
import os, io, time, json, re
import duckdb
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from typing import Dict, List, Optional, Tuple

st.set_page_config(page_title="DailyDeck Sections — Wow", page_icon="📚", layout="wide")

# --------------------
# Styling
# --------------------
st.markdown("""
<style>
.sidebar .sidebar-content { width: 360px; }
.stButton>button { border-radius: 14px; padding:.55rem 1rem; box-shadow:0 8px 22px rgba(0,0,0,.08); }
.card { border:1px solid #eef0f6; border-radius:18px; padding:14px; background:linear-gradient(180deg,#fafafd 0%,#f4f6fc 100%);}
.hero { border-radius:22px; padding:20px; border:1px solid #e9eef7; background: radial-gradient(1400px 360px at 100% 0%, #f3f7ff 0%, #ffffff 70%);}
</style>
""", unsafe_allow_html=True)

# --------------------
# Sections (provided by user)
# --------------------
RAW_SECTIONS = """*Install required tools
*Import Libraries
*Upload Data set
*Convert Dates to Datetime
*Clean Numeric Columns for analysis
SALES
Global sales Overview
Global Net Sales Distribution by Sales Channel
Global Net Sales Distribution by SHIFT
Night vs Day Shift Sales Ratio — Stores with Night Shifts
Global Day vs Night Sales — Only Stores with NIGHT Shift
2nd-Highest Channel Share
Bottom 30 — 2nd Highest Channel
*Sales Workings_1
*Sales Workings_2
*Sales Workings_3
*Sales Workings_4
*Sales Workings_5
Stores Sales Summary
OPERATIONS
Customer Traffic-Storewise
Active Tills During the day
Average Customers Served per Till
Store Customer Traffic Storewise
*Customer Traffic Computation 1
*Customer Traffic Computation2
*Customer Traffic Computation 3
Customer Traffic-Departmentwise
Cashiers Perfomance
Till Usage
Tax Compliance
Customer Baskets Overview
Global Category Overview-Sales
Global Category Overview-Baskets
Supplier Contribution
Category Overview
Branch Comparison
Product Perfomance
Global Loyalty Overview
Branch Loyalty Overview
Customer Loyalty Overview
Global Pricing Overview
Branch Brach Overview
Global Refunds Overview
Branch Refunds Overview"""
SECTIONS = [s for s in [ln.strip() for ln in RAW_SECTIONS.splitlines()] if s and not s.startswith("*")]

# Group headline labels (no charts), to separate nav
GROUPS = {"SALES","OPERATIONS"}

if "csv_path" not in st.session_state:
    st.session_state["csv_path"] = None
if "duckdb_loaded" not in st.session_state:
    st.session_state["duckdb_loaded"] = False
if "columns" not in st.session_state:
    st.session_state["columns"] = []

# --------------------
# Sidebar: Upload + Filters + Nav
# --------------------
with st.sidebar:
    st.header("📤 Upload")
    csv_file = st.file_uploader("Upload CSV (up to ~500MB)", type=["csv"])
    lim = st.number_input("Preview/Compute row cap (for safety)", min_value=50000, max_value=5000000, value=1000000, step=50000, help="DuckDB will scan with this LIMIT to keep things responsive.")

    st.divider()
    st.header("🔎 Filters")

    # Allow user to map key columns if auto-detect fails
    st.caption("Auto-detected columns can be overridden below.")

# Keep a connection alive per run
def get_con() -> duckdb.DuckDBPyConnection:
    if not hasattr(st.session_state, "con") or st.session_state.con is None:
        st.session_state.con = duckdb.connect(database=":memory:")
    return st.session_state.con

# Save upload to /tmp for DuckDB
if csv_file is not None:
    tmp_path = os.path.join("/tmp", f"upload_{int(time.time())}.csv")
    with open(tmp_path, "wb") as f:
        f.write(csv_file.getvalue())
    st.session_state["csv_path"] = tmp_path
    st.session_state["duckdb_loaded"] = False

# Define synonym sets for robust detection
SYN = {
    "date": ["TRN_DATE","DATE","TXN_DATE","DATETIME","TIME","TRANS_DATE","SALE_DATE"],
    "store": ["STORE_NAME","STORE","BRANCH","STORE_CODE","BRANCH_NAME"],
    "department": ["DEPARTMENT","DEPT","DEPT_NAME"],
    "category": ["CATEGORY","CAT","CAT_NAME"],
    "supplier": ["SUPPLIER","VENDOR","SUPPLIER_NAME"],
    "till": ["TILL","TILL_NO","TILL_ID","REGISTER","POS"],
    "cashier": ["CASHIER","CASHIER_NAME","USER","CLERK"],
    "customer_id": ["CUST_CODE","CUSTOMER_ID","RECEIPT_ID","TXN_ID","DOC_NO","RCT","INVOICE_NO"],
    "channel": ["CHANNEL","SALES_CHANNEL","MODE","TENDER_TYPE","PAYMENT_CHANNEL"],
    "shift": ["SHIFT","WORK_SHIFT"],
    "net_sales": ["SALES_PRE_VAT","NET_SALES","NET_SALES_PRE_VAT","SP_PRE_VAT","AMOUNT","TOTAL_SALES"],
    "gross_profit": ["GROSS_PROFIT","GP","MARGIN_AMOUNT"],
    "qty": ["QUANTITY","QTY","UNITS","PCS"],
    "price": ["AVG_SP_PRE_VAT","UNIT_PRICE","PRICE","SP_PRE_VAT"],
    "tax_flag": ["CU_DEVICE_SERIAL","TAX_FLAG","TAX_COMPLIANT"]
}

def infer_columns(con, path) -> Dict[str,str]:
    # Sample small to read header
    df = con.execute(f"SELECT * FROM read_csv_auto('{path}', SAMPLE_SIZE=200000, HEADER=TRUE) LIMIT 20000").df()
    cols = list(df.columns)
    st.session_state["columns"] = cols
    found = {}
    for key, cands in SYN.items():
        for c in cands:
            if c in cols:
                found[key] = c
                break
    # derive SHIFT if not present
    if "shift" not in found and "date" in found:
        found["shift"] = "__DERIVED_SHIFT__"
    return found

def ensure_loaded(path):
    con = get_con()
    if not st.session_state["duckdb_loaded"]:
        con.execute("DROP VIEW IF EXISTS v;")
        con.execute(f"CREATE VIEW v AS SELECT * FROM read_csv_auto('{path}', HEADER=TRUE, SAMPLE_SIZE=2000000);")
        st.session_state["duckdb_loaded"] = True

def derive_shift_expr(date_col):
    # Day 06:00-17:59, Night 18:00-05:59
    return f"CASE WHEN EXTRACT(HOUR FROM CAST({date_col} AS TIMESTAMP)) BETWEEN 6 AND 17 THEN 'DAY' ELSE 'NIGHT' END"

def safe_select(expr, where="", groupby="", orderby="", limit_clause=""):
    con = get_con()
    ensure_loaded(st.session_state["csv_path"])
    where_clause = f"WHERE {where}" if where else ""
    group_clause = f"GROUP BY {groupby}" if groupby else ""
    order_clause = f"ORDER BY {orderby}" if orderby else ""
    limit_clause = f"LIMIT {int(st.session_state.get('row_cap', 1000000))}" if not limit_clause else limit_clause
    q = f"SELECT {expr} FROM v {where_clause} {group_clause} {order_clause} {limit_clause};"
    return con.execute(q).df()

# Load data if present
if st.session_state["csv_path"]:
    con = get_con()
    ensure_loaded(st.session_state["csv_path"])
    found = infer_columns(con, st.session_state["csv_path"])
    with st.sidebar:
        st.caption("Detected / override:")
        date_col = st.selectbox("Date column", options=["(none)"] + st.session_state["columns"], index=(st.session_state["columns"].index(found.get("date"))+1 if found.get("date") in st.session_state["columns"] else 0))
        store_col = st.selectbox("Store column", options=["(none)"] + st.session_state["columns"], index=(st.session_state["columns"].index(found.get("store"))+1 if found.get("store") in st.session_state["columns"] else 0))
        channel_col = st.selectbox("Channel column", options=["(none)"] + st.session_state["columns"], index=(st.session_state["columns"].index(found.get("channel"))+1 if found.get("channel") in st.session_state["columns"] else 0))
        shift_col = st.selectbox("Shift column (or derive)", options=["(derive from Date)"] + st.session_state["columns"], index=(st.session_state["columns"].index(found.get("shift"))+1 if found.get("shift") in st.session_state["columns"] else 0))
        dept_col = st.selectbox("Department column", options=["(none)"] + st.session_state["columns"], index=(st.session_state["columns"].index(found.get("department"))+1 if found.get("department") in st.session_state["columns"] else 0))
        cat_col = st.selectbox("Category column", options=["(none)"] + st.session_state["columns"], index=(st.session_state["columns"].index(found.get("category"))+1 if found.get("category") in st.session_state["columns"] else 0))
        supplier_col = st.selectbox("Supplier column", options=["(none)"] + st.session_state["columns"], index=(st.session_state["columns"].index(found.get("supplier"))+1 if found.get("supplier") in st.session_state["columns"] else 0))
        till_col = st.selectbox("Till column", options=["(none)"] + st.session_state["columns"], index=(st.session_state["columns"].index(found.get("till"))+1 if found.get("till") in st.session_state["columns"] else 0))
        cashier_col = st.selectbox("Cashier column", options=["(none)"] + st.session_state["columns"], index=(st.session_state["columns"].index(found.get("cashier"))+1 if found.get("cashier") in st.session_state["columns"] else 0))
        cust_col = st.selectbox("Receipt/Customer ID column", options=["(none)"] + st.session_state["columns"], index=(st.session_state["columns"].index(found.get("customer_id"))+1 if found.get("customer_id") in st.session_state["columns"] else 0))
        sales_col = st.selectbox("Net Sales column", options=["(none)"] + st.session_state["columns"], index=(st.session_state["columns"].index(found.get("net_sales"))+1 if found.get("net_sales") in st.session_state["columns"] else 0))
        gp_col = st.selectbox("Gross Profit column", options=["(none)"] + st.session_state["columns"], index=(st.session_state["columns"].index(found.get("gross_profit"))+1 if found.get("gross_profit") in st.session_state["columns"] else 0))
        qty_col = st.selectbox("Quantity column", options=["(none)"] + st.session_state["columns"], index=(st.session_state["columns"].index(found.get("qty"))+1 if found.get("qty") in st.session_state["columns"] else 0))
        tax_col = st.selectbox("Tax Compliance flag", options=["(none)"] + st.session_state["columns"], index=(st.session_state["columns"].index(found.get("tax_flag"))+1 if found.get("tax_flag") in st.session_state["columns"] else 0))

        st.markdown("**Base Filters**")
        store_filter = st.text_input("Store filter (comma-separated values)", value="")
        dept_filter = st.text_input("Department filter (comma-separated)", value="")
        cat_filter = st.text_input("Category filter (comma-separated)", value="")

    st.session_state["row_cap"] = lim

    # Build WHERE from filters
    filters = []
    def csv_to_in(colname, val):
        vals = [v.strip() for v in val.split(",") if v.strip()]
        if not vals: return ""
        arr = ",".join([f"'{v.replace("'","''")}'" for v in vals])
        return f"{colname} IN ({arr})"

    if store_col != "(none)" and store_filter:
        filters.append(csv_to_in(store_col, store_filter))
    if dept_col != "(none)" and dept_filter:
        filters.append(csv_to_in(dept_col, dept_filter))
    if cat_col != "(none)" and cat_filter:
        filters.append(csv_to_in(cat_col, cat_filter))

    where = " AND ".join([f for f in filters if f])

else:
    found = {}
    where = ""

st.markdown("<div class='hero'><h3>📚 DailyDeck — Click a section to view</h3><p>Upload your CSV, set column mappings if needed, and browse sections below. Starred items are hidden automatically.</p></div>", unsafe_allow_html=True)
st.write("")

# Build nav
cols = st.columns([1,3,12])
with cols[0]:
    st.markdown("#### Sections")
    for s in SECTIONS:
        if s in GROUPS:
            st.markdown(f"**{s}**")
        else:
            if st.button(s, key=f"nav_{s}"):
                st.session_state["active_section"] = s

active = st.session_state.get("active_section", None)
if active is None:
    active = next((s for s in SECTIONS if s not in GROUPS), None)
    st.session_state["active_section"] = active

with cols[2]:
    st.markdown(f"### {active if active else ''}")

    def ensure_shift_alias():
        if not st.session_state.get("csv_path"): return None
        if shift_col != "(derive from Date)":
            return shift_col
        if date_col != "(none)":
            return derive_shift_expr(date_col)
        return None

    if not st.session_state.get("csv_path"):
        st.info("Upload a CSV on the left to activate outputs.")
    else:
        def chart_df(df, kind="bar", x=None, y=None, title=""):
            if df is None or df.empty:
                st.warning("No data for the selected filters/columns.")
                return
            if kind == "bar":
                fig = px.bar(df, x=x, y=y)
            elif kind == "line":
                fig = px.line(df, x=x, y=y)
            elif kind == "pie":
                fig = px.pie(df, names=x, values=y)
            else:
                fig = px.bar(df, x=x, y=y)
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download data", df.to_csv(index=False).encode("utf-8"), file_name="section.csv", use_container_width=True)

        def q_total_sales_by(expr_group):
            if sales_col == "(none)":
                return pd.DataFrame()
            expr = f"{expr_group} AS grp, SUM(CAST({sales_col} AS DOUBLE)) AS net_sales"
            dfq = safe_select(expr, where=where, groupby="grp", orderby="net_sales DESC", limit_clause="")
            return dfq

        if active == "Global sales Overview":
            if sales_col == "(none)":
                st.error("Map the Net Sales column in the sidebar.")
            else:
                expr = f"SUM(CAST({sales_col} AS DOUBLE)) AS net_sales, COUNT(*) AS rows"
                dfq = safe_select(expr, where=where, limit_clause="")
                st.metric("Net Sales (preview scope)", f"{dfq['net_sales'].iloc[0]:,.0f}")
                st.metric("Rows", f"{dfq['rows'].iloc[0]:,}")

        elif active == "Global Net Sales Distribution by Sales Channel":
            if channel_col == "(none)" or sales_col == "(none)":
                st.error("Map Channel and Net Sales columns.")
            else:
                dfq = q_total_sales_by(channel_col)
                chart_df(dfq, "bar", x="grp", y="net_sales")

        elif active == "Global Net Sales Distribution by SHIFT":
            sc = ensure_shift_alias()
            if not sc or sales_col == "(none)":
                st.error("Need Shift (or Date to derive) and Net Sales.")
            else:
                dfq = q_total_sales_by(sc)
                chart_df(dfq, "pie", x="grp", y="net_sales")

        elif active == "Night vs Day Shift Sales Ratio — Stores with Night Shifts":
            sc = ensure_shift_alias()
            if not sc or store_col == "(none)" or sales_col == "(none)":
                st.error("Need Store, Shift (or Date), and Net Sales.")
            else:
                expr = f"{store_col} AS store, CASE WHEN ({sc})='NIGHT' THEN 1 ELSE 0 END AS is_night, CAST({sales_col} AS DOUBLE) AS ns"
                dfbase = safe_select(expr, where=where, limit_clause="")
                if dfbase.empty:
                    st.warning("No data.")
                else:
                    have_night = dfbase.groupby("store")["is_night"].sum().reset_index()
                    have_night = have_night[have_night["is_night"]>0]["store"]
                    filt = dfbase[dfbase["store"].isin(have_night)]
                    agg = filt.groupby([ "store", filt["is_night"].map({1:"NIGHT",0:"DAY"}) ] )["ns"].sum().reset_index().rename(columns={"is_night":"shift","ns":"net_sales"})
                    chart_df(agg, "bar", x="store", y="net_sales")

        elif active == "Global Day vs Night Sales — Only Stores with NIGHT Shift":
            sc = ensure_shift_alias()
            if not sc or store_col == "(none)" or sales_col == "(none)":
                st.error("Need Store, Shift (or Date), and Net Sales.")
            else:
                expr = f"{store_col} AS store, ({sc}) AS shift, SUM(CAST({sales_col} AS DOUBLE)) AS net_sales"
                dfq = safe_select(expr, where=where, groupby="store, shift", orderby="store, shift", limit_clause="")
                have_night = dfq[dfq["shift"]=="NIGHT"]["store"].unique().tolist()
                dfq = dfq[dfq["store"].isin(have_night)]
                chart_df(dfq, "bar", x="store", y="net_sales")

        elif active == "2nd-Highest Channel Share":
            if channel_col == "(none)" or sales_col == "(none)":
                st.error("Map Channel and Net Sales.")
            else:
                expr = f"{channel_col} AS ch, SUM(CAST({sales_col} AS DOUBLE)) AS ns"
                dfq = safe_select(expr, where=where, groupby="ch", orderby="ns DESC", limit_clause="")
                if dfq.empty:
                    st.warning("No data.")
                else:
                    if len(dfq)>=2:
                        chart_df(dfq.iloc[1:2], "bar", x="ch", y="ns")
                    else:
                        st.info("Less than two channels found.")

        elif active == "Bottom 30 — 2nd Highest Channel":
            if store_col == "(none)" or channel_col == "(none)" or sales_col == "(none)":
                st.error("Map Store, Channel, Net Sales.")
            else:
                expr = f"{store_col} AS store, {channel_col} AS ch, SUM(CAST({sales_col} AS DOUBLE)) AS ns"
                dfq = safe_select(expr, where=where, groupby="store, ch", orderby="store, ns DESC", limit_clause="")
                second = dfq.sort_values(["store","ns"], ascending=[True, False]).groupby("store").nth(1).reset_index()
                chart_df(second.sort_values("ns").head(30), "bar", x="store", y="ns")

        elif active == "Stores Sales Summary":
            if store_col == "(none)" or sales_col == "(none)":
                st.error("Map Store and Net Sales.")
            else:
                expr = f"{store_col} AS store, SUM(CAST({sales_col} AS DOUBLE)) AS net_sales, COUNT(*) AS txns"
                dfq = safe_select(expr, where=where, groupby="store", orderby="net_sales DESC", limit_clause="")
                chart_df(dfq, "bar", x="store", y="net_sales")

        elif active == "Customer Traffic-Storewise":
            if store_col == "(none)" or cust_col == "(none)":
                st.error("Map Store and Receipt/Customer ID.")
            else:
                expr = f"{store_col} AS store, COUNT(DISTINCT {cust_col}) AS baskets"
                dfq = safe_select(expr, where=where, groupby="store", orderby="baskets DESC", limit_clause="")
                chart_df(dfq, "bar", x="store", y="baskets")

        elif active == "Active Tills During the day":
            if till_col == "(none)" or date_col == "(none)":
                st.error("Map Till and Date.")
            else:
                expr = f"DATE_TRUNC('hour', CAST({date_col} AS TIMESTAMP)) AS hour, COUNT(DISTINCT {till_col}) AS active_tills"
                dfq = safe_select(expr, where=where, groupby="hour", orderby="hour", limit_clause="")
                chart_df(dfq, "line", x="hour", y="active_tills")

        elif active == "Average Customers Served per Till":
            if till_col == "(none)" or cust_col == "(none)":
                st.error("Map Till and Receipt/Customer ID.")
            else:
                expr = f"{till_col} AS till, COUNT(DISTINCT {cust_col})*1.0 / NULLIF(COUNT(DISTINCT {till_col}),0) AS avg_cust_per_till"
                dfq = safe_select(expr, where=where, groupby="till", orderby="avg_cust_per_till DESC", limit_clause="")
                chart_df(dfq, "bar", x="till", y="avg_cust_per_till")

        elif active == "Store Customer Traffic Storewise":
            if store_col == "(none)" or cust_col == "(none)":
                st.error("Map Store and Receipt/Customer ID.")
            else:
                expr = f"{store_col} AS store, COUNT(DISTINCT {cust_col}) AS baskets"
                dfq = safe_select(expr, where=where, groupby="store", orderby="baskets DESC", limit_clause="")
                chart_df(dfq, "bar", x="store", y="baskets")

        elif active == "Customer Traffic-Departmentwise":
            if dept_col == "(none)" or cust_col == "(none)":
                st.error("Map Department and Receipt/Customer ID.")
            else:
                expr = f"{dept_col} AS dept, COUNT(DISTINCT {cust_col}) AS baskets"
                dfq = safe_select(expr, where=where, groupby="dept", orderby="baskets DESC", limit_clause="")
                chart_df(dfq, "bar", x="dept", y="baskets")

        elif active == "Cashiers Perfomance":
            if cashier_col == "(none)" or sales_col == "(none)":
                st.error("Map Cashier and Net Sales.")
            else:
                expr = f"{cashier_col} AS cashier, SUM(CAST({sales_col} AS DOUBLE)) AS net_sales, COUNT(*) AS txns"
                dfq = safe_select(expr, where=where, groupby="cashier", orderby="net_sales DESC", limit_clause="")
                chart_df(dfq, "bar", x="cashier", y="net_sales")

        elif active == "Till Usage":
            if till_col == "(none)":
                st.error("Map Till.")
            else:
                expr = f"{till_col} AS till, COUNT(*) AS uses"
                dfq = safe_select(expr, where=where, groupby="till", orderby="uses DESC", limit_clause="")
                chart_df(dfq, "bar", x="till", y="uses")

        elif active == "Tax Compliance":
            if tax_col == "(none)":
                st.error("Map Tax flag (e.g., CU_DEVICE_SERIAL).")
            else:
                expr = f"CASE WHEN {tax_col} IS NULL OR {tax_col}='' THEN 'NON_COMPLIANT' ELSE 'COMPLIANT' END AS status, COUNT(*) AS txns"
                dfq = safe_select(expr, where=where, groupby="status", orderby="txns DESC", limit_clause="")
                chart_df(dfq, "pie", x="status", y="txns")

        elif active == "Customer Baskets Overview":
            if cust_col == "(none)" or sales_col == "(none)":
                st.error("Map Receipt/Customer ID and Net Sales.")
            else:
                expr = f"{cust_col} AS basket, SUM(CAST({sales_col} AS DOUBLE)) AS basket_value"
                dfq = safe_select(expr, where=where, groupby="basket", orderby="basket_value DESC", limit_clause="LIMIT 10000")
                st.write(dfq.describe())
                chart_df(dfq.sort_values("basket_value", ascending=False).head(100), "bar", x="basket", y="basket_value")

        elif active == "Global Category Overview-Sales":
            if cat_col == "(none)" or sales_col == "(none)":
                st.error("Map Category and Net Sales.")
            else:
                expr = f"{cat_col} AS cat, SUM(CAST({sales_col} AS DOUBLE)) AS net_sales"
                dfq = safe_select(expr, where=where, groupby="cat", orderby="net_sales DESC", limit_clause="")
                chart_df(dfq, "bar", x="cat", y="net_sales")

        elif active == "Global Category Overview-Baskets":
            if cat_col == "(none)" or cust_col == "(none)":
                st.error("Map Category and Receipt/Customer ID.")
            else:
                expr = f"{cat_col} AS cat, COUNT(DISTINCT {cust_col}) AS baskets"
                dfq = safe_select(expr, where=where, groupby="cat", orderby="baskets DESC", limit_clause="")
                chart_df(dfq, "bar", x="cat", y="baskets")

        elif active == "Supplier Contribution":
            if supplier_col == "(none)" or sales_col == "(none)":
                st.error("Map Supplier and Net Sales.")
            else:
                expr = f"{supplier_col} AS supplier, SUM(CAST({sales_col} AS DOUBLE)) AS net_sales"
                dfq = safe_select(expr, where=where, groupby="supplier", orderby="net_sales DESC", limit_clause="")
                chart_df(dfq.head(50), "bar", x="supplier", y="net_sales")

        elif active == "Category Overview":
            if dept_col == "(none)" or cat_col == "(none)" or sales_col == "(none)":
                st.error("Map Department, Category, and Net Sales.")
            else:
                expr = f"{dept_col} AS dept, {cat_col} AS cat, SUM(CAST({sales_col} AS DOUBLE)) AS net_sales"
                dfq = safe_select(expr, where=where, groupby="dept, cat", orderby="net_sales DESC", limit_clause="")
                chart_df(dfq, "bar", x="cat", y="net_sales")

        elif active == "Branch Comparison":
            if store_col == "(none)" or sales_col == "(none)":
                st.error("Map Store and Net Sales.")
            else:
                expr = f"{store_col} AS store, SUM(CAST({sales_col} AS DOUBLE)) AS net_sales"
                dfq = safe_select(expr, where=where, groupby="store", orderby="net_sales DESC", limit_clause="")
                chart_df(dfq, "bar", x="store", y="net_sales")

        elif active == "Product Perfomance":
            item_code = next((c for c in st.session_state["columns"] if c.upper() in ["ITEM_CODE","SKU","PRODUCT_CODE"]), None)
            item_name = next((c for c in st.session_state["columns"] if c.upper() in ["ITEM_NAME","PRODUCT","SKU_NAME"]), None)
            if (item_code or item_name) and sales_col != "(none)":
                expr = f"COALESCE({item_name or item_code}, {item_code or item_name}) AS item, SUM(CAST({sales_col} AS DOUBLE)) AS net_sales"
                dfq = safe_select(expr, where=where, groupby="item", orderby="net_sales DESC", limit_clause="")
                chart_df(dfq.head(50), "bar", x="item", y="net_sales")
            else:
                st.error("Need ITEM_NAME/ITEM_CODE and Net Sales.")

        elif active == "Global Loyalty Overview":
            loy = next((c for c in st.session_state["columns"] if "LOYAL" in c.upper()), None)
            if not loy:
                st.error("No loyalty column detected (name containing 'LOYAL').")
            else:
                expr = f"CASE WHEN {loy} IN ('1','Y','YES',1,TRUE) THEN 'LOYAL' ELSE 'NON-LOYAL' END AS segment, COUNT(*) AS txns"
                dfq = safe_select(expr, where=where, groupby="segment", orderby="txns DESC", limit_clause="")
                chart_df(dfq, "pie", x="segment", y="txns")

        elif active == "Branch Loyalty Overview":
            loy = next((c for c in st.session_state["columns"] if "LOYAL" in c.upper()), None)
            if not loy or store_col == "(none)":
                st.error("Need a loyalty column and Store mapping.")
            else:
                expr = f"{store_col} AS store, CASE WHEN {loy} IN ('1','Y','YES',1,TRUE) THEN 'LOYAL' ELSE 'NON-LOYAL' END AS segment, COUNT(*) AS txns"
                dfq = safe_select(expr, where=where, groupby="store, segment", orderby="store, segment", limit_clause="")
                chart_df(dfq, "bar", x="store", y="txns")

        elif active == "Customer Loyalty Overview":
            loy = next((c for c in st.session_state["columns"] if "LOYAL" in c.upper()), None)
            if not loy or cust_col == "(none)":
                st.error("Need a loyalty column and Receipt/Customer ID.")
            else:
                expr = f"{cust_col} AS cust, CASE WHEN {loy} IN ('1','Y','YES',1,TRUE) THEN 'LOYAL' ELSE 'NON-LOYAL' END AS segment, COUNT(*) AS txns"
                dfq = safe_select(expr, where=where, groupby="cust, segment", orderby="txns DESC", limit_clause="")
                chart_df(dfq.head(200), "bar", x="cust", y="txns")

        elif active == "Global Pricing Overview":
            price_col = next((c for c in st.session_state["columns"] if c.upper() in ["AVG_SP_PRE_VAT","UNIT_PRICE","PRICE","SP_PRE_VAT"]), None)
            if not price_col:
                st.error("No price column detected.")
            else:
                expr = f"ROUND(CAST({price_col} AS DOUBLE), 0) AS price_bucket, COUNT(*) AS rows"
                dfq = safe_select(expr, where=where, groupby="price_bucket", orderby="price_bucket", limit_clause="")
                chart_df(dfq, "line", x="price_bucket", y="rows")

        elif active == "Branch Brach Overview":
            if store_col == "(none)" or dept_col == "(none)" or sales_col == "(none)":
                st.error("Map Store, Department, Net Sales.")
            else:
                expr = f"{store_col} AS store, {dept_col} AS dept, SUM(CAST({sales_col} AS DOUBLE)) AS net_sales"
                dfq = safe_select(expr, where=where, groupby="store, dept", orderby="store, net_sales DESC", limit_clause="")
                chart_df(dfq, "bar", x="dept", y="net_sales")

        elif active == "Global Refunds Overview":
            ref = next((c for c in st.session_state["columns"] if "REFUND" in c.upper() or "VOID" in c.upper()), None)
            if not ref:
                st.error("No refund/void indicator column detected.")
            else:
                expr = f"CASE WHEN {ref} IN ('1','Y','YES',1,TRUE) THEN 'REFUND/VOID' ELSE 'NORMAL' END AS txntype, COUNT(*) AS txns"
                dfq = safe_select(expr, where=where, groupby="txntype", orderby="txns DESC", limit_clause="")
                chart_df(dfq, "pie", x="txntype", y="txns")

        elif active == "Branch Refunds Overview":
            ref = next((c for c in st.session_state["columns"] if "REFUND" in c.upper() or "VOID" in c.upper()), None)
            if not ref or store_col == "(none)":
                st.error("Need refund/void indicator and Store mapping.")
            else:
                expr = f"{store_col} AS store, CASE WHEN {ref} IN ('1','Y','YES',1,TRUE) THEN 'REFUND/VOID' ELSE 'NORMAL' END AS txntype, COUNT(*) AS txns"
                dfq = safe_select(expr, where=where, groupby="store, txntype", orderby="store, txns DESC", limit_clause="")
                chart_df(dfq, "bar", x="store", y="txns")

        else:
            st.info("Section logic not recognized or is a group heading.")
