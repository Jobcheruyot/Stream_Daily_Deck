"""
Superdeck — Streamlit Analytics Dashboard (UI-first, Colab->Streamlit port)
- Shows three prominent category cards (SALES, OPERATIONS, INSIGHTS)
- When a category is chosen the subsections for that category are shown
- Selecting a subsection renders the corresponding visualization (translated from Colab notebook)
- Background precompute runs all dependency steps so subsections render instantly
- Robust to missing columns (shows friendly messages), supports chunked CSV reads
- Numeric formatting, download buttons, runtime config debug

Instructions:
- Place this file as app.py in your repo.
- Add `.streamlit/config.toml` with server.maxUploadSize = 1024 if you want Streamlit to accept 1GB uploads (and restart).
- Recommended packages: streamlit, pandas, numpy, plotly, boto3 (optional), pyarrow (optional)
"""
from __future__ import annotations
import os
import io
import textwrap
from datetime import timedelta
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------
# Page config + styling
# -----------------------
st.set_page_config(page_title="Superdeck — Analytics", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      /* Sidebar width */
      [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
          width: 360px; min-width:320px;
      }
      .card { border-radius:10px; padding:14px; box-shadow: 0 4px 14px rgba(20,20,20,0.06); }
      .card h2 { margin:4px 0 8px 0; }
      .muted { color:#6c757d; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Helper utilities
# -----------------------
def safe_read_csv_bytes(uploaded, chunksize: int = 200_000) -> pd.DataFrame:
    """
    Read uploaded file-like (Streamlit UploadedFile or BytesIO).
    If small -> plain read, if large -> chunked read and concat.
    """
    try:
        # Convert to BytesIO if necessary
        if hasattr(uploaded, "getvalue"):
            b = io.BytesIO(uploaded.getvalue())
        else:
            uploaded.seek(0)
            b = io.BytesIO(uploaded.read())
        size_mb = len(b.getvalue()) / (1024 * 1024)
    except Exception:
        b = uploaded
        size_mb = None

    b.seek(0)
    # If we detect >200MB, use chunking
    if size_mb and size_mb > 200:
        parts = []
        try:
            for chunk in pd.read_csv(b, on_bad_lines="skip", low_memory=False, chunksize=chunksize):
                parts.append(chunk)
            df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        except Exception as e:
            raise RuntimeError(f"Chunked CSV load failed: {e}")
    else:
        try:
            df = pd.read_csv(b, on_bad_lines="skip", low_memory=False)
        except Exception as e:
            raise RuntimeError(f"CSV load failed: {e}")
    return df

def to_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
    return df

def fmt_int(v):
    try:
        return f"{int(v):,}"
    except Exception:
        return v

def fmt_float(v, d=2):
    try:
        return f"{float(v):,.{d}f}"
    except Exception:
        return v

def download_df_button(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# -----------------------
# Data preparation + precomputations (cached)
# -----------------------
@st.cache_data(show_spinner=True)
def preprocess_all(uploaded_file) -> Dict[str, Any]:
    """
    Load CSV and compute derived tables used across subsections.
    Returns a dictionary of dataframes and computed objects.
    """
    # 1) Load
    df = safe_read_csv_bytes(uploaded_file)

    # Normalize column names (trim)
    df.columns = [c.strip() for c in df.columns]

    # Parse dates if present
    for col in ["TRN_DATE", "ZED_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Clean numeric columns used widely
    numeric_cols = ["QTY", "CP_PRE_VAT", "SP_PRE_VAT", "COST_PRE_VAT", "NET_SALES", "VAT_AMT"]
    df = to_numeric_cols(df, numeric_cols)

    # Ensure IDs are strings
    idcols = ["STORE_CODE", "TILL", "SESSION", "RCT"]
    for c in idcols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()

    # Construct CUST_CODE if missing and components exist
    if "CUST_CODE" not in df.columns and all(c in df.columns for c in idcols):
        df["CUST_CODE"] = (
            df["STORE_CODE"].astype(str).str.strip()
            + "-"
            + df["TILL"].astype(str).str.strip()
            + "-"
            + df["SESSION"].astype(str).str.strip()
            + "-"
            + df["RCT"].astype(str).str.strip()
        )

    # Ensure useful text columns exist so access won't KeyError
    text_cols_default = [
        "STORE_NAME", "ITEM_NAME", "ITEM_CODE", "DEPARTMENT", "CATEGORY",
        "CASHIER", "CAP_CUSTOMER_CODE", "LOYALTY_CUSTOMER_CODE", "CU_DEVICE_SERIAL",
        "SHIFT", "SALES_CHANNEL_L1", "SALES_CHANNEL_L2"
    ]
    for c in text_cols_default:
        if c not in df.columns:
            df[c] = ""

    # Ensure NET_SALES exists numeric
    if "NET_SALES" not in df.columns:
        df["NET_SALES"] = 0
    else:
        df["NET_SALES"] = pd.to_numeric(df["NET_SALES"], errors="coerce").fillna(0)

    # Derived: GROSS_SALES
    if "GROSS_SALES" not in df.columns:
        if "VAT_AMT" in df.columns:
            df["GROSS_SALES"] = df["NET_SALES"] + df["VAT_AMT"]
        else:
            df["GROSS_SALES"] = df["NET_SALES"]

    # Derived: Till_Code
    if "Till_Code" not in df.columns:
        df["TILL"] = df["TILL"].astype(str).fillna("").str.strip()
        df["STORE_CODE"] = df["STORE_CODE"].astype(str).fillna("").str.strip()
        df["Till_Code"] = df["TILL"] + "-" + df["STORE_CODE"]

    # Precompute common items used across visuals
    results: Dict[str, Any] = {"df": df}

    # SALES-level aggregations
    try:
        gs = df.groupby("SALES_CHANNEL_L1", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        gs["NET_SALES_M"] = gs["NET_SALES"] / 1_000_000
        gs["PCT"] = 100 * gs["NET_SALES"] / gs["NET_SALES"].sum() if gs["NET_SALES"].sum() != 0 else 0.0
        results["global_sales"] = gs
    except Exception:
        results["global_sales"] = pd.DataFrame()

    try:
        channel2 = df.groupby("SALES_CHANNEL_L2", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        channel2["NET_SALES_M"] = channel2["NET_SALES"] / 1_000_000
        channel2["PCT"] = 100 * channel2["NET_SALES"] / channel2["NET_SALES"].sum() if channel2["NET_SALES"].sum() != 0 else 0.0
        results["channel2_sales"] = channel2
    except Exception:
        results["channel2_sales"] = pd.DataFrame()

    try:
        shift_sales = df.groupby("SHIFT", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        shift_sales["PCT"] = 100 * shift_sales["NET_SALES"] / shift_sales["NET_SALES"].sum() if shift_sales["NET_SALES"].sum() != 0 else 0.0
        results["shift_sales"] = shift_sales
    except Exception:
        results["shift_sales"] = pd.DataFrame()

    # Night vs day per store
    try:
        stores_with_night = df[df["SHIFT"].astype(str).str.upper().str.contains("NIGHT", na=False)]["STORE_NAME"].unique()
        if len(stores_with_night) > 0:
            nd = df[df["STORE_NAME"].isin(stores_with_night)].copy()
            nd["Shift_Bucket"] = np.where(nd["SHIFT"].astype(str).str.upper().str.contains("NIGHT", na=False), "Night", "Day")
            ratio = nd.groupby(["STORE_NAME", "Shift_Bucket"], as_index=False)["NET_SALES"].sum()
            store_tot = ratio.groupby("STORE_NAME")["NET_SALES"].transform("sum")
            ratio["PCT"] = 100 * ratio["NET_SALES"] / store_tot
            pivot = ratio.pivot(index="STORE_NAME", columns="Shift_Bucket", values="PCT").fillna(0)
            results["night_day_pivot"] = pivot
            results["global_nd"] = ratio
        else:
            results["night_day_pivot"] = pd.DataFrame()
            results["global_nd"] = pd.DataFrame()
    except Exception:
        results["night_day_pivot"] = pd.DataFrame()
        results["global_nd"] = pd.DataFrame()

    # 2nd-highest channel share per store
    try:
        store_chan = df.groupby(["STORE_NAME", "SALES_CHANNEL_L1"], as_index=False)["NET_SALES"].sum()
        store_tot = store_chan.groupby("STORE_NAME")["NET_SALES"].transform("sum")
        store_chan["PCT"] = np.where(store_tot > 0, 100 * store_chan["NET_SALES"] / store_tot, 0.0)
        store_chan = store_chan.sort_values(["STORE_NAME", "PCT"], ascending=[True, False])
        store_chan["RANK"] = store_chan.groupby("STORE_NAME").cumcount() + 1
        second = store_chan[store_chan["RANK"] == 2].copy()
        # include stores with only 1 channel
        all_stores = store_chan["STORE_NAME"].drop_duplicates()
        missing_stores = set(all_stores) - set(second["STORE_NAME"])
        if missing_stores:
            second = pd.concat([second, pd.DataFrame({"STORE_NAME": list(missing_stores), "SALES_CHANNEL_L1": "(None)", "NET_SALES": 0, "PCT": 0})], ignore_index=True)
        results["second_channel_table"] = second.sort_values("PCT", ascending=False)
    except Exception:
        results["second_channel_table"] = pd.DataFrame()

    # Stores sales summary (gross, customers)
    try:
        if "GROSS_SALES" not in df.columns:
            df["GROSS_SALES"] = df["NET_SALES"] + df.get("VAT_AMT", 0)
        ss = df.groupby("STORE_NAME", as_index=False).agg(NET_SALES=("NET_SALES", "sum"), GROSS_SALES=("GROSS_SALES", "sum"))
        ss["Customer_Numbers"] = df.groupby("STORE_NAME")["CUST_CODE"].nunique().reindex(ss["STORE_NAME"]).fillna(0).astype(int).values
        total_gross = ss["GROSS_SALES"].sum()
        ss["Pct_Contribution"] = 100 * ss["GROSS_SALES"] / total_gross if total_gross != 0 else 0.0
        results["stores_sales_summary"] = ss.sort_values("GROSS_SALES", ascending=False)
    except Exception:
        results["stores_sales_summary"] = pd.DataFrame()

    # OPERATIONS: Customer traffic per 30-min slot
    try:
        if "TRN_DATE" in df.columns:
            ft = df.dropna(subset=["TRN_DATE"]).copy()
            ft["TRN_DATE_ONLY"] = ft["TRN_DATE"].dt.date
            first_touch = ft.groupby(["STORE_NAME", "TRN_DATE_ONLY", "CUST_CODE"], as_index=False)["TRN_DATE"].min()
            first_touch["TIME_INTERVAL"] = first_touch["TRN_DATE"].dt.floor("30T")
            first_touch["TIME_ONLY"] = first_touch["TIME_INTERVAL"].dt.time
            counts = first_touch.groupby(["STORE_NAME", "TIME_ONLY"])["CUST_CODE"].nunique().reset_index(name="RECEIPT_COUNT")
            # pivot with full time grid
            start_time = pd.Timestamp("00:00:00")
            intervals = [(start_time + timedelta(minutes=30 * i)).time() for i in range(48)]
            heat = counts.pivot(index="STORE_NAME", columns="TIME_ONLY", values="RECEIPT_COUNT").fillna(0)
            for t in intervals:
                if t not in heat.columns:
                    heat[t] = 0
            heat = heat[intervals]
            results["customer_traffic_heat"] = heat
            results["customer_first_touch"] = first_touch
        else:
            results["customer_traffic_heat"] = pd.DataFrame()
    except Exception:
        results["customer_traffic_heat"] = pd.DataFrame()

    # OPERATIONS: Till usage / active tills
    try:
        df2 = df.dropna(subset=["TRN_DATE"]).copy()
        df2["TIME_INTERVAL"] = df2["TRN_DATE"].dt.floor("30T")
        df2["TIME_ONLY"] = df2["TIME_INTERVAL"].dt.time
        till_counts = df2.groupby(["STORE_NAME", "TIME_ONLY"])["Till_Code"].nunique().reset_index(name="UNIQUE_TILLS")
        heat_tills = till_counts.pivot(index="STORE_NAME", columns="TIME_ONLY", values="UNIQUE_TILLS").fillna(0)
        for t in results.get("customer_traffic_heat", pd.DataFrame()).columns:
            if t not in heat_tills.columns:
                heat_tills[t] = 0
        # ensure consistent column order if possible
        results["till_heat"] = heat_tills
    except Exception:
        results["till_heat"] = pd.DataFrame()

    # OPERATIONS: Cashier performance (receipt durations, customers served)
    try:
        df3 = df.dropna(subset=["TRN_DATE"]).copy()
        for c in ["STORE_CODE", "TILL", "SESSION", "RCT"]:
            if c in df3.columns:
                df3[c] = df3[c].astype(str).fillna("").str.strip()
        if "CUST_CODE" not in df3.columns and all(c in df3.columns for c in idcols):
            df3["CUST_CODE"] = df3["STORE_CODE"] + "-" + df3["TILL"] + "-" + df3["SESSION"] + "-" + df3["RCT"]
        receipt_duration = df3.groupby(["STORE_NAME", "CUST_CODE"], as_index=False).agg(Start_Time=("TRN_DATE", "min"), End_Time=("TRN_DATE", "max"))
        receipt_duration["Duration_Sec"] = (receipt_duration["End_Time"] - receipt_duration["Start_Time"]).dt.total_seconds().fillna(0)
        cashier_stats = df3.merge(receipt_duration[["STORE_NAME", "CUST_CODE", "Duration_Sec"]], on=["STORE_NAME", "CUST_CODE"], how="left")
        cashier_summary = cashier_stats.groupby(["STORE_NAME", "CASHIER"], as_index=False).agg(Hours_Worked=("Duration_Sec", lambda s: s.sum() / 3600.0), Customers_Served=("CUST_CODE", "nunique"))
        cashier_summary["Hours_Worked"] = cashier_summary["Hours_Worked"].round(2)
        cashier_summary["Customers_per_Hour"] = np.where(cashier_summary["Hours_Worked"] > 0, (cashier_summary["Customers_Served"] / cashier_summary["Hours_Worked"]).round(1), 0.0)
        results["cashier_summary"] = cashier_summary
    except Exception:
        results["cashier_summary"] = pd.DataFrame()

    # INSIGHTS: baskets and top items
    try:
        dfb = df.copy()
        dfb = df.dropna(subset=["ITEM_NAME", "CUST_CODE"])
        basket_count = df.groupby("ITEM_NAME")["CUST_CODE"].nunique().rename("Count_of_Baskets")
        agg = df.groupby("ITEM_NAME")[["QTY", "NET_SALES"]].sum()
        top_items = basket_count.to_frame().join(agg).reset_index().sort_values("Count_of_Baskets", ascending=False)
        results["top_items_global"] = top_items
    except Exception:
        results["top_items_global"] = pd.DataFrame()

    # INSIGHTS: pricing multi-price and refunds
    try:
        dprice = df.copy()
        dprice["DATE"] = dprice["TRN_DATE"].dt.date if "TRN_DATE" in dprice.columns else pd.NaT
        grp = dprice.groupby(["STORE_NAME", "DATE", "ITEM_CODE", "ITEM_NAME"], as_index=False).agg(
            Num_Prices=("SP_PRE_VAT", lambda s: s.dropna().nunique()),
            Price_Min=("SP_PRE_VAT", "min"),
            Price_Max=("SP_PRE_VAT", "max"),
            Total_QTY=("QTY", "sum"),
        )
        grp["Price_Spread"] = (grp["Price_Max"] - grp["Price_Min"]).round(2)
        multi_price = grp[(grp["Num_Prices"] > 1) & (grp["Price_Spread"] > 0)].copy()
        if not multi_price.empty:
            multi_price["Diff_Value"] = (multi_price["Total_QTY"] * multi_price["Price_Spread"]).round(2)
        results["multi_price"] = multi_price
    except Exception:
        results["multi_price"] = pd.DataFrame()

    try:
        dneg = df.copy()
        dneg["NET_SALES"] = pd.to_numeric(dneg["NET_SALES"], errors="coerce").fillna(0)
        neg = dneg[dneg["NET_SALES"] < 0].copy()
        if not neg.empty:
            group_cols = ["STORE_NAME", "CAP_CUSTOMER_CODE"]
            val_summ = neg.groupby(group_cols)["NET_SALES"].sum().rename("Total_Neg_Value")
            if "CUST_CODE" in neg.columns:
                cnt_summ = neg.groupby(group_cols)["CUST_CODE"].nunique().rename("Total_Count")
            else:
                cnt_summ = neg.groupby(group_cols).size().rename("Total_Count")
            summary_neg = pd.concat([val_summ, cnt_summ], axis=1).reset_index()
            summary_neg["Abs_Neg_Value"] = summary_neg["Total_Neg_Value"].abs()
            results["refunds_summary"] = summary_neg
        else:
            results["refunds_summary"] = pd.DataFrame()
    except Exception:
        results["refunds_summary"] = pd.DataFrame()

    # INSIGHTS: Loyalty
    try:
        dfL = df.copy()
        dfL["TRN_DATE"] = pd.to_datetime(dfL["TRN_DATE"], errors="coerce")
        dfL = dfL.dropna(subset=["TRN_DATE", "STORE_NAME", "CUST_CODE"])
        dfL["LOYALTY_CUSTOMER_CODE"] = dfL["LOYALTY_CUSTOMER_CODE"].astype(str).str.strip()
        dfL = dfL[dfL["LOYALTY_CUSTOMER_CODE"].replace({"nan": "", "NaN": "", "None": ""}).str.len() > 0]
        receipts = dfL.groupby(["STORE_NAME", "CUST_CODE", "LOYALTY_CUSTOMER_CODE"], as_index=False).agg(Basket_Value=("NET_SALES", "sum"), First_Time=("TRN_DATE", "min"))
        results["loyalty_receipts"] = receipts
    except Exception:
        results["loyalty_receipts"] = pd.DataFrame()

    # done
    return results

# -----------------------
# Sidebar upload + debug
# -----------------------
st.sidebar.header("Upload & Runtime")
st.sidebar.markdown("Upload CSV (server limit applies). Use S3 direct-upload if host blocks large uploads.")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Large files >200MB: prefer direct S3 upload or increase server.maxUploadSize")
st.sidebar.markdown("---")
st.sidebar.caption("Runtime debug:")
st.sidebar.write("STREAMLIT_SERVER_MAX_UPLOAD_SIZE (env):", os.environ.get("STREAMLIT_SERVER_MAX_UPLOAD_SIZE"))
try:
    st.sidebar.write("streamlit server.maxUploadSize (config):", st.config.get_option("server.maxUploadSize"))
except Exception:
    st.sidebar.write("streamlit server.maxUploadSize (config): unavailable")

# -----------------------
# Main UI layout
# -----------------------
st.title("Superdeck — Sales, Operations & Insights")
st.markdown("A clean, responsive UI that surfaces the notebook visuals. Select a main category below.")

# Large category cards (visual)
col1, col2, col3 = st.columns([1,1,1])
if "main_section" not in st.session_state:
    st.session_state["main_section"] = "SALES"
with col1:
    if st.button("📈 SALES", key="btn_sales"):
        st.session_state["main_section"] = "SALES"
    st.write("<div class='muted'>Global & per-store sales breakdowns</div>", unsafe_allow_html=True)
with col2:
    if st.button("⚙️ OPERATIONS", key="btn_ops"):
        st.session_state["main_section"] = "OPERATIONS"
    st.write("<div class='muted'>Traffic, tills, cashier & compliance</div>", unsafe_allow_html=True)
with col3:
    if st.button("🔎 INSIGHTS", key="btn_insights"):
        st.session_state["main_section"] = "INSIGHTS"
    st.write("<div class='muted'>Top items, pricing, loyalty, refunds</div>", unsafe_allow_html=True)

st.markdown("---")

# left: subsections; right: visual area
left_col, right_col = st.columns([1,3])

# subsections list for selected main
MAIN_SECTIONS = {
    "SALES": [
        "Global sales Overview",
        "Global Net Sales Distribution by Sales Channel",
        "Global Net Sales Distribution by SHIFT",
        "Night vs Day Shift Sales Ratio — Stores with Night Shifts",
        "Global Day vs Night Sales — Only Stores with NIGHT Shift",
        "2nd-Highest Channel Share",
        "Bottom 30 — 2nd Highest Channel",
        "Stores Sales Summary",
    ],
    "OPERATIONS": [
        "Customer Traffic-Storewise",
        "Active Tills During the day",
        "Average Customers Served per Till",
        "Store Customer Traffic Storewise",
        "Customer Traffic-Departmentwise",
        "Cashiers Perfomance",
        "Till Usage",
        "Tax Compliance",
    ],
    "INSIGHTS": [
        "Customer Baskets Overview",
        "Global Category Overview-Sales",
        "Global Category Overview-Baskets",
        "Supplier Contribution",
        "Category Overview",
        "Branch Comparison",
        "Product Perfomance",
        "Global Loyalty Overview",
        "Branch Loyalty Overview",
        "Customer Loyalty Overview",
        "Global Pricing Overview",
        "Branch Brach Overview",
        "Global Refunds Overview",
        "Branch Refunds Overview",
    ]
}

main_sel = st.session_state["main_section"]
subchoices = MAIN_SECTIONS.get(main_sel, [])
with left_col:
    st.header(f"{main_sel} — Subcategories")
    sub = st.radio("Select Subsection", subchoices, index=0, key="subsection_radio")

# -----------------------
# If no data loaded show hint
# -----------------------
if uploaded is None:
    with right_col:
        st.info("No CSV uploaded yet. Upload a CSV in the sidebar to enable the visuals. If files >200MB are blocked, use direct S3 upload method (see earlier guidance).")
        st.stop()

# -----------------------
# Run preprocessing (cached)
# -----------------------
with st.spinner("Preparing data in background (this can take a moment)..."):
    try:
        DATA = preprocess_all(uploaded)
        df = DATA.get("df", pd.DataFrame())
    except Exception as e:
        st.error(f"Data preparation failed: {e}")
        st.stop()

# -----------------------
# Helper: show missing columns message
# -----------------------
def check_cols(required: List[str]) -> Tuple[bool, List[str]]:
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0, missing)

# -----------------------
# Render subsection visuals
# -----------------------
with right_col:
    st.header(sub)
    # SALES
    if main_sel == "SALES":
        if sub == "Global sales Overview":
            gs = DATA.get("global_sales", pd.DataFrame())
            if gs.empty:
                st.warning("No SALES_CHANNEL_L1 / NET_SALES data available.")
            else:
                labels = [f"{r['SALES_CHANNEL_L1']} ({r['PCT']:.1f}% | {r['NET_SALES_M']:.1f}M)" for _, r in gs.iterrows()]
                fig = go.Figure(go.Pie(labels=labels, values=gs["NET_SALES_M"], hole=0.6, text=[f"{p:.1f}%" for p in gs["PCT"]], textinfo="text"))
                fig.update_layout(title="SALES CHANNEL TYPE — Global Overview", height=540)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(gs.rename(columns={"NET_SALES": "NET_SALES (KSh)", "NET_SALES_M": "NET_SALES (M)", "PCT":"Pct %"}), use_container_width=True)
                download_df = gs.copy()
                download_df_button(download_df, "global_sales_overview.csv", "⬇️ Download Table")

        elif sub == "Global Net Sales Distribution by Sales Channel":
            channel2 = DATA.get("channel2_sales", pd.DataFrame())
            if channel2.empty:
                st.warning("No SALES_CHANNEL_L2 data.")
            else:
                fig = px.pie(channel2, names="SALES_CHANNEL_L2", values="NET_SALES_M", hole=0.6, title="Net Sales by Sales Mode (L2)")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(channel2, use_container_width=True)
                download_df_button(channel2, "channel2_sales.csv", "⬇️ Download Table")

        elif sub == "Global Net Sales Distribution by SHIFT":
            sh = DATA.get("shift_sales", pd.DataFrame())
            if sh.empty:
                st.warning("No SHIFT data.")
            else:
                fig = px.pie(sh, names="SHIFT", values="NET_SALES", hole=0.6, title="Net Sales by Shift")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(sh, use_container_width=True)
                download_df_button(sh, "shift_sales.csv", "⬇️ Download Table")

        elif sub == "Night vs Day Shift Sales Ratio — Stores with Night Shifts":
            pivot = DATA.get("night_day_pivot", pd.DataFrame())
            if pivot.empty:
                st.info("No stores with NIGHT shift found or insufficient data.")
            else:
                pivot_sorted = pivot.sort_values("Night", ascending=False)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=pivot_sorted["Night"], y=pivot_sorted.index, orientation="h", name="Night", marker_color="#d62728"))
                fig.add_trace(go.Bar(x=pivot_sorted["Day"], y=pivot_sorted.index, orientation="h", name="Day", marker_color="#1f77b4"))
                fig.update_layout(barmode="group", title="Night vs Day % (by Store)", xaxis_title="% of Store Sales", height=700)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(pivot_sorted.reset_index().rename(columns={"Night":"Night %","Day":"Day %"}), use_container_width=True)

        elif sub == "Global Day vs Night Sales — Only Stores with NIGHT Shift":
            gnd = DATA.get("global_nd", pd.DataFrame())
            if gnd.empty:
                st.info("No NIGHT shift store totals.")
            else:
                gb = gnd.groupby("Shift_Bucket", as_index=False)["NET_SALES"].sum()
                gb["PCT"] = 100 * gb["NET_SALES"] / gb["NET_SALES"].sum()
                fig = px.pie(gb, names="Shift_Bucket", values="NET_SALES", hole=0.6, title="Global Day vs Night Sales (NIGHT Shift only)")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(gb, use_container_width=True)

        elif sub == "2nd-Highest Channel Share":
            second = DATA.get("second_channel_table", pd.DataFrame())
            if second.empty:
                st.info("No second-channel data.")
            else:
                top30 = second.sort_values("PCT", ascending=False).head(30)
                fig = px.bar(top30, x="PCT", y="STORE_NAME", orientation="h", title="Top 30 Stores by 2nd-Highest Channel %", text="PCT")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(top30.rename(columns={"PCT":"2nd Channel %"}), use_container_width=True)

        elif sub == "Bottom 30 — 2nd Highest Channel":
            second = DATA.get("second_channel_table", pd.DataFrame())
            if second.empty:
                st.info("No second-channel data.")
            else:
                bottom30 = second.sort_values("PCT", ascending=True).head(30)
                fig = px.bar(bottom30, x="PCT", y="STORE_NAME", orientation="h", title="Bottom 30 Stores by 2nd-Highest Channel %", text="PCT", color_discrete_sequence=["#d62728"])
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(bottom30.rename(columns={"PCT":"2nd Channel %"}), use_container_width=True)

        elif sub == "Stores Sales Summary":
            ss = DATA.get("stores_sales_summary", pd.DataFrame())
            if ss.empty:
                st.info("No stores sales summary.")
            else:
                st.dataframe(ss, use_container_width=True)
                download_df_button(ss, "stores_sales_summary.csv", "⬇️ Download Table")
                fig = px.bar(ss.sort_values("GROSS_SALES", ascending=True), x="GROSS_SALES", y="STORE_NAME", orientation="h", title="Gross Sales by Store")
                st.plotly_chart(fig, use_container_width=True)

    # OPERATIONS
    elif main_sel == "OPERATIONS":
        if sub == "Customer Traffic-Storewise":
            heat = DATA.get("customer_traffic_heat", pd.DataFrame())
            if heat.empty:
                st.info("Customer traffic heatmap not available (insufficient TRN_DATE/CUST_CODE data).")
            else:
                # choose a store
                store = st.selectbox("Store", options=sorted(heat.index.tolist()))
                if store:
                    row = heat.loc[store]
                    times = [t.strftime("%H:%M") if hasattr(t, "strftime") else str(t) for t in heat.columns]
                    fig = px.bar(x=times, y=row.values.astype(int), labels={"x":"Time","y":"Receipts"}, title=f"Receipts by Time - {store}")
                    st.plotly_chart(fig, use_container_width=True)
                    df_out = pd.DataFrame({"TIME": times, "Receipts": row.values.astype(int)})
                    st.dataframe(df_out, use_container_width=True)
                    download_df_button(df_out, f"{store}_traffic_by_time.csv", "⬇️ Download Table")

        elif sub == "Active Tills During the day":
            till_heat = DATA.get("till_heat", pd.DataFrame())
            if till_heat.empty:
                st.info("No till activity data available.")
            else:
                # show heatmap for all stores aggregated
                # reorder columns to time strings if time objects
                x_labels = [t.strftime("%H:%M") if hasattr(t, "strftime") else str(t) for t in till_heat.columns]
                fig = px.imshow(till_heat.values, x=x_labels, y=till_heat.index, text_auto=True, aspect="auto", title="Active Tills by 30-min interval")
                fig.update_xaxes(side="top")
                st.plotly_chart(fig, use_container_width=True)
                summary = till_heat.max(axis=1).rename("MAX_TILLS").reset_index()
                st.dataframe(summary.sort_values("MAX_TILLS", ascending=False), use_container_width=True)

        elif sub == "Average Customers Served per Till":
            # compute customers per till matrix
            try:
                cust_heat = DATA.get("customer_traffic_heat", pd.DataFrame())
                till_heat = DATA.get("till_heat", pd.DataFrame())
                if cust_heat.empty or till_heat.empty:
                    st.info("Insufficient data to compute customers per till.")
                else:
                    # align indices and columns
                    common_idx = cust_heat.index.intersection(till_heat.index)
                    cust = cust_heat.loc[common_idx]
                    till = till_heat.loc[common_idx]
                    ratio = cust.divide(till.replace(0, np.nan)).fillna(0)
                    x_labels = [t.strftime("%H:%M") if hasattr(t, "strftime") else str(t) for t in ratio.columns]
                    fig = px.imshow(ratio.values, x=x_labels, y=ratio.index, text_auto=True, aspect="auto", title="Customers per Till (30-min slots)")
                    fig.update_xaxes(side="top")
                    st.plotly_chart(fig, use_container_width=True)
                    # table top ratios
                    top = ratio.max(axis=1).sort_values(ascending=False).rename("MAX_RATIO").reset_index()
                    st.dataframe(top, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

        elif sub == "Store Customer Traffic Storewise":
            branch_data = {}
            if "customer_first_touch" not in DATA:
                st.info("No detailed customer touch data available.")
            else:
                first_touch = DATA["customer_first_touch"]
                stores = sorted(first_touch["STORE_NAME"].unique().tolist())
                store = st.selectbox("Store", stores)
                if store:
                    ft = first_touch[first_touch["STORE_NAME"] == store]
                    ft["TIME_SLOT"] = ft["TRN_DATE"].dt.floor("30T")
                    counts = ft.groupby("TIME_SLOT")["CUST_CODE"].nunique()
                    times = counts.index.strftime("%H:%M")
                    fig = px.bar(x=times, y=counts.values, labels={"x":"Time","y":"Receipts"}, title=f"Customer receipts by time — {store}")
                    st.plotly_chart(fig, use_container_width=True)
                    df_out = pd.DataFrame({"TIME": times, "Receipts": counts.values.astype(int)})
                    st.dataframe(df_out, use_container_width=True)
                    download_df_button(df_out, f"{store}_customer_traffic.csv", "⬇️ Download Table")

        elif sub == "Customer Traffic-Departmentwise":
            # Use build_branch_pivot logic: show department heatmap for selected store
            if "df" not in DATA or "TRN_DATE" not in DATA["df"].columns:
                st.info("Insufficient data.")
            else:
                df_full = DATA["df"].dropna(subset=["TRN_DATE"]).copy()
                stores = sorted(df_full["STORE_NAME"].unique().tolist())
                store = st.selectbox("Store for Department Traffic", stores)
                if store:
                    dfb = df_full[df_full["STORE_NAME"] == store].copy()
                    dfb["TIME_ONLY"] = dfb["TRN_DATE"].dt.floor("30T").dt.time
                    tmp = dfb.groupby(["DEPARTMENT", "TIME_ONLY"])["CUST_CODE"].nunique().reset_index(name="Unique_Customers")
                    if tmp.empty:
                        st.info("No department data for this store.")
                    else:
                        # pivot and plot
                        start_time = pd.Timestamp("00:00:00")
                        intervals = [(start_time + timedelta(minutes=30 * i)).time() for i in range(48)]
                        pivot = tmp.pivot(index="DEPARTMENT", columns="TIME_ONLY", values="Unique_Customers").fillna(0)
                        for t in intervals:
                            if t not in pivot.columns:
                                pivot[t] = 0
                        pivot = pivot[intervals]
                        x_labels = [t.strftime("%H:%M") for t in pivot.columns]
                        fig = px.imshow(pivot.values, x=x_labels, y=pivot.index, text_auto=True, aspect="auto", title=f"Department Traffic — {store}")
                        fig.update_xaxes(side="top")
                        st.plotly_chart(fig, use_container_width=True)
                        disp = pivot.sum(axis=1).rename("TOTAL").reset_index()
                        st.dataframe(disp.sort_values("TOTAL", ascending=False), use_container_width=True)

        elif sub == "Cashiers Perfomance":
            cs = DATA.get("cashier_summary", pd.DataFrame())
            if cs.empty:
                st.info("No cashier performance data available.")
            else:
                branches = sorted(cs["STORE_NAME"].unique().tolist())
                branch = st.selectbox("Branch", branches)
                if branch:
                    dfb = cs[cs["STORE_NAME"] == branch].copy()
                    fig = px.bar(dfb.sort_values("Customers_per_Hour", ascending=True), x="Customers_per_Hour", y="CASHIER", orientation="h", title=f"Customers per Hour — {branch}", color="Customers_per_Hour", color_continuous_scale="Blues")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(dfb.sort_values("Customers_per_Hour", ascending=False), use_container_width=True)

        elif sub == "Till Usage":
            # Branch summary from earlier code
            bs = DATA.get("till_heat", pd.DataFrame())
            if bs.empty:
                st.info("No till usage data.")
            else:
                # show Unique Tills per slot heat + summary
                # compute max tills per store
                max_tills = bs.max(axis=1).rename("MAX_TILLS").reset_index()
                st.dataframe(max_tills.sort_values("MAX_TILLS", ascending=False), use_container_width=True)
                download_df_button(max_tills, "till_usage_summary.csv", "⬇️ Download Table")

        elif sub == "Tax Compliance":
            d = DATA["df"].copy()
            if "CU_DEVICE_SERIAL" not in d.columns:
                st.info("CU_DEVICE_SERIAL column not present; tax compliance unavailable.")
            else:
                d["Tax_Compliant"] = np.where(d["CU_DEVICE_SERIAL"].astype(str).str.strip().replace({"nan": "", "None": ""}) != "", "Compliant", "Non-Compliant")
                global_summary = d.groupby("Tax_Compliant", as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"Receipts"})
                fig = px.pie(global_summary, names="Tax_Compliant", values="Receipts", hole=0.5, title="Global Tax Compliance Overview", color_discrete_map={"Compliant":"#2ca02c","Non-Compliant":"#d62728"})
                st.plotly_chart(fig, use_container_width=True)
                # store table
                store_till = d.groupby(["STORE_NAME", "Tax_Compliant"], as_index=False)["CUST_CODE"].nunique().reset_index(drop=True)
                pivot = store_till.pivot(index="STORE_NAME", columns="Tax_Compliant", values="CUST_CODE").fillna(0)
                if "Compliant" not in pivot.columns:
                    pivot["Compliant"] = 0
                if "Non-Compliant" not in pivot.columns:
                    pivot["Non-Compliant"] = 0
                pivot["Total"] = pivot["Compliant"] + pivot["Non-Compliant"]
                pivot["Compliance_%"] = np.where(pivot["Total"]>0, (100 * pivot["Compliant"] / pivot["Total"]).round(1), 0.0)
                st.dataframe(pivot.reset_index().sort_values("Total", ascending=False), use_container_width=True)

    # INSIGHTS
    elif main_sel == "INSIGHTS":
        if sub == "Customer Baskets Overview":
            top_items = DATA.get("top_items_global", pd.DataFrame())
            if top_items.empty:
                st.info("Top items data unavailable.")
            else:
                metric = st.selectbox("Metric", ["QTY", "NET_SALES", "Count_of_Baskets"], index=2)
                top_n = st.slider("Top N", 5, 100, 10)
                ti = top_items.copy().head(top_n)
                # show bar chart by metric
                if metric not in ti.columns:
                    # if Count_of_Baskets not present compute
                    if metric == "Count_of_Baskets" and "Count_of_Baskets" not in ti.columns:
                        tc = df.groupby("ITEM_NAME")["CUST_CODE"].nunique().rename("Count_of_Baskets").reset_index()
                        ti = tc.merge(df.groupby("ITEM_NAME")[["QTY","NET_SALES"]].sum().reset_index(), on="ITEM_NAME", how="left").sort_values(metric, ascending=False).head(top_n)
                fig = px.bar(ti, x=metric, y="ITEM_NAME", orientation="h", title=f"Top {top_n} items by {metric}")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(ti, use_container_width=True)

        elif sub == "Global Category Overview-Sales":
            if "CATEGORY" not in df.columns:
                st.info("CATEGORY not present.")
            else:
                cat = df.groupby("CATEGORY", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
                fig = px.bar(cat, x="NET_SALES", y="CATEGORY", orientation="h", title="Global Category Sales")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(cat, use_container_width=True)

        elif sub == "Global Category Overview-Baskets":
            if "CATEGORY" not in df.columns:
                st.info("CATEGORY not present.")
            else:
                cat = df.groupby("CATEGORY")["CUST_CODE"].nunique().rename("Unique_Baskets").reset_index().sort_values("Unique_Baskets", ascending=False)
                fig = px.bar(cat, x="Unique_Baskets", y="CATEGORY", orientation="h", title="Global Category — Unique Baskets")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(cat, use_container_width=True)

        elif sub == "Supplier Contribution":
            st.info("Supplier contribution visualization is not in the attached subset. If your dataset has a supplier column, we can add it — provide column name (e.g., SUPPLIER_NAME).")

        elif sub == "Category Overview":
            st.info("Category overview is available under Global Category Overview. Expand if you need more KPIs.")

        elif sub == "Branch Comparison":
            branches = sorted(df["STORE_NAME"].dropna().unique().tolist())
            if len(branches) < 2:
                st.info("Need at least 2 branches to compare.")
            else:
                colA, colB = st.columns(2)
                with colA:
                    A = st.selectbox("Branch A", branches, index=0)
                with colB:
                    B = st.selectbox("Branch B", branches, index=1)
                metric = st.selectbox("Metric", ["QTY", "NET_SALES"], index=1)
                top_n = st.slider("Top N items", 5, 100, 10, key="branch_topn")
                if st.button("Compare branches"):
                    dfA = df[df["STORE_NAME"]==A].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(top_n)
                    dfB = df[df["STORE_NAME"]==B].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(top_n)
                    comb = pd.concat([dfA.assign(Branch=A), dfB.assign(Branch=B)], ignore_index=True)
                    fig = px.bar(comb, x=metric, y="ITEM_NAME", color="Branch", orientation="h", barmode="group", title=f"Top {top_n} items: {A} vs {B}")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(comb, use_container_width=True)

        elif sub == "Product Perfomance":
            if "ITEM_CODE" not in df.columns:
                st.info("ITEM_CODE not found.")
            else:
                lookup = df[["ITEM_CODE","ITEM_NAME"]].drop_duplicates().sort_values("ITEM_NAME")
                opts = (lookup["ITEM_CODE"] + " — " + lookup["ITEM_NAME"]).tolist()
                sel = st.selectbox("Select SKU", opts)
                if sel:
                    code = sel.split("—")[0].strip()
                    item_df = df[df["ITEM_CODE"]==code]
                    if item_df.empty:
                        st.info("No data for selected SKU.")
                    else:
                        store_summary = item_df.groupby("STORE_NAME", as_index=False).agg(Baskets_With_Item=("CUST_CODE","nunique"), Total_QTY=("QTY","sum"))
                        store_total_customers = df.groupby("STORE_NAME")["CUST_CODE"].nunique()
                        store_summary["Store_Customers"] = store_summary["STORE_NAME"].map(store_total_customers).fillna(0).astype(int)
                        store_summary["Pct_of_Store_Customers"] = np.where(store_summary["Store_Customers"]>0, (100 * store_summary["Baskets_With_Item"] / store_summary["Store_Customers"]).round(1), 0.0)
                        st.dataframe(store_summary.sort_values("Baskets_With_Item", ascending=False), use_container_width=True)
                        fig = px.bar(store_summary, x="Baskets_With_Item", y="STORE_NAME", orientation="h", title=f"Baskets with {code} by Store")
                        st.plotly_chart(fig, use_container_width=True)

        elif sub == "Global Loyalty Overview":
            receipts = DATA.get("loyalty_receipts", pd.DataFrame())
            if receipts.empty:
                st.info("No loyalty data found (LOYALTY_CUSTOMER_CODE).")
            else:
                per_branch_multi = receipts.groupby(["STORE_NAME","LOYALTY_CUSTOMER_CODE"]).agg(Baskets_in_Store=("CUST_CODE","nunique"), Total_Value_in_Store=("Basket_Value","sum")).reset_index()
                per_branch_multi = per_branch_multi[per_branch_multi["Baskets_in_Store"]>1]
                overview = per_branch_multi.groupby("STORE_NAME", as_index=False).agg(Loyal_Customers_Multi=("LOYALTY_CUSTOMER_CODE","nunique"), Total_Baskets_of_Those=("Baskets_in_Store","sum"), Total_Value_of_Those=("Total_Value_in_Store","sum"))
                overview["Avg_Baskets_per_Customer"] = np.where(overview["Loyal_Customers_Multi"]>0, (overview["Total_Baskets_of_Those"]/overview["Loyal_Customers_Multi"]).round(2), 0.0)
                st.dataframe(overview.sort_values(["Loyal_Customers_Multi","Total_Baskets_of_Those"], ascending=False), use_container_width=True)

        elif sub == "Branch Loyalty Overview":
            receipts = DATA.get("loyalty_receipts", pd.DataFrame())
            if receipts.empty:
                st.info("No loyalty receipts.")
            else:
                branch = st.selectbox("Branch for loyalty overview", sorted(receipts["STORE_NAME"].unique()))
                per_store = receipts[receipts["STORE_NAME"]==branch].groupby("LOYALTY_CUSTOMER_CODE", as_index=False).agg(Baskets_in_Store=("CUST_CODE","nunique"), Total_Value_in_Store=("Basket_Value","sum"))
                per_store = per_store[per_store["Baskets_in_Store"]>1].sort_values(["Baskets_in_Store","Total_Value_in_Store"], ascending=False)
                st.dataframe(per_store, use_container_width=True)

        elif sub == "Customer Loyalty Overview":
            receipts = DATA.get("loyalty_receipts", pd.DataFrame())
            if receipts.empty:
                st.info("No loyalty receipts.")
            else:
                st.dataframe(receipts.head(200), use_container_width=True)

        elif sub == "Global Pricing Overview":
            multi_price = DATA.get("multi_price", pd.DataFrame())
            if multi_price.empty:
                st.info("No multi-priced SKUs detected.")
            else:
                summary = multi_price.groupby("STORE_NAME", as_index=False).agg(Items_with_MultiPrice=("ITEM_CODE","nunique"), Total_Diff_Value=("Diff_Value","sum"), Avg_Spread=("Price_Spread","mean"), Max_Spread=("Price_Spread","max"))
                st.dataframe(summary.sort_values("Total_Diff_Value", ascending=False), use_container_width=True)
                fig = px.bar(summary.sort_values("Total_Diff_Value"), x="Total_Diff_Value", y="STORE_NAME", orientation="h", title="Stores by pricing spread impact")
                st.plotly_chart(fig, use_container_width=True)

        elif sub == "Branch Brach Overview":
            st.info("This subsection name appears duplicated/typoed (Branch Brach Overview). If you mean Branch Pricing Overview or Branch Refunds Overview, choose the correct subsection.")

        elif sub == "Global Refunds Overview":
            refunds = DATA.get("refunds_summary", pd.DataFrame())
            if refunds.empty:
                st.info("No negative receipts found.")
            else:
                st.dataframe(refunds.sort_values("Abs_Neg_Value", ascending=False).head(200), use_container_width=True)

        elif sub == "Branch Refunds Overview":
            refunds = DATA.get("refunds_summary", pd.DataFrame())
            if refunds.empty:
                st.info("No negative receipts.")
            else:
                branch = st.selectbox("Branch", sorted(refunds["STORE_NAME"].unique()))
                st.dataframe(refunds[refunds["STORE_NAME"]==branch].sort_values("Abs_Neg_Value", ascending=False), use_container_width=True)

    # End of all subsections
    st.markdown("---")
    st.caption("Note: If a visualization says data missing, check that the uploaded CSV contains the named columns used by that view (the Colab notebook used TRN_DATE, NET_SALES, STORE_NAME, CUST_CODE, TILL, etc.).")
