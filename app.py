
import os, time
import duckdb
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="DailyDeck — Minimal Sidebar", page_icon="📊", layout="wide")

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
SECTIONS = [s for s in [ln.strip() for ln in RAW_SECTIONS.splitlines()] if s and not s.startswith('*')]
GROUPS = ['SALES','OPERATIONS']
SECTION_TITLES = [s for s in SECTIONS if s not in GROUPS]

# ---------------- Sidebar (ONLY upload + titles) ----------------
with st.sidebar:
    st.header("📤 Upload CSV")
    csv_file = st.file_uploader("CSV (up to ~500MB)", type=["csv"], label_visibility="collapsed")
    st.markdown("---")
    chosen = st.radio("Sections", options=SECTION_TITLES, index=0)

# ---------------- Data loading via DuckDB ----------------
if 'csv_path' not in st.session_state:
    st.session_state['csv_path'] = None
if 'con' not in st.session_state:
    st.session_state['con'] = duckdb.connect(database=':memory:')
if csv_file is not None:
    tmp_path = os.path.join('/tmp', 'upload_' + str(int(time.time())) + '.csv')
    with open(tmp_path, 'wb') as f:
        f.write(csv_file.getvalue())
    st.session_state['csv_path'] = tmp_path
con = st.session_state['con']

def ensure_view():
    con.execute("DROP VIEW IF EXISTS v;")
    con.execute("CREATE VIEW v AS SELECT * FROM read_csv_auto('" + st.session_state['csv_path'] + "', HEADER=TRUE, SAMPLE_SIZE=2000000);")

# ---------------- Main: Controls (collapsed) ----------------
st.markdown("## " + chosen)
if not st.session_state['csv_path']:
    st.info("Upload a CSV on the left to activate controls and outputs.")
    st.stop()

ensure_view()
cols = con.execute("PRAGMA table_info('v')").df()['name'].tolist()

# Heuristic defaults
def pick(names):
    for n in names:
        if n in cols:
            return n
    return None

default_map = dict(
    date = pick(['TRN_DATE','DATE','TXN_DATE','DATETIME']),
    store = pick(['STORE_NAME','STORE','BRANCH','STORE_CODE']),
    channel = pick(['CHANNEL','SALES_CHANNEL','MODE','TENDER_TYPE']),
    shift = pick(['SHIFT']),
    dept = pick(['DEPARTMENT','DEPT']),
    cat = pick(['CATEGORY','CAT']),
    supplier = pick(['SUPPLIER','VENDOR']),
    till = pick(['TILL','REGISTER','POS']),
    cashier = pick(['CASHIER','USER']),
    cust = pick(['CUST_CODE','RECEIPT_ID','DOC_NO','RCT']),
    sales = pick(['SALES_PRE_VAT','NET_SALES','SP_PRE_VAT','AMOUNT']),
    gp = pick(['GROSS_PROFIT','GP']),
    qty = pick(['QUANTITY','QTY']),
    price = pick(['AVG_SP_PRE_VAT','UNIT_PRICE','PRICE','SP_PRE_VAT']),
    tax = pick(['CU_DEVICE_SERIAL','TAX_FLAG'])
)

with st.expander("Controls", expanded=False):
    row_cap = st.number_input("Row cap (DuckDB LIMIT)", 50000, 5000000, 1000000, 50000)
    c1,c2,c3 = st.columns(3)
    with c1:
        date_col = st.selectbox("Date", ["(none)"]+cols, index=(cols.index(default_map['date'])+1 if default_map['date'] in cols else 0))
        store_col = st.selectbox("Store", ["(none)"]+cols, index=(cols.index(default_map['store'])+1 if default_map['store'] in cols else 0))
        channel_col = st.selectbox("Channel", ["(none)"]+cols, index=(cols.index(default_map['channel'])+1 if default_map['channel'] in cols else 0))
    with c2:
        shift_col = st.selectbox("Shift (or derive from Date)", ["(derive)"]+cols, index=(cols.index(default_map['shift'])+1 if default_map['shift'] in cols else 0))
        dept_col = st.selectbox("Department", ["(none)"]+cols, index=(cols.index(default_map['dept'])+1 if default_map['dept'] in cols else 0))
        cat_col = st.selectbox("Category", ["(none)"]+cols, index=(cols.index(default_map['cat'])+1 if default_map['cat'] in cols else 0))
    with c3:
        supplier_col = st.selectbox("Supplier", ["(none)"]+cols, index=(cols.index(default_map['supplier'])+1 if default_map['supplier'] in cols else 0))
        cust_col = st.selectbox("Receipt/Customer ID", ["(none)"]+cols, index=(cols.index(default_map['cust'])+1 if default_map['cust'] in cols else 0))
        sales_col = st.selectbox("Net Sales", ["(none)"]+cols, index=(cols.index(default_map['sales'])+1 if default_map['sales'] in cols else 0))

    st.caption("Quick filters")
    f1,f2,f3 = st.columns(3)
    with f1:
        store_filter = st.text_input("Store IN (comma-separated)", value="")
    with f2:
        dept_filter = st.text_input("Department IN", value="")
    with f3:
        cat_filter = st.text_input("Category IN", value="")

def sql_in(colname, csv_vals):
    vals = [v.strip() for v in csv_vals.split(',') if v.strip()]
    if not vals: return ""
    quoted = ",".join(["'" + v.replace("'", "''") + "'" for v in vals])
    return colname + " IN (" + quoted + ")"

where_parts = []
if 'store_filter' in locals() and store_filter and store_col != "(none)": where_parts.append(sql_in(store_col, store_filter))
if 'dept_filter' in locals() and dept_filter and dept_col != "(none)": where_parts.append(sql_in(dept_col, dept_filter))
if 'cat_filter' in locals() and cat_filter and cat_col != "(none)": where_parts.append(sql_in(cat_col, cat_filter))
WHERE = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

def shift_expr():
    if 'shift_col' in locals() and shift_col != "(derive)":
        return shift_col
    if 'date_col' in locals() and date_col != "(none)":
        return "CASE WHEN EXTRACT(HOUR FROM CAST(" + date_col + " AS TIMESTAMP)) BETWEEN 6 AND 17 THEN 'DAY' ELSE 'NIGHT' END"
    return None

def q(sql):
    return con.execute(sql).df()

def limited(select, groupby=None, orderby=None):
    gb = (" GROUP BY " + groupby) if groupby else ""
    ob = (" ORDER BY " + orderby) if orderby else ""
    return q("SELECT " + select + " FROM v" + WHERE + gb + ob + " LIMIT " + str(row_cap) + ";")

def chart(df, kind, x, y):
    if df is None or df.empty:
        st.warning("No data for current selection.")
        return
    if kind=='bar':
        fig = px.bar(df, x=x, y=y)
    elif kind=='line':
        fig = px.line(df, x=x, y=y)
    elif kind=='pie':
        fig = px.pie(df, names=x, values=y)
    else:
        fig = px.bar(df, x=x, y=y)
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download data", df.to_csv(index=False).encode('utf-8'), file_name="section.csv", use_container_width=True)

# ---------------- Section logic ----------------
s = chosen

if s == "Global sales Overview":
    if sales_col == "(none)":
        st.error("Map Net Sales in Controls.")
    else:
        df = limited("SUM(CAST(" + sales_col + " AS DOUBLE)) AS net_sales, COUNT(*) AS rows")
        st.metric("Net Sales", f"{'{'}df['net_sales'].iloc[0]{'}'}:,.0f}")
        st.metric("Rows", f"{'{'}df['rows'].iloc[0]{'}'}:,}")

elif s == "Global Net Sales Distribution by Sales Channel":
    if channel_col == "(none)" or sales_col == "(none)":
        st.error("Map Channel and Net Sales in Controls.")
    else:
        df = limited(channel_col + " AS grp, SUM(CAST(" + sales_col + " AS DOUBLE)) AS net_sales", groupby="grp", orderby="net_sales DESC")
        chart(df, "bar", "grp", "net_sales")

elif s == "Global Net Sales Distribution by SHIFT":
    se = shift_expr()
    if not se or sales_col == "(none)":
        st.error("Need Shift (or Date to derive) and Net Sales.")
    else:
        df = limited("(" + se + ") AS grp, SUM(CAST(" + sales_col + " AS DOUBLE)) AS net_sales", groupby="grp", orderby="net_sales DESC")
        chart(df, "pie", "grp", "net_sales")

elif s == "Night vs Day Shift Sales Ratio — Stores with Night Shifts":
    se = shift_expr()
    if not se or store_col == "(none)" or sales_col == "(none)":
        st.error("Map Store, Shift/Date, Net Sales.")
    else:
        base = q("SELECT " + store_col + " AS store, CASE WHEN (" + se + ")='NIGHT' THEN 1 ELSE 0 END AS is_night, CAST(" + sales_col + " AS DOUBLE) AS ns FROM v" + WHERE + " LIMIT " + str(row_cap) + ";")
        if base.empty:
            st.warning("No data.")
        else:
            night_stores = base.groupby('store')['is_night'].sum()
            have_night = night_stores[night_stores>0].index.tolist()
            filt = base[base['store'].isin(have_night)]
            agg = filt.groupby(['store', filt['is_night'].map({1:'NIGHT',0:'DAY'})])['ns'].sum().reset_index().rename(columns={'is_night':'shift','ns':'net_sales'})
            chart(agg, 'bar', 'store', 'net_sales')

elif s == "Global Day vs Night Sales — Only Stores with NIGHT Shift":
    se = shift_expr()
    if not se or store_col == "(none)" or sales_col == "(none)":
        st.error("Map Store, Shift/Date, Net Sales.")
    else:
        df = limited(store_col + " AS store, (" + se + ") AS shift, SUM(CAST(" + sales_col + " AS DOUBLE)) AS net_sales",
                     groupby="store, shift", orderby="store, shift")
        have_night = df[df['shift']=='NIGHT']['store'].unique().tolist()
        df = df[df['store'].isin(have_night)]
        chart(df, 'bar', 'store', 'net_sales')

elif s == "2nd-Highest Channel Share":
    if channel_col == "(none)" or sales_col == "(none)":
        st.error("Map Channel and Net Sales.")
    else:
        df = limited(channel_col + " AS ch, SUM(CAST(" + sales_col + " AS DOUBLE)) AS ns", groupby="ch", orderby="ns DESC")
        if len(df)>=2:
            chart(df.iloc[1:2], 'bar', 'ch', 'ns')
        else:
            st.info("Less than two channels found.")

elif s == "Bottom 30 — 2nd Highest Channel":
    if store_col == "(none)" or channel_col == "(none)" or sales_col == "(none)":
        st.error("Map Store, Channel, Net Sales.")
    else:
        df = limited(store_col + " AS store, " + channel_col + " AS ch, SUM(CAST(" + sales_col + " AS DOUBLE)) AS ns",
                     groupby="store, ch", orderby="store, ns DESC")
        second = df.sort_values(['store','ns'], ascending=[True, False]).groupby('store').nth(1).reset_index()
        chart(second.sort_values('ns').head(30), 'bar', 'store', 'ns')

elif s == "Stores Sales Summary":
    if store_col == "(none)" or sales_col == "(none)":
        st.error("Map Store and Net Sales.")
    else:
        df = limited(store_col + " AS store, SUM(CAST(" + sales_col + " AS DOUBLE)) AS net_sales, COUNT(*) AS txns",
                     groupby="store", orderby="net_sales DESC")
        chart(df, 'bar', 'store', 'net_sales')

elif s == "Customer Traffic-Storewise":
    if store_col == "(none)" or cust_col == "(none)":
        st.error("Map Store and Receipt/Customer ID.")
    else:
        df = limited(store_col + " AS store, COUNT(DISTINCT " + cust_col + ") AS baskets",
                     groupby="store", orderby="baskets DESC")
        chart(df, 'bar', 'store', 'baskets')

elif s == "Active Tills During the day":
    if date_col == "(none)":
        st.error("Map Date.")
    elif till_col == "(none)":
        st.error("Map Till.")
    else:
        df = limited("DATE_TRUNC('hour', CAST(" + date_col + " AS TIMESTAMP)) AS hour, COUNT(DISTINCT " + till_col + ") AS active_tills",
                     groupby="hour", orderby="hour")
        chart(df, 'line', 'hour', 'active_tills')

elif s == "Average Customers Served per Till":
    if till_col == "(none)" or cust_col == "(none)":
        st.error("Map Till and Receipt/Customer ID.")
    else:
        df = limited(till_col + " AS till, COUNT(DISTINCT " + cust_col + ")*1.0 / NULLIF(COUNT(DISTINCT " + till_col + "),0) AS avg_cust_per_till",
                     groupby="till", orderby="avg_cust_per_till DESC")
        chart(df, 'bar', 'till', 'avg_cust_per_till')

elif s == "Store Customer Traffic Storewise":
    if store_col == "(none)" or cust_col == "(none)":
        st.error("Map Store and Receipt/Customer ID.")
    else:
        df = limited(store_col + " AS store, COUNT(DISTINCT " + cust_col + ") AS baskets",
                     groupby="store", orderby="baskets DESC")
        chart(df, 'bar', 'store', 'baskets')

elif s == "Customer Traffic-Departmentwise":
    if dept_col == "(none)" or cust_col == "(none)":
        st.error("Map Department and Receipt/Customer ID.")
    else:
        df = limited(dept_col + " AS dept, COUNT(DISTINCT " + cust_col + ") AS baskets",
                     groupby="dept", orderby="baskets DESC")
        chart(df, 'bar', 'dept', 'baskets')

elif s == "Cashiers Perfomance":
    if cashier_col == "(none)" or sales_col == "(none)":
        st.error("Map Cashier and Net Sales.")
    else:
        df = limited(cashier_col + " AS cashier, SUM(CAST(" + sales_col + " AS DOUBLE)) AS net_sales, COUNT(*) AS txns",
                     groupby="cashier", orderby="net_sales DESC")
        chart(df, 'bar', 'cashier', 'net_sales')

elif s == "Till Usage":
    if till_col == "(none)":
        st.error("Map Till.")
    else:
        df = limited(till_col + " AS till, COUNT(*) AS uses",
                     groupby="till", orderby="uses DESC")
        chart(df, 'bar', 'till', 'uses')

elif s == "Tax Compliance":
    if tax_col == "(none)":
        st.error("Map Tax flag.")
    else:
        df = limited("CASE WHEN " + tax_col + " IS NULL OR " + tax_col + "='' THEN 'NON_COMPLIANT' ELSE 'COMPLIANT' END AS status, COUNT(*) AS txns",
                     groupby="status", orderby="txns DESC")
        chart(df, 'pie', 'status', 'txns')

elif s == "Customer Baskets Overview":
    if cust_col == "(none)" or sales_col == "(none)":
        st.error("Map Receipt/Customer ID and Net Sales.")
    else:
        df = limited(cust_col + " AS basket, SUM(CAST(" + sales_col + " AS DOUBLE)) AS basket_value",
                     groupby="basket", orderby="basket_value DESC")
        st.write(df.describe())
        chart(df.head(100), 'bar', 'basket', 'basket_value')

elif s == "Global Category Overview-Sales":
    if cat_col == "(none)" or sales_col == "(none)":
        st.error("Map Category and Net Sales.")
    else:
        df = limited(cat_col + " AS cat, SUM(CAST(" + sales_col + " AS DOUBLE)) AS net_sales",
                     groupby="cat", orderby="net_sales DESC")
        chart(df, 'bar', 'cat', 'net_sales')

elif s == "Global Category Overview-Baskets":
    if cat_col == "(none)" or cust_col == "(none)":
        st.error("Map Category and Receipt/Customer ID.")
    else:
        df = limited(cat_col + " AS cat, COUNT(DISTINCT " + cust_col + ") AS baskets",
                     groupby="cat", orderby="baskets DESC")
        chart(df, 'bar', 'cat', 'baskets')

elif s == "Supplier Contribution":
    if supplier_col == "(none)" or sales_col == "(none)":
        st.error("Map Supplier and Net Sales.")
    else:
        df = limited(supplier_col + " AS supplier, SUM(CAST(" + sales_col + " AS DOUBLE)) AS net_sales",
                     groupby="supplier", orderby="net_sales DESC")
        chart(df.head(50), 'bar', 'supplier', 'net_sales')

elif s == "Category Overview":
    if dept_col == "(none)" or cat_col == "(none)" or sales_col == "(none)":
        st.error("Map Department, Category, Net Sales.")
    else:
        df = limited(dept_col + " AS dept, " + cat_col + " AS cat, SUM(CAST(" + sales_col + " AS DOUBLE)) AS net_sales",
                     groupby="dept, cat", orderby="net_sales DESC")
        chart(df, 'bar', 'cat', 'net_sales')

elif s == "Branch Comparison":
    if store_col == "(none)" or sales_col == "(none)":
        st.error("Map Store and Net Sales.")
    else:
        df = limited(store_col + " AS store, SUM(CAST(" + sales_col + " AS DOUBLE)) AS net_sales",
                     groupby="store", orderby="net_sales DESC")
        chart(df, 'bar', 'store', 'net_sales')

elif s == "Product Perfomance":
    item_code = next((c for c in cols if c.upper() in ["ITEM_CODE","SKU","PRODUCT_CODE"]), None)
    item_name = next((c for c in cols if c.upper() in ["ITEM_NAME","PRODUCT","SKU_NAME"]), None)
    if (item_code or item_name) and sales_col != "(none)":
        df = limited("COALESCE(" + (item_name or item_code) + ", " + (item_code or item_name) + ") AS item, SUM(CAST(" + sales_col + " AS DOUBLE)) AS net_sales",
                     groupby="item", orderby="net_sales DESC")
        chart(df.head(50), 'bar', 'item', 'net_sales')
    else:
        st.error("Need ITEM_NAME/ITEM_CODE and Net Sales.")

elif s == "Global Loyalty Overview":
    loy = next((c for c in cols if "LOYAL" in c.upper()), None)
    if not loy:
        st.error("No loyalty column detected (contains 'LOYAL').")
    else:
        df = limited("CASE WHEN " + loy + " IN ('1','Y','YES',1,TRUE) THEN 'LOYAL' ELSE 'NON-LOYAL' END AS segment, COUNT(*) AS txns",
                     groupby="segment", orderby="txns DESC")
        chart(df, 'pie', 'segment', 'txns')

elif s == "Branch Loyalty Overview":
    loy = next((c for c in cols if "LOYAL" in c.upper()), None)
    if not loy or store_col == "(none)":
        st.error("Need loyalty column and Store.")
    else:
        df = limited(store_col + " AS store, CASE WHEN " + loy + " IN ('1','Y','YES',1,TRUE) THEN 'LOYAL' ELSE 'NON-LOYAL' END AS segment, COUNT(*) AS txns",
                     groupby="store, segment", orderby="store, segment")
        chart(df, 'bar', 'store', 'txns')

elif s == "Customer Loyalty Overview":
    loy = next((c for c in cols if "LOYAL" in c.upper()), None)
    if not loy or cust_col == "(none)":
        st.error("Need loyalty column and Receipt/Customer ID.")
    else:
        df = limited(cust_col + " AS cust, CASE WHEN " + loy + " IN ('1','Y','YES',1,TRUE) THEN 'LOYAL' ELSE 'NON-LOYAL' END AS segment, COUNT(*) AS txns",
                     groupby="cust, segment", orderby="txns DESC")
        chart(df.head(200), 'bar', 'cust', 'txns')

elif s == "Global Pricing Overview":
    price_col = next((c for c in cols if c.upper() in ["AVG_SP_PRE_VAT","UNIT_PRICE","PRICE","SP_PRE_VAT"]), None)
    if not price_col:
        st.error("No price column detected.")
    else:
        df = limited("ROUND(CAST(" + price_col + " AS DOUBLE),0) AS price_bucket, COUNT(*) AS rows",
                     groupby="price_bucket", orderby="price_bucket")
        chart(df, 'line', 'price_bucket', 'rows')

elif s == "Branch Brach Overview":
    if store_col == "(none)" or dept_col == "(none)" or sales_col == "(none)":
        st.error("Map Store, Department, Net Sales.")
    else:
        df = limited(store_col + " AS store, " + dept_col + " AS dept, SUM(CAST(" + sales_col + " AS DOUBLE)) AS net_sales",
                     groupby="store, dept", orderby="store, net_sales DESC")
        chart(df, 'bar', 'dept', 'net_sales')

elif s == "Global Refunds Overview":
    ref = next((c for c in cols if "REFUND" in c.upper() or "VOID" in c.upper()), None)
    if not ref:
        st.error("No refund/void indicator detected.")
    else:
        df = limited("CASE WHEN " + ref + " IN ('1','Y','YES',1,TRUE) THEN 'REFUND/VOID' ELSE 'NORMAL' END AS txntype, COUNT(*) AS txns",
                     groupby="txntype", orderby="txns DESC")
        chart(df, 'pie', 'txntype', 'txns')

elif s == "Branch Refunds Overview":
    ref = next((c for c in cols if "REFUND" in c.upper() or "VOID" in c.upper()), None)
    if not ref or store_col == "(none)":
        st.error("Need refund/void indicator and Store.")
    else:
        df = limited(store_col + " AS store, CASE WHEN " + ref + " IN ('1','Y','YES',1,TRUE) THEN 'REFUND/VOID' ELSE 'NORMAL' END AS txntype, COUNT(*) AS txns",
                     groupby="store, txntype", orderby="store, txns DESC")
        chart(df, 'bar', 'store', 'txns')

else:
    st.info("Section not recognized.")
