"""
Superdeck — Streamlit edition (complete, cleaned)
- Full app: SALES, OPERATIONS, INSIGHTS
- Consistent 30-minute bucketing for all time-slot views
- Rows ranked by their row-max (so top activity rows appear first)
- Sidebar Top-N control and other small UI niceties
- Cached expensive parsing/derivations for responsiveness
Run:
    streamlit run app.py
"""
from io import BytesIO
from datetime import timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Superdeck (Streamlit)")

# -----------------------
# Helpers
# -----------------------
def get_30min_intervals():
    """Return list of datetime.time objects from 00:00 to 23:30 at 30-min resolution."""
    return pd.date_range("00:00", "23:30", freq="30min").time.tolist()

def ensure_time_indexed_pivot(pivot: pd.DataFrame):
    """Reindex pivot columns to canonical 30-min slots and fill missing with 0."""
    full_intervals = get_30min_intervals()
    # If pivot has datetime.time columns already, reindex directly; else try to coerce
    return pivot.reindex(columns=full_intervals, fill_value=0)

# -----------------------
# Data Loading & Caching
# -----------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines='skip', low_memory=False)

@st.cache_data
def load_uploaded_file(contents: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(contents), on_bad_lines='skip', low_memory=False)

def smart_load():
    st.sidebar.markdown("### Upload data (CSV) or use default")
    uploaded = st.sidebar.file_uploader("Upload DAILY_POS_TRN_ITEMS CSV", type=['csv'])
    if uploaded is not None:
        with st.spinner("Parsing uploaded CSV..."):
            df = load_uploaded_file(uploaded.getvalue())
        st.sidebar.success("Loaded uploaded CSV")
        return df

    # Optional default path (useful in notebooks). If not present, prompt upload.
    default_path = "/content/DAILY_POS_TRN_ITEMS_2025-10-21.csv"
    try:
        with st.spinner(f"Loading default CSV: {default_path}"):
            df = load_csv(default_path)
        st.sidebar.info(f"Loaded default path: {default_path}")
        return df
    except Exception:
        st.sidebar.warning("No default CSV found. Please upload a CSV to run the app.")
        return None

# -----------------------
# Global UI controls
# -----------------------
TOP_N = st.sidebar.slider("Top N for lists", min_value=5, max_value=200, value=30, step=5)
CHART_ROW_HEIGHT = 28  # px per row for heatmap height calculation

# -----------------------
# Cleaning & Derivations (cached)
# -----------------------
@st.cache_data
def clean_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-time cached cleaning:
    - Normalize strings
    - Parse TRN_DATE and create 30-min buckets (TIME_INTERVAL, TIME_ONLY)
    - Remove commas and coerce numeric columns
    - Build composite keys (CUST_CODE, Till_Code, CASHIER-COUNT)
    - Build Shift_Bucket
    """
    if df is None:
        return df
    d = df.copy()

    # Normalize string columns
    str_cols = ['STORE_CODE','TILL','SESSION','RCT','STORE_NAME','CASHIER','ITEM_CODE',
                'ITEM_NAME','DEPARTMENT','CATEGORY','CU_DEVICE_SERIAL','CAP_CUSTOMER_CODE',
                'LOYALTY_CUSTOMER_CODE','SUPPLIER_NAME','SALES_CHANNEL_L1','SALES_CHANNEL_L2','SHIFT']
    for c in str_cols:
        if c in d.columns:
            d[c] = d[c].fillna('').astype(str).str.strip()

    # Dates: parse once
    if 'TRN_DATE' in d.columns:
        d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
        d = d.dropna(subset=['TRN_DATE']).copy()
        d['DATE'] = d['TRN_DATE'].dt.date
        d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30T')     # canonical 30-min floor
        d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time           # datetime.time objects

    if 'ZED_DATE' in d.columns:
        d['ZED_DATE'] = pd.to_datetime(d['ZED_DATE'], errors='coerce')

    # Numeric parsing (remove commas)
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(',', '', regex=False).str.strip(), errors='coerce').fillna(0)

    # GROSS_SALES
    if 'GROSS_SALES' not in d.columns:
        d['GROSS_SALES'] = d.get('NET_SALES', 0) + d.get('VAT_AMT', 0)

    # Composite keys
    if all(col in d.columns for col in ['STORE_CODE','TILL','SESSION','RCT']):
        d['CUST_CODE'] = d['STORE_CODE'].astype(str) + '-' + d['TILL'].astype(str) + '-' + d['SESSION'].astype(str) + '-' + d['RCT'].astype(str)
    else:
        if 'CUST_CODE' not in d.columns:
            d['CUST_CODE'] = ''

    if 'TILL' in d.columns and 'STORE_CODE' in d.columns:
        d['Till_Code'] = d['TILL'].astype(str) + '-' + d['STORE_CODE'].astype(str)

    if 'STORE_NAME' in d.columns and 'CASHIER' in d.columns:
        d['CASHIER-COUNT'] = d['CASHIER'].astype(str) + '-' + d['STORE_NAME'].astype(str)

    # Shift bucket
    if 'SHIFT' in d.columns:
        d['Shift_Bucket'] = np.where(d['SHIFT'].str.upper().str.contains('NIGHT', na=False), 'Night', 'Day')

    # Force numeric types for safety
    if 'SP_PRE_VAT' in d.columns:
        d['SP_PRE_VAT'] = d['SP_PRE_VAT'].astype(float)
    if 'NET_SALES' in d.columns:
        d['NET_SALES'] = d['NET_SALES'].astype(float)

    return d

# -----------------------
# Display & formatting helpers
# -----------------------
def format_and_display(df: pd.DataFrame, numeric_cols: list | None = None, index_col: str | None = None, total_label: str = 'TOTAL'):
    """
    Append totals row and pretty-format numeric columns (commas; 2 decimals for floats).
    """
    if df is None or df.empty:
        st.dataframe(df)
        return

    df_display = df.copy()

    # Detect numeric columns if not provided
    if numeric_cols is None:
        numeric_cols = list(df_display.select_dtypes(include=[np.number]).columns)

    totals = {}
    for col in df_display.columns:
        if col in numeric_cols:
            try:
                totals[col] = df_display[col].astype(float).sum()
            except Exception:
                totals[col] = ''
        else:
            totals[col] = ''

    # Choose label column
    if index_col and index_col in df_display.columns:
        label_col = index_col
    else:
        non_numeric_cols = [c for c in df_display.columns if c not in numeric_cols]
        label_col = non_numeric_cols[0] if non_numeric_cols else df_display.columns[0]

    totals[label_col] = total_label
    tot_df = pd.DataFrame([totals], columns=df_display.columns)
    appended = pd.concat([df_display, tot_df], ignore_index=True)

    # Formatting numeric columns
    for col in numeric_cols:
        if col in appended.columns:
            vals = appended[col].dropna().astype(float)
            is_int_like = False
            if len(vals) > 0:
                is_int_like = np.allclose(vals.fillna(0).round(0), vals.fillna(0))
            if is_int_like:
                appended[col] = appended[col].map(lambda v: f"{int(v):,}" if pd.notna(v) and str(v) != '' else '')
            else:
                appended[col] = appended[col].map(lambda v: f"{v:,.2f}" if pd.notna(v) and str(v) != '' else '')

    st.dataframe(appended, use_container_width=True)

def donut_from_agg(df_agg, label_col, value_col, title, hole=0.55, value_is_millions=False):
    labels = df_agg[label_col].astype(str).tolist()
    vals = df_agg[value_col].astype(float).tolist()
    if value_is_millions:
        vals_plot = [v/1_000_000 for v in vals]
        hover = 'KSh %{value:,.2f} M'
    else:
        vals_plot = vals
        hover = 'KSh %{value:,.2f}'
    s = sum(vals) if sum(vals) != 0 else 1
    legend_labels = [f"{lab} ({100*val/s:.1f}%)" for lab,val in zip(labels, vals)]
    fig = go.Figure(data=[go.Pie(labels=legend_labels, values=vals_plot, hole=hole,
                                 hovertemplate='<b>%{label}</b><br>' + hover + '<extra></extra>')])
    fig.update_layout(title=title)
    return fig

# -----------------------
# SALES implementations
# -----------------------
def sales_global_overview(df):
    st.header("Global Sales Overview")
    if 'SALES_CHANNEL_L1' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SALES_CHANNEL_L1 or NET_SALES")
        return
    g = df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False).head(TOP_N)
    g['NET_SALES_M'] = g['NET_SALES'] / 1_000_000
    fig = donut_from_agg(g, 'SALES_CHANNEL_L1', 'NET_SALES', "Sales Channel — Global Overview", hole=0.65, value_is_millions=True)
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(g[['SALES_CHANNEL_L1','NET_SALES']], numeric_cols=['NET_SALES'], index_col='SALES_CHANNEL_L1')

def sales_by_channel_l2(df):
    st.header("Global Net Sales Distribution by SALES_CHANNEL_L2")
    if 'SALES_CHANNEL_L2' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SALES_CHANNEL_L2 or NET_SALES")
        return
    g = df.groupby('SALES_CHANNEL_L2', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False).head(TOP_N)
    fig = donut_from_agg(g, 'SALES_CHANNEL_L2', 'NET_SALES', "Net Sales by SALES_CHANNEL_L2", hole=0.65, value_is_millions=False)
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(g[['SALES_CHANNEL_L2','NET_SALES']], numeric_cols=['NET_SALES'], index_col='SALES_CHANNEL_L2')

def sales_by_shift(df):
    st.header("Net Sales by SHIFT")
    if 'SHIFT' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SHIFT or NET_SALES")
        return
    g = df.groupby('SHIFT', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
    g['PCT'] = 100 * g['NET_SALES'] / g['NET_SALES'].sum()
    fig = go.Figure(data=[go.Pie(labels=g['SHIFT'], values=g['NET_SALES'], hole=0.65,
                                 hovertemplate='%{label}: KSh %{value:,.2f} (<b>%{percent}</b>)<extra></extra>')])
    fig.update_layout(title="Net Sales by SHIFT")
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(g[['SHIFT','NET_SALES','PCT']], numeric_cols=['NET_SALES','PCT'], index_col='SHIFT')

def second_highest_channel_share(df):
    st.header("2nd-Highest Channel Share per Store")
    req = ['STORE_NAME','SALES_CHANNEL_L1','NET_SALES']
    if not all(c in df.columns for c in req):
        st.warning("Missing required columns")
        return
    store_chan = df.groupby(['STORE_NAME','SALES_CHANNEL_L1'], as_index=False)['NET_SALES'].sum()
    store_chan['STORE_TOTAL'] = store_chan.groupby('STORE_NAME')['NET_SALES'].transform('sum')
    store_chan['PCT'] = 100 * store_chan['NET_SALES'] / store_chan['STORE_TOTAL']
    store_chan = store_chan.sort_values(['STORE_NAME','PCT'], ascending=[True, False])
    store_chan['RANK'] = store_chan.groupby('STORE_NAME').cumcount() + 1
    second = store_chan[store_chan['RANK']==2][['STORE_NAME','SALES_CHANNEL_L1','PCT']].rename(columns={'SALES_CHANNEL_L1':'SECOND_CHANNEL','PCT':'SECOND_PCT'})
    # ensure all stores present
    all_stores = store_chan['STORE_NAME'].unique()
    missing = set(all_stores) - set(second['STORE_NAME'])
    if missing:
        add = pd.DataFrame({'STORE_NAME': list(missing), 'SECOND_CHANNEL': ['(None)']*len(missing), 'SECOND_PCT': [0.0]*len(missing)})
        second = pd.concat([second, add], ignore_index=True)
    second_sorted = second.sort_values('SECOND_PCT', ascending=False).head(TOP_N)
    fig = px.bar(second_sorted, x='SECOND_PCT', y='STORE_NAME', orientation='h', text=second_sorted['SECOND_PCT'].map(lambda v: f"{v:.1f}%"))
    fig.update_layout(title=f"Top {len(second_sorted)} Stores by 2nd-Highest Channel Share", height=max(400, 24*len(second_sorted)))
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(second_sorted, numeric_cols=['SECOND_PCT'], index_col='STORE_NAME')

def bottom_30_2nd_highest(df):
    st.header("Bottom 30 — 2nd Highest Channel")
    req = ['STORE_NAME','SALES_CHANNEL_L1','NET_SALES']
    if not all(c in df.columns for c in req):
        st.warning("Missing required columns")
        return
    store_chan = df.groupby(['STORE_NAME','SALES_CHANNEL_L1'], as_index=False)['NET_SALES'].sum()
    store_chan['STORE_TOTAL'] = store_chan.groupby('STORE_NAME')['NET_SALES'].transform('sum')
    store_chan['PCT'] = 100 * store_chan['NET_SALES'] / store_chan['STORE_TOTAL']
    store_chan = store_chan.sort_values(['STORE_NAME','PCT'], ascending=[True, False])
    store_chan['RANK'] = store_chan.groupby('STORE_NAME').cumcount() + 1
    top_tbl = store_chan[store_chan['RANK']==1][['STORE_NAME','SALES_CHANNEL_L1','PCT']].rename(columns={'SALES_CHANNEL_L1':'TOP_CHANNEL','PCT':'TOP_PCT'})
    second_tbl = store_chan[store_chan['RANK']==2][['STORE_NAME','SALES_CHANNEL_L1','PCT']].rename(columns={'SALES_CHANNEL_L1':'SECOND_CHANNEL','PCT':'SECOND_PCT'})
    ranking = pd.merge(top_tbl, second_tbl, on='STORE_NAME', how='left').fillna({'SECOND_CHANNEL':'(None)','SECOND_PCT':0})
    bottom = ranking.sort_values('SECOND_PCT', ascending=True).head(30)
    fig = px.bar(bottom, x='SECOND_PCT', y='STORE_NAME', orientation='h', text=bottom['SECOND_PCT'].map(lambda v: f"{v:.1f}%"))
    fig.update_layout(title="Bottom 30 Stores by 2nd-Highest Channel Share", height=max(500, 24*len(bottom)))
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(bottom, numeric_cols=['SECOND_PCT','TOP_PCT'], index_col='STORE_NAME')

def stores_sales_summary(df):
    st.header("Stores Sales Summary")
    if 'STORE_NAME' not in df.columns:
        st.warning("Missing STORE_NAME")
        return
    df2 = df.copy()
    df2['NET_SALES'] = pd.to_numeric(df2.get('NET_SALES', 0), errors='coerce').fillna(0)
    df2['VAT_AMT'] = pd.to_numeric(df2.get('VAT_AMT', 0), errors='coerce').fillna(0)
    df2['GROSS_SALES'] = df2['NET_SALES'] + df2['VAT_AMT']
    sales_summary = df2.groupby('STORE_NAME', as_index=False)[['NET_SALES','GROSS_SALES']].sum().sort_values('GROSS_SALES', ascending=False)
    sales_summary['% Contribution'] = (sales_summary['GROSS_SALES'] / sales_summary['GROSS_SALES'].sum() * 100).round(2)
    if 'CUST_CODE' in df2.columns and df2['CUST_CODE'].astype(bool).any():
        cust_counts = df2.groupby('STORE_NAME')['CUST_CODE'].nunique().reset_index().rename(columns={'CUST_CODE':'Customer Numbers'})
        sales_summary = sales_summary.merge(cust_counts, on='STORE_NAME', how='left')
    # Use MAX (GROSS_SALES) in row to rank and take Top N
    sales_summary = sales_summary.sort_values('GROSS_SALES', ascending=False).head(TOP_N)
    format_and_display(sales_summary[['STORE_NAME','NET_SALES','GROSS_SALES','% Contribution','Customer Numbers']], numeric_cols=['NET_SALES','GROSS_SALES','% Contribution','Customer Numbers'], index_col='STORE_NAME')

# -----------------------
# OPERATIONS implementations (30-min consistent)
# -----------------------
def customer_traffic_storewise(df):
    st.header("Customer Traffic Heatmap — Storewise (30-min slots)")
    required = ['TRN_DATE','CUST_CODE']
    if not all(c in df.columns for c in required):
        st.warning("Missing TRN_DATE or CUST_CODE")
        return

    d = df.copy()
    first_touch = d.groupby(['STORE_NAME','DATE','CUST_CODE'], as_index=False)['TRN_DATE'].min()
    first_touch['TIME_INTERVAL'] = first_touch['TRN_DATE'].dt.floor('30T')
    first_touch['TIME_ONLY'] = first_touch['TIME_INTERVAL'].dt.time
    counts = first_touch.groupby(['STORE_NAME','TIME_ONLY'])['CUST_CODE'].nunique().reset_index(name='RECEIPT_COUNT')

    pivot = counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='RECEIPT_COUNT').fillna(0)
    pivot = ensure_time_indexed_pivot(pivot)

    if pivot.empty:
        st.info("No customer traffic data to display")
        return

    # Rank rows by max value
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    x = [t.strftime('%H:%M') for t in pivot.columns]
    y = pivot.index.tolist()
    z = pivot.values

    fig = px.imshow(z, x=x, y=y, labels=dict(x="Time Interval (30 min)", y="Store Name", color="Receipts"),
                    text_auto=False, aspect='auto')
    fig.update_xaxes(side='top')
    fig.update_layout(title="Customer Traffic — Storewise (30-min slots)", height=max(400, CHART_ROW_HEIGHT*len(y)))
    st.plotly_chart(fig, use_container_width=True)

    totals = pivot.sum(axis=1).reset_index()
    totals.columns = ['STORE_NAME','Total_Receipts']
    format_and_display(totals, numeric_cols=['Total_Receipts'], index_col='STORE_NAME')

def active_tills_during_day(df):
    st.header("Active Tills During the Day (30-min slots)")
    if 'TRN_DATE' not in df.columns:
        st.warning("Missing TRN_DATE")
        return

    d = df.copy()
    if 'Till_Code' not in d.columns and 'TILL' in d.columns and 'STORE_CODE' in d.columns:
        d['Till_Code'] = d['TILL'].astype(str) + '-' + d['STORE_CODE'].astype(str)
    d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30T')
    d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time

    till_counts = d.groupby(['STORE_NAME','TIME_ONLY'])['Till_Code'].nunique().reset_index(name='UNIQUE_TILLS')
    pivot = till_counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='UNIQUE_TILLS').fillna(0)
    pivot = ensure_time_indexed_pivot(pivot)

    if pivot.empty:
        st.info("No till activity data")
        return

    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    x = [t.strftime('%H:%M') for t in pivot.columns]
    y = pivot.index.tolist()
    z = pivot.values

    fig = px.imshow(z, x=x, y=y, labels=dict(x="Time Interval (30 min)", y="Store Name", color="Unique Tills"),
                    text_auto=False, aspect='auto')
    fig.update_xaxes(side='top')
    fig.update_layout(title="Active Tills — 30-min slots", height=max(400, CHART_ROW_HEIGHT*len(y)))
    st.plotly_chart(fig, use_container_width=True)

    pivot_totals = pivot.max(axis=1).reset_index()
    pivot_totals.columns = ['STORE_NAME','MAX_ACTIVE_TILLS']
    format_and_display(pivot_totals, numeric_cols=['MAX_ACTIVE_TILLS'], index_col='STORE_NAME')

def avg_customers_per_till(df):
    st.header("Average Customers Served per Till (30-min slots)")
    if 'TRN_DATE' not in df.columns:
        st.warning("Missing TRN_DATE")
        return

    d = df.copy()
    # ensure CUST_CODE exists
    if 'CUST_CODE' not in d.columns or not d['CUST_CODE'].astype(bool).any():
        for c in ['STORE_CODE','TILL','SESSION','RCT']:
            if c in d.columns:
                d[c] = d[c].astype(str).fillna('').str.strip()
        d['CUST_CODE'] = d.get('STORE_CODE','').astype(str) + '-' + d.get('TILL','').astype(str) + '-' + d.get('SESSION','').astype(str) + '-' + d.get('RCT','').astype(str)

    # first touch per receipt/day
    first_touch = d.groupby(['STORE_NAME','DATE','CUST_CODE'], as_index=False)['TRN_DATE'].min()
    first_touch['TIME_INTERVAL'] = first_touch['TRN_DATE'].dt.floor('30T')
    first_touch['TIME_ONLY'] = first_touch['TIME_INTERVAL'].dt.time
    cust_counts = first_touch.groupby(['STORE_NAME','TIME_ONLY'])['CUST_CODE'].nunique().reset_index(name='CUSTOMERS')

    d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30T')
    d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time
    if 'Till_Code' not in d.columns and 'TILL' in d.columns and 'STORE_CODE' in d.columns:
        d['Till_Code'] = d['TILL'].astype(str) + '-' + d['STORE_CODE'].astype(str)
    till_counts = d.groupby(['STORE_NAME','TIME_ONLY'])['Till_Code'].nunique().reset_index(name='TILLS')

    cust_pivot = cust_counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='CUSTOMERS').fillna(0)
    cust_pivot = ensure_time_indexed_pivot(cust_pivot)
    till_pivot = till_counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='TILLS').fillna(0)
    till_pivot = ensure_time_indexed_pivot(till_pivot)

    # align indices
    all_stores = sorted(set(cust_pivot.index) | set(till_pivot.index))
    cust_pivot = cust_pivot.reindex(index=all_stores, fill_value=0)
    till_pivot = till_pivot.reindex(index=all_stores, fill_value=0)

    ratio = cust_pivot.divide(till_pivot.replace(0, np.nan)).fillna(0)
    ratio = np.ceil(ratio).astype(int)

    if ratio.empty:
        st.info("No data to display")
        return

    ratio = ratio.loc[ratio.max(axis=1).sort_values(ascending=False).index]

    x = [t.strftime('%H:%M') for t in ratio.columns]
    y = ratio.index.tolist()
    z = ratio.values

    fig = px.imshow(z, x=x, y=y, labels=dict(x="Time Interval (30 min)", y="Store Name", color="Customers per Till"),
                    text_auto=False, aspect='auto')
    fig.update_xaxes(side='top')
    fig.update_layout(title="Avg Customers per Till — 30-min slots", height=max(400, CHART_ROW_HEIGHT*len(y)))
    st.plotly_chart(fig, use_container_width=True)

    pivot_totals = pd.DataFrame({'STORE_NAME': ratio.index, 'MAX_CUSTOMERS_PER_TILL': ratio.max(axis=1).astype(int)})
    format_and_display(pivot_totals, numeric_cols=['MAX_CUSTOMERS_PER_TILL'], index_col='STORE_NAME')

def store_customer_traffic_storewise(df):
    st.header("Store Customer Traffic (per Department) — 30-min slots")
    if 'STORE_NAME' not in df.columns:
        st.warning("Missing STORE_NAME")
        return
    branches = sorted(df['STORE_NAME'].unique())
    branch = st.selectbox("Select Branch", branches)
    d = df[df['STORE_NAME']==branch].copy()
    if d.empty:
        st.info("No data for selected branch")
        return

    first_touch = d.groupby(['DEPARTMENT','DATE','CUST_CODE'], as_index=False)['TRN_DATE'].min()
    first_touch['TIME_INTERVAL'] = first_touch['TRN_DATE'].dt.floor('30T')
    first_touch['TIME_ONLY'] = first_touch['TIME_INTERVAL'].dt.time
    tmp = first_touch.groupby(['DEPARTMENT','TIME_ONLY'])['CUST_CODE'].nunique().reset_index(name='Unique_Customers')

    pivot = tmp.pivot(index='DEPARTMENT', columns='TIME_ONLY', values='Unique_Customers').fillna(0)
    pivot = ensure_time_indexed_pivot(pivot)

    if pivot.empty:
        st.info("No department traffic data")
        return

    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    x = [t.strftime('%H:%M') for t in pivot.columns]
    y = pivot.index.tolist()
    z = pivot.values

    fig = px.imshow(z, x=x, y=y, labels=dict(x="Time Interval (30 min)", y="Department", color="Unique Customers"),
                    text_auto=False, aspect='auto')
    fig.update_xaxes(side='top')
    fig.update_layout(title=f"{branch} — Customer Traffic by Department (30-min slots)", height=max(400, CHART_ROW_HEIGHT*len(y)))
    st.plotly_chart(fig, use_container_width=True)

    totals = pivot.sum(axis=1).reset_index()
    totals.columns = ['DEPARTMENT','TOTAL_CUSTOMERS']
    format_and_display(totals, numeric_cols=['TOTAL_CUSTOMERS'], index_col='DEPARTMENT')

def till_usage(df):
    st.header("Till Usage (30-min slots)")
    d = df.copy()
    if 'TRN_DATE' not in d.columns:
        st.warning("Missing TRN_DATE")
        return
    if 'Till_Code' not in d.columns and 'TILL' in d.columns and 'STORE_CODE' in d.columns:
        d['Till_Code'] = d['TILL'].astype(str) + '-' + d['STORE_CODE'].astype(str)

    d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30T')
    d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time

    till_activity = d.groupby(['STORE_NAME','Till_Code','TIME_ONLY'], as_index=False).agg(Receipts=('CUST_CODE','nunique'))

    branches = sorted(d['STORE_NAME'].unique())
    branch = st.selectbox("Select Branch for Till Usage", branches)
    dfb = till_activity[till_activity['STORE_NAME']==branch]
    if dfb.empty:
        st.info("No till activity")
        return

    pivot = dfb.pivot(index='Till_Code', columns='TIME_ONLY', values='Receipts').fillna(0)
    pivot = ensure_time_indexed_pivot(pivot)
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    x = [t.strftime('%H:%M') for t in pivot.columns]
    y = pivot.index.tolist()
    z = pivot.values

    fig = px.imshow(z, x=x, y=y, labels=dict(x="Time Interval (30 min)", y="Till", color="Receipts"),
                    text_auto=False, aspect='auto')
    fig.update_xaxes(side='top')
    fig.update_layout(title=f"{branch} — Till Usage (30-min slots)", height=max(400, CHART_ROW_HEIGHT*len(y)))
    st.plotly_chart(fig, use_container_width=True)

    totals = pivot.sum(axis=1).reset_index()
    totals.columns = ['Till_Code','Total_Receipts']
    format_and_display(totals, numeric_cols=['Total_Receipts'], index_col='Till_Code')

# -----------------------
# INSIGHTS implementations (pricing + loyalty + suppliers + categories)
# -----------------------
def global_pricing_overview(df):
    st.header("Global Pricing Overview — Multi-Priced SKUs per Day")
    required = ['TRN_DATE','STORE_NAME','ITEM_CODE','ITEM_NAME','QTY','SP_PRE_VAT']
    if not all(c in df.columns for c in required):
        st.warning("Missing pricing columns")
        return
    d = df.copy()
    d['DATE'] = d['TRN_DATE'].dt.date
    grp = d.groupby(['STORE_NAME','DATE','ITEM_CODE','ITEM_NAME'], as_index=False).agg(
        Num_Prices=('SP_PRE_VAT', lambda s: s.dropna().nunique()),
        Price_Min=('SP_PRE_VAT','min'),
        Price_Max=('SP_PRE_VAT','max'),
        Total_QTY=('QTY','sum')
    )
    grp['Price_Spread'] = grp['Price_Max'] - grp['Price_Min']
    multi_price = grp[(grp['Num_Prices']>1) & (grp['Price_Spread']>0)].copy()
    if multi_price.empty:
        st.info("No multi-priced SKUs found")
        return
    multi_price['Diff_Value'] = multi_price['Total_QTY'] * multi_price['Price_Spread']
    summary = multi_price.groupby('STORE_NAME', as_index=False).agg(
        Items_with_MultiPrice=('ITEM_CODE','nunique'),
        Total_Diff_Value=('Diff_Value','sum'),
        Avg_Spread=('Price_Spread','mean'),
        Max_Spread=('Price_Spread','max')
    ).sort_values('Total_Diff_Value', ascending=False).head(TOP_N)
    format_and_display(summary, numeric_cols=['Items_with_MultiPrice','Total_Diff_Value','Avg_Spread','Max_Spread'], index_col='STORE_NAME')

def branch_pricing_overview(df):
    st.header("Branch Pricing Overview (drilldown)")
    required = ['TRN_DATE','STORE_NAME','ITEM_CODE','ITEM_NAME','QTY','SP_PRE_VAT','CUST_CODE']
    if not all(c in df.columns for c in required):
        st.warning("Missing required columns for Branch Pricing Overview")
        return

    d0 = df.copy()
    d0['TRN_DATE'] = pd.to_datetime(d0['TRN_DATE'], errors='coerce')
    d0 = d0.dropna(subset=['TRN_DATE','STORE_NAME','ITEM_CODE','ITEM_NAME','QTY','SP_PRE_VAT','CUST_CODE']).copy()
    for c in ['STORE_NAME','ITEM_CODE','ITEM_NAME','CUST_CODE']:
        if c in d0.columns:
            d0[c] = d0[c].astype(str).str.strip()
    d0['SP_PRE_VAT'] = d0['SP_PRE_VAT'].astype(str).str.replace(',', '', regex=False).str.strip()
    d0['SP_PRE_VAT'] = pd.to_numeric(d0['SP_PRE_VAT'], errors='coerce').fillna(0.0)
    d0['QTY'] = pd.to_numeric(d0['QTY'], errors='coerce').fillna(0.0)
    d0['DATE'] = d0['TRN_DATE'].dt.date

    branches = sorted(d0['STORE_NAME'].unique())
    if not branches:
        st.info("No branches available")
        return
    branch = st.selectbox("Select Branch (pricing drilldown)", branches)
    d = d0[d0['STORE_NAME']==branch].copy()
    if d.empty:
        st.info("No rows for this branch")
        return

    per_item_day = d.groupby(['DATE','ITEM_CODE','ITEM_NAME'], as_index=False).agg(
        Num_Prices=('SP_PRE_VAT', lambda s: s.dropna().nunique()),
        Price_Min=('SP_PRE_VAT','min'),
        Price_Max=('SP_PRE_VAT','max'),
        Total_QTY=('QTY','sum')
    )
    per_item_day['Price_Spread'] = per_item_day['Price_Max'] - per_item_day['Price_Min']
    multi = per_item_day[(per_item_day['Num_Prices']>1) & (per_item_day['Price_Spread']>0)].copy()
    if multi.empty:
        st.success(f"{branch}: No SKUs with multiple prices (spread>0) on same day.")
        return

    multi['Diff_Value'] = (multi['Total_QTY'] * multi['Price_Spread']).round(2)
    # Rank by Diff_Value (impact) and take Top N
    multi = multi.sort_values(['Diff_Value','Price_Spread','Total_QTY'], ascending=[False, False, False]).reset_index(drop=True).head(TOP_N)
    multi.insert(0,'#', range(1, len(multi)+1))

    st.subheader("Per Item/Day — Summary (ranked by Diff_Value)")
    format_and_display(multi[['#','DATE','ITEM_CODE','ITEM_NAME','Num_Prices','Price_Min','Price_Max','Price_Spread','Total_QTY','Diff_Value']],
                       numeric_cols=['Num_Prices','Price_Min','Price_Max','Price_Spread','Total_QTY','Diff_Value'],
                       index_col='ITEM_CODE')

    # Price-level breakdown
    price_brk = (
        d.merge(multi[['DATE','ITEM_CODE']], on=['DATE','ITEM_CODE'], how='inner')
         .groupby(['DATE','ITEM_CODE','ITEM_NAME','SP_PRE_VAT'], as_index=False)
         .agg(Qty_At_Price=('QTY','sum'),
              Receipts_At_Price=('CUST_CODE','nunique'),
              First_Time=('TRN_DATE','min'),
              Last_Time=('TRN_DATE','max'))
    )
    price_brk['First_Time'] = price_brk['First_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    price_brk['Last_Time'] = price_brk['Last_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    price_brk['Time_Window'] = price_brk['First_Time'] + ' → ' + price_brk['Last_Time']
    price_compact = price_brk[['DATE','ITEM_CODE','ITEM_NAME','SP_PRE_VAT','Qty_At_Price','Receipts_At_Price','Time_Window']].rename(columns={
        'SP_PRE_VAT':'Price','Qty_At_Price':'QTY','Receipts_At_Price':'Receipts'
    })

    st.subheader("Price-level breakdown for selected item-days")
    format_and_display(price_compact, numeric_cols=['Price','QTY','Receipts'], index_col='ITEM_CODE')

    # Full receipt-level rows
    receipts_detail = d.merge(multi[['DATE','ITEM_CODE']], on=['DATE','ITEM_CODE'], how='inner')
    cols = ['DATE','ITEM_CODE','ITEM_NAME','CUST_CODE','TRN_DATE','SP_PRE_VAT','QTY']
    for extra in ['CASHIER','Till_Code','SHIFT','SALES_CHANNEL_L1','SALES_CHANNEL_L2']:
        if extra in receipts_detail.columns:
            cols.append(extra)
    cols = [c for c in cols if c in receipts_detail.columns]
    receipts_detail = receipts_detail[cols].sort_values(['DATE','ITEM_CODE','TRN_DATE'], ascending=[False,True,True]).reset_index(drop=True)
    receipts_detail['TRN_DATE'] = receipts_detail['TRN_DATE'].dt.strftime('%Y-%m-%d %H:%M:%S')

    st.subheader("All receipt-level rows for affected item-days")
    st.info(f"Showing {len(receipts_detail):,} receipt rows for selected branch/item-days.")
    with st.expander("Show receipts"):
        st.dataframe(receipts_detail, use_container_width=True)

    # CSV download
    try:
        csv_bytes = receipts_detail.to_csv(index=False).encode('utf-8')
        st.download_button("Download receipts CSV", data=csv_bytes, file_name=f"{branch}_multi_price_receipts.csv", mime='text/csv')
    except Exception:
        pass

def global_loyalty_overview(df):
    st.header("Global Loyalty Overview")
    required = ['TRN_DATE','STORE_NAME','CUST_CODE','LOYALTY_CUSTOMER_CODE','NET_SALES']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns: {missing}")
        return

    dfL = df.copy()
    dfL['TRN_DATE'] = pd.to_datetime(dfL['TRN_DATE'], errors='coerce')
    dfL = dfL.dropna(subset=['TRN_DATE','STORE_NAME','CUST_CODE']).copy()
    for c in ['STORE_NAME','CUST_CODE','LOYALTY_CUSTOMER_CODE']:
        if c in dfL.columns:
            dfL[c] = dfL[c].astype(str).str.strip()
    dfL['NET_SALES'] = pd.to_numeric(dfL['NET_SALES'], errors='coerce').fillna(0)
    dfL = dfL[dfL['LOYALTY_CUSTOMER_CODE'].replace({'nan':'','NaN':'','None':''}).str.len()>0].copy()

    receipts = dfL.groupby(['STORE_NAME','CUST_CODE','LOYALTY_CUSTOMER_CODE'], as_index=False).agg(Basket_Value=('NET_SALES','sum'), First_Time=('TRN_DATE','min'))
    per_branch_multi = receipts.groupby(['STORE_NAME','LOYALTY_CUSTOMER_CODE']).agg(Baskets_in_Store=('CUST_CODE','nunique'), Total_Value_in_Store=('Basket_Value','sum')).reset_index()
    per_branch_multi = per_branch_multi[per_branch_multi['Baskets_in_Store']>1]
    overview = per_branch_multi.groupby('STORE_NAME', as_index=False).agg(Loyal_Customers_Multi=('LOYALTY_CUSTOMER_CODE','nunique'), Total_Baskets_of_Those=('Baskets_in_Store','sum'), Total_Value_of_Those=('Total_Value_in_Store','sum'))
    overview['Avg_Baskets_per_Customer'] = np.where(overview['Loyal_Customers_Multi']>0, (overview['Total_Baskets_of_Those']/overview['Loyal_Customers_Multi']).round(2), 0.0)
    overview = overview.sort_values('Loyal_Customers_Multi', ascending=False).head(TOP_N)
    format_and_display(overview, numeric_cols=['Loyal_Customers_Multi','Total_Baskets_of_Those','Total_Value_of_Those','Avg_Baskets_per_Customer'], index_col='STORE_NAME')

def supplier_contribution(df):
    st.header("Supplier Contribution — Top suppliers by net sales")
    if 'SUPPLIER_NAME' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SUPPLIER_NAME or NET_SALES")
        return
    g = df.groupby('SUPPLIER_NAME', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False).head(TOP_N)
    format_and_display(g, numeric_cols=['NET_SALES'], index_col='SUPPLIER_NAME')
    fig = px.bar(g, x='NET_SALES', y='SUPPLIER_NAME', orientation='h', title="Top Suppliers by Net Sales")
    st.plotly_chart(fig, use_container_width=True)

def category_overview(df):
    st.header("Category Overview")
    if 'CATEGORY' not in df.columns:
        st.warning("Missing CATEGORY")
        return
    g = df.groupby('CATEGORY', as_index=False).agg(Baskets=('CUST_CODE','nunique'), Net_Sales=('NET_SALES','sum')).sort_values('Net_Sales', ascending=False).head(TOP_N)
    format_and_display(g, numeric_cols=['Baskets','Net_Sales'], index_col='CATEGORY')

# -----------------------
# App Entrypoint
# -----------------------
def main():
    st.title("Superdeck — Streamlit edition")
    raw_df = smart_load()
    if raw_df is None:
        st.stop()

    with st.spinner("Preparing data (cached) ..."):
        df = clean_and_derive(raw_df)

    section = st.sidebar.selectbox("Section", ["SALES","OPERATIONS","INSIGHTS"])

    if section == "SALES":
        sales_items = [
            "Global Sales Overview",
            "Net Sales by Sales Channel (L2)",
            "Net Sales by SHIFT",
            "2nd-Highest Channel Share",
            "Bottom 30 — 2nd Highest Channel",
            "Stores Sales Summary"
        ]
        choice = st.sidebar.selectbox("Sales Subsection", sales_items)
        if choice == sales_items[0]:
            sales_global_overview(df)
        elif choice == sales_items[1]:
            sales_by_channel_l2(df)
        elif choice == sales_items[2]:
            sales_by_shift(df)
        elif choice == sales_items[3]:
            second_highest_channel_share(df)
        elif choice == sales_items[4]:
            bottom_30_2nd_highest(df)
        elif choice == sales_items[5]:
            stores_sales_summary(df)

    elif section == "OPERATIONS":
        ops_items = [
            "Customer Traffic-Storewise",
            "Active Tills During the day",
            "Average Customers Served per Till",
            "Store Customer Traffic Storewise",
            "Till Usage"
        ]
        choice = st.sidebar.selectbox("Operations Subsection", ops_items)
        if choice == ops_items[0]:
            customer_traffic_storewise(df)
        elif choice == ops_items[1]:
            active_tills_during_day(df)
        elif choice == ops_items[2]:
            avg_customers_per_till(df)
        elif choice == ops_items[3]:
            store_customer_traffic_storewise(df)
        elif choice == ops_items[4]:
            till_usage(df)

    elif section == "INSIGHTS":
        ins_items = [
            "Global Pricing Overview",
            "Branch Pricing Overview",
            "Global Loyalty Overview",
            "Supplier Contribution",
            "Category Overview"
        ]
        choice = st.sidebar.selectbox("Insights Subsection", ins_items)
        if choice == ins_items[0]:
            global_pricing_overview(df)
        elif choice == ins_items[1]:
            branch_pricing_overview(df)
        elif choice == ins_items[2]:
            global_loyalty_overview(df)
        elif choice == ins_items[3]:
            supplier_contribution(df)
        elif choice == ins_items[4]:
            category_overview(df)

if __name__ == "__main__":
    main()
