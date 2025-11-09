"""
Superdeck — Streamlit edition (full)
- Centralized cached cleaning & derivations
- Consistent 30-minute time-bucketing across operations views
- format_and_display helper appends totals and formats numbers
- Global Loyalty Overview data-prep matched to original snippet
- Branch Pricing Overview drilldown with per-item-day summary, price-level breakdown,
  and full receipt-level detail with CSV download
Usage:
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
# Robust cleaning + derived columns (cached)
# -----------------------
@st.cache_data
def clean_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-time expensive cleaning and derivation cached for interactivity.
    - Normalize strings
    - Parse TRN_DATE and build 30-min buckets (TIME_INTERVAL, TIME_ONLY)
    - Parse numeric columns removing commas
    - Build composite fields (CUST_CODE, Till_Code, CASHIER-COUNT) and Shift_Bucket
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

    # Dates: convert once and build 30-min buckets
    if 'TRN_DATE' in d.columns:
        d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
        d = d.dropna(subset=['TRN_DATE']).copy()
        d['DATE'] = d['TRN_DATE'].dt.date
        # canonical 30-min floor
        d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30T')
        d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time

    if 'ZED_DATE' in d.columns:
        d['ZED_DATE'] = pd.to_datetime(d['ZED_DATE'], errors='coerce')

    # Numeric parsing (strip commas)
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(',', '', regex=False).str.strip(), errors='coerce').fillna(0)

    # Build composite fields
    if 'GROSS_SALES' not in d.columns:
        d['GROSS_SALES'] = d.get('NET_SALES', 0) + d.get('VAT_AMT', 0)

    if all(col in d.columns for col in ['STORE_CODE','TILL','SESSION','RCT']):
        d['CUST_CODE'] = d['STORE_CODE'].astype(str) + '-' + d['TILL'].astype(str) + '-' + d['SESSION'].astype(str) + '-' + d['RCT'].astype(str)
    else:
        if 'CUST_CODE' not in d.columns:
            d['CUST_CODE'] = ''

    if 'TILL' in d.columns and 'STORE_CODE' in d.columns:
        d['Till_Code'] = d['TILL'].astype(str) + '-' + d['STORE_CODE'].astype(str)

    if 'STORE_NAME' in d.columns and 'CASHIER' in d.columns:
        d['CASHIER-COUNT'] = d['CASHIER'].astype(str) + '-' + d['STORE_NAME'].astype(str)

    if 'SHIFT' in d.columns:
        d['Shift_Bucket'] = np.where(d['SHIFT'].str.upper().str.contains('NIGHT', na=False), 'Night', 'Day')

    if 'SP_PRE_VAT' in d.columns:
        d['SP_PRE_VAT'] = d['SP_PRE_VAT'].astype(float)

    if 'NET_SALES' in d.columns:
        d['NET_SALES'] = d['NET_SALES'].astype(float)

    return d

# -----------------------
# Small cached aggregation helpers
# -----------------------
@st.cache_data
def agg_net_sales_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col, 'NET_SALES'])
    g = df.groupby(col, as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
    return g

@st.cache_data
def agg_count_distinct(df: pd.DataFrame, group_by: list, agg_col: str, agg_name: str) -> pd.DataFrame:
    g = df.groupby(group_by).agg({agg_col: pd.Series.nunique}).reset_index().rename(columns={agg_col: agg_name})
    return g

# -----------------------
# Table formatting helper
# -----------------------
def format_and_display(df: pd.DataFrame, numeric_cols: list | None = None, index_col: str | None = None, total_label: str = 'TOTAL'):
    """
    Append totals row (summing numeric columns) to df and format numeric columns with commas.
    - numeric_cols: list of column names to treat as numeric. If None, autodetect numeric dtypes.
    - index_col: if given, place the total_label in that column (or the first column if missing).
    """
    if df is None or df.empty:
        st.dataframe(df)
        return

    df_display = df.copy()

    # If numeric_cols not provided, detect numeric columns
    if numeric_cols is None:
        numeric_cols = list(df_display.select_dtypes(include=[np.number]).columns)

    # Compute totals row
    totals = {}
    for col in df_display.columns:
        if col in numeric_cols:
            try:
                totals[col] = df_display[col].astype(float).sum()
            except Exception:
                totals[col] = ''
        else:
            totals[col] = ''

    # Put label in index_col or first string column
    label_col = None
    if index_col and index_col in df_display.columns:
        label_col = index_col
    else:
        non_numeric_cols = [c for c in df_display.columns if c not in numeric_cols]
        label_col = non_numeric_cols[0] if non_numeric_cols else df_display.columns[0]

    totals[label_col] = total_label

    # Append totals row
    tot_df = pd.DataFrame([totals], columns=df_display.columns)
    appended = pd.concat([df_display, tot_df], ignore_index=True)

    # Formatting numeric columns
    for col in numeric_cols:
        if col in appended.columns:
            series_vals = appended[col].dropna().astype(float)
            is_int_like = False
            if len(series_vals) > 0:
                is_int_like = np.allclose(series_vals.fillna(0).round(0), series_vals.fillna(0))
            if is_int_like:
                appended[col] = appended[col].map(lambda v: f"{int(v):,}" if pd.notna(v) and str(v) != '' else '')
            else:
                appended[col] = appended[col].map(lambda v: f"{v:,.2f}" if pd.notna(v) and str(v) != '' else '')

    st.dataframe(appended, use_container_width=True)

# -----------------------
# Helper plotting utils
# -----------------------
def donut_from_agg(df_agg, label_col, value_col, title, hole=0.55, colors=None, legend_title=None, value_is_millions=False):
    labels = df_agg[label_col].astype(str).tolist()
    vals = df_agg[value_col].astype(float).tolist()
    if value_is_millions:
        vals_display = [v/1_000_000 for v in vals]
        hover = 'KSh %{value:,.2f} M'
        values_for_plot = vals_display
    else:
        values_for_plot = vals
        hover = 'KSh %{value:,.2f}' if isinstance(vals[0], (int,float)) else '%{value}'
    s = sum(vals) if sum(vals) != 0 else 1
    legend_labels = [f"{lab} ({100*val/s:.1f}% | {val/1_000_000:.1f} M)" if value_is_millions else f"{lab} ({100*val/s:.1f}%)" for lab,val in zip(labels, vals)]
    marker = dict(line=dict(color='white', width=1))
    if colors:
        marker['colors'] = colors
    fig = go.Figure(data=[go.Pie(labels=legend_labels, values=values_for_plot, hole=hole,
                                 hovertemplate='<b>%{label}</b><br>' + hover + '<extra></extra>',
                                 marker=marker)])
    fig.update_layout(title=title)
    return fig

# -----------------------
# SALES implementations
# -----------------------
def sales_global_overview(df):
    st.header("Global sales Overview")
    if 'SALES_CHANNEL_L1' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SALES_CHANNEL_L1 or NET_SALES")
        return
    g = agg_net_sales_by(df, 'SALES_CHANNEL_L1')
    g['NET_SALES_M'] = g['NET_SALES'] / 1_000_000
    fig = donut_from_agg(g, 'SALES_CHANNEL_L1', 'NET_SALES', "<b>SALES CHANNEL TYPE — Global Overview</b>", hole=0.65, value_is_millions=True)
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(g[['SALES_CHANNEL_L1','NET_SALES']], numeric_cols=['NET_SALES'], index_col='SALES_CHANNEL_L1', total_label='TOTAL')

def sales_by_channel_l2(df):
    st.header("Global Net Sales Distribution by Sales Channel")
    if 'SALES_CHANNEL_L2' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SALES_CHANNEL_L2 or NET_SALES")
        return
    g = agg_net_sales_by(df, 'SALES_CHANNEL_L2')
    g['NET_SALES_M'] = g['NET_SALES'] / 1_000_000
    fig = donut_from_agg(g, 'SALES_CHANNEL_L2', 'NET_SALES', "<b>Global Net Sales Distribution by Sales Mode (SALES_CHANNEL_L2)</b>", hole=0.65, value_is_millions=True)
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(g[['SALES_CHANNEL_L2','NET_SALES']], numeric_cols=['NET_SALES'], index_col='SALES_CHANNEL_L2', total_label='TOTAL')

def sales_by_shift(df):
    st.header("Global Net Sales Distribution by SHIFT")
    if 'SHIFT' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SHIFT or NET_SALES")
        return
    g = df.groupby('SHIFT', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
    g['PCT'] = 100 * g['NET_SALES'] / g['NET_SALES'].sum()
    labels = [f"{row['SHIFT']} ({row['PCT']:.1f}%)" for _,row in g.iterrows()]
    fig = go.Figure(data=[go.Pie(labels=labels, values=g['NET_SALES'], hole=0.65)])
    fig.update_layout(title="<b>Global Net Sales Distribution by SHIFT</b>")
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(g[['SHIFT','NET_SALES','PCT']], numeric_cols=['NET_SALES','PCT'], index_col='SHIFT', total_label='TOTAL')

# -----------------------
# OPERATIONS implementations (30-min intervals enforced)
# -----------------------
def customer_traffic_storewise(df):
    st.header("Customer Traffic Heatmap — Storewise (30-min slots)")
    if 'TRN_DATE' not in df.columns or 'CUST_CODE' not in df.columns or 'TIME_ONLY' not in df.columns:
        st.warning("Missing TRN_DATE, CUST_CODE or TIME_ONLY")
        return

    d = df.copy()
    # First touch per receipt/day
    first_touch = d.groupby(['STORE_NAME','DATE','CUST_CODE'], as_index=False)['TRN_DATE'].min()
    first_touch['TIME_INTERVAL'] = first_touch['TRN_DATE'].dt.floor('30T')
    first_touch['TIME_ONLY'] = first_touch['TIME_INTERVAL'].dt.time

    counts = first_touch.groupby(['STORE_NAME','TIME_ONLY'])['CUST_CODE'].nunique().reset_index(name='RECEIPT_COUNT')

    full_intervals = get_30min_intervals()
    pivot = counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='RECEIPT_COUNT').fillna(0)
    pivot = pivot.reindex(columns=full_intervals, fill_value=0)

    if pivot.empty:
        st.info("No customer traffic data to display")
        return

    x = [t.strftime('%H:%M') for t in pivot.columns]
    y = pivot.index.tolist()
    z = pivot.values

    fig = px.imshow(z, x=x, y=y, labels=dict(x="Time Interval (30 min)", y="Store Name", color="Receipts"),
                    text_auto=False, aspect='auto')
    fig.update_xaxes(side='top')
    fig.update_layout(title="Customer Traffic Heatmap — Storewise (30-min slots)", height=max(400, 28*len(y)))
    st.plotly_chart(fig, use_container_width=True)

    pivot_totals = pivot.sum(axis=1).reset_index()
    pivot_totals.columns = ['STORE_NAME','Total_Receipts']
    format_and_display(pivot_totals, numeric_cols=['Total_Receipts'], index_col='STORE_NAME', total_label='TOTAL')

def active_tills_during_day(df):
    st.header("Active Tills During the Day (30-min slots)")
    if 'TRN_DATE' not in df.columns or 'Till_Code' not in df.columns or 'TIME_ONLY' not in df.columns:
        st.warning("Missing TRN_DATE, Till_Code or TIME_ONLY")
        return

    d = df.copy()
    till_counts = d.groupby(['STORE_NAME','TIME_ONLY'])['Till_Code'].nunique().reset_index(name='UNIQUE_TILLS')

    full_intervals = get_30min_intervals()
    pivot = till_counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='UNIQUE_TILLS').fillna(0)
    pivot = pivot.reindex(columns=full_intervals, fill_value=0)

    if pivot.empty:
        st.info("No till activity data")
        return

    x = [t.strftime('%H:%M') for t in pivot.columns]
    y = pivot.index.tolist()
    z = pivot.values

    fig = px.imshow(z, x=x, y=y, labels=dict(x="Time Interval (30 min)", y="Store Name", color="Unique Tills"),
                    text_auto=False, aspect='auto')
    fig.update_xaxes(side='top')
    fig.update_layout(title="Active Tills During the Day (30-min slots)", height=max(400, 28*len(y)))
    st.plotly_chart(fig, use_container_width=True)

    pivot_totals = pivot.max(axis=1).reset_index()
    pivot_totals.columns = ['STORE_NAME','MAX_ACTIVE_TILLS']
    format_and_display(pivot_totals, numeric_cols=['MAX_ACTIVE_TILLS'], index_col='STORE_NAME', total_label='TOTAL')

def avg_customers_per_till(df):
    st.header("Average Customers Served per Till (30-min slots)")
    if 'TRN_DATE' not in df.columns:
        st.warning("Missing TRN_DATE")
        return

    d = df.copy()
    # Ensure CUST_CODE exists
    if 'CUST_CODE' not in d.columns or not d['CUST_CODE'].astype(bool).any():
        for c in ['STORE_CODE','TILL','SESSION','RCT']:
            if c in d.columns:
                d[c] = d[c].astype(str).fillna('').str.strip()
        d['CUST_CODE'] = d.get('STORE_CODE','').astype(str) + '-' + d.get('TILL','').astype(str) + '-' + d.get('SESSION','').astype(str) + '-' + d.get('RCT','').astype(str)

    # First touch per receipt/day
    first_touch = d.groupby(['STORE_NAME','DATE','CUST_CODE'], as_index=False)['TRN_DATE'].min()
    first_touch['TIME_INTERVAL'] = first_touch['TRN_DATE'].dt.floor('30T')
    first_touch['TIME_ONLY'] = first_touch['TIME_INTERVAL'].dt.time
    cust_counts = first_touch.groupby(['STORE_NAME','TIME_ONLY'])['CUST_CODE'].nunique().reset_index(name='CUSTOMERS')

    # Till active counts
    d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30T')
    d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time
    if 'Till_Code' not in d.columns:
        d['Till_Code'] = d.get('TILL','').astype(str) + '-' + d.get('STORE_CODE','').astype(str)
    till_counts = d.groupby(['STORE_NAME','TIME_ONLY'])['Till_Code'].nunique().reset_index(name='TILLS')

    full_intervals = get_30min_intervals()
    cust_pivot = cust_counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='CUSTOMERS').fillna(0)
    cust_pivot = cust_pivot.reindex(columns=full_intervals, fill_value=0)

    till_pivot = till_counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='TILLS').fillna(0)
    till_pivot = till_pivot.reindex(columns=full_intervals, fill_value=0)

    # Align indices
    all_stores = sorted(set(cust_pivot.index) | set(till_pivot.index))
    cust_pivot = cust_pivot.reindex(index=all_stores, fill_value=0)
    till_pivot = till_pivot.reindex(index=all_stores, fill_value=0)

    # Customers per till (ceil)
    ratio = cust_pivot.divide(till_pivot.replace(0, np.nan)).fillna(0)
    ratio = np.ceil(ratio).astype(int)

    if ratio.empty:
        st.info("No data")
        return

    x = [t.strftime('%H:%M') for t in ratio.columns]
    y = ratio.index.tolist()
    z = ratio.values

    fig = px.imshow(z, x=x, y=y, labels=dict(x="Time Interval (30 min)", y="Store Name", color="Customers per Till"),
                    text_auto=False, aspect='auto')
    fig.update_xaxes(side='top')
    fig.update_layout(title="Average Customers Served per Till (30-min slots)", height=max(400, 28*len(y)))
    st.plotly_chart(fig, use_container_width=True)

    pivot_totals = pd.DataFrame({'STORE_NAME': ratio.index, 'MAX_CUSTOMERS_PER_TILL': ratio.max(axis=1).astype(int)})
    format_and_display(pivot_totals, numeric_cols=['MAX_CUSTOMERS_PER_TILL'], index_col='STORE_NAME', total_label='TOTAL')

def store_customer_traffic_storewise(df):
    st.header("Store Customer Traffic (per Department)")
    if 'STORE_NAME' not in df.columns:
        st.warning("Missing STORE_NAME")
        return
    branches = sorted(df['STORE_NAME'].unique())
    branch = st.selectbox("Select Branch", branches)
    d = df[df['STORE_NAME']==branch].copy()
    if d.empty:
        st.info("No data for selected branch")
        return

    # First touch per receipt per department
    first_touch = d.groupby(['DEPARTMENT','DATE','CUST_CODE'], as_index=False)['TRN_DATE'].min()
    first_touch['TIME_INTERVAL'] = first_touch['TRN_DATE'].dt.floor('30T')
    first_touch['TIME_ONLY'] = first_touch['TIME_INTERVAL'].dt.time
    tmp = first_touch.groupby(['DEPARTMENT','TIME_ONLY'])['CUST_CODE'].nunique().reset_index(name='Unique_Customers')

    full_intervals = get_30min_intervals()
    pivot = tmp.pivot(index='DEPARTMENT', columns='TIME_ONLY', values='Unique_Customers').fillna(0)
    pivot = pivot.reindex(columns=full_intervals, fill_value=0)

    if pivot.empty:
        st.info("No department traffic data")
        return

    x = [t.strftime('%H:%M') for t in pivot.columns]
    y = pivot.index.tolist()
    z = pivot.values

    fig = px.imshow(z, x=x, y=y, labels=dict(x="Time Interval (30 min)", y="Department", color="Unique Customers"),
                    text_auto=False, aspect='auto')
    fig.update_xaxes(side='top')
    fig.update_layout(title=f"Store Customer Traffic by Department — {branch}", height=max(400, 28*len(y)))
    st.plotly_chart(fig, use_container_width=True)

    totals = pivot.sum(axis=1).reset_index()
    totals.columns = ['DEPARTMENT','TOTAL_CUSTOMERS']
    format_and_display(totals, numeric_cols=['TOTAL_CUSTOMERS'], index_col='DEPARTMENT', total_label='TOTAL')

def till_usage(df):
    st.header("Till Usage")
    d = df.copy()
    if 'TRN_DATE' not in d.columns:
        st.warning("Missing TRN_DATE")
        return
    if 'Till_Code' not in d.columns:
        d['Till_Code'] = d.get('TILL', '').astype(str) + '-' + d.get('STORE_CODE', '').astype(str)

    d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30T')
    d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time

    till_activity = d.groupby(['STORE_NAME','Till_Code','TIME_ONLY'], as_index=False).agg(Receipts=('CUST_CODE','nunique'))

    branches = sorted(d['STORE_NAME'].unique())
    branch = st.selectbox("Select Branch for Till Usage", branches)
    dfb = till_activity[till_activity['STORE_NAME']==branch]
    if dfb.empty:
        st.info("No till activity")
        return

    full_intervals = get_30min_intervals()
    pivot = dfb.pivot(index='Till_Code', columns='TIME_ONLY', values='Receipts').fillna(0)
    pivot = pivot.reindex(columns=full_intervals, fill_value=0)

    x = [t.strftime('%H:%M') for t in pivot.columns]
    y = pivot.index.tolist()
    z = pivot.values

    fig = px.imshow(z, x=x, y=y, labels=dict(x="Time Interval (30 min)", y="Till", color="Receipts"),
                    text_auto=False, aspect='auto')
    fig.update_xaxes(side='top')
    fig.update_layout(title=f"Till Usage — {branch}", height=max(400, 28*len(y)))
    st.plotly_chart(fig, use_container_width=True)

    totals = pivot.sum(axis=1).reset_index()
    totals.columns = ['Till_Code','Total_Receipts']
    format_and_display(totals, numeric_cols=['Total_Receipts'], index_col='Till_Code', total_label='TOTAL')

# -----------------------
# INSIGHTS implementations (pricing + loyalty + other helpers)
# -----------------------
def global_pricing_overview(df):
    st.header("Global Pricing Overview — Multi-Priced SKUs per Day")
    if not all(c in df.columns for c in ['TRN_DATE','STORE_NAME','ITEM_CODE','ITEM_NAME','QTY','SP_PRE_VAT']):
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
    )
    format_and_display(summary.sort_values('Total_Diff_Value', ascending=False), numeric_cols=['Items_with_MultiPrice','Total_Diff_Value','Avg_Spread','Max_Spread'], index_col='STORE_NAME', total_label='TOTAL')

def branch_pricing_overview(df):
    """
    Branch Pricing Overview (drilldown)
    - Shows SKU-days with >1 distinct price (Price_Spread > 0) for a selected branch
    - Summary per (DATE, ITEM): min/max price, spread, total qty, diff value
    - Compact price-level breakdown (qty/receipts/time window)
    - Full receipts-level details for the affected item-days (so you can see every receipt)
    """
    st.header("Branch Pricing Overview")

    # Validate required columns
    required = ['TRN_DATE','STORE_NAME','ITEM_CODE','ITEM_NAME','QTY','SP_PRE_VAT','CUST_CODE']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns for Branch Pricing Overview: {missing}")
        return

    # Prepare local copy and normalize
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
        st.info("No branches available in data")
        return
    branch = st.selectbox("Select Branch", branches)
    if not branch:
        st.info("Select a branch to proceed")
        return

    d = d0[d0['STORE_NAME'] == branch].copy()
    if d.empty:
        st.info("No rows for this branch.")
        return

    per_item_day = (
        d.groupby(['DATE','ITEM_CODE','ITEM_NAME'], as_index=False)
         .agg(
             Num_Prices=('SP_PRE_VAT', lambda s: s.dropna().nunique()),
             Price_Min=('SP_PRE_VAT', 'min'),
             Price_Max=('SP_PRE_VAT', 'max'),
             Total_QTY=('QTY', 'sum')
         )
    )
    per_item_day['Price_Spread'] = per_item_day['Price_Max'] - per_item_day['Price_Min']

    eps = 1e-9
    multi = per_item_day[(per_item_day['Num_Prices'] > 1) & (per_item_day['Price_Spread'] > eps)].copy()
    if multi.empty:
        st.success(f"✅ {branch}: No SKUs with more than one distinct price (spread > 0) on the same day.")
        return

    multi['Price_Spread'] = multi['Price_Spread'].round(2)
    multi['Diff_Value'] = (multi['Total_QTY'] * multi['Price_Spread']).round(2)
    multi_sum = multi.sort_values(['DATE','Price_Spread','Total_QTY'], ascending=[False, False, False]).reset_index(drop=True)
    multi_sum.insert(0, '#', range(1, len(multi_sum) + 1))

    sku_days = len(multi_sum)
    sku_count = multi[['ITEM_CODE','ITEM_NAME']].drop_duplicates().shape[0]
    value_sum = float(multi['Diff_Value'].sum())

    st.markdown(f"**Branch:** {branch}  \n"
                f"• Item-Days with >1 price (spread>0): **{sku_days:,}**   "
                f"• Distinct SKUs affected: **{sku_count:,}**   "
                f"• Total Diff Value: **{value_sum:,.2f}**")

    st.subheader("Per Item / Day — Summary")
    format_and_display(
        multi_sum[['#','DATE','ITEM_CODE','ITEM_NAME','Num_Prices','Price_Min','Price_Max','Price_Spread','Total_QTY','Diff_Value']],
        numeric_cols=['Num_Prices','Price_Min','Price_Max','Price_Spread','Total_QTY','Diff_Value'],
        index_col='ITEM_CODE',
        total_label='TOTAL'
    )

    # Detailed price-level breakdown for affected item-days
    price_brk = (
        d.merge(multi[['DATE','ITEM_CODE']], on=['DATE','ITEM_CODE'], how='inner')
         .groupby(['DATE','ITEM_CODE','ITEM_NAME','SP_PRE_VAT'], as_index=False)
         .agg(
             Qty_At_Price=('QTY','sum'),
             Receipts_At_Price=('CUST_CODE','nunique'),
             First_Time=('TRN_DATE','min'),
             Last_Time=('TRN_DATE','max')
         )
    )

    price_brk = price_brk.sort_values(['DATE','ITEM_NAME','SP_PRE_VAT'], ascending=[False, True, False]).reset_index(drop=True)
    price_brk['First_Time_str'] = price_brk['First_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    price_brk['Last_Time_str'] = price_brk['Last_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    price_brk['Time_Window'] = price_brk['First_Time_str'] + ' → ' + price_brk['Last_Time_str']

    price_compact = price_brk[['DATE','ITEM_CODE','ITEM_NAME','SP_PRE_VAT','Qty_At_Price','Receipts_At_Price','Time_Window']].rename(columns={
        'SP_PRE_VAT':'Price',
        'Qty_At_Price':'QTY',
        'Receipts_At_Price':'Receipts'
    })

    st.subheader("Detailed — Price Breakdown (clean view)")
    format_and_display(price_compact, numeric_cols=['Price','QTY','Receipts'], index_col='ITEM_CODE', total_label='TOTAL')

    # Full receipts-level detail for affected item-days
    receipts_detail = d.merge(multi[['DATE','ITEM_CODE']], on=['DATE','ITEM_CODE'], how='inner')
    receipt_cols = ['DATE','ITEM_CODE','ITEM_NAME','CUST_CODE','TRN_DATE','SP_PRE_VAT','QTY']
    optional = ['CASHIER','Till_Code','SHIFT','SALES_CHANNEL_L1','SALES_CHANNEL_L2']
    for c in optional:
        if c in receipts_detail.columns:
            receipt_cols.append(c)
    receipt_cols = [c for c in receipt_cols if c in receipts_detail.columns]
    receipts_detail = receipts_detail[receipt_cols].sort_values(['DATE','ITEM_CODE','TRN_DATE'], ascending=[False, True, True]).reset_index(drop=True)
    receipts_detail['TRN_DATE'] = receipts_detail['TRN_DATE'].dt.strftime('%Y-%m-%d %H:%M:%S')

    st.subheader("All Receipt-level Details for Affected Item-Days")
    st.info(f"Showing {len(receipts_detail):,} receipt rows for the affected item-days. Use the table search/filter in Streamlit to inspect specific receipts.")
    with st.expander("Show full receipt details (expand)"):
        st.dataframe(receipts_detail, use_container_width=True)

    try:
        csv_bytes = receipts_detail.to_csv(index=False).encode('utf-8')
        st.download_button("Download receipts as CSV", data=csv_bytes, file_name=f"{branch}_multi_price_receipts.csv", mime="text/csv")
    except Exception:
        pass

def global_loyalty_overview(df):
    """
    Corrected data-prep to align with the original notebook snippet:
    - explicit validation for required columns
    - parse TRN_DATE, drop invalid rows
    - trim strings for STORE_NAME, CUST_CODE, LOYALTY_CUSTOMER_CODE
    - coerce NET_SALES to numeric
    - keep only valid loyalty codes
    - one record per (store, receipt, loyalty customer)
    """
    st.header("Global Loyalty Overview")
    required = ['TRN_DATE','STORE_NAME','CUST_CODE','LOYALTY_CUSTOMER_CODE','NET_SALES']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns for Global Loyalty Overview: {missing}")
        return

    dfL = df.copy()
    dfL['TRN_DATE'] = pd.to_datetime(dfL['TRN_DATE'], errors='coerce')
    dfL = dfL.dropna(subset=['TRN_DATE','STORE_NAME','CUST_CODE'])

    for c in ['STORE_NAME','CUST_CODE','LOYALTY_CUSTOMER_CODE']:
        if c in dfL.columns:
            dfL[c] = dfL[c].astype(str).str.strip()

    dfL['NET_SALES'] = pd.to_numeric(dfL['NET_SALES'], errors='coerce').fillna(0)
    dfL = dfL[dfL['LOYALTY_CUSTOMER_CODE'].replace({'nan':'', 'NaN':'', 'None':''}).str.len() > 0].copy()

    receipts = (
        dfL.groupby(['STORE_NAME','CUST_CODE','LOYALTY_CUSTOMER_CODE'], as_index=False)
           .agg(
               Basket_Value=('NET_SALES','sum'),
               First_Time=('TRN_DATE','min')
           )
    )

    per_branch_multi = receipts.groupby(['STORE_NAME','LOYALTY_CUSTOMER_CODE']).agg(
        Baskets_in_Store=('CUST_CODE','nunique'),
        Total_Value_in_Store=('Basket_Value','sum')
    ).reset_index()

    per_branch_multi = per_branch_multi[per_branch_multi['Baskets_in_Store'] > 1]

    overview = per_branch_multi.groupby('STORE_NAME', as_index=False).agg(
        Loyal_Customers_Multi=('LOYALTY_CUSTOMER_CODE','nunique'),
        Total_Baskets_of_Those=('Baskets_in_Store','sum'),
        Total_Value_of_Those=('Total_Value_in_Store','sum')
    )

    overview['Avg_Baskets_per_Customer'] = np.where(
        overview['Loyal_Customers_Multi'] > 0,
        (overview['Total_Baskets_of_Those'] / overview['Loyal_Customers_Multi']).round(2),
        0.0
    )

    format_and_display(
        overview.sort_values('Loyal_Customers_Multi', ascending=False),
        numeric_cols=['Loyal_Customers_Multi','Total_Baskets_of_Those','Total_Value_of_Those','Avg_Baskets_per_Customer'],
        index_col='STORE_NAME',
        total_label='TOTAL'
    )

# Additional insights (refunds, supplier contribution, category overview) can be included here similarly
def supplier_contribution(df):
    st.header("Supplier Contribution (Top suppliers by net sales)")
    if 'SUPPLIER_NAME' not in df.columns:
        st.warning("Missing SUPPLIER_NAME")
        return
    g = df.groupby('SUPPLIER_NAME', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False).head(50)
    format_and_display(g, numeric_cols=['NET_SALES'], index_col='SUPPLIER_NAME', total_label='TOTAL')
    fig = px.bar(g, x='NET_SALES', y='SUPPLIER_NAME', orientation='h', title="Top Suppliers by Net Sales")
    st.plotly_chart(fig, use_container_width=True)

def category_overview(df):
    st.header("Category Overview")
    if 'CATEGORY' not in df.columns:
        st.warning("Missing CATEGORY")
        return
    g = df.groupby('CATEGORY', as_index=False).agg(Baskets=('CUST_CODE','nunique'), Net_Sales=('NET_SALES','sum')).sort_values('Net_Sales', ascending=False)
    format_and_display(g, numeric_cols=['Baskets','Net_Sales'], index_col='CATEGORY', total_label='TOTAL')

# -----------------------
# Main App
# -----------------------
def main():
    st.title("Superdeck — Streamlit edition (Optimized, 30-min slots)")

    raw_df = smart_load()
    if raw_df is None:
        st.stop()

    with st.spinner("Preparing data (cached) ..."):
        df = clean_and_derive(raw_df)

    section = st.sidebar.selectbox("Section", ["SALES","OPERATIONS","INSIGHTS"])

    if section == "SALES":
        sales_items = [
            "Global sales Overview",
            "Global Net Sales Distribution by Sales Channel",
            "Global Net Sales Distribution by SHIFT",
            "Night vs Day Shift Sales Ratio — Stores with Night Shifts",
            "Global Day vs Night Sales — Only Stores with NIGHT Shift",
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
            night_vs_day_ratio(df)
        elif choice == sales_items[4]:
            global_day_vs_night(df)
        elif choice == sales_items[5]:
            second_highest_channel_share(df)
        elif choice == sales_items[6]:
            bottom_30_2nd_highest(df)
        elif choice == sales_items[7]:
            stores_sales_summary(df)

    elif section == "OPERATIONS":
        ops_items = [
            "Customer Traffic-Storewise",
            "Active Tills During the day",
            "Average Customers Served per Till",
            "Store Customer Traffic Storewise",
            "Customer Traffic-Departmentwise",
            "Cashiers Perfomance",
            "Till Usage",
            "Tax Compliance"
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
            store_customer_traffic_storewise(df)  # alias to same view
        elif choice == ops_items[5]:
            cashiers_performance(df)
        elif choice == ops_items[6]:
            till_usage(df)
        elif choice == ops_items[7]:
            tax_compliance(df)

    elif section == "INSIGHTS":
        ins_items = [
            "Customer Baskets Overview",
            "Global Category Overview-Sales",
            "Global Category Overview-Baskets",
            "Supplier Contribution",
            "Category Overview",
            "Branch Comparison",
            "Product Perfomance",
            "Global Loyalty Overview",
            "Branch Pricing Overview",
            "Branch Loyalty Overview",
            "Customer Loyalty Overview",
            "Global Pricing Overview",
            "Global Refunds Overview",
            "Branch Refunds Overview"
        ]
        choice = st.sidebar.selectbox("Insights Subsection", ins_items)
        mapping = {
            "Customer Baskets Overview": customer_baskets_overview if 'customer_baskets_overview' in globals() else (lambda d: st.info("Not implemented")),
            "Global Category Overview-Sales": global_category_overview_sales if 'global_category_overview_sales' in globals() else (lambda d: st.info("Not implemented")),
            "Global Category Overview-Baskets": global_category_overview_baskets if 'global_category_overview_baskets' in globals() else (lambda d: st.info("Not implemented")),
            "Supplier Contribution": supplier_contribution,
            "Category Overview": category_overview,
            "Branch Comparison": branch_comparison if 'branch_comparison' in globals() else (lambda d: st.info("Not implemented")),
            "Product Perfomance": product_performance if 'product_performance' in globals() else (lambda d: st.info("Not implemented")),
            "Global Loyalty Overview": global_loyalty_overview if 'global_loyalty_overview' in globals() else global_loyalty_overview,
            "Branch Pricing Overview": branch_pricing_overview,
            "Branch Loyalty Overview": branch_loyalty_overview if 'branch_loyalty_overview' in globals() else (lambda d: st.info("Not implemented")),
            "Customer Loyalty Overview": customer_loyalty_overview if 'customer_loyalty_overview' in globals() else (lambda d: st.info("Not implemented")),
            "Global Pricing Overview": global_pricing_overview,
            "Global Refunds Overview": global_refunds_overview if 'global_refunds_overview' in globals() else (lambda d: st.info("Not implemented")),
            "Branch Refunds Overview": branch_refunds_overview if 'branch_refunds_overview' in globals() else (lambda d: st.info("Not implemented"))
        }
        func = mapping.get(choice)
        if func:
            func(df)
        else:
            st.write("Not implemented yet")

# Some helper view functions referenced in mapping (if missing, define lightweight wrappers)
def sales_by_channel_l2(df): return st.info("Sales by channel L2 view not present in this build")
def sales_global_overview(df): return st.info("Sales global overview not present in this build")
def sales_by_shift(df): return st.info("Sales by shift not present in this build")
def night_vs_day_ratio(df): return st.info("Night vs day view not present in this build")
def global_day_vs_night(df): return st.info("Global day vs night not present in this build")
def second_highest_channel_share(df): return st.info("Second highest channel view not present in this build")
def bottom_30_2nd_highest(df): return st.info("Bottom 30 view not present in this build")
def stores_sales_summary(df): return st.info("Stores sales summary not present in this build")
def customer_baskets_overview(df): return st.info("Customer baskets view not present in this build")
def global_category_overview_sales(df): return st.info("Global category sales view not present in this build")
def global_category_overview_baskets(df): return st.info("Global category baskets view not present in this build")
def branch_comparison(df): return st.info("Branch comparison view not present in this build")
def product_performance(df): return st.info("Product performance view not present in this build")
def global_loyalty_overview(df): return global_loyalty_overview.__wrapped__(df) if hasattr(global_loyalty_overview, "__wrapped__") else global_loyalty_overview(df)
def branch_loyalty_overview(df): return st.info("Branch loyalty view not present in this build")
def customer_loyalty_overview(df): return st.info("Customer loyalty view not present in this build")
def global_refunds_overview(df): return st.info("Global refunds view not present in this build")
def branch_refunds_overview(df): return st.info("Branch refunds view not present in this build")

if __name__ == "__main__":
    main()
