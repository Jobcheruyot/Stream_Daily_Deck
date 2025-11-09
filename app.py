# Streamlit app converted from Colab notebook "Superdeck"
# Updated: Added table formatting helper to append totals and format numeric columns with commas.
# Usage:
#   streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import textwrap
from datetime import timedelta

st.set_page_config(layout="wide", page_title="Superdeck (Streamlit)")

# -----------------------
# Utility / Data Loading
# -----------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path, on_bad_lines='skip', low_memory=False)

def smart_load():
    st.sidebar.markdown("### Upload data (CSV) or use default")
    uploaded = st.sidebar.file_uploader("Upload DAILY_POS_TRN_ITEMS CSV", type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded, on_bad_lines='skip', low_memory=False)
        st.sidebar.success("Loaded uploaded CSV")
        return df
    # try default path
    default_path = "/content/DAILY_POS_TRN_ITEMS_2025-10-21.csv"
    try:
        df = load_csv(default_path)
        st.sidebar.info(f"Loaded default path: {default_path}")
        return df
    except Exception:
        st.sidebar.warning("No default CSV found. Please upload a CSV to run the app.")
        return None

def clean_common(df):
    # Minimal robust cleaning used across pages
    df = df.copy()
    # Dates
    for dcol in ['TRN_DATE', 'ZED_DATE']:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors='coerce')
    # numeric columns
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(',', '', regex=False).str.strip()
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # string cleanup for keys
    for c in ['STORE_CODE','TILL','SESSION','RCT','STORE_NAME','CASHIER','ITEM_CODE','ITEM_NAME','DEPARTMENT','CATEGORY','CU_DEVICE_SERIAL','CAP_CUSTOMER_CODE','LOYALTY_CUSTOMER_CODE','SUPPLIER_NAME','SALES_CHANNEL_L1','SALES_CHANNEL_L2','SHIFT']:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna('').str.strip()
    # Build composite fields used in many analyses
    if all(col in df.columns for col in ['STORE_CODE','TILL','SESSION','RCT']):
        df['CUST_CODE'] = df['STORE_CODE'] + '-' + df['TILL'] + '-' + df['SESSION'] + '-' + df['RCT']
    if 'TILL' in df.columns and 'STORE_CODE' in df.columns:
        df['Till_Code'] = df['TILL'].astype(str) + '-' + df['STORE_CODE'].astype(str)
    if 'STORE_NAME' in df.columns and 'CASHIER' in df.columns:
        df['CASHIER-COUNT'] = df['CASHIER'].astype(str) + '-' + df['STORE_NAME'].astype(str)
    # Fill net_sales/vat
    if 'NET_SALES' in df.columns:
        df['NET_SALES'] = df['NET_SALES'].fillna(0)
    if 'VAT_AMT' in df.columns:
        df['VAT_AMT'] = df['VAT_AMT'].fillna(0)
    if 'GROSS_SALES' not in df.columns and 'NET_SALES' in df.columns:
        df['GROSS_SALES'] = df.get('NET_SALES', 0) + df.get('VAT_AMT', 0)
    # Ensure TRN_DATE exists as datetime where available
    if 'TRN_DATE' in df.columns:
        df = df.dropna(subset=['TRN_DATE']).copy()
    return df

# -----------------------
# Table formatting helper
# -----------------------
def format_and_display(df: pd.DataFrame, numeric_cols: list | None = None, index_col: str | None = None, total_label: str = 'TOTAL'):
    """
    Append a totals row (summing numeric columns) to df and format numeric columns with commas.
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
        # find first non-numeric column to host label
        non_numeric_cols = [c for c in df_display.columns if c not in numeric_cols]
        label_col = non_numeric_cols[0] if non_numeric_cols else df_display.columns[0]

    totals[label_col] = total_label

    # Append totals row
    tot_df = pd.DataFrame([totals], columns=df_display.columns)
    appended = pd.concat([df_display, tot_df], ignore_index=True)

    # Formatting
    for col in numeric_cols:
        if col in appended.columns:
            # detect integer-like
            series_vals = appended[col].dropna().astype(float)
            is_int_like = np.allclose(series_vals.fillna(0).round(0), series_vals.fillna(0))
            if is_int_like:
                appended[col] = appended[col].map(lambda v: f"{int(v):,}" if pd.notna(v) and str(v) != '' else '')
            else:
                appended[col] = appended[col].map(lambda v: f"{v:,.2f}" if pd.notna(v) and str(v) != '' else '')

    # Present with st.dataframe (strings will render nicely)
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
    # compute pct for labels
    s = sum(vals) if sum(vals) != 0 else 1
    legend_labels = [f"{lab} ({100*val/s:.1f}% | {val/1_000_000:.1f} M)" if value_is_millions else f"{lab} ({100*val/s:.1f}%)" for lab,val in zip(labels, vals)]
    # build marker dict only if needed; Pie expects 'colors' not 'color' for marker
    marker = dict(line=dict(color='white', width=1))
    if colors:
        marker['colors'] = colors
    fig = go.Figure(data=[go.Pie(labels=legend_labels, values=values_for_plot, hole=hole,
                                 hovertemplate='<b>%{label}</b><br>' + hover + '<extra></extra>',
                                 marker=marker)])
    fig.update_layout(title=title)
    return fig

# -----------------------
# SALES section implementations
# -----------------------
def sales_global_overview(df):
    st.header("Global sales Overview")
    if 'SALES_CHANNEL_L1' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SALES_CHANNEL_L1 or NET_SALES")
        return
    g = df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
    g['NET_SALES_M'] = g['NET_SALES'] / 1_000_000
    fig = donut_from_agg(g, 'SALES_CHANNEL_L1', 'NET_SALES', "<b>SALES CHANNEL TYPE — Global Overview</b>", hole=0.65, value_is_millions=True)
    st.plotly_chart(fig, use_container_width=True)
    # Also show table with totals
    format_and_display(g[['SALES_CHANNEL_L1','NET_SALES']], numeric_cols=['NET_SALES'], index_col='SALES_CHANNEL_L1', total_label='TOTAL')

def sales_by_channel_l2(df):
    st.header("Global Net Sales Distribution by Sales Channel")
    if 'SALES_CHANNEL_L2' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SALES_CHANNEL_L2 or NET_SALES")
        return
    g = df.groupby('SALES_CHANNEL_L2', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
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
    # Table with totals
    format_and_display(g[['SHIFT','NET_SALES','PCT']], numeric_cols=['NET_SALES','PCT'], index_col='SHIFT', total_label='TOTAL')

def night_vs_day_ratio(df):
    st.header("Night vs Day Shift Sales Ratio — Stores with Night Shifts")
    # Build store-level percent Night/Day
    if 'SHIFT' not in df.columns or 'STORE_NAME' not in df.columns:
        st.warning("Missing SHIFT or STORE_NAME")
        return
    df2 = df.copy()
    df2['Shift_Bucket'] = np.where(df2['SHIFT'].str.upper().str.contains('NIGHT', na=False), 'Night', 'Day')
    stores_with_night = df2[df2['Shift_Bucket'] == 'Night']['STORE_NAME'].unique()
    df_nd = df2[df2['STORE_NAME'].isin(stores_with_night)].copy()
    ratio = df_nd.groupby(['STORE_NAME','Shift_Bucket'])['NET_SALES'].sum().reset_index()
    ratio['STORE_TOTAL'] = ratio.groupby('STORE_NAME')['NET_SALES'].transform('sum')
    ratio['PCT'] = 100 * ratio['NET_SALES'] / ratio['STORE_TOTAL']
    pivot = ratio.pivot(index='STORE_NAME', columns='Shift_Bucket', values='PCT').fillna(0)
    if pivot.empty:
        st.info("No stores with NIGHT shift found")
        return
    pivot_sorted = pivot.sort_values('Night', ascending=False)
    numbered_labels = [f"{i+1}. {s}" for i,s in enumerate(pivot_sorted.index)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pivot_sorted['Night'], y=numbered_labels, orientation='h', name='Night', marker_color='#d62728', text=[f"{v:.1f}%" for v in pivot_sorted['Night']], textposition='inside'))
    # Day percent as annotations
    for i,(n_val, d_val) in enumerate(zip(pivot_sorted['Night'], pivot_sorted['Day'])):
        fig.add_annotation(x=n_val + 1, y=numbered_labels[i], text=f"{d_val:.1f}% Day", showarrow=False, xanchor='left')
    fig.update_layout(title="Night vs Day Shift Sales Ratio — Stores with Night Shifts", xaxis_title="% of Store Sales", height=700)
    st.plotly_chart(fig, use_container_width=True)
    # Table with totals
    table = pivot_sorted.reset_index().rename(columns={'Night':'Night_%','Day':'Day_%'})
    format_and_display(table, numeric_cols=['Night_%','Day_%'], index_col='STORE_NAME', total_label='TOTAL')

def global_day_vs_night(df):
    st.header("Global Day vs Night Sales — Only Stores with NIGHT Shifts")
    if 'SHIFT' not in df.columns:
        st.warning("Missing SHIFT")
        return
    df2 = df.copy()
    df2['Shift_Bucket'] = np.where(df2['SHIFT'].str.upper().str.contains('NIGHT', na=False), 'Night', 'Day')
    stores_with_night = df2[df2['Shift_Bucket']=='Night']['STORE_NAME'].unique()
    df_nd = df2[df2['STORE_NAME'].isin(stores_with_night)]
    if df_nd.empty:
        st.info("No stores with night shifts")
        return
    agg = df_nd.groupby('Shift_Bucket', as_index=False)['NET_SALES'].sum()
    agg['PCT'] = 100 * agg['NET_SALES'] / agg['NET_SALES'].sum()
    labels = [f"{r.Shift_Bucket} ({r.PCT:.1f}%)" for _,r in agg.iterrows()]
    fig = go.Figure(go.Pie(labels=labels, values=agg['NET_SALES'], hole=0.65))
    fig.update_layout(title="<b>Global Day vs Night Sales — Only Stores with NIGHT Shifts</b>")
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(agg, numeric_cols=['NET_SALES','PCT'], index_col='Shift_Bucket', total_label='TOTAL')

def second_highest_channel_share(df):
    st.header("2nd-Highest Channel Share")
    if not all(col in df.columns for col in ['STORE_NAME','SALES_CHANNEL_L1','NET_SALES']):
        st.warning("Missing columns required")
        return
    data = df.copy()
    data['NET_SALES'] = pd.to_numeric(data['NET_SALES'], errors='coerce').fillna(0)
    store_chan = data.groupby(['STORE_NAME','SALES_CHANNEL_L1'], as_index=False)['NET_SALES'].sum()
    store_tot = store_chan.groupby('STORE_NAME')['NET_SALES'].transform('sum')
    store_chan['PCT'] = 100 * store_chan['NET_SALES'] / store_tot
    store_chan = store_chan.sort_values(['STORE_NAME','PCT'], ascending=[True, False])
    store_chan['RANK'] = store_chan.groupby('STORE_NAME').cumcount() + 1
    second = store_chan[store_chan['RANK']==2][['STORE_NAME','SALES_CHANNEL_L1','PCT']].rename(columns={'SALES_CHANNEL_L1':'SECOND_CHANNEL','PCT':'SECOND_PCT'})
    all_stores = store_chan['STORE_NAME'].drop_duplicates()
    missing_stores = set(all_stores) - set(second['STORE_NAME'])
    if missing_stores:
        add = pd.DataFrame({'STORE_NAME':list(missing_stores),'SECOND_CHANNEL':['(None)']*len(missing_stores),'SECOND_PCT':[0.0]*len(missing_stores)})
        second = pd.concat([second, add], ignore_index=True)
    second_sorted = second.sort_values('SECOND_PCT', ascending=False)
    top_n = st.sidebar.slider("Top N", min_value=10, max_value=100, value=30)
    top_ = second_sorted.head(top_n).copy()
    if top_.empty:
        st.info("No stores to display")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top_['SECOND_PCT'], y=top_['STORE_NAME'], orientation='h',
                         marker_color='#9aa0a6', name='Stem', hoverinfo='none',
                         text=[f"{p:.1f}%" for p in top_['SECOND_PCT']], textposition='outside'))
    fig.add_trace(go.Scatter(x=top_['SECOND_PCT'], y=top_['STORE_NAME'], mode='markers',
                             marker=dict(color='#1f77b4', size=10), name='2nd Channel %',
                             hovertemplate='%{x:.1f}%<extra></extra>'))
    annotations = []
    for idx, row in top_.iterrows():
        annotations.append(dict(x=row['SECOND_PCT'] + 1, y=row['STORE_NAME'], text=f"{row['SECOND_CHANNEL']}", showarrow=False, xanchor='left', font=dict(size=10)))
    fig.update_layout(title=f"Top {top_n} Stores by 2nd-Highest Channel Share (SALES_CHANNEL_L1)",
                      xaxis_title="2nd-Highest Channel Share (% of Store NET_SALES)",
                      height=max(500, 24*len(top_)),
                      annotations=annotations, yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig, use_container_width=True)
    # show table with totals
    format_and_display(second_sorted[['STORE_NAME','SECOND_CHANNEL','SECOND_PCT']], numeric_cols=['SECOND_PCT'], index_col='STORE_NAME', total_label='TOTAL')

def bottom_30_2nd_highest(df):
    st.header("Bottom 30 — 2nd Highest Channel")
    if not all(col in df.columns for col in ['STORE_NAME','SALES_CHANNEL_L1','NET_SALES']):
        st.warning("Missing required columns")
        return
    data = df.copy()
    data['NET_SALES'] = pd.to_numeric(data['NET_SALES'], errors='coerce').fillna(0)
    store_chan = data.groupby(['STORE_NAME','SALES_CHANNEL_L1'], as_index=False)['NET_SALES'].sum()
    store_tot = store_chan.groupby('STORE_NAME')['NET_SALES'].transform('sum')
    store_chan['PCT'] = 100 * store_chan['NET_SALES'] / store_tot
    store_chan = store_chan.sort_values(['STORE_NAME','PCT'], ascending=[True, False])
    store_chan['RANK'] = store_chan.groupby('STORE_NAME').cumcount() + 1
    top_tbl = store_chan[store_chan['RANK']==1][['STORE_NAME','SALES_CHANNEL_L1','PCT']].rename(columns={'SALES_CHANNEL_L1':'TOP_CHANNEL','PCT':'TOP_PCT'})
    second_tbl = store_chan[store_chan['RANK']==2][['STORE_NAME','SALES_CHANNEL_L1','PCT']].rename(columns={'SALES_CHANNEL_L1':'SECOND_CHANNEL','PCT':'SECOND_PCT'})
    ranking = pd.merge(top_tbl, second_tbl, on='STORE_NAME', how='left').fillna({'SECOND_CHANNEL':'(None)','SECOND_PCT':0})
    bottom_30 = ranking.sort_values('SECOND_PCT', ascending=True).head(30)
    if bottom_30.empty:
        st.info("No stores to display")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bottom_30['SECOND_PCT'], y=bottom_30['STORE_NAME'], orientation='h', marker_color='#9aa0a6', name='Stem', text=[f"{v:.1f}%" for v in bottom_30['SECOND_PCT']], textposition='outside'))
    fig.add_trace(go.Scatter(x=bottom_30['SECOND_PCT'], y=bottom_30['STORE_NAME'], mode='markers', marker=dict(color='#1f77b4', size=10), name='2nd Channel %'))
    annotations = []
    for idx, row in bottom_30.iterrows():
        annotations.append(dict(x=row['SECOND_PCT'] + 1, y=row['STORE_NAME'], text=f"{row['SECOND_CHANNEL']}", showarrow=False, xanchor='left', font=dict(size=10)))
    fig.update_layout(title="Bottom 30 Stores by 2nd-Highest Channel Share (SALES_CHANNEL_L1)", xaxis_title="2nd-Highest Channel Share (% of Store NET_SALES)", height=max(500, 24*len(bottom_30)), annotations=annotations, yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(bottom_30, numeric_cols=['SECOND_PCT','TOP_PCT'], index_col='STORE_NAME', total_label='TOTAL')

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
    if 'CUST_CODE' in df2.columns:
        cust_counts = df2.groupby('STORE_NAME')['CUST_CODE'].nunique().reset_index().rename(columns={'CUST_CODE':'Customer Numbers'})
        sales_summary = sales_summary.merge(cust_counts, on='STORE_NAME', how='left')
    # Format & totals display
    format_and_display(sales_summary[['STORE_NAME','NET_SALES','GROSS_SALES','% Contribution','Customer Numbers']], numeric_cols=['NET_SALES','GROSS_SALES','% Contribution','Customer Numbers'], index_col='STORE_NAME', total_label='TOTAL')

# -----------------------
# OPERATIONS implementations (tables formatted where used)
# -----------------------
def customer_traffic_storewise(df):
    st.header("Customer Traffic Heatmap — Storewise (30-min slots)")
    if 'TRN_DATE' not in df.columns or 'CUST_CODE' not in df.columns:
        st.warning("Missing TRN_DATE or CUST_CODE")
        return
    df2 = df.copy()
    df2['TRN_DATE'] = pd.to_datetime(df2['TRN_DATE'], errors='coerce')
    df2['TRN_DATE_ONLY'] = df2['TRN_DATE'].dt.date
    first_touch = df2.dropna(subset=['TRN_DATE']).groupby(['STORE_NAME','TRN_DATE_ONLY','CUST_CODE'], as_index=False)['TRN_DATE'].min()
    first_touch['TIME_INTERVAL'] = first_touch['TRN_DATE'].dt.floor('30T')
    first_touch['TIME_ONLY'] = first_touch['TIME_INTERVAL'].dt.time
    counts = first_touch.groupby(['STORE_NAME','TIME_ONLY'])['CUST_CODE'].nunique().reset_index(name='RECEIPT_COUNT')
    pivot = counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='RECEIPT_COUNT').fillna(0)
    if pivot.empty:
        st.info("No customer traffic data to display")
        return
    intervals = sorted(pivot.columns)
    z = pivot.values
    x = [t.strftime('%H:%M') for t in intervals]
    y = pivot.index.tolist()
    fig = px.imshow(z, x=x, y=y, labels=dict(x="Time Interval (30 min)", y="Store Name", color="Receipts"), text_auto=True)
    fig.update_xaxes(side='top')
    st.plotly_chart(fig, use_container_width=True)
    # provide totals table per store
    pivot_totals = pivot.sum(axis=1).reset_index().rename(columns={0:'Total_Receipts'}) if isinstance(pivot.sum(axis=1), pd.Series) else pd.DataFrame()
    pivot_totals = pivot.sum(axis=1).reset_index()
    pivot_totals.columns = ['STORE_NAME','Total_Receipts']
    format_and_display(pivot_totals, numeric_cols=['Total_Receipts'], index_col='STORE_NAME', total_label='TOTAL')

def active_tills_during_day(df):
    st.header("Active Tills During the Day (30-min slots)")
    if 'TRN_DATE' not in df.columns or 'Till_Code' not in df.columns:
        st.warning("Missing TRN_DATE or Till_Code")
        return
    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30T')
    d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time
    till_counts = d.groupby(['STORE_NAME','TIME_ONLY'])['Till_Code'].nunique().reset_index(name='UNIQUE_TILLS')
    pivot = till_counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='UNIQUE_TILLS').fillna(0)
    if pivot.empty:
        st.info("No till activity data")
        return
    intervals = sorted(pivot.columns)
    z = pivot.values
    x = [t.strftime('%H:%M') for t in intervals]
    y = pivot.index.tolist()
    fig = px.imshow(z, x=x, y=y, labels=dict(x="Time Interval (30 min)", y="Store Name", color="Unique Tills"), text_auto=True)
    fig.update_xaxes(side='top')
    st.plotly_chart(fig, use_container_width=True)
    # Totals per store
    pivot_totals = pivot.max(axis=1).reset_index()
    pivot_totals.columns = ['STORE_NAME','MAX_ACTIVE_TILLS']
    format_and_display(pivot_totals, numeric_cols=['MAX_ACTIVE_TILLS'], index_col='STORE_NAME', total_label='TOTAL')

def avg_customers_per_till(df):
    st.header("Average Customers Served per Till (30-min slots)")
    if 'TRN_DATE' not in df.columns:
        st.warning("Missing TRN_DATE")
        return
    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    for c in ['STORE_CODE','TILL','SESSION','RCT']:
        if c in d.columns:
            d[c] = d[c].astype(str).fillna('').str.strip()
    if 'CUST_CODE' not in d.columns:
        d['CUST_CODE'] = d['STORE_CODE'] + '-' + d['TILL'] + '-' + d['SESSION'] + '-' + d['RCT']
    d['TRN_DATE_ONLY'] = d['TRN_DATE'].dt.date
    first_touch = d.groupby(['STORE_NAME','TRN_DATE_ONLY','CUST_CODE'], as_index=False)['TRN_DATE'].min()
    first_touch['TIME_INTERVAL'] = first_touch['TRN_DATE'].dt.floor('30T')
    first_touch['TIME_ONLY'] = first_touch['TIME_INTERVAL'].dt.time
    cust_counts = first_touch.groupby(['STORE_NAME','TIME_ONLY'])['CUST_CODE'].nunique().reset_index(name='CUSTOMERS')
    d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30T')
    d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time
    d['Till_Code'] = d['TILL'].astype(str) + '-' + d['STORE_CODE'].astype(str)
    till_counts = d.groupby(['STORE_NAME','TIME_ONLY'])['Till_Code'].nunique().reset_index(name='TILLS')
    cust_pivot = cust_counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='CUSTOMERS').fillna(0)
    till_pivot = till_counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='TILLS').fillna(0)
    cols = sorted(set(cust_pivot.columns) | set(till_pivot.columns))
    cust_pivot = cust_pivot.reindex(columns=cols).fillna(0)
    till_pivot = till_pivot.reindex(columns=cols).fillna(0)
    ratio = cust_pivot / till_pivot.replace(0, np.nan)
    ratio = np.ceil(ratio.fillna(0)).astype(int)
    if ratio.empty:
        st.info("No data")
        return
    intervals = sorted(ratio.columns)
    z = ratio.values
    x = [t.strftime('%H:%M') for t in intervals]
    y = ratio.index.tolist()
    fig = px.imshow(z, x=x, y=y, labels=dict(x="Time Interval (30 min)", y="Store Name", color="Customers per Till"), text_auto=True)
    fig.update_xaxes(side='top')
    st.plotly_chart(fig, use_container_width=True)
    # Totals: max ratio per store
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
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d = d.dropna(subset=['TRN_DATE'])
    d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30T')
    d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time
    tmp = d.groupby(['DEPARTMENT','TIME_ONLY'])['CUST_CODE'].nunique().reset_index(name='Unique_Customers')
    pivot = tmp.pivot(index='DEPARTMENT', columns='TIME_ONLY', values='Unique_Customers').fillna(0)
    if pivot.empty:
        st.info("No department traffic data")
        return
    intervals = sorted(pivot.columns)
    z = pivot.values
    x = [t.strftime('%H:%M') for t in intervals]
    y = pivot.index.tolist()
    fig = px.imshow(z, x=x, y=y, labels=dict(x="Time of Day", y="Department", color="Unique Customers"), text_auto=True)
    fig.update_xaxes(side='top')
    st.plotly_chart(fig, use_container_width=True)
    # Totals per department
    totals = pivot.sum(axis=1).reset_index()
    totals.columns = ['DEPARTMENT','TOTAL_CUSTOMERS']
    format_and_display(totals, numeric_cols=['TOTAL_CUSTOMERS'], index_col='DEPARTMENT', total_label='TOTAL')

def customer_traffic_departmentwise(df):
    st.header("Customer Traffic — Departmentwise (branch selectable)")
    store_customer_traffic_storewise(df)

def cashiers_performance(df):
    st.header("Cashiers Performance")
    if not all(c in df.columns for c in ['TRN_DATE','CUST_CODE','CASHIER-COUNT','STORE_NAME']):
        st.warning("Missing required columns for cashier performance")
        return
    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    receipt_duration = d.groupby(['STORE_NAME','CUST_CODE'], as_index=False).agg(Start_Time=('TRN_DATE','min'), End_Time=('TRN_DATE','max'))
    receipt_duration['Duration_Sec'] = (receipt_duration['End_Time'] - receipt_duration['Start_Time']).dt.total_seconds().fillna(0)
    merged = d.merge(receipt_duration[['STORE_NAME','CUST_CODE','Duration_Sec']], on=['STORE_NAME','CUST_CODE'], how='left')
    cashier_summary = merged.groupby(['STORE_NAME','CASHIER-COUNT'], as_index=False).agg(Avg_Duration_Sec=('Duration_Sec','mean'), Customers_Served=('CUST_CODE','nunique'))
    cashier_summary['Avg_Serve_Min'] = (cashier_summary['Avg_Duration_Sec']/60).round(1)
    branches = sorted(cashier_summary['STORE_NAME'].unique())
    branch = st.selectbox("Select Branch for Cashier Performance", branches)
    dfb = cashier_summary[cashier_summary['STORE_NAME']==branch].sort_values('Avg_Serve_Min')
    if dfb.empty:
        st.info("No cashier data for this branch")
        return
    dfb['Label'] = dfb['Avg_Serve_Min'].astype(str) + ' min (' + dfb['Customers_Served'].astype(str) + ' customers)'
    fig = px.bar(dfb, x='Avg_Serve_Min', y='CASHIER-COUNT', orientation='h', text='Label', color='Avg_Serve_Min', color_continuous_scale='Blues', title=f"Avg Serving Time per Cashier — {branch}")
    fig.update_layout(coloraxis_showscale=False, height=max(400, 25*len(dfb)))
    st.plotly_chart(fig, use_container_width=True)
    # Show formatted table with totals
    format_and_display(dfb[['CASHIER-COUNT','Avg_Serve_Min','Customers_Served']], numeric_cols=['Avg_Serve_Min','Customers_Served'], index_col='CASHIER-COUNT', total_label='TOTAL')

def till_usage(df):
    st.header("Till Usage")
    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d['TIME_SLOT'] = d['TRN_DATE'].dt.floor('30T')
    d['TIME_ONLY'] = d['TIME_SLOT'].dt.time
    if 'Till_Code' not in d.columns:
        d['Till_Code'] = d['TILL'].astype(str) + '-' + d['STORE_CODE'].astype(str)
    till_activity = d.groupby(['STORE_NAME','Till_Code','TIME_ONLY'], as_index=False).agg(Receipts=('CUST_CODE','nunique'))
    branches = sorted(d['STORE_NAME'].unique())
    branch = st.selectbox("Select Branch for Till Usage", branches)
    dfb = till_activity[till_activity['STORE_NAME']==branch]
    if dfb.empty:
        st.info("No till activity")
        return
    pivot = dfb.pivot(index='Till_Code', columns='TIME_ONLY', values='Receipts').fillna(0)
    x = [t.strftime('%H:%M') for t in sorted(pivot.columns)]
    fig = px.imshow(pivot.values, x=x, y=pivot.index, labels=dict(x="Time of Day (30-min slot)", y="Till", color="Receipts"), text_auto=True)
    fig.update_xaxes(side='top')
    st.plotly_chart(fig, use_container_width=True)
    # Totals per till
    totals = pivot.sum(axis=1).reset_index()
    totals.columns = ['Till_Code','Total_Receipts']
    format_and_display(totals, numeric_cols=['Total_Receipts'], index_col='Till_Code', total_label='TOTAL')

def tax_compliance(df):
    st.header("Tax Compliance")
    if 'CU_DEVICE_SERIAL' not in df.columns or 'CUST_CODE' not in df.columns:
        st.warning("Missing CU_DEVICE_SERIAL or CUST_CODE")
        return
    d = df.copy()
    d['Tax_Compliant'] = np.where(d['CU_DEVICE_SERIAL'].replace({'nan':'','NaN':'','None':''}).str.strip().str.len()>0, 'Compliant', 'Non-Compliant')
    store_till = d.groupby(['STORE_NAME','Till_Code','Tax_Compliant'], as_index=False).agg(Receipts=('CUST_CODE','nunique'))
    branch = st.selectbox("Select Branch for Tax Compliance", sorted(d['STORE_NAME'].unique()))
    dfb = store_till[store_till['STORE_NAME']==branch]
    if dfb.empty:
        st.info("No compliance data for branch")
        return
    pivot = dfb.pivot(index='Till_Code', columns='Tax_Compliant', values='Receipts').fillna(0)
    pivot = pivot.reindex(columns=['Compliant','Non-Compliant'], fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=pivot.index, x=pivot['Compliant'], orientation='h', name='Compliant', marker_color='#2ca02c'))
    fig.add_trace(go.Bar(y=pivot.index, x=pivot['Non-Compliant'], orientation='h', name='Non-Compliant', marker_color='#d62728', text=pivot['Non-Compliant'], textposition='outside'))
    fig.update_layout(barmode='stack', title=f"Tax Compliance by Till — {branch}", height=max(400, 24*len(pivot.index)))
    st.plotly_chart(fig, use_container_width=True)
    # Summary table per store (compliant/non-compliant + total + pct)
    store_summary = d.groupby(['STORE_NAME','Tax_Compliant'], as_index=False).agg(Receipts=('CUST_CODE','nunique')).pivot(index='STORE_NAME', columns='Tax_Compliant', values='Receipts').fillna(0)
    store_summary['Total'] = store_summary.sum(axis=1)
    store_summary['Compliance_%'] = np.where(store_summary['Total']>0, (store_summary.get('Compliant',0)/store_summary['Total']*100).round(1), 0.0)
    format_and_display(store_summary.reset_index(), numeric_cols=['Compliant','Non-Compliant','Total','Compliance_%'], index_col='STORE_NAME', total_label='TOTAL')

# -----------------------
# INSIGHTS implementations (tables formatted)
# -----------------------
def customer_baskets_overview(df):
    st.header("Customer Baskets Overview")
    d = df.copy()
    d = d.dropna(subset=['ITEM_NAME','CUST_CODE','STORE_NAME','DEPARTMENT'])
    d = d[~d['DEPARTMENT'].str.upper().eq('LUGGAGE & BAGS')]
    branches = sorted(d['STORE_NAME'].unique())
    branch = st.selectbox("Branch for comparison (branch vs global)", branches)
    metric = st.selectbox("Metric", ['QTY','NET_SALES'])
    top_x = st.number_input("Top X", min_value=5, max_value=200, value=10)
    departments = sorted(d['DEPARTMENT'].unique())
    selected_depts = st.multiselect("Departments (empty = all)", options=departments, default=None)
    temp = d.copy()
    if selected_depts:
        temp = temp[temp['DEPARTMENT'].isin(selected_depts)]
    basket_count = temp.groupby('ITEM_NAME')['CUST_CODE'].nunique().rename('Count_of_Baskets')
    agg_data = temp.groupby('ITEM_NAME')[['QTY','NET_SALES']].sum()
    global_top = basket_count.to_frame().join(agg_data).reset_index().sort_values(metric, ascending=False).head(int(top_x))
    global_top.insert(0,'#', range(1, len(global_top)+1))
    st.subheader("Global Top Items")
    format_and_display(global_top.reset_index(drop=True), numeric_cols=['Count_of_Baskets','QTY','NET_SALES'], index_col='ITEM_NAME', total_label='TOTAL')
    branch_df = temp[temp['STORE_NAME']==branch]
    if branch_df.empty:
        st.info("No data for selected branch")
        return
    basket_count_b = branch_df.groupby('ITEM_NAME')['CUST_CODE'].nunique().rename('Count_of_Baskets')
    agg_b = branch_df.groupby('ITEM_NAME')[['QTY','NET_SALES']].sum()
    branch_top = basket_count_b.to_frame().join(agg_b).reset_index().sort_values(metric, ascending=False).head(int(top_x))
    branch_top.insert(0,'#', range(1, len(branch_top)+1))
    st.subheader(f"{branch} Top Items")
    format_and_display(branch_top.reset_index(drop=True), numeric_cols=['Count_of_Baskets','QTY','NET_SALES'], index_col='ITEM_NAME', total_label='TOTAL')
    missing = set(global_top['ITEM_NAME']) - set(branch_top['ITEM_NAME'])
    if missing:
        st.subheader(f"Items in Global Top {top_x} but missing/underperforming in {branch}")
        missing_df = global_top[global_top['ITEM_NAME'].isin(missing)].reset_index(drop=True)
        format_and_display(missing_df, numeric_cols=['Count_of_Baskets','QTY','NET_SALES'], index_col='ITEM_NAME', total_label='TOTAL')
    else:
        st.success(f"All top {top_x} global items also present in {branch}")

def global_category_overview_sales(df):
    st.header("Global Category Overview — Sales")
    if 'CATEGORY' not in df.columns:
        st.warning("Missing CATEGORY")
        return
    g = df.groupby('CATEGORY', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
    format_and_display(g, numeric_cols=['NET_SALES'], index_col='CATEGORY', total_label='TOTAL')
    fig = px.bar(g.head(20), x='NET_SALES', y='CATEGORY', orientation='h', title="Top Categories by Net Sales")
    st.plotly_chart(fig, use_container_width=True)

def global_category_overview_baskets(df):
    st.header("Global Category Overview — Baskets")
    if 'CATEGORY' not in df.columns:
        st.warning("Missing CATEGORY")
        return
    g = df.groupby('CATEGORY', as_index=False)['CUST_CODE'].nunique().rename(columns={'CUST_CODE':'Baskets'}).sort_values('Baskets', ascending=False)
    format_and_display(g, numeric_cols=['Baskets'], index_col='CATEGORY', total_label='TOTAL')
    fig = px.bar(g.head(20), x='Baskets', y='CATEGORY', orientation='h', title="Top Categories by Baskets")
    st.plotly_chart(fig, use_container_width=True)

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

def branch_comparison(df):
    st.header("Branch Comparison")
    branches = sorted(df['STORE_NAME'].unique())
    col1, col2 = st.columns(2)
    with col1:
        a = st.selectbox("Branch A", branches, index=0)
    with col2:
        b = st.selectbox("Branch B", branches, index=1 if len(branches)>1 else 0)
    metric = st.selectbox("Metric", ['QTY','NET_SALES'])
    top_x = st.number_input("Top X", min_value=5, max_value=200, value=10)
    def top_items(branch):
        temp = df[df['STORE_NAME']==branch]
        baskets = temp.groupby('ITEM_NAME')['CUST_CODE'].nunique().rename('Count_of_Baskets')
        totals = temp.groupby('ITEM_NAME')[[ 'QTY','NET_SALES']].sum()
        merged = baskets.to_frame().join(totals, how='outer').fillna(0).reset_index().sort_values(metric, ascending=False).head(int(top_x))
        merged.insert(0,'#', range(1, len(merged)+1))
        return merged
    topA = top_items(a)
    topB = top_items(b)
    combined = pd.concat([topA.assign(Branch=a), topB.assign(Branch=b)], ignore_index=True)
    fig = px.bar(combined, x=metric, y='ITEM_NAME', color='Branch', orientation='h', text='Count_of_Baskets', barmode='group', title=f"Branch Comparison — {a} vs {b}")
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    st.subheader(f"{a} Top Items")
    format_and_display(topA, numeric_cols=['Count_of_Baskets','QTY','NET_SALES'], index_col='ITEM_NAME', total_label='TOTAL')
    st.subheader(f"{b} Top Items")
    format_and_display(topB, numeric_cols=['Count_of_Baskets','QTY','NET_SALES'], index_col='ITEM_NAME', total_label='TOTAL')

def product_performance(df):
    st.header("Product Performance")
    if 'ITEM_CODE' not in df.columns:
        st.warning("Missing ITEM_CODE")
        return
    lookup = df[['ITEM_CODE','ITEM_NAME']].drop_duplicates().sort_values(['ITEM_CODE','ITEM_NAME'])
    options = (lookup['ITEM_CODE'] + ' — ' + lookup['ITEM_NAME']).tolist()
    choice = st.selectbox("Choose SKU (CODE — NAME)", options)
    item_code = choice.split('—')[0].strip()
    item_data = df[df['ITEM_CODE']==item_code]
    if item_data.empty:
        st.info("No data for this SKU")
        return
    any_name = item_data['ITEM_NAME'].iloc[0]
    st.subheader(f"SKU: {item_code} — {any_name}")
    baskets = item_data[['STORE_NAME','CUST_CODE']].drop_duplicates().assign(Has_Item=1)
    basket_counts = baskets.groupby('STORE_NAME').agg(Baskets_With_Item=('Has_Item','sum')).reset_index()
    total_qty = item_data.groupby('STORE_NAME')['QTY'].sum().reset_index().rename(columns={'QTY':'Total_QTY_Sold_Branch'})
    summary = basket_counts.merge(total_qty, on='STORE_NAME', how='left').fillna(0).sort_values('Baskets_With_Item', ascending=False)
    format_and_display(summary, numeric_cols=['Baskets_With_Item','Total_QTY_Sold_Branch'], index_col='STORE_NAME', total_label='TOTAL')
    fig = px.bar(summary, x='Baskets_With_Item', y='STORE_NAME', orientation='h', title="Baskets with Item by Store")
    st.plotly_chart(fig, use_container_width=True)

def global_loyalty_overview(df):
    st.header("Global Loyalty Overview")
    if not all(c in df.columns for c in ['TRN_DATE','STORE_NAME','CUST_CODE','LOYALTY_CUSTOMER_CODE','NET_SALES']):
        st.warning("Missing required loyalty columns")
        return
    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d = d[d['LOYALTY_CUSTOMER_CODE'].str.replace('nan','').str.strip().astype(bool)]
    receipts = d.groupby(['STORE_NAME','CUST_CODE','LOYALTY_CUSTOMER_CODE'], as_index=False).agg(Basket_Value=('NET_SALES','sum'), First_Time=('TRN_DATE','min'))
    per_branch_multi = receipts.groupby(['STORE_NAME','LOYALTY_CUSTOMER_CODE']).agg(Baskets_in_Store=('CUST_CODE','nunique'), Total_Value_in_Store=('Basket_Value','sum')).reset_index()
    per_branch_multi = per_branch_multi[per_branch_multi['Baskets_in_Store']>1]
    overview = per_branch_multi.groupby('STORE_NAME', as_index=False).agg(Loyal_Customers_Multi=('LOYALTY_CUSTOMER_CODE','nunique'), Total_Baskets_of_Those=('Baskets_in_Store','sum'), Total_Value_of_Those=('Total_Value_in_Store','sum'))
    overview['Avg_Baskets_per_Customer'] = (overview['Total_Baskets_of_Those'] / overview['Loyal_Customers_Multi']).round(2)
    format_and_display(overview.sort_values('Loyal_Customers_Multi', ascending=False), numeric_cols=['Loyal_Customers_Multi','Total_Baskets_of_Those','Total_Value_of_Those','Avg_Baskets_per_Customer'], index_col='STORE_NAME', total_label='TOTAL')

def branch_loyalty_overview(df):
    st.header("Branch Loyalty Overview (per-branch loyal customers with >1 baskets)")
    if not all(c in df.columns for c in ['TRN_DATE','STORE_NAME','CUST_CODE','LOYALTY_CUSTOMER_CODE','NET_SALES']):
        st.warning("Missing required loyalty columns")
        return
    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d = d[d['LOYALTY_CUSTOMER_CODE'].str.replace('nan','').str.strip().astype(bool)]
    receipts = d.groupby(['STORE_NAME','CUST_CODE','LOYALTY_CUSTOMER_CODE'], as_index=False).agg(Basket_Value=('NET_SALES','sum'), First_Time=('TRN_DATE','min'))
    stores = sorted(receipts['STORE_NAME'].unique())
    store = st.selectbox("Select Branch", stores)
    per_store = receipts[receipts['STORE_NAME']==store].groupby('LOYALTY_CUSTOMER_CODE', as_index=False).agg(Baskets_in_Store=('CUST_CODE','nunique'), Total_Value_in_Store=('Basket_Value','sum'), First_Time=('First_Time','min'))
    per_store = per_store[per_store['Baskets_in_Store']>1].sort_values(['Baskets_in_Store','Total_Value_in_Store'], ascending=False)
    format_and_display(per_store, numeric_cols=['Baskets_in_Store','Total_Value_in_Store'], index_col='LOYALTY_CUSTOMER_CODE', total_label='TOTAL')

def customer_loyalty_overview(df):
    st.header("Customer Loyalty Overview (global)")
    if not all(c in df.columns for c in ['TRN_DATE','STORE_NAME','CUST_CODE','LOYALTY_CUSTOMER_CODE','NET_SALES']):
        st.warning("Missing required loyalty columns")
        return
    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d = d[d['LOYALTY_CUSTOMER_CODE'].str.replace('nan','').str.strip().astype(bool)]
    receipts = d.groupby(['STORE_NAME','CUST_CODE','LOYALTY_CUSTOMER_CODE'], as_index=False).agg(Basket_Value=('NET_SALES','sum'), First_Time=('TRN_DATE','min'))
    stores_per_cust = receipts.groupby('LOYALTY_CUSTOMER_CODE')['STORE_NAME'].nunique().reset_index(name='Stores_Visited')
    customers = sorted(stores_per_cust[stores_per_cust['Stores_Visited']>1]['LOYALTY_CUSTOMER_CODE'].unique())
    if not customers:
        st.info("No loyalty customers with >1 baskets found")
        return
    cust = st.selectbox("Select Loyalty Customer (multi-store)", customers)
    rc = receipts[receipts['LOYALTY_CUSTOMER_CODE']==cust]
    if rc.empty:
        st.info("No receipts for this customer")
        return
    per_store = rc.groupby('STORE_NAME', as_index=False).agg(Baskets=('CUST_CODE','nunique'), Total_Value=('Basket_Value','sum'), First_Time=('First_Time','min'), Last_Time=('First_Time','max')).sort_values(['Baskets','Total_Value'], ascending=False)
    format_and_display(per_store, numeric_cols=['Baskets','Total_Value'], index_col='STORE_NAME', total_label='TOTAL')
    # Receipt-level detail
    rc_disp = rc.copy()
    rc_disp['First_Time'] = rc_disp['First_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    rc_disp = rc_disp.rename(columns={'CUST_CODE':'Receipt_No','Basket_Value':'Basket_Value_KSh'})
    format_and_display(rc_disp[['STORE_NAME','Receipt_No','Basket_Value_KSh','First_Time']], numeric_cols=['Basket_Value_KSh'], index_col='STORE_NAME', total_label='TOTAL')

def global_pricing_overview(df):
    st.header("Global Pricing Overview — Multi-Priced SKUs per Day")
    if not all(c in df.columns for c in ['TRN_DATE','STORE_NAME','ITEM_CODE','ITEM_NAME','QTY','SP_PRE_VAT']):
        st.warning("Missing pricing columns")
        return
    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d['DATE'] = d['TRN_DATE'].dt.date
    d['SP_PRE_VAT'] = pd.to_numeric(d['SP_PRE_VAT'].astype(str).str.replace(',','',regex=False), errors='coerce').fillna(0)
    grp = d.groupby(['STORE_NAME','DATE','ITEM_CODE','ITEM_NAME'], as_index=False).agg(Num_Prices=('SP_PRE_VAT', lambda s: s.dropna().nunique()), Price_Min=('SP_PRE_VAT','min'), Price_Max=('SP_PRE_VAT','max'), Total_QTY=('QTY','sum'))
    grp['Price_Spread'] = grp['Price_Max'] - grp['Price_Min']
    multi_price = grp[(grp['Num_Prices']>1) & (grp['Price_Spread']>0)].copy()
    multi_price['Diff_Value'] = multi_price['Total_QTY'] * multi_price['Price_Spread']
    summary = multi_price.groupby('STORE_NAME', as_index=False).agg(Items_with_MultiPrice=('ITEM_CODE','nunique'), Total_Diff_Value=('Diff_Value','sum'), Avg_Spread=('Price_Spread','mean'), Max_Spread=('Price_Spread','max'))
    format_and_display(summary.sort_values('Total_Diff_Value', ascending=False), numeric_cols=['Items_with_MultiPrice','Total_Diff_Value','Avg_Spread','Max_Spread'], index_col='STORE_NAME', total_label='TOTAL')

def branch_pricing_overview(df):
    st.header("Branch Pricing Overview")
    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d['DATE'] = d['TRN_DATE'].dt.date
    d['SP_PRE_VAT'] = pd.to_numeric(d['SP_PRE_VAT'].astype(str).str.replace(',','',regex=False), errors='coerce').fillna(0)
    branches = sorted(d['STORE_NAME'].unique())
    branch = st.selectbox("Select Branch", branches)
    per_item_day = d[d['STORE_NAME']==branch].groupby(['DATE','ITEM_CODE','ITEM_NAME'], as_index=False).agg(Num_Prices=('SP_PRE_VAT', lambda s: s.dropna().nunique()), Price_Min=('SP_PRE_VAT','min'), Price_Max=('SP_PRE_VAT','max'), Total_QTY=('QTY','sum'))
    per_item_day['Price_Spread'] = per_item_day['Price_Max'] - per_item_day['Price_Min']
    multi = per_item_day[(per_item_day['Num_Prices']>1) & (per_item_day['Price_Spread']>0)]
    if multi.empty:
        st.success(f"{branch}: No SKUs with multiple prices on same day")
        return
    multi['Diff_Value'] = multi['Total_QTY'] * multi['Price_Spread']
    format_and_display(multi.sort_values('Diff_Value', ascending=False), numeric_cols=['Num_Prices','Price_Min','Price_Max','Price_Spread','Total_QTY','Diff_Value'], index_col='ITEM_CODE', total_label='TOTAL')

def global_refunds_overview(df):
    st.header("Global Refunds Overview (Negative receipts)")
    d = df.copy()
    d['NET_SALES'] = pd.to_numeric(d['NET_SALES'].astype(str).str.replace(',','',regex=False), errors='coerce').fillna(0)
    neg = d[d['NET_SALES']<0]
    if neg.empty:
        st.info("No negative receipts found")
        return
    if 'CAP_CUSTOMER_CODE' in neg.columns:
        neg['Sale_Type'] = np.where(neg['CAP_CUSTOMER_CODE'].str.replace('nan','').str.strip().astype(bool), 'On_account sales', 'General sales')
    else:
        neg['Sale_Type'] = 'General sales'
    summary = neg.groupby(['STORE_NAME','Sale_Type'], as_index=False).agg(Total_Neg_Value=('NET_SALES','sum'), Total_Count=('CUST_CODE','nunique'))
    format_and_display(summary.sort_values('Total_Neg_Value'), numeric_cols=['Total_Neg_Value','Total_Count'], index_col='STORE_NAME', total_label='TOTAL')

def branch_refunds_overview(df):
    st.header("Branch Refunds Overview (Negative receipts per store)")
    d = df.copy()
    d['NET_SALES'] = pd.to_numeric(d['NET_SALES'].astype(str).str.replace(',','',regex=False), errors='coerce').fillna(0)
    neg = d[d['NET_SALES']<0]
    branches = sorted(neg['STORE_NAME'].unique())
    if not branches:
        st.info("No negative receipts")
        return
    branch = st.selectbox("Select Branch", branches)
    dfb = neg[neg['STORE_NAME']==branch]
    agg = dfb.groupby(['STORE_NAME','CUST_CODE'], as_index=False).agg(Total_Value=('NET_SALES','sum'), First_Time=('TRN_DATE','min'), Cashier=('CASHIER','first'))
    format_and_display(agg.sort_values('Total_Value'), numeric_cols=['Total_Value'], index_col='CUST_CODE', total_label='TOTAL')

# -----------------------
# Main App
# -----------------------
def main():
    st.title("Superdeck — Streamlit edition")
    df = smart_load()
    if df is None:
        st.stop()
    df = clean_common(df)

    # Top-level sections
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
            customer_traffic_departmentwise(df)
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
            "Branch Loyalty Overview",
            "Customer Loyalty Overview",
            "Global Pricing Overview",
            "Branch Pricing Overview",
            "Global Refunds Overview",
            "Branch Refunds Overview"
        ]
        choice = st.sidebar.selectbox("Insights Subsection", ins_items)
        mapping = {
            ins_items[0]: customer_baskets_overview,
            ins_items[1]: global_category_overview_sales,
            ins_items[2]: global_category_overview_baskets,
            ins_items[3]: supplier_contribution,
            ins_items[4]: category_overview,
            ins_items[5]: branch_comparison,
            ins_items[6]: product_performance,
            ins_items[7]: global_loyalty_overview,
            ins_items[8]: branch_loyalty_overview,
            ins_items[9]: customer_loyalty_overview,
            ins_items[10]: global_pricing_overview,
            ins_items[11]: branch_pricing_overview,
            ins_items[12]: global_refunds_overview,
            ins_items[13]: branch_refunds_overview
        }
        func = mapping.get(choice)
        if func:
            func(df)
        else:
            st.write("Not implemented yet")

if __name__ == "__main__":
    main()
