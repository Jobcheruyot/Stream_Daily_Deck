
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta, datetime
import os
from dotenv import load_dotenv

# Try to import Supabase, with fallback
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    st.warning("Supabase client not available. Please install with: pip install supabase")

st.set_page_config(layout="wide", page_title="Superdeck (Supabase)")

# -----------------------
# Supabase Configuration
# -----------------------
@st.cache_resource
def init_supabase():
    SUPABASE_URL = st.secrets.get("https://nyeolmhfbuomnnphcrdm.supabase.co"
    SUPABASE_KEY = st.secrets.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im55ZW9sbWhmYnVvbW5ucGhjcmRtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMwMjY0NzEsImV4cCI6MjA3ODYwMjQ3MX0.3yAN1VTGWhJFy5y5Bn5vhcDTjp3-grjr7cXGpxGXR-E", "your_supabase_key_here")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------
# Data Loading from Supabase
# -----------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_supabase_data(date_basis: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    supabase_client = init_supabase()
    
    # Convert dates to string format for Supabase
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Query based on selected date basis
    if date_basis == 'TRN_DATE':
        response = supabase_client.table('DAILY_POS_TRN_ITEMS')\
            .select('*')\
            .gte('TRN_DATE', start_date_str)\
            .lte('TRN_DATE', end_date_str)\
            .execute()
    else:  # ZED_DATE
        response = supabase_client.table('DAILY_POS_TRN_ITEMS')\
            .select('*')\
            .gte('ZED_DATE', start_date_str)\
            .lte('ZED_DATE', end_date_str)\
            .execute()
    
    return pd.DataFrame(response.data)

# -----------------------
# Date Range Selector
# -----------------------
def setup_date_filters():
    st.sidebar.markdown("### Date Configuration")
    
    # Date basis selection
    date_basis = st.sidebar.radio(
        "Date Basis",
        ['TRN_DATE', 'ZED_DATE'],
        help="Choose which date column to use for filtering"
    )
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=7),
            max_value=datetime.now().date()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            max_value=datetime.now().date()
        )
    
    # Validate date range
    if start_date > end_date:
        st.sidebar.error("Error: End date must be after start date.")
        return None, None, None
    
    return date_basis, start_date, end_date

# -----------------------
# Robust cleaning + derived columns (updated for multi-day)
# -----------------------
@st.cache_data
def clean_and_derive(df: pd.DataFrame, date_basis: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    
    d = df.copy()

    # Normalize string columns
    str_cols = [
        'STORE_CODE','TILL','SESSION','RCT','STORE_NAME','CASHIER','ITEM_CODE',
        'ITEM_NAME','DEPARTMENT','CATEGORY','CU_DEVICE_SERIAL','CAP_CUSTOMER_CODE',
        'LOYALTY_CUSTOMER_CODE','SUPPLIER_NAME','SALES_CHANNEL_L1','SALES_CHANNEL_L2','SHIFT'
    ]
    for c in str_cols:
        if c in d.columns:
            d[c] = d[c].fillna('').astype(str).str.strip()

    # Dates - handle both TRN_DATE and ZED_DATE
    for date_col in ['TRN_DATE', 'ZED_DATE']:
        if date_col in d.columns:
            d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
    
    # Use the selected date basis as primary date
    primary_date_col = date_basis
    d = d.dropna(subset=[primary_date_col]).copy()
    d['DATE'] = d[primary_date_col].dt.date
    d['TIME_INTERVAL'] = d[primary_date_col].dt.floor('30min')
    d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time
    d['DAY_OF_WEEK'] = d[primary_date_col].dt.day_name()
    d['WEEK_NUMBER'] = d[primary_date_col].dt.isocalendar().week

    # Numeric parsing
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(
                d[c].astype(str).str.replace(',', '', regex=False).str.strip(),
                errors='coerce'
            ).fillna(0)

    # GROSS_SALES
    if 'GROSS_SALES' not in d.columns:
        d['GROSS_SALES'] = d.get('NET_SALES', 0) + d.get('VAT_AMT', 0)

    # CUST_CODE
    if all(col in d.columns for col in ['STORE_CODE','TILL','SESSION','RCT']):
        d['CUST_CODE'] = (
            d['STORE_CODE'].astype(str) + '-' +
            d['TILL'].astype(str) + '-' +
            d['SESSION'].astype(str) + '-' +
            d['RCT'].astype(str)
        )
    else:
        if 'CUST_CODE' not in d.columns:
            d['CUST_CODE'] = ''

    # Till_Code
    if 'TILL' in d.columns and 'STORE_CODE' in d.columns:
        d['Till_Code'] = d['TILL'].astype(str) + '-' + d['STORE_CODE'].astype(str)

    # CASHIER-COUNT
    if 'STORE_NAME' in d.columns and 'CASHIER' in d.columns:
        d['CASHIER-COUNT'] = d['CASHIER'].astype(str) + '-' + d['STORE_NAME'].astype(str)

    # Shift bucket
    if 'SHIFT' in d.columns:
        d['Shift_Bucket'] = np.where(
            d['SHIFT'].str.upper().str.contains('NIGHT', na=False),
            'Night',
            'Day'
        )

    if 'SP_PRE_VAT' in d.columns:
        d['SP_PRE_VAT'] = d['SP_PRE_VAT'].astype(float)
    if 'NET_SALES' in d.columns:
        d['NET_SALES'] = d['NET_SALES'].astype(float)

    return d

# -----------------------
# Enhanced Aggregation Helpers with Date Trends
# -----------------------
@st.cache_data
def agg_net_sales_by_with_trend(df: pd.DataFrame, col: str, date_col: str = 'DATE') -> tuple:
    """Returns both daily aggregate and trend data"""
    if col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    
    # Daily aggregation
    daily_g = df.groupby([date_col, col], as_index=False)['NET_SALES'].sum()
    
    # Total aggregation
    total_g = df.groupby(col, as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
    
    return total_g, daily_g

@st.cache_data
def create_trend_chart(daily_data, x_col, y_col, title, color_col=None):
    """Create a trend line chart with daily data"""
    if daily_data.empty:
        return go.Figure()
    
    if color_col:
        fig = px.line(daily_data, x=x_col, y=y_col, color=color_col, 
                     title=title, markers=True)
    else:
        fig = px.line(daily_data, x=x_col, y=y_col, title=title, markers=True)
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=y_col,
        hovermode='x unified'
    )
    return fig

# -----------------------
# Table formatting helper (unchanged)
# -----------------------
def format_and_display(df: pd.DataFrame, numeric_cols: list | None = None,
                       index_col: str | None = None, total_label: str = 'TOTAL'):
    if df is None or df.empty:
        st.dataframe(df)
        return

    df_display = df.copy()

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

    if index_col and index_col in df_display.columns:
        label_col = index_col
    else:
        non_numeric_cols = [c for c in df_display.columns if c not in numeric_cols]
        label_col = non_numeric_cols[0] if non_numeric_cols else df_display.columns[0]

    totals[label_col] = total_label

    tot_df = pd.DataFrame([totals], columns=df_display.columns)
    appended = pd.concat([df_display, tot_df], ignore_index=True)

    for col in numeric_cols:
        if col in appended.columns:
            series_vals = appended[col].dropna()
            try:
                series_vals = series_vals.astype(float)
            except Exception:
                continue
            is_int_like = len(series_vals) > 0 and np.allclose(
                series_vals.fillna(0).round(0),
                series_vals.fillna(0)
            )
            if is_int_like:
                appended[col] = appended[col].map(
                    lambda v: f"{int(v):,}" if pd.notna(v) and str(v) != '' else ''
                )
            else:
                appended[col] = appended[col].map(
                    lambda v: f"{float(v):,.2f}" if pd.notna(v) and str(v) != '' else ''
                )

    st.dataframe(appended, use_container_width=True)

# -----------------------
# Enhanced SALES functions with trends
# -----------------------
def sales_global_overview(df, date_basis):
    st.header("Global Sales Overview")
    
    if 'SALES_CHANNEL_L1' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SALES_CHANNEL_L1 or NET_SALES")
        return
        
    # Get aggregated data with trends
    total_g, daily_g = agg_net_sales_by_with_trend(df, 'SALES_CHANNEL_L1')
    
    if total_g.empty:
        st.warning("No data available for the selected period")
        return
    
    # Display trend chart
    st.subheader("Daily Sales Trend by Channel")
    trend_fig = create_trend_chart(daily_g, 'DATE', 'NET_SALES', 
                                  "Daily Net Sales by Channel", 'SALES_CHANNEL_L1')
    st.plotly_chart(trend_fig, use_container_width=True)
    
    # Display donut chart for total period
    total_g['NET_SALES_M'] = total_g['NET_SALES'] / 1_000_000
    
    fig = go.Figure(data=[go.Pie(
        labels=total_g['SALES_CHANNEL_L1'],
        values=total_g['NET_SALES'],
        hole=0.65,
        hovertemplate='<b>%{label}</b><br>KSh %{value:,.2f} M<extra></extra>'
    )])
    fig.update_layout(title="<b>SALES CHANNEL TYPE — Selected Period Overview</b>")
    st.plotly_chart(fig, use_container_width=True)
    
    format_and_display(
        total_g[['SALES_CHANNEL_L1', 'NET_SALES']],
        numeric_cols=['NET_SALES'],
        index_col='SALES_CHANNEL_L1',
        total_label='TOTAL'
    )

def sales_by_channel_l2(df, date_basis):
    st.header("Global Net Sales Distribution by Sales Channel")
    
    if 'SALES_CHANNEL_L2' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SALES_CHANNEL_L2 or NET_SALES")
        return
        
    total_g, daily_g = agg_net_sales_by_with_trend(df, 'SALES_CHANNEL_L2')
    
    if total_g.empty:
        st.warning("No data available for the selected period")
        return
    
    # Display trend
    st.subheader("Daily Sales Trend by Sales Mode")
    trend_fig = create_trend_chart(daily_g, 'DATE', 'NET_SALES', 
                                  "Daily Net Sales by Sales Mode", 'SALES_CHANNEL_L2')
    st.plotly_chart(trend_fig, use_container_width=True)
    
    total_g['NET_SALES_M'] = total_g['NET_SALES'] / 1_000_000
    
    fig = go.Figure(data=[go.Pie(
        labels=total_g['SALES_CHANNEL_L2'],
        values=total_g['NET_SALES'],
        hole=0.65,
        hovertemplate='<b>%{label}</b><br>KSh %{value:,.2f} M<extra></extra>'
    )])
    fig.update_layout(title="<b>Global Net Sales Distribution by Sales Mode (SALES_CHANNEL_L2)</b>")
    st.plotly_chart(fig, use_container_width=True)
    
    format_and_display(
        total_g[['SALES_CHANNEL_L2', 'NET_SALES']],
        numeric_cols=['NET_SALES'],
        index_col='SALES_CHANNEL_L2',
        total_label='TOTAL'
    )

def sales_by_shift(df, date_basis):
    st.header("Global Net Sales Distribution by SHIFT")
    
    if 'SHIFT' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SHIFT or NET_SALES")
        return
        
    # Daily trend by shift
    daily_shift = df.groupby(['DATE', 'SHIFT'], as_index=False)['NET_SALES'].sum()
    
    # Total aggregation
    total_g = df.groupby('SHIFT', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
    total_g['PCT'] = 100 * total_g['NET_SALES'] / total_g['NET_SALES'].sum()
    
    # Display trend
    st.subheader("Daily Sales Trend by Shift")
    trend_fig = create_trend_chart(daily_shift, 'DATE', 'NET_SALES', 
                                  "Daily Net Sales by Shift", 'SHIFT')
    st.plotly_chart(trend_fig, use_container_width=True)
    
    labels = [f"{row['SHIFT']} ({row['PCT']:.1f}%)" for _, row in total_g.iterrows()]
    fig = go.Figure(data=[go.Pie(labels=labels, values=total_g['NET_SALES'], hole=0.65)])
    fig.update_layout(title="<b>Global Net Sales Distribution by SHIFT</b>")
    st.plotly_chart(fig, use_container_width=True)
    
    format_and_display(
        total_g[['SHIFT', 'NET_SALES', 'PCT']],
        numeric_cols=['NET_SALES', 'PCT'],
        index_col='SHIFT',
        total_label='TOTAL'
    )

def night_vs_day_ratio(df, date_basis):
    st.header("Night vs Day Shift Sales Ratio — Stores with Night Shifts")
    
    if 'Shift_Bucket' not in df.columns or 'STORE_NAME' not in df.columns:
        st.warning("Missing Shift_Bucket or STORE_NAME")
        return
        
    stores_with_night = df[df['Shift_Bucket'] == 'Night']['STORE_NAME'].unique()
    df_nd = df[df['STORE_NAME'].isin(stores_with_night)].copy()
    
    if df_nd.empty:
        st.info("No stores with NIGHT shift found in selected period")
        return
        
    ratio = df_nd.groupby(['STORE_NAME', 'Shift_Bucket'])['NET_SALES'].sum().reset_index()
    ratio['STORE_TOTAL'] = ratio.groupby('STORE_NAME')['NET_SALES'].transform('sum')
    ratio['PCT'] = 100 * ratio['NET_SALES'] / ratio['STORE_TOTAL']
    
    pivot = ratio.pivot(index='STORE_NAME', columns='Shift_Bucket', values='PCT').fillna(0)
    
    if pivot.empty:
        st.info("No stores with NIGHT shift found")
        return
        
    pivot_sorted = pivot.sort_values('Night', ascending=False)
    numbered_labels = [f"{i+1}. {s}" for i, s in enumerate(pivot_sorted.index)]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pivot_sorted['Night'],
        y=numbered_labels,
        orientation='h',
        name='Night',
        marker_color='#d62728',
        text=[f"{v:.1f}%" for v in pivot_sorted['Night']],
        textposition='inside'
    ))
    
    for i, (n_val, d_val) in enumerate(zip(pivot_sorted['Night'], pivot_sorted['Day'])):
        fig.add_annotation(
            x=n_val + 1,
            y=numbered_labels[i],
            text=f"{d_val:.1f}% Day",
            showarrow=False,
            xanchor='left'
        )
        
    fig.update_layout(
        title="Night vs Day Shift Sales Ratio — Stores with Night Shifts",
        xaxis_title="% of Store Sales",
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)
    
    table = pivot_sorted.reset_index().rename(columns={'Night': 'Night_%', 'Day': 'Day_%'})
    format_and_display(
        table,
        numeric_cols=['Night_%', 'Day_%'],
        index_col='STORE_NAME',
        total_label='TOTAL'
    )

# -----------------------
# Enhanced OPERATIONS functions with trends
# -----------------------
def customer_traffic_storewise(df, date_basis):
    st.header("Customer Traffic Heatmap — Storewise (30-min slots, deduped)")

    if 'DATE' not in df.columns or 'STORE_NAME' not in df.columns:
        st.warning("Missing DATE or STORE_NAME — cannot compute traffic.")
        return

    d = df.copy()
    
    # Build/ensure CUST_CODE
    if 'CUST_CODE' in d.columns and d['CUST_CODE'].astype(str).str.strip().astype(bool).any():
        d['CUST_CODE'] = d['CUST_CODE'].astype(str).str.strip()
    else:
        required_parts = ['STORE_CODE', 'TILL', 'SESSION', 'RCT']
        if not all(c in d.columns for c in required_parts):
            st.warning("Missing CUST_CODE and/or its components (STORE_CODE, TILL, SESSION, RCT).")
            return
        for col in required_parts:
            d[col] = d[col].astype(str).fillna('').str.strip()
        d['CUST_CODE'] = d['STORE_CODE'] + '-' + d['TILL'] + '-' + d['SESSION'] + '-' + d['RCT']

    # Get first touch per customer per day
    first_touch = (
        d.groupby(['STORE_NAME', 'DATE', 'CUST_CODE'], as_index=False)['TIME_INTERVAL']
         .min()
    )
    first_touch['TIME_ONLY'] = first_touch['TIME_INTERVAL'].dt.time

    # 30-min grid
    start_time = pd.Timestamp("00:00:00")
    intervals = [(start_time + timedelta(minutes=30 * i)).time() for i in range(48)]
    col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]

    # Aggregate across all dates in range
    counts = (
        first_touch.groupby(['STORE_NAME', 'TIME_ONLY'])['CUST_CODE']
                   .nunique()
                   .reset_index(name='RECEIPT_COUNT')
    )
    
    if counts.empty:
        st.info("No customer traffic data to display.")
        return

    heatmap = counts.pivot(index='STORE_NAME', columns='TIME_ONLY',
                           values='RECEIPT_COUNT').fillna(0)

    for t in intervals:
        if t not in heatmap.columns:
            heatmap[t] = 0
    heatmap = heatmap[intervals]

    heatmap['TOTAL'] = heatmap.sum(axis=1)
    heatmap = heatmap.sort_values('TOTAL', ascending=False)

    totals = heatmap['TOTAL'].astype(int).copy()
    heatmap_matrix = heatmap.drop(columns=['TOTAL'])

    if heatmap_matrix.empty:
        st.info("No customer traffic data to display.")
        return

    colorscale = [
        [0.0,   '#E6E6E6'],
        [0.001, '#FFFFCC'],
        [0.25,  '#FED976'],
        [0.50,  '#FEB24C'],
        [0.75,  '#FD8D3C'],
        [1.0,   '#E31A1C']
    ]

    z = heatmap_matrix.values
    zmax = float(z.max()) if z.size else 1.0
    if zmax <= 0:
        zmax = 1.0

    fig = px.imshow(
        z,
        x=col_labels,
        y=heatmap_matrix.index,
        text_auto=True,
        aspect='auto',
        color_continuous_scale=colorscale,
        zmin=0,
        zmax=zmax,
        labels=dict(
            x="Time Interval (30 min)",
            y="Store Name",
            color="Receipts"
        )
    )

    fig.update_xaxes(side='top')

    # totals annotation
    for i, total in enumerate(totals):
        fig.add_annotation(
            x=-0.6,
            y=i,
            text=f"{total:,}",
            showarrow=False,
            xanchor='right',
            yanchor='middle',
            font=dict(size=11, color='black')
        )
    fig.add_annotation(
        x=-0.6,
        y=-1,
        text="<b>TOTAL</b>",
        showarrow=False,
        xanchor='right',
        yanchor='top',
        font=dict(size=12, color='black')
    )

    fig.update_layout(
        title="Customer Traffic Heatmap (Aggregated Period)",
        xaxis_title="Time of Day",
        yaxis_title="Store Name",
        height=max(600, 25 * len(heatmap_matrix.index)),
        margin=dict(l=185, r=20, t=85, b=45),
        coloraxis_colorbar=dict(title="Receipt Count")
    )

    st.plotly_chart(fig, use_container_width=True)

    # Daily trend of customer traffic
    st.subheader("Daily Customer Traffic Trend")
    daily_traffic = first_touch.groupby('DATE')['CUST_CODE'].nunique().reset_index()
    daily_traffic.columns = ['DATE', 'Unique_Customers']
    
    trend_fig = px.line(daily_traffic, x='DATE', y='Unique_Customers', 
                       title="Daily Unique Customers Trend", markers=True)
    st.plotly_chart(trend_fig, use_container_width=True)

    totals_df = totals.reset_index()
    totals_df.columns = ['STORE_NAME', 'Total_Receipts']
    st.subheader("Storewise Total Receipts (Deduped)")
    format_and_display(
        totals_df,
        numeric_cols=['Total_Receipts'],
        index_col='STORE_NAME',
        total_label='TOTAL'
    )

def cashiers_performance(df, date_basis):
    st.header("Cashiers Performance")

    if 'DATE' not in df.columns:
        st.warning("Missing DATE")
        return

    d = df.copy()

    # Ensure identifiers exist
    required_id_cols = ['STORE_CODE', 'TILL', 'SESSION', 'RCT', 'CASHIER', 'ITEM_CODE']
    missing = [c for c in required_id_cols if c not in d.columns]
    if missing:
        st.warning(f"Missing column(s) in dataset: {missing}")
        return
        
    for c in required_id_cols + ['STORE_NAME']:
        if c in d.columns:
            d[c] = d[c].astype(str).fillna('').str.strip()

    # Build CUST_CODE if not present
    if 'CUST_CODE' not in d.columns:
        d['CUST_CODE'] = d['STORE_CODE'] + '-' + d['TILL'] + '-' + d['SESSION'] + '-' + d['RCT']
    else:
        d['CUST_CODE'] = d['CUST_CODE'].astype(str).fillna('').str.strip()

    # Create unique cashier per store
    if 'CASHIER-COUNT' not in d.columns:
        d['CASHIER-COUNT'] = d['STORE_NAME'] + '-' + d['CASHIER']

    # Receipt-level duration and item count
    receipt_duration = (
        d.groupby(['STORE_NAME', 'CUST_CODE', 'DATE'], as_index=False)
         .agg(Start_Time=('TIME_INTERVAL', 'min'),
              End_Time=('TIME_INTERVAL', 'max'))
    )
    receipt_duration['Duration_Sec'] = (
        receipt_duration['End_Time'] - receipt_duration['Start_Time']
    ).dt.total_seconds().fillna(0)

    receipt_items = (
        d.groupby(['STORE_NAME', 'CUST_CODE', 'DATE'], as_index=False)['ITEM_CODE']
         .nunique()
         .rename(columns={'ITEM_CODE': 'Unique_Items'})
    )

    receipt_stats = pd.merge(
        receipt_duration, receipt_items,
        on=['STORE_NAME', 'CUST_CODE', 'DATE'], how='left'
    )

    # Store-level summary with daily trends
    store_daily = (
        receipt_stats.groupby(['STORE_NAME', 'DATE'])
        .agg(
            Avg_Time_per_Customer_Min=('Duration_Sec', lambda s: s.mean() / 60),
            Avg_Items_per_Receipt=('Unique_Items', 'mean')
        )
        .reset_index()
    )
    
    store_summary = (
        receipt_stats.groupby('STORE_NAME')
        .agg(
            Total_Customers=('CUST_CODE', 'nunique'),
            Avg_Time_per_Customer_Min=('Duration_Sec', lambda s: s.mean() / 60),
            Avg_Items_per_Receipt=('Unique_Items', 'mean')
        )
        .reset_index()
    )
    
    store_summary['Avg_Time_per_Customer_Min'] = store_summary['Avg_Time_per_Customer_Min'].round(1)
    store_summary['Avg_Items_per_Receipt'] = store_summary['Avg_Items_per_Receipt'].round(1)
    store_summary = store_summary.sort_values('Avg_Time_per_Customer_Min', ascending=True).reset_index(drop=True)
    store_summary.index = np.arange(1, len(store_summary) + 1)
    store_summary.index.name = '#'

    # Display daily trends
    st.subheader("Daily Performance Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        trend_fig1 = create_trend_chart(store_daily, 'DATE', 'Avg_Time_per_Customer_Min', 
                                      "Daily Avg Time per Customer (min)", 'STORE_NAME')
        st.plotly_chart(trend_fig1, use_container_width=True)
    
    with col2:
        trend_fig2 = create_trend_chart(store_daily, 'DATE', 'Avg_Items_per_Receipt', 
                                      "Daily Avg Items per Receipt", 'STORE_NAME')
        st.plotly_chart(trend_fig2, use_container_width=True)

    # Cashier-level analysis (same as before but with date context)
    merged_for_duration = d.merge(
        receipt_stats[['STORE_NAME', 'CUST_CODE', 'DATE', 'Duration_Sec']],
        on=['STORE_NAME', 'CUST_CODE', 'DATE'], how='left'
    )
    
    cashier_durations = (
        merged_for_duration
        .groupby(['STORE_NAME', 'CASHIER-COUNT'], as_index=False)
        .agg(
            Avg_Duration_Sec=('Duration_Sec', 'mean'),
            Customers_Served=('CUST_CODE', 'nunique')
        )
    )
    cashier_durations['Avg_Serve_Min'] = (cashier_durations['Avg_Duration_Sec'] / 60.0).round(1)

    # Branch selection for cashier details
    branches = sorted(store_summary['STORE_NAME'].unique().tolist())
    if not branches:
        st.info("No branches found.")
        return

    branch_data = {
        b: cashier_durations[cashier_durations['STORE_NAME'] == b].sort_values('Avg_Serve_Min')
        for b in branches
    }
    
    init_branch = branches[0]
    df_branch = branch_data[init_branch].copy()
    df_branch['Label_Text'] = (
        df_branch['Avg_Serve_Min'].astype(str) + ' min (' +
        df_branch['Customers_Served'].astype(str) + ' customers)'
    )

    fig = px.bar(
        df_branch,
        x='Avg_Serve_Min',
        y='CASHIER-COUNT',
        orientation='h',
        text='Label_Text',
        color='Avg_Serve_Min',
        color_continuous_scale='Blues',
        title=f"🕒 Avg Serving Time per Cashier — {init_branch}",
        labels={'Avg_Serve_Min': 'Avg Time per Customer (min)', 'CASHIER-COUNT': 'Cashier'}
    )
    fig.update_traces(textposition='outside', textfont=dict(size=10))
    fig.update_layout(
        xaxis_title="Average Serving Time (minutes)",
        yaxis_title="Cashier",
        coloraxis_showscale=False,
        height=max(500, 25 * len(df_branch))
    )

    # Dropdown to switch branches
    buttons = []
    for b in branches:
        dfb = branch_data[b].copy()
        dfb['Label_Text'] = (
            dfb['Avg_Serve_Min'].astype(str) + ' min (' +
            dfb['Customers_Served'].astype(str) + ' customers)'
        )
        buttons.append(dict(
            label=b,
            method='update',
            args=[{
                'x': [dfb['Avg_Serve_Min']],
                'y': [dfb['CASHIER-COUNT']],
                'text': [dfb['Label_Text']],
                'marker': {'color': dfb['Avg_Serve_Min'], 'colorscale': 'Blues'}
            }, {
                'title': f"🕒 Avg Serving Time per Cashier — {b}",
                'height': max(500, 25 * len(dfb))
            }]
        ))

    fig.update_layout(
        updatemenus=[dict(
            type='dropdown',
            x=0, xanchor='left',
            y=1.15, yanchor='top',
            buttons=buttons,
            showactive=True
        )]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display store summary table
    st.subheader("Store Performance Summary")
    format_and_display(
        store_summary.reset_index(),
        numeric_cols=['Total_Customers', 'Avg_Time_per_Customer_Min', 'Avg_Items_per_Receipt'],
        index_col=None,
        total_label='TOTAL'
    )

# -----------------------
# Enhanced INSIGHTS functions with trends
# -----------------------
def customer_baskets_overview(df, date_basis):
    st.header("Customer Baskets Overview")
    
    d = df.copy()
    d = d.dropna(subset=['ITEM_NAME', 'CUST_CODE', 'STORE_NAME', 'DEPARTMENT'])
    d = d[~d['DEPARTMENT'].str.upper().eq('LUGGAGE & BAGS')]
    
    branches = sorted(d['STORE_NAME'].unique())
    branch = st.selectbox("Branch for comparison (branch vs global)", branches)
    metric = st.selectbox("Metric", ['QTY', 'NET_SALES'])
    top_x = st.number_input("Top X", min_value=5, max_value=200, value=10)
    departments = sorted(d['DEPARTMENT'].unique())
    selected_depts = st.multiselect(
        "Departments (empty = all)",
        options=departments,
        default=None
    )
    
    temp = d.copy()
    if selected_depts:
        temp = temp[temp['DEPARTMENT'].isin(selected_depts)]
        
    # Daily trend of top items
    st.subheader("Daily Top Items Trend")
    daily_top = temp.groupby(['DATE', 'ITEM_NAME'])[['QTY', 'NET_SALES']].sum().reset_index()
    daily_top = daily_top.sort_values(['DATE', metric], ascending=[True, False])
    
    # Get top items for the entire period
    basket_count = temp.groupby('ITEM_NAME')['CUST_CODE'].nunique().rename('Count_of_Baskets')
    agg_data = temp.groupby('ITEM_NAME')[['QTY', 'NET_SALES']].sum()
    
    global_top = (
        basket_count.to_frame()
        .join(agg_data)
        .reset_index()
        .sort_values(metric, ascending=False)
        .head(int(top_x))
    )
    global_top.insert(0, '#', range(1, len(global_top) + 1))
    
    # Show trend for top 5 items
    top_5_items = global_top.head(5)['ITEM_NAME'].tolist()
    daily_top_5 = daily_top[daily_top['ITEM_NAME'].isin(top_5_items)]
    
    if not daily_top_5.empty:
        trend_fig = create_trend_chart(daily_top_5, 'DATE', metric, 
                                      f"Daily {metric} Trend for Top 5 Items", 'ITEM_NAME')
        st.plotly_chart(trend_fig, use_container_width=True)

    st.subheader("Global Top Items")
    format_and_display(
        global_top.reset_index(drop=True),
        numeric_cols=['Count_of_Baskets', 'QTY', 'NET_SALES'],
        index_col='ITEM_NAME',
        total_label='TOTAL'
    )
    
    # Branch analysis
    branch_df = temp[temp['STORE_NAME'] == branch]
    if branch_df.empty:
        st.info("No data for selected branch")
        return
        
    basket_count_b = branch_df.groupby('ITEM_NAME')['CUST_CODE'].nunique().rename('Count_of_Baskets')
    agg_b = branch_df.groupby('ITEM_NAME')[['QTY', 'NET_SALES']].sum()
    
    branch_top = (
        basket_count_b.to_frame()
        .join(agg_b)
        .reset_index()
        .sort_values(metric, ascending=False)
        .head(int(top_x))
    )
    branch_top.insert(0, '#', range(1, len(branch_top) + 1))
    
    st.subheader(f"{branch} Top Items")
    format_and_display(
        branch_top.reset_index(drop=True),
        numeric_cols=['Count_of_Baskets', 'QTY', 'NET_SALES'],
        index_col='ITEM_NAME',
        total_label='TOTAL'
    )

def global_category_overview_sales(df, date_basis):
    st.header("Global Category Overview — Sales")
    
    if 'CATEGORY' not in df.columns:
        st.warning("Missing CATEGORY")
        return
        
    # Get data with trends
    total_g, daily_g = agg_net_sales_by_with_trend(df, 'CATEGORY')
    
    # Display daily trend
    st.subheader("Daily Sales Trend by Category")
    trend_fig = create_trend_chart(daily_g, 'DATE', 'NET_SALES', 
                                  "Daily Net Sales by Category", 'CATEGORY')
    st.plotly_chart(trend_fig, use_container_width=True)
    
    format_and_display(
        total_g,
        numeric_cols=['NET_SALES'],
        index_col='CATEGORY',
        total_label='TOTAL'
    )
    
    fig = px.bar(
        total_g.head(20),
        x='NET_SALES',
        y='CATEGORY',
        orientation='h',
        title="Top Categories by Net Sales"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Main App with Supabase Integration
# -----------------------
def main():
    st.title("DailyDeck: The Story Behind the Numbers - Supabase Edition")

    # Date filters
    date_basis, start_date, end_date = setup_date_filters()
    
    if date_basis is None:
        st.stop()

    # Load data
    with st.spinner(f"Loading data from {start_date} to {end_date} using {date_basis}..."):
        try:
            raw_df = load_supabase_data(date_basis, start_date, end_date)
            if raw_df.empty:
                st.error("No data found for the selected date range and criteria.")
                st.stop()
                
            df = clean_and_derive(raw_df, date_basis)
            
            # Display data summary
            st.sidebar.markdown("### Data Summary")
            st.sidebar.info(f"**Period:** {start_date} to {end_date}\n"
                          f"**Date Basis:** {date_basis}\n"
                          f"**Total Records:** {len(df):,}\n"
                          f"**Date Range:** {df['DATE'].min()} to {df['DATE'].max()}\n"
                          f"**Stores:** {df['STORE_NAME'].nunique()}")
                          
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

    section = st.sidebar.selectbox(
        "Section",
        ["SALES", "OPERATIONS", "INSIGHTS"]
    )

    # Enhanced function mapping with date_basis parameter
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
        
        sales_functions = {
            sales_items[0]: sales_global_overview,
            sales_items[1]: sales_by_channel_l2,
            sales_items[2]: sales_by_shift,
            sales_items[3]: night_vs_day_ratio,
            # Add other sales functions here with the same pattern
        }
        
        if choice in sales_functions:
            sales_functions[choice](df, date_basis)
        else:
            st.info("Function coming soon - pattern established")

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
        
        ops_functions = {
            ops_items[0]: customer_traffic_storewise,
            ops_items[5]: cashiers_performance,
            # Add other operations functions here
        }
        
        if choice in ops_functions:
            ops_functions[choice](df, date_basis)
        else:
            st.info("Function coming soon - pattern established")

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
        
        insights_functions = {
            ins_items[0]: customer_baskets_overview,
            ins_items[1]: global_category_overview_sales,
            # Add other insights functions here
        }
        
        if choice in insights_functions:
            insights_functions[choice](df, date_basis)
        else:
            st.info("Function coming soon - pattern established")

if __name__ == "__main__":

    main()


