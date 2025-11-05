import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(layout="wide", page_title="Superdeck Analytics", initial_sidebar_state="expanded")
st.title("🦸 Superdeck Analytics Dashboard")
st.markdown("> Upload your sales CSV - all analytics available as tabs and dropdowns.")

# ====== Data Upload and Preprocessing ======
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
if uploaded is None:
    st.info("Please upload a dataset to proceed.")
    st.stop()

@st.cache_data(show_spinner=True)
def load_and_prepare(uploaded):
    df = pd.read_csv(uploaded, on_bad_lines='skip', low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # Dates
    for col in ['TRN_DATE', 'ZED_DATE']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    # Numeric columns
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    for nc in numeric_cols:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors='coerce').fillna(0)
    # CUST_CODE construction
    idcols = ['STORE_CODE', 'TILL', 'SESSION', 'RCT']
    for col in idcols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('').str.strip()
    if 'CUST_CODE' not in df.columns:
        if all(c in df.columns for c in idcols):
            df['CUST_CODE'] = (
                df['STORE_CODE'].str.strip() + '-' +
                df['TILL'].str.strip() + '-' +
                df['SESSION'].str.strip() + '-' +
                df['RCT'].str.strip()
            )
        else:
            missing = [c for c in idcols if c not in df.columns]
            st.error(f"Your data is missing columns required to build CUST_CODE: {missing}. Cannot proceed.")
            st.stop()
    df['CUST_CODE'] = df['CUST_CODE'].astype(str).str.strip()
    return df

df = load_and_prepare(uploaded)

@st.cache_data(show_spinner=False)
def get_time_grid():
    start_time = pd.Timestamp("00:00:00")
    intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
    col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
    return intervals, col_labels

intervals, col_labels = get_time_grid()

tab_names = [
    "Sales Channel L1", "Sales Mode (L2)", "Net Sales by Shift", "Night vs Day/Store", "Day vs Night Pie",
    "2nd Channel Share", "Sales Summary", "Cust Traffic Heatmap", "Till Heatmap", "Custs per Till",
    "Dept/Branch Heatmap", "Tax Compliance", "Top Items", "Branch Compare", "Multi-price SKUs", "Refunds"
]
tabs = st.tabs(tab_names)

# 1. SALES CHANNEL PIE
with tabs[0]:
    st.header("Sales Channel Type (L1) Distribution")
    if "SALES_CHANNEL_L1" in df.columns and "NET_SALES" in df.columns:
        g = df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
        g['NET_SALES_M'] = g['NET_SALES']/1_000_000
        g['PCT'] = g['NET_SALES']/g['NET_SALES'].sum()*100
        fig = go.Figure(go.Pie(
            labels=[f"{row['SALES_CHANNEL_L1']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f}M)" for _,row in g.iterrows()],
            values=g['NET_SALES_M'], 
            hole=0.6,
            marker=dict(colors=px.colors.qualitative.Plotly),
            text=[f"{p:.1f}%" for p in g['PCT']],
            textinfo='text'
        ))
        fig.update_layout(title="Sales Channel Type (L1) - Global", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing SALES_CHANNEL_L1 or NET_SALES.")

# 2. SALES CHANNEL L2
with tabs[1]:
    st.header("Net Sales by Mode (L2)")
    if "SALES_CHANNEL_L2" in df.columns and "NET_SALES" in df.columns:
        g = df.groupby('SALES_CHANNEL_L2', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
        g['NET_SALES_M'] = g['NET_SALES']/1_000_000
        g['PCT'] = g['NET_SALES']/g['NET_SALES'].sum()*100
        fig = go.Figure(go.Pie(
            labels=[f"{row['SALES_CHANNEL_L2']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f}M)" for _,row in g.iterrows()],
            values=g['NET_SALES_M'], 
            hole=0.6,
            marker=dict(colors=px.colors.qualitative.Vivid),
            text=[f"{p:.1f}%" for p in g['PCT']],
            textinfo='text'
        ))
        fig.update_layout(title="Sales Mode (L2)", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing SALES_CHANNEL_L2 or NET_SALES.")

# 3. SALES SHIFT PIE
with tabs[2]:
    st.header("Net Sales by SHIFT")
    if "SHIFT" in df.columns and "NET_SALES" in df.columns:
        g = df.groupby('SHIFT', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
        g['PCT'] = g['NET_SALES']/g['NET_SALES'].sum()*100
        colors = px.colors.qualitative.Bold
        fig = go.Figure(go.Pie(
            labels=[f"{row['SHIFT']} ({row['PCT']:.1f}%)" for _,row in g.iterrows()],
            values=g['NET_SALES'],
            hole=0.6,
            marker=dict(colors=colors),
            text=[f"{p:.1f}%" for p in g['PCT']],
            textinfo='text'
        ))
        fig.update_layout(title="Global Net Sales by SHIFT")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing SHIFT or NET_SALES.")

# 4. Night vs Day Ratio by Store (bars, bright colors, proper x/y)
with tabs[3]:
    st.header("Night vs Day Sales Ratio per Store")
    req = ["SHIFT", "STORE_NAME", "NET_SALES"]
    if all(x in df.columns for x in req):
        night_stores = df[df['SHIFT'].str.upper().str.contains('NIGHT', na=False)]['STORE_NAME'].unique()
        df_nd = df[df['STORE_NAME'].isin(night_stores)].copy()
        df_nd['Shift_Bucket'] = np.where(df_nd['SHIFT'].str.upper().str.contains('NIGHT', na=False),'Night','Day')
        ratio_df = df_nd.groupby(['STORE_NAME','Shift_Bucket'], as_index=False)['NET_SALES'].sum()
        sum_sales = ratio_df.groupby("STORE_NAME")["NET_SALES"].transform("sum")
        ratio_df['PCT'] = 100 * ratio_df['NET_SALES'] / sum_sales
        pivot_df = ratio_df.pivot(index='STORE_NAME', columns='Shift_Bucket', values='PCT').fillna(0)
        pivot_sorted = pivot_df.sort_values('Night', ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pivot_sorted['Night'],
            y=pivot_sorted.index,
            orientation='h',
            name='Night',
            marker_color='#d62728',
            text=[f"{v:.1f}%" for v in pivot_sorted['Night']],
            textposition='auto'
        ))
        fig.add_trace(go.Bar(
            x=pivot_sorted['Day'],
            y=pivot_sorted.index,
            orientation='h',
            name='Day',
            marker_color='#1f77b4',
            text=[f"{v:.1f}%" for v in pivot_sorted['Day']],
            textposition='auto'
        ))
        fig.update_layout(barmode='group', title="Night vs Day Sales % (by Store)", xaxis_title="% of Store Sales", yaxis_title="Store", height=550)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"One or more of these columns missing: {req}")

# 5. Day vs Night Pie (Global)
with tabs[4]:
    st.header("Global Day vs Night (NIGHT shift stores)")
    req = ["SHIFT", "STORE_NAME", "NET_SALES"]
    if all(x in df.columns for x in req):
        night_stores = df[df['SHIFT'].str.upper().str.contains('NIGHT', na=False)]['STORE_NAME'].unique()
        df_nd = df[df['STORE_NAME'].isin(night_stores)].copy()
        df_nd['Shift_Bucket'] = np.where(df_nd['SHIFT'].str.upper().str.contains('NIGHT', na=False),'Night','Day')
        gb = df_nd.groupby('Shift_Bucket', as_index=False)['NET_SALES'].sum()
        gb['PCT'] = 100 * gb['NET_SALES'] / gb['NET_SALES'].sum()
        colors = ['#1f77b4', '#d62728']
        fig = go.Figure(go.Pie(
            labels=[f"{b} ({p:.1f}%)" for b, p in zip(gb['Shift_Bucket'], gb['PCT'])],
            values=gb['NET_SALES'], hole=0.6, marker=dict(colors=colors),
            text=[f"{p:.1f}%" for p in gb['PCT']], textinfo='text'))
        fig.update_layout(title="Day vs Night Sales (NIGHT shift stores)")
        st.plotly_chart(fig, use_container_width=True)

# Remaining tabs omitted for brevity. You can apply the same color/axis logic as above to all visuals.

st.sidebar.markdown("---\nBuilt with ❤️ using Streamlit\nContact: Jobcheruyot")
