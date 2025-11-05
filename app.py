import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import textwrap
from datetime import timedelta

st.set_page_config(layout="wide")
st.title("🦸 Superdeck Streamlit Analytics")
st.markdown("Upload your sales/operations CSV below. Use the tabs to switch insights.")

# === Data Upload ===
uploaded = st.sidebar.file_uploader("Upload your CSV", type="csv")
if uploaded is None:
    st.info("Please upload a CSV file to use the dashboard.")
    st.stop()

df = pd.read_csv(uploaded, on_bad_lines='skip', low_memory=False)

# === Core Data Cleaning ===
date_cols = ['TRN_DATE', 'ZED_DATE']
for dc in date_cols:
    if dc in df.columns:
        df[dc] = pd.to_datetime(df[dc], errors='coerce')
numeric_cols = ['QTY','CP_PRE_VAT','SP_PRE_VAT','COST_PRE_VAT','NET_SALES','VAT_AMT']
for nc in numeric_cols:
    if nc in df.columns:
        df[nc] = pd.to_numeric(df[nc], errors='coerce').fillna(0)

tab_titles = [
    "Sales Channel Overview",
    "Sales Mode Dist.",
    "Net Sales by Shift",
    "Night vs Day Ratio",
    "Day v Night (Stores)",
    "2nd Channel Share",
    "Sales Workings",
    "Store Sales Summary",
    "Customer Traffic",
    "Till Activity",
    "Customers per Till",
    # Add more as needed
]
tabs = st.tabs(tab_titles)

# ==== Tab 0: Sales Channel Type Overview ====
with tabs[0]:
    st.header("Sales Channel Type — Global Overview")
    if not all(x in df.columns for x in ["SALES_CHANNEL_L1", "NET_SALES"]):
        st.warning("File must include SALES_CHANNEL_L1 and NET_SALES columns.")
    else:
        grouped = df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
        grouped['NET_SALES_M'] = grouped['NET_SALES'] / 1_000_000
        grouped['PCT'] = grouped['NET_SALES'] / grouped['NET_SALES'].sum() * 100
        labels = [f"{row['SALES_CHANNEL_L1']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f} M)" for _, row in grouped.iterrows()]
        fig = go.Figure(go.Pie(
            labels=labels,
            values=grouped['NET_SALES_M'],
            hole=0.65,
            text=[f"{p:.1f}%" for p in grouped['PCT']],
            textinfo='text',
            marker=dict(colors=px.colors.qualitative.Plotly)
        ))
        fig.update_layout(title="Global Sales by Channel (L1)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(grouped)

# ==== Tab 1: Sales Channel L2 ====
with tabs[1]:
    st.header("Global Net Sales Distribution by Sales Mode (L2)")
    if not all(x in df.columns for x in ["SALES_CHANNEL_L2", "NET_SALES"]):
        st.warning("File must include SALES_CHANNEL_L2 and NET_SALES columns.")
    else:
        grouped = df.groupby('SALES_CHANNEL_L2', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
        grouped['NET_SALES_M'] = grouped['NET_SALES'] / 1_000_000
        grouped['PCT'] = grouped['NET_SALES'] / grouped['NET_SALES'].sum() * 100
        labels = [f"{row['SALES_CHANNEL_L2']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f} M)" for _, row in grouped.iterrows()]
        fig = go.Figure(go.Pie(
            labels=labels,
            values=grouped['NET_SALES_M'],
            hole=0.65,
            text=[f"{p:.1f}%" for p in grouped['PCT']],
            textinfo='text',
            marker=dict(colors=px.colors.qualitative.Plotly)
        ))
        fig.update_layout(title="Net Sales by Channel (L2)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(grouped)

# ==== Tab 2: Net Sales by SHIFT ====
with tabs[2]:
    st.header("Net Sales by SHIFT")
    if not all(x in df.columns for x in ["SHIFT", "NET_SALES"]):
        st.warning("File must include SHIFT and NET_SALES columns.")
    else:
        grouped = df.groupby('SHIFT', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
        grouped['PCT'] = grouped['NET_SALES'] / grouped['NET_SALES'].sum() * 100
        labels = [f"{row['SHIFT']} ({row['PCT']:.1f}%)" for _, row in grouped.iterrows()]
        fig = go.Figure(go.Pie(
            labels=labels,
            values=grouped['NET_SALES'],
            hole=0.65,
            text=[f"{p:.1f}%" for p in grouped['PCT']],
            textinfo='text',
            marker=dict(colors=px.colors.qualitative.Plotly)
        ))
        fig.update_layout(title="Global Net Sales by SHIFT")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(grouped)

# ==== Tab 3: Night vs Day Ratio — Storewise  ====
with tabs[3]:
    st.header("Night vs Day Shift Sales Ratio — Storewise")
    for col in ["SHIFT", "STORE_NAME", "NET_SALES"]:
        if col not in df.columns:
            st.warning(f"File must have {col}")
            st.stop()
    stores_with_night = df[df['SHIFT'].str.upper().str.contains('NIGHT', na=False)]['STORE_NAME'].unique()
    df_nd = df[df['STORE_NAME'].isin(stores_with_night)].copy()
    df_nd['Shift_Bucket'] = np.where(df_nd['SHIFT'].str.upper().str.contains('NIGHT', na=False), 'Night', 'Day')
    ratio_df = (
        df_nd.groupby(["STORE_NAME","Shift_Bucket"], as_index=False)['NET_SALES'].sum())
    store_totals = ratio_df.groupby('STORE_NAME')['NET_SALES'].transform('sum')
    ratio_df['PCT'] = ratio_df['NET_SALES'] / store_totals * 100
    piv = ratio_df.pivot(index="STORE_NAME", columns="Shift_Bucket", values="PCT").fillna(0)
    st.dataframe(piv.style.background_gradient(axis=None))
    # Optionally: plot horizontal bars

# ==== Tab 4: Global Day vs Night — Only Stores with Night Shift ====
with tabs[4]:
    st.header("Global Day vs Night Sales — Only Stores with NIGHT Shift")
    stores_with_night = df[df['SHIFT'].str.upper().str.contains('NIGHT', na=False)]['STORE_NAME'].unique()
    df_nd = df[df['STORE_NAME'].isin(stores_with_night)].copy()
    df_nd['Shift_Bucket'] = np.where(df_nd['SHIFT'].str.upper().str.contains('NIGHT', na=False), 'Night', 'Day')
    global_nd = (df_nd.groupby('Shift_Bucket', as_index=False)['NET_SALES']
               .sum().sort_values('NET_SALES', ascending=False))
    global_nd['PCT'] = 100 * global_nd['NET_SALES'] / global_nd['NET_SALES'].sum()
    labels = [f"{b} ({p:.1f}%)" for b, p in zip(global_nd['Shift_Bucket'], global_nd['PCT'])]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=global_nd['NET_SALES'],
        hole=0.65,
        text=[f"{p:.1f}%" for p in global_nd['PCT']],
        textinfo='text',
        marker=dict(colors=['#1f77b4','#d62728'], line=dict(color='white', width=1)),
        sort=False
    ))
    fig.update_layout(
        title="Global Day vs Night Sales — Only Stores with NIGHT Shifts",
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(global_nd)

# ==== Tab 5: 2nd-Highest Channel Share ====
with tabs[5]:
    st.header("2nd-Highest Channel Share (Top 30 shown)")
    cols_needed = {"STORE_NAME", "SALES_CHANNEL_L1", "NET_SALES"}
    if not cols_needed.issubset(df.columns):
        st.warning(f"Missing columns: {cols_needed - set(df.columns)}")
    else:
        data = df.copy()
        data["NET_SALES"] = pd.to_numeric(data["NET_SALES"], errors="coerce").fillna(0)
        store_chan = data.groupby(["STORE_NAME", "SALES_CHANNEL_L1"], as_index=False)["NET_SALES"].sum()
        store_tot = store_chan.groupby("STORE_NAME")["NET_SALES"].transform("sum")
        store_chan["PCT"] = 100 * store_chan["NET_SALES"] / store_tot
        store_chan = store_chan.sort_values(["STORE_NAME", "PCT"], ascending=[True, False])
        store_chan["RANK"] = store_chan.groupby("STORE_NAME").cumcount() + 1
        second_tbl = store_chan[store_chan["RANK"] == 2][["STORE_NAME", "SALES_CHANNEL_L1", "PCT"]]
        st.dataframe(second_tbl.head(30))

# ==== Tab 6: Sales Workings Section ====
with tabs[6]:
    st.header("Sales Workings Quick View")
    if 'VAT_AMT' in df.columns:
        df['GROSS_SALES'] = df['NET_SALES'] + df['VAT_AMT']
        st.dataframe(df[['NET_SALES','VAT_AMT','GROSS_SALES']].head())
    else:
        st.write("VAT_AMT column not in file, cannot compute gross sales.")

# ==== Tab 7: Store Sales Summary ====
with tabs[7]:
    st.header("Store Sales Summary")
    if 'GROSS_SALES' not in df.columns and 'VAT_AMT' in df.columns:
        df['GROSS_SALES'] = df['NET_SALES'] + df['VAT_AMT']
    if 'STORE_NAME' in df.columns:
        sales_summary = (df.groupby('STORE_NAME', as_index=False)[['NET_SALES','GROSS_SALES']].sum())
        sales_summary['% Contribution'] = (sales_summary['GROSS_SALES'] / sales_summary['GROSS_SALES'].sum() * 100).round(2)
        sales_summary = sales_summary.sort_values('GROSS_SALES', ascending=False)
        st.dataframe(sales_summary)

# ==== Tab 8: Customer Traffic-Storewise (heatmap example) ====
with tabs[8]:
    st.header("Storewise Customer Traffic (30 min buckets)")
    try:
        df['TRN_DATE'] = pd.to_datetime(df['TRN_DATE'], errors='coerce')
        df['TRN_DATE_ONLY'] = df['TRN_DATE'].dt.date
        for col in ['STORE_CODE','TILL','SESSION','RCT']:
            df[col] = df[col].astype(str).fillna('').str.strip()
        df['CUST_CODE'] = df['STORE_CODE'] + '-' + df['TILL'] + '-' + df['SESSION'] + '-' + df['RCT']
        first_touch = (
            df.dropna(subset=['TRN_DATE'])
              .groupby(['STORE_NAME','TRN_DATE_ONLY','CUST_CODE'], as_index=False)['TRN_DATE']
              .min()
        )
        first_touch['TIME_INTERVAL'] = first_touch['TRN_DATE'].dt.floor('30T')
        first_touch['TIME_ONLY'] = first_touch['TIME_INTERVAL'].dt.time
        intervals = [(pd.Timestamp("00:00:00") + timedelta(minutes=30*i)).time() for i in range(48)]
        col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
        counts = (
            first_touch.groupby(['STORE_NAME','TIME_ONLY'])['CUST_CODE']
                       .nunique()
                       .reset_index(name='RECEIPT_COUNT')
        )
        heatmap = counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='RECEIPT_COUNT').fillna(0)
        for t in intervals:
            if t not in heatmap.columns:
                heatmap[t] = 0
        heatmap = heatmap[intervals]
        fig = px.imshow(
            heatmap.values,
            x=col_labels,
            y=heatmap.index,
            text_auto=True,
            color_continuous_scale='RdYlBu',
            zmin=0, zmax=np.max(heatmap.values)
        )
        fig.update_xaxes(side='top')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(heatmap)
    except Exception as ex:
        st.error(f"Error building traffic heatmap: {ex}")

# ==== Tab 9: Till Heatmap ====
with tabs[9]:
    st.header("Peak Active Tills per Store")
    try:
        df['TILL'] = df['TILL'].astype(str).fillna('').str.strip()
        df['STORE_CODE'] = df['STORE_CODE'].astype(str).fillna('').str.strip()
        df['TRN_DATE'] = pd.to_datetime(df['TRN_DATE'], errors='coerce')
        df = df.dropna(subset=['TRN_DATE'])
        df['TIME_INTERVAL'] = df['TRN_DATE'].dt.floor('30T')
        df['TIME_ONLY'] = df['TIME_INTERVAL'].dt.time
        intervals = [(pd.Timestamp("00:00:00") + timedelta(minutes=30*i)).time() for i in range(48)]
        col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
        df['Till_Code'] = df['TILL'] + '-' + df['STORE_CODE']
        till_counts = (
            df.groupby(['STORE_NAME','TIME_ONLY'])['Till_Code']
              .nunique()
              .reset_index(name='UNIQUE_TILLS')
        )
        heatmap = till_counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='UNIQUE_TILLS').fillna(0)
        for t in intervals:
            if t not in heatmap.columns:
                heatmap[t] = 0
        heatmap = heatmap[intervals]
        fig = px.imshow(
            heatmap.values,
            x=col_labels,
            y=heatmap.index,
            text_auto=True,
            color_continuous_scale='YlOrRd',
            zmin=0, zmax=np.max(heatmap.values)
        )
        fig.update_xaxes(side='top')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as ex:
        st.error(f"Till heatmap error: {ex}")

# ==== Tab 10: Customers per Till by Time ====
with tabs[10]:
    st.header("Customers Served per Till")
    # ... Similar processing as the notebook (see full code for logic)
    st.info("Coming soon — see full notebook for details! Add as above.")

st.sidebar.info("Superdeck Streamlit — from Colab/Notebook. [YourName]")
