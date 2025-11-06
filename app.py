import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import io

# === Wide sidebar fix & better main output width ===
st.markdown("""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 370px;
        min-width: 340px;
        max-width: 480px;
        padding-right: 18px;
    }
    .block-container {padding-top:1rem;}
    </style>
    """, unsafe_allow_html=True)
st.set_page_config(layout="wide", page_title="Superdeck Analytics Dashboard", initial_sidebar_state="expanded")

st.title("🦸 Superdeck Analytics Dashboard")
st.markdown("> Upload your sales CSV, choose a main section and subsection for live analytics.")

# --- SIDEBAR: Upload block ---
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV (up to 500MB, check server settings)", type="csv")
if uploaded is None:
    st.info("Please upload a dataset to proceed.")
    st.stop()

@st.cache_data(show_spinner=True)
def load_and_prepare(uploaded):
    df = pd.read_csv(uploaded, on_bad_lines='skip', low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # Date columns
    for col in ['TRN_DATE', 'ZED_DATE']: 
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    for nc in numeric_cols:
        if nc in df.columns: df[nc] = pd.to_numeric(df[nc], errors='coerce').fillna(0)
    idcols = ['STORE_CODE', 'TILL', 'SESSION', 'RCT']
    for col in idcols: 
        if col in df.columns: df[col] = df[col].astype(str).fillna('').str.strip()
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
            st.error(f"Missing columns for CUST_CODE: {missing}")
            st.stop()
    df['CUST_CODE'] = df['CUST_CODE'].astype(str).str.strip()
    return df

df = load_and_prepare(uploaded)

@st.cache_data
def get_time_grid():
    start_time = pd.Timestamp("00:00:00")
    intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
    col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
    return intervals, col_labels

intervals, col_labels = get_time_grid()

def download_button(obj, filename, label, use_xlsx=False):
    if use_xlsx:
        towrite = io.BytesIO()
        obj.to_excel(towrite, encoding="utf-8", index=False, engine='openpyxl')
        towrite.seek(0)
        st.download_button(label, towrite, file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.download_button(label, obj.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")

def download_plot(fig, filename):
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600)
        st.download_button("⬇️ Download Plot as PNG", img_bytes, filename=filename, mime="image/png")
    except Exception:
        st.info("Image download is unavailable (likely missing `kaleido`). Table will still download fine.")

main_sections = {
    "SALES": [
        "Global sales Overview",
        "Global Net Sales Distribution by Sales Channel",
        "Global Net Sales Distribution by SHIFT",
        "Night vs Day Shift Sales Ratio — Stores with Night Shifts",
        "Global Day vs Night Sales — Only Stores with NIGHT Shift",
        "2nd-Highest Channel Share",
        "Bottom 30 — 2nd Highest Channel",
        "Stores Sales Summary"
    ],
    "OPERATIONS": [
        "Customer Traffic-Storewise",
        "Active Tills During the day",
        "Average Customers Served per Till",
        "Store Customer Traffic Storewise",
        "Customer Traffic-Departmentwise",
        "Cashiers Perfomance",
        "Till Usage",
        "Tax Compliance"
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
        "Branch Branch Overview",
        "Global Refunds Overview",
        "Branch Refunds Overview"
    ]
}

section = st.sidebar.radio("Main Section", list(main_sections.keys()))
subsection = st.sidebar.selectbox("Subsection", main_sections[section], key="subsection")

st.markdown(f"##### {section} ➔ {subsection}")

# === SALES ===
if section == "SALES":
    # 1. Global sales Overview
    if subsection == "Global sales Overview":
        gs = df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum()
        gs['NET_SALES_M'] = gs['NET_SALES'] / 1_000_000
        gs['PCT'] = (gs['NET_SALES'] / gs['NET_SALES'].sum()) * 100
        labels = [f"{row['SALES_CHANNEL_L1']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f}M)" for _, row in gs.iterrows()]
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=gs['NET_SALES_M'],
            hole=0.57,
            marker=dict(colors=px.colors.qualitative.Plotly),
            text=[f"{p:.1f}%" for p in gs['PCT']],
            textinfo='text',
            sort=False,
        )])
        fig.update_layout(title="SALES CHANNEL TYPE — Global Overview", height=400, margin=dict(t=60))
        col1, col2 = st.columns([2, 2])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(gs, use_container_width=True)
            download_button(gs, "global_sales_overview.csv", "⬇️ Download Table")
            download_plot(fig, "global_sales_overview.png")

    # --- Add other outputs using similar layout ---
    elif subsection == "Global Net Sales Distribution by Sales Channel":
        g2 = df.groupby('SALES_CHANNEL_L2', as_index=False)['NET_SALES'].sum()
        g2['NET_SALES_M'] = g2['NET_SALES']/1_000_000
        g2['PCT'] = g2['NET_SALES']/g2['NET_SALES'].sum()*100
        colors = px.colors.qualitative.Vivid
        labels = [f"{row['SALES_CHANNEL_L2']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f}M)" for _, row in g2.iterrows()]
        fig = go.Figure(go.Pie(
            labels=labels,
            values=g2['NET_SALES_M'],
            hole=0.58,
            marker=dict(colors=colors),
            text=[f"{p:.1f}%" for p in g2['PCT']],
            textinfo='text'
        ))
        fig.update_layout(title="Net Sales by Sales Mode (L2)", height=400, margin=dict(t=60))
        col1, col2 = st.columns([2, 2])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(g2, use_container_width=True)
            download_button(g2, "sales_channel_l2.csv", "⬇️ Download Table")
            download_plot(fig, "sales_channel_l2_pie.png")

    # ...continue all other outputs...

# --- OPERATIONS ---
elif section == "OPERATIONS":
    if subsection == "Customer Traffic-Storewise":
        stores = df["STORE_NAME"].dropna().unique().tolist()
        selected_store = st.selectbox("Select Store", stores)
        dff = df[df["STORE_NAME"]==selected_store].copy()
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        dff = dff.dropna(subset=['TRN_DATE'])
        for c in ["STORE_CODE","TILL","SESSION","RCT"]:
            dff[c] = dff[c].astype(str).fillna('').str.strip()
        dff['CUST_CODE'] = dff['STORE_CODE']+'-'+dff['TILL']+'-'+dff['SESSION']+'-'+dff['RCT']
        dff['TIME_ONLY'] = dff['TRN_DATE'].dt.floor('30T').dt.time
        heat = dff.groupby('TIME_ONLY')['CUST_CODE'].nunique().reindex(intervals, fill_value=0)
        fig = px.bar(x=col_labels, y=heat.values, labels={"x":"Time","y":"Receipts"}, text=heat.values,
                     color_discrete_sequence=['#3192e1'], title=f"Receipts by Time - {selected_store}", height=360)
        col1, col2 = st.columns([2, 2])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(heat, use_container_width=True)
            download_button(heat.reset_index(), "customer_traffic_storewise.csv", "⬇️ Download Table")
            download_plot(fig, "customer_traffic_storewise.png")

    # ...continue all other outputs...

# --- INSIGHTS ---
elif section == "INSIGHTS":
    if subsection == "Branch Comparison":
        branches = sorted(df['STORE_NAME'].dropna().unique().tolist())
        selected_A = st.selectbox("Branch A", branches, key="bc_a")
        selected_B = st.selectbox("Branch B", branches, key="bc_b")
        metric = st.selectbox("Metric", ["QTY","NET_SALES"], key="bc_metric")
        N = st.slider("Top N", 5, 50, 10, key="bc_n")
        dfA = df[df["STORE_NAME"]==selected_A].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N)
        dfB = df[df["STORE_NAME"]==selected_B].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N)
        combA = dfA.copy(); combA['Branch'] = selected_A
        combB = dfB.copy(); combB['Branch'] = selected_B
        both = pd.concat([combA, combB], ignore_index=True)
        fig = px.bar(both, x=metric, y="ITEM_NAME", color="Branch", orientation="h", barmode="group",
                     title=f"Top {N} items: {selected_A} vs {selected_B}", color_discrete_sequence=["#1f77b4","#ff7f0e"], height=450)
        col1, col2 = st.columns([2, 2])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(both, use_container_width=True)
            download_button(both, "branch_comparison.csv", f"⬇️ Download Branch Comparison Table")
            download_plot(fig, "branch_comparison_bar.png")

    # ...continue all other INSIGHTS outputs...

st.sidebar.markdown("---\nSidebar auto-expands for easy selection. All tables and plots can be downloaded. If image download fails, check `kaleido` install.")
