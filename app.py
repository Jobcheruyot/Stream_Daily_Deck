# Updated Streamlit app with support for larger uploads and chunked CSV parsing.
# Note: To actually allow uploads up to 1024 MB you must set the Streamlit server config:
#   - Environment: export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024
#   - OR .streamlit/config.toml: server.maxUploadSize = 1024
#
# Also ensure the host has enough RAM to hold/process a ~1GB CSV in memory, or
# consider converting to Parquet / using Dask for out-of-core processing.

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import io

# Show current STREAMLIT_SERVER_MAX_UPLOAD_SIZE (MB) if set
env_max_upload_mb = os.environ.get("STREAMLIT_SERVER_MAX_UPLOAD_SIZE")
try:
    env_max_upload_mb_int = int(env_max_upload_mb) if env_max_upload_mb is not None else None
except Exception:
    env_max_upload_mb_int = None

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

uploader_hint = "Upload CSV (up to 1024 MB; set server config to allow 1 GB uploads)"
if env_max_upload_mb_int:
    uploader_hint = f"Upload CSV (server allows up to {env_max_upload_mb_int} MB)"

uploaded = st.sidebar.file_uploader(uploader_hint, type="csv")
if uploaded is None:
    st.info("Please upload a dataset to proceed.")
    st.stop()

# Helper: recommended action to increase limit if needed
if env_max_upload_mb_int is None or env_max_upload_mb_int < 1024:
    st.sidebar.warning(
        "If you need 1 GB uploads set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024 (MB) on the server "
        "or add .streamlit/config.toml with server.maxUploadSize = 1024. "
        "See sidebar instructions at the bottom for examples."
    )

@st.cache_data(show_spinner=True)
def load_and_prepare(uploaded_file):
    # Process large CSVs in chunks to make parsing more robust on big files.
    # This still ends up assembling a full DataFrame in memory; ensure you have enough RAM.
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    idcols = ['STORE_CODE', 'TILL', 'SESSION', 'RCT']

    # Determine uploaded file size in MB if possible
    size_mb = None
    try:
        size_mb = getattr(uploaded_file, "size", None)
        if size_mb:
            size_mb = size_mb / (1024 * 1024)
    except Exception:
        size_mb = None

    # If large file, use chunked reading
    CHUNK_THRESHOLD_MB = 200  # when to switch to chunked parsing
    CHUNK_SIZE = 200_000      # rows per chunk; tune for your data/profile

    def process_chunk(df_chunk):
        df_chunk.columns = [c.strip() for c in df_chunk.columns]
        for col in ['TRN_DATE', 'ZED_DATE']:
            if col in df_chunk.columns:
                df_chunk[col] = pd.to_datetime(df_chunk[col], errors='coerce')
        for nc in numeric_cols:
            if nc in df_chunk.columns:
                df_chunk[nc] = pd.to_numeric(df_chunk[nc], errors='coerce').fillna(0)
        for col in idcols:
            if col in df_chunk.columns:
                df_chunk[col] = df_chunk[col].astype(str).fillna('').str.strip()
        if 'CUST_CODE' not in df_chunk.columns:
            if all(c in df_chunk.columns for c in idcols):
                df_chunk['CUST_CODE'] = (
                    df_chunk['STORE_CODE'].str.strip() + '-' +
                    df_chunk['TILL'].str.strip() + '-' +
                    df_chunk['SESSION'].str.strip() + '-' +
                    df_chunk['RCT'].str.strip()
                )
            else:
                # If required id cols are missing, we will raise later at top-level (keeps chunk logic simpler)
                pass
        if 'CUST_CODE' in df_chunk.columns:
            df_chunk['CUST_CODE'] = df_chunk['CUST_CODE'].astype(str).str.strip()
        return df_chunk

    try:
        if size_mb is not None and size_mb > CHUNK_THRESHOLD_MB:
            # Use chunked reader
            chunks = []
            reader = pd.read_csv(uploaded_file, on_bad_lines='skip', low_memory=False, chunksize=CHUNK_SIZE)
            for chunk in reader:
                processed = process_chunk(chunk)
                chunks.append(processed)
            df = pd.concat(chunks, ignore_index=True)
        else:
            # Normal read
            df = pd.read_csv(uploaded_file, on_bad_lines='skip', low_memory=False)
            df = process_chunk(df)
    except pd.errors.EmptyDataError:
        st.error("Uploaded CSV appears to be empty or malformed.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # Validate presence of CUST_CODE or idcols, else show helpful error
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
        # If obj is a Series or index-like, convert to DataFrame for CSV
        if isinstance(obj, (pd.Series, pd.Index)):
            obj = obj.reset_index()
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
        "Branch Branch Overview",
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

st.sidebar.markdown(
    "---\nSidebar auto-expands for easy selection. All tables and plots can be downloaded. "
    "If image download fails, check `kaleido` install.\n\n"
    "To allow 1 GB uploads, you must set the Streamlit server config: \n\n"
    "Linux/macOS (env): export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024\n\n"
    "Or add repository file: .streamlit/config.toml with content:\nserver.maxUploadSize = 1024\n"
)
