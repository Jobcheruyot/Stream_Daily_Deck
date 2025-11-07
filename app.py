import os
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

# ONLY present the 1 GB option/hint (no alternative small-file text)
uploader_hint = "Upload CSV (up to 1024 MB — ensure server.maxUploadSize is set to 1024 MB)"
uploaded = st.sidebar.file_uploader(uploader_hint, type="csv")
if uploaded is None:
    st.info("Please upload a dataset to proceed.")
    st.stop()

# We'll always use chunked reading (no "small-file alternative").
# Use a non-cached loader so we can show progress and live messages while reading.
def _process_chunk(df_chunk, numeric_cols, idcols):
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
    if 'CUST_CODE' in df_chunk.columns:
        df_chunk['CUST_CODE'] = df_chunk['CUST_CODE'].astype(str).str.strip()
    return df_chunk

def load_and_prepare_chunked(uploaded_file, chunksize=200_000):
    """
    Always reads CSV in chunks, processes each chunk and returns the concatenated dataframe.
    Provides live progress messages to Streamlit.
    """
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    idcols = ['STORE_CODE', 'TILL', 'SESSION', 'RCT']

    # Attempt to estimate file size (MB) if available for user info
    size_mb = None
    try:
        size = getattr(uploaded_file, "size", None)
        if size:
            size_mb = size / (1024 * 1024)
    except Exception:
        size_mb = None

    status = st.empty()
    progress_placeholder = st.empty()
    status.info("Starting chunked load...")

    # Ensure file pointer at start
    try:
        uploaded_file.seek(0)
    except Exception:
        # Some UploadedFile may not support seek - convert to BytesIO
        uploaded_file = io.BytesIO(uploaded_file.getvalue())

    chunks = []
    total_rows = 0
    chunk_count = 0

    try:
        reader = pd.read_csv(uploaded_file, on_bad_lines='skip', low_memory=False, chunksize=chunksize)
    except Exception as e:
        status.error(f"Failed to open CSV for chunked reading: {e}")
        st.stop()

    with st.spinner("Reading CSV in chunks..."):
        for chunk in reader:
            chunk_count += 1
            processed = _process_chunk(chunk, numeric_cols, idcols)
            rows = len(processed)
            total_rows += rows
            chunks.append(processed)

            # update lightweight progress info
            progress_placeholder.write(f"Chunks processed: {chunk_count} — Rows read so far: {total_rows:,}")
        if chunk_count == 0:
            status.error("No data found in uploaded CSV.")
            st.stop()

    status.success(f"Finished reading CSV: {chunk_count} chunks, {total_rows:,} rows.")
    # Concatenate
    try:
        df = pd.concat(chunks, ignore_index=True)
    except Exception as e:
        status.error(f"Failed to concatenate chunks: {e}")
        st.stop()

    # Validate presence of CUST_CODE or idcols
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
            status.error(f"Missing columns for CUST_CODE: {missing}")
            st.stop()

    # Clean up placeholders
    progress_placeholder.empty()
    return df

# Load data (always chunked). Show progress and immediate debug outputs.
df = load_and_prepare_chunked(uploaded, chunksize=200_000)

# Quick diagnostics & visible outputs so user can see what's happened
st.subheader("Upload Summary & Diagnostics")
col_a, col_b = st.columns(2)
with col_a:
    st.write("Shape:")
    st.write({"rows": df.shape[0], "columns": df.shape[1]})
    st.write("Columns and dtypes:")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]).reset_index().rename(columns={"index": "column"}))
    st.write("Sample (first 10 rows):")
    st.dataframe(df.head(10))
with col_b:
    st.write("Basic stats for numeric columns:")
    if any(col for col in df.columns if np.issubdtype(df[col].dtype, np.number)):
        st.dataframe(df.describe().transpose())
    else:
        st.write("No numeric columns detected to describe.")
    st.write("Unique counts of key IDs:")
    id_summary = {
        "unique_STORE_NAME": int(df['STORE_NAME'].nunique()) if 'STORE_NAME' in df.columns else None,
        "unique_STORE_CODE": int(df['STORE_CODE'].nunique()) if 'STORE_CODE' in df.columns else None,
        "unique_CUST_CODE": int(df['CUST_CODE'].nunique()) if 'CUST_CODE' in df.columns else None,
    }
    st.write(id_summary)

# create time grid for other parts of the app
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
        if isinstance(obj, (pd.Series, pd.Index)):
            obj = obj.reset_index()
        st.download_button(label, obj.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")

def download_plot(fig, filename):
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600)
        st.download_button("⬇️ Download Plot as PNG", img_bytes, filename=filename, mime="image/png")
    except Exception:
        st.info("Image download is unavailable (likely missing `kaleido`). Table will still download fine.")

# Main UI sections (kept minimal here) — default to showing a Global Sales Overview output so the user sees app outputs immediately
st.subheader("Quick Insights (auto-generated)")

if 'SALES_CHANNEL_L1' in df.columns and 'NET_SALES' in df.columns:
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
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(gs)
    download_button(gs, "global_sales_overview.csv", "⬇️ Download Table")
    download_plot(fig, "global_sales_overview.png")
else:
    st.warning("Columns SALES_CHANNEL_L1 and/or NET_SALES are missing; cannot produce Global Sales Overview.")

# Also produce a small store-wise receipts by time chart if TRN_DATE and STORE_NAME exist
if 'TRN_DATE' in df.columns and 'STORE_NAME' in df.columns:
    try:
        dff = df.dropna(subset=['TRN_DATE']).copy()
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        # pick a store sample to display if many exist
        stores = dff["STORE_NAME"].dropna().unique().tolist()
        sample_store = stores[0] if stores else None
        if sample_store:
            st.write(f"Receipts by time for a sample store: {sample_store}")
            dff_sample = dff[dff["STORE_NAME"] == sample_store].copy()
            for c in ["STORE_CODE","TILL","SESSION","RCT"]:
                if c in dff_sample.columns:
                    dff_sample[c] = dff_sample[c].astype(str).fillna('').str.strip()
            dff_sample['CUST_CODE'] = dff_sample.get('CUST_CODE', dff_sample.get('CUST_CODE', ''))  # safe-get
            dff_sample['TIME_ONLY'] = dff_sample['TRN_DATE'].dt.floor('30T').dt.time
            heat = dff_sample.groupby('TIME_ONLY')['CUST_CODE'].nunique().reindex(intervals, fill_value=0)
            fig2 = px.bar(x=col_labels, y=heat.values, labels={"x":"Time","y":"Receipts"}, text=heat.values,
                          color_discrete_sequence=['#3192e1'], title=f"Receipts by Time - {sample_store}", height=360)
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(heat.reset_index().rename(columns={0: 'receipts'}))
            download_button(heat.reset_index(), "customer_traffic_sample_store.csv", "⬇️ Download Table")
        else:
            st.info("No STORE_NAME values found to display time-based receipts.")
    except Exception as e:
        st.error(f"Error while generating time chart: {e}")
else:
    st.info("Skipping store time chart: TRN_DATE and/or STORE_NAME columns not present.")

# Sidebar note (only the 1GB instruction)
st.sidebar.markdown(
    "---\nTo allow 1 GB uploads, set the Streamlit server config:\n\n"
    "Linux/macOS (env): export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024\n\n"
    "Or add repository file: .streamlit/config.toml with content:\nserver.maxUploadSize = 1024\n"
)
