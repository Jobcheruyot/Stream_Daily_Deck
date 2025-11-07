import os
import io
from datetime import timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# === Page config & style tweaks ===
st.set_page_config(layout="wide", page_title="Superdeck Analytics Dashboard (1GB upload-ready)")
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

# === Debug: show streamlit upload limit (env + config) ===
st.sidebar.header("Upload configuration & tips")
env_val = os.environ.get("STREAMLIT_SERVER_MAX_UPLOAD_SIZE")
try:
    cfg_val = st.config.get_option("server.maxUploadSize")
except Exception:
    cfg_val = None

st.sidebar.write("STREAMLIT_SERVER_MAX_UPLOAD_SIZE (env):", env_val)
st.sidebar.write("streamlit server.maxUploadSize (config):", cfg_val)
desired_mb = 1024
st.sidebar.markdown(
    f"- Desired maximum upload size: **{desired_mb} MB (1 GB)**\n"
    "- If the config or env value is below this, uploads larger than that will be rejected by the Streamlit frontend."
)

st.title("🦸 Superdeck Analytics Dashboard — Upload (1GB guidance)")
st.markdown("> Upload your sales CSV (app supports chunked parsing). If you need to accept 1GB uploads, set Streamlit/server and proxy limits before launching the app (instructions below).")

# Provide immediate troubleshooting snippet and instructions
st.sidebar.markdown("### How to enable 1 GB uploads")
st.sidebar.markdown(
    "1) Add `.streamlit/config.toml` with: `server.maxUploadSize = 1024` and restart Streamlit.\n\n"
    "2) OR start Streamlit with CLI: `streamlit run app.py --server.maxUploadSize=1024`.\n\n"
    "3) OR set env var before start: `export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024`.\n\n"
    "4) If behind NGINX or another proxy, set `client_max_body_size 1024M;` in NGINX and reload.\n\n"
    "5) If you're on a managed platform (Cloud Run, Heroku, Streamlit Cloud), check provider limits — they may block large uploads regardless of Streamlit config."
)

# --- Uploader ---
uploader_hint = f"Upload CSV (server allows up to {cfg_val or env_val or 'UNKNOWN'} MB)"
uploaded = st.sidebar.file_uploader(uploader_hint, type="csv")
if uploaded is None:
    st.info("Please upload a dataset to proceed.")
    st.stop()

# === Utility functions ===
@st.cache_data(show_spinner=True)
def load_and_prepare(uploaded_file, chunk_threshold_mb=200, chunk_rows=200_000):
    """
    Load CSV using chunked reading for large files. This still constructs the full DataFrame in memory,
    so ensure the host has enough RAM. If you cannot increase RAM, consider switching to S3 upload + processing
    or using Dask/vaex for out-of-core processing.
    """
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    idcols = ['STORE_CODE', 'TILL', 'SESSION', 'RCT']

    # Try to get approximate uploaded file size (in bytes)
    size_mb = None
    try:
        size_bytes = getattr(uploaded_file, "size", None)
        if size_bytes:
            size_mb = size_bytes / (1024 * 1024)
    except Exception:
        size_mb = None

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
        if 'CUST_CODE' not in df_chunk.columns and all(c in df_chunk.columns for c in idcols):
            df_chunk['CUST_CODE'] = (
                df_chunk['STORE_CODE'].str.strip() + '-' +
                df_chunk['TILL'].str.strip() + '-' +
                df_chunk['SESSION'].str.strip() + '-' +
                df_chunk['RCT'].str.strip()
            )
        if 'CUST_CODE' in df_chunk.columns:
            df_chunk['CUST_CODE'] = df_chunk['CUST_CODE'].astype(str).str.strip()
        return df_chunk

    try:
        if size_mb is not None and size_mb > chunk_threshold_mb:
            # chunked read
            chunks = []
            reader = pd.read_csv(uploaded_file, on_bad_lines='skip', low_memory=False, chunksize=chunk_rows)
            for chunk in reader:
                chunks.append(process_chunk(chunk))
            df = pd.concat(chunks, ignore_index=True)
        else:
            # read at once
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, on_bad_lines='skip', low_memory=False)
            df = process_chunk(df)
    except pd.errors.EmptyDataError:
        st.error("Uploaded CSV appears to be empty or malformed.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # Ensure CUST_CODE exists or build it, otherwise fail
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
        if isinstance(obj, (pd.Series, pd.Index)):
            obj = obj.reset_index()
        st.download_button(label, obj.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")

def download_plot(fig, filename):
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600)
        st.download_button("⬇️ Download Plot as PNG", img_bytes, filename=filename, mime="image/png")
    except Exception:
        st.info("Image download is unavailable (likely missing `kaleido`). Table will still download fine.")

# Minimal sections to demonstrate charts / tables; extend as needed
main_sections = {
    "SALES": ["Global sales Overview", "Global Net Sales Distribution by Sales Channel"],
    "OPERATIONS": ["Customer Traffic-Storewise"],
    "INSIGHTS": ["Branch Comparison"]
}

section = st.sidebar.radio("Main Section", list(main_sections.keys()))
subsection = st.sidebar.selectbox("Subsection", main_sections[section], key="subsection")
st.markdown(f"##### {section} ➔ {subsection}")

# === SALES ===
if section == "SALES":
    if subsection == "Global sales Overview":
        if 'SALES_CHANNEL_L1' not in df.columns or 'NET_SALES' not in df.columns:
            st.error("Required columns SALES_CHANNEL_L1 or NET_SALES missing.")
        else:
            gs = df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum()
            gs['NET_SALES_M'] = gs['NET_SALES'] / 1_000_000
            gs['PCT'] = (gs['NET_SALES'] / gs['NET_SALES'].sum()) * 100
            labels = [f"{row['SALES_CHANNEL_L1']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f}M)" for _, row in gs.iterrows()]
            fig = go.Figure(data=[go.Pie(labels=labels, values=gs['NET_SALES_M'], hole=0.57,
                                         marker=dict(colors=px.colors.qualitative.Plotly),
                                         text=[f"{p:.1f}%" for p in gs['PCT']], textinfo='text', sort=False)])
            fig.update_layout(title="SALES CHANNEL TYPE — Global Overview", height=400, margin=dict(t=60))
            col1, col2 = st.columns([2, 2])
            with col1:
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.dataframe(gs, use_container_width=True)
                download_button(gs, "global_sales_overview.csv", "⬇️ Download Table")
                download_plot(fig, "global_sales_overview.png")

    elif subsection == "Global Net Sales Distribution by Sales Channel":
        if 'SALES_CHANNEL_L2' not in df.columns or 'NET_SALES' not in df.columns:
            st.error("Required columns SALES_CHANNEL_L2 or NET_SALES missing.")
        else:
            g2 = df.groupby('SALES_CHANNEL_L2', as_index=False)['NET_SALES'].sum()
            g2['NET_SALES_M'] = g2['NET_SALES'] / 1_000_000
            g2['PCT'] = g2['NET_SALES'] / g2['NET_SALES'].sum() * 100
            labels = [f"{row['SALES_CHANNEL_L2']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f}M)" for _, row in g2.iterrows()]
            fig = go.Figure(go.Pie(labels=labels, values=g2['NET_SALES_M'], hole=0.58,
                                   marker=dict(colors=px.colors.qualitative.Vivid),
                                   text=[f"{p:.1f}%" for p in g2['PCT']], textinfo='text'))
            fig.update_layout(title="Net Sales by Sales Mode (L2)", height=400, margin=dict(t=60))
            col1, col2 = st.columns([2, 2])
            with col1:
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.dataframe(g2, use_container_width=True)
                download_button(g2, "sales_channel_l2.csv", "⬇️ Download Table")
                download_plot(fig, "sales_channel_l2_pie.png")

# === OPERATIONS ===
elif section == "OPERATIONS":
    if subsection == "Customer Traffic-Storewise":
        if 'STORE_NAME' not in df.columns or 'TRN_DATE' not in df.columns:
            st.error("Required columns STORE_NAME or TRN_DATE missing.")
        else:
            stores = df["STORE_NAME"].dropna().unique().tolist()
            selected_store = st.selectbox("Select Store", stores)
            dff = df[df["STORE_NAME"] == selected_store].copy()
            dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
            dff = dff.dropna(subset=['TRN_DATE'])
            for c in ["STORE_CODE", "TILL", "SESSION", "RCT"]:
                if c in dff.columns:
                    dff[c] = dff[c].astype(str).fillna('').str.strip()
            if all(c in dff.columns for c in ["STORE_CODE", "TILL", "SESSION", "RCT"]):
                dff['CUST_CODE'] = dff['STORE_CODE'] + '-' + dff['TILL'] + '-' + dff['SESSION'] + '-' + dff['RCT']
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

# === INSIGHTS ===
elif section == "INSIGHTS":
    if subsection == "Branch Comparison":
        if 'STORE_NAME' not in df.columns or 'ITEM_NAME' not in df.columns:
            st.error("Required columns STORE_NAME or ITEM_NAME missing.")
        else:
            branches = sorted(df['STORE_NAME'].dropna().unique().tolist())
            selected_A = st.selectbox("Branch A", branches, key="bc_a")
            selected_B = st.selectbox("Branch B", branches, key="bc_b")
            metric = st.selectbox("Metric", ["QTY", "NET_SALES"], key="bc_metric")
            N = st.slider("Top N", 5, 50, 10, key="bc_n")
            dfA = df[df["STORE_NAME"] == selected_A].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N)
            dfB = df[df["STORE_NAME"] == selected_B].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N)
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

st.sidebar.markdown("---\nIf you still get the '200MB' error after setting server.maxUploadSize to 1024 and restarting Streamlit, the blocking component is almost certainly your reverse proxy or the hosting platform. Update nginx / load balancer settings or use an external upload (S3 presigned) flow and then process the file from cloud storage.")
