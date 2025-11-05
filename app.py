
import os
import io
import json
import time
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc
import streamlit as st
import plotly.express as px
from typing import List, Tuple

st.set_page_config(page_title="DailyDeck — Wow UI", page_icon="📊", layout="wide")

# -----------------------------
# Minimal "wow" styling
# -----------------------------
st.markdown("""
    <style>
        .main {padding: 1.5rem 2rem;}
        .block-container {padding-top: 1rem;}
        .stButton>button {
            border-radius: 14px;
            padding: 0.6rem 1rem;
            box-shadow: 0 6px 18px rgba(0,0,0,.08);
        }
        .metric-card {
            border-radius: 18px;
            padding: 1rem 1.2rem;
            background: linear-gradient(180deg, rgba(250,250,253,1) 0%, rgba(244,246,252,1) 100%);
            border: 1px solid #eef0f6;
        }
        .pill {
            display:inline-block; padding:.25rem .6rem; border-radius:999px; background:#eef2ff; font-size:.8rem; margin-right:.4rem;
        }
        .hero {
            border-radius: 24px; padding: 24px; border: 1px solid #e9eef7; 
            background: radial-gradient(1200px 300px at 100% 0%, #f3f7ff 0%, #ffffff 70%);
        }
        .footer-note {font-size: 12px; color: #6b7280; text-align:center; margin-top:1rem;}
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='hero'><h2>📈 DailyDeck — Big CSV Uploader & Notebook Viewer</h2><p>Upload huge CSVs (≈500MB), explore fast with DuckDB/Arrow, render notebook markdown (without lines starting with <code>*</code>), and download results.</p></div>", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def arrow_read_csv(file_bytes: bytes, sample_rows: int = 50000) -> pa.Table:
    # Read to Arrow table using streaming reader
    reader = pacsv.open_csv(io.BytesIO(file_bytes))
    tbl = reader.read_next_batch()
    batches = [tbl]
    rows = tbl.num_rows
    # Limit sample_rows to keep memory reasonable while previewing
    while True:
        try:
            b = reader.read_next_batch()
        except StopIteration:
            break
        if b is None or b.num_rows == 0:
            break
        if rows + b.num_rows > sample_rows:
            # take partial to stay within sample_rows
            remaining = max(0, sample_rows - rows)
            if remaining <= 0:
                break
            b = b.slice(0, remaining)
        batches.append(b)
        rows += b.num_rows
        if rows >= sample_rows:
            break
    return pa.Table.from_batches(batches)

@st.cache_data(show_spinner=False)
def duckdb_load_to_df(tmp_path: str, limit_rows: int = 100000) -> pd.DataFrame:
    # DuckDB can scan large CSVs quickly without loading full file into memory
    con = duckdb.connect(database=':memory:')
    # AUTO-DETECT types; change delim/quote as needed
    query = f"""
        SELECT * FROM read_csv_auto('{tmp_path}', HEADER=TRUE, SAMPLE_SIZE=2000000)
        LIMIT {limit_rows}
    """
    df = con.execute(query).df()
    con.close()
    return df

def render_markdown_filtered(md_text: str) -> str:
    # Drop lines that start with '*'
    filtered_lines = []
    for line in md_text.splitlines():
        if line.strip().startswith('*'):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)

def extract_md_from_ipynb(nb_bytes: bytes) -> str:
    nb = json.loads(nb_bytes.decode('utf-8', errors='ignore'))
    md_chunks = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            # join source; filter later
            src = ''.join(cell.get('source', ''))
            md_chunks.append(src)
    return "\n\n---\n\n".join(md_chunks)

def safe_action_labels(labels: List[str]) -> List[str]:
    # Return only labels that do NOT start with '*'
    return [lbl for lbl in labels if not lbl.strip().startswith('*')]

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

# -----------------------------
# Sidebar — uploads & options
# -----------------------------
with st.sidebar:
    st.header("📤 Uploads & Options")
    csv_file = st.file_uploader("Upload CSV (up to ~500MB)", type=["csv"], accept_multiple_files=False)
    nb_file = st.file_uploader("Optional: Upload .ipynb (markdown will render)", type=["ipynb"], accept_multiple_files=False)

    st.subheader("Read Options")
    limit_rows = st.number_input("Preview rows (DuckDB LIMIT)", min_value=1000, max_value=2_000_000, value=200_000, step=10_000)
    sample_rows = st.number_input("Markdown-only mode sample rows (Arrow)", min_value=10_000, max_value=1_000_000, value=100_000, step=10_000)

    st.divider()
    st.caption("Only actions not starting with '*' will be shown.")
    raw_actions = st.text_area("Action labels (one per line)", value="Show Summary\n*Hidden Experimental\nTop Categories\nTime Series\nDownload Current View")
    actions = safe_action_labels(raw_actions.splitlines())

# -----------------------------
# Notebook markdown (filtered)
# -----------------------------
if nb_file:
    try:
        md = extract_md_from_ipynb(nb_file.getvalue())
        md = render_markdown_filtered(md)
        with st.expander("📘 Notebook Markdown (filtered)", expanded=True):
            st.markdown(md)
        # Offer download of the filtered markdown
        st.download_button("Download filtered Markdown", data=md.encode("utf-8"), file_name="notebook_filtered.md")
    except Exception as e:
        st.warning(f"Could not parse notebook markdown: {e}")

# -----------------------------
# Data ingest
# -----------------------------
df = None
if csv_file is not None:
    # Save to a temp file so DuckDB can scan it
    tmp_path = os.path.join(st.experimental_get_query_params().get("tmpdir", ["/tmp"])[0], f"upload_{int(time.time())}.csv")
    with open(tmp_path, "wb") as f:
        f.write(csv_file.getvalue())

    with st.spinner("Loading preview via DuckDB..."):
        df = duckdb_load_to_df(tmp_path, limit_rows=int(limit_rows))

    st.success(f"Loaded preview with {len(df):,} rows × {len(df.columns)} cols (DuckDB LIMIT).")

    # Basic info
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Rows (preview)", f"{len(df):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Columns", f"{len(df.columns):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("File Size", f"{csv_file.size/1024/1024:.1f} MB")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

# -----------------------------
# Action panel (only non-*)
# -----------------------------
st.subheader("⚙️ Actions")
cols = st.columns(min(4, max(1, len(actions))))
clicked = None
for i, label in enumerate(actions):
    with cols[i % len(cols)]:
        if st.button(label, use_container_width=True):
            clicked = label

if df is None and clicked:
    st.info("Upload a CSV first to use the actions.")

# -----------------------------
# Action handlers
# -----------------------------
if df is not None:
    # Column selectors
    with st.expander("🔎 Column selectors", expanded=False):
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c])]
        dt_cols  = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        st.caption("Detected types are heuristic; you can still select any column below.")
        sel_num = st.multiselect("Numeric columns", options=list(df.columns), default=num_cols[:3])
        sel_cat = st.multiselect("Categorical columns", options=list(df.columns), default=cat_cols[:2])
        sel_dt  = st.selectbox("Datetime column (optional)", options=["(none)"] + list(df.columns))

    # Default view if nothing clicked
    if clicked is None:
        st.markdown("#### 👋 Start by choosing an action above, or explore quick summaries below.")

    # Show Summary
    if clicked == "Show Summary" or clicked is None:
        with st.container():
            st.markdown("### 📊 Quick Summary")
            st.write(df.describe(include='all').transpose())

    # Top Categories
    if clicked == "Top Categories":
        if sel_cat:
            target = sel_cat[0]
            vc = df[target].value_counts().head(25).reset_index()
            vc.columns = [target, "count"]
            st.markdown(f"### 🏷️ Top {target}")
            fig = px.bar(vc, x=target, y="count")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download Top Categories CSV", data=to_csv_bytes(vc), file_name=f"top_{target}.csv")
        else:
            st.info("Select at least one categorical column in the expand section above.")

    # Time Series
    if clicked == "Time Series":
        if sel_dt and sel_dt != "(none)":
            dt_series = pd.to_datetime(df[sel_dt], errors='coerce')
            ts = df.assign(__dt=dt_series).dropna(subset=["__dt"]).groupby(pd.Grouper(key="__dt", freq="D")).size().reset_index(name="count")
            st.markdown(f"### ⏱️ Counts per day ({sel_dt})")
            fig = px.line(ts, x="__dt", y="count")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download Time Series CSV", data=to_csv_bytes(ts), file_name="time_series.csv")
        else:
            st.info("Choose a datetime column to build a series.")

    # Download Current View
    if clicked == "Download Current View":
        st.download_button("Download current preview as CSV", data=to_csv_bytes(df), file_name="preview.csv", use_container_width=True)

st.markdown("<div class='footer-note'>Built with Streamlit • DuckDB • Arrow • Plotly</div>", unsafe_allow_html=True)
