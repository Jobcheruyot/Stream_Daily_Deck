"""
Streamlit Superdeck Analytics Dashboard (S3-direct upload + large file safe loader)
- Supports:
  - normal Streamlit uploader (subject to server.maxUploadSize)
  - direct browser → S3 uploads using a presigned POST (bypasses Streamlit limit)
  - server-side processing of S3 object (download & chunked read)
- Requires AWS credentials (in st.secrets or environment) for presigned POST generation
  and for server-side file download.
"""
import os
import io
import time
from datetime import timedelta
from typing import List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import boto3
from botocore.exceptions import ClientError

# ---------- Page UI and debug ----------
st.set_page_config(layout="wide", page_title="Superdeck Analytics Dashboard (Large Uploads)")
st.title("🦸 Superdeck Analytics Dashboard — Large Upload Ready")
st.markdown("Use the S3 direct upload option for files larger than your Streamlit server upload limit.")

# Runtime debug: show effective Streamlit upload settings
st.sidebar.header("Upload debug info")
st.sidebar.write("ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE:", os.environ.get("STREAMLIT_SERVER_MAX_UPLOAD_SIZE"))
try:
    st.sidebar.write("streamlit server.maxUploadSize (config):", st.config.get_option("server.maxUploadSize"))
except Exception as e:
    st.sidebar.write("Could not read streamlit config:", e)

st.sidebar.markdown("---")
st.sidebar.markdown("If `streamlit server.maxUploadSize` is < desired upload size, add `.streamlit/config.toml` and restart the app.")

# ---------- S3 helper utilities ----------
def get_boto3_client():
    # Prefer st.secrets if present, else environment variables
    aws = {}
    if "aws" in st.secrets:
        aws_conf = st.secrets["aws"]
        aws["aws_access_key_id"] = aws_conf.get("AWS_ACCESS_KEY_ID")
        aws["aws_secret_access_key"] = aws_conf.get("AWS_SECRET_ACCESS_KEY")
        aws["region_name"] = aws_conf.get("AWS_REGION")
    else:
        aws["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID")
        aws["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws["region_name"] = os.environ.get("AWS_REGION")

    if not aws["aws_access_key_id"] or not aws["aws_secret_access_key"]:
        return None

    return boto3.client("s3",
                        aws_access_key_id=aws["aws_access_key_id"],
                        aws_secret_access_key=aws["aws_secret_access_key"],
                        region_name=aws["region_name"])

def make_presigned_post(bucket_name: str, object_name: str, fields: dict = None, conditions: list = None, expiration: int = 3600):
    """
    Generate a presigned POST dict to allow direct browser upload to S3.
    Returns dict with url and fields to include in the multipart POST.
    """
    s3_client = get_boto3_client()
    if s3_client is None:
        raise RuntimeError("AWS credentials not found in st.secrets or environment variables.")
    try:
        response = s3_client.generate_presigned_post(Bucket=bucket_name,
                                                     Key=object_name,
                                                     Fields=fields or {},
                                                     Conditions=conditions or [],
                                                     ExpiresIn=expiration)
    except ClientError as e:
        raise RuntimeError(f"Could not generate presigned POST: {e}")
    return response

def download_s3_to_buffer(bucket: str, key: str) -> io.BytesIO:
    """Download S3 object into a BytesIO buffer"""
    s3_client = get_boto3_client()
    if s3_client is None:
        raise RuntimeError("AWS credentials not found in st.secrets or environment variables.")
    bio = io.BytesIO()
    try:
        s3_client.download_fileobj(bucket, key, bio)
    except ClientError as e:
        raise RuntimeError(f"Failed to download S3 object: {e}")
    bio.seek(0)
    return bio

# ---------- Local CSV loader (chunked safe) ----------
@st.cache_data(show_spinner=True)
def read_csv_chunked(file_like, numeric_cols: Optional[List[str]] = None, idcols: Optional[List[str]] = None):
    """
    Read CSV from file-like object using a chunked approach and return prepared DataFrame.
    - file_like: a file-like object (BytesIO, TemporaryFile, etc) positioned at 0
    - numeric_cols, idcols: optional lists of columns to coerce
    """
    numeric_cols = numeric_cols or ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    idcols = idcols or ['STORE_CODE', 'TILL', 'SESSION', 'RCT']

    try:
        file_like.seek(0)
    except Exception:
        pass

    CHUNK_ROWS = 200_000
    chunks = []
    try:
        reader = pd.read_csv(file_like, on_bad_lines='skip', low_memory=False, chunksize=CHUNK_ROWS)
        for chunk in reader:
            # cleanup chunk
            chunk.columns = [c.strip() for c in chunk.columns]
            for col in ['TRN_DATE', 'ZED_DATE']:
                if col in chunk.columns:
                    chunk[col] = pd.to_datetime(chunk[col], errors='coerce')
            for nc in numeric_cols:
                if nc in chunk.columns:
                    # Remove thousand separators if present, then to numeric
                    chunk[nc] = pd.to_numeric(chunk[nc].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            for col in idcols:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype(str).fillna('').str.strip()
            # build CUST_CODE if not present
            if 'CUST_CODE' not in chunk.columns and all(c in chunk.columns for c in idcols):
                chunk['CUST_CODE'] = (chunk['STORE_CODE'].str.strip() + '-' + chunk['TILL'].str.strip() + '-' +
                                     chunk['SESSION'].str.strip() + '-' + chunk['RCT'].str.strip())
            chunks.append(chunk)
        if len(chunks) == 0:
            return pd.DataFrame()
        df = pd.concat(chunks, ignore_index=True)
        # final cleanup
        if 'CUST_CODE' in df.columns:
            df['CUST_CODE'] = df['CUST_CODE'].astype(str).str.strip()
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        raise

# ---------- UI: Choose upload method ----------
st.header("1) Upload your CSV")
st.markdown("Option A: Streamlit uploader (subject to server upload limit). Option B: Direct upload from your browser to S3 (bypasses Streamlit limits).")

colA, colB = st.columns(2)

with colA:
    st.subheader("A — Streamlit uploader")
    uploaded_file = st.file_uploader("Upload CSV (via Streamlit)", type="csv", accept_multiple_files=False)
    if uploaded_file is not None:
        st.success(f"Received file: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        try:
            df_local = read_csv_chunked(uploaded_file)
            st.info(f"Loaded {len(df_local)} rows.")
            st.session_state["latest_df_rows"] = len(df_local)
            st.session_state["latest_df_sample"] = df_local.head(3).to_dict()
        except Exception as e:
            st.error(f"Failed to parse uploaded CSV: {e}")

with colB:
    st.subheader("B — Direct browser → S3 upload (recommended for >200MB)")
    st.markdown("Instructions:")
    st.markdown("1) Enter S3 bucket name and a target object key (filename) -> 2) Click Generate Form -> 3) Use the browser file control to select and upload file directly to S3 -> 4) Copy the uploaded object key and use 'Process S3 file' below to download & process it.")
    s3_bucket = st.text_input("S3 bucket name", key="s3_bucket")
    s3_key_prefix = st.text_input("S3 key prefix (optional)", value="", key="s3_prefix")
    desired_filename = st.text_input("Target object key (e.g. uploads/mybigfile.csv)", key="s3_key")
    if s3_key_prefix and not desired_filename:
        desired_filename = s3_key_prefix.rstrip("/") + "/"
    if st.button("Generate S3 Upload Form"):
        if not s3_bucket or not desired_filename:
            st.error("Provide S3 bucket name and object key.")
        else:
            # Generate a presigned POST
            # Ensure credentials exist
            client = get_boto3_client()
            if client is None:
                st.error("AWS credentials not found. Put your keys in Streamlit Secrets under [aws] or set environment variables.")
            else:
                # allow small content-type checks but keep minimal conditions to ensure broad uploads
                object_key = desired_filename
                try:
                    presigned = make_presigned_post(s3_bucket, object_key, expiration=3600)
                except Exception as e:
                    st.error(str(e))
                    presigned = None
                if presigned:
                    # Build HTML/JS uploader using presigned POST fields
                    upload_url = presigned["url"]
                    fields = presigned["fields"]
                    # Create simple HTML uploader - user uploads directly to S3
                    post_fields = "".join([f'<input type="hidden" name="{k}" value="{v}"/>' for k, v in fields.items()])
                    html = f"""
                    <html>
                      <body>
                        <p><b>Direct S3 Upload form</b></p>
                        <input id="file" type="file" />
                        <button onclick="upload()">Upload to S3</button>
                        <div id="status"></div>
                        <script>
                          async function upload() {{
                            const fileInput = document.getElementById('file');
                            if (!fileInput.files.length) {{
                              alert('Select a file first');
                              return;
                            }}
                            const file = fileInput.files[0];
                            const url = "{upload_url}";
                            const form = new FormData();
                            {''.join([f'form.append("{k}", "{v}");\\n                            ' for k, v in fields.items()])}
                            // Key must match the presigned Key field; if using dynamic key you would set here
                            form.append('file', file);
                            document.getElementById('status').innerText = "Uploading...";
                            try {{
                              const resp = await fetch(url, {{
                                method: 'POST',
                                body: form
                              }});
                              if (resp.ok) {{
                                document.getElementById('status').innerHTML = "Upload succeeded. Object key: <code>{object_key}</code>";
                              }} else {{
                                document.getElementById('status').innerText = "Upload failed: " + resp.status + " " + resp.statusText;
                              }}
                            }} catch (err) {{
                              document.getElementById('status').innerText = "Upload error: " + err;
                            }}
                          }}
                        </script>
                      </body>
                    </html>
                    """
                    st.components.v1.html(html, height=220, scrolling=True)
                    st.success("S3 upload form generated. After upload, use the object key to process the file below.")

# ---------- Process S3 uploaded file ----------
st.markdown("---")
st.header("2) Process a CSV that's already in S3 (download & parse server-side)")
st.markdown("If you used the Direct S3 Upload flow above, paste the exact object key here and click Process. The app will download directly from S3 and parse in chunks.")

s3_bucket_proc = st.text_input("S3 bucket to download from (if different than above)", value=s3_bucket or "", key="proc_bucket")
s3_object_key = st.text_input("S3 object key to process (e.g. uploads/mybigfile.csv)", key="proc_key")
if st.button("Process S3 file"):
    if not s3_bucket_proc or not s3_object_key:
        st.error("Provide both bucket and object key.")
    else:
        st.info("Downloading file from S3 and processing (chunked)... This may take a while for very large files.")
        try:
            bio = download_s3_to_buffer(s3_bucket_proc, s3_object_key)
            df_from_s3 = read_csv_chunked(bio)
            st.success(f"Downloaded and parsed {len(df_from_s3)} rows from s3://{s3_bucket_proc}/{s3_object_key}")
            # Basic example analytics: group NET_SALES by SALES_CHANNEL_L1 if present
            if 'NET_SALES' in df_from_s3.columns and 'SALES_CHANNEL_L1' in df_from_s3.columns:
                gs = df_from_s3.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
                gs['NET_SALES_M'] = gs['NET_SALES'] / 1_000_000
                fig = px.pie(gs, names='SALES_CHANNEL_L1', values='NET_SALES_M', title="Sales by Channel (M)")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(gs.head(50), use_container_width=True)
            else:
                st.info("Parsed file. No SALES_CHANNEL_L1 or NET_SALES columns found; showing sample rows.")
                st.dataframe(df_from_s3.head(50), use_container_width=True)
            # cache last processed summary for convenience
            st.session_state['last_loaded_rows'] = len(df_from_s3)
        except Exception as e:
            st.error(f"Failed to process S3 file: {e}")

# ---------- Small diagnostics ----------
st.markdown("---")
st.write("Diagnostics / tips:")
st.write("- If you see the Streamlit uploader rejecting >200MB, use the Direct S3 Upload form instead.")
st.write("- To enable direct S3 uploads you must provide AWS credentials in Streamlit Secrets or environment variables.")
st.write("- Recommended: set server.maxUploadSize = 1024 in .streamlit/config.toml and restart app.")
