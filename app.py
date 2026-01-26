import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from io import BytesIO

st.set_page_config(layout="wide", page_title="Superdeck (Streamlit)")
# Add the following at the very top of your Streamlit script, AFTER st.set_page_config


# -----------------------
# Data Loading & Caching
# -----------------------
def _read_csv_in_chunks(source, chunksize=100_000, read_kwargs=None):
    """
    Read a CSV from a path or a file-like object using chunks, return a single DataFrame.
    This reduces peak memory during parsing and is more robust for very large files.

    source: path (str) or file-like (BytesIO)
    chunksize: number of rows per chunk
    read_kwargs: additional kwargs passed to pd.read_csv
    """
    if read_kwargs is None:
        read_kwargs = {}
    chunks = []
    try:
        reader = pd.read_csv(source, chunksize=chunksize, **read_kwargs)
        for chunk in reader:
            chunks.append(chunk)
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.DataFrame()
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except ValueError:
        # Some pandas versions may raise ValueError for certain malformed CSVs when using chunks.
        # Fall back to single-shot read as a last resort.
        try:
            return pd.read_csv(source, **read_kwargs)
        except Exception:
            raise

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """
    Robust loader for a CSV file on disk. Uses chunked reading by default.
    """
    read_kwargs = dict(on_bad_lines='skip', low_memory=False)
    try:
        # Try chunked reading first (helps with huge files)
        return _read_csv_in_chunks(path, chunksize=100_000, read_kwargs=read_kwargs)
    except Exception:
        # fallback to a direct read if chunked fails for some reason
        return pd.read_csv(path, on_bad_lines='skip', low_memory=False)

@st.cache_data
def load_uploaded_file(contents: bytes) -> pd.DataFrame:
    """
    Robust loader for uploaded bytes. Uses a BytesIO buffer and chunked reading.
    """
    buf = BytesIO(contents)
    read_kwargs = dict(on_bad_lines='skip', low_memory=False)
    try:
        return _read_csv_in_chunks(buf, chunksize=100_000, read_kwargs=read_kwargs)
    except Exception:
        # fallback
        buf.seek(0)
        return pd.read_csv(buf, on_bad_lines='skip', low_memory=False)


def smart_load():
    st.sidebar.markdown("### Upload data (CSV) or use default")
    uploaded = st.sidebar.file_uploader("Upload DAILY_POS_TRN_ITEMS CSV", type=['csv'])
    if uploaded is not None:
        with st.spinner("Parsing uploaded CSV..."):
            df = load_uploaded_file(uploaded.getvalue())
        st.sidebar.success("Loaded uploaded CSV")
        return df

    # try default path (optional)
    default_path = "/content/DAILY_POS_TRN_ITEMS_2025-10-21.csv"
    try:
        with st.spinner(f"Loading default CSV: {default_path}"):
            df = load_csv(default_path)
        st.sidebar.info(f"Loaded default path: {default_path}")
        return df
    except Exception:
        st.sidebar.warning("No default CSV found. Please upload a CSV to run the app.")
        return None

# -----------------------
# Robust cleaning + derived columns (cached)
# -----------------------
@st.cache_data
def clean_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
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

    # Dates
    if 'TRN_DATE' in d.columns:
        d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
        d = d.dropna(subset=['TRN_DATE']).copy()
        d['DATE'] = d['TRN_DATE'].dt.date
        d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30min')
        d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time

    if 'ZED_DATE' in d.columns:
        d['ZED_DATE'] = pd.to_datetime(d['ZED_DATE'], errors='coerce')

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
# Small cached aggregation helpers
# -----------------------
@st.cache_data
def agg_net_sales_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col, 'NET_SALES'])
    g = df.groupby(col, as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
    return g

@st.cache_data
def agg_count_distinct(df: pd.DataFrame, group_by: list, agg_col: str, agg_name: str) -> pd.DataFrame:
    g = df.groupby(group_by).agg({agg_col: pd.Series.nunique}).reset_index().rename(columns={agg_col: agg_name})
    return g

# -----------------------
# Table formatting helper
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
# Helper plotting utils
# -----------------------
# (rest of file unchanged)

# Note: For brevity the rest of app.py remains identical to original and is not repeated here.
# When applying this update ensure the unchanged parts of the original app.py are preserved exactly below this point.
