import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from supabase import create_client, Client

st.set_page_config(layout="wide", page_title="DailyDeck (Supabase Millions)")

# =========================================================
#  SUPABASE CONNECTION
# =========================================================
@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

# =========================================================
#  CHUNKED LOADER: FETCH *ALL* ROWS IN DATE RANGE
#  - Uses range() pagination so we are NOT capped at 1,000
#  - No artificial row limit
#  - chunk_size can be tuned (100k is a good start)
# =========================================================
@st.cache_data(show_spinner=True)
def load_from_supabase(
    start_date: date,
    end_date: date,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    client = get_supabase_client()

    start_iso = f"{start_date}T00:00:00"
    end_iso   = f"{end_date}T23:59:59.999999"

    all_rows = []
    start = 0

    while True:
        end = start + chunk_size - 1
        resp = (
            client.table("daily_pos_trn_items_clean")
            .select("*")
            .gte("trn_date", start_iso)
            .lte("trn_date", end_iso)
            .range(start, end)         # <-- no .limit(), use ranged pagination
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break

        all_rows.extend(rows)

        # if we got less than a full chunk, Supabase has no more rows
        if len(rows) < chunk_size:
            break

        start += chunk_size

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # IMPORTANT: normalise to uppercase so your original code
    # (TRN_DATE, STORE_NAME, SALES_CHANNEL_L1, etc.) still works
    df.columns = [c.upper() for c in df.columns]
    return df

# =========================================================
#  CLEANING & DERIVED COLUMNS (your existing version)
#  (this is almost exactly what you already had)
# =========================================================
@st.cache_data
def clean_and_derive(df: pd.DataFrame) -> pd.DataFrame:
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

# =========================================================
#  SMALL HELPERS (same as your script)
# =========================================================
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

def format_and_display(df: pd.DataFrame, numeric_cols: list | None = None,
                       index_col: str | None = None, total_label: str = 'TOTAL'):
    # (you can paste your existing format_and_display body here – unchanged)
    ...
    # [KEEP YOUR ORIGINAL IMPLEMENTATION]
    # -----------------------------------

def donut_from_agg(df_agg, label_col, value_col, title,
                   hole=0.55, colors=None,
                   legend_title=None, value_is_millions=False):
    # (same body as in your script – unchanged)
    ...
