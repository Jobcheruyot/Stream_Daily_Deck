import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from supabase import create_client

st.set_page_config(layout="wide", page_title="Superdeck")

TABLE_NAME = "daily_pos_trn_items_clean"

# ---------------------------------------------------------
# CONNECT TO SUPABASE
# ---------------------------------------------------------
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except:
        st.error("Supabase credentials missing in Streamlit Secrets.")
        st.stop()

    return create_client(url, key)


# ---------------------------------------------------------
# LOAD FROM SUPABASE (LOWERCASE-FRIENDLY)
# ---------------------------------------------------------
def load_supabase_data(date_basis, start_date, end_date):
    client = init_supabase()

    # force lowercase because Supabase stores columns in lowercase
    date_basis_lower = date_basis.lower()

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Query Supabase
    try:
        res = (
            client.table(TABLE_NAME)
            .select("*")
            .gte(date_basis_lower, start_str)
            .lte(date_basis_lower, end_str)
            .execute()
        )
    except Exception as e:
        st.error(f"Supabase query failed: {e}")
        return pd.DataFrame()

    data = res.data or []
    df = pd.DataFrame(data)

    if df.empty:
        return df

    # 🟢 Convert all columns to uppercase (CRITICAL FIX)
    df.columns = [c.upper() for c in df.columns]

    # 🟢 Create uppercase aliases for date columns if they exist
    if "TRN_DATE" not in df.columns and "TRN_DATE".lower() in df.columns:
        df["TRN_DATE"] = df["TRN_DATE".lower()]
    if "ZED_DATE" not in df.columns and "ZED_DATE".lower() in df.columns:
        df["ZED_DATE"] = df["ZED_DATE".lower()]

    # convert dates
    for col in ["TRN_DATE", "ZED_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
def sidebar_config():
    date_basis = st.sidebar.radio(
        "Use which date?",
        ["TRN_DATE", "ZED_DATE"],
        index=0
    )
    today = datetime.today().date()
    start_date = st.sidebar.date_input("Start date", today - timedelta(days=7))
    end_date = st.sidebar.date_input("End date", today)

    section = st.sidebar.selectbox(
        "Section", ["SALES", "OPERATIONS", "INSIGHTS"]
    )

    return date_basis, start_date, end_date, section


# ---------------------------------------------------------
# CLEAN DATAFRAME
# ---------------------------------------------------------
def clean_df(df, date_basis):
    if df.empty:
        return df

    df = df.copy()

    # derived fields
    df["DATE"] = df[date_basis].dt.date
    df["HOUR"] = df[date_basis].dt.hour
    df["HALF_HOUR"] = df[date_basis].dt.floor("30min")

    # numeric cleanup
    for col in ["QTY", "NET_SALES", "SP_PRE_VAT", "CP_PRE_VAT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # receipt key
    for c in ["STORE_CODE", "TILL", "SESSION", "RCT"]:
        if c not in df.columns:
            df[c] = ""
    df["CUST_CODE"] = (
        df["STORE_CODE"] + "-" + df["TILL"] + "-" + df["SESSION"] + "-" + df["RCT"]
    )

    return df


# ---------------------------------------------------------
# SALES REPORTS
# ---------------------------------------------------------
def sales_global_overview(df):
    st.header("Global sales Overview")
    if 'SALES_CHANNEL_L1' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SALES_CHANNEL_L1 or NET_SALES")
        return
    g = agg_net_sales_by(df, 'SALES_CHANNEL_L1')
    g['NET_SALES_M'] = g['NET_SALES'] / 1_000_000
    fig = donut_from_agg(
        g,
        'SALES_CHANNEL_L1',
        'NET_SALES',
        "<b>SALES CHANNEL TYPE — Global Overview</b>",
        hole=0.65,
        value_is_millions=True
    )
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(
        g[['SALES_CHANNEL_L1', 'NET_SALES']],
        numeric_cols=['NET_SALES'],
        index_col='SALES_CHANNEL_L1',
        total_label='TOTAL'
    )


# ---------------------------------------------------------
# OPERATIONS
# ---------------------------------------------------------
def cashier_perf(df):
    st.subheader("Cashier Performance")

    if "CUST_CODE" not in df.columns or "NET_SALES" not in df.columns:
        st.warning("Missing required columns.")
        return

    if "CASHIER" not in df.columns:
        df["CASHIER"] = "Unknown"

    perf = df.groupby("CASHIER").agg(
        NET_SALES=("NET_SALES", "sum"),
        RECEIPTS=("CUST_CODE", "nunique")
    ).reset_index()

    st.dataframe(perf)


# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
def main():
    st.title("Daily Deck – Supabase Version")

    date_basis, start_date, end_date, section = sidebar_config()

    st.write(f"**Loading data for {date_basis}: {start_date} → {end_date}**")

    df = load_supabase_data(date_basis, start_date, end_date)

    if df.empty:
        st.warning("No data for selected period.")
        return

    df = clean_df(df, date_basis)

    if section == "SALES":
        sales_global(df)

    elif section == "OPERATIONS":
        cashier_perf(df)

    else:
        st.info("INSIGHTS coming soon.")


if __name__ == "__main__":
    main()

