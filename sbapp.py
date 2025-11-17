###############################################################
#  DAILYDECK v2  (SUPABASE-ONLY, MULTI-DAY, LOWERCASE VERSION)
###############################################################

import os
from datetime import date, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from supabase import create_client, Client

st.set_page_config(
    page_title="DailyDeck",
    layout="wide"
)

################################################################
# 1. SUPABASE CONNECTION
################################################################

@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

################################################################
# 2. LOAD DATA (ALL LOWERCASE COLUMNS)
################################################################

@st.cache_data(show_spinner=True)
def load_from_supabase(start_date: date, end_date: date) -> pd.DataFrame:

    client = get_supabase_client()

    start_iso = f"{start_date}T00:00:00"
    end_iso   = f"{end_date}T23:59:59.999999"

    data = (
        client.table("daily_pos_trn_items_clean")
        .select("*")
        .gte("trn_date", start_iso)     # FIXED (lowercase)
        .lte("trn_date", end_iso)
        .limit(1_000_000)
        .execute()
        .data
    )

    return pd.DataFrame(data)

################################################################
# 3. CLEAN & PREPARE (LOWERCASE EVERYTHING)
################################################################

@st.cache_data
def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df

    d = df.copy()

    # Convert date
    d["trn_date"] = pd.to_datetime(d["trn_date"], errors="coerce")
    d = d.dropna(subset=["trn_date"]).copy()

    d["date"] = d["trn_date"].dt.date
    d["time_interval"] = d["trn_date"].dt.floor("30min")
    d["time_only"] = d["time_interval"].dt.time

    # Convert numeric fields
    num_cols = ["qty", "sp_pre_vat", "net_sales", "vat_amt"]
    for c in num_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(
                d[c].astype(str).str.replace(",", ""), errors="coerce"
            ).fillna(0)

    # Gross sales
    d["gross_sales"] = d["net_sales"] + d["vat_amt"]

    # CUST_CODE
    if all(c in d.columns for c in ["store_code", "till", "session", "rct"]):
        d["cust_code"] = (
            d["store_code"].astype(str) + "-" +
            d["till"].astype(str)        + "-" +
            d["session"].astype(str)     + "-" +
            d["rct"].astype(str)
        )

    # Day/Night shift
    if "shift" in d.columns:
        d["shift_bucket"] = np.where(
            d["shift"].str.contains("night", case=False, na=False),
            "night", "day"
        )

    return d

################################################################
# 4. TRENDS PANEL
################################################################

def show_trends(df: pd.DataFrame, section: str):

    st.markdown("---")
    st.subheader("📈 Trends in Selected Period")

    if df.empty:
        st.info("No data available for trends.")
        return

    col1, col2 = st.columns(2)

    # Daily net sales
    with col1:
        daily = df.groupby("date", as_index=False)["net_sales"].sum()
        fig = px.line(daily, x="date", y="net_sales", markers=True,
                      title="Daily Net Sales Trend")
        st.plotly_chart(fig, use_container_width=True)

    # Section-specific trends
    with col2:
        if section == "SALES" and "sales_channel_l1" in df.columns:
            trend = (
                df.groupby(["date", "sales_channel_l1"], as_index=False)
                ["net_sales"].sum()
            )
            fig = px.line(
                trend, x="date", y="net_sales",
                color="sales_channel_l1", markers=True,
                title="Sales by Channel Trend"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif section == "OPERATIONS":
            tr = df.groupby("date")["cust_code"].nunique().reset_index()
            fig = px.line(
                tr, x="date", y="cust_code", markers=True,
                title="Customer Traffic Trend"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif section == "INSIGHTS":
            tr = df.groupby("date")["cust_code"].nunique().reset_index()
            fig = px.line(
                tr, x="date", y="cust_code", markers=True,
                title="Basket Count Trend"
            )
            st.plotly_chart(fig, use_container_width=True)

################################################################
# 5. SUBSECTIONS
################################################################

def sales_global_overview(df):
    st.subheader("🌍 Global Sales Overview")
    g = df.groupby("store_name", as_index=False)["net_sales"].sum()\
          .sort_values("net_sales", ascending=False)
    st.dataframe(g, use_container_width=True)
    st.plotly_chart(px.bar(g, x="store_name", y="net_sales"), use_container_width=True)

def sales_by_channel_l2(df):
    st.subheader("Sales by Channel L2")
    if "sales_channel_l2" not in df.columns:
        st.warning("Column 'sales_channel_l2' missing")
        return
    g = df.groupby("sales_channel_l2", as_index=False)["net_sales"].sum()
    st.plotly_chart(px.pie(g, names="sales_channel_l2", values="net_sales"), use_container_width=True)

def sales_by_shift(df):
    st.subheader("Sales by Shift")
    g = df.groupby("shift_bucket", as_index=False)["net_sales"].sum()
    st.plotly_chart(px.bar(g, x="shift_bucket", y="net_sales"), use_container_width=True)

def customer_traffic_storewise(df):
    st.subheader("Customer Traffic (Storewise)")
    g = df.groupby("store_name")["cust_code"].nunique().reset_index()
    st.plotly_chart(px.bar(g, x="store_name", y="cust_code"), use_container_width=True)

def cashiers_performance(df):
    st.subheader("Cashiers Performance")
    if "cashier" not in df.columns:
        st.warning("Column 'cashier' missing")
        return
    g = df.groupby("cashier", as_index=False)["net_sales"].sum()
    st.plotly_chart(px.bar(g, x="cashier", y="net_sales"), use_container_width=True)

def category_overview(df):
    st.subheader("Category Overview")
    if "category" not in df.columns:
        st.warning("Column 'category' missing")
        return
    g = df.groupby("category", as_index=False)["net_sales"].sum()
    st.plotly_chart(px.bar(g, x="category", y="net_sales"), use_container_width=True)

################################################################
# 6. MAIN APP
################################################################

def main():

    st.title("📊 DailyDeck — Multi-Day Retail Dashboard (Lowercase Edition)")

    st.sidebar.markdown("### Select Date Range")
    today = date.today()

    start_date = st.sidebar.date_input("Start date", today - timedelta(days=7))
    end_date   = st.sidebar.date_input("End date", today)

    if start_date > end_date:
        st.sidebar.error("❌ Start date cannot be after end date")
        st.stop()

    with st.spinner("Fetching data from Supabase..."):
        raw = load_from_supabase(start_date, end_date)

    if raw.empty:
        st.warning("No records found for the selected period.")
        st.stop()

    df = clean_and_prepare(raw)

    section = st.sidebar.radio("Section", ["SALES", "OPERATIONS", "INSIGHTS"])

    if section == "SALES":
        opt = st.sidebar.selectbox("Choose metric", [
            "Global Sales Overview",
            "Sales by Channel",
            "Sales by Shift"
        ])
        if opt == "Global Sales Overview": sales_global_overview(df)
        elif opt == "Sales by Channel":    sales_by_channel_l2(df)
        elif opt == "Sales by Shift":      sales_by_shift(df)
        show_trends(df, section)

    elif section == "OPERATIONS":
        opt = st.sidebar.selectbox("Choose metric", [
            "Customer Traffic Storewise",
            "Cashiers Performance"
        ])
        if opt == "Customer Traffic Storewise": customer_traffic_storewise(df)
        elif opt == "Cashiers Performance":     cashiers_performance(df)
        show_trends(df, section)

    elif section == "INSIGHTS":
        opt = st.sidebar.selectbox("Choose metric", ["Category Overview"])
        if opt == "Category Overview": category_overview(df)
        show_trends(df, section)

################################################################
# RUN
################################################################

if __name__ == "__main__":
    main()
