###############################################################
#  DAILYDECK v2  (SUPABASE-ONLY, MULTI-DAY, TREND DRIVEN)
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
# 2. LOAD DATA FROM SUPABASE (MULTI-DAY)
################################################################

@st.cache_data(show_spinner=True)
def load_from_supabase(start_date: date, end_date: date) -> pd.DataFrame:

    client = get_supabase_client()

    start_iso = f"{start_date}T00:00:00"
    end_iso   = f"{end_date}T23:59:59.999999"

    data = (
        client.table("daily_pos_trn_items_clean")
        .select("*")
        .gte("TRN_DATE", start_iso)
        .lte("TRN_DATE", end_iso)
        .limit(1_000_000)
        .execute()
        .data
    )

    df = pd.DataFrame(data)
    return df


################################################################
# 3. DATA CLEANING & DERIVED COLUMNS
################################################################

@st.cache_data
def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df

    d = df.copy()

    # Convert dates
    d["TRN_DATE"] = pd.to_datetime(d["TRN_DATE"], errors="coerce")
    d = d.dropna(subset=["TRN_DATE"]).copy()
    d["DATE"] = d["TRN_DATE"].dt.date
    d["TIME_INTERVAL"] = d["TRN_DATE"].dt.floor("30min")
    d["TIME_ONLY"] = d["TIME_INTERVAL"].dt.time

    # Convert numeric fields
    num_cols = ["QTY", "SP_PRE_VAT", "NET_SALES", "VAT_AMT"]
    for c in num_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(
                d[c].astype(str).str.replace(",", ""), errors="coerce"
            ).fillna(0)

    # Gross
    d["GROSS_SALES"] = d["NET_SALES"] + d["VAT_AMT"]

    # CUST_CODE
    if all(c in d.columns for c in ["STORE_CODE", "TILL", "SESSION", "RCT"]):
        d["CUST_CODE"] = (
            d["STORE_CODE"].astype(str) + "-" +
            d["TILL"].astype(str)        + "-" +
            d["SESSION"].astype(str)     + "-" +
            d["RCT"].astype(str)
        )

    # Day/Night shift bucket
    if "SHIFT" in d.columns:
        d["Shift_Bucket"] = np.where(
            d["SHIFT"].str.contains("NIGHT", case=False, na=False),
            "Night", "Day"
        )

    return d


################################################################
# 4. GENERIC TREND PANEL
################################################################

def show_trends(df: pd.DataFrame, section: str):

    st.markdown("---")
    st.subheader("📈 Trends in Selected Period")

    if df.empty:
        st.info("No data available for trends.")
        return

    col1, col2 = st.columns(2)

    # Universal daily net sales trend
    with col1:
        if "NET_SALES" in df.columns:
            daily = df.groupby("DATE", as_index=False)["NET_SALES"].sum()
            fig = px.line(
                daily, x="DATE", y="NET_SALES",
                markers=True, title="Daily Net Sales Trend"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Section-specific trend
    with col2:
        if section == "SALES" and "SALES_CHANNEL_L1" in df.columns:
            trend = (
                df.groupby(["DATE", "SALES_CHANNEL_L1"], as_index=False)
                ["NET_SALES"].sum()
            )
            fig = px.line(
                trend, x="DATE", y="NET_SALES",
                color="SALES_CHANNEL_L1", markers=True,
                title="Sales by Channel Trend"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif section == "OPERATIONS":
            tr = df.groupby("DATE")["CUST_CODE"].nunique().reset_index()
            fig = px.line(
                tr, x="DATE", y="CUST_CODE", markers=True,
                title="Customer Traffic Trend"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif section == "INSIGHTS":
            tr = df.groupby("DATE")["CUST_CODE"].nunique().reset_index()
            fig = px.line(
                tr, x="DATE", y="CUST_CODE", markers=True,
                title="Basket Count Trend"
            )
            st.plotly_chart(fig, use_container_width=True)


################################################################
# 5. SALES / OPERATIONS / INSIGHTS SUBSECTIONS
################################################################

def sales_global_overview(df):
    st.subheader("🌍 Global Sales Overview")

    g = df.groupby("STORE_NAME", as_index=False)["NET_SALES"].sum().sort_values(
        "NET_SALES", ascending=False
    )
    st.dataframe(g, use_container_width=True)

    fig = px.bar(
        g, x="STORE_NAME", y="NET_SALES",
        title="Net Sales by Store"
    )
    st.plotly_chart(fig, use_container_width=True)


def sales_by_channel_l2(df):
    st.subheader("Sales by Channel L2")

    if "SALES_CHANNEL_L2" not in df.columns:
        st.warning("SALES_CHANNEL_L2 missing.")
        return

    g = df.groupby("SALES_CHANNEL_L2", as_index=False)["NET_SALES"].sum()
    fig = px.pie(g, names="SALES_CHANNEL_L2", values="NET_SALES")
    st.plotly_chart(fig, use_container_width=True)


def sales_by_shift(df):
    st.subheader("Sales by Shift")

    g = df.groupby("Shift_Bucket", as_index=False)["NET_SALES"].sum()
    fig = px.bar(g, x="Shift_Bucket", y="NET_SALES")
    st.plotly_chart(fig, use_container_width=True)


def customer_traffic_storewise(df):
    st.subheader("Customer Traffic (Storewise)")

    g = df.groupby("STORE_NAME")["CUST_CODE"].nunique().reset_index()
    fig = px.bar(g, x="STORE_NAME", y="CUST_CODE")
    st.plotly_chart(fig, use_container_width=True)


def cashiers_performance(df):
    st.subheader("Cashiers Performance")

    if "CASHIER" not in df.columns:
        st.warning("Missing CASHIER column")
        return

    g = df.groupby("CASHIER", as_index=False)["NET_SALES"].sum()
    fig = px.bar(g, x="CASHIER", y="NET_SALES")
    st.plotly_chart(fig, use_container_width=True)


def category_overview(df):
    st.subheader("Category Sales Overview")

    if "CATEGORY" not in df.columns:
        st.warning("Category column missing")
        return

    g = df.groupby("CATEGORY", as_index=False)["NET_SALES"].sum()
    fig = px.bar(g, x="CATEGORY", y="NET_SALES")
    st.plotly_chart(fig, use_container_width=True)


################################################################
# 6. MAIN APP
################################################################

def main():

    st.title("📊 DailyDeck — Multi-Day Retail Performance Dashboard")

    ###############################################################
    # Sidebar: Date Range
    ###############################################################

    st.sidebar.markdown("### Select Date Range")

    today = date.today()
    start_date = st.sidebar.date_input(
        "Start date", today - timedelta(days=7)
    )
    end_date = st.sidebar.date_input(
        "End date", today
    )

    if start_date > end_date:
        st.sidebar.error("❌ Start date cannot be after end date")
        st.stop()

    ###############################################################
    # Load Data
    ###############################################################

    with st.spinner("Fetching data from Supabase..."):
        raw_df = load_from_supabase(start_date, end_date)

    if raw_df.empty:
        st.warning("No transactions found for this period.")
        st.stop()

    df = clean_and_prepare(raw_df)

    ###############################################################
    # Main Navigation
    ###############################################################

    section = st.sidebar.radio(
        "Section",
        ["SALES", "OPERATIONS", "INSIGHTS"]
    )

    ###############################################################
    # SALES
    ###############################################################

    if section == "SALES":
        opt = st.sidebar.selectbox(
            "Choose metric",
            [
                "Global Sales Overview",
                "Sales by Channel",
                "Sales by Shift"
            ]
        )

        if opt == "Global Sales Overview":
            sales_global_overview(df)

        elif opt == "Sales by Channel":
            sales_by_channel_l2(df)

        elif opt == "Sales by Shift":
            sales_by_shift(df)

        show_trends(df, "SALES")

    ###############################################################
    # OPERATIONS
    ###############################################################

    elif section == "OPERATIONS":
        opt = st.sidebar.selectbox(
            "Choose metric",
            [
                "Customer Traffic Storewise",
                "Cashiers Performance"
            ]
        )

        if opt == "Customer Traffic Storewise":
            customer_traffic_storewise(df)

        elif opt == "Cashiers Performance":
            cashiers_performance(df)

        show_trends(df, "OPERATIONS")

    ###############################################################
    # INSIGHTS
    ###############################################################

    elif section == "INSIGHTS":
        opt = st.sidebar.selectbox(
            "Choose metric",
            ["Category Overview"]
        )

        if opt == "Category Overview":
            category_overview(df)

        show_trends(df, "INSIGHTS")


################################################################
# RUN APP
################################################################

if __name__ == "__main__":
    main()
