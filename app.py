import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import os

st.set_page_config(layout="wide", page_title="Superdeck (Streamlit)")
# Add the following at the very top of your Streamlit script, AFTER st.set_page_config



# -----------------------
# Data Loading & Caching
# -----------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines='skip', low_memory=False)

@st.cache_data
def load_uploaded_file(contents: bytes) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_csv(BytesIO(contents), on_bad_lines='skip', low_memory=False)

def smart_load():
    st.sidebar.markdown("### Upload data (CSV) or use default")

    # 1) Prefer uploaded file
    uploaded = st.sidebar.file_uploader(
        "Upload DAILY_POS_TRN_ITEMS CSV",
        type=["csv"]
    )
    if uploaded is not None:
        with st.spinner("Parsing uploaded CSV..."):
            df = load_uploaded_file(uploaded.getvalue())
        st.sidebar.success("Loaded uploaded CSV")
        return df

    # 2) Otherwise, try bundled default CSV in the repo
    base_dir = os.path.dirname(__file__)
    default_path = os.path.join(
        base_dir,
        "data",
        "DAILY_POS_TRN_ITEMS_2025-10-21.csv"
    )

    if os.path.exists(default_path):
        try:
            with st.spinner(f"Loading default CSV from {default_path}"):
                df = load_csv(default_path)
            st.sidebar.info("Using bundled default dataset")
            return df
        except Exception as e:
            st.sidebar.error(f"Failed to load default CSV: {e}")

    # 3) If nothing worked, tell the user and return None
    st.sidebar.warning(
        "No data loaded. Please upload a DAILY_POS_TRN_ITEMS CSV in the sidebar."
    )
    return None

# -----------------------
# Robust cleaning + derived columns (cached)
# -----------------------
@st.cache_data
def clean_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Basic column standardization ---
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Rename key columns if needed (just in case of slight mismatches)
    rename_map = {
        "BUSINESS_DAY": "BUSINESS_DAY",
        "BRANCH": "BRANCH",
        "DEPARTMENT": "DEPARTMENT",
        "SUB_DEPARTMENT": "SUB_DEPARTMENT",
        "CATEGORY": "CATEGORY",
        "SUB_CATEGORY": "SUB_CATEGORY",
        "ITEM_CODE": "ITEM_CODE",
        "ITEM_NAME": "ITEM_NAME",
        "QUANTITY": "QUANTITY",
        "NET_SALES": "NET_SALES",
        "GROSS_SALES": "GROSS_SALES",
        "REFUND_QTY": "REFUND_QTY",
        "REFUND_AMT": "REFUND_AMT",
        "VOID_QTY": "VOID_QTY",
        "VOID_AMT": "VOID_AMT",
        "DISC_AMT": "DISC_AMT",
        "VAT_AMT": "VAT_AMT",
        "SALES_CHANNEL": "SALES_CHANNEL",
        "SHIFT": "SHIFT",
        "CASHIER_NAME": "CASHIER_NAME",
        "TILL_NO": "TILL_NO",
        "BUSINESS_HOUR": "BUSINESS_HOUR",
        "BUSINESS_DAY_NAME": "BUSINESS_DAY_NAME",
        "INVOICE_NO": "INVOICE_NO",
        "CUSTOMER_TYPE": "CUSTOMER_TYPE",
        "LOYALTY_CARD_NO": "LOYALTY_CARD_NO",
        "LOYALTY_CUSTOMER_CODE": "LOYALTY_CUSTOMER_CODE",
        "SUPPLIER_NAME": "SUPPLIER_NAME",
        "UNIT_COST": "UNIT_COST"
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # --- Ensure required columns exist ---
    required_cols = [
        "BUSINESS_DAY", "BRANCH", "DEPARTMENT", "CATEGORY", "ITEM_CODE", "ITEM_NAME",
        "QUANTITY", "NET_SALES", "GROSS_SALES", "REFUND_QTY", "REFUND_AMT",
        "VOID_QTY", "VOID_AMT", "DISC_AMT", "VAT_AMT", "SALES_CHANNEL", "SHIFT",
        "CASHIER_NAME", "TILL_NO", "BUSINESS_HOUR", "BUSINESS_DAY_NAME",
        "INVOICE_NO", "CUSTOMER_TYPE", "LOYALTY_CARD_NO", "LOYALTY_CUSTOMER_CODE",
        "SUPPLIER_NAME", "UNIT_COST"
    ]
    for col in required_cols:
        if col not in df.columns:
            if col in ["REFUND_QTY", "VOID_QTY", "QUANTITY"]:
                df[col] = 0
            elif col in ["REFUND_AMT", "VOID_AMT", "DISC_AMT", "VAT_AMT", "NET_SALES", "GROSS_SALES", "UNIT_COST"]:
                df[col] = 0.0
            else:
                df[col] = "UNKNOWN"

    # --- Convert dtypes ---
    numeric_cols = [
        "QUANTITY", "NET_SALES", "GROSS_SALES", "REFUND_QTY", "REFUND_AMT",
        "VOID_QTY", "VOID_AMT", "DISC_AMT", "VAT_AMT", "UNIT_COST"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Date + time
    df["BUSINESS_DAY"] = pd.to_datetime(df["BUSINESS_DAY"], errors="coerce")
    df["BUSINESS_HOUR"] = pd.to_numeric(df["BUSINESS_HOUR"], errors="coerce")
    df["BUSINESS_DAY_NAME"] = df["BUSINESS_DAY_NAME"].astype(str).str.strip()

    # Clean strings
    for col in ["BRANCH", "DEPARTMENT", "SUB_DEPARTMENT", "CATEGORY",
                "SUB_CATEGORY", "ITEM_NAME", "SALES_CHANNEL", "SHIFT",
                "CASHIER_NAME", "TILL_NO", "INVOICE_NO", "CUSTOMER_TYPE",
                "LOYALTY_CARD_NO", "LOYALTY_CUSTOMER_CODE", "SUPPLIER_NAME"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # --- Derived Metrics ---
    # Effective quantity = quantity - refunds - voids
    df["EFFECTIVE_QTY"] = df["QUANTITY"] - df["REFUND_QTY"] - df["VOID_QTY"]

    # Effective sales = net sales - refunds - voids
    df["EFFECTIVE_SALES"] = df["NET_SALES"] - df["REFUND_AMT"] - df["VOID_AMT"]

    # Basket size / invoice metrics
    invoice_group = df.groupby("INVOICE_NO").agg(
        BASKET_QTY=("QUANTITY", "sum"),
        BASKET_LINES=("ITEM_CODE", "nunique"),
        BASKET_NET_SALES=("NET_SALES", "sum"),
        BASKET_GROSS_SALES=("GROSS_SALES", "sum")
    ).reset_index()

    df = df.merge(invoice_group, on="INVOICE_NO", how="left")

    # Branch-day-level aggregates
    branch_day_group = df.groupby(["BUSINESS_DAY", "BRANCH"]).agg(
        BRANCH_DAY_NET_SALES=("NET_SALES", "sum"),
        BRANCH_DAY_GROSS_SALES=("GROSS_SALES", "sum"),
        BRANCH_DAY_CUSTOMERS=("INVOICE_NO", "nunique"),
        BRANCH_DAY_LINES=("ITEM_CODE", "nunique"),
        BRANCH_DAY_QTY=("QUANTITY", "sum")
    ).reset_index()

    df = df.merge(branch_day_group,
                  on=["BUSINESS_DAY", "BRANCH"],
                  how="left")

    # Category level metrics
    cat_group = df.groupby(["BUSINESS_DAY", "BRANCH", "CATEGORY"]).agg(
        CAT_NET_SALES=("NET_SALES", "sum"),
        CAT_GROSS_SALES=("GROSS_SALES", "sum"),
        CAT_QTY=("QUANTITY", "sum")
    ).reset_index()

    df = df.merge(cat_group,
                  on=["BUSINESS_DAY", "BRANCH", "CATEGORY"],
                  how="left",
                  suffixes=("", "_CATLVL"))

    # Day-of-week, hour-of-day
    df["DAY_OF_WEEK"] = df["BUSINESS_DAY"].dt.day_name()
    df["HOUR_OF_DAY"] = df["BUSINESS_HOUR"].astype(int)

    # Flags
    df["IS_REFUND"] = np.where(df["REFUND_QTY"] > 0, 1, 0)
    df["IS_VOID"] = np.where(df["VOID_QTY"] > 0, 1, 0)

    # Basket metrics at branch level
    branch_basket = df.groupby("BRANCH").agg(
        AVG_BASKET_VALUE=("BASKET_NET_SALES", "mean"),
        AVG_BASKET_QTY=("BASKET_QTY", "mean"),
        AVG_BASKET_LINES=("BASKET_LINES", "mean"),
        TOTAL_BASKETS=("INVOICE_NO", "nunique")
    ).reset_index()
    df = df.merge(branch_basket, on="BRANCH", how="left")

    # Price / margin proxies
    df["UNIT_PRICE"] = np.where(df["QUANTITY"] != 0,
                                df["NET_SALES"] / df["QUANTITY"],
                                0)
    df["UNIT_MARGIN"] = df["UNIT_PRICE"] - df["UNIT_COST"]

    return df

# -----------------------
# Helper functions
# -----------------------
def format_currency(x):
    return f"KES {x:,.0f}"

def safe_div(n, d):
    if d == 0 or pd.isna(d):
        return 0
    return n / d

# -----------------------
# Sales Section Functions
# -----------------------
def sales_global_overview(df: pd.DataFrame):
    st.subheader("Global Sales Overview")

    total_net_sales = df["NET_SALES"].sum()
    total_gross_sales = df["GROSS_SALES"].sum()
    total_customers = df["INVOICE_NO"].nunique()
    total_items_sold = df["QUANTITY"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Net Sales", format_currency(total_net_sales))
    col2.metric("Total Gross Sales", format_currency(total_gross_sales))
    col3.metric("Total Customers", f"{total_customers:,}")
    col4.metric("Total Items Sold", f"{total_items_sold:,}")

    # Sales trend over time
    daily = df.groupby("BUSINESS_DAY").agg(
        NET_SALES=("NET_SALES", "sum"),
        GROSS_SALES=("GROSS_SALES", "sum"),
        CUSTOMERS=("INVOICE_NO", "nunique")
    ).reset_index()

    fig = px.line(
        daily,
        x="BUSINESS_DAY",
        y="NET_SALES",
        title="Net Sales Trend Over Time",
        markers=True
    )
    fig.update_layout(yaxis_title="Net Sales (KES)")
    st.plotly_chart(fig, use_container_width=True)

    # Sales by branch
    branch_sales = df.groupby("BRANCH").agg(
        NET_SALES=("NET_SALES", "sum"),
        GROSS_SALES=("GROSS_SALES", "sum")
    ).reset_index().sort_values("NET_SALES", ascending=False)

    fig2 = px.bar(
        branch_sales,
        x="BRANCH",
        y="NET_SALES",
        title="Net Sales by Branch",
        text_auto=".2s"
    )
    fig2.update_layout(yaxis_title="Net Sales (KES)")
    st.plotly_chart(fig2, use_container_width=True)

def sales_by_channel_l2(df: pd.DataFrame):
    st.subheader("Global Net Sales Distribution by Sales Channel")

    channel_sales = df.groupby("SALES_CHANNEL").agg(
        NET_SALES=("NET_SALES", "sum"),
        GROSS_SALES=("GROSS_SALES", "sum"),
        CUSTOMERS=("INVOICE_NO", "nunique")
    ).reset_index()

    channel_sales["CONTRIBUTION"] = channel_sales["NET_SALES"] / channel_sales["NET_SALES"].sum()

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.bar(
            channel_sales.sort_values("NET_SALES", ascending=False),
            x="SALES_CHANNEL",
            y="NET_SALES",
            text_auto=".2s",
            title="Net Sales by Channel"
        )
        fig.update_layout(yaxis_title="Net Sales (KES)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            channel_sales,
            values="NET_SALES",
            names="SALES_CHANNEL",
            title="Share of Net Sales by Channel"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.dataframe(
        channel_sales[["SALES_CHANNEL", "NET_SALES", "GROSS_SALES", "CUSTOMERS", "CONTRIBUTION"]]
        .sort_values("NET_SALES", ascending=False)
        .style.format({
            "NET_SALES": "KES {:,.0f}",
            "GROSS_SALES": "KES {:,.0f}",
            "CONTRIBUTION": "{:.1%}"
        })
    )

def sales_by_shift(df: pd.DataFrame):
    st.subheader("Global Net Sales Distribution by Shift")

    shift_sales = df.groupby("SHIFT").agg(
        NET_SALES=("NET_SALES", "sum"),
        GROSS_SALES=("GROSS_SALES", "sum"),
        CUSTOMERS=("INVOICE_NO", "nunique")
    ).reset_index()

    shift_sales["CONTRIBUTION"] = shift_sales["NET_SALES"] / shift_sales["NET_SALES"].sum()

    fig = px.bar(
        shift_sales.sort_values("NET_SALES", ascending=False),
        x="SHIFT",
        y="NET_SALES",
        text_auto=".2s",
        title="Net Sales by Shift"
    )
    fig.update_layout(yaxis_title="Net Sales (KES)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        shift_sales[["SHIFT", "NET_SALES", "GROSS_SALES", "CUSTOMERS", "CONTRIBUTION"]]
        .sort_values("NET_SALES", ascending=False)
        .style.format({
            "NET_SALES": "KES {:,.0f}",
            "GROSS_SALES": "KES {:,.0f}",
            "CONTRIBUTION": "{:.1%}"
        })
    )

def night_vs_day_ratio(df: pd.DataFrame):
    st.subheader("Night vs Day Shift Sales Ratio — Stores with Night Shifts")

    # Consider 'NIGHT' vs everything else as 'DAY'
    df_night_stores = df[df["SHIFT"].str.upper() == "NIGHT"]["BRANCH"].unique()
    df_filtered = df[df["BRANCH"].isin(df_night_stores)]

    df_filtered["SHIFT_BUCKET"] = np.where(df_filtered["SHIFT"].str.upper() == "NIGHT", "NIGHT", "DAY")

    ratio = df_filtered.groupby(["BRANCH", "SHIFT_BUCKET"]).agg(
        NET_SALES=("NET_SALES", "sum")
    ).reset_index()

    pivot = ratio.pivot(index="BRANCH", columns="SHIFT_BUCKET", values="NET_SALES").fillna(0)
    pivot["NIGHT_TO_DAY_RATIO"] = pivot.apply(
        lambda row: safe_div(row.get("NIGHT", 0), row.get("DAY", 0)),
        axis=1
    )
    pivot = pivot.reset_index()

    fig = px.bar(
        pivot.sort_values("NIGHT_TO_DAY_RATIO", ascending=False),
        x="BRANCH",
        y="NIGHT_TO_DAY_RATIO",
        title="Night to Day Net Sales Ratio by Branch (Night-Shift Stores Only)",
        text_auto=".2f"
    )
    fig.update_layout(yaxis_title="Night/Day Sales Ratio")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        pivot[["BRANCH", "DAY", "NIGHT", "NIGHT_TO_DAY_RATIO"]]
        .sort_values("NIGHT_TO_DAY_RATIO", ascending=False)
        .style.format({
            "DAY": "KES {:,.0f}",
            "NIGHT": "KES {:,.0f}",
            "NIGHT_TO_DAY_RATIO": "{:.2f}"
        })
    )

def global_day_vs_night(df: pd.DataFrame):
    st.subheader("Global Day vs Night Sales — Only Stores with NIGHT Shift")

    df_night_stores = df[df["SHIFT"].str.upper() == "NIGHT"]["BRANCH"].unique()
    df_filtered = df[df["BRANCH"].isin(df_night_stores)]

    df_filtered["SHIFT_BUCKET"] = np.where(df_filtered["SHIFT"].str.upper() == "NIGHT", "NIGHT", "DAY")

    agg = df_filtered.groupby("SHIFT_BUCKET").agg(
        NET_SALES=("NET_SALES", "sum"),
        GROSS_SALES=("GROSS_SALES", "sum"),
        CUSTOMERS=("INVOICE_NO", "nunique")
    ).reset_index()

    fig = px.bar(
        agg,
        x="SHIFT_BUCKET",
        y="NET_SALES",
        text_auto=".2s",
        title="Global Net Sales by Day vs Night (Night-Shift Stores)"
    )
    fig.update_layout(yaxis_title="Net Sales (KES)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        agg[["SHIFT_BUCKET", "NET_SALES", "GROSS_SALES", "CUSTOMERS"]]
        .sort_values("NET_SALES", ascending=False)
        .style.format({
            "NET_SALES": "KES {:,.0f}",
            "GROSS_SALES": "KES {:,.0f}"
        })
    )

def second_highest_channel_share(df: pd.DataFrame):
    st.subheader("2nd-Highest Channel Share by Branch")

    # Calculate channel share within each branch
    branch_channel = df.groupby(["BRANCH", "SALES_CHANNEL"]).agg(
        NET_SALES=("NET_SALES", "sum")
    ).reset_index()

    branch_totals = branch_channel.groupby("BRANCH")["NET_SALES"].transform("sum")
    branch_channel["SHARE"] = branch_channel["NET_SALES"] / branch_totals

    # Rank channels within each branch
    branch_channel["RANK_IN_BRANCH"] = branch_channel.groupby("BRANCH")["SHARE"] \
        .rank(method="dense", ascending=False)

    second_highest = branch_channel[branch_channel["RANK_IN_BRANCH"] == 2] \
        .sort_values("SHARE", ascending=False)

    fig = px.bar(
        second_highest,
        x="BRANCH",
        y="SHARE",
        color="SALES_CHANNEL",
        title="2nd-Highest Channel Share by Branch",
        text_auto=".1%"
    )
    fig.update_layout(yaxis_title="Share of Branch Net Sales")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        second_highest[["BRANCH", "SALES_CHANNEL", "NET_SALES", "SHARE"]]
        .sort_values("SHARE", ascending=False)
        .style.format({
            "NET_SALES": "KES {:,.0f}",
            "SHARE": "{:.1%}"
        })
    )

def bottom_30_2nd_highest(df: pd.DataFrame):
    st.subheader("Bottom 30 Branches by 2nd-Highest Channel Share")

    branch_channel = df.groupby(["BRANCH", "SALES_CHANNEL"]).agg(
        NET_SALES=("NET_SALES", "sum")
    ).reset_index()

    branch_totals = branch_channel.groupby("BRANCH")["NET_SALES"].transform("sum")
    branch_channel["SHARE"] = branch_channel["NET_SALES"] / branch_totals

    branch_channel["RANK_IN_BRANCH"] = branch_channel.groupby("BRANCH")["SHARE"] \
        .rank(method="dense", ascending=False)

    second_highest = branch_channel[branch_channel["RANK_IN_BRANCH"] == 2]

    bottom_30 = second_highest.sort_values("SHARE", ascending=True).head(30)

    fig = px.bar(
        bottom_30,
        x="BRANCH",
        y="SHARE",
        color="SALES_CHANNEL",
        title="Bottom 30 Branches by 2nd-Highest Channel Share",
        text_auto=".1%"
    )
    fig.update_layout(yaxis_title="Share of Branch Net Sales")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        bottom_30[["BRANCH", "SALES_CHANNEL", "NET_SALES", "SHARE"]]
        .sort_values("SHARE", ascending=True)
        .style.format({
            "NET_SALES": "KES {:,.0f}",
            "SHARE": "{:.1%}"
        })
    )

def stores_sales_summary(df: pd.DataFrame):
    st.subheader("Stores Sales Summary")

    summary = df.groupby("BRANCH").agg(
        NET_SALES=("NET_SALES", "sum"),
        GROSS_SALES=("GROSS_SALES", "sum"),
        CUSTOMERS=("INVOICE_NO", "nunique"),
        ITEMS_SOLD=("QUANTITY", "sum"),
        AVG_BASKET_VALUE=("BASKET_NET_SALES", "mean"),
        AVG_BASKET_QTY=("BASKET_QTY", "mean")
    ).reset_index()

    summary["SALES_PER_CUSTOMER"] = summary["NET_SALES"] / summary["CUSTOMERS"]
    summary["ITEMS_PER_CUSTOMER"] = summary["ITEMS_SOLD"] / summary["CUSTOMERS"]

    st.dataframe(
        summary.sort_values("NET_SALES", ascending=False)
        .style.format({
            "NET_SALES": "KES {:,.0f}",
            "GROSS_SALES": "KES {:,.0f}",
            "AVG_BASKET_VALUE": "KES {:,.0f}",
            "SALES_PER_CUSTOMER": "KES {:,.0f}"
        })
    )

    fig = px.bar(
        summary.sort_values("NET_SALES", ascending=False),
        x="BRANCH",
        y="NET_SALES",
        title="Net Sales by Store",
        text_auto=".2s"
    )
    fig.update_layout(yaxis_title="Net Sales (KES)")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Operations Section Functions
# -----------------------
def customer_traffic_storewise(df: pd.DataFrame):
    st.subheader("Customer Traffic-Storewise")

    branch_traffic = df.groupby("BRANCH").agg(
        CUSTOMERS=("INVOICE_NO", "nunique")
    ).reset_index().sort_values("CUSTOMERS", ascending=False)

    fig = px.bar(
        branch_traffic,
        x="BRANCH",
        y="CUSTOMERS",
        title="Number of Customers by Store",
        text_auto=".2s"
    )
    fig.update_layout(yaxis_title="Customers")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        branch_traffic
        .style.format({
            "CUSTOMERS": "{:,.0f}"
        })
    )

def active_tills_during_day(df: pd.DataFrame):
    st.subheader("Active Tills During the Day")

    active_tills = df.groupby(["BUSINESS_DAY", "BRANCH"]).agg(
        ACTIVE_TILLS=("TILL_NO", "nunique")
    ).reset_index()

    fig = px.line(
        active_tills,
        x="BUSINESS_DAY",
        y="ACTIVE_TILLS",
        color="BRANCH",
        title="Active Tills per Day by Branch"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        active_tills
        .sort_values(["BUSINESS_DAY", "BRANCH"])
        .style.format({
            "ACTIVE_TILLS": "{:,.0f}"
        })
    )

def avg_customers_per_till(df: pd.DataFrame):
    st.subheader("Average Customers Served per Till")

    till_customers = df.groupby(["BRANCH", "TILL_NO"]).agg(
        CUSTOMERS=("INVOICE_NO", "nunique")
    ).reset_index()

    till_stats = till_customers.groupby("BRANCH").agg(
        AVG_CUSTOMERS_PER_TILL=("CUSTOMERS", "mean"),
        TILLS=("TILL_NO", "nunique")
    ).reset_index()

    fig = px.bar(
        till_stats.sort_values("AVG_CUSTOMERS_PER_TILL", ascending=False),
        x="BRANCH",
        y="AVG_CUSTOMERS_PER_TILL",
        title="Average Customers per Till by Branch",
        text_auto=".1f"
    )
    fig.update_layout(yaxis_title="Avg Customers per Till")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        till_stats
        .sort_values("AVG_CUSTOMERS_PER_TILL", ascending=False)
        .style.format({
            "AVG_CUSTOMERS_PER_TILL": "{:,.1f}",
            "TILLS": "{:,.0f}"
        })
    )

def store_customer_traffic_storewise(df: pd.DataFrame):
    st.subheader("Store Customer Traffic Storewise")

    period = st.sidebar.selectbox("Aggregate Period", ["Daily", "Weekly", "Monthly"])

    temp = df.copy()
    temp["DATE"] = temp["BUSINESS_DAY"].dt.date

    if period == "Weekly":
        temp["PERIOD"] = temp["BUSINESS_DAY"] - temp["BUSINESS_DAY"].dt.weekday * timedelta(days=1)
    elif period == "Monthly":
        temp["PERIOD"] = temp["BUSINESS_DAY"].values.astype("datetime64[M]")
    else:
        temp["PERIOD"] = temp["BUSINESS_DAY"]

    traffic = temp.groupby(["PERIOD", "BRANCH"]).agg(
        CUSTOMERS=("INVOICE_NO", "nunique")
    ).reset_index()

    fig = px.line(
        traffic,
        x="PERIOD",
        y="CUSTOMERS",
        color="BRANCH",
        title=f"Customer Traffic ({period})"
    )
    st.plotly_chart(fig, use_container_width=True)

def customer_traffic_departmentwise(df: pd.DataFrame):
    st.subheader("Customer Traffic-Departmentwise")

    dept_traffic = df.groupby("DEPARTMENT").agg(
        CUSTOMERS=("INVOICE_NO", "nunique"),
        NET_SALES=("NET_SALES", "sum")
    ).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(
            dept_traffic.sort_values("CUSTOMERS", ascending=False),
            x="DEPARTMENT",
            y="CUSTOMERS",
            title="Customer Traffic by Department",
            text_auto=".2s"
        )
        fig1.update_layout(yaxis_title="Customers")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(
            dept_traffic.sort_values("NET_SALES", ascending=False),
            x="DEPARTMENT",
            y="NET_SALES",
            title="Net Sales by Department",
            text_auto=".2s"
        )
        fig2.update_layout(yaxis_title="Net Sales (KES)")
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        dept_traffic
        .sort_values("NET_SALES", ascending=False)
        .style.format({
            "CUSTOMERS": "{:,.0f}",
            "NET_SALES": "KES {:,.0f}"
        })
    )

def cashiers_performance(df: pd.DataFrame):
    st.subheader("Cashiers Performance")

    cashier_stats = df.groupby("CASHIER_NAME").agg(
        TRANSACTIONS=("INVOICE_NO", "nunique"),
        NET_SALES=("NET_SALES", "sum"),
        ITEMS_SCANNED=("QUANTITY", "sum")
    ).reset_index()

    cashier_stats["AVG_ITEMS_PER_TXN"] = cashier_stats["ITEMS_SCANNED"] / cashier_stats["TRANSACTIONS"]
    cashier_stats["AVG_SALES_PER_TXN"] = cashier_stats["NET_SALES"] / cashier_stats["TRANSACTIONS"]

    st.dataframe(
        cashier_stats
        .sort_values("NET_SALES", ascending=False)
        .style.format({
            "NET_SALES": "KES {:,.0f}",
            "AVG_ITEMS_PER_TXN": "{:,.1f}",
            "AVG_SALES_PER_TXN": "KES {:,.0f}"
        })
    )

    fig = px.bar(
        cashier_stats.sort_values("NET_SALES", ascending=False).head(20),
        x="CASHIER_NAME",
        y="NET_SALES",
        title="Top 20 Cashiers by Net Sales",
        text_auto=".2s"
    )
    fig.update_layout(yaxis_title="Net Sales (KES)")
    st.plotly_chart(fig, use_container_width=True)

def till_usage(df: pd.DataFrame):
    st.subheader("Till Usage")

    till_stats = df.groupby(["BRANCH", "TILL_NO"]).agg(
        TRANSACTIONS=("INVOICE_NO", "nunique"),
        NET_SALES=("NET_SALES", "sum")
    ).reset_index()

    fig = px.scatter(
        till_stats,
        x="TRANSACTIONS",
        y="NET_SALES",
        color="BRANCH",
        hover_data=["TILL_NO"],
        title="Till Usage: Transactions vs Net Sales"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        till_stats
        .sort_values("NET_SALES", ascending=False)
        .style.format({
            "TRANSACTIONS": "{:,.0f}",
            "NET_SALES": "KES {:,.0f}"
        })
    )

def tax_compliance(df: pd.DataFrame):
    st.subheader("Tax Compliance")

    # Here we approximate taxable sales as NET_SALES and VAT_AMT as VAT
    tax_stats = df.groupby("BRANCH").agg(
        NET_SALES=("NET_SALES", "sum"),
        VAT_AMT=("VAT_AMT", "sum")
    ).reset_index()

    tax_stats["VAT_RATE"] = tax_stats.apply(
        lambda row: safe_div(row["VAT_AMT"], row["NET_SALES"]),
        axis=1
    )

    fig = px.bar(
        tax_stats,
        x="BRANCH",
        y="VAT_RATE",
        title="Effective VAT Rate by Branch",
        text_auto=".1%"
    )
    fig.update_layout(yaxis_title="Effective VAT Rate")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        tax_stats
        .sort_values("VAT_RATE", ascending=False)
        .style.format({
            "NET_SALES": "KES {:,.0f}",
            "VAT_AMT": "KES {:,.0f}",
            "VAT_RATE": "{:.1%}"
        })
    )

# -----------------------
# Insights Section Functions
# -----------------------
def customer_baskets_overview(df: pd.DataFrame):
    st.subheader("Customer Baskets Overview")

    basket_stats = df.groupby("INVOICE_NO").agg(
        BASKET_QTY=("BASKET_QTY", "max"),
        BASKET_LINES=("BASKET_LINES", "max"),
        BASKET_NET_SALES=("BASKET_NET_SALES", "max")
    ).reset_index()

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Basket Qty", f"{basket_stats['BASKET_QTY'].mean():.1f}")
    col2.metric("Avg Basket Lines", f"{basket_stats['BASKET_LINES'].mean():.1f}")
    col3.metric("Avg Basket Value", format_currency(basket_stats["BASKET_NET_SALES"].mean()))

    fig = px.histogram(
        basket_stats,
        x="BASKET_NET_SALES",
        nbins=50,
        title="Distribution of Basket Values"
    )
    fig.update_layout(xaxis_title="Basket Net Sales (KES)")
    st.plotly_chart(fig, use_container_width=True)

def global_category_overview_sales(df: pd.DataFrame):
    st.subheader("Global Category Overview - Sales")

    cat_sales = df.groupby("CATEGORY").agg(
        NET_SALES=("NET_SALES", "sum"),
        GROSS_SALES=("GROSS_SALES", "sum"),
        QTY=("QUANTITY", "sum")
    ).reset_index()

    cat_sales["CONTRIBUTION"] = cat_sales["NET_SALES"] / cat_sales["NET_SALES"].sum()

    fig = px.bar(
        cat_sales.sort_values("NET_SALES", ascending=False),
        x="CATEGORY",
        y="NET_SALES",
        title="Net Sales by Category",
        text_auto=".2s"
    )
    fig.update_layout(yaxis_title="Net Sales (KES)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        cat_sales
        .sort_values("NET_SALES", ascending=False)
        .style.format({
            "NET_SALES": "KES {:,.0f}",
            "GROSS_SALES": "KES {:,.0f}",
            "CONTRIBUTION": "{:.1%}"
        })
    )

def global_category_overview_baskets(df: pd.DataFrame):
    st.subheader("Global Category Overview - Baskets")

    cat_baskets = df.groupby("CATEGORY").agg(
        BASKETS=("INVOICE_NO", "nunique"),
        NET_SALES=("NET_SALES", "sum")
    ).reset_index()

    cat_baskets["AVG_SALES_PER_BASKET"] = cat_baskets["NET_SALES"] / cat_baskets["BASKETS"]

    fig = px.bar(
        cat_baskets.sort_values("BASKETS", ascending=False),
        x="CATEGORY",
        y="BASKETS",
        title="Number of Baskets by Category",
        text_auto=".2s"
    )
    fig.update_layout(yaxis_title="Baskets")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        cat_baskets
        .sort_values("NET_SALES", ascending=False)
        .style.format({
            "NET_SALES": "KES {:,.0f}",
            "AVG_SALES_PER_BASKET": "KES {:,.0f}"
        })
    )

def supplier_contribution(df: pd.DataFrame):
    st.subheader("Supplier Contribution")

    supplier_stats = df.groupby("SUPPLIER_NAME").agg(
        NET_SALES=("NET_SALES", "sum"),
        QTY=("QUANTITY", "sum")
    ).reset_index()

    supplier_stats["CONTRIBUTION"] = supplier_stats["NET_SALES"] / supplier_stats["NET_SALES"].sum()

    fig = px.bar(
        supplier_stats.sort_values("NET_SALES", ascending=False).head(50),
        x="SUPPLIER_NAME",
        y="NET_SALES",
        title="Top 50 Suppliers by Net Sales",
        text_auto=".2s"
    )
    fig.update_layout(
        yaxis_title="Net Sales (KES)",
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        supplier_stats
        .sort_values("NET_SALES", ascending=False)
        .style.format({
            "NET_SALES": "KES {:,.0f}",
            "CONTRIBUTION": "{:.1%}"
        })
    )

def category_overview(df: pd.DataFrame):
    st.subheader("Category Overview")

    branch_filter = st.selectbox("Select Branch (or All)", ["All"] + sorted(df["BRANCH"].unique().tolist()))
    temp = df.copy()
    if branch_filter != "All":
        temp = temp[temp["BRANCH"] == branch_filter]

    cat_stats = temp.groupby("CATEGORY").agg(
        NET_SALES=("NET_SALES", "sum"),
        BASKETS=("INVOICE_NO", "nunique"),
        QTY=("QUANTITY", "sum")
    ).reset_index()

    cat_stats["AVG_BASKET_VALUE"] = cat_stats["NET_SALES"] / cat_stats["BASKETS"]

    fig = px.bar(
        cat_stats.sort_values("NET_SALES", ascending=False),
        x="CATEGORY",
        y="NET_SALES",
        title=f"Net Sales by Category ({branch_filter})",
        text_auto=".2s"
    )
    fig.update_layout(yaxis_title="Net Sales (KES)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        cat_stats
        .sort_values("NET_SALES", ascending=False)
        .style.format({
            "NET_SALES": "KES {:,.0f}",
            "AVG_BASKET_VALUE": "KES {:,.0f}"
        })
    )

def branch_comparison(df: pd.DataFrame):
    st.subheader("Branch Comparison")

    metric = st.selectbox("Metric", ["NET_SALES", "CUSTOMERS", "AVG_BASKET_VALUE"])

    branch_stats = df.groupby("BRANCH").agg(
        NET_SALES=("NET_SALES", "sum"),
        CUSTOMERS=("INVOICE_NO", "nunique"),
        AVG_BASKET_VALUE=("BASKET_NET_SALES", "mean")
    ).reset_index()

    fig = px.bar(
        branch_stats.sort_values(metric, ascending=False),
        x="BRANCH",
        y=metric,
        title=f"Branch Comparison by {metric}",
        text_auto=".2s"
    )
    if metric == "NET_SALES":
        fig.update_layout(yaxis_title="Net Sales (KES)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        branch_stats
        .sort_values(metric, ascending=False)
        .style.format({
            "NET_SALES": "KES {:,.0f}",
            "AVG_BASKET_VALUE": "KES {:,.0f}"
        })
    )

def product_performance(df: pd.DataFrame):
    st.subheader("Product Performance (Top/Bottom)")

    top_n = st.slider("Top/Bottom N", min_value=5, max_value=50, value=20, step=5)

    prod_stats = df.groupby(["ITEM_CODE", "ITEM_NAME"]).agg(
        NET_SALES=("NET_SALES", "sum"),
        QTY=("QUANTITY", "sum"),
        BASKETS=("INVOICE_NO", "nunique")
    ).reset_index()

    prod_stats["AVG_PRICE"] = prod_stats["NET_SALES"] / prod_stats["QTY"].replace(0, np.nan)

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"Top {top_n} Products by Net Sales")
        top_products = prod_stats.sort_values("NET_SALES", ascending=False).head(top_n)
        st.dataframe(
            top_products
            .style.format({
                "NET_SALES": "KES {:,.0f}",
                "AVG_PRICE": "KES {:,.0f}"
            })
        )

    with col2:
        st.write(f"Bottom {top_n} Products by Net Sales (Non-zero)")
        bottom_products = prod_stats[prod_stats["NET_SALES"] > 0].sort_values("NET_SALES", ascending=True).head(top_n)
        st.dataframe(
            bottom_products
            .style.format({
                "NET_SALES": "KES {:,.0f}",
                "AVG_PRICE": "KES {:,.0f}"
            })
        )

def global_loyalty_overview(df: pd.DataFrame):
    st.subheader("Global Loyalty Overview")

    df["IS_LOYALTY"] = np.where(df["LOYALTY_CARD_NO"].str.upper() == "UNKNOWN", 0, 1)

    loyalty_stats = df.groupby("IS_LOYALTY").agg(
        NET_SALES=("NET_SALES", "sum"),
        CUSTOMERS=("INVOICE_NO", "nunique")
    ).reset_index()

    loyalty_stats["SEGMENT"] = loyalty_stats["IS_LOYALTY"].map({0: "Non-Loyalty", 1: "Loyalty"})

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            loyalty_stats,
            x="SEGMENT",
            y="NET_SALES",
            title="Net Sales by Loyalty vs Non-Loyalty",
            text_auto=".2s"
        )
        fig.update_layout(yaxis_title="Net Sales (KES)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            loyalty_stats,
            x="SEGMENT",
            y="CUSTOMERS",
            title="Customers by Loyalty vs Non-Loyalty",
            text_auto=".2s"
        )
        fig2.update_layout(yaxis_title="Customers")
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        loyalty_stats[["SEGMENT", "NET_SALES", "CUSTOMERS"]]
        .style.format({
            "NET_SALES": "KES {:,.0f}"
        })
    )

def branch_loyalty_overview(df: pd.DataFrame):
    st.subheader("Branch Loyalty Overview")

    df["IS_LOYALTY"] = np.where(df["LOYALTY_CARD_NO"].str.upper() == "UNKNOWN", 0, 1)

    branch_loyalty = df.groupby(["BRANCH", "IS_LOYALTY"]).agg(
        NET_SALES=("NET_SALES", "sum"),
        CUSTOMERS=("INVOICE_NO", "nunique")
    ).reset_index()

    branch_loyalty["SEGMENT"] = branch_loyalty["IS_LOYALTY"].map({0: "Non-Loyalty", 1: "Loyalty"})

    fig = px.bar(
        branch_loyalty,
        x="BRANCH",
        y="NET_SALES",
        color="SEGMENT",
        title="Net Sales by Branch and Loyalty Segment",
        text_auto=".2s"
    )
    fig.update_layout(yaxis_title="Net Sales (KES)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        branch_loyalty
        .sort_values(["BRANCH", "SEGMENT"])
        .style.format({
            "NET_SALES": "KES {:,.0f}"
        })
    )

def customer_loyalty_overview(df: pd.DataFrame):
    st.subheader("Customer Loyalty Overview")

    df["IS_LOYALTY"] = np.where(df["LOYALTY_CARD_NO"].str.upper() == "UNKNOWN", 0, 1)

    customer_stats = df.groupby("INVOICE_NO").agg(
        NET_SALES=("BASKET_NET_SALES", "max"),
        IS_LOYALTY=("IS_LOYALTY", "max")
    ).reset_index()

    loyal = customer_stats[customer_stats["IS_LOYALTY"] == 1]
    non_loyal = customer_stats[customer_stats["IS_LOYALTY"] == 0]

    col1, col2 = st.columns(2)
    col1.metric("Avg Basket Value (Loyalty)", format_currency(loyal["NET_SALES"].mean()))
    col2.metric("Avg Basket Value (Non-Loyalty)", format_currency(non_loyal["NET_SALES"].mean()))

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=loyal["NET_SALES"], name="Loyalty"))
    fig.add_trace(go.Histogram(x=non_loyal["NET_SALES"], name="Non-Loyalty"))

    fig.update_layout(
        barmode="overlay",
        title="Distribution of Basket Values by Loyalty",
        xaxis_title="Basket Net Sales (KES)"
    )
    fig.update_traces(opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)

def global_pricing_overview(df: pd.DataFrame):
    st.subheader("Global Pricing Overview")

    price_stats = df.groupby("ITEM_CODE").agg(
        ITEM_NAME=("ITEM_NAME", "first"),
        AVG_PRICE=("UNIT_PRICE", "mean"),
        AVG_COST=("UNIT_COST", "mean"),
        QTY=("QUANTITY", "sum"),
        NET_SALES=("NET_SALES", "sum")
    ).reset_index()

    price_stats["UNIT_MARGIN"] = price_stats["AVG_PRICE"] - price_stats["AVG_COST"]
    price_stats["MARGIN_RATE"] = price_stats["UNIT_MARGIN"] / price_stats["AVG_PRICE"].replace(0, np.nan)

    st.dataframe(
        price_stats
        .sort_values("NET_SALES", ascending=False)
        .style.format({
            "AVG_PRICE": "KES {:,.0f}",
            "AVG_COST": "KES {:,.0f}",
            "UNIT_MARGIN": "KES {:,.0f}",
            "MARGIN_RATE": "{:.1%}",
            "NET_SALES": "KES {:,.0f}"
        })
    )

    fig = px.scatter(
        price_stats,
        x="AVG_PRICE",
        y="MARGIN_RATE",
        size="NET_SALES",
        hover_data=["ITEM_NAME"],
        title="Price vs Margin Rate (Bubble size = Net Sales)"
    )
    fig.update_layout(xaxis_title="Average Price (KES)", yaxis_title="Margin Rate")
    st.plotly_chart(fig, use_container_width=True)

def branch_pricing_overview(df: pd.DataFrame):
    st.subheader("Branch Pricing Overview")

    branch = st.selectbox("Select Branch", sorted(df["BRANCH"].unique().tolist()))

    temp = df[df["BRANCH"] == branch]

    price_stats = temp.groupby("ITEM_CODE").agg(
        ITEM_NAME=("ITEM_NAME", "first"),
        AVG_PRICE=("UNIT_PRICE", "mean"),
        AVG_COST=("UNIT_COST", "mean"),
        QTY=("QUANTITY", "sum"),
        NET_SALES=("NET_SALES", "sum")
    ).reset_index()

    price_stats["UNIT_MARGIN"] = price_stats["AVG_PRICE"] - price_stats["AVG_COST"]
    price_stats["MARGIN_RATE"] = price_stats["UNIT_MARGIN"] / price_stats["AVG_PRICE"].replace(0, np.nan)

    st.dataframe(
        price_stats
        .sort_values("NET_SALES", ascending=False)
        .style.format({
            "AVG_PRICE": "KES {:,.0f}",
            "AVG_COST": "KES {:,.0f}",
            "UNIT_MARGIN": "KES {:,.0f}",
            "MARGIN_RATE": "{:.1%}",
            "NET_SALES": "KES {:,.0f}"
        })
    )

    fig = px.scatter(
        price_stats,
        x="AVG_PRICE",
        y="MARGIN_RATE",
        size="NET_SALES",
        hover_data=["ITEM_NAME"],
        title=f"Price vs Margin Rate - {branch}"
    )
    fig.update_layout(xaxis_title="Average Price (KES)", yaxis_title="Margin Rate")
    st.plotly_chart(fig, use_container_width=True)

def global_refunds_overview(df: pd.DataFrame):
    st.subheader("Global Refunds Overview")

    refund_stats = df.groupby("BRANCH").agg(
        REFUND_QTY=("REFUND_QTY", "sum"),
        REFUND_AMT=("REFUND_AMT", "sum"),
        NET_SALES=("NET_SALES", "sum")
    ).reset_index()

    refund_stats["REFUND_RATE_SALES"] = refund_stats["REFUND_AMT"] / refund_stats["NET_SALES"].replace(0, np.nan)
    refund_stats["REFUND_RATE_QTY"] = refund_stats["REFUND_QTY"] / (df.groupby("BRANCH")["QUANTITY"].sum().replace(0, np.nan))

    fig = px.bar(
        refund_stats.sort_values("REFUND_AMT", ascending=False),
        x="BRANCH",
        y="REFUND_AMT",
        title="Refund Amount by Branch",
        text_auto=".2s"
    )
    fig.update_layout(yaxis_title="Refund Amount (KES)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        refund_stats
        .sort_values("REFUND_AMT", ascending=False)
        .style.format({
            "REFUND_AMT": "KES {:,.0f}",
            "REFUND_RATE_SALES": "{:.2%}"
        })
    )

def branch_refunds_overview(df: pd.DataFrame):
    st.subheader("Branch Refunds Overview")

    branch = st.selectbox("Select Branch", sorted(df["BRANCH"].unique().tolist()))
    temp = df[df["BRANCH"] == branch]

    refund_stats = temp.groupby("ITEM_CODE").agg(
        ITEM_NAME=("ITEM_NAME", "first"),
        REFUND_QTY=("REFUND_QTY", "sum"),
        REFUND_AMT=("REFUND_AMT", "sum"),
        NET_SALES=("NET_SALES", "sum")
    ).reset_index()

    refund_stats["REFUND_RATE_SALES"] = refund_stats["REFUND_AMT"] / refund_stats["NET_SALES"].replace(0, np.nan)

    st.dataframe(
        refund_stats
        .sort_values("REFUND_AMT", ascending=False)
        .style.format({
            "REFUND_AMT": "KES {:,.0f}",
            "REFUND_RATE_SALES": "{:.2%}",
            "NET_SALES": "KES {:,.0f}"
        })
    )

    fig = px.bar(
        refund_stats.sort_values("REFUND_AMT", ascending=False).head(30),
        x="ITEM_NAME",
        y="REFUND_AMT",
        title=f"Top Refund Items - {branch}",
        text_auto=".2s"
    )
    fig.update_layout(
        yaxis_title="Refund Amount (KES)",
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Main App
# -----------------------
def main():
    st.title("DailyDeck: The Story Behind the Numbers")

    raw_df = smart_load()
    if raw_df is None:
        st.info("👈 Please upload a DAILY_POS_TRN_ITEMS CSV in the sidebar to see the dashboards.")
        return

    with st.spinner("Preparing data (cached) ..."):
        df = clean_and_derive(raw_df)

    section = st.sidebar.selectbox(
        "Section",
        ["SALES", "OPERATIONS", "INSIGHTS"]
    )

    if section == "SALES":
        sales_items = [
            "Global sales Overview",
            "Global Net Sales Distribution by Sales Channel",
            "Global Net Sales Distribution by SHIFT",
            "Night vs Day Shift Sales Ratio — Stores with Night Shifts",
            "Global Day vs Night Sales — Only Stores with NIGHT Shift",
            "2nd-Highest Channel Share",
            "Bottom 30 — 2nd Highest Channel",
            "Stores Sales Summary"
        ]
        choice = st.sidebar.selectbox(
            "Sales Subsection",
            sales_items
        )
        if choice == sales_items[0]:
            sales_global_overview(df)
        elif choice == sales_items[1]:
            sales_by_channel_l2(df)
        elif choice == sales_items[2]:
            sales_by_shift(df)
        elif choice == sales_items[3]:
            night_vs_day_ratio(df)
        elif choice == sales_items[4]:
            global_day_vs_night(df)
        elif choice == sales_items[5]:
            second_highest_channel_share(df)
        elif choice == sales_items[6]:
            bottom_30_2nd_highest(df)
        elif choice == sales_items[7]:
            stores_sales_summary(df)

    elif section == "OPERATIONS":
        ops_items = [
            "Customer Traffic-Storewise",
            "Active Tills During the day",
            "Average Customers Served per Till",
            "Store Customer Traffic Storewise",
            "Customer Traffic-Departmentwise",
            "Cashiers Perfomance",
            "Till Usage",
            "Tax Compliance"
        ]
        choice = st.sidebar.selectbox(
            "Operations Subsection",
            ops_items
        )
        if choice == ops_items[0]:
            customer_traffic_storewise(df)
        elif choice == ops_items[1]:
            active_tills_during_day(df)
        elif choice == ops_items[2]:
            avg_customers_per_till(df)
        elif choice == ops_items[3]:
            store_customer_traffic_storewise(df)
        elif choice == ops_items[4]:
            customer_traffic_departmentwise(df)
        elif choice == ops_items[5]:
            cashiers_performance(df)
        elif choice == ops_items[6]:
            till_usage(df)
        elif choice == ops_items[7]:
            tax_compliance(df)

    elif section == "INSIGHTS":
        ins_items = [
            "Customer Baskets Overview",
            "Global Category Overview-Sales",
            "Global Category Overview-Baskets",
            "Supplier Contribution",
            "Category Overview",
            "Branch Comparison",
            "Product Perfomance",
            "Global Loyalty Overview",
            "Branch Loyalty Overview",
            "Customer Loyalty Overview",
            "Global Pricing Overview",
            "Branch Pricing Overview",
            "Global Refunds Overview",
            "Branch Refunds Overview"
        ]
        choice = st.sidebar.selectbox(
            "Insights Subsection",
            ins_items
        )
        mapping = {
            ins_items[0]: customer_baskets_overview,
            ins_items[1]: global_category_overview_sales,
            ins_items[2]: global_category_overview_baskets,
            ins_items[3]: supplier_contribution,
            ins_items[4]: category_overview,
            ins_items[5]: branch_comparison,
            ins_items[6]: product_performance,
            ins_items[7]: global_loyalty_overview,
            ins_items[8]: branch_loyalty_overview,
            ins_items[9]: customer_loyalty_overview,
            ins_items[10]: global_pricing_overview,
            ins_items[11]: branch_pricing_overview,
            ins_items[12]: global_refunds_overview,
            ins_items[13]: branch_refunds_overview
        }
        func = mapping.get(choice)
        if func:
            func(df)
        else:
            st.write("Not implemented yet")

if __name__ == "__main__":
    main()
