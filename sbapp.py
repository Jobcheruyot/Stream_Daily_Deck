import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from supabase import create_client, Client

# ============================================================================
# Streamlit Page Config
# ============================================================================
st.set_page_config(
    layout="wide",
    page_title="Superdeck (Supabase)",
)

# ============================================================================
# Supabase Configuration
# ============================================================================
TABLE_NAME = "daily_pos_trn_items_clean"


@st.cache_resource
def init_supabase() -> Client:
    """
    Initialize Supabase client using secrets.toml.

    Required in .streamlit/secrets.toml:
        SUPABASE_URL = "https://eifqphcrwzddrmvdmtek.supabase.co"
        SUPABASE_KEY = "<your anon public key>"
    """
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception:
        st.error(
            "Supabase credentials not found in secrets.\n\n"
            "Add SUPABASE_URL and SUPABASE_KEY to .streamlit/secrets.toml "
            "or to the Streamlit Cloud Secrets panel."
        )
        st.stop()

    return create_client(url, key)


# ============================================================================
# Data Loading from Supabase
# ============================================================================

def load_supabase_data(
    date_basis: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Load public.daily_pos_trn_items_clean from Supabase using TRN_DATE or ZED_DATE
    as the basis for filtering.

    We:
    - Map the UI date_basis (TRN_DATE / ZED_DATE) to the actual Supabase column name
      which is stored in lowercase (trn_date / zed_date).
    - Standardise all DataFrame column names to UPPERCASE so the rest of the app
      can continue to use the original naming convention.
    """
    client = init_supabase()

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    if date_basis not in ("TRN_DATE", "ZED_DATE"):
        raise ValueError("date_basis must be 'TRN_DATE' or 'ZED_DATE'.")

    # Map to actual Supabase column name (lowercase)
    supa_date_col = date_basis.lower()

    # Supabase query
    response = (
        client.table(TABLE_NAME)
        .select("*")
        .gte(supa_date_col, start_str)
        .lte(supa_date_col, end_str)
        .execute()
    )

    data = response.data or []
    df = pd.DataFrame(data)

    # Normalise column names to uppercase so downstream logic continues to work
    if not df.empty:
        df.columns = [c.upper() for c in df.columns]

    return df


# ============================================================================
# Sidebar and Global Config
# ============================================================================


def sidebar_config():
    st.sidebar.markdown("### Filters")

    # Date basis selector
    date_basis = st.sidebar.radio(
        "Use which date for filtering?",
        ["TRN_DATE", "ZED_DATE"],
        index=0,
        help="Choose which date column to use for filtering.",
    )

    # Date range
    today = datetime.now().date()
    default_start = today - timedelta(days=7)

    c1, c2 = st.sidebar.columns(2)
    with c1:
        start_date = st.sidebar.date_input(
            "Start Date",
            value=default_start,
            max_value=today,
        )
    with c2:
        end_date = st.sidebar.date_input(
            "End Date",
            value=today,
            max_value=today,
        )

    if start_date > end_date:
        st.sidebar.error("Start Date cannot be after End Date.")
        st.stop()

    # Main section
    section = st.sidebar.selectbox(
        "Main Section",
        ["SALES", "OPERATIONS", "INSIGHTS"],
    )

    # Subsections handled later in main()
    return date_basis, start_date, end_date, section, None


# ============================================================================
# Data Cleaning & Derived Columns
# ============================================================================


def clean_and_derive(df: pd.DataFrame, date_basis: str) -> pd.DataFrame:
    """
    Clean the raw Supabase dataframe and create derived columns needed
    across all sections.
    """
    d = df.copy()

    # --- Ensure we have at least some required columns
    if "STORE_CODE" not in d.columns or "ITEM_CODE" not in d.columns:
        st.error("Required columns missing from Supabase data.")
        st.stop()

    # --- Standard string columns
    str_cols = [
        "STORE_CODE", "TILL", "SESSION", "RCT", "STORE_NAME", "CASHIER",
        "ITEM_CODE", "ITEM_NAME", "DEPARTMENT", "CATEGORY", "CU_DEVICE_SERIAL",
        "CAP_CUSTOMER_CODE", "LOYALTY_CUSTOMER_CODE", "SUPPLIER_NAME",
        "SALES_CHANNEL_L1", "SALES_CHANNEL_L2", "SHIFT",
    ]
    for c in str_cols:
        if c in d.columns:
            d[c] = d[c].astype(str).fillna("").str.strip()

    # --- Parse dates (TRN_DATE & ZED_DATE if present)
    for col in ["TRN_DATE", "ZED_DATE"]:
        if col in d.columns:
            d[col] = pd.to_datetime(d[col], errors="coerce")

    # --- Use selected date_basis as primary
    if date_basis not in d.columns:
        st.error(f"Selected date_basis '{date_basis}' not found in data columns.")
        st.stop()

    d = d.dropna(subset=[date_basis]).copy()
    d["DATE"] = d[date_basis].dt.date
    d["HOUR"] = d[date_basis].dt.hour
    d["HALF_HOUR"] = d[date_basis].dt.floor("30min")
    d["WEEKDAY"] = d[date_basis].dt.day_name()
    d["WEEK_NUMBER"] = d[date_basis].dt.isocalendar().week

    # --- Numeric columns
    numeric_cols = [
        "QTY",
        "CP_PRE_VAT",
        "SP_PRE_VAT",
        "COST_PRE_VAT",
        "NET_SALES",
        "VAT_AMT",
    ]
    for c in numeric_cols:
        if c in d.columns:
            d[c] = (
                pd.to_numeric(d[c], errors="coerce")
                .fillna(0)
            )

    # --- Derived metrics
    if "NET_SALES" in d.columns and "COST_PRE_VAT" in d.columns:
        d["GROSS_MARGIN"] = d["NET_SALES"] - d["COST_PRE_VAT"]
        with np.errstate(divide="ignore", invalid="ignore"):
            d["MARGIN_PCT"] = np.where(
                d["NET_SALES"] != 0,
                d["GROSS_MARGIN"] / d["NET_SALES"] * 100,
                0,
            )

    # --- Unique receipt
    required = ["STORE_CODE", "TILL", "SESSION", "RCT"]
    if all(c in d.columns for c in required):
        d["CUST_CODE"] = (
            d["STORE_CODE"] + "-" + d["TILL"] + "-" + d["SESSION"] + "-" + d["RCT"]
        )

    # --- Cashier per store
    if "STORE_NAME" in d.columns and "CASHIER" in d.columns:
        d["CASHIER-COUNT"] = d["STORE_NAME"] + "-" + d["CASHIER"]

    # --- Channel placeholder
    if "SALES_CHANNEL_L1" not in d.columns:
        d["SALES_CHANNEL_L1"] = "UNKNOWN"
    if "SALES_CHANNEL_L2" not in d.columns:
        d["SALES_CHANNEL_L2"] = "UNKNOWN"

    return d


# ============================================================================
# SALES SECTION FUNCTIONS
# ============================================================================


def sales_global_overview(df: pd.DataFrame):
    st.subheader("Global Sales Overview")

    if "NET_SALES" not in df.columns:
        st.warning("NET_SALES column missing.")
        return

    total_sales = df["NET_SALES"].sum()
    total_qty = df["QTY"].sum() if "QTY" in df.columns else np.nan

    c1, c2 = st.columns(2)
    c1.metric("Total Net Sales", f"{total_sales:,.0f}")
    c2.metric("Total Quantity Sold", f"{total_qty:,.0f}")

    # Daily trend
    if "DATE" in df.columns:
        daily = df.groupby("DATE", as_index=False)["NET_SALES"].sum()
        fig = px.line(daily, x="DATE", y="NET_SALES", title="Daily Net Sales Trend")
        st.plotly_chart(fig, use_container_width=True)


def sales_by_channel_l2(df: pd.DataFrame):
    st.subheader("Global Net Sales Distribution by Sales Channel (L2)")

    if "SALES_CHANNEL_L2" not in df.columns or "NET_SALES" not in df.columns:
        st.warning("Required columns missing for channel analysis.")
        return

    ch = (
        df.groupby("SALES_CHANNEL_L2", as_index=False)["NET_SALES"]
        .sum()
        .sort_values("NET_SALES", ascending=False)
    )
    fig = px.pie(ch, names="SALES_CHANNEL_L2", values="NET_SALES")
    st.plotly_chart(fig, use_container_width=True)


def sales_by_shift(df: pd.DataFrame):
    st.subheader("Global Net Sales Distribution by SHIFT")

    if "SHIFT" not in df.columns or "NET_SALES" not in df.columns:
        st.warning("SHIFT or NET_SALES column missing.")
        return

    sh = (
        df.groupby("SHIFT", as_index=False)["NET_SALES"]
        .sum()
        .sort_values("NET_SALES", ascending=False)
    )
    fig = px.bar(sh, x="SHIFT", y="NET_SALES")
    st.plotly_chart(fig, use_container_width=True)


def night_vs_day_ratio(df: pd.DataFrame):
    st.subheader("Night vs Day Shift Sales Ratio — Stores with Night Shifts")

    if "SHIFT" not in df.columns or "NET_SALES" not in df.columns:
        st.warning("SHIFT or NET_SALES column missing.")
        return

    # Only stores that have at least some night shift
    store_shift = df.groupby(["STORE_NAME", "SHIFT"], as_index=False)["NET_SALES"].sum()
    stores_with_night = store_shift[store_shift["SHIFT"].str.upper().str.contains("NIGHT")]["STORE_NAME"].unique()

    subset = df[df["STORE_NAME"].isin(stores_with_night)].copy()
    subset["SHIFT_GROUP"] = np.where(
        subset["SHIFT"].str.upper().str.contains("NIGHT"), "NIGHT", "DAY"
    )

    g = subset.groupby(["STORE_NAME", "SHIFT_GROUP"], as_index=False)["NET_SALES"].sum()
    pivot = g.pivot(index="STORE_NAME", columns="SHIFT_GROUP", values="NET_SALES").fillna(0)
    pivot["NIGHT/DAY_RATIO"] = np.where(
        pivot["DAY"] != 0,
        pivot["NIGHT"] / pivot["DAY"],
        np.nan,
    )
    pivot = pivot.sort_values("NIGHT/DAY_RATIO", ascending=False)

    st.dataframe(pivot)


def stores_sales_summary(df: pd.DataFrame):
    st.subheader("Stores Sales Summary")

    if "STORE_NAME" not in df.columns or "NET_SALES" not in df.columns:
        st.warning("Required columns missing for store summary.")
        return

    summary = (
        df.groupby("STORE_NAME", as_index=False)
        .agg({"NET_SALES": "sum", "QTY": "sum"})
        .sort_values("NET_SALES", ascending=False)
    )

    st.dataframe(summary)


# ============================================================================
# OPERATIONS SECTION
# ============================================================================


def customer_traffic_storewise(df: pd.DataFrame):
    st.subheader("Customer Traffic-Storewise (30-min Heatmap)")

    if "STORE_NAME" not in df.columns or "HALF_HOUR" not in df.columns or "CUST_CODE" not in df.columns:
        st.warning("Required columns missing for traffic analysis.")
        return

    # unique receipts per half-hour per store
    traffic = (
        df.groupby(["STORE_NAME", "HALF_HOUR"])["CUST_CODE"]
        .nunique()
        .reset_index(name="NUM_CUSTOMERS")
    )

    store = st.selectbox("Select Store", sorted(df["STORE_NAME"].unique()))
    subset = traffic[traffic["STORE_NAME"] == store].copy()

    if subset.empty:
        st.info("No data for selected store.")
        return

    subset["DATE"] = subset["HALF_HOUR"].dt.date
    subset["TIME"] = subset["HALF_HOUR"].dt.time

    pivot = subset.pivot(index="TIME", columns="DATE", values="NUM_CUSTOMERS").fillna(0)
    fig = px.imshow(
        pivot,
        labels=dict(x="Date", y="Time (30-min)", color="Customers"),
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)


def cashiers_performance(df: pd.DataFrame):
    st.subheader("Cashiers Performance")

    required = ["CASHIER-COUNT", "NET_SALES", "CUST_CODE"]
    if not all(c in df.columns for c in required):
        st.warning("Required columns missing for cashier performance.")
        return

    perf = (
        df.groupby("CASHIER-COUNT", as_index=False)
        .agg(
            NET_SALES=("NET_SALES", "sum"),
            NUM_RECEIPTS=("CUST_CODE", "nunique"),
        )
    )
    perf["AVG_SALE_PER_RECEIPT"] = np.where(
        perf["NUM_RECEIPTS"] != 0,
        perf["NET_SALES"] / perf["NUM_RECEIPTS"],
        0,
    )

    st.dataframe(perf.sort_values("NET_SALES", ascending=False))

    fig = px.bar(
        perf.sort_values("NET_SALES", ascending=False).head(30),
        x="CASHIER-COUNT",
        y="NET_SALES",
        title="Top 30 Cashiers by Net Sales",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# INSIGHTS SECTION (placeholder)
# ============================================================================


def customer_baskets_overview(df: pd.DataFrame):
    st.subheader("Customer Baskets Overview (Coming Soon)")
    st.info("Basket-level insights will be added here in a later version.")


# ============================================================================
# MAIN APP LAYOUT
# ============================================================================


def main():
    # Header
    st.markdown(
        """
        <div style='width:100%;padding:2rem 1.5rem 1rem;background:#fff;
        border-bottom:1px solid #eef1f6;text-align:center;position:relative;'>
            <div>
                <h1 style='margin:.2rem 0 0 0;font-size:2rem;color:#111827;'>
                    Quick Mart Limited
                </h1>
                <p style='margin:0 .2rem 0 0;font-size:1rem;color:#d72638;font-weight:600;'>
                    Let the data Speak
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    date_basis, start_date, end_date, section, _ = sidebar_config()

    # Load data
    with st.spinner("Loading data from Supabase..."):
        raw_df = load_supabase_data(date_basis, start_date, end_date)

    if raw_df is None or raw_df.empty:
        st.warning("No data returned from Supabase for the selected period.")
        return

    df = clean_and_derive(raw_df, date_basis)

    st.caption(
        f"Data source: Supabase • Table: {TABLE_NAME} • "
        f"Date Basis: {date_basis} • Period: {start_date} to {end_date}"
    )

    # Route to selected report
    if section == "SALES":
        sub = st.sidebar.selectbox(
            "Sales Report",
            [
                "Global Sales Overview",
                "Global Net Sales Distribution by Sales Channel (L2)",
                "Global Net Sales Distribution by SHIFT",
                "Night vs Day Shift Sales Ratio — Stores with Night Shifts",
                "Stores Sales Summary",
            ],
        )

        if sub == "Global Sales Overview":
            sales_global_overview(df)
        elif sub == "Global Net Sales Distribution by Sales Channel (L2)":
            sales_by_channel_l2(df)
        elif sub == "Global Net Sales Distribution by SHIFT":
            sales_by_shift(df)
        elif sub == "Night vs Day Shift Sales Ratio — Stores with Night Shifts":
            night_vs_day_ratio(df)
        elif sub == "Stores Sales Summary":
            stores_sales_summary(df)

    elif section == "OPERATIONS":
        sub = st.sidebar.selectbox(
            "Operations Report",
            [
                "Customer Traffic-Storewise (30-min Heatmap)",
                "Cashiers Performance",
            ],
        )
        if sub == "Customer Traffic-Storewise (30-min Heatmap)":
            customer_traffic_storewise(df)
        elif sub == "Cashiers Performance":
            cashiers_performance(df)

    elif section == "INSIGHTS":
        sub = st.sidebar.selectbox(
            "Insights Report",
            [
                "Customer Baskets Overview (Coming Soon)",
            ],
        )
        if sub == "Customer Baskets Overview (Coming Soon)":
            customer_baskets_overview(df)


if __name__ == "__main__":
    main()
