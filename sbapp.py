import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
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


@st.cache_resource
def init_supabase() -> Client:
    """
    Initialize Supabase client using Streamlit secrets.

    IMPORTANT (on Streamlit Cloud):
    - App → Settings → Advanced settings → Secrets
      Add:

        SUPABASE_URL = "https://eifqphcrwzddrmvdmtek.supabase.co"
        SUPABASE_KEY = "<YOUR_ANON_PUBLIC_KEY>"

    Locally you can also use .streamlit/secrets.toml with the same keys.
    """
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception:
        st.error(
            "Supabase credentials not found.\n\n"
            "Add SUPABASE_URL and SUPABASE_KEY in Streamlit secrets "
            "or in .streamlit/secrets.toml."
        )
        st.stop()

    try:
        client = create_client(url, key)
    except Exception as e:
        st.error(f"Failed to create Supabase client: {e}")
        st.stop()

    return client


# ============================================================================
# Data Loading from Supabase
# ============================================================================


@st.cache_data(ttl=3600)
def load_supabase_data(
    date_basis: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Load data from Supabase table public.daily_pos_trn_items_clean,
    filtered by TRN_DATE or ZED_DATE.

    Parameters
    ----------
    date_basis : "TRN_DATE" or "ZED_DATE"
    start_date, end_date : datetime.date
    """
    client = init_supabase()

    if date_basis not in {"TRN_DATE", "ZED_DATE"}:
        raise ValueError("date_basis must be 'TRN_DATE' or 'ZED_DATE'")

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    try:
        response = (
            client.table("daily_pos_trn_items_clean")
            .select("*")
            .gte(date_basis, start_str)
            .lte(date_basis, end_str)
            .execute()
        )
    except Exception as e:
        st.error(f"Error querying Supabase: {e}")
        return pd.DataFrame()

    data = response.data or []
    df = pd.DataFrame(data)

    if df.empty:
        return df

    # Normalize column names to UPPER CASE so the rest of the app is stable
    df.columns = [c.upper() for c in df.columns]

    # Convert date columns to datetime if present
    for col in ["TRN_DATE", "ZED_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


# ============================================================================
# Utility helpers
# ============================================================================


def pick_sales_column(df: pd.DataFrame):
    """
    Try to guess the main 'sales value' column.
    Adjust priorities as needed.
    """
    for col in [
        "NET_SALES",
        "SALES_PRE_VAT",
        "SALES_VALUE",
        "GROSS_SALES",
        "TOTAL_SALES",
        "NET_SALES_PRE_VAT",
    ]:
        if col in df.columns:
            return col
    return None


def safe_group_sum(df: pd.DataFrame, group_cols, value_col: str) -> pd.DataFrame:
    """
    Group by and sum, but handle missing columns and empty frames gracefully.
    """
    if df.empty:
        return df
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols or value_col not in df.columns:
        return pd.DataFrame()
    g = df.groupby(group_cols, dropna=False)[value_col].sum().reset_index()
    return g


def ensure_column(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        st.info(
            f"Column `{col}` not found in dataset. "
            f"This report will be skipped."
        )
        return False
    return True


# ============================================================================
# Period Trend Summary (for the selected date range)
# ============================================================================


def render_period_trend_summary(
    df: pd.DataFrame,
    date_basis: str,
    start_date: date,
    end_date: date,
) -> None:
    """
    High-level trend summary for the selected period:
    - Total & average sales per day
    - Change between first and last day
    - Best and worst day
    - Top channels (if available)
    """
    st.markdown("### Period Trend Summary")

    value_col = pick_sales_column(df)
    if not value_col:
        st.info(
            "Cannot compute trend summary because no sales value column "
            "was found (e.g. NET_SALES)."
        )
        return

    date_col = date_basis if date_basis in df.columns else None
    if not date_col:
        # Fallback: try TRN_DATE
        if "TRN_DATE" in df.columns:
            date_col = "TRN_DATE"
        elif "ZED_DATE" in df.columns:
            date_col = "ZED_DATE"

    if not date_col:
        st.info("No date column found for trend analysis.")
        return

    daily = (
        df.groupby(date_col)[value_col]
        .sum()
        .reset_index()
        .sort_values(date_col)
    )

    if daily.empty:
        st.info("No data available in the selected period.")
        return

    # Basic stats
    num_days = daily[date_col].nunique()
    total_sales = float(daily[value_col].sum())
    avg_per_day = float(daily[value_col].mean())

    first_row = daily.iloc[0]
    last_row = daily.iloc[-1]
    first_val = float(first_row[value_col])
    last_val = float(last_row[value_col])

    change = last_val - first_val
    pct_change = (change / first_val * 100) if first_val > 0 else None

    best_row = daily.loc[daily[value_col].idxmax()]
    worst_row = daily.loc[daily[value_col].idxmin()]

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Sales (Period)", f"{total_sales:,.0f}")
    with c2:
        st.metric("Average per Day", f"{avg_per_day:,.0f}")
    with c3:
        st.metric("Number of Days", num_days)
    with c4:
        if pct_change is not None:
            st.metric(
                "Last vs First Day",
                f"{change:,.0f}",
                f"{pct_change:+.1f}%",
            )
        else:
            st.metric("Last vs First Day", f"{change:,.0f}")

    # Small text narrative
    period_str = f"{start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')}"
    first_date_str = first_row[date_col].strftime("%d-%b-%Y")
    last_date_str = last_row[date_col].strftime("%d-%b-%Y")
    best_date_str = best_row[date_col].strftime("%d-%b-%Y")
    worst_date_str = worst_row[date_col].strftime("%d-%b-%Y")

    trend_word = "up" if (pct_change or 0) > 3 else "down" if (pct_change or 0) < -3 else "flat"

    st.markdown(
        f"""
- **Period covered:** `{period_str}`  
- **Trend:** Overall sales trended **{trend_word}** from {first_date_str} to {last_date_str}.  
- **Best day:** {best_date_str} with ≈ **{best_row[value_col]:,.0f}** in sales.  
- **Softest day:** {worst_date_str} with ≈ **{worst_row[value_col]:,.0f}** in sales.
"""
    )

    # Top channels summary (if available)
    channel_col = None
    if "SALES_CHANNEL_L2" in df.columns:
        channel_col = "SALES_CHANNEL_L2"
    elif "SALES_CHANNEL_L1" in df.columns:
        channel_col = "SALES_CHANNEL_L1"

    if channel_col:
        g_ch = safe_group_sum(df, [channel_col], value_col)
        if not g_ch.empty:
            g_ch = g_ch.sort_values(value_col, ascending=False)
            top_n = g_ch.head(3)
            parts = [
                f"{row[channel_col]} ({row[value_col]/total_sales*100:,.1f}%)"
                for _, row in top_n.iterrows()
            ]
            st.markdown(
                "- **Top channels in this period:** " + ", ".join(parts)
            )


# ============================================================================
# SALES REPORTS
# ============================================================================


def sales_global_overview(df: pd.DataFrame) -> None:
    st.subheader("Global Sales Overview")

    value_col = pick_sales_column(df)
    if not value_col:
        st.warning(
            "Could not find a sales value column "
            "(e.g., NET_SALES, SALES_PRE_VAT)."
        )
        st.dataframe(df.head())
        return

    date_col = "TRN_DATE" if "TRN_DATE" in df.columns else None
    if not date_col:
        st.warning("TRN_DATE column not found. Showing overall totals only.")
        total_sales = df[value_col].sum()
        st.metric("Total Sales", f"{total_sales:,.0f}")
        return

    daily = (
        df.groupby(date_col)[value_col]
        .sum()
        .reset_index()
        .sort_values(date_col)
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.line(
            daily,
            x=date_col,
            y=value_col,
            title="Daily Net Sales",
            markers=True,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        total_sales = daily[value_col].sum()
        avg_per_day = daily[value_col].mean()
        st.metric("Total Sales", f"{total_sales:,.0f}")
        st.metric("Average per Day", f"{avg_per_day:,.0f}")


def sales_channel_distribution(df: pd.DataFrame) -> None:
    st.subheader("Global Net Sales Distribution by Sales Channel (L2)")

    value_col = pick_sales_column(df)
    if not value_col:
        st.warning("Sales value column not found.")
        return

    if "SALES_CHANNEL_L2" not in df.columns:
        st.info(
            "Column `SALES_CHANNEL_L2` not found. "
            "Using `SALES_CHANNEL_L1` if available."
        )
        channel_col = "SALES_CHANNEL_L1" if "SALES_CHANNEL_L1" in df.columns else None
    else:
        channel_col = "SALES_CHANNEL_L2"

    if not channel_col:
        st.warning("No channel column found.")
        return

    g = safe_group_sum(df, [channel_col], value_col)
    if g.empty:
        st.info("No data available for channel distribution.")
        return

    fig = px.pie(
        g,
        values=value_col,
        names=channel_col,
        title="Net Sales Share by Channel",
        hole=0.4,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def sales_shift_distribution(df: pd.DataFrame) -> None:
    st.subheader("Global Net Sales Distribution by SHIFT")

    value_col = pick_sales_column(df)
    if not value_col:
        st.warning("Sales value column not found.")
        return

    if not ensure_column(df, "SHIFT"):
        return

    g = safe_group_sum(df, ["SHIFT"], value_col)
    if g.empty:
        st.info("No data available for shift distribution.")
        return

    fig = px.bar(
        g,
        x="SHIFT",
        y=value_col,
        title="Sales by Shift",
        text_auto=True,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def sales_night_vs_day(df: pd.DataFrame) -> None:
    st.subheader("Night vs Day Shift Sales Ratio — Stores with Night Shifts")

    value_col = pick_sales_column(df)
    if not value_col:
        st.warning("Sales value column not found.")
        return

    needed = {"STORE_NAME", "SHIFT"}
    if not needed.issubset(df.columns):
        st.info("Columns STORE_NAME and SHIFT are required for this view.")
        return

    # Keep only stores that have NIGHT shift data
    has_night = (
        df[df["SHIFT"].astype(str).str.upper() == "NIGHT"]["STORE_NAME"]
        .dropna()
        .unique()
    )
    sub = df[df["STORE_NAME"].isin(has_night)].copy()
    if sub.empty:
        st.info("No stores with NIGHT shift found in the selected period.")
        return

    sub["DAY_NIGHT"] = np.where(
        sub["SHIFT"].astype(str).str.upper() == "NIGHT", "NIGHT", "DAY"
    )
    g = safe_group_sum(sub, ["STORE_NAME", "DAY_NIGHT"], value_col)
    if g.empty:
        st.info("No data available for this view.")
        return

    fig = px.bar(
        g,
        x="STORE_NAME",
        y=value_col,
        color="DAY_NIGHT",
        barmode="group",
        title="Day vs Night Sales (Stores with Night Shift)",
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=10, r=10, t=40, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# OPERATIONS
# ============================================================================


def operations_customer_traffic(df: pd.DataFrame) -> None:
    st.subheader("Customer Traffic — Storewise (Approximate)")

    if not ensure_column(df, "CUST_CODE"):
        st.info(
            "CUST_CODE column missing.\n\n"
            "If you want basket-level traffic, create CUST_CODE as "
            "STORE_CODE-TILL-SESSION-RCT in your pipeline."
        )
        return

    if "STORE_NAME" not in df.columns:
        st.info("STORE_NAME column missing; cannot show storewise traffic.")
        return

    g = (
        df.groupby("STORE_NAME")["CUST_CODE"]
        .nunique()
        .reset_index()
        .rename(columns={"CUST_CODE": "Num_Customers"})
        .sort_values("Num_Customers", ascending=False)
    )

    fig = px.bar(
        g,
        x="STORE_NAME",
        y="Num_Customers",
        title="Approximate Customer Count by Store",
        text_auto=True,
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=10, r=10, t=40, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)


# (You can later add more OPERATIONS views like Tills, Cashiers, etc.
#  using the same df that now has UPPER CASE columns.)


# ============================================================================
# INSIGHTS
# ============================================================================


def insights_category_overview(df: pd.DataFrame) -> None:
    st.subheader("Global Category Overview — Sales")

    value_col = pick_sales_column(df)
    if not value_col:
        st.warning("Sales value column not found.")
        return

    cat_cols = [c for c in ["DEPARTMENT", "CATEGORY"] if c in df.columns]
    if not cat_cols:
        st.info("No DEPARTMENT or CATEGORY column found.")
        return

    g = safe_group_sum(df, cat_cols, value_col)
    if g.empty:
        st.info("No data available for category overview.")
        return

    fig = px.treemap(
        g,
        path=cat_cols,
        values=value_col,
        title="Sales Contribution by Department / Category",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MAIN APP
# ============================================================================


def main():
    # ---------------- Header ----------------
    st.markdown(
        """
        <div style="text-align:center; margin-bottom: 1rem;">
            <h1 style="margin-bottom:0;">Quick Mart Limited</h1>
            <p style="margin-top:0; color:#d72638; font-weight:600;">
                Let the data Speak
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- Sidebar filters ----------------
    today = datetime.today().date()
    default_start = today - timedelta(days=7)

    st.sidebar.header("Filters")

    basis_label = st.sidebar.radio(
        "Date Basis",
        ["TRN_DATE", "ZED_DATE"],
        index=0,
        help="Choose which date column to use for filtering.",
    )
    date_basis = basis_label

    c1, c2 = st.sidebar.columns(2)
    with c1:
        start_date = st.sidebar.date_input(
            "Start Date", value=default_start, max_value=today
        )
    with c2:
        end_date = st.sidebar.date_input(
            "End Date", value=today, max_value=today
        )

    if start_date > end_date:
        st.sidebar.error("End date must be on or after Start date.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Section")

    section = st.sidebar.selectbox(
        "Main Section",
        ["SALES", "OPERATIONS", "INSIGHTS"],
    )

    if section == "SALES":
        sub = st.sidebar.selectbox(
            "Sales Report",
            [
                "Global Sales Overview",
                "Global Net Sales Distribution by Sales Channel (L2)",
                "Global Net Sales Distribution by SHIFT",
                "Night vs Day Shift Sales Ratio — Stores with Night Shifts",
            ],
        )
    elif section == "OPERATIONS":
        sub = st.sidebar.selectbox(
            "Operations Report",
            ["Customer Traffic — Storewise"],
        )
    else:  # INSIGHTS
        sub = st.sidebar.selectbox(
            "Insights Report",
            ["Global Category Overview — Sales"],
        )

    # ---------------- Load data ----------------
    with st.spinner("Loading data from Supabase..."):
        df = load_supabase_data(date_basis, start_date, end_date)

    if df.empty:
        st.warning("No data returned from Supabase for the selected filters.")
        return

    st.caption(
        f"Data source: Supabase • Table: daily_pos_trn_items_clean • "
        f"Rows: {len(df):,}"
    )

    # ---------------- Period Trend Summary (Option B add-on) ----------------
    render_period_trend_summary(df, date_basis, start_date, end_date)

    st.markdown("---")

    # ---------------- Route to report ----------------
    if section == "SALES":
        if sub == "Global Sales Overview":
            sales_global_overview(df)
        elif sub == "Global Net Sales Distribution by Sales Channel (L2)":
            sales_channel_distribution(df)
        elif sub == "Global Net Sales Distribution by SHIFT":
            sales_shift_distribution(df)
        elif sub == (
            "Night vs Day Shift Sales Ratio — Stores with Night Shifts"
        ):
            sales_night_vs_day(df)

    elif section == "OPERATIONS":
        if sub == "Customer Traffic — Storewise":
            operations_customer_traffic(df)

    elif section == "INSIGHTS":
        if sub == "Global Category Overview — Sales":
            insights_category_overview(df)


if __name__ == "__main__":
    main()
