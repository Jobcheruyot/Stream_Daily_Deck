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

@st.cache_resource
def init_supabase() -> Client:
    """
    Initialize Supabase client using secrets.toml.

    Required in .streamlit/secrets.toml:
        SUPABASE_URL = "https://eifqphcrwzddrmvdmtek.supabase.co"
        SUPABASE_KEY = "<eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVpZnFwaGNyd3pkZHJtdmRtdGVrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMxMTAyODQsImV4cCI6MjA3ODY4NjI4NH0.DlC8ZBUO0BZAcKR1IzVmaCyc4hD7OGVPlcYWj3sTXec>"
    """
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception as e:
        st.error(
            "Supabase credentials not found in secrets.\n\n"
            "Add SUPABASE_URL and SUPABASE_KEY to .streamlit/secrets.toml."
        )
        st.stop()

    return create_client(url, key)


# ============================================================================
# Data Loading from Supabase
# ============================================================================

@st.cache_data(ttl=3600)
def load_supabase_data(
    date_basis: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Load public.daily_pos_trn_items_clean from Supabase using TRN_DATE or ZED_DATE
    as the basis for filtering.
    """
    client = init_supabase()

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    if date_basis not in ("TRN_DATE", "ZED_DATE"):
        raise ValueError("date_basis must be 'TRN_DATE' or 'ZED_DATE'.")

    # Supabase query
    response = (
        client.table("public.daily_pos_trn_items_clean")
        .select("*")
        .gte(date_basis, start_str)
        .lte(date_basis, end_str)
        .execute()
    )

    data = response.data or []
    df = pd.DataFrame(data)

    return df


# ============================================================================
# Sidebar: Date Filters + Section Selection
# ============================================================================

def sidebar_config():
    st.sidebar.markdown("## Configuration")

    # Date basis
    date_basis = st.sidebar.radio(
        "Date Basis",
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
                "Stores Sales Summary",
            ],
        )
    elif section == "OPERATIONS":
        sub = st.sidebar.selectbox(
            "Operations Report",
            [
                "Customer Traffic-Storewise (30-min Heatmap)",
                "Cashiers Performance",
            ],
        )
    else:  # INSIGHTS
        sub = st.sidebar.selectbox(
            "Insights Report",
            [
                "Customer Baskets Overview (Coming Soon)",
            ],
        )

    return date_basis, start_date, end_date, section, sub


# ============================================================================
# Data Cleaning & Derived Columns
# ============================================================================

@st.cache_data
def clean_and_derive(df: pd.DataFrame, date_basis: str) -> pd.DataFrame:
    """
    Clean dataset and create derived columns used across reports.
    """
    if df is None or df.empty:
        return df

    d = df.copy()

    # --- Normalize string columns
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
        st.error(f"{date_basis} column not found in data.")
        st.stop()

    d = d.dropna(subset=[date_basis]).copy()
    d["DATE"] = d[date_basis].dt.date
    d["TIME_INTERVAL"] = d[date_basis].dt.floor("30min")
    d["TIME_ONLY"] = d["TIME_INTERVAL"].dt.time
    d["DAY_OF_WEEK"] = d[date_basis].dt.day_name()
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
                d[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

    # --- GROSS_SALES
    if "GROSS_SALES" not in d.columns:
        d["GROSS_SALES"] = d.get("NET_SALES", 0) + d.get("VAT_AMT", 0)

    # --- CUST_CODE
    if all(col in d.columns for col in ["STORE_CODE", "TILL", "SESSION", "RCT"]):
        d["CUST_CODE"] = (
            d["STORE_CODE"].astype(str)
            + "-"
            + d["TILL"].astype(str)
            + "-"
            + d["SESSION"].astype(str)
            + "-"
            + d["RCT"].astype(str)
        )
    else:
        if "CUST_CODE" not in d.columns:
            d["CUST_CODE"] = ""

    # --- Till_Code
    if "TILL" in d.columns and "STORE_CODE" in d.columns:
        d["Till_Code"] = d["TILL"].astype(str) + "-" + d["STORE_CODE"].astype(str)

    # --- CASHIER-COUNT
    if "STORE_NAME" in d.columns and "CASHIER" in d.columns:
        d["CASHIER-COUNT"] = d["STORE_NAME"] + "-" + d["CASHIER"]

    # --- Shift bucket
    if "SHIFT" in d.columns:
        d["Shift_Bucket"] = np.where(
            d["SHIFT"].str.upper().str.contains("NIGHT", na=False),
            "Night",
            "Day",
        )

    return d


# ============================================================================
# Helper: Table Formatting
# ============================================================================

def format_and_display(
    df: pd.DataFrame,
    numeric_cols: list | None = None,
    index_col: str | None = None,
    total_label: str = "TOTAL",
):
    """
    Format DataFrame with totals row and numeric formatting, then display.
    """
    if df is None or df.empty:
        st.info("No data to display.")
        return

    df_disp = df.copy()

    if numeric_cols is None:
        numeric_cols = list(df_disp.select_dtypes(include=[np.number]).columns)

    totals = {}
    for col in df_disp.columns:
        if col in numeric_cols:
            try:
                totals[col] = float(df_disp[col].sum())
            except Exception:
                totals[col] = ""
        else:
            totals[col] = ""

    # Label column for TOTAL
    if index_col and index_col in df_disp.columns:
        label_col = index_col
    else:
        non_num = [c for c in df_disp.columns if c not in numeric_cols]
        label_col = non_num[0] if non_num else df_disp.columns[0]

    totals[label_col] = total_label

    total_row = pd.DataFrame([totals], columns=df_disp.columns)
    out = pd.concat([df_disp, total_row], ignore_index=True)

    # numeric formatting
    for col in numeric_cols:
        if col not in out.columns:
            continue
        series = out[col]
        try:
            series = series.astype(float)
        except Exception:
            continue
        is_int_like = np.allclose(
            series.fillna(0).round(0),
            series.fillna(0),
        )
        if is_int_like:
            out[col] = out[col].map(
                lambda v: f"{int(v):,}" if pd.notna(v) and v != "" else ""
            )
        else:
            out[col] = out[col].map(
                lambda v: f"{float(v):,.2f}" if pd.notna(v) and v != "" else ""
            )

    st.dataframe(out, use_container_width=True)


# ============================================================================
# SALES REPORTS
# ============================================================================

def sales_global_overview(df: pd.DataFrame):
    st.header("Global Sales Overview")

    if "SALES_CHANNEL_L1" not in df.columns or "NET_SALES" not in df.columns:
        st.warning("SALES_CHANNEL_L1 or NET_SALES missing in data.")
        return

    # Total by channel (L1)
    summary = (
        df.groupby("SALES_CHANNEL_L1", as_index=False)["NET_SALES"]
        .sum()
        .sort_values("NET_SALES", ascending=False)
    )
    if summary.empty:
        st.info("No sales data for selected period.")
        return

    # Donut chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=summary["SALES_CHANNEL_L1"],
                values=summary["NET_SALES"],
                hole=0.6,
                hovertemplate="<b>%{label}</b><br>KSh %{value:,.0f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(title="<b>SALES CHANNEL TYPE — Selected Period Overview</b>")
    st.plotly_chart(fig, use_container_width=True)

    format_and_display(
        summary[["SALES_CHANNEL_L1", "NET_SALES"]],
        numeric_cols=["NET_SALES"],
        index_col="SALES_CHANNEL_L1",
        total_label="TOTAL",
    )


def sales_by_channel_l2(df: pd.DataFrame):
    st.header("Global Net Sales Distribution by Sales Channel (L2)")

    if "SALES_CHANNEL_L2" not in df.columns or "NET_SALES" not in df.columns:
        st.warning("SALES_CHANNEL_L2 or NET_SALES missing in data.")
        return

    summary = (
        df.groupby("SALES_CHANNEL_L2", as_index=False)["NET_SALES"]
        .sum()
        .sort_values("NET_SALES", ascending=False)
    )
    if summary.empty:
        st.info("No sales data for selected period.")
        return

    fig = go.Figure(
        data=[
            go.Pie(
                labels=summary["SALES_CHANNEL_L2"],
                values=summary["NET_SALES"],
                hole=0.6,
                hovertemplate="<b>%{label}</b><br>KSh %{value:,.0f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="<b>Global Net Sales Distribution by Sales Mode (SALES_CHANNEL_L2)</b>"
    )
    st.plotly_chart(fig, use_container_width=True)

    format_and_display(
        summary[["SALES_CHANNEL_L2", "NET_SALES"]],
        numeric_cols=["NET_SALES"],
        index_col="SALES_CHANNEL_L2",
        total_label="TOTAL",
    )


def sales_by_shift(df: pd.DataFrame):
    st.header("Global Net Sales Distribution by SHIFT")

    if "SHIFT" not in df.columns or "NET_SALES" not in df.columns:
        st.warning("SHIFT or NET_SALES missing in data.")
        return

    summary = (
        df.groupby("SHIFT", as_index=False)["NET_SALES"]
        .sum()
        .sort_values("NET_SALES", ascending=False)
    )
    if summary.empty:
        st.info("No sales data for selected period.")
        return

    summary["PCT"] = 100 * summary["NET_SALES"] / summary["NET_SALES"].sum()

    labels = [f"{row['SHIFT']} ({row['PCT']:.1f}%)" for _, row in summary.iterrows()]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=summary["NET_SALES"],
                hole=0.6,
            )
        ]
    )
    fig.update_layout(title="<b>Global Net Sales Distribution by SHIFT</b>")
    st.plotly_chart(fig, use_container_width=True)

    format_and_display(
        summary[["SHIFT", "NET_SALES", "PCT"]],
        numeric_cols=["NET_SALES", "PCT"],
        index_col="SHIFT",
        total_label="TOTAL",
    )


def night_vs_day_ratio(df: pd.DataFrame):
    st.header("Night vs Day Shift Sales Ratio — Stores with Night Shifts")

    if "Shift_Bucket" not in df.columns or "STORE_NAME" not in df.columns:
        st.warning("Shift_Bucket or STORE_NAME missing in data.")
        return

    stores_with_night = df[df["Shift_Bucket"] == "Night"]["STORE_NAME"].unique()
    d = df[df["STORE_NAME"].isin(stores_with_night)].copy()

    if d.empty:
        st.info("No stores with Night shift in selected period.")
        return

    ratio = (
        d.groupby(["STORE_NAME", "Shift_Bucket"], as_index=False)["NET_SALES"]
        .sum()
        .rename(columns={"NET_SALES": "Sales"})
    )
    ratio["STORE_TOTAL"] = ratio.groupby("STORE_NAME")["Sales"].transform("sum")
    ratio["PCT"] = 100 * ratio["Sales"] / ratio["STORE_TOTAL"]

    pivot = ratio.pivot(
        index="STORE_NAME", columns="Shift_Bucket", values="PCT"
    ).fillna(0)

    if pivot.empty:
        st.info("No data for Night vs Day.")
        return

    pivot = pivot.sort_values("Night", ascending=False)
    labels = [f"{i+1}. {name}" for i, name in enumerate(pivot.index)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=pivot["Night"],
            y=labels,
            orientation="h",
            name="Night",
            text=[f"{v:.1f}%" for v in pivot["Night"]],
            textposition="inside",
        )
    )

    for i, (n_val, d_val) in enumerate(
        zip(pivot["Night"], pivot.get("Day", pd.Series([0] * len(pivot))))
    ):
        fig.add_annotation(
            x=n_val + 1,
            y=labels[i],
            text=f"{d_val:.1f}% Day",
            showarrow=False,
            xanchor="left",
        )

    fig.update_layout(
        title="Night vs Day Shift Sales Ratio — Stores with Night Shifts",
        xaxis_title="% of Store Sales",
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)


def stores_sales_summary(df: pd.DataFrame):
    st.header("Stores Sales Summary")

    if "STORE_NAME" not in df.columns or "NET_SALES" not in df.columns:
        st.warning("STORE_NAME or NET_SALES missing in data.")
        return

    summary = (
        df.groupby("STORE_NAME", as_index=False)["NET_SALES"]
        .sum()
        .sort_values("NET_SALES", ascending=False)
    )
    if summary.empty:
        st.info("No sales data for selected period.")
        return

    # Rank
    summary["Rank"] = np.arange(1, len(summary) + 1)
    summary = summary[["Rank", "STORE_NAME", "NET_SALES"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=summary["STORE_NAME"],
                y=summary["NET_SALES"],
                text=[f"{v:,.0f}" for v in summary["NET_SALES"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Net Sales by Store",
        xaxis_title="Store",
        yaxis_title="Net Sales",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)

    format_and_display(
        summary,
        numeric_cols=["NET_SALES"],
        index_col="STORE_NAME",
        total_label="TOTAL",
    )


# ============================================================================
# OPERATIONS REPORTS
# ============================================================================

def customer_traffic_storewise(df: pd.DataFrame):
    st.header("Customer Traffic-Storewise (30-min Heatmap)")

    if "STORE_NAME" not in df.columns or "DATE" not in df.columns:
        st.warning("STORE_NAME or DATE missing in data.")
        return

    d = df.copy()

    # Build CUST_CODE if missing
    if "CUST_CODE" not in d.columns or d["CUST_CODE"].astype(str).str.strip().eq("").all():
        needed = ["STORE_CODE", "TILL", "SESSION", "RCT"]
        if not all(c in d.columns for c in needed):
            st.warning("CUST_CODE and its components are missing.")
            return
        for c in needed:
            d[c] = d[c].astype(str).fillna("").str.strip()
        d["CUST_CODE"] = (
            d["STORE_CODE"] + "-" + d["TILL"] + "-" + d["SESSION"] + "-" + d["RCT"]
        )
    else:
        d["CUST_CODE"] = d["CUST_CODE"].astype(str).fillna("").str.strip()

    # First-tap per customer/day
    ft = (
        d.groupby(["STORE_NAME", "DATE", "CUST_CODE"], as_index=False)["TIME_INTERVAL"]
        .min()
        .rename(columns={"TIME_INTERVAL": "FIRST_TIME"})
    )
    ft["TIME_ONLY"] = ft["FIRST_TIME"].dt.time

    # 30-min grid
    base = datetime.combine(datetime.today(), datetime.min.time())
    intervals = [(base + timedelta(minutes=30 * i)).time() for i in range(48)]
    labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]

    counts = (
        ft.groupby(["STORE_NAME", "TIME_ONLY"])["CUST_CODE"]
        .nunique()
        .reset_index(name="RECEIPT_COUNT")
    )
    if counts.empty:
        st.info("No traffic data for selected period.")
        return

    heat = counts.pivot(
        index="STORE_NAME",
        columns="TIME_ONLY",
        values="RECEIPT_COUNT",
    ).fillna(0)

    for t in intervals:
        if t not in heat.columns:
            heat[t] = 0

    heat = heat[intervals]
    heat["TOTAL"] = heat.sum(axis=1)
    heat = heat.sort_values("TOTAL", ascending=False)

    totals = heat["TOTAL"]
    heat_matrix = heat.drop(columns=["TOTAL"])

    z = heat_matrix.values
    if z.size == 0:
        st.info("No traffic data for selected period.")
        return
    zmax = float(z.max()) if z.max() > 0 else 1.0

    fig = px.imshow(
        z,
        x=labels,
        y=heat_matrix.index,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[
            [0.0, "#E6E6E6"],
            [0.001, "#FFFFCC"],
            [0.25, "#FED976"],
            [0.50, "#FEB24C"],
            [0.75, "#FD8D3C"],
            [1.0, "#E31A1C"],
        ],
        zmin=0,
        zmax=zmax,
        labels={
            "x": "Time Interval (30 min)",
            "y": "Store Name",
            "color": "Receipts",
        },
    )
    fig.update_xaxes(side="top")

    # Totals annotation
    for i, total in enumerate(totals):
        fig.add_annotation(
            x=-0.6,
            y=i,
            text=f"{int(total):,}",
            showarrow=False,
            xanchor="right",
            yanchor="middle",
        )
    fig.add_annotation(
        x=-0.6,
        y=-1,
        text="<b>TOTAL</b>",
        showarrow=False,
        xanchor="right",
        yanchor="top",
    )

    fig.update_layout(
        title="Customer Traffic Heatmap (Aggregated Over Selected Period)",
        xaxis_title="Time of Day",
        yaxis_title="Store Name",
        height=max(600, 24 * len(heat_matrix.index)),
        margin=dict(l=185, r=20, t=80, b=40),
        coloraxis_colorbar=dict(title="Receipt Count"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Store totals table
    totals_df = totals.reset_index()
    totals_df.columns = ["STORE_NAME", "Total_Receipts"]
    st.subheader("Storewise Total Receipts (Deduped)")
    format_and_display(
        totals_df,
        numeric_cols=["Total_Receipts"],
        index_col="STORE_NAME",
        total_label="TOTAL",
    )


def cashiers_performance(df: pd.DataFrame):
    st.header("Cashiers Performance")

    if "DATE" not in df.columns:
        st.warning("DATE column missing in data.")
        return

    d = df.copy()

    # Ensure identifiers
    required_cols = ["STORE_CODE", "TILL", "SESSION", "RCT", "CASHIER", "ITEM_CODE"]
    missing = [c for c in required_cols if c not in d.columns]
    if missing:
        st.warning(f"Missing columns: {missing}")
        return

    for c in required_cols + ["STORE_NAME"]:
        if c in d.columns:
            d[c] = d[c].astype(str).fillna("").str.strip()

    # CUST_CODE
    if "CUST_CODE" not in d.columns:
        d["CUST_CODE"] = (
            d["STORE_CODE"] + "-" + d["TILL"] + "-" + d["SESSION"] + "-" + d["RCT"]
        )
    else:
        d["CUST_CODE"] = d["CUST_CODE"].astype(str).fillna("").str.strip()

    # CASHIER-COUNT
    if "CASHIER-COUNT" not in d.columns:
        d["CASHIER-COUNT"] = d["STORE_NAME"] + "-" + d["CASHIER"]

    # Receipt-level duration & item count
    receipt_duration = (
        d.groupby(["STORE_NAME", "CUST_CODE", "DATE"], as_index=False)
        .agg(
            Start_Time=("TIME_INTERVAL", "min"),
            End_Time=("TIME_INTERVAL", "max"),
        )
    )
    receipt_duration["Duration_Sec"] = (
        receipt_duration["End_Time"] - receipt_duration["Start_Time"]
    ).dt.total_seconds().fillna(0)

    receipt_items = (
        d.groupby(["STORE_NAME", "CUST_CODE", "DATE"], as_index=False)["ITEM_CODE"]
        .nunique()
        .rename(columns={"ITEM_CODE": "Unique_Items"})
    )

    receipt_stats = receipt_duration.merge(
        receipt_items,
        on=["STORE_NAME", "CUST_CODE", "DATE"],
        how="left",
    )

    # Store-level summary
    store_summary = (
        receipt_stats.groupby("STORE_NAME", as_index=False)
        .agg(
            Total_Customers=("CUST_CODE", "nunique"),
            Avg_Time_per_Customer_Min=(
                "Duration_Sec",
                lambda s: s.mean() / 60.0,
            ),
            Avg_Items_per_Receipt=("Unique_Items", "mean"),
        )
        .sort_values("Avg_Time_per_Customer_Min", ascending=True)
    )

    if store_summary.empty:
        st.info("No cashier data for selected period.")
        return

    store_summary["Avg_Time_per_Customer_Min"] = (
        store_summary["Avg_Time_per_Customer_Min"].round(1)
    )
    store_summary["Avg_Items_per_Receipt"] = (
        store_summary["Avg_Items_per_Receipt"].round(1)
    )

    # Rank
    store_summary.insert(0, "#", np.arange(1, len(store_summary) + 1))

    fig = go.Figure(
        data=[
            go.Bar(
                x=store_summary["STORE_NAME"],
                y=store_summary["Avg_Time_per_Customer_Min"],
                text=[
                    f"{v:.1f} min" for v in store_summary["Avg_Time_per_Customer_Min"]
                ],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Average Time per Customer (Minutes) by Store",
        xaxis_title="Store",
        yaxis_title="Avg Time per Customer (min)",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Store-level Cashier Performance Summary")
    format_and_display(
        store_summary[
            [
                "#",
                "STORE_NAME",
                "Total_Customers",
                "Avg_Time_per_Customer_Min",
                "Avg_Items_per_Receipt",
            ]
        ],
        numeric_cols=[
            "Total_Customers",
            "Avg_Time_per_Customer_Min",
            "Avg_Items_per_Receipt",
        ],
        index_col="STORE_NAME",
        total_label="TOTAL",
    )


# ============================================================================
# INSIGHTS (Placeholder for now)
# ============================================================================

def customer_baskets_overview(df: pd.DataFrame):
    st.header("Customer Baskets Overview (Coming Soon)")
    st.info(
        "This section will show basket size, mix, and key categories per basket. "
        "For now, focus on SALES and OPERATIONS sections above."
    )


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown(
        """
        <div style="text-align:center; margin-bottom: 1.5rem;">
            <h1 style="margin-bottom:0;">Quick Mart Limited</h1>
            <p style="color:#d72638; font-weight:600; margin-top:0;">Let the data Speak</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    date_basis, start_date, end_date, section, sub = sidebar_config()

    # Load data
    with st.spinner("Loading data from Supabase..."):
        raw_df = load_supabase_data(date_basis, start_date, end_date)

    if raw_df is None or raw_df.empty:
        st.warning("No data returned from Supabase for the selected period.")
        return

    df = clean_and_derive(raw_df, date_basis)

    st.caption(
        f"Data source: Supabase • Table: public.daily_pos_trn_items_clean • "
        f"Date Basis: {date_basis} • Period: {start_date} to {end_date}"
    )

    # Route to selected report
    if section == "SALES":
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
        if sub == "Customer Traffic-Storewise (30-min Heatmap)":
            customer_traffic_storewise(df)
        elif sub == "Cashiers Performance":
            cashiers_performance(df)

    elif section == "INSIGHTS":
        if sub == "Customer Baskets Overview (Coming Soon)":
            customer_baskets_overview(df)


if __name__ == "__main__":
    main()



