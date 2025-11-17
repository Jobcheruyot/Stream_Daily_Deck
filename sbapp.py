###############################################################
#  DAILYDECK – SUPABASE, MULTI-DAY, MILLION-ROW READY
###############################################################

import os
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from supabase import create_client, Client

# ------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="DailyDeck – Multi-Day Retail Dashboard",
    layout="wide",
)

# ------------------------------------------------------------
# 1. SUPABASE CONNECTION
# ------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_supabase_client() -> Client:
    """
    Create a single cached Supabase client instance.
    Reads credentials from Streamlit secrets.
    """
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


# ------------------------------------------------------------
# 2. LOAD DATA FROM SUPABASE (NO ROW LIMIT, CHUNKED)
# ------------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_from_supabase(start_date: date, end_date: date,
                       page_size: int = 200_000) -> pd.DataFrame:
    """
    Load ALL rows from Supabase for the selected date range.

    Uses chunked pagination with .range() so we are not stuck at 1 000 rows.
    It keeps pulling pages until the API returns fewer than page_size rows.

    NOTE: This still has to fit in RAM – for *true* 20M+ scale,
    you eventually want to push more aggregation into SQL.
    """

    client = get_supabase_client()

    # We'll filter on ZED_DATE (the "trading day"),
    # but you can switch to TRN_DATE if you prefer.
    start_iso = start_date.isoformat()
    end_iso = end_date.isoformat()

    all_frames: list[pd.DataFrame] = []
    offset = 0

    while True:
        resp = (
            client.table("daily_pos_trn_items_clean")
            .select("*")
            .gte("zed_date", start_iso)
            .lte("zed_date", end_iso)
            .range(offset, offset + page_size - 1)
            .execute()
        )

        data = resp.data
        if not data:
            break

        chunk_df = pd.DataFrame(data)
        all_frames.append(chunk_df)

        # If we got less than a full page, we're done
        if len(chunk_df) < page_size:
            break

        offset += page_size

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)

    # Normalise column names to UPPERCASE so the rest of the
    # analytics code (ported from your CSV app) can remain similar.
    df.columns = [c.upper() for c in df.columns]

    return df


# ------------------------------------------------------------
# 3. SMART LOAD – WIRE INTO SIDEBAR DATE PICKER
# ------------------------------------------------------------

def smart_load() -> pd.DataFrame:
    """
    Supabase-based loader:
    - user selects date range in sidebar
    - we fetch ALL rows in that range in chunks (no hard row limit)
    """
    st.sidebar.markdown("### Select Date Range (Supabase)")
    today = date.today()
    default_start = today - timedelta(days=7)

    start_date = st.sidebar.date_input("Start date", default_start)
    end_date = st.sidebar.date_input("End date", today)

    if start_date > end_date:
        st.sidebar.error("Start date cannot be after end date")
        st.stop()

    with st.spinner("Loading data from Supabase (all rows in range) ..."):
        df = load_from_supabase(start_date, end_date)

    if df is None or df.empty:
        st.sidebar.warning("No data returned from Supabase for this period.")
        return None

    st.sidebar.success(
        f"Loaded {len(df):,} rows from Supabase\n"
        f"{start_date} → {end_date}"
    )
    return df


# ------------------------------------------------------------
# 4. CLEANING & DERIVED COLUMNS
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def clean_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise key fields, derive DATE / TIME_INTERVAL / CUST_CODE etc.
    This is an UPPERCASE version of your earlier cleaning logic.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    # Normalise strings
    str_cols = [
        "STORE_CODE", "TILL", "SESSION", "RCT", "STORE_NAME", "CASHIER",
        "ITEM_CODE", "ITEM_NAME", "DEPARTMENT", "CATEGORY",
        "CU_DEVICE_SERIAL", "CAP_CUSTOMER_CODE", "LOYALTY_CUSTOMER_CODE",
        "SUPPLIER_NAME", "SALES_CHANNEL_L1", "SALES_CHANNEL_L2", "SHIFT",
    ]
    for c in str_cols:
        if c in d.columns:
            d[c] = (
                d[c]
                .astype(str)
                .fillna("")
                .str.strip()
            )

    # TRN_DATE / ZED_DATE
    if "TRN_DATE" in d.columns:
        d["TRN_DATE"] = pd.to_datetime(d["TRN_DATE"], errors="coerce")
        d = d.dropna(subset=["TRN_DATE"]).copy()
        d["DATE"] = d["TRN_DATE"].dt.date
        d["TIME_INTERVAL"] = d["TRN_DATE"].dt.floor("30min")
        d["TIME_ONLY"] = d["TIME_INTERVAL"].dt.time

    if "ZED_DATE" in d.columns:
        d["ZED_DATE"] = pd.to_datetime(d["ZED_DATE"], errors="coerce")

    # Numerics
    numeric_cols = [
        "QTY", "CP_PRE_VAT", "SP_PRE_VAT",
        "COST_PRE_VAT", "NET_SALES", "VAT_AMT",
    ]
    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(
                d[c].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            ).fillna(0.0)

    # GROSS_SALES
    if "GROSS_SALES" not in d.columns:
        d["GROSS_SALES"] = d.get("NET_SALES", 0) + d.get("VAT_AMT", 0)

    # CUST_CODE
    if all(c in d.columns for c in ["STORE_CODE", "TILL", "SESSION", "RCT"]):
        d["CUST_CODE"] = (
            d["STORE_CODE"].astype(str) + "-"
            + d["TILL"].astype(str) + "-"
            + d["SESSION"].astype(str) + "-"
            + d["RCT"].astype(str)
        )
    else:
        if "CUST_CODE" not in d.columns:
            d["CUST_CODE"] = ""

    # Till_Code
    if "TILL" in d.columns and "STORE_CODE" in d.columns:
        d["TILL_CODE"] = d["TILL"].astype(str) + "-" + d["STORE_CODE"].astype(str)

    # CASHIER-COUNT
    if "STORE_NAME" in d.columns and "CASHIER" in d.columns:
        d["CASHIER-COUNT"] = d["CASHIER"].astype(str) + "-" + d["STORE_NAME"].astype(str)

    # Shift bucket
    if "SHIFT" in d.columns:
        d["SHIFT_BUCKET"] = np.where(
            d["SHIFT"].str.upper().str.contains("NIGHT", na=False),
            "Night",
            "Day",
        )

    return d


# ------------------------------------------------------------
# 5. SMALL AGG HELPERS & TABLE FORMATTER
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def agg_net_sales_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col, "NET_SALES"])
    g = (
        df.groupby(col, as_index=False)["NET_SALES"]
        .sum()
        .sort_values("NET_SALES", ascending=False)
    )
    return g


def format_and_display(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    index_col: str | None = None,
    total_label: str = "TOTAL",
):
    """
    Simple "TOTAL" row + formatted numeric columns.
    """
    if df is None or df.empty:
        st.dataframe(df, use_container_width=True)
        return

    df_display = df.copy()

    if numeric_cols is None:
        numeric_cols = list(df_display.select_dtypes(include=[np.number]).columns)

    totals = {}
    for col in df_display.columns:
        if col in numeric_cols:
            try:
                totals[col] = float(df_display[col].sum())
            except Exception:
                totals[col] = ""
        else:
            totals[col] = ""

    # choose label column
    if index_col and index_col in df_display.columns:
        label_col = index_col
    else:
        non_nums = [c for c in df_display.columns if c not in numeric_cols]
        label_col = non_nums[0] if non_nums else df_display.columns[0]

    totals[label_col] = total_label
    tot_df = pd.DataFrame([totals], columns=df_display.columns)
    appended = pd.concat([df_display, tot_df], ignore_index=True)

    for col in numeric_cols:
        if col not in appended.columns:
            continue
        series_vals = appended[col].dropna()
        if len(series_vals) == 0:
            continue
        try:
            series_vals = series_vals.astype(float)
        except Exception:
            continue

        is_int_like = np.allclose(
            series_vals.round(0),
            series_vals,
            rtol=0,
            atol=1e-6,
        )
        if is_int_like:
            appended[col] = appended[col].map(
                lambda v: f"{int(v):,}" if pd.notna(v) and v != "" else ""
            )
        else:
            appended[col] = appended[col].map(
                lambda v: f"{float(v):,.2f}" if pd.notna(v) and v != "" else ""
            )

    st.dataframe(appended, use_container_width=True)


def donut_from_agg(
    df_agg: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    hole: float = 0.55,
):
    labels = df_agg[label_col].astype(str).tolist()
    vals = df_agg[value_col].astype(float).tolist()
    s = sum(vals) or 1
    legend_labels = [f"{lab} ({100 * v / s:.1f}%)" for lab, v in zip(labels, vals)]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=legend_labels,
                values=vals,
                hole=hole,
                hovertemplate="<b>%{label}</b><br>KSh %{value:,.2f}<extra></extra>",
                marker=dict(line=dict(color="white", width=1)),
            )
        ]
    )
    fig.update_layout(title=title)
    return fig


# ------------------------------------------------------------
# 6. TREND STRIP – SHOWN FOR EVERY SECTION
# ------------------------------------------------------------

def show_trends(df: pd.DataFrame, section_name: str):
    """
    Lightweight "trend strip" – Net Sales & Receipts per day
    for whatever window the user picked.
    """
    if df is None or df.empty or "DATE" not in df.columns:
        return

    trend = (
        df.groupby("DATE", as_index=False)
        .agg(
            NET_SALES=("NET_SALES", "sum"),
            RECEIPTS=("CUST_CODE", pd.Series.nunique),
        )
        .sort_values("DATE")
    )

    if trend.empty:
        return

    st.subheader(f"{section_name.title()} – Period Trends")

    c1, c2 = st.columns(2)

    with c1:
        fig1 = px.line(
            trend,
            x="DATE",
            y="NET_SALES",
            markers=True,
            title="Net Sales by Day",
        )
        fig1.update_traces(mode="lines+markers")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.line(
            trend,
            x="DATE",
            y="RECEIPTS",
            markers=True,
            title="Receipt Count by Day",
        )
        fig2.update_traces(mode="lines+markers")
        st.plotly_chart(fig2, use_container_width=True)


# ------------------------------------------------------------
# 7. SALES SECTION – KEY SUBSECTIONS
# (You can keep extending with your full original logic)
# ------------------------------------------------------------

def sales_global_overview(df: pd.DataFrame):
    st.header("Global Sales Overview")

    if "SALES_CHANNEL_L1" not in df.columns or "NET_SALES" not in df.columns:
        st.warning("Missing SALES_CHANNEL_L1 or NET_SALES")
        return

    g = agg_net_sales_by(df, "SALES_CHANNEL_L1")
    fig = donut_from_agg(
        g,
        "SALES_CHANNEL_L1",
        "NET_SALES",
        "<b>Sales Channel Type – Global Overview</b>",
        hole=0.65,
    )
    st.plotly_chart(fig, use_container_width=True)

    format_and_display(
        g[["SALES_CHANNEL_L1", "NET_SALES"]],
        numeric_cols=["NET_SALES"],
        index_col="SALES_CHANNEL_L1",
        total_label="TOTAL",
    )


def sales_by_channel_l2(df: pd.DataFrame):
    st.header("Global Net Sales Distribution by Sales Channel (L2)")

    if "SALES_CHANNEL_L2" not in df.columns or "NET_SALES" not in df.columns:
        st.warning("Missing SALES_CHANNEL_L2 or NET_SALES")
        return

    g = agg_net_sales_by(df, "SALES_CHANNEL_L2")
    fig = donut_from_agg(
        g,
        "SALES_CHANNEL_L2",
        "NET_SALES",
        "<b>Global Net Sales by Sales Mode (SALES_CHANNEL_L2)</b>",
        hole=0.65,
    )
    st.plotly_chart(fig, use_container_width=True)

    format_and_display(
        g[["SALES_CHANNEL_L2", "NET_SALES"]],
        numeric_cols=["NET_SALES"],
        index_col="SALES_CHANNEL_L2",
        total_label="TOTAL",
    )


def sales_by_shift(df: pd.DataFrame):
    st.header("Global Net Sales Distribution by SHIFT")

    if "SHIFT" not in df.columns or "NET_SALES" not in df.columns:
        st.warning("Missing SHIFT or NET_SALES")
        return

    g = (
        df.groupby("SHIFT", as_index=False)["NET_SALES"]
        .sum()
        .sort_values("NET_SALES", ascending=False)
    )
    g["PCT"] = 100 * g["NET_SALES"] / g["NET_SALES"].sum()

    labels = [f"{row.SHIFT} ({row.PCT:.1f}%)" for _, row in g.iterrows()]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=g["NET_SALES"],
                hole=0.65,
            )
        ]
    )
    fig.update_layout(title="<b>Global Net Sales Distribution by SHIFT</b>")
    st.plotly_chart(fig, use_container_width=True)

    format_and_display(
        g[["SHIFT", "NET_SALES", "PCT"]],
        numeric_cols=["NET_SALES", "PCT"],
        index_col="SHIFT",
        total_label="TOTAL",
    )


def stores_sales_summary(df: pd.DataFrame):
    st.header("Stores Sales Summary")

    if "STORE_NAME" not in df.columns:
        st.warning("Missing STORE_NAME")
        return

    d = df.copy()
    d["NET_SALES"] = pd.to_numeric(d.get("NET_SALES", 0), errors="coerce").fillna(0)
    d["VAT_AMT"] = pd.to_numeric(d.get("VAT_AMT", 0), errors="coerce").fillna(0)
    d["GROSS_SALES"] = d["NET_SALES"] + d["VAT_AMT"]

    summary = (
        d.groupby("STORE_NAME", as_index=False)[["NET_SALES", "GROSS_SALES"]]
        .sum()
        .sort_values("GROSS_SALES", ascending=False)
    )
    summary["% Contribution"] = (
        summary["GROSS_SALES"] / summary["GROSS_SALES"].sum() * 100
    ).round(2)

    if "CUST_CODE" in d.columns:
        cust_counts = (
            d.groupby("STORE_NAME")["CUST_CODE"]
            .nunique()
            .reset_index()
            .rename(columns={"CUST_CODE": "Customer Numbers"})
        )
        summary = summary.merge(cust_counts, on="STORE_NAME", how="left")

    cols = [
        "STORE_NAME",
        "NET_SALES",
        "GROSS_SALES",
        "% Contribution",
        "Customer Numbers",
    ]
    cols = [c for c in cols if c in summary.columns]

    format_and_display(
        summary[cols].fillna(0),
        numeric_cols=[c for c in cols if c != "STORE_NAME"],
        index_col="STORE_NAME",
        total_label="TOTAL",
    )


# ------------------------------------------------------------
# 8. OPERATIONS SECTION – A FEW HEAVY LIFTERS
# (You can expand with all your detailed heatmaps later)
# ------------------------------------------------------------

def customer_traffic_storewise(df: pd.DataFrame):
    st.header("Customer Traffic – Storewise (by Day)")

    required = ["DATE", "STORE_NAME", "CUST_CODE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns: {missing}")
        return

    traffic = (
        df.groupby(["STORE_NAME", "DATE"], as_index=False)["CUST_CODE"]
        .nunique()
        .rename(columns={"CUST_CODE": "RECEIPTS"})
    )

    fig = px.line(
        traffic,
        x="DATE",
        y="RECEIPTS",
        color="STORE_NAME",
        markers=True,
        title="Receipts per Store by Day",
    )
    st.plotly_chart(fig, use_container_width=True)

    pivot = traffic.pivot(index="STORE_NAME", columns="DATE", values="RECEIPTS").fillna(0)
    st.subheader("Storewise Receipts per Day")
    format_and_display(
        pivot.reset_index(),
        numeric_cols=[c for c in pivot.columns],
        index_col="STORE_NAME",
        total_label="TOTAL",
    )


def active_tills_during_day(df: pd.DataFrame):
    st.header("Peak Active Tills (simple overview)")

    required = ["TRN_DATE", "STORE_NAME", "TILL_CODE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(
            "Missing required columns for this view (need TRN_DATE, STORE_NAME, TILL_CODE)"
        )
        return

    d = df.copy()
    d["TRN_DATE"] = pd.to_datetime(d["TRN_DATE"], errors="coerce")
    d["TIME_INTERVAL"] = d["TRN_DATE"].dt.floor("30min")

    till_counts = (
        d.groupby(["STORE_NAME", "TIME_INTERVAL"])["TILL_CODE"]
        .nunique()
        .reset_index(name="UNIQUE_TILLS")
    )

    fig = px.line(
        till_counts,
        x="TIME_INTERVAL",
        y="UNIQUE_TILLS",
        color="STORE_NAME",
        title="Number of Active Tills by 30-min Interval",
    )
    st.plotly_chart(fig, use_container_width=True)


def tax_compliance(df: pd.DataFrame):
    st.header("Tax Compliance")

    required = ["CU_DEVICE_SERIAL", "CUST_CODE", "STORE_NAME"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns for tax view: {missing}")
        return

    d = df.copy()
    d["Tax_Compliant"] = np.where(
        d["CU_DEVICE_SERIAL"]
        .astype(str)
        .replace({"nan": "", "NaN": "", "None": ""})
        .str.strip()
        .astype(bool),
        "Compliant",
        "Non-Compliant",
    )

    store_summary = (
        d.groupby(["STORE_NAME", "Tax_Compliant"], as_index=False)["CUST_CODE"]
        .nunique()
        .rename(columns={"CUST_CODE": "Receipts"})
    )

    pivot = store_summary.pivot(
        index="STORE_NAME", columns="Tax_Compliant", values="Receipts"
    ).fillna(0)

    pivot["Total"] = pivot.sum(axis=1)
    pivot["Compliance_%"] = np.where(
        pivot["Total"] > 0,
        (pivot.get("Compliant", 0) / pivot["Total"] * 100).round(1),
        0.0,
    )

    format_and_display(
        pivot.reset_index(),
        numeric_cols=["Compliant", "Non-Compliant", "Total", "Compliance_%"],
        index_col="STORE_NAME",
        total_label="TOTAL",
    )


# ------------------------------------------------------------
# 9. INSIGHTS SECTION – SAMPLE SUBSECTIONS
# ------------------------------------------------------------

def customer_baskets_overview(df: pd.DataFrame):
    st.header("Customer Baskets Overview")

    required = ["ITEM_NAME", "CUST_CODE", "STORE_NAME", "DEPARTMENT", "QTY", "NET_SALES"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns: {missing}")
        return

    d = df.copy()
    d = d.dropna(subset=["ITEM_NAME", "CUST_CODE", "STORE_NAME"])

    branches = sorted(d["STORE_NAME"].unique())
    if not branches:
        st.info("No branches in data.")
        return

    branch = st.selectbox("Branch for comparison", branches)
    metric = st.selectbox("Metric", ["QTY", "NET_SALES"])
    top_x = st.number_input("Top X", min_value=5, max_value=200, value=10)

    basket_count = (
        d.groupby("ITEM_NAME")["CUST_CODE"]
        .nunique()
        .rename("Count_of_Baskets")
    )
    agg_data = d.groupby("ITEM_NAME")[["QTY", "NET_SALES"]].sum()

    global_top = (
        basket_count.to_frame()
        .join(agg_data)
        .reset_index()
        .sort_values(metric, ascending=False)
        .head(int(top_x))
    )
    global_top.insert(0, "#", range(1, len(global_top) + 1))

    st.subheader("Global Top Items")
    format_and_display(
        global_top,
        numeric_cols=["Count_of_Baskets", "QTY", "NET_SALES"],
        index_col="ITEM_NAME",
        total_label="TOTAL",
    )

    branch_df = d[d["STORE_NAME"] == branch]
    if branch_df.empty:
        st.info("No data for selected branch")
        return

    basket_count_b = (
        branch_df.groupby("ITEM_NAME")["CUST_CODE"]
        .nunique()
        .rename("Count_of_Baskets")
    )
    agg_b = branch_df.groupby("ITEM_NAME")[["QTY", "NET_SALES"]].sum()
    branch_top = (
        basket_count_b.to_frame()
        .join(agg_b)
        .reset_index()
        .sort_values(metric, ascending=False)
        .head(int(top_x))
    )
    branch_top.insert(0, "#", range(1, len(branch_top) + 1))

    st.subheader(f"{branch} Top Items")
    format_and_display(
        branch_top,
        numeric_cols=["Count_of_Baskets", "QTY", "NET_SALES"],
        index_col="ITEM_NAME",
        total_label="TOTAL",
    )


def global_category_overview_sales(df: pd.DataFrame):
    st.header("Global Category Overview – Sales")

    if "CATEGORY" not in df.columns:
        st.warning("Missing CATEGORY")
        return

    g = agg_net_sales_by(df, "CATEGORY")
    format_and_display(
        g,
        numeric_cols=["NET_SALES"],
        index_col="CATEGORY",
        total_label="TOTAL",
    )

    fig = px.bar(
        g.head(20),
        x="NET_SALES",
        y="CATEGORY",
        orientation="h",
        title="Top Categories by Net Sales",
    )
    st.plotly_chart(fig, use_container_width=True)


def global_category_overview_baskets(df: pd.DataFrame):
    st.header("Global Category Overview – Baskets")

    if "CATEGORY" not in df.columns or "CUST_CODE" not in df.columns:
        st.warning("Missing CATEGORY or CUST_CODE")
        return

    g = (
        df.groupby("CATEGORY", as_index=False)["CUST_CODE"]
        .nunique()
        .rename(columns={"CUST_CODE": "Baskets"})
        .sort_values("Baskets", ascending=False)
    )

    format_and_display(
        g,
        numeric_cols=["Baskets"],
        index_col="CATEGORY",
        total_label="TOTAL",
    )

    fig = px.bar(
        g.head(20),
        x="Baskets",
        y="CATEGORY",
        orientation="h",
        title="Top Categories by Baskets",
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# 10. MAIN APP ROUTER
# ------------------------------------------------------------

def main():
    st.title("DailyDeck – The Story Behind the Numbers (Supabase Edition)")

    # 1) Load & clean data
    raw_df = smart_load()
    if raw_df is None:
        st.stop()

    with st.spinner("Preparing data (cached) ..."):
        df = clean_and_derive(raw_df)

    # 2) Top-level section
    section = st.sidebar.selectbox(
        "Section",
        ["SALES", "OPERATIONS", "INSIGHTS"],
    )

    # ----------------------- SALES ----------------------------
    if section == "SALES":
        sales_items = [
            "Global Sales Overview",
            "Global Net Sales Distribution by Sales Channel",
            "Global Net Sales Distribution by SHIFT",
            "Stores Sales Summary",
        ]
        choice = st.sidebar.selectbox("Sales Subsection", sales_items)

        # Trend strip first
        show_trends(df, section)

        if choice == sales_items[0]:
            sales_global_overview(df)
        elif choice == sales_items[1]:
            sales_by_channel_l2(df)
        elif choice == sales_items[2]:
            sales_by_shift(df)
        elif choice == sales_items[3]:
            stores_sales_summary(df)

    # -------------------- OPERATIONS --------------------------
    elif section == "OPERATIONS":
        ops_items = [
            "Customer Traffic-Storewise",
            "Active Tills During the Day",
            "Tax Compliance",
        ]
        choice = st.sidebar.selectbox("Operations Subsection", ops_items)

        show_trends(df, section)

        if choice == ops_items[0]:
            customer_traffic_storewise(df)
        elif choice == ops_items[1]:
            active_tills_during_day(df)
        elif choice == ops_items[2]:
            tax_compliance(df)

    # ---------------------- INSIGHTS --------------------------
    elif section == "INSIGHTS":
        ins_items = [
            "Customer Baskets Overview",
            "Global Category Overview-Sales",
            "Global Category Overview-Baskets",
        ]
        choice = st.sidebar.selectbox("Insights Subsection", ins_items)

        show_trends(df, section)

        if choice == ins_items[0]:
            customer_baskets_overview(df)
        elif choice == ins_items[1]:
            global_category_overview_sales(df)
        elif choice == ins_items[2]:
            global_category_overview_baskets(df)


if __name__ == "__main__":
    main()
