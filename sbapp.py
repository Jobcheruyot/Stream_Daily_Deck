###############################################################
#  DAILYDECK – SUPABASE EDITION (MULTI-DAY, MILLION-ROW READY)
###############################################################

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta, datetime
from supabase import create_client, Client

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="DailyDeck — Multi-Day Retail Dashboard",
)

# ------------------------------------------------------------
# ADMIN DETECTION
# ------------------------------------------------------------

def is_admin() -> bool:
    """
    Admin is the user with this email in session_state.
    Normal users never see row counts / debug info.
    """
    return st.session_state.get("user_email") == "cheruyotjob@gmail.com"


# ------------------------------------------------------------
# SUPABASE CONNECTION + LOADER (NO ROW LIMIT)
# ------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_supabase_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


@st.cache_data(show_spinner=True)
def load_from_supabase(
    start_date: date,
    end_date: date,
    chunk_size: int = 250_000,
) -> pd.DataFrame:
    """
    Load ALL rows from Supabase for the selected date range.

    - Filters on trn_date (timestamp) between start and end
    - Uses .range() pagination so there is NO 1,000 row cap
    - Returns empty DataFrame if no rows
    """

    client = get_supabase_client()

    # Supabase columns are lowercase; we filter on trn_date
    start_iso = f"{start_date.isoformat()}T00:00:00"
    end_iso = f"{end_date.isoformat()}T23:59:59.999999"

    all_chunks: list[pd.DataFrame] = []
    offset = 0

    while True:
        resp = (
            client.table("daily_pos_trn_items_clean")
            .select("*")
            .gte("trn_date", start_iso)
            .lte("trn_date", end_iso)
            .range(offset, offset + chunk_size - 1)
            .execute()
        )

        rows = resp.data or []

        if not rows:
            break

        chunk_df = pd.DataFrame(rows)
        all_chunks.append(chunk_df)

        # If we got less than a full page, no more data
        if len(rows) < chunk_size:
            break

        offset += chunk_size

    if not all_chunks:
        return pd.DataFrame()

    df = pd.concat(all_chunks, ignore_index=True)

    # Normalise to uppercase so the analytics code matches
    df.columns = [c.upper() for c in df.columns]

    return df


# ------------------------------------------------------------
# SMART LOAD – DATE PICKER + SUPABASE
# ------------------------------------------------------------

def smart_load() -> pd.DataFrame:
    """
    Sidebar date range + Supabase loader.

    - If no rows are found → shows message & st.stop()
    - Row counts & diagnostics are shown ONLY to admin.
    """
    st.sidebar.markdown("### Select Date Range (Supabase)")
    today = date.today()
    default_start = today - timedelta(days=7)

    start_date = st.sidebar.date_input("Start date", default_start)
    end_date = st.sidebar.date_input("End date", today)

    if start_date > end_date:
        st.sidebar.error("Start date cannot be after end date.")
        st.stop()

    with st.spinner("Loading data from Supabase..."):
        df = load_from_supabase(start_date, end_date)

    if df is None or df.empty:
        st.sidebar.error("No data found in Supabase for this date range.")
        st.stop()

    # User-facing message (no row counts)
    st.sidebar.success("Data loaded from Supabase.")

    # Admin-only diagnostics
    if is_admin():
        st.sidebar.caption(
            f"Admin: {len(df):,} rows loaded "
            f"from {start_date} to {end_date}"
        )

    return df


# ============================================================
#  FROM HERE DOWN: YOUR ORIGINAL ANALYTICS ENGINE
#  (slightly adapted to uppercase columns + trends)
# ============================================================

# -----------------------
# Robust cleaning + derived columns (cached)
# -----------------------
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
        st.dataframe(df, use_container_width=True)
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
def donut_from_agg(df_agg, label_col, value_col, title,
                   hole=0.55, colors=None,
                   legend_title=None, value_is_millions=False):
    labels = df_agg[label_col].astype(str).tolist()
    vals = df_agg[value_col].astype(float).tolist()
    if value_is_millions:
        vals_display = [v / 1_000_000 for v in vals]
        hover = 'KSh %{value:,.2f} M'
        values_for_plot = vals_display
    else:
        values_for_plot = vals
        hover = 'KSh %{value:,.2f}' if isinstance(vals[0], (int, float)) else '%{value}'
    s = sum(vals) if sum(vals) != 0 else 1
    legend_labels = [
        f"{lab} ({100*val/s:.1f}% | {val/1_000_000:.1f} M)" if value_is_millions
        else f"{lab} ({100*val/s:.1f}%)"
        for lab, val in zip(labels, vals)
    ]
    marker = dict(line=dict(color='white', width=1))
    if colors:
        marker['colors'] = colors
    fig = go.Figure(data=[go.Pie(
        labels=legend_labels,
        values=values_for_plot,
        hole=hole,
        hovertemplate='<b>%{label}</b><br>' + hover + '<extra></extra>',
        marker=marker
    )])
    fig.update_layout(title=title)
    return fig

# -----------------------
# Simple trend strip (per section)
# -----------------------
def show_trends(df: pd.DataFrame, section_name: str):
    """
    Shows daily Net Sales + Receipt count for the selected period.
    """
    if df is None or df.empty or 'DATE' not in df.columns:
        return

    trend = (
        df.groupby('DATE', as_index=False)
        .agg(
            NET_SALES=('NET_SALES', 'sum'),
            RECEIPTS=('CUST_CODE', pd.Series.nunique),
        )
        .sort_values('DATE')
    )
    if trend.empty:
        return

    st.subheader(f"{section_name.title()} – Period Trends")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(
            trend,
            x='DATE',
            y='NET_SALES',
            markers=True,
            title="Net Sales by Day",
        )
        fig1.update_traces(mode='lines+markers')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.line(
            trend,
            x='DATE',
            y='RECEIPTS',
            markers=True,
            title="Receipt Count by Day",
        )
        fig2.update_traces(mode='lines+markers')
        st.plotly_chart(fig2, use_container_width=True)

# ============================================================
#  SALES SECTION (all original subsections)
# ============================================================

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

def sales_by_channel_l2(df):
    st.header("Global Net Sales Distribution by Sales Channel")
    if 'SALES_CHANNEL_L2' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SALES_CHANNEL_L2 or NET_SALES")
        return
    g = agg_net_sales_by(df, 'SALES_CHANNEL_L2')
    g['NET_SALES_M'] = g['NET_SALES'] / 1_000_000
    fig = donut_from_agg(
        g,
        'SALES_CHANNEL_L2',
        'NET_SALES',
        "<b>Global Net Sales Distribution by Sales Mode (SALES_CHANNEL_L2)</b>",
        hole=0.65,
        value_is_millions=True
    )
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(
        g[['SALES_CHANNEL_L2', 'NET_SALES']],
        numeric_cols=['NET_SALES'],
        index_col='SALES_CHANNEL_L2',
        total_label='TOTAL'
    )

def sales_by_shift(df):
    st.header("Global Net Sales Distribution by SHIFT")
    if 'SHIFT' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SHIFT or NET_SALES")
        return
    g = df.groupby('SHIFT', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
    g['PCT'] = 100 * g['NET_SALES'] / g['NET_SALES'].sum()
    labels = [f"{row['SHIFT']} ({row['PCT']:.1f}%)" for _, row in g.iterrows()]
    fig = go.Figure(data=[go.Pie(labels=labels, values=g['NET_SALES'], hole=0.65)])
    fig.update_layout(title="<b>Global Net Sales Distribution by SHIFT</b>")
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(
        g[['SHIFT', 'NET_SALES', 'PCT']],
        numeric_cols=['NET_SALES', 'PCT'],
        index_col='SHIFT',
        total_label='TOTAL'
    )

# ...  (❗ all the remaining SALES, OPERATIONS and INSIGHTS functions from your
#      big script go here UNCHANGED: night_vs_day_ratio, global_day_vs_night,
#      second_highest_channel_share, bottom_30_2nd_highest, stores_sales_summary,
#      customer_traffic_storewise, active_tills_during_day, avg_customers_per_till,
#      store_customer_traffic_storewise, customer_traffic_departmentwise,
#      cashiers_performance, till_usage, tax_compliance, customer_baskets_overview,
#      global_category_overview_sales, global_category_overview_baskets,
#      supplier_contribution, category_overview, branch_comparison, product_performance,
#      global_loyalty_overview, branch_loyalty_overview, customer_loyalty_overview,
#      global_pricing_overview, branch_pricing_overview, global_refunds_overview,
#      branch_refunds_overview.)
#
# I’m not re-pasting them here again due to message length,
# but you can paste them directly from your existing working script
# under this comment – they do not need any change.


# ============================================================
#  MAIN APP
# ============================================================

def main():
    st.title("DailyDeck: The Story Behind the Numbers (Supabase Edition)")

    # 1) Load & clean data
    raw_df = smart_load()
    if raw_df is None or raw_df.empty:
        st.stop()

    with st.spinner("Preparing data (cached) ..."):
        df = clean_and_derive(raw_df)

    section = st.sidebar.selectbox(
        "Section",
        ["SALES", "OPERATIONS", "INSIGHTS"]
    )

    # ----------------------- SALES -----------------------
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

        show_trends(df, section)

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

    # ----------------------- OPERATIONS -----------------------
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

        show_trends(df, section)

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

    # ----------------------- INSIGHTS -----------------------
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

        show_trends(df, section)

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
