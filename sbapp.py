# app.py

import os
from datetime import timedelta, date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from supabase import create_client, Client

# ------------------------------------------------
# Streamlit page config
# ------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Superdeck (Streamlit)",
)


# =================================================
#  Supabase helpers
# =================================================
@st.cache_data(show_spinner=False)
def get_supabase_client() -> Client:
    """
    Create a Supabase client using either Streamlit secrets
    or environment variables.
    """
    url = None
    key = None

    # Prefer st.secrets if available
    try:
        url = st.secrets.get("SUPABASE_URL", None)
        key = st.secrets.get("SUPABASE_KEY", None)
    except Exception:
        pass

    # Fallback to environment variables
    url = url or os.getenv("SUPABASE_URL")
    key = key or os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError(
            "Supabase credentials not found. "
            "Set SUPABASE_URL and SUPABASE_KEY in Streamlit secrets or environment variables."
        )

    return create_client(url, key)


@st.cache_data(show_spinner=True)
def load_from_supabase(
    start_date: date,
    end_date: date,
    table_name: str = "daily_pos_trn_items_clean",
) -> pd.DataFrame:
    """
    Pull rows from Supabase for the selected date period.
    Assumes TRN_DATE exists in the table.
    """
    client = get_supabase_client()

    # Convert to ISO strings and make end_date inclusive
    start_iso = f"{start_date}T00:00:00"
    end_iso = f"{end_date}T23:59:59.999999"

    query = (
        client.table(table_name)
        .select("*")
        .gte("TRN_DATE", start_iso)
        .lte("TRN_DATE", end_iso)
        .limit(1_000_000)  # adjust if you expect more rows
    )

    res = query.execute()
    data = res.data or []
    df = pd.DataFrame(data)

    return df


# =================================================
#  CSV helpers (fallback / debugging)
# =================================================
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines='skip', low_memory=False)


@st.cache_data
def load_uploaded_file(contents: bytes) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_csv(BytesIO(contents), on_bad_lines='skip', low_memory=False)


def smart_load() -> pd.DataFrame | None:
    """
    Sidebar widget:
      - primary: Supabase with date range
      - secondary: CSV upload (as before)
    """
    st.sidebar.markdown("### Data source")

    source = st.sidebar.radio(
        "Choose data source",
        ["Supabase DB (recommended)", "Upload CSV file"],
        index=0,
    )

    if source == "Upload CSV file":
        uploaded = st.sidebar.file_uploader(
            "Upload DAILY_POS_TRN_ITEMS CSV",
            type=['csv']
        )
        if uploaded is not None:
            with st.spinner("Parsing uploaded CSV..."):
                df = load_uploaded_file(uploaded.getvalue())
            st.sidebar.success("Loaded uploaded CSV")
            return df

        # Optional fallback default path
        default_path = "/content/DAILY_POS_TRN_ITEMS_2025-10-21.csv"
        try:
            with st.spinner(f"Loading default CSV: {default_path}"):
                df = load_csv(default_path)
            st.sidebar.info(f"Loaded default path: {default_path}")
            return df
        except Exception:
            st.sidebar.warning("No default CSV found. Please upload a CSV.")
            return None

    # -------- Supabase path ----------
    st.sidebar.markdown("#### Date range for Supabase")

    today = date.today()
    default_start = today - timedelta(days=6)

    start_date = st.sidebar.date_input(
        "Start date", value=default_start
    )
    end_date = st.sidebar.date_input(
        "End date", value=today
    )

    if isinstance(start_date, list):  # just in case user picks a range widget
        start_date = start_date[0]
    if isinstance(end_date, list):
        end_date = end_date[-1]

    if start_date > end_date:
        st.sidebar.error("Start date must be on or before end date.")
        return None

    st.sidebar.caption(
        f"Pulling data from Supabase between **{start_date}** and **{end_date}**."
    )

    try:
        with st.spinner("Loading data from Supabase..."):
            df = load_from_supabase(start_date, end_date)
        if df.empty:
            st.sidebar.warning("No rows found for selected period.")
            return None
        st.sidebar.success(f"Loaded {len(df):,} rows from Supabase")
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading from Supabase: {e}")
        return None


# =================================================
#  Cleaning & derived columns (from your previous app)
# =================================================
@st.cache_data
def clean_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df.copy()

    # Normalize string columns
    str_cols = [
        'STORE_CODE', 'TILL', 'SESSION', 'RCT', 'STORE_NAME', 'CASHIER',
        'ITEM_CODE', 'ITEM_NAME', 'DEPARTMENT', 'CATEGORY', 'CU_DEVICE_SERIAL',
        'CAP_CUSTOMER_CODE', 'LOYALTY_CUSTOMER_CODE', 'SUPPLIER_NAME',
        'SALES_CHANNEL_L1', 'SALES_CHANNEL_L2', 'SHIFT'
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
    numeric_cols = [
        'QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT',
        'NET_SALES', 'VAT_AMT'
    ]
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
    if all(col in d.columns for col in ['STORE_CODE', 'TILL', 'SESSION', 'RCT']):
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


# =================================================
#  Small cached aggregation helpers
# =================================================
@st.cache_data
def agg_net_sales_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col, 'NET_SALES'])
    g = df.groupby(col, as_index=False)['NET_SALES'].sum().sort_values(
        'NET_SALES', ascending=False
    )
    return g


@st.cache_data
def agg_count_distinct(df: pd.DataFrame, group_by: list, agg_col: str, agg_name: str) -> pd.DataFrame:
    g = (
        df.groupby(group_by)
        .agg({agg_col: pd.Series.nunique})
        .reset_index()
        .rename(columns={agg_col: agg_name})
    )
    return g


# =================================================
#  Table formatting helper
# =================================================
def format_and_display(
    df: pd.DataFrame,
    numeric_cols: list | None = None,
    index_col: str | None = None,
    total_label: str = 'TOTAL'
):
    if df is None or df.empty:
        st.dataframe(df)
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
            is_int_like = (
                len(series_vals) > 0 and
                np.allclose(series_vals.fillna(0).round(0), series_vals.fillna(0))
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


# =================================================
#  Plotting helpers (your original functions)
# =================================================
def donut_from_agg(df_agg, label_col, value_col, title,
                   hole=0.55, colors=None,
                   legend_title=None, value_is_millions=False):
    labels = df_agg[label_col].astype(str).tolist()
    vals = df_agg[value_col].astype(float).tolist()
    if not vals:
        st.info("No data for chart.")
        return go.Figure()

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


# =================================================
#  SALES / OPERATIONS / INSIGHTS FUNCTIONS
#  (all copied from your previous script, unchanged)
#  NOTE: For brevity here, they are identical to the
#  functions you already had (sales_global_overview,
#  sales_by_channel_l2, sales_by_shift, night_vs_day_ratio,
#  global_day_vs_night, second_highest_channel_share,
#  bottom_30_2nd_highest, stores_sales_summary,
#  customer_traffic_storewise, active_tills_during_day,
#  avg_customers_per_till, store_customer_traffic_storewise,
#  customer_traffic_departmentwise, cashiers_performance,
#  till_usage, tax_compliance, customer_baskets_overview,
#  global_category_overview_sales, global_category_overview_baskets,
#  supplier_contribution, category_overview, branch_comparison,
#  product_performance, global_loyalty_overview,
#  branch_loyalty_overview, customer_loyalty_overview,
#  global_pricing_overview, branch_pricing_overview,
#  global_refunds_overview, branch_refunds_overview).
#
#  👉 Paste those function definitions here exactly as in your
#  previous file – they work unchanged on multi-day data.
#  (They were omitted in this snippet purely to save space.)
# =================================================

# --- PASTE ALL YOUR ANALYTIC FUNCTIONS HERE UNCHANGED ---
# (Everything from `def sales_global_overview(df):` down to
#  `def branch_refunds_overview(df):` from your old script.)
# ---------------------------------------------------------
#  🔴 IMPORTANT: Do NOT change those functions; they will
#  automatically respect the date range because df is already
#  filtered by TRN_DATE.
# ---------------------------------------------------------


# =================================================
#  NEW: Trend panel shown under every subsection
# =================================================
def show_trends(df: pd.DataFrame, section_name: str):
    """
    Generic trends panel that appears under every subsection.

    - Left chart: Daily NET_SALES (or QTY) for the selected period
    - Right chart: Section-specific metric
    """
    if df is None or df.empty or 'TRN_DATE' not in df.columns:
        return

    st.markdown("---")
    st.subheader("📈 Trends in selected period")

    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d = d.dropna(subset=['TRN_DATE'])

    if d.empty:
        st.info("No valid TRN_DATE rows for trends.")
        return

    d['DATE'] = d['TRN_DATE'].dt.date

    col1, col2 = st.columns(2)

    # 1) Daily value trend
    with col1:
        if 'NET_SALES' in d.columns:
            daily_sales = d.groupby('DATE', as_index=False)['NET_SALES'].sum()
            fig = px.line(
                daily_sales,
                x='DATE',
                y='NET_SALES',
                markers=True,
                title="Daily Net Sales"
            )
            st.plotly_chart(fig, use_container_width=True)
        elif 'QTY' in d.columns:
            daily_qty = d.groupby('DATE', as_index=False)['QTY'].sum()
            fig = px.line(
                daily_qty,
                x='DATE',
                y='QTY',
                markers=True,
                title="Daily Quantity"
            )
            st.plotly_chart(fig, use_container_width=True)

    # 2) Section-specific trend
    with col2:
        if section_name == "SALES" and \
           'SALES_CHANNEL_L1' in d.columns and 'NET_SALES' in d.columns:
            ch_trend = (
                d.groupby(['DATE', 'SALES_CHANNEL_L1'], as_index=False)['NET_SALES']
                .sum()
            )
            fig2 = px.line(
                ch_trend,
                x='DATE',
                y='NET_SALES',
                color='SALES_CHANNEL_L1',
                markers=True,
                title="Net Sales by Sales Channel (L1)"
            )
            st.plotly_chart(fig2, use_container_width=True)

        elif section_name == "OPERATIONS" and 'CUST_CODE' in d.columns:
            traffic = d.groupby('DATE', as_index=False)['CUST_CODE'].nunique()
            fig2 = px.line(
                traffic,
                x='DATE',
                y='CUST_CODE',
                markers=True,
                title="Unique Receipts (Customer Traffic) per Day"
            )
            st.plotly_chart(fig2, use_container_width=True)

        elif section_name == "INSIGHTS" and \
             'CUST_CODE' in d.columns and 'NET_SALES' in d.columns:
            baskets = d.groupby('DATE', as_index=False)['CUST_CODE'].nunique()
            fig2 = px.line(
                baskets,
                x='DATE',
                y='CUST_CODE',
                markers=True,
                title="Basket Count per Day"
            )
            st.plotly_chart(fig2, use_container_width=True)


# =================================================
#  MAIN APP
# =================================================
def main():
    st.title("DailyDeck: The Story Behind the Numbers")

    raw_df = smart_load()
    if raw_df is None or raw_df.empty:
        st.stop()

    with st.spinner("Preparing data (cached) ..."):
        df = clean_and_derive(raw_df)

    if df is None or df.empty:
        st.error("No data after cleaning. Check your source or date range.")
        st.stop()

    section = st.sidebar.selectbox(
        "Section",
        ["SALES", "OPERATIONS", "INSIGHTS"]
    )

    # -----------------------------------------------
    # SALES
    # -----------------------------------------------
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
        choice = st.sidebar.selectbox("Sales Subsection", sales_items)

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

        # New trends panel for SALES
        show_trends(df, "SALES")

    # -----------------------------------------------
    # OPERATIONS
    # -----------------------------------------------
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
        choice = st.sidebar.selectbox("Operations Subsection", ops_items)

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

        # New trends panel for OPERATIONS
        show_trends(df, "OPERATIONS")

    # -----------------------------------------------
    # INSIGHTS
    # -----------------------------------------------
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
        choice = st.sidebar.selectbox("Insights Subsection", ins_items)

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
            ins_items[13]: branch_refunds_overview,
        }

        func = mapping.get(choice)
        if func:
            func(df)
        else:
            st.write("Not implemented yet")

        # New trends panel for INSIGHTS
        show_trends(df, "INSIGHTS")


if __name__ == "__main__":
    main()
