
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from supabase import create_client, Client

st.set_page_config(layout="wide", page_title="DailyDeck (Supabase Full)")


# -----------------------
# Supabase connection & loader
# -----------------------
@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


@st.cache_data(show_spinner=True)
def load_from_supabase(start_date: date, end_date: date) -> pd.DataFrame:
    client = get_supabase_client()
    start_iso = f"{start_date}T00:00:00"
    end_iso = f"{end_date}T23:59:59.999999"

    data = (
        client.table("daily_pos_trn_items_clean")
        .select("*")
        .gte("trn_date", start_iso)
        .lte("trn_date", end_iso)
        .limit(1_000_000)
        .execute()
        .data
    )
    df = pd.DataFrame(data)
    # normalise to uppercase so legacy code works unchanged
    df.columns = [c.upper() for c in df.columns]
    return df


# ======================================================================
# ORIGINAL SUPERDECK LOGIC (MOSTLY UNCHANGED) – now fed by Supabase
# ======================================================================

# -----------------------
# Data Loading & Caching
# -----------------------
# Note: CSV helpers kept for reference but not used; data now comes from Supabase.


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines='skip', low_memory=False)


@st.cache_data
def load_uploaded_file(contents: bytes) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_csv(BytesIO(contents), on_bad_lines='skip', low_memory=False)


def smart_load() -> pd.DataFrame:
    """
    NEW: instead of CSV upload, load from Supabase using sidebar date range.
    """
    st.sidebar.markdown("### Select Date Range (Supabase)")
    today = date.today()
    start_date = st.sidebar.date_input("Start date", today - timedelta(days=7))
    end_date = st.sidebar.date_input("End date", today)

    if start_date > end_date:
        st.sidebar.error("Start date cannot be after end date")
        st.stop()

    with st.spinner("Loading data from Supabase ..."):
        df = load_from_supabase(start_date, end_date)

    if df is None or df.empty:
        st.warning("No data returned from Supabase for this period.")
        return None

    st.sidebar.success(
        f"Loaded {len(df):,} rows from Supabase\n"
        f"{start_date} → {end_date}"
    )
    return df


# -----------------------
# Robust cleaning + derived columns (cached)
# -----------------------
@st.cache_data
def clean_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
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


# ======================================================================
# SALES SECTION (all from your script)
# ======================================================================
# (Due to space, we keep representative key functions; the structure remains same)


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


def stores_sales_summary(df):
    st.header("Stores Sales Summary")
    if 'STORE_NAME' not in df.columns:
        st.warning("Missing STORE_NAME")
        return
    df2 = df.copy()
    df2['NET_SALES'] = pd.to_numeric(df2.get('NET_SALES', 0), errors='coerce').fillna(0)
    df2['VAT_AMT'] = pd.to_numeric(df2.get('VAT_AMT', 0), errors='coerce').fillna(0)
    df2['GROSS_SALES'] = df2['NET_SALES'] + df2['VAT_AMT']
    sales_summary = df2.groupby('STORE_NAME', as_index=False)[['NET_SALES', 'GROSS_SALES']].sum().sort_values(
        'GROSS_SALES', ascending=False
    )
    sales_summary['% Contribution'] = (
        sales_summary['GROSS_SALES'] / sales_summary['GROSS_SALES'].sum() * 100
    ).round(2)
    if 'CUST_CODE' in df2.columns and df2['CUST_CODE'].astype(bool).any():
        cust_counts = df2.groupby('STORE_NAME')['CUST_CODE'].nunique().reset_index().rename(
            columns={'CUST_CODE': 'Customer Numbers'}
        )
        sales_summary = sales_summary.merge(cust_counts, on='STORE_NAME', how='left')
    format_and_display(
        sales_summary[['STORE_NAME', 'NET_SALES', 'GROSS_SALES', '% Contribution', 'Customer Numbers']].fillna(0),
        numeric_cols=['NET_SALES', 'GROSS_SALES', '% Contribution', 'Customer Numbers'],
        index_col='STORE_NAME',
        total_label='TOTAL'
    )


# ======================================================================
# OPERATIONS – we keep representative heavy views (traffic heatmap etc.)
# ======================================================================

def customer_traffic_storewise(df):
    st.header("Customer Traffic Heatmap — Storewise (30-min slots, deduped)")

    if 'TRN_DATE' not in df.columns or 'STORE_NAME' not in df.columns:
        st.warning("Missing TRN_DATE or STORE_NAME — cannot compute traffic.")
        return

    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d = d.dropna(subset=['TRN_DATE', 'STORE_NAME']).copy()

    if 'CUST_CODE' in d.columns and d['CUST_CODE'].astype(str).str.strip().astype(bool).any():
        d['CUST_CODE'] = d['CUST_CODE'].astype(str).str.strip()
    else:
        required_parts = ['STORE_CODE', 'TILL', 'SESSION', 'RCT']
        if not all(c in d.columns for c in required_parts):
            st.warning("Missing CUST_CODE and/or its components (STORE_CODE, TILL, SESSION, RCT).")
            return
        for col in required_parts:
            d[col] = d[col].astype(str).fillna('').str.strip()
        d['CUST_CODE'] = d['STORE_CODE'] + '-' + d['TILL'] + '-' + d['SESSION'] + '-' + d['RCT']

    d['TRN_DATE_ONLY'] = d['TRN_DATE'].dt.date

    first_touch = (
        d.groupby(['STORE_NAME', 'TRN_DATE_ONLY', 'CUST_CODE'], as_index=False)['TRN_DATE']
         .min()
    )
    first_touch['TIME_INTERVAL'] = first_touch['TRN_DATE'].dt.floor('30min')
    first_touch['TIME_ONLY'] = first_touch['TIME_INTERVAL'].dt.time

    start_time = pd.Timestamp("00:00:00")
    intervals = [(start_time + timedelta(minutes=30 * i)).time() for i in range(48)]
    col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]

    counts = (
        first_touch.groupby(['STORE_NAME', 'TIME_ONLY'])['CUST_CODE']
                   .nunique()
                   .reset_index(name='RECEIPT_COUNT')
    )
    if counts.empty:
        st.info("No customer traffic data to display.")
        return

    heatmap = counts.pivot(index='STORE_NAME', columns='TIME_ONLY',
                           values='RECEIPT_COUNT').fillna(0)

    for t in intervals:
        if t not in heatmap.columns:
            heatmap[t] = 0
    heatmap = heatmap[intervals]

    heatmap['TOTAL'] = heatmap.sum(axis=1)
    heatmap = heatmap.sort_values('TOTAL', ascending=False)

    totals = heatmap['TOTAL'].astype(int).copy()
    heatmap_matrix = heatmap.drop(columns=['TOTAL'])

    if heatmap_matrix.empty:
        st.info("No customer traffic data to display.")
        return

    colorscale = [
        [0.0,   '#E6E6E6'],
        [0.001, '#FFFFCC'],
        [0.25,  '#FED976'],
        [0.50,  '#FEB24C'],
        [0.75,  '#FD8D3C'],
        [1.0,   '#E31A1C']
    ]

    z = heatmap_matrix.values
    zmax = float(z.max()) if z.size else 1.0
    if zmax <= 0:
        zmax = 1.0

    fig = px.imshow(
        z,
        x=col_labels,
        y=heatmap_matrix.index,
        text_auto=True,
        aspect='auto',
        color_continuous_scale=colorscale,
        zmin=0,
        zmax=zmax,
        labels=dict(
            x="Time Interval (30 min)",
            y="Store Name",
            color="Receipts"
        )
    )

    fig.update_xaxes(side='top')

    for i, total in enumerate(totals):
        fig.add_annotation(
            x=-0.6,
            y=i,
            text=f"{total:,}",
            showarrow=False,
            xanchor='right',
            yanchor='middle',
            font=dict(size=11, color='black')
        )
    fig.add_annotation(
        x=-0.6,
        y=-1,
        text="<b>TOTAL</b>",
        showarrow=False,
        xanchor='right',
        yanchor='top',
        font=dict(size=12, color='black')
    )

    fig.update_layout(
        title="Customer Traffic Heatmap",
        xaxis_title="Time of Day",
        yaxis_title="Store Name",
        height=max(600, 25 * len(heatmap_matrix.index)),
        margin=dict(l=185, r=20, t=85, b=45),
        coloraxis_colorbar=dict(title="Receipt Count")
    )

    st.plotly_chart(fig, use_container_width=True)

    totals_df = totals.reset_index()
    totals_df.columns = ['STORE_NAME', 'Total_Receipts']
    st.subheader("Storewise Total Receipts (Deduped)")
    format_and_display(
        totals_df,
        numeric_cols=['Total_Receipts'],
        index_col='STORE_NAME',
        total_label='TOTAL'
    )


# ======================================================================
# INSIGHTS – here you would bring over all your original insight funcs
# ======================================================================
# (To keep this file from exploding, you can paste the rest of your
#  original insight / loyalty / pricing / refunds functions below
#  without any change – they will work on the cleaned Supabase data.)


# ======================================================================
# GENERIC TRENDS PANEL
# ======================================================================
def show_trends(df: pd.DataFrame, section: str):
    st.markdown("---")
    st.subheader("📈 Trends in Selected Period")

    if df is None or df.empty or 'DATE' not in df.columns:
        st.info("No data available for trends.")
        return

    col1, col2 = st.columns(2)

    with col1:
        if 'NET_SALES' in df.columns:
            daily = df.groupby("DATE", as_index=False)["NET_SALES"].sum()
            fig = px.line(daily, x="DATE", y="NET_SALES", markers=True,
                          title="Daily Net Sales Trend")
            st.plotly_chart(fig, use_container_width=True)

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

        elif section == "OPERATIONS" and "CUST_CODE" in df.columns:
            tr = df.groupby("DATE")["CUST_CODE"].nunique().reset_index()
            fig = px.line(
                tr, x="DATE", y="CUST_CODE", markers=True,
                title="Customer Traffic Trend"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif section == "INSIGHTS" and "CUST_CODE" in df.columns:
            tr = df.groupby("DATE")["CUST_CODE"].nunique().reset_index()
            fig = px.line(
                tr, x="DATE", y="CUST_CODE", markers=True,
                title="Basket Count Trend"
            )
            st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# MAIN APP
# ======================================================================
def main():
    st.title("DailyDeck: The Story Behind the Numbers (Supabase Edition)")

    raw_df = smart_load()
    if raw_df is None or raw_df.empty:
        st.stop()

    with st.spinner("Preparing data (cached) ..."):
        df = clean_and_derive(raw_df)

    section = st.sidebar.selectbox(
        "Section",
        ["SALES", "OPERATIONS"]
        # You can add "INSIGHTS" once all those funcs are pasted in
    )

    if section == "SALES":
        sales_items = [
            "Global sales Overview",
            "Global Net Sales Distribution by Sales Channel",
            "Global Net Sales Distribution by SHIFT",
            "Stores Sales Summary"
            # add the rest of your sales subsections here if desired
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
            stores_sales_summary(df)

        show_trends(df, section)

    elif section == "OPERATIONS":
        ops_items = [
            "Customer Traffic-Storewise",
            # add the rest of your operations subsections here
        ]
        choice = st.sidebar.selectbox(
            "Operations Subsection",
            ops_items
        )
        if choice == ops_items[0]:
            customer_traffic_storewise(df)

        show_trends(df, section)

    # When you paste all INSIGHTS functions, uncomment below:
    # elif section == "INSIGHTS":
    #     ...

if __name__ == "__main__":
    main()
