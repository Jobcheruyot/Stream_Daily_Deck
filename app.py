import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from supabase import create_client

st.set_page_config(layout="wide", page_title="Superdeck")

TABLE_NAME = "daily_pos_trn_items_clean"


# ---------------------------------------------------------
# COLUMN HARMONIZER
# ---------------------------------------------------------
def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize dataframe column names to UPPERCASE with stripped whitespace."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    return df


# ---------------------------------------------------------
# CONNECT TO SUPABASE
# ---------------------------------------------------------
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except:
        st.error("Supabase credentials missing in Streamlit Secrets.")
        st.stop()

    return create_client(url, key)


# ---------------------------------------------------------
# LOAD FROM SUPABASE (LOWERCASE-FRIENDLY)
# ---------------------------------------------------------
def load_supabase_data(date_basis, start_date, end_date):
    client = init_supabase()

    # force lowercase because Supabase stores columns in lowercase
    date_basis_lower = date_basis.lower()

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Query Supabase
    try:
        res = (
            client.table(TABLE_NAME)
            .select("*")
            .gte(date_basis_lower, start_str)
            .lte(date_basis_lower, end_str)
            .execute()
        )
    except Exception as e:
        st.error(f"Supabase query failed: {e}")
        return pd.DataFrame()

    data = res.data or []
    df = pd.DataFrame(data)

    if df.empty:
        return df

    # Harmonize column names so the rest of the app can safely use UPPERCASE
    df = harmonize_columns(df)

    # Convert key date columns to datetime if present
    for col in ["TRN_DATE", "ZED_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
def sidebar_config():
    date_basis = st.sidebar.radio(
        "Use which date?",
        ["TRN_DATE", "ZED_DATE"],
        index=0
    )
    today = datetime.today().date()
    start_date = st.sidebar.date_input("Start date", today - timedelta(days=7))
    end_date = st.sidebar.date_input("End date", today)

    section = st.sidebar.selectbox(
        "Section", ["SALES", "OPERATIONS", "INSIGHTS"]
    )

    return date_basis, start_date, end_date, section


# ---------------------------------------------------------
# CLEAN DATAFRAME
# ---------------------------------------------------------
def clean_df(df, date_basis):
    if df.empty:
        return df

    df = df.copy()

    # derived fields
    df["DATE"] = df[date_basis].dt.date
    df["HOUR"] = df[date_basis].dt.hour
    df["HALF_HOUR"] = df[date_basis].dt.floor("30min")

    # numeric cleanup
    for col in ["QTY", "NET_SALES", "SP_PRE_VAT", "CP_PRE_VAT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # receipt key
    for c in ["STORE_CODE", "TILL", "SESSION", "RCT"]:
        if c not in df.columns:
            df[c] = ""
    df["CUST_CODE"] = (
        df["STORE_CODE"] + "-" + df["TILL"] + "-" + df["SESSION"] + "-" + df["RCT"]
    )

    return df


# ---------------------------------------------------------
# UTILITY FUNCTIONS (agg, donut, format)
# ---------------------------------------------------------
def agg_net_sales_by(df, group_col):
    df2 = df.copy()
    df2["NET_SALES"] = pd.to_numeric(df2.get("NET_SALES", 0), errors="coerce").fillna(0)
    g = df2.groupby(group_col, as_index=False)["NET_SALES"].sum()
    g = g.sort_values("NET_SALES", ascending=False)
    return g


def donut_from_agg(g, label_col, value_col, title, hole=0.4, value_is_millions=False):
    import plotly.graph_objects as go

    values = g[value_col].astype(float)
    if value_is_millions:
        values = values / 1_000_000.0
    labels = g[label_col].astype(str)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=hole,
                textinfo="label+percent",
                hovertemplate="%{label}<br>%{value:.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(title=title)
    return fig


def format_and_display(df, numeric_cols=None, index_col=None, total_label="TOTAL"):
    if df is None or df.empty:
        st.info("No data to display.")
        return

    d = df.copy()
    if numeric_cols:
        for col in numeric_cols:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0)

    if index_col and index_col in d.columns:
        d = d.set_index(index_col)

    if numeric_cols:
        totals = {}
        for col in d.columns:
            if col in numeric_cols:
                totals[col] = d[col].sum()
            else:
                totals[col] = ""
        total_row = pd.DataFrame(totals, index=[total_label])
        d = pd.concat([d, total_row])

    st.dataframe(d.style.format(thousands=",", precision=2))


# -----------------------
# SALES
# -----------------------
import plotly.graph_objects as go


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


def night_vs_day_ratio(df):
    st.header("Night vs Day Shift Sales Ratio — Stores with Night Shifts")
    if 'Shift_Bucket' not in df.columns or 'STORE_NAME' not in df.columns:
        st.warning("Missing Shift_Bucket or STORE_NAME")
        return
    stores_with_night = df[df['Shift_Bucket'] == 'Night']['STORE_NAME'].unique()
    df_nd = df[df['STORE_NAME'].isin(stores_with_night)].copy()
    ratio = df_nd.groupby(['STORE_NAME', 'Shift_Bucket'])['NET_SALES'].sum().reset_index()
    ratio['STORE_TOTAL'] = ratio.groupby('STORE_NAME')['NET_SALES'].transform('sum')
    ratio['PCT'] = 100 * ratio['NET_SALES'] / ratio['STORE_TOTAL']
    pivot = ratio.pivot(index='STORE_NAME', columns='Shift_Bucket', values='PCT').fillna(0)
    if pivot.empty:
        st.info("No stores with NIGHT shift found")
        return
    pivot_sorted = pivot.sort_values('Night', ascending=False)
    numbered_labels = [f"{i+1}. {s}" for i, s in enumerate(pivot_sorted.index)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pivot_sorted['Night'],
        y=numbered_labels,
        orientation='h',
        name='Night',
        marker_color='#d62728',
        text=[f"{v:.1f}%" for v in pivot_sorted['Night']],
        textposition='inside'
    ))
    for i, (n_val, d_val) in enumerate(zip(pivot_sorted['Night'], pivot_sorted['Day'])):
        fig.add_annotation(
            x=n_val + 1,
            y=numbered_labels[i],
            text=f"{d_val:.1f}% Day",
            showarrow=False,
            xanchor='left'
        )
    fig.update_layout(
        title="Night vs Day Shift Sales Ratio — Stores with Night Shifts",
        xaxis_title="% of Store Sales",
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)
    table = pivot_sorted.reset_index().rename(columns={'Night': 'Night_%', 'Day': 'Day_%'})
    format_and_display(
        table,
        numeric_cols=['Night_%', 'Day_%'],
        index_col='STORE_NAME',
        total_label='TOTAL'
    )


def global_day_vs_night(df):
    st.header("Global Day vs Night Sales — Only Stores with NIGHT Shifts")
    if 'Shift_Bucket' not in df.columns:
        st.warning("Missing Shift_Bucket")
        return
    stores_with_night = df[df['Shift_Bucket'] == 'Night']['STORE_NAME'].unique()
    df_nd = df[df['STORE_NAME'].isin(stores_with_night)]
    if df_nd.empty:
        st.info("No stores with night shifts")
        return
    agg = df_nd.groupby('Shift_Bucket', as_index=False)['NET_SALES'].sum()
    agg['PCT'] = 100 * agg['NET_SALES'] / agg['NET_SALES'].sum()
    labels = [f"{r.Shift_Bucket} ({r.PCT:.1f}%)" for _, r in agg.iterrows()]
    fig = go.Figure(go.Pie(labels=labels, values=agg['NET_SALES'], hole=0.65))
    fig.update_layout(title="<b>Global Day vs Night Sales — Only Stores with NIGHT Shifts</b>")
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(
        agg,
        numeric_cols=['NET_SALES', 'PCT'],
        index_col='Shift_Bucket',
        total_label='TOTAL'
    )


def second_highest_channel_share(df):
    st.header("2nd-Highest Channel Share")
    if not all(col in df.columns for col in ['STORE_NAME', 'SALES_CHANNEL_L1', 'NET_SALES']):
        st.warning("Missing columns required")
        return
    data = df.copy()
    store_chan = data.groupby(['STORE_NAME', 'SALES_CHANNEL_L1'], as_index=False)['NET_SALES'].sum()
    store_tot = store_chan.groupby('STORE_NAME')['NET_SALES'].transform('sum')
    store_chan['PCT'] = 100 * store_chan['NET_SALES'] / store_tot
    store_chan = store_chan.sort_values(['STORE_NAME', 'PCT'], ascending=[True, False])
    store_chan['RANK'] = store_chan.groupby('STORE_NAME').cumcount() + 1
    second = store_chan[store_chan['RANK'] == 2][['STORE_NAME', 'SALES_CHANNEL_L1', 'PCT']].rename(
        columns={'SALES_CHANNEL_L1': 'SECOND_CHANNEL', 'PCT': 'SECOND_PCT'}
    )
    all_stores = store_chan['STORE_NAME'].drop_duplicates()
    missing_stores = set(all_stores) - set(second['STORE_NAME'])
    if missing_stores:
        add = pd.DataFrame({
            'STORE_NAME': list(missing_stores),
            'SECOND_CHANNEL': ['(None)'] * len(missing_stores),
            'SECOND_PCT': [0.0] * len(missing_stores)
        })
        second = pd.concat([second, add], ignore_index=True)
    second_sorted = second.sort_values('SECOND_PCT', ascending=False)
    top_n = st.sidebar.slider("Top N", min_value=10, max_value=100, value=30)
    top_ = second_sorted.head(top_n).copy()
    if top_.empty:
        st.info("No stores to display")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_['SECOND_PCT'],
        y=top_['STORE_NAME'],
        orientation='h',
        marker_color='#9aa0a6',
        name='Stem',
        hoverinfo='none',
        text=[f"{p:.1f}%" for p in top_['SECOND_PCT']],
        textposition='outside'
    ))
    fig.add_trace(go.Scatter(
        x=top_['SECOND_PCT'],
        y=top_['STORE_NAME'],
        mode='markers',
        marker=dict(color='#1f77b4', size=10),
        name='2nd Channel %',
        hovertemplate='%{x:.1f}%<extra></extra>'
    ))
    annotations = []
    for _, row in top_.iterrows():
        annotations.append(dict(
            x=row['SECOND_PCT'] + 1,
            y=row['STORE_NAME'],
            text=f"{row['SECOND_CHANNEL']}",
            showarrow=False,
            xanchor='left',
            font=dict(size=10)
        ))
    fig.update_layout(
        title=f"Top {top_n} Stores by 2nd-Highest Channel Share (SALES_CHANNEL_L1)",
        xaxis_title="2nd-Highest Channel Share (% of Store NET_SALES)",
        height=max(500, 24 * len(top_)),
        annotations=annotations,
        yaxis=dict(autorange='reversed')
    )
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(
        second_sorted[['STORE_NAME', 'SECOND_CHANNEL', 'SECOND_PCT']],
        numeric_cols=['SECOND_PCT'],
        index_col='STORE_NAME',
        total_label='TOTAL'
    )


def bottom_30_2nd_highest(df):
    st.header("Bottom 30 — 2nd Highest Channel")
    if not all(col in df.columns for col in ['STORE_NAME', 'SALES_CHANNEL_L1', 'NET_SALES']):
        st.warning("Missing required columns")
        return
    data = df.copy()
    store_chan = data.groupby(['STORE_NAME', 'SALES_CHANNEL_L1'], as_index=False)['NET_SALES'].sum()
    store_tot = store_chan.groupby('STORE_NAME')['NET_SALES'].transform('sum')
    store_chan['PCT'] = 100 * store_chan['NET_SALES'] / store_tot
    store_chan = store_chan.sort_values(['STORE_NAME', 'PCT'], ascending=[True, False])
    store_chan['RANK'] = store_chan.groupby('STORE_NAME').cumcount() + 1
    top_tbl = store_chan[store_chan['RANK'] == 1][['STORE_NAME', 'SALES_CHANNEL_L1', 'PCT']].rename(
        columns={'SALES_CHANNEL_L1': 'TOP_CHANNEL', 'PCT': 'TOP_PCT'}
    )
    second_tbl = store_chan[store_chan['RANK'] == 2][['STORE_NAME', 'SALES_CHANNEL_L1', 'PCT']].rename(
        columns={'SALES_CHANNEL_L1': 'SECOND_CHANNEL', 'PCT': 'SECOND_PCT'}
    )
    ranking = pd.merge(top_tbl, second_tbl, on='STORE_NAME', how='left').fillna(
        {'SECOND_CHANNEL': '(None)', 'SECOND_PCT': 0}
    )
    bottom_30 = ranking.sort_values('SECOND_PCT', ascending=True).head(30)
    if bottom_30.empty:
        st.info("No stores to display")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bottom_30['SECOND_PCT'],
        y=bottom_30['STORE_NAME'],
        orientation='h',
        marker_color='#9aa0a6',
        name='Stem',
        text=[f"{v:.1f}%" for v in bottom_30['SECOND_PCT']],
        textposition='outside'
    ))
    fig.add_trace(go.Scatter(
        x=bottom_30['SECOND_PCT'],
        y=bottom_30['STORE_NAME'],
        mode='markers',
        marker=dict(color='#1f77b4', size=10),
        name='2nd Channel %'
    ))
    annotations = []
    for _, row in bottom_30.iterrows():
        annotations.append(dict(
            x=row['SECOND_PCT'] + 1,
            y=row['STORE_NAME'],
            text=f"{row['SECOND_CHANNEL']}",
            showarrow=False,
            xanchor='left',
            font=dict(size=10)
        ))
    fig.update_layout(
        title="Bottom 30 Stores by 2nd-Highest Channel Share (SALES_CHANNEL_L1)",
        xaxis_title="2nd-Highest Channel Share (% of Store NET_SALES)",
        height=max(500, 24 * len(bottom_30)),
        annotations=annotations,
        yaxis=dict(autorange='reversed')
    )
    st.plotly_chart(fig, use_container_width=True)
    format_and_display(
        bottom_30,
        numeric_cols=['SECOND_PCT', 'TOP_PCT'],
        index_col='STORE_NAME',
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


# -----------------------
# OPERATIONS
# -----------------------
def customer_traffic_storewise(df):
    st.header("Customer Traffic Heatmap — Storewise (30-min slots, deduped)")

    if 'TRN_DATE' not in df.columns or 'STORE_NAME' not in df.columns:
        st.warning("Missing TRN_DATE or STORE_NAME — cannot compute traffic.")
        return

    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d = d.dropna(subset=['TRN_DATE', 'STORE_NAME']).copy()

    # Build/ensure CUST_CODE
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

    first_touch['HALF_HOUR'] = first_touch['TRN_DATE'].dt.floor('30min')

    heat = (
        first_touch.groupby(['STORE_NAME', 'HALF_HOUR'])['CUST_CODE']
        .nunique()
        .reset_index(name='CUSTOMERS')
    )

    if heat.empty:
        st.info("No customer traffic to display.")
        return

    pivot = heat.pivot_table(
        index='STORE_NAME',
        columns='HALF_HOUR',
        values='CUSTOMERS',
        fill_value=0
    )

    st.dataframe(pivot.style.format(thousands=","))


# (… your remaining OPERATIONS & INSIGHTS functions stay unchanged …)
# I’ve left them intact from your previous file.


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    st.title("Superdeck")

    # Sidebar configuration
    date_basis, start_date, end_date, section = sidebar_config()

    # Load data
    with st.spinner("Loading data from Supabase..."):
        raw_df = load_supabase_data(date_basis, start_date, end_date)

    if raw_df is None or raw_df.empty:
        st.warning("No data returned from Supabase for the selected period.")
        return

    df = clean_df(raw_df, date_basis)

    # Route by section (SALES / OPERATIONS / INSIGHTS)
    if section == "SALES":
        sales_global_overview(df)
        sales_by_channel_l2(df)
        sales_by_shift(df)
        night_vs_day_ratio(df)
        global_day_vs_night(df)
        second_highest_channel_share(df)
        bottom_30_2nd_highest(df)
        stores_sales_summary(df)

    elif section == "OPERATIONS":
        customer_traffic_storewise(df)
        # … other OPERATIONS subsections …

    elif section == "INSIGHTS":
        # … INSIGHTS subsections …
        pass


if __name__ == "__main__":
    main()
