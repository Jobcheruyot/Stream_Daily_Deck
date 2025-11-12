import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(layout="wide", page_title="Superdeck (Streamlit)")
# Add the following at the very top of your Streamlit script, AFTER st.set_page_config

# ========= SAFE UI SHELL (No sidebar, no JS, no internal imports) =========
import streamlit as st
import pandas as pd
import numpy as np
from contextlib import suppress

# 0) Minimal heartbeat to prevent blank page even if CSS fails
st.write("")  # keep — forces something to render

# 1) Hide chrome + layout styles
st.markdown("""
<style>
#MainMenu, header, footer {visibility:hidden;}
[data-testid="stSidebar"] {display:none;}
.block-container {padding-top:0; padding-bottom:0; max-width:1600px;}
.app-viewport{min-height:100vh; display:grid; grid-template-rows:auto 1fr auto;}
.topbar{display:flex; justify-content:center; align-items:center; padding:1.2rem 1.5rem;
        border-bottom:1px solid #eef1f6; background:#fff; position:relative;}
.brand{text-align:center;}
.brand h1{margin:0; font-size:1.9rem; color:#111827;}
.brand p{margin:.2rem 0 0; font-size:1rem; color:#d72638; font-weight:600;}
.top-right{position:absolute; right:1.2rem; top:1rem;}
.btn{display:inline-block; padding:.55rem 1rem; border-radius:10px; text-decoration:none;
     font-weight:700; border:2px solid transparent; transition:.15s; cursor:pointer}
.btn-primary{background:#d72638; color:#fff;}
.btn-outline{background:#fff; border-color:#16a34a; color:#16a34a;}
.btn:hover{transform:translateY(-1px); filter:brightness(.98)}
.sidebar-green{ background:#e6f6ea; border:1px solid #c5e4cf; border-radius:12px; padding:1rem; }
.hero{ background:#f8fafc; border:1px solid #e5e7eb; border-radius:12px; padding:1.2rem; }
.kpi{ background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:1rem 1.2rem; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.05);}
.kpi-label{ font-size:.9rem; color:#6b7280;}
.kpi-value{ font-size:1.5rem; font-weight:700; color:#111827;}
.err{ background:#fff3cd; color:#8a6d3b; border:1px solid #ffeeba; padding:.6rem .8rem; border-radius:8px; margin:.5rem 0; font-size:.9rem;}
.foot{ text-align:left; padding:.4rem 1rem; font-size:.8rem; background:#111827; color:#fff;}
</style>
""", unsafe_allow_html=True)

# 2) State
if "ui_section" not in st.session_state: st.session_state.ui_section = None  # "sales"|"ops"|"insights"
if "ui_sub" not in st.session_state: st.session_state.ui_sub = None
def go(section=None, sub=None):
    if section is not None: st.session_state.ui_section = section
    if sub is not None: st.session_state.ui_sub = sub

# 3) Very safe KPI helper (works with empty/None df)
def _safe_kpis(df: pd.DataFrame | None):
    if df is None or len(df.index)==0:
        return 0.0, 0.0, 0.0, 0
    d = df.copy()
    for c in ["NET_SALES","QTY"]:
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    if "CUST_CODE" not in d.columns and all(x in d.columns for x in ["STORE_CODE","TILL","SESSION","RCT"]):
        d["CUST_CODE"] = (d["STORE_CODE"].astype(str)+"-"+d["TILL"].astype(str)+"-"+
                          d["SESSION"].astype(str)+"-"+d["RCT"].astype(str))
    net = float(d.get("NET_SALES", pd.Series(dtype=float)).sum())
    baskets = float(d.get("CUST_CODE", pd.Series(dtype=str)).nunique() or 0)
    with suppress(Exception):
        items_per_rcpt = d.groupby("CUST_CODE")["ITEM_CODE"].nunique().mean()
    if 'items_per_rcpt' not in locals() or pd.isna(items_per_rcpt): items_per_rcpt = 0.0
    if "Till_Code" in d.columns:
        tills = int(d["Till_Code"].nunique())
    elif all(x in d.columns for x in ["TILL","STORE_CODE"]):
        tills = int((d["TILL"].astype(str)+"-"+d["STORE_CODE"].astype(str)).nunique())
    else:
        tills = 0
    return net/1_000_000, baskets/1_000, float(items_per_rcpt), tills

# 4) Load your df if possible — but NEVER crash if helpers are missing
df, warn = None, None
try:
    if 'smart_load' in globals():
        df = smart_load()
    if df is not None and 'clean_and_derive' in globals():
        df = clean_and_derive(df)
except Exception as e:
    warn = f"{type(e).__name__}: {e}"
net_m, baskets_k, avg_items, tills = _safe_kpis(df)

# 5) Top bar (always renders)
st.markdown('<div class="app-viewport">', unsafe_allow_html=True)
st.markdown("""
<div class="topbar">
  <div class="brand">
    <h1>Quick Mart Limited</h1>
    <p>Let the data Speak</p>
  </div>
  <div class="top-right"><a class="btn btn-outline" href="#" onclick="return false;">Home</a></div>
</div>
""", unsafe_allow_html=True)

# 6) Body layout — weights (1,5) to avoid collapse
left, right = st.columns([1,5], gap="large")

with left:
    st.markdown('<div class="sidebar-green"><h3>Sections</h3>', unsafe_allow_html=True)
    st.button("Sales Analytics", use_container_width=True, on_click=go, kwargs=dict(section="sales", sub=None))
    st.button("Operations", use_container_width=True, on_click=go, kwargs=dict(section="ops", sub=None))
    st.button("Insights", use_container_width=True, on_click=go, kwargs=dict(section="insights", sub=None))
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f'<div class="kpi"><div class="kpi-label">Net Sales</div><div class="kpi-value">KSh {net_m:,.1f}M</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi"><div class="kpi-label">Baskets</div><div class="kpi-value">{baskets_k:,.1f}K</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi"><div class="kpi-label">Avg Items/Receipt</div><div class="kpi-value">{avg_items:,.1f}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi"><div class="kpi-label">Active Tills</div><div class="kpi-value">{tills:,}</div></div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    if c1.button("Sales Analytics ▶", use_container_width=True): go("sales", None)
    if c2.button("Operations ▶", use_container_width=True): go("ops", None)
    if c3.button("Insights ▶", use_container_width=True): go("insights", None)

    # Bottom uploader (centered across row)
    u1,u2 = st.columns([4,1])
    uploaded = u1.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed", key="hero_csv_safe")
    if u2.button("Upload CSV", use_container_width=True) and uploaded is not None:
        try:
            if 'load_uploaded_file' in globals():
                df = load_uploaded_file(uploaded.getvalue())
            else:
                from io import BytesIO
                df = pd.read_csv(BytesIO(uploaded.getvalue()), on_bad_lines='skip', low_memory=False)
            if 'clean_and_derive' in globals():
                df = clean_and_derive(df)
            net_m, baskets_k, avg_items, tills = _safe_kpis(df)
            st.success("CSV uploaded.")
        except Exception as e:
            warn = f"{type(e).__name__}: {e}"

    if warn:
        st.markdown(f'<div class="err">Startup warning: {warn}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="foot">© 2025 Quick Mart Limited</div></div>', unsafe_allow_html=True)

# 7) Router — call your existing functions AFTER hero is visible
def render_page(section: str | None, sub: str | None, df: pd.DataFrame | None):
    if df is None:
        return
    try:
        if section == "sales":
            if sub is None or sub == "global_overview": sales_global_overview(df)
            elif sub == "channel_distribution":        sales_by_channel_l2(df)
            elif sub == "store_summary":                stores_sales_summary(df)
        elif section == "ops":
            if sub is None or sub == "cashiers_performance": cashiers_performance(df)
            elif sub == "till_usage":                       till_usage(df)
            elif sub == "store_traffic":                    customer_traffic_storewise(df)
        elif section == "insights":
            if sub is None or sub == "customer_baskets":    customer_baskets_overview(df)
            elif sub == "category_performance":             global_category_overview_sales(df)
            elif sub == "supplier_contribution":            global_category_overview_baskets(df)
    except Exception as e:
        st.markdown(f'<div class="err">Render error: {type(e).__name__}: {e}</div>', unsafe_allow_html=True)

render_page(st.session_state.ui_section, st.session_state.ui_sub, df)
# ================= END SAFE SHELL =================


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
    uploaded = st.sidebar.file_uploader("Upload DAILY_POS_TRN_ITEMS CSV", type=['csv'])
    if uploaded is not None:
        with st.spinner("Parsing uploaded CSV..."):
            df = load_uploaded_file(uploaded.getvalue())
        st.sidebar.success("Loaded uploaded CSV")
        return df

    # try default path (optional)
    default_path = "/content/DAILY_POS_TRN_ITEMS_2025-10-21.csv"
    try:
        with st.spinner(f"Loading default CSV: {default_path}"):
            df = load_csv(default_path)
        st.sidebar.info(f"Loaded default path: {default_path}")
        return df
    except Exception:
        st.sidebar.warning("No default CSV found. Please upload a CSV to run the app.")
        return None

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
# SALES
# -----------------------
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
    first_touch['TIME_INTERVAL'] = first_touch['TRN_DATE'].dt.floor('30min')
    first_touch['TIME_ONLY'] = first_touch['TIME_INTERVAL'].dt.time

    # 30-min grid
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

    # totals annotation
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

def active_tills_during_day(df):
    st.header("Peak Active Tills")

    required = ['TRN_DATE', 'STORE_NAME', 'TILL', 'STORE_CODE']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns for Active Tills view: {missing}")
        return

    d = df.copy()

    # Ensure strings
    d['TILL'] = d['TILL'].astype(str).fillna('').str.strip()
    d['STORE_CODE'] = d['STORE_CODE'].astype(str).fillna('').str.strip()
    if 'Till_Code' not in d.columns:
        d['Till_Code'] = d['TILL'] + '-' + d['STORE_CODE']

    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d = d.dropna(subset=['TRN_DATE'])

    d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30min')
    d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time

    # Time grid
    start_time = pd.Timestamp("00:00:00")
    intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
    col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]

    # Count unique tills per interval
    till_counts = (
        d.groupby(['STORE_NAME','TIME_ONLY'])['Till_Code']
          .nunique()
          .reset_index(name='UNIQUE_TILLS')
    )

    if till_counts.empty:
        st.info("No till activity data to display.")
        return

    heatmap = till_counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='UNIQUE_TILLS').fillna(0)

    for t in intervals:
        if t not in heatmap.columns:
            heatmap[t] = 0
    heatmap = heatmap[intervals]

    # Max active tills per store
    heatmap['MAX_TILLS'] = heatmap.max(axis=1).astype(int)
    heatmap = heatmap.sort_values('MAX_TILLS', ascending=False)

    max_vals = heatmap['MAX_TILLS'].copy()
    mat = heatmap.drop(columns=['MAX_TILLS']).values

    colorscale = [
        [0.0,  '#E6E6E6'],
        [0.001,'#FFFFCC'],
        [0.25, '#FED976'],
        [0.50, '#FEB24C'],
        [0.75, '#FD8D3C'],
        [1.0,  '#E31A1C']
    ]
    zmax = float(mat.max()) if mat.size else 1.0
    if zmax <= 0:
        zmax = 1.0

    fig = px.imshow(
        mat,
        x=col_labels,
        y=heatmap.index,
        text_auto=True,
        aspect='auto',
        color_continuous_scale=colorscale,
        zmin=0, zmax=zmax,
        labels=dict(x="Time Interval (30 min)", y="Store Name", color="Unique Tills")
    )

    fig.update_xaxes(side='top')

    # Max labels
    for i, max_till in enumerate(max_vals):
        fig.add_annotation(
            x=-0.6, y=i,
            text=f"{int(max_till):,}",
            showarrow=False,
            xanchor='right',
            yanchor='middle',
            font=dict(size=11, color='black')
        )

    fig.add_annotation(
        x=-0.6, y=-1,
        text="<b>MAX</b>",
        showarrow=False,
        xanchor='right',
        yanchor='top',
        font=dict(size=12, color='black')
    )

    fig.update_layout(
        title="Peak Active Tills",
        xaxis_title="Time of Day",
        yaxis_title="Store Name",
        height=max(600, 25 * len(heatmap.index)),
        margin=dict(l=180, r=20, t=60, b=40),
        coloraxis_colorbar=dict(title="Unique Tills")
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    summary = max_vals.reset_index()
    summary.columns = ['STORE_NAME', 'MAX_ACTIVE_TILLS']
    st.subheader("Peak Active Tills per Store")
    format_and_display(
        summary,
        numeric_cols=['MAX_ACTIVE_TILLS'],
        index_col='STORE_NAME',
        total_label='TOTAL'
    )

def avg_customers_per_till(df):
    st.header("Average Customers Served per Till (30-min slots)")

    if 'TRN_DATE' not in df.columns:
        st.warning("Missing TRN_DATE")
        return

    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d = d.dropna(subset=['TRN_DATE'])

    # --- 1) Build time intervals ---
    start_time = pd.Timestamp("00:00:00")
    intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
    col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]

    # --- 2) Customer counts (earliest receipt per till) ---
    for c in ['STORE_CODE','TILL','SESSION','RCT']:
        if c not in d.columns:
            st.warning("Missing required columns for computing CUST_CODE")
            return
        d[c] = d[c].astype(str).fillna('').str.strip()

    d['CUST_CODE'] = d['STORE_CODE'] + '-' + d['TILL'] + '-' + d['SESSION'] + '-' + d['RCT']
    d['TRN_DATE_ONLY'] = d['TRN_DATE'].dt.date

    first_touch = (
        d.groupby(['STORE_NAME','TRN_DATE_ONLY','CUST_CODE'], as_index=False)['TRN_DATE'].min()
    )
    first_touch['TIME_INTERVAL'] = first_touch['TRN_DATE'].dt.floor('30min')
    first_touch['TIME_ONLY'] = first_touch['TIME_INTERVAL'].dt.time

    customer_counts = (
        first_touch.groupby(['STORE_NAME','TIME_ONLY'])['CUST_CODE']
        .nunique().reset_index(name='CUSTOMERS')
    )
    cust_pivot = customer_counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='CUSTOMERS').fillna(0)
    for t in intervals:
        if t not in cust_pivot.columns:
            cust_pivot[t] = 0
    cust_pivot = cust_pivot[intervals]

    # --- 3) Till counts ---
    d['TILL'] = d['TILL'].astype(str).fillna('').str.strip()
    d['STORE_CODE'] = d['STORE_CODE'].astype(str).fillna('').str.strip()
    d['Till_Code'] = d['TILL'] + '-' + d['STORE_CODE']
    d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30min')
    d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time

    till_counts = (
        d.groupby(['STORE_NAME','TIME_ONLY'])['Till_Code']
        .nunique().reset_index(name='TILLS')
    )
    till_pivot = till_counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='TILLS').fillna(0)
    for t in intervals:
        if t not in till_pivot.columns:
            till_pivot[t] = 0
    till_pivot = till_pivot[intervals]

    # --- 4) Calculate and round up Customers per Till ---
    ratio_matrix = cust_pivot / till_pivot.replace(0, np.nan)
    ratio_matrix = np.ceil(ratio_matrix).fillna(0).astype(int)

    if ratio_matrix.empty:
        st.info("No data")
        return

    ratio_matrix['MAX_RATIO'] = ratio_matrix.max(axis=1)
    ratio_matrix = ratio_matrix.sort_values('MAX_RATIO', ascending=False)
    max_vals = ratio_matrix['MAX_RATIO']
    ratio_data = ratio_matrix.drop(columns=['MAX_RATIO']).values

    # --- 5) Plot the heatmap ---
    colorscale = [
        [0.0,  '#E6E6E6'],   # 0 = gray
        [0.001,'#e0f3db'],
        [0.25, '#a8ddb5'],
        [0.50, '#43a2ca'],
        [0.75, '#0868ac'],
        [1.0,  '#084081']
    ]
    zmax = float(ratio_data.max()) if ratio_data.size else 1.0
    if zmax <= 0:
        zmax = 1.0

    fig = px.imshow(
        ratio_data,
        x=col_labels,
        y=ratio_matrix.index,
        text_auto=True,
        aspect='auto',
        color_continuous_scale=colorscale,
        zmin=0,
        zmax=zmax,
        labels=dict(x="Time Interval (30 min)", y="Store Name", color="Customers per Till")
    )

    fig.update_xaxes(side='top')

    # Add max labels on the left
    for i, val in enumerate(max_vals):
        fig.add_annotation(
            x=-0.6, y=i,
            text=f"{val}",
            showarrow=False,
            xanchor='right', yanchor='middle',
            font=dict(size=11, color='black')
        )

    # Header for max column
    fig.add_annotation(
        x=-0.6, y=-1,
        text="<b>MAX</b>",
        showarrow=False,
        xanchor='right', yanchor='top',
        font=dict(size=12, color='black')
    )

    fig.update_layout(
        title="Customers Served per Till ",
        xaxis_title="Time of Day",
        yaxis_title="Store Name",
        height=max(600, 25 * len(ratio_matrix.index)),
        margin=dict(l=190, r=30, t=60, b=60),
        coloraxis_colorbar=dict(title="Customers / Till")
    )

    st.plotly_chart(fig, use_container_width=True)

    pivot_totals = pd.DataFrame({
        'STORE_NAME': ratio_matrix.index,
        'MAX_CUSTOMERS_PER_TILL': max_vals.astype(int)
    })
    format_and_display(
        pivot_totals,
        numeric_cols=['MAX_CUSTOMERS_PER_TILL'],
        index_col='STORE_NAME',
        total_label='TOTAL'
    )

def store_customer_traffic_storewise(df):
    st.header("Store Customer Traffic (per Department)")

    if 'TRN_DATE' not in df.columns or 'STORE_NAME' not in df.columns:
        st.warning("Missing TRN_DATE or STORE_NAME")
        return

    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d = d.dropna(subset=['TRN_DATE']).copy()

    # Ensure CUST_CODE
    for col in ['STORE_CODE','TILL','SESSION','RCT']:
        if col in d.columns:
            d[col] = d[col].astype(str).fillna('').str.strip()
    if 'CUST_CODE' not in d.columns:
        if not all(c in d.columns for c in ['STORE_CODE','TILL','SESSION','RCT']):
            st.warning("Missing columns to build CUST_CODE")
            return
        d['CUST_CODE'] = d['STORE_CODE'] + '-' + d['TILL'] + '-' + d['SESSION'] + '-' + d['RCT']

    d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30min')
    d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time

    # Time grid
    start_time = pd.Timestamp("00:00:00")
    intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
    col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]

    def build_branch_pivot(branch_name: str):
        branch_df = d[d['STORE_NAME'] == branch_name]
        tmp = (
            branch_df.groupby(['DEPARTMENT','TIME_ONLY'])['CUST_CODE']
            .nunique()
            .reset_index(name='Unique_Customers')
        )

        pivot = tmp.pivot(index='DEPARTMENT', columns='TIME_ONLY', values='Unique_Customers').fillna(0)
        for t in intervals:
            if t not in pivot.columns:
                pivot[t] = 0
        pivot = pivot[intervals]

        pivot['TOTAL'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('TOTAL', ascending=False)
        totals = pivot['TOTAL'].astype(int).tolist()
        y_labels = pivot.index.tolist()
        mat = pivot.drop(columns=['TOTAL']).values

        # total customers (unique CUST_CODE) in that store
        total_customers = branch_df['CUST_CODE'].nunique()

        return mat, y_labels, totals, total_customers

    branches = sorted(d['STORE_NAME'].dropna().unique().tolist())
    if not branches:
        st.info("No branches found")
        return

    branch_data = {}
    global_zmax = 1
    for b in branches:
        mat, y_labels, totals, total_customers = build_branch_pivot(b)
        branch_data[b] = {'z': mat, 'y': y_labels, 'totals': totals, 'total_customers': total_customers}
        if mat.size:
            global_zmax = max(global_zmax, int(np.max(mat)))

    init_branch = branches[0]
    z0 = branch_data[init_branch]['z']
    y0 = branch_data[init_branch]['y']
    totals0 = branch_data[init_branch]['totals']
    total_customers0 = branch_data[init_branch]['total_customers']

    colorscale = [
        [0.0,  '#E6E6E6'],
        [0.001,'#e0f3db'],
        [0.25, '#a8ddb5'],
        [0.50, '#43a2ca'],
        [0.75, '#0868ac'],
        [1.0,  '#084081']
    ]

    fig = px.imshow(
        z0,
        x=col_labels,
        y=y0,
        text_auto=True,
        aspect='auto',
        color_continuous_scale=colorscale,
        zmin=0,
        zmax=global_zmax,
        labels=dict(x="Time of Day", y="Department", color="Unique Customers")
    )

    fig.update_xaxes(side='top')

    def make_total_annotations(totals, y_labels):
        ann = []
        for i, total in enumerate(totals):
            ann.append(dict(
                x=-0.6, y=i, text=f"{int(total):,}",
                showarrow=False, xanchor='right', yanchor='middle',
                font=dict(size=11, color='black')
            ))
        ann.append(dict(
            x=-0.6, y=-1, text="<b>TOTAL</b>",
            showarrow=False, xanchor='right', yanchor='top',
            font=dict(size=12, color='black')
        ))
        return ann

    fig.update_layout(
        title=f"🕒 Customer Traffic Patterns — {init_branch} | Total Customers: {total_customers0:,}",
        xaxis_title="Time of Day",
        yaxis_title="Department",
        height=max(600, 25 * len(y0)),
        margin=dict(l=180, r=20, t=60, b=40),
        coloraxis_colorbar=dict(title="Customers"),
        annotations=make_total_annotations(totals0, y0)
    )

    buttons = []
    for b in branches:
        z = branch_data[b]['z']
        y = branch_data[b]['y']
        totals = branch_data[b]['totals']
        total_customers = branch_data[b]['total_customers']
        new_height = max(600, 25 * len(y))

        buttons.append(dict(
            label=b,
            method='update',
            args=[
                {'z': [z], 'y': [y], 'x': [col_labels]},
                {
                    'title': f"🕒 Customer Traffic Patterns — {b} | Total Customers: {total_customers:,}",
                    'annotations': make_total_annotations(totals, y),
                    'height': new_height
                }
            ]
        ))

    fig.update_layout(
        updatemenus=[dict(
            type='dropdown',
            x=0, xanchor='left',
            y=1.12, yanchor='top',
            buttons=buttons,
            direction='down',
            showactive=True
        )]
    )

    st.plotly_chart(fig, use_container_width=True)

def customer_traffic_departmentwise(df):
    # Same engine as Store Customer Traffic Storewise
    store_customer_traffic_storewise(df)

def cashiers_performance(df: pd.DataFrame):
    import plotly.express as px
    import numpy as np
    import pandas as pd
    import streamlit as st

    st.header("Cashiers Perfomance")

    # ===== 1) Prepare Data =====
    if 'TRN_DATE' not in df.columns:
        st.warning("Missing TRN_DATE")
        return

    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d = d.dropna(subset=['TRN_DATE']).copy()

    # Ensure identifiers exist (and are strings)
    required_id_cols = ['STORE_CODE', 'TILL', 'SESSION', 'RCT', 'CASHIER', 'ITEM_CODE']
    missing = [c for c in required_id_cols if c not in d.columns]
    if missing:
        st.warning(f"Missing column(s) in dataset: {missing}")
        return
    for c in required_id_cols + ['STORE_NAME']:
        if c in d.columns:
            d[c] = d[c].astype(str).fillna('').str.strip()

    # Build CUST_CODE if not present
    if 'CUST_CODE' not in d.columns:
        d['CUST_CODE'] = d['STORE_CODE'] + '-' + d['TILL'] + '-' + d['SESSION'] + '-' + d['RCT']
    else:
        d['CUST_CODE'] = d['CUST_CODE'].astype(str).fillna('').str.strip()

    # Create unique cashier per store (match your naming)
    if 'CASHIER-COUNT' not in d.columns:
        d['CASHIER-COUNT'] = d['STORE_NAME'] + '-' + d['CASHIER']

    # ===== 2) Receipt-level duration (min) and item count =====
    receipt_duration = (
        d.groupby(['STORE_NAME', 'CUST_CODE'], as_index=False)
         .agg(Start_Time=('TRN_DATE', 'min'),
              End_Time=('TRN_DATE', 'max'))
    )
    receipt_duration['Duration_Sec'] = (
        receipt_duration['End_Time'] - receipt_duration['Start_Time']
    ).dt.total_seconds().fillna(0)

    receipt_items = (
        d.groupby(['STORE_NAME', 'CUST_CODE'], as_index=False)['ITEM_CODE']
         .nunique()
         .rename(columns={'ITEM_CODE': 'Unique_Items'})
    )

    receipt_stats = pd.merge(
        receipt_duration, receipt_items,
        on=['STORE_NAME', 'CUST_CODE'], how='left'
    )

    # ===== 3) Store-level summary (table at the end) =====
    store_summary = (
        receipt_stats.groupby('STORE_NAME', as_index=False)
        .agg(
            Total_Customers=('CUST_CODE', 'nunique'),
            Avg_Time_per_Customer_Min=('Duration_Sec', lambda s: s.mean() / 60),
            Avg_Items_per_Receipt=('Unique_Items', 'mean')
        )
    )
    store_summary['Avg_Time_per_Customer_Min'] = store_summary['Avg_Time_per_Customer_Min'].round(1)
    store_summary['Avg_Items_per_Receipt'] = store_summary['Avg_Items_per_Receipt'].round(1)
    store_summary = store_summary.sort_values('Avg_Time_per_Customer_Min', ascending=True).reset_index(drop=True)
    store_summary.index = np.arange(1, len(store_summary) + 1)
    store_summary.index.name = '#'
    store_summary['Total_Customers'] = store_summary['Total_Customers'].map('{:,.0f}'.format)

    # ===== 4) Cashier summary (duration + customer count + avg items/receipt) =====
    merged_for_duration = d.merge(
        receipt_stats[['STORE_NAME', 'CUST_CODE', 'Duration_Sec']],
        on=['STORE_NAME', 'CUST_CODE'], how='left'
    )
    cashier_durations = (
        merged_for_duration
        .groupby(['STORE_NAME', 'CASHIER-COUNT'], as_index=False)
        .agg(
            Avg_Duration_Sec=('Duration_Sec', 'mean'),
            Customers_Served=('CUST_CODE', 'nunique')
        )
    )
    cashier_durations['Avg_Serve_Min'] = (cashier_durations['Avg_Duration_Sec'] / 60.0).round(1)

    # --- Avg items per receipt per cashier ---
    _receipt_cashier = d[['STORE_NAME', 'CUST_CODE', 'CASHIER-COUNT']].drop_duplicates()
    _rc_items = _receipt_cashier.merge(receipt_items, on=['STORE_NAME', 'CUST_CODE'], how='left')
    _avg_items_cashier = (
        _rc_items.groupby(['STORE_NAME', 'CASHIER-COUNT'], as_index=False)['Unique_Items']
                .mean()
                .rename(columns={'Unique_Items': 'Avg_Items_per_Receipt_Cashier'})
    )
    cashier_durations = cashier_durations.merge(
        _avg_items_cashier, on=['STORE_NAME', 'CASHIER-COUNT'], how='left'
    )
    cashier_durations['Avg_Items_per_Receipt_Cashier'] = (
        cashier_durations['Avg_Items_per_Receipt_Cashier'].round(1)
    )

    # ===== 5) Dropdown chart per branch (matches your intended look) =====
    branches = sorted(store_summary['STORE_NAME'].unique().tolist())
    if not branches:
        st.info("No branches found.")
        return

    # Initial branch
    init_branch = branches[0]
    branch_data = {
        b: cashier_durations[cashier_durations['STORE_NAME'] == b].sort_values('Avg_Serve_Min')
        for b in branches
    }
    df_branch = branch_data[init_branch].copy()
    df_branch['Label_Text'] = (
        df_branch['Avg_Serve_Min'].astype(str) + ' min (' +
        df_branch['Customers_Served'].astype(str) + ' customers) — ' +
        df_branch['Avg_Items_per_Receipt_Cashier'].fillna(0).map('{:.1f}'.format) + ' items'
    )

    fig = px.bar(
        df_branch,
        x='Avg_Serve_Min',
        y='CASHIER-COUNT',
        orientation='h',
        text='Label_Text',
        color='Avg_Serve_Min',
        color_continuous_scale='Blues',
        title=f"🕒 Avg Serving Time per Cashier — {init_branch}",
        labels={'Avg_Serve_Min': 'Avg Time per Customer (min)', 'CASHIER-COUNT': 'Cashier'}
    )
    fig.update_traces(textposition='outside', textfont=dict(size=10))
    fig.update_layout(
        xaxis_title="Average Serving Time (minutes)",
        yaxis_title="Cashier",
        coloraxis_showscale=False,
        height=max(500, 25 * len(df_branch))
    )

    # Dropdown to switch branches
    buttons = []
    for b in branches:
        dfb = branch_data[b].copy()
        dfb['Label_Text'] = (
            dfb['Avg_Serve_Min'].astype(str) + ' min (' +
            dfb['Customers_Served'].astype(str) + ' customers) — ' +
            dfb['Avg_Items_per_Receipt_Cashier'].fillna(0).map('{:.1f}'.format) + ' items'
        )
        buttons.append(dict(
            label=b,
            method='update',
            args=[{
                'x': [dfb['Avg_Serve_Min']],
                'y': [dfb['CASHIER-COUNT']],
                'text': [dfb['Label_Text']],
                'marker': {'color': dfb['Avg_Serve_Min'], 'colorscale': 'Blues'}
            }, {
                'title': f"🕒 Avg Serving Time per Cashier — {b}",
                'height': max(500, 25 * len(dfb))
            }]
        ))

    fig.update_layout(
        updatemenus=[dict(
            type='dropdown',
            x=0, xanchor='left',
            y=1.15, yanchor='top',
            buttons=buttons,
            showactive=True
        )]
    )
    st.plotly_chart(fig, use_container_width=True)

    # ===== 7) Display Final Table =====
    try:
        # If you have the helper in your app
        format_and_display(
            store_summary.reset_index(),
            numeric_cols=['Total_Customers', 'Avg_Time_per_Customer_Min', 'Avg_Items_per_Receipt'],
            index_col=None,
            total_label='TOTAL'
        )
    except Exception:
        st.dataframe(store_summary, use_container_width=True)


def till_usage(df):
    st.header("Till Usage")
    d = df.copy()
    if 'Till_Code' not in d.columns:
        if not all(c in d.columns for c in ['TILL','STORE_CODE']):
            st.warning("Missing TILL/STORE_CODE for Till_Code")
            return
        d['Till_Code'] = d['TILL'].astype(str) + '-' + d['STORE_CODE'].astype(str)
    branches = sorted(d['STORE_NAME'].unique())
    if not branches:
        st.info("No branches")
        return
    branch = st.selectbox("Select Branch for Till Usage", branches)
    till_activity = d.groupby(['STORE_NAME', 'Till_Code', 'TIME_ONLY'], as_index=False).agg(
        Receipts=('CUST_CODE', 'nunique')
    )
    dfb = till_activity[till_activity['STORE_NAME'] == branch]
    if dfb.empty:
        st.info("No till activity")
        return
    pivot = dfb.pivot(index='Till_Code', columns='TIME_ONLY', values='Receipts').fillna(0)
    x = [t.strftime('%H:%M') for t in sorted(pivot.columns)]
    fig = px.imshow(
        pivot.values,
        x=x,
        y=pivot.index,
        labels=dict(x="Time of Day (30-min slot)", y="Till", color="Receipts"),
        text_auto=True
    )
    fig.update_xaxes(side='top')
    st.plotly_chart(fig, use_container_width=True)
    totals = pivot.sum(axis=1).reset_index()
    totals.columns = ['Till_Code', 'Total_Receipts']
    format_and_display(
        totals,
        numeric_cols=['Total_Receipts'],
        index_col='Till_Code',
        total_label='TOTAL'
    )

def tax_compliance(df):
    st.header("Tax Compliance")
    if 'CU_DEVICE_SERIAL' not in df.columns or 'CUST_CODE' not in df.columns:
        st.warning("Missing CU_DEVICE_SERIAL or CUST_CODE")
        return
    d = df.copy()
    d['Tax_Compliant'] = np.where(
        d['CU_DEVICE_SERIAL'].replace(
            {'nan': '', 'NaN': '', 'None': ''}
        ).str.strip().str.len() > 0,
        'Compliant',
        'Non-Compliant'
    )
    if 'Till_Code' not in d.columns and all(c in d.columns for c in ['TILL','STORE_CODE']):
        d['Till_Code'] = d['TILL'].astype(str) + '-' + d['STORE_CODE'].astype(str)
    store_till = d.groupby(
        ['STORE_NAME', 'Till_Code', 'Tax_Compliant'],
        as_index=False
    ).agg(Receipts=('CUST_CODE', 'nunique'))
    branch = st.selectbox(
        "Select Branch for Tax Compliance",
        sorted(d['STORE_NAME'].unique())
    )
    dfb = store_till[store_till['STORE_NAME'] == branch]
    if dfb.empty:
        st.info("No compliance data for branch")
        return
    pivot = dfb.pivot(
        index='Till_Code',
        columns='Tax_Compliant',
        values='Receipts'
    ).fillna(0)
    pivot = pivot.reindex(columns=['Compliant', 'Non-Compliant'], fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=pivot.index,
        x=pivot['Compliant'],
        orientation='h',
        name='Compliant',
        marker_color='#2ca02c'
    ))
    fig.add_trace(go.Bar(
        y=pivot.index,
        x=pivot['Non-Compliant'],
        orientation='h',
        name='Non-Compliant',
        marker_color='#d62728',
        text=pivot['Non-Compliant'],
        textposition='outside'
    ))
    fig.update_layout(
        barmode='stack',
        title=f"Tax Compliance by Till — {branch}",
        height=max(400, 24 * len(pivot.index))
    )
    st.plotly_chart(fig, use_container_width=True)
    store_summary = d.groupby(
        ['STORE_NAME', 'Tax_Compliant'],
        as_index=False
    ).agg(Receipts=('CUST_CODE', 'nunique')).pivot(
        index='STORE_NAME',
        columns='Tax_Compliant',
        values='Receipts'
    ).fillna(0)
    store_summary['Total'] = store_summary.sum(axis=1)
    store_summary['Compliance_%'] = np.where(
        store_summary['Total'] > 0,
        (store_summary.get('Compliant', 0) / store_summary['Total'] * 100).round(1),
        0.0
    )
    format_and_display(
        store_summary.reset_index(),
        numeric_cols=['Compliant', 'Non-Compliant', 'Total', 'Compliance_%'],
        index_col='STORE_NAME',
        total_label='TOTAL'
    )

# -----------------------
# INSIGHTS
# -----------------------
def customer_baskets_overview(df):
    st.header("Customer Baskets Overview")
    d = df.copy()
    d = d.dropna(subset=['ITEM_NAME', 'CUST_CODE', 'STORE_NAME', 'DEPARTMENT'])
    d = d[~d['DEPARTMENT'].str.upper().eq('LUGGAGE & BAGS')]
    branches = sorted(d['STORE_NAME'].unique())
    branch = st.selectbox("Branch for comparison (branch vs global)", branches)
    metric = st.selectbox("Metric", ['QTY', 'NET_SALES'])
    top_x = st.number_input("Top X", min_value=5, max_value=200, value=10)
    departments = sorted(d['DEPARTMENT'].unique())
    selected_depts = st.multiselect(
        "Departments (empty = all)",
        options=departments,
        default=None
    )
    temp = d.copy()
    if selected_depts:
        temp = temp[temp['DEPARTMENT'].isin(selected_depts)]
    basket_count = temp.groupby('ITEM_NAME')['CUST_CODE'].nunique().rename(
        'Count_of_Baskets'
    )
    agg_data = temp.groupby('ITEM_NAME')[['QTY', 'NET_SALES']].sum()
    global_top = (
        basket_count.to_frame()
        .join(agg_data)
        .reset_index()
        .sort_values(metric, ascending=False)
        .head(int(top_x))
    )
    global_top.insert(0, '#', range(1, len(global_top) + 1))
    st.subheader("Global Top Items")
    format_and_display(
        global_top.reset_index(drop=True),
        numeric_cols=['Count_of_Baskets', 'QTY', 'NET_SALES'],
        index_col='ITEM_NAME',
        total_label='TOTAL'
    )
    branch_df = temp[temp['STORE_NAME'] == branch]
    if branch_df.empty:
        st.info("No data for selected branch")
        return
    basket_count_b = branch_df.groupby('ITEM_NAME')['CUST_CODE'].nunique().rename(
        'Count_of_Baskets'
    )
    agg_b = branch_df.groupby('ITEM_NAME')[['QTY', 'NET_SALES']].sum()
    branch_top = (
        basket_count_b.to_frame()
        .join(agg_b)
        .reset_index()
        .sort_values(metric, ascending=False)
        .head(int(top_x))
    )
    branch_top.insert(0, '#', range(1, len(branch_top) + 1))
    st.subheader(f"{branch} Top Items")
    format_and_display(
        branch_top.reset_index(drop=True),
        numeric_cols=['Count_of_Baskets', 'QTY', 'NET_SALES'],
        index_col='ITEM_NAME',
        total_label='TOTAL'
    )
    # --- Items in Global Top X but Missing/Underperforming in Y (Zero Sales Only) ---
    # Only show items where the selected branch has ZERO NET_SALES (or no record).
    if 'NET_SALES' in branch_df.columns:
        branch_item_sales = (
            branch_df.groupby('ITEM_NAME', as_index=False)['NET_SALES']
            .sum()
            .rename(columns={'NET_SALES': 'Branch_NET_SALES'})
        )

        comp = global_top.merge(branch_item_sales, on='ITEM_NAME', how='left')
        comp['Branch_NET_SALES'] = comp['Branch_NET_SALES'].fillna(0)

        zero_sales = comp[comp['Branch_NET_SALES'] == 0].copy()

        if not zero_sales.empty:
            st.subheader(
                f'Items in Global Top {top_x} but Missing/Underperforming in "{branch}" (Zero Sales Only)'
            )
            cols = ['#', 'ITEM_NAME', 'Count_of_Baskets', 'QTY', 'NET_SALES', 'Branch_NET_SALES']
            zero_sales = zero_sales[cols].reset_index(drop=True)
            format_and_display(
                zero_sales,
                numeric_cols=['Count_of_Baskets', 'QTY', 'NET_SALES', 'Branch_NET_SALES'],
                index_col='ITEM_NAME',
                total_label='TOTAL'
            )
        else:
            st.success(
                f'No items in Global Top {top_x} with zero sales in "{branch}".'
            )
    else:
        st.info("NET_SALES column missing — cannot compute zero-sales underperformance view.")

def global_category_overview_sales(df):
    st.header("Global Category Overview — Sales")
    if 'CATEGORY' not in df.columns:
        st.warning("Missing CATEGORY")
        return
    g = agg_net_sales_by(df, 'CATEGORY')
    format_and_display(
        g,
        numeric_cols=['NET_SALES'],
        index_col='CATEGORY',
        total_label='TOTAL'
    )
    fig = px.bar(
        g.head(20),
        x='NET_SALES',
        y='CATEGORY',
        orientation='h',
        title="Top Categories by Net Sales"
    )
    st.plotly_chart(fig, use_container_width=True)

def global_category_overview_baskets(df):
    st.header("Global Category Overview — Baskets")
    if 'CATEGORY' not in df.columns:
        st.warning("Missing CATEGORY")
        return
    g = df.groupby('CATEGORY', as_index=False)['CUST_CODE'].nunique().rename(
        columns={'CUST_CODE': 'Baskets'}
    ).sort_values('Baskets', ascending=False)
    format_and_display(
        g,
        numeric_cols=['Baskets'],
        index_col='CATEGORY',
        total_label='TOTAL'
    )
    fig = px.bar(
        g.head(20),
        x='Baskets',
        y='CATEGORY',
        orientation='h',
        title="Top Categories by Baskets"
    )
    st.plotly_chart(fig, use_container_width=True)

def supplier_contribution(df):
    st.header("Supplier Contribution (Top suppliers by net sales)")
    if 'SUPPLIER_NAME' not in df.columns:
        st.warning("Missing SUPPLIER_NAME")
        return
    g = df.groupby('SUPPLIER_NAME', as_index=False)['NET_SALES'].sum().sort_values(
        'NET_SALES', ascending=False
    ).head(50)
    format_and_display(
        g,
        numeric_cols=['NET_SALES'],
        index_col='SUPPLIER_NAME',
        total_label='TOTAL'
    )
    fig = px.bar(
        g,
        x='NET_SALES',
        y='SUPPLIER_NAME',
        orientation='h',
        title="Top Suppliers by Net Sales"
    )
    st.plotly_chart(fig, use_container_width=True)

def category_overview(df):
    st.header("Category Overview")
    if 'CATEGORY' not in df.columns:
        st.warning("Missing CATEGORY")
        return
    g = df.groupby('CATEGORY', as_index=False).agg(
        Baskets=('CUST_CODE', 'nunique'),
        Net_Sales=('NET_SALES', 'sum')
    ).sort_values('Net_Sales', ascending=False)
    format_and_display(
        g,
        numeric_cols=['Baskets', 'Net_Sales'],
        index_col='CATEGORY',
        total_label='TOTAL'
    )

def branch_comparison(df):
    st.header("Branch Comparison")
    branches = sorted(df['STORE_NAME'].unique())
    if len(branches) < 1:
        st.info("No branches")
        return
    col1, col2 = st.columns(2)
    with col1:
        a = st.selectbox("Branch A", branches, index=0)
    with col2:
        b = st.selectbox("Branch B", branches, index=1 if len(branches) > 1 else 0)
    metric = st.selectbox("Metric", ['QTY', 'NET_SALES'])
    top_x = st.number_input("Top X", min_value=5, max_value=200, value=10)

    def top_items(branch):
        temp = df[df['STORE_NAME'] == branch]
        baskets = temp.groupby('ITEM_NAME')['CUST_CODE'].nunique().rename(
            'Count_of_Baskets'
        )
        totals = temp.groupby('ITEM_NAME')[['QTY', 'NET_SALES']].sum()
        merged = (
            baskets.to_frame()
            .join(totals, how='outer')
            .fillna(0)
            .reset_index()
            .sort_values(metric, ascending=False)
            .head(int(top_x))
        )
        merged.insert(0, '#', range(1, len(merged) + 1))
        return merged

    topA = top_items(a)
    topB = top_items(b)
    combined = pd.concat(
        [topA.assign(Branch=a), topB.assign(Branch=b)],
        ignore_index=True
    )
    fig = px.bar(
        combined,
        x=metric,
        y='ITEM_NAME',
        color='Branch',
        orientation='h',
        text='Count_of_Baskets',
        barmode='group',
        title=f"Branch Comparison — {a} vs {b}"
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    st.subheader(f"{a} Top Items")
    format_and_display(
        topA,
        numeric_cols=['Count_of_Baskets', 'QTY', 'NET_SALES'],
        index_col='ITEM_NAME',
        total_label='TOTAL'
    )
    st.subheader(f"{b} Top Items")
    format_and_display(
        topB,
        numeric_cols=['Count_of_Baskets', 'QTY', 'NET_SALES'],
        index_col='ITEM_NAME',
        total_label='TOTAL'
    )

def product_performance(df):
    st.header("Product Performance")

    required = ['ITEM_CODE','ITEM_NAME','CUST_CODE','STORE_NAME','QTY','CATEGORY','DEPARTMENT']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns: {missing}")
        return

    # Clean
    d = df.copy()
    for c in ['ITEM_CODE','ITEM_NAME','CUST_CODE','STORE_NAME','CATEGORY','DEPARTMENT']:
        d[c] = d[c].astype(str).str.strip()
    d['QTY'] = pd.to_numeric(d['QTY'], errors='coerce').fillna(0)

    # SKU picker
    lookup = d[['ITEM_CODE','ITEM_NAME']].drop_duplicates().sort_values(['ITEM_CODE','ITEM_NAME'])
    options = (lookup['ITEM_CODE'] + ' — ' + lookup['ITEM_NAME']).tolist()
    choice = st.selectbox("Choose SKU (CODE — NAME)", options)

    if not choice:
        return

    item_code = choice.split('—')[0].strip()
    item_data = d[d['ITEM_CODE'] == item_code]
    if item_data.empty:
        st.info("No data for this SKU")
        return

    item_name = item_data['ITEM_NAME'].iloc[0]
    item_cat = item_data['CATEGORY'].mode().iloc[0]
    item_dept = item_data['DEPARTMENT'].mode().iloc[0]

    st.subheader(f"SKU: {item_code} — {item_name}")
    st.caption(f"Category: {item_cat} | Department: {item_dept}")

    # --------------------------
    # Basket Computations
    # --------------------------
    store_total_customers = d.groupby('STORE_NAME')['CUST_CODE'].nunique()
    cat_pool = d[d['CATEGORY'] == item_cat].groupby('STORE_NAME')['CUST_CODE'].nunique()
    dept_pool = d[d['DEPARTMENT'] == item_dept].groupby('STORE_NAME')['CUST_CODE'].nunique()

    basket_item_counts = (
        d.groupby(['STORE_NAME','CUST_CODE'])['ITEM_CODE'].nunique()
         .rename('Distinct_SKUs').reset_index()
    )

    baskets_with_item = (
        item_data[['STORE_NAME','CUST_CODE']].drop_duplicates().assign(Has_Item=1)
    )

    with_comp = baskets_with_item.merge(
        basket_item_counts,
        on=['STORE_NAME','CUST_CODE'],
        how='left'
    )
    with_comp['Only_Item'] = (with_comp['Distinct_SKUs'] == 1).astype(int)
    with_comp['With_Others'] = (with_comp['Distinct_SKUs'] > 1).astype(int)

    # store-level summaries
    max_qty_per_basket = (
        item_data.groupby(['STORE_NAME','CUST_CODE'])['QTY'].sum()
                 .groupby('STORE_NAME').max()
    )
    total_qty_per_store = item_data.groupby('STORE_NAME')['QTY'].sum()

    store_summary = (
        with_comp.groupby('STORE_NAME', as_index=False)
                 .agg(
                    Baskets_With_Item=('Has_Item','sum'),
                    Only_Item_Baskets=('Only_Item','sum'),
                    With_Other_Items=('With_Others','sum')
                 )
    )

    store_summary = (
        store_summary
        .merge(max_qty_per_basket.rename('Highest_QTY_In_Basket'), on='STORE_NAME', how='left')
        .merge(total_qty_per_store.rename('Total_QTY_Sold_Branch'), on='STORE_NAME', how='left')
        .fillna(0)
    )

    # global totals
    g_baskets = store_summary['Baskets_With_Item'].sum()
    g_only = store_summary['Only_Item_Baskets'].sum()
    g_with = store_summary['With_Other_Items'].sum()

    g_only_pct = round(100 * g_only / g_baskets, 1) if g_baskets else 0
    g_with_pct = round(100 * g_with / g_baskets, 1) if g_baskets else 0

    g_store = int(store_total_customers.sum())
    g_cat = int(cat_pool.sum())
    g_dept = int(dept_pool.sum())

    g_pct_store = round(100 * g_baskets / g_store, 1) if g_store else 0
    g_pct_cat = round(100 * g_baskets / g_cat, 1) if g_cat else 0
    g_pct_dept = round(100 * g_baskets / g_dept, 1) if g_dept else 0

    # store percentages
    per_store = store_summary.copy()

    per_store['Only_Item_Baskets'] = (
        (per_store['Only_Item_Baskets'] / per_store['Baskets_With_Item'] * 100)
        .replace([np.nan, np.inf], 0).round(1)
    )
    per_store['With_Other_Items'] = (
        (per_store['With_Other_Items'] / per_store['Baskets_With_Item'] * 100)
        .replace([np.nan, np.inf], 0).round(1)
    )

    per_store['Pct_of_Store_Customers'] = (
        per_store['STORE_NAME'].map(store_total_customers)
        .rdiv(per_store['Baskets_With_Item']) * 100
    ).replace([np.nan, np.inf], 0).round(1)

    per_store['Pct_of_Category_Customers'] = (
        per_store['STORE_NAME'].map(cat_pool)
        .rdiv(per_store['Baskets_With_Item']) * 100
    ).replace([np.nan, np.inf], 0).round(1)

    per_store['Pct_of_Department_Customers'] = (
        per_store['STORE_NAME'].map(dept_pool)
        .rdiv(per_store['Baskets_With_Item']) * 100
    ).replace([np.nan, np.inf], 0).round(1)

    per_store = per_store.sort_values('Baskets_With_Item', ascending=False)

    # TOTAL row
    total_row = pd.DataFrame([{
        'STORE_NAME': 'TOTAL',
        'Baskets_With_Item': g_baskets,
        'Only_Item_Baskets': g_only_pct,
        'With_Other_Items': g_with_pct,
        'Highest_QTY_In_Basket': int(per_store['Highest_QTY_In_Basket'].max()),
        'Total_QTY_Sold_Branch': total_qty_per_store.sum(),
        'Pct_of_Store_Customers': g_pct_store,
        'Pct_of_Category_Customers': g_pct_cat,
        'Pct_of_Department_Customers': g_pct_dept
    }])

    final = pd.concat([total_row, per_store], ignore_index=True)
    final.insert(0, '#', ['' if i == 0 else i for i in range(len(final))])

    # formatting
    final['Baskets_With_Item'] = final['Baskets_With_Item'].map('{:,.0f}'.format)
    for col in ['Only_Item_Baskets','With_Other_Items',
                'Pct_of_Store_Customers','Pct_of_Category_Customers','Pct_of_Department_Customers']:
        final[col] = final[col].map(lambda x: f"{float(x):.1f}%")
    for col in ['Highest_QTY_In_Basket','Total_QTY_Sold_Branch']:
        final[col] = final[col].map('{:,.0f}'.format)

    st.subheader("Store Breakdown")
    st.dataframe(final, use_container_width=True)

    # chart
    chart_df = per_store[['STORE_NAME','Only_Item_Baskets','With_Other_Items']].melt(
        id_vars='STORE_NAME',
        var_name='Type',
        value_name='Percent'
    )
    chart_df['Type'] = chart_df['Type'].map({
        'Only_Item_Baskets': 'Only Item (%)',
        'With_Other_Items': 'With Other Items (%)'
    })

    fig = px.bar(
        chart_df,
        x='Percent', y='STORE_NAME',
        color='Type', orientation='h',
        barmode='stack',
        title=f"Stores — Basket Split (%) for {item_code} ({item_name})"
    )
    fig.update_layout(height=max(420, 22*len(chart_df['STORE_NAME'].unique())))
    st.plotly_chart(fig, use_container_width=True)

def global_loyalty_overview(df):
    st.header("Global Loyalty Overview")
    required = [
        'TRN_DATE', 'STORE_NAME', 'CUST_CODE',
        'LOYALTY_CUSTOMER_CODE', 'NET_SALES'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(
            f"Missing required columns for Global Loyalty Overview: {missing}"
        )
        return

    dfL = df.copy()
    dfL['TRN_DATE'] = pd.to_datetime(dfL['TRN_DATE'], errors='coerce')
    dfL = dfL.dropna(subset=['TRN_DATE', 'STORE_NAME', 'CUST_CODE'])

    for c in ['STORE_NAME', 'CUST_CODE', 'LOYALTY_CUSTOMER_CODE']:
        dfL[c] = dfL[c].astype(str).str.strip()

    dfL['NET_SALES'] = pd.to_numeric(
        dfL['NET_SALES'], errors='coerce'
    ).fillna(0)

    dfL = dfL[
        dfL['LOYALTY_CUSTOMER_CODE']
        .replace({'nan': '', 'NaN': '', 'None': ''})
        .str.len() > 0
    ].copy()

    receipts = (
        dfL.groupby(
            ['STORE_NAME', 'CUST_CODE', 'LOYALTY_CUSTOMER_CODE'],
            as_index=False
        )
        .agg(
            Basket_Value=('NET_SALES', 'sum'),
            First_Time=('TRN_DATE', 'min')
        )
    )

    per_branch_multi = receipts.groupby(
        ['STORE_NAME', 'LOYALTY_CUSTOMER_CODE']
    ).agg(
        Baskets_in_Store=('CUST_CODE', 'nunique'),
        Total_Value_in_Store=('Basket_Value', 'sum')
    ).reset_index()

    per_branch_multi = per_branch_multi[
        per_branch_multi['Baskets_in_Store'] > 1
    ]

    overview = per_branch_multi.groupby(
        'STORE_NAME', as_index=False
    ).agg(
        Loyal_Customers_Multi=(
            'LOYALTY_CUSTOMER_CODE', 'nunique'
        ),
        Total_Baskets_of_Those=(
            'Baskets_in_Store', 'sum'
        ),
        Total_Value_of_Those=(
            'Total_Value_in_Store', 'sum'
        )
    )

    overview['Avg_Baskets_per_Customer'] = np.where(
        overview['Loyal_Customers_Multi'] > 0,
        (
            overview['Total_Baskets_of_Those']
            / overview['Loyal_Customers_Multi']
        ).round(2),
        0.0
    )

    format_and_display(
        overview.sort_values(
            'Loyal_Customers_Multi', ascending=False
        ),
        numeric_cols=[
            'Loyal_Customers_Multi',
            'Total_Baskets_of_Those',
            'Total_Value_of_Those',
            'Avg_Baskets_per_Customer'
        ],
        index_col='STORE_NAME',
        total_label='TOTAL'
    )

def branch_loyalty_overview(df):
    st.header("Branch Loyalty Overview (per-branch loyal customers with >1 baskets)")
    if not all(
        c in df.columns for c in [
            'TRN_DATE', 'STORE_NAME', 'CUST_CODE',
            'LOYALTY_CUSTOMER_CODE', 'NET_SALES'
        ]
    ):
        st.warning("Missing required loyalty columns")
        return
    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d = d[
        d['LOYALTY_CUSTOMER_CODE']
        .replace('nan', '')
        .str.strip()
        .astype(bool)
    ]
    receipts = d.groupby(
        ['STORE_NAME', 'CUST_CODE', 'LOYALTY_CUSTOMER_CODE'],
        as_index=False
    ).agg(
        Basket_Value=('NET_SALES', 'sum'),
        First_Time=('TRN_DATE', 'min')
    )
    stores = sorted(receipts['STORE_NAME'].unique())
    store = st.selectbox("Select Branch", stores)
    per_store = receipts[receipts['STORE_NAME'] == store].groupby(
        'LOYALTY_CUSTOMER_CODE', as_index=False
    ).agg(
        Baskets_in_Store=('CUST_CODE', 'nunique'),
        Total_Value_in_Store=('Basket_Value', 'sum'),
        First_Time=('First_Time', 'min')
    )
    per_store = per_store[
        per_store['Baskets_in_Store'] > 1
    ].sort_values(
        ['Baskets_in_Store', 'Total_Value_in_Store'],
        ascending=False
    )
    format_and_display(
        per_store,
        numeric_cols=['Baskets_in_Store', 'Total_Value_in_Store'],
        index_col='LOYALTY_CUSTOMER_CODE',
        total_label='TOTAL'
    )

def customer_loyalty_overview(df):
    st.header("Customer Loyalty Overview (global)")
    if not all(
        c in df.columns for c in [
            'TRN_DATE', 'STORE_NAME', 'CUST_CODE',
            'LOYALTY_CUSTOMER_CODE', 'NET_SALES'
        ]
    ):
        st.warning("Missing required loyalty columns")
        return
    d = df.copy()
    d = d[
        d['LOYALTY_CUSTOMER_CODE']
        .replace('nan', '')
        .str.strip()
        .astype(bool)
    ]
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    receipts = d.groupby(
        ['STORE_NAME', 'CUST_CODE', 'LOYALTY_CUSTOMER_CODE'],
        as_index=False
    ).agg(
        Basket_Value=('NET_SALES', 'sum'),
        First_Time=('TRN_DATE', 'min')
    )
    stores_per_cust = receipts.groupby(
        'LOYALTY_CUSTOMER_CODE'
    )['STORE_NAME'].nunique().reset_index(
        name='Stores_Visited'
    )
    customers = sorted(
        stores_per_cust[
            stores_per_cust['Stores_Visited'] > 1
        ]['LOYALTY_CUSTOMER_CODE'].unique()
    )
    if not customers:
        st.info("No loyalty customers with >1 baskets found")
        return
    cust = st.selectbox(
        "Select Loyalty Customer (multi-store)",
        customers
    )
    rc = receipts[receipts['LOYALTY_CUSTOMER_CODE'] == cust]
    if rc.empty:
        st.info("No receipts for this customer")
        return
    per_store = rc.groupby(
        'STORE_NAME', as_index=False
    ).agg(
        Baskets=('CUST_CODE', 'nunique'),
        Total_Value=('Basket_Value', 'sum'),
        First_Time=('First_Time', 'min'),
        Last_Time=('First_Time', 'max')
    ).sort_values(
        ['Baskets', 'Total_Value'],
        ascending=False
    )
    format_and_display(
        per_store,
        numeric_cols=['Baskets', 'Total_Value'],
        index_col='STORE_NAME',
        total_label='TOTAL'
    )
    rc_disp = rc.copy()
    rc_disp['First_Time'] = rc_disp['First_Time'].dt.strftime(
        '%Y-%m-%d %H:%M:%S'
    )
    rc_disp = rc_disp.rename(
        columns={
            'CUST_CODE': 'Receipt_No',
            'Basket_Value': 'Basket_Value_KSh'
        }
    )
    format_and_display(
        rc_disp[
            ['STORE_NAME', 'Receipt_No',
             'Basket_Value_KSh', 'First_Time']
        ],
        numeric_cols=['Basket_Value_KSh'],
        index_col='STORE_NAME',
        total_label='TOTAL'
    )

def global_pricing_overview(df):
    st.header("Global Pricing Overview — Multi-Priced SKUs per Day")
    required = [
        'TRN_DATE', 'STORE_NAME', 'ITEM_CODE',
        'ITEM_NAME', 'QTY', 'SP_PRE_VAT'
    ]
    if not all(c in df.columns for c in required):
        st.warning("Missing pricing columns")
        return
    d = df.copy()
    d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
    d['DATE'] = d['TRN_DATE'].dt.date
    grp = d.groupby(
        ['STORE_NAME', 'DATE', 'ITEM_CODE', 'ITEM_NAME'],
        as_index=False
    ).agg(
        Num_Prices=('SP_PRE_VAT', lambda s: s.dropna().nunique()),
        Price_Min=('SP_PRE_VAT', 'min'),
        Price_Max=('SP_PRE_VAT', 'max'),
        Total_QTY=('QTY', 'sum')
    )
    grp['Price_Spread'] = grp['Price_Max'] - grp['Price_Min']
    multi_price = grp[
        (grp['Num_Prices'] > 1) &
        (grp['Price_Spread'] > 0)
    ].copy()
    if multi_price.empty:
        st.info("No multi-priced SKUs found")
        return
    multi_price['Diff_Value'] = (
        multi_price['Total_QTY'] * multi_price['Price_Spread']
    )
    summary = multi_price.groupby(
        'STORE_NAME', as_index=False
    ).agg(
        Items_with_MultiPrice=('ITEM_CODE', 'nunique'),
        Total_Diff_Value=('Diff_Value', 'sum'),
        Avg_Spread=('Price_Spread', 'mean'),
        Max_Spread=('Price_Spread', 'max')
    )
    format_and_display(
        summary.sort_values(
            'Total_Diff_Value', ascending=False
        ),
        numeric_cols=[
            'Items_with_MultiPrice',
            'Total_Diff_Value',
            'Avg_Spread',
            'Max_Spread'
        ],
        index_col='STORE_NAME',
        total_label='TOTAL'
    )

def branch_pricing_overview(df):
    st.header("Branch Pricing Overview")

    required = [
        'TRN_DATE', 'STORE_NAME', 'ITEM_CODE',
        'ITEM_NAME', 'QTY', 'SP_PRE_VAT', 'CUST_CODE'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(
            f"Missing required columns for Branch Pricing Overview: {missing}"
        )
        return

    d0 = df.copy()
    d0['TRN_DATE'] = pd.to_datetime(d0['TRN_DATE'], errors='coerce')
    d0 = d0.dropna(
        subset=[
            'TRN_DATE', 'STORE_NAME', 'ITEM_CODE',
            'ITEM_NAME', 'QTY', 'SP_PRE_VAT', 'CUST_CODE'
        ]
    ).copy()

    for c in ['STORE_NAME', 'ITEM_CODE', 'ITEM_NAME', 'CUST_CODE']:
        d0[c] = d0[c].astype(str).str.strip()

    d0['SP_PRE_VAT'] = d0['SP_PRE_VAT'].astype(str).str.replace(
        ',', '', regex=False
    ).str.strip()
    d0['SP_PRE_VAT'] = pd.to_numeric(
        d0['SP_PRE_VAT'], errors='coerce'
    ).fillna(0.0)
    d0['QTY'] = pd.to_numeric(
        d0['QTY'], errors='coerce'
    ).fillna(0.0)
    d0['DATE'] = d0['TRN_DATE'].dt.date

    branches = sorted(d0['STORE_NAME'].unique())
    if not branches:
        st.info("No branches available in data")
        return
    branch = st.selectbox("Select Branch", branches)

    if not branch:
        st.info("Select a branch to proceed")
        return

    d = d0[d0['STORE_NAME'] == branch].copy()
    if d.empty:
        st.info("No rows for this branch.")
        return

    per_item_day = d.groupby(
        ['DATE', 'ITEM_CODE', 'ITEM_NAME'],
        as_index=False
    ).agg(
        Num_Prices=('SP_PRE_VAT', lambda s: s.dropna().nunique()),
        Price_Min=('SP_PRE_VAT', 'min'),
        Price_Max=('SP_PRE_VAT', 'max'),
        Total_QTY=('QTY', 'sum')
    )
    per_item_day['Price_Spread'] = (
        per_item_day['Price_Max'] - per_item_day['Price_Min']
    )

    eps = 1e-9
    multi = per_item_day[
        (per_item_day['Num_Prices'] > 1) &
        (per_item_day['Price_Spread'] > eps)
    ].copy()

    if multi.empty:
        st.success(
            f"✅ {branch}: No SKUs with more than one distinct price "
            f"(spread > 0) on the same day."
        )
        return

    multi['Price_Spread'] = multi['Price_Spread'].round(2)
    multi['Diff_Value'] = (
        multi['Total_QTY'] * multi['Price_Spread']
    ).round(2)
    multi_sum = multi.sort_values(
        ['DATE', 'Price_Spread', 'Total_QTY'],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    multi_sum.insert(0, '#', range(1, len(multi_sum) + 1))

    sku_days = len(multi_sum)
    sku_count = multi[
        ['ITEM_CODE', 'ITEM_NAME']
    ].drop_duplicates().shape[0]
    value_sum = float(multi['Diff_Value'].sum())

    st.markdown(
        f"**Branch:** {branch}  \n"
        f"• Item-Days with >1 price (spread>0): **{sku_days:,}**   "
        f"• Distinct SKUs affected: **{sku_count:,}**   "
        f"• Total Diff Value: **{value_sum:,.2f}**"
    )

    st.subheader("Per Item / Day — Summary")
    format_and_display(
        multi_sum[
            [
                '#', 'DATE', 'ITEM_CODE', 'ITEM_NAME',
                'Num_Prices', 'Price_Min', 'Price_Max',
                'Price_Spread', 'Total_QTY', 'Diff_Value'
            ]
        ],
        numeric_cols=[
            'Num_Prices', 'Price_Min', 'Price_Max',
            'Price_Spread', 'Total_QTY', 'Diff_Value'
        ],
        index_col='ITEM_CODE',
        total_label='TOTAL'
    )

    price_brk = d.merge(
        multi[['DATE', 'ITEM_CODE']],
        on=['DATE', 'ITEM_CODE'],
        how='inner'
    ).groupby(
        ['DATE', 'ITEM_CODE', 'ITEM_NAME', 'SP_PRE_VAT'],
        as_index=False
    ).agg(
        Qty_At_Price=('QTY', 'sum'),
        Receipts_At_Price=('CUST_CODE', 'nunique'),
        First_Time=('TRN_DATE', 'min'),
        Last_Time=('TRN_DATE', 'max')
    )

    price_brk = price_brk.sort_values(
        ['DATE', 'ITEM_NAME', 'SP_PRE_VAT'],
        ascending=[False, True, False]
    ).reset_index(drop=True)
    price_brk['First_Time_str'] = price_brk['First_Time'].dt.strftime(
        '%Y-%m-%d %H:%M:%S'
    )
    price_brk['Last_Time_str'] = price_brk['Last_Time'].dt.strftime(
        '%Y-%m-%d %H:%M:%S'
    )
    price_brk['Time_Window'] = (
        price_brk['First_Time_str'] +
        ' → ' +
        price_brk['Last_Time_str']
    )

    price_compact = price_brk[
        [
            'DATE', 'ITEM_CODE', 'ITEM_NAME',
            'SP_PRE_VAT', 'Qty_At_Price',
            'Receipts_At_Price', 'Time_Window'
        ]
    ].rename(
        columns={
            'SP_PRE_VAT': 'Price',
            'Qty_At_Price': 'QTY',
            'Receipts_At_Price': 'Receipts'
        }
    )

    st.subheader("Detailed — Price Breakdown (clean view)")
    format_and_display(
        price_compact,
        numeric_cols=['Price', 'QTY', 'Receipts'],
        index_col='ITEM_CODE',
        total_label='TOTAL'
    )

    receipts_detail = d.merge(
        multi[['DATE', 'ITEM_CODE']],
        on=['DATE', 'ITEM_CODE'],
        how='inner'
    )
    receipt_cols = [
        'DATE', 'ITEM_CODE', 'ITEM_NAME',
        'CUST_CODE', 'TRN_DATE',
        'SP_PRE_VAT', 'QTY'
    ]
    optional = [
        'CASHIER', 'Till_Code', 'SHIFT',
        'SALES_CHANNEL_L1', 'SALES_CHANNEL_L2'
    ]
    for c in optional:
        if c in receipts_detail.columns:
            receipt_cols.append(c)
    receipt_cols = [c for c in receipt_cols if c in receipts_detail.columns]
    receipts_detail = receipts_detail[
        receipt_cols
    ].sort_values(
        ['DATE', 'ITEM_CODE', 'TRN_DATE'],
        ascending=[False, True, True]
    ).reset_index(drop=True)
    receipts_detail['TRN_DATE'] = receipts_detail[
        'TRN_DATE'
    ].dt.strftime('%Y-%m-%d %H:%M:%S')

    st.subheader(
        "All Receipt-level Details for Affected Item-Days"
    )
    st.info(
        f"Showing {len(receipts_detail):,} receipt rows for the "
        f"affected item-days. Use the table search/filter in "
        f"Streamlit to inspect specific receipts."
    )
    with st.expander("Show full receipt details (expand)"):
        st.dataframe(
            receipts_detail,
            use_container_width=True
        )

    try:
        csv_bytes = receipts_detail.to_csv(
            index=False
        ).encode('utf-8')
        st.download_button(
            "Download receipts as CSV",
            data=csv_bytes,
            file_name=f"{branch}_multi_price_receipts.csv",
            mime="text/csv"
        )
    except Exception:
        pass

def global_refunds_overview(df):
    st.header("Global Refunds Overview (Negative receipts)")
    d = df.copy()
    d['NET_SALES'] = pd.to_numeric(
        d['NET_SALES'].astype(str).str.replace(
            ',', '', regex=False
        ),
        errors='coerce'
    ).fillna(0)
    neg = d[d['NET_SALES'] < 0]
    if neg.empty:
        st.info("No negative receipts found")
        return
    if 'CAP_CUSTOMER_CODE' in neg.columns:
        neg['Sale_Type'] = np.where(
            neg['CAP_CUSTOMER_CODE']
            .str.replace('nan', '')
            .str.strip()
            .astype(bool),
            'On_account sales',
            'General sales'
        )
    else:
        neg['Sale_Type'] = 'General sales'
    summary = neg.groupby(
        ['STORE_NAME', 'Sale_Type'],
        as_index=False
    ).agg(
        Total_Neg_Value=('NET_SALES', 'sum'),
        Total_Count=('CUST_CODE', 'nunique')
    )
    format_and_display(
        summary.sort_values(
            'Total_Neg_Value'
        ),
        numeric_cols=['Total_Neg_Value', 'Total_Count'],
        index_col='STORE_NAME',
        total_label='TOTAL'
    )

def branch_refunds_overview(df):
    st.header("Branch Refunds Overview (Negative receipts per store)")
    d = df.copy()
    d['NET_SALES'] = pd.to_numeric(
        d['NET_SALES'].astype(str).str.replace(
            ',', '', regex=False
        ),
        errors='coerce'
    ).fillna(0)
    neg = d[d['NET_SALES'] < 0]
    branches = sorted(neg['STORE_NAME'].unique())
    if not branches:
        st.info("No negative receipts")
        return
    branch = st.selectbox("Select Branch", branches)
    dfb = neg[neg['STORE_NAME'] == branch]
    agg = dfb.groupby(
        ['STORE_NAME', 'CUST_CODE'],
        as_index=False
    ).agg(
        Total_Value=('NET_SALES', 'sum'),
        First_Time=('TRN_DATE', 'min'),
        Cashier=('CASHIER', 'first')
    )
    format_and_display(
        agg.sort_values('Total_Value'),
        numeric_cols=['Total_Value'],
        index_col='CUST_CODE',
        total_label='TOTAL'
    )

# -----------------------
# Main App
# -----------------------
def main():
    st.title("DailyDeck: The Story Behind the Numbers")

    raw_df = smart_load()
    if raw_df is None:
        st.stop()

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
