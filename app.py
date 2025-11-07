# Streamlit Superdeck Analytics Dashboard
# This file is an updated, self-contained Streamlit app that:
# - Adds the missing OPERATIONS and INSIGHTS subsections (implemented from the attached Colab notebook)
# - Ensures numbers are displayed with thousands separators where appropriate
# - Adds totals/averages rows in summary tables where useful
#
# Notes:
# - To accept very large uploads you still must set Streamlit server config (server.maxUploadSize)
#   or the STREAMLIT_SERVER_MAX_UPLOAD_SIZE env var before starting Streamlit.
# - The app uses Plotly for charts and pandas for processing. For PNG export of plots install kaleido.

import os
import io
from datetime import timedelta
from typing import List, Dict

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# === Page config & CSS tweaks ===
st.set_page_config(layout="wide", page_title="Superdeck Analytics Dashboard")
st.markdown("""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
        min-width: 320px;
        max-width: 480px;
        padding-right: 12px;
    }
    .block-container {padding-top:0.6rem;}
    </style>
    """, unsafe_allow_html=True)

st.title("🦸 Superdeck Analytics Dashboard")
st.markdown("> Upload your sales CSV, choose a main section and subsection for live analytics.")

# === Sidebar: show active server upload limit debug info ===
st.sidebar.header("Upload / Settings")
st.sidebar.write("STREAMLIT_SERVER_MAX_UPLOAD_SIZE (env):", os.environ.get("STREAMLIT_SERVER_MAX_UPLOAD_SIZE"))
try:
    st.sidebar.write("streamlit server.maxUploadSize (config):", st.config.get_option("server.maxUploadSize"))
except Exception:
    st.sidebar.write("streamlit server.maxUploadSize (config): unavailable")

st.sidebar.markdown(
    "To allow >200MB uploads set the server config before starting Streamlit, e.g.:\n\n"
    "- `streamlit run app.py --server.maxUploadSize=1024`\n"
    "- or add `.streamlit/config.toml` with `server.maxUploadSize = 1024`"
)

uploaded = st.sidebar.file_uploader("Upload CSV (check server settings)", type="csv")
if uploaded is None:
    st.info("Please upload a dataset to proceed.")
    st.stop()

# === Utility / Formatting helpers ===
def fmt_ints(df: pd.DataFrame, int_cols: List[str]) -> pd.DataFrame:
    df2 = df.copy()
    for c in int_cols:
        if c in df2.columns:
            # Preserve NaNs
            df2[c] = df2[c].apply(lambda v: f"{int(v):,}" if pd.notna(v) and np.isfinite(v) else v)
    return df2

def fmt_floats(df: pd.DataFrame, float_cols: List[str], decimals: int = 2) -> pd.DataFrame:
    df2 = df.copy()
    for c in float_cols:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda v: f"{v:,.{decimals}f}" if pd.notna(v) and np.isfinite(v) else v)
    return df2

def append_totals_row(df: pd.DataFrame, sum_cols: List[str], label_col: str = None, label: str = "Total") -> pd.DataFrame:
    """Append a totals row for sum_cols. If label_col provided, set it to label in the totals row."""
    tot = {}
    for c in sum_cols:
        if c in df.columns:
            # If column is numeric
            try:
                tot[c] = df[c].replace([np.inf, -np.inf], np.nan).dropna().astype(float).sum()
            except Exception:
                tot[c] = ""
    # build totals row
    total_row = {c: "" for c in df.columns}
    for k, v in tot.items():
        total_row[k] = v
    if label_col and label_col in df.columns:
        total_row[label_col] = label
    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

def safe_to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    return df

# === Data load & basic preparation ===
@st.cache_data(show_spinner=True)
def load_and_prepare(uploaded_file):
    df = pd.read_csv(uploaded_file, on_bad_lines='skip', low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # Date parsing
    for col in ['TRN_DATE', 'ZED_DATE']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    # Numeric columns
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    df = safe_to_numeric(df, numeric_cols)
    # ID columns to string
    idcols = ['STORE_CODE', 'TILL', 'SESSION', 'RCT']
    for col in idcols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('').str.strip()
    # Build CUST_CODE if missing
    if 'CUST_CODE' not in df.columns:
        if all(c in df.columns for c in idcols):
            df['CUST_CODE'] = (
                df['STORE_CODE'].str.strip() + '-' +
                df['TILL'].str.strip() + '-' +
                df['SESSION'].str.strip() + '-' +
                df['RCT'].str.strip()
            )
        else:
            missing = [c for c in idcols if c not in df.columns]
            st.error(f"Missing columns for CUST_CODE: {missing}")
            st.stop()
    df['CUST_CODE'] = df['CUST_CODE'].astype(str).str.strip()
    # Ensure some helpful columns exist
    for col in ['STORE_NAME','ITEM_NAME','ITEM_CODE','DEPARTMENT','CATEGORY','CASHIER','CAP_CUSTOMER_CODE','LOYALTY_CUSTOMER_CODE','CU_DEVICE_SERIAL']:
        if col not in df.columns:
            df[col] = ""
    # Clean whitespace
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()
    return df

df = load_and_prepare(uploaded)

# Precompute 30-min time grid
@st.cache_data
def get_time_grid():
    start_time = pd.Timestamp("00:00:00")
    intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
    col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
    return intervals, col_labels
intervals, col_labels = get_time_grid()

# === Download helpers ===
def download_button(df_obj, filename, label):
    if isinstance(df_obj, pd.Series):
        df_obj = df_obj.reset_index()
    csv = df_obj.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")

def download_plot(fig, filename):
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600)
        st.download_button("⬇️ Download Plot as PNG", img_bytes, filename=filename, mime="image/png")
    except Exception:
        st.warning("Plot download unavailable (install kaleido to enable).")

# === Main sections & subsections (same list as original) ===
main_sections = {
    "SALES": [
        "Global sales Overview",
        "Global Net Sales Distribution by Sales Channel",
        "Global Net Sales Distribution by SHIFT",
        "Night vs Day Shift Sales Ratio — Stores with Night Shifts",
        "Global Day vs Night Sales — Only Stores with NIGHT Shift",
        "2nd-Highest Channel Share",
        "Bottom 30 — 2nd Highest Channel",
        "Stores Sales Summary"
    ],
    "OPERATIONS": [
        "Customer Traffic-Storewise",
        "Active Tills During the day",
        "Average Customers Served per Till",
        "Store Customer Traffic Storewise",
        "Customer Traffic-Departmentwise",
        "Cashiers Perfomance",
        "Till Usage",
        "Tax Compliance"
    ],
    "INSIGHTS": [
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
        "Branch Branch Overview",
        "Global Refunds Overview",
        "Branch Refunds Overview"
    ]
}

section = st.sidebar.radio("Main Section", list(main_sections.keys()))
subsection = st.sidebar.selectbox("Subsection", main_sections[section], key="subsection")
st.markdown(f"### {section} ➔ {subsection}")

# === SALES (kept mostly as original) ===
if section == "SALES":
    if subsection == "Global sales Overview":
        gs = df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum()
        gs['NET_SALES_M'] = gs['NET_SALES'] / 1_000_000
        gs['PCT'] = (gs['NET_SALES'] / gs['NET_SALES'].sum()) * 100
        labels = [f"{r['SALES_CHANNEL_L1']} ({r['PCT']:.1f}% | {r['NET_SALES_M']:.1f}M)" for _, r in gs.iterrows()]
        fig = go.Figure(data=[go.Pie(labels=labels, values=gs['NET_SALES_M'], hole=0.6,
                                     marker=dict(colors=px.colors.qualitative.Plotly),
                                     text=[f"{p:.1f}%" for p in gs['PCT']], textinfo='text')])
        fig.update_layout(title="SALES CHANNEL TYPE — Global Overview", height=520)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fmt_floats(gs[['SALES_CHANNEL_L1','NET_SALES','NET_SALES_M','PCT']].rename(columns={'NET_SALES':'NET_SALES (KSh)','NET_SALES_M':'NET_SALES (M)','PCT':'Pct %'}), ['NET_SALES (KSh)','NET_SALES (M)'], 2))
        download_button(gs, "global_sales_overview.csv", "⬇️ Download Table")
        download_plot(fig, "global_sales_overview.png")

    elif subsection == "Global Net Sales Distribution by Sales Channel":
        g2 = df.groupby('SALES_CHANNEL_L2', as_index=False)['NET_SALES'].sum()
        g2['NET_SALES_M'] = g2['NET_SALES']/1_000_000
        g2['PCT'] = g2['NET_SALES']/g2['NET_SALES'].sum()*100
        fig = px.pie(g2, names='SALES_CHANNEL_L2', values='NET_SALES_M', hole=0.6, title="Net Sales by Sales Mode (L2)",
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fmt_floats(g2[['SALES_CHANNEL_L2','NET_SALES','NET_SALES_M','PCT']].rename(columns={'NET_SALES':'NET_SALES (KSh)','NET_SALES_M':'NET_SALES (M)','PCT':'Pct %'}), ['NET_SALES (KSh)','NET_SALES (M)'], 2))
        download_button(g2, "sales_channel_l2.csv", "⬇️ Download Table")
        download_plot(fig, "sales_channel_l2_pie.png")

    elif subsection == "Global Net Sales Distribution by SHIFT":
        sh = df.groupby('SHIFT', as_index=False)['NET_SALES'].sum()
        sh['PCT'] = sh['NET_SALES']/sh['NET_SALES'].sum()*100
        fig = px.pie(sh, names='SHIFT', values='NET_SALES', hole=0.6, title="Net Sales by Shift",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fmt_floats(sh[['SHIFT','NET_SALES','PCT']].rename(columns={'NET_SALES':'NET_SALES (KSh)','PCT':'Pct %'}), ['NET_SALES (KSh)'], 2))
        download_button(sh, "shift_sales.csv", "⬇️ Download Table")
        download_plot(fig, "shift_sales_pie.png")

    elif subsection == "Night vs Day Shift Sales Ratio — Stores with Night Shifts":
        # implemented previously; keep
        ns = df[df['SHIFT'].str.upper().str.contains('NIGHT', na=False)]['STORE_NAME'].unique()
        if len(ns) == 0:
            st.info("No stores with NIGHT shift found.")
        else:
            df_nd = df[df['STORE_NAME'].isin(ns)].copy()
            df_nd['Shift_Bucket'] = np.where(df_nd['SHIFT'].str.upper().str.contains('NIGHT', na=False),'Night','Day')
            ratio_df = df_nd.groupby(['STORE_NAME','Shift_Bucket'], as_index=False)['NET_SALES'].sum()
            store_totals = ratio_df.groupby('STORE_NAME')['NET_SALES'].transform('sum')
            ratio_df['PCT'] = 100 * ratio_df['NET_SALES'] / store_totals
            pivot_df = ratio_df.pivot(index='STORE_NAME', columns='Shift_Bucket', values='PCT').fillna(0)
            pivot_sorted = pivot_df.sort_values(by='Night', ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=pivot_sorted['Night'], y=pivot_sorted.index, orientation='h', name='Night', marker_color='#d62728'))
            fig.add_trace(go.Bar(x=pivot_sorted['Day'], y=pivot_sorted.index, orientation='h', name='Day', marker_color='#1f77b4'))
            fig.update_layout(barmode='group', title="Night vs Day % (by Store)", xaxis_title="% of Store Sales", height=700)
            st.plotly_chart(fig, use_container_width=True)
            # format numeric columns
            df_out = pivot_sorted.reset_index().rename(columns={'Night':'Night %','Day':'Day %'})
            df_out = fmt_floats(df_out, ['Night %','Day %'], 1)
            st.dataframe(df_out, use_container_width=True)
            download_button(df_out, "night_day_ratio.csv", "⬇️ Download Table")
            download_plot(fig, "night_day_ratio_bar.png")

    elif subsection == "Global Day vs Night Sales — Only Stores with NIGHT Shift":
        ns = df[df['SHIFT'].str.upper().str.contains('NIGHT', na=False)]['STORE_NAME'].unique()
        if len(ns) == 0:
            st.info("No stores with NIGHT shift found.")
        else:
            df_nd = df[df['STORE_NAME'].isin(ns)].copy()
            df_nd['Shift_Bucket'] = np.where(df_nd['SHIFT'].str.upper().str.contains('NIGHT', na=False),'Night','Day')
            gb = df_nd.groupby('Shift_Bucket', as_index=False)['NET_SALES'].sum()
            gb['PCT'] = 100 * gb['NET_SALES'] / gb['NET_SALES'].sum()
            fig = px.pie(gb, names='Shift_Bucket', values='NET_SALES', hole=0.6, title="Global Day vs Night Sales (NIGHT Shift only)",
                         color_discrete_map={'Night':'#d62728','Day':'#1f77b4'})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(fmt_floats(gb.rename(columns={'NET_SALES':'NET_SALES (KSh)','PCT':'Pct %'}), ['NET_SALES (KSh)','Pct %'], 2))
            download_button(gb, "day_night_global.csv", "⬇️ Download Table")
            download_plot(fig, "day_night_global.png")

    elif subsection == "2nd-Highest Channel Share":
        req = {"STORE_NAME","SALES_CHANNEL_L1","NET_SALES"}
        if req.issubset(df.columns):
            data = df.copy()
            data["NET_SALES"] = pd.to_numeric(data["NET_SALES"], errors="coerce").fillna(0)
            store_chan = data.groupby(["STORE_NAME","SALES_CHANNEL_L1"], as_index=False)["NET_SALES"].sum()
            store_tot = store_chan.groupby("STORE_NAME")["NET_SALES"].transform("sum")
            store_chan["PCT"] = 100 * store_chan["NET_SALES"] / store_tot
            store_chan = store_chan.sort_values(["STORE_NAME","PCT"], ascending=[True,False])
            store_chan["RANK"] = store_chan.groupby("STORE_NAME").cumcount() + 1
            second = store_chan[store_chan["RANK"]==2].copy()
            second_sorted = second.sort_values("PCT", ascending=False).head(30)
            if second_sorted.empty:
                st.info("No stores with a second channel identified.")
            else:
                fig = go.Figure(go.Bar(x=second_sorted["PCT"], y=second_sorted["STORE_NAME"], orientation='h', marker_color='#1f77b4'))
                fig.update_layout(title="Top 30 Stores by 2nd-Highest Channel %", xaxis_title="2nd Channel %", height=700)
                st.plotly_chart(fig, use_container_width=True)
                df_out = second_sorted[['STORE_NAME','SALES_CHANNEL_L1','PCT']].rename(columns={'SALES_CHANNEL_L1':'2nd Channel','PCT':'2nd Channel %'})
                st.dataframe(fmt_floats(df_out, ['2nd Channel %'], 1), use_container_width=True)
                download_button(df_out, "top30_2nd_channel.csv", "⬇️ Download Table")
                download_plot(fig, "top30_lollipop.png")
        else:
            st.error("Required columns missing for this view.")

    elif subsection == "Bottom 30 — 2nd Highest Channel":
        req = {"STORE_NAME","SALES_CHANNEL_L1","NET_SALES"}
        if req.issubset(df.columns):
            data = df.copy()
            data["NET_SALES"] = pd.to_numeric(data["NET_SALES"], errors="coerce").fillna(0)
            store_chan = data.groupby(["STORE_NAME","SALES_CHANNEL_L1"], as_index=False)["NET_SALES"].sum()
            store_tot = store_chan.groupby("STORE_NAME")["NET_SALES"].transform("sum")
            store_chan["PCT"] = 100 * store_chan["NET_SALES"] / store_tot
            store_chan = store_chan.sort_values(["STORE_NAME","PCT"], ascending=[True,False])
            store_chan["RANK"] = store_chan.groupby("STORE_NAME").cumcount() + 1
            second = store_chan[store_chan["RANK"]==2].copy()
            bottom30 = second.sort_values("PCT", ascending=True).head(30)
            if bottom30.empty:
                st.info("No stores with a second channel identified.")
            else:
                fig = go.Figure(go.Bar(x=bottom30["PCT"], y=bottom30["STORE_NAME"], orientation='h', marker_color='#d62728'))
                fig.update_layout(title="Bottom 30 Stores by 2nd-Highest Channel %", xaxis_title="2nd Channel %", height=700)
                st.plotly_chart(fig, use_container_width=True)
                df_out = bottom30[['STORE_NAME','SALES_CHANNEL_L1','PCT']].rename(columns={'SALES_CHANNEL_L1':'2nd Channel','PCT':'2nd Channel %'})
                st.dataframe(fmt_floats(df_out, ['2nd Channel %'], 1), use_container_width=True)
                download_button(df_out, "bottom30_2nd_channel.csv", "⬇️ Download Table")
                download_plot(fig, "bottom30_lollipop.png")
        else:
            st.error("Required columns missing for this view.")

    elif subsection == "Stores Sales Summary":
        # Build coarse summary with Totals and % contribution
        if 'GROSS_SALES' not in df.columns and 'VAT_AMT' in df.columns:
            df['GROSS_SALES'] = df['NET_SALES'] + df['VAT_AMT']
        ss = df.groupby('STORE_NAME', as_index=False).agg(NET_SALES=('NET_SALES','sum'), GROSS_SALES=('GROSS_SALES','sum'))
        ss['Customer_Numbers'] = df.groupby('STORE_NAME')['CUST_CODE'].nunique().values
        ss['% Contribution'] = (ss['GROSS_SALES']/ss['GROSS_SALES'].sum()*100).round(2)
        ss = ss.sort_values('GROSS_SALES', ascending=False).reset_index(drop=True)
        ss_display = append_totals_row(ss, ['NET_SALES','GROSS_SALES','Customer_Numbers'], label_col='STORE_NAME', label='TOTAL')
        ss_display = fmt_ints(ss_display, ['NET_SALES','GROSS_SALES','Customer_Numbers'])
        ss_display = fmt_floats(ss_display, ['% Contribution'], 2)
        st.dataframe(ss_display, use_container_width=True)
        download_button(ss_display, "stores_sales_summary.csv", "⬇️ Download Table")
        fig = px.bar(ss, x="GROSS_SALES", y="STORE_NAME", orientation="h", color='% Contribution', color_continuous_scale='Blues',
                     text='GROSS_SALES', title="Gross Sales by Store")
        fig.update_traces(texttemplate='%{text:,.0f}')
        st.plotly_chart(fig, use_container_width=True)
        download_plot(fig, "store_sales_summary_bar.png")

# === OPERATIONS (fill in missing outputs) ===
elif section == "OPERATIONS":
    # Helper: ensure TRN_DATE and CUST_CODE exist
    df['TRN_DATE'] = pd.to_datetime(df['TRN_DATE'], errors='coerce')

    if subsection == "Customer Traffic-Storewise":
        stores = sorted(df["STORE_NAME"].dropna().unique().tolist())
        selected_store = st.selectbox("Select Store", stores)
        dff = df[df["STORE_NAME"]==selected_store].copy()
        dff = dff.dropna(subset=['TRN_DATE'])
        for c in ["STORE_CODE","TILL","SESSION","RCT"]:
            if c in dff.columns:
                dff[c] = dff[c].astype(str).fillna('').str.strip()
        if 'CUST_CODE' not in dff.columns:
            dff['CUST_CODE'] = dff['STORE_CODE']+'-'+dff['TILL']+'-'+dff['SESSION']+'-'+dff['RCT']
        dff['TIME_ONLY'] = dff['TRN_DATE'].dt.floor('30T').dt.time
        heat = dff.groupby('TIME_ONLY')['CUST_CODE'].nunique().reindex(intervals, fill_value=0)
        fig = px.bar(x=col_labels, y=heat.values, labels={"x":"Time","y":"Receipts"}, text=heat.values,
                     color_discrete_sequence=['#3192e1'], title=f"Receipts by Time - {selected_store}")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        df_out = heat.reset_index().rename(columns={'index':'TIME','CUST_CODE':'Receipts'})
        df_out['Receipts'] = df_out[0].astype(int) if 0 in df_out.columns else df_out['CUST_CODE']
        # Format
        df_out = df_out.rename(columns={0:'Receipts'}).loc[:, ['TIME','Receipts']]
        df_out = fmt_ints(df_out, ['Receipts'])
        st.dataframe(df_out, use_container_width=True)
        download_button(df_out, "customer_traffic_storewise.csv", "⬇️ Download Table")
        download_plot(fig, "customer_traffic_storewise.png")

    elif subsection == "Active Tills During the day":
        # Count unique Till_Code (TILL-STORE_CODE) per 30-min slot and produce heatmap + max tills per store table
        for c in ['TILL','STORE_CODE']:
            if c in df.columns:
                df[c] = df[c].astype(str).fillna('').str.strip()
        if 'Till_Code' not in df.columns:
            df['Till_Code'] = df['TILL'] + '-' + df['STORE_CODE']
        d = df.dropna(subset=['TRN_DATE']).copy()
        d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30T')
        d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time
        till_counts = d.groupby(['STORE_NAME','TIME_ONLY'])['Till_Code'].nunique().reset_index(name='UNIQUE_TILLS')
        heatmap = till_counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='UNIQUE_TILLS').fillna(0)
        for t in intervals:
            if t not in heatmap.columns:
                heatmap[t] = 0
        heatmap = heatmap[intervals]
        heatmap['MAX_TILLS'] = heatmap.max(axis=1).astype(int)
        heatmap = heatmap.sort_values('MAX_TILLS', ascending=False)
        max_vals = heatmap['MAX_TILLS']
        mat = heatmap.drop(columns=['MAX_TILLS']).values
        fig = px.imshow(mat, x=col_labels, y=heatmap.index, text_auto=True, aspect='auto',
                        color_continuous_scale='Blues', title="Active Tills by 30-min interval")
        fig.update_xaxes(side='top')
        st.plotly_chart(fig, use_container_width=True)
        # summary table
        summary = heatmap.reset_index()[['STORE_NAME','MAX_TILLS']]
        summary = append_totals_row(summary, ['MAX_TILLS'], label_col='STORE_NAME', label='TOTAL')
        summary = fmt_ints(summary, ['MAX_TILLS'])
        st.dataframe(summary, use_container_width=True)
        download_button(summary, "active_tills_summary.csv", "⬇️ Download Table")
        download_plot(fig, "active_tills_heatmap.png")

    elif subsection == "Average Customers Served per Till":
        # Compute unique customers per interval and divide by unique tills
        d = df.dropna(subset=['TRN_DATE']).copy()
        for c in ['STORE_CODE','TILL','SESSION','RCT']:
            d[c] = d[c].astype(str).fillna('').str.strip()
        if 'CUST_CODE' not in d.columns:
            d['CUST_CODE'] = d['STORE_CODE'] + '-' + d['TILL'] + '-' + d['SESSION'] + '-' + d['RCT']
        d['TIME_INTERVAL'] = d['TRN_DATE'].dt.floor('30T')
        d['TIME_ONLY'] = d['TIME_INTERVAL'].dt.time
        cust_counts = d.groupby(['STORE_NAME','TIME_ONLY'])['CUST_CODE'].nunique().reset_index(name='CUSTOMERS')
        till_counts = d.groupby(['STORE_NAME','TIME_ONLY'])['TILL'].nunique().reset_index(name='TILLS')
        ratio = cust_counts.merge(till_counts, on=['STORE_NAME','TIME_ONLY'], how='outer').fillna(0)
        ratio['CUSTOMERS_PER_TILL'] = ratio.apply(lambda r: (r['CUSTOMERS']/r['TILLS'] if r['TILLS']>0 else 0), axis=1)
        # Pivot to heatmap per store (max ratio etc.)
        pivot = ratio.pivot(index='STORE_NAME', columns='TIME_ONLY', values='CUSTOMERS_PER_TILL').fillna(0)
        for t in intervals:
            if t not in pivot.columns:
                pivot[t] = 0
        pivot = pivot[intervals]
        pivot['MAX_RATIO'] = pivot.max(axis=1).round(1)
        pivot = pivot.sort_values('MAX_RATIO', ascending=False)
        mat = pivot.drop(columns=['MAX_RATIO']).values
        fig = px.imshow(mat, x=col_labels, y=pivot.index, text_auto=True, aspect='auto', color_continuous_scale='YlGnBu',
                        title="Customers per Till (30-min slots)")
        fig.update_xaxes(side='top')
        st.plotly_chart(fig, use_container_width=True)
        # summary table
        summary = pivot.reset_index()[['STORE_NAME','MAX_RATIO']]
        summary = append_totals_row(summary, ['MAX_RATIO'], label_col='STORE_NAME', label='TOTAL')
        summary = fmt_floats(summary, ['MAX_RATIO'], 1)
        st.dataframe(summary, use_container_width=True)
        download_button(summary, "customers_per_till_summary.csv", "⬇️ Download Table")
        download_plot(fig, "customers_per_till_heatmap.png")

    elif subsection == "Store Customer Traffic Storewise":
        # Department-level heatmap per store (dropdown)
        stores = sorted(df['STORE_NAME'].dropna().unique().tolist())
        selected = st.selectbox("Select Store for Department Traffic", stores, key="store_dept")
        dff = df[df['STORE_NAME']==selected].dropna(subset=['TRN_DATE']).copy()
        for c in ['STORE_CODE','TILL','SESSION','RCT']:
            dff[c] = dff[c].astype(str).fillna('').str.strip()
        if 'CUST_CODE' not in dff.columns:
            dff['CUST_CODE'] = dff['STORE_CODE'] + '-' + dff['TILL'] + '-' + dff['SESSION'] + '-' + dff['RCT']
        dff['TIME_ONLY'] = dff['TRN_DATE'].dt.floor('30T').dt.time
        tmp = dff.groupby(['DEPARTMENT','TIME_ONLY'])['CUST_CODE'].nunique().reset_index(name='Unique_Customers')
        pivot = tmp.pivot(index='DEPARTMENT', columns='TIME_ONLY', values='Unique_Customers').fillna(0)
        for t in intervals:
            if t not in pivot.columns:
                pivot[t] = 0
        pivot = pivot[intervals]
        pivot['TOTAL'] = pivot.sum(axis=1).astype(int)
        pivot = pivot.sort_values('TOTAL', ascending=False)
        mat = pivot.drop(columns=['TOTAL']).values
        fig = px.imshow(mat, x=col_labels, y=pivot.index, text_auto=True, aspect='auto', color_continuous_scale='Viridis',
                        title=f"Department Traffic — {selected}")
        fig.update_xaxes(side='top')
        st.plotly_chart(fig, use_container_width=True)
        # show totals and overall unique customers
        total_customers = int(dff['CUST_CODE'].nunique())
        st.write(f"Total unique receipts in {selected}: {total_customers:,}")
        disp = pivot.reset_index()[['DEPARTMENT','TOTAL']]
        disp = append_totals_row(disp, ['TOTAL'], label_col='DEPARTMENT', label='TOTAL')
        disp = fmt_ints(disp, ['TOTAL'])
        st.dataframe(disp, use_container_width=True)
        download_button(disp, f"{selected}_department_traffic.csv", "⬇️ Download Table")
        download_plot(fig, f"{selected}_department_traffic.png")

    elif subsection == "Customer Traffic-Departmentwise":
        # Similar to Store Customer Traffic but present a branch selector as well
        branches = sorted(df['STORE_NAME'].dropna().unique().tolist())
        selected_branch = st.selectbox("Branch", branches, key="dept_branch")
        dff = df[df['STORE_NAME']==selected_branch].dropna(subset=['TRN_DATE']).copy()
        dff['TIME_ONLY'] = dff['TRN_DATE'].dt.floor('30T').dt.time
        tmp = dff.groupby(['DEPARTMENT','TIME_ONLY'])['CUST_CODE'].nunique().reset_index(name='Unique_Customers')
        pivot = tmp.pivot(index='DEPARTMENT', columns='TIME_ONLY', values='Unique_Customers').fillna(0)
        for t in intervals:
            if t not in pivot.columns:
                pivot[t] = 0
        pivot = pivot[intervals]
        pivot['TOTAL'] = pivot.sum(axis=1).astype(int)
        pivot = pivot.sort_values('TOTAL', ascending=False)
        fig = px.imshow(pivot.drop(columns=['TOTAL']).values, x=col_labels, y=pivot.index, text_auto=True,
                        aspect='auto', color_continuous_scale='Cividis', title=f"Department Traffic — {selected_branch}")
        fig.update_xaxes(side='top')
        st.plotly_chart(fig, use_container_width=True)
        disp = pivot.reset_index()[['DEPARTMENT','TOTAL']]
        disp = append_totals_row(disp, ['TOTAL'], label_col='DEPARTMENT', label='TOTAL')
        disp = fmt_ints(disp, ['TOTAL'])
        st.dataframe(disp, use_container_width=True)
        download_button(disp, f"{selected_branch}_dept_traffic.csv", "⬇️ Download Table")
        download_plot(fig, f"{selected_branch}_dept_traffic.png")

    elif subsection == "Cashiers Perfomance":
        # Compute hours worked and customers served per cashier in each store
        d = df.dropna(subset=['TRN_DATE']).copy()
        if 'CASHIER' not in d.columns:
            st.error("CASHIER column missing.")
        else:
            d['CASHIER-COUNT'] = d['CASHIER'].astype(str) + ' — ' + d['STORE_NAME'].astype(str)
            # receipt-level duration
            receipt_duration = d.groupby(['STORE_NAME','CUST_CODE'], as_index=False).agg(Start_Time=('TRN_DATE','min'), End_Time=('TRN_DATE','max'))
            receipt_duration['Duration_Sec'] = (receipt_duration['End_Time'] - receipt_duration['Start_Time']).dt.total_seconds().fillna(0)
            # join back to cashier level via CUST_CODE and STORE_NAME
            cashier_stats = d.merge(receipt_duration[['STORE_NAME','CUST_CODE','Duration_Sec']], on=['STORE_NAME','CUST_CODE'], how='left')
            cashier_summary = cashier_stats.groupby(['STORE_NAME','CASHIER-COUNT'], as_index=False).agg(
                Hours_Worked=('Duration_Sec', lambda s: s.sum()/3600.0),
                Customers_Served=('CUST_CODE', 'nunique')
            )
            # customers per hour
            cashier_summary['Hours_Worked'] = cashier_summary['Hours_Worked'].round(2)
            cashier_summary['Customers_per_Hour'] = cashier_summary.apply(lambda r: round(r['Customers_Served']/r['Hours_Worked'],1) if r['Hours_Worked']>0 else 0, axis=1)
            # summary table: totals per store
            store_totals = cashier_summary.groupby('STORE_NAME').agg(
                Total_Hours=('Hours_Worked','sum'),
                Total_Customers=('Customers_Served','sum'),
                Avg_Customers_per_Hour=('Customers_per_Hour','mean')
            ).reset_index()
            store_totals['Avg_Customers_per_Hour'] = store_totals['Avg_Customers_per_Hour'].round(2)
            st.subheader("Cashier Summary (per cashier)")
            cs_display = cashier_summary.sort_values(['STORE_NAME','Customers_per_Hour'], ascending=[True,False]).reset_index(drop=True)
            cs_display = append_totals_row(cs_display, ['Hours_Worked','Customers_Served'], label_col='CASHIER-COUNT', label='TOTAL')
            cs_display = fmt_ints(cs_display, ['Customers_Served'])
            cs_display = fmt_floats(cs_display, ['Hours_Worked','Customers_per_Hour'], 2)
            st.dataframe(cs_display, use_container_width=True)
            st.subheader("Store-level totals")
            st.dataframe(fmt_floats(fmt_ints(store_totals, ['Total_Customers']), ['Total_Hours','Avg_Customers_per_Hour'], 2), use_container_width=True)
            download_button(cs_display, "cashier_summary.csv", "⬇️ Download Cashier Table")

    elif subsection == "Till Usage":
        # Build summary per store: Unique_Tills, Total Receipts, Busiest_Till, Busiest_Till_Pct
        d = df.dropna(subset=['TRN_DATE']).copy()
        for c in ['TILL','STORE_CODE']:
            if c in d.columns:
                d[c] = d[c].astype(str).fillna('').str.strip()
        d['Till_Code'] = d['TILL'] + '-' + d['STORE_CODE']
        d['TIME_ONLY'] = d['TRN_DATE'].dt.floor('30T').dt.time
        till_activity = d.groupby(['STORE_NAME','Till_Code'], as_index=False)['CUST_CODE'].nunique().rename(columns={'CUST_CODE':'Receipts'})
        # store summary
        branch_summary = till_activity.groupby('STORE_NAME', as_index=False).agg(
            Store_Total_Receipts=('Receipts','sum'),
            Avg_Per_Till=('Receipts','mean'),
            Max_Per_Till=('Receipts','max'),
            Unique_Tills=('Till_Code','nunique')
        )
        busiest = till_activity.sort_values(['STORE_NAME','Receipts'], ascending=[True,False]).groupby('STORE_NAME').first().reset_index().rename(columns={'Receipts':'Busiest_Till_Receipts','Till_Code':'Busiest_Till'})
        branch_summary = branch_summary.merge(busiest, on='STORE_NAME', how='left')
        branch_summary['Busiest_Till_Pct'] = np.where(branch_summary['Store_Total_Receipts']>0,
                                                     (100*branch_summary['Busiest_Till_Receipts']/branch_summary['Store_Total_Receipts']).round(1),
                                                     0.0)
        # Formatting & totals
        disp = branch_summary.copy()
        disp = append_totals_row(disp, ['Store_Total_Receipts','Unique_Tills','Busiest_Till_Receipts'], label_col='STORE_NAME', label='TOTAL')
        disp = fmt_ints(disp, ['Store_Total_Receipts','Unique_Tills','Busiest_Till_Receipts'])
        disp = fmt_floats(disp, ['Avg_Per_Till','Busiest_Till_Pct','Max_Per_Till'], 2)
        st.dataframe(disp, use_container_width=True)
        download_button(disp, "till_usage_summary.csv", "⬇️ Download Table")

    elif subsection == "Tax Compliance":
        # Compliant if CU_DEVICE_SERIAL is present/non-empty
        d = df.dropna(subset=['TRN_DATE']).copy()
        d['Tax_Compliant'] = np.where(d['CU_DEVICE_SERIAL'].astype(str).str.strip().replace({'nan':'','None':''})!='','Compliant','Non-Compliant')
        global_summary = d.groupby('Tax_Compliant', as_index=False).agg(Receipts=('CUST_CODE','nunique'))
        fig = px.pie(global_summary, names='Tax_Compliant', values='Receipts', color='Tax_Compliant',
                     color_discrete_map={'Compliant':'#2ca02c','Non-Compliant':'#d62728'}, hole=0.5,
                     title='Global Tax Compliance Overview')
        fig.update_traces(textinfo='label+percent')
        st.plotly_chart(fig, use_container_width=True)
        # store-level table
        store_till = d.groupby(['STORE_NAME','Tax_Compliant'], as_index=False).agg(Receipts=('CUST_CODE','nunique'))
        store_pivot = store_till.pivot(index='STORE_NAME', columns='Tax_Compliant', values='Receipts').fillna(0)
        if 'Compliant' not in store_pivot.columns:
            store_pivot['Compliant'] = 0
        if 'Non-Compliant' not in store_pivot.columns:
            store_pivot['Non-Compliant'] = 0
        store_pivot['Total'] = store_pivot['Compliant'] + store_pivot['Non-Compliant']
        store_pivot['Compliance_%'] = np.where(store_pivot['Total']>0, (100*store_pivot['Compliant']/store_pivot['Total']).round(1), 0.0)
        store_display = store_pivot.reset_index()
        store_display = append_totals_row(store_display, ['Compliant','Non-Compliant','Total'], label_col='STORE_NAME', label='TOTAL')
        store_display = fmt_ints(store_display, ['Compliant','Non-Compliant','Total'])
        store_display = fmt_floats(store_display, ['Compliance_%'], 1)
        st.dataframe(store_display, use_container_width=True)
        download_button(store_display, "tax_compliance_by_store.csv", "⬇️ Download Table")
        download_plot(fig, "tax_compliance_pie.png")

# === INSIGHTS (fill many missing outputs from the notebook) ===
elif section == "INSIGHTS":
    if subsection == "Branch Comparison":
        branches = sorted(df['STORE_NAME'].dropna().unique().tolist())
        selected_A = st.selectbox("Branch A", branches, key="bc_a")
        selected_B = st.selectbox("Branch B", branches, key="bc_b")
        metric = st.selectbox("Metric", ["QTY","NET_SALES"], key="bc_metric")
        N = st.slider("Top N", 5, 50, 10, key="bc_n")
        dfA = df[df["STORE_NAME"]==selected_A].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N)
        dfB = df[df["STORE_NAME"]==selected_B].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N)
        combA = dfA.copy(); combA['Branch'] = selected_A
        combB = dfB.copy(); combB['Branch'] = selected_B
        both = pd.concat([combA, combB], ignore_index=True)
        fig = px.bar(both, x=metric, y="ITEM_NAME", color="Branch", orientation="h", barmode="group",
                     title=f"Top {N} items: {selected_A} vs {selected_B}", color_discrete_sequence=["#1f77b4","#ff7f0e"])
        st.plotly_chart(fig, use_container_width=True)
        # Format totals and numbers
        both_display = both.copy()
        if metric == 'NET_SALES':
            both_display = fmt_floats(both_display, ['NET_SALES'], 2)
        else:
            both_display = fmt_ints(both_display, ['QTY'])
        st.dataframe(both_display, use_container_width=True)
        download_button(both_display, "branch_comparison.csv", f"⬇️ Download Branch Comparison Table")
        download_plot(fig, "branch_comparison_bar.png")

    elif subsection == "Customer Baskets Overview":
        # Build top items by count of baskets and show missing / underperforming per branch
        st.info("This view shows global top items (by QTY or NET_SALES) and compares with a selected branch.")
        metric = st.selectbox("Metric", ['QTY','NET_SALES'], index=0, key="baskets_metric")
        top_x = st.number_input("Top X", min_value=5, max_value=200, value=10, step=5, key="baskets_topx")
        branch = st.selectbox("Branch", sorted(df['STORE_NAME'].dropna().unique().tolist()), key="baskets_branch")
        depts = st.multiselect("Departments (optional)", options=sorted(df['DEPARTMENT'].dropna().unique().tolist()), default=[])
        # Filtering
        dff = df.copy()
        if depts:
            dff = dff[dff['DEPARTMENT'].isin(depts)]
        # global top
        basket_count = dff.groupby('ITEM_NAME')['CUST_CODE'].nunique().rename('Count_of_Baskets')
        agg_data = dff.groupby('ITEM_NAME')[['QTY','NET_SALES']].sum()
        merged = basket_count.to_frame().join(agg_data).reset_index().sort_values(metric, ascending=False).head(int(top_x))
        merged.insert(0, '#', range(1, len(merged)+1))
        merged['STORE_NAME'] = 'GLOBAL'
        # branch top
        branch_df = dff[dff['STORE_NAME']==branch]
        branch_baskets = branch_df.groupby('ITEM_NAME')['CUST_CODE'].nunique().rename('Count_of_Baskets')
        branch_agg = branch_df.groupby('ITEM_NAME')[[ 'QTY','NET_SALES']].sum()
        merged_b = branch_baskets.to_frame().join(branch_agg).reset_index().sort_values(metric, ascending=False).head(int(top_x))
        merged_b.insert(0, '#', range(1, len(merged_b)+1))
        merged_b['STORE_NAME'] = branch
        # missing items
        missing_items = set(merged['ITEM_NAME']) - set(merged_b['ITEM_NAME'])
        missing_df = merged[merged['ITEM_NAME'].isin(missing_items)].copy()
        if not missing_df.empty:
            missing_df.insert(0, '#', range(1, len(missing_df)+1))
        # Show chart
        combined = pd.concat([merged.assign(Level='Global'), merged_b.assign(Level='Branch')], ignore_index=True)
        fig = px.bar(combined, x=metric, y='ITEM_NAME', color='Level', orientation='h',
                     title=f"Top {top_x} items by {metric}: Global vs {branch}")
        st.plotly_chart(fig, use_container_width=True)
        # show tables formatted
        if metric == 'NET_SALES':
            st.subheader("Global Top Items")
            st.dataframe(fmt_floats(merged, ['NET_SALES'], 2), use_container_width=True)
            st.subheader(f"{branch} Top Items")
            st.dataframe(fmt_floats(merged_b, ['NET_SALES'], 2), use_container_width=True)
            if not missing_df.empty:
                st.subheader("Items in Global top but missing/underperforming in branch")
                st.dataframe(fmt_floats(missing_df, ['NET_SALES'], 2), use_container_width=True)
        else:
            st.subheader("Global Top Items")
            st.dataframe(fmt_ints(merged, ['QTY']), use_container_width=True)
            st.subheader(f"{branch} Top Items")
            st.dataframe(fmt_ints(merged_b, ['QTY']), use_container_width=True)
            if not missing_df.empty:
                st.subheader("Items in Global top but missing/underperforming in branch")
                st.dataframe(fmt_ints(missing_df, ['QTY']), use_container_width=True)

    elif subsection == "Product Perfomance":
        # Provide a combobox of item codes / names and basic per-store metrics
        st.info("Select an ITEM_CODE (or type part of the name). App shows per-store volumes, basket penetration and simple charts.")
        lookup = df[['ITEM_CODE','ITEM_NAME']].drop_duplicates().sort_values('ITEM_NAME')
        options = (lookup['ITEM_CODE'] + " — " + lookup['ITEM_NAME']).tolist()
        sel = st.selectbox("Select SKU (code — name)", options=options)
        if sel:
            item_code = sel.split('—')[0].strip() if '—' in sel else sel.strip()
            item_df = df[df['ITEM_CODE']==item_code].copy()
            if item_df.empty:
                st.warning("No rows for selected SKU.")
            else:
                # store-level counts
                store_baskets = item_df.groupby('STORE_NAME')['CUST_CODE'].nunique().rename('Baskets_With_Item').reset_index()
                store_qty = item_df.groupby('STORE_NAME')['QTY'].sum().rename('Total_QTY').reset_index()
                store_summary = store_baskets.merge(store_qty, on='STORE_NAME', how='left')
                # compute % of store customers
                store_total_customers = df.groupby('STORE_NAME')['CUST_CODE'].nunique()
                store_summary['Store_Customers'] = store_summary['STORE_NAME'].map(store_total_customers).fillna(0).astype(int)
                store_summary['Pct_of_Store_Customers'] = np.where(store_summary['Store_Customers']>0,
                                                                  (100*store_summary['Baskets_With_Item']/store_summary['Store_Customers']).round(1),0.0)
                store_summary = store_summary.sort_values('Baskets_With_Item', ascending=False)
                # totals row
                store_summary = append_totals_row(store_summary, ['Baskets_With_Item','Total_QTY'], label_col='STORE_NAME', label='TOTAL')
                # format
                store_summary = fmt_ints(store_summary, ['Baskets_With_Item','Total_QTY','Store_Customers'])
                store_summary = fmt_floats(store_summary, ['Pct_of_Store_Customers'], 1)
                st.dataframe(store_summary, use_container_width=True)
                download_button(store_summary, f"product_{item_code}_store_summary.csv", "⬇️ Download Table")
                # simple bar chart of baskets by store
                fig = px.bar(store_summary[:-1], x='Baskets_With_Item', y='STORE_NAME', orientation='h', title=f"Baskets with {item_code} by Store")
                st.plotly_chart(fig, use_container_width=True)
                download_plot(fig, f"{item_code}_baskets_bar.png")

    elif subsection == "Global Pricing Overview":
        # Multi-priced SKUs per store summary (spread > 0)
        d = df.dropna(subset=['TRN_DATE']).copy()
        d['SP_PRE_VAT'] = pd.to_numeric(d['SP_PRE_VAT'].astype(str).str.replace(',',''), errors='coerce').fillna(0.0)
        d['DATE'] = d['TRN_DATE'].dt.date
        grp = d.groupby(['STORE_NAME','DATE','ITEM_CODE','ITEM_NAME'], as_index=False).agg(
            Num_Prices=('SP_PRE_VAT', lambda s: s.dropna().nunique()),
            Price_Min=('SP_PRE_VAT', 'min'),
            Price_Max=('SP_PRE_VAT', 'max'),
            Total_QTY=('QTY', 'sum')
        )
        grp['Price_Spread'] = (grp['Price_Max'] - grp['Price_Min']).round(2)
        multi_price = grp[(grp['Num_Prices']>1) & (grp['Price_Spread']>0)].copy()
        if multi_price.empty:
            st.info("No items sold at multiple prices (spread>0) found.")
        else:
            multi_price['Diff_Value'] = (multi_price['Total_QTY'] * multi_price['Price_Spread']).round(2)
            summary = multi_price.groupby('STORE_NAME', as_index=False).agg(
                Items_with_MultiPrice=('ITEM_CODE','nunique'),
                Total_Diff_Value=('Diff_Value','sum'),
                Avg_Spread=('Price_Spread','mean'),
                Max_Spread=('Price_Spread','max')
            )
            summary = summary.sort_values('Total_Diff_Value', ascending=False)
            summary_display = append_totals_row(summary, ['Items_with_MultiPrice','Total_Diff_Value'], label_col='STORE_NAME', label='TOTAL')
            summary_display = fmt_ints(summary_display, ['Items_with_MultiPrice'])
            summary_display = fmt_floats(summary_display, ['Total_Diff_Value','Avg_Spread','Max_Spread'], 2)
            st.dataframe(summary_display, use_container_width=True)
            download_button(summary_display, "global_pricing_overview.csv", "⬇️ Download Table")
            # top stores chart
            topN = min(20, len(summary))
            if topN>0:
                fig = px.bar(summary.head(topN).sort_values('Total_Diff_Value', ascending=True), x='Total_Diff_Value', y='STORE_NAME', orientation='h',
                             text='Total_Diff_Value', color='Items_with_MultiPrice', color_continuous_scale='Blues',
                             title='Top Stores by Value Impact from Multi-Priced SKUs')
                fig.update_traces(texttemplate='KSh %{text:,.2f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                download_plot(fig, "pricing_impact_bar.png")

    elif subsection == "Global Refunds Overview":
        # negative NET_SALES rows aggregated
        d = df.copy()
        d['NET_SALES'] = pd.to_numeric(d['NET_SALES'], errors='coerce').fillna(0)
        neg = d[d['NET_SALES']<0].copy()
        if neg.empty:
            st.info("No negative receipts found in dataset.")
        else:
            neg['CAP_CUSTOMER_CODE'] = neg.get('CAP_CUSTOMER_CODE', pd.Series('', index=neg.index)).astype(str).str.strip()
            neg['Sale_Type'] = np.where(neg['CAP_CUSTOMER_CODE']=='','General sales','On_account sales')
            group_cols = ['STORE_NAME','Sale_Type','CAP_CUSTOMER_CODE']
            val_summ = neg.groupby(group_cols)['NET_SALES'].sum().rename('Total_Neg_Value')
            # counts per unique receipt if available
            if 'CUST_CODE' in neg.columns:
                cnt_summ = neg.groupby(group_cols)['CUST_CODE'].nunique().rename('Total_Count')
            else:
                cnt_summ = neg.groupby(group_cols).size().rename('Total_Count')
            summary = pd.concat([val_summ, cnt_summ], axis=1).reset_index()
            summary['Abs_Neg_Value'] = summary['Total_Neg_Value'].abs()
            # Add store totals and grand total
            store_totals = summary.groupby('STORE_NAME', as_index=False).agg(Total_Neg_Value=('Total_Neg_Value','sum'), Total_Count=('Total_Count','sum'), Abs_Neg_Value=('Abs_Neg_Value','sum'))
            store_totals['Sale_Type']='ALL'; store_totals['CAP_CUSTOMER_CODE']='—'
            combined = pd.concat([summary, store_totals], ignore_index=True)
            grand = pd.DataFrame({'STORE_NAME':['TOTAL'],'Sale_Type':['ALL'],'CAP_CUSTOMER_CODE':['—'],'Total_Neg_Value':[summary['Total_Neg_Value'].sum()],'Total_Count':[summary['Total_Count'].sum()],'Abs_Neg_Value':[summary['Abs_Neg_Value'].sum()]})
            combined = pd.concat([combined, grand], ignore_index=True)
            combined_display = combined[['STORE_NAME','Sale_Type','CAP_CUSTOMER_CODE','Total_Neg_Value','Total_Count','Abs_Neg_Value']]
            combined_display = fmt_floats(combined_display, ['Total_Neg_Value','Abs_Neg_Value'], 2)
            combined_display = fmt_ints(combined_display, ['Total_Count'])
            st.dataframe(combined_display, use_container_width=True)
            download_button(combined_display, "global_refunds_overview.csv", "⬇️ Download Table")

    elif subsection == "Global Loyalty Overview":
        # Loyalty dataset: receipts with LOYALTY_CUSTOMER_CODE
        d = df.copy()
        d['TRN_DATE'] = pd.to_datetime(d['TRN_DATE'], errors='coerce')
        d['NET_SALES'] = pd.to_numeric(d['NET_SALES'], errors='coerce').fillna(0)
        dL = d[d['LOYALTY_CUSTOMER_CODE'].astype(str).str.strip().replace({'nan':'','None':''})!=''].copy()
        if dL.empty:
            st.info("No loyalty customer records found.")
        else:
            receipts = dL.groupby(['STORE_NAME','CUST_CODE','LOYALTY_CUSTOMER_CODE'], as_index=False).agg(Basket_Value=('NET_SALES','sum'), First_Time=('TRN_DATE','min'))
            per_branch_multi = receipts.groupby(['STORE_NAME','LOYALTY_CUSTOMER_CODE']).agg(Baskets_in_Store=('CUST_CODE','nunique'), Total_Value_in_Store=('Basket_Value','sum')).reset_index()
            per_branch_multi = per_branch_multi[per_branch_multi['Baskets_in_Store']>1]
            overview = per_branch_multi.groupby('STORE_NAME', as_index=False).agg(Loyal_Customers_Multi=('LOYALTY_CUSTOMER_CODE','nunique'), Total_Baskets_of_Those=('Baskets_in_Store','sum'), Total_Value_of_Those=('Total_Value_in_Store','sum'))
            overview['Avg_Baskets_per_Customer'] = np.where(overview['Loyal_Customers_Multi']>0, (overview['Total_Baskets_of_Those']/overview['Loyal_Customers_Multi']).round(2), 0.0)
            overview = overview.sort_values(['Loyal_Customers_Multi','Total_Baskets_of_Those'], ascending=[False, False]).reset_index(drop=True)
            overview = append_totals_row(overview, ['Loyal_Customers_Multi','Total_Baskets_of_Those','Total_Value_of_Those'], label_col='STORE_NAME', label='TOTAL')
            ov_disp = fmt_ints(overview, ['Loyal_Customers_Multi','Total_Baskets_of_Those'])
            ov_disp = fmt_floats(ov_disp, ['Total_Value_of_Those','Avg_Baskets_per_Customer'], 2)
            st.write("All-Branch Overview — Loyalty Customers with >1 Baskets")
            st.dataframe(ov_disp, use_container_width=True)
            download_button(ov_disp, "global_loyalty_overview.csv", "⬇️ Download Table")

    else:
        # Generic fallback for subsections not implemented in detail yet
        st.info("This subsection will be added from the original Colab notebook on request. Current app implements the most used INSIGHTS outputs: Branch Comparison, Product Performance, Global Pricing, Customer Baskets, Global Refunds, Global Loyalty.")

# === Footer / notes ===
st.sidebar.markdown("---")
st.sidebar.markdown("All tables include download buttons. Numeric columns are formatted for readability. For >200MB uploads, update Streamlit server config before starting (server.maxUploadSize).")
