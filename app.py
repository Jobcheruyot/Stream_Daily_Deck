import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import io

# --- Wider sidebar CSS ---
st.markdown("""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
        min-width: 325px;
        max-width: 450px;
    }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="Superdeck Analytics Dashboard", initial_sidebar_state="expanded")
st.title("🦸 Superdeck Analytics Dashboard")
st.markdown("> Upload your sales CSV, then choose a main section and subsection for live analytics.")

# --- SIDEBAR: Upload block ---
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV (up to 500MB, check server settings)", type="csv")
if uploaded is None:
    st.info("Please upload a dataset to proceed.")
    st.stop()

@st.cache_data(show_spinner=True)
def load_and_prepare(uploaded):
    df = pd.read_csv(uploaded, on_bad_lines='skip', low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for col in ['TRN_DATE', 'ZED_DATE']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    for nc in numeric_cols:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors='coerce').fillna(0)
    idcols = ['STORE_CODE', 'TILL', 'SESSION', 'RCT']
    for col in idcols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('').str.strip()
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
    return df

df = load_and_prepare(uploaded)

@st.cache_data
def get_time_grid():
    start_time = pd.Timestamp("00:00:00")
    intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
    col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
    return intervals, col_labels
intervals, col_labels = get_time_grid()

def download_button(obj, filename, label, use_xlsx=False):
    if use_xlsx:
        towrite = io.BytesIO()
        obj.to_excel(towrite, encoding="utf-8", index=False, engine='openpyxl')
        towrite.seek(0)
        st.download_button(label, towrite, file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.download_button(label, obj.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")

def download_plot(fig, filename):
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600)
        st.download_button("⬇️ Download Plot as PNG", img_bytes, filename=filename, mime="image/png")
    except Exception:
        st.warning("Plot download not available (kaleido not installed?)")

# --- MAIN SECTION + SUBSECTION dropdowns ---
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

# --- SALES ---
if section == "SALES":
    if subsection == "Global sales Overview":
        gs = df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum()
        gs['NET_SALES_M'] = gs['NET_SALES'] / 1_000_000
        gs['PCT'] = (gs['NET_SALES'] / gs['NET_SALES'].sum()) * 100
        fig = go.Figure(data=[go.Pie(
            labels=[f"{row['SALES_CHANNEL_L1']} ({row['PCT']:.1f}%| {row['NET_SALES_M']:.1f}M)" for _,row in gs.iterrows()],
            values=gs['NET_SALES_M'], hole=0.65,
            marker=dict(colors=px.colors.qualitative.Plotly),
            text=[f"{p:.1f}%" for p in gs['PCT']],
            textinfo='text'
        )])
        fig.update_layout(title="SALES CHANNEL TYPE — Global Overview", height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(gs, use_container_width=True)
        download_button(gs, "global_sales_overview.csv", "⬇️ Download Table")
        download_plot(fig, "global_sales_overview.png")

    elif subsection == "Global Net Sales Distribution by Sales Channel":
        g2 = df.groupby('SALES_CHANNEL_L2', as_index=False)['NET_SALES'].sum()
        g2['NET_SALES_M'] = g2['NET_SALES']/1_000_000
        g2['PCT'] = g2['NET_SALES']/g2['NET_SALES'].sum()*100
        colors = px.colors.qualitative.Vivid
        fig = px.pie(g2, names='SALES_CHANNEL_L2', values='NET_SALES_M', color='SALES_CHANNEL_L2',
             color_discrete_sequence=colors, title="Net Sales by Sales Mode (L2)", hole=0.6)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(g2, use_container_width=True)
        download_button(g2, "sales_channel_l2.csv", "⬇️ Download Table")
        download_plot(fig, "sales_channel_l2_pie.png")

    elif subsection == "Global Net Sales Distribution by SHIFT":
        sh = df.groupby('SHIFT', as_index=False)['NET_SALES'].sum()
        sh['PCT'] = sh['NET_SALES']/sh['NET_SALES'].sum()*100
        colors = px.colors.qualitative.Bold
        fig = px.pie(sh, names='SHIFT', values='NET_SALES', color='SHIFT',
                color_discrete_sequence=colors, title="Net Sales by Shift", hole=0.6)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sh, use_container_width=True)
        download_button(sh, "shift_sales.csv", "⬇️ Download Table")
        download_plot(fig, "shift_sales_pie.png")

    elif subsection == "Night vs Day Shift Sales Ratio — Stores with Night Shifts":
        ns = df[df['SHIFT'].str.upper().str.contains('NIGHT', na=False)]['STORE_NAME'].unique()
        df_nd = df[df['STORE_NAME'].isin(ns)].copy()
        df_nd['Shift_Bucket'] = np.where(df_nd['SHIFT'].str.upper().str.contains('NIGHT', na=False),'Night','Day')
        ratio_df = df_nd.groupby(['STORE_NAME','Shift_Bucket'], as_index=False)['NET_SALES'].sum()
        store_totals = ratio_df.groupby('STORE_NAME')['NET_SALES'].transform('sum')
        ratio_df['PCT'] = 100 * ratio_df['NET_SALES'] / store_totals
        pivot_df = ratio_df.pivot(index='STORE_NAME', columns='Shift_Bucket', values='PCT').fillna(0)
        pivot_sorted = pivot_df.sort_values(by='Night', ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pivot_sorted['Night'], y=pivot_sorted.index, orientation='h',
            name='Night', marker_color='#d62728', text=pivot_sorted['Night'], textposition='outside'
        ))
        fig.add_trace(go.Bar(
            x=pivot_sorted['Day'], y=pivot_sorted.index, orientation='h',
            name='Day', marker_color='#1f77b4', text=pivot_sorted['Day'], textposition='outside'
        ))
        fig.update_layout(barmode='group', title="Night vs Day % (by Store)",
                          xaxis_title="% Night Sales", yaxis_title="Store", height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pivot_sorted, use_container_width=True)
        download_button(pivot_sorted.reset_index(), "night_day_ratio.csv", "⬇️ Download Table")
        download_plot(fig, "night_day_ratio_bar.png")

    elif subsection == "Global Day vs Night Sales — Only Stores with NIGHT Shift":
        ns = df[df['SHIFT'].str.upper().str.contains('NIGHT', na=False)]['STORE_NAME'].unique()
        df_nd = df[df['STORE_NAME'].isin(ns)].copy()
        df_nd['Shift_Bucket'] = np.where(df_nd['SHIFT'].str.upper().str.contains('NIGHT', na=False),'Night','Day')
        gb = df_nd.groupby('Shift_Bucket', as_index=False)['NET_SALES'].sum()
        gb['PCT'] = 100 * gb['NET_SALES'] / gb['NET_SALES'].sum()
        fig = px.pie(gb, names='Shift_Bucket', values='NET_SALES', color='Shift_Bucket',
            color_discrete_map={'Night':'#d62728','Day':'#1f77b4'}, hole=0.6, title="Global Day vs Night Sales (NIGHT Shift only)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(gb, use_container_width=True)
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
            top30 = store_chan[store_chan["RANK"]==2].sort_values("PCT", ascending=False).head(30)
            fig = go.Figure(go.Scatter(
                x=top30["PCT"], y=top30["STORE_NAME"],
                mode="markers+lines",
                marker=dict(size=14, color="#1f77b4"), name="2nd Channel %",
                line=dict(color="#aaaaaa", width=2)
            ))
            fig.update_layout(title="Top 30 Stores by 2nd-Highest Channel %", xaxis_title="2nd Channel %", yaxis_title="Store", height=700)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top30, use_container_width=True)
            download_button(top30.reset_index(), "top30_2nd_channel.csv", "⬇️ Download Table")
            download_plot(fig, "top30_lollipop.png")

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
            bottom30 = store_chan[store_chan["RANK"]==2].sort_values("PCT", ascending=True).head(30)
            fig = go.Figure(go.Scatter(
                x=bottom30["PCT"], y=bottom30["STORE_NAME"],
                mode="markers+lines",
                marker=dict(size=14, color="#d62728"), name="2nd Channel %",
                line=dict(color="gray", width=2)
            ))
            fig.update_layout(title="Bottom 30 Stores by 2nd-Highest Channel %", xaxis_title="2nd Channel %", yaxis_title="Store", height=700)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(bottom30, use_container_width=True)
            download_button(bottom30.reset_index(), "bottom30_2nd_channel.csv", "⬇️ Download Table")
            download_plot(fig, "bottom30_lollipop.png")

    elif subsection == "Stores Sales Summary":
        if 'GROSS_SALES' not in df.columns and 'VAT_AMT' in df.columns:
            df['GROSS_SALES'] = df['NET_SALES'] + df['VAT_AMT']
        ss = df.groupby('STORE_NAME', as_index=False)[['NET_SALES','GROSS_SALES']].sum()
        ss['Customer_Numbers'] = df.groupby('STORE_NAME')['CUST_CODE'].nunique().values
        ss['% Contribution'] = (ss['GROSS_SALES']/ss['GROSS_SALES'].sum()*100).round(2)
        ss = ss.sort_values('GROSS_SALES', ascending=False).reset_index(drop=True)
        st.dataframe(ss, use_container_width=True)
        download_button(ss, "stores_sales_summary.csv", "⬇️ Download Table")
        fig = px.bar(ss, x="GROSS_SALES", y="STORE_NAME", orientation="h", color="% Contribution", color_continuous_scale='Blues',
                     text="GROSS_SALES", title="Gross Sales by Store")
        st.plotly_chart(fig, use_container_width=True)
        download_plot(fig, "store_sales_summary_bar.png")

# --- OPERATIONS ---
elif section == "OPERATIONS":
    if subsection == "Customer Traffic-Storewise":
        stores = df["STORE_NAME"].dropna().unique().tolist()
        selected_store = st.selectbox("Select Store", stores)
        dff = df[df["STORE_NAME"]==selected_store].copy()
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        dff = dff.dropna(subset=['TRN_DATE'])
        for c in ["STORE_CODE","TILL","SESSION","RCT"]:
            dff[c] = dff[c].astype(str).fillna('').str.strip()
        dff['CUST_CODE'] = dff['STORE_CODE']+'-'+dff['TILL']+'-'+dff['SESSION']+'-'+dff['RCT']
        dff['TIME_ONLY'] = dff['TRN_DATE'].dt.floor('30T').dt.time
        heat = dff.groupby('TIME_ONLY')['CUST_CODE'].nunique().reindex(intervals, fill_value=0)
        fig = px.bar(x=col_labels, y=heat.values, labels={"x":"Time","y":"Receipts"}, text=heat.values,
                     color_discrete_sequence=['#3192e1'], title=f"Receipts by Time - {selected_store}")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(heat, use_container_width=True)
        download_button(heat.reset_index(), "customer_traffic_storewise.csv", "⬇️ Download Table")
        download_plot(fig, "customer_traffic_storewise.png")

    # ...add other Operations outputs from your original code, following these patterns...

# --- INSIGHTS ---
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
        st.dataframe(both, use_container_width=True)
        download_button(both, "branch_comparison.csv", f"⬇️ Download Branch Comparison Table")
        download_plot(fig, "branch_comparison_bar.png")

    # ...add other INSIGHTS outputs from your original code, following these patterns...

st.sidebar.markdown("---\nSelect a main section and subsection. All tables and plots are downloadable. Sidebar auto-expands for easier selection.")
