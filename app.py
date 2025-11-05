import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(layout="wide", page_title="Superdeck Analytics", initial_sidebar_state="expanded")
st.title("🦸 Superdeck Analytics Dashboard")
st.markdown("> Upload your sales CSV and explore all analytics of the original Superdeck notebook. Use the tabs and dropdowns for deep-dive.")

# ---------------- Data Upload and Preparation ----------------
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload your data as CSV", type="csv")
if uploaded is None:
    st.info("Please upload a dataset.")
    st.stop()

df = pd.read_csv(uploaded, on_bad_lines='skip', low_memory=False)
date_cols = ['TRN_DATE', 'ZED_DATE']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
for nc in numeric_cols:
    if nc in df.columns:
        df[nc] = pd.to_numeric(df[nc], errors='coerce').fillna(0)

# For-indexed time bucket features (for heatmaps etc)
def get_time_grid():
    start_time = pd.Timestamp("00:00:00")
    intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
    col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
    return intervals, col_labels

intervals, col_labels = get_time_grid()

# ---------------- Tabs Structure (broadly matching your notebook) ----------------
tab_names = [
    "Sales Channel Overview",
    "Net Sales by Mode",
    "Net Sales by Shift",
    "Night vs Day Ratio (Store)",
    "Global Day/Night (Pie)",
    "2nd Channel Share (Lollipop)",
    "Sales Summary",
    "Customer Traffic (Heatmap)",
    "Active Tills Heatmap",
    "Customers per Till",
    "Dept Traffic Heatmap",
    "Tax Compliance",
    "Top Items/Branch",
    "Branch Comparison",
    "Pricing Spread",
    "Refunds",
]
tabs = st.tabs(tab_names)

# ----------- 1. Sales Channel Overview (Pie - L1) -----------
with tabs[0]:
    st.header("Sales Channel Type (L1) Overview")
    if "SALES_CHANNEL_L1" in df.columns and "NET_SALES" in df.columns:
        sales = df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
        sales['NET_SALES_M'] = sales['NET_SALES'] / 1_000_000
        sales['PCT'] = (sales['NET_SALES'] / sales['NET_SALES'].sum()) * 100
        fig = go.Figure(go.Pie(
            labels=[f"{row['SALES_CHANNEL_L1']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f} M)" for _, row in sales.iterrows()],
            values=sales['NET_SALES_M'],
            hole=0.65,
            text=[f"{p:.1f}%" for p in sales['PCT']],
            textinfo='text',
            marker=dict(colors=px.colors.qualitative.Plotly)
        ))
        fig.update_layout(title="<b>SALES CHANNEL TYPE — Global Overview</b>")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing SALES_CHANNEL_L1 or NET_SALES column.")

# ----------- 2. Net Sales by Mode (Pie - L2) -----------
with tabs[1]:
    st.header("Net Sales by Mode (L2)")
    if "SALES_CHANNEL_L2" in df.columns and "NET_SALES" in df.columns:
        sales = df.groupby('SALES_CHANNEL_L2', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
        sales['NET_SALES_M'] = sales['NET_SALES'] / 1_000_000
        sales['PCT'] = (sales['NET_SALES'] / sales['NET_SALES'].sum()) * 100
        total_sales_m = sales['NET_SALES_M'].sum()
        fig = go.Figure(go.Pie(
            labels=[f"{row['SALES_CHANNEL_L2']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f} M)" for _, row in sales.iterrows()],
            values=sales['NET_SALES_M'],
            hole=0.65,
            text=[f"{p:.1f}%" for p in sales['PCT']],
            textinfo='text',
            marker=dict(colors=px.colors.qualitative.Plotly)
        ))
        fig.update_layout(
            title=f"<b>Global Net Sales Distribution by Sales Mode (L2)</b>",
            legend_title_text=f"Sales Mode (% | KSh M)\n<b>Total Sales: {total_sales_m:,.1f} M</b>"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing SALES_CHANNEL_L2 or NET_SALES column.")

# ----------- 3. Net Sales by Shift (Pie) -----------
with tabs[2]:
    st.header("Net Sales by SHIFT")
    if "SHIFT" in df.columns and "NET_SALES" in df.columns:
        shift_sales = df.groupby('SHIFT', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
        shift_sales['PCT'] = (shift_sales['NET_SALES'] / shift_sales['NET_SALES'].sum()) * 100
        colors = px.colors.qualitative.Plotly
        fig = go.Figure(go.Pie(
            labels=[f"{row['SHIFT']} ({row['PCT']:.1f}%)" for _, row in shift_sales.iterrows()],
            values=shift_sales['NET_SALES'],
            hole=0.65,
            text=[f"{p:.1f}%" for p in shift_sales['PCT']],
            textinfo='text',
            marker=dict(colors=colors)
        ))
        fig.update_layout(
            title="<b>Global Net Sales Distribution by SHIFT</b>"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing SHIFT or NET_SALES.")

# ----------- 4. Night vs Day Ratio (Store, Horizontal Bar) -----------
with tabs[3]:
    st.header("Night vs Day Sales Ratio per Store")
    req = ["SHIFT", "STORE_NAME", "NET_SALES"]
    if all(x in df.columns for x in req):
        night_stores = df[df['SHIFT'].str.upper().str.contains("NIGHT", na=False)]['STORE_NAME'].unique()
        df_nd = df[df['STORE_NAME'].isin(night_stores)].copy()
        df_nd['Shift_Bucket'] = np.where(df_nd['SHIFT'].str.upper().str.contains('NIGHT', na=False), 'Night', 'Day')
        ratio_df = df_nd.groupby(['STORE_NAME','Shift_Bucket'], as_index=False)['NET_SALES'].sum()
        store_totals = ratio_df.groupby('STORE_NAME')['NET_SALES'].transform('sum')
        ratio_df['PCT'] = 100 * ratio_df['NET_SALES'] / store_totals
        pivot_df = ratio_df.pivot(index='STORE_NAME', columns='Shift_Bucket', values='PCT').fillna(0)
        # Horizontal bar: store, sorted by Night%
        pivot_sorted = pivot_df.sort_values(by='Night', ascending=False)
        st.bar_chart(pivot_sorted[['Night','Day']], height=(40+16*len(pivot_sorted)))
        st.write("Highest 'Night' % stores are top.")
    else:
        st.warning(f"One or more required columns missing: {req}")

# ----------- 5. Global Day vs Night Pie -----------
with tabs[4]:
    st.header("Global Day vs Night Sales (Night Stores Only)")
    req = ["SHIFT", "STORE_NAME", "NET_SALES"]
    if all(x in df.columns for x in req):
        night_stores = df[df['SHIFT'].str.upper().str.contains('NIGHT', na=False)]['STORE_NAME'].unique()
        df_nd = df[df['STORE_NAME'].isin(night_stores)].copy()
        df_nd['Shift_Bucket'] = np.where(df_nd['SHIFT'].str.upper().str.contains('NIGHT', na=False), 'Night', 'Day')
        gb = df_nd.groupby('Shift_Bucket', as_index=False)['NET_SALES'].sum()
        gb['PCT'] = 100 * gb['NET_SALES'] / gb['NET_SALES'].sum()
        legend_labels = [f"{b} ({p:.1f}%)" for b, p in zip(gb['Shift_Bucket'], gb['PCT'])]
        fig = go.Figure(go.Pie(
            labels=legend_labels,
            values=gb['NET_SALES'],
            hole=0.65,
            text=[f"{p:.1f}%" for p in gb['PCT']],
            textinfo='text',
            marker=dict(colors=['#1f77b4', '#d62728'], line=dict(color='white', width=1)),
            sort=False
        ))
        fig.update_layout(title="<b>Day vs Night (NIGHT shift stores)</b>")
        st.plotly_chart(fig, use_container_width=True)

# ----------- 6. 2nd-Highest Channel Share Lollipop Bar -----------
with tabs[5]:
    st.header("2nd-Highest Channel Share (Storewise Lollipop)")
    req = {"STORE_NAME", "SALES_CHANNEL_L1", "NET_SALES"}
    if req.issubset(df.columns):
        data = df.copy()
        data["NET_SALES"] = pd.to_numeric(data["NET_SALES"], errors="coerce").fillna(0)
        store_chan = data.groupby(["STORE_NAME", "SALES_CHANNEL_L1"], as_index=False)["NET_SALES"].sum()
        store_tot = store_chan.groupby("STORE_NAME")["NET_SALES"].transform("sum")
        store_chan["PCT"] = 100 * store_chan["NET_SALES"] / store_tot
        store_chan = store_chan.sort_values(["STORE_NAME", "PCT"], ascending=[True, False])
        store_chan["RANK"] = store_chan.groupby("STORE_NAME").cumcount() + 1
        n = st.slider("Top N", 5, 60, 30)
        plot_df = store_chan[store_chan["RANK"]==2].sort_values("PCT", ascending=False).head(n)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df["PCT"],
            y=plot_df["STORE_NAME"],
            mode='markers+lines',
            marker=dict(size=13, color="#1f77b4", line=dict(width=2, color="gray")),
            name="2nd Channel %"
        ))
        fig.update_layout(
            title="Top Stores by 2nd-Highest Channel %",
            xaxis_title="2nd Channel % of Store Net Sales",
            yaxis_title="Store",
            height=(36*n+150 if n>10 else 600),
            margin=dict(l=220, r=60, t=90, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------- 7. Sales Summary Table & Visual (less table, more bar) -----------
with tabs[6]:
    st.header("Sales Summary (Net, Gross & Contribution, Customer #)")
    if 'GROSS_SALES' not in df.columns and 'VAT_AMT' in df.columns:
        df['GROSS_SALES'] = df['NET_SALES'] + df['VAT_AMT']
    if "STORE_NAME" in df.columns:
        group = df.groupby("STORE_NAME", as_index=False).agg(
            NET_SALES=('NET_SALES','sum'),
            GROSS_SALES=('GROSS_SALES','sum'),
            Customer_Numbers=('CUST_CODE','nunique')
        )
        group['%_Contribution'] = (group['GROSS_SALES'] / group['GROSS_SALES'].sum() * 100).round(2)
        group = group.sort_values('GROSS_SALES', ascending=False)
        fig = px.bar(group, x="GROSS_SALES", y="STORE_NAME", orientation="h",
                     title="Gross Sales by Store", color="%_Contribution",
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(group)
    else:
        st.warning("STORE_NAME missing.")

# ----------- 8. Customer Traffic Heatmap (Storewise) -----------
with tabs[7]:
    st.header("Customer Traffic Heatmap (Store x Time)")
    if all(c in df.columns for c in ["STORE_NAME","TRN_DATE","STORE_CODE","TILL","SESSION","RCT"]):
        store_opts = df["STORE_NAME"].dropna().unique().tolist()
        branch = st.selectbox("Store", store_opts, key="traffic_store")
        # robust CUST_CODE
        dff = df[df["STORE_NAME"]==branch].copy()
        for c in ["STORE_CODE","TILL","SESSION","RCT"]:
            dff[c] = dff[c].astype(str).fillna('').str.strip()
        dff['CUST_CODE'] = dff['STORE_CODE']+"-"+dff['TILL']+"-"+dff['SESSION']+"-"+dff['RCT']
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        dff = dff.dropna(subset=["TRN_DATE"])
        dff['TRN_DATE_ONLY'] = dff['TRN_DATE'].dt.date
        first_touch = (dff.groupby(["TRN_DATE_ONLY", "CUST_CODE"], as_index=False)["TRN_DATE"].min())
        first_touch['TIME_SLOT'] = first_touch['TRN_DATE'].dt.floor('30T')
        first_touch['TIME_ONLY'] = first_touch['TIME_SLOT'].dt.time
        cnts = first_touch.groupby("TIME_ONLY")['CUST_CODE'].nunique().reindex(intervals, fill_value=0)
        fig = px.bar(x=col_labels, y=cnts.values, text=cnts.values, labels={"x":"Time", "y":"Unique Receipts"},
                     title=f"Customer Traffic by Time (Unique Receipts) — {branch}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing necessary columns for heatmap.")

# ----------- 9. Active Tills Heatmap (Store x Time) -----------
with tabs[8]:
    st.header("Active Tills Heatmap (Store x Time)")
    if all(c in df.columns for c in ["STORE_NAME","TRN_DATE","TILL","STORE_CODE"]):
        stores = df["STORE_NAME"].dropna().unique()
        store = st.selectbox("Store", stores, key="heatmap_store")
        dff = df[df["STORE_NAME"]==store].copy()
        dff['TILL'] = dff['TILL'].astype(str).fillna('').str.strip()
        dff['STORE_CODE'] = dff['STORE_CODE'].astype(str).fillna('').str.strip()
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        dff = dff.dropna(subset=["TRN_DATE"])
        dff['TIME_INTERVAL'] = dff['TRN_DATE'].dt.floor('30T')
        dff['TIME_ONLY'] = dff['TIME_INTERVAL'].dt.time
        dff['Till_Code'] = dff['TILL'] + '-' + dff['STORE_CODE']
        cnts = dff.groupby(['TIME_ONLY'])['Till_Code'].nunique().reindex(intervals, fill_value=0)
        fig = px.bar(x=col_labels, y=cnts.values, text=cnts.values, labels={"x":"Time", "y":"Active Tills"},
                     title=f"Till Activity by Time — {store}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Columns missing for tills.")

# ----------- 10. Customers per Till (Store x Time) -----------
with tabs[9]:
    st.header("Customers per Till (Heatmap Store x Time)")
    if all(c in df.columns for c in ["STORE_NAME","TRN_DATE","TILL","STORE_CODE","SESSION","RCT"]):
        stores = df["STORE_NAME"].dropna().unique()
        store = st.selectbox("Pick Store", stores, key="custpertill_store")
        dff = df[df["STORE_NAME"]==store].copy()
        for c in ["STORE_CODE","TILL","SESSION","RCT"]:
            dff[c] = dff[c].astype(str).fillna('').str.strip()
        dff['CUST_CODE'] = dff['STORE_CODE']+"-"+dff['TILL']+"-"+dff['SESSION']+"-"+dff['RCT']
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        dff = dff.dropna(subset=["TRN_DATE"])
        dff['TRN_DATE_ONLY'] = dff['TRN_DATE'].dt.date
        dff['Till_Code'] = dff['TILL'] + '-' + dff['STORE_CODE']
        dff['TIME_INTERVAL'] = dff['TRN_DATE'].dt.floor('30T')
        dff['TIME_ONLY'] = dff['TIME_INTERVAL'].dt.time
        # Customer counts
        cust_per_time = dff.groupby('TIME_ONLY')['CUST_CODE'].nunique().reindex(intervals, fill_value=0)
        till_per_time = dff.groupby('TIME_ONLY')['Till_Code'].nunique().reindex(intervals, fill_value=0)
        ratio = np.ceil(cust_per_time / till_per_time.replace(0, np.nan)).fillna(0)
        fig = px.bar(x=col_labels, y=ratio, text=ratio.astype(int), labels={"x":"Time", "y":"Customers/Till"},
                     title=f"Customers Served per Till by Time — {store}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Required columns missing.")

# ----------- 11. Department Traffic Heatmap (per store) -----------
with tabs[10]:
    st.header("Department Traffic Heatmap")
    if all(c in df.columns for c in ["STORE_NAME","DEPARTMENT","TRN_DATE","STORE_CODE","TILL","SESSION","RCT"]):
        store_opts = df["STORE_NAME"].dropna().unique().tolist()
        branch = st.selectbox("Store", store_opts, key="dept_heatmap_store")
        dff = df[df["STORE_NAME"]==branch].copy()
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        for c in ["STORE_CODE","TILL","SESSION","RCT"]:
            dff[c] = dff[c].astype(str).fillna('').str.strip()
        dff['CUST_CODE'] = dff['STORE_CODE']+"-"+dff['TILL']+"-"+dff['SESSION']+"-"+dff['RCT']
        dff['TIME_INTERVAL'] = dff['TRN_DATE'].dt.floor('30T')
        dff['TIME_ONLY'] = dff['TIME_INTERVAL'].dt.time
        traffic = dff.groupby(['DEPARTMENT','TIME_ONLY'])['CUST_CODE'].nunique().reset_index()
        heatmap = traffic.pivot(index="DEPARTMENT", columns="TIME_ONLY", values="CUST_CODE").fillna(0)
        for t in intervals:
            if t not in heatmap.columns: heatmap[t]=0
        heatmap = heatmap[intervals]
        fig = px.imshow(
            heatmap.values,
            x=col_labels,
            y=heatmap.index,
            text_auto=True,
            aspect='auto',
            color_continuous_scale="Viridis"
        )
        fig.update_xaxes(side='top')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing department/store columns.")

# ----------- 12. Tax Compliance Overview (Pie + Bar) -----------
with tabs[11]:
    st.header("Tax Compliance (Pie + Till bar per store)")
    req = ["CU_DEVICE_SERIAL", "CUST_CODE", "STORE_NAME", "Till_Code"]
    if all(c in df.columns for c in req):
        df['Tax_Compliant'] = np.where(df['CU_DEVICE_SERIAL'].replace({'nan':'','NaN':'','None':''}).str.len()>0,
                                       'Compliant','Non-Compliant')
        g = df.groupby('Tax_Compliant', as_index=False)['CUST_CODE'].nunique()
        pie = px.pie(g, names='Tax_Compliant', values='CUST_CODE', color='Tax_Compliant',
                     title="🌍 Global Tax Compliance")
        st.plotly_chart(pie, use_container_width=True)
        stores = df['STORE_NAME'].dropna().unique()
        store = st.selectbox("Pick Store", stores, key="tax_store")
        dfb = df[df['STORE_NAME']==store]
        by_till = dfb.groupby(['Till_Code','Tax_Compliant'])['CUST_CODE'].nunique().reset_index()
        till_pivot = by_till.pivot(index="Till_Code", columns="Tax_Compliant", values="CUST_CODE").fillna(0)
        fig = go.Figure()
        fig.add_bar(y=till_pivot.index, x=till_pivot.get('Compliant',[]), orientation='h', name="Compliant", marker_color='#2ca02c')
        fig.add_bar(y=till_pivot.index, x=till_pivot.get('Non-Compliant',[]), orientation='h', name="Non-Compliant", marker_color='#d62728')
        fig.update_layout(
            barmode='stack', title=f"Tax Compliance by Till — {store}",
            xaxis_title="Unique Receipts", yaxis_title="Till"
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------- 13. Top Items/Branch (bar) -----------
with tabs[12]:
    st.header("Top Items by Branch")
    if all(c in df.columns for c in ["STORE_NAME","ITEM_NAME","QTY"]):
        stores = df["STORE_NAME"].dropna().unique()
        store = st.selectbox("Store", stores, key="topitem_branch")
        metric = st.selectbox("Metric", ["QTY","NET_SALES"], key="topitem_metric")
        n = st.slider("Top N", 5, 50, 10, key="topitem_n")
        branch_df = df[df["STORE_NAME"]==store]
        top_prod = branch_df.groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(n)
        fig = px.bar(top_prod, x=metric, y="ITEM_NAME", orientation="h", title=f"Top {n} Items by {metric} — {store}")
        st.plotly_chart(fig, use_container_width=True)

# ----------- 14. Branch Comparison -----------
with tabs[13]:
    st.header("Branch Comparison: Top N items by QTY/NET_SALES")
    if all(c in df.columns for c in ["STORE_NAME","ITEM_NAME","QTY"]):
        stores = list(df["STORE_NAME"].dropna().unique())
        a = st.selectbox("Branch A", stores, key="br_comp_a")
        b = st.selectbox("Branch B", stores, index=1 if len(stores)>1 else 0, key="br_comp_b")
        metric = st.selectbox("Metric", ["QTY","NET_SALES"], key="br_comp_metric")
        n = st.slider("Top N", 5, 50, 10, key="br_comp_n")
        dfA = df[df["STORE_NAME"]==a].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(n)
        dfB = df[df["STORE_NAME"]==b].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(n)
        combA = dfA.copy(); combA['Branch'] = a
        combB = dfB.copy(); combB['Branch'] = b
        both = pd.concat([combA, combB], ignore_index=True)
        fig = px.bar(both, x=metric, y="ITEM_NAME", color="Branch", orientation="h", barmode="group",
                     title=f"Top {n} items: {a} vs {b}")
        st.plotly_chart(fig, use_container_width=True)

# ----------- 15. Pricing Spread (multi-priced SKUs) -----------
with tabs[14]:
    st.header("Pricing Spread: Multi-Priced SKUs")
    if all(c in df.columns for c in ["STORE_NAME", "ITEM_CODE", "SP_PRE_VAT", "QTY", "ITEM_NAME"]):
        stores = df['STORE_NAME'].dropna().unique()
        store = st.selectbox("Store", stores, key="multi_price_store")
        dff = df[df['STORE_NAME'] == store].copy()
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        dff['SP_PRE_VAT'] = pd.to_numeric(dff['SP_PRE_VAT'], errors='coerce').fillna(0)
        dff['QTY'] = pd.to_numeric(dff['QTY'], errors='coerce').fillna(0)
        dff['DATE'] = dff['TRN_DATE'].dt.date
        grp = dff.groupby(['DATE','ITEM_CODE','ITEM_NAME'], as_index=False).agg(
            Num_Prices=('SP_PRE_VAT', lambda s: s.dropna().nunique()),
            Price_Min=('SP_PRE_VAT','min'),
            Price_Max=('SP_PRE_VAT','max'),
            Total_QTY=('QTY','sum')
        )
        grp['Spread'] = grp['Price_Max']-grp['Price_Min']
        grp = grp[(grp['Num_Prices']>1)&(grp['Spread']>0)]
        fig = px.bar(grp, x='Spread', y='ITEM_NAME', orientation='h', title=f"SKUs with Multi-Price (Spread>0) — {store}",
                     color='Total_QTY', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

# ----------- 16. Refunds (Negative sales) -----------
with tabs[15]:
    st.header("Refunds / Negative Receipts")
    if "NET_SALES" in df.columns and "STORE_NAME" in df.columns:
        stores = df["STORE_NAME"].dropna().unique()
        store = st.selectbox("Store", stores, key="neg_receipts_store")
        dff = df[(df["STORE_NAME"]==store) & (df["NET_SALES"] < 0)]
        if not dff.empty:
            fig = px.bar(dff, x="NET_SALES", y="ITEM_NAME", orientation="h", color_discrete_sequence=["#d62728"],
                         title=f"Refund/Negative Receipts by Item — {store}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No negative receipts found for this branch.")

st.sidebar.markdown("---\n**Built with Streamlit**\nAll analytics from original Superdeck notebook\nJobcheruyot")
