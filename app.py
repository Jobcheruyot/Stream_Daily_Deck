import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(layout="wide", page_title="Superdeck Analytics", initial_sidebar_state="expanded")
st.title("🦸 Superdeck Analytics Dashboard")
st.markdown("> Upload your sales CSV - all analytics available as tabs and dropdowns.")

# ====== Data Upload and Preprocessing ======
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
if uploaded is None:
    st.info("Please upload a dataset to proceed.")
    st.stop()

@st.cache_data(show_spinner=True)
def load_and_prepare(uploaded):
    df = pd.read_csv(uploaded, on_bad_lines='skip', low_memory=False)
    # Standardize all column names (remove whitespace from start/end but keep case)
    df.columns = [c.strip() for c in df.columns]
    # Dates
    for col in ['TRN_DATE', 'ZED_DATE']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    # Ensure numeric columns
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    for nc in numeric_cols:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors='coerce').fillna(0)
    # -- CUST_CODE construction --
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
            st.error(f"Your data is missing columns required to build CUST_CODE: {missing}. Cannot proceed.")
            st.stop()
    # Remove leading/trailing spaces in CUST_CODE
    df['CUST_CODE'] = df['CUST_CODE'].astype(str).str.strip()
    return df

df = load_and_prepare(uploaded)

@st.cache_data(show_spinner=False)
def get_time_grid():
    start_time = pd.Timestamp("00:00:00")
    intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
    col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
    return intervals, col_labels

intervals, col_labels = get_time_grid()

tab_names = [
    "Sales Channel L1", "Sales Mode (L2)", "Net Sales by Shift", "Night vs Day/Store", "Day vs Night Pie",
    "2nd Channel Share", "Sales Summary", "Cust Traffic Heatmap", "Till Heatmap", "Custs per Till",
    "Dept/Branch Heatmap", "Tax Compliance", "Top Items", "Branch Compare", "Multi-price SKUs", "Refunds"
]
tabs = st.tabs(tab_names)

# ----- 1. SALES CHANNEL PIE -----
with tabs[0]:
    st.header("Sales Channel Type (L1) Distribution")
    if "SALES_CHANNEL_L1" in df.columns and "NET_SALES" in df.columns:
        g = df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
        g['NET_SALES_M'] = g['NET_SALES']/1_000_000
        g['PCT'] = g['NET_SALES']/g['NET_SALES'].sum()*100
        fig = go.Figure(go.Pie(
            labels=[f"{row['SALES_CHANNEL_L1']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f}M)" for _,row in g.iterrows()],
            values=g['NET_SALES_M'], hole=0.65, text=[f"{p:.1f}%" for p in g['PCT']], textinfo='text'))
        fig.update_layout(title="Sales Channel Type (L1) - Global", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing SALES_CHANNEL_L1 or NET_SALES.")

# ----- 2. SALES CHANNEL L2 -----
with tabs[1]:
    st.header("Net Sales by Mode (L2)")
    if "SALES_CHANNEL_L2" in df.columns and "NET_SALES" in df.columns:
        g = df.groupby('SALES_CHANNEL_L2', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
        g['NET_SALES_M'] = g['NET_SALES']/1_000_000
        g['PCT'] = g['NET_SALES']/g['NET_SALES'].sum()*100
        fig = go.Figure(go.Pie(
            labels=[f"{row['SALES_CHANNEL_L2']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f}M)" for _,row in g.iterrows()],
            values=g['NET_SALES_M'], hole=0.65, text=[f"{p:.1f}%" for p in g['PCT']], textinfo='text'))
        fig.update_layout(title="Sales Mode (L2)", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

# ----- 3. SALES SHIFT PIE -----
with tabs[2]:
    st.header("Net Sales by SHIFT")
    if "SHIFT" in df.columns and "NET_SALES" in df.columns:
        g = df.groupby('SHIFT', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
        g['PCT'] = g['NET_SALES']/g['NET_SALES'].sum()*100
        fig = go.Figure(go.Pie(labels=[f"{row['SHIFT']} ({row['PCT']:.1f}%)" for _,row in g.iterrows()],
                               values=g['NET_SALES'], hole=0.65, text=[f"{p:.1f}%" for p in g['PCT']], textinfo='text'))
        fig.update_layout(title="Global Net Sales by SHIFT")
        st.plotly_chart(fig, use_container_width=True)

# (4) Night vs Day Ratio by Store (bars)
with tabs[3]:
    st.header("Night vs Day Sales Ratio per Store")
    req = ["SHIFT","STORE_NAME","NET_SALES"]
    if all(x in df.columns for x in req):
        night_stores = df[df['SHIFT'].str.upper().str.contains('NIGHT', na=False)]['STORE_NAME'].unique()
        df_nd = df[df['STORE_NAME'].isin(night_stores)].copy()
        df_nd['Shift_Bucket'] = np.where(df_nd['SHIFT'].str.upper().str.contains('NIGHT', na=False),'Night','Day')
        ratio_df = df_nd.groupby(['STORE_NAME','Shift_Bucket'], as_index=False)['NET_SALES'].sum()
        sum_sales = ratio_df.groupby("STORE_NAME")["NET_SALES"].transform("sum")
        ratio_df['PCT'] = 100 * ratio_df['NET_SALES'] / sum_sales
        pivot_df = ratio_df.pivot(index='STORE_NAME', columns='Shift_Bucket', values='PCT').fillna(0)
        pivot_sorted = pivot_df.sort_values('Night', ascending=False)
        st.bar_chart(pivot_sorted[['Night','Day']], height=min(1200, 40+16*len(pivot_sorted)))
    else:
        st.warning(f"One or more of these columns missing: {req}")

# (5) Day vs Night Pie (Global)
with tabs[4]:
    st.header("Global Day vs Night (NIGHT shift stores)")
    req = ["SHIFT","STORE_NAME","NET_SALES"]
    if all(x in df.columns for x in req):
        night_stores = df[df['SHIFT'].str.upper().str.contains('NIGHT', na=False)]['STORE_NAME'].unique()
        df_nd = df[df['STORE_NAME'].isin(night_stores)].copy()
        df_nd['Shift_Bucket'] = np.where(df_nd['SHIFT'].str.upper().str.contains('NIGHT', na=False),'Night','Day')
        gb = df_nd.groupby('Shift_Bucket', as_index=False)['NET_SALES'].sum()
        gb['PCT'] = 100 * gb['NET_SALES'] / gb['NET_SALES'].sum()
        fig = go.Figure(go.Pie(
            labels=[f"{b} ({p:.1f}%)" for b, p in zip(gb['Shift_Bucket'], gb['PCT'])],
            values=gb['NET_SALES'], hole=0.65, text=[f"{p:.1f}%" for p in gb['PCT']], textinfo='text'))
        fig.update_layout(title="Day vs Night Sales (NIGHT shift stores)")
        st.plotly_chart(fig, use_container_width=True)

# (6) 2nd-Highest Channel Share (lollipop bar)
with tabs[5]:
    st.header("Stores by 2nd-Highest Channel Share (L1)")
    req = {"STORE_NAME","SALES_CHANNEL_L1","NET_SALES"}
    if req.issubset(df.columns):
        data = df.copy()
        data["NET_SALES"] = pd.to_numeric(data["NET_SALES"], errors="coerce").fillna(0)
        store_chan = data.groupby(["STORE_NAME","SALES_CHANNEL_L1"], as_index=False)["NET_SALES"].sum()
        store_tot = store_chan.groupby("STORE_NAME")["NET_SALES"].transform("sum")
        store_chan["PCT"] = 100 * store_chan["NET_SALES"] / store_tot
        store_chan = store_chan.sort_values(["STORE_NAME","PCT"], ascending=[True,False])
        store_chan["RANK"] = store_chan.groupby("STORE_NAME").cumcount() + 1
        n = st.slider("Top N", 5, 60, 30)
        top2 = store_chan[store_chan["RANK"]==2].sort_values("PCT", ascending=False).head(n)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=top2["PCT"], y=top2["STORE_NAME"], mode='markers+lines',
            marker=dict(size=12, color="#1f77b4", line=dict(width=2,color='#aaa')), name="2nd Channel %"
        ))
        fig.update_layout(title="Top Stores by 2nd Channel %", xaxis_title="2nd Channel %", yaxis_title="Store")
        st.plotly_chart(fig, use_container_width=True)

# (7) Sales Summary - bar+table
with tabs[6]:
    st.header("Sales Summary by Store")
    if 'GROSS_SALES' not in df.columns and 'VAT_AMT' in df.columns:
        df['GROSS_SALES'] = df['NET_SALES'] + df['VAT_AMT']
    if "STORE_NAME" in df.columns:
        group = df.groupby("STORE_NAME", as_index=False).agg(
            NET_SALES=('NET_SALES','sum'),
            GROSS_SALES=('GROSS_SALES','sum'),
            CUSTS=('CUST_CODE','nunique')
        )
        group['%_Contribution'] = (group['GROSS_SALES']/group['GROSS_SALES'].sum()*100).round(2)
        group = group.sort_values('GROSS_SALES', ascending=False)
        fig = px.bar(group, x="GROSS_SALES", y="STORE_NAME", orientation="h",
                     title="Gross Sales by Store", color="%_Contribution",
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(group)

# (8) Customer Traffic - quick by store line bar
with tabs[7]:
    st.header("Customer Traffic Heatmap")
    if all(c in df.columns for c in ["STORE_NAME","TRN_DATE","CUST_CODE"]):
        stores = df["STORE_NAME"].dropna().unique()
        store = st.selectbox("Pick Store", stores, key="heatmap_traffic_store")
        dff = df[df["STORE_NAME"]==store].copy()
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        dff = dff.dropna(subset=["TRN_DATE"])
        dff['TIME_INT'] = dff['TRN_DATE'].dt.floor('30T')
        dff['TIME_ONLY'] = dff['TIME_INT'].dt.time
        cnts = dff.groupby('TIME_ONLY')['CUST_CODE'].nunique().reindex(intervals, fill_value=0)
        fig = px.bar(x=col_labels, y=cnts.values, text=cnts.values, labels={"x":"Time", "y":"Receipts"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("STORE_NAME, TRN_DATE, CUST_CODE missing.")

# (9) Active Tills - quick heatmap
with tabs[8]:
    st.header("Active Tills Heatmap")
    if all(c in df.columns for c in ["STORE_NAME","TRN_DATE","TILL","STORE_CODE"]):
        stores = df["STORE_NAME"].dropna().unique()
        store = st.selectbox("Store", stores, key="tills_heatmap_store")
        dff = df[df["STORE_NAME"]==store]
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        dff['TIME'] = dff['TRN_DATE'].dt.floor('30T').dt.time
        dff['Till_Code'] = dff['TILL'].astype(str).fillna('') + '-' + dff['STORE_CODE'].astype(str).fillna('')
        res = dff.groupby('TIME')['Till_Code'].nunique().reindex(intervals, fill_value=0)
        fig = px.bar(x=col_labels, y=res.values, labels={"x":"Time","y":"Active Tills"}, text=res.values)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("STORE_NAME,TRN_DATE,TILL,STORE_CODE missing.")

# (10) Customers per Till (ratio by store x time)
with tabs[9]:
    st.header("Customers per Till over Time")
    reqs = ["STORE_NAME","TRN_DATE","CUST_CODE","TILL","STORE_CODE"]
    if all(c in df.columns for c in reqs):
        stores = df['STORE_NAME'].dropna().unique()
        store = st.selectbox("Store", stores, key="custpertill_store")
        dff = df[df["STORE_NAME"]==store].copy()
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        dff['TIME'] = dff['TRN_DATE'].dt.floor('30T').dt.time
        dff['Till_Code'] = dff['TILL'].astype(str).fillna('') + '-' + dff['STORE_CODE'].astype(str).fillna('')
        cust_count = dff.groupby('TIME')['CUST_CODE'].nunique().reindex(intervals, fill_value=0)
        till_count = dff.groupby('TIME')['Till_Code'].nunique().reindex(intervals, fill_value=0)
        ratio = np.ceil(cust_count / till_count.replace(0,np.nan)).fillna(0)
        fig = px.bar(x=col_labels, y=ratio, text=ratio.astype(int), labels={"x":"Time","y":"Cust/Till"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(str(reqs) + " missing.")

# (11) Dept/Branch Heatmap
with tabs[10]:
    st.header("Department Traffic Heatmap by Store")
    req = ["STORE_NAME","DEPARTMENT","TRN_DATE","CUST_CODE"]
    if all(c in df.columns for c in req):
        stores = df['STORE_NAME'].dropna().unique()
        store = st.selectbox("Store", stores, key="dept_heatmap_branch")
        dff = df[df['STORE_NAME']==store].copy()
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        dff['TIME'] = dff['TRN_DATE'].dt.floor('30T').dt.time
        traffic = dff.groupby(['DEPARTMENT','TIME'])['CUST_CODE'].nunique().reset_index()
        heatmap = traffic.pivot(index="DEPARTMENT", columns="TIME", values="CUST_CODE").fillna(0)
        for t in intervals: 
            if t not in heatmap.columns:
                heatmap[t] = 0
        heatmap = heatmap[intervals]
        fig = px.imshow(heatmap.values, x=col_labels, y=heatmap.index, text_auto=True, aspect='auto')
        fig.update_xaxes(side='top')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(str(req) + " missing.")

# (12) Tax Compliance Pie + bar (till)
with tabs[11]:
    st.header("Tax Compliance Overview")
    req = ["CU_DEVICE_SERIAL","CUST_CODE","STORE_NAME","TILL","STORE_CODE"]
    if all(c in df.columns for c in req):
        df['Tax_Compliant'] = np.where(df['CU_DEVICE_SERIAL'].replace({'nan':'','NaN':'','None':''}).str.len()>0,
                                       'Compliant','Non-Compliant')
        g = df.groupby("Tax_Compliant", as_index=False)["CUST_CODE"].nunique()
        pie = px.pie(g, names="Tax_Compliant", values="CUST_CODE", color="Tax_Compliant",
                 color_discrete_map={'Compliant':'#2ca02c','Non-Compliant':'#d62728'}, hole=0.45)
        st.plotly_chart(pie, use_container_width=True)
        stores = df['STORE_NAME'].dropna().unique()
        store = st.selectbox("Pick Store", stores, key="tax_br")
        dfb = df[df['STORE_NAME']==store]
        dfb['Till_Code'] = dfb['TILL'].astype(str).fillna('') + '-' + dfb['STORE_CODE'].astype(str).fillna('')
        by_till = dfb.groupby(['Till_Code','Tax_Compliant'])['CUST_CODE'].nunique().reset_index()
        till_pivot = by_till.pivot(index="Till_Code", columns="Tax_Compliant", values="CUST_CODE").fillna(0)
        fig = go.Figure()
        if 'Compliant' in till_pivot: 
            fig.add_bar(y=till_pivot.index, x=till_pivot.get('Compliant',[]), orientation='h', name='Compliant', marker_color='#2ca02c')
        if 'Non-Compliant' in till_pivot: 
            fig.add_bar(y=till_pivot.index, x=till_pivot.get('Non-Compliant',[]), orientation='h', name='Non-Compliant', marker_color='#d62728')
        fig.update_layout(barmode='stack', title=f"Tax Compliance by Till — {store}", xaxis_title="Receipts", yaxis_title="Till")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(str(req) + " missing.")

# (13) Top Items selectable
with tabs[12]:
    st.header("Top Items, Select Branch")
    reqs = ["STORE_NAME","ITEM_NAME","QTY"]
    if all(c in df.columns for c in reqs):
        stores = df['STORE_NAME'].dropna().unique()
        store = st.selectbox("Store", stores, key="topitem_store")
        metric = st.selectbox("Metric", ["QTY","NET_SALES"], key="topitem_metric")
        N = st.slider("Top N", 5, 50, 10)
        dff = df[df["STORE_NAME"]==store]
        gg = dff.groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N)
        fig = px.bar(gg, x=metric, y="ITEM_NAME", orientation="h", title=f"Top {N} by {metric} — {store}")
        st.plotly_chart(fig, use_container_width=True)

# (14) Branch Comparison (side by side)
with tabs[13]:
    st.header("Branch Comparison, Top N by Metric")
    reqs = ["STORE_NAME","ITEM_NAME","QTY"]
    if all(c in df.columns for c in reqs):
        stores = df['STORE_NAME'].dropna().unique()
        a = st.selectbox("Branch A", stores, key="brcompA"); b = st.selectbox("Branch B", stores, key="brcompB")
        metric = st.selectbox("Metric", ["QTY","NET_SALES"], key="brcomp_metric")
        N = st.slider("Top N", 5, 50, 10, key="brcomp_n")
        dfA = df[df["STORE_NAME"]==a].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N); dfA['Branch']=a
        dfB = df[df["STORE_NAME"]==b].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N); dfB['Branch']=b
        zz = pd.concat([dfA,dfB], ignore_index=True)
        fig = px.bar(zz, x=metric, y="ITEM_NAME", color='Branch', barmode="group", orientation="h",
                     title=f"Top {N} by {metric}: {a} vs {b}")
        st.plotly_chart(fig, use_container_width=True)

# (15) Pricing Spread
with tabs[14]:
    st.header("Multi-Priced SKUs / Pricing Spread (per branch)")
    req = ["STORE_NAME","ITEM_CODE","SP_PRE_VAT","QTY","ITEM_NAME"]
    if all(c in df.columns for c in req):
        stores = df['STORE_NAME'].dropna().unique()
        store = st.selectbox("Store", stores, key="mprice_store")
        dff = df[df['STORE_NAME']==store].copy()
        dff['SP_PRE_VAT'] = pd.to_numeric(dff['SP_PRE_VAT'], errors='coerce').fillna(0)
        dff['QTY'] = pd.to_numeric(dff['QTY'], errors='coerce').fillna(0)
        dff['DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce').dt.date
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

# (16) Negative Receipts/Refunds
with tabs[15]:
    st.header("Refunds / Negative Receipts")
    reqs = ["NET_SALES","STORE_NAME","ITEM_NAME"]
    if all(c in df.columns for c in reqs):
        stores = df["STORE_NAME"].dropna().unique()
        store = st.selectbox("Store", stores, key="neg_receipt_store")
        dff = df[(df["STORE_NAME"]==store) & (df["NET_SALES"] < 0)]
        if not dff.empty:
            fig = px.bar(dff, x="NET_SALES", y="ITEM_NAME", orientation="h", color_discrete_sequence=["#d62728"],
                         title=f"Refund/Negative Receipts by Item — {store}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No negative receipts found for this branch.")

st.sidebar.markdown("---\nBuilt with ❤️ using Streamlit\nContact: Jobcheruyot")
