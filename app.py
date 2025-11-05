import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(layout="wide")
st.title("🦸 Superdeck Analytics")

# === Upload Data Block ===
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload sales CSV", type="csv")
if uploaded is None:
    st.stop()

df = pd.read_csv(uploaded, on_bad_lines="skip", low_memory=False)
date_cols = ["TRN_DATE", "ZED_DATE"]
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
num_cols = ["QTY","CP_PRE_VAT","SP_PRE_VAT","COST_PRE_VAT","NET_SALES","VAT_AMT"]
for nc in num_cols:
    if nc in df.columns:
        df[nc] = pd.to_numeric(df[nc], errors="coerce").fillna(0)

# === Tabs for All Analysis Areas ===
tabs = st.tabs([
    "Sales Channel Overview",        # Pie by L1
    "Sales Mode Overview",           # Pie by L2
    "Net Sales by Shift",            # Pie by SHIFT
    "Night vs Day Ratio (Store)",    # Horizontal bar Night% by store
    "Day vs Night Global",           # Pie Night/Day
    "2nd-Highest Channel Share",     # Lollipop/ranks
    "Traffic Heatmap",               # Customers over time, storewise
    "Active Tills Heatmap",          # Tills over time
    "Customers per Till Heatmap",    # Ratio customers/till
    "Branch-Dept Traffic",           # Heatmap drilldown
    "Tax Compliance",                # Compliance pies/bars
    "Product/Top Items",             # Item wise - dropdowns
    "Branch Comparison",             # Branch vs branch - dropdowns
    "Pricing Spread",                # SKUs with mult. prices
    "Refunds"                        # Negative sales
])

# ==== 1. Sales Channel Pie ====
with tabs[0]:
    st.header("Sales Channel (L1) Overview")
    if "SALES_CHANNEL_L1" in df.columns and "NET_SALES" in df.columns:
        g = df.groupby("SALES_CHANNEL_L1", as_index=False)["NET_SALES"].sum()
        g["PCT"] = g["NET_SALES"] / g["NET_SALES"].sum() * 100
        fig = go.Figure(go.Pie(
            labels=g["SALES_CHANNEL_L1"],
            values=g["NET_SALES"],
            hole=0.65,
            text=[f"{p:.1f}%" for p in g["PCT"]],
            textinfo="text+label"
        ))
        fig.update_layout(title="Sales Channel Type (L1)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Upload data with SALES_CHANNEL_L1 and NET_SALES columns.")

# ==== 2. Sales Mode Pie ====
with tabs[1]:
    st.header("Sales Mode (L2) Overview")
    if "SALES_CHANNEL_L2" in df.columns and "NET_SALES" in df.columns:
        g = df.groupby("SALES_CHANNEL_L2", as_index=False)["NET_SALES"].sum()
        g["PCT"] = g["NET_SALES"] / g["NET_SALES"].sum() * 100
        fig = go.Figure(go.Pie(
            labels=g["SALES_CHANNEL_L2"],
            values=g["NET_SALES"],
            hole=0.65,
            text=[f"{p:.1f}%" for p in g["PCT"]],
            textinfo='text+label'
        ))
        fig.update_layout(title="Sales Mode (L2)")
        st.plotly_chart(fig, use_container_width=True)

# ==== 3. Shift Pie ====
with tabs[2]:
    st.header("Net Sales by Shift")
    if "SHIFT" in df.columns and "NET_SALES" in df.columns:
        g = df.groupby("SHIFT", as_index=False)["NET_SALES"].sum()
        g["PCT"] = g["NET_SALES"] / g["NET_SALES"].sum() * 100
        fig = go.Figure(go.Pie(
            labels=g["SHIFT"],
            values=g["NET_SALES"],
            hole=0.65,
            text=[f"{p:.1f}%" for p in g["PCT"]],
            textinfo='text+label'
        ))
        fig.update_layout(title="Shift Distribution")
        st.plotly_chart(fig, use_container_width=True)

# ==== 4. Night vs Day Ratio Storewise ====
with tabs[3]:
    st.header("Night vs Day Sales: Storewise Ratio")
    if "SHIFT" in df.columns and "STORE_NAME" in df.columns and "NET_SALES" in df.columns:
        df2 = df.copy()
        df2['Shift_Bucket'] = np.where(df2['SHIFT'].str.upper().str.contains('NIGHT', na=False), 'Night', 'Day')
        s = df2.groupby(['STORE_NAME','Shift_Bucket'], as_index=False)["NET_SALES"].sum()
        total = s.groupby('STORE_NAME')["NET_SALES"].transform('sum')
        s['PCT'] = 100 * s["NET_SALES"] / total
        pivot = s.pivot(index="STORE_NAME", columns="Shift_Bucket", values="PCT").fillna(0)
        # Bar plot, stores sorted by Night %
        plot_df = pivot.sort_values("Night", ascending=False)
        fig = go.Figure(go.Bar(
            x=plot_df['Night'],
            y=plot_df.index,
            orientation='h',
            name='Night %',
            marker_color="#d62728",
            text=[f"{v:.1f}%" for v in plot_df['Night']],
            textposition='auto'
        ))
        fig.update_layout(title="Stores Ranked by Night Shift Sales %",
                          xaxis_title="% Night Sales", yaxis_title="Store")
        st.plotly_chart(fig, use_container_width=True)

# ==== 5. Day vs Night Global Pie ====
with tabs[4]:
    st.header("Global Day vs Night (Only stores with NIGHT shift)")
    if "SHIFT" in df.columns and "STORE_NAME" in df.columns and "NET_SALES" in df.columns:
        stores_with_night = df[df['SHIFT'].str.upper().str.contains('NIGHT', na=False)]['STORE_NAME'].unique()
        dfx = df[df['STORE_NAME'].isin(stores_with_night)].copy()
        dfx['Shift_Bucket'] = np.where(dfx['SHIFT'].str.upper().str.contains('NIGHT', na=False), 'Night', 'Day')
        gg = dfx.groupby('Shift_Bucket', as_index=False)["NET_SALES"].sum()
        gg['PCT'] = gg['NET_SALES'] / gg['NET_SALES'].sum() * 100
        fig = go.Figure(go.Pie(
            labels=gg['Shift_Bucket'],
            values=gg["NET_SALES"],
            hole=0.65,
            text=[f"{v:.1f}%" for v in gg['PCT']],
            textinfo='text+label'
        ))
        fig.update_layout(title="Day vs Night (All NIGHT stores)")
        st.plotly_chart(fig, use_container_width=True)

# ==== 6. 2nd-Highest Channel Share ====
with tabs[5]:
    st.header("2nd-Highest Channel Share")
    if all(x in df.columns for x in ("STORE_NAME","SALES_CHANNEL_L1","NET_SALES")):
        top_n = st.slider("Show Top N Stores", 10, 100, 30)
        data = df.copy()
        data["NET_SALES"] = pd.to_numeric(data["NET_SALES"], errors="coerce").fillna(0)
        store_chan = data.groupby(["STORE_NAME", "SALES_CHANNEL_L1"], as_index=False)["NET_SALES"].sum()
        store_tot = store_chan.groupby("STORE_NAME")["NET_SALES"].transform("sum")
        store_chan["PCT"] = 100 * store_chan["NET_SALES"] / store_tot
        store_chan = store_chan.sort_values(["STORE_NAME", "PCT"], ascending=[True, False])
        store_chan["RANK"] = store_chan.groupby("STORE_NAME").cumcount() + 1
        second_tbl = store_chan[store_chan["RANK"]==2][["STORE_NAME","SALES_CHANNEL_L1","PCT"]]
        second_tbl = second_tbl.sort_values("PCT", ascending=False).head(top_n)
        # Lollipop plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=second_tbl["PCT"],
            y=second_tbl["STORE_NAME"],
            mode="markers+lines",
            marker=dict(size=14, color="#1f77b4"),
            line=dict(color="#aaaaaa", width=2),
            name="2nd Channel %"
        ))
        fig.update_layout(title="Top Stores by 2nd-Highest Channel %", xaxis_title="% of Store Net Sales", yaxis_title="Store")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Required columns missing.")

# ==== 7. Customer Traffic Heatmap ====
with tabs[6]:
    st.header("Customer Traffic Heatmap (Store x Time)")
    store_opts = df["STORE_NAME"].dropna().unique().tolist()
    branch = st.selectbox("Branch:", store_opts)
    if all(c in df.columns for c in ["STORE_NAME","TRN_DATE","STORE_CODE","TILL","SESSION","RCT"]):
        dff = df[df["STORE_NAME"]==branch].copy()
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        dff = dff.dropna(subset=['TRN_DATE'])
        for col in ["STORE_CODE","TILL","SESSION","RCT"]:
            dff[col] = dff[col].astype(str).fillna('').str.strip()
        dff['CUST_CODE'] = dff['STORE_CODE'] + '-' + dff['TILL'] + '-' + dff['SESSION'] + '-' + dff['RCT']
        dff['TIME_INTERVAL'] = dff['TRN_DATE'].dt.floor('30T')
        dff['TIME_ONLY'] = dff['TIME_INTERVAL'].dt.time
        intervals = [(pd.Timestamp("00:00:00") + timedelta(minutes=30*i)).time() for i in range(48)]
        col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
        cust_per_time = dff.groupby('TIME_ONLY')["CUST_CODE"].nunique().reindex(intervals, fill_value=0)
        fig = px.bar(
            x=col_labels, y=cust_per_time.values, labels={"x":"Time Interval","y":"Customer Count"},
            title=f"Traffic by time — {branch}", text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

# ==== 8: Active Tills Heatmap ====
with tabs[7]:
    st.header("Active Tills Heatmap (Store x Time)")
    if all(x in df.columns for x in ["STORE_NAME","TRN_DATE","TILL","STORE_CODE"]):
        branch_list = df["STORE_NAME"].dropna().unique().tolist()
        branch = st.selectbox("Branch", branch_list, key="till_branch")
        dff = df[df["STORE_NAME"]==branch].copy()
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        dff['TILL'] = dff["TILL"].astype(str).fillna('').str.strip()
        dff['STORE_CODE'] = dff["STORE_CODE"].astype(str).fillna("").str.strip()
        dff['TIME_INTERVAL'] = dff['TRN_DATE'].dt.floor('30T')
        dff['TIME_ONLY'] = dff['TIME_INTERVAL'].dt.time
        intervals = [(pd.Timestamp("00:00:00") + timedelta(minutes=30*i)).time() for i in range(48)]
        col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
        dff['Till_Code'] = dff['TILL'] + "-" + dff['STORE_CODE']
        til_per_time = dff.groupby('TIME_ONLY')["Till_Code"].nunique().reindex(intervals, fill_value=0)
        fig = px.bar(
            x=col_labels, y=til_per_time.values, labels={"x":"Time Interval","y":"Active Tills"},
            title=f"Tills Active by time — {branch}", text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

# ==== 9: Customers per Till Heatmap ====
with tabs[8]:
    st.header("Customers per Till Heatmap (Store x Time)")
    # You can mirror notebook blocks for more complex calculation and use heatmap/imshow visuals
    st.info("Implement customers per till per time grid as in the .py.")

# ==== 10: Branch Department Traffic====
with tabs[9]:
    st.header("Branch-Dept Customer Heatmap")
    branch = st.selectbox("Branch", df["STORE_NAME"].dropna().unique().tolist(), key="dept_branch")
    if all(c in df.columns for c in ["STORE_NAME","DEPARTMENT","TRN_DATE","STORE_CODE","TILL","SESSION","RCT"]):
        dff = df[df["STORE_NAME"]==branch].copy()
        for col in ["STORE_CODE","TILL","SESSION","RCT"]:
            dff[col] = dff[col].astype(str).fillna('').str.strip()
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        dff['CUST_CODE'] = dff['STORE_CODE'] + '-' + dff['TILL'] + '-' + dff['SESSION'] + '-' + dff['RCT']
        dff['TIME_INTERVAL'] = dff['TRN_DATE'].dt.floor('30T')
        dff['TIME_ONLY'] = dff['TIME_INTERVAL'].dt.time
        intervals = [(pd.Timestamp("00:00:00") + timedelta(minutes=30*i)).time() for i in range(48)]
        col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
        traffic = dff.groupby(['DEPARTMENT','TIME_ONLY'])["CUST_CODE"].nunique().reset_index()
        heatmap = traffic.pivot(index="DEPARTMENT", columns="TIME_ONLY", values="CUST_CODE").fillna(0)
        for t in intervals:
            if t not in heatmap.columns:
                heatmap[t]=0
        heatmap = heatmap[intervals]
        fig = px.imshow(
            heatmap.values,
            x=col_labels,
            y=heatmap.index,
            text_auto=True,
            aspect='auto',
        )
        fig.update_xaxes(side='top')
        fig.update_layout(title=f"Dept Traffic — {branch}")
        st.plotly_chart(fig, use_container_width=True)

# ==== 11: Tax Compliance Pie/Bar ====
with tabs[10]:
    st.header("Tax Compliance Overview")
    if all(x in df.columns for x in ("CU_DEVICE_SERIAL","CUST_CODE","STORE_NAME")):
        df['Tax_Compliant'] = np.where(df['CU_DEVICE_SERIAL'].replace({'nan':'','NaN':'','None':''}).str.len() > 0,
                                       "Compliant","Non-Compliant")
        g = df.groupby('Tax_Compliant', as_index=False)["CUST_CODE"].nunique()
        fig = go.Figure(go.Pie(
            labels=g['Tax_Compliant'], values=g['CUST_CODE'],
            hole=0.4,
            text=g['CUST_CODE'],
            textinfo='text+label'
        ))
        fig.update_layout(title="Global Tax Compliance (Receipts)")
        st.plotly_chart(fig, use_container_width=True)

# ==== 12: Product/Top Items ====
with tabs[11]:
    st.header("Top Items / Product Analytics")
    if "STORE_NAME" in df.columns and "ITEM_NAME" in df.columns and "QTY" in df.columns:
        store_opts = df["STORE_NAME"].dropna().unique().tolist()
        store = st.selectbox("Branch", store_opts, key="prod_branch")
        metric = st.radio("Metric", ["QTY","NET_SALES"])
        top_n = st.slider("Top N", 5, 50, 10)
        data = df[df["STORE_NAME"]==store]
        top_prod = data.groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(top_n)
        fig = px.bar(top_prod, x=metric, y="ITEM_NAME", orientation="h", title=f"Top {top_n} Items by {metric} — {store}")
        st.plotly_chart(fig, use_container_width=True)

# ==== 13: Branch Comparison ====
with tabs[12]:
    st.header("Branch-Branch Comparison")
    stores = df["STORE_NAME"].dropna().unique().tolist()
    a, b = st.selectbox("Branch A", stores, key="compA"), st.selectbox("Branch B", stores, key="compB")
    metric = st.radio("Metric", ["QTY","NET_SALES"], key="brn_metric")
    n = st.slider("Top N", 5, 50, 10, key="brn_n")
    topA = df[df["STORE_NAME"]==a].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(n)
    topB = df[df["STORE_NAME"]==b].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(n)
    comb = pd.concat([topA.assign(branch=a), topB.assign(branch=b)], ignore_index=True)
    fig = px.bar(comb, x=metric, y="ITEM_NAME", color='branch', barmode="group", orientation="h",
                 title=f"Top {n}: {a} vs {b}")
    st.plotly_chart(fig, use_container_width=True)

# ==== 14: Pricing Spread ====
with tabs[13]:
    st.header("Pricing Spread for SKUs (Branches with Multi-Price Same Day)")
    if "STORE_NAME" in df.columns and "ITEM_CODE" in df.columns and "SP_PRE_VAT" in df.columns:
        branch = st.selectbox("Branch", df["STORE_NAME"].dropna().unique().tolist(), key="price_branch")
        dff = df[df["STORE_NAME"]==branch].copy()
        dff['TRN_DATE'] = pd.to_datetime(dff['TRN_DATE'], errors='coerce')
        dff = dff.dropna(subset=['TRN_DATE'])
        dff['SP_PRE_VAT'] = pd.to_numeric(dff["SP_PRE_VAT"], errors='coerce').fillna(0)
        dff['QTY'] = pd.to_numeric(dff["QTY"], errors='coerce').fillna(0)
        dff['DATE'] = dff['TRN_DATE'].dt.date
        grp = dff.groupby(['DATE','ITEM_CODE','ITEM_NAME'], as_index=False).agg(
            Num_Prices=('SP_PRE_VAT', lambda s: s.dropna().nunique()),
            Price_Min=('SP_PRE_VAT','min'),
            Price_Max=('SP_PRE_VAT','max'),
            Total_QTY=('QTY','sum')
        )
        grp['Spread'] = grp['Price_Max']-grp['Price_Min']
        grp = grp[(grp['Num_Prices']>1)&(grp['Spread']>0)]
        fig = px.bar(
            grp, x='Spread', y='ITEM_NAME', orientation='h',
            title="SKUs with Multi-Price (spread>0)", color='Total_QTY'
        )
        st.plotly_chart(fig, use_container_width=True)

# ==== 15: Refunds (Neg Sales) ====
with tabs[14]:
    st.header("Negative Receipts / Refunds")
    if "NET_SALES" in df.columns and 'STORE_NAME' in df.columns:
        branch = st.selectbox("Branch", df["STORE_NAME"].dropna().unique().tolist(), key="refund_branch")
        dff = df[(df["STORE_NAME"]==branch) & (df["NET_SALES"] < 0)]
        if not dff.empty:
            fig = px.bar(
                dff, x="NET_SALES", y="ITEM_NAME", orientation='h',
                color_discrete_sequence=['#d62728'], title="Refund/Negative Receipts"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No negative receipts for this branch.")

st.sidebar.markdown("---\nBuilt with ❤️ using Streamlit\n[Your Name]")
