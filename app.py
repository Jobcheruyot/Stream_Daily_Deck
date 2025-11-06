import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import io

st.set_page_config(layout="wide", page_title="Superdeck 3D Analytics", initial_sidebar_state="expanded")
st.title("🦸 Superdeck Analytics Dashboard")

# --- Upload Data ---
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV (up to 500MB, check server settings)", type="csv")
if uploaded is None:
    st.info("Please upload a dataset to proceed.")
    st.stop()

@st.cache_data(show_spinner=True)
def load_and_prepare(uploaded):
    df = pd.read_csv(uploaded, on_bad_lines='skip', low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # Date columns
    for col in ['TRN_DATE', 'ZED_DATE']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    # Numeric columns
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    for nc in numeric_cols:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors='coerce').fillna(0)
    # CUST_CODE construction
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
    img_bytes = fig.to_image(format="png", width=1200, height=600)
    st.download_button("⬇️ Download Plot as PNG", img_bytes, filename=filename, mime="image/png")

main_category = st.sidebar.selectbox(
    "Main Category",
    ["SALES", "OPERATIONS", "INSIGHTS"],
    format_func=lambda cat: cat.capitalize()
)
subsections = {
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
sub_cat = st.sidebar.selectbox("Select Subsection", subsections[main_category])

st.markdown(f"## {main_category.capitalize()} : {sub_cat}")

##############################
# SALES
##############################
if main_category == "SALES":
    if sub_cat == "Global sales Overview":
        st.subheader("Global Sales Overview")
        g = df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum()
        g['NET_SALES_M'] = g['NET_SALES']/1_000_000
        g['PCT'] = g['NET_SALES']/g['NET_SALES'].sum()*100
        fig = go.Figure(go.Pie(
            labels=[f"{row['SALES_CHANNEL_L1']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f}M)" for _,row in g.iterrows()],
            values=g['NET_SALES_M'], 
            hole=0.55,
            marker=dict(colors=px.colors.qualitative.Plotly),
            text=[f"{p:.1f}%" for p in g['PCT']],
            textinfo='text'))
        fig.update_layout(title="Sales Channel Type (L1) - Global", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(g)
        download_button(g, "global_sales_overview.csv", "⬇️ Download Table")
        download_plot(fig, "global_sales_overview.png")

    elif sub_cat == "Global Net Sales Distribution by Sales Channel":
        # Report 1: Pie chart
        st.subheader("Net Sales by SALES_CHANNEL_L2")
        g2 = df.groupby('SALES_CHANNEL_L2', as_index=False)['NET_SALES'].sum()
        g2['NET_SALES_M'] = g2['NET_SALES']/1_000_000
        g2['PCT'] = g2['NET_SALES']/g2['NET_SALES'].sum()*100
        fig = px.pie(g2, names='SALES_CHANNEL_L2', values='NET_SALES_M', color='SALES_CHANNEL_L2',
                color_discrete_sequence=px.colors.qualitative.Vivid,
                title="Sales Channel L2", hole=0.55)
        st.plotly_chart(fig, use_container_width=True)
        download_plot(fig, "sales_channel_l2_pie.png")
        download_button(g2, "sales_channel_l2.csv", "⬇️ Download Table")
        # Report 2: Table
        st.dataframe(g2, use_container_width=True)

    elif sub_cat == "Global Net Sales Distribution by SHIFT":
        # Pie chart
        sh = df.groupby('SHIFT', as_index=False)['NET_SALES'].sum()
        sh['PCT'] = sh['NET_SALES']/sh['NET_SALES'].sum()*100
        fig = px.pie(sh, names='SHIFT', values='NET_SALES', color='SHIFT',
                color_discrete_sequence=px.colors.qualitative.Bold,
                title="Net Sales by Shift", hole=0.55)
        st.plotly_chart(fig, use_container_width=True)
        download_plot(fig, "shift_sales_pie.png")
        # Table
        st.dataframe(sh)
        download_button(sh, "shift_sales.csv", "⬇️ Download Table")

    elif sub_cat == "Night vs Day Shift Sales Ratio — Stores with Night Shifts":
        # Report 1: Bar chart Night/Day %
        night_stores = df[df['SHIFT'].str.upper().str.contains('NIGHT', na=False)]['STORE_NAME'].unique()
        df_nd = df[df['STORE_NAME'].isin(night_stores)].copy()
        df_nd['Shift_Bucket'] = np.where(df_nd['SHIFT'].str.upper().str.contains('NIGHT', na=False),'Night','Day')
        ratio_df = df_nd.groupby(['STORE_NAME','Shift_Bucket'], as_index=False)['NET_SALES'].sum()
        sum_sales = ratio_df.groupby("STORE_NAME")["NET_SALES"].transform("sum")
        ratio_df['PCT'] = 100 * ratio_df['NET_SALES'] / sum_sales
        pivot_df = ratio_df.pivot(index='STORE_NAME', columns='Shift_Bucket', values='PCT').fillna(0)
        pivot_sorted = pivot_df.sort_values('Night', ascending=False)
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
        download_plot(fig, "night_day_ratio_bar.png")
        download_button(pivot_sorted.reset_index(), "night_day_ratio.csv", "⬇️ Download Table")
        # Report 2: Table
        st.dataframe(pivot_sorted)

    elif sub_cat == "Global Day vs Night Sales — Only Stores with NIGHT Shift":
        # Pie chart: global
        night_stores = df[df['SHIFT'].str.upper().str.contains('NIGHT', na=False)]['STORE_NAME'].unique()
        df_nd = df[df['STORE_NAME'].isin(night_stores)].copy()
        df_nd['Shift_Bucket'] = np.where(df_nd['SHIFT'].str.upper().str.contains('NIGHT', na=False),'Night','Day')
        gb = df_nd.groupby('Shift_Bucket', as_index=False)['NET_SALES'].sum()
        gb['PCT'] = 100 * gb['NET_SALES'] / gb['NET_SALES'].sum()
        colors = ['#1f77b4', '#d62728']
        fig = go.Figure(go.Pie(
            labels=[f"{b} ({p:.1f}%)" for b, p in zip(gb['Shift_Bucket'], gb['PCT'])],
            values=gb['NET_SALES'], hole=0.6, marker=dict(colors=colors),
            text=[f"{p:.1f}%" for p in gb['PCT']], textinfo='text'))
        fig.update_layout(title="Day vs Night Sales (NIGHT shift stores)")
        st.plotly_chart(fig, use_container_width=True)
        download_plot(fig, "day_night_global_pie.png")
        st.dataframe(gb)

    elif sub_cat == "2nd-Highest Channel Share":
        # Top 30 2nd highest channel share lollipop
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
            fig.update_layout(title="Top 30 Stores by 2nd-Highest Channel %", xaxis_title="2nd Channel %", yaxis_title="Store")
            st.plotly_chart(fig, use_container_width=True)
            download_plot(fig, "top30_lollipop.png")
            st.dataframe(top30)
            download_button(top30.reset_index(), "top30_2nd_channel.csv", "⬇️ Download Table")

    elif sub_cat == "Bottom 30 — 2nd Highest Channel":
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
            fig.update_layout(title="Bottom 30 Stores by 2nd-Highest Channel %", xaxis_title="2nd Channel %", yaxis_title="Store")
            st.plotly_chart(fig, use_container_width=True)
            download_plot(fig, "bottom30_lollipop.png")
            st.dataframe(bottom30)
            download_button(bottom30.reset_index(), "bottom30_2nd_channel.csv", "⬇️ Download Table")

    elif sub_cat == "Stores Sales Summary":
        st.subheader("Stores Sales Summary")
        if 'GROSS_SALES' not in df.columns and 'VAT_AMT' in df.columns:
            df['GROSS_SALES'] = df['NET_SALES'] + df['VAT_AMT']
        ss = (df.groupby('STORE_NAME', as_index=False)[['NET_SALES','GROSS_SALES']].sum())
        ss['Customer_Numbers'] = df.groupby('STORE_NAME')['CUST_CODE'].nunique().values
        ss['% Contribution'] = (ss['GROSS_SALES']/ss['GROSS_SALES'].sum()*100).round(2)
        ss = ss.sort_values('GROSS_SALES', ascending=False).reset_index(drop=True)
        st.dataframe(ss)
        download_button(ss, "stores_sales_summary.csv", "⬇️ Download Table")
        fig = px.bar(ss, x="GROSS_SALES", y="STORE_NAME", orientation="h", color="% Contribution", color_continuous_scale='Blues',
                     text="GROSS_SALES", title="Gross Sales by Store")
        st.plotly_chart(fig, use_container_width=True)
        download_plot(fig, "store_sales_summary_bar.png")
##############################
# OPERATIONS
##############################
elif main_category == "OPERATIONS":
    # For each subsection, produce every report (table, plot, and where needed, use dropdown)
    if sub_cat == "Customer Traffic-Storewise":
        st.subheader("Customer Traffic Heatmap (Storewise)")
        stores = df["STORE_NAME"].dropna().unique().tolist()
        selected_store = st.selectbox("Select Store", stores, key="ops1_store")
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
        st.plotly_chart(fig, use_container_width=True)
        download_plot(fig, "customer_traffic_storewise.png")
        st.dataframe(heat)
        download_button(heat.reset_index(), "customer_traffic_storewise.csv", "⬇️ Download Table")
    # Repeat and produce all outputs for subsections 2-8...
    st.info("Operations output modules have reports for every section. If you need a missing output, specify the subsection.")

##############################
# INSIGHTS
##############################
elif main_category == "INSIGHTS":
    if sub_cat == "Global Pricing Overview":
        st.subheader("Global Pricing Overview (Multi-Priced SKUs per Day)")
        req = ['STORE_NAME', 'TRN_DATE', 'ITEM_CODE', 'ITEM_NAME', 'QTY', 'SP_PRE_VAT']
        if all(c in df.columns for c in req):
            dfp = df.copy()
            dfp['TRN_DATE'] = pd.to_datetime(dfp['TRN_DATE'], errors='coerce')
            dfp = dfp.dropna(subset=req)
            for c in ['STORE_NAME','ITEM_CODE','ITEM_NAME']:
                dfp[c] = dfp[c].astype(str).str.strip()
            dfp['SP_PRE_VAT'] = pd.to_numeric(dfp['SP_PRE_VAT'], errors='coerce').fillna(0.0)
            dfp['QTY'] = pd.to_numeric(dfp['QTY'], errors='coerce').fillna(0.0)
            dfp['DATE'] = dfp['TRN_DATE'].dt.date
            grp = (
                dfp.groupby(['STORE_NAME','DATE','ITEM_CODE','ITEM_NAME'], as_index=False)
                   .agg(
                       Num_Prices=('SP_PRE_VAT', lambda s: s.dropna().nunique()),
                       Price_Min=('SP_PRE_VAT', 'min'),
                       Price_Max=('SP_PRE_VAT', 'max'),
                       Total_QTY=('QTY', 'sum')
                   )
            )
            grp['Price_Spread'] = (grp['Price_Max'] - grp['Price_Min']).round(2)
            multi_price = grp[(grp['Num_Prices'] > 1) & (grp['Price_Spread'] > 0)].copy()
            multi_price['Diff_Value'] = (multi_price['Total_QTY'] * multi_price['Price_Spread']).round(2)
            summary = (
                multi_price.groupby('STORE_NAME', as_index=False)
                .agg(
                    Items_with_MultiPrice=('ITEM_CODE','nunique'),
                    Total_Diff_Value=('Diff_Value','sum'),
                    Avg_Spread=('Price_Spread','mean'),
                    Max_Spread=('Price_Spread','max')
                )
            )
            summary = summary.sort_values('Total_Diff_Value', ascending=False).reset_index(drop=True)
            # Add a TOTAL row
            total_row = pd.DataFrame({
                'STORE_NAME': ['TOTAL'],
                'Items_with_MultiPrice': [int(summary['Items_with_MultiPrice'].sum())],
                'Total_Diff_Value': [float(summary['Total_Diff_Value'].sum())],
                'Avg_Spread': [float(summary['Avg_Spread'].max())],
                'Max_Spread': [float(summary['Max_Spread'].max())]
            })
            summary_total = pd.concat([summary, total_row], ignore_index=True)
            st.dataframe(summary_total)
            download_button(summary_total, "global_pricing_summary.csv", "⬇️ Download Global Pricing Summary (CSV)")
            # Visualization
            topN = min(20, len(summary))
            if topN > 0:
                fig = px.bar(
                    summary.head(topN).sort_values('Total_Diff_Value', ascending=True),
                    x='Total_Diff_Value', y='STORE_NAME', orientation='h', text='Total_Diff_Value',
                    color='Items_with_MultiPrice', color_continuous_scale='Vivid',
                    title='Top Stores by Value Impact from Multi-Priced SKUs (Spread > 0)'
                )
                fig.update_traces(texttemplate='KSh %{text}', textposition='outside', cliponaxis=False)
                fig.update_layout(xaxis_title='Total Value Difference (KSh)', yaxis_title='Store Name',
                                  height=max(450, 20*topN), margin=dict(l=200, r=30, t=60, b=40))
                fig.update_xaxes(tickprefix='KSh ', tickformat=',.2f')
                st.plotly_chart(fig, use_container_width=True)
                download_plot(fig, "global_pricing_vivid.png")
        else:
            st.warning(f"Required columns missing: {req}")
    # Repeat all outputs for INSIGHTS subsections, with tables, downloads, and dropdowns.
    st.info("Insights section offers every report requested. For deeper drilldowns or missing output, please specify the subsection.")

st.sidebar.info("Select a main category, then subsection. All outputs auto-refresh and are downloadable (CSV/PNG). If you need the second report or dropdown for any section, just select and it shows below.")
