import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import io

# --- App config
st.set_page_config(layout="wide", page_title="Superdeck Analytics", initial_sidebar_state="expanded")
st.title("🦸 Superdeck Analytics Dashboard")
st.markdown("> Upload your sales CSV, explore & download tables and visuals. Every dropdown change refreshes instantly.")

def download_button(obj, filename, label, use_xlsx=False):
    """Generate a Streamlit download button for tables."""
    if use_xlsx:
        towrite = io.BytesIO()
        obj.to_excel(towrite, encoding="utf-8", index=False, engine='openpyxl')
        towrite.seek(0)
        st.download_button(label, towrite, file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.download_button(label, obj.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")

def download_plot(fig, filename):
    """A helper to save a Plotly fig and offer for download."""
    img_bytes = fig.to_image(format="png", width=1200, height=600)
    st.download_button("⬇️ Download Plot as PNG", img_bytes, filename=filename, mime="image/png")

# --- Upload data
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
            st.error(f"Your data is missing columns required to build CUST_CODE: {missing}. Cannot proceed.")
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

tab_names = [
    "Sales Channel L1", "Sales Mode (L2)", "Net Sales by Shift",
    "Night vs Day/Store", "Day vs Night Pie", "2nd Channel Share",
    "Sales Summary", "Store Traffic Heatmap", "Till Heatmap", "Customers per Till",
    "Dept/Branch Heatmap", "Tax Compliance", "Top Items", "Branch Comparison",
    "Pricing Spread", "Refunds", "Global Pricing Overview"
]
tabs = st.tabs(tab_names)

## ... previous tabs unchanged ...

## Global Pricing Overview Tab
with tabs[16]:
    st.header("Global Pricing Overview (Multi-Priced SKUs per Day)")
    required_cols = ['STORE_NAME', 'TRN_DATE', 'ITEM_CODE', 'ITEM_NAME', 'QTY', 'SP_PRE_VAT']
    if all(c in df.columns for c in required_cols):
        dfp = df.copy()
        dfp['TRN_DATE'] = pd.to_datetime(dfp['TRN_DATE'], errors='coerce')
        dfp = dfp.dropna(subset=['TRN_DATE','STORE_NAME','ITEM_CODE','ITEM_NAME','QTY','SP_PRE_VAT'])
        for c in ['STORE_NAME','ITEM_CODE','ITEM_NAME']:
            dfp[c] = dfp[c].astype(str).str.strip()
        dfp['SP_PRE_VAT'] = dfp['SP_PRE_VAT'].astype(str).str.replace(',', '', regex=False).str.strip()
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
        summary.insert(0, '#', range(1, len(summary)+1))
        # Add a TOTAL row
        total_row = pd.DataFrame({
            '#': [''],
            'STORE_NAME': ['TOTAL'],
            'Items_with_MultiPrice': [int(summary['Items_with_MultiPrice'].sum())],
            'Total_Diff_Value': [float(summary['Total_Diff_Value'].sum())],
            'Avg_Spread': [float(summary['Avg_Spread'].max())],
            'Max_Spread': [float(summary['Max_Spread'].max())]
        })
        summary_total = pd.concat([summary, total_row], ignore_index=True)
        # Formatting
        summary_total['Items_with_MultiPrice'] = summary_total['Items_with_MultiPrice'].apply(lambda x: f"{int(x):,}" if pd.notna(x) and str(x).isdigit() else x)
        for c in ['Total_Diff_Value','Avg_Spread','Max_Spread']:
            summary_total[c] = summary_total[c].apply(lambda x: f"{float(x):,.2f}" if pd.notna(x) and str(x).replace('.', '', 1).isdigit() else x)
        st.dataframe(summary_total, use_container_width=True)
        download_button(summary_total, "global_pricing_summary.csv", "⬇️ Download Global Pricing Summary (CSV)")
        # Visualization
        topN = min(20, len(summary))
        st.subheader("Top Stores by Value Impact from Multi-Priced SKUs")
        if topN > 0:
            fig = px.bar(
                summary.head(topN).sort_values('Total_Diff_Value', ascending=True),
                x='Total_Diff_Value',
                y='STORE_NAME',
                orientation='h',
                text='Total_Diff_Value',
                color='Items_with_MultiPrice',
                color_continuous_scale='Vivid',
                title='Top Stores by Value Impact from Multi-Priced SKUs (Spread > 0)'
            )
            fig.update_traces(texttemplate='KSh %{text}', textposition='outside', cliponaxis=False)
            fig.update_layout(
                xaxis_title='Total Value Difference (KSh)',
                yaxis_title='Store Name',
                coloraxis_colorbar=dict(title='SKUs with >1 Price'),
                height=max(450, 20*topN),
                margin=dict(l=200, r=30, t=60, b=40)
            )
            fig.update_xaxes(tickprefix='KSh ', tickformat=',.2f')
            st.plotly_chart(fig, use_container_width=True)
            download_plot(fig, "global_pricing_vivid.png")
        else:
            st.success("No multi-priced items with positive spread found across stores.")
    else:
        st.warning(f"Required columns missing: {required_cols}")

st.sidebar.success("You can download tables (CSV/Excel) and all visuals as PNGs from each tab!")
st.sidebar.markdown("---\nBuilt with ❤️ using Streamlit\nContact: Jobcheruyot")
