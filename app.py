#!/usr/bin/env python3
"""
Superdeck — Robust replacement app.py

This file is a defensive replacement to address the errors shown in your logs:
- TypeError from formatting functions (handled with safe formatters)
- KeyError: 'CUST_CODE' (guarded and placeholder columns created)
- ValueError: cannot insert '#' already exists (safe insertion)
- pyarrow ArrowInvalid: convert datetime columns before st.dataframe
- Deprecation: replace use_container_width with width='stretch'

Deployment:
- Replace the repository's app.py with this file and push to main.
- Keep .streamlit/config.toml minimal (only server.maxUploadSize if needed).
- Restart the app in Streamlit Cloud (Manage app → Restart).
"""
from __future__ import annotations
import io
import traceback
from typing import Dict, List, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Page config & styling
# -------------------------
st.set_page_config(page_title="Superdeck — Safe Start", layout="wide")
st.markdown(
    """
    <style>
      [data-testid="stSidebar"][aria-expanded="true"] > div:first-child { width: 340px; }
      .muted { color: #6c757d; font-size: 13px; }
      .card { padding:12px; border-radius:8px; box-shadow:0 6px 18px rgba(0,0,0,0.04); background: linear-gradient(180deg,#fff,#f8fbff); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Sidebar: uploader + subsections (uploader on top)
# -------------------------
st.sidebar.header("Upload CSV")
uploaded = st.sidebar.file_uploader("Upload a CSV file (required)", type=["csv"])

with st.sidebar.expander("Advanced: large uploads / S3 (optional)"):
    st.write("If your host blocks large uploads, upload to S3 externally and paste the object key here (advanced).")

st.sidebar.markdown("---")
st.sidebar.header("Views")
MAIN_SECTIONS = {
    "SALES": [
        "Global sales Overview",
        "Global Net Sales Distribution by Sales Channel",
        "Global Net Sales Distribution by SHIFT",
        "Night vs Day Shift Sales Ratio — Stores with Night Shifts",
        "Global Day vs Night Sales — Only Stores with NIGHT Shift",
        "2nd-Highest Channel Share",
        "Bottom 30 — 2nd Highest Channel",
        "Stores Sales Summary",
    ],
    "OPERATIONS": [
        "Customer Traffic-Storewise",
        "Active Tills During the day",
        "Average Customers Served per Till",
        "Store Customer Traffic Storewise",
        "Customer Traffic-Departmentwise",
        "Cashiers Perfomance",
        "Till Usage",
        "Tax Compliance",
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
        "Branch Brach Overview",
        "Global Refunds Overview",
        "Branch Refunds Overview",
    ],
}
category = st.sidebar.selectbox("Category", list(MAIN_SECTIONS.keys()))
subsection = st.sidebar.selectbox("Subsection", MAIN_SECTIONS[category])

# -------------------------
# Utilities (robust)
# -------------------------
def safe_read_csv_bytes(b: bytes) -> pd.DataFrame:
    bio = io.BytesIO(b)
    bio.seek(0)
    try:
        df = pd.read_csv(bio, on_bad_lines="skip", low_memory=False)
    except Exception:
        # chunked fallback
        bio.seek(0)
        parts = []
        for chunk in pd.read_csv(bio, on_bad_lines="skip", low_memory=False, chunksize=200_000):
            parts.append(chunk)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    return df

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    return df

def fmt_ints(df: pd.DataFrame, int_cols: List[str]) -> pd.DataFrame:
    df2 = df.copy()
    for c in int_cols:
        if c in df2.columns:
            def _fmt(v):
                try:
                    if pd.isna(v):
                        return ""
                    return f"{int(float(v)):,}"
                except Exception:
                    return v if v is not None else ""
            df2[c] = df2[c].apply(_fmt)
    return df2

def fmt_floats(df: pd.DataFrame, float_cols: List[str], decimals: int = 2) -> pd.DataFrame:
    df2 = df.copy()
    for c in float_cols:
        if c in df2.columns:
            def _fmt(v):
                try:
                    if pd.isna(v):
                        return ""
                    return f"{float(v):,.{decimals}f}"
                except Exception:
                    return v if v is not None else ""
            df2[c] = df2[c].apply(_fmt)
    return df2

def safe_insert_number_col(df: pd.DataFrame, col_name: str = "#") -> pd.DataFrame:
    df2 = df.copy()
    # If column exists already, overwrite it with sequential numbers
    if col_name in df2.columns:
        try:
            df2[col_name] = range(1, len(df2) + 1)
            cols = [col for col in df2.columns if col != col_name]
            df2 = df2.loc[:, [col_name] + cols]
            return df2
        except Exception:
            # fallback: create a new column with a different name
            new_col = col_name + "_1"
            df2.insert(0, new_col, range(1, len(df2) + 1))
            return df2
    else:
        df2.insert(0, col_name, range(1, len(df2) + 1))
        return df2

def convert_datetimes_for_display(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for col in df2.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df2[col]):
                df2[col] = df2[col].dt.strftime("%Y-%m-%d %H:%M:%S")
            elif df2[col].dtype == object:
                sample = df2[col].dropna().astype(str).head(10)
                if not sample.empty:
                    parsed = False
                    for s in sample:
                        try:
                            pd.to_datetime(s)
                            parsed = True
                            break
                        except Exception:
                            parsed = False
                    if parsed:
                        df2[col] = pd.to_datetime(df2[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
    return df2

# -------------------------
# Main UI
# -------------------------
st.title("Superdeck — Safe Dashboard")
st.write("Upload a CSV in the sidebar to enable the selected view. Subsections are below the uploader so the main area stays wide.")

if uploaded is None:
    st.info("No CSV uploaded. Upload to continue.")
    st.stop()

# Load CSV safely and show errors in-app (no crash)
try:
    with st.spinner("Loading CSV..."):
        df = safe_read_csv_bytes(uploaded.getvalue())
except Exception as e:
    st.error("Failed to read CSV. Expand error details for traceback.")
    with st.expander("Error details"):
        st.text(str(e))
        st.text(traceback.format_exc())
    st.stop()

# Ensure safe presence of common columns
placeholder_str_cols = ["CUST_CODE", "STORE_NAME", "ITEM_NAME", "ITEM_CODE", "SALES_CHANNEL_L1", "SALES_CHANNEL_L2", "SHIFT", "CU_DEVICE_SERIAL"]
for c in placeholder_str_cols:
    if c not in df.columns:
        df[c] = ""

if "TRN_DATE" not in df.columns:
    df["TRN_DATE"] = pd.NaT

# Coerce numeric columns which are commonly used
df = coerce_numeric(df, ["NET_SALES", "QTY", "VAT_AMT", "GROSS_SALES", "SP_PRE_VAT", "CP_PRE_VAT"])

# Convert TRN_DATE if possible
try:
    df["TRN_DATE"] = pd.to_datetime(df["TRN_DATE"], errors="coerce")
except Exception:
    pass

st.success(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")

# -------------------------
# Views
# -------------------------
def sales_global_overview(df: pd.DataFrame):
    if "SALES_CHANNEL_L1" not in df.columns or "NET_SALES" not in df.columns:
        st.error("Missing columns: SALES_CHANNEL_L1 and/or NET_SALES.")
        return
    agg = df.groupby("SALES_CHANNEL_L1", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
    if agg.empty:
        st.info("No sales data.")
        return
    agg["NET_SALES_M"] = agg["NET_SALES"] / 1_000_000
    agg["PCT"] = (100 * agg["NET_SALES"] / agg["NET_SALES"].sum()).round(1)
    labels = [f"{r['SALES_CHANNEL_L1']} ({r['PCT']:.1f}% | {r['NET_SALES_M']:.1f}M)" for _, r in agg.iterrows()]
    fig = go.Figure(go.Pie(labels=labels, values=agg["NET_SALES_M"], hole=0.6, text=[f"{p:.1f}%" for p in agg["PCT"]]))
    fig.update_layout(title="SALES CHANNEL TYPE — Global Overview", height=520)
    st.plotly_chart(fig, width="stretch")
    disp = agg[["SALES_CHANNEL_L1", "NET_SALES", "NET_SALES_M", "PCT"]].rename(columns={"SALES_CHANNEL_L1":"Channel","NET_SALES":"Net Sales (KSh)","NET_SALES_M":"Net Sales (M)","PCT":"Pct"})
    disp = convert_datetimes_for_display(disp)
    disp = fmt_ints(disp, ["Net Sales (KSh)"])
    disp = fmt_floats(disp, ["Pct"], 1)
    disp = safe_insert_number_col(disp, "#")
    st.dataframe(disp, width="stretch")
    st.download_button("⬇️ Download table", disp.to_csv(index=False).encode("utf-8"), "global_sales_overview.csv", "text/csv")

def stores_sales_summary(df: pd.DataFrame):
    if "NET_SALES" not in df.columns:
        st.error("Missing NET_SALES.")
        return
    ss = df.groupby("STORE_NAME", as_index=False).agg(NET_SALES=("NET_SALES","sum"), GROSS_SALES=("GROSS_SALES","sum"))
    if "CUST_CODE" in df.columns and df["CUST_CODE"].astype(str).str.strip().ne("").any():
        ss["Customer_Numbers"] = df.groupby("STORE_NAME")["CUST_CODE"].nunique().reindex(ss["STORE_NAME"]).fillna(0).astype(int).values
    else:
        ss["Customer_Numbers"] = 0
    total_gross = ss["GROSS_SALES"].sum()
    ss["% Contribution"] = (100 * ss["GROSS_SALES"] / total_gross).round(2) if total_gross != 0 else 0.0
    ss = ss.sort_values("GROSS_SALES", ascending=False).reset_index(drop=True)
    ss_display = ss.copy()
    ss_display = fmt_ints(ss_display, ["NET_SALES", "GROSS_SALES", "Customer_Numbers"])
    ss_display = fmt_floats(ss_display, ["% Contribution"], 2)
    ss_display = convert_datetimes_for_display(ss_display)
    ss_display = safe_insert_number_col(ss_display, "#")
    st.dataframe(ss_display, width="stretch")
    st.download_button("⬇️ Download Stores Summary", ss.to_csv(index=False).encode("utf-8"), "stores_sales_summary.csv", "text/csv")
    fig = px.bar(ss.sort_values("GROSS_SALES", ascending=True), x="GROSS_SALES", y="STORE_NAME", orientation="h", title="Gross Sales by Store")
    st.plotly_chart(fig, width="stretch")

# Dispatch selected subsection (kept small / robust)
try:
    if category == "SALES":
        if subsection == "Global sales Overview":
            sales_global_overview(df)
        elif subsection == "Stores Sales Summary":
            stores_sales_summary(df)
        else:
            st.info("Other SALES subsections will be added once this safe starter runs. Choose 'Global sales Overview' or 'Stores Sales Summary' to verify the app.")
    elif category == "OPERATIONS":
        if subsection == "Customer Traffic-Storewise":
            if "TRN_DATE" not in df.columns or "STORE_NAME" not in df.columns or "CUST_CODE" not in df.columns:
                st.error("Missing TRN_DATE and/or STORE_NAME and/or CUST_CODE.")
            else:
                ft = df.dropna(subset=["TRN_DATE"]).copy()
                ft["TRN_DATE"] = pd.to_datetime(ft["TRN_DATE"], errors="coerce")
                ft["DATE_ONLY"] = ft["TRN_DATE"].dt.date
                first_touch = ft.groupby(["STORE_NAME", "DATE_ONLY", "CUST_CODE"], as_index=False)["TRN_DATE"].min()
                first_touch["TIME_SLOT"] = first_touch["TRN_DATE"].dt.floor("30T").dt.time
                counts = first_touch.groupby(["STORE_NAME", "TIME_SLOT"])["CUST_CODE"].nunique().reset_index(name="Receipts")
                counts_disp = convert_datetimes_for_display(counts)
                counts_disp = safe_insert_number_col(counts_disp, "#")
                st.dataframe(counts_disp, width="stretch")
        elif subsection == "Tax Compliance":
            if "CU_DEVICE_SERIAL" not in df.columns or "CUST_CODE" not in df.columns:
                st.error("Missing CU_DEVICE_SERIAL and/or CUST_CODE.")
            else:
                d = df.copy()
                d["Tax_Compliant"] = np.where(d["CU_DEVICE_SERIAL"].astype(str).str.strip().replace({"nan": "", "None": ""}) != "", "Compliant", "Non-Compliant")
                summary = d.groupby("Tax_Compliant", as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"Receipts"})
                fig = px.pie(summary, names="Tax_Compliant", values="Receipts", hole=0.5, title="Tax Compliance")
                st.plotly_chart(fig, width="stretch")
                st.dataframe(convert_datetimes_for_display(summary), width="stretch")
        else:
            st.info("This OPERATIONS subsection isn't included in the safe starter yet.")
    elif category == "INSIGHTS":
        if subsection == "Customer Baskets Overview":
            if "ITEM_NAME" not in df.columns or "CUST_CODE" not in df.columns:
                st.error("Missing ITEM_NAME or CUST_CODE.")
            else:
                topn = st.slider("Top N", 5, 100, 10)
                top_items = df.groupby("ITEM_NAME")["CUST_CODE"].nunique().rename("Count_of_Baskets").reset_index().sort_values("Count_of_Baskets", ascending=False).head(topn)
                fig = px.bar(top_items, x="Count_of_Baskets", y="ITEM_NAME", orientation="h", title=f"Top {topn} Items by Baskets")
                st.plotly_chart(fig, width="stretch")
                st.dataframe(convert_datetimes_for_display(top_items), width="stretch")
        else:
            st.info("This INSIGHTS subsection isn't included in the safe starter yet.")
except Exception as e:
    st.error("Unexpected error while rendering view. App will not crash — see details.")
    with st.expander("Error details"):
        st.text(str(e))
        st.text(traceback.format_exc())

st.markdown("---")
st.caption("If the app still fails after replacing this file, open Manage app → Logs and paste the first ERROR/Traceback lines here.")
