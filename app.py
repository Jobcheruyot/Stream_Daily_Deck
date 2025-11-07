#!/usr/bin/env python3
"""
Superdeck — Safe, robust Streamlit app replacement.

Why this file:
- Replaces the previous app.py with a robust, defensive implementation so the app
  starts reliably on Streamlit Cloud (avoids the "Oh no." crash).
- Sidebar layout: uploader at top; subsections shown under uploader (keeps main area wide).
- Runs heavy processing only after upload, wrapped in try/except so errors are shown in-app.
- Fixes the crashes seen in your logs:
  - Avoids KeyError('CUST_CODE') by checking for columns before use.
  - Avoids formatting errors (ufunc 'isfinite') by coercing to numeric before formatting.
  - Prevents "cannot insert #, already exists" by adding safeties when inserting numbering.
  - Converts datetime columns to strings before sending DataFrames to Streamlit to avoid pyarrow ArrowInvalid.
  - Replaces deprecated `use_container_width` calls with `width='stretch'` where applicable.
- Minimal but complete set of views implemented for verification. We can re-add the rest incrementally.

Deployment:
- Replace your repository's app.py with this file and push.
- If you set .streamlit/config.toml, keep only server.maxUploadSize = 1024 (do not set server.address/server.port).
- In Streamlit Cloud -> Manage app -> Restart / Deploy.
- If it still fails, paste the first ERROR/Traceback lines from the app logs here.

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
# Sidebar: uploader + subsections
# -------------------------
st.sidebar.header("Upload CSV")
uploaded = st.sidebar.file_uploader("Upload a CSV file (required)", type=["csv"])

with st.sidebar.expander("Advanced: large uploads / S3 (optional)"):
    st.write("If your host blocks large uploads, you can upload directly to S3 and then provide the S3 object key to this app.")
    st.write("Only open this if you need it.")

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
# Helpers: safe loading, formatting
# -------------------------
def safe_read_csv_bytes(b: bytes) -> pd.DataFrame:
    """Read CSV from bytes. Use chunking only if necessary and handle errors."""
    bio = io.BytesIO(b)
    bio.seek(0)
    # Try reading directly; if fails due to memory for very large, Streamlit host will likely block anyway.
    try:
        df = pd.read_csv(bio, on_bad_lines="skip", low_memory=False)
    except Exception as e:
        # Try chunked fallback
        try:
            bio.seek(0)
            parts = []
            for chunk in pd.read_csv(bio, on_bad_lines="skip", low_memory=False, chunksize=200_000):
                parts.append(chunk)
            df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        except Exception:
            raise
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    return df

def fmt_ints(df: pd.DataFrame, int_cols: List[str]) -> pd.DataFrame:
    """Safely format integer-like columns for display (returns copy)."""
    df2 = df.copy()
    for c in int_cols:
        if c in df2.columns:
            # preserve NaN, handle non-numeric
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
    if col_name in df2.columns:
        # ensure it's leftmost but avoid duplicate insert error
        cols = [col for col in df2.columns if col != col_name]
        df2 = df2.loc[:, cols]
        df2.insert(0, col_name, range(1, len(df2) + 1))
    else:
        df2.insert(0, col_name, range(1, len(df2) + 1))
    return df2

def convert_datetimes_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime-like columns to ISO strings to avoid pyarrow/pandas->arrow issues in Streamlit display."""
    df2 = df.copy()
    for col in df2.columns:
        if pd.api.types.is_datetime64_any_dtype(df2[col]) or pd.api.types.is_datetime64_ns_dtype(df2[col]):
            df2[col] = df2[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        # also handle object columns that may contain datetimes mixed types
        elif df2[col].dtype == object:
            # detect if many values are datetime-like
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
                    try:
                        df2[col] = pd.to_datetime(df2[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass
    return df2

# -------------------------
# Main logic: load file then show chosen subsection
# -------------------------
st.title("Superdeck — Safe Dashboard")
st.write("Upload a CSV in the sidebar to enable the selected view. Subsections are below the uploader so the main area stays wide.")

if uploaded is None:
    st.info("No CSV uploaded yet. Upload to continue.")
    st.stop()

# Try loading with safe wrapper and show errors in-app rather than letting the process crash
try:
    with st.spinner("Loading CSV..."):
        df = safe_read_csv_bytes(uploaded.getvalue())
except Exception as e:
    st.error("Failed to read CSV. See details below.")
    with st.expander("Error details"):
        st.text(str(e))
        st.text(traceback.format_exc())
    st.stop()

# Normalize a few common columns so downstream code doesn't KeyError
# If missing, create placeholder columns with empty values (safer than crashing)
for col in ["CUST_CODE", "STORE_NAME", "TRN_DATE", "ITEM_NAME", "ITEM_CODE", "SALES_CHANNEL_L1", "SALES_CHANNEL_L2", "SHIFT", "VAT_AMT"]:
    if col not in df.columns:
        df[col] = "" if col != "TRN_DATE" else pd.NaT

# Coerce typical numeric columns
df = coerce_numeric(df, ["NET_SALES", "QTY", "VAT_AMT", "GROSS_SALES", "SP_PRE_VAT", "CP_PRE_VAT"])

# Convert datetimes to proper dtype where possible
if "TRN_DATE" in df.columns:
    try:
        df["TRN_DATE"] = pd.to_datetime(df["TRN_DATE"], errors="coerce")
    except Exception:
        pass

st.success(f"Loaded {len(df):,} rows, {len(df.columns):,} columns.")

# -------------------------
# Render views (robust)
# -------------------------
def sales_global_overview(df: pd.DataFrame):
    if "SALES_CHANNEL_L1" not in df.columns or "NET_SALES" not in df.columns:
        st.error("Missing columns: SALES_CHANNEL_L1 and/or NET_SALES.")
        return
    agg = df.groupby("SALES_CHANNEL_L1", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
    if agg.empty:
        st.info("No sales data found.")
        return
    agg["NET_SALES_M"] = agg["NET_SALES"] / 1_000_000
    agg["PCT"] = (100 * agg["NET_SALES"] / agg["NET_SALES"].sum()).round(1)
    labels = [f"{r['SALES_CHANNEL_L1']} ({r['PCT']:.1f}% | {r['NET_SALES_M']:.1f}M)" for _, r in agg.iterrows()]
    fig = go.Figure(go.Pie(labels=labels, values=agg["NET_SALES_M"], hole=0.6, text=[f"{p:.1f}%" for p in agg["PCT"]]))
    fig.update_layout(title="SALES CHANNEL TYPE — Global Overview", height=520)
    st.plotly_chart(fig, use_container_width=True)  # warnings ok but not fatal
    disp = agg[["SALES_CHANNEL_L1", "NET_SALES", "NET_SALES_M", "PCT"]].rename(columns={"SALES_CHANNEL_L1":"Channel","NET_SALES":"Net Sales (KSh)","NET_SALES_M":"Net Sales (M)","PCT":"Pct"})
    # Convert datetimes or problematic dtypes before display
    disp = convert_datetimes_for_display(disp)
    st.dataframe(disp, width="stretch")
    st.download_button("⬇️ Download table", disp.to_csv(index=False).encode("utf-8"), "global_sales_overview.csv", "text/csv")

def sales_channel2(df: pd.DataFrame):
    if "SALES_CHANNEL_L2" not in df.columns or "NET_SALES" not in df.columns:
        st.error("Missing columns: SALES_CHANNEL_L2 and/or NET_SALES.")
        return
    agg = df.groupby("SALES_CHANNEL_L2", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
    if agg.empty:
        st.info("No data.")
        return
    agg["NET_SALES_M"] = agg["NET_SALES"] / 1_000_000
    fig = px.pie(agg, names="SALES_CHANNEL_L2", values="NET_SALES_M", hole=0.6, title="Global Net Sales Distribution by SALES_CHANNEL_L2")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(convert_datetimes_for_display(agg), width="stretch")

def sales_by_shift(df: pd.DataFrame):
    if "SHIFT" not in df.columns or "NET_SALES" not in df.columns:
        st.error("Missing columns: SHIFT and/or NET_SALES.")
        return
    agg = df.groupby("SHIFT", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
    if agg.empty:
        st.info("No data.")
        return
    fig = px.pie(agg, names="SHIFT", values="NET_SALES", hole=0.6, title="Global Net Sales by SHIFT")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(convert_datetimes_for_display(agg), width="stretch")

def night_day_ratio(df: pd.DataFrame):
    if "SHIFT" not in df.columns or "STORE_NAME" not in df.columns or "NET_SALES" not in df.columns:
        st.error("Missing columns: SHIFT, STORE_NAME, and/or NET_SALES.")
        return
    stores_with_night = df[df["SHIFT"].astype(str).str.upper().str.contains("NIGHT", na=False)]["STORE_NAME"].unique()
    if len(stores_with_night) == 0:
        st.info("No stores with NIGHT shift found.")
        return
    dnd = df[df["STORE_NAME"].isin(stores_with_night)].copy()
    dnd["Shift_Bucket"] = np.where(dnd["SHIFT"].astype(str).str.upper().str.contains("NIGHT", na=False), "Night", "Day")
    r = dnd.groupby(["STORE_NAME", "Shift_Bucket"], as_index=False)["NET_SALES"].sum()
    tot = r.groupby("STORE_NAME")["NET_SALES"].transform("sum")
    r["PCT"] = np.where(tot > 0, 100 * r["NET_SALES"] / tot, 0.0)
    pivot = r.pivot(index="STORE_NAME", columns="Shift_Bucket", values="PCT").fillna(0)
    pivot = pivot.sort_values("Night", ascending=False)
    if pivot.empty:
        st.info("No results after aggregation.")
        return
    fig = go.Figure()
    if "Night" in pivot.columns:
        fig.add_trace(go.Bar(x=pivot["Night"], y=pivot.index, orientation="h", name="Night", marker_color="#d62728"))
    if "Day" in pivot.columns:
        fig.add_trace(go.Bar(x=pivot["Day"], y=pivot.index, orientation="h", name="Day", marker_color="#1f77b4"))
    fig.update_layout(barmode="group", title="Night vs Day % by Store", height=max(400, 24 * len(pivot)))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(convert_datetimes_for_display(pivot.reset_index().rename(columns={"Night":"Night %","Day":"Day %"})), width="stretch")

def stores_sales_summary(df: pd.DataFrame):
    if "NET_SALES" not in df.columns:
        st.error("Missing NET_SALES column.")
        return
    ss = df.groupby("STORE_NAME", as_index=False).agg(NET_SALES=("NET_SALES", "sum"), GROSS_SALES=("GROSS_SALES", "sum"))
    # customer numbers if available
    if "CUST_CODE" in df.columns:
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
    # ensure no duplicate '#' insertion
    ss_display = safe_insert_number_col(ss_display, "#")
    st.dataframe(ss_display, width="stretch")
    st.download_button("⬇️ Download Stores Summary", ss.to_csv(index=False).encode("utf-8"), "stores_sales_summary.csv", "text/csv")
    # Small chart
    fig = px.bar(ss.sort_values("GROSS_SALES", ascending=True), x="GROSS_SALES", y="STORE_NAME", orientation="h", title="Gross Sales by Store")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Dispatch selected subsection
# -------------------------
try:
    if category == "SALES":
        if subsection == "Global sales Overview":
            sales_global_overview(df)
        elif subsection == "Global Net Sales Distribution by Sales Channel":
            sales_channel2(df)
        elif subsection == "Global Net Sales Distribution by SHIFT":
            sales_by_shift(df)
        elif subsection == "Night vs Day Shift Sales Ratio — Stores with Night Shifts":
            night_day_ratio(df)
        elif subsection == "Global Day vs Night Sales — Only Stores with NIGHT Shift":
            night_day_ratio(df)  # reused view for verification
        elif subsection == "Stores Sales Summary":
            stores_sales_summary(df)
        else:
            st.info("This SALES subsection is not implemented in the safe starter yet.")
    elif category == "OPERATIONS":
        if subsection == "Customer Traffic-Storewise":
            # Simple display; more advanced views can be added after this is stable
            if "TRN_DATE" not in df.columns or "CUST_CODE" not in df.columns or "STORE_NAME" not in df.columns:
                st.error("Missing TRN_DATE, CUST_CODE, or STORE_NAME required for this view.")
            else:
                ft = df.dropna(subset=["TRN_DATE"]).copy()
                ft["TRN_DATE"] = pd.to_datetime(ft["TRN_DATE"], errors="coerce")
                ft["DATE_ONLY"] = ft["TRN_DATE"].dt.date
                first_touch = ft.groupby(["STORE_NAME", "DATE_ONLY", "CUST_CODE"], as_index=False)["TRN_DATE"].min()
                first_touch["TIME_SLOT"] = first_touch["TRN_DATE"].dt.floor("30T").dt.time
                counts = first_touch.groupby(["STORE_NAME", "TIME_SLOT"])["CUST_CODE"].nunique().reset_index(name="Receipts")
                # convert datetimes for display
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
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(convert_datetimes_for_display(summary), width="stretch")
        else:
            st.info("This OPERATIONS subsection is not implemented in the safe starter yet.")
    elif category == "INSIGHTS":
        if subsection == "Customer Baskets Overview":
            if "ITEM_NAME" not in df.columns or "CUST_CODE" not in df.columns:
                st.error("Missing ITEM_NAME or CUST_CODE.")
            else:
                topn = st.slider("Top N", 5, 100, 10)
                top_items = df.groupby("ITEM_NAME")["CUST_CODE"].nunique().rename("Count_of_Baskets").reset_index().sort_values("Count_of_Baskets", ascending=False).head(topn)
                fig = px.bar(top_items, x="Count_of_Baskets", y="ITEM_NAME", orientation="h", title=f"Top {topn} Items by Baskets")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(convert_datetimes_for_display(top_items), width="stretch")
        else:
            st.info("This INSIGHTS subsection is not implemented in the safe starter yet.")
except Exception as e:
    st.error("An unexpected error occurred while rendering the view. The app will not crash — see details below.")
    with st.expander("Error details"):
        st.text(str(e))
        st.text(traceback.format_exc())

# -------------------------
# Final hint for debugging
# -------------------------
st.markdown("---")
st.caption("If anything still fails, open the Streamlit app logs (Manage app → Logs) and paste the first ERROR/Traceback lines here. I'll analyze and provide a targeted patch.")
