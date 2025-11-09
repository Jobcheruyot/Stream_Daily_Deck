"""
Superdeck Analytics Dashboard - Fixed table rendering for Global sales Overview

This patch addresses:
- "Unable to render table due to an internal error" observed when showing the
  Global sales Overview table in Streamlit.
Root cause and fix:
- The prior table-rendering helper relied on pyarrow/Streamlit converting mixed-type
  columns (e.g. floats + empty strings + occasional datetime.time) which could raise
  serialization errors. Some totals rows also introduced mixed dtypes (numbers + "").
- Solution: before calling st.dataframe, convert the DataFrame to a safe string-only
  display copy where numeric columns are formatted into strings and any datetime.time
  values are converted to HH:MM strings. This guarantees Streamlit won't fail during
  pyarrow serialization and preserves the visual formatting (commas, decimals, %).
- The Global sales Overview section is updated to use the safe formatter.

Usage:
  streamlit run app.py
"""
from datetime import timedelta, time as dtime
import io
import hashlib
import traceback
import sys

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide", page_title="Superdeck Analytics Dashboard", initial_sidebar_state="expanded")

# ---------- Colors ----------
COLOR_BLUE = "#1f77b4"
COLOR_ORANGE = "#ff7f0e"
COLOR_GREEN = "#2ca02c"
COLOR_RED = "#d62728"
PALETTE10 = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_RED, "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
DIVERGING = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]

# ---------- Safe display helpers ----------
def _safe_time_to_str(x):
    if isinstance(x, dtime):
        return x.strftime("%H:%M")
    return x

def _format_display_df(df: pd.DataFrame, int_cols=None, float_cols=None, pct_cols=None, percent_fmt="1"):
    """
    Return a copy of df where:
    - int_cols are formatted with thousands separators,
    - float_cols are formatted with comma and 2 decimals,
    - pct_cols are formatted as 'x.x%' strings,
    - any datetime.time objects are converted to "HH:MM" strings,
    - all other values are converted to strings (preserves empty strings).
    This guarantees Streamlit/pyarrow will not error when serializing the table.
    """
    df = df.copy()
    # Convert datetime.time objects in any column
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna().head(20)
            if any(isinstance(v, dtime) for v in sample):
                df[c] = df[c].map(lambda v: v.strftime("%H:%M") if isinstance(v, dtime) else v)

    # Format numeric columns into strings
    if int_cols:
        for c in int_cols:
            if c in df.columns:
                df[c] = df[c].map(lambda v: f"{int(v):,}" if (pd.notna(v) and str(v) != "") else "")
    if float_cols:
        for c in float_cols:
            if c in df.columns:
                df[c] = df[c].map(lambda v: f"{float(v):,.2f}" if (pd.notna(v) and str(v) != "") else "")
    if pct_cols:
        for c in pct_cols:
            if c in df.columns:
                df[c] = df[c].map(lambda v: f"{float(v):.1f}%" if (pd.notna(v) and str(v) != "") else "")

    # Convert any remaining values to strings (so dtype is object everywhere)
    for c in df.columns:
        # Avoid converting already formatted numeric strings
        df[c] = df[c].map(lambda v: "" if pd.isna(v) else str(v))

    return df

def display_table_with_format_safe(df: pd.DataFrame, int_cols=None, float_cols=None, pct_cols=None, height=300):
    """
    Safely display formatted table in Streamlit using _format_display_df to avoid pyarrow errors.
    """
    if df is None or df.empty:
        st.info("No data available for this view.")
        return
    try:
        df_out = _format_display_df(df.copy(), int_cols=int_cols, float_cols=float_cols, pct_cols=pct_cols)
        st.dataframe(df_out, width='stretch', height=height)
    except Exception:
        # Fallback: display as plain text table to avoid crash and log the detailed traceback.
        st.error("Unable to render formatted table in the normal way. Showing a plain representation and logging details.")
        st.text(df.head(50).to_string())
        traceback.print_exc(file=sys.stdout)

def add_total_row(df: pd.DataFrame, numeric_cols: list, label_col: str = None, total_label="Total"):
    """
    Insert a top total row; totals computed only for numeric_cols.
    The function returns the DataFrame with potentially mixed dtypes — caller should
    send it through display_table_with_format_safe before st.dataframe.
    """
    if df is None or df.empty:
        return df.copy()
    totals = {}
    for c in numeric_cols:
        if c in df.columns:
            try:
                totals[c] = df[c].sum()
            except Exception:
                totals[c] = ""
    row = {c: "" for c in df.columns}
    for c, v in totals.items():
        row[c] = v
    if label_col and label_col in df.columns:
        row[label_col] = total_label
    else:
        row[df.columns[0]] = total_label
    top = pd.DataFrame([row])
    return pd.concat([top, df], ignore_index=True)

# ---------- Data loader and precompute ----------
@st.cache_data(show_spinner=True)
def load_and_precompute(file_bytes: bytes) -> dict:
    df = pd.read_csv(io.BytesIO(file_bytes), on_bad_lines="skip", low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # parse dates
    for col in ["TRN_DATE", "ZED_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # numeric conversion
    numeric_cols = ["QTY", "CP_PRE_VAT", "SP_PRE_VAT", "COST_PRE_VAT", "NET_SALES", "VAT_AMT"]
    for nc in numeric_cols:
        if nc in df.columns:
            df[nc] = df[nc].astype(str).str.replace(",", "", regex=False)
            df[nc] = pd.to_numeric(df[nc], errors="coerce").fillna(0)

    # ensure id columns are strings
    idcols = ["STORE_CODE", "TILL", "SESSION", "RCT"]
    for c in idcols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()

    # Build CUST_CODE if missing
    if "CUST_CODE" not in df.columns:
        if all(c in df.columns for c in idcols):
            df["CUST_CODE"] = (df["STORE_CODE"].str.strip() + "-" + df["TILL"].str.strip() + "-" + df["SESSION"].str.strip() + "-" + df["RCT"].str.strip())
        else:
            df["CUST_CODE"] = df.index.astype(str)
    df["CUST_CODE"] = df["CUST_CODE"].astype(str).str.strip()

    out = {"df": df}

    # Global sales by SALES_CHANNEL_L1
    if "SALES_CHANNEL_L1" in df.columns and "NET_SALES" in df.columns:
        gs = df.groupby("SALES_CHANNEL_L1", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        gs["NET_SALES_M"] = gs["NET_SALES"] / 1_000_000
        total = gs["NET_SALES"].sum()
        gs["PCT"] = gs["NET_SALES"] / total * 100 if total != 0 else 0
        out["sales_channel_l1"] = gs
    else:
        out["sales_channel_l1"] = pd.DataFrame()

    # other precomputations omitted for brevity (they can be added as in prior versions)
    out["sample_rows"] = df.head(200)
    return out

# ---------- UI ----------
st.title("🦸 Superdeck Analytics Dashboard (fixed table renderer)")
uploaded = st.file_uploader("Upload CSV (CSV)", type="csv")
if uploaded is None:
    st.info("Please upload a dataset to proceed.")
    st.stop()

file_bytes = uploaded.getvalue()
try:
    state = load_and_precompute(file_bytes)
except Exception:
    st.error("Failed to load dataset; check the file and server logs.")
    traceback.print_exc(file=sys.stdout)
    st.stop()

st.sidebar.download_button("⬇️ Download sample rows", state["sample_rows"].to_csv(index=False).encode("utf-8"), "sample_rows.csv", "text/csv")

# Minimal navigation for demonstration: only implement Global sales Overview here (the user requested fix)
section = st.sidebar.selectbox("Section", ["SALES"])
subsection = st.sidebar.selectbox("Subsection", ["Global sales Overview"])

if section == "SALES" and subsection == "Global sales Overview":
    gs = state.get("sales_channel_l1", pd.DataFrame())
    if gs.empty:
        st.warning("SALES_CHANNEL_L1 or NET_SALES not present in uploaded CSV.")
    else:
        # Ensure NET_SALES is numeric (as notebook)
        # (precompute already converted NET_SALES but double-ensure here)
        gs_local = gs.copy()
        gs_local["NET_SALES"] = pd.to_numeric(gs_local["NET_SALES"], errors="coerce").fillna(0)
        gs_local["NET_SALES_M"] = (gs_local["NET_SALES"] / 1_000_000).round(2)
        total_sales = gs_local["NET_SALES"].sum()
        gs_local["PCT"] = (gs_local["NET_SALES"] / total_sales * 100).round(1) if total_sales != 0 else 0.0

        # Legend labels as notebook
        legend_labels = [f"{row['SALES_CHANNEL_L1']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f} M)" for _, row in gs_local.iterrows()]
        values = gs_local["NET_SALES_M"]

        colors = PALETTE10  # safe explicit palette
        try:
            fig = go.Figure(data=[go.Pie(
                labels=legend_labels,
                values=values,
                hole=0.65,
                text=[f"{p:.1f}%" for p in gs_local["PCT"]],
                textinfo='text',
                textposition='inside',
                insidetextorientation='auto',
                sort=True,
                marker=dict(colors=colors, line=dict(color='white', width=1)),
                hovertemplate='<b>%{label}</b><br>KSh %{value:,.2f} M<extra></extra>'
            )])
            fig.update_layout(
                title="<b>SALES CHANNEL TYPE — Global Overview</b>",
                title_x=0.42,
                margin=dict(l=40, r=40, t=70, b=40),
                legend_title_text="Sales Channels (% | KSh Millions)",
                showlegend=True,
                height=600
            )
            st.plotly_chart(fig, width='stretch')
        except Exception:
            st.error("Failed to render pie chart (non-fatal). Showing table instead.")
            traceback.print_exc(file=sys.stdout)

        # Table with totals and safe formatting — THIS is the robust path that prevents the pyarrow error.
        df_out = gs_local[["SALES_CHANNEL_L1", "NET_SALES", "NET_SALES_M", "PCT"]].copy()
        df_out = add_total_row(df_out, numeric_cols=["NET_SALES"], label_col="SALES_CHANNEL_L1")
        # Use the safe display function which converts all columns to strings and formats numbers
        display_table_with_format_safe(df_out, int_cols=["NET_SALES"], float_cols=["NET_SALES_M"], pct_cols=["PCT"], height=420)
        st.download_button("⬇️ Download table CSV", _format_display_df(df_out, int_cols=["NET_SALES"], float_cols=["NET_SALES_M"], pct_cols=["PCT"]).to_csv(index=False).encode("utf-8"), file_name="global_sales_overview.csv", mime="text/csv")
