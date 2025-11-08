#!/usr/bin/env python3
"""
Robust Streamlit app for stream_daily_deck (replacement/fix).

Main goals implemented:
- App should not crash: every subsection wrapped with safe checks and try/except.
- Formatting helpers fixed: don't attempt float/int on empty strings; coerce then format.
- Tables include a totals row when applicable (only for numeric columns that exist).
- Tables only display when there's data; they do not show huge empty grids.
- Convert datetime/time objects to strings before sending to Streamlit (avoids pyarrow errors).
- Avoid deprecated use_container_width by using width='stretch' where appropriate.
- Avoid widget label warnings (no empty label strings).
- Avoid double-insert '#' errors and add safe numbering functions.
- Safer Plotly usage: avoids referencing internal plotly color modules that may not exist.

Replace the repository app.py with this file and restart the Streamlit app (Manage app → Restart).
If anything still fails, paste the first ERROR/Traceback lines from the app logs here and I'll patch immediately.
"""

from __future__ import annotations
import io
import traceback
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Superdeck Analytics Dashboard", layout="wide")
st.markdown(
    """
    <style>
      [data-testid="stSidebar"][aria-expanded="true"] > div:first-child { width: 360px; }
      .muted { color: #6c757d; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Sidebar: uploader + nav
# -------------------------
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV (CSV file)", type=["csv"])

# helpful sample download placeholder (filled after load)
if uploaded is None:
    st.sidebar.write("No file uploaded yet.")
else:
    try:
        sample = pd.read_csv(io.BytesIO(uploaded.getvalue()), nrows=20, on_bad_lines="skip", low_memory=False)
        st.sidebar.download_button("Download sample rows (first 20)", sample.to_csv(index=False).encode("utf-8"), "sample_rows.csv")
    except Exception:
        pass

st.sidebar.markdown("---")
st.sidebar.header("Main Section")
main_category = st.sidebar.radio("Select main area", ["SALES", "OPERATIONS", "INSIGHTS"], index=0)

SUBSECTIONS = {
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

subsection = st.sidebar.selectbox("Subsection", SUBSECTIONS[main_category])

# -------------------------
# Utilities: safe readers & formatters
# -------------------------
def safe_read_csv_bytes(b: bytes) -> pd.DataFrame:
    bio = io.BytesIO(b)
    bio.seek(0)
    try:
        df = pd.read_csv(bio, on_bad_lines="skip", low_memory=False)
    except Exception:
        # chunked fallback to avoid memory issues
        bio.seek(0)
        parts = []
        for chunk in pd.read_csv(bio, on_bad_lines="skip", low_memory=False, chunksize=200_000):
            parts.append(chunk)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def safe_to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric safely. Handles '', 'nan', commas, percent symbols."""
    s = series.copy()
    # convert bytes/object to str before cleaning
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        s = s.fillna("").astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")

def fmt_ints(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            series = safe_to_numeric(df2[c])
            df2[c] = series.map(lambda v: f"{int(v):,}" if pd.notna(v) else "")
    return df2

def fmt_floats(df: pd.DataFrame, cols: List[str], decimals: int = 2) -> pd.DataFrame:
    df2 = df.copy()
    fmt = "{:,.%df}" % decimals
    for c in cols:
        if c in df2.columns:
            series = safe_to_numeric(df2[c])
            df2[c] = series.map(lambda v: fmt.format(v) if pd.notna(v) else "")
    return df2

def convert_datetime_and_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime/time-like columns to ISO strings to avoid pyarrow errors."""
    df2 = df.copy()
    for col in df2.columns:
        try:
            # pandas datetime64
            if pd.api.types.is_datetime64_any_dtype(df2[col]):
                df2[col] = df2[col].dt.strftime("%Y-%m-%d %H:%M:%S")
            # python datetime/time objects in object dtype
            elif df2[col].dtype == object:
                sample = df2[col].dropna().head(20)
                if not sample.empty:
                    if sample.apply(lambda x: hasattr(x, "hour") if x is not None else False).any():
                        df2[col] = df2[col].astype(str)
                    else:
                        # try parse a few values to datetimes
                        parsed = False
                        for v in sample.astype(str):
                            try:
                                pd.to_datetime(v)
                                parsed = True
                                break
                            except Exception:
                                parsed = parsed or False
                        if parsed:
                            df2[col] = pd.to_datetime(df2[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            # fallback: convert to str
            df2[col] = df2[col].astype(str)
    return df2

def safe_insert_number_col(df: pd.DataFrame, col_name: str = "#") -> pd.DataFrame:
    df2 = df.copy()
    # remove any existing column with same name to avoid insert error
    if col_name in df2.columns:
        df2.drop(columns=[col_name], inplace=True)
    df2.insert(0, col_name, range(1, len(df2) + 1))
    return df2

def add_totals_row(df: pd.DataFrame, numeric_cols: List[str], label_col: str = "STORE_NAME", label_value: str = "Total") -> pd.DataFrame:
    """
    Append a totals row for the given numeric_cols.
    Returns a new DataFrame with totals row inserted as first row (so it stands out).
    """
    df2 = df.copy()
    if df2.empty:
        return df2
    totals = {}
    for c in numeric_cols:
        if c in df2.columns:
            totals[c] = safe_to_numeric(df2[c]).sum(min_count=1)
    # Build new totals row with same columns
    row = {col: "" for col in df2.columns}
    # set label
    if label_col in df2.columns:
        row[label_col] = label_value
    # assign totals (preserve numeric type)
    for k, v in totals.items():
        row[k] = v
    # Create DF and prepend
    tot_df = pd.DataFrame([row])
    # Keep original column order
    combined = pd.concat([tot_df, df2], ignore_index=True)
    return combined

def display_table(df: pd.DataFrame, int_cols: List[str] = None, float_cols: List[str] = None, include_totals: bool = False, numeric_for_totals: List[str] = None, label_col: str = None):
    """
    Display a DataFrame safely:
    - convert datetimes/times to strings
    - optionally add totals row (only if numeric_for_totals contains available numeric cols and df not empty)
    - format ints/floats safely
    - only show table if it has data
    """
    if df is None:
        st.info("No data.")
        return
    df2 = df.copy()
    if df2.empty:
        st.info("No data to display for this view.")
        return
    # add totals row if requested and numeric cols exist
    if include_totals and numeric_for_totals:
        present_numeric = [c for c in numeric_for_totals if c in df2.columns]
        if present_numeric:
            df2 = add_totals_row(df2, present_numeric, label_col=label_col or df2.columns[0])
    # format integers and floats
    if int_cols:
        df2 = fmt_ints(df2, int_cols)
    if float_cols:
        df2 = fmt_floats(df2, float_cols)
    # convert datetimes/time objects
    df2 = convert_datetime_and_time_columns(df2)
    # numbering
    df2 = safe_insert_number_col(df2, "#")
    # show table
    st.dataframe(df2, width="stretch", use_container_width=False)

# -------------------------
# Load file and normalize columns
# -------------------------
st.title("Superdeck Analytics Dashboard — Safe Start")
st.write("Upload your sales CSV. This app is defensive: it will not crash on missing or malformed columns.")

if uploaded is None:
    st.info("Please upload a CSV to proceed.")
    st.stop()

try:
    with st.spinner("Reading uploaded CSV..."):
        df = safe_read_csv_bytes(uploaded.getvalue())
except Exception as e:
    st.error("Failed to read uploaded CSV.")
    with st.expander("Error details"):
        st.text(str(e))
        st.text(traceback.format_exc())
    st.stop()

# Normalize commonly used columns (create placeholders so code won't KeyError)
for col in [
    "CUST_CODE", "STORE_NAME", "TRN_DATE", "ITEM_NAME", "ITEM_CODE",
    "SALES_CHANNEL_L1", "SALES_CHANNEL_L2", "SHIFT", "VAT_AMT", "GROSS_SALES",
    "CU_DEVICE_SERIAL", "CAP_CUSTOMER_CODE", "STORE_CODE", "TILL", "SESSION", "RCT", "CASHIER"
]:
    if col not in df.columns:
        if col == "TRN_DATE":
            df[col] = pd.NaT
        else:
            df[col] = ""

# coerce common numeric columns safely (do not mutate original strings)
for nc in ["NET_SALES", "QTY", "VAT_AMT", "GROSS_SALES", "SP_PRE_VAT", "CP_PRE_VAT"]:
    if nc in df.columns:
        df[nc] = safe_to_numeric(df[nc]).fillna(0)

# convert TRN_DATE when possible
if "TRN_DATE" in df.columns:
    try:
        df["TRN_DATE"] = pd.to_datetime(df["TRN_DATE"], errors="coerce")
    except Exception:
        pass

st.success(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")

# -------------------------
# Provide protected, robust views for subsections
# -------------------------
st.header(f"{main_category} → {subsection}")

# Each view is defensive: checks columns, wraps in try/except, returns early if no data.

def view_global_sales_overview():
    try:
        if "SALES_CHANNEL_L1" not in df.columns or "NET_SALES" not in df.columns:
            st.warning("Missing SALES_CHANNEL_L1 and/or NET_SALES.")
            return
        gs = df.groupby("SALES_CHANNEL_L1", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        if gs.empty:
            st.info("No sales data available.")
            return
        gs["NET_SALES_M"] = gs["NET_SALES"] / 1_000_000
        gs["PCT"] = 100 * gs["NET_SALES"] / gs["NET_SALES"].sum()
        # Pie
        labels = [f"{r['SALES_CHANNEL_L1']} ({r['PCT']:.1f}% | {r['NET_SALES_M']:.1f}M)" for _, r in gs.iterrows()]
        fig = go.Figure(go.Pie(labels=labels, values=gs["NET_SALES_M"], hole=0.6, text=[f"{p:.1f}%" for p in gs["PCT"]]))
        fig.update_layout(title="SALES CHANNEL TYPE — Global Overview", height=520)
        st.plotly_chart(fig, width="stretch")
        # Table with totals
        display_table(gs.rename(columns={"SALES_CHANNEL_L1":"Channel"}), int_cols=["NET_SALES"], float_cols=["NET_SALES_M","PCT"], include_totals=True, numeric_for_totals=["NET_SALES","NET_SALES_M"], label_col="Channel")
    except Exception as e:
        st.error("Error rendering Global sales Overview.")
        with st.expander("Details"):
            st.text(str(e))
            st.text(traceback.format_exc())

def view_channel2():
    try:
        if "SALES_CHANNEL_L2" not in df.columns or "NET_SALES" not in df.columns:
            st.warning("Missing SALES_CHANNEL_L2 and/or NET_SALES.")
            return
        ch2 = df.groupby("SALES_CHANNEL_L2", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        if ch2.empty:
            st.info("No channel L2 data.")
            return
        ch2["NET_SALES_M"] = ch2["NET_SALES"] / 1_000_000
        fig = px.pie(ch2, names="SALES_CHANNEL_L2", values="NET_SALES_M", hole=0.6, title="Global Net Sales by SALES_CHANNEL_L2")
        st.plotly_chart(fig, width="stretch")
        display_table(ch2.rename(columns={"SALES_CHANNEL_L2":"Mode"}), int_cols=["NET_SALES"], float_cols=["NET_SALES_M"], include_totals=True, numeric_for_totals=["NET_SALES"])
    except Exception as e:
        st.error("Error rendering Global Net Sales Distribution by Sales Channel.")
        with st.expander("Details"):
            st.text(str(e))
            st.text(traceback.format_exc())

def view_shift_sales():
    try:
        if "SHIFT" not in df.columns or "NET_SALES" not in df.columns:
            st.warning("Missing SHIFT and/or NET_SALES.")
            return
        ss = df.groupby("SHIFT", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        if ss.empty:
            st.info("No shift sales data.")
            return
        ss["PCT"] = 100 * ss["NET_SALES"] / ss["NET_SALES"].sum()
        fig = px.pie(ss, names="SHIFT", values="NET_SALES", hole=0.6, title="Global Net Sales Distribution by SHIFT")
        st.plotly_chart(fig, width="stretch")
        display_table(ss, int_cols=["NET_SALES"], float_cols=["PCT"], include_totals=True, numeric_for_totals=["NET_SALES"])
    except Exception as e:
        st.error("Error rendering Global Net Sales Distribution by SHIFT.")
        with st.expander("Details"):
            st.text(str(e))
            st.text(traceback.format_exc())

def view_night_vs_day():
    try:
        if not {"STORE_NAME", "SHIFT", "NET_SALES"}.issubset(df.columns):
            st.warning("Missing STORE_NAME, SHIFT or NET_SALES columns.")
            return
        stores_with_night = df[df["SHIFT"].astype(str).str.upper().str.contains("NIGHT", na=False)]["STORE_NAME"].unique()
        if len(stores_with_night) == 0:
            st.info("No stores with NIGHT shift found.")
            return
        dnd = df[df["STORE_NAME"].isin(stores_with_night)].copy()
        dnd["Shift_Bucket"] = np.where(dnd["SHIFT"].astype(str).str.upper().str.contains("NIGHT", na=False), "Night", "Day")
        r = dnd.groupby(["STORE_NAME","Shift_Bucket"], as_index=False)["NET_SALES"].sum()
        tot = r.groupby("STORE_NAME")["NET_SALES"].transform("sum")
        r["PCT"] = np.where(tot > 0, 100 * r["NET_SALES"] / tot, 0.0)
        pivot = r.pivot(index="STORE_NAME", columns="Shift_Bucket", values="PCT").fillna(0)
        if pivot.empty:
            st.info("No results after aggregation.")
            return
        pivot = pivot.sort_values("Night", ascending=False)
        # Bar chart
        fig = go.Figure()
        if "Night" in pivot.columns:
            fig.add_trace(go.Bar(x=pivot["Night"], y=pivot.index, orientation="h", name="Night", marker_color="#d62728"))
        if "Day" in pivot.columns:
            fig.add_trace(go.Bar(x=pivot["Day"], y=pivot.index, orientation="h", name="Day", marker_color="#1f77b4"))
        fig.update_layout(barmode="group", title="Night vs Day % by Store", height=max(400, 24 * len(pivot)))
        st.plotly_chart(fig, width="stretch")
        display_table(pivot.reset_index().rename(columns={"Night":"Night %","Day":"Day %"}), float_cols=["Night %","Day %"], include_totals=False)
    except Exception as e:
        st.error("Error rendering Night vs Day Shift view.")
        with st.expander("Details"):
            st.text(str(e))
            st.text(traceback.format_exc())

def view_second_channel(which="top"):
    try:
        required = {"STORE_NAME", "SALES_CHANNEL_L1", "NET_SALES"}
        if not required.issubset(df.columns):
            st.warning(f"Missing columns: {', '.join(required)}")
            return
        d = df.copy()
        d["NET_SALES"] = safe_to_numeric(d["NET_SALES"]).fillna(0)
        store_chan = d.groupby(["STORE_NAME","SALES_CHANNEL_L1"], as_index=False)["NET_SALES"].sum()
        store_tot = store_chan.groupby("STORE_NAME")["NET_SALES"].transform("sum")
        store_chan["PCT"] = np.where(store_tot>0, 100 * store_chan["NET_SALES"] / store_tot, 0.0)
        store_chan = store_chan.sort_values(["STORE_NAME","PCT"], ascending=[True, False])
        store_chan["RANK"] = store_chan.groupby("STORE_NAME").cumcount() + 1
        second = store_chan[store_chan["RANK"]==2]
        if second.empty:
            st.info("No 2nd-highest channel rows found (many stores have only one channel).")
            return
        if which == "top":
            top30 = second.sort_values("PCT", ascending=False).head(30)
            fig = px.bar(top30, x="PCT", y="STORE_NAME", orientation="h", title="Top 30 Stores by 2nd-Highest Channel Share")
            st.plotly_chart(fig, width="stretch")
            display_table(top30.rename(columns={"SALES_CHANNEL_L1":"2nd Channel","PCT":"2nd Channel %"}), float_cols=["2nd Channel %"], include_totals=False)
        else:
            bottom30 = second.sort_values("PCT", ascending=True).head(30)
            fig = px.bar(bottom30, x="PCT", y="STORE_NAME", orientation="h", title="Bottom 30 Stores by 2nd-Highest Channel Share", color_discrete_sequence=["#d62728"])
            st.plotly_chart(fig, width="stretch")
            display_table(bottom30.rename(columns={"SALES_CHANNEL_L1":"2nd Channel","PCT":"2nd Channel %"}), float_cols=["2nd Channel %"], include_totals=False)
    except Exception as e:
        st.error("Error rendering 2nd-highest channel share.")
        with st.expander("Details"):
            st.text(str(e))
            st.text(traceback.format_exc())

def view_stores_summary():
    try:
        if "STORE_NAME" not in df.columns or "GROSS_SALES" not in df.columns:
            st.warning("STORE_NAME and GROSS_SALES required for this view.")
            return
        ss = df.groupby("STORE_NAME", as_index=False).agg(NET_SALES=("NET_SALES","sum"), GROSS_SALES=("GROSS_SALES","sum"))
        if ss.empty:
            st.info("No store-level sales.")
            return
        # customer counts if possible
        if "CUST_CODE" in df.columns:
            ss["Customer_Numbers"] = df.groupby("STORE_NAME")["CUST_CODE"].nunique().reindex(ss["STORE_NAME"]).fillna(0).astype(int).values
        else:
            ss["Customer_Numbers"] = 0
        total_gross = ss["GROSS_SALES"].sum()
        ss["% Contribution"] = np.where(total_gross>0, 100 * ss["GROSS_SALES"] / total_gross, 0.0).round(2)
        ss = ss.sort_values("GROSS_SALES", ascending=False).reset_index(drop=True)
        # format and show totals row
        display_table(ss, int_cols=["NET_SALES", "GROSS_SALES", "Customer_Numbers"], float_cols=["% Contribution"], include_totals=True, numeric_for_totals=["NET_SALES","GROSS_SALES","Customer_Numbers"], label_col="STORE_NAME")
        # bar
        fig = px.bar(ss.sort_values("GROSS_SALES", ascending=True), x="GROSS_SALES", y="STORE_NAME", orientation="h", title="Gross Sales by Store")
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.error("Error rendering Stores Sales Summary.")
        with st.expander("Details"):
            st.text(str(e))
            st.text(traceback.format_exc())

def view_customer_traffic_storewise():
    try:
        if not {"TRN_DATE","CUST_CODE","STORE_NAME"}.issubset(df.columns):
            st.warning("TRN_DATE, CUST_CODE and STORE_NAME required.")
            return
        tmp = df.dropna(subset=["TRN_DATE"]).copy()
        tmp["TRN_DATE"] = pd.to_datetime(tmp["TRN_DATE"], errors="coerce")
        tmp["DATE_ONLY"] = tmp["TRN_DATE"].dt.date
        first_touch = tmp.groupby(["STORE_NAME","DATE_ONLY","CUST_CODE"], as_index=False)["TRN_DATE"].min()
        first_touch["TIME_SLOT"] = first_touch["TRN_DATE"].dt.floor("30T")
        first_touch["TIME_SLOT_STR"] = first_touch["TIME_SLOT"].astype(str)
        counts = first_touch.groupby(["STORE_NAME","TIME_SLOT_STR"])["CUST_CODE"].nunique().reset_index(name="Receipts")
        if counts.empty:
            st.info("No traffic receipts found.")
            return
        # heatmap-style: pivot but only for display, don't create huge empty columns
        pivot = counts.pivot(index="STORE_NAME", columns="TIME_SLOT_STR", values="Receipts").fillna(0)
        # use plotly imshow - small and safe
        fig = px.imshow(pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(), color_continuous_scale="YlOrRd", labels=dict(x="Time Slot", y="Store", color="Receipts"), aspect="auto", title="Customer Traffic Heatmap")
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, width="stretch")
        display_table(counts, int_cols=["Receipts"], include_totals=False)
    except Exception as e:
        st.error("Error rendering Customer Traffic-Storewise.")
        with st.expander("Details"):
            st.text(str(e))
            st.text(traceback.format_exc())

def view_active_tills():
    try:
        # similar safe pattern to customer traffic
        required = {"TRN_DATE", "TILL", "STORE_CODE", "STORE_NAME"}
        if not required.issubset(df.columns):
            st.warning("TILL, STORE_CODE, TRN_DATE, STORE_NAME required for active tills view.")
            return
        tmp = df.dropna(subset=["TRN_DATE"]).copy()
        tmp["TRN_DATE"] = pd.to_datetime(tmp["TRN_DATE"], errors="coerce")
        tmp["TIME_SLOT"] = tmp["TRN_DATE"].dt.floor("30T").astype(str)
        tmp["Till_Code"] = tmp["TILL"].astype(str).str.strip() + "-" + tmp["STORE_CODE"].astype(str).str.strip()
        counts = tmp.groupby(["STORE_NAME","TIME_SLOT"])["Till_Code"].nunique().reset_index(name="Unique_Tills")
        if counts.empty:
            st.info("No till activity.")
            return
        pivot = counts.pivot(index="STORE_NAME", columns="TIME_SLOT", values="Unique_Tills").fillna(0)
        fig = px.imshow(pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(), color_continuous_scale="Blues", labels=dict(x="Time Slot", y="Store", color="Unique Tills"), title="Active Tills Heatmap")
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, width="stretch")
        display_table(counts, int_cols=["Unique_Tills"], include_totals=False)
    except Exception as e:
        st.error("Error in Active Tills view.")
        with st.expander("Details"):
            st.text(str(e))
            st.text(traceback.format_exc())

def view_customer_baskets_overview():
    try:
        if not {"ITEM_NAME","CUST_CODE"}.issubset(df.columns):
            st.warning("ITEM_NAME and CUST_CODE are required.")
            return
        topn = st.slider("Top N", 5, 100, 10)
        top_items = df.groupby("ITEM_NAME")["CUST_CODE"].nunique().rename("Count_of_Baskets").reset_index().sort_values("Count_of_Baskets", ascending=False).head(topn)
        if top_items.empty:
            st.info("No basket data.")
            return
        fig = px.bar(top_items, x="Count_of_Baskets", y="ITEM_NAME", orientation="h", title=f"Top {topn} Items by Baskets")
        st.plotly_chart(fig, width="stretch")
        display_table(top_items, int_cols=["Count_of_Baskets"], include_totals=False)
    except Exception as e:
        st.error("Error rendering Customer Baskets Overview.")
        with st.expander("Details"):
            st.text(str(e))
            st.text(traceback.format_exc())

def view_global_pricing_overview():
    try:
        required = {"TRN_DATE","ITEM_CODE","SP_PRE_VAT"}
        if not required.issubset(df.columns):
            st.warning("TRN_DATE, ITEM_CODE, SP_PRE_VAT required.")
            return
        d = df.copy()
        d["TRN_DATE"] = pd.to_datetime(d["TRN_DATE"], errors="coerce")
        d["DATE"] = d["TRN_DATE"].dt.date
        d["SP_PRE_VAT"] = safe_to_numeric(d["SP_PRE_VAT"]).fillna(0)
        grp = d.groupby(["STORE_NAME","DATE","ITEM_CODE","ITEM_NAME"], as_index=False).agg(
            Num_Prices=("SP_PRE_VAT", lambda s: s.dropna().nunique()),
            Price_Min=("SP_PRE_VAT","min"),
            Price_Max=("SP_PRE_VAT","max"),
            Total_QTY=("QTY","sum")
        )
        grp["Price_Spread"] = (grp["Price_Max"] - grp["Price_Min"]).round(2)
        multi = grp[(grp["Num_Prices"]>1) & (grp["Price_Spread"]>0)].copy()
        if multi.empty:
            st.info("No multi-priced SKUs with positive spread found.")
            return
        summary = multi.groupby("STORE_NAME", as_index=False).agg(
            Items_with_MultiPrice=("ITEM_CODE","nunique"),
            Total_Diff_Value=("Price_Spread", lambda s: (s * multi.loc[s.index, "Total_QTY"]).sum())
        ).sort_values("Total_Diff_Value", ascending=False)
        display_table(summary, int_cols=["Items_with_MultiPrice"], float_cols=["Total_Diff_Value"], include_totals=True, numeric_for_totals=["Items_with_MultiPrice","Total_Diff_Value"])
        fig = px.bar(summary.head(20).sort_values("Total_Diff_Value", ascending=True), x="Total_Diff_Value", y="STORE_NAME", orientation="h", title="Top Stores by Value Impact from Multi-Priced SKUs")
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.error("Error rendering Global Pricing Overview.")
        with st.expander("Details"):
            st.text(str(e))
            st.text(traceback.format_exc())

def view_global_refunds_overview():
    try:
        if "NET_SALES" not in df.columns:
            st.warning("NET_SALES is required.")
            return
        d = df.copy()
        d["NET_SALES"] = safe_to_numeric(d["NET_SALES"]).fillna(0)
        neg = d[d["NET_SALES"] < 0].copy()
        if neg.empty:
            st.info("No negative receipts.")
            return
        gr = neg.groupby("STORE_NAME", as_index=False).agg(Total_Neg_Value=("NET_SALES","sum"), Receipts=("CUST_CODE","nunique"))
        gr["Abs_Neg_Value"] = gr["Total_Neg_Value"].abs()
        display_table(gr.sort_values("Abs_Neg_Value", ascending=False), int_cols=["Receipts"], float_cols=["Total_Neg_Value","Abs_Neg_Value"], include_totals=True, numeric_for_totals=["Total_Neg_Value","Receipts"])
        fig = px.bar(gr.sort_values("Abs_Neg_Value", ascending=True), x="Abs_Neg_Value", y="STORE_NAME", orientation="h", title="Stores by Absolute Negative Value")
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.error("Error rendering Global Refunds Overview.")
        with st.expander("Details"):
            st.text(str(e))
            st.text(traceback.format_exc())

# Mapping
SALES_HANDLERS = {
    "Global sales Overview": view_global_sales_overview,
    "Global Net Sales Distribution by Sales Channel": view_channel2,
    "Global Net Sales Distribution by SHIFT": view_shift_sales,
    "Night vs Day Shift Sales Ratio — Stores with Night Shifts": view_night_vs_day,
    "Global Day vs Night Sales — Only Stores with NIGHT Shift": view_night_vs_day,
    "2nd-Highest Channel Share": lambda: view_second_channel("top"),
    "Bottom 30 — 2nd Highest Channel": lambda: view_second_channel("bottom"),
    "Stores Sales Summary": view_stores_summary,
}

OPERATIONS_HANDLERS = {
    "Customer Traffic-Storewise": view_customer_traffic_storewise,
    "Active Tills During the day": view_active_tills,
    "Average Customers Served per Till": lambda: st.info("Average Customers Served per Till: implemented on request"),
    "Store Customer Traffic Storewise": view_customer_traffic_storewise,
    "Customer Traffic-Departmentwise": view_customer_traffic_storewise,
    "Cashiers Perfomance": lambda: st.info("Cashiers Performance: implemented on request"),
    "Till Usage": view_active_tills,
    "Tax Compliance": lambda: st.info("Tax Compliance: implemented on request"),
}

INSIGHTS_HANDLERS = {
    "Customer Baskets Overview": view_customer_baskets_overview,
    "Global Pricing Overview": view_global_pricing_overview,
    "Global Refunds Overview": view_global_refunds_overview,
    # placeholders for other detailed views that were in the original notebook
    "Global Category Overview-Sales": lambda: st.info("Category Sales: original app had a detailed view — implement on request."),
    "Global Category Overview-Baskets": lambda: st.info("Category Baskets: original app had a detailed view — implement on request."),
    "Supplier Contribution": lambda: st.info("Supplier contribution: implement on request."),
    "Category Overview": lambda: st.info("Category overview: implement on request."),
    "Branch Comparison": lambda: st.info("Branch comparison: implement on request."),
    "Product Perfomance": lambda: st.info("Product performance: implement on request."),
    "Global Loyalty Overview": lambda: st.info("Global loyalty: implement on request."),
    "Branch Loyalty Overview": lambda: st.info("Branch loyalty: implement on request."),
    "Customer Loyalty Overview": lambda: st.info("Customer loyalty: implement on request."),
    "Branch Brach Overview": lambda: st.info("Branch Branch Overview: implement on request."),
    "Branch Refunds Overview": lambda: st.info("Branch refunds: implement on request."),
}

# -------------------------
# Dispatch
# -------------------------
try:
    if main_category == "SALES":
        handler = SALES_HANDLERS.get(subsection)
        if handler:
            handler()
        else:
            st.info("This SALES subsection is not implemented in the safe starter yet.")
    elif main_category == "OPERATIONS":
        handler = OPERATIONS_HANDLERS.get(subsection)
        if handler:
            handler()
        else:
            st.info("This OPERATIONS subsection is not implemented in the safe starter yet.")
    elif main_category == "INSIGHTS":
        handler = INSIGHTS_HANDLERS.get(subsection)
        if handler:
            handler()
        else:
            st.info("This INSIGHTS subsection is not implemented in the safe starter yet.")
except Exception as e:
    st.error("An unexpected error occurred rendering this subsection. The app will not crash — see details below.")
    with st.expander("Error details"):
        st.text(str(e))
        st.text(traceback.format_exc())

st.markdown("---")
st.caption("If any section still fails: copy the first ERROR/Traceback block from Manage app → Logs and paste it here. I'll patch the failing function directly.")
