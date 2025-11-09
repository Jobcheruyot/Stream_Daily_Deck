"""
Superdeck Analytics Dashboard - Full faithful Streamlit port (SALES → INSIGHTS)

- Implements all requested sections and subsections from SALES, OPERATIONS and INSIGHTS.
- Precomputes summaries at upload time (cached) for fast subsection switching.
- Tables include a single totals row where applicable and use thousands separators.
- Percentages formatted with one decimal and shown consistently.
- Converts datetime.time objects to strings before sending DataFrames to Streamlit to avoid pyarrow errors.
- Defensive plotting and table rendering so one failing visual won't crash the app.
- Uses explicit color palettes (avoids internal Plotly palettes that caused AttributeError).
- Replaces deprecated use_container_width with width='stretch'.

Usage:
    streamlit run app.py

Notes:
- If your CSV is very large, deploy on a node with sufficient memory; consider a streaming/aggregation approach.
- To allow uploads larger than default, set STREAMLIT_SERVER_MAX_UPLOAD_SIZE in .streamlit/config.toml or env.
"""

from datetime import timedelta, time as dtime
import io
import hashlib
import textwrap
import traceback
import sys

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- Page config ----------
st.set_page_config(layout="wide", page_title="Superdeck Analytics Dashboard", initial_sidebar_state="expanded")

# ---------- Theme / Colors ----------
COLOR_BLUE = "#1f77b4"
COLOR_ORANGE = "#ff7f0e"
COLOR_GREEN = "#2ca02c"
COLOR_RED = "#d62728"
PALETTE10 = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_RED, "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
DIVERGING = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]

# ---------- Helpers (defined before UI usage) ----------
def _sha256_bytes(b: bytes) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def _safe_time_to_str(val):
    if isinstance(val, dtime):
        return val.strftime("%H:%M")
    return val

def _format_display_df(df: pd.DataFrame, int_cols=None, float_cols=None, pct_cols=None, pct_decimals=1, float_decimals=2):
    """
    Return a copy of df where:
    - int_cols formatted with thousands separators
    - float_cols formatted with comma and float_decimals
    - pct_cols formatted with pct_decimals and a trailing '%'
    - datetime.time objects converted to HH:MM strings
    - all values converted to strings to avoid pyarrow serialization issues
    """
    df2 = df.copy()
    # convert time objects
    for c in df2.columns:
        if df2[c].dtype == object:
            sample = df2[c].dropna().head(20)
            if any(isinstance(v, dtime) for v in sample):
                df2[c] = df2[c].map(lambda v: v.strftime("%H:%M") if isinstance(v, dtime) else v)
    # format numeric columns
    if int_cols:
        for c in int_cols:
            if c in df2.columns:
                df2[c] = df2[c].map(lambda v: f"{int(v):,}" if pd.notna(v) and str(v) != "" else "")
    if float_cols:
        for c in float_cols:
            if c in df2.columns:
                fmt = f"{{:,.{float_decimals}f}}"
                df2[c] = df2[c].map(lambda v: fmt.format(float(v)) if pd.notna(v) and str(v) != "" else "")
    if pct_cols:
        for c in pct_cols:
            if c in df2.columns:
                fmt = f"{{:.{pct_decimals}f}}%"
                df2[c] = df2[c].map(lambda v: fmt.format(float(v)) if pd.notna(v) and str(v) != "" else "")
    # convert other values to string
    for c in df2.columns:
        df2[c] = df2[c].map(lambda v: "" if pd.isna(v) else str(v))
    return df2

def add_total_row(df: pd.DataFrame, numeric_cols: list, label_col: str = None, total_label="Total"):
    """
    Add a single totals row at the top summing numeric_cols where present.
    Returns a new DataFrame (may contain mixed types). Caller should run through _format_display_df.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    totals = {}
    for c in numeric_cols:
        if c in df.columns:
            try:
                totals[c] = df[c].sum()
            except Exception:
                totals[c] = ""
    total_row = {c: "" for c in df.columns}
    for c, v in totals.items():
        total_row[c] = v
    if label_col and label_col in df.columns:
        total_row[label_col] = total_label
    else:
        total_row[df.columns[0]] = total_label
    top = pd.DataFrame([total_row])
    out = pd.concat([top, df], ignore_index=True)
    return out

def display_table_with_format(df: pd.DataFrame, int_cols=None, float_cols=None, pct_cols=None, height=360):
    """
    Safely render a DataFrame in Streamlit with formatted strings (avoids pyarrow errors).
    """
    if df is None or df.empty:
        st.info("No data available for this view.")
        return
    try:
        df_out = _format_display_df(df.copy(), int_cols=int_cols, float_cols=float_cols, pct_cols=pct_cols)
        st.dataframe(df_out, width='stretch', height=height)
    except Exception:
        st.error("Unable to render table due to an internal error. See logs for details.")
        traceback.print_exc(file=sys.stdout)
        # fallback: plain text
        st.text(df.head(200).to_string())

def try_plot(fig, height=None):
    """
    Plot with defensive try/except so visualization failures don't crash the app.
    """
    try:
        if height:
            fig.update_layout(height=height)
        st.plotly_chart(fig, width='stretch')
    except Exception:
        st.error("Plot rendering failed (non-fatal). See logs for details.")
        traceback.print_exc(file=sys.stdout)

def safe_download_df(df: pd.DataFrame, filename: str, label: str = "⬇️ Download CSV"):
    try:
        csv_bytes = _format_display_df(df.copy()).to_csv(index=False).encode("utf-8")
        st.download_button(label, csv_bytes, file_name=filename, mime="text/csv")
    except Exception:
        st.warning("Download unavailable for this table.")

# ---------- Load and precompute (cached) ----------
@st.cache_data(show_spinner=True)
def load_and_precompute(file_bytes: bytes) -> dict:
    """Load CSV bytes and compute ALL notebook summaries used across subsections."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), on_bad_lines="skip", low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

    df.columns = [c.strip() for c in df.columns]

    # parse dates
    for col in ["TRN_DATE", "ZED_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # numeric conversions (keep numeric types for calculations)
    numeric_cols = ["QTY", "CP_PRE_VAT", "SP_PRE_VAT", "COST_PRE_VAT", "NET_SALES", "VAT_AMT"]
    for nc in numeric_cols:
        if nc in df.columns:
            df[nc] = df[nc].astype(str).str.replace(",", "", regex=False)
            df[nc] = pd.to_numeric(df[nc], errors="coerce").fillna(0)

    # ensure identifiers are strings
    idcols = ["STORE_CODE", "TILL", "SESSION", "RCT"]
    for c in idcols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()

    # create CUST_CODE if missing
    if "CUST_CODE" not in df.columns:
        if all(c in df.columns for c in idcols):
            df["CUST_CODE"] = (df["STORE_CODE"].str.strip() + "-" + df["TILL"].str.strip() + "-" + df["SESSION"].str.strip() + "-" + df["RCT"].str.strip())
        else:
            df["CUST_CODE"] = df.index.astype(str)
    df["CUST_CODE"] = df["CUST_CODE"].astype(str).str.strip()

    out = {"df": df}

    # SALES: sales_channel_l1
    if "SALES_CHANNEL_L1" in df.columns and "NET_SALES" in df.columns:
        s1 = df.groupby("SALES_CHANNEL_L1", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        s1["NET_SALES_M"] = s1["NET_SALES"] / 1_000_000
        total_s1 = s1["NET_SALES"].sum()
        s1["PCT"] = s1["NET_SALES"] / total_s1 * 100 if total_s1 != 0 else 0
        out["sales_channel_l1"] = s1
    else:
        out["sales_channel_l1"] = pd.DataFrame()

    # SALES: sales_channel_l2
    if "SALES_CHANNEL_L2" in df.columns and "NET_SALES" in df.columns:
        s2 = df.groupby("SALES_CHANNEL_L2", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        s2["NET_SALES_M"] = s2["NET_SALES"] / 1_000_000
        total_s2 = s2["NET_SALES"].sum()
        s2["PCT"] = s2["NET_SALES"] / total_s2 * 100 if total_s2 != 0 else 0
        out["sales_channel_l2"] = s2
    else:
        out["sales_channel_l2"] = pd.DataFrame()

    # SHIFT derived
    if "TRN_DATE" in df.columns and "NET_SALES" in df.columns:
        tmp = df.copy()
        tmp["HOUR"] = tmp["TRN_DATE"].dt.hour.fillna(-1).astype(int)
        tmp["SHIFT_TYPE"] = np.where(tmp["HOUR"].between(7, 18), "Day", "Night")
        shift_tot = tmp.groupby("SHIFT_TYPE", as_index=False)["NET_SALES"].sum()
        total_shift = shift_tot["NET_SALES"].sum()
        shift_tot["PCT"] = shift_tot["NET_SALES"] / total_shift * 100 if total_shift != 0 else 0
        out["sales_by_shift"] = shift_tot

        per_store_shift = tmp.groupby(["STORE_NAME", "SHIFT_TYPE"], as_index=False)["NET_SALES"].sum().pivot(index="STORE_NAME", columns="SHIFT_TYPE", values="NET_SALES").fillna(0)
        per_store_shift["total"] = per_store_shift.sum(axis=1)
        per_store_shift = per_store_shift.reset_index().sort_values("total", ascending=False)
        if not per_store_shift.empty:
            for c in ["Day", "Night"]:
                if c in per_store_shift.columns:
                    per_store_shift[c + "_PCT"] = np.where(per_store_shift["total"] > 0, per_store_shift[c] / per_store_shift["total"] * 100, 0)
        out["per_store_shift"] = per_store_shift
    else:
        out["sales_by_shift"] = pd.DataFrame()
        out["per_store_shift"] = pd.DataFrame()

    # 2nd highest channel per store
    if {"STORE_NAME", "SALES_CHANNEL_L1", "NET_SALES"}.issubset(df.columns):
        store_chan = df.groupby(["STORE_NAME", "SALES_CHANNEL_L1"], as_index=False)["NET_SALES"].sum()
        store_chan["STORE_TOTAL"] = store_chan.groupby("STORE_NAME")["NET_SALES"].transform("sum")
        store_chan["PCT"] = np.where(store_chan["STORE_TOTAL"] > 0, store_chan["NET_SALES"] / store_chan["STORE_TOTAL"] * 100, 0)
        store_chan = store_chan.sort_values(["STORE_NAME", "PCT"], ascending=[True, False])
        store_chan["RANK"] = store_chan.groupby("STORE_NAME").cumcount() + 1
        second_tbl = store_chan[store_chan["RANK"] == 2][["STORE_NAME", "SALES_CHANNEL_L1", "PCT"]].rename(columns={"SALES_CHANNEL_L1":"SECOND_CHANNEL", "PCT":"SECOND_PCT"})
        all_stores = store_chan["STORE_NAME"].drop_duplicates()
        missing = set(all_stores) - set(second_tbl["STORE_NAME"])
        if missing:
            second_tbl = pd.concat([second_tbl, pd.DataFrame({"STORE_NAME": list(missing), "SECOND_CHANNEL": ["(None)"]*len(missing), "SECOND_PCT":[0.0]*len(missing)})], ignore_index=True)
        second_sorted = second_tbl.sort_values("SECOND_PCT", ascending=False)
        out["second_channel_table"] = second_sorted
        out["second_top_30"] = second_sorted.head(30)
        out["second_bottom_30"] = second_sorted.tail(30).sort_values("SECOND_PCT", ascending=True)
    else:
        out["second_channel_table"] = pd.DataFrame()
        out["second_top_30"] = pd.DataFrame()
        out["second_bottom_30"] = pd.DataFrame()

    # store sales summary
    if {"STORE_NAME", "NET_SALES", "QTY", "CUST_CODE"}.issubset(df.columns):
        store_sum = df.groupby("STORE_NAME", as_index=False).agg(NET_SALES=("NET_SALES","sum"), QTY=("QTY","sum"), RECEIPTS=("CUST_CODE", pd.Series.nunique)).sort_values("NET_SALES", ascending=False)
        out["store_sales_summary"] = store_sum
    else:
        out["store_sales_summary"] = pd.DataFrame()

    # receipts by time (dedupe by CUST_CODE earliest)
    if "TRN_DATE" in df.columns:
        rec = df.drop_duplicates(subset=["CUST_CODE"]).copy()
        rec["TRN_DATE"] = pd.to_datetime(rec["TRN_DATE"], errors="coerce")
        rec = rec.dropna(subset=["TRN_DATE"])
        rec["TIME_SLOT"] = rec["TRN_DATE"].dt.floor("30T").dt.time
        heat = rec.groupby(["STORE_NAME", "TIME_SLOT"], as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"RECEIPT_COUNT"})
        out["receipts_by_time"] = heat
    else:
        out["receipts_by_time"] = pd.DataFrame()

    # active tills average per store per day
    if {"TRN_DATE", "TILL", "STORE_NAME"}.issubset(df.columns):
        ttmp = df.copy()
        ttmp["TRN_DATE"] = pd.to_datetime(ttmp["TRN_DATE"], errors="coerce")
        ttmp = ttmp.dropna(subset=["TRN_DATE"])
        ttmp["TRN_DATE_DATE"] = ttmp["TRN_DATE"].dt.date
        active_tills = ttmp.groupby(["STORE_NAME", "TRN_DATE_DATE"], as_index=False)["TILL"].nunique()
        active_avg = active_tills.groupby("STORE_NAME", as_index=False)["TILL"].mean().rename(columns={"TILL":"avg_active_tills"})
        out["active_tills_avg"] = active_avg.sort_values("avg_active_tills", ascending=False)
    else:
        out["active_tills_avg"] = pd.DataFrame()

    # first_touch receipts helpful for customers-per-till
    if {"TRN_DATE", "STORE_CODE", "TILL", "SESSION", "RCT", "STORE_NAME"}.issubset(df.columns):
        tmp = df.copy()
        tmp["TRN_DATE"] = pd.to_datetime(tmp["TRN_DATE"], errors="coerce")
        tmp = tmp.dropna(subset=["TRN_DATE"])
        for c in ["STORE_CODE", "TILL", "SESSION", "RCT"]:
            tmp[c] = tmp[c].astype(str).fillna("").str.strip()
        tmp["CUST_CODE"] = tmp["STORE_CODE"] + "-" + tmp["TILL"] + "-" + tmp["SESSION"] + "-" + tmp["RCT"]
        tmp["TRN_DATE_ONLY"] = tmp["TRN_DATE"].dt.date
        first_touch = tmp.groupby(["STORE_NAME", "TRN_DATE_ONLY", "CUST_CODE"], as_index=False)["TRN_DATE"].min()
        first_touch["TIME_SLOT"] = first_touch["TRN_DATE"].dt.floor("30T").dt.time
        out["first_touch"] = first_touch
    else:
        out["first_touch"] = pd.DataFrame()

    # cashier perf
    if "CASHIER" in df.columns or "CASHIER_NAME" in df.columns:
        cash_col = "CASHIER" if "CASHIER" in df.columns else "CASHIER_NAME"
        cashier_perf = df.groupby(cash_col, as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        out["cashier_perf"] = cashier_perf
    else:
        out["cashier_perf"] = pd.DataFrame()

    # dept and supplier
    if "DEPARTMENT" in df.columns:
        out["dept_sales"] = df.groupby("DEPARTMENT", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
    else:
        out["dept_sales"] = pd.DataFrame()

    if "SUPPLIER" in df.columns:
        out["supplier_sales"] = df.groupby("SUPPLIER", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
    else:
        out["supplier_sales"] = pd.DataFrame()

    # top items
    if "ITEM_NAME" in df.columns:
        out["top_items_sales"] = df.groupby("ITEM_NAME", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        out["top_items_qty"] = df.groupby("ITEM_NAME", as_index=False)["QTY"].sum().sort_values("QTY", ascending=False)
    else:
        out["top_items_sales"] = pd.DataFrame()
        out["top_items_qty"] = pd.DataFrame()

    # loyalty receipts
    if {"LOYALTY_CUSTOMER_CODE", "CUST_CODE", "STORE_NAME", "TRN_DATE", "NET_SALES"}.issubset(df.columns):
        loy = df.copy()
        loy["TRN_DATE"] = pd.to_datetime(loy["TRN_DATE"], errors="coerce")
        loy = loy.dropna(subset=["TRN_DATE"])
        loy["LOYALTY_CUSTOMER_CODE"] = loy["LOYALTY_CUSTOMER_CODE"].astype(str).str.strip()
        loy = loy[loy["LOYALTY_CUSTOMER_CODE"].replace({"nan":"", "NaN":"", "None":""}).str.len() > 0]
        receipts = loy.groupby(["STORE_NAME", "CUST_CODE", "LOYALTY_CUSTOMER_CODE"], as_index=False).agg(Basket_Value=("NET_SALES", "sum"), First_Time=("TRN_DATE", "min"))
        out["loyalty_receipts"] = receipts
    else:
        out["loyalty_receipts"] = pd.DataFrame()

    # pricing multi-price summary
    if {"TRN_DATE", "STORE_NAME", "ITEM_CODE", "ITEM_NAME", "SP_PRE_VAT", "QTY"}.issubset(df.columns):
        dpp = df.copy()
        dpp["TRN_DATE"] = pd.to_datetime(dpp["TRN_DATE"], errors="coerce")
        dpp = dpp.dropna(subset=["TRN_DATE"])
        dpp["DATE"] = dpp["TRN_DATE"].dt.date
        dpp["SP_PRE_VAT"] = pd.to_numeric(dpp["SP_PRE_VAT"].astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0.0)
        dpp["QTY"] = pd.to_numeric(dpp["QTY"].astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0.0)
        grp = dpp.groupby(["STORE_NAME", "DATE", "ITEM_CODE", "ITEM_NAME"], as_index=False).agg(
            Num_Prices=("SP_PRE_VAT", lambda s: s.dropna().nunique()),
            Price_Min=("SP_PRE_VAT", "min"),
            Price_Max=("SP_PRE_VAT", "max"),
            Total_QTY=("QTY", "sum")
        )
        grp["Price_Spread"] = grp["Price_Max"] - grp["Price_Min"]
        multi_price = grp[(grp["Num_Prices"] > 1) & (grp["Price_Spread"] > 0)].copy()
        if not multi_price.empty:
            multi_price["Diff_Value"] = (multi_price["Total_QTY"] * multi_price["Price_Spread"]).round(2)
            summary_pr = multi_price.groupby("STORE_NAME", as_index=False).agg(
                Items_with_MultiPrice=("ITEM_CODE", "nunique"),
                Total_Diff_Value=("Diff_Value", "sum"),
                Avg_Spread=("Price_Spread", "mean"),
                Max_Spread=("Price_Spread", "max")
            ).sort_values("Total_Diff_Value", ascending=False)
            out["global_pricing_summary"] = summary_pr
            out["multi_price_detail"] = multi_price
        else:
            out["global_pricing_summary"] = pd.DataFrame()
            out["multi_price_detail"] = pd.DataFrame()
    else:
        out["global_pricing_summary"] = pd.DataFrame()
        out["multi_price_detail"] = pd.DataFrame()

    # refunds
    if {"NET_SALES", "STORE_NAME", "CUST_CODE"}.issubset(df.columns):
        neg = df[df["NET_SALES"] < 0].copy()
        if not neg.empty:
            neg["Abs_Neg"] = neg["NET_SALES"].abs()
            out["global_refunds"] = neg.groupby("STORE_NAME", as_index=False).agg(Total_Neg_Value=("NET_SALES", "sum"), Receipts=("CUST_CODE", pd.Series.nunique)).sort_values("Total_Neg_Value")
            out["branch_refunds_detail"] = neg.groupby(["STORE_NAME", "CUST_CODE"], as_index=False).agg(Value=("NET_SALES", "sum"), First_Time=("TRN_DATE", "min"))
        else:
            out["global_refunds"] = pd.DataFrame()
            out["branch_refunds_detail"] = pd.DataFrame()
    else:
        out["global_refunds"] = pd.DataFrame()
        out["branch_refunds_detail"] = pd.DataFrame()

    # lists for dropdowns
    out["stores"] = sorted(df["STORE_NAME"].dropna().unique().tolist()) if "STORE_NAME" in df.columns else []
    out["channels_l1"] = sorted(df["SALES_CHANNEL_L1"].dropna().unique().tolist()) if "SALES_CHANNEL_L1" in df.columns else []
    out["channels_l2"] = sorted(df["SALES_CHANNEL_L2"].dropna().unique().tolist()) if "SALES_CHANNEL_L2" in df.columns else []
    out["items"] = sorted(df["ITEM_NAME"].dropna().unique().tolist()) if "ITEM_NAME" in df.columns else []
    out["departments"] = sorted(df["DEPARTMENT"].dropna().unique().tolist()) if "DEPARTMENT" in df.columns else []

    start_time = pd.Timestamp("00:00:00")
    intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
    out["time_intervals"] = intervals
    out["time_labels"] = [t.strftime("%H:%M") for t in intervals]

    out["sample_rows"] = df.head(200)
    return out

# ---------- App UI ----------
st.title("🦸 Superdeck Analytics Dashboard")
st.markdown("Upload your sales CSV, the app will precompute analytics and show all subsections from SALES → INSIGHTS. Tables include totals and use thousands separators.")

uploaded = st.file_uploader("Upload sales CSV (CSV)", type="csv")
if uploaded is None:
    st.info("Please upload a CSV to proceed.")
    st.stop()

file_bytes = uploaded.getvalue()
try:
    state = load_and_precompute(file_bytes)
except Exception:
    st.error("Failed to load dataset. See server logs for details.")
    traceback.print_exc(file=sys.stdout)
    st.stop()

# Sample download
st.sidebar.download_button("⬇️ Download sample rows", state["sample_rows"].to_csv(index=False).encode("utf-8"), "sample_rows.csv", "text/csv")
st.sidebar.markdown("---")
st.sidebar.markdown("Theme: Red & Green — Positive (green), Negative/alerts (red)")

# Sections + subsections
main_sections = {
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

section = st.sidebar.radio("Main Section", list(main_sections.keys()))
subsection = st.sidebar.selectbox("Subsection", main_sections[section])
st.markdown(f"##### {section} ➔ {subsection}")

df = state["df"]  # raw df available for on-demand uses

# ---------- SALES ----------
if section == "SALES":
    # 1 Global sales Overview
    if subsection == "Global sales Overview":
        gs = state["sales_channel_l1"]
        if gs.empty:
            st.warning("Missing SALES_CHANNEL_L1 or NET_SALES columns.")
        else:
            gs_disp = gs.copy()
            # ensure numeric and compute measures
            gs_disp["NET_SALES"] = pd.to_numeric(gs_disp["NET_SALES"], errors="coerce").fillna(0)
            gs_disp["NET_SALES_M"] = (gs_disp["NET_SALES"] / 1_000_000).round(2)
            total = gs_disp["NET_SALES"].sum()
            gs_disp["PCT"] = (gs_disp["NET_SALES"] / total * 100).round(1) if total != 0 else 0.0
            legend_labels = [f"{r['SALES_CHANNEL_L1']} ({r['PCT']:.1f}% | {r['NET_SALES_M']:.1f} M)" for _, r in gs_disp.iterrows()]

            fig = go.Figure(go.Pie(
                labels=legend_labels,
                values=gs_disp["NET_SALES_M"],
                hole=0.65,
                text=[f"{p:.1f}%" for p in gs_disp["PCT"]],
                textinfo='text',
                textposition='inside',
                marker=dict(colors=PALETTE10, line=dict(color='white', width=1)),
                hovertemplate='<b>%{label}</b><br>KSh %{value:,.2f} M<extra></extra>'
            ))
            fig.update_layout(title="<b>SALES CHANNEL TYPE — Global Overview</b>", title_x=0.42, height=600)
            try_plot(fig)

            # table with totals and formatted numbers
            df_out = gs_disp[["SALES_CHANNEL_L1", "NET_SALES", "NET_SALES_M", "PCT"]].copy()
            df_out = add_total_row(df_out, numeric_cols=["NET_SALES"], label_col="SALES_CHANNEL_L1")
            display_table_with_format(df_out, int_cols=["NET_SALES"], float_cols=["NET_SALES_M"], pct_cols=["PCT"], height=420)
            safe_download_df(df_out, "global_sales_overview.csv", "⬇️ Download Table")

    # 2 Global Net Sales Distribution by Sales Channel (L2)
    elif subsection == "Global Net Sales Distribution by Sales Channel":
        g2 = state["sales_channel_l2"]
        if g2.empty:
            st.warning("Missing SALES_CHANNEL_L2 or NET_SALES columns.")
        else:
            g2_disp = g2.copy()
            g2_disp["NET_SALES"] = pd.to_numeric(g2_disp["NET_SALES"], errors="coerce").fillna(0)
            g2_disp["NET_SALES_M"] = (g2_disp["NET_SALES"] / 1_000_000).round(2)
            total = g2_disp["NET_SALES"].sum()
            g2_disp["PCT"] = (g2_disp["NET_SALES"] / total * 100).round(1) if total != 0 else 0.0
            legend_labels = [f"{r['SALES_CHANNEL_L2']} ({r['PCT']:.1f}% | {r['NET_SALES_M']:.1f} M)" for _, r in g2_disp.iterrows()]

            fig = go.Figure(go.Pie(labels=legend_labels, values=g2_disp["NET_SALES_M"], hole=0.65,
                                   text=[f"{p:.1f}%" for p in g2_disp["PCT"]],
                                   marker=dict(colors=PALETTE10, line=dict(color='white', width=1)),
                                   hovertemplate='<b>%{label}</b><br>KSh %{value:,.2f} M<extra></extra>'))
            fig.update_layout(title="<b>Global Net Sales Distribution by Sales Mode (L2)</b>", height=620)
            try_plot(fig)

            df_out = g2_disp[["SALES_CHANNEL_L2", "NET_SALES", "NET_SALES_M", "PCT"]].copy()
            df_out = add_total_row(df_out, numeric_cols=["NET_SALES"], label_col="SALES_CHANNEL_L2")
            display_table_with_format(df_out, int_cols=["NET_SALES"], float_cols=["NET_SALES_M"], pct_cols=["PCT"], height=420)
            safe_download_df(df_out, "sales_channel_l2.csv", "⬇️ Download Table")

    # 3 Global Net Sales Distribution by SHIFT
    elif subsection == "Global Net Sales Distribution by SHIFT":
        sb = state["sales_by_shift"]
        if sb.empty:
            st.warning("No SHIFT data.")
        else:
            sb_disp = sb.copy()
            sb_disp["NET_SALES"] = pd.to_numeric(sb_disp["NET_SALES"], errors="coerce").fillna(0)
            sb_disp["PCT"] = sb_disp["PCT"].round(1)
            fig = px.bar(sb_disp, x="SHIFT_TYPE", y="NET_SALES", color="SHIFT_TYPE", color_discrete_map={"Day": COLOR_GREEN, "Night": COLOR_RED}, title="Net Sales by SHIFT")
            try_plot(fig)
            df_out = add_total_row(sb_disp.copy(), numeric_cols=["NET_SALES"], label_col="SHIFT_TYPE")
            display_table_with_format(df_out, int_cols=["NET_SALES"], pct_cols=["PCT"], height=420)
            safe_download_df(df_out, "sales_by_shift.csv", "⬇️ Download Table")

    # 4 Night vs Day Shift Sales Ratio — Stores with Night Shifts
    elif subsection == "Night vs Day Shift Sales Ratio — Stores with Night Shifts":
        per_store = state["per_store_shift"]
        if per_store.empty:
            st.info("No per-store shift data available.")
        else:
            # show Day/Night % columns if present
            cols_pct = [c for c in per_store.columns if c.endswith("_PCT")]
            df_disp = per_store.copy()
            if "Night_PCT" in df_disp.columns:
                fig = px.bar(df_disp.sort_values("Night_PCT", ascending=True), x="Night_PCT", y="STORE_NAME", orientation="h", color_discrete_sequence=[COLOR_RED], title="Night % by Store (stores with night activity)")
                try_plot(fig)
            # show table with totals (Day and Night sums)
            tbl = add_total_row(df_disp[[c for c in df_disp.columns if c in ["STORE_NAME", "Day", "Night"]]].copy(), numeric_cols=["Day", "Night"], label_col="STORE_NAME")
            display_table_with_format(tbl, int_cols=["Day", "Night"], height=480)
            safe_download_df(tbl, "night_vs_day_store_pct.csv", "⬇️ Download Table")

    # 5 Global Day vs Night Sales — Only Stores with NIGHT Shift
    elif subsection == "Global Day vs Night Sales — Only Stores with NIGHT Shift":
        # Recompute from df for stores with night
        if "TRN_DATE" in df.columns and "NET_SALES" in df.columns:
            tmp = df.copy()
            tmp["SHIFT_BUCKET"] = np.where(tmp.get("SHIFT", "").astype(str).str.upper().str.contains("NIGHT", na=False), "Night", np.where(tmp.get("SHIFT", "").astype(str).str.strip()== "", "Day", "Day"))
            stores_with_night = df[df.get("SHIFT", "").astype(str).str.upper().str.contains("NIGHT", na=False)]["STORE_NAME"].unique()
            tmp = tmp[tmp["STORE_NAME"].isin(stores_with_night)]
            global_nd = tmp.groupby("SHIFT_BUCKET", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
            total = global_nd["NET_SALES"].sum()
            global_nd["PCT"] = (global_nd["NET_SALES"] / total * 100).round(1) if total != 0 else 0.0
            labels = [f"{r['SHIFT_BUCKET']} ({r['PCT']:.1f}%)" for _, r in global_nd.iterrows()]
            fig = go.Figure(go.Pie(labels=labels, values=global_nd["NET_SALES"], hole=0.65, text=[f"{p:.1f}%" for p in global_nd["PCT"]], marker=dict(colors=[COLOR_BLUE, COLOR_RED])))
            try_plot(fig)
            df_out = add_total_row(global_nd.copy(), numeric_cols=["NET_SALES"], label_col="SHIFT_BUCKET")
            display_table_with_format(df_out, int_cols=["NET_SALES"], pct_cols=["PCT"], height=420)
            safe_download_df(df_out, "global_day_night_stores_with_night.csv", "⬇️ Download Table")
        else:
            st.info("TRN_DATE or NET_SALES missing to compute day/night global stats.")

    # 6 2nd-Highest Channel Share
    elif subsection == "2nd-Highest Channel Share":
        sc = state["second_channel_table"]
        if sc.empty:
            st.info("No second-channel information.")
        else:
            df_out = add_total_row(sc.copy(), numeric_cols=["SECOND_PCT"], label_col="STORE_NAME")
            display_table_with_format(df_out[["STORE_NAME", "SECOND_CHANNEL", "SECOND_PCT"]], pct_cols=["SECOND_PCT"], height=520)
            # lollipop-ish horizontal bar for top stores
            top = sc.sort_values("SECOND_PCT", ascending=False).head(50)
            fig = px.bar(top.sort_values("SECOND_PCT", ascending=True), x="SECOND_PCT", y="STORE_NAME", orientation="h", color_discrete_sequence=[COLOR_BLUE], title="2nd-Highest Channel Share — Top stores")
            try_plot(fig)

    # 7 Bottom 30 — 2nd Highest Channel
    elif subsection == "Bottom 30 — 2nd Highest Channel":
        bottom = state["second_bottom_30"]
        if bottom.empty:
            st.info("No bottom 30 data.")
        else:
            df_out = add_total_row(bottom.copy(), numeric_cols=["SECOND_PCT"], label_col="STORE_NAME")
            display_table_with_format(df_out[["STORE_NAME", "SECOND_CHANNEL", "SECOND_PCT"]], pct_cols=["SECOND_PCT"], height=520)
            fig = px.bar(bottom.sort_values("SECOND_PCT", ascending=True), x="SECOND_PCT", y="STORE_NAME", orientation="h", color_discrete_sequence=[COLOR_RED], title="Bottom 30 — 2nd Highest Channel")
            try_plot(fig)

    # 8 Stores Sales Summary
    elif subsection == "Stores Sales Summary":
        ss = state["store_sales_summary"]
        if ss.empty:
            st.info("No store sales summary available.")
        else:
            df_out = add_total_row(ss.copy(), numeric_cols=["NET_SALES", "QTY"], label_col="STORE_NAME")
            display_table_with_format(df_out, int_cols=["NET_SALES", "QTY", "RECEIPTS"], height=520)
            safe_download_df(df_out, "stores_sales_summary.csv", "⬇️ Download Table")

# ---------- OPERATIONS ----------
elif section == "OPERATIONS":
    # 1 Customer Traffic-Storewise
    if subsection == "Customer Traffic-Storewise":
        heat = state["receipts_by_time"]
        stores = state["stores"]
        if heat.empty or not stores:
            st.info("No receipts-by-time data available.")
        else:
            sel_store = st.selectbox("Select Store", stores)
            df_heat = heat[heat["STORE_NAME"] == sel_store].copy()
            if df_heat.empty:
                st.info("No time data for selected store.")
            else:
                times = state["time_intervals"]
                time_idx = pd.Index(times, name="TIME_SLOT")
                pivot = df_heat.set_index("TIME_SLOT")["RECEIPT_COUNT"].reindex(time_idx, fill_value=0)
                labels = state["time_labels"]
                fig = px.bar(x=labels, y=pivot.values, labels={"x":"Time","y":"Receipts"}, color_discrete_sequence=[COLOR_BLUE], title=f"Receipts by Time - {sel_store}")
                try_plot(fig)
                out_df = pivot.reset_index().rename(columns={"TIME_SLOT":"TIME", 0:"RECEIPT_COUNT"})
                out_df.columns = ["TIME", "RECEIPT_COUNT"]
                out_df = add_total_row(out_df, numeric_cols=["RECEIPT_COUNT"], label_col="TIME")
                display_table_with_format(out_df, int_cols=["RECEIPT_COUNT"], height=420)
                safe_download_df(out_df, f"receipts_by_time_{sel_store}.csv", "⬇️ Download Table")

    # 2 Active Tills During the day
    elif subsection == "Active Tills During the day":
        at = state["active_tills_avg"]
        if at.empty:
            st.info("No active tills data.")
        else:
            df_out = add_total_row(at.copy(), numeric_cols=["avg_active_tills"], label_col="STORE_NAME")
            display_table_with_format(df_out, float_cols=["avg_active_tills"], height=420)
            safe_download_df(df_out, "active_tills_avg.csv", "⬇️ Download Table")

    # 3 Average Customers Served per Till
    elif subsection == "Average Customers Served per Till":
        first_touch = state["first_touch"]
        if first_touch.empty:
            st.info("Insufficient first-touch data.")
        else:
            receipts = first_touch.copy()
            receipts["TIME_SLOT"] = receipts["TRN_DATE"].dt.floor("30T").dt.time
            cust_counts = receipts.groupby(["STORE_NAME", "TIME_SLOT"], as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"CUSTOMERS"})
            tmp = df.copy()
            if {"TRN_DATE", "TILL", "STORE_NAME"}.issubset(tmp.columns):
                tmp["TRN_DATE"] = pd.to_datetime(tmp["TRN_DATE"], errors="coerce")
                tmp = tmp.dropna(subset=["TRN_DATE"])
                tmp["TIME_SLOT"] = tmp["TRN_DATE"].dt.floor("30T").dt.time
                till_counts = tmp.groupby(["STORE_NAME", "TIME_SLOT"], as_index=False)["TILL"].nunique().rename(columns={"TILL":"TILLS"})
                merged = cust_counts.merge(till_counts, on=["STORE_NAME", "TIME_SLOT"], how="left").fillna({"TILLS":0})
                merged["CUST_PER_TILL"] = merged["CUSTOMERS"] / merged["TILLS"].replace(0, np.nan)
                per_store = merged.groupby("STORE_NAME", as_index=False).agg(Max_Cust_Per_Till=("CUST_PER_TILL", "max"), Avg_Cust_Per_Till=("CUST_PER_TILL", "mean")).fillna(0)
                df_out = add_total_row(per_store.copy(), numeric_cols=["Max_Cust_Per_Till","Avg_Cust_Per_Till"], label_col="STORE_NAME")
                display_table_with_format(df_out, float_cols=["Max_Cust_Per_Till","Avg_Cust_Per_Till"], height=420)
                safe_download_df(df_out, "cust_per_till.csv", "⬇️ Download Table")
            else:
                st.info("Missing TILL or TRN_DATE columns to compute tills-per-slot.")

    # 4 Store Customer Traffic Storewise
    elif subsection == "Store Customer Traffic Storewise":
        stores = state["stores"]
        if not stores:
            st.info("No stores available.")
        else:
            sel_store = st.selectbox("Select Store", stores)
            branch_df = df[df["STORE_NAME"] == sel_store].copy()
            if branch_df.empty:
                st.info("No data for store.")
            else:
                branch_df["TRN_DATE"] = pd.to_datetime(branch_df["TRN_DATE"], errors="coerce")
                branch_df = branch_df.dropna(subset=["TRN_DATE"])
                for c in ["STORE_CODE","TILL","SESSION","RCT"]:
                    if c in branch_df.columns:
                        branch_df[c] = branch_df[c].astype(str).fillna("").str.strip()
                if "CUST_CODE" not in branch_df.columns:
                    branch_df["CUST_CODE"] = branch_df["STORE_CODE"] + "-" + branch_df["TILL"] + "-" + branch_df["SESSION"] + "-" + branch_df["RCT"]
                branch_df["TIME_SLOT"] = branch_df["TRN_DATE"].dt.floor("30T").dt.time
                tmp = branch_df.groupby(["DEPARTMENT", "TIME_SLOT"], as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"UNIQUE_CUSTOMERS"})
                display_table_with_format(tmp, int_cols=["UNIQUE_CUSTOMERS"], height=520)
                safe_download_df(tmp, f"dept_traffic_{sel_store}.csv", "⬇️ Download Table")

    # 5 Customer Traffic-Departmentwise
    elif subsection == "Customer Traffic-Departmentwise":
        depts = state["departments"]
        if not depts:
            st.info("No departments available.")
        else:
            sel = st.selectbox("Select Department", depts)
            dept_df = df[df["DEPARTMENT"] == sel].copy()
            if dept_df.empty:
                st.info("No rows for selected department.")
            else:
                dept_df["TRN_DATE"] = pd.to_datetime(dept_df["TRN_DATE"], errors="coerce")
                dept_df = dept_df.dropna(subset=["TRN_DATE"])
                dept_df["TIME_SLOT"] = dept_df["TRN_DATE"].dt.floor("30T").dt.time
                counts = dept_df.groupby(["STORE_NAME", "TIME_SLOT"], as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"UNIQUE_CUSTOMERS"})
                display_table_with_format(counts, int_cols=["UNIQUE_CUSTOMERS"], height=520)
                safe_download_df(counts, f"dept_{sel}_traffic.csv", "⬇️ Download Table")

    # 6 Cashiers Perfomance
    elif subsection == "Cashiers Perfomance":
        cp = state["cashier_perf"]
        if cp.empty:
            st.info("No cashier information available.")
        else:
            df_out = add_total_row(cp.copy(), numeric_cols=["NET_SALES"], label_col=cp.columns[0])
            display_table_with_format(df_out, int_cols=["NET_SALES"], height=420)
            safe_download_df(df_out, "cashier_perf.csv", "⬇️ Download Table")

    # 7 Till Usage
    elif subsection == "Till Usage":
        tmp = df.copy()
        tmp["TRN_DATE"] = pd.to_datetime(tmp["TRN_DATE"], errors="coerce")
        tmp = tmp.dropna(subset=["TRN_DATE"])
        tmp["TIME_SLOT"] = tmp["TRN_DATE"].dt.floor("30T").dt.time
        tmp["Till_Code"] = tmp["TILL"].astype(str).fillna("") + "-" + tmp["STORE_CODE"].astype(str).fillna("")
        till_activity = tmp.groupby(["STORE_NAME", "Till_Code", "TIME_SLOT"], as_index=False).agg(Receipts=("CUST_CODE","nunique"))
        branch_summary = till_activity.groupby("STORE_NAME", as_index=False).agg(Store_Total_Receipts=("Receipts","sum"), Avg_Per_Till=("Receipts","mean"), Max_Per_Till=("Receipts","max"), Unique_Tills=("Till_Code","nunique"))
        df_out = add_total_row(branch_summary.copy(), numeric_cols=["Store_Total_Receipts"], label_col="STORE_NAME")
        display_table_with_format(df_out, int_cols=["Store_Total_Receipts","Unique_Tills"], float_cols=["Avg_Per_Till"], height=520)
        safe_download_df(df_out, "till_usage_summary.csv", "⬇️ Download Table")

    # 8 Tax Compliance
    elif subsection == "Tax Compliance":
        if {"CU_DEVICE_SERIAL", "CUST_CODE", "STORE_NAME"}.issubset(df.columns):
            tdf = df.copy()
            tdf["Tax_Compliant"] = np.where(tdf["CU_DEVICE_SERIAL"].astype(str).str.strip().replace({"nan":"", "NaN":"", "None":""}).str.len() > 0, "Compliant", "Non-Compliant")
            global_summary = tdf.groupby("Tax_Compliant", as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"Receipts"})
            fig = px.pie(global_summary, names="Tax_Compliant", values="Receipts", color="Tax_Compliant", color_discrete_map={"Compliant": COLOR_GREEN, "Non-Compliant": COLOR_RED}, hole=0.45, title="Global Tax Compliance Overview")
            try_plot(fig)
            store_till = tdf.groupby(["STORE_NAME", "Till_Code", "Tax_Compliant"], as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"Receipts"})
            branch = st.selectbox("Select Branch", state["stores"])
            dfb = store_till[store_till["STORE_NAME"] == branch]
            if dfb.empty:
                st.info("No compliance data for this branch.")
            else:
                pivot = dfb.pivot(index="Till_Code", columns="Tax_Compliant", values="Receipts").fillna(0).reset_index()
                pivot = add_total_row(pivot.copy(), numeric_cols=[c for c in pivot.columns if c != "Till_Code"], label_col="Till_Code")
                display_table_with_format(pivot, int_cols=[c for c in pivot.columns if c != "Till_Code"], height=520)
                safe_download_df(pivot, f"tax_compliance_{branch}.csv", "⬇️ Download Table")
        else:
            st.info("Missing columns for Tax Compliance (CU_DEVICE_SERIAL, CUST_CODE, STORE_NAME).")

# ---------- INSIGHTS ----------
elif section == "INSIGHTS":
    # 1 Customer Baskets Overview
    if subsection == "Customer Baskets Overview":
        receipts = df.drop_duplicates(subset=["CUST_CODE"])
        if receipts.empty or "NET_SALES" not in receipts.columns:
            st.info("Insufficient basket-level data.")
        else:
            total = receipts["NET_SALES"].sum()
            avg = receipts["NET_SALES"].mean()
            med = receipts["NET_SALES"].median()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total basket net sales", f"KSh {total:,.2f}")
            c2.metric("Average basket value", f"KSh {avg:,.2f}")
            c3.metric("Median basket value", f"KSh {med:,.2f}")
            fig = px.histogram(receipts, x="NET_SALES", nbins=50, title="Basket Value Distribution", color_discrete_sequence=[COLOR_BLUE])
            try_plot(fig)

    # 2 Global Category Overview-Sales
    elif subsection == "Global Category Overview-Sales":
        dept = state["dept_sales"]
        if dept.empty:
            st.info("No department/category sales data.")
        else:
            df_out = add_total_row(dept.copy(), numeric_cols=["NET_SALES"], label_col="DEPARTMENT")
            display_table_with_format(df_out, int_cols=["NET_SALES"], height=420)
            fig = px.bar(dept.head(50), x="DEPARTMENT", y="NET_SALES", title="Category Sales (Global)", color_discrete_sequence=[COLOR_GREEN])
            try_plot(fig)

    # 3 Global Category Overview-Baskets
    elif subsection == "Global Category Overview-Baskets":
        if "DEPARTMENT" in df.columns:
            rec = df.drop_duplicates(subset=["CUST_CODE"])
            dept_counts = rec.groupby("DEPARTMENT", as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"BASKETS"})
            df_out = add_total_row(dept_counts.copy(), numeric_cols=["BASKETS"], label_col="DEPARTMENT")
            display_table_with_format(df_out, int_cols=["BASKETS"], height=420)
            fig = px.bar(dept_counts.sort_values("BASKETS", ascending=False), x="DEPARTMENT", y="BASKETS", title="Baskets by Department (Global)", color_discrete_sequence=[COLOR_BLUE])
            try_plot(fig)
        else:
            st.info("DEPARTMENT column missing.")

    # 4 Supplier Contribution
    elif subsection == "Supplier Contribution":
        sup = state["supplier_sales"]
        if sup.empty:
            st.info("No supplier data.")
        else:
            df_out = add_total_row(sup.copy(), numeric_cols=["NET_SALES"], label_col="SUPPLIER")
            display_table_with_format(df_out, int_cols=["NET_SALES"], height=420)
            safe_download_df(df_out, "supplier_contribution.csv", "⬇️ Download Table")

    # 5 Category Overview
    elif subsection == "Category Overview":
        if "CATEGORY" in df.columns and "NET_SALES" in df.columns:
            cat = df.groupby("CATEGORY", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
            df_out = add_total_row(cat.copy(), numeric_cols=["NET_SALES"], label_col="CATEGORY")
            display_table_with_format(df_out, int_cols=["NET_SALES"], height=420)
            fig = px.bar(cat.head(50), x="CATEGORY", y="NET_SALES", title="Category Sales", color_discrete_sequence=[COLOR_GREEN])
            try_plot(fig)
        else:
            st.info("CATEGORY or NET_SALES missing.")

    # 6 Branch Comparison
    elif subsection == "Branch Comparison":
        stores = state["stores"]
        if len(stores) < 2:
            st.info("Not enough branches to compare.")
        else:
            A = st.selectbox("Branch A", stores, key="bc_a")
            B = st.selectbox("Branch B", stores, key="bc_b")
            metric = st.selectbox("Metric", ["QTY", "NET_SALES"], key="bc_metric")
            N = st.slider("Top N", 5, 50, 10, key="bc_n")
            if metric not in df.columns:
                st.error(f"Metric {metric} not found in data.")
            else:
                dfA = df[df["STORE_NAME"] == A].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N)
                dfB = df[df["STORE_NAME"] == B].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N)
                combA = dfA.copy(); combA["Branch"] = A
                combB = dfB.copy(); combB["Branch"] = B
                both = pd.concat([combA, combB], ignore_index=True)
                fig = px.bar(both, x=metric, y="ITEM_NAME", color="Branch", orientation="h", barmode="group", color_discrete_sequence=[COLOR_BLUE, COLOR_GREEN], title=f"Top {N} items: {A} vs {B}")
                try_plot(fig)
                display_table_with_format(both, int_cols=[metric] if metric == "QTY" else ["NET_SALES"], height=450)

    # 7 Product Perfomance
    elif subsection == "Product Perfomance":
        items = state["items"]
        if not items:
            st.info("No ITEM_NAME present in data.")
        else:
            sel_item = st.selectbox("Select Item", items)
            item_df = df[df["ITEM_NAME"] == sel_item]
            if item_df.empty:
                st.info("No rows for selected item.")
            else:
                qty_by_store = item_df.groupby("STORE_NAME", as_index=False)["QTY"].sum().sort_values("QTY", ascending=False)
                sales_by_store = item_df.groupby("STORE_NAME", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
                st.subheader("Top stores by quantity")
                display_table_with_format(add_total_row(qty_by_store.copy(), numeric_cols=["QTY"], label_col="STORE_NAME"), int_cols=["QTY"], height=300)
                st.subheader("Top stores by sales")
                display_table_with_format(add_total_row(sales_by_store.copy(), numeric_cols=["NET_SALES"], label_col="STORE_NAME"), int_cols=["NET_SALES"], height=300)

    # 8 Global Loyalty Overview
    elif subsection == "Global Loyalty Overview":
        lr = state["loyalty_receipts"]
        if lr.empty:
            st.info("No loyalty data.")
        else:
            branch_loyal = lr.groupby("STORE_NAME", as_index=False)["LOYALTY_CUSTOMER_CODE"].nunique().rename(columns={"LOYALTY_CUSTOMER_CODE":"Loyal_Customers"})
            df_out = add_total_row(branch_loyal.copy(), numeric_cols=["Loyal_Customers"], label_col="STORE_NAME")
            display_table_with_format(df_out, int_cols=["Loyal_Customers"], height=420)
            safe_download_df(df_out, "global_loyalty_overview.csv", "⬇️ Download Table")

    # 9 Branch Loyalty Overview
    elif subsection == "Branch Loyalty Overview":
        lr = state["loyalty_receipts"]
        if lr.empty:
            st.info("No loyalty receipts.")
        else:
            branch = st.selectbox("Select Branch", sorted(lr["STORE_NAME"].unique().tolist()))
            per_store = lr[lr["STORE_NAME"] == branch].groupby("LOYALTY_CUSTOMER_CODE", as_index=False).agg(Baskets=("CUST_CODE","nunique"), Total_Value=("Basket_Value","sum")).sort_values(["Baskets", "Total_Value"], ascending=[False, False])
            df_out = add_total_row(per_store.copy(), numeric_cols=["Baskets", "Total_Value"], label_col="LOYALTY_CUSTOMER_CODE")
            display_table_with_format(df_out, int_cols=["Baskets"], float_cols=["Total_Value"], height=520)
            safe_download_df(df_out, f"branch_loyalty_{branch}.csv", "⬇️ Download Table")

    # 10 Customer Loyalty Overview
    elif subsection == "Customer Loyalty Overview":
        lr = state["loyalty_receipts"]
        if lr.empty:
            st.info("No loyalty receipts.")
        else:
            custs = sorted(lr["LOYALTY_CUSTOMER_CODE"].unique().tolist())
            sel = st.selectbox("Select Loyalty Customer", custs)
            rc = lr[lr["LOYALTY_CUSTOMER_CODE"] == sel]
            if rc.empty:
                st.info("No receipts for selected loyalty customer.")
            else:
                display_table_with_format(rc.sort_values("First_Time", ascending=False), float_cols=["Basket_Value"], height=520)

    # 11 Global Pricing Overview
    elif subsection == "Global Pricing Overview":
        pr = state["global_pricing_summary"]
        if pr.empty:
            st.info("No multi-priced SKU summary found.")
        else:
            df_out = add_total_row(pr.copy(), numeric_cols=["Total_Diff_Value"], label_col="STORE_NAME")
            display_table_with_format(df_out, int_cols=["Items_with_MultiPrice"], float_cols=["Total_Diff_Value", "Avg_Spread", "Max_Spread"], height=520)
            fig = px.bar(pr.head(20).sort_values("Total_Diff_Value", ascending=True), x="Total_Diff_Value", y="STORE_NAME", orientation="h", color="Items_with_MultiPrice", color_continuous_scale=DIVERGING, title="Top Stores by Value Impact from Multi-Priced SKUs")
            try_plot(fig)
            safe_download_df(df_out, "global_pricing_summary.csv", "⬇️ Download Table")

    # 12 Branch Brach Overview (store sales summary)
    elif subsection == "Branch Brach Overview":
        ssum = state["store_sales_summary"]
        if ssum.empty:
            st.info("No store-level summary.")
        else:
            df_out = add_total_row(ssum.copy(), numeric_cols=["NET_SALES", "QTY"], label_col="STORE_NAME")
            display_table_with_format(df_out, int_cols=["NET_SALES", "QTY", "RECEIPTS"], height=520)
            safe_download_df(df_out, "branch_overview.csv", "⬇️ Download Table")

    # 13 Global Refunds Overview
    elif subsection == "Global Refunds Overview":
        gr = state["global_refunds"]
        if gr.empty:
            st.info("No negative receipts found.")
        else:
            df_out = add_total_row(gr.copy(), numeric_cols=["Total_Neg_Value"], label_col="STORE_NAME")
            display_table_with_format(df_out, int_cols=["Receipts"], float_cols=["Total_Neg_Value"], height=520)
            fig = px.bar(gr.sort_values("Total_Neg_Value", ascending=True), x="Total_Neg_Value", y="STORE_NAME", orientation="h", color_discrete_sequence=[COLOR_RED], title="Global Refunds by Store")
            try_plot(fig)
            safe_download_df(df_out, "global_refunds.csv", "⬇️ Download Table")

    # 14 Branch Refunds Overview
    elif subsection == "Branch Refunds Overview":
        br = state["branch_refunds_detail"]
        if br.empty:
            st.info("No refunds detail available.")
        else:
            branch = st.selectbox("Select Branch", sorted(br["STORE_NAME"].unique().tolist()))
            dfb = br[br["STORE_NAME"] == branch].copy().sort_values("Value")
            dfb_out = add_total_row(dfb.copy(), numeric_cols=["Value"], label_col="CUST_CODE")
            display_table_with_format(dfb_out, float_cols=["Value"], height=520)
            safe_download_df(dfb_out, f"branch_refunds_{branch}.csv", "⬇️ Download Table")

# ---------- Footer ----------
st.sidebar.markdown("---")
st.sidebar.markdown("All tables include totals, formatted with thousands separators. If numbers appear off, please provide a small CSV sample and I'll reproduce exact steps.")
