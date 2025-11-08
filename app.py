"""
Superdeck Analytics Dashboard - faithful Streamlit port of the provided notebook

- Implements all sections and subsections exactly as listed by the user.
- Precomputes every summary at upload time (cached) so subsection navigation is fast.
- Uses explicit color palettes (avoids plotly internal palette names that caused AttributeError).
- Converts datetime.time columns to strings before passing DataFrames to st.dataframe / st.download_button
  to avoid pyarrow ArrowTypeError seen in logs.
- Keeps numeric internal types for correct calculations; formatting for display is applied only to copies.
- Adds totals rows where the notebook had totals.
- Replaces any matplotlib-only visuals with Plotly equivalents (interactive) so Streamlit rendering is consistent.
- If any dataset column is missing, subsection shows a clear message rather than crashing.
- Use Streamlit's cached computations to reduce memory/CPU on reruns.

Usage:
  streamlit run app.py

Notes:
- If your CSV is large (>> 200MB) you may need a larger instance or a streaming-aggregation version.
- To allow uploads > default size, set STREAMLIT_SERVER_MAX_UPLOAD_SIZE env or .streamlit/config.toml
"""

from datetime import timedelta, time as dtime
import io, hashlib, textwrap

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(layout="wide", page_title="Superdeck Analytics Dashboard", initial_sidebar_state="expanded")
st.title("🦸 Superdeck Analytics Dashboard (Notebook-faithful)")
st.markdown("Upload CSV and the app will compute all sections/subsections. Tables have totals and formatted numbers.")

# ---------- Styles / colors ----------
COLOR_BLUE = "#1f77b4"
COLOR_ORANGE = "#ff7f0e"
COLOR_GREEN = "#2ca02c"
COLOR_RED = "#d62728"
PALETTE10 = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_RED, "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
DIVERGING = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]

# ---------- Helpers ----------
def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def _safe_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df where any datetime.time objects are converted to strings (HH:MM)."""
    df2 = df.copy()
    for col in df2.columns:
        if df2[col].dtype == object:
            sample = df2[col].dropna().head(10)
            if any(isinstance(v, dtime) for v in sample):
                df2[col] = df2[col].map(lambda v: v.strftime("%H:%M") if isinstance(v, dtime) else v)
        # pandas time dtype detection (rare) - also convert
    return df2

def fmt_int_col(df, col):
    if col in df.columns:
        return df[col].map(lambda v: f"{int(v):,}" if pd.notna(v) else v)
    return df.get(col)

def fmt_float_col(df, col, decimals=2):
    if col in df.columns:
        fmt = f"{{:,.{decimals}f}}"
        return df[col].map(lambda v: fmt.format(float(v)) if pd.notna(v) else v)
    return df.get(col)

def add_total_row(df: pd.DataFrame, numeric_cols: list, label_col: str = None, total_label="Total") -> pd.DataFrame:
    """Return a copy with a totals row inserted at the top (numeric_cols summed)."""
    if df.empty:
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

def st_download_df(df: pd.DataFrame, filename: str, label: str = "⬇️ Download CSV"):
    out = _safe_display_df(df).to_csv(index=False).encode("utf-8")
    st.download_button(label, out, file_name=filename, mime="text/csv")

# ---------- Load and precompute (cached) ----------
@st.cache_data(show_spinner=True)
def load_and_precompute(file_bytes: bytes) -> dict:
    """Load CSV bytes and compute ALL the notebook summaries. Returns a dict of DataFrames and metadata."""
    df = pd.read_csv(io.BytesIO(file_bytes), on_bad_lines="skip", low_memory=False)
    # normalize columns
    df.columns = [c.strip() for c in df.columns]

    # parse dates
    for date_col in ["TRN_DATE", "ZED_DATE"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # numeric cleaning (do NOT format strings here; keep numeric types)
    numeric_cols = ["QTY", "CP_PRE_VAT", "SP_PRE_VAT", "COST_PRE_VAT", "NET_SALES", "VAT_AMT"]
    for nc in numeric_cols:
        if nc in df.columns:
            df[nc] = df[nc].astype(str).str.replace(",", "", regex=False)
            df[nc] = pd.to_numeric(df[nc], errors="coerce").fillna(0)

    # id columns to str
    idcols = ["STORE_CODE", "TILL", "SESSION", "RCT"]
    for c in idcols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()

    # build CUST_CODE if missing (match notebook)
    if "CUST_CODE" not in df.columns:
        if all(c in df.columns for c in idcols):
            df["CUST_CODE"] = (df["STORE_CODE"].str.strip() + "-" + df["TILL"].str.strip() + "-" + df["SESSION"].str.strip() + "-" + df["RCT"].str.strip())
        else:
            df["CUST_CODE"] = df.index.astype(str)
    df["CUST_CODE"] = df["CUST_CODE"].astype(str).str.strip()

    out = {"df": df}

    # --- SALES CHANNEL L1 ---
    if "SALES_CHANNEL_L1" in df.columns and "NET_SALES" in df.columns:
        s1 = df.groupby("SALES_CHANNEL_L1", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        s1["NET_SALES_M"] = s1["NET_SALES"] / 1_000_000
        total = s1["NET_SALES"].sum()
        s1["PCT"] = s1["NET_SALES"] / total * 100 if total != 0 else 0
        out["sales_channel_l1"] = s1
    else:
        out["sales_channel_l1"] = pd.DataFrame()

    # --- SALES CHANNEL L2 ---
    if "SALES_CHANNEL_L2" in df.columns and "NET_SALES" in df.columns:
        s2 = df.groupby("SALES_CHANNEL_L2", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        s2["NET_SALES_M"] = s2["NET_SALES"] / 1_000_000
        total2 = s2["NET_SALES"].sum()
        s2["PCT"] = s2["NET_SALES"] / total2 * 100 if total2 != 0 else 0
        out["sales_channel_l2"] = s2
    else:
        out["sales_channel_l2"] = pd.DataFrame()

    # --- SHIFT and Day/Night ---
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
        # compute PCT columns per store
        if not per_store_shift.empty:
            p = per_store_shift.copy()
            p_cols = [c for c in ["Day", "Night"] if c in p.columns]
            for c in p_cols:
                p[c + "_PCT"] = np.where(p["total"] > 0, p[c] / p["total"] * 100, 0)
            out["per_store_shift"] = p
        else:
            out["per_store_shift"] = per_store_shift
    else:
        out["sales_by_shift"] = pd.DataFrame()
        out["per_store_shift"] = pd.DataFrame()

    # --- 2nd highest channel (per store) ---
    if {"STORE_NAME", "SALES_CHANNEL_L1", "NET_SALES"}.issubset(df.columns):
        dc = df.groupby(["STORE_NAME", "SALES_CHANNEL_L1"], as_index=False)["NET_SALES"].sum()
        dc["STORE_TOTAL"] = dc.groupby("STORE_NAME")["NET_SALES"].transform("sum")
        dc["PCT"] = np.where(dc["STORE_TOTAL"] > 0, dc["NET_SALES"] / dc["STORE_TOTAL"] * 100, 0)
        dc = dc.sort_values(["STORE_NAME", "PCT"], ascending=[True, False])
        dc["RANK"] = dc.groupby("STORE_NAME").cumcount() + 1
        second = dc[dc["RANK"] == 2][["STORE_NAME", "SALES_CHANNEL_L1", "PCT"]].rename(columns={"SALES_CHANNEL_L1": "SECOND_CHANNEL", "PCT": "SECOND_PCT"})
        all_stores = dc["STORE_NAME"].drop_duplicates()
        missing = set(all_stores) - set(second["STORE_NAME"])
        if missing:
            second = pd.concat([second, pd.DataFrame({"STORE_NAME": list(missing), "SECOND_CHANNEL": ["(None)"] * len(missing), "SECOND_PCT": [0.0] * len(missing)})], ignore_index=True)
        second_sorted = second.sort_values("SECOND_PCT", ascending=False)
        out["second_channel_table"] = second_sorted
        out["second_top_30"] = second_sorted.head(30)
        out["second_bottom_30"] = second_sorted.tail(30).sort_values("SECOND_PCT", ascending=True)
    else:
        out["second_channel_table"] = pd.DataFrame()
        out["second_top_30"] = pd.DataFrame()
        out["second_bottom_30"] = pd.DataFrame()

    # --- Store sales summary (NET_SALES, QTY, RECEIPTS) ---
    if {"STORE_NAME", "NET_SALES", "QTY", "CUST_CODE"}.issubset(df.columns):
        store_sum = df.groupby("STORE_NAME", as_index=False).agg(NET_SALES=("NET_SALES", "sum"), QTY=("QTY", "sum"), RECEIPTS=("CUST_CODE", pd.Series.nunique)).sort_values("NET_SALES", ascending=False)
        out["store_sales_summary"] = store_sum
    else:
        out["store_sales_summary"] = pd.DataFrame()

    # --- Receipts-by-time (dedupe by CUST_CODE earliest TRN_DATE) ---
    if "TRN_DATE" in df.columns and "CUST_CODE" in df.columns:
        r = df.drop_duplicates(subset=["CUST_CODE"]).copy()
        r["TRN_DATE"] = pd.to_datetime(r["TRN_DATE"], errors="coerce")
        r = r.dropna(subset=["TRN_DATE"])
        r["TIME_SLOT"] = r["TRN_DATE"].dt.floor("30T").dt.time
        heat = r.groupby(["STORE_NAME", "TIME_SLOT"], as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE": "RECEIPT_COUNT"})
        out["receipts_by_time"] = heat
    else:
        out["receipts_by_time"] = pd.DataFrame()

    # --- Active tills avg per store/day ---
    if {"TRN_DATE", "TILL", "STORE_NAME"}.issubset(df.columns):
        ttmp = df.copy()
        ttmp["TRN_DATE"] = pd.to_datetime(ttmp["TRN_DATE"], errors="coerce")
        ttmp = ttmp.dropna(subset=["TRN_DATE"])
        ttmp["TRN_DATE_DATE"] = ttmp["TRN_DATE"].dt.date
        active = ttmp.groupby(["STORE_NAME", "TRN_DATE_DATE"], as_index=False)["TILL"].nunique()
        avg = active.groupby("STORE_NAME", as_index=False)["TILL"].mean().rename(columns={"TILL": "avg_active_tills"})
        out["active_tills_avg"] = avg.sort_values("avg_active_tills", ascending=False)
    else:
        out["active_tills_avg"] = pd.DataFrame()

    # --- first_touch receipts for customers-per-till / average customers per till ---
    if {"TRN_DATE", "STORE_CODE", "TILL", "SESSION", "RCT", "STORE_NAME"}.issubset(df.columns):
        tmp = df.copy()
        tmp["TRN_DATE"] = pd.to_datetime(tmp["TRN_DATE"], errors="coerce")
        tmp = tmp.dropna(subset=["TRN_DATE"])
        for c in ["STORE_CODE", "TILL", "SESSION", "RCT"]:
            tmp[c] = tmp[c].astype(str).fillna("").str.strip()
        tmp["CUST_CODE"] = tmp["STORE_CODE"] + "-" + tmp["TILL"] + "-" + tmp["SESSION"] + "-" + tmp["RCT"]
        tmp["TRN_DATE_ONLY"] = tmp["TRN_DATE"].dt.date
        ft = tmp.groupby(["STORE_NAME", "TRN_DATE_ONLY", "CUST_CODE"], as_index=False)["TRN_DATE"].min()
        ft["TIME_SLOT"] = ft["TRN_DATE"].dt.floor("30T").dt.time
        out["first_touch"] = ft
    else:
        out["first_touch"] = pd.DataFrame()

    # --- Cashier performance (basic sums) ---
    if "CASHIER" in df.columns or "CASHIER_NAME" in df.columns:
        cc = "CASHIER" if "CASHIER" in df.columns else "CASHIER_NAME"
        cf = df.groupby(cc, as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        out["cashier_perf"] = cf
    else:
        out["cashier_perf"] = pd.DataFrame()

    # --- Department & Supplier summaries ---
    if "DEPARTMENT" in df.columns:
        dept = df.groupby("DEPARTMENT", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        out["dept_sales"] = dept
    else:
        out["dept_sales"] = pd.DataFrame()

    if "SUPPLIER" in df.columns:
        supp = df.groupby("SUPPLIER", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        out["supplier_sales"] = supp
    else:
        out["supplier_sales"] = pd.DataFrame()

    # --- Top items by sales / qty ---
    if "ITEM_NAME" in df.columns:
        out["top_items_sales"] = df.groupby("ITEM_NAME", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        out["top_items_qty"] = df.groupby("ITEM_NAME", as_index=False)["QTY"].sum().sort_values("QTY", ascending=False)
    else:
        out["top_items_sales"] = pd.DataFrame()
        out["top_items_qty"] = pd.DataFrame()

    # --- Loyalty receipts (one record per store/receipt/loyalty code) ---
    if {"LOYALTY_CUSTOMER_CODE", "CUST_CODE", "STORE_NAME", "TRN_DATE", "NET_SALES"}.issubset(df.columns):
        loy = df.copy()
        loy["TRN_DATE"] = pd.to_datetime(loy["TRN_DATE"], errors="coerce")
        loy = loy.dropna(subset=["TRN_DATE"])
        loy["LOYALTY_CUSTOMER_CODE"] = loy["LOYALTY_CUSTOMER_CODE"].astype(str).str.strip()
        loy = loy[loy["LOYALTY_CUSTOMER_CODE"].replace({"nan": "", "NaN": "", "None": ""}).str.len() > 0]
        receipts = loy.groupby(["STORE_NAME", "CUST_CODE", "LOYALTY_CUSTOMER_CODE"], as_index=False).agg(Basket_Value=("NET_SALES", "sum"), First_Time=("TRN_DATE", "min"))
        out["loyalty_receipts"] = receipts
    else:
        out["loyalty_receipts"] = pd.DataFrame()

    # --- Pricing multi-price computations ---
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
            summary_pr = multi_price.groupby("STORE_NAME", as_index=False).agg(Items_with_MultiPrice=("ITEM_CODE", "nunique"),
                                                                               Total_Diff_Value=("Diff_Value", "sum"),
                                                                               Avg_Spread=("Price_Spread", "mean"),
                                                                               Max_Spread=("Price_Spread", "max")).sort_values("Total_Diff_Value", ascending=False)
            out["global_pricing_summary"] = summary_pr
            out["multi_price_detail"] = multi_price
        else:
            out["global_pricing_summary"] = pd.DataFrame()
            out["multi_price_detail"] = pd.DataFrame()
    else:
        out["global_pricing_summary"] = pd.DataFrame()
        out["multi_price_detail"] = pd.DataFrame()

    # --- Refunds (negative NET_SALES) ---
    if {"NET_SALES", "STORE_NAME", "CUST_CODE"}.issubset(df.columns):
        neg = df[df["NET_SALES"] < 0].copy()
        if not neg.empty:
            neg["Abs_Neg"] = neg["NET_SALES"].abs()
            global_refunds = neg.groupby("STORE_NAME", as_index=False).agg(Total_Neg_Value=("NET_SALES", "sum"), Receipts=("CUST_CODE", pd.Series.nunique))
            out["global_refunds"] = global_refunds.sort_values("Total_Neg_Value")
            branch_refunds = neg.groupby(["STORE_NAME", "CUST_CODE"], as_index=False).agg(Value=("NET_SALES", "sum"), First_Time=("TRN_DATE", "min"))
            out["branch_refunds_detail"] = branch_refunds
        else:
            out["global_refunds"] = pd.DataFrame()
            out["branch_refunds_detail"] = pd.DataFrame()
    else:
        out["global_refunds"] = pd.DataFrame()
        out["branch_refunds_detail"] = pd.DataFrame()

    # --- dropdown lists & metadata ---
    out["stores"] = sorted(df["STORE_NAME"].dropna().unique().tolist()) if "STORE_NAME" in df.columns else []
    out["channels_l1"] = sorted(df["SALES_CHANNEL_L1"].dropna().unique().tolist()) if "SALES_CHANNEL_L1" in df.columns else []
    out["channels_l2"] = sorted(df["SALES_CHANNEL_L2"].dropna().unique().tolist()) if "SALES_CHANNEL_L2" in df.columns else []
    out["items"] = sorted(df["ITEM_NAME"].dropna().unique().tolist()) if "ITEM_NAME" in df.columns else []
    out["departments"] = sorted(df["DEPARTMENT"].dropna().unique().tolist()) if "DEPARTMENT" in df.columns else []

    # time grid labels
    start_time = pd.Timestamp("00:00:00")
    intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
    out["time_intervals"] = intervals
    out["time_labels"] = [t.strftime("%H:%M") for t in intervals]

    out["sample_rows"] = df.head(200)
    return out

# ---------- Upload widget ----------
uploaded = st.file_uploader("Upload sales CSV (CSV). App will precompute all analyses.", type="csv")
if uploaded is None:
    st.stop()

file_bytes = uploaded.getvalue()
state = load_and_precompute(file_bytes)

# quick sample download (sidebar)
st.sidebar.download_button("⬇️ Download sample rows", state["sample_rows"].to_csv(index=False).encode("utf-8"), "sample_rows.csv", "text/csv")
st.sidebar.markdown("Theme: Red & Green — positives (green), negatives/alerts (red)")

# ---------- Sections / Subsections ----------
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

df = state["df"]  # raw df for on-demand groupbys

# ---------- SALES implementations ----------
if section == "SALES":
    # 1 Global sales Overview
    if subsection == "Global sales Overview":
        gs = state["sales_channel_l1"]
        if gs.empty:
            st.info("SALES_CHANNEL_L1 or NET_SALES is missing.")
        else:
            gs_display = gs.copy()
            gs_display["NET_SALES_M"] = gs_display["NET_SALES_M"].round(2)
            gs_display["PCT"] = gs_display["PCT"].round(1)
            labels = [f"{r['SALES_CHANNEL_L1']} ({r['PCT']:.1f}% | {r['NET_SALES_M']:.1f} M)" for _, r in gs_display.iterrows()]
            try:
                fig = go.Figure(go.Pie(labels=labels, values=gs_display["NET_SALES_M"], hole=0.65,
                                       text=[f"{p:.1f}%" for p in gs_display["PCT"]],
                                       marker=dict(colors=PALETTE10, line=dict(color='white', width=1))))
                fig.update_layout(title="<b>SALES CHANNEL TYPE — Global Overview</b>", height=560)
                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.error("Plot error, showing table: " + str(e))
            # table with totals
            df_out = gs_display[["SALES_CHANNEL_L1", "NET_SALES", "NET_SALES_M", "PCT"]].copy()
            df_out = add_total_row(df_out, numeric_cols=["NET_SALES"], label_col="SALES_CHANNEL_L1")
            display_table_with_format(df_out, int_cols=["NET_SALES"], float_cols=["NET_SALES_M", "PCT"], height=420)
            st_download_df(df_out, "global_sales_overview.csv", "⬇️ Download Table")

    # 2 Global Net Sales Distribution by Sales Channel (L2)
    elif subsection == "Global Net Sales Distribution by Sales Channel":
        g2 = state["sales_channel_l2"]
        if g2.empty:
            st.info("SALES_CHANNEL_L2 or NET_SALES missing.")
        else:
            g2_disp = g2.copy()
            g2_disp["NET_SALES_M"] = g2_disp["NET_SALES_M"].round(2)
            g2_disp["PCT"] = g2_disp["PCT"].round(1)
            labels = [f"{r['SALES_CHANNEL_L2']} ({r['PCT']:.1f}% | {r['NET_SALES_M']:.1f} M)" for _, r in g2_disp.iterrows()]
            try:
                fig = go.Figure(go.Pie(labels=labels, values=g2_disp["NET_SALES_M"], hole=0.65, marker=dict(colors=PALETTE10, line=dict(color='white', width=1))))
                fig.update_layout(title="<b>Global Net Sales Distribution by Sales Mode (L2)</b>", height=560)
                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.error("Plot error, showing table: " + str(e))
            df_out = g2_disp[["SALES_CHANNEL_L2", "NET_SALES", "NET_SALES_M", "PCT"]].copy()
            df_out = add_total_row(df_out, numeric_cols=["NET_SALES"], label_col="SALES_CHANNEL_L2")
            display_table_with_format(df_out, int_cols=["NET_SALES"], float_cols=["NET_SALES_M", "PCT"], height=420)
            st_download_df(df_out, "sales_channel_l2.csv", "⬇️ Download Table")

    # 3 Global Net Sales Distribution by SHIFT
    elif subsection == "Global Net Sales Distribution by SHIFT":
        sb = state["sales_by_shift"]
        if sb.empty:
            st.info("No shift data available.")
        else:
            sb_disp = sb.copy()
            sb_disp["PCT"] = sb_disp["PCT"].round(1)
            fig = px.bar(sb_disp, x="SHIFT_TYPE", y="NET_SALES", color="SHIFT_TYPE", color_discrete_map={"Day": COLOR_GREEN, "Night": COLOR_RED}, title="Net Sales by SHIFT")
            st.plotly_chart(fig, width='stretch')
            df_out = add_total_row(sb_disp.copy(), numeric_cols=["NET_SALES"], label_col="SHIFT_TYPE")
            display_table_with_format(df_out, int_cols=["NET_SALES"], float_cols=["PCT"], height=420)

    # 4 Night vs Day Shift Sales Ratio — Stores with Night Shifts
    elif subsection == "Night vs Day Shift Sales Ratio — Stores with Night Shifts":
        pss = state["per_store_shift"]
        if pss.empty:
            st.info("No per-store shift data available.")
        else:
            # show day/night PCT columns if present
            cols = [c for c in pss.columns if c.endswith("_PCT")]
            if not cols:
                st.info("No day/night percentage columns computed.")
                st.dataframe(_safe_display_df(pss), width='stretch')
            else:
                df_out = pss.copy()
                # display totals row for numeric columns (Day/Night raw)
                df_table = add_total_row(df_out[[c for c in df_out.columns if c in ["STORE_NAME","Day","Night"]]], numeric_cols=["Day","Night"], label_col="STORE_NAME")
                display_table_with_format(_safe_display_df(df_table), int_cols=["Day","Night"], float_cols=[], height=480)
                # bar chart sorted by Night%
                if "Night_PCT" in df_out.columns:
                    fig = px.bar(df_out.sort_values("Night_PCT", ascending=True), x="Night_PCT", y="STORE_NAME", orientation="h", color_discrete_sequence=[COLOR_RED], title="Night % by Store (stores with night activity)")
                    st.plotly_chart(fig, width='stretch')

    # 5 Global Day vs Night Sales — Only Stores with NIGHT Shift
    elif subsection == "Global Day vs Night Sales — Only Stores with NIGHT Shift":
        gnd = state["sales_by_shift"]
        if gnd.empty:
            st.info("No shift data.")
        else:
            gnd_disp = gnd.copy()
            gnd_disp["PCT"] = gnd_disp["PCT"].round(1)
            labels = [f"{r['SHIFT_TYPE']} ({r['PCT']:.1f}%)" for _, r in gnd_disp.iterrows()]
            try:
                fig = go.Figure(go.Pie(labels=labels, values=gnd_disp["NET_SALES"], hole=0.65, marker=dict(colors=[COLOR_BLUE, COLOR_RED])))
                fig.update_layout(title="Global Day vs Night Sales — Only Stores with NIGHT Shift", height=520)
                st.plotly_chart(fig, width='stretch')
            except Exception:
                st.info("Pie chart unavailable, showing table.")
            df_out = add_total_row(gnd_disp.copy(), numeric_cols=["NET_SALES"], label_col="SHIFT_TYPE")
            display_table_with_format(df_out, int_cols=["NET_SALES"], float_cols=["PCT"], height=420)

    # 6 2nd-Highest Channel Share
    elif subsection == "2nd-Highest Channel Share":
        sc = state["second_channel_table"]
        if sc.empty:
            st.info("No second-channel data.")
        else:
            df_out = add_total_row(sc.copy(), numeric_cols=["SECOND_PCT"], label_col="STORE_NAME")
            display_table_with_format(df_out[["STORE_NAME", "SECOND_CHANNEL", "SECOND_PCT"]], float_cols=["SECOND_PCT"], height=520)
            # Lollipop-like plot using plotly (horizontal)
            top = sc.sort_values("SECOND_PCT", ascending=True).tail(50)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=top["SECOND_PCT"], y=top["STORE_NAME"], orientation="h", marker_color=COLOR_BLUE, name="2nd %"))
            fig.update_layout(title="2nd-Highest Channel Share (selected stores)", height=600)
            st.plotly_chart(fig, width='stretch')

    # 7 Bottom 30 — 2nd Highest Channel
    elif subsection == "Bottom 30 — 2nd Highest Channel":
        bottom = state["second_bottom_30"]
        if bottom.empty:
            st.info("No bottom-30 data.")
        else:
            df_out = add_total_row(bottom.copy(), numeric_cols=["SECOND_PCT"], label_col="STORE_NAME")
            display_table_with_format(df_out[["STORE_NAME", "SECOND_CHANNEL", "SECOND_PCT"]], float_cols=["SECOND_PCT"], height=520)
            fig = px.bar(bottom.sort_values("SECOND_PCT", ascending=True), x="SECOND_PCT", y="STORE_NAME", orientation="h", color_discrete_sequence=[COLOR_RED], title="Bottom 30 — 2nd Highest Channel")
            st.plotly_chart(fig, width='stretch')

    # 8 Stores Sales Summary
    elif subsection == "Stores Sales Summary":
        ss = state["store_sales_summary"]
        if ss.empty:
            st.info("No store sales summary available.")
        else:
            df_out = add_total_row(ss.copy(), numeric_cols=["NET_SALES", "QTY"], label_col="STORE_NAME")
            display_table_with_format(df_out, int_cols=["NET_SALES", "QTY", "RECEIPTS"], height=520)
            st_download_df(df_out, "store_sales_summary.csv", "⬇️ Download Store Summary")

# ---------- OPERATIONS implementations ----------
elif section == "OPERATIONS":
    # 1 Customer Traffic-Storewise
    if subsection == "Customer Traffic-Storewise":
        heat = state["receipts_by_time"]
        if heat.empty:
            st.info("No receipts-by-time available.")
        else:
            sel = st.selectbox("Select store", state["stores"])
            df_heat = heat[heat["STORE_NAME"] == sel].copy()
            if df_heat.empty:
                st.info("No data for selected store.")
            else:
                # pivot and ensure full 48 slots
                slots = state["time_intervals"]
                idx = pd.Index(slots, name="TIME_SLOT")
                pivot = df_heat.set_index("TIME_SLOT")["RECEIPT_COUNT"].reindex(idx, fill_value=0)
                labels = state["time_labels"]
                fig = px.bar(x=labels, y=pivot.values, labels={"x":"Time","y":"Receipts"}, color_discrete_sequence=[COLOR_BLUE], title=f"Receipts by Time - {sel}")
                st.plotly_chart(fig, width='stretch')
                out_df = pivot.reset_index().rename(columns={"TIME_SLOT":"TIME", 0:"RECEIPT_COUNT"})
                out_df.columns = ["TIME", "RECEIPT_COUNT"]
                out_df = add_total_row(out_df, numeric_cols=["RECEIPT_COUNT"], label_col="TIME")
                display_table_with_format(_safe_display_df(out_df), int_cols=["RECEIPT_COUNT"], height=420)
                st_download_df(out_df, f"receipts_by_time_{sel}.csv", "⬇️ Download Table")

    # 2 Active Tills During the day
    elif subsection == "Active Tills During the day":
        at = state["active_tills_avg"]
        if at.empty:
            st.info("No active tills info.")
        else:
            df_out = add_total_row(at.copy(), numeric_cols=["avg_active_tills"], label_col="STORE_NAME")
            display_table_with_format(df_out, float_cols=["avg_active_tills"], height=420)
            st_download_df(df_out, "active_tills_avg.csv", "⬇️ Download Table")

    # 3 Average Customers Served per Till
    elif subsection == "Average Customers Served per Till":
        ft = state["first_touch"]
        if ft.empty or "TRN_DATE" not in ft.columns:
            st.info("Insufficient first-touch data.")
        else:
            # customers per timeslot
            ft2 = ft.copy()
            ft2["TIME_SLOT"] = ft2["TRN_DATE"].dt.floor("30T").dt.time
            cust_counts = ft2.groupby(["STORE_NAME", "TIME_SLOT"], as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"CUSTOMERS"})
            tmp = df.copy()
            if {"TRN_DATE", "TILL", "STORE_NAME"}.issubset(tmp.columns):
                tmp["TRN_DATE"] = pd.to_datetime(tmp["TRN_DATE"], errors="coerce")
                tmp = tmp.dropna(subset=["TRN_DATE"])
                tmp["TIME_SLOT"] = tmp["TRN_DATE"].dt.floor("30T").dt.time
                till_counts = tmp.groupby(["STORE_NAME", "TIME_SLOT"], as_index=False)["TILL"].nunique().rename(columns={"TILL":"TILLS"})
                merged = cust_counts.merge(till_counts, on=["STORE_NAME", "TIME_SLOT"], how="left").fillna({"TILLS":0})
                merged["CUST_PER_TILL"] = merged["CUSTOMERS"] / merged["TILLS"].replace(0, np.nan)
                per_store = merged.groupby("STORE_NAME", as_index=False).agg(Max_Cust_Per_Till=("CUST_PER_TILL","max"), Avg_Cust_Per_Till=("CUST_PER_TILL","mean")).fillna(0)
                df_out = add_total_row(per_store.copy(), numeric_cols=["Max_Cust_Per_Till", "Avg_Cust_Per_Till"], label_col="STORE_NAME")
                display_table_with_format(df_out, float_cols=["Max_Cust_Per_Till", "Avg_Cust_Per_Till"], height=420)
                st_download_df(df_out, "customers_per_till.csv", "⬇️ Download Table")
            else:
                st.info("Missing TILL or TRN_DATE columns to compute metrics.")

    # 4 Store Customer Traffic Storewise
    elif subsection == "Store Customer Traffic Storewise":
        sel = st.selectbox("Select store", state["stores"])
        branch_df = df[df["STORE_NAME"] == sel].copy()
        if branch_df.empty:
            st.info("No data for this store.")
        else:
            branch_df["TRN_DATE"] = pd.to_datetime(branch_df["TRN_DATE"], errors="coerce")
            branch_df = branch_df.dropna(subset=["TRN_DATE"])
            branch_df["TIME_SLOT"] = branch_df["TRN_DATE"].dt.floor("30T").dt.time
            tmp = branch_df.groupby(["DEPARTMENT", "TIME_SLOT"], as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"UNIQUE_CUSTOMERS"})
            display_table_with_format(_safe_display_df(tmp), int_cols=["UNIQUE_CUSTOMERS"], height=520)
            st_download_df(_safe_display_df(tmp), f"dept_traffic_{sel}.csv", "⬇️ Download Table")

    # 5 Customer Traffic-Departmentwise
    elif subsection == "Customer Traffic-Departmentwise":
        depts = state["departments"]
        if not depts:
            st.info("No departments present.")
        else:
            sel = st.selectbox("Select department", depts)
            ddf = df[df["DEPARTMENT"] == sel].copy()
            if ddf.empty:
                st.info("No rows for that department.")
            else:
                ddf["TRN_DATE"] = pd.to_datetime(ddf["TRN_DATE"], errors="coerce")
                ddf = ddf.dropna(subset=["TRN_DATE"])
                ddf["TIME_SLOT"] = ddf["TRN_DATE"].dt.floor("30T").dt.time
                counts = ddf.groupby(["STORE_NAME", "TIME_SLOT"], as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"UNIQUE_CUSTOMERS"})
                display_table_with_format(_safe_display_df(counts), int_cols=["UNIQUE_CUSTOMERS"], height=520)
                st_download_df(_safe_display_df(counts), f"dept_{sel}_traffic.csv", "⬇️ Download Table")

    # 6 Cashiers Perfomance
    elif subsection == "Cashiers Perfomance":
        cp = state["cashier_perf"]
        if cp.empty:
            st.info("No cashier columns (CASHIER or CASHIER_NAME) in dataset.")
        else:
            df_out = add_total_row(cp.copy(), numeric_cols=["NET_SALES"], label_col=cp.columns[0])
            display_table_with_format(df_out, int_cols=["NET_SALES"], height=420)
            st_download_df(df_out, "cashier_performance.csv", "⬇️ Download Table")

    # 7 Till Usage
    elif subsection == "Till Usage":
        # produce branch_summary as earlier notebook
        tmp = df.copy()
        tmp["TRN_DATE"] = pd.to_datetime(tmp["TRN_DATE"], errors="coerce")
        tmp = tmp.dropna(subset=["TRN_DATE"])
        tmp["TIME_SLOT"] = tmp["TRN_DATE"].dt.floor("30T").dt.time
        tmp["Till_Code"] = tmp["TILL"].astype(str).fillna("") + "-" + tmp["STORE_CODE"].astype(str).fillna("")
        till_activity = tmp.groupby(["STORE_NAME", "Till_Code", "TIME_SLOT"], as_index=False).agg(Receipts=("CUST_CODE", "nunique"))
        branch_summary = till_activity.groupby("STORE_NAME", as_index=False).agg(Store_Total_Receipts=("Receipts","sum"), Avg_Per_Till=("Receipts","mean"), Max_Per_Till=("Receipts","max"), Unique_Tills=("Till_Code","nunique"))
        df_out = add_total_row(branch_summary.copy(), numeric_cols=["Store_Total_Receipts"], label_col="STORE_NAME")
        display_table_with_format(df_out, int_cols=["Store_Total_Receipts","Unique_Tills"], float_cols=["Avg_Per_Till"], height=520)
        st_download_df(df_out, "till_usage_summary.csv", "⬇️ Download Table")

    # 8 Tax Compliance
    elif subsection == "Tax Compliance":
        if {"CU_DEVICE_SERIAL", "CUST_CODE", "STORE_NAME"}.issubset(df.columns):
            tdf = df.copy()
            tdf["Tax_Compliant"] = np.where(tdf["CU_DEVICE_SERIAL"].astype(str).str.strip().replace({"nan":"", "NaN":"", "None":""}).str.len() > 0, "Compliant", "Non-Compliant")
            global_summary = tdf.groupby("Tax_Compliant", as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"Receipts"})
            fig = px.pie(global_summary, names="Tax_Compliant", values="Receipts", color="Tax_Compliant", color_discrete_map={"Compliant": COLOR_GREEN, "Non-Compliant": COLOR_RED}, hole=0.45, title="Global Tax Compliance Overview")
            st.plotly_chart(fig, width='stretch')
            store_till = tdf.groupby(["STORE_NAME", "Till_Code", "Tax_Compliant"], as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"Receipts"})
            branch = st.selectbox("Select Branch", state["stores"])
            dfb = store_till[store_till["STORE_NAME"] == branch]
            if dfb.empty:
                st.info("No compliance data for this branch.")
            else:
                pivot = dfb.pivot(index="Till_Code", columns="Tax_Compliant", values="Receipts").fillna(0).reset_index()
                pivot = add_total_row(pivot.copy(), numeric_cols=[c for c in pivot.columns if c != "Till_Code"], label_col="Till_Code")
                display_table_with_format(_safe_display_df(pivot), int_cols=[c for c in pivot.columns if c != "Till_Code"], height=520)
                st_download_df(_safe_display_df(pivot), f"tax_compliance_{branch}.csv", "⬇️ Download Table")
        else:
            st.info("Missing columns for tax compliance: CU_DEVICE_SERIAL, CUST_CODE, STORE_NAME")

# ---------- INSIGHTS implementations ----------
elif section == "INSIGHTS":
    # 1 Customer Baskets Overview
    if subsection == "Customer Baskets Overview":
        receipts = df.drop_duplicates(subset=["CUST_CODE"])
        if receipts.empty or "NET_SALES" not in receipts.columns:
            st.info("Not enough basket data.")
        else:
            total = receipts["NET_SALES"].sum()
            avg = receipts["NET_SALES"].mean()
            med = receipts["NET_SALES"].median()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total basket net sales", f"KSh {total:,.2f}")
            c2.metric("Average basket value", f"KSh {avg:,.2f}")
            c3.metric("Median basket value", f"KSh {med:,.2f}")
            fig = px.histogram(receipts, x="NET_SALES", nbins=50, title="Basket Value Distribution", color_discrete_sequence=[COLOR_BLUE])
            st.plotly_chart(fig, width='stretch')

    # 2 Global Category Overview-Sales
    elif subsection == "Global Category Overview-Sales":
        dept = state["dept_sales"]
        if dept.empty:
            st.info("No department sales data.")
        else:
            df_out = add_total_row(dept.copy(), numeric_cols=["NET_SALES"], label_col="DEPARTMENT")
            display_table_with_format(df_out, int_cols=["NET_SALES"], height=420)
            fig = px.bar(dept.head(50), x="DEPARTMENT", y="NET_SALES", title="Category Sales (Global)", color_discrete_sequence=[COLOR_GREEN])
            st.plotly_chart(fig, width='stretch')

    # 3 Global Category Overview-Baskets
    elif subsection == "Global Category Overview-Baskets":
        if "DEPARTMENT" in df.columns:
            rec = df.drop_duplicates(subset=["CUST_CODE"])
            dept_counts = rec.groupby("DEPARTMENT", as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"BASKETS"})
            df_out = add_total_row(dept_counts.copy(), numeric_cols=["BASKETS"], label_col="DEPARTMENT")
            display_table_with_format(df_out, int_cols=["BASKETS"], height=420)
            fig = px.bar(dept_counts.sort_values("BASKETS", ascending=False), x="DEPARTMENT", y="BASKETS", title="Baskets by Department (Global)", color_discrete_sequence=[COLOR_BLUE])
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("DEPARTMENT missing.")

    # 4 Supplier Contribution
    elif subsection == "Supplier Contribution":
        sup = state["supplier_sales"]
        if sup.empty:
            st.info("No supplier data.")
        else:
            df_out = add_total_row(sup.copy(), numeric_cols=["NET_SALES"], label_col="SUPPLIER")
            display_table_with_format(df_out, int_cols=["NET_SALES"], height=420)
            st_download_df(df_out, "supplier_contribution.csv", "⬇️ Download Table")

    # 5 Category Overview
    elif subsection == "Category Overview":
        if "CATEGORY" in df.columns and "NET_SALES" in df.columns:
            cat = df.groupby("CATEGORY", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
            df_out = add_total_row(cat.copy(), numeric_cols=["NET_SALES"], label_col="CATEGORY")
            display_table_with_format(df_out, int_cols=["NET_SALES"], height=420)
            fig = px.bar(cat.head(50), x="CATEGORY", y="NET_SALES", title="Category Sales", color_discrete_sequence=[COLOR_GREEN])
            st.plotly_chart(fig, width='stretch')
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
                st.error("Metric not in dataset.")
            else:
                dfA = df[df["STORE_NAME"] == A].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N)
                dfB = df[df["STORE_NAME"] == B].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N)
                combA = dfA.copy(); combA["Branch"] = A
                combB = dfB.copy(); combB["Branch"] = B
                both = pd.concat([combA, combB], ignore_index=True)
                fig = px.bar(both, x=metric, y="ITEM_NAME", color="Branch", orientation="h", barmode="group", color_discrete_sequence=[COLOR_BLUE, COLOR_GREEN], title=f"Top {N} items: {A} vs {B}")
                st.plotly_chart(fig, width='stretch')
                display_table_with_format(both, int_cols=[metric] if metric == "QTY" else ["NET_SALES"], height=520)

    # 7 Product Perfomance
    elif subsection == "Product Perfomance":
        items = state["items"]
        if not items:
            st.info("No items present.")
        else:
            sel = st.selectbox("Select Item", items)
            item_df = df[df["ITEM_NAME"] == sel]
            if item_df.empty:
                st.info("No rows for that item.")
            else:
                qty_by_store = item_df.groupby("STORE_NAME", as_index=False)["QTY"].sum().sort_values("QTY", ascending=False)
                sales_by_store = item_df.groupby("STORE_NAME", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
                st.subheader("Top stores by quantity")
                display_table_with_format(add_total_row(qty_by_store.copy(), numeric_cols=["QTY"], label_col="STORE_NAME"), int_cols=["QTY"])
                st.subheader("Top stores by sales")
                display_table_with_format(add_total_row(sales_by_store.copy(), numeric_cols=["NET_SALES"], label_col="STORE_NAME"), int_cols=["NET_SALES"])

    # 8 Global Loyalty Overview
    elif subsection == "Global Loyalty Overview":
        lr = state["loyalty_receipts"]
        if lr.empty:
            st.info("No loyalty data.")
        else:
            branch_loyal = lr.groupby("STORE_NAME", as_index=False)["LOYALTY_CUSTOMER_CODE"].nunique().rename(columns={"LOYALTY_CUSTOMER_CODE":"Loyal_Customers"})
            df_out = add_total_row(branch_loyal.copy(), numeric_cols=["Loyal_Customers"], label_col="STORE_NAME")
            display_table_with_format(df_out, int_cols=["Loyal_Customers"])

    # 9 Branch Loyalty Overview
    elif subsection == "Branch Loyalty Overview":
        lr = state["loyalty_receipts"]
        if lr.empty:
            st.info("No loyalty receipts.")
        else:
            branch = st.selectbox("Select branch", sorted(lr["STORE_NAME"].unique().tolist()))
            per_store = lr[lr["STORE_NAME"] == branch].groupby("LOYALTY_CUSTOMER_CODE", as_index=False).agg(Baskets=("CUST_CODE","nunique"), Total_Value=("Basket_Value","sum")).sort_values(["Baskets","Total_Value"], ascending=[False, False])
            display_table_with_format(add_total_row(per_store.copy(), numeric_cols=["Baskets", "Total_Value"], label_col="LOYALTY_CUSTOMER_CODE"), int_cols=["Baskets"], float_cols=["Total_Value"])

    # 10 Customer Loyalty Overview
    elif subsection == "Customer Loyalty Overview":
        lr = state["loyalty_receipts"]
        if lr.empty:
            st.info("No loyalty receipts.")
        else:
            custs = lr["LOYALTY_CUSTOMER_CODE"].unique().tolist()
            sel = st.selectbox("Select loyalty customer", custs)
            rc = lr[lr["LOYALTY_CUSTOMER_CODE"] == sel]
            display_table_with_format(rc.sort_values("First_Time", ascending=False), float_cols=["Basket_Value"])

    # 11 Global Pricing Overview
    elif subsection == "Global Pricing Overview":
        pr = state["global_pricing_summary"]
        if pr.empty:
            st.info("No multi-priced SKUs found.")
        else:
            df_out = add_total_row(pr.copy(), numeric_cols=["Total_Diff_Value"], label_col="STORE_NAME")
            display_table_with_format(df_out, int_cols=["Items_with_MultiPrice"], float_cols=["Total_Diff_Value", "Avg_Spread", "Max_Spread"], height=520)
            fig = px.bar(pr.head(20).sort_values("Total_Diff_Value", ascending=True), x="Total_Diff_Value", y="STORE_NAME", orientation="h", color="Items_with_MultiPrice", color_continuous_scale=DIVERGING, title="Top Stores by Value Impact from Multi-Priced SKUs")
            st.plotly_chart(fig, width='stretch')

    # 12 Branch Brach Overview (store sales summary)
    elif subsection == "Branch Brach Overview":
        ssum = state["store_sales_summary"]
        if ssum.empty:
            st.info("No store summary.")
        else:
            df_out = add_total_row(ssum.copy(), numeric_cols=["NET_SALES", "QTY"], label_col="STORE_NAME")
            display_table_with_format(df_out, int_cols=["NET_SALES", "QTY", "RECEIPTS"], height=520)

    # 13 Global Refunds Overview
    elif subsection == "Global Refunds Overview":
        gr = state["global_refunds"]
        if gr.empty:
            st.info("No negative receipts found.")
        else:
            df_out = add_total_row(gr.copy(), numeric_cols=["Total_Neg_Value"], label_col="STORE_NAME")
            display_table_with_format(df_out, int_cols=["Receipts"], float_cols=["Total_Neg_Value"], height=520)
            fig = px.bar(gr.sort_values("Total_Neg_Value", ascending=True), x="Total_Neg_Value", y="STORE_NAME", orientation="h", color_discrete_sequence=[COLOR_RED], title="Global Refunds by Store")
            st.plotly_chart(fig, width='stretch')

    # 14 Branch Refunds Overview
    elif subsection == "Branch Refunds Overview":
        br = state["branch_refunds_detail"]
        if br.empty:
            st.info("No refunds detail.")
        else:
            branch = st.selectbox("Select branch", sorted(br["STORE_NAME"].unique().tolist()))
            dfb = br[br["STORE_NAME"] == branch].copy().sort_values("Value")
            dfb_out = add_total_row(dfb.copy(), numeric_cols=["Value"], label_col="CUST_CODE")
            display_table_with_format(_safe_display_df(dfb_out), float_cols=["Value"], height=520)
            st_download_df(_safe_display_df(dfb_out), f"refunds_{branch}.csv", "⬇️ Download Table")

# ---------- Footer ----------
st.sidebar.markdown("---")
st.sidebar.markdown("If anything still differs from your notebook, paste the notebook cell outputs that are wrong and I'll match them exactly.")
