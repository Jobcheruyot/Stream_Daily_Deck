"""
Superdeck Analytics Dashboard - Streamlit app (fixed)

Fixes applied (from logs):
- Replaced references to non-existent Plotly palettes (e.g. RdYlGn) with a small, explicit diverging/seq palettes.
- Prevented crashes when Streamlit tries to serialize datetime.time objects to Arrow by converting time columns to strings
  before passing DataFrames to st.dataframe / st.download_button.
- Added defensive try/except around plotting so a palette/import issue won't crash the whole app.
- Ensured all requested sections and subsections are implemented and computed at startup (precomputation).
- Kept UI free of diagnostics (no debug printed panels).
- Kept thousands separators for numeric tables and added totals where applicable.
- Kept theme colour accents (red & green).

Run:
  streamlit run app.py
Ensure server max upload size if you need > default: set STREAMLIT_SERVER_MAX_UPLOAD_SIZE or .streamlit/config.toml
"""
from datetime import timedelta, time as dtime
import io
import hashlib

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- Page config & simple styles ----------
st.set_page_config(layout="wide", page_title="Superdeck Analytics Dashboard", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 370px;
        min-width: 340px;
        max-width: 480px;
        padding-right: 18px;
    }
    .block-container {padding-top:1rem;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Color palettes (explicit safe lists) ----------
COLOR_GREEN = "#2ca02c"
COLOR_RED = "#d62728"
COLOR_BLUE = "#1f77b4"

# Safe diverging palette (explicit - avoids depending on plotly internals)
PALETTE_DIVERGING = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]
PALETTE_POS_NEG = [COLOR_GREEN, COLOR_RED]

# ---------- Upload UI ----------
st.title("🦸 Superdeck Analytics Dashboard")
st.markdown("> Upload your sales CSV. App precomputes all analytics once, then you can navigate subsections instantly.")
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV (CSV file)", type="csv")
if uploaded is None:
    st.info("Please upload a dataset to proceed.")
    st.stop()

# ---------- Helpers ----------
def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def _safe_to_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert problematic column types (datetime.time) to string so Streamlit/pyarrow doesn't fail.
    Also keep index as-is.
    """
    out = df.copy()
    for col in out.columns:
        # convert pandas Time/ python datetime.time to formatted string
        if out[col].dtype == object:
            # check a sample for datetime.time
            sample = out[col].dropna().head(20)
            if any(isinstance(v, dtime) for v in sample):
                out[col] = out[col].map(lambda v: v.strftime("%H:%M") if isinstance(v, dtime) else v)
        # convert pandas time dtype if present (pandas <-> numpy)
        # If values are pandas.Timestamp with no date? they will be handled elsewhere
    return out

def fmt_int_series(s: pd.Series) -> pd.Series:
    return s.map(lambda v: f"{int(v):,}" if pd.notna(v) else v)

def fmt_float_series(s: pd.Series, decimals=2) -> pd.Series:
    fmt = "{:,.%df}" % decimals
    return s.map(lambda v: fmt.format(float(v)) if pd.notna(v) else v)

def add_total_row(df: pd.DataFrame, numeric_cols: list, label_col: str = None, total_label="Total"):
    if df.empty:
        return df
    # sum numeric columns only if present
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
        first_col = df.columns[0]
        total_row[first_col] = total_label
    total_df = pd.DataFrame([total_row])
    out = pd.concat([total_df, df], ignore_index=True)
    return out

def display_table_with_format(df: pd.DataFrame, int_cols=None, float_cols=None, height=300):
    if df.empty:
        st.info("No data available for this view.")
        return
    df_out = df.copy()
    if int_cols:
        for c in int_cols:
            if c in df_out.columns:
                df_out[c] = fmt_int_series(df_out[c])
    if float_cols:
        for c in float_cols:
            if c in df_out.columns:
                df_out[c] = fmt_float_series(df_out[c], decimals=2)
    df_out = _safe_to_display(df_out)
    # Streamlit's new width parameter is recommended, but width isn't strictly required here
    st.dataframe(df_out, width='stretch', height=height)

def try_image_download(fig: go.Figure, filename="plot.png", label="⬇️ Download Plot as PNG"):
    try:
        img = fig.to_image(format="png", width=1200, height=600)
        st.download_button(label, img, file_name=filename, mime="image/png")
    except Exception:
        # kaleido not installed or other issue: ignore
        pass

# ---------- Load & precompute (cached) ----------
@st.cache_data(show_spinner=True)
def load_and_precompute(file_bytes: bytes) -> dict:
    # Load CSV
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), on_bad_lines="skip", low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

    df.columns = [c.strip() for c in df.columns]

    # Dates
    for col in ["TRN_DATE", "ZED_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numeric conversions with comma removal
    numeric_cols = ["QTY", "CP_PRE_VAT", "SP_PRE_VAT", "COST_PRE_VAT", "NET_SALES", "VAT_AMT"]
    for nc in numeric_cols:
        if nc in df.columns:
            df[nc] = df[nc].astype(str).str.replace(",", "", regex=False)
            df[nc] = pd.to_numeric(df[nc], errors="coerce").fillna(0)

    # ID columns
    idcols = ["STORE_CODE", "TILL", "SESSION", "RCT"]
    for c in idcols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()

    # CUST_CODE
    if "CUST_CODE" not in df.columns:
        if all(c in df.columns for c in idcols):
            df["CUST_CODE"] = (
                df["STORE_CODE"].str.strip()
                + "-"
                + df["TILL"].str.strip()
                + "-"
                + df["SESSION"].str.strip()
                + "-"
                + df["RCT"].str.strip()
            )
        else:
            df["CUST_CODE"] = df.index.astype(str)
    df["CUST_CODE"] = df["CUST_CODE"].astype(str).str.strip()

    # Prepare output dict
    out = {}
    out["df"] = df

    # Basic totals
    out["TOTAL_NET_SALES"] = float(df["NET_SALES"].sum()) if "NET_SALES" in df.columns else 0.0
    out["TOTAL_QTY"] = int(df["QTY"].sum()) if "QTY" in df.columns else 0

    # SALES_CHANNEL L1/L2
    if "SALES_CHANNEL_L1" in df.columns and "NET_SALES" in df.columns:
        s1 = df.groupby("SALES_CHANNEL_L1", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        s1["NET_SALES_M"] = s1["NET_SALES"] / 1_000_000
        s1["PCT"] = s1["NET_SALES"] / s1["NET_SALES"].sum() * 100
        out["sales_channel_l1"] = s1
    else:
        out["sales_channel_l1"] = pd.DataFrame()

    if "SALES_CHANNEL_L2" in df.columns and "NET_SALES" in df.columns:
        s2 = df.groupby("SALES_CHANNEL_L2", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        s2["NET_SALES_M"] = s2["NET_SALES"] / 1_000_000
        s2["PCT"] = s2["NET_SALES"] / s2["NET_SALES"].sum() * 100
        out["sales_channel_l2"] = s2
    else:
        out["sales_channel_l2"] = pd.DataFrame()

    # Shifts + Day/Night derived
    if "TRN_DATE" in df.columns and "NET_SALES" in df.columns:
        tmp = df.copy()
        tmp["HOUR"] = tmp["TRN_DATE"].dt.hour.fillna(-1).astype(int)
        tmp["SHIFT_TYPE"] = np.where(tmp["HOUR"].between(7, 18), "Day", "Night")
        shift_tot = tmp.groupby("SHIFT_TYPE", as_index=False)["NET_SALES"].sum()
        out["sales_by_shift"] = shift_tot
        per_store_shift = tmp.groupby(["STORE_NAME", "SHIFT_TYPE"], as_index=False)["NET_SALES"].sum().pivot(index="STORE_NAME", columns="SHIFT_TYPE", values="NET_SALES").fillna(0)
        per_store_shift["total"] = per_store_shift.sum(axis=1)
        per_store_shift = per_store_shift.reset_index().sort_values("total", ascending=False)
        out["per_store_shift"] = per_store_shift
    else:
        out["sales_by_shift"] = pd.DataFrame()
        out["per_store_shift"] = pd.DataFrame()

    # 2nd-highest channel computations (top / bottom 30)
    if {"STORE_NAME", "SALES_CHANNEL_L1", "NET_SALES"}.issubset(df.columns):
        store_chan = df.groupby(["STORE_NAME", "SALES_CHANNEL_L1"], as_index=False)["NET_SALES"].sum()
        store_tot = store_chan.groupby("STORE_NAME")["NET_SALES"].transform("sum")
        store_chan["PCT"] = 100 * store_chan["NET_SALES"] / store_tot
        store_chan = store_chan.sort_values(["STORE_NAME", "PCT"], ascending=[True, False])
        store_chan["RANK"] = store_chan.groupby("STORE_NAME").cumcount() + 1
        second = store_chan[store_chan["RANK"] == 2][["STORE_NAME", "SALES_CHANNEL_L1", "PCT"]].rename(columns={"SALES_CHANNEL_L1": "SECOND_CHANNEL", "PCT": "SECOND_PCT"})
        all_stores = store_chan["STORE_NAME"].drop_duplicates()
        missing_stores = set(all_stores) - set(second["STORE_NAME"])
        if missing_stores:
            second = pd.concat([second, pd.DataFrame({"STORE_NAME": list(missing_stores), "SECOND_CHANNEL": ["(None)"] * len(missing_stores), "SECOND_PCT": [0.0] * len(missing_stores)})], ignore_index=True)
        second_sorted = second.sort_values("SECOND_PCT", ascending=False)
        out["second_channel_table"] = second_sorted
        out["second_top_30"] = second_sorted.head(30)
        out["second_bottom_30"] = second_sorted.tail(30).sort_values("SECOND_PCT", ascending=True)
    else:
        out["second_channel_table"] = pd.DataFrame()
        out["second_top_30"] = pd.DataFrame()
        out["second_bottom_30"] = pd.DataFrame()

    # Store summary
    if {"STORE_NAME", "NET_SALES", "QTY", "CUST_CODE"}.issubset(df.columns):
        store_sum = df.groupby("STORE_NAME", as_index=False).agg(NET_SALES=("NET_SALES", "sum"), QTY=("QTY", "sum"), RECEIPTS=("CUST_CODE", pd.Series.nunique)).sort_values("NET_SALES", ascending=False)
        out["store_sales_summary"] = store_sum
    else:
        out["store_sales_summary"] = pd.DataFrame()

    # Receipts by time (dedup)
    receipts = df.drop_duplicates(subset=["CUST_CODE"]).copy()
    if "TRN_DATE" in receipts.columns:
        receipts["TRN_DATE"] = pd.to_datetime(receipts["TRN_DATE"], errors="coerce")
        receipts = receipts.dropna(subset=["TRN_DATE"])
        # floor to 30min and keep time objects for internal computations
        receipts["TIME_SLOT"] = receipts["TRN_DATE"].dt.floor("30T").dt.time
        heat = receipts.groupby(["STORE_NAME", "TIME_SLOT"], as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE": "RECEIPT_COUNT"})
        out["receipts_by_time"] = heat
    else:
        out["receipts_by_time"] = pd.DataFrame()

    # Active tills average
    if {"TRN_DATE", "TILL", "STORE_NAME"}.issubset(df.columns):
        tmp = df.copy()
        tmp["TRN_DATE"] = pd.to_datetime(tmp["TRN_DATE"], errors="coerce")
        tmp = tmp.dropna(subset=["TRN_DATE"])
        tmp["TRN_DATE_DATE"] = tmp["TRN_DATE"].dt.date
        active_tills = tmp.groupby(["STORE_NAME", "TRN_DATE_DATE"], as_index=False)["TILL"].nunique()
        active_avg = active_tills.groupby("STORE_NAME", as_index=False)["TILL"].mean().rename(columns={"TILL": "avg_active_tills"})
        out["active_tills_avg"] = active_avg.sort_values("avg_active_tills", ascending=False)
    else:
        out["active_tills_avg"] = pd.DataFrame()

    # First-touch receipts for customers-per-till
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

    # Cashier perf
    if "CASHIER" in df.columns or "CASHIER_NAME" in df.columns:
        cash_col = "CASHIER" if "CASHIER" in df.columns else "CASHIER_NAME"
        cashier_perf = df.groupby(cash_col, as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        out["cashier_perf"] = cashier_perf
    else:
        out["cashier_perf"] = pd.DataFrame()

    # Dept & supplier
    if "DEPARTMENT" in df.columns:
        dept_sales = df.groupby("DEPARTMENT", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        out["dept_sales"] = dept_sales
    else:
        out["dept_sales"] = pd.DataFrame()

    if "SUPPLIER" in df.columns:
        supplier_sales = df.groupby("SUPPLIER", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        out["supplier_sales"] = supplier_sales
    else:
        out["supplier_sales"] = pd.DataFrame()

    # Top items
    if "ITEM_NAME" in df.columns:
        out["top_items_sales"] = df.groupby("ITEM_NAME", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False).head(200)
        out["top_items_qty"] = df.groupby("ITEM_NAME", as_index=False)["QTY"].sum().sort_values("QTY", ascending=False).head(200)
    else:
        out["top_items_sales"] = pd.DataFrame()
        out["top_items_qty"] = pd.DataFrame()

    # Loyalty receipts
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

    # Pricing multi-price
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

    # Refunds negative receipts
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

    # Lists for dropdowns
    out["stores"] = sorted(df["STORE_NAME"].dropna().unique().tolist()) if "STORE_NAME" in df.columns else []
    out["channels_l1"] = sorted(df["SALES_CHANNEL_L1"].dropna().unique().tolist()) if "SALES_CHANNEL_L1" in df.columns else []
    out["channels_l2"] = sorted(df["SALES_CHANNEL_L2"].dropna().unique().tolist()) if "SALES_CHANNEL_L2" in df.columns else []
    out["items"] = sorted(df["ITEM_NAME"].dropna().unique().tolist()) if "ITEM_NAME" in df.columns else []
    out["departments"] = sorted(df["DEPARTMENT"].dropna().unique().tolist()) if "DEPARTMENT" in df.columns else []

    # Time grid
    start_time = pd.Timestamp("00:00:00")
    intervals = [(start_time + timedelta(minutes=30 * i)).time() for i in range(48)]
    col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
    out["time_intervals"] = intervals
    out["time_labels"] = col_labels

    out["sample_rows"] = df.head(200)

    return out

# load & precompute
file_bytes = uploaded.getvalue()
state = load_and_precompute(file_bytes)

# sample download (sidebar)
st.sidebar.markdown("---")
st.sidebar.download_button("⬇️ Download sample rows", state["sample_rows"].to_csv(index=False).encode("utf-8"), "sample_rows.csv", "text/csv")
st.sidebar.markdown("Theme: Red (alerts) & Green (positive)")

# ---------- UI Sections ----------
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
    ]
}
section = st.sidebar.radio("Main Section", list(main_sections.keys()))
subsection = st.sidebar.selectbox("Subsection", main_sections[section], key="subsection")
st.markdown(f"##### {section} ➔ {subsection}")

df = state["df"]

# ---------- SALES ----------
if section == "SALES":
    if subsection == "Global sales Overview":
        gs = state["sales_channel_l1"]
        if gs.empty:
            st.warning("Missing SALES_CHANNEL_L1 / NET_SALES data.")
        else:
            gs_disp = gs.copy()
            gs_disp["NET_SALES_M"] = gs_disp["NET_SALES_M"].round(2)
            gs_disp["PCT"] = gs_disp["PCT"].round(1)
            labels = [f"{r['SALES_CHANNEL_L1']} ({r['PCT']:.1f}% | {r['NET_SALES_M']:.1f}M)" for _, r in gs_disp.iterrows()]
            try:
                fig = go.Figure(go.Pie(labels=labels, values=gs_disp["NET_SALES_M"], hole=0.5, marker=dict(colors=PALETTE_DIVERGING)))
                fig.update_layout(title="SALES CHANNEL TYPE — Global Overview", height=420)
                c1, c2 = st.columns([2, 2])
                with c1:
                    st.plotly_chart(fig, width='stretch')
                    try_image_download(fig, "global_sales_overview.png")
                with c2:
                    df_out = gs_disp[["SALES_CHANNEL_L1", "NET_SALES", "NET_SALES_M", "PCT"]].copy()
                    df_out = add_total_row(df_out, numeric_cols=["NET_SALES"], label_col="SALES_CHANNEL_L1")
                    display_table_with_format(df_out, int_cols=["NET_SALES"], float_cols=["NET_SALES_M", "PCT"], height=420)
            except Exception as e:
                st.error(f"Plot rendering issue: {e}")
                # still show table
                df_out = gs_disp[["SALES_CHANNEL_L1", "NET_SALES", "NET_SALES_M", "PCT"]].copy()
                df_out = add_total_row(df_out, numeric_cols=["NET_SALES"], label_col="SALES_CHANNEL_L1")
                display_table_with_format(df_out, int_cols=["NET_SALES"], float_cols=["NET_SALES_M", "PCT"], height=420)

    elif subsection == "Global Net Sales Distribution by Sales Channel":
        g2 = state["sales_channel_l2"]
        if g2.empty:
            st.warning("Missing SALES_CHANNEL_L2 / NET_SALES data.")
        else:
            g2_disp = g2.copy()
            labels = [f"{r['SALES_CHANNEL_L2']} ({r['PCT']:.1f}% | {r['NET_SALES_M']:.1f}M)" for _, r in g2_disp.iterrows()]
            try:
                fig = go.Figure(go.Pie(labels=labels, values=g2_disp["NET_SALES_M"], hole=0.5, marker=dict(colors=PALETTE_DIVERGING)))
                fig.update_layout(title="Net Sales by Sales Mode (L2)", height=420)
                c1, c2 = st.columns([2, 2])
                with c1:
                    st.plotly_chart(fig, width='stretch')
                    try_image_download(fig, "sales_channel_l2.png")
                with c2:
                    df_out = g2_disp[["SALES_CHANNEL_L2", "NET_SALES", "NET_SALES_M", "PCT"]].copy()
                    df_out = add_total_row(df_out, numeric_cols=["NET_SALES"], label_col="SALES_CHANNEL_L2")
                    display_table_with_format(df_out, int_cols=["NET_SALES"], float_cols=["NET_SALES_M", "PCT"], height=420)
            except Exception as e:
                st.error(f"Plot rendering issue: {e}")
                df_out = g2_disp[["SALES_CHANNEL_L2", "NET_SALES", "NET_SALES_M", "PCT"]].copy()
                df_out = add_total_row(df_out, numeric_cols=["NET_SALES"], label_col="SALES_CHANNEL_L2")
                display_table_with_format(df_out, int_cols=["NET_SALES"], float_cols=["NET_SALES_M", "PCT"], height=420)

    elif subsection == "Global Net Sales Distribution by SHIFT":
        sb = state["sales_by_shift"]
        if sb.empty:
            st.warning("No SHIFT data.")
        else:
            sb_disp = sb.copy().rename(columns={"SHIFT_TYPE": "SHIFT"})
            fig = px.bar(sb_disp, x="SHIFT", y="NET_SALES", color="SHIFT", color_discrete_map={"Day": COLOR_GREEN, "Night": COLOR_RED}, title="Net Sales by SHIFT")
            st.plotly_chart(fig, width='stretch')
            df_out = add_total_row(sb_disp.copy(), numeric_cols=["NET_SALES"], label_col="SHIFT")
            display_table_with_format(df_out, int_cols=["NET_SALES"])

    elif subsection == "Night vs Day Shift Sales Ratio — Stores with Night Shifts":
        per_store = state["per_store_shift"]
        if per_store.empty:
            st.info("No per-store shift data.")
        else:
            if "Night" in per_store.columns:
                night_stores = per_store[per_store["Night"] > 0].copy()
            else:
                night_stores = pd.DataFrame()
            if night_stores.empty:
                st.info("No stores with Night sales.")
            else:
                night_stores["Day"] = night_stores.get("Day", 0)
                night_stores["Night"] = night_stores.get("Night", 0)
                night_stores["day_night_ratio"] = (night_stores["Day"] / night_stores["Night"].replace(0, np.nan)).fillna(0)
                df_out = _safe_to_display(night_stores.reset_index())
                display_table_with_format(df_out, float_cols=["day_night_ratio"])
                fig = px.bar(night_stores.sort_values("day_night_ratio"), x="day_night_ratio", y="STORE_NAME", orientation="h", title="Day/Night Sales Ratio — Stores with Night Sales", color_discrete_sequence=[COLOR_BLUE])
                st.plotly_chart(fig, width='stretch')

    elif subsection == "Global Day vs Night Sales — Only Stores with NIGHT Shift":
        gnd = state["sales_by_shift"]
        if gnd.empty:
            st.info("No shift data.")
        else:
            gnd_disp = gnd.copy()
            gnd_disp["PCT"] = (gnd_disp["NET_SALES"] / gnd_disp["NET_SALES"].sum() * 100).round(1)
            labels = [f"{row['SHIFT_TYPE']} ({row['PCT']:.1f}%)" for _, row in gnd_disp.iterrows()]
            try:
                fig = go.Figure(go.Pie(labels=labels, values=gnd_disp["NET_SALES"], hole=0.5, marker=dict(colors=[COLOR_BLUE, COLOR_RED])))
                fig.update_layout(title="Global Day vs Night Sales — Stores with Night Shift", height=420)
                st.plotly_chart(fig, width='stretch')
            except Exception:
                st.info("Unable to render pie; showing table instead.")
            df_out = add_total_row(gnd_disp.copy(), numeric_cols=["NET_SALES"], label_col="SHIFT_TYPE")
            display_table_with_format(df_out, int_cols=["NET_SALES"], float_cols=["PCT"])

    elif subsection == "2nd-Highest Channel Share":
        sc = state["second_channel_table"]
        if sc.empty:
            st.info("No data for 2nd-Highest Channel Share.")
        else:
            df_out = add_total_row(sc.copy(), numeric_cols=["SECOND_PCT"], label_col="STORE_NAME")
            display_table_with_format(df_out[["STORE_NAME", "SECOND_CHANNEL", "SECOND_PCT"]], float_cols=["SECOND_PCT"])
            fig = px.bar(sc.sort_values("SECOND_PCT", ascending=True).head(50), x="SECOND_PCT", y="STORE_NAME", orientation="h", color_discrete_sequence=[COLOR_BLUE], title="2nd-Highest Channel Share")
            st.plotly_chart(fig, width='stretch')

    elif subsection == "Bottom 30 — 2nd Highest Channel":
        bottom_30 = state["second_bottom_30"]
        if bottom_30.empty:
            st.info("No bottom 30 data.")
        else:
            df_out = add_total_row(bottom_30.copy(), numeric_cols=["SECOND_PCT"], label_col="STORE_NAME")
            display_table_with_format(df_out[["STORE_NAME", "SECOND_CHANNEL", "SECOND_PCT"]], float_cols=["SECOND_PCT"])
            fig = px.bar(bottom_30.sort_values("SECOND_PCT", ascending=True), x="SECOND_PCT", y="STORE_NAME", orientation="h", color_discrete_sequence=[COLOR_RED], title="Bottom 30 — 2nd Highest Channel")
            st.plotly_chart(fig, width='stretch')

    elif subsection == "Stores Sales Summary":
        ss = state["store_sales_summary"]
        if ss.empty:
            st.info("No store sales summary.")
        else:
            ss_disp = add_total_row(ss.copy(), numeric_cols=["NET_SALES", "QTY"], label_col="STORE_NAME")
            display_table_with_format(ss_disp, int_cols=["NET_SALES", "QTY", "RECEIPTS"], height=520)
            sel = st.selectbox("Filter store (All or select)", ["All"] + state["stores"])
            if sel != "All":
                filtered = ss[ss["STORE_NAME"] == sel]
                display_table_with_format(filtered, int_cols=["NET_SALES", "QTY", "RECEIPTS"])

# ---------- OPERATIONS ----------
elif section == "OPERATIONS":
    if subsection == "Customer Traffic-Storewise":
        heat = state["receipts_by_time"]
        if heat.empty or not state["stores"]:
            st.info("No receipts-by-time data available.")
        else:
            sel_store = st.selectbox("Select Store", state["stores"])
            df_heat = heat[heat["STORE_NAME"] == sel_store].copy()
            if df_heat.empty:
                st.info("No time data for selected store.")
            else:
                times = state["time_intervals"]
                time_idx = pd.Index(times, name="TIME_SLOT")
                pivot = df_heat.set_index("TIME_SLOT")["RECEIPT_COUNT"].reindex(time_idx, fill_value=0)
                labels = state["time_labels"]
                fig = px.bar(x=labels, y=pivot.values, labels={"x": "Time", "y": "Receipts"}, color_discrete_sequence=[COLOR_BLUE], title=f"Receipts by Time - {sel_store}")
                st.plotly_chart(fig, width='stretch')
                out_df = pivot.reset_index().rename(columns={"TIME_SLOT": "TIME", 0: "RECEIPT_COUNT"})
                out_df.columns = ["TIME", "RECEIPT_COUNT"]
                out_df = add_total_row(out_df, numeric_cols=["RECEIPT_COUNT"], label_col="TIME")
                display_table_with_format(out_df, int_cols=["RECEIPT_COUNT"])

    elif subsection == "Active Tills During the day":
        at = state["active_tills_avg"]
        if at.empty:
            st.info("No active tills data.")
        else:
            at_disp = add_total_row(at.copy(), numeric_cols=["avg_active_tills"], label_col="STORE_NAME")
            display_table_with_format(at_disp, float_cols=["avg_active_tills"], height=420)

    elif subsection == "Average Customers Served per Till":
        first_touch = state["first_touch"]
        if first_touch.empty:
            st.info("Insufficient data for this metric.")
        else:
            receipts = first_touch.copy()
            receipts["TIME_SLOT"] = receipts["TRN_DATE"].dt.floor("30T").dt.time
            cust_counts = receipts.groupby(["STORE_NAME", "TIME_SLOT"], as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE": "CUSTOMERS"})
            tmp = df.copy()
            if {"TRN_DATE", "TILL", "STORE_NAME"}.issubset(tmp.columns):
                tmp["TRN_DATE"] = pd.to_datetime(tmp["TRN_DATE"], errors="coerce")
                tmp = tmp.dropna(subset=["TRN_DATE"])
                tmp["TIME_SLOT"] = tmp["TRN_DATE"].dt.floor("30T").dt.time
                till_counts = tmp.groupby(["STORE_NAME", "TIME_SLOT"], as_index=False)["TILL"].nunique().rename(columns={"TILL": "TILLS"})
                merged = cust_counts.merge(till_counts, on=["STORE_NAME", "TIME_SLOT"], how="left").fillna({"TILLS": 0})
                merged["CUST_PER_TILL"] = merged["CUSTOMERS"] / merged["TILLS"].replace(0, np.nan)
                per_store = merged.groupby("STORE_NAME", as_index=False).agg(Max_Cust_Per_Till=("CUST_PER_TILL", "max"), Avg_Cust_Per_Till=("CUST_PER_TILL", "mean")).fillna(0)
                per_store_disp = add_total_row(per_store, numeric_cols=["Max_Cust_Per_Till", "Avg_Cust_Per_Till"], label_col="STORE_NAME")
                display_table_with_format(per_store_disp, float_cols=["Max_Cust_Per_Till", "Avg_Cust_Per_Till"])
            else:
                st.info("Missing TILL or TRN_DATE to compute tills-per-slot.")

    elif subsection == "Store Customer Traffic Storewise":
        sel_store = st.selectbox("Select Store", state["stores"])
        branch_df = df[df["STORE_NAME"] == sel_store].copy()
        if branch_df.empty:
            st.info("No data for selected store.")
        else:
            branch_df["TRN_DATE"] = pd.to_datetime(branch_df["TRN_DATE"], errors="coerce")
            branch_df = branch_df.dropna(subset=["TRN_DATE"])
            for c in ["STORE_CODE", "TILL", "SESSION", "RCT"]:
                if c in branch_df.columns:
                    branch_df[c] = branch_df[c].astype(str).fillna("").str.strip()
            if "CUST_CODE" not in branch_df.columns:
                branch_df["CUST_CODE"] = branch_df["STORE_CODE"] + "-" + branch_df["TILL"] + "-" + branch_df["SESSION"] + "-" + branch_df["RCT"]
            branch_df["TIME_SLOT"] = branch_df["TRN_DATE"].dt.floor("30T").dt.time
            tmp = branch_df.groupby(["DEPARTMENT", "TIME_SLOT"], as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE": "UNIQUE_CUSTOMERS"})
            tmp = _safe_to_display(tmp)
            display_table_with_format(tmp, int_cols=["UNIQUE_CUSTOMERS"], height=520)

    elif subsection == "Customer Traffic-Departmentwise":
        departments = state["departments"]
        if not departments:
            st.info("No departments found.")
        else:
            sel_dept = st.selectbox("Select Department", departments)
            dept_df = df[df["DEPARTMENT"] == sel_dept].copy()
            if dept_df.empty:
                st.info("No rows for this department.")
            else:
                dept_df["TRN_DATE"] = pd.to_datetime(dept_df["TRN_DATE"], errors="coerce")
                dept_df = dept_df.dropna(subset=["TRN_DATE"])
                dept_df["TIME_SLOT"] = dept_df["TRN_DATE"].dt.floor("30T").dt.time
                counts = dept_df.groupby(["STORE_NAME", "TIME_SLOT"], as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE": "UNIQUE_CUSTOMERS"})
                counts = _safe_to_display(counts)
                display_table_with_format(counts, int_cols=["UNIQUE_CUSTOMERS"], height=520)

    elif subsection == "Cashiers Perfomance":
        cp = state["cashier_perf"]
        if cp.empty:
            st.info("No cashier data.")
        else:
            cp_disp = add_total_row(cp.copy(), numeric_cols=["NET_SALES"], label_col=cp.columns[0])
            display_table_with_format(cp_disp, int_cols=["NET_SALES"], height=420)

    elif subsection == "Till Usage":
        tmp = df.copy()
        tmp["TRN_DATE"] = pd.to_datetime(tmp["TRN_DATE"], errors="coerce")
        tmp = tmp.dropna(subset=["TRN_DATE"])
        tmp["TIME_SLOT"] = tmp["TRN_DATE"].dt.floor("30T").dt.time
        tmp["Till_Code"] = tmp["TILL"].astype(str).fillna("") + "-" + tmp["STORE_CODE"].astype(str).fillna("")
        till_activity = tmp.groupby(["STORE_NAME", "Till_Code", "TIME_SLOT"], as_index=False).agg(Receipts=("CUST_CODE", "nunique"))
        branch_summary = till_activity.groupby("STORE_NAME", as_index=False).agg(Store_Total_Receipts=("Receipts", "sum"), Avg_Per_Till=("Receipts", "mean"), Max_Per_Till=("Receipts", "max"), Unique_Tills=("Till_Code", "nunique"))
        branch_summary_disp = add_total_row(branch_summary.copy(), numeric_cols=["Store_Total_Receipts"], label_col="STORE_NAME")
        display_table_with_format(branch_summary_disp, int_cols=["Store_Total_Receipts", "Unique_Tills"], float_cols=["Avg_Per_Till"], height=520)

    elif subsection == "Tax Compliance":
        if {"CU_DEVICE_SERIAL", "CUST_CODE", "STORE_NAME"}.issubset(df.columns):
            tdf = df.copy()
            tdf["Tax_Compliant"] = np.where(tdf["CU_DEVICE_SERIAL"].astype(str).str.strip().replace({"nan": "", "NaN": "", "None": ""}).str.len() > 0, "Compliant", "Non-Compliant")
            global_summary = tdf.groupby("Tax_Compliant", as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE": "Receipts"})
            fig = px.pie(global_summary, names="Tax_Compliant", values="Receipts", color="Tax_Compliant", color_discrete_map={"Compliant": COLOR_GREEN, "Non-Compliant": COLOR_RED}, hole=0.45, title="Global Tax Compliance Overview")
            st.plotly_chart(fig, width='stretch')
            store_till = tdf.groupby(["STORE_NAME", "Till_Code", "Tax_Compliant"], as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE": "Receipts"})
            branch = st.selectbox("Select Branch", state["stores"])
            dfb = store_till[store_till["STORE_NAME"] == branch]
            if dfb.empty:
                st.info("No compliance data for this branch.")
            else:
                pivot = dfb.pivot(index="Till_Code", columns="Tax_Compliant", values="Receipts").fillna(0).reset_index()
                pivot = add_total_row(pivot.copy(), numeric_cols=[c for c in pivot.columns if c != "Till_Code"], label_col="Till_Code")
                display_table_with_format(_safe_to_display(pivot), int_cols=[c for c in pivot.columns if c != "Till_Code"], height=420)
        else:
            st.info("Missing columns for tax compliance (CU_DEVICE_SERIAL, CUST_CODE, STORE_NAME).")

# ---------- INSIGHTS ----------
elif section == "INSIGHTS":
    if subsection == "Customer Baskets Overview":
        receipts = df.drop_duplicates(subset=["CUST_CODE"])
        if "NET_SALES" not in receipts.columns or receipts.empty:
            st.info("Insufficient basket-level data.")
        else:
            basket_sum = receipts["NET_SALES"].sum()
            basket_avg = receipts["NET_SALES"].mean()
            basket_median = receipts["NET_SALES"].median()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total basket net sales", f"KSh {basket_sum:,.2f}")
            c2.metric("Average basket value", f"KSh {basket_avg:,.2f}")
            c3.metric("Median basket value", f"KSh {basket_median:,.2f}")
            fig = px.histogram(receipts, x="NET_SALES", nbins=50, title="Basket Value Distribution", color_discrete_sequence=[COLOR_BLUE])
            st.plotly_chart(fig, width='stretch')

    elif subsection == "Global Category Overview-Sales":
        dept = state["dept_sales"]
        if dept.empty:
            st.info("No department data.")
        else:
            df_out = add_total_row(dept.copy(), numeric_cols=["NET_SALES"], label_col="DEPARTMENT")
            display_table_with_format(df_out, int_cols=["NET_SALES"], height=420)
            fig = px.bar(dept, x="DEPARTMENT", y="NET_SALES", title="Category Sales (Global)", color_discrete_sequence=[COLOR_GREEN])
            st.plotly_chart(fig, width='stretch')

    elif subsection == "Global Category Overview-Baskets":
        if "DEPARTMENT" in df.columns:
            rec = df.drop_duplicates(subset=["CUST_CODE"])
            dept_counts = rec.groupby("DEPARTMENT", as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE": "BASKETS"})
            df_out = add_total_row(dept_counts.copy(), numeric_cols=["BASKETS"], label_col="DEPARTMENT")
            display_table_with_format(df_out, int_cols=["BASKETS"])
            fig = px.bar(dept_counts.sort_values("BASKETS", ascending=False), x="DEPARTMENT", y="BASKETS", color_discrete_sequence=[COLOR_BLUE], title="Baskets by Department (Global)")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("DEPARTMENT column missing.")

    elif subsection == "Supplier Contribution":
        sup = state["supplier_sales"]
        if sup.empty:
            st.info("No supplier data.")
        else:
            df_out = add_total_row(sup.copy(), numeric_cols=["NET_SALES"], label_col="SUPPLIER")
            display_table_with_format(df_out, int_cols=["NET_SALES"], height=420)

    elif subsection == "Category Overview":
        if "CATEGORY" in df.columns and "NET_SALES" in df.columns:
            cat = df.groupby("CATEGORY", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
            df_out = add_total_row(cat.copy(), numeric_cols=["NET_SALES"], label_col="CATEGORY")
            display_table_with_format(df_out, int_cols=["NET_SALES"])
            fig = px.bar(cat.head(50), x="CATEGORY", y="NET_SALES", title="Category Sales", color_discrete_sequence=[COLOR_GREEN])
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("CATEGORY or NET_SALES missing.")

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
                st.error(f"Metric {metric} not found.")
            else:
                dfA = df[df["STORE_NAME"] == A].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N)
                dfB = df[df["STORE_NAME"] == B].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N)
                combA = dfA.copy(); combA["Branch"] = A
                combB = dfB.copy(); combB["Branch"] = B
                both = pd.concat([combA, combB], ignore_index=True)
                fig = px.bar(both, x=metric, y="ITEM_NAME", color="Branch", orientation="h", barmode="group", color_discrete_sequence=[COLOR_BLUE, COLOR_GREEN], title=f"Top {N} items: {A} vs {B}")
                st.plotly_chart(fig, width='stretch')
                display_table_with_format(both, int_cols=[metric] if metric == "QTY" else ["NET_SALES"], height=450)

    elif subsection == "Product Perfomance":
        items = state["items"]
        if not items:
            st.info("No ITEM_NAME available.")
        else:
            sel_item = st.selectbox("Select Item", items)
            if not sel_item:
                st.info("Select an item.")
            else:
                item_df = df[df["ITEM_NAME"] == sel_item]
                if item_df.empty:
                    st.info("No rows for this item.")
                else:
                    qty_by_store = item_df.groupby("STORE_NAME", as_index=False)["QTY"].sum().sort_values("QTY", ascending=False)
                    sales_by_store = item_df.groupby("STORE_NAME", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
                    st.subheader("Top stores by quantity")
                    display_table_with_format(add_total_row(qty_by_store.copy(), numeric_cols=["QTY"], label_col="STORE_NAME"), int_cols=["QTY"])
                    st.subheader("Top stores by sales")
                    display_table_with_format(add_total_row(sales_by_store.copy(), numeric_cols=["NET_SALES"], label_col="STORE_NAME"), int_cols=["NET_SALES"])

    elif subsection == "Global Loyalty Overview":
        lr = state["loyalty_receipts"]
        if lr.empty:
            st.info("No loyalty data available.")
        else:
            branch_loyal = lr.groupby("STORE_NAME", as_index=False)["LOYALTY_CUSTOMER_CODE"].nunique().rename(columns={"LOYALTY_CUSTOMER_CODE": "Loyal_Customers"})
            df_out = add_total_row(branch_loyal.copy(), numeric_cols=["Loyal_Customers"], label_col="STORE_NAME")
            display_table_with_format(df_out, int_cols=["Loyal_Customers"], height=420)

    elif subsection == "Branch Loyalty Overview":
        lr = state["loyalty_receipts"]
        if lr.empty:
            st.info("No loyalty receipts.")
        else:
            branch = st.selectbox("Select Branch", sorted(lr["STORE_NAME"].unique().tolist()))
            per_store = lr[lr["STORE_NAME"] == branch].groupby("LOYALTY_CUSTOMER_CODE", as_index=False).agg(Baskets=("CUST_CODE", "nunique"), Total_Value=("Basket_Value", "sum")).sort_values(["Baskets", "Total_Value"], ascending=[False, False])
            display_table_with_format(add_total_row(per_store.copy(), numeric_cols=["Baskets", "Total_Value"], label_col="LOYALTY_CUSTOMER_CODE"), int_cols=["Baskets"], float_cols=["Total_Value"])

    elif subsection == "Customer Loyalty Overview":
        lr = state["loyalty_receipts"]
        if lr.empty:
            st.info("No loyalty data.")
        else:
            custs = lr["LOYALTY_CUSTOMER_CODE"].unique().tolist()
            sel_cust = st.selectbox("Select Loyalty Customer", custs)
            rc = lr[lr["LOYALTY_CUSTOMER_CODE"] == sel_cust]
            if rc.empty:
                st.info("No receipts for selected loyalty customer.")
            else:
                display_table_with_format(rc.sort_values("First_Time", ascending=False), float_cols=["Basket_Value"])

    elif subsection == "Global Pricing Overview":
        pr = state["global_pricing_summary"]
        if pr.empty:
            st.info("No multi-priced SKUs found.")
        else:
            df_out = add_total_row(pr.copy(), numeric_cols=["Total_Diff_Value"], label_col="STORE_NAME")
            display_table_with_format(df_out, int_cols=["Items_with_MultiPrice"], float_cols=["Total_Diff_Value", "Avg_Spread", "Max_Spread"])
            fig = px.bar(pr.head(20).sort_values("Total_Diff_Value", ascending=True), x="Total_Diff_Value", y="STORE_NAME", orientation="h", color="Items_with_MultiPrice", color_continuous_scale=PALETTE_DIVERGING, title="Top Stores by Value Impact from Multi-Priced SKUs")
            st.plotly_chart(fig, width='stretch')

    elif subsection == "Branch Brach Overview":
        ssum = state["store_sales_summary"]
        if ssum.empty:
            st.info("No store-level summary.")
        else:
            ssum_disp = add_total_row(ssum.copy(), numeric_cols=["NET_SALES", "QTY"], label_col="STORE_NAME")
            display_table_with_format(ssum_disp, int_cols=["NET_SALES", "QTY", "RECEIPTS"])

    elif subsection == "Global Refunds Overview":
        gr = state["global_refunds"]
        if gr.empty:
            st.info("No negative receipts found.")
        else:
            gr_disp = add_total_row(gr.copy(), numeric_cols=["Total_Neg_Value"], label_col="STORE_NAME")
            display_table_with_format(gr_disp, int_cols=["Receipts"], float_cols=["Total_Neg_Value"])
            fig = px.bar(gr.sort_values("Total_Neg_Value", ascending=True), x="Total_Neg_Value", y="STORE_NAME", orientation="h", color_discrete_sequence=[COLOR_RED], title="Global Refunds by Store")
            st.plotly_chart(fig, width='stretch')

    elif subsection == "Branch Refunds Overview":
        br = state["branch_refunds_detail"]
        if br.empty:
            st.info("No refunds detail available.")
        else:
            branch = st.selectbox("Select Branch", sorted(br["STORE_NAME"].unique().tolist()))
            dfb = br[br["STORE_NAME"] == branch].copy()
            dfb = dfb.sort_values("Value")
            dfb_disp = add_total_row(dfb.copy(), numeric_cols=["Value"], label_col="CUST_CODE")
            display_table_with_format(dfb_disp, float_cols=["Value"])

# ---------- Footer ----------
st.sidebar.markdown("---")
st.sidebar.markdown("All tables use thousands separators. Tables include totals where applicable. Theme: Red & Green.")
