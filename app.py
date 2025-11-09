import io
import hashlib
from datetime import timedelta

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

#
# Superdeck Analytics Dashboard (fast navigation: precompute all summaries at startup)
#
# - Reads uploaded CSV into memory once and caches it (keyed by file hash).
# - Precomputes a set of summary tables & lightweight "receipts" view up front (cached).
# - UI subsections render using precomputed DataFrames/figures so switching is very fast.
# - Dropdowns and filters read from cached lists and apply to precomputed tables (fast).
#
# NOTE / CAVEATS:
# - This implementation reads the entire uploaded file into memory. For very large files
#   (multi-hundred-MB / ~1GB) ensure the host has sufficient RAM. If you need streaming
#   (no full-DataFrame in memory), tell me and I will convert the precomputation to a
#   streaming-aggregate version that never keeps the full DataFrame.
# - To allow uploads up to 1 GB, set Streamlit server config:
#     export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024
#   or add .streamlit/config.toml with server.maxUploadSize = 1024.
#

st.set_page_config(layout="wide", page_title="Superdeck Analytics Dashboard", initial_sidebar_state="expanded")

# === Styles ===
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
    unsafe_allow_html=True,
)


st.title("🦸 Superdeck Analytics Dashboard")
st.markdown("> Upload your sales CSV, precompute all analytics once, then navigate subsections quickly.")

# --- Sidebar: file upload ---
st.sidebar.header("Upload Data")
st.sidebar.markdown("Upload CSV (up to 1024 MB if server configured). App will precompute summaries at startup.")

uploaded = st.sidebar.file_uploader("Upload CSV (CSV file)", type="csv")
if uploaded is None:
    st.info("Please upload a dataset to proceed.")
    st.stop()


# ---------- Helpers & Caching ----------
def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


@st.cache_data(show_spinner=True)
def load_dataframe(file_bytes: bytes) -> pd.DataFrame:
    """
    Load the CSV bytes into a pandas DataFrame and do light cleaning:
    - Strip column names
    - Parse TRN_DATE and ZED_DATE if present
    - Convert numeric columns
    - Normalize ID columns and create CUST_CODE if missing
    The loaded DataFrame is returned and cached keyed by file_bytes.
    """
    # Read CSV from bytes (robust to BOM)
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), on_bad_lines="skip", low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Parse date columns (safe)
    for col in ["TRN_DATE", "ZED_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numeric conversions
    numeric_cols = ["QTY", "CP_PRE_VAT", "SP_PRE_VAT", "COST_PRE_VAT", "NET_SALES", "VAT_AMT"]
    for nc in numeric_cols:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors="coerce").fillna(0)

    # ID columns to str
    idcols = ["STORE_CODE", "TILL", "SESSION", "RCT"]
    for col in idcols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()

    # CUST_CODE creation if missing
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
            # If CUST_CODE and idcols are missing, create a synthetic receipt id when possible
            if "TRN_DATE" in df.columns and "STORE_NAME" in df.columns:
                # use row index fallback (not ideal but prevents errors)
                df["CUST_CODE"] = df.index.astype(str)
            else:
                df["CUST_CODE"] = df.index.astype(str)

    df["CUST_CODE"] = df["CUST_CODE"].astype(str).str.strip()

    return df


@st.cache_data(show_spinner=True)
def precompute_summaries(df: pd.DataFrame) -> dict:
    """
    Precompute the set of summary tables and small lookup lists that the UI requires.
    Returns a dict of DataFrames and metadata for fast rendering.
    Cached keyed by the DataFrame object identity (Streamlit handles cache invalidation).
    """
    out = {}

    # Basic totals
    out["TOTAL_NET_SALES"] = float(df["NET_SALES"].sum()) if "NET_SALES" in df.columns else 0.0
    out["TOTAL_QTY"] = int(df["QTY"].sum()) if "QTY" in df.columns else 0

    # Sales by L1 and L2 channels
    if "SALES_CHANNEL_L1" in df.columns and "NET_SALES" in df.columns:
        s1 = df.groupby("SALES_CHANNEL_L1", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        s1["NET_SALES_M"] = s1["NET_SALES"] / 1_000_000
        s1["PCT"] = s1["NET_SALES"] / s1["NET_SALES"].sum() * 100
        out["sales_channel_l1"] = s1
    else:
        out["sales_channel_l1"] = pd.DataFrame(columns=["SALES_CHANNEL_L1", "NET_SALES", "NET_SALES_M", "PCT"])

    if "SALES_CHANNEL_L2" in df.columns and "NET_SALES" in df.columns:
        s2 = df.groupby("SALES_CHANNEL_L2", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        s2["NET_SALES_M"] = s2["NET_SALES"] / 1_000_000
        s2["PCT"] = s2["NET_SALES"] / s2["NET_SALES"].sum() * 100
        out["sales_channel_l2"] = s2
    else:
        out["sales_channel_l2"] = pd.DataFrame(columns=["SALES_CHANNEL_L2", "NET_SALES", "NET_SALES_M", "PCT"])

    # Shifts / Day vs Night: determine hour from TRN_DATE if available
    if "TRN_DATE" in df.columns and "NET_SALES" in df.columns:
        df = df.copy()
        df["HOUR"] = df["TRN_DATE"].dt.hour.fillna(-1).astype(int)
        # Define day: 07:00-18:59, night: otherwise
        df["SHIFT_TYPE"] = np.where(df["HOUR"].between(7, 18), "Day", "Night")
        shift_tot = df.groupby("SHIFT_TYPE", as_index=False)["NET_SALES"].sum()
        out["sales_by_shift"] = shift_tot
        # Per-store day/night (stores that have night shift activity)
        per_store_shift = df.groupby(["STORE_NAME", "SHIFT_TYPE"], as_index=False)["NET_SALES"].sum().pivot(index="STORE_NAME", columns="SHIFT_TYPE", values="NET_SALES").fillna(0)
        per_store_shift["total"] = per_store_shift.sum(axis=1)
        per_store_shift = per_store_shift.reset_index().sort_values("total", ascending=False)
        out["per_store_shift"] = per_store_shift
    else:
        out["sales_by_shift"] = pd.DataFrame()
        out["per_store_shift"] = pd.DataFrame()

    # Store-level sales summary
    if "STORE_NAME" in df.columns and "NET_SALES" in df.columns:
        store_sum = (
            df.groupby("STORE_NAME", as_index=False)
            .agg(NET_SALES=("NET_SALES", "sum"), QTY=("QTY", "sum"), RECEIPTS=("CUST_CODE", pd.Series.nunique))
            .sort_values("NET_SALES", ascending=False)
        )
        out["store_sales_summary"] = store_sum
    else:
        out["store_sales_summary"] = pd.DataFrame()

    # Top items overall (by NET_SALES and QTY)
    top_n = 50
    if "ITEM_NAME" in df.columns:
        if "NET_SALES" in df.columns:
            top_sales = df.groupby("ITEM_NAME", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False).head(top_n)
        else:
            top_sales = pd.DataFrame(columns=["ITEM_NAME", "NET_SALES"])
        if "QTY" in df.columns:
            top_qty = df.groupby("ITEM_NAME", as_index=False)["QTY"].sum().sort_values("QTY", ascending=False).head(top_n)
        else:
            top_qty = pd.DataFrame(columns=["ITEM_NAME", "QTY"])
        out["top_items_sales"] = top_sales
        out["top_items_qty"] = top_qty
    else:
        out["top_items_sales"] = pd.DataFrame()
        out["top_items_qty"] = pd.DataFrame()

    # Receipts-level view: deduplicate by CUST_CODE to get receipts, then time slicing
    receipts = df.drop_duplicates(subset=["CUST_CODE"]).copy()
    if "TRN_DATE" in receipts.columns:
        receipts["TRN_DATE"] = pd.to_datetime(receipts["TRN_DATE"], errors="coerce")
        receipts = receipts.dropna(subset=["TRN_DATE"])
        # 30-minute slot
        receipts["TIME_SLOT"] = receipts["TRN_DATE"].dt.floor("30T").dt.time
        # Per-store time heat counts (count receipts per slot)
        if "STORE_NAME" in receipts.columns:
            heat = receipts.groupby(["STORE_NAME", "TIME_SLOT"], as_index=False)["CUST_CODE"].nunique()
            out["receipts_by_time"] = heat  # tall format; UI will pivot per store as needed
        else:
            out["receipts_by_time"] = pd.DataFrame()
    else:
        out["receipts_by_time"] = pd.DataFrame()

    # Active tills: unique tills per store per day (average / distribution)
    if "TRN_DATE" in df.columns and "TILL" in df.columns and "STORE_NAME" in df.columns:
        df["TRN_DATE_DATE"] = df["TRN_DATE"].dt.date
        active_tills = df.groupby(["STORE_NAME", "TRN_DATE_DATE"], as_index=False)["TILL"].nunique()
        active_tills_avg = active_tills.groupby("STORE_NAME", as_index=False)["TILL"].mean().rename(columns={"TILL": "avg_active_tills"})
        out["active_tills_avg"] = active_tills_avg.sort_values("avg_active_tills", ascending=False)
    else:
        out["active_tills_avg"] = pd.DataFrame()

    # Cashier / Till usage / tax compliance basics (if relevant columns exist)
    if "CASHIER" in df.columns or "CASHIER_NAME" in df.columns:
        ch_col = "CASHIER" if "CASHIER" in df.columns else "CASHIER_NAME"
        cashier_perf = df.groupby(ch_col, as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        out["cashier_perf"] = cashier_perf
    else:
        out["cashier_perf"] = pd.DataFrame()

    # Category & Supplier summaries
    if "DEPARTMENT" in df.columns and "NET_SALES" in df.columns:
        dept = df.groupby("DEPARTMENT", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        out["dept_sales"] = dept
    else:
        out["dept_sales"] = pd.DataFrame()

    if "SUPPLIER" in df.columns and "NET_SALES" in df.columns:
        supp = df.groupby("SUPPLIER", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        out["supplier_sales"] = supp
    else:
        out["supplier_sales"] = pd.DataFrame()

    # Branch list and other dropdown lists
    out["stores"] = sorted(df["STORE_NAME"].dropna().unique().tolist()) if "STORE_NAME" in df.columns else []
    out["channels_l1"] = sorted(df["SALES_CHANNEL_L1"].dropna().unique().tolist()) if "SALES_CHANNEL_L1" in df.columns else []
    out["channels_l2"] = sorted(df["SALES_CHANNEL_L2"].dropna().unique().tolist()) if "SALES_CHANNEL_L2" in df.columns else []
    out["items"] = sorted(df["ITEM_NAME"].dropna().unique().tolist()) if "ITEM_NAME" in df.columns else []

    # Time grid labels used by UI
    start_time = pd.Timestamp("00:00:00")
    intervals = [(start_time + timedelta(minutes=30 * i)).time() for i in range(48)]
    col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
    out["time_intervals"] = intervals
    out["time_labels"] = col_labels

    # Keep a small sample of raw rows for quick inspection (first 200)
    out["sample_rows"] = df.head(200)

    return out


# ---------- Load file and precompute ----------
# Use file bytes to key cache
file_bytes = uploaded.getvalue()
file_hash = _sha256_bytes(file_bytes)

with st.spinner("Loading and precomputing analytics (this runs once)..."):
    try:
        df = load_dataframe(file_bytes)
    except Exception as e:
        st.exception(e)
        st.stop()

    try:
        summaries = precompute_summaries(df)
    except Exception as e:
        st.exception(e)
        st.stop()

# Expose some top-level diagnostics
st.sidebar.markdown("---")
st.sidebar.write("Diagnostics")
st.sidebar.write(f"Rows: {df.shape[0]:,}")
st.sidebar.write(f"Columns: {df.shape[1]}")
st.sidebar.write(f"Cached file hash: {file_hash[:12]}")

# ---------- UI Sections and fast render using precomputed summaries ----------
main_sections = {
    "SALES": [
        "Global sales Overview",
        "Global Net Sales Distribution by Sales Channel",
        "Global Net Sales Distribution by SHIFT",
        "Night vs Day Shift Sales Ratio — Stores with Night Shifts",
        "Stores Sales Summary",
        "Top Items Overview",
    ],
    "OPERATIONS": [
        "Customer Traffic-Storewise",
        "Active Tills During the day",
        "Average Customers Served per Till",
        "Cashiers Performance",
    ],
    "INSIGHTS": [
        "Customer Baskets Overview",
        "Global Category Overview-Sales",
        "Supplier Contribution",
        "Branch Comparison",
    ],
}

section = st.sidebar.radio("Main Section", list(main_sections.keys()))
subsection = st.sidebar.selectbox("Subsection", main_sections[section], key="subsection")

st.markdown(f"##### {section} ➔ {subsection}")

# Helper: download helpers
def download_button_df(df_obj: pd.DataFrame, filename: str, label: str):
    st.download_button(label, df_obj.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")


# --- SALES ---
if section == "SALES":
    # 1. Global sales Overview
    if subsection == "Global sales Overview":
        gs = summaries["sales_channel_l1"]
        if gs.empty:
            st.warning("No SALES_CHANNEL_L1 / NET_SALES data available.")
        else:
            gs_display = gs.copy()
            labels = [f"{r['SALES_CHANNEL_L1']} ({r['PCT']:.1f}% | {r['NET_SALES_M']:.1f}M)" for _, r in gs_display.iterrows()]
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=gs_display["NET_SALES_M"],
                        hole=0.57,
                        marker=dict(colors=px.colors.qualitative.Plotly),
                        text=[f"{p:.1f}%" for p in gs_display["PCT"]],
                        textinfo="text",
                        sort=False,
                    )
                ]
            )
            fig.update_layout(title="SALES CHANNEL TYPE — Global Overview", height=420, margin=dict(t=60))
            c1, c2 = st.columns([2, 2])
            with c1:
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(gs_display, use_container_width=True)
                download_button_df(gs_display, "global_sales_overview.csv", "⬇️ Download Table")
                # image may require kaleido; optional
                try:
                    img_bytes = fig.to_image(format="png", width=1200, height=600)
                    st.download_button("⬇️ Download Plot as PNG", img_bytes, filename="global_sales_overview.png", mime="image/png")
                except Exception:
                    st.info("Image download disabled: install kaleido to enable.")

    elif subsection == "Global Net Sales Distribution by Sales Channel":
        g2 = summaries["sales_channel_l2"]
        if g2.empty:
            st.warning("No SALES_CHANNEL_L2 / NET_SALES data available.")
        else:
            labels = [f"{r['SALES_CHANNEL_L2']} ({r['PCT']:.1f}% | {r['NET_SALES_M']:.1f}M)" for _, r in g2.iterrows()]
            fig = go.Figure(go.Pie(labels=labels, values=g2["NET_SALES_M"], hole=0.58, marker=dict(colors=px.colors.qualitative.Vivid), text=[f"{p:.1f}%" for p in g2["PCT"]], textinfo="text"))
            fig.update_layout(title="Net Sales by Sales Mode (L2)", height=420, margin=dict(t=60))
            c1, c2 = st.columns([2, 2])
            with c1:
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(g2, use_container_width=True)
                download_button_df(g2, "sales_channel_l2.csv", "⬇️ Download Table")

    elif subsection == "Global Net Sales Distribution by SHIFT":
        sb = summaries["sales_by_shift"]
        if sb.empty:
            st.warning("No shift or NET_SALES data available.")
        else:
            fig = px.bar(sb, x="SHIFT_TYPE", y="NET_SALES", text="NET_SALES", color="SHIFT_TYPE", title="Net Sales by Shift", color_discrete_sequence=["#1f77b4", "#ff7f0e"])
            fig.update_layout(yaxis_title="Net Sales")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sb)
            download_button_df(sb, "sales_by_shift.csv", "⬇️ Download Table")

    elif subsection == "Night vs Day Shift Sales Ratio — Stores with Night Shifts":
        per_store_shift = summaries["per_store_shift"]
        if per_store_shift.empty:
            st.warning("No per-store shift data available.")
        else:
            # show only stores with Night > 0
            if "Night" in per_store_shift.columns:
                night_stores = per_store_shift[per_store_shift["Night"] > 0].copy()
            else:
                night_stores = pd.DataFrame()
            if night_stores.empty:
                st.info("No stores with night sales found.")
            else:
                night_stores["day_night_ratio"] = night_stores.get("Day", 0) / (night_stores.get("Night", 1))
                fig = px.bar(night_stores.sort_values("day_night_ratio"), x="day_night_ratio", y="STORE_NAME", orientation="h", title="Day/Night Sales Ratio — Stores with Night Sales")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(night_stores.reset_index(drop=True))
                download_button_df(night_stores.reset_index(), "night_stores_ratio.csv", "⬇️ Download Table")

    elif subsection == "Stores Sales Summary":
        store_sum = summaries["store_sales_summary"]
        if store_sum.empty:
            st.warning("No store sales summary available.")
        else:
            # Allow filtering
            store_filter = st.selectbox("Select store to inspect (All = show top N)", ["All"] + summaries["stores"], index=0)
            top_n = st.slider("Top N rows", 5, 200, 25)
            if store_filter == "All":
                st.dataframe(store_sum.head(top_n), use_container_width=True)
                download_button_df(store_sum, "store_sales_summary_all.csv", "⬇️ Download Full Store Summary")
            else:
                st.dataframe(store_sum[store_sum["STORE_NAME"] == store_filter], use_container_width=True)
                download_button_df(store_sum[store_sum["STORE_NAME"] == store_filter], f"store_{store_filter}_summary.csv", "⬇️ Download Store Summary")

    elif subsection == "Top Items Overview":
        # Top items by sales and qty (precomputed)
        st.write("Top items by Net Sales")
        st.dataframe(summaries["top_items_sales"].head(50))
        st.write("Top items by Quantity sold")
        st.dataframe(summaries["top_items_qty"].head(50))
        download_button_df(summaries["top_items_sales"], "top_items_by_sales.csv", "⬇️ Download Top Items (Sales)")
        download_button_df(summaries["top_items_qty"], "top_items_by_qty.csv", "⬇️ Download Top Items (Qty)")

# --- OPERATIONS ---
elif section == "OPERATIONS":
    if subsection == "Customer Traffic-Storewise":
        receipts_heat = summaries["receipts_by_time"]
        if receipts_heat.empty or not summaries["stores"]:
            st.warning("No receipts-by-time or store data available.")
        else:
            sel_store = st.selectbox("Select Store", summaries["stores"])
            # pivot for selected store
            df_heat = receipts_heat[receipts_heat["STORE_NAME"] == sel_store].copy()
            if df_heat.empty:
                st.info("No time data for selected store.")
            else:
                # Build full time index to ensure zeros
                intervals = summaries["time_intervals"]
                time_index = pd.Index(intervals, name="TIME_SLOT")
                pivot = df_heat.set_index("TIME_SLOT")["CUST_CODE"].reindex(time_index, fill_value=0)
                labels = summaries["time_labels"]
                fig = px.bar(x=labels, y=pivot.values, labels={"x": "Time", "y": "Receipts"}, title=f"Receipts by Time - {sel_store}", color_discrete_sequence=["#3192e1"])
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(pivot.reset_index().rename(columns={"TIME_SLOT": "time_slot", "CUST_CODE": "receipts"}))
                # download
                out_df = pivot.reset_index()
                download_button_df(out_df, f"customer_traffic_{sel_store}.csv", "⬇️ Download Table")

    elif subsection == "Active Tills During the day":
        at = summaries["active_tills_avg"]
        if at.empty:
            st.warning("No active tills data.")
        else:
            st.dataframe(at.head(100))
            download_button_df(at, "active_tills_avg.csv", "⬇️ Download Table")

    elif subsection == "Average Customers Served per Till":
        # compute on the fly using precomputed receipts / active tills if possible
        if "store_sales_summary" in summaries and not summaries["store_sales_summary"].empty and not summaries["active_tills_avg"].empty:
            merged = summaries["store_sales_summary"].merge(summaries["active_tills_avg"], on="STORE_NAME", how="left")
            merged["cust_per_till"] = merged["RECEIPTS"] / merged["avg_active_tills"].replace(0, np.nan)
            st.dataframe(merged.sort_values("cust_per_till", ascending=False).head(200))
            download_button_df(merged, "customers_per_till.csv", "⬇️ Download Table")
        else:
            st.info("Insufficient data for customers-per-till calculation.")

    elif subsection == "Cashiers Performance":
        cp = summaries["cashier_perf"]
        if cp.empty:
            st.info("No cashier data found (columns CASHIER or CASHIER_NAME missing).")
        else:
            st.dataframe(cp.head(200))
            download_button_df(cp, "cashier_performance.csv", "⬇️ Download Table")

# --- INSIGHTS ---
elif section == "INSIGHTS":
    if subsection == "Branch Comparison":
        stores = summaries["stores"]
        if len(stores) < 2:
            st.info("Not enough branches to compare.")
        else:
            selected_A = st.selectbox("Branch A", stores, key="bc_a")
            selected_B = st.selectbox("Branch B", stores, key="bc_b")
            metric = st.selectbox("Metric", ["QTY", "NET_SALES"], key="bc_metric")
            N = st.slider("Top N", 5, 50, 10, key="bc_n")
            # groupby per branch for chosen metric from the full df (fast because df is in memory)
            if metric not in df.columns:
                st.error(f"Metric {metric} not in data.")
            else:
                dfA = df[df["STORE_NAME"] == selected_A].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N)
                dfB = df[df["STORE_NAME"] == selected_B].groupby("ITEM_NAME", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(N)
                combA = dfA.copy()
                combA["Branch"] = selected_A
                combB = dfB.copy()
                combB["Branch"] = selected_B
                both = pd.concat([combA, combB], ignore_index=True)
                fig = px.bar(both, x=metric, y="ITEM_NAME", color="Branch", orientation="h", barmode="group", title=f"Top {N} items: {selected_A} vs {selected_B}", color_discrete_sequence=["#1f77b4", "#ff7f0e"], height=450)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(both)
                download_button_df(both, "branch_comparison.csv", "⬇️ Download Branch Comparison Table")

    elif subsection == "Global Category Overview-Sales":
        dept = summaries["dept_sales"]
        if dept.empty:
            st.info("No department/category sales data.")
        else:
            fig = px.bar(dept, x="DEPARTMENT", y="NET_SALES", title="Category Sales (Global)", text="NET_SALES")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dept)
            download_button_df(dept, "category_sales.csv", "⬇️ Download Table")

    elif subsection == "Supplier Contribution":
        sup = summaries["supplier_sales"]
        if sup.empty:
            st.info("No supplier data.")
        else:
            st.dataframe(sup.head(200))
            download_button_df(sup, "supplier_sales.csv", "⬇️ Download Table")

    elif subsection == "Customer Baskets Overview":
        # Quick basket-level metrics using receipts (deduplicated by CUST_CODE)
        receipts = df.drop_duplicates(subset=["CUST_CODE"])
        if receipts.empty or "NET_SALES" not in receipts.columns:
            st.info("Insufficient basket-level data")
        else:
            basket_sum = receipts["NET_SALES"].sum()
            basket_avg = receipts["NET_SALES"].mean()
            basket_median = receipts["NET_SALES"].median()
            st.write("Basket metrics (dedup by CUST_CODE):")
            st.metric("Total net sales (baskets)", f"{basket_sum:,.2f}")
            st.metric("Average basket value", f"{basket_avg:,.2f}")
            st.metric("Median basket value", f"{basket_median:,.2f}")
            # Distribution
            fig = px.histogram(receipts, x="NET_SALES", nbins=50, title="Basket Value Distribution")
            st.plotly_chart(fig, use_container_width=True)

# Footer / tips
st.sidebar.markdown("---")
st.sidebar.markdown("Sidebar auto-expands for easy selection. All tables and plots can be downloaded. If image download fails, install `kaleido` in the environment.")

# Quick inspection sample & download
st.sidebar.markdown("---")
st.sidebar.write("Quick sample of raw data")
st.sidebar.dataframe(summaries["sample_rows"].head(10))
st.sidebar.download_button("⬇️ Download sample CSV", summaries["sample_rows"].to_csv(index=False).encode("utf-8"), "sample_rows.csv", "text/csv")
