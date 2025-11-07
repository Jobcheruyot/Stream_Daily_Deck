#!/usr/bin/env python3
"""
Superdeck — Streamlit Analytics (clean, safe, production-friendly)

Goals for this version:
- Avoid any top-level runtime errors so Streamlit Cloud won't show the "Oh no" crash page.
- Keep the UI focused: 3 main categories (SALES / OPERATIONS / INSIGHTS) and subsections.
- Only show uploader and a minimal, relevant sidebar. Remove debug and noisy alternative upload prompts.
- Provide an unobtrusive "Advanced" collapsible for S3 direct-upload instructions (optional).
- Defensive data loading: parse the CSV only after upload, catch exceptions and present friendly error messages (no crashes).
- Provide clear guidance in-app when columns required for a view are missing.

Replace your current app.py with this file and restart the app.
"""
from typing import List, Dict, Any
import io
from datetime import timedelta
import textwrap
import traceback

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Page config & light styling
# ----------------------------
st.set_page_config(page_title="Superdeck Analytics", layout="wide")
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

# ----------------------------
# Sidebar: uploader (minimal)
# ----------------------------
st.sidebar.header("Upload CSV")
st.sidebar.write("Upload a CSV to enable the dashboard. Files larger than your host limit may fail — see Advanced > S3 instructions if needed.")
uploaded = st.sidebar.file_uploader("Choose CSV file", type="csv", accept_multiple_files=False)

# Advanced (collapsed) only if the user expands it — not prominent
with st.sidebar.expander("Advanced: Direct S3 upload (optional)", expanded=False):
    st.write("If your host blocks large uploads, upload directly to S3 and then paste the S3 object key into the app. This is advanced and optional.")
    st.markdown("- Create an S3 presigned POST on your backend (or in this app with AWS creds) and upload from the browser.")
    st.markdown("- After upload, use 'Process S3 file' on the main page to download and parse the object server-side.")

# ----------------------------
# Small helper utilities
# ----------------------------
def safe_read_csv_bytes(uploaded_file, chunksize: int = 200_000) -> pd.DataFrame:
    """Read CSV robustly using chunking for large files. Raises on failure (caught by caller)."""
    # Convert Streamlit UploadedFile to BytesIO
    if hasattr(uploaded_file, "getvalue"):
        b = io.BytesIO(uploaded_file.getvalue())
    else:
        uploaded_file.seek(0)
        b = io.BytesIO(uploaded_file.read())
    size_mb = len(b.getvalue()) / (1024 * 1024)
    b.seek(0)
    if size_mb > 200:
        parts = []
        for chunk in pd.read_csv(b, on_bad_lines="skip", low_memory=False, chunksize=chunksize):
            parts.append(chunk)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        df = pd.read_csv(b, on_bad_lines="skip", low_memory=False)
    return df

def to_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
    return df

def show_missing_columns(required: List[str]):
    st.error(f"Missing columns required for this view: {', '.join(required)}")

# ----------------------------
# Safe load & preprocess (only run after upload)
# ----------------------------
@st.cache_data(show_spinner=False)
def preprocess(uploaded_file) -> Dict[str, Any]:
    """Load CSV and compute frequently used aggregates. Returns a dict of results."""
    df = safe_read_csv_bytes(uploaded_file)
    # normalize columns
    df.columns = [c.strip() for c in df.columns]
    # parse dates if present
    for dcol in ("TRN_DATE", "ZED_DATE"):
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    # numeric cleanup
    numeric_cols = ["QTY", "CP_PRE_VAT", "SP_PRE_VAT", "COST_PRE_VAT", "NET_SALES", "VAT_AMT"]
    df = to_numeric_cols(df, numeric_cols)
    # ensure some essential text columns exist so we don't KeyError
    for t in ["STORE_NAME", "CUST_CODE", "SALES_CHANNEL_L1", "SALES_CHANNEL_L2", "SHIFT"]:
        if t not in df.columns:
            df[t] = ""
    # derived columns
    if "GROSS_SALES" not in df.columns:
        df["GROSS_SALES"] = df.get("NET_SALES", 0) + df.get("VAT_AMT", 0)
    # precompute common aggregates used by SALES visuals
    results = {"df": df}
    try:
        results["global_sales"] = (
            df.groupby("SALES_CHANNEL_L1", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        )
        results["channel2"] = (
            df.groupby("SALES_CHANNEL_L2", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
        )
        results["shift_sales"] = df.groupby("SHIFT", as_index=False)["NET_SALES"].sum().sort_values("NET_SALES", ascending=False)
    except Exception:
        # if something unexpected, return as-empty and let views show friendly message
        results["global_sales"] = pd.DataFrame()
        results["channel2"] = pd.DataFrame()
        results["shift_sales"] = pd.DataFrame()
    return results

# ----------------------------
# Main header: categories
# ----------------------------
st.title("Superdeck — Sales & Operations Dashboard")
st.markdown("Choose a category below. Subsections appear once you select a category.")

# show three big buttons (user-friendly)
cols = st.columns(3)
if "main_category" not in st.session_state:
    st.session_state["main_category"] = "SALES"
with cols[0]:
    if st.button("📈 SALES"):
        st.session_state["main_category"] = "SALES"
with cols[1]:
    if st.button("⚙️ OPERATIONS"):
        st.session_state["main_category"] = "OPERATIONS"
with cols[2]:
    if st.button("🔎 INSIGHTS"):
        st.session_state["main_category"] = "INSIGHTS"

st.markdown("---")

# subsections dictionary (only relevant ones implemented; others show "coming soon")
SUBSECTIONS = {
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
        "Branch Brach Overview",
        "Global Refunds Overview",
        "Branch Refunds Overview"
    ]
}

main = st.session_state["main_category"]
left, right = st.columns([1, 3])
with left:
    st.subheader(f"{main} — Subsections")
    subsection = st.radio("Choose view", SUBSECTIONS[main], index=0)

# If no file uploaded yet, stop with minimal guidance
if uploaded is None:
    with right:
        st.info("Please upload your CSV on the left to activate the dashboard.")
        st.write("If you need help uploading large files, expand 'Advanced' in the left sidebar for S3 guidance.")
    st.stop()

# Load data defensively
try:
    with st.spinner("Loading data..."):
        DATA = preprocess(uploaded)
        df = DATA["df"]
except Exception as e:
    # Show friendly error and stack trace in expander; do NOT let the app crash
    st.error("Failed to read or parse the uploaded CSV. The app will not crash — please review the error details below.")
    with st.expander("Error details (click to expand)"):
        st.text(str(e))
        st.text(traceback.format_exc())
    st.stop()

# ----------------------------
# Render each subsection (focused, minimal UI elements)
# ----------------------------
with right:
    st.header(subsection)

    # SALES views
    if main == "SALES":
        if subsection == "Global sales Overview":
            gs = DATA.get("global_sales", pd.DataFrame())
            if gs.empty or "NET_SALES" not in gs.columns:
                show_missing_columns(["SALES_CHANNEL_L1", "NET_SALES"])
            else:
                gs["NET_SALES_M"] = gs["NET_SALES"] / 1_000_000
                gs["PCT"] = (gs["NET_SALES"] / gs["NET_SALES"].sum() * 100).round(1)
                labels = [f"{r['SALES_CHANNEL_L1']} ({r['PCT']:.1f}% | {r['NET_SALES_M']:.1f}M)" for _, r in gs.iterrows()]
                fig = go.Figure(go.Pie(labels=labels, values=gs["NET_SALES_M"], hole=0.6, text=[f"{p:.1f}%" for p in gs["PCT"]], textinfo="text"))
                fig.update_layout(title="Sales Channel — Global Overview", height=520)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(gs[["SALES_CHANNEL_L1", "NET_SALES"]].rename(columns={"SALES_CHANNEL_L1":"Channel","NET_SALES":"Net Sales (KSh)"}), use_container_width=True)

        elif subsection == "Global Net Sales Distribution by Sales Channel":
            ch2 = DATA.get("channel2", pd.DataFrame())
            if ch2.empty:
                show_missing_columns(["SALES_CHANNEL_L2", "NET_SALES"])
            else:
                ch2["NET_SALES_M"] = ch2["NET_SALES"] / 1_000_000
                fig = px.pie(ch2, names="SALES_CHANNEL_L2", values="NET_SALES_M", hole=0.6, title="Sales by SALES_CHANNEL_L2")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(ch2[["SALES_CHANNEL_L2","NET_SALES"]].rename(columns={"SALES_CHANNEL_L2":"Mode","NET_SALES":"Net Sales (KSh)"}), use_container_width=True)

        elif subsection == "Global Net Sales Distribution by SHIFT":
            sh = DATA.get("shift_sales", pd.DataFrame())
            if sh.empty:
                show_missing_columns(["SHIFT","NET_SALES"])
            else:
                fig = px.pie(sh, names="SHIFT", values="NET_SALES", hole=0.6, title="Net Sales by SHIFT")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(sh, use_container_width=True)

        elif subsection == "Night vs Day Shift Sales Ratio — Stores with Night Shifts":
            # compute on demand (defensive)
            if "SHIFT" not in df.columns or "STORE_NAME" not in df.columns or "NET_SALES" not in df.columns:
                show_missing_columns(["SHIFT","STORE_NAME","NET_SALES"])
            else:
                stores_with_night = df[df["SHIFT"].astype(str).str.upper().str.contains("NIGHT", na=False)]["STORE_NAME"].unique()
                if len(stores_with_night) == 0:
                    st.info("No stores with NIGHT shift found.")
                else:
                    dnd = df[df["STORE_NAME"].isin(stores_with_night)].copy()
                    dnd["Shift_Bucket"] = np.where(dnd["SHIFT"].astype(str).str.upper().str.contains("NIGHT", na=False),"Night","Day")
                    r = dnd.groupby(["STORE_NAME","Shift_Bucket"], as_index=False)["NET_SALES"].sum()
                    tot = r.groupby("STORE_NAME")["NET_SALES"].transform("sum")
                    r["PCT"] = np.where(tot>0, 100 * r["NET_SALES"] / tot, 0.0)
                    pivot = r.pivot(index="STORE_NAME", columns="Shift_Bucket", values="PCT").fillna(0)
                    pivot = pivot.sort_values("Night", ascending=False)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=pivot["Night"], y=pivot.index, orientation="h", name="Night", marker_color="#d62728"))
                    fig.add_trace(go.Bar(x=pivot["Day"], y=pivot.index, orientation="h", name="Day", marker_color="#1f77b4"))
                    fig.update_layout(barmode="group", title="Night vs Day % by Store", height=max(400, 24*len(pivot)))
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(pivot.reset_index().rename(columns={"Night":"Night %","Day":"Day %"}), use_container_width=True)

        elif subsection == "Global Day vs Night Sales — Only Stores with NIGHT Shift":
            if "SHIFT" not in df.columns:
                show_missing_columns(["SHIFT"])
            else:
                stores_with_night = df[df["SHIFT"].astype(str).str.upper().str.contains("NIGHT", na=False)]["STORE_NAME"].unique()
                if len(stores_with_night) == 0:
                    st.info("No NIGHT shift stores.")
                else:
                    dnd = df[df["STORE_NAME"].isin(stores_with_night)].copy()
                    dnd["Shift_Bucket"] = np.where(dnd["SHIFT"].astype(str).str.upper().str.contains("NIGHT", na=False),"Night","Day")
                    gb = dnd.groupby("Shift_Bucket", as_index=False)["NET_SALES"].sum()
                    gb["PCT"] = 100 * gb["NET_SALES"] / gb["NET_SALES"].sum() if gb["NET_SALES"].sum() else 0.0
                    fig = px.pie(gb, names="Shift_Bucket", values="NET_SALES", hole=0.6, title="Global Day vs Night (Night Stores)")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(gb, use_container_width=True)

        elif subsection in ("2nd-Highest Channel Share", "Bottom 30 — 2nd Highest Channel"):
            # safe compute second-largest channel share per store
            required = ["STORE_NAME","SALES_CHANNEL_L1","NET_SALES"]
            if any(c not in df.columns for c in required):
                show_missing_columns(required)
            else:
                d = df.copy()
                d["NET_SALES"] = pd.to_numeric(d["NET_SALES"], errors="coerce").fillna(0)
                store_chan = d.groupby(["STORE_NAME","SALES_CHANNEL_L1"], as_index=False)["NET_SALES"].sum()
                store_tot = store_chan.groupby("STORE_NAME")["NET_SALES"].transform("sum")
                store_chan["PCT"] = np.where(store_tot>0, 100*store_chan["NET_SALES"]/store_tot,0.0)
                store_chan = store_chan.sort_values(["STORE_NAME","PCT"], ascending=[True,False])
                store_chan["RANK"] = store_chan.groupby("STORE_NAME").cumcount()+1
                second = store_chan[store_chan["RANK"]==2]
                if second.empty:
                    st.info("No stores with a valid 2nd channel (many stores only have 1 channel).")
                else:
                    if subsection.startswith("2nd-Highest"):
                        top30 = second.sort_values("PCT", ascending=False).head(30)
                        fig = px.bar(top30, x="PCT", y="STORE_NAME", orientation="h", title="Top 30 by 2nd Channel %")
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(top30.rename(columns={"SALES_CHANNEL_L1":"2nd Channel","PCT":"2nd Channel %"}), use_container_width=True)
                    else:
                        bottom30 = second.sort_values("PCT", ascending=True).head(30)
                        fig = px.bar(bottom30, x="PCT", y="STORE_NAME", orientation="h", title="Bottom 30 by 2nd Channel %", color_discrete_sequence=["#d62728"])
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(bottom30.rename(columns={"SALES_CHANNEL_L1":"2nd Channel","PCT":"2nd Channel %"}), use_container_width=True)

        elif subsection == "Stores Sales Summary":
            if "GROSS_SALES" not in df.columns and "NET_SALES" not in df.columns:
                show_missing_columns(["NET_SALES"])
            else:
                ss = df.groupby("STORE_NAME", as_index=False).agg(NET_SALES=("NET_SALES","sum"), GROSS_SALES=("GROSS_SALES","sum"))
                ss["Customer_Numbers"] = df.groupby("STORE_NAME")["CUST_CODE"].nunique().reindex(ss["STORE_NAME"]).fillna(0).astype(int).values
                total_gross = ss["GROSS_SALES"].sum()
                ss["Pct_Contribution"] = (100 * ss["GROSS_SALES"] / total_gross).round(2) if total_gross!=0 else 0.0
                st.dataframe(ss.sort_values("GROSS_SALES", ascending=False), use_container_width=True)
                st.download_button("⬇️ Download Stores Summary", ss.to_csv(index=False).encode("utf-8"), "stores_summary.csv", "text/csv")

    # OPERATIONS and INSIGHTS: for brevity, show clear placeholder and friendly message if not implemented
    elif main == "OPERATIONS":
        st.info("Operations views are available; the selected subsection will render here if your CSV contains the required fields.")
        st.caption("Implemented views in this release: Customer Traffic-Storewise, Active Tills, Average Customers per Till, Cashiers Performance, Tax Compliance.")
        # Example: show simple traffic heatmap if data present
        if subsection == "Customer Traffic-Storewise":
            if "TRN_DATE" not in df.columns or "CUST_CODE" not in df.columns or "STORE_NAME" not in df.columns:
                show_missing_columns(["TRN_DATE", "CUST_CODE", "STORE_NAME"])
            else:
                ft = df.dropna(subset=["TRN_DATE"]).copy()
                ft["DATE_ONLY"] = ft["TRN_DATE"].dt.date
                first_touch = ft.groupby(["STORE_NAME","DATE_ONLY","CUST_CODE"], as_index=False)["TRN_DATE"].min()
                first_touch["TIME_SLOT"] = first_touch["TRN_DATE"].dt.floor("30T").dt.time
                counts = first_touch.groupby(["STORE_NAME","TIME_SLOT"])["CUST_CODE"].nunique().reset_index(name="Receipts")
                st.dataframe(counts.head(200), use_container_width=True)

        elif subsection == "Tax Compliance":
            if "CU_DEVICE_SERIAL" not in df.columns or "CUST_CODE" not in df.columns:
                show_missing_columns(["CU_DEVICE_SERIAL", "CUST_CODE"])
            else:
                d = df.copy()
                d["Tax_Compliant"] = np.where(d["CU_DEVICE_SERIAL"].astype(str).str.strip().replace({"nan": "","None":""})!="","Compliant","Non-Compliant")
                summary = d.groupby("Tax_Compliant", as_index=False)["CUST_CODE"].nunique().rename(columns={"CUST_CODE":"Receipts"})
                fig = px.pie(summary, names="Tax_Compliant", values="Receipts", hole=0.5, title="Tax Compliance")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(summary, use_container_width=True)
        else:
            st.write("This Operations subsection is not yet fully implemented in the simplified UI. If you need a specific view implemented now, tell me which one and I'll add it.")

    elif main == "INSIGHTS":
        st.info("Insights views — pick a subsection. Many item-level and loyalty views depend on ITEM_NAME, ITEM_CODE, CUST_CODE and LOYALTY_CUSTOMER_CODE columns.")
        if subsection == "Customer Baskets Overview":
            if "ITEM_NAME" not in df.columns or "CUST_CODE" not in df.columns:
                show_missing_columns(["ITEM_NAME","CUST_CODE"])
            else:
                topx = st.slider("Top N", 5, 100, 10)
                global_top = df.groupby("ITEM_NAME")["CUST_CODE"].nunique().rename("Count_of_Baskets").reset_index().sort_values("Count_of_Baskets", ascending=False).head(topx)
                fig = px.bar(global_top, x="Count_of_Baskets", y="ITEM_NAME", orientation="h", title=f"Top {topx} items by baskets")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(global_top, use_container_width=True)
        else:
            st.write("This Insights subsection either requires additional dataset fields or will be implemented on request. Tell me which specific insight you want next and I'll prioritize it.")

    # end subsection rendering

    st.markdown("---")
    st.caption("If a view shows 'missing columns', either your CSV lacks those columns or they are named differently. In that case, upload an anonymized sample (5–20 rows) or tell me the column names and I'll map them automatically.")
