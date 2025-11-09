# app.py
# Superdeck / DailyDeck – Notebook-aligned Streamlit App
# - Uses one CSV uploader
# - Computes metrics once (cached)
# - Provides sections/subsections from the notebook TOC
# - Hides starred sections
# - Each subsection has deterministic, defensive logic

from datetime import timedelta, time as dtime
import io
import sys
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# Page config & theme
# =============================================================================
st.set_page_config(
    page_title="Superdeck Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊",
)

# Color palette
COLOR_BLUE = "#2563eb"
COLOR_ORANGE = "#fb923c"
COLOR_GREEN = "#22c55e"
COLOR_RED = "#ef4444"
PALETTE10 = [
    COLOR_BLUE,
    COLOR_ORANGE,
    COLOR_GREEN,
    COLOR_RED,
    "#6366f1",
    "#a855f7",
    "#ec4899",
    "#6b7280",
    "#22d3ee",
    "#facc15",
]


# =============================================================================
# Utility helpers
# =============================================================================
def _safe_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert problematic types (e.g. datetime.time) to strings for Streamlit."""
    if df is None or df.empty:
        return df
    df2 = df.copy()
    for c in df2.columns:
        if df2[c].dtype == object:
            sample = df2[c].dropna().head(20)
            if any(isinstance(v, dtime) for v in sample):
                df2[c] = df2[c].map(
                    lambda v: v.strftime("%H:%M") if isinstance(v, dtime) else v
                )
    return df2


def fmt_int(s: pd.Series) -> pd.Series:
    return s.map(lambda v: f"{int(v):,}" if pd.notna(v) else "")


def fmt_float(s: pd.Series, d: int = 2) -> pd.Series:
    fmt = "{:,.%df}" % d
    return s.map(lambda v: fmt.format(float(v)) if pd.notna(v) else "")


def add_total_row(df: pd.DataFrame, numeric_cols, label_col=None, label="Total"):
    if df is None or df.empty:
        return df
    total_row = {c: "" for c in df.columns}
    for c in numeric_cols:
        if c in df.columns:
            try:
                total_row[c] = df[c].sum()
            except Exception:
                total_row[c] = ""
    if label_col and label_col in df.columns:
        total_row[label_col] = label
    else:
        total_row[df.columns[0]] = label
    return pd.concat([pd.DataFrame([total_row]), df], ignore_index=True)


def show_table(
    df: pd.DataFrame,
    int_cols=None,
    float_cols=None,
    height: int = 380,
    download_name: str | None = None,
):
    if df is None or df.empty:
        st.info("No data for this view.")
        return
    try:
        df_out = df.copy()
        int_cols = int_cols or []
        float_cols = float_cols or []
        for c in int_cols:
            if c in df_out.columns:
                df_out[c] = fmt_int(df_out[c])
        for c in float_cols:
            if c in df_out.columns:
                df_out[c] = fmt_float(df_out[c])
        df_out = _safe_display_df(df_out)
        st.dataframe(df_out, use_container_width=True, height=height)
        if download_name:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                csv_bytes,
                file_name=download_name,
                mime="text/csv",
                use_container_width=True,
            )
    except Exception:
        st.error("Error rendering table; see logs.")
        traceback.print_exc(file=sys.stdout)


def pie(labels, values, title, text=None):
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            text=text,
            textinfo="text" if text is not None else "percent",
            marker=dict(colors=PALETTE10),
        )
    )
    fig.update_layout(title=title, legend_title_text="", height=520)
    st.plotly_chart(fig, use_container_width=True)


def bar(df, x, y, title, orientation="v"):
    if df is None or df.empty:
        st.info("No chart data.")
        return
    fig = px.bar(
        df,
        x=x,
        y=y,
        orientation=orientation,
        color_discrete_sequence=PALETTE10,
        title=title,
    )
    st.plotly_chart(fig, use_container_width=True)


def line(df, x, y, title):
    if df is None or df.empty:
        st.info("No chart data.")
        return
    fig = px.line(
        df, x=x, y=y, color_discrete_sequence=PALETTE10, title=title
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Data loading & core precomputations
# =============================================================================
@st.cache_data(show_spinner=True)
def load_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(
        io.BytesIO(file_bytes),
        on_bad_lines="skip",
        low_memory=False,
    )

    # Strip column names
    df.columns = [c.strip() for c in df.columns]

    # Parse dates
    for c in ["TRN_DATE", "ZED_DATE"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Clean numerics commonly used in notebook
    num_candidates = [
        "QTY",
        "CP_PRE_VAT",
        "SP_PRE_VAT",
        "COST_PRE_VAT",
        "NET_SALES",
        "VAT_AMT",
    ]
    for c in num_candidates:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Normalize some IDs/text
    for c in ["STORE_CODE", "STORE_NAME", "TILL", "SESSION", "RCT"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Build CUST_CODE if missing (store-till-session-rct)
    if "CUST_CODE" not in df.columns:
        if all(c in df.columns for c in ["STORE_CODE", "TILL", "SESSION", "RCT"]):
            df["CUST_CODE"] = (
                df["STORE_CODE"]
                + "-"
                + df["TILL"]
                + "-"
                + df["SESSION"]
                + "-"
                + df["RCT"]
            )
        else:
            df["CUST_CODE"] = df.index.astype(str)

    return df


@st.cache_data(show_spinner=False)
def compute_metrics(df: pd.DataFrame) -> dict:
    m = {}
    # Common existence helpers
    has = lambda *cols: all(c in df.columns for c in cols)

    # ---- SALES ----
    # Global sales overview by L1
    if has("SALES_CHANNEL_L1", "NET_SALES"):
        t = (
            df.groupby("SALES_CHANNEL_L1", as_index=False)["NET_SALES"]
            .sum()
            .sort_values("NET_SALES", ascending=False)
        )
        total = t["NET_SALES"].sum()
        t["NET_SALES_M"] = t["NET_SALES"] / 1_000_000
        t["PCT"] = np.where(total > 0, t["NET_SALES"] / total * 100, 0.0)
        m["sales_l1"] = t
    else:
        m["sales_l1"] = pd.DataFrame()

    # Global Net Sales by L2 (or use L1 if L2 missing)
    if has("SALES_CHANNEL_L2", "NET_SALES"):
        t = (
            df.groupby("SALES_CHANNEL_L2", as_index=False)["NET_SALES"]
            .sum()
            .sort_values("NET_SALES", ascending=False)
        )
        total = t["NET_SALES"].sum()
        t["PCT"] = np.where(total > 0, t["NET_SALES"] / total * 100, 0.0)
        m["sales_l2"] = t
    else:
        m["sales_l2"] = pd.DataFrame()

    # Shift (Day/Night) using TRN_DATE hour if SHIFT not present
    if has("TRN_DATE", "NET_SALES"):
        tmp = df.copy()
        tmp["HOUR"] = tmp["TRN_DATE"].dt.hour
        tmp["SHIFT"] = np.where(
            tmp["HOUR"].between(7, 18), "DAY", "NIGHT"
        )
        m["df_shift"] = tmp

        # Global by shift
        s = (
            tmp.groupby("SHIFT", as_index=False)["NET_SALES"]
            .sum()
            .sort_values("NET_SALES", ascending=False)
        )
        tot = s["NET_SALES"].sum()
        s["PCT"] = np.where(tot > 0, s["NET_SALES"] / tot * 100, 0.0)
        m["sales_shift"] = s

        # Per store shift breakdown
        if has("STORE_NAME"):
            per = (
                tmp.groupby(["STORE_NAME", "SHIFT"], as_index=False)["NET_SALES"]
                .sum()
            )
            m["store_shift"] = per
        else:
            m["store_shift"] = pd.DataFrame()
    else:
        m["df_shift"] = pd.DataFrame()
        m["sales_shift"] = pd.DataFrame()
        m["store_shift"] = pd.DataFrame()

    # 2nd highest channel share per store (L1)
    if has("STORE_NAME", "SALES_CHANNEL_L1", "NET_SALES"):
        g = (
            df.groupby(
                ["STORE_NAME", "SALES_CHANNEL_L1"], as_index=False
            )["NET_SALES"]
            .sum()
        )
        g["STORE_TOTAL"] = g.groupby("STORE_NAME")["NET_SALES"].transform(
            "sum"
        )
        g["PCT"] = np.where(
            g["STORE_TOTAL"] > 0, g["NET_SALES"] / g["STORE_TOTAL"] * 100, 0
        )
        g = g.sort_values(
            ["STORE_NAME", "PCT"], ascending=[True, False]
        )
        g["RANK"] = g.groupby("STORE_NAME")["PCT"].rank(
            "first", ascending=False
        )
        second = g[g["RANK"] == 2][
            ["STORE_NAME", "SALES_CHANNEL_L1", "PCT"]
        ].rename(
            columns={
                "SALES_CHANNEL_L1": "2ND_CHANNEL",
                "PCT": "2ND_PCT",
            }
        )
        second = second.sort_values("2ND_PCT", ascending=False)
        m["second_channel"] = second
        m["second_channel_bottom_30"] = second.tail(30).sort_values(
            "2ND_PCT"
        )
    else:
        m["second_channel"] = pd.DataFrame()
        m["second_channel_bottom_30"] = pd.DataFrame()

    # Store sales summary
    if has("STORE_NAME", "NET_SALES"):
        agg = {
            "NET_SALES": ("NET_SALES", "sum"),
        }
        if "QTY" in df.columns:
            agg["QTY"] = ("QTY", "sum")
        if "CUST_CODE" in df.columns:
            agg["RECEIPTS"] = ("CUST_CODE", pd.Series.nunique)
        t = (
            df.groupby("STORE_NAME", as_index=False)
            .agg(**agg)
            .sort_values("NET_SALES", ascending=False)
        )
        m["store_summary"] = t
    else:
        m["store_summary"] = pd.DataFrame()

    # ---- OPERATIONS / TRAFFIC ----
    # Customer Traffic - Storewise
    if has("STORE_NAME", "CUST_CODE"):
        t = (
            df.groupby("STORE_NAME", as_index=False)["CUST_CODE"]
            .nunique()
            .rename(columns={"CUST_CODE": "BASKETS"})
            .sort_values("BASKETS", ascending=False)
        )
        m["traffic_storewise"] = t
    else:
        m["traffic_storewise"] = pd.DataFrame()

    # Active Tills During the day
    if has("TRN_DATE", "TILL", "STORE_NAME"):
        t = df.copy()
        t["TRN_DATE"] = pd.to_datetime(t["TRN_DATE"], errors="coerce")
        t = t.dropna(subset=["TRN_DATE"])
        t["HOUR"] = t["TRN_DATE"].dt.floor("30T")
        act = (
            t.groupby(["STORE_NAME", "HOUR"], as_index=False)["TILL"]
            .nunique()
            .rename(columns={"TILL": "ACTIVE_TILLS"})
        )
        m["active_tills"] = act
    else:
        m["active_tills"] = pd.DataFrame()

    # Avg Customers Served per Till
    if has("TILL", "CUST_CODE"):
        t = (
            df.groupby("TILL", as_index=False)["CUST_CODE"]
            .nunique()
            .rename(columns={"CUST_CODE": "CUSTOMERS"})
        )
        m["avg_cust_per_till"] = t
    else:
        m["avg_cust_per_till"] = pd.DataFrame()

    # Store Customer Traffic Storewise (similar to first, but we keep)
    m["store_customer_traffic"] = m["traffic_storewise"]

    # Customer Traffic-Departmentwise
    if has("DEPARTMENT", "CUST_CODE"):
        t = (
            df.groupby("DEPARTMENT", as_index=False)["CUST_CODE"]
            .nunique()
            .rename(columns={"CUST_CODE": "BASKETS"})
            .sort_values("BASKETS", ascending=False)
        )
        m["traffic_dept"] = t
    else:
        m["traffic_dept"] = pd.DataFrame()

    # Cashiers Performance
    cashier_col = "CASHIER" if "CASHIER" in df.columns else None
    if cashier_col and "NET_SALES" in df.columns:
        t = (
            df.groupby(cashier_col, as_index=False)["NET_SALES"]
            .sum()
            .sort_values("NET_SALES", ascending=False)
        )
        m["cashiers"] = t
    else:
        m["cashiers"] = pd.DataFrame()

    # Till Usage
    if "TILL" in df.columns:
        t = (
            df.groupby("TILL", as_index=False)
            .size()
            .rename(columns={"size": "TXNS"})
            .sort_values("TXNS", ascending=False)
        )
        m["till_usage"] = t
    else:
        m["till_usage"] = pd.DataFrame()

    # Tax Compliance (simple: CU_DEVICE_SERIAL or TAX_FLAG presence)
    tax_col = None
    for c in ["CU_DEVICE_SERIAL", "TAX_FLAG"]:
        if c in df.columns:
            tax_col = c
            break
    if tax_col:
        t = df.copy()
        t["STATUS"] = np.where(
            t[tax_col].astype(str).str.strip().eq(""), "NON_COMPLIANT", "COMPLIANT"
        )
        s = (
            t.groupby("STATUS", as_index=False)
            .size()
            .rename(columns={"size": "TXNS"})
        )
        m["tax_compliance"] = s
    else:
        m["tax_compliance"] = pd.DataFrame()

    # ---- BASKETS & CATEGORIES ----
    # Customer Baskets Overview: basket value per CUST_CODE
    if has("CUST_CODE", "NET_SALES"):
        bt = (
            df.groupby("CUST_CODE", as_index=False)["NET_SALES"]
            .sum()
            .rename(columns={"NET_SALES": "BASKET_VALUE"})
        )
        m["baskets"] = bt
    else:
        m["baskets"] = pd.DataFrame()

    # Global Category Overview - Sales
    if has("CATEGORY", "NET_SALES"):
        t = (
            df.groupby("CATEGORY", as_index=False)["NET_SALES"]
            .sum()
            .sort_values("NET_SALES", ascending=False)
        )
        m["cat_sales"] = t
    else:
        m["cat_sales"] = pd.DataFrame()

    # Global Category Overview - Baskets
    if has("CATEGORY", "CUST_CODE"):
        t = (
            df.groupby("CATEGORY", as_index=False)["CUST_CODE"]
            .nunique()
            .rename(columns={"CUST_CODE": "BASKETS"})
            .sort_values("BASKETS", ascending=False)
        )
        m["cat_baskets"] = t
    else:
        m["cat_baskets"] = pd.DataFrame()

    # Supplier Contribution
    if has("SUPPLIER", "NET_SALES"):
        t = (
            df.groupby("SUPPLIER", as_index=False)["NET_SALES"]
            .sum()
            .sort_values("NET_SALES", ascending=False)
        )
        m["supplier"] = t
    else:
        m["supplier"] = pd.DataFrame()

    # Category Overview (Dept x Cat)
    if has("DEPARTMENT", "CATEGORY", "NET_SALES"):
        t = (
            df.groupby(["DEPARTMENT", "CATEGORY"], as_index=False)["NET_SALES"]
            .sum()
            .sort_values("NET_SALES", ascending=False)
        )
        m["dept_cat"] = t
    else:
        m["dept_cat"] = pd.DataFrame()

    # Branch Comparison (store vs NET_SALES)
    m["branch_comparison"] = m["store_summary"]

    # Product Performance
    if has("ITEM_NAME", "NET_SALES"):
        t = (
            df.groupby("ITEM_NAME", as_index=False)["NET_SALES"]
            .sum()
            .sort_values("NET_SALES", ascending=False)
        )
        m["product_perf"] = t
    else:
        m["product_perf"] = pd.DataFrame()

    # ---- LOYALTY ----
    if has("LOYALTY_CUSTOMER_CODE", "CUST_CODE", "NET_SALES"):
        lo = df.copy()
        lo["HAS_LOYALTY"] = np.where(
            lo["LOYALTY_CUSTOMER_CODE"].astype(str).str.strip().eq(""),
            "NON-LOYAL",
            "LOYAL",
        )
        t = (
            lo.groupby("HAS_LOYALTY", as_index=False)["CUST_CODE"]
            .nunique()
            .rename(columns={"CUST_CODE": "BASKETS"})
        )
        m["loyalty_global"] = t

        if "STORE_NAME" in lo.columns:
            b = (
                lo.groupby(["STORE_NAME", "HAS_LOYALTY"], as_index=False)[
                    "CUST_CODE"
                ]
                .nunique()
                .rename(columns={"CUST_CODE": "BASKETS"})
            )
            m["loyalty_branch"] = b
        else:
            m["loyalty_branch"] = pd.DataFrame()
    else:
        m["loyalty_global"] = pd.DataFrame()
        m["loyalty_branch"] = pd.DataFrame()

    # Customer Loyalty Overview: top loyalty customers
    if has("LOYALTY_CUSTOMER_CODE", "NET_SALES"):
        lo = df.copy()
        lo["LOYALTY_CUSTOMER_CODE"] = lo["LOYALTY_CUSTOMER_CODE"].astype(str)
        t = (
            lo.groupby("LOYALTY_CUSTOMER_CODE", as_index=False)["NET_SALES"]
            .sum()
            .sort_values("NET_SALES", ascending=False)
        )
        m["loyalty_customer"] = t
    else:
        m["loyalty_customer"] = pd.DataFrame()

    # ---- PRICING ----
    if has("ITEM_CODE", "ITEM_NAME", "SP_PRE_VAT", "QTY", "TRN_DATE", "STORE_NAME"):
        pp = df.copy()
        pp["TRN_DATE"] = pd.to_datetime(pp["TRN_DATE"], errors="coerce")
        pp = pp.dropna(subset=["TRN_DATE"])
        pp["DATE"] = pp["TRN_DATE"].dt.date

        g = (
            pp.groupby(
                ["STORE_NAME", "DATE", "ITEM_CODE", "ITEM_NAME"], as_index=False
            )
            .agg(
                Num_Prices=("SP_PRE_VAT", lambda s: s.dropna().nunique()),
                Price_Min=("SP_PRE_VAT", "min"),
                Price_Max=("SP_PRE_VAT", "max"),
                Total_QTY=("QTY", "sum"),
            )
        )
        g["Spread"] = g["Price_Max"] - g["Price_Min"]
        multi = g[(g["Num_Prices"] > 1) & (g["Spread"] > 0)].copy()
        if not multi.empty:
            multi["Value_Impact"] = multi["Total_QTY"] * multi["Spread"]
            summ = (
                multi.groupby("STORE_NAME", as_index=False)["Value_Impact"]
                .sum()
                .sort_values("Value_Impact", ascending=False)
            )
            m["pricing_global"] = summ
            m["pricing_detail"] = multi
        else:
            m["pricing_global"] = pd.DataFrame()
            m["pricing_detail"] = pd.DataFrame()
    else:
        m["pricing_global"] = pd.DataFrame()
        m["pricing_detail"] = pd.DataFrame()

    # ---- REFUNDS ----
    if has("NET_SALES"):
        neg = df[df["NET_SALES"] < 0].copy()
        if not neg.empty:
            if "STORE_NAME" in neg.columns:
                g = (
                    neg.groupby("STORE_NAME", as_index=False)["NET_SALES"]
                    .sum()
                    .rename(columns={"NET_SALES": "REFUND_VALUE"})
                    .sort_values("REFUND_VALUE")
                )
                m["refunds_global"] = g
                if "CUST_CODE" in neg.columns:
                    d = (
                        neg.groupby(["STORE_NAME", "CUST_CODE"], as_index=False)[
                            "NET_SALES"
                        ]
                        .sum()
                        .rename(columns={"NET_SALES": "REFUND_VALUE"})
                    )
                    m["refunds_branch"] = d
                else:
                    m["refunds_branch"] = pd.DataFrame()
            else:
                m["refunds_global"] = pd.DataFrame()
                m["refunds_branch"] = pd.DataFrame()
        else:
            m["refunds_global"] = pd.DataFrame()
            m["refunds_branch"] = pd.DataFrame()
    else:
        m["refunds_global"] = pd.DataFrame()
        m["refunds_branch"] = pd.DataFrame()

    return m


# =============================================================================
# UI – Upload & Navigation
# =============================================================================
st.title("🦸 Superdeck Analytics Dashboard")
st.caption(
    "CSV → notebook-aligned views. Only non-starred sections from your DailyDeck notebook are exposed."
)

uploaded = st.sidebar.file_uploader("Upload DAILY POS CSV", type=["csv"])
if not uploaded:
    st.info("Upload your CSV in the sidebar to begin.")
    st.stop()

try:
    df = load_data(uploaded.getvalue())
    metrics = compute_metrics(df)
except Exception as e:
    st.error("Failed to load/prepare data. Check logs for details.")
    traceback.print_exc(file=sys.stdout)
    st.stop()

# Navigation structure (no starred entries)
NAV = {
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

main_section = st.sidebar.radio("Main section", list(NAV.keys()))
subsection = st.sidebar.selectbox(
    "View", NAV[main_section], key="sub_view"
)

st.subheader(f"{main_section} ➜ {subsection}")


# =============================================================================
# Section renderers
# =============================================================================
try:
    # ---------- SALES ----------
    if main_section == "SALES":
        if subsection == "Global sales Overview":
            t = metrics["sales_l1"]
            if t.empty:
                st.info("Need SALES_CHANNEL_L1 & NET_SALES.")
            else:
                labels = [
                    f"{r.SALES_CHANNEL_L1} ({r.PCT:.1f}%)"
                    for _, r in t.iterrows()
                ]
                pie(labels, t["NET_SALES_M"], "Sales by Channel (L1)")
                out = t[
                    ["SALES_CHANNEL_L1", "NET_SALES", "NET_SALES_M", "PCT"]
                ].copy()
                out = add_total_row(
                    out, ["NET_SALES"], "SALES_CHANNEL_L1"
                )
                show_table(
                    out,
                    int_cols=["NET_SALES"],
                    float_cols=["NET_SALES_M", "PCT"],
                    download_name="global_sales_overview.csv",
                )

        elif subsection == "Global Net Sales Distribution by Sales Channel":
            t = metrics["sales_l2"]
            if t.empty:
                st.info("Need SALES_CHANNEL_L2 & NET_SALES.")
            else:
                bar(
                    t,
                    "SALES_CHANNEL_L2",
                    "NET_SALES",
                    "Net Sales by Channel (L2)",
                )
                show_table(
                    t,
                    int_cols=["NET_SALES"],
                    float_cols=["PCT"],
                    download_name="global_sales_by_channel.csv",
                )

        elif subsection == "Global Net Sales Distribution by SHIFT":
            t = metrics["sales_shift"]
            if t.empty:
                st.info("Need TRN_DATE & NET_SALES.")
            else:
                pie(t["SHIFT"], t["NET_SALES"], "Net Sales by Shift")
                show_table(
                    t,
                    int_cols=["NET_SALES"],
                    float_cols=["PCT"],
                    download_name="global_sales_by_shift.csv",
                )

        elif subsection == "Night vs Day Shift Sales Ratio — Stores with Night Shifts":
            per = metrics["store_shift"]
            if per.empty:
                st.info("Need TRN_DATE, NET_SALES, STORE_NAME.")
            else:
                piv = (
                    per.pivot(
                        index="STORE_NAME",
                        columns="SHIFT",
                        values="NET_SALES",
                    )
                    .fillna(0)
                    .reset_index()
                )
                # keep only stores with some NIGHT sales
                piv = piv[piv.get("NIGHT", 0) > 0]
                if piv.empty:
                    st.info("No stores with NIGHT shift sales found.")
                else:
                    piv["TOTAL"] = piv.sum(axis=1, numeric_only=True)
                    for c in ["DAY", "NIGHT"]:
                        if c in piv.columns:
                            piv[c + "_PCT"] = np.where(
                                piv["TOTAL"] > 0,
                                piv[c] / piv["TOTAL"] * 100,
                                0,
                            )
                    bar(
                        piv.sort_values("NIGHT_PCT", ascending=False),
                        "STORE_NAME",
                        "NIGHT_PCT",
                        "Night Shift Share (Stores with Night Sales)",
                    )
                    show_table(
                        piv,
                        float_cols=[
                            c
                            for c in piv.columns
                            if c.endswith("_PCT")
                        ],
                        download_name="night_vs_day_stores_with_night.csv",
                    )

        elif subsection == "Global Day vs Night Sales — Only Stores with NIGHT Shift":
            per = metrics["store_shift"]
            if per.empty:
                st.info("Need TRN_DATE, NET_SALES, STORE_NAME.")
            else:
                piv = (
                    per.pivot(
                        index="STORE_NAME",
                        columns="SHIFT",
                        values="NET_SALES",
                    )
                    .fillna(0)
                    .reset_index()
                )
                piv = piv[piv.get("NIGHT", 0) > 0]
                if piv.empty:
                    st.info("No stores with NIGHT shift sales.")
                else:
                    melted = piv.melt(
                        id_vars="STORE_NAME",
                        value_vars=[
                            c
                            for c in ["DAY", "NIGHT"]
                            if c in piv.columns
                        ],
                        var_name="SHIFT",
                        value_name="NET_SALES",
                    )
                    bar(
                        melted,
                        "STORE_NAME",
                        "NET_SALES",
                        "Day vs Night Sales (Stores with Night Shift)",
                    )
                    show_table(
                        melted,
                        int_cols=["NET_SALES"],
                        download_name="day_vs_night_stores_with_night.csv",
                    )

        elif subsection == "2nd-Highest Channel Share":
            t = metrics["second_channel"]
            if t.empty:
                st.info(
                    "Need STORE_NAME, SALES_CHANNEL_L1, NET_SALES."
                )
            else:
                bar(
                    t.head(30),
                    "STORE_NAME",
                    "2ND_PCT",
                    "Top 30 Stores by 2nd Highest Channel Share",
                )
                show_table(
                    t,
                    float_cols=["2ND_PCT"],
                    download_name="second_highest_channel_share.csv",
                )

        elif subsection == "Bottom 30 — 2nd Highest Channel":
            t = metrics["second_channel_bottom_30"]
            if t.empty:
                st.info(
                    "Need STORE_NAME, SALES_CHANNEL_L1, NET_SALES."
                )
            else:
                bar(
                    t,
                    "STORE_NAME",
                    "2ND_PCT",
                    "Bottom 30 Stores by 2nd Highest Channel Share",
                )
                show_table(
                    t,
                    float_cols=["2ND_PCT"],
                    download_name="bottom30_second_channel_share.csv",
                )

        elif subsection == "Stores Sales Summary":
            t = metrics["store_summary"]
            show_table(
                t,
                int_cols=[c for c in t.columns if c != "STORE_NAME"],
                download_name="stores_sales_summary.csv",
            )

    # ---------- OPERATIONS ----------
    elif main_section == "OPERATIONS":
        if subsection == "Customer Traffic-Storewise":
            t = metrics["traffic_storewise"]
            bar(t, "STORE_NAME", "BASKETS", "Customer Baskets by Store")
            show_table(
                t,
                int_cols=["BASKETS"],
                download_name="customer_traffic_storewise.csv",
            )

        elif subsection == "Active Tills During the day":
            t = metrics["active_tills"]
            if t.empty:
                st.info("Need TRN_DATE, TILL, STORE_NAME.")
            else:
                line(
                    t,
                    "HOUR",
                    "ACTIVE_TILLS",
                    "Active Tills Over Time (All Stores)",
                )
                show_table(
                    t,
                    int_cols=["ACTIVE_TILLS"],
                    download_name="active_tills.csv",
                )

        elif subsection == "Average Customers Served per Till":
            t = metrics["avg_cust_per_till"]
            show_table(
                t,
                int_cols=["CUSTOMERS"],
                download_name="avg_customers_per_till.csv",
            )

        elif subsection == "Store Customer Traffic Storewise":
            t = metrics["store_customer_traffic"]
            bar(
                t,
                "STORE_NAME",
                "BASKETS",
                "Store Customer Traffic (Baskets)",
            )
            show_table(
                t,
                int_cols=["BASKETS"],
                download_name="store_customer_traffic.csv",
            )

        elif subsection == "Customer Traffic-Departmentwise":
            t = metrics["traffic_dept"]
            bar(
                t,
                "DEPARTMENT",
                "BASKETS",
                "Customer Traffic by Department",
            )
            show_table(
                t,
                int_cols=["BASKETS"],
                download_name="customer_traffic_departmentwise.csv",
            )

        elif subsection == "Cashiers Perfomance":
            t = metrics["cashiers"]
            show_table(
                t,
                int_cols=["NET_SALES"]
                if "NET_SALES" in t.columns
                else None,
                download_name="cashiers_performance.csv",
            )

        elif subsection == "Till Usage":
            t = metrics["till_usage"]
            show_table(
                t,
                int_cols=["TXNS"],
                download_name="till_usage.csv",
            )

        elif subsection == "Tax Compliance":
            t = metrics["tax_compliance"]
            pie(t["STATUS"], t["TXNS"], "Tax Compliance Status")
            show_table(
                t,
                int_cols=["TXNS"],
                download_name="tax_compliance.csv",
            )

    # ---------- INSIGHTS ----------
    elif main_section == "INSIGHTS":
        if subsection == "Customer Baskets Overview":
            t = metrics["baskets"]
            if t.empty:
                st.info("Need CUST_CODE & NET_SALES.")
            else:
                st.write("Basket value distribution (sample):")
                show_table(
                    t.sort_values(
                        "BASKET_VALUE", ascending=False
                    ).head(200),
                    int_cols=["BASKET_VALUE"],
                    download_name="customer_baskets_overview.csv",
                )

        elif subsection == "Global Category Overview-Sales":
            t = metrics["cat_sales"]
            bar(t, "CATEGORY", "NET_SALES", "Sales by Category")
            show_table(
                t,
                int_cols=["NET_SALES"],
                download_name="global_category_sales.csv",
            )

        elif subsection == "Global Category Overview-Baskets":
            t = metrics["cat_baskets"]
            bar(t, "CATEGORY", "BASKETS", "Baskets by Category")
            show_table(
                t,
                int_cols=["BASKETS"],
                download_name="global_category_baskets.csv",
            )

        elif subsection == "Supplier Contribution":
            t = metrics["supplier"]
            show_table(
                t,
                int_cols=["NET_SALES"]
                if "NET_SALES" in t.columns
                else None,
                download_name="supplier_contribution.csv",
            )

        elif subsection == "Category Overview":
            t = metrics["dept_cat"]
            show_table(
                t,
                int_cols=["NET_SALES"],
                download_name="category_overview_dept_cat.csv",
            )

        elif subsection == "Branch Comparison":
            t = metrics["branch_comparison"]
            bar(t, "STORE_NAME", "NET_SALES", "Net Sales by Store")
            show_table(
                t,
                int_cols=[c for c in t.columns if c != "STORE_NAME"],
                download_name="branch_comparison.csv",
            )

        elif subsection == "Product Perfomance":
            t = metrics["product_perf"]
            show_table(
                t.head(200),
                int_cols=["NET_SALES"]
                if "NET_SALES" in t.columns
                else None,
                download_name="product_performance.csv",
            )

        elif subsection == "Global Loyalty Overview":
            t = metrics["loyalty_global"]
            pie(
                t["HAS_LOYALTY"],
                t["BASKETS"],
                "Loyal vs Non-Loyal Baskets",
            )
            show_table(
                t,
                int_cols=["BASKETS"],
                download_name="loyalty_global.csv",
            )

        elif subsection == "Branch Loyalty Overview":
            t = metrics["loyalty_branch"]
            show_table(
                t,
                int_cols=["BASKETS"],
                download_name="branch_loyalty_overview.csv",
            )

        elif subsection == "Customer Loyalty Overview":
            t = metrics["loyalty_customer"]
            show_table(
                t.head(200),
                int_cols=["NET_SALES"]
                if "NET_SALES" in t.columns
                else None,
                download_name="customer_loyalty_overview.csv",
            )

        elif subsection == "Global Pricing Overview":
            t = metrics["pricing_global"]
            show_table(
                t,
                float_cols=["Value_Impact"]
                if "Value_Impact" in t.columns
                else None,
                download_name="global_pricing_overview.csv",
            )

        elif subsection == "Branch Brach Overview":
            t = metrics["dept_cat"]
            if t.empty:
                st.info("Need DEPARTMENT, CATEGORY, NET_SALES.")
            else:
                show_table(
                    t,
                    int_cols=["NET_SALES"],
                    download_name="branch_branch_overview.csv",
                )

        elif subsection == "Global Refunds Overview":
            t = metrics["refunds_global"]
            show_table(
                t,
                int_cols=["REFUND_VALUE"]
                if "REFUND_VALUE" in t.columns
                else None,
                download_name="global_refunds_overview.csv",
            )

        elif subsection == "Branch Refunds Overview":
            t = metrics["refunds_branch"]
            show_table(
                t,
                int_cols=["REFUND_VALUE"]
                if "REFUND_VALUE" in t.columns
                else None,
                download_name="branch_refunds_overview.csv",
            )

    else:
        st.info("Unknown section.")

except Exception:
    st.error(
        "Unexpected error while rendering this section. See logs for details."
    )
    traceback.print_exc(file=sys.stdout)
