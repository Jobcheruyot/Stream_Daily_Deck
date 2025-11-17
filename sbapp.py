def smart_load() -> pd.DataFrame:
    """
    Supabase-based loader:
    - user selects date range in sidebar
    - we fetch ALL rows in that range in chunks
    """
    st.sidebar.markdown("### Select Date Range (Supabase)")
    today = date.today()
    start_date = st.sidebar.date_input("Start date", today - timedelta(days=7))
    end_date   = st.sidebar.date_input("End date", today)

    if start_date > end_date:
        st.sidebar.error("Start date cannot be after end date")
        st.stop()

    with st.spinner("Loading data from Supabase ..."):
        df = load_from_supabase(start_date, end_date)

    if df is None or df.empty:
        st.sidebar.warning("No data returned from Supabase for this period.")
        return None

    st.sidebar.success(
        f"Loaded {len(df):,} rows from Supabase\n"
        f"{start_date} → {end_date}"
    )
    return df


def main():
    st.title("DailyDeck: The Story Behind the Numbers (Supabase Edition)")

    raw_df = smart_load()
    if raw_df is None:
        st.stop()

    with st.spinner("Preparing data (cached) ..."):
        df = clean_and_derive(raw_df)

    section = st.sidebar.selectbox(
        "Section",
        ["SALES", "OPERATIONS", "INSIGHTS"]
    )

    # -----------------------
    # SALES
    # -----------------------
    if section == "SALES":
        sales_items = [
            "Global sales Overview",
            "Global Net Sales Distribution by Sales Channel",
            "Global Net Sales Distribution by SHIFT",
            "Night vs Day Shift Sales Ratio — Stores with Night Shifts",
            "Global Day vs Night Sales — Only Stores with NIGHT Shift",
            "2nd-Highest Channel Share",
            "Bottom 30 — 2nd Highest Channel",
            "Stores Sales Summary"
        ]
        choice = st.sidebar.selectbox("Sales Subsection", sales_items)

        if choice == sales_items[0]:
            sales_global_overview(df)
        elif choice == sales_items[1]:
            sales_by_channel_l2(df)
        elif choice == sales_items[2]:
            sales_by_shift(df)
        elif choice == sales_items[3]:
            night_vs_day_ratio(df)
        elif choice == sales_items[4]:
            global_day_vs_night(df)
        elif choice == sales_items[5]:
            second_highest_channel_share(df)
        elif choice == sales_items[6]:
            bottom_30_2nd_highest(df)
        elif choice == sales_items[7]:
            stores_sales_summary(df)

        # Trend strip for SALES
        show_trends(df, section)

    # -----------------------
    # OPERATIONS
    # -----------------------
    elif section == "OPERATIONS":
        ops_items = [
            "Customer Traffic-Storewise",
            "Active Tills During the day",
            "Average Customers Served per Till",
            "Store Customer Traffic Storewise",
            "Customer Traffic-Departmentwise",
            "Cashiers Perfomance",
            "Till Usage",
            "Tax Compliance"
        ]
        choice = st.sidebar.selectbox("Operations Subsection", ops_items)

        if choice == ops_items[0]:
            customer_traffic_storewise(df)
        elif choice == ops_items[1]:
            active_tills_during_day(df)
        elif choice == ops_items[2]:
            avg_customers_per_till(df)
        elif choice == ops_items[3]:
            store_customer_traffic_storewise(df)
        elif choice == ops_items[4]:
            customer_traffic_departmentwise(df)
        elif choice == ops_items[5]:
            cashiers_performance(df)
        elif choice == ops_items[6]:
            till_usage(df)
        elif choice == ops_items[7]:
            tax_compliance(df)

        # Trend strip for OPERATIONS
        show_trends(df, section)

    # -----------------------
    # INSIGHTS
    # -----------------------
    elif section == "INSIGHTS":
        ins_items = [
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
            "Branch Pricing Overview",
            "Global Refunds Overview",
            "Branch Refunds Overview"
        ]
        choice = st.sidebar.selectbox("Insights Subsection", ins_items)

        if choice == ins_items[0]:
            customer_baskets_overview(df)
        elif choice == ins_items[1]:
            global_category_overview_sales(df)
        elif choice == ins_items[2]:
            global_category_overview_baskets(df)
        elif choice == ins_items[3]:
            supplier_contribution(df)
        elif choice == ins_items[4]:
            category_overview(df)
        elif choice == ins_items[5]:
            branch_comparison(df)
        elif choice == ins_items[6]:
            product_performance(df)
        elif choice == ins_items[7]:
            global_loyalty_overview(df)
        elif choice == ins_items[8]:
            branch_loyalty_overview(df)
        elif choice == ins_items[9]:
            customer_loyalty_overview(df)
        elif choice == ins_items[10]:
            global_pricing_overview(df)
        elif choice == ins_items[11]:
            branch_pricing_overview(df)
        elif choice == ins_items[12]:
            global_refunds_overview(df)
        elif choice == ins_items[13]:
            branch_refunds_overview(df)

        # Trend strip for INSIGHTS
        show_trends(df, section)


if __name__ == "__main__":
    main()
