def yesterday_story(df):
    if "TRN_DATE" not in df.columns or "SP_PRE_VAT" not in df.columns:
        st.info("Yesterday’s narrative will appear once TRN_DATE and SP_PRE_VAT are available.")
        return

    df = df.dropna(subset=["TRN_DATE"])
    df["DATE"] = df["TRN_DATE"].dt.date

    if df["DATE"].nunique() < 2:
        st.info("Need at least two days of data to describe yesterday vs previous day.")
        return

    totals = df.groupby("DATE")["SP_PRE_VAT"].sum().sort_index()
    yest, prev = totals.index[-1], totals.index[-2]
    yest_val, prev_val = totals.iloc[-1], totals.iloc[-2]
    delta = ((yest_val - prev_val) / prev_val) * 100 if prev_val else 0

    # Category + branch quick signals (if columns exist)
    cat_txt = ""
    if "CATEGORY" in df.columns:
        top_cat = (
            df[df["DATE"] == yest]
            .groupby("CATEGORY")["SP_PRE_VAT"].sum()
            .sort_values(ascending=False)
            .head(2)
            .index.tolist()
        )
        if top_cat:
            cat_txt = f" led mainly by **{', '.join(top_cat)}**."

    branch_txt = ""
    if "STORE_NAME" in df.columns:
        by_store = (
            df[df["DATE"] == yest]
            .groupby("STORE_NAME")["SP_PRE_VAT"].sum()
            .sort_values(ascending=False)
        )
        leaders = by_store.head(3).index.tolist()
        laggards = by_store.tail(3).index.tolist()
        if leaders:
            branch_txt += f" Top performers were **{', '.join(leaders)}**."
        if laggards:
            branch_txt += f" Stores needing attention include **{', '.join(laggards)}**."

    direction = "higher" if delta > 0 else "lower" if delta < 0 else "flat"
    delta_txt = f"{abs(delta):.1f}%" if prev_val else "N/A"

    text = (
        f"Yesterday (**{yest}**) the business closed at **KSh {yest_val:,.0f}**, "
        f"{direction} than the previous day (**{delta_txt}**)."
        f"{cat_txt}"
        f"{branch_txt}"
    )

    st.subheader("📰 Yesterday in Our Stores")
    st.write(text)

safe_section("📰 Yesterday’s Trading Narrative", yesterday_story, df)
