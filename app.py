import streamlit as st
from logic.momentum import get_momentum_df
from logic.backtest import run_backtest  # keep for future use
import pandas as pd

st.set_page_config(page_title="Momentum + FIP Screener", layout="wide")

st.title("📈 Momentum Investing Strategy Dashboard")

tab1, tab2, tab3 = st.tabs(["📋 Nifty 500 List", "🚀 Momentum + FIP Picks", "⏳ Backtest Strategy"])

# --- Tab 1: Nifty 500 List ---
with tab1:
    st.header("📋 Nifty 500 Stocks")
    nifty_df = pd.read_csv("data/nifty500.csv")
    st.dataframe(nifty_df)

# --- Tab 2: Momentum + FIP Picks ---
with tab2:
    st.header("🚀 Stable Momentum + FIP Picks (Live)")
    with st.spinner("Fetching latest data..."):
        momentum_df = get_momentum_df()

    if momentum_df.empty:
        st.warning("No stocks passed the filters.")
    else:
        # Columns to show: company name, symbol, ISIN, 11 month returns, momentum score, FIP score
        month_range = pd.date_range(
            start=pd.Timestamp.today().replace(day=1) - pd.DateOffset(months=12),
            periods=11,
            freq='MS'
        )
        month_labels = [d.strftime('%b %Y') for d in month_range]

        display_columns = ["Company Name", "Symbol", "ISIN Code"] + month_labels + ["momentum_score", "fip_score"]
        st.dataframe(momentum_df[display_columns])

# --- Tab 3: Backtest Strategy ---
with tab3:
    st.header("⏳ Backtest Historical Performance")

    selected_date = st.date_input("Select backtest month (for example: 2025-04-01)", value=pd.to_datetime("2025-04-01"))

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            bt_df = run_backtest(selected_date)
        if bt_df.empty:
            st.warning("No stocks available for this backtest.")
        else:
            st.dataframe(bt_df)
