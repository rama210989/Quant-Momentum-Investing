# app.py
import streamlit as st
from logic.momentum import get_momentum_df
from logic.backtest import run_backtest
import pandas as pd

st.set_page_config(page_title="Momentum + FIP Screener", layout="wide")

st.title("📈 Momentum Investing Strategy Dashboard")

# Tabbed layout
tab1, tab2, tab3 = st.tabs(["📋 Nifty 500 List", "🚀 Top 30 Momentum Picks", "⏳ Backtest Strategy"])

# --- Tab 1: Nifty 500 List ---
with tab1:
    st.header("📋 Nifty 500 Stocks")
    nifty_df = pd.read_csv("data/nifty500.csv")
    st.dataframe(nifty_df)

# --- Tab 2: Top 30 Momentum Picks ---
with tab2:
    st.header("🚀 Top 30 Momentum + FIP Picks (Live)")
    with st.spinner("Fetching latest data..."):
        top_df = get_momentum_df()
    if top_df.empty:
        st.warning("No stocks passed the filters.")
    else:
        st.dataframe(top_df)

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
