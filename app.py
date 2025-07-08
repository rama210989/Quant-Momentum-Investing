import streamlit as st
from logic.momentum import get_momentum_df
import pandas as pd

st.set_page_config(page_title="Momentum + FIP Screener", layout="wide")

st.title("📈 Momentum Investing Strategy Dashboard")

tab1, tab2 = st.tabs(["📋 Nifty 500 List", "🚀 Momentum + FIP Picks"])

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
        st.warning("No stocks available.")
    else:
        # Get month labels dynamically from momentum_df columns (they match momentum.py output)
        month_labels = []
        for col in momentum_df.columns:
            # Month columns are 3-letter month + 4-digit year format, e.g. 'Jul 2024'
            if len(col) == 8 and col[3] == " " and col[:3].isalpha() and col[4:].isdigit():
                month_labels.append(col)

        display_columns = ["Company Name", "Symbol", "ISIN Code"] + month_labels + ["momentum_score", "pct_negative_months", "pct_positive_months", "fip_score"]

        st.dataframe(momentum_df[display_columns])
