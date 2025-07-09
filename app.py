import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import io
import xlsxwriter
from typing import Optional, Dict, Any, List
from stock_data_fetcher import fetch_stock_data_cache

# Page configuration for mobile compatibility
st.set_page_config(
    page_title="Stock Momentum & FIP Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"  # Mobile-friendly: collapsed sidebar
)

# Custom CSS for mobile responsiveness
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 12px;
        padding-right: 12px;
    }
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            padding-left: 8px;
            padding-right: 8px;
            font-size: 14px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Stock Momentum & FIP Analysis")
st.markdown("**Analyze NIFTY 100 + NIFTY MIDCAP 150 stocks with dynamic momentum & FIP scoring**")

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None

# Fixed parameters (matching Colab exactly)
today = datetime.datetime.today()
end_date = datetime.datetime(today.year, today.month, 1) + pd.DateOffset(months=1)
start_date = end_date - pd.DateOffset(months=13)

# Month labels: exactly like Colab
month_range = pd.date_range(start=start_date, periods=11, freq='MS')
month_labels = [d.strftime('%b %Y') for d in month_range]

def validate_csv(df: pd.DataFrame) -> bool:
    """Validate uploaded CSV file format"""
    required_columns = ['Symbol', 'Company Name', 'ISIN Code']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return False
    
    if df.empty:
        st.error("CSV file is empty")
        return False
    
    return True

def calculate_momentum(symbol: str, start_date: datetime.datetime, 
                     end_date: datetime.datetime, month_labels: List[str]) -> Optional[Dict[str, Any]]:
    """Calculate momentum score exactly like Colab"""
    try:
        data = yf.download(f"{symbol}.NS", start=start_date, end=end_date, 
                          interval="1mo", progress=False, auto_adjust=True)

        if data.shape[0] < 13 or "Close" not in data.columns:
            return None

        close_prices = data["Close"].dropna()
        monthly_returns = close_prices.pct_change().dropna()

        if isinstance(monthly_returns, pd.DataFrame):
            monthly_returns = monthly_returns.squeeze()

        last_11 = monthly_returns[-12:-1]  # Jul 2024 to May 2025
        if len(last_11) < 11:
            return None

        returns_dict = {}
        for i in range(11):
            val = float(last_11.iloc[i]) * 100  # Convert to percent
            returns_dict[month_labels[i]] = val

        # Momentum score
        momentum_score = (last_11 + 1).prod() - 1
        momentum_score = float(momentum_score) * 100  # Convert to percent
        returns_dict["momentum_score"] = momentum_score

        return returns_dict

    except Exception:
        return None

def calculate_fip(row, month_labels: List[str]) -> pd.Series:
    """Calculate FIP score exactly like Colab"""
    # Extract monthly returns columns as numeric (decimal, e.g. 0.108)
    returns = row[month_labels].astype(float) / 100  # convert from percentage string to decimal

    # Count negative and positive months
    pct_negative = (returns < 0).sum() / len(returns)
    pct_positive = (returns > 0).sum() / len(returns)

    # Get sign of momentum score (convert from percentage string to decimal)
    momentum = float(row['momentum_score'])
    sign_momentum = 1 if momentum > 0 else -1

    # Calculate FIP
    fip = sign_momentum * (pct_negative - pct_positive)

    return pd.Series({
        "pct_negative_months": pct_negative * 100,
        "pct_positive_months": pct_positive * 100,
        "fip_score": fip
    })

# Analysis function - Define before usage
def run_analysis(stock_data: pd.DataFrame, month_labels: List[str]):
    """Run the momentum and FIP analysis"""
    with st.spinner("Analyzing stock momentum and FIP scores..."):
        results_list = []
        progress_bar = st.progress(0)
        
        for i, (idx, row) in enumerate(stock_data.iterrows()):
            symbol = row['Symbol'].replace('.NS', '')  # Remove .NS if present
            company_name = row['Company Name']
            isin_code = row['ISIN Code']
            
            # Calculate momentum
            momentum_data = calculate_momentum(symbol, start_date, end_date, month_labels)
            
            if momentum_data is not None:
                # Calculate FIP
                fip_data = calculate_fip(pd.Series(momentum_data), month_labels)
                
                # Combine results
                result = {
                    'Company Name': company_name,
                    'Symbol': symbol,
                    'ISIN Code': isin_code,
                    **momentum_data,
                    **fip_data
                }
                results_list.append(result)
            
            # Update progress
            progress_bar.progress((i + 1) / len(stock_data))
        
        if results_list:
            results_df = pd.DataFrame(results_list)
            results_df = results_df.sort_values('momentum_score', ascending=False)
            
            st.session_state.results_df = results_df
            st.session_state.analysis_complete = True
            
            st.success(f"✅ Analysis complete! Found {len(results_df)} valid results out of {len(stock_data)} stocks.")
            st.balloons()
        else:
            st.error("No valid results found. Please check your data and try again.")

# Create main tab structure
tab1, tab2, tab3 = st.tabs(["📊 Fetch Data", "📋 All Results", "🎯 Top 25 Stable Stocks"])

with tab1:
    st.header("📊 Fetch Stock Data")
    
    # Data source selection
    col1, col2 = st.columns(2)
    
    with col1:
        data_source = st.selectbox(
            "Select Data Source",
            ["NIFTY 100 + NIFTY MIDCAP 150 (Combined)", "NIFTY 100 Only", "NIFTY MIDCAP 150 Only", "Upload CSV File"],
            index=0
        )
    
    with col2:
        st.info(f"""
        **Analysis Settings:**
        - Period: {month_labels[0]} to {month_labels[-1]}
        - Rolling Window: 11 months
        - Auto-updates monthly
        """)
    
    # Data fetching based on selection
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload CSV file with stock symbols",
            type=['csv'],
            help="CSV should contain columns: 'Symbol', 'Company Name', 'ISIN Code'"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if validate_csv(df):
                    st.session_state.stock_data = df
                    st.success(f"✅ Loaded {len(df)} stocks from uploaded file")
                    st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
    else:
        # Fetch from predefined indices
        if st.button("🔄 Fetch Current Index Data", type="primary"):
            with st.spinner("Fetching latest stock data..."):
                try:
                    if data_source == "NIFTY 100 + NIFTY MIDCAP 150 (Combined)":
                        stock_data = fetch_stock_data_cache("combined")
                    elif data_source == "NIFTY 100 Only":
                        stock_data = fetch_stock_data_cache("nifty100")
                    elif data_source == "NIFTY MIDCAP 150 Only":
                        stock_data = fetch_stock_data_cache("midcap150")
                    
                    st.session_state.stock_data = stock_data
                    st.success(f"✅ Fetched {len(stock_data)} stocks from {data_source}")
                    
                    # Display sample data
                    st.subheader("Sample Stock Data")
                    st.dataframe(stock_data.head(10), use_container_width=True)
                    
                    # Show summary statistics
                    st.subheader("Data Summary")
                    st.write(f"**Total Stocks:** {len(stock_data)}")
                    st.write(f"**Unique Companies:** {stock_data['Company Name'].nunique()}")
                    st.write(f"**Data Source:** {data_source}")
                    
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
    
    # Run analysis button
    if st.session_state.stock_data is not None:
        st.divider()
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Run Momentum & FIP Analysis", type="primary", use_container_width=True):
                run_analysis(st.session_state.stock_data, month_labels)
        
        with col2:
            stock_count = len(st.session_state.stock_data)
            st.metric("Stocks to Analyze", stock_count)
        
        with col3:
            if st.session_state.analysis_complete:
                st.metric("Analysis Status", "✅ Complete")
            else:
                st.metric("Analysis Status", "⏳ Pending")

with tab2:
    st.header("📋 All Results")
    
    if st.session_state.analysis_complete and st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_momentum = results_df['momentum_score'].mean()
            st.metric("Avg Momentum Score", f"{avg_momentum:.3f}")
        
        with col2:
            positive_momentum = (results_df['momentum_score'] > 0).sum()
            st.metric("Positive Momentum", f"{positive_momentum}/{len(results_df)}")
        
        with col3:
            avg_fip = results_df['fip_score'].mean()
            st.metric("Avg FIP Score", f"{avg_fip:.3f}")
        
        # Results table
        st.subheader("All Results (Sorted by Momentum Score)")
        
        # Define display columns
        display_columns = ["Company Name", "Symbol", "ISIN Code"] + month_labels + ["momentum_score"]
        display_columns_extended = display_columns + ["pct_negative_months", "pct_positive_months", "fip_score"]
        
        st.dataframe(
            results_df[display_columns_extended],
            use_container_width=True,
            hide_index=True
        )
        
        # Download option
        if st.button("📥 Download Results as Excel"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                results_df.to_excel(writer, sheet_name='Analysis Results', index=False)
            
            st.download_button(
                label="Download Excel File",
                data=buffer.getvalue(),
                file_name=f"stock_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("👆 Please fetch data and run analysis first in the 'Fetch Data' tab.")

with tab3:
    st.header("🎯 Top 25 Stable Stocks")
    
    if st.session_state.analysis_complete and st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        
        # Filter stocks with negative FIP scores
        stable_stocks = results_df[results_df['fip_score'] < 0]
        
        if len(stable_stocks) > 0:
            # Get top 25 by momentum score
            top_25 = stable_stocks.head(25)
            
            # Summary metrics for top 25
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Stable Stocks Found", len(stable_stocks))
            
            with col2:
                avg_momentum_top25 = top_25['momentum_score'].mean()
                st.metric("Avg Momentum (Top 25)", f"{avg_momentum_top25:.3f}")
            
            with col3:
                avg_fip_top25 = top_25['fip_score'].mean()
                st.metric("Avg FIP (Top 25)", f"{avg_fip_top25:.3f}")
            
            st.success(f"Showing top 25 stocks with highest momentum from {len(stable_stocks)} stable stocks (negative FIP)")
            
            # Display top 25 results
            display_columns = ["Company Name", "Symbol", "ISIN Code"] + month_labels + ["momentum_score"]
            display_columns_extended = display_columns + ["pct_negative_months", "pct_positive_months", "fip_score"]
            
            st.dataframe(
                top_25[display_columns_extended],
                use_container_width=True,
                hide_index=True
            )
            
            # Download option for top 25
            if st.button("📥 Download Top 25 as Excel"):
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    top_25.to_excel(writer, sheet_name='Top 25 Stable Stocks', index=False)
                
                st.download_button(
                    label="Download Top 25 Excel File",
                    data=buffer.getvalue(),
                    file_name=f"top_25_stable_stocks_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("No stocks found with negative FIP scores.")
    else:
        st.info("👆 Please fetch data and run analysis first in the 'Fetch Data' tab.")


