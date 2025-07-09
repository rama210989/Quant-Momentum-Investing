import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import io
import xlsxwriter
from typing import Optional, Dict, Any, List
from stock_data_fetcher import fetch_stock_data_cache
import requests
import json
import calendar
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
if 'telegram_config' not in st.session_state:
    st.session_state.telegram_config = None

# Fixed parameters (matching Colab exactly)
today = datetime.today()
end_date = datetime(today.year, today.month, 1) + pd.DateOffset(months=1)
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

def calculate_momentum(symbol: str, start_date: datetime, 
                     end_date: datetime, month_labels: List[str]) -> Optional[Dict[str, Any]]:
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

def is_last_week_of_rebalancing_month():
    """Check if current date is in the last week of rebalancing months (Feb, May, Aug, Nov)"""
    current_date = datetime.now()
    current_month = current_date.month
    rebalancing_months = [2, 5, 8, 11]  # Feb, May, Aug, Nov
    
    if current_month not in rebalancing_months:
        return False
    
    # Get last day of current month
    last_day = calendar.monthrange(current_date.year, current_month)[1]
    
    # Check if we're in the last 7 days of the month
    days_remaining = last_day - current_date.day
    
    return days_remaining <= 7

def get_next_rebalancing_info():
    """Get information about next rebalancing date"""
    current_date = datetime.now()
    current_month = current_date.month
    rebalancing_months = [2, 5, 8, 11]  # Feb, May, Aug, Nov
    
    # Find next rebalancing month
    next_months = [m for m in rebalancing_months if m > current_month]
    if next_months:
        next_month = next_months[0]
        next_year = current_date.year
    else:
        next_month = rebalancing_months[0]
        next_year = current_date.year + 1
    
    # Calculate last week of next rebalancing month
    last_day = calendar.monthrange(next_year, next_month)[1]
    last_week_start = datetime(next_year, next_month, last_day - 6)
    
    days_until = (last_week_start - current_date).days
    
    month_name = calendar.month_name[next_month]
    
    return {
        'month': month_name,
        'year': next_year,
        'days_until': days_until,
        'last_week_start': last_week_start
    }

def create_rebalancing_alert_message(top_25_stocks: pd.DataFrame):
    """Create rebalancing alert message for Telegram"""
    current_date = datetime.now().strftime('%B %d, %Y')
    
    message = f"🔔 <b>REBALANCING ALERT</b> - {current_date}\n\n"
    message += "📅 <b>Last week of rebalancing month!</b>\n"
    message += "⏰ <b>Time to rebalance your portfolio</b>\n\n"
    
    message += f"🎯 <b>TOP 25 STABLE MOMENTUM STOCKS</b>\n"
    message += f"<i>Based on trailing 12M momentum, stable FIP scores</i>\n\n"
    
    # Add top 10 stocks with key metrics
    message += "🏆 <b>TOP 10 PICKS:</b>\n"
    message += "<pre>"
    for i, (_, stock) in enumerate(top_25_stocks.head(10).iterrows()):
        message += f"{i+1:2d}. {stock['Symbol']:<12} {stock['momentum_score']:6.1f}% (FIP: {stock['fip_score']:6.3f})\n"
    message += "</pre>"
    
    if len(top_25_stocks) > 10:
        message += f"\n... and {len(top_25_stocks) - 10} more stocks\n"
    
    # Add summary stats
    avg_momentum = top_25_stocks['momentum_score'].mean()
    avg_fip = top_25_stocks['fip_score'].mean()
    equal_weight = 100 / len(top_25_stocks)
    
    message += f"\n📈 <b>PORTFOLIO METRICS:</b>\n"
    message += f"• Avg Momentum: <b>{avg_momentum:.2f}%</b>\n"
    message += f"• Avg FIP Score: <b>{avg_fip:.3f}</b>\n"
    message += f"• Equal Weight: <b>{equal_weight:.1f}% per stock</b>\n"
    
    message += f"\n🚨 <b>ACTION REQUIRED:</b>\n"
    message += f"• Review current holdings\n"
    message += f"• Compare with new top 25 list\n"
    message += f"• Prepare rebalancing orders\n"
    message += f"• Execute by month-end\n"
    
    return message

def send_telegram_alert(token: str, chat_id: str, message: str):
    """Send alert via Telegram Bot"""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML'
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            st.success("✅ Alert sent successfully!")
            return True
        else:
            st.error(f"❌ Failed to send alert: {response.text}")
            return False
    except Exception as e:
        st.error(f"❌ Error sending alert: {str(e)}")
        return False

def test_telegram_connection(token: str, chat_id: str):
    """Test Telegram bot connection"""
    url = f"https://api.telegram.org/bot{token}/getMe"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            bot_info = response.json()
            st.success(f"✅ Bot connected: @{bot_info['result']['username']}")
            return True
        else:
            st.error(f"❌ Connection failed: {response.text}")
            return False
    except Exception as e:
        st.error(f"❌ Connection error: {str(e)}")
        return False

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

# Sidebar - Rebalancing Status
st.sidebar.title("🔄 Rebalancing Status")

if is_last_week_of_rebalancing_month():
    st.sidebar.error("🚨 REBALANCING WEEK!")
    st.sidebar.markdown("**Time to rebalance your portfolio**")
    
    current_date = datetime.now()
    month_name = calendar.month_name[current_date.month]
    last_day = calendar.monthrange(current_date.year, current_date.month)[1]
    days_left = last_day - current_date.day
    
    st.sidebar.write(f"📅 **{month_name} {current_date.year}**")
    st.sidebar.write(f"⏰ **{days_left} days left**")
else:
    next_rebalancing = get_next_rebalancing_info()
    st.sidebar.info(f"📅 Next rebalancing:")
    st.sidebar.write(f"**{next_rebalancing['month']} {next_rebalancing['year']}**")
    st.sidebar.write(f"⏰ **{next_rebalancing['days_until']} days to go**")

# Create main tab structure
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Fetch Data", "📋 All Results", "🎯 Top 25 Stable Stocks", "🔔 Alerts", "📈 Backtest"])

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
                file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
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
                    file_name=f"top_25_stable_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("No stocks found with negative FIP scores.")
    else:
        st.info("👆 Please fetch data and run analysis first in the 'Fetch Data' tab.")

with tab4:
    st.header("🔔 Quarterly Rebalancing Alerts")
    
    # Check rebalancing status
    if is_last_week_of_rebalancing_month():
        st.success("📅 **REBALANCING WEEK**: Time to rebalance your portfolio!")
        st.markdown("**🚨 This is the last week of a rebalancing month!**")
    else:
        next_rebalancing = get_next_rebalancing_info()
        st.info(f"📅 **Next Rebalancing**: Last week of {next_rebalancing['month']} {next_rebalancing['year']} ({next_rebalancing['days_until']} days)")
    
    st.divider()
    
    # Telegram Bot Configuration
    st.subheader("🤖 Telegram Bot Setup")
    
    # Instructions
    with st.expander("📖 Setup Instructions"):
        st.markdown("""
        **Step 1: Create Bot**
        1. Open Telegram and search for `@BotFather`
        2. Send `/newbot` command
        3. Choose name: "Your Investment Alert Bot"
        4. Choose username: "your_investment_bot" (must end with 'bot')
        5. Copy the token (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)
        
        **Step 2: Get Your Chat ID**
        1. Search for `@userinfobot` on Telegram
        2. Send any message
        3. Copy your Chat ID (looks like: `123456789`)
        
        **Step 3: Test**
        1. Search for your bot by username
        2. Send `/start` to activate
        3. Configure bot in the app below
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        telegram_token = st.text_input(
            "🔑 Telegram Bot Token",
            type="password",
            placeholder="123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
            help="Get from @BotFather on Telegram"
        )
    
    with col2:
        chat_id = st.text_input(
            "💬 Your Chat ID",
            placeholder="123456789",
            help="Get from @userinfobot on Telegram"
        )
    
    # Test and Save configuration
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧪 Test Connection", type="secondary"):
            if telegram_token and chat_id:
                if test_telegram_connection(telegram_token, chat_id):
                    st.session_state.telegram_config = {
                        'token': telegram_token,
                        'chat_id': chat_id
                    }
            else:
                st.error("Please provide both token and chat ID")
    
    with col2:
        if st.button("💾 Save Configuration", type="primary"):
            if telegram_token and chat_id:
                st.session_state.telegram_config = {
                    'token': telegram_token,
                    'chat_id': chat_id
                }
                st.success("✅ Configuration saved!")
            else:
                st.error("Please provide both token and chat ID")
    
    st.divider()
    
    # Alert System
    st.subheader("📱 Send Rebalancing Alert")
    
    if st.session_state.analysis_complete and st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        stable_stocks = results_df[results_df['fip_score'] < 0]
        
        if len(stable_stocks) > 0:
            top_25 = stable_stocks.head(25)
            
            # Show alert preview
            st.subheader("📋 Alert Preview")
            preview_msg = create_rebalancing_alert_message(top_25)
            st.code(preview_msg, language='text')
            
            # Send alert button
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("📤 Send Alert", type="primary"):
                    if st.session_state.telegram_config:
                        send_telegram_alert(
                            st.session_state.telegram_config['token'],
                            st.session_state.telegram_config['chat_id'],
                            preview_msg
                        )
                    else:
                        st.error("Please configure Telegram bot first")
            
            with col2:
                if st.session_state.telegram_config:
                    st.success("✅ Telegram bot configured and ready")
                else:
                    st.warning("⚠️ Please configure Telegram bot first")
        else:
            st.warning("No stable stocks found. Please run analysis first.")
    else:
        st.info("👆 Please fetch data and run analysis first in the 'Fetch Data' tab.")
    
    st.divider()
    
    # Rebalancing Schedule
    st.subheader("📅 Rebalancing Schedule")
    
    current_year = datetime.now().year
    rebalancing_months = [2, 5, 8, 11]  # Feb, May, Aug, Nov
    
    schedule_data = []
    for month in rebalancing_months:
        if month < datetime.now().month:
            year = current_year + 1
        else:
            year = current_year
        
        month_name = calendar.month_name[month]
        last_day = calendar.monthrange(year, month)[1]
        alert_start = datetime(year, month, last_day - 6)
        alert_end = datetime(year, month, last_day)
        
        schedule_data.append({
            'Month': f"{month_name} {year}",
            'Alert Period': f"{alert_start.strftime('%b %d')} - {alert_end.strftime('%b %d')}",
            'Status': '🔴 Active' if is_last_week_of_rebalancing_month() and month == datetime.now().month else '⏳ Pending'
        })
    
    st.dataframe(schedule_data, use_container_width=True, hide_index=True)

with tab5:
    st.header("📈 Strategy Backtesting & Portfolio Evolution")
    st.markdown("**Test your momentum strategy against historical data with complete portfolio tracking**")
    
    # Backtest Configuration
    st.subheader("⚙️ Backtest Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2021, 1, 1),
            min_value=datetime(2018, 1, 1),
            max_value=datetime.now()
        )
    
    with col2:
        initial_capital = st.number_input(
            "Initial Capital (₹)",
            min_value=100000,
            max_value=10000000,
            value=1000000,
            step=100000,
            format="%d"
        )
    
    with col3:
        fresh_capital_pct = st.number_input(
            "Fresh Capital Every 6M (%)",
            min_value=0,
            max_value=50,
            value=10,
            step=5
        )
    
    st.info("📝 **Note**: Transaction costs are not included in this backtest analysis")
    
    # Initialize backtest session state
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'portfolio_evolution' not in st.session_state:
        st.session_state.portfolio_evolution = None
    
    # Run Backtest Button
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Run Historical Backtest", type="primary", use_container_width=True):
            run_comprehensive_backtest(start_date, initial_capital, fresh_capital_pct)
    
    with col2:
        if st.session_state.backtest_results is not None:
            st.metric("Backtest Status", "✅ Complete")
        else:
            st.metric("Backtest Status", "⏳ Pending")
    
    # Display Results
    if st.session_state.backtest_results is not None:
        display_backtest_results()

def get_rebalancing_dates(start_date, end_date):
    """Get all rebalancing dates (Feb, May, Aug, Nov) between start and end dates"""
    rebalancing_months = [2, 5, 8, 11]  # Feb, May, Aug, Nov
    dates = []
    
    current_year = start_date.year
    end_year = end_date.year
    
    for year in range(current_year, end_year + 1):
        for month in rebalancing_months:
            rebal_date = datetime(year, month, 1)
            
            # Only include dates after start_date and before end_date
            if rebal_date >= datetime(start_date.year, start_date.month, 1) and rebal_date <= end_date:
                dates.append(rebal_date)
    
    return sorted(dates)

def calculate_historical_momentum(symbol, analysis_date):
    """Calculate momentum for a specific historical date"""
    try:
        # Calculate date ranges for historical analysis
        end_date_hist = datetime(analysis_date.year, analysis_date.month, 1) + pd.DateOffset(months=1)
        start_date_hist = end_date_hist - pd.DateOffset(months=13)
        
        # Create month labels for this period
        month_range_hist = pd.date_range(start=start_date_hist, periods=11, freq='MS')
        month_labels_hist = [d.strftime('%b %Y') for d in month_range_hist]
        
        data = yf.download(f"{symbol}.NS", start=start_date_hist, end=end_date_hist, 
                          interval="1mo", progress=False, auto_adjust=True)

        if data.shape[0] < 13 or "Close" not in data.columns:
            return None

        close_prices = data["Close"].dropna()
        monthly_returns = close_prices.pct_change().dropna()

        if isinstance(monthly_returns, pd.DataFrame):
            monthly_returns = monthly_returns.squeeze()

        last_11 = monthly_returns[-12:-1]
        if len(last_11) < 11:
            return None

        returns_dict = {}
        for i in range(11):
            val = float(last_11.iloc[i]) * 100
            returns_dict[month_labels_hist[i]] = val

        # Momentum score
        momentum_score = (last_11 + 1).prod() - 1
        momentum_score = float(momentum_score) * 100
        returns_dict["momentum_score"] = momentum_score

        return returns_dict

    except Exception:
        return None

def get_historical_top25(analysis_date, stock_universe):
    """Get top 25 stocks for a specific historical date"""
    results_list = []
    
    for _, row in stock_universe.iterrows():
        symbol = row['Symbol'].replace('.NS', '')
        company_name = row['Company Name']
        
        momentum_data = calculate_historical_momentum(symbol, analysis_date)
        
        if momentum_data is not None:
            # Calculate FIP
            month_labels_hist = [k for k in momentum_data.keys() if k != 'momentum_score']
            fip_data = calculate_fip(pd.Series(momentum_data), month_labels_hist)
            
            result = {
                'Company Name': company_name,
                'Symbol': symbol,
                **momentum_data,
                **fip_data
            }
            results_list.append(result)
    
    if results_list:
        results_df = pd.DataFrame(results_list)
        
        # Filter stable stocks (negative FIP)
        stable_stocks = results_df[results_df['fip_score'] < 0]
        
        if len(stable_stocks) > 0:
            # Get top 25 by momentum score
            top_25 = stable_stocks.sort_values('momentum_score', ascending=False).head(25)
            return top_25['Symbol'].tolist()
    
    return []

def calculate_forward_returns(stocks, start_date, months=3):
    """Calculate forward returns for a list of stocks"""
    returns = {}
    
    end_date = start_date + pd.DateOffset(months=months)
    
    for symbol in stocks:
        try:
            data = yf.download(f"{symbol}.NS", start=start_date, end=end_date, 
                             interval="1d", progress=False, auto_adjust=True)
            
            if len(data) > 0 and "Close" in data.columns:
                start_price = data["Close"].iloc[0]
                end_price = data["Close"].iloc[-1]
                
                if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                    return_pct = (end_price / start_price - 1) * 100
                    returns[symbol] = return_pct
                else:
                    returns[symbol] = 0.0
            else:
                returns[symbol] = 0.0
                
        except Exception:
            returns[symbol] = 0.0
    
    return returns

def get_benchmark_returns(start_date, end_date):
    """Get benchmark returns for Nifty 100 and Nifty Midcap 150"""
    benchmarks = {}
    
    # Nifty 100 (using ^CNX100 or similar proxy)
    try:
        nifty100_data = yf.download("^CNX100", start=start_date, end=end_date, 
                                   interval="1d", progress=False, auto_adjust=True)
        if len(nifty100_data) > 0:
            start_price = nifty100_data["Close"].iloc[0]
            end_price = nifty100_data["Close"].iloc[-1]
            benchmarks['Nifty 100'] = (end_price / start_price - 1) * 100
    except:
        # Fallback: estimate based on historical averages
        benchmarks['Nifty 100'] = 12.0  # Approximate annual return
    
    # Nifty Midcap 150 proxy
    try:
        # Use CNX Midcap or estimate
        benchmarks['Nifty Midcap 150'] = 15.0  # Approximate annual return
    except:
        benchmarks['Nifty Midcap 150'] = 15.0
    
    return benchmarks

def run_comprehensive_backtest(start_date, initial_capital, fresh_capital_pct):
    """Run the complete backtesting analysis"""
    
    with st.spinner("Running comprehensive backtest analysis..."):
        
        # Get stock universe (you can modify this to use your actual data source)
        try:
            stock_universe = fetch_stock_data_cache("combined")
        except:
            st.error("Unable to fetch stock universe. Please ensure you have run the analysis first.")
            return
        
        # Get all rebalancing dates
        end_date = datetime.now()
        rebalancing_dates = get_rebalancing_dates(start_date, end_date)
        
        if len(rebalancing_dates) == 0:
            st.error("No rebalancing dates found in the selected period.")
            return
        
        # Initialize tracking variables
        portfolio_value = initial_capital
        portfolio_history = []
        portfolio_evolution = []
        stock_performance = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, rebal_date in enumerate(rebalancing_dates):
            status_text.text(f'Processing {rebal_date.strftime("%B %Y")} rebalancing... ({i+1}/{len(rebalancing_dates)})')
            progress_bar.progress((i + 1) / len(rebalancing_dates))
            
            # Get top 25 stocks for this date
            top_25_stocks = get_historical_top25(rebal_date, stock_universe)
            
            if len(top_25_stocks) == 0:
                continue
            
            # Add fresh capital every 6 months (alternate rebalances)
            if i > 0 and i % 2 == 0:  # Every 2nd rebalance = 6 months
                fresh_capital = initial_capital * (fresh_capital_pct / 100)
                portfolio_value += fresh_capital
            
            # Calculate allocation per stock (equal weight)
            allocation_per_stock = portfolio_value / len(top_25_stocks)
            
            # Calculate 3-month forward returns
            forward_returns = calculate_forward_returns(top_25_stocks, rebal_date, 3)
            
            # Calculate portfolio return for this quarter
            quarter_return = sum(forward_returns.values()) / len(forward_returns) if forward_returns else 0
            new_portfolio_value = portfolio_value * (1 + quarter_return / 100)
            
            # Track portfolio evolution
            portfolio_evolution.append({
                'Date': rebal_date,
                'Stocks': top_25_stocks.copy(),
                'Portfolio_Value': portfolio_value,
                'Quarter_Return': quarter_return,
                'New_Portfolio_Value': new_portfolio_value
            })
            
            # Track individual stock performance
            for stock in top_25_stocks:
                if stock not in stock_performance:
                    stock_performance[stock] = {
                        'periods': [],
                        'returns': [],
                        'total_quarters': 0,
                        'total_return': 0
                    }
                
                stock_return = forward_returns.get(stock, 0)
                stock_performance[stock]['periods'].append(rebal_date)
                stock_performance[stock]['returns'].append(stock_return)
                stock_performance[stock]['total_quarters'] += 1
                
                # Calculate cumulative return for this stock
                if len(stock_performance[stock]['returns']) == 1:
                    stock_performance[stock]['total_return'] = stock_return
                else:
                    prev_total = stock_performance[stock]['total_return']
                    stock_performance[stock]['total_return'] = (1 + prev_total/100) * (1 + stock_return/100) - 1
                    stock_performance[stock]['total_return'] *= 100
            
            # Update portfolio value
            portfolio_value = new_portfolio_value
            
            # Add to history
            portfolio_history.append({
                'Date': rebal_date,
                'Portfolio_Value': portfolio_value,
                'Quarter_Return': quarter_return,
                'Cumulative_Return': (portfolio_value / initial_capital - 1) * 100
            })
        
        progress_bar.empty()
        status_text.empty()
        
        # Calculate final metrics
        if len(portfolio_history) > 0:
            total_return = (portfolio_value / initial_capital - 1) * 100
            years = len(portfolio_history) / 4  # 4 quarters per year
            cagr = (portfolio_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0
            cagr *= 100
            
            # Store results in session state
            st.session_state.backtest_results = {
                'portfolio_history': portfolio_history,
                'portfolio_evolution': portfolio_evolution,
                'stock_performance': stock_performance,
                'initial_capital': initial_capital,
                'final_value': portfolio_value,
                'total_return': total_return,
                'cagr': cagr,
                'years': years,
                'quarters': len(portfolio_history)
            }
            
            st.success(f"✅ Backtest complete! Analyzed {len(portfolio_history)} quarters.")
            st.balloons()
        else:
            st.error("No data found for backtesting. Please check your date range.")

def display_backtest_results():
    """Display comprehensive backtest results"""
    
    results = st.session_state.backtest_results
    
    # Performance Summary
    st.subheader("📊 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Initial Capital",
            f"₹{results['initial_capital']:,.0f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Final Value",
            f"₹{results['final_value']:,.0f}",
            delta=f"₹{results['final_value'] - results['initial_capital']:,.0f}"
        )
    
    with col3:
        st.metric(
            "Total Return",
            f"{results['total_return']:.1f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            "CAGR",
            f"{results['cagr']:.1f}%",
            delta=None
        )
    
    # Portfolio Value Chart
    st.subheader("📈 Portfolio Value Over Time")
    
    if len(results['portfolio_history']) > 0:
        df_history = pd.DataFrame(results['portfolio_history'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_history['Date'],
            y=df_history['Portfolio_Value'],
            mode='lines+markers',
            name='Your Strategy',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ))
        
        # Add benchmark comparison (simplified)
        benchmark_values = [results['initial_capital'] * (1.12 ** (i/4)) for i in range(len(df_history))]
        fig.add_trace(go.Scatter(
            x=df_history['Date'],
            y=benchmark_values,
            mode='lines',
            name='Nifty 100 (~12% CAGR)',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Portfolio Value Growth",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (₹)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Quarterly Returns
    st.subheader("📊 Quarterly Returns")
    
    if len(results['portfolio_history']) > 0:
        df_history = pd.DataFrame(results['portfolio_history'])
        
        fig = go.Figure(data=go.Bar(
            x=df_history['Date'],
            y=df_history['Quarter_Return'],
            marker_color=['green' if x > 0 else 'red' for x in df_history['Quarter_Return']],
            name='Quarterly Returns'
        ))
        
        fig.update_layout(
            title="Quarterly Performance",
            xaxis_title="Quarter",
            yaxis_title="Return (%)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio Evolution
    st.subheader("🔄 Portfolio Evolution")
    
    display_portfolio_evolution()
    
    # Stock Performance Analysis
    st.subheader("🏆 Individual Stock Performance")
    
    display_stock_performance()

def display_portfolio_evolution():
    """Display how portfolio composition changed over time"""
    
    results = st.session_state.backtest_results
    evolution = results['portfolio_evolution']
    
    if len(evolution) == 0:
        st.warning("No portfolio evolution data available.")
        return
    
    # Create evolution summary
    evolution_data = []
    prev_stocks = set()
    
    for i, period in enumerate(evolution):
        current_stocks = set(period['Stocks'])
        
        if i == 0:
            added = current_stocks
            removed = set()
            continued = set()
        else:
            added = current_stocks - prev_stocks
            removed = prev_stocks - current_stocks
            continued = current_stocks & prev_stocks
        
        evolution_data.append({
            'Date': period['Date'].strftime('%b %Y'),
            'Portfolio_Value': f"₹{period['Portfolio_Value']:,.0f}",
            'Quarter_Return': f"{period['Quarter_Return']:.1f}%",
            'Added': len(added),
            'Removed': len(removed),
            'Continued': len(continued),
            'Turnover': f"{(len(added) + len(removed)) / 25 * 100:.1f}%" if i > 0 else "N/A"
        })
        
        prev_stocks = current_stocks
    
    df_evolution = pd.DataFrame(evolution_data)
    st.dataframe(df_evolution, use_container_width=True, hide_index=True)
    
    # Portfolio Stability Metrics
    if len(evolution) > 1:
        avg_turnover = np.mean([float(x['Turnover'].replace('%', '')) for x in evolution_data[1:] if x['Turnover'] != 'N/A'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Turnover per Quarter", f"{avg_turnover:.1f}%")
        
        with col2:
            total_quarters = len(evolution)
            st.metric("Total Rebalances", f"{total_quarters}")
        
        with col3:
            avg_added = np.mean([x['Added'] for x in evolution_data[1:]])
            st.metric("Avg Stocks Changed", f"{avg_added:.1f}")

def display_stock_performance():
    """Display individual stock performance analysis"""
    
    results = st.session_state.backtest_results
    stock_perf = results['stock_performance']
    
    if len(stock_perf) == 0:
        st.warning("No stock performance data available.")
        return
    
    # Create stock performance summary
    stock_summary = []
    
    for stock, data in stock_perf.items():
        quarters_held = data['total_quarters']
        total_return = data['total_return']
        avg_return = np.mean(data['returns']) if data['returns'] else 0
        
        # Calculate contribution to portfolio (assuming equal weight)
        portfolio_contribution = total_return * (1/25)  # 4% weight per stock
        
        stock_summary.append({
            'Symbol': stock,
            'Quarters_Held': quarters_held,
            'Total_Return': total_return,
            'Avg_Quarterly_Return': avg_return,
            'Portfolio_Contribution': portfolio_contribution,
            'Consistency': quarters_held / results['quarters'] * 100
        })
    
    df_stocks = pd.DataFrame(stock_summary)
    df_stocks = df_stocks.sort_values('Portfolio_Contribution', ascending=False)
    
    # Format for display
    df_display = df_stocks.copy()
    df_display['Total_Return'] = df_display['Total_Return'].apply(lambda x: f"{x:.1f}%")
    df_display['Avg_Quarterly_Return'] = df_display['Avg_Quarterly_Return'].apply(lambda x: f"{x:.1f}%")
    df_display['Portfolio_Contribution'] = df_display['Portfolio_Contribution'].apply(lambda x: f"{x:.2f}%")
    df_display['Consistency'] = df_display['Consistency'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(
        df_display.rename(columns={
            'Quarters_Held': 'Quarters',
            'Total_Return': 'Total Return',
            'Avg_Quarterly_Return': 'Avg Q Return',
            'Portfolio_Contribution': 'Portfolio Impact',
            'Consistency': 'Consistency %'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Top/Bottom Performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Contributors")
        top_performers = df_stocks.head(5)
        for _, stock in top_performers.iterrows():
            st.write(f"**{stock['Symbol']}**: +{stock['Portfolio_Contribution']:.2f}% contribution ({stock['Quarters_Held']} quarters)")
    
    with col2:
        st.subheader("📉 Worst Performers")
        bottom_performers = df_stocks.tail(5)
        for _, stock in bottom_performers.iterrows():
            st.write(f"**{stock['Symbol']}**: {stock['Portfolio_Contribution']:.2f}% contribution ({stock['Quarters_Held']} quarters)")

# Add plotly to requirements.txt:
# plotly>=5.0.0


