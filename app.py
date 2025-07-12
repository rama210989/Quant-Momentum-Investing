import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import io
import xlsxwriter
from typing import Optional, Dict, Any, List
import requests
import json
import calendar
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the new Supabase-powered stock data fetcher
try:
    from stock_data_fetcher import fetch_stock_data_cache, show_database_management
    SUPABASE_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Error importing stock_data_fetcher: {e}")
    st.error("💡 Please ensure you have the new Supabase-powered stock_data_fetcher.py")
    SUPABASE_AVAILABLE = False

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

# Database status check
if SUPABASE_AVAILABLE:
    try:
        # Test if we can fetch data
        test_data = fetch_stock_data_cache("combined")
        if len(test_data) > 0:
            st.success(f"✅ Connected to database: {len(test_data)} stocks available")
        else:
            st.warning("⚠️ Database connected but no stocks found. Please populate the database.")
    except Exception as e:
        st.error(f"❌ Database connection issue: {e}")
        st.error("💡 **Solution**: Either fix the database connection or upload a CSV file")

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Fetch Data", "📋 All Results", "🎯 Top 25 Stable Stocks", "🔔 Alerts", "💾 Database"])

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
        # Fetch from Supabase database
        if st.button("🔄 Fetch Current Index Data", type="primary"):
            if not SUPABASE_AVAILABLE:
                st.error("❌ Supabase connection not available. Please upload a CSV file instead.")
            else:
                try:
                    with st.spinner("Fetching latest stock data from database..."):
                        if data_source == "NIFTY 100 + NIFTY MIDCAP 150 (Combined)":
                            stock_data = fetch_stock_data_cache("combined")
                        elif data_source == "NIFTY 100 Only":
                            stock_data = fetch_stock_data_cache("nifty100")
                        elif data_source == "NIFTY MIDCAP 150 Only":
                            stock_data = fetch_stock_data_cache("midcap150")
                        
                        st.session_state.stock_data = stock_data
                        st.success(f"✅ Fetched {len(stock_data)} stocks from database")
                        
                        # Display sample data
                        st.subheader("Sample Stock Data")
                        st.dataframe(stock_data.head(10), use_container_width=True)
                        
                        # Show summary statistics
                        st.subheader("Data Summary")
                        st.write(f"**Total Stocks:** {len(stock_data)}")
                        st.write(f"**Unique Companies:** {stock_data['Company Name'].nunique()}")
                        st.write(f"**Data Source:** Database ({data_source})")
                        
                except Exception as e:
                    st.error(f"❌ Error fetching data from database: {e}")
                    st.error("💡 **Solution**: Please upload a CSV file or check your database connection")
    
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
    st.header("💾 Database Management")
    
    if SUPABASE_AVAILABLE:
        # Show database management interface
        show_database_management()
        
        st.divider()
        
        # CSV Import Instructions
        st.subheader("📂 Import CSV Data to Database")
        
        st.info("""
        **To populate your database with stock data:**
        
        1. **Run the CSV importer in Colab:**
           - Upload your CSV files (MW-NIFTY-100-12-Jul-2025.csv, MW-NIFTY-MIDCAP-150-12-Jul-2025.csv)
           - Run the csv_importer.py script
           - This will populate your Supabase database
        
        2. **Expected result:**
           - ~95 Nifty 100 stocks
           - ~145 Nifty Midcap 150 stocks
           - Total ~240 stocks in database
        
        3. **After import:**
           - Your app will automatically use database data
           - No more hardcoded stock lists
           - Monthly updates possible
        """)
        
        # Test database connection
        if st.button("🧪 Test Database Connection"):
            try:
                test_data = fetch_stock_data_cache("combined")
                st.success(f"✅ Database connection successful: {len(test_data)} stocks available")
                
                # Show sample data
                if len(test_data) > 0:
                    st.subheader("Sample Database Data")
                    st.dataframe(test_data.head(5), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Database connection failed: {e}")
                st.error("💡 Please run the CSV importer in Colab first")
        
        # CSV Import Helper
        st.subheader("🚀 Quick CSV Import Guide")
        
        with st.expander("📋 Step-by-Step Colab Instructions"):
            st.code("""
# 1. In Google Colab, upload your CSV files
from google.colab import files
uploaded = files.upload()
# Upload: MW-NIFTY-100-12-Jul-2025.csv and MW-NIFTY-MIDCAP-150-12-Jul-2025.csv

# 2. Install dependencies
!pip install supabase pandas

# 3. Copy the csv_importer.py script to Colab and update file paths:
nifty100_csv = "MW-NIFTY-100-12-Jul-2025.csv"
midcap150_csv = "MW-NIFTY-MIDCAP-150-12-Jul-2025.csv"

# 4. Run the script
python csv_importer.py

# Expected output:
# ✅ NIFTY100: 95 stocks
# ✅ NIFTY_MIDCAP_150: 145 stocks
# 🎉 Total stocks imported: 240
            """, language='python')
        
    else:
        st.error("❌ Supabase connection not available")
        st.error("💡 Please check your stock_data_fetcher.py configuration")
        
        st.subheader("🔧 Troubleshooting")
        st.info("""
        **If you see this error:**
        
        1. **Check your stock_data_fetcher.py:**
           - Make sure you have the new Supabase version
           - Update the SUPABASE_URL and SUPABASE_KEY
        
        2. **Install dependencies:**
           ```bash
           pip install supabase
           ```
        
        3. **Alternative: Use CSV Upload**
           - Go to "Fetch Data" tab
           - Select "Upload CSV File"
           - Upload your stock list manually
        """)

# Add some helpful information at the bottom
st.divider()
st.markdown("---")

# Status indicator
col1, col2, col3 = st.columns(3)

with col1:
    if SUPABASE_AVAILABLE:
        st.success("✅ Database Ready")
    else:
        st.error("❌ Database Issue")

with col2:
    if st.session_state.stock_data is not None:
        st.success(f"✅ Data Loaded ({len(st.session_state.stock_data)} stocks)")
    else:
        st.warning("⏳ No Data Loaded")

with col3:
    if st.session_state.analysis_complete:
        st.success("✅ Analysis Complete")
    else:
        st.info("⏳ Analysis Pending")

# Help section
with st.expander("❓ Need Help?"):
    st.markdown("""
    **Current Status & Next Steps:**
    
    1. **If Database is Ready:** Click "Fetch Current Index Data" to get ~240 stocks from database
    2. **If Database Issue:** Use "Upload CSV File" option as backup
    3. **After Data Loaded:** Click "Run Momentum & FIP Analysis" 
    4. **View Results:** Check "All Results" and "Top 25 Stable Stocks" tabs
    
    **Troubleshooting:**
    - Database issues? Run the CSV importer in Colab first
    - No data? Try uploading CSV manually
    - Analysis errors? Check if stock symbols are valid
    
    **For Supabase Setup:**
    - Run the SQL schema in your Supabase dashboard
    - Use CSV importer to populate data
    - Update credentials in stock_data_fetcher.py
    """)

st.markdown("---")
st.caption("🚀 Powered by Supabase Database | 📈 Gray & Vogel Momentum Strategy")
