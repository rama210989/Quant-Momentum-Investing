# Stock Momentum & FIP Analysis

A Streamlit application for analyzing stock momentum and FIP (Frequency of Positive Returns) scores for Indian stocks.

## 🚀 Features

- **Momentum Analysis**: Calculate 12-month rolling momentum scores
- **FIP Scoring**: Frequency of Positive Returns analysis for stability
- **Top 25 Stable Stocks**: Identify stocks with negative FIP scores and high momentum
- **Quarterly Rebalancing Alerts**: Telegram integration for rebalancing reminders
- **Data Export**: Download results as Excel files
- **Mobile Responsive**: Optimized for mobile and desktop use

## 📊 Analysis Methodology

### Momentum Score
- Calculates 12-month rolling momentum using monthly returns
- Uses the formula: `(1 + r1) * (1 + r2) * ... * (1 + r11) - 1`
- Expressed as a percentage

### FIP Score
- Frequency of Positive Returns analysis
- Calculates percentage of positive vs negative months
- Formula: `sign(momentum) * (pct_negative - pct_positive)`
- Negative FIP scores indicate more stable performance

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-momentum-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Supabase (Optional)**
   ```bash
   cp .env.example .env
   # Edit .env with your Supabase credentials
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📁 Data Sources

### Option 1: Upload CSV File
Upload a CSV file with the following columns:
- `Symbol`: Stock symbol (e.g., RELIANCE, TCS)
- `Company Name`: Full company name
- `ISIN Code`: ISIN code for the stock

### Option 2: Use Sample Data
The application includes sample NIFTY 500 data in `data/nifty500.csv`

### Option 3: Supabase Database (Advanced)
Configure Supabase credentials in `.env` file for database integration.

## 🎯 Usage

1. **Data Input Tab**: Choose data source and upload/load stock data
2. **All Results Tab**: View complete analysis results
3. **Top 25 Stable Stocks Tab**: Focus on stocks with negative FIP scores
4. **Alerts Tab**: Configure Telegram alerts for rebalancing

## 🔔 Rebalancing Schedule

The application tracks quarterly rebalancing periods:
- **February**: Last week of the month
- **May**: Last week of the month  
- **August**: Last week of the month
- **November**: Last week of the month

## 📱 Telegram Integration

1. Create a bot with @BotFather on Telegram
2. Get your Chat ID from @userinfobot
3. Configure in the Alerts tab
4. Receive automatic rebalancing alerts

## 🛠️ Recent Changes & Fixes

### What Was Fixed:
1. **Removed Complex Database Logic**: Simplified data handling to focus on CSV uploads and sample data
2. **Eliminated Backtesting Code**: Removed all backtesting-related functionality as requested
3. **Improved Error Handling**: Better error messages and fallback options
4. **Simplified Code Structure**: Reduced app.py from 700+ lines to cleaner, more maintainable code
5. **Enhanced Data Validation**: Better CSV validation and sample data loading
6. **Preserved Core Logic**: All momentum and FIP calculation logic remains unchanged

### Key Improvements:
- **Cleaner UI**: Simplified tab structure and better user guidance
- **Better Data Flow**: Clear separation between data input and analysis
- **Improved Documentation**: Better help text and instructions
- **Mobile Optimization**: Enhanced responsive design
- **Environment Configuration**: Support for .env files for Supabase credentials

## 📈 Analysis Period

The application analyzes the trailing 12 months of data:
- **Start Date**: 13 months ago from current month
- **End Date**: Current month
- **Analysis Window**: 11 months of returns (excluding current month)

## 🎨 UI Features

- **Progress Tracking**: Real-time progress bars during analysis
- **Summary Metrics**: Key statistics displayed prominently
- **Interactive Tables**: Sortable and downloadable results
- **Status Indicators**: Clear visual feedback for all operations
- **Responsive Design**: Works on mobile and desktop

## 🔒 Data Privacy

- All data processing happens locally
- No stock data is stored permanently
- Supabase integration is optional and configurable
- CSV uploads are processed in memory only

## 🐛 Troubleshooting

### Common Issues:

1. **"No valid results found"**
   - Check that stock symbols are correct (without .NS suffix)
   - Ensure sufficient historical data is available
   - Verify CSV format matches requirements

2. **"Database connection failed"**
   - Use CSV upload option instead
   - Check Supabase credentials if using database
   - Verify network connectivity

3. **"Analysis taking too long"**
   - Reduce number of stocks in analysis
   - Check internet connection for data fetching
   - Consider using sample data for testing

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your data format matches requirements
3. Test with sample data first
4. Review error messages for specific guidance

## 📄 License

This project is for educational and research purposes. Please ensure compliance with data usage terms and regulations.