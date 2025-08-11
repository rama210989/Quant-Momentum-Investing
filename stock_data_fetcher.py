"""
Google Sheets-Powered Stock Data Fetcher
========================================
Fetches stock data from Google Sheets with CSV upload fallback.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import streamlit as st
import logging
from datetime import datetime
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Sheets configuration
GOOGLE_SHEET_ID = os.getenv('GOOGLE_SHEET_ID', '')
CREDENTIALS_FILE = 'google_credentials.json'

class GoogleSheetsStockFetcher:
    def __init__(self):
        """Initialize Google Sheets client"""
        self.gc = None
        self.sheet = None
        self.connection_status = "not_initialized"
        
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            
            # Try to load credentials
            credentials_path = self._find_credentials_file()
            if not credentials_path:
                self.connection_status = "credentials_missing"
                logger.error("❌ Google credentials file not found")
                return
            
            # Set up credentials
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
            self.gc = gspread.authorize(creds)
            
            # Try to open the sheet
            sheet_id = self._get_sheet_id()
            if not sheet_id:
                self.connection_status = "sheet_id_missing"
                logger.error("❌ Google Sheet ID not configured")
                return
            
            self.sheet = self.gc.open_by_key(sheet_id).sheet1
            self._test_connection()
            self.connection_status = "connected"
            logger.info("✅ Google Sheets client initialized successfully")
            
        except ImportError:
            logger.error("❌ gspread library not installed. Run: pip install gspread google-auth")
            self.connection_status = "library_missing"
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Sheets client: {e}")
            self.connection_status = "connection_failed"
    
    def _find_credentials_file(self) -> Optional[str]:
        """Find Google credentials file"""
        possible_paths = [
            CREDENTIALS_FILE,
            f'./{CREDENTIALS_FILE}',
            f'../{CREDENTIALS_FILE}',
            os.path.expanduser(f'~/{CREDENTIALS_FILE}')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found credentials file: {path}")
                return path
        
        # Check if credentials are in Streamlit secrets
        try:
            if hasattr(st, 'secrets') and 'google_credentials' in st.secrets:
                # Write secrets to temporary file
                with open(CREDENTIALS_FILE, 'w') as f:
                    json.dump(dict(st.secrets['google_credentials']), f)
                return CREDENTIALS_FILE
        except Exception as e:
            logger.warning(f"Could not load from Streamlit secrets: {e}")
        
        return None
    
    def _get_sheet_id(self) -> Optional[str]:
        """Get Google Sheet ID from environment or Streamlit secrets"""
        # Try environment variable first
        sheet_id = os.getenv('GOOGLE_SHEET_ID')
        if sheet_id:
            return sheet_id
        
        # Try Streamlit secrets
        try:
            if hasattr(st, 'secrets') and 'GOOGLE_SHEET_ID' in st.secrets:
                return st.secrets['GOOGLE_SHEET_ID']
        except Exception as e:
            logger.warning(f"Could not load sheet ID from secrets: {e}")
        
        return None
    
    def _test_connection(self):
        """Test Google Sheets connection"""
        if not self.sheet:
            raise Exception("Google Sheets not initialized")
        
        try:
            # Try to read first row to test connection
            values = self.sheet.row_values(1)
            if not values:
                logger.warning("⚠️ Sheet appears to be empty")
            logger.info("✅ Google Sheets connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ Google Sheets connection test failed: {e}")
            raise e
    
    def fetch_stocks_by_type(self, stock_type: str = 'combined') -> pd.DataFrame:
        """
        Fetch stocks from Google Sheets
        
        Args:
            stock_type: 'nifty100', 'midcap150', or 'combined'
        
        Returns:
            DataFrame with columns: Company Name, Symbol, ISIN Code
        """
        if not self.sheet or self.connection_status != "connected":
            raise Exception(f"Google Sheets not available: {self.connection_status}")
        
        try:
            # Get all data from sheet
            all_values = self.sheet.get_all_values()
            
            if not all_values or len(all_values) < 2:
                logger.warning("No data found in Google Sheet")
                return pd.DataFrame(columns=['Company Name', 'Symbol', 'ISIN Code'])
            
            # Convert to DataFrame
            headers = all_values[0]
            data = all_values[1:]
            
            df = pd.DataFrame(data, columns=headers)
            
            # Filter by stock type
            if stock_type == 'nifty100':
                df = df[df['Index Type'].str.upper() == 'NIFTY100']
            elif stock_type == 'midcap150':
                df = df[df['Index Type'].str.upper().isin(['MIDCAP150', 'NIFTY_MIDCAP_150'])]
            elif stock_type == 'combined':
                df = df[df['Index Type'].str.upper().isin(['NIFTY100', 'MIDCAP150', 'NIFTY_MIDCAP_150'])]
            
            # Filter active stocks
            if 'Is Active' in df.columns:
                df = df[df['Is Active'].str.upper().isin(['TRUE', 'YES', '1'])]
            
            # Clean and validate data
            if 'Symbol' in df.columns:
                df['Symbol'] = df['Symbol'].str.replace('.NS', '', regex=False)
                df = df[df['Symbol'].str.len() > 0]  # Remove empty symbols
            
            # Ensure all required columns exist
            required_columns = ['Company Name', 'Symbol', 'ISIN Code']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ''
            
            # Filter out rows with empty symbols or company names
            df = df[(df['Symbol'].str.len() > 0) & (df['Company Name'].str.len() > 0)]
            
            logger.info(f"Fetched {len(df)} stocks for {stock_type}")
            return df[required_columns].reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error fetching stocks from Google Sheets: {e}")
            raise e
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the Google Sheet data"""
        if not self.sheet or self.connection_status != "connected":
            return {'database_status': 'error', 'error': f'Not connected: {self.connection_status}'}
        
        try:
            # Get all data
            all_values = self.sheet.get_all_values()
            
            if not all_values or len(all_values) < 2:
                return {
                    'nifty100_count': 0,
                    'midcap150_count': 0,
                    'total_count': 0,
                    'database_status': 'empty',
                    'connection_status': self.connection_status
                }
            
            # Convert to DataFrame for analysis
            headers = all_values[0]
            data = all_values[1:]
            df = pd.DataFrame(data, columns=headers)
            
            # Filter active stocks
            if 'Is Active' in df.columns:
                df = df[df['Is Active'].str.upper().isin(['TRUE', 'YES', '1'])]
            
            # Count by index type
            nifty100_count = len(df[df['Index Type'].str.upper() == 'NIFTY100'])
            midcap150_count = len(df[df['Index Type'].str.upper().isin(['MIDCAP150', 'NIFTY_MIDCAP_150'])])
            
            return {
                'nifty100_count': nifty100_count,
                'midcap150_count': midcap150_count,
                'total_count': nifty100_count + midcap150_count,
                'database_status': 'healthy',
                'connection_status': self.connection_status,
                'total_rows': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'database_status': 'error', 'error': str(e), 'connection_status': self.connection_status}
    
    def update_stock_data(self, stocks_df: pd.DataFrame) -> bool:
        """
        Update Google Sheet with new stock data
        
        Args:
            stocks_df: DataFrame with stock data to upload
        
        Returns:
            bool: Success status
        """
        if not self.sheet or self.connection_status != "connected":
            logger.error(f"Cannot update: Google Sheets not available: {self.connection_status}")
            return False
        
        try:
            # Prepare data for upload
            required_columns = ['Company Name', 'Symbol', 'ISIN Code', 'Index Type', 'Is Active']
            
            # Ensure all required columns exist
            for col in required_columns:
                if col not in stocks_df.columns:
                    stocks_df[col] = 'TRUE' if col == 'Is Active' else ''
            
            # Convert to list of lists for gspread
            headers = required_columns
            values = [headers] + stocks_df[required_columns].fillna('').astype(str).values.tolist()
            
            # Clear existing data and update
            self.sheet.clear()
            self.sheet.update('A1', values)
            
            logger.info(f"✅ Updated Google Sheet with {len(stocks_df)} stocks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating Google Sheet: {e}")
            return False

# Initialize fetcher with proper error handling
fetcher = None
DATABASE_AVAILABLE = False

try:
    fetcher = GoogleSheetsStockFetcher()
    if fetcher.connection_status == "connected":
        DATABASE_AVAILABLE = True
        logger.info("✅ Google Sheets connection established")
    else:
        logger.warning(f"⚠️ Google Sheets not available: {fetcher.connection_status}")
        DATABASE_AVAILABLE = False
except Exception as e:
    logger.error(f"❌ Failed to create fetcher: {e}")
    DATABASE_AVAILABLE = False

def validate_csv(df: pd.DataFrame) -> bool:
    """Validate uploaded CSV file format"""
    required_columns = ['Symbol', 'Company Name', 'ISIN Code']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        st.info("Required columns: Symbol, Company Name, ISIN Code")
        return False
    
    if df.empty:
        st.error("❌ CSV file is empty")
        return False
    
    # Check for empty symbols
    empty_symbols = df['Symbol'].isna().sum() + (df['Symbol'] == '').sum()
    if empty_symbols > 0:
        st.warning(f"⚠️ Found {empty_symbols} rows with empty symbols - these will be filtered out")
    
    return True

@st.cache_data(ttl=3600)
def fetch_stock_data_cache(source: str = "combined") -> pd.DataFrame:
    """
    Cached function to fetch stock data from Google Sheets
    Falls back to requiring CSV upload if sheets unavailable
    """
    if not DATABASE_AVAILABLE or not fetcher:
        st.error("❌ Google Sheets connection not available")
        st.info("💡 Please upload a CSV file with your stock data")
        raise Exception("Google Sheets not available - CSV upload required")
    
    try:
        if source == "combined":
            return fetcher.fetch_stocks_by_type('combined')
        elif source == "nifty100":
            return fetcher.fetch_stocks_by_type('nifty100')
        elif source == "midcap150":
            return fetcher.fetch_stocks_by_type('midcap150')
        else:
            return fetcher.fetch_stocks_by_type('combined')
            
    except Exception as e:
        st.error(f"❌ Error fetching from Google Sheets: {e}")
        st.info("💡 Please upload a CSV file with your stock data")
        raise e

def show_database_status():
    """Show current Google Sheets connection status"""
    if DATABASE_AVAILABLE and fetcher:
        stats = fetcher.get_database_stats()
        
        if stats.get('database_status') == 'healthy':
            st.success("✅ Google Sheets Connected")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("NIFTY 100", stats.get('nifty100_count', 0))
            with col2:
                st.metric("MIDCAP 150", stats.get('midcap150_count', 0))
            with col3:
                st.metric("Total Stocks", stats.get('total_count', 0))
                
        elif stats.get('database_status') == 'empty':
            st.warning("⚠️ Google Sheet is empty")
            st.info("💡 Please add stock data to your Google Sheet or run the data scraper")
        else:
            st.error(f"❌ Google Sheets Error: {stats.get('error', 'Unknown error')}")
            
    else:
        if fetcher:
            if fetcher.connection_status == "library_missing":
                st.error("❌ Required libraries not installed")
                st.code("pip install gspread google-auth", language="bash")
            elif fetcher.connection_status == "credentials_missing":
                st.error("❌ Google credentials file not found")
                st.info("💡 Please add google_credentials.json to your project or configure Streamlit secrets")
            elif fetcher.connection_status == "sheet_id_missing":
                st.error("❌ Google Sheet ID not configured")
                st.info("💡 Please set GOOGLE_SHEET_ID environment variable or Streamlit secret")
            elif fetcher.connection_status == "connection_failed":
                st.error("❌ Google Sheets connection failed")
                st.info("💡 Check your credentials and sheet permissions")
            else:
                st.warning(f"⚠️ Google Sheets not available: {fetcher.connection_status}")
        else:
            st.warning("⚠️ Google Sheets not configured")

def clear_cache():
    """Clear the Streamlit cache"""
    try:
        st.cache_data.clear()
        st.success("✅ Cache cleared")
        logger.info("Cache cleared successfully")
    except Exception as e:
        st.error(f"❌ Error clearing cache: {e}")
        logger.error(f"Error clearing cache: {e}")

# Test function for development
def test_google_sheets_connection():
    """Test the Google Sheets connection and data retrieval"""
    print("🧪 Testing Google Sheets stock fetcher...")
    
    if not fetcher:
        print("❌ Failed to create fetcher")
        return False
    
    if not DATABASE_AVAILABLE:
        print(f"❌ Google Sheets not available: {fetcher.connection_status}")
        return False
    
    try:
        print("\n1. Testing connection...")
        fetcher._test_connection()
        print("✅ Connection OK")
        
        print("\n2. Testing database stats...")
        stats = fetcher.get_database_stats()
        print(f"✅ Stats: {stats}")
        
        print("\n3. Testing data fetch...")
        combined = fetcher.fetch_stocks_by_type('combined')
        print(f"✅ Found {len(combined)} combined stocks")
        
        if len(combined) > 0:
            print("\n4. Sample data:")
            print(combined.head())
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_google_sheets_connection()
