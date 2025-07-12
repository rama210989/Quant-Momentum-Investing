"""
Supabase-Powered Stock Data Fetcher
===================================
Fetches stock data from Supabase database with CSV upload fallback.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import streamlit as st
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase configuration - UPDATE THESE WITH YOUR CREDENTIALS
SUPABASE_URL = "https://koujyyumgqlththajmui.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtvdWp5eXVtZ3FsdGh0aGFqbXVpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MjEyMTY4OSwiZXhwIjoyMDY3Njk3Njg5fQ.q6nhzZUdc3SxM5Wezd1D7rkTXB8Ur48PP-AvZm8Erp0"

class SupabaseStockFetcher:
    def __init__(self):
        """Initialize Supabase client"""
        self.supabase: Optional[Any] = None
        self.connection_status = "not_initialized"
        
        try:
            from supabase import create_client, Client
            self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            self._test_connection()
            self.connection_status = "connected"
            logger.info("✅ Supabase client initialized successfully")
        except ImportError:
            logger.error("❌ Supabase library not installed. Run: pip install supabase")
            self.connection_status = "library_missing"
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            self.connection_status = "connection_failed"
    
    def _test_connection(self):
        """Test Supabase connection"""
        if not self.supabase:
            raise Exception("Supabase client not initialized")
        
        try:
            # Simple query to test connection
            result = self.supabase.table('stock_universe').select('id').limit(1).execute()
            logger.info("✅ Supabase connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ Supabase connection test failed: {e}")
            raise e
    
    def fetch_stocks_by_type(self, stock_type: str = 'combined') -> pd.DataFrame:
        """
        Fetch stocks from Supabase database
        
        Args:
            stock_type: 'nifty100', 'midcap150', or 'combined'
        
        Returns:
            DataFrame with columns: Company Name, Symbol, ISIN Code
        """
        if not self.supabase or self.connection_status != "connected":
            raise Exception(f"Supabase not available: {self.connection_status}")
        
        try:
            # Try using RPC function first, fallback to direct table query
            try:
                result = self.supabase.rpc('get_stocks_by_type', {'stock_type': stock_type}).execute()
                if result.data:
                    stocks_df = pd.DataFrame(result.data)
                else:
                    stocks_df = self._fetch_stocks_direct(stock_type)
            except Exception as rpc_error:
                logger.warning(f"RPC function failed, using direct query: {rpc_error}")
                stocks_df = self._fetch_stocks_direct(stock_type)
            
            if stocks_df.empty:
                logger.warning(f"No stocks found for type: {stock_type}")
                return pd.DataFrame(columns=['Company Name', 'Symbol', 'ISIN Code'])
            
            # Standardize column names
            column_mapping = {
                'company_name': 'Company Name',
                'symbol': 'Symbol', 
                'isin_code': 'ISIN Code'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in stocks_df.columns:
                    stocks_df = stocks_df.rename(columns={old_col: new_col})
            
            # Clean symbol column
            if 'Symbol' in stocks_df.columns:
                stocks_df['Symbol'] = stocks_df['Symbol'].str.replace('.NS', '', regex=False)
            
            # Filter out empty symbols
            stocks_df = stocks_df[stocks_df['Symbol'].str.len() > 0]
            
            # Ensure all required columns exist
            required_columns = ['Company Name', 'Symbol', 'ISIN Code']
            for col in required_columns:
                if col not in stocks_df.columns:
                    stocks_df[col] = ''
            
            logger.info(f"Fetched {len(stocks_df)} stocks for {stock_type}")
            return stocks_df[required_columns]
            
        except Exception as e:
            logger.error(f"Error fetching stocks from Supabase: {e}")
            raise e
    
    def _fetch_stocks_direct(self, stock_type: str) -> pd.DataFrame:
        """Direct table query fallback method"""
        try:
            if stock_type == 'nifty100':
                result = self.supabase.table('stock_universe').select('*').eq('index_type', 'NIFTY100').eq('is_active', True).execute()
            elif stock_type == 'midcap150':
                result = self.supabase.table('stock_universe').select('*').eq('index_type', 'NIFTY_MIDCAP_150').eq('is_active', True).execute()
            elif stock_type == 'combined':
                result = self.supabase.table('stock_universe').select('*').in_('index_type', ['NIFTY100', 'NIFTY_MIDCAP_150']).eq('is_active', True).execute()
            else:
                result = self.supabase.table('stock_universe').select('*').eq('is_active', True).execute()
            
            return pd.DataFrame(result.data) if result.data else pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Direct query failed: {e}")
            return pd.DataFrame()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database"""
        if not self.supabase or self.connection_status != "connected":
            return {'database_status': 'error', 'error': f'Not connected: {self.connection_status}'}
        
        try:
            nifty100_result = self.supabase.table('stock_universe').select('id').eq('index_type', 'NIFTY100').eq('is_active', True).execute()
            midcap150_result = self.supabase.table('stock_universe').select('id').eq('index_type', 'NIFTY_MIDCAP_150').eq('is_active', True).execute()
            
            return {
                'nifty100_count': len(nifty100_result.data) if nifty100_result.data else 0,
                'midcap150_count': len(midcap150_result.data) if midcap150_result.data else 0,
                'total_count': (len(nifty100_result.data) if nifty100_result.data else 0) + (len(midcap150_result.data) if midcap150_result.data else 0),
                'database_status': 'healthy',
                'connection_status': self.connection_status
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'database_status': 'error', 'error': str(e), 'connection_status': self.connection_status}

# Initialize fetcher with proper error handling
fetcher = None
DATABASE_AVAILABLE = False

try:
    fetcher = SupabaseStockFetcher()
    if fetcher.connection_status == "connected":
        DATABASE_AVAILABLE = True
        logger.info("✅ Database connection established")
    else:
        logger.warning(f"⚠️ Database not available: {fetcher.connection_status}")
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
    Cached function to fetch stock data from Supabase database
    Falls back to requiring CSV upload if database unavailable
    """
    if not DATABASE_AVAILABLE or not fetcher:
        st.error("❌ Database connection not available")
        st.info("💡 Please upload a CSV file with your stock data")
        raise Exception("Database not available - CSV upload required")
    
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
        st.error(f"❌ Error fetching from database: {e}")
        st.info("💡 Please upload a CSV file with your stock data")
        raise e

def show_database_status():
    """Show current database connection status"""
    if DATABASE_AVAILABLE and fetcher:
        stats = fetcher.get_database_stats()
        
        if stats.get('database_status') == 'healthy':
            st.success("✅ Database Connected")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("NIFTY 100", stats.get('nifty100_count', 0))
            with col2:
                st.metric("MIDCAP 150", stats.get('midcap150_count', 0))
            with col3:
                st.metric("Total Stocks", stats.get('total_count', 0))
                
        else:
            st.error(f"❌ Database Error: {stats.get('error', 'Unknown error')}")
            
    else:
        if fetcher and fetcher.connection_status == "library_missing":
            st.error("❌ Supabase library not installed")
            st.code("pip install supabase", language="bash")
        elif fetcher and fetcher.connection_status == "connection_failed":
            st.error("❌ Database connection failed")
            st.info("Check your Supabase URL and API key")
        else:
            st.warning("⚠️ Database not configured")

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
def test_database_connection():
    """Test the database connection and data retrieval"""
    print("🧪 Testing Supabase stock fetcher...")
    
    if not fetcher:
        print("❌ Failed to create fetcher")
        return False
    
    if not DATABASE_AVAILABLE:
        print(f"❌ Database not available: {fetcher.connection_status}")
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
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_database_connection()
