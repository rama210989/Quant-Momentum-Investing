"""
Supabase Stock Data Fetcher
===========================
Clean implementation for fetching stock data from Supabase database.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import streamlit as st
from supabase import create_client, Client
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseStockFetcher:
    def __init__(self):
        """Initialize Supabase client with environment variables or fallback"""
        self.supabase: Optional[Client] = None
        
        # Try to get credentials from environment variables first
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        # Fallback to hardcoded values if env vars not set
        if not supabase_url:
            supabase_url = "https://koujyyumgqlththajmui.supabase.co"
        if not supabase_key:
            supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtvdWp5eXVtZ3FsdGh0aGFqbXVpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MjEyMTY4OSwiZXhwIjoyMDY3Njk3Njg5fQ.q6nhzZUdc3SxM5Wezd1D7rkTXB8Ur48PP-AvZm8Erp0"
        
        try:
            self.supabase = create_client(supabase_url, supabase_key)
            self._test_connection()
            logger.info("✅ Supabase connection successful")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            self.supabase = None
    
    def _test_connection(self):
        """Test Supabase connection"""
        if not self.supabase:
            raise Exception("Supabase client not initialized")
        
        try:
            # Simple query to test connection
            result = self.supabase.table('stock_universe').select('id').limit(1).execute()
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
        if not self.supabase:
            raise Exception("Supabase client not initialized")
        
        try:
            # Use the SQL function we created
            result = self.supabase.rpc('get_stocks_by_type', {'stock_type': stock_type}).execute()
            
            if not result.data:
                logger.warning(f"No stocks found for type: {stock_type}")
                return pd.DataFrame(columns=['Company Name', 'Symbol', 'ISIN Code'])
            
            # Convert to DataFrame and format for your app
            stocks_df = pd.DataFrame(result.data)
            
            # Rename columns to match your app's expectations
            stocks_df = stocks_df.rename(columns={
                'company_name': 'Company Name',
                'symbol': 'Symbol',
                'isin_code': 'ISIN Code'
            })
            
            # Ensure Symbol column doesn't have .NS suffix
            stocks_df['Symbol'] = stocks_df['Symbol'].str.replace('.NS', '', regex=False)
            
            # Remove any empty symbols
            stocks_df = stocks_df[stocks_df['Symbol'].str.len() > 0]
            
            logger.info(f"Fetched {len(stocks_df)} stocks for {stock_type}")
            return stocks_df[['Company Name', 'Symbol', 'ISIN Code']]
            
        except Exception as e:
            logger.error(f"Error fetching stocks from Supabase: {e}")
            raise e
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database"""
        if not self.supabase:
            return {'database_status': 'not_connected'}
        
        try:
            # Count stocks by index type
            nifty100_count = self.supabase.table('stock_universe').select('id').eq('index_type', 'NIFTY100').eq('is_active', True).execute()
            midcap150_count = self.supabase.table('stock_universe').select('id').eq('index_type', 'NIFTY_MIDCAP_150').eq('is_active', True).execute()
            
            # Get last update info
            last_update = self.supabase.table('data_update_log').select('*').order('created_at', desc=True).limit(1).execute()
            
            return {
                'nifty100_count': len(nifty100_count.data),
                'midcap150_count': len(midcap150_count.data),
                'total_count': len(nifty100_count.data) + len(midcap150_count.data),
                'last_update': last_update.data[0] if last_update.data else None,
                'database_status': 'healthy'
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'database_status': 'error', 'error': str(e)}

# Global fetcher instance
fetcher = SupabaseStockFetcher()

def is_database_available() -> bool:
    """Check if database is available and working"""
    return fetcher.supabase is not None

def get_nifty_100_stocks() -> pd.DataFrame:
    """Fetch current NIFTY 100 stock list from Supabase database"""
    if not is_database_available():
        raise Exception("Database connection not available")
    
    try:
        df = fetcher.fetch_stocks_by_type('nifty100')
        
        if df.empty:
            raise Exception("No NIFTY 100 data in database")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in get_nifty_100_stocks: {e}")
        raise e

def get_nifty_midcap_150_stocks() -> pd.DataFrame:
    """Fetch current NIFTY MIDCAP 150 stock list from Supabase database"""
    if not is_database_available():
        raise Exception("Database connection not available")
    
    try:
        df = fetcher.fetch_stocks_by_type('midcap150')
        
        if df.empty:
            raise Exception("No NIFTY MIDCAP 150 data in database")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in get_nifty_midcap_150_stocks: {e}")
        raise e

def get_combined_nifty_stocks() -> pd.DataFrame:
    """Get combined NIFTY 100 + NIFTY MIDCAP 150 stocks from Supabase database"""
    if not is_database_available():
        raise Exception("Database connection not available")
    
    try:
        df = fetcher.fetch_stocks_by_type('combined')
        
        if df.empty:
            raise Exception("No stock data in database")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in get_combined_nifty_stocks: {e}")
        raise e

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data_cache(source: str = "combined") -> pd.DataFrame:
    """
    Cached function to fetch stock data from Supabase database
    TTL (Time To Live) = 3600 seconds = 1 hour
    """
    try:
        if source == "combined":
            return get_combined_nifty_stocks()
        elif source == "nifty100":
            return get_nifty_100_stocks()
        elif source == "midcap150":
            return get_nifty_midcap_150_stocks()
        else:
            return get_combined_nifty_stocks()
            
    except Exception as e:
        logger.error(f"Error in fetch_stock_data_cache: {e}")
        raise e

def get_database_info() -> Dict[str, Any]:
    """Get database information for display"""
    if not is_database_available():
        return {
            'status': 'not_available',
            'message': 'Database connection not available'
        }
    
    try:
        stats = fetcher.get_database_stats()
        return {
            'status': 'available',
            'stats': stats
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

def test_database_connection() -> bool:
    """Test the database connection and data retrieval"""
    if not is_database_available():
        return False
    
    try:
        # Test each function
        nifty100 = fetcher.fetch_stocks_by_type('nifty100')
        midcap150 = fetcher.fetch_stocks_by_type('midcap150')
        combined = fetcher.fetch_stocks_by_type('combined')
        stats = fetcher.get_database_stats()
        
        logger.info(f"✅ Database test successful: {len(nifty100)} NIFTY 100, {len(midcap150)} MIDCAP 150, {len(combined)} combined")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests
    print("🧪 Testing Supabase stock fetcher...")
    success = test_database_connection()
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Tests failed!")
