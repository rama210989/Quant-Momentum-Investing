"""
Supabase-Powered Stock Data Fetcher
===================================
Replaces the original stock_data_fetcher.py to read from Supabase database
instead of hardcoded lists or unreliable web scraping.
"""

import pandas as pd
from typing import List, Dict, Any
import streamlit as st
from supabase import create_client, Client
import logging
from datetime import datetime, timedelta
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase configuration - UPDATE THESE
SUPABASE_URL = "https://koujyyumgqlththajmui.supabase.co"  # Your existing URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtvdWp5eXVtZ3FsdGh0aGFqbXVpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MjEyMTY4OSwiZXhwIjoyMDY3Njk3Njg5fQ.q6nhzZUdc3SxM5Wezd1D7rkTXB8Ur48PP-AvZm8Erp0"  # Your existing key

class SupabaseStockFetcher:
    def __init__(self):
        """Initialize Supabase client"""
        try:
            self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
            self._test_connection()
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            self.supabase = None
    
    def _test_connection(self):
        """Test Supabase connection"""
        try:
            # Simple query to test connection
            result = self.supabase.table('stock_universe').select('id').limit(1).execute()
            logger.info("✅ Supabase connection successful")
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
            return {}
        
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
try:
    fetcher = SupabaseStockFetcher()
except Exception as e:
    logger.error(f"Failed to create global fetcher: {e}")
    fetcher = None

def get_nifty_100_stocks() -> pd.DataFrame:
    """
    Fetch current NIFTY 100 stock list from Supabase database
    """
    if not fetcher:
        st.error("❌ Database connection not available")
        st.error("💡 **Solution**: Please check your Supabase configuration or upload a CSV file")
        raise Exception("Database connection failed")
    
    try:
        with st.spinner("Fetching NIFTY 100 stocks from database..."):
            df = fetcher.fetch_stocks_by_type('nifty100')
            
            if df.empty:
                st.warning("⚠️ No NIFTY 100 stocks found in database")
                st.error("💡 **Solution**: Please run the data population script or upload a CSV file")
                raise Exception("No NIFTY 100 data in database")
            
            st.success(f"✅ Fetched {len(df)} NIFTY 100 stocks from database")
            return df
            
    except Exception as e:
        st.error(f"❌ Error fetching NIFTY 100: {e}")
        st.error("💡 **Solution**: Please upload a CSV file with NIFTY 100 stocks in the 'Upload CSV File' option")
        logger.error(f"Error in get_nifty_100_stocks: {e}")
        raise e

def get_nifty_midcap_150_stocks() -> pd.DataFrame:
    """
    Fetch current NIFTY MIDCAP 150 stock list from Supabase database
    """
    if not fetcher:
        st.error("❌ Database connection not available")
        st.error("💡 **Solution**: Please check your Supabase configuration or upload a CSV file")
        raise Exception("Database connection failed")
    
    try:
        with st.spinner("Fetching NIFTY MIDCAP 150 stocks from database..."):
            df = fetcher.fetch_stocks_by_type('midcap150')
            
            if df.empty:
                st.warning("⚠️ No NIFTY MIDCAP 150 stocks found in database")
                st.error("💡 **Solution**: Please run the data population script or upload a CSV file")
                raise Exception("No NIFTY MIDCAP 150 data in database")
            
            st.success(f"✅ Fetched {len(df)} NIFTY MIDCAP 150 stocks from database")
            return df
            
    except Exception as e:
        st.error(f"❌ Error fetching NIFTY MIDCAP 150: {e}")
        st.error("💡 **Solution**: Please upload a CSV file with NIFTY MIDCAP 150 stocks in the 'Upload CSV File' option")
        logger.error(f"Error in get_nifty_midcap_150_stocks: {e}")
        raise e

def get_combined_nifty_stocks() -> pd.DataFrame:
    """
    Get combined NIFTY 100 + NIFTY MIDCAP 150 stocks from Supabase database
    """
    if not fetcher:
        st.error("❌ Database connection not available")
        st.error("💡 **Solution**: Please check your Supabase configuration or upload a CSV file")
        raise Exception("Database connection failed")
    
    try:
        with st.spinner("Fetching combined NIFTY 100 + MIDCAP 150 stocks from database..."):
            df = fetcher.fetch_stocks_by_type('combined')
            
            if df.empty:
                st.warning("⚠️ No stocks found in database")
                st.error("💡 **Solution**: Please run the data population script or upload a CSV file")
                raise Exception("No stock data in database")
            
            # Get database stats for display
            stats = fetcher.get_database_stats()
            
            st.success(f"✅ Combined dataset ready: {len(df)} unique stocks from database")
            
            # Show breakdown if stats available
            if stats.get('database_status') == 'healthy':
                st.info(f"📊 **Database**: {stats.get('nifty100_count', 0)} NIFTY 100 + {stats.get('midcap150_count', 0)} MIDCAP 150 = {len(df)} total stocks")
                
                # Show last update info
                if stats.get('last_update'):
                    last_update = stats['last_update']
                    update_date = last_update.get('completed_at', 'Unknown')
                    st.info(f"🕒 **Last Updated**: {update_date}")
            
            return df
            
    except Exception as e:
        st.error(f"❌ Error creating combined dataset: {e}")
        st.error("💡 **Solution**: Please upload a CSV file with your stock list in the 'Upload CSV File' option")
        logger.error(f"Error in get_combined_nifty_stocks: {e}")
        raise e

@st.cache_data(ttl=3600)  # Cache for 1 hour (database is already fast)
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
        st.error(f"❌ Error in fetch_stock_data_cache: {e}")
        st.error("💡 **Solution**: Please use the 'Upload CSV File' option to provide your stock list")
        logger.error(f"Error in fetch_stock_data_cache: {e}")
        raise e

def show_database_management():
    """Show database management interface in Streamlit"""
    st.subheader("💾 Database Management")
    
    if not fetcher:
        st.error("❌ Database connection not available")
        st.code("""
# To fix this, update the credentials in stock_data_fetcher.py:
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-service-role-key"
        """)
        return
    
    # Get and display database stats
    stats = fetcher.get_database_stats()
    
    if stats.get('database_status') == 'healthy':
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("NIFTY 100 Stocks", stats.get('nifty100_count', 0))
        
        with col2:
            st.metric("MIDCAP 150 Stocks", stats.get('midcap150_count', 0))
        
        with col3:
            st.metric("Total Stocks", stats.get('total_count', 0))
        
        # Last update info
        if stats.get('last_update'):
            last_update = stats['last_update']
            st.info(f"""
            **Last Update**: {last_update.get('completed_at', 'Unknown')}  
            **Type**: {last_update.get('update_type', 'Unknown')}  
            **Status**: {last_update.get('status', 'Unknown')}  
            **Records Added**: {last_update.get('records_added', 0)}
            """)
        
        # Database actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Cache"):
                st.cache_data.clear()
                st.success("✅ Cache refreshed")
                st.rerun()
        
        with col2:
            st.info("💡 To update stocks, run the population script")
        
        with col3:
            if st.button("📊 Test Connection"):
                try:
                    fetcher._test_connection()
                    st.success("✅ Database connection OK")
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")
    
    else:
        st.error(f"❌ Database error: {stats.get('error', 'Unknown error')}")
        st.info("💡 Please check your Supabase configuration and run the population script")

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
    
    try:
        # Test each function
        print("\n1. Testing NIFTY 100:")
        nifty100 = fetcher.fetch_stocks_by_type('nifty100')
        print(f"✅ Found {len(nifty100)} NIFTY 100 stocks")
        
        print("\n2. Testing NIFTY MIDCAP 150:")
        midcap150 = fetcher.fetch_stocks_by_type('midcap150')
        print(f"✅ Found {len(midcap150)} MIDCAP 150 stocks")
        
        print("\n3. Testing Combined:")
        combined = fetcher.fetch_stocks_by_type('combined')
        print(f"✅ Found {len(combined)} combined stocks")
        
        print("\n4. Testing Database Stats:")
        stats = fetcher.get_database_stats()
        print(f"✅ Stats: {stats}")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests
    test_database_connection()).execute()
            midcap150_count = self.supabase.table('stock_universe').select('id').eq('index_type', 'NIFTY_MIDCAP_150').eq('is_active', True
