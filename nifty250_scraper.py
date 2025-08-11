"""
NIFTY 250 Stock Data Scraper
============================
Scrapes NIFTY 250 constituents and updates Google Sheets with fresh data.
"""

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
import time
import logging
from typing import Dict, List, Optional, Tuple
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NIFTY250Scraper:
    def __init__(self):
        """Initialize the scraper"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.stocks_data = []
        
    def scrape_nse_indices(self) -> pd.DataFrame:
        """
        Scrape stock data from NSE indices
        Returns combined data from NIFTY 100 + NIFTY MIDCAP 150 = NIFTY 250
        """
        all_stocks = []
        
        # Define index mappings - Only NIFTY 250 constituents
        indices_map = {
            'NIFTY 100': 'https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv',
            'NIFTY MIDCAP 150': 'https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv'
        }
        
        for index_name, url in indices_map.items():
            logger.info(f"📊 Fetching {index_name} constituents...")
            
            try:
                # Fetch CSV data directly from NSE
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Save to temporary file and read with pandas
                with open(f'temp_{index_name.lower().replace(" ", "_")}.csv', 'wb') as f:
                    f.write(response.content)
                
                # Read CSV
                df = pd.read_csv(f'temp_{index_name.lower().replace(" ", "_")}.csv')
                
                # Standardize column names (NSE CSV format may vary)
                df.columns = df.columns.str.strip()
                
                # Map columns to our standard format
                column_mapping = {
                    'Company Name': 'Company Name',
                    'Symbol': 'Symbol',
                    'ISIN Code': 'ISIN Code',
                    'ISIN': 'ISIN Code'
                }
                
                # Rename columns if they exist
                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns and new_col != old_col:
                        df = df.rename(columns={old_col: new_col})
                
                # Add index type
                if index_name == 'NIFTY 100':
                    df['Index Type'] = 'NIFTY100'
                elif index_name == 'NIFTY MIDCAP 150':
                    df['Index Type'] = 'MIDCAP150'
                
                df['Is Active'] = True
                
                # Clean symbol column
                if 'Symbol' in df.columns:
                    df['Symbol'] = df['Symbol'].str.replace('.NS', '', regex=False)
                    df['Symbol'] = df['Symbol'].str.strip()
                
                # Filter required columns
                required_cols = ['Company Name', 'Symbol', 'ISIN Code', 'Index Type', 'Is Active']
                available_cols = [col for col in required_cols if col in df.columns]
                
                if len(available_cols) >= 3:  # At least Company Name, Symbol, and one more
                    df_filtered = df[available_cols].copy()
                    
                    # Fill missing ISIN codes if needed
                    if 'ISIN Code' not in df_filtered.columns:
                        df_filtered['ISIN Code'] = ''
                    
                    all_stocks.append(df_filtered)
                    logger.info(f"✅ Found {len(df_filtered)} stocks in {index_name}")
                else:
                    logger.warning(f"⚠️ Insufficient columns in {index_name}")
                
            except Exception as e:
                logger.error(f"❌ Error fetching {index_name}: {e}")
                
                # Fallback: Try alternative approach for this index
                fallback_data = self._fallback_scrape_index(index_name)
                if fallback_data is not None and len(fallback_data) > 0:
                    all_stocks.append(fallback_data)
        
        # Combine all data
        if all_stocks:
            combined_df = pd.concat(all_stocks, ignore_index=True)
            
            # Remove duplicates (in case a stock appears in both indices)
            combined_df = combined_df.drop_duplicates(subset=['Symbol'], keep='first')
            
            logger.info(f"🎯 Total NIFTY 250 stocks collected: {len(combined_df)}")
            return combined_df
        else:
            logger.error("❌ No stock data could be retrieved")
            return pd.DataFrame()
    
    def _fallback_scrape_index(self, index_name: str) -> Optional[pd.DataFrame]:
        """
        Fallback method to get index constituents from alternative sources
        """
        logger.info(f"🔄 Trying fallback method for {index_name}")
        
        try:
            # Use yfinance to get some major stocks as fallback
            if index_name == 'NIFTY 100':
                major_stocks = [
                    ('Reliance Industries', 'RELIANCE', 'INE002A01018'),
                    ('Tata Consultancy Services', 'TCS', 'INE467B01029'),
                    ('HDFC Bank', 'HDFCBANK', 'INE040A01034'),
                    ('Infosys', 'INFY', 'INE009A01021'),
                    ('ICICI Bank', 'ICICIBANK', 'INE090A01021'),
                    ('Hindustan Unilever', 'HINDUNILVR', 'INE030A01027'),
                    ('ITC', 'ITC', 'INE154A01025'),
                    ('State Bank of India', 'SBIN', 'INE062A01020'),
                    ('Bharti Airtel', 'BHARTIARTL', 'INE397D01024'),
                    ('Kotak Mahindra Bank', 'KOTAKBANK', 'INE237A01028')
                ]
                
                df = pd.DataFrame(major_stocks, columns=['Company Name', 'Symbol', 'ISIN Code'])
                df['Index Type'] = 'NIFTY100'
                df['Is Active'] = True
                
                logger.info(f"✅ Fallback: Added {len(df)} major NIFTY 100 stocks")
                return df
                
        except Exception as e:
            logger.error(f"❌ Fallback method failed for {index_name}: {e}")
        
        return None
    
    def enrich_with_isin_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich missing ISIN codes using Yahoo Finance and other sources
        """
        logger.info("🔍 Enriching missing ISIN codes...")
        
        missing_isin = df[df['ISIN Code'].isna() | (df['ISIN Code'] == '')].copy()
        
        if len(missing_isin) == 0:
            logger.info("✅ All stocks already have ISIN codes")
            return df
        
        logger.info(f"📝 Found {len(missing_isin)} stocks with missing ISIN codes")
        
        for idx, row in missing_isin.iterrows():
            symbol = row['Symbol']
            try:
                # Try to get ISIN from Yahoo Finance
                ticker = yf.Ticker(f"{symbol}.NS")
                info = ticker.info
                
                if 'isin' in info and info['isin']:
                    df.loc[idx, 'ISIN Code'] = info['isin']
                    logger.info(f"✅ Found ISIN for {symbol}: {info['isin']}")
                else:
                    logger.warning(f"⚠️ No ISIN found for {symbol}")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"⚠️ Error getting ISIN for {symbol}: {e}")
        
        return df
    
    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the scraped data
        """
        logger.info("🧹 Validating and cleaning data...")
        
        initial_count = len(df)
        
        # Remove rows with missing essential data
        df = df.dropna(subset=['Company Name', 'Symbol'])
        df = df[df['Company Name'].str.len() > 0]
        df = df[df['Symbol'].str.len() > 0]
        
        # Clean company names
        df['Company Name'] = df['Company Name'].str.strip()
        df['Symbol'] = df['Symbol'].str.strip().str.upper()
        
        # Clean ISIN codes
        df['ISIN Code'] = df['ISIN Code'].fillna('').str.strip()
        
        # Ensure Is Active is boolean
        df['Is Active'] = df['Is Active'].fillna(True)
        
        # Add last updated timestamp
        df['Last Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Symbol'], keep='first')
        
        final_count = len(df)
        logger.info(f"✅ Data cleaned: {initial_count} → {final_count} stocks")
        
        return df
    
    def update_google_sheet(self, df: pd.DataFrame) -> bool:
        """
        Update Google Sheets with the scraped data
        """
        try:
            # Import the fetcher from existing module
            from stock_data_fetcher import fetcher, DATABASE_AVAILABLE
            
            if not DATABASE_AVAILABLE or not fetcher:
                logger.error("❌ Google Sheets connection not available")
                return False
            
            logger.info("📤 Updating Google Sheets...")
            
            # Ensure all required columns are present
            required_columns = ['Company Name', 'Symbol', 'ISIN Code', 'Index Type', 'Is Active']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ''
            
            # Update the sheet
            success = fetcher.update_stock_data(df)
            
            if success:
                logger.info(f"✅ Successfully updated Google Sheets with {len(df)} stocks")
                return True
            else:
                logger.error("❌ Failed to update Google Sheets")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating Google Sheets: {e}")
            return False
    
    def save_backup_csv(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save data as backup CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"nifty250_backup_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        logger.info(f"💾 Backup saved: {filename}")
        return filename
    
    def run_full_scrape(self) -> bool:
        """
        Execute the complete scraping process
        """
        logger.info("🚀 Starting NIFTY 250 scraping process...")
        
        try:
            # Step 1: Scrape NSE indices
            df = self.scrape_nse_indices()
            
            if df.empty:
                logger.error("❌ No data scraped, aborting")
                return False
            
            # Step 2: Enrich with ISIN codes
            df = self.enrich_with_isin_codes(df)
            
            # Step 3: Validate and clean
            df = self.validate_and_clean_data(df)
            
            # Step 4: Save backup
            backup_file = self.save_backup_csv(df)
            
            # Step 5: Update Google Sheets
            success = self.update_google_sheet(df)
            
            if success:
                logger.info("🎉 NIFTY 250 scraping completed successfully!")
                logger.info(f"📊 Final count: {len(df)} stocks")
                
                # Print summary
                nifty100_count = len(df[df['Index Type'] == 'NIFTY100'])
                midcap150_count = len(df[df['Index Type'] == 'MIDCAP150'])
                
                logger.info(f"📈 NIFTY 100: {nifty100_count} stocks")
                logger.info(f"📈 MIDCAP 150: {midcap150_count} stocks")
                logger.info(f"💾 Backup saved: {backup_file}")
                
                return True
            else:
                logger.error("❌ Failed to update Google Sheets")
                return False
                
        except Exception as e:
            logger.error(f"❌ Scraping process failed: {e}")
            return False

def main():
    """
    Main function to run the scraper
    """
    print("🚀 NIFTY 250 Stock Data Scraper")
    print("=" * 40)
    
    scraper = NIFTY250Scraper()
    success = scraper.run_full_scrape()
    
    if success:
        print("\n✅ Scraping completed successfully!")
        print("📊 Check your Google Sheet for updated data")
    else:
        print("\n❌ Scraping failed!")
        print("📋 Check the logs for error details")

if __name__ == "__main__":
    main()
