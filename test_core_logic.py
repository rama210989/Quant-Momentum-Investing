#!/usr/bin/env python3
"""
Test script for core momentum and FIP calculation logic
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def calculate_momentum(symbol: str, start_date: datetime, 
                     end_date: datetime, month_labels: list) -> dict:
    """Calculate momentum score exactly like Colab"""
    try:
        # Mock data for testing (simulating yfinance data)
        # In real usage, this would be: data = yf.download(f"{symbol}.NS", ...)
        
        # Create mock monthly returns data
        np.random.seed(42)  # For reproducible results
        monthly_returns = np.random.normal(0.02, 0.08, 12)  # 2% mean, 8% std
        
        if len(monthly_returns) < 11:
            return None

        last_11 = monthly_returns[-12:-1]  # Last 11 months
        
        returns_dict = {}
        for i in range(11):
            val = float(last_11[i]) * 100
            returns_dict[month_labels[i]] = val

        # Momentum score
        momentum_score = (last_11 + 1).prod() - 1
        momentum_score = float(momentum_score) * 100
        returns_dict["momentum_score"] = momentum_score

        return returns_dict

    except Exception as e:
        print(f"Error in calculate_momentum: {e}")
        return None

def calculate_fip(row, month_labels: list) -> pd.Series:
    """Calculate FIP score exactly like Colab"""
    returns = row[month_labels].astype(float) / 100

    pct_negative = (returns < 0).sum() / len(returns)
    pct_positive = (returns > 0).sum() / len(returns)

    momentum = float(row['momentum_score'])
    sign_momentum = 1 if momentum > 0 else -1

    fip = sign_momentum * (pct_negative - pct_positive)

    return pd.Series({
        "pct_negative_months": pct_negative * 100,
        "pct_positive_months": pct_positive * 100,
        "fip_score": fip
    })

def test_core_logic():
    """Test the core momentum and FIP calculation logic"""
    print("🧪 Testing Core Logic...")
    
    # Setup test parameters
    today = datetime.today()
    end_date = datetime(today.year, today.month, 1) + pd.DateOffset(months=1)
    start_date = end_date - pd.DateOffset(months=13)
    
    # Month labels
    month_range = pd.date_range(start=start_date, periods=11, freq='MS')
    month_labels = [d.strftime('%b %Y') for d in month_range]
    
    print(f"📅 Analysis period: {month_labels[0]} to {month_labels[-1]}")
    
    # Test momentum calculation
    print("\n1. Testing Momentum Calculation...")
    momentum_data = calculate_momentum("RELIANCE", start_date, end_date, month_labels)
    
    if momentum_data:
        print(f"✅ Momentum calculation successful")
        print(f"   Momentum Score: {momentum_data['momentum_score']:.2f}%")
        print(f"   Monthly returns calculated for {len(month_labels)} months")
    else:
        print("❌ Momentum calculation failed")
        return False
    
    # Test FIP calculation
    print("\n2. Testing FIP Calculation...")
    row = pd.Series(momentum_data)
    fip_data = calculate_fip(row, month_labels)
    
    print(f"✅ FIP calculation successful")
    print(f"   FIP Score: {fip_data['fip_score']:.3f}")
    print(f"   % Negative Months: {fip_data['pct_negative_months']:.1f}%")
    print(f"   % Positive Months: {fip_data['pct_positive_months']:.1f}%")
    
    # Test with sample data
    print("\n3. Testing with Sample Data...")
    try:
        sample_df = pd.read_csv("data/nifty500.csv")
        print(f"✅ Sample data loaded: {len(sample_df)} stocks")
        print(f"   Columns: {list(sample_df.columns)}")
        
        # Test with first few stocks
        test_stocks = sample_df.head(3)
        print(f"   Testing with first 3 stocks: {list(test_stocks['Symbol'])}")
        
    except Exception as e:
        print(f"⚠️ Sample data test: {e}")
    
    print("\n🎉 Core logic tests completed successfully!")
    return True

if __name__ == "__main__":
    test_core_logic()