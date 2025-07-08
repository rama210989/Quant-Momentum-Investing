# logic/backtest.py

import yfinance as yf
import pandas as pd
import datetime
import numpy as np

def run_backtest(as_of_date, top_n=30):
    # Dates
    lookback_months = 12
    forward_months = 3

    end_date = as_of_date
    start_date = end_date - pd.DateOffset(months=lookback_months + 1)
    future_end_date = as_of_date + pd.DateOffset(months=forward_months + 1)

    # Month labels
    month_range = pd.date_range(start=start_date + pd.DateOffset(months=1), periods=12, freq='MS')
    month_labels = [d.strftime('%b %Y') for d in month_range[:-1]]

    df = pd.read_csv("data/nifty500.csv")
    results = []

    for _, row in df.iterrows():
        symbol = str(row['Symbol']).strip()
        try:
            data = yf.download(f"{symbol}.NS", start=start_date, end=future_end_date, interval="1mo", progress=False, auto_adjust=True)
            if data.shape[0] < (lookback_months + forward_months) or "Close" not in data.columns:
                continue

            close_prices = data["Close"].dropna()
            monthly_returns = close_prices.pct_change().dropna()

            last_returns = monthly_returns[-(lookback_months + 1):-1]
            if len(last_returns) < 11:
                continue

            returns_dict = {month_labels[i]: last_returns.iloc[i].item() * 100 for i in range(11)}

            # Momentum score
            momentum_score = (last_returns + 1).prod() - 1
            momentum_pct = momentum_score * 100

            # FIP
            pct_neg = (last_returns < 0).sum() / len(last_returns)
            pct_pos = (last_returns > 0).sum() / len(last_returns)
            sign = 1 if momentum_score > 0 else -1
            fip = sign * (pct_neg - pct_pos)

            # Forward returns
            future_returns = monthly_returns[-forward_months:]
            if len(future_returns) < forward_months:
                continue

            abs_return = (future_returns + 1).prod() - 1

            result = {
                "Company Name": row['Company Name'],
                "Symbol": symbol,
                "ISIN Code": row['ISIN Code'],
                **returns_dict,
                "momentum_score": round(momentum_pct, 2),
                "fip_score": round(fip, 3),
                "3M_abs_return_%": round(abs_return.item() * 100, 2)
            }

            if result["fip_score"] < 0:
                results.append(result)
        except Exception:
            continue

    backtest_df = pd.DataFrame(results)
    if backtest_df.empty:
        return None, None, month_labels

    # Get Nifty 50 return over same 3-month forward period
    try:
        nifty = yf.download("^NSEI", start=as_of_date, end=future_end_date, interval="1mo", progress=False, auto_adjust=True)
        nifty_ret = nifty["Close"].pct_change().dropna()[-forward_months:]
        nifty_cum = (nifty_ret + 1).prod() - 1
        nifty_3m_return = round(nifty_cum.item() * 100, 2)
    except:
        nifty_3m_return = None

    top_df = backtest_df.sort_values(by="momentum_score", ascending=False).head(top_n).reset_index(drop=True)
    return top_df, nifty_3m_return, month_labels
