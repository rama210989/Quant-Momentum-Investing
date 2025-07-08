import yfinance as yf
import pandas as pd
import datetime
import time

def get_momentum_df():
    df = pd.read_csv("data/nifty500.csv")

    today = datetime.datetime.today()
    end_date = datetime.datetime(today.year, today.month, 1)  # last completed month
    start_date = end_date - pd.DateOffset(months=13)

    month_range = pd.date_range(start=start_date, periods=12, freq='MS')
    month_labels = [d.strftime('%b %Y') for d in month_range[:-1]]

    results = []

    for _, row in df.iterrows():
        symbol = str(row["Symbol"]).strip()
        try:
            print(f"Downloading data for {symbol}...")
            data = yf.download(f"{symbol}.NS", start=start_date, end=end_date, interval="1mo", auto_adjust=True, progress=False)
            print(f"Data rows: {data.shape[0]}")
            if data.shape[0] < 13 or "Close" not in data.columns:
                print(f"Skipping {symbol}: insufficient data")
                continue

            close = data["Close"].dropna()
            monthly_returns = close.pct_change().dropna()

            last_returns = monthly_returns[-12:-1]
            if len(last_returns) < 11:
                print(f"Skipping {symbol}: not enough returns")
                continue

            returns_dict = {month_labels[i]: round(last_returns.iloc[i] * 100, 2) for i in range(11)}

            momentum_score = (last_returns + 1).prod() - 1
            momentum_pct = round(momentum_score * 100, 2)

            pct_neg = (last_returns < 0).sum() / len(last_returns)
            pct_pos = (last_returns > 0).sum() / len(last_returns)
            sign = 1 if momentum_score > 0 else -1
            fip = sign * (pct_neg - pct_pos)

            result = {
                "Company Name": row["Company Name"],
                "Symbol": symbol,
                "ISIN Code": row["ISIN Code"],
                **returns_dict,
                "momentum_score": momentum_pct,
                "pct_negative_months": round(pct_neg * 100, 2),
                "pct_positive_months": round(pct_pos * 100, 2),
                "fip_score": round(fip, 3)
            }

            results.append(result)

            time.sleep(0.5)  # avoid throttling

        except Exception as e:
            print(f"Error for {symbol}: {e}")
            continue

    momentum_df = pd.DataFrame(results)
    if not momentum_df.empty:
        momentum_df = momentum_df.sort_values(by="momentum_score", ascending=False).reset_index(drop=True)

    return momentum_df
