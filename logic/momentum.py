import yfinance as yf
import pandas as pd
import datetime
import numpy as np

def get_momentum_df():
    # Load stock list
    df = pd.read_csv("data/nifty500.csv")

    # Dynamic date range setup
    today = datetime.datetime.today()
    end_date = datetime.datetime(today.year, today.month, 1) + pd.DateOffset(months=1)  # next month's 1st day
    start_date = end_date - pd.DateOffset(months=13)  # 13 months back

    # Month labels: 11 months from start_date
    month_range = pd.date_range(start=start_date, periods=11, freq='MS')
    month_labels = [d.strftime('%b %Y') for d in month_range]

    results = []

    for _, row in df.iterrows():
        symbol = str(row["Symbol"]).strip()
        try:
            data = yf.download(f"{symbol}.NS", start=start_date, end=end_date, interval="1mo",
                               auto_adjust=True, progress=False)

            # Require minimum 13 rows (to compute 12 pct changes for 11 months)
            if data.shape[0] < 13 or "Close" not in data.columns:
                continue

            close_prices = data["Close"].dropna()
            monthly_returns = close_prices.pct_change().dropna()

            last_11 = monthly_returns[-12:-1]

            if len(last_11) < 11:
                continue

            returns_dict = {}
            for i in range(11):
                returns_dict[month_labels[i]] = round(float(last_11.iloc[i]) * 100, 2)

            momentum_score = (last_11 + 1).prod() - 1
            momentum_score_pct = round(float(momentum_score) * 100, 2)
            returns_dict["momentum_score"] = momentum_score_pct

            returns_decimal = pd.Series(returns_dict).drop("momentum_score") / 100
            pct_negative = (returns_decimal < 0).sum() / len(returns_decimal)
            pct_positive = (returns_decimal > 0).sum() / len(returns_decimal)
            sign_momentum = 1 if momentum_score_pct > 0 else -1
            fip_score = round(sign_momentum * (pct_negative - pct_positive), 3)

            result = {
                "Company Name": row["Company Name"],
                "Symbol": symbol,
                "ISIN Code": row["ISIN Code"],
                **returns_dict,
                "pct_negative_months": round(pct_negative * 100, 2),
                "pct_positive_months": round(pct_positive * 100, 2),
                "fip_score": fip_score
            }

            results.append(result)

        except Exception:
            continue

    momentum_df = pd.DataFrame(results)

    if not momentum_df.empty:
        momentum_df = momentum_df.sort_values(by="momentum_score", ascending=False).reset_index(drop=True)

    return momentum_df
