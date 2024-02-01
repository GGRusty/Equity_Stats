import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression


def get_daily_data(ticker, period="3y", interval="1d", close_only=True):
    yf_ticker = yf.Ticker(ticker)
    data = yf_ticker.history(period=period, interval=interval)
    if close_only:
        close_data = data[["Close"]].rename(columns={"Close": ticker})
        return close_data
    else:
        return data


def calculate_beta(index, stock, period="3y", interval="1d", just_beta=False):
    index_data = get_daily_data(index, period=period, interval=interval)
    stock_data = get_daily_data(stock, period=period, interval=interval)

    data = pd.merge(index_data, stock_data, left_index=True, right_index=True)

    data[f"{index}_return"] = data[index].pct_change()
    data[f"{stock}_return"] = data[stock].pct_change()

    data.dropna(inplace=True)

    X = data[f"{index}_return"].values.reshape(-1, 1)
    y = data[f"{stock}_return"].values

    model = LinearRegression().fit(X, y)

    beta = model.coef_[0]

    if just_beta:
        return beta
    else:
        return beta, model
