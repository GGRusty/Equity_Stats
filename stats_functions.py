import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader as pdr
from datetime import datetime, timedelta
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_price_data(ticker, period="3y", interval="1d", close_only=True):
    yf_ticker = yf.Ticker(ticker)
    data = yf_ticker.history(period=period, interval=interval)
    if close_only:
        close_data = data[["Close"]].rename(columns={"Close": ticker})
        return close_data
    else:
        return data


def calculate_beta(index, stock, period="3y", interval="1d", just_beta=False):
    index_data = get_price_data(index, period=period, interval=interval)
    stock_data = get_price_data(stock, period=period, interval=interval)

    data = pd.merge(index_data, stock_data, left_index=True, right_index=True)
    data[f"{index}_return"] = data[index].pct_change()
    data[f"{stock}_return"] = data[stock].pct_change()
    data.dropna(inplace=True)
    X = data[f"{index}_return"].values
    y = data[f"{stock}_return"].values
    X = sm.add_constant(X)

    ols = sm.OLS(y, X).fit()

    beta = ols.params[1]

    if just_beta:
        return beta
    else:
        return beta, ols


def get_rf_rate(treasury_type="3m"):
    type_to_series = {
        "4wk": "DTB4WK",
        "3m": "DGS3MO",
        "6m": "DGS6MO",
        "1y": "DGS1",
        "3y": "DGS3",
        "5y": "DGS5",
        "7y": "DGS7",
        "10y": "DGS10",
    }
    series_id = type_to_series.get(treasury_type.lower())
    if not series_id:
        valid_options = ", ".join(f'"{option}"' for option in type_to_series.keys())
        raise ValueError(
            f"Treasury type '{treasury_type}' is not recognized. Valid options are: {valid_options}."
        )
    start = datetime.today() - timedelta(days=10)
    end = datetime.today()
    rate_data = pdr.get_data_fred(series_id, start, end)
    most_recent_rate = rate_data.iloc[-1, 0]
    most_recent_rate_pct = float(most_recent_rate) / 100  # type: ignore
    return most_recent_rate_pct


def get_market_return(market_ticker="^GSPC", period="10y"):
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=int(period[:-1]))
    data = yf.download(market_ticker, start=start_date, end=end_date)
    data["Year"] = data.index.year
    annual_returns = []

    for year in range(start_date.year, end_date.year + 1):
        if year in data["Year"].values:
            yearly_data = data[data["Year"] == year]
            yearly_return = (
                yearly_data["Adj Close"][-1] / yearly_data["Adj Close"][0]
            ) - 1
            annual_returns.append(
                1 + yearly_return
            )

    # we use Geometric mean to account for compounding
    geom_mean_annual_return = np.prod(annual_returns) ** (1 / len(annual_returns)) - 1

    return float(geom_mean_annual_return)  # type: ignore


def calculate_capm(
    stock="AAPL",
    index="^GSPC",
    beta_period="3y",
    beta_interval="1wk",
    market_period="10y",
    treasury_type="3m",
):
    beta = calculate_beta(
        index=index,
        stock=stock,
        period=beta_period,
        interval=beta_interval,
        just_beta=True,
    )
    rf_rate = get_rf_rate(treasury_type=treasury_type)
    market_rate = get_market_return(market_ticker=index, period=market_period)
    equity_risk_premium = market_rate - rf_rate
    capm = rf_rate + (beta * equity_risk_premium)
    return capm


def get_russell1000_tickers():
    url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
    tables = pd.read_html(url)  # This returns a list of all tables on the page
    sp500_table = tables[2]  # Assuming the first table is the one we want
    tickers = sp500_table["Ticker"].tolist()
    return tickers


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)  # This returns a list of all tables on the page
    sp500_table = tables[0]  # Assuming the first table is the one we want
    tickers = sp500_table["Symbol"].tolist()
    return tickers
