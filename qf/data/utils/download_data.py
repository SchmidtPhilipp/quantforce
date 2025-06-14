"""This module provides a function to download historical financial data using yfinance or a simulated downloader."""

from functools import lru_cache

import yfinance as yf
from curl_cffi import requests

from qf import (
    DEFAULT_DOWNLOADER,
    DEFAULT_INTERVAL,
    DEFAULT_USE_ADJUSTED_CLOSE,
    DEFAULT_USE_AUTOREPAIR,
    VERBOSITY,
)

from .generate_random_data import generate_random_data


@lru_cache(maxsize=128)
def download_data(
    start,
    end,
    ticker,
    interval=DEFAULT_INTERVAL,
    downloader=DEFAULT_DOWNLOADER,
    verbosity=VERBOSITY,
):
    """
    Downloads historical financial data using yfinance or a simulated downloader.

    Parameters:
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
        ticker (str): Ticker symbol (e.g. "AAPL").
        interval (str): Frequency of data ('1d', '1wk', '1mo', etc.). Default is '1d'.
        downloader (str): The downloader to use ('yfinance' or 'simulate'). Default is 'simulate'.

    Returns:
        pd.DataFrame: A DataFrame with historical OHLCV data.
    """
    if downloader == "simulate":
        return generate_random_data(start, end, ticker, interval=interval)
    elif downloader == "yfinance":

        session = requests.Session(impersonate="chrome")
        return yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            progress=bool(verbosity),
            auto_adjust=DEFAULT_USE_ADJUSTED_CLOSE,
            session=session,
            repair=DEFAULT_USE_AUTOREPAIR,
            keepna=True,
        )  # We use ajusted close prices everytime.
    else:
        raise ValueError("Downloader not supported.")
