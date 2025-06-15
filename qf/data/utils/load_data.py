"""Load historical financial data from cache or download it.

This module provides functions to load historical financial data either from cache
or by downloading it using the specified downloader.
"""

import os
from typing import List, Union

import pandas as pd
import yfinance as yf

from qf import DEFAULT_CACHE_DIR, VERBOSITY

from .trading_calendar import get_trading_days


def _get_cache_path(ticker: str, start: str, end: str, cache_dir: str) -> str:
    """Get the path to the cached data file for a ticker."""
    return os.path.join(cache_dir, f"{ticker}_{start}_{end}.parquet")


def _save_to_cache(
    data: pd.DataFrame, ticker: str, start: str, end: str, cache_dir: str
):
    """Save data to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _get_cache_path(ticker, start, end, cache_dir)
    data.to_parquet(cache_path)


def _load_from_cache(
    ticker: str, start: str, end: str, cache_dir: str
) -> Union[pd.DataFrame, None]:
    """Load data from cache if it exists."""
    cache_path = _get_cache_path(ticker, start, end, cache_dir)
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    return None


def _download_data(
    ticker: str, start: str, end: str, verbosity: int = VERBOSITY
) -> pd.DataFrame:
    """Download data for a single ticker using yfinance."""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            if verbosity > 0:
                print(f"No data found for {ticker}")
            return pd.DataFrame()
        return data
    except Exception as e:
        if verbosity > 0:
            print(f"Error downloading {ticker}: {str(e)}")
        return pd.DataFrame()


def _align_to_trading_days(data: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Align data to trading days, filling missing days with NaN."""
    # Get trading days for the period
    trading_days = get_trading_days(start, end)

    # If data is empty, create empty DataFrame with trading days
    if data.empty:
        return pd.DataFrame(index=trading_days)

    # Reindex to trading days, filling missing values with NaN
    return data.reindex(trading_days)


def load_data(
    tickers: Union[str, List[str]],
    start: str,
    end: str,
    verbosity: int = VERBOSITY,
    cache_dir: str = DEFAULT_CACHE_DIR,
    downloader: str = "yfinance",
) -> pd.DataFrame:
    """
    Load historical financial data either from cache or by downloading it.

    Args:
        tickers: Single ticker or list of tickers
        start: Start date
        end: End date
        verbosity: Verbosity level
        cache_dir: Directory to store cached data
        downloader: Data downloader to use (currently only yfinance supported)

    Returns:
        pd.DataFrame: Multi-index DataFrame with tickers and OHLCV data
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    all_data = []

    for ticker in tickers:
        # Try to load from cache first
        data = _load_from_cache(ticker, start, end, cache_dir)

        # If not in cache, download and save to cache
        if data is None:
            data = _download_data(ticker, start, end, verbosity)
            if not data.empty:
                # Align to trading days before caching
                data = _align_to_trading_days(data, start, end)
                _save_to_cache(data, ticker, start, end, cache_dir)

        # If we have data, add it to our collection
        if not data.empty:
            # Add ticker as a column level
            data.columns = pd.MultiIndex.from_product([[ticker], data.columns])
            all_data.append(data)

    if not all_data:
        # If no data was found for any ticker, create empty DataFrame with trading days
        trading_days = get_trading_days(start, end)
        return pd.DataFrame(index=trading_days)

    # Combine all ticker data
    combined_data = pd.concat(all_data, axis=1)

    return combined_data


################################################################################################
################################################################################################
# Test the downloader


def test():
    import matplotlib.pyplot as plt

    # test simulated data
    downloader = "yfinance"

    # Delete the chache directory if it exists
    cache_dir = "data/test_data"
    if os.path.exists(cache_dir):
        import shutil

        shutil.rmtree(cache_dir)
    # Create the cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Example usage
    tickers = ["AAPL", "MSFT"]
    start = "2020-01-01"
    end = "2021-01-01"
    data = load_data(
        tickers, start, end, verbosity=1, cache_dir=cache_dir, downloader=downloader
    )
    print(data.head())
    plt.figure()
    data["AAPL"]["Close"].plot(title="AAPL Close Price")
    # save the plot
    plt.savefig(cache_dir + "/" + "AAPL_Close_Price.png")

    # extend the date range at the front
    start = "2019-01-01"
    data = load_data(
        tickers, start, end, verbosity=1, cache_dir=cache_dir, downloader=downloader
    )
    print(data.head())
    data["AAPL"]["Close"].plot(title="AAPL Close Price Extended")
    # save the plot
    plt.savefig(cache_dir + "/" + "AAPL_Close_Price_Extended.png")

    # extend the date range at the end
    end = "2022-01-01"
    data = load_data(
        tickers, start, end, verbosity=1, cache_dir=cache_dir, downloader=downloader
    )
    print(data.head())
    plt.figure()
    data["AAPL"]["Close"].plot(title="AAPL Close Price Extended")
    # save the plot
    plt.savefig(cache_dir + "/" + "AAPL_Close_Price_Extended_End.png")

    # Extend the data range with a new ticker
    tickers = ["AAPL", "MSFT", "GOOGL"]
    start = "1900-01-01"
    end = "2025-01-01"
    data = load_data(
        tickers, start, end, verbosity=1, cache_dir=cache_dir, downloader=downloader
    )
    print(data.head())
    plt.figure()
    data["AAPL"]["Close"].plot(title="AAPL Close Price Extended with GOOGL")
    data["GOOGL"]["Close"].plot(title="AAPL Close Price Extended with GOOGL")
    # save the plot
    plt.savefig(cache_dir + "/" + "AAPL_Close_Price_Extended_with_GOOGL.png")


if __name__ == "__main__":
    # test()

    start = "1950-01-01"
    end = "2025-01-01"

    from qf.data.tickers.tickers import DOWJONES, NASDAQ100, SNP_500

    tickers = NASDAQ100 + DOWJONES + SNP_500
    tickers = list(set(tickers))  # Remove duplicates
    tickers.sort()

    data = load_data(tickers, start, end, verbosity=1)

    # tickers consisting only of nan
    tickers = data.columns.levels[0]
    empty_tickers = []
    for ticker in tickers:
        if data[ticker].isnull().all().all():
            print(f"Ticker {ticker} consists only of NaN values.")
            empty_tickers.append(ticker)

    Nasda1q = list(set(NASDAQ100) - set(empty_tickers))
    DowJones1q = list(set(DOWJONES) - set(empty_tickers))
    Snp500q = list(set(SNP_500) - set(empty_tickers))

    file = {"NASDAQ100": Nasda1q, "DOWJONES": DowJones1q, "SNP_500": Snp500q}

    # save as json
    import json

    # create the files
    os.makedirs("data/tickers", exist_ok=True)
    with open("data/tickers/tickers.json", "w") as f:
        json.dump(file, f)

    print(data.head())
    data["AAPL"]["Close"].plot(title="AAPL Close Price")
