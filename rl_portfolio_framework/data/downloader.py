import yfinance as yf
import pandas as pd
import os
import warnings
from .preprocessor import add_technical_indicators

def download_data(tickers, start, end, interval="1d", progress=False, cache_dir="data_cache", verbosity=0):
    """
    Downloads historical financial data using yfinance and performs forward/backward filling.

    Parameters:
        tickers (list[str] or str): A single ticker or a list of ticker symbols (e.g. "AAPL", ["AAPL", "MSFT"]).
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
        interval (str): Frequency of data ('1d', '1wk', '1mo', etc.). Default is '1d'.
        progress (bool): Whether to show download progress. Default is False.
        cache_dir (str): Directory to cache the downloaded data. Default is 'data_cache'.

    Returns:
        pd.DataFrame: A multi-indexed DataFrame (ticker, OHLCV) with cleaned historical data.
    """

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache file path
    tickers_str = tickers if isinstance(tickers, str) else "-".join(tickers)
    cache_file = os.path.join(cache_dir, f"{tickers_str}_{start}_{end}_{interval}.csv")

    # Check if cached data exists
    if os.path.exists(cache_file):
        if verbosity > 0:
            print(f"Loading data from cache: {cache_file}")
        data = pd.read_csv(cache_file, header=[0, 1], index_col=0, parse_dates=True)
    else:
        # Download data with separate groups per ticker (multi-indexed columns)
        warnings.filterwarnings("ignore", category=ResourceWarning)
        data = yf.download(tickers, start=start, end=end, interval=interval, group_by="tickers", progress=progress)

        # Fill missing values forward and backward to ensure continuity
        data = data.ffill().bfill()

        # Save data to cache
        data.to_csv(cache_file)
        if verbosity > 0:
            print(f"Data cached to: {cache_file}")


    return data

def get_data(tickers, start, end, indicators=("sma", "rsi", "macd", "ema", "adx", "bb", "atr", "obv"), verbosity=0):
    """
    Downloads historical financial data and adds technical indicators.  
    """
    train_data = download_data(tickers, start, end, verbosity=verbosity)
    train_data = add_technical_indicators(train_data)  
    train_data = drop_columns(train_data, indicators)
    return train_data


def drop_columns(data, indicators):
    """
    Drops all columns that are not 'Close' or specified indicators.

    Parameters:
        data (pd.DataFrame): The input data with multi-indexed columns.
        indicators (list[str]): List of indicator names to retain.

    Returns:
        pd.DataFrame: Filtered data containing only 'Close' and specified indicator columns.
    """
    # Keep only 'Close' and columns matching the specified indicators
    filtered_columns = [
        col for col in data.columns if col[1] == "Close" or col[1] in indicators
    ]
    return data.loc[:, filtered_columns]