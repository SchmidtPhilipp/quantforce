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
    # Ensure tickers is a list
    if isinstance(tickers, str):
        tickers = [tickers]

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    all_data = []

    for ticker in tickers:
        # Generate cache file path for each ticker
        cache_file = os.path.join(cache_dir, f"{ticker}_{start}_{end}_{interval}.csv")

        # Check if cached data exists
        if os.path.exists(cache_file):
            if verbosity > 0:
                print(f"Loading data from cache: {cache_file}")
            data = pd.read_csv(cache_file, header=[0, 1], index_col=0, parse_dates=True)
        else:
            # Download data for the ticker
            warnings.filterwarnings("ignore", category=ResourceWarning)
            data = yf.download(ticker, start=start, end=end, interval=interval, progress=progress)

            # Fill missing values forward and backward to ensure continuity
            data = data.ffill().bfill()

            # Save data to cache
            data.to_csv(cache_file)
            if verbosity > 0:
                print(f"Data cached to: {cache_file}")

        # Append the data to the list
        all_data.append(data)

    # Combine all ticker data into a single DataFrame
    combined_data = pd.concat(all_data, axis=1)

    # Debugging: Check the structure of the combined data
    if verbosity > 0:
        print(f"Combined data structure:\n{combined_data.head()}")


    # Swap levels of the MultiIndex columns to have 'Ticker' as the first level
    combined_data.columns = pd.MultiIndex.from_tuples(
        [(ticker, field) for ticker, field in combined_data.columns]
    )
    combined_data = combined_data.swaplevel(axis=1)


    return combined_data


def get_data(tickers, start, end, indicators=("sma", "rsi", "macd", "ema", "adx", "bb", "atr", "obv"), verbosity=0):
    """
    Downloads historical financial data and adds technical indicators.  
    """
    train_data = download_data(tickers, start, end, verbosity=verbosity)
    train_data = add_technical_indicators(train_data, indicators=indicators)  
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