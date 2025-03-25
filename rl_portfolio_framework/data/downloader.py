import yfinance as yf
import pandas as pd
import os
import warnings

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