import os
import pandas as pd
import yfinance as yf
import warnings

def download_and_cache_by_time(ticker, start, end, interval="1d", cache_dir="data/data_cache", verbosity=0):
    """
    Downloads and caches historical data for a single ticker, organized by time intervals.

    Parameters:
        ticker (str): The ticker symbol to download.
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
        interval (str): Frequency of data ('1d', '1h', '1m', etc.). Default is '1d'.
        cache_dir (str): Directory to cache the downloaded data.
        verbosity (int): Verbosity level for logging.

    Returns:
        None
    """
    # Download data
    warnings.filterwarnings("ignore", category=ResourceWarning)
    data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)

    # Fill missing values forward and backward
    data = data.ffill().bfill()

    # Iterate over each time interval and save data
    for timestamp, row in data.iterrows():
        date_str = timestamp.strftime("%Y-%m-%d")  # Adjust format for hourly/minute data if needed
        cache_file = os.path.join(cache_dir, f"{date_str}.csv")

        # Check if the file exists
        if os.path.exists(cache_file):
            # Append to the existing file
            existing_data = pd.read_csv(cache_file, index_col=0)
            existing_data.loc[ticker] = row
            existing_data.to_csv(cache_file)
        else:
            # Create a new file
            new_data = pd.DataFrame([row], index=[ticker])
            new_data.to_csv(cache_file)

        if verbosity > 0:
            print(f"Data for {ticker} on {date_str} cached to: {cache_file}")