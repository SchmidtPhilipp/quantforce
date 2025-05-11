from datetime import timedelta
import os
import pandas as pd
from data.utils.cache import load_cache, save_cache, update_cache
from data.utils.download_data import download_data
from data.utils.wait import wait

def load_data(
    tickers, start, end, interval="1d", progress=False, cache_dir="data/cache", verbosity=0, downloader="yfinance"
):
    """
    Downloads historical financial data using yfinance or a simulated downloader and performs forward/backward filling.

    Parameters:
        tickers (list[str] or str): A single ticker or a list of ticker symbols (e.g. "AAPL", ["AAPL", "MSFT"]).
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
        interval (str): Frequency of data ('1d', '1wk', '1mo', etc.). Default is '1d'.
        progress (bool): Whether to show download progress. Default is False.
        cache_dir (str): Directory to cache the downloaded data. Default is 'data_cache'.
        verbosity (int): Verbosity level for logging.
        downloader (str): The downloader to use ('yfinance' or 'simulate'). Default is 'simulate'.

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
        ticker = ticker.replace("/", "_")  # Replace '/' with '_' for file naming
        # Generate cache file path for each ticker
        cache_file = os.path.join(cache_dir, f"{ticker}_{interval}.csv")

        # Load cached data if available
        cached_data = load_cache(cache_file, verbosity)


        # Determine missing date ranges
        if not cached_data.empty:
            # Extract the start and end dates from the 'Date' level of the MultiIndex
            cached_start = cached_data.index.min()
            cached_end = cached_data.index.max()

            # Check for missing data at the beginning
            if pd.Timestamp(start) < cached_start:
                download_start = start
                download_end = (cached_start - timedelta(days=1)).strftime("%Y-%m-%d")
                if verbosity > 0:
                    print(f"Downloading missing data at the beginning for {ticker}: {download_start} to {download_end}")
                new_data_start = download_data(download_start, download_end, ticker, interval=interval, downloader=downloader, verbosity=verbosity)
                wait(10) 
                cached_data = update_cache(new_data_start, cached_data)

            # Check for missing data at the end
            if pd.Timestamp(end) > cached_end:
                download_start = (cached_end + timedelta(days=1)).strftime("%Y-%m-%d")
                download_end = end
                if verbosity > 0:
                    print(f"Downloading missing data at the end for {ticker}: {download_start} to {download_end}")
                new_data_end = download_data(download_start, download_end, ticker, interval=interval, downloader=downloader, verbosity=verbosity)
                wait(10) 
                cached_data = update_cache(cached_data, new_data_end)
        else:
            # No cached data, download the full range
            if verbosity > 0:
                print(f"Downloading full data range for {ticker}: {start} to {end}")
            new_data = download_data(start, end, ticker, interval=interval, downloader=downloader, verbosity=verbosity)
            wait(10) 

            cached_data = new_data

        # Save updated cache
        save_cache(cached_data, cache_file, verbosity)  # Save only the single ticker's data
        all_data.append(cached_data)

    # Combine all ticker data into a single MultiIndex DataFrame
    all_data = pd.concat(all_data, axis=1)

    # Flip MultiIndex levels to have 'Ticker' as the first level
    all_data = all_data.swaplevel(axis=1).sort_index(axis=1)

    return all_data

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
    data = load_data(tickers, start, end, verbosity=1, cache_dir=cache_dir, downloader=downloader)
    print(data.head())
    plt.figure()
    data["AAPL"]["Close"].plot(title="AAPL Close Price")   
    # save the plot
    plt.savefig(cache_dir + "/" + "AAPL_Close_Price.png")
    
    # extend the date range at the front   
    start = "2019-01-01"
    data = load_data(tickers, start, end, verbosity=1, cache_dir=cache_dir, downloader=downloader)
    print(data.head())
    data["AAPL"]["Close"].plot(title="AAPL Close Price Extended")
    # save the plot
    plt.savefig(cache_dir + "/" + "AAPL_Close_Price_Extended.png")

    # extend the date range at the end
    end = "2022-01-01"
    data = load_data(tickers, start, end, verbosity=1, cache_dir=cache_dir, downloader=downloader)
    print(data.head())
    plt.figure()
    data["AAPL"]["Close"].plot(title="AAPL Close Price Extended")
    # save the plot
    plt.savefig(cache_dir + "/" + "AAPL_Close_Price_Extended_End.png")


    # Extend the data range with a new ticker   
    tickers = ["AAPL", "MSFT", "GOOGL"]
    start = "1900-01-01"
    end = "2025-01-01"
    data = load_data(tickers, start, end, verbosity=1, cache_dir=cache_dir, downloader=downloader)
    print(data.head())
    plt.figure()
    data["AAPL"]["Close"].plot(title="AAPL Close Price Extended with GOOGL")
    data["GOOGL"]["Close"].plot(title="AAPL Close Price Extended with GOOGL")
    # save the plot
    plt.savefig(cache_dir + "/" + "AAPL_Close_Price_Extended_with_GOOGL.png")



if __name__ == "__main__":
    #test()

    start = "1950-01-01"
    end = "2025-01-01"

    from data.tickers import NASDAQ100, DOWJONES, SNP_500
    tickers = NASDAQ100 + DOWJONES + SNP_500
    tickers = list(set(tickers))  # Remove duplicates

    tickers = ["CRM", "AAPL"]

    data = load_data(tickers, start, end, verbosity=0)
    print(data.head())
    data["AAPL"]["Close"].plot(title="AAPL Close Price")