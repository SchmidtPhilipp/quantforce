import os
import sys
import pandas as pd
from datetime import timedelta

import exchange_calendars as ecals

from qf.data.utils.cache import load_cache, save_cache, update_cache
from qf.data.utils.download_data import download_data
from qf.data.utils.clean_data import reindex_data
from qf.data.utils.make_cache_dir import make_cache_dir


from qf import DEFAULT_INTERVAL, DEFAULT_USE_CACHE, DEFAULT_CACHE_DIR, VERBOSITY, DEFAULT_DOWNLOADER, DEFAULT_FORCE_DOWNLOAD, DEFAULT_N_TRADING_DAYS

def load_data(
    tickers,
    start,
    end,
    interval=DEFAULT_INTERVAL,
    progress=False,
    use_cache=DEFAULT_USE_CACHE,
    cache_dir=DEFAULT_CACHE_DIR,
    verbosity=VERBOSITY,
    downloader=DEFAULT_DOWNLOADER,
    force_download=DEFAULT_FORCE_DOWNLOAD,
    extend_days=10  
):
    """
    Load historical financial data with optional caching and cleaning.

    Parameters:
        tickers (str or list): One or multiple ticker symbols.
        start, end (str): Date strings in 'YYYY-MM-DD' format.
        interval (str): Data interval, e.g., '1d', '1wk'.
        progress (bool): Show download progress.
        use_cache (bool): Whether to use cache or ignore it completely.
        cache_dir (str or None): Absolute path to cache directory. If None, no caching is done.
        verbosity (int): Verbosity level.
        downloader (str): Source for downloading data.
        force_download (bool): Force redownload of data.
        extend_days (int): Number of additional days to extend the range when downloading data.

    Returns:
        pd.DataFrame: Multi-index DataFrame with (ticker, OHLCV) data.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    start, end = adjust_start_end_dates(start, end)

    cache_dir = make_cache_dir(cache_dir, use_cache)

    all_data = []

    for ticker in tickers:
        # Clean ticker symbols
        ticker = ticker.replace("/", "_") 

        # Create cache file path for ticker
        cache_file = os.path.join(cache_dir, f"{ticker}_{interval}.csv") if use_cache and cache_dir else None

        cached_data = pd.DataFrame()

        if use_cache and cache_file and os.path.exists(cache_file):
            cached_data = load_cache(cache_file, verbosity)

        if not cached_data.empty and not force_download:
            cached_start = cached_data.index.min()
            cached_end = cached_data.index.max()

            # Check if the cached data is sufficient
            if pd.Timestamp(start) < cached_start:
                # Extend the range by `extend_days`
                new_start = (pd.Timestamp(start) - pd.Timedelta(days=extend_days)).strftime("%Y-%m-%d")
                new_data_start = download_data(
                    new_start, (cached_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    ticker, interval, downloader, verbosity
                )
                cached_data = update_cache(new_data_start, cached_data)

            if pd.Timestamp(end) > cached_end:
                # Extend the range by `extend_days`
                new_end = (pd.Timestamp(end) + pd.Timedelta(days=extend_days)).strftime("%Y-%m-%d")
                new_data_end = download_data(
                    (cached_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    new_end, ticker, interval, downloader, verbosity
                )
                cached_data = update_cache(cached_data, new_data_end)

        else:
            if verbosity > 0:
                print(f"Downloading full data range for {ticker}: {start} to {end}")
            cached_data = download_data(start, end, ticker, interval, downloader, verbosity)

        if use_cache and cache_file:
            save_cache(cached_data, cache_file, verbosity)  # Save raw, uncleaned data

        all_data.append(cached_data)

    all_data = pd.concat(all_data, axis=1).swaplevel(axis=1).sort_index(axis=1)


    return all_data


def adjust_start_end_dates(start, end, verbosity=VERBOSITY):
    cal = ecals.get_calendar("XNAS")  # NASDAQ calendar, can be changed to other exchanges
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # is session expects a date, so we convert start and end to dates
    if isinstance(start, pd.Timestamp):
        start = start.date()
    if isinstance(end, pd.Timestamp):
        end = end.date()

    if not cal.is_session(start):
        # If start is not a trading day, find the previous trading day
        while not cal.is_session(start):
            start -= timedelta(days=1)
        
    if not cal.is_session(end):
        while not cal.is_session(end):
            end -= timedelta(days=1)
            if end > pd.Timestamp("today").date():
                while not cal.is_session(end):
                    end -= timedelta(days=1)
                break

    # Back to string format
    start = start.strftime("%Y-%m-%d")
    end = end.strftime("%Y-%m-%d")

    if verbosity > 0:
        print(f"Adjusted start date: {start}, end date: {end}")

    return start, end

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

    from qf.data.tickers.tickers import NASDAQ100, DOWJONES, SNP_500
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

    file = {"NASDAQ100": Nasda1q,
            "DOWJONES": DowJones1q,
            "SNP_500": Snp500q}

    # save as json
    import json
    # create the files
    os.makedirs("data/tickers", exist_ok=True)
    with open("data/tickers/tickers.json", "w") as f:
        json.dump(file, f)




    print(data.head())
    data["AAPL"]["Close"].plot(title="AAPL Close Price")