import yfinance as yf
from .generate_random_data import generate_random_data

def download_data(start, end, ticker, interval="1d", downloader="simulate", verbosity=0):
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
        from curl_cffi import requests
        session = requests.Session(impersonate="chrome")
        return yf.download(ticker, start=start, end=end, interval=interval, progress=bool(verbosity), auto_adjust=True)#, session=session)
    else:
        raise ValueError("Downloader not supported.")