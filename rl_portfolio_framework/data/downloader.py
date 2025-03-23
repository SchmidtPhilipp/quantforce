import yfinance as yf
import pandas as pd
import warnings

def download_data(tickers, start, end, interval="1d", progress=False):
    """
    Downloads historical financial data using yfinance and performs forward/backward filling.

    Parameters:
        tickers (list[str] or str): A single ticker or a list of ticker symbols (e.g. "AAPL", ["AAPL", "MSFT"]).
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
        interval (str): Frequency of data ('1d', '1wk', '1mo', etc.). Default is '1d'.

    Returns:
        pd.DataFrame: A multi-indexed DataFrame (ticker, OHLCV) with cleaned historical data.
    """

    # Download data with separate groups per ticker (multi-indexed columns)
    warnings.filterwarnings("ignore", category=ResourceWarning)
    data = yf.download(tickers, start=start, end=end, interval=interval, group_by="tickers",  progress=progress)

    # Fill missing values forward and backward to ensure continuity
    data = data.ffill().bfill()

    return data