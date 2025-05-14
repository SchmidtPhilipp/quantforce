from .preprocessor import add_technical_indicators
from .utils.load_data import load_data  
from .utils.clean_data import clean_data, drop_columns


def get_data(tickers, start, end, indicators=("sma", "rsi", "macd", "ema", "adx", "bb", "atr", "obv"), verbosity=0, cache_dir="data/cache", downloader="yfinance"):
    """
    Downloads historical financial data and adds technical indicators.  
    """
    # load the data either from cache or download it
    data = load_data(tickers, start, end, verbosity=verbosity, cache_dir=cache_dir, downloader=downloader)

    # clean the data
    data = clean_data(data, start, end)

    # Remove the Adj Close column if it exists but only if the ticker has the adj close column
    # This is to avoid dropping the Adj Close column for all tickers
    if "Adj Close" in data.columns.get_level_values(1):
        adj_close_tickers = set(t for t, field in data.columns if field == "Adj Close")
        data = data.drop(columns=[(ticker, "Adj Close") for ticker in adj_close_tickers])

    # Add the technical indicators to the data
    data = add_technical_indicators(data, indicators=indicators, verbosity=verbosity)  

    # remove the columns that are not 'Close' or the specified indicators
    data = drop_columns(data, indicators)

    # Ensure that close is in the dataframe
    if "Close" not in data.columns.get_level_values(1):
        raise ValueError("The 'Close' column is missing from the data.")

    return data




# TODO: Verify the data is correct, has no missing values or outliers, is continuous (stetig) to a certain degree. 
