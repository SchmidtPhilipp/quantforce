from datetime import datetime
import numpy as np
import pandas as pd


def generate_random_data(start: str,  # 'YYYY-MM-DD'
                         end: str,  # 'YYYY-MM-DD'
                         tickers: list[str],
                         interval: str = "1d") -> pd.DataFrame:
                         
    """
    Generates random financial data with the same structure as yfinance.

    Parameters:
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
        interval (str): Frequency of data ('1d', '1wk', '1mo', etc.). Default is '1d'.
        tickers (list[str]): List of ticker symbols to include in the MultiIndex. Default is ['AAPL', 'MSFT'].

    Returns:
        pd.DataFrame: Randomly generated financial data with a MultiIndex (Ticker, Date) and columns 
                      ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'].

    Raises:
        ValueError: If any input parameter is invalid or the output is not as expected.
    """
    # Input validation
    if not isinstance(start, str) or not isinstance(end, str):
        raise ValueError("Start and end dates must be strings in 'YYYY-MM-DD' format.")
    try:
        pd.Timestamp(start)
        pd.Timestamp(end)
    except ValueError:
        raise ValueError("Start and end dates must be valid dates in 'YYYY-MM-DD' format.")

    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("Start date must be earlier than or equal to the end date.")

    if interval not in ["1d", "1wk", "1mo"]:
        raise ValueError("Interval must be one of ['1d', '1wk', '1mo'].")

    if isinstance(tickers, str):
        tickers = [tickers]

    if not isinstance(tickers, list) or not all(isinstance(ticker, str) for ticker in tickers):
        raise ValueError("Tickers must be a list of strings.")

    if len(tickers) == 0:
        raise ValueError("Tickers list cannot be empty.")

    # Generate date range
    date_range = pd.date_range(start=start, end=end, freq=interval)
    num_rows = len(date_range)

    if num_rows == 0:
        raise ValueError("Date range must contain at least one date.")

    all_data = []

    for ticker in tickers:
        # Generate random walk for the 'Close' price
        close_prices = np.cumsum(np.random.normal(loc=0, scale=1, size=num_rows)) + 100  # Start at 100
        high_prices = close_prices + np.random.uniform(1, 5, num_rows)
        low_prices = close_prices - np.random.uniform(1, 5, num_rows)
        open_prices = close_prices + np.random.uniform(-2, 2, num_rows)
        adj_close_prices = close_prices + np.random.uniform(-1, 1, num_rows)
        volumes = np.random.randint(1_000_000, 10_000_000, num_rows)

        # Create a DataFrame for the ticker
        data = {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Adj Close": adj_close_prices,
            "Volume": volumes,
        }

        # Create a DataFrame for the ticker
        df = pd.DataFrame(data, index=date_range)
        df.index.name = "Date"

        # Add the ticker as the first level of the MultiIndex
        df = pd.concat({ticker: df}, axis=1)
        all_data.append(df)

    # Combine all tickers into a single DataFrame
    combined_data = pd.concat(all_data, axis=1)

    # Output validation
    if not isinstance(combined_data, pd.DataFrame):
        raise ValueError("The output must be a pandas DataFrame.")

    if not isinstance(combined_data.columns, pd.MultiIndex):
        raise ValueError("The output DataFrame must have a MultiIndex for columns.")

    expected_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for ticker in tickers:
        if not all(col in combined_data[ticker].columns for col in expected_columns):
            raise ValueError(f"Missing expected columns for ticker '{ticker}'.")
        
    # Add column names to the MultiIndex
    combined_data.columns.names = ["Ticker", "Price"]

    # Flip the MultiIndex to have Price as the first level
    combined_data = combined_data.swaplevel(axis=1).sort_index(axis=1)

    return combined_data


if __name__ == "__main__":
    # Example usage
    try:
        start_date = "2022-01-01"
        end_date = "2022-12-31"
        random_data = generate_random_data(start_date, end_date, tickers=["AAPL", "MSFT", "GOOG"])
        print(random_data.head())
        # Plot 
        random_data["AAPL"]["Close"].plot()

    except ValueError as e:
        print(f"Error: {e}")