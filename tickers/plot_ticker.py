import json
import random
import matplotlib.pyplot as plt
from data.downloader import download_data

def load_tickers_from_file(file_path):
    """
    Loads tickers from a JSON file if it exists.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        list: A list of tickers loaded from the file.
    """
    if not file_path or not file_path.endswith(".json"):
        raise ValueError("Invalid file path. Please provide a valid JSON file.")
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading tickers from file: {e}")
        return []

def download_random_ticker_data(file_path, n, start_date, end_date, cache_dir="data_cache"):
    """
    Downloads historical data for n random tickers using the downloader function.

    Parameters:
        file_path (str): Path to the JSON file containing valid tickers.
        n (int): Number of random tickers to download data for.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        cache_dir (str): Directory to cache the downloaded data.

    Returns:
        dict: A dictionary containing the downloaded data for the selected tickers.
    """
    # Load tickers from the file
    tickers = load_tickers_from_file(file_path)
    if not tickers:
        print("No tickers found in the file.")
        return {}

    # Select n random tickers
    random_tickers = random.sample(tickers, min(n, len(tickers)))
    print(f"Selected random tickers: {random_tickers}")

    # Use the downloader function to fetch data for the selected tickers
    try:
        data = download_data(
            tickers=random_tickers,
            start=start_date,
            end=end_date,
            cache_dir=cache_dir,
            verbosity=1
        )
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return {}

def plot_ticker_data(data, n, start_date, end_date):
    """
    Plots the historical data for the given tickers in the same figure.

    Parameters:
        data (pd.DataFrame): A multi-index DataFrame containing the downloaded data for the tickers.
        n (int): Number of tickers to plot.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    """

    # Create a single figure for all tickers
    plt.figure(figsize=(12, 8))

    # Iterate over the tickers in the multi-index DataFrame
    for ticker in data.columns.levels[0]:  # Extract tickers from the first level of the multi-index
        try:
            # Extract the 'Close' column for the current ticker
            ticker_data = data[ticker]["Close"]
            if ticker_data.empty:
                print(f"No data found for ticker: {ticker}")
                continue

            # Plot the closing price
            plt.plot(ticker_data.index, ticker_data.values, label=f"{ticker} Closing Price")
        except Exception as e:
            print(f"Error processing data for ticker {ticker}: {e}")

    # Add title, labels, and legend
    plt.title(f"Closing Prices of {n} Random Tickers from {start_date} to {end_date}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    # Place the legend outside the plot (eastoutside)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.tight_layout()  # Adjust layout to fit the legend
    plt.show()

def main():
    # Path to the JSON file containing valid tickers
    file_path = "tickers/valid_tickers.json"

    # Number of tickers to plot
    n = 5

    # Define the timeframe
    start_date = "2000-01-01"
    end_date = "2023-01-01"

    # Download data for n random tickers
    data = download_random_ticker_data(file_path, n, start_date, end_date)

    # Plot the downloaded data
    plot_ticker_data(data, n, start_date, end_date)

if __name__ == "__main__":
    main()