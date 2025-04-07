import json
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.downloader import download_data
from data.dataset import TimeBasedDataset  # Import the new dataset class

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

def plot_ticker_data(dataloader, n, start_date, end_date):
    """
    Plots the historical data for the given tickers in the same figure.
    """
    plt.figure(figsize=(12, 8))

    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx} shape: {batch.shape}")  # Debugging
        if batch_idx >= n:  # Limit to `n` tickers
            break

        # Extract ticker data
        ticker_data = batch.squeeze(0).numpy()  # Convert to NumPy array
        if ticker_data.size == 0:
            print(f"Empty data for batch {batch_idx}")
            continue

        dates = dataloader.dataset.dates[-len(ticker_data):]  # Get corresponding dates
        ticker = dataloader.dataset.tickers[batch_idx]  # Get ticker name

        # Plot the closing price
        plt.plot(dates, ticker_data[:, 3], label=f"{ticker} Closing Price")  # Assuming column 3 is 'Close'

    plt.title(f"Closing Prices of {n} Random Tickers from {start_date} to {end_date}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    # Path to the JSON file containing valid tickers
    file_path = "tickers/valid_tickers.json"

    # Number of tickers to plot
    n = 5

    # Define the timeframe
    start_date = "2021-01-01"
    end_date = "2023-01-01"

    # Load tickers from the file
    tickers = load_tickers_from_file(file_path)
    if not tickers:
        print("No tickers found in the file.")
        return

    # Extract only the ticker symbols
    ticker_symbols = [ticker["ticker"] for ticker in tickers if "ticker" in ticker]

    # Select n random tickers
    random_tickers = random.sample(ticker_symbols, min(n, len(ticker_symbols)))


    random_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # For testing purposes
    print(f"Selected random tickers: {random_tickers}")

    # Create the dataset and dataloader
    dataset = TimeBasedDataset(
        tickers=random_tickers,
        timesteps=1,  # Number of timesteps to load
        start_date=start_date,
        end_date=end_date,
        interval="1d"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Plot the downloaded data
    plot_ticker_data(dataloader, n, start_date, end_date)

if __name__ == "__main__":
    main()