import json
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
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


def plot_ticker_data_live(dataloader, n, start_date, end_date, interval=2):
    """
    Continuously plots the latest data for the given tickers in a loop.

    Parameters:
        dataloader (DataLoader): DataLoader containing the ticker data.
        n (int): Number of tickers to plot.
        start_date (str): Start date for the data.
        end_date (str): End date for the data.
        interval (int): Time in seconds to wait between updates.
    """
    plt.ion()  # Turn on interactive mode for live plotting
    fig, ax = plt.subplots(figsize=(12, 4))

    for batch_idx, batch in enumerate(dataloader):
        # Clear the previous plot
        ax.clear()
        # iterate over the tickers
        for i in range(dataloader.dataset.tickers.__len__()):
            if batch_idx >= n:  # Limit to `n` tickers
                break

            # Extract ticker data
            ticker_data = batch.squeeze(0).numpy()  # Convert to NumPy array
            if ticker_data.size == 0:
                print(f"Empty data for batch {batch_idx}")
                continue

            # Ensure dates are available
            if hasattr(dataloader.dataset, "data"):
                dates = dataloader.dataset.data.index[-len(ticker_data):]
            else:
                raise ValueError("Dataset does not provide dates for plotting.")

            # Ensure tickers are available
            if hasattr(dataloader.dataset, "tickers"):
                ticker = dataloader.dataset.tickers[i]
            else:
                raise ValueError("Dataset does not provide tickers for plotting.")



            # Plot the closing price
            closing_prices = ticker_data[:, int(i*(batch.shape[-1]/dataloader.dataset.tickers.__len__()))]  # Assuming column 3 is 'Close'

            ax.plot(dates, closing_prices, label=f"{ticker} Closing Price", color="blue")



        #ax.set_title(f"Live Closing Prices of {ticker} from {start_date} to {end_date}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.grid()

        # Pause to simulate live updates
        plt.pause(interval)

    plt.ioff()  # Turn off interactive mode
    plt.show()


def main():

    # Number of tickers to plot
    n = 5

    # Define the timeframe
    start_date = "2000-01-01"
    end_date = "2023-01-01"


    # For testing purposes, override random selection
    tickers = ["AAPL"]

    # Create the dataset and dataloader
    dataset = TimeBasedDataset(
        tickers=tickers,
        window_size=100,  # Number of timesteps to load
        start_date=start_date,
        end_date=end_date,
        interval="1d"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Plot the downloaded data live
    plot_ticker_data_live(dataloader, n, start_date, end_date, interval=1)


if __name__ == "__main__":
    main()