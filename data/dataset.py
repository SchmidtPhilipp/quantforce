import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import yfinance as yf
import warnings
from tqdm import tqdm  # Import tqdm for the progress bar

class TimeBasedDataset(Dataset):
    """
    A PyTorch Dataset for time-based financial data.

    This dataset dynamically loads financial data for a specified list of tickers
    and organizes it into samples of consecutive time intervals (e.g., days, hours).
    If data for a specific time interval is missing, it is downloaded during
    initialization and cached for future use.

    Parameters:
        cache_dir (str): 
            Directory where data is cached. Each time interval (e.g., day) is stored
            as a separate CSV file in this directory.
        tickers (list[str]): 
            List of ticker symbols to include in the dataset. If None, all tickers
            in the cached data will be used.
        timesteps (int): 
            Number of consecutive time intervals to include in each sample. For example,
            if `timesteps=60` and `interval="1d"`, each sample will contain data for
            60 consecutive trading days.
        start_date (str): 
            Start date for filtering data (in 'YYYY-MM-DD' format). Only data from
            this date onward will be included in the dataset. If None, no filtering
            is applied.
        end_date (str): 
            End date for filtering data (in 'YYYY-MM-DD' format). Only data up to
            this date will be included in the dataset. If None, no filtering is applied.
        interval (str): 
            Frequency of the data. Supported values include:
                - "1d": Daily data (default).
                - "1h": Hourly data.
                - "1m": Minute-level data.
            This parameter determines the granularity of the data.

    Methods:
        __len__():
            Returns the number of samples in the dataset. The number of samples is
            determined by the number of available dates minus the number of timesteps
            required for each sample.
        __getitem__(idx):
            Retrieves a single sample from the dataset. A sample consists of data for
            all specified tickers over a range of consecutive time intervals.
    """

    def __init__(self, cache_dir="data/data_cache", 
                 tickers=None, 
                 timesteps=1, 
                 start_date=None, 
                 end_date=None, 
                 interval="1d"):
        
        self.cache_dir = cache_dir
        self.tickers = tickers
        self.timesteps = timesteps
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.interval = interval
        self.metadata = {}  # Initialize metadata

        # Ensure the cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Generate the list of dates dynamically
        self.dates = self._generate_dates()

        # Download missing data for all required dates
        self._prepare_data()

        # Debugging: Print dataset initialization details
        print(f"Initialized TimeBasedDataset with {len(self.dates)} dates and {self.timesteps} timesteps.")
        if len(self.dates) < self.timesteps:
            print("Warning: Not enough dates to provide a single sample.")

    def __len__(self):
        # Ensure the length is non-negative
        return max(0, len(self.dates) - self.timesteps)

    def __getitem__(self, idx):
        # Get the range of dates for this sample
        date_range = self.dates[idx:idx + self.timesteps]

        # Load data for each date in the range
        data_frames = []
        for date in date_range:
            file_path = os.path.join(self.cache_dir, f"{date}_{self.interval}.csv")
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            # Load the data
            df = pd.read_csv(file_path, index_col=0)
            if self.tickers:
                df = df.loc[self.tickers]  # Filter by tickers if specified
            # Handle NaN values (e.g., forward-fill)
            df = df.fillna(method="ffill").fillna(method="bfill")
            data_frames.append(df)

        if not data_frames:
            print(f"No data found for idx {idx} with date range {date_range}")
            return torch.tensor([])  # Return an empty tensor if no data is found

        # Concatenate data and convert to tensor
        data = pd.concat(data_frames)
        print(f"Loaded data for idx {idx}: {data.shape}")
        return torch.tensor(data.values, dtype=torch.float32)

    def _generate_dates(self):
        """
        Generates a list of dates based on the start_date, end_date, and interval.

        Returns:
            list: A list of dates as strings in 'YYYY-MM-DD' format.
        """
        if not self.start_date or not self.end_date:
            raise ValueError("Both start_date and end_date must be specified to generate dates.")

        # Generate a date range based on the interval
        if self.interval == "1d":
            date_range = pd.date_range(self.start_date, self.end_date, freq="D")
        elif self.interval == "1h":
            date_range = pd.date_range(self.start_date, self.end_date, freq="H")
        elif self.interval == "1m":
            date_range = pd.date_range(self.start_date, self.end_date, freq="T")
        else:
            raise ValueError(f"Unsupported interval: {self.interval}")

        return [date.strftime("%Y-%m-%d") for date in date_range]

    def _prepare_data(self):
        """
        Ensures that all required data is downloaded and cached.

        This method downloads data for all tickers in a single batch and splits
        it into individual CSV files for each date. If a date is missing, it creates
        an empty file with NaN values for all tickers.
        """
        # Check which tickers and intervals need to be downloaded
        tickers_to_download = []
        for ticker in self.tickers:
            if ticker in self.metadata and self.interval in self.metadata[ticker]:
                print(f"Data for ticker {ticker} with interval {self.interval} is already cached.")
            else:
                tickers_to_download.append(ticker)

        if tickers_to_download:
            print(f"Downloading data for tickers: {tickers_to_download}")
            self._download_data_for_tickers(tickers_to_download)

        # Ensure every date in the range has a corresponding file
        for date in self.dates:
            file_path = os.path.join(self.cache_dir, f"{date}_{self.interval}.csv")
            if not os.path.exists(file_path):
                # Create an empty file with NaN values for all tickers
                print(f"Creating empty file for missing date: {date}")
                empty_data = pd.DataFrame(index=self.tickers)  # Create a DataFrame with tickers as the index
                empty_data.to_csv(file_path)

    def _download_data_for_tickers(self, tickers):
        """
        Downloads data for all specified tickers in a single batch and splits it
        into individual CSV files for each date.

        Parameters:
            tickers (list): List of ticker symbols to download data for.
        """
        try:
            # Download data for all tickers in a single batch
            data = yf.download(
                tickers=tickers,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                group_by="ticker",
                progress=False
            )

            # Check if data is empty
            if data.empty:
                print(f"No data found for tickers {tickers} in the specified timeframe.")
                return

            # Process data for each ticker
            for ticker in tickers:
                if ticker not in data:
                    print(f"No data found for ticker {ticker}. Skipping.")
                    continue

                ticker_data = data[ticker]
                ticker_data = ticker_data.ffill().bfill()  # Fill missing values

                # Split the data by date and save to individual CSV files
                for date, row in ticker_data.iterrows():
                    date_str = date.strftime("%Y-%m-%d")
                    file_path = os.path.join(self.cache_dir, f"{date_str}_{self.interval}.csv")

                    # If the file already exists, append to it
                    if os.path.exists(file_path):
                        existing_data = pd.read_csv(file_path, index_col=0)
                        existing_data.loc[ticker] = row
                        existing_data.to_csv(file_path)
                    else:
                        # Create a new file
                        new_data = pd.DataFrame([row], index=[ticker])
                        new_data.to_csv(file_path)

                # Update metadata for the ticker
                self._update_metadata(
                    ticker,
                    self.interval,
                    self.start_date.strftime("%Y-%m-%d"),
                    self.end_date.strftime("%Y-%m-%d")
                )

        except Exception as e:
            print(f"Error downloading data for tickers {tickers}: {e}")

    def _download_data_for_ticker(self, ticker, missing_dates):
        """
        Downloads data for a specific ticker for the entire timeframe and splits it
        into individual CSV files for each date.

        Parameters:
            ticker (str): The ticker symbol to download data for.
            missing_dates (list): List of dates for which data is missing.
        """
        print(f"Downloading data for ticker: {ticker}")
        try:
            # Download data for the entire timeframe
            data = yf.download(ticker, start=self.start_date, end=self.end_date, interval=self.interval, progress=False)

            # Check if data is empty
            if data.empty:
                print(f"No data found for ticker {ticker} in the specified timeframe.")
                return

            # Fill missing values forward and backward
            data = data.ffill().bfill()

            # Split the data by date and save to individual CSV files
            for date, row in data.iterrows():
                date_str = date.strftime("%Y-%m-%d")
                if date_str not in missing_dates:
                    continue  # Skip dates that are already cached

                file_path = os.path.join(self.cache_dir, f"{date_str}_{self.interval}.csv")

                # If the file already exists, append to it
                if os.path.exists(file_path):
                    existing_data = pd.read_csv(file_path, index_col=0)
                    existing_data.loc[ticker] = row
                    existing_data.to_csv(file_path)
                else:
                    # Create a new file
                    new_data = pd.DataFrame([row], index=[ticker])
                    new_data.to_csv(file_path)

            # Update metadata
            self._update_metadata(ticker, self.interval, self.start_date.strftime("%Y-%m-%d"), self.end_date.strftime("%Y-%m-%d"))

        except Exception as e:
            print(f"Error downloading data for ticker {ticker}: {e}")

    def _update_metadata(self, ticker, interval, start_date, end_date):
        """
        Updates the metadata to include the specified ticker, interval, and date range.

        Parameters:
            ticker (str): The ticker symbol.
            interval (str): The interval (e.g., "1d", "1h").
            start_date (str): The start date of the downloaded data.
            end_date (str): The end date of the downloaded data.
        """
        if ticker not in self.metadata:
            self.metadata[ticker] = {}
        if interval not in self.metadata[ticker]:
            self.metadata[ticker][interval] = {"start_date": start_date, "end_date": end_date}
        else:
            # Update the date range if necessary
            existing = self.metadata[ticker][interval]
            existing["start_date"] = min(existing["start_date"], start_date)
            existing["end_date"] = max(existing["end_date"], end_date)
        self._save_metadata()

    def _save_metadata(self):
        """
        Saves the metadata to a file in the cache directory.
        """
        metadata_path = os.path.join(self.cache_dir, "metadata.json")
        pd.DataFrame(self.metadata).to_json(metadata_path)
