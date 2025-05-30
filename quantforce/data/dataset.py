import torch
from torch.utils.data import Dataset, DataLoader
from .get_data import get_data

class TimeBasedDataset(Dataset):
    """
    A PyTorch Dataset for time-based financial data with a sliding window of size t.
    """

    def __init__(self, tickers, start_date, end_date, interval="1d", window_size=60, cache_dir="data/data_cache", indicators=("sma", "rsi", "macd")):
        """
        Initializes the dataset.

        Parameters:
            tickers (list[str]): List of ticker symbols to include in the dataset.
            start_date (str): Start date for filtering data (in 'YYYY-MM-DD' format).
            end_date (str): End date for filtering data (in 'YYYY-MM-DD' format).
            interval (str): Frequency of the data (e.g., "1d", "1h").
            window_size (int): Number of consecutive time intervals to include in each sample.
            cache_dir (str): Directory where data is cached.
            indicators (tuple[str]): List of technical indicators to include in the dataset.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.window_size = window_size
        self.cache_dir = cache_dir
        self.indicators = indicators

        # Download and preprocess data using get_data
        self.data = get_data(tickers, start_date, end_date, indicators=indicators)
        
        # Some tickers may not have data so we need to remove them attention multiindex
        self.tickers = self.data.columns.levels[0].tolist()

        # Ensure data is sorted by date
        self.data = self.data.sort_index()

        # Convert data to a PyTorch tensor
        self.data_tensor = torch.tensor(self.data.values, dtype=torch.float32)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return max(0, len(self.data_tensor) - self.window_size + 1)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: A tensor containing data for the specified window.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        return self.data_tensor[idx:idx + self.window_size]

