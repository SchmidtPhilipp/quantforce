"""Data management for financial data.

This module provides a DataManagerV2 class to handle data operations including
loading, caching, and updating financial data in a single cache file.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

from qf import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DOWNLOADER,
    DEFAULT_FORCE_DOWNLOAD,
    DEFAULT_INTERVAL,
    DEFAULT_USE_ADJUSTED_CLOSE,
    DEFAULT_USE_AUTOREPAIR,
    DEFAULT_USE_CACHE,
    VERBOSITY,
)

from .trading_calendar import get_trading_days


class DataManagerV2:
    """Manages financial data operations including loading, caching, and updating in a single cache file."""

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        interval: str = DEFAULT_INTERVAL,
        downloader: str = DEFAULT_DOWNLOADER,
        verbosity: int = VERBOSITY,
        use_adjusted_close: bool = DEFAULT_USE_ADJUSTED_CLOSE,
        use_autorepair: bool = DEFAULT_USE_AUTOREPAIR,
    ):
        """
        Initialize the DataManagerV2.

        Args:
            cache_dir: Directory to store cached data
            interval: Data interval (e.g., '1d', '1wk')
            downloader: Data downloader to use ('yfinance' or 'simulate')
            verbosity: Verbosity level
            use_adjusted_close: Whether to use adjusted close prices
            use_autorepair: Whether to automatically repair data
        """
        self.cache_dir = cache_dir
        self.interval = interval
        self.downloader = downloader
        self.verbosity = verbosity
        self.use_adjusted_close = use_adjusted_close
        self.use_autorepair = use_autorepair
        os.makedirs(cache_dir, exist_ok=True)

        # Single cache file for all data
        self.cache_file = os.path.join(cache_dir, f"data_cache_{interval}.csv")

        # Initialize cache if it doesn't exist
        if not os.path.exists(self.cache_file):
            self._initialize_cache()

    def _initialize_cache(self) -> None:
        """Initialize empty cache file with proper MultiIndex columns."""
        # Create empty DataFrame with MultiIndex columns
        columns = pd.MultiIndex.from_tuples(
            [
                ("META", "Open"),
                ("META", "High"),
                ("META", "Low"),
                ("META", "Close"),
                ("META", "Volume"),
            ],
            names=["Ticker", "Attribute"],
        )
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(self.cache_file)

    def _load_cache(self) -> pd.DataFrame:
        """Load data from cache file."""
        try:
            if os.path.exists(self.cache_file):
                data = pd.read_csv(
                    self.cache_file, header=[0, 1], index_col=0, parse_dates=True
                )
                return data
            return pd.DataFrame()
        except Exception as e:
            if self.verbosity > 0:
                print(f"Error loading cache: {e}")
            return pd.DataFrame()

    def _save_cache(self, data: pd.DataFrame) -> None:
        """Save data to cache file."""
        try:
            data.to_csv(self.cache_file)
            if self.verbosity > 0:
                print(f"Saved data to cache")
        except Exception as e:
            if self.verbosity > 0:
                print(f"Error saving cache: {e}")

    def _update_cache(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Update cache with new data, replacing NaN values with new data.

        Args:
            new_data: New data to merge with cache

        Returns:
            pd.DataFrame: Updated data
        """
        if new_data.empty:
            return new_data

        cached_data = self._load_cache()

        # If cache is empty, return new data
        if cached_data.empty:
            return new_data

        try:
            # Ensure both DataFrames have datetime index
            if not isinstance(new_data.index, pd.DatetimeIndex):
                new_data.index = pd.to_datetime(new_data.index)
            if not isinstance(cached_data.index, pd.DatetimeIndex):
                cached_data.index = pd.to_datetime(cached_data.index)

            # Combine data
            combined_data = pd.concat([cached_data, new_data])

            # For each column, keep new values if old values were NaN
            for col in combined_data.columns:
                mask = combined_data[col].isna()
                if mask.any():
                    combined_data.loc[mask, col] = new_data.loc[mask, col]

            # Remove duplicates and sort
            combined_data = combined_data[~combined_data.index.duplicated(keep="last")]
            combined_data = combined_data.sort_index()

            return combined_data
        except Exception as e:
            if self.verbosity > 0:
                print(f"Error updating cache: {e}")
            return new_data

    def _download_data(self, tickers: list, start: str, end: str) -> pd.DataFrame:
        """Download data for a list of tickers."""
        if self.verbosity > 0:
            print(f"Downloading data for {tickers} from {start} to {end}")

        if self.downloader == "yfinance":
            try:
                data = yf.download(
                    tickers,
                    start=start,
                    end=end,
                    interval=self.interval,
                    group_by="ticker",
                    progress=bool(self.verbosity),
                    auto_adjust=self.use_adjusted_close,
                    repair=self.use_autorepair,
                )
                if data.empty or data.isna().all().all():
                    if self.verbosity > 0:
                        print(f"No data available for {tickers}")
                    return pd.DataFrame()
                return data
            except Exception as e:
                if self.verbosity > 0:
                    print(f"Error downloading {tickers}: {e}")
                return pd.DataFrame()
        elif self.downloader == "simulate":
            from .generate_random_data import generate_random_data

            try:
                known_tickers = ["AAPL", "MSFT"]
                valid_tickers = [t for t in tickers if t in known_tickers]
                if not valid_tickers:
                    if self.verbosity > 0:
                        print(f"No known tickers in {tickers} for simulate mode")
                    return pd.DataFrame()
                data = generate_random_data(
                    start, end, valid_tickers, interval=self.interval
                )
                return data
            except Exception as e:
                if self.verbosity > 0:
                    print(f"Error generating random data for {tickers}: {e}")
                return pd.DataFrame()
        else:
            raise ValueError(f"Unknown downloader: {self.downloader}")

    def get_data(
        self,
        tickers: Union[str, List[str]],
        start: str,
        end: str,
        use_cache: bool = DEFAULT_USE_CACHE,
        force_download: bool = DEFAULT_FORCE_DOWNLOAD,
    ) -> pd.DataFrame:
        """Get data for multiple tickers.

        Args:
            tickers: Single ticker or list of tickers
            start: Start date
            end: End date
            use_cache: Whether to use cached data
            force_download: If True, forces download even if data exists in cache

        Returns:
            pd.DataFrame: Multi-index DataFrame with data for all tickers
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        # Load from cache if not forcing download
        if use_cache and not force_download:
            data = self._load_cache()

            if not data.empty:
                # Filter to requested tickers and date range
                data = data.loc[start:end, pd.IndexSlice[tickers, :]]

                # Check if we need to extend the data
                if (data.index[0] - pd.Timestamp(start)).days > 3 or (
                    data.index[-1] - pd.Timestamp(end)
                ).days < -3:
                    # Download missing data
                    new_data = self._download_data(tickers, start, end)
                    # Update cache with new data
                    data = self._update_cache(new_data)
                    # Filter again to requested date range
                    data = data.loc[start:end, pd.IndexSlice[tickers, :]]
            else:
                # Download fresh data
                data = self._download_data(tickers, start, end)
                if not data.empty:
                    self._save_cache(data)
        else:
            # Download fresh data
            data = self._download_data(tickers, start, end)
            if use_cache and not data.empty:
                self._save_cache(data)

        return data

    def clear_cache(self):
        """Clear the entire cache file."""
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
            if self.verbosity > 0:
                print("Cleared cache file")
            self._initialize_cache()

    def get_cache_info(self) -> Dict[str, str]:
        """Get information about cached data.

        Returns:
            Dict[str, str]: Dictionary with ticker as key and date range as value
        """
        cache_info = {}
        data = self._load_cache()

        if not data.empty:
            for ticker in data.columns.get_level_values(0).unique():
                ticker_data = data[ticker]
                if not ticker_data.empty:
                    cache_info[ticker] = f"{data.index[0]} to {data.index[-1]}"

        return cache_info
