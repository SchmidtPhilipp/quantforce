"""Data management for financial data.

This module provides a DataManager class to handle data operations including
loading, caching, and updating financial data.
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


class DataManager:
    """Manages financial data operations including loading, caching, and updating."""

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
        Initialize the DataManager.

        Args:
            cache_dir: Directory to store cached data
            interval: Data interval (e.g., '1d', '1wk')
            downloader: Data downloader to use ('yfinance' or 'simulate')
            verbosity: Verbosity level
        """
        self.cache_dir = cache_dir
        self.interval = interval
        self.downloader = downloader
        self.verbosity = verbosity
        self.use_adjusted_close = use_adjusted_close
        self.use_autorepair = use_autorepair
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, ticker: str) -> str:
        """Get the path to the cached data file for a ticker."""
        return os.path.join(self.cache_dir, f"{ticker}_{self.interval}.csv")

    def _save_to_cache(
        self, ticker_list: list, data: pd.DataFrame, start: str, end: str
    ) -> None:
        """Save data to cache with MultiIndex columns."""

        if data.empty:
            if self.verbosity > 0:
                print(f"Not saving empty data for {ticker_list} to cache")
            return
        cache_path = self._get_cache_path("_".join(ticker_list))

        try:
            data.to_csv(cache_path)
            if self.verbosity > 0:
                print(f"Saved data for {ticker_list} to cache")
        except Exception as e:
            if self.verbosity > 0:
                print(f"Error saving cache for {ticker_list}: {e}")

    def _load_from_cache(self, ticker_list: list) -> pd.DataFrame:
        """Load data from cache with MultiIndex columns."""
        cache_path = self._get_cache_path("_".join(ticker_list))
        if not os.path.exists(cache_path):
            return pd.DataFrame()
        try:
            data = pd.read_csv(cache_path, header=[0, 1], index_col=0, parse_dates=True)
            return data
        except Exception as e:
            if self.verbosity > 0:
                print(f"Error loading cache for {ticker_list}: {e}")
            return pd.DataFrame()

    def _update_cache(
        self, ticker: str, new_data: pd.DataFrame, start: str, end: str
    ) -> pd.DataFrame:
        """Update cached data with new data.

        Args:
            ticker: Stock ticker symbol
            new_data: New data to merge with cache

        Returns:
            pd.DataFrame: Updated data
        """
        if new_data.empty:
            return new_data

        cached_data = self._load_from_cache([ticker])

        # If cached data is empty, return new data
        if cached_data.empty:
            return new_data

        try:
            # Ensure both DataFrames have datetime index
            if not isinstance(new_data.index, pd.DatetimeIndex):
                new_data.index = pd.to_datetime(new_data.index)
            if not isinstance(cached_data.index, pd.DatetimeIndex):
                cached_data.index = pd.to_datetime(cached_data.index)

            # Combine data and remove duplicates
            combined_data = pd.concat([cached_data, new_data])
            combined_data = combined_data[~combined_data.index.duplicated(keep="last")]
            combined_data = combined_data.sort_index()

            # Cast the data to a df with the complete date range but use the trading calendar to get the dates, non existing dates are filled with NaNs
            # date_range = get_trading_days(start, end)
            # combined_data = combined_data.reindex(date_range)

            return combined_data
        except Exception as e:
            if self.verbosity > 0:
                print(f"Error updating cache for {ticker}: {e}")
            return new_data

    def _align_to_trading_days(self, data: pd.DataFrame) -> pd.DataFrame:
        """Align data to trading days.

        Args:
            data: DataFrame to align
        Returns:
            pd.DataFrame: Aligned DataFrame
        """
        if data.empty:
            return data

        # Ensure index is timezone-naive and has only date components (Y-m-d)
        if isinstance(data.index, pd.DatetimeIndex):
            data.index = data.index.tz_localize(None).normalize()

        # Get trading days for the date range
        trading_days = get_trading_days(data.index[0], data.index[-1])
        # Ensure trading_days is also timezone-naive and has only date components
        trading_days = trading_days.tz_localize(None).normalize()

        # Reindex to trading days
        return data.reindex(trading_days)

    def _download_data(self, tickers: list, start: str, end: str) -> pd.DataFrame:
        """Download data for a list of tickers, always return MultiIndex columns."""
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
                # yfinance returns MultiIndex columns by default
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
                        print(
                            f"No known tickers in {tickers} for simulate mode, returning empty DataFrame"
                        )
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

    def _create_multiindex_dataframe(
        self, ticker: str, data: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Create multi-index DataFrame for a single ticker.

        Args:
            ticker: Stock ticker symbol
            data: DataFrame with OHLCV data
        Returns:
            pd.DataFrame: Multi-index DataFrame
        """
        # If data is None or empty, create empty DataFrame with DatetimeIndex
        if data is None or data.empty:
            # Create a single-day date range for empty DataFrame
            date_range = pd.date_range(
                start="2024-01-01", end="2024-01-01", freq=self.interval
            ).tz_localize(None)
            columns = ["Open", "High", "Low", "Close", "Volume"]
            # Create DataFrame with proper index and columns
            empty_df = pd.DataFrame(
                index=date_range,
                columns=pd.MultiIndex.from_tuples([(ticker, col) for col in columns]),
            )
            return empty_df.iloc[
                0:0
            ]  # Return truly empty DataFrame but with proper index

        # Remove 'Adj Close' if present
        cols = [c for c in data.columns if c != "Adj Close"]
        data = data[cols]

        return data

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

        all_data = []
        for ticker in tickers:
            # Try to load from cache first if not forcing download
            if use_cache and not force_download:
                data = self._load_from_cache([ticker])

                if not data.empty:
                    # Check if we need to extend the data
                    # I am using a tolerance of 3 days to avoid issues with the data being slightly off and constantly reloading.
                    if (data.index[0] - pd.Timestamp(start)).days > 3 or (
                        data.index[-1] - pd.Timestamp(end)
                    ).days < -3:
                        # Download missing data
                        new_data = self._download_data([ticker], start, end)
                        # Update cache with new data
                        data = self._update_cache(ticker, new_data, start, end)
                    # Always filter to requested date range
                    data = data[start:end]
                else:
                    # Download fresh data
                    data = self._download_data([ticker], start, end)
                    if not data.empty:
                        self._save_to_cache([ticker], data, start, end)
            else:
                # Download fresh data
                data = self._download_data([ticker], start, end)
                if use_cache and not data.empty:
                    self._save_to_cache([ticker], data, start, end)

            # Create multi-index DataFrame for this ticker
            data = self._create_multiindex_dataframe(ticker, data)
            if not data.empty:
                all_data.append(data)

        if not all_data:
            # Create empty DataFrame with DatetimeIndex and proper columns
            date_range = pd.date_range(
                start=start, end=end, freq=self.interval
            ).tz_localize(None)
            columns = ["Open", "High", "Low", "Close", "Volume"]
            empty_df = pd.DataFrame(
                index=date_range,
                columns=pd.MultiIndex.from_tuples(
                    [(tickers[0], col) for col in columns]
                ),
            )
            return empty_df.iloc[
                0:0
            ]  # Return truly empty DataFrame but with proper index

        # Combine all DataFrames
        result = pd.concat(all_data, axis=1)

        # Align to trading days
        result: pd.DataFrame = self._align_to_trading_days(result)

        # Ensure timezone-naive index
        if isinstance(result.index, pd.DatetimeIndex):
            result.index = result.index.tz_localize(None)

        return result

    def clear_cache(self, tickers: Optional[Union[str, List[str]]] = None):
        """
        Clear the cache for a specific ticker or all tickers.

        Args:
            tickers: Ticker to clear cache for, or None to clear all caches
        """
        if tickers is None:
            # Clear all cache files
            for file in os.listdir(self.cache_dir):
                if file.endswith(".csv"):
                    os.remove(os.path.join(self.cache_dir, file))
            if self.verbosity > 0:
                print("Cleared all cached data")
        else:
            if isinstance(tickers, str):
                tickers = [tickers]
            for ticker in tickers:
                cache_path = self._get_cache_path(ticker)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    if self.verbosity > 0:
                        print(f"Cleared cache for {ticker}")

    def get_cache_info(self) -> Dict[str, str]:
        """Get information about cached tickers.

        Returns:
            Dict[str, str]: Dictionary with ticker as key and date range as value
        """
        cache_info = {}
        for file in os.listdir(self.cache_dir):
            if file.endswith(".csv"):
                ticker = file.split("_")[0]
                try:
                    data = pd.read_csv(
                        os.path.join(self.cache_dir, file),
                        index_col=0,
                        parse_dates=True,
                    )
                    if not data.empty:
                        cache_info[ticker] = f"{data.index[0]} to {data.index[-1]}"
                except Exception as e:
                    if self.verbosity > 0:
                        print(f"Error reading cache info for {ticker}: {e}")
        return cache_info
