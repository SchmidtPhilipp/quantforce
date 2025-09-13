"""Data management for financial data.

This module provides a DataManager class to handle data operations including
loading, caching, and updating financial data with smart segment-based loading.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

from qf.settings import VERBOSITY
from qf.utils.logging_config import get_logger

from .trading_calendar import get_trading_days

logger = get_logger(__name__)


class DataManager:
    """Manages financial data operations with smart segment-based loading and sparse cache structure."""

    def __init__(
        self,
        cache_dir: str = os.path.join(os.path.expanduser("~"), "qf_cache"),
        interval: str = "1d",
        downloader: str = "yfinance",
        verbosity: int = VERBOSITY,
        use_adjusted_close: bool = True,
        use_autorepair: bool = False,
        cache_format: str = "parquet",
        buffer_days: int = 50,  # Additional days to download before and after
    ):
        """
        Initialize the DataManager.

        Args:
            cache_dir: Directory to store cached data
            interval: Data interval (e.g., '1d', '1wk')
            downloader: Data downloader to use ('yfinance' or 'simulate')
            verbosity: Verbosity level
            use_adjusted_close: Whether to use adjusted close prices
            use_autorepair: Whether to automatically repair data
            cache_format: Cache format ('parquet', 'csv', 'hdf5')
            buffer_days: Additional days to download before and after requested range
        """
        self.cache_dir = cache_dir
        self.interval = interval
        self.downloader = downloader
        self.verbosity = verbosity
        self.use_adjusted_close = use_adjusted_close
        self.use_autorepair = use_autorepair
        self.cache_format = cache_format.lower()
        self.buffer_days = buffer_days
        os.makedirs(cache_dir, exist_ok=True)

        # Single cache file for all data
        if self.cache_format == "parquet":
            self.cache_file = os.path.join(cache_dir, f"data_cache_{interval}.parquet")
        elif self.cache_format == "csv":
            self.cache_file = os.path.join(cache_dir, f"data_cache_{interval}.csv")
        elif self.cache_format == "hdf5":
            self.cache_file = os.path.join(cache_dir, f"data_cache_{interval}.h5")
        else:
            raise ValueError(
                f"Unsupported cache format: {cache_format}. Use 'parquet', 'csv', or 'hdf5'"
            )

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

        # Ensure the cache directory exists
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

        # Save the empty DataFrame to the cache file
        self._save_cache(empty_df)

    def _load_cache(self) -> pd.DataFrame:
        """Load data from cache file."""
        try:
            if os.path.exists(self.cache_file):
                if self.cache_format == "parquet":
                    data = pd.read_parquet(self.cache_file)
                elif self.cache_format == "csv":
                    data = pd.read_csv(
                        self.cache_file, header=[0, 1], index_col=0, parse_dates=True
                    )
                elif self.cache_format == "hdf5":
                    data = pd.read_hdf(self.cache_file, key="data")
                else:
                    raise ValueError(f"Unsupported cache format: {self.cache_format}")
                return data
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return pd.DataFrame()

    def _save_cache(self, data: pd.DataFrame) -> None:
        """Save data to cache file."""
        try:
            if self.cache_format == "parquet":
                data.to_parquet(self.cache_file, compression="snappy")
            elif self.cache_format == "csv":
                data.to_csv(self.cache_file)
            elif self.cache_format == "hdf5":
                data.to_hdf(self.cache_file, key="data", mode="w", complevel=9)
            else:
                raise ValueError(f"Unsupported cache format: {self.cache_format}")

            file_size = os.path.getsize(self.cache_file) / (1024 * 1024)  # MB
            logger.info(f"Saved data to cache ({file_size:.2f} MB)")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def get_cache_size_mb(self) -> float:
        """Get the size of the cache file in MB."""
        if os.path.exists(self.cache_file):
            return os.path.getsize(self.cache_file) / (1024 * 1024)
        return 0.0

    def get_cache_stats(self) -> Dict[str, Union[int, float, str]]:
        """Get comprehensive cache statistics."""
        stats = {
            "format": self.cache_format,
            "file_path": self.cache_file,
            "buffer_days": self.buffer_days,
            "file_size_mb": self.get_cache_size_mb(),
            "exists": os.path.exists(self.cache_file),
        }

        if os.path.exists(self.cache_file):
            try:
                data = self._load_cache()
                if not data.empty:
                    stats.update(
                        {
                            "rows": len(data),
                            "columns": len(data.columns),
                            "tickers": len(data.columns.get_level_values(0).unique()),
                            "date_range": f"{data.index[0]} to {data.index[-1]}",
                            "memory_usage_mb": round(
                                data.memory_usage(deep=True).sum() / (1024 * 1024), 2
                            ),
                        }
                    )
            except Exception as e:
                stats["error"] = str(e)

        return stats

    def _normalize_dataframe_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame to have (Ticker, Attribute) MultiIndex structure."""
        if data.empty:
            return data

        # Check if we have a MultiIndex
        if not isinstance(data.columns, pd.MultiIndex):
            return data

        # If yfinance format: (Price, Ticker) -> (Ticker, Price)
        if data.columns.names[0] in [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        ]:
            data = data.swaplevel(axis=1).sort_index(axis=1)
            data.columns.names = ["Ticker", "Attribute"]
        # If already in (Ticker, Attribute) format, just ensure names are correct
        elif data.columns.names[0] == "Ticker":
            data.columns.names = ["Ticker", "Attribute"]

        return data

    def _get_missing_date_ranges(
        self,
        ticker: str,
        requested_start: str,
        requested_end: str,
        cached_data: pd.DataFrame,
    ) -> List[tuple]:
        """
        Determine which date ranges are missing for a given ticker.

        Args:
            ticker: Ticker symbol
            requested_start: Requested start date
            requested_end: Requested end date
            cached_data: Currently cached data

        Returns:
            List of (start_date, end_date) tuples for missing ranges
        """
        requested_start_ts = pd.Timestamp(requested_start)
        requested_end_ts = pd.Timestamp(requested_end)

        # Check if ticker exists in cache
        if ticker not in cached_data.columns.get_level_values(0):
            # Ticker completely missing - download entire range with buffer
            buffer_start = requested_start_ts - pd.Timedelta(days=self.buffer_days)
            buffer_end = requested_end_ts + pd.Timedelta(days=self.buffer_days)
            return [
                (buffer_start.strftime("%Y-%m-%d"), buffer_end.strftime("%Y-%m-%d"))
            ]

        # Get ticker data and find actual date range
        ticker_data = cached_data[ticker].dropna()
        if ticker_data.empty:
            # Ticker exists but no valid data
            buffer_start = requested_start_ts - pd.Timedelta(days=self.buffer_days)
            buffer_end = requested_end_ts + pd.Timedelta(days=self.buffer_days)
            return [
                (buffer_start.strftime("%Y-%m-%d"), buffer_end.strftime("%Y-%m-%d"))
            ]

        cached_start = ticker_data.index[0]
        cached_end = ticker_data.index[-1]

        missing_ranges = []

        # Check if we need data before cached range
        if requested_start_ts < cached_start:
            buffer_start = requested_start_ts - pd.Timedelta(days=self.buffer_days)
            # Download up to a few days past cached start to ensure continuity
            download_end = cached_start + pd.Timedelta(days=5)
            missing_ranges.append(
                (buffer_start.strftime("%Y-%m-%d"), download_end.strftime("%Y-%m-%d"))
            )

        # Check if we need data after cached range
        if requested_end_ts > cached_end:
            # Download from a few days before cached end to ensure continuity
            download_start = cached_end - pd.Timedelta(days=5)
            buffer_end = requested_end_ts + pd.Timedelta(days=self.buffer_days)
            missing_ranges.append(
                (download_start.strftime("%Y-%m-%d"), buffer_end.strftime("%Y-%m-%d"))
            )

        return missing_ranges

    def _download_data(self, tickers: list, start: str, end: str) -> pd.DataFrame:
        """Download data for a list of tickers."""
        logger.info(f"Downloading data for {tickers} from {start} to {end}")

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
                    logger.warning(f"No data available for {tickers}")
                    return pd.DataFrame()
                return data
            except Exception as e:
                logger.error(f"Error downloading {tickers}: {e}")
                return pd.DataFrame()
        elif self.downloader == "simulate":
            from .generate_random_data import generate_random_data

            try:
                known_tickers = ["AAPL", "MSFT"]
                valid_tickers = [t for t in tickers if t in known_tickers]
                if not valid_tickers:
                    logger.warning(f"No known tickers in {tickers} for simulate mode")
                    return pd.DataFrame()
                data = generate_random_data(
                    start, end, valid_tickers, interval=self.interval
                )
                return data
            except Exception as e:
                logger.error(f"Error generating random data for {tickers}: {e}")
                return pd.DataFrame()
        else:
            raise ValueError(f"Unknown downloader: {self.downloader}")

    def _merge_data(
        self, cached_data: pd.DataFrame, new_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge new data with cached data, prioritizing new data for overlapping periods.

        Args:
            cached_data: Existing cached data
            new_data: Newly downloaded data

        Returns:
            Merged DataFrame
        """
        if new_data.empty:
            return cached_data

        if cached_data.empty:
            return self._normalize_dataframe_structure(new_data)

        # Normalize both DataFrames
        cached_data = self._normalize_dataframe_structure(cached_data)
        new_data = self._normalize_dataframe_structure(new_data)

        try:
            # Ensure both DataFrames have datetime index
            if not isinstance(new_data.index, pd.DatetimeIndex):
                new_data.index = pd.to_datetime(new_data.index)
            if not isinstance(cached_data.index, pd.DatetimeIndex):
                cached_data.index = pd.to_datetime(cached_data.index)

            # Get union of all indices and columns
            all_index = cached_data.index.union(new_data.index)
            all_columns = cached_data.columns.union(new_data.columns)

            # Reindex both DataFrames to common structure
            cached_data = cached_data.reindex(index=all_index, columns=all_columns)
            new_data = new_data.reindex(index=all_index, columns=all_columns)

            # For overlapping data, prefer new_data (overwrite cached data)
            # Start with cached data as base
            merged_data = cached_data.copy()

            # Overwrite with new data where it exists (not NaN)
            mask = new_data.notna()
            merged_data[mask] = new_data[mask]

            # Sort columns for consistency
            if isinstance(merged_data.columns, pd.MultiIndex):
                merged_data = merged_data.sort_index(axis=1, level=[0, 1])
            else:
                merged_data = merged_data.sort_index(axis=1)

            return merged_data

        except Exception as e:
            logger.error(f"Error merging data: {e}")
            return new_data

    def get_data(
        self,
        tickers: Union[str, List[str]],
        start: str,
        end: str,
        use_cache: bool = True,
        force_download: bool = False,
    ) -> pd.DataFrame:
        """
        Get data for multiple tickers with smart segment-based loading.

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

        # Load cached data
        cached_data = (
            self._load_cache() if use_cache and not force_download else pd.DataFrame()
        )

        if force_download:
            # Force download all data with buffer
            start_with_buffer = (
                pd.Timestamp(start) - pd.Timedelta(days=self.buffer_days)
            ).strftime("%Y-%m-%d")
            end_with_buffer = (
                pd.Timestamp(end) + pd.Timedelta(days=self.buffer_days)
            ).strftime("%Y-%m-%d")

            new_data = self._download_data(tickers, start_with_buffer, end_with_buffer)
            if not new_data.empty:
                new_data = self._normalize_dataframe_structure(new_data)
                if use_cache:
                    # Merge with existing cache and save
                    merged_data = self._merge_data(cached_data, new_data)
                    self._save_cache(merged_data)
                    cached_data = merged_data
                else:
                    cached_data = new_data
        else:
            # Smart loading: check each ticker individually
            tickers_to_download = {}  # ticker -> list of (start, end) ranges

            for ticker in tickers:
                missing_ranges = self._get_missing_date_ranges(
                    ticker, start, end, cached_data
                )
                if missing_ranges:
                    tickers_to_download[ticker] = missing_ranges
                    logger.info(f"Missing data for {ticker}: {missing_ranges}")

            # Download missing data
            if tickers_to_download:
                # Group tickers by date ranges to minimize API calls
                range_to_tickers = {}
                for ticker, ranges in tickers_to_download.items():
                    for date_range in ranges:
                        if date_range not in range_to_tickers:
                            range_to_tickers[date_range] = []
                        range_to_tickers[date_range].append(ticker)

                # Download data for each unique range
                for (range_start, range_end), range_tickers in range_to_tickers.items():
                    logger.info(
                        f"Downloading {range_tickers} for range {range_start} to {range_end}"
                    )

                    new_data = self._download_data(
                        range_tickers, range_start, range_end
                    )
                    if not new_data.empty:
                        # Merge with existing data
                        cached_data = self._merge_data(cached_data, new_data)

                # Save updated cache
                if use_cache and not cached_data.empty:
                    self._save_cache(cached_data)

        # Filter to requested tickers and date range
        if not cached_data.empty:
            # Filter by tickers
            available_tickers = [
                t for t in tickers if t in cached_data.columns.get_level_values(0)
            ]

            if available_tickers:
                # Extract ticker data
                ticker_data_list = []
                for ticker in available_tickers:
                    ticker_df = cached_data[ticker]
                    ticker_data_list.append(ticker_df)

                if ticker_data_list:
                    # Combine ticker data
                    result = pd.concat(ticker_data_list, axis=1, keys=available_tickers)
                    result.columns.names = ["Ticker", "Attribute"]

                    # Filter by date range
                    try:
                        result = result.loc[start:end]
                    except Exception as e:
                        logger.warning(f"Error filtering date range: {e}")

                    return result
            else:
                logger.warning(
                    f"No data available for any of the requested tickers: {tickers}"
                )

        return pd.DataFrame()

    def clear_cache(self):
        """Clear the entire cache file."""
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
            logger.info("Cleared cache file")
            self._initialize_cache()

    def get_cache_info(self) -> Dict[str, str]:
        """Get information about cached data.

        Returns:
            Dict[str, str]: Dictionary with ticker as key and date range as value
        """
        cache_info = {}
        data = self._load_cache()

        if not data.empty:
            data = self._normalize_dataframe_structure(data)
            for ticker in data.columns.get_level_values(0).unique():
                ticker_data = data[ticker].dropna()
                if not ticker_data.empty:
                    cache_info[ticker] = (
                        f"{ticker_data.index[0]} to {ticker_data.index[-1]}"
                    )

        return cache_info
