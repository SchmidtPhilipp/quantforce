"""Unit tests for the DataManager class."""

import os
import shutil
import unittest
from datetime import datetime, timedelta

import pandas as pd

from qf.data.utils.data_manager import DataManager


class TestDataManager(unittest.TestCase):
    """Test cases for the DataManager class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary cache directory for testing
        self.test_cache_dir = "test_cache"
        os.makedirs(self.test_cache_dir, exist_ok=True)

        # Initialize DataManager with test settings
        self.data_manager = DataManager(
            cache_dir=self.test_cache_dir,
            interval="1d",
            downloader="simulate",  # Use simulate downloader for testing
            verbosity=2,
        )

        # Test parameters
        self.tickers = ["AAPL", "MSFT"]
        self.start = "2024-01-01"
        self.end = "2024-03-01"

    def tearDown(self):
        """Clean up after each test."""
        # Remove test cache directory
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def _run_all_tests(self, downloader):
        data_manager = DataManager(
            cache_dir=self.test_cache_dir,
            interval="1d",
            downloader=downloader,
            verbosity=2,
        )
        # test_get_data_basic
        with self.subTest(downloader=downloader, test="get_data_basic"):
            data = data_manager.get_data(self.tickers, self.start, self.end)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertFalse(data.empty)
            self.assertIsInstance(data.index, pd.DatetimeIndex)
            expected_columns = ["Open", "High", "Low", "Close", "Volume"]
            for ticker in self.tickers:
                for col in expected_columns:
                    self.assertIn((ticker, col), data.columns)

        # test_cache_functionality
        with self.subTest(downloader=downloader, test="cache_functionality"):
            data1 = data_manager.get_data(self.tickers, self.start, self.end)
            data2 = data_manager.get_data(self.tickers, self.start, self.end)
            pd.testing.assert_frame_equal(data1, data2)
            cache_info = data_manager.get_cache_info()
            self.assertGreater(len(cache_info), 0)

        # test_empty_data_handling
        with self.subTest(downloader=downloader, test="empty_data_handling"):
            data = data_manager.get_data(["NONEXISTENT"], self.start, self.end)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertTrue(data.empty or data.isna().all().all())
            self.assertIsInstance(data.index, pd.DatetimeIndex)

        # test_date_range_handling
        with self.subTest(downloader=downloader, test="date_range_handling"):
            date_ranges = [
                ("2024-01-01", "2024-01-31"),
                ("2024-02-01", "2024-02-29"),
                ("2024-03-01", "2024-03-31"),
            ]
            for start, end in date_ranges:
                data = data_manager.get_data(self.tickers, start, end)
                self.assertIsInstance(data, pd.DataFrame)
                self.assertIsInstance(data.index, pd.DatetimeIndex)
                if not data.empty:
                    start_ts = pd.Timestamp(start).tz_localize(None)
                    end_ts = pd.Timestamp(end).tz_localize(None)
                    self.assertGreaterEqual(data.index[0], start_ts)
                    self.assertLessEqual(data.index[-1], end_ts)

        # test_multiindex_creation
        with self.subTest(downloader=downloader, test="multiindex_creation"):
            data = data_manager.get_data(self.tickers, self.start, self.end)
            self.assertIsInstance(data.columns, pd.MultiIndex)
            for ticker in self.tickers:
                self.assertIn(ticker, data.columns.get_level_values(0))

    def test_simulate_and_yfinance(self):
        for downloader in ["simulate", "yfinance"]:
            with self.subTest(downloader=downloader):
                self._run_all_tests(downloader)

    def test_cache_functionality(self):
        """Test caching functionality."""
        print("\nTesting cache functionality...")

        # First load - should download
        data1 = self.data_manager.get_data(self.tickers, self.start, self.end)

        # Second load - should use cache
        data2 = self.data_manager.get_data(self.tickers, self.start, self.end)

        # Check if data is the same
        pd.testing.assert_frame_equal(data1, data2)

        # Check cache info
        cache_info = self.data_manager.get_cache_info()
        self.assertGreater(len(cache_info), 0)

        print("Cache info:")
        for ticker, date_range in cache_info.items():
            print(f"{ticker}: {date_range}")

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        print("\nTesting empty data handling...")

        # Try loading data for a non-existent ticker
        data = self.data_manager.get_data(["NONEXISTENT"], self.start, self.end)

        self.assertIsInstance(data, pd.DataFrame)
        # Accept both empty DataFrames and all-NaN DataFrames as valid empty results
        self.assertTrue(data.empty or data.isna().all().all())
        self.assertIsInstance(data.index, pd.DatetimeIndex)

        print(f"Empty data shape: {data.shape}")
        print(f"Empty data columns: {data.columns}")

    def test_date_range_handling(self):
        """Test handling of different date ranges."""
        print("\nTesting date range handling...")

        # Test different date ranges
        date_ranges = [
            ("2024-01-01", "2024-01-31"),
            ("2024-02-01", "2024-02-29"),
            ("2024-03-01", "2024-03-31"),
        ]

        for start, end in date_ranges:
            print(f"\nTesting range: {start} to {end}")
            data = self.data_manager.get_data(self.tickers, start, end)

            self.assertIsInstance(data, pd.DataFrame)
            self.assertIsInstance(data.index, pd.DatetimeIndex)

            # Skip date range checks for empty DataFrames
            if not data.empty:
                # Convert start to timezone-naive timestamp for comparison
                start_ts = pd.Timestamp(start).tz_localize(None)
                end_ts = pd.Timestamp(end).tz_localize(None)

                self.assertGreaterEqual(data.index[0], start_ts)
                self.assertLessEqual(data.index[-1], end_ts)

            print(f"Data shape: {data.shape}")
            if not data.empty:
                print(f"Date range: {data.index[0]} to {data.index[-1]}")

    def test_multiindex_creation(self):
        """Test creation of multi-index DataFrame."""
        print("\nTesting multi-index creation...")

        data = self.data_manager.get_data(self.tickers, self.start, self.end)

        # Check if it's a multi-index DataFrame
        self.assertIsInstance(data.columns, pd.MultiIndex)

        # Check structure
        for ticker in self.tickers:
            self.assertIn(ticker, data.columns.get_level_values(0))

        print(f"Multi-index levels: {data.columns.levels}")
        print(f"Multi-index names: {data.columns.names}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
