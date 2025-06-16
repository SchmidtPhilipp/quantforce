import os
import unittest
from datetime import datetime, timedelta

import pandas as pd

from qf.data.utils.get_data import get_data


class TestGetData(unittest.TestCase):

    def setUp(self):
        self.tickers = ["AAPL", "MSFT"]  # Using known tickers for simulate mode
        self.start = "2006-01-01"
        self.end = "2021-01-01"
        self.cache_dir = "tests/test_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.indicators = ["sma", "rsi"]
        self.n_trading_days = 365
        self.imputation_method = "bfill"

        self.df = get_data(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            indicators=self.indicators,
            verbosity=1,
            cache_dir=self.cache_dir,
            n_trading_days=self.n_trading_days,
            imputation_method=self.imputation_method,
        )

    def test_index_is_datetime(self):
        """Test that index is datetime and properly formatted."""
        self.assertTrue(
            isinstance(self.df.index, pd.DatetimeIndex), "Index is not datetime"
        )
        self.assertTrue(self.df.index.is_monotonic_increasing, "Index is not sorted")
        self.assertTrue(self.df.index.is_unique, "Index has duplicate dates")
        # Test that index is timezone-naive
        self.assertIsNone(self.df.index.tz, "Index should be timezone-naive")

    def test_date_range(self):
        """Test date range constraints."""
        self.assertLessEqual(self.df.index.min(), pd.to_datetime(self.start))
        n_delta_max = 10
        self.assertGreaterEqual(
            self.df.index.max(),
            pd.to_datetime(self.end) - pd.Timedelta(days=n_delta_max),
        )
        self.assertLessEqual(self.df.index.max(), pd.Timestamp("today"))

    def test_no_missing_values(self):
        """Test data completeness."""
        self.assertFalse(self.df.isnull().any().any(), "DataFrame contains NaNs")

    def test_indicator_calculation(self):
        """Test that indicators are calculated correctly."""
        # Test SMA
        for ticker in self.tickers:
            sma_col = (ticker, "sma")
            close_col = (ticker, "Close")
            self.assertTrue(sma_col in self.df.columns, f"SMA not found for {ticker}")
            # SMA should be less volatile than Close
            self.assertLess(
                self.df[sma_col].std(),
                self.df[close_col].std(),
                f"SMA volatility not less than Close for {ticker}",
            )

        # Test RSI
        for ticker in self.tickers:
            rsi_col = (ticker, "rsi")
            self.assertTrue(rsi_col in self.df.columns, f"RSI not found for {ticker}")
            # RSI should be between 0 and 100
            self.assertTrue(
                (self.df[rsi_col] >= 0).all() and (self.df[rsi_col] <= 100).all(),
                f"RSI values out of range for {ticker}",
            )

    def tearDown(self):
        """Clean up test cache directory."""
        if os.path.exists(self.cache_dir):
            import shutil

            shutil.rmtree(self.cache_dir)


if __name__ == "__main__":
    unittest.main()
