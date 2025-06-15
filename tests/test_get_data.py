import os
import unittest
from datetime import datetime, timedelta

import pandas as pd

from qf.data.utils.get_data import get_data


class TestGetData(unittest.TestCase):

    def setUp(self):
        self.tickers = ["TSLA", "MSFT"]
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

    def test_cache_functionality(self):
        """Test that caching works correctly."""
        # First call should download data
        df1 = get_data(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            verbosity=1,
            cache_dir=self.cache_dir,
        )

        # Second call should use cache
        df2 = get_data(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            verbosity=1,
            cache_dir=self.cache_dir,
        )

        pd.testing.assert_frame_equal(df1, df2, "Cached data differs from original")

        # Test future dates
        future_start = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        future_end = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        future_df = get_data(
            tickers=self.tickers,
            start=future_start,
            end=future_end,
            verbosity=1,
            cache_dir=self.cache_dir,
        )
        self.assertTrue(future_df.empty, "Data found for future dates")

    def test_empty_data_handling(self):
        """Test handling of non-existent tickers."""
        non_existent_tickers = ["NONEXISTENT1", "NONEXISTENT2"]
        empty_df = get_data(
            tickers=non_existent_tickers,
            start=self.start,
            end=self.end,
            verbosity=1,
            cache_dir=self.cache_dir,
        )
        self.assertTrue(empty_df.empty, "Data found for non-existent tickers")

    def test_different_downloaders(self):
        """Test both yfinance and simulate downloaders."""
        # Test yfinance
        yf_df = get_data(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            downloader="yfinance",
            verbosity=1,
            cache_dir=self.cache_dir,
        )
        self.assertTrue(len(yf_df) > 0, "No data from yfinance")

        # Test simulate
        sim_df = get_data(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            downloader="simulate",
            verbosity=1,
            cache_dir=self.cache_dir,
        )
        self.assertTrue(len(sim_df) > 0, "No data from simulate")

    def test_invalid_n_trading_days(self):
        """Test validation of n_trading_days parameter."""
        with self.assertRaises(ValueError) as context:
            get_data(
                tickers=self.tickers,
                start=self.start,
                end=self.end,
                n_trading_days=300,  # Invalid value
                verbosity=1,
                cache_dir=self.cache_dir,
            )
        self.assertEqual(
            str(context.exception), "n_trading_days must be either 252 or 365."
        )

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test single ticker
        single_df = get_data(
            tickers=[self.tickers[0]],
            start=self.start,
            end=self.end,
            verbosity=1,
            cache_dir=self.cache_dir,
        )
        self.assertTrue(len(single_df) > 0, "No data for single ticker")

        # Test same start and end date
        same_date = "2020-01-01"
        same_date_df = get_data(
            tickers=self.tickers,
            start=same_date,
            end=same_date,
            verbosity=1,
            cache_dir=self.cache_dir,
        )
        self.assertTrue(len(same_date_df) > 0, "No data for same start/end date")

        # Test very old dates
        old_df = get_data(
            tickers=self.tickers,
            start="1900-01-01",
            end="1900-12-31",
            verbosity=1,
            cache_dir=self.cache_dir,
        )
        self.assertTrue(old_df.empty, "Data found for very old dates")

    def tearDown(self):
        """Clean up test cache directory."""
        if os.path.exists(self.cache_dir):
            import shutil

            shutil.rmtree(self.cache_dir)


if __name__ == "__main__":
    unittest.main()
