import unittest
import pandas as pd
import os
from qf.data.utils.load_data import load_data


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.tickers = ["TSLA", "MSFT"]
        self.start = "2006-01-01"
        self.end = "2021-01-01"
        self.cache_dir = "tests/test_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.df = load_data(self.tickers, self.start, self.end, verbosity=1, cache_dir=self.cache_dir)

    def test_shape_and_columns(self):
        self.assertTrue(len(self.df) > 0, "DataFrame is empty")
        self.assertIsInstance(self.df.columns, pd.MultiIndex, "Columns are not MultiIndex")

        expected_fields = {"Open", "High", "Low", "Close"}
        tickers_in_df = {t for t, _ in self.df.columns}
        fields_in_df = {f for _, f in self.df.columns}

        self.assertTrue(set(self.tickers).issubset(tickers_in_df), "Missing tickers")
        self.assertTrue(expected_fields.issubset(fields_in_df), "Missing fields in data")

    def test_index_is_datetime(self):
        self.assertTrue(isinstance(self.df.index, pd.DatetimeIndex), "Index is not datetime")

    def test_date_range(self):
        # The start date should be before or at the the given start date
        self.assertLessEqual(self.df.index.min(), pd.to_datetime(self.start))
        # The end date can be larger or smaller than the given end date. But it can neverbe larger than todays date. It may be smaller than the given date by a maximum of 10 days.
        n_delta_max = 10
        self.assertGreaterEqual(self.df.index.max(), pd.to_datetime(self.end) - pd.Timedelta(days=n_delta_max))
        self.assertLessEqual(self.df.index.max(), pd.Timestamp("today"))

    def test_no_missing_values(self):
        self.assertFalse(self.df.isnull().any().any(), "DataFrame contains NaNs")

    def test_single_ticker_consistency(self):
        single_df = load_data("AAPL", self.start, self.end, verbosity=1, cache_dir=self.cache_dir)
        self.assertIsInstance(single_df.columns, pd.MultiIndex, "Single ticker should return MultiIndex columns")
        self.assertIn(("AAPL", "Close"), single_df.columns, "Expected ('AAPL', 'Close') column not found")

    def test_extended_date_range(self):
        extended_start = "2019-01-01"
        extended_df = load_data(self.tickers, extended_start, self.end, verbosity=1, cache_dir=self.cache_dir)
        self.assertLessEqual(extended_df.index.min(), pd.to_datetime(extended_start))
        n_delta_max = 10
        self.assertGreaterEqual(self.df.index.max(), pd.to_datetime(self.end) - pd.Timedelta(days=n_delta_max))
        self.assertLessEqual(self.df.index.max(), pd.Timestamp("today"))

        extended_end = "2022-01-01"
        extended_df = load_data(self.tickers, self.start, extended_end, verbosity=1, cache_dir=self.cache_dir)
        self.assertLessEqual(extended_df.index.min(), pd.to_datetime(self.start))
        self.assertGreaterEqual(self.df.index.max(), pd.to_datetime(self.end) - pd.Timedelta(days=n_delta_max))
        self.assertLessEqual(self.df.index.max(), pd.Timestamp("today"))

    def test_new_ticker_integration(self):
        tickers = ["AAPL", "MSFT", "GOOGL"]
        extended_df = load_data(tickers, self.start, self.end, verbosity=1, cache_dir=self.cache_dir)
        self.assertIn(("GOOGL", "Close"), extended_df.columns, "Expected ('GOOGL', 'Close') column not found")

    def test_empty_tickers(self):
        tickers = ["AAPL", "MSFT", "INVALID_TICKER"]
        df = load_data(tickers, self.start, self.end, verbosity=1, cache_dir=self.cache_dir)
        empty_tickers = []
        for ticker in tickers:
            if df[ticker].isnull().all().all():
                empty_tickers.append(ticker)
        self.assertIn("INVALID_TICKER", empty_tickers, "Expected INVALID_TICKER to be empty")

    def tearDown(self):
        # Lösche den Cache-Ordner nach den Tests
        if os.path.exists(self.cache_dir):
            import shutil
            shutil.rmtree(self.cache_dir)


if __name__ == "__main__":
    unittest.main()
