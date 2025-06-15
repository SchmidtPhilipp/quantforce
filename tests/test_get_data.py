import os
import unittest

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

    def test_shape_and_columns(self):
        self.assertTrue(len(self.df) > 0, "DataFrame is empty")
        self.assertIsInstance(
            self.df.columns, pd.MultiIndex, "Columns are not MultiIndex"
        )

        expected_fields = {"Open", "High", "Low", "Close"}
        tickers_in_df = {t for t, _ in self.df.columns}
        fields_in_df = {f for _, f in self.df.columns}

        self.assertTrue(set(self.tickers).issubset(tickers_in_df), "Missing tickers")
        self.assertTrue(
            expected_fields.issubset(fields_in_df), "Missing fields in data"
        )

    def test_index_is_datetime(self):
        self.assertTrue(
            isinstance(self.df.index, pd.DatetimeIndex), "Index is not datetime"
        )

    def test_date_range(self):
        self.assertLessEqual(self.df.index.min(), pd.to_datetime(self.start))
        n_delta_max = 10
        self.assertGreaterEqual(
            self.df.index.max(),
            pd.to_datetime(self.end) - pd.Timedelta(days=n_delta_max),
        )
        self.assertLessEqual(self.df.index.max(), pd.Timestamp("today"))

    def test_no_missing_values(self):
        self.assertFalse(self.df.isnull().any().any(), "DataFrame contains NaNs")

    def test_invalid_n_trading_days(self):
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

    def test_invalid_imputation_method(self):
        with self.assertRaises(ValueError) as context:
            get_data(
                tickers=self.tickers,
                start=self.start,
                end=self.end,
                imputation_method="invalid_method",  # Invalid value
                verbosity=1,
                cache_dir=self.cache_dir,
            )
        self.assertEqual(
            str(context.exception),
            "Unknown imputation method: invalid_method. Use 'bfill' or 'shrinkage'.",
        )

    def tearDown(self):
        # Lösche den Cache-Ordner nach den Tests
        if os.path.exists(self.cache_dir):
            import shutil

            shutil.rmtree(self.cache_dir)


if __name__ == "__main__":
    unittest.main()
