import unittest
import pandas as pd
from data.get_data import download_data
import unittest
import pandas as pd
from data.get_data import download_data


# python -m unittest tests.test_downloader


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.tickers = ["AAPL", "MSFT"]
        self.start = "2020-01-01"
        self.end = "2020-03-01"
        self.df = download_data(self.tickers, self.start, self.end)

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
        self.assertGreaterEqual(self.df.index.min(), pd.to_datetime(self.start))
        self.assertLessEqual(self.df.index.max(), pd.to_datetime(self.end))

    def test_no_missing_values(self):
        self.assertFalse(self.df.isnull().any().any(), "DataFrame contains NaNs")

    def test_single_ticker_consistency(self):
        single_df = download_data("AAPL", self.start, self.end)

        self.assertIsInstance(single_df.columns, pd.MultiIndex, "Single ticker should return MultiIndex columns")

        self.assertIn(("AAPL", "Close"), single_df.columns, "Expected ('AAPL', 'Close') column not found")



if __name__ == "__main__":
    unittest.main()
