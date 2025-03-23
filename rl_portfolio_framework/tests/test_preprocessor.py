import unittest
import pandas as pd
import numpy as np
from data.downloader import download_data
from data.preprocessor import compute_sma, compute_rsi, compute_macd


class TestIndicators(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.series = pd.Series(np.linspace(100, 110, 100))  # lineare Serie
        self.noisy_series = pd.Series(100 + np.random.randn(100).cumsum())  # simulierte Preisreihe

    def test_sma(self):
        sma = compute_sma(self.series, window=10)
        self.assertEqual(len(sma), 100)
        self.assertTrue(np.all(pd.isna(sma[:9])))  # erste 9 Werte = NaN
        self.assertAlmostEqual(sma.iloc[10], np.mean(self.series[1:11]), places=5)

    def test_rsi(self):
        rsi = compute_rsi(self.series, window=14)
        self.assertEqual(len(rsi), 100)
        self.assertTrue(rsi.max() <= 100)
        self.assertTrue(rsi.min() >= 0)

    def test_macd(self):
        # Beispieldaten laden
        tickers = ["AAPL"]
        start = "2022-01-01"
        end = "2022-06-01"
        df = download_data(tickers, start, end)

        close = df[("AAPL", "Close")]
        macd = compute_macd(close)

        valid_macd = macd.dropna()
        self.assertGreater(len(valid_macd), 0, "MACD has no valid values")

        # Optionaler Richtwertvergleich: z. B. Trend erkennen
        self.assertGreater(valid_macd.iloc[-1], valid_macd.iloc[0], "MACD did not increase")



    def test_rsi_random_data(self):
        rsi = compute_rsi(self.noisy_series)
        self.assertTrue((0 <= rsi.dropna()).all() and (rsi.dropna() <= 100).all())

    def test_macd_noisy_data(self):
        macd = compute_macd(self.noisy_series)
        self.assertEqual(len(macd), len(self.noisy_series))


if __name__ == "__main__":
    unittest.main()
