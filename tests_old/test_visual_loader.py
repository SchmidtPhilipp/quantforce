import unittest
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
from ta.momentum import RSIIndicator
from utils.plot import plot_lines_grayscale

import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
from data.get_data import download_data
from data.preprocessor import compute_rsi, add_technical_indicators

# Configure matplotlib to export PGF for LaTeX
mpl.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "pgf.rcfonts": False,
})



# Test the visual loader
#  python -m unittest discover -s tests         
# python -m unittest tests.test_visual_loader

class TestVisualLoader(unittest.TestCase):

    line_width = 1.8
    gray_colors = {
        "close": "black",
        "sma": "dimgray",
        "rsi": "black",
        "macd": "black",
        "ema": "slategray",
        "bb_upper": "lightgray",
        "bb_lower": "lightgray"
    }

    def setUp(self):
        os.makedirs("tmp/plots", exist_ok=True)

    def test_plot_all_indicators(self):
        tickers = ["AAPL"]
        start = "2020-01-01"
        end = "2021-01-01"
        df = download_data(tickers, start, end)
        df = add_technical_indicators(df, indicators=("sma", "rsi", "macd", "ema", "bb"))

        ticker = tickers[0]
        df_single = df.xs(ticker, axis=1, level=0)

        # Plot 1: Preis + SMA, EMA, BB
        price_df = df_single[["Close", "sma", "ema", "bb_upper", "bb_lower"]]
        plot_lines_grayscale(
            df=price_df,
            ylabel="Price",
            title=f"{ticker} – Price Trend Indicators",
            filename="aapl_price_indicators"
        )

        # Plot 2: RSI
        rsi_df = df_single[["rsi"]]
        plot_lines_grayscale(
            df=rsi_df,
            ylabel="RSI",
            title=f"{ticker} – Relative Strength Index",
            filename="aapl_rsi", 
            y_limits=(0, 100)
        )

        # Plot 3: MACD
        macd_df = df_single[["macd"]]
        plot_lines_grayscale(
            df=macd_df,
            ylabel="MACD",
            title=f"{ticker} – MACD",
            filename="aapl_macd"
        )

        # Plot 4: Volume
        volume_df = df_single[["Volume"]]
        plot_lines_grayscale(
            df=volume_df,
            ylabel="Volume",
            title=f"{ticker} – Volume",
            filename="aapl_volume"
        )

        #print("✅ test_plot_all_indicators – All indicator plots saved.")

    def test_plot_preprocessed_indicators(self):
        tickers = ["AAPL"]
        start = "2022-01-01"
        end = "2022-06-01"

        df = download_data(tickers, start, end)
        close = df[("AAPL", "Close")]

        rsi_custom = compute_rsi(close)
        rsi_ref = RSIIndicator(close, window=14).rsi()

        # Combine indicators and reference lines
        rsi_df = pd.DataFrame({
            "TA RSI": rsi_ref,
            "Overbought (70)": 70,
            "Oversold (30)": 30
        })

        # Plot with unified grayscale function
        plot_lines_grayscale(
            df=rsi_df,
            xlabel="Date",
            ylabel="RSI",
            title="RSI Comparison – AAPL",
            filename="aapl_rsi_comparison",
            save_dir="tmp/plots",
            y_limits=(0, 100)
        )

        #print("✅ test_plot_preprocessed_indicators – Plot saved to tmp/plots/aapl_rsi_comparison.(pgf|png)")



if __name__ == "__main__":
    unittest.main()




