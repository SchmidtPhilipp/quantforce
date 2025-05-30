
import os
import sys
import os

# Include ../../ to access the get_data and tickers modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


from datetime import datetime
from data import get_data
from utils.plot import plot_lines_grayscale, plot_dual_axis

def test_plot_all_indicators():
    tickers = ["AAPL"]
    start = "2024-01-01"
    end = "2025-01-01"

    df = get_data(tickers, start, end, indicators=("sma25", 
                                                   "sma50", 
                                                   "sma200", 
                                                   "ema25", 
                                                   "ema50", 
                                                   "ema200", 
                                                   "rsi", 
                                                   "macd",
                                                    "ema", 
                                                   "bb", 
                                                   "bb_upper", 
                                                   "bb_lower", 
                                                   "Volume", 
                                                   "adx",
                                                   "atr",
                                                   "obv"),)
    folder = os.path.dirname(os.path.abspath(__file__))
    ticker = tickers[0]
    df_single = df.xs(ticker, axis=1, level=0)

    linewidth = 1


    # Plot 1: Preis +  BB
    price_df = df_single[["Close", "bb_upper", "bb_lower"]]
    plot_lines_grayscale(
        df=price_df,
        ylabel="Price",
        #title=f"{ticker} – Price Trend Indicators",
        filename="aapl_bollinger_bands",
        save_dir=folder,
        linewidth=linewidth,
        figsize=(8, 2.5)
    )

    # Plot 1.1: SMA
    price_df = df_single[["Close", "sma25", "sma50", "sma200"]]
    plot_lines_grayscale(
        df=price_df,
        ylabel="Price",
        #title=f"{ticker} – Price Trend Indicators",
        filename="aapl_sma",
        save_dir=folder,
        linewidth=linewidth,
        figsize=(8, 2.5)
    )
    
    # Plot 1.2: EMA
    price_df = df_single[["Close", "ema25", "ema50", "ema200"]]
    plot_lines_grayscale(
        df=price_df,
        ylabel="Price",
        #title=f"{ticker} – Price Trend Indicators",
        filename="aapl_ema",
        save_dir=folder,
        linewidth=linewidth,
        figsize=(8, 2.5)
    )
    

    # Plot 2: RSI
    rsi_df = df_single[["rsi"]]
    plot_lines_grayscale(
        df=rsi_df,
        ylabel="RSI",
        #title=f"{ticker} – Relative Strength Index",
        filename="aapl_rsi", 
        y_limits=(0, 100), 
        save_dir=folder,
        linewidth=linewidth,
        figsize=(8, 2.5)
    )

    # Plot 3: MACD
    macd_df = df_single[["macd"]]
    plot_lines_grayscale(
        df=macd_df,
        ylabel="MACD",
        #title=f"{ticker} – MACD",
        filename="aapl_macd",
        save_dir=folder,
        linewidth=linewidth,
        figsize=(8, 2.5)
    )
    plot_dual_axis(
        df=df_single[["macd", "ema"]],  # MACD and Signal Line
        ylabel_left="MACD",
        filename="aapl_macd_dual",
        save_dir=folder,
        linewidth=linewidth,
        ylabel_right="Close",
        y_limits_left=(-4, 4),
        y_limits_right=(150, 250),
    )

    # Plot 4: Volume
    volume_df = df_single[["Volume"]]
    plot_lines_grayscale(
        df=volume_df,
        ylabel="Volume",
        #title=f"{ticker} – Volume",
        filename="aapl_volume",
        save_dir=folder,
        linewidth=linewidth,
        figsize=(8, 2.5)
    )

    # Plot 7: ADX
    adx_df = df_single[["adx"]]
    plot_lines_grayscale(
        df=adx_df,
        ylabel="ADX",
        #title=f"{ticker} – Average Directional Index",
        filename="aapl_adx",
        save_dir=folder,
        linewidth=linewidth,
        figsize=(8, 2.5)
    )


    # Plot 8: ATR
    atr_df = df_single[["atr"]]
    plot_lines_grayscale(
        df=atr_df,
        ylabel="ATR",
        #title=f"{ticker} – Average True Range",
        filename="aapl_atr",
        save_dir=folder,
        linewidth=linewidth,
        figsize=(8, 2.5)
    )

    # Plot 9 OBV
    obv_df = df_single[["obv"]]
    plot_lines_grayscale(
        df=obv_df,
        ylabel="OBV",
        #title=f"{ticker} – On-Balance Volume",
        filename="aapl_obv",
        save_dir=folder,
        linewidth=linewidth,
        figsize=(8, 2.5)
    )


    #print("✅ test_plot_all_indicators – All indicator plots saved.")


if __name__ == "__main__":
    test_plot_all_indicators()