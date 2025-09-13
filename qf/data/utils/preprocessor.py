import numpy as np
import pandas as pd

# Import Series
from pandas import Series
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

from qf.settings import VERBOSITY
from qf.utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_sma(series, window=14):
    return SMAIndicator(series, window=window).sma_indicator()


def compute_rsi(series, window=14):
    return RSIIndicator(series, window=window).rsi()


def compute_macd(series, fast=12, slow=26, signal=9):
    macd = MACD(series, window_fast=fast, window_slow=slow, window_sign=signal)
    return macd.macd_diff()


def compute_ema(series, window=14):
    return EMAIndicator(series, window=window).ema_indicator()


def compute_adx(high, low, close, window=14):
    return ADXIndicator(high=high, low=low, close=close, window=window).adx()


def compute_bollinger_bands(series, window=20):
    bb = BollingerBands(series, window=window)
    return bb.bollinger_hband(), bb.bollinger_lband()


def compute_atr(high, low, close, window=14):
    return AverageTrueRange(
        high=high, low=low, close=close, window=window
    ).average_true_range()


def compute_obv(close, volume):
    return OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()


def add_technical_indicators(
    data,
    indicators=("sma", "rsi", "macd", "ema", "adx", "bb", "atr", "obv"),
    verbosity=VERBOSITY,
):
    """
    Adds technical indicators to OHLCV data with MultiIndex columns (ticker, field).

    Parameters:
        data (pd.DataFrame): MultiIndex columns (ticker, field) from yfinance.
        indicators (tuple): List of indicators to compute.

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    if data.empty:
        logger.info("Empty DataFrame provided. Returning empty DataFrame.")
        return data

    if not indicators:
        logger.info("No indicators specified. Returning the original data.")
        return data

    valid_indicators = {"sma", "rsi", "macd", "ema", "adx", "bb", "atr", "obv"}
    selected_indicators = set(indicators).intersection(valid_indicators)

    # Check for indicators with specific window sizes (e.g., "sma50", "ema20", "adx14", "bb20", "atr14")
    windowed_indicators = [
        ind
        for ind in indicators
        if any(
            ind.startswith(prefix) and ind[len(prefix) :].isdigit()
            for prefix in ["sma", "ema", "adx", "bb", "atr"]
        )
    ]

    if not selected_indicators and not windowed_indicators:
        logger.info(
            "No valid indicators found in the provided list. Returning the original data."
        )
        return data

    enriched = {}

    tickers = sorted(set(t for t, field in data.columns if field == "Close"))

    for ticker in tickers:
        try:
            close = data[(ticker, "Close")]
            high = data[(ticker, "High")]
            low = data[(ticker, "Low")]
            volume = data[(ticker, "Volume")]

            # Compute standard indicators
            if "sma" in selected_indicators:
                enriched[(ticker, "sma")] = compute_sma(close)

            if "rsi" in selected_indicators:
                enriched[(ticker, "rsi")] = compute_rsi(close)

            if "macd" in selected_indicators:
                enriched[(ticker, "macd")] = compute_macd(close)

            if "ema" in selected_indicators:
                enriched[(ticker, "ema")] = compute_ema(close)

            if "adx" in selected_indicators:
                enriched[(ticker, "adx")] = compute_adx(high, low, close)

            if "bb" in selected_indicators:
                bb_upper, bb_lower = compute_bollinger_bands(close)
                enriched[(ticker, "bb_upper")] = bb_upper
                enriched[(ticker, "bb_lower")] = bb_lower

            if "atr" in selected_indicators:
                enriched[(ticker, "atr")] = compute_atr(high, low, close)

            if "obv" in selected_indicators:
                enriched[(ticker, "obv")] = compute_obv(close, volume)

            # Compute indicators with specific window sizes
            for indicator in windowed_indicators:
                if indicator.startswith("sma"):
                    window = int(indicator[3:])  # Extract the window size
                    enriched[(ticker, f"sma{window}")] = compute_sma(
                        close, window=window
                    )
                elif indicator.startswith("ema"):
                    window = int(indicator[3:])
                    enriched[(ticker, f"ema{window}")] = compute_ema(
                        close, window=window
                    )
                elif indicator.startswith("adx"):
                    window = int(indicator[3:])
                    enriched[(ticker, f"adx{window}")] = compute_adx(
                        high, low, close, window=window
                    )
                elif indicator.startswith("bb"):
                    window = int(indicator[2:])
                    bb_upper, bb_lower = compute_bollinger_bands(close, window=window)
                    enriched[(ticker, f"bb_upper{window}")] = bb_upper
                    enriched[(ticker, f"bb_lower{window}")] = bb_lower
                elif indicator.startswith("atr"):
                    window = int(indicator[3:])
                    enriched[(ticker, f"atr{window}")] = compute_atr(
                        high, low, close, window=window
                    )
        except Exception as e:
            logger.error(f"Error calculating indicators for {ticker}: {str(e)}")
            continue

    if not enriched:
        logger.info("No indicators were computed. Returning the original data.")
        return data

    enriched_df = pd.DataFrame(enriched, index=data.index)

    # Ensure MultiIndex columns only if enriched_df is not empty
    if not enriched_df.empty:
        enriched_df.columns = pd.MultiIndex.from_tuples(enriched_df.columns)

    result = pd.concat([data, enriched_df], axis=1)
    return result.ffill().bfill().dropna()
