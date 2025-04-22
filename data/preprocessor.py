import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import numpy as np

# Import Series
from pandas import Series

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
    return AverageTrueRange(high=high, low=low, close=close, window=window).average_true_range()

def compute_obv(close, volume):
    return OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

def add_technical_indicators(data, indicators=("sma", "rsi", "macd", "ema", "adx", "bb", "atr", "obv"), verbosity=0):
    """
    Adds technical indicators to OHLCV data with MultiIndex columns (ticker, field).

    Parameters:
        data (pd.DataFrame): MultiIndex columns (ticker, field) from yfinance.
        indicators (tuple): List of indicators to compute.

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    if not indicators:
        print("No indicators specified. Returning the original data.")
        return data

    valid_indicators = {"sma", "rsi", "macd", "ema", "adx", "bb", "atr", "obv"}
    selected_indicators = set(indicators).intersection(valid_indicators)

    if not selected_indicators:
        if verbosity > 0:
            print("No valid indicators found in the provided list. Returning the original data.")
        return data

    data = data.copy()
    enriched = {}

    tickers = sorted(set(t for t, field in data.columns if field == "Close"))

    for ticker in tickers:
        close = data[(ticker, "Close")].squeeze()
        high = data[(ticker, "High")].squeeze()
        low = data[(ticker, "Low")].squeeze()
        volume = data[(ticker, "Volume")].squeeze()

        # Ensure that close, high, low, and volume are Series
        if not isinstance(close, Series):
            close = pd.Series(close.values, index=data.index, name="Close")
        if not isinstance(high, Series):
            high = pd.Series(high.values, index=data.index, name="High")
        if not isinstance(low, Series):
            low = pd.Series(low.values, index=data.index, name="Low")
        if not isinstance(volume, Series):
            volume = pd.Series(volume.values, index=data.index, name="Volume")

        # Compute indicators as before
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
            enriched[(ticker, "bb_upper")], enriched[(ticker, "bb_lower")] = compute_bollinger_bands(close)

        if "atr" in selected_indicators:
            enriched[(ticker, "atr")] = compute_atr(high, low, close)

        if "obv" in selected_indicators:
            enriched[(ticker, "obv")] = compute_obv(close, volume)

    if not enriched:
        print("No indicators were computed. Returning the original data.")
        return data

    enriched_df = pd.DataFrame(enriched, index=data.index)

    # Ensure MultiIndex columns only if enriched_df is not empty
    if not enriched_df.empty:
        enriched_df.columns = pd.MultiIndex.from_tuples(enriched_df.columns)

    result = pd.concat([data, enriched_df], axis=1)
    return result.ffill().bfill().dropna()