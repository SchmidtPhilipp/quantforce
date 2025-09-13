"""
Data Configuration Module.

This module provides the DataConfig class which defines the configuration for
data sources, preprocessing, and management in the QuantForce framework.
It includes settings for ticker selection, date ranges, technical indicators,
and data imputation methods.
"""

import os
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional, Set

from qf.data.tickers.tickers import (
    DOWJONES,
    MSCIACWI,
    MSCIEU,
    MSCIWORLD,
    NASDAQ100,
    SNP500,
)

AVAILABLE_TICKER_SETS = {
    "DOWJONES": DOWJONES,
    "NASDAQ100": NASDAQ100,
    "SNP500": SNP500,
    "MSCIACWI": MSCIACWI,
    "MSCIWORLD": MSCIWORLD,
    "MSCIEU": MSCIEU,
}

ALL_INDICATORS = [
    "sma",
    "rsi",
    "macd",
    "ema",
    "adx",
    "bb",
    "atr",
    "obv",
    "open",
    "high",
    "low",
    "volume",
]


@dataclass
class DataConfig:
    """
    Configuration for data sources, preprocessing, and management.

    This class defines all parameters needed to configure data sources,
    preprocessing steps, and data management in the QuantForce framework.
    It includes settings for ticker selection, date ranges, technical
    indicators, and various data imputation methods.

    The DataConfig supports multiple ticker sets, various technical indicators,
    and sophisticated imputation methods to handle missing data in financial
    time series.

    Attributes:
        tickers (List[str]): List of stock ticker symbols to include.
            Changed from Set[str] to List[str] for JSON serialization.
        start (str): Start date for data collection in 'YYYY-MM-DD' format.
        end (str): End date for data collection in 'YYYY-MM-DD' format.
        interval (Literal["1d"]): Time interval for data collection.
            Currently only "1d" (daily) is supported. Default: "1d".
        indicators (List[str]): List of technical indicators to compute.
            Changed from Set[str] to List[str] for JSON serialization.
            Default: All available indicators.
        cache_dir (Optional[str]): Directory for caching downloaded data.
            Default: ~/qf_cache.
        downloader (Literal["yfinance"]): Data downloader to use.
            Currently only "yfinance" is supported. Default: "yfinance".
        n_trading_days (Optional[int]): Number of trading days per year.
            Used for annualization calculations. Default: 252.
        imputation_method (Literal): Method for handling missing data. Options:
            - "ffill": Forward fill
            - "linear_interpolation": Linear interpolation
            - "log_interpolation": Log-linear interpolation
            - "spline_interpolation": Spline interpolation
            - "polynomial_interpolation": Polynomial interpolation
            - "quadratic_interpolation": Quadratic interpolation
            - "cubic_interpolation": Cubic interpolation
            - "keep_nan": Keep NaN values
            - "shrinkage": Shrinkage estimation
            - "linear_kalman_global": Linear Kalman filter (global)
            - "linear_kalman_velocity_global": Linear Kalman with velocity
            - "ekf_gbm_deterministic_global": Extended Kalman GBM (deterministic)
            - "ekf_gbm_stochastic_global": Extended Kalman GBM (stochastic)
            - "ekf_gbm_stochastic_local": Extended Kalman GBM (local)
            - "gbm_monte_carlo_unconstrained_end_median": GBM Monte Carlo
            - "gbm_monte_carlo_unconstrained_end_sample": GBM Monte Carlo sample
            - "gbm_monte_carlo_unconstrained_end_mean": GBM Monte Carlo mean
            - "gbm_bridge_impute_with_unit_variance": GBM bridge with unit variance
            - "gbm_bridge_impute_with_global_variance": GBM bridge with global variance
            - "gbm_bridge_impute_with_local_variance": GBM bridge with local variance
            - "gbm_bridge_impute_with_rolling_variance": GBM bridge with rolling variance
            Default: "log_interpolation".
        backfill_method (Literal): Method for backfilling missing data at start.
            Options: "bfill", "shrinkage", "remove_short_stocks", "keep_nan",
            "insert_zeros", "none". Default: "bfill".
        force_download (bool): Force re-download of cached data.
            Default: False.
        use_cache (bool): Whether to use cached data when available.
            Default: True.
        use_adjusted_close (bool): Whether to use adjusted close prices.
            Default: True.
        use_autorepair (bool): Whether to automatically repair data issues.
            Default: False.

    Example:
        >>> from qf.data.config.data_config import DataConfig
        >>>
        >>> # Create a basic data configuration
        >>> config = DataConfig(
        ...     tickers=["AAPL", "MSFT", "GOOGL"],
        ...     start="2020-01-01",
        ...     end="2023-01-01",
        ...     indicators=["rsi", "sma", "macd"]
        ... )
        >>>
        >>> # Create a configuration for training data
        >>> train_config = DataConfig.get_default_train_config()
        >>>
        >>> # Create a configuration for evaluation data
        >>> eval_config = DataConfig.get_default_eval_config()
        >>>
        >>> # Create a configuration with advanced imputation
        >>> advanced_config = DataConfig(
        ...     tickers=list(DOWJONES),
        ...     start="2018-01-01",
        ...     end="2023-01-01",
        ...     imputation_method="ekf_gbm_stochastic_global",
        ...     backfill_method="shrinkage"
        ... )
    """

    tickers: List[str]  # Changed from Set[str] to List[str] for JSON serialization
    start: str
    end: str
    interval: Literal["1d"] = "1d"  # Other time frames are not supported yet
    indicators: List[str] = field(
        default_factory=lambda: sorted(ALL_INDICATORS)
    )  # Changed from Set[str] to List[str]
    cache_dir: Optional[str] = os.path.join(os.path.expanduser("~"), "qf_cache")
    downloader: Literal["yfinance"] = "yfinance"
    n_trading_days: Optional[int] = 252
    imputation_method: Literal[
        "ffill",
        "linear_interpolation",
        "log_interpolation",
        "spline_interpolation",
        "polynomial_interpolation",
        "quadratic_interpolation",
        "cubic_interpolation",
        "keep_nan",
        "shrinkage",
        "linear_kalman_global",
        "linear_kalman_velocity_global",
        "ekf_gbm_deterministic_global",
        "ekf_gbm_stochastic_global",
        "ekf_gbm_stochastic_local",
        "gbm_monte_carlo_unconstrained_end_median",
        "gbm_monte_carlo_unconstrained_end_sample",
        "gbm_monte_carlo_unconstrained_end_mean",
        "gbm_bridge_impute_with_unit_variance",
        "gbm_bridge_impute_with_global_variance",
        "gbm_bridge_impute_with_local_variance",
        "gbm_bridge_impute_with_rolling_variance",
    ] = "log_interpolation"
    backfill_method: Literal[
        "bfill", "shrinkage", "remove_short_stocks", "keep_nan", "insert_zeros", "none"
    ] = "bfill"

    force_download: bool = False
    use_cache: bool = True
    use_adjusted_close: bool = True
    use_autorepair: bool = False

    def __post_init__(self):
        """
        Ensure tickers and indicators are sorted and unique.

        This method is called after initialization to clean up the
        tickers and indicators lists, ensuring they are sorted and
        contain no duplicates.
        """
        self.tickers = sorted(list(set(self.tickers)))
        self.indicators = sorted(list(set(self.indicators)))

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for JSON serialization.

        Returns:
            dict: Dictionary representation of the configuration.

        Example:
            >>> config = DataConfig.get_default_train_config()
            >>> config_dict = config.to_dict()
            >>> print("tickers" in config_dict)  # True
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DataConfig":
        """
        Create configuration instance from dictionary.

        Args:
            config_dict (dict): Dictionary containing configuration parameters.

        Returns:
            DataConfig: New configuration instance.

        Example:
            >>> config_dict = {
            ...     "tickers": ["AAPL", "MSFT"],
            ...     "start": "2020-01-01",
            ...     "end": "2023-01-01"
            ... }
            >>> config = DataConfig.from_dict(config_dict)
        """
        return cls(**config_dict)

    def get_tickers_set(self) -> Set[str]:
        """
        Get tickers as a set for internal use.

        Returns:
            Set[str]: Set of ticker symbols.

        Example:
            >>> config = DataConfig.get_default_train_config()
            >>> ticker_set = config.get_tickers_set()
            >>> print(len(ticker_set))  # Number of unique tickers
        """
        return set(self.tickers)

    def get_indicators_set(self) -> Set[str]:
        """
        Get indicators as a set for internal use.

        Returns:
            Set[str]: Set of indicator names.

        Example:
            >>> config = DataConfig.get_default_train_config()
            >>> indicator_set = config.get_indicators_set()
            >>> print("rsi" in indicator_set)  # True
        """
        return set(self.indicators)

    def copy(self) -> "DataConfig":
        """
        Create a copy of the configuration.
        """
        return DataConfig(**asdict(self))

    @classmethod
    def get_default_train_config(cls, **overrides) -> "DataConfig":
        """
        Get default configuration for training data.

        Returns a configuration with settings suitable for training
        reinforcement learning agents, including appropriate date ranges
        and essential technical indicators.

        Returns:
            DataConfig: Default training configuration.

        Example:
            >>> train_config = DataConfig.get_default_train_config()
            >>> print(train_config.start)  # "2008-06-01"
            >>> print(train_config.end)    # "2020-01-01"
        """
        default_config = cls(
            tickers=sorted(list(DOWJONES)),
            start="2008-06-01",
            end="2020-01-01",
            interval="1d",
            indicators=["rsi", "sma", "macd", "atr"],
            cache_dir=os.path.join(os.path.expanduser("~"), "qf_cache"),
            downloader="yfinance",
            n_trading_days=252,
            imputation_method="log_interpolation",
            backfill_method="bfill",
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(default_config, key):
                setattr(default_config, key, value)

        return default_config

    @classmethod
    def get_default_eval_config(cls, **overrides) -> "DataConfig":
        """
        Get default configuration for evaluation data.

        Returns a configuration with settings suitable for evaluating
        trained agents, including appropriate date ranges and essential
        technical indicators.

        Returns:
            DataConfig: Default evaluation configuration.

        Example:
            >>> eval_config = DataConfig.get_default_eval_config()
            >>> print(eval_config.start)  # "2020-01-01"
            >>> print(eval_config.end)    # "2025-01-01"
        """
        default_config = cls(
            tickers=sorted(list(DOWJONES)),
            start="2020-01-01",
            end="2025-01-01",
            interval="1d",
            indicators=["rsi", "sma", "macd", "atr"],
            cache_dir=os.path.join(os.path.expanduser("~"), "qf_cache"),
            downloader="yfinance",
            n_trading_days=252,
            imputation_method="log_interpolation",
            backfill_method="bfill",
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(default_config, key):
                setattr(default_config, key, value)

        return default_config
