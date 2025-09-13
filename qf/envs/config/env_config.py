"""
Environment Configuration Module.

This module provides the EnvConfig class which defines the configuration for
portfolio trading environments in the QuantForce framework. The configuration
includes data settings, trading parameters, reward functions, observation settings,
and performance optimization options.

The EnvConfig class supports various datatypes for tensor operations and includes
validation to ensure proper configuration parameters.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import torch

import qf
from qf import DEFAULT_LOG_DIR, VERBOSITY
from qf.data.config.data_config import DataConfig
from qf.envs.dataclass.observation_config import ObservationConfig
from qf.envs.reward_functions.config.reward_function_config import (
    BaseRewardConfig,
    get_default_config,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


@dataclass
class EnvConfig:
    """
    Configuration class for portfolio trading environments.

    This class defines all parameters needed to configure a portfolio trading
    environment, including data sources, trading costs, reward functions,
    observation settings, and performance optimizations.

    Attributes:
        data_config (DataConfig): Configuration for data sources and preprocessing.
        obs_window_size (int): Number of historical time steps to include in observations.
        initial_balance (float): Starting portfolio value in currency units.
        trade_cost_percent (float): Proportional trading cost as percentage (e.g., 0.01 = 1%).
        trade_cost_fixed (float): Fixed trading cost per transaction in currency units.
        reward_function_config (BaseRewardConfig): Configuration for the reward function.
        observation_config (ObservationConfig): Configuration for observation space.
        final_reward (float): Reward given at episode termination (default: 0.0).
        verbosity (Optional[int]): Logging verbosity level (default: uses global VERBOSITY).
        log_dir (Optional[str]): Directory for logging outputs (default: DEFAULT_LOG_DIR).
        config_name (Optional[str]): Name identifier for this configuration.
        risk_free_rate (Optional[float]): Risk-free rate for calculations (default: 0.0).
        rebalancing_period (Optional[int]): Steps between portfolio rebalancing
            (1=daily, None=no rebalancing, default: 1).
        datatype (str): Tensor datatype for computations
            ("float32", "float16", "bfloat16", "int8", default: "float16").
        enable_tensor_preallocation (bool): Enable tensor memory preallocation (default: True).
        enable_memory_pooling (bool): Enable memory pooling for better performance (default: True).
        enable_batch_processing (bool): Enable batch processing optimizations (default: True).
        enable_caching (bool): Enable result caching (default: True).
        performance_monitoring (bool): Enable performance monitoring (default: False).
        use_detailed_logging (bool): Enable detailed step-by-step logging (default: False).

    Example:
        >>> from qf.envs.config.env_config import EnvConfig
        >>> from qf.data.config.data_config import DataConfig
        >>> from qf.envs.dataclass.observation_config import ObservationConfig
        >>>
        >>> # Create a basic configuration
        >>> data_config = DataConfig.get_default_train_config()
        >>> obs_config = ObservationConfig(include_actions=True, include_portfolio=True)
        >>>
        >>> config = EnvConfig(
        ...     data_config=data_config,
        ...     obs_window_size=60,
        ...     initial_balance=1_000_000,
        ...     trade_cost_percent=0.01,
        ...     trade_cost_fixed=1.0,
        ...     reward_function_config=get_default_config(),
        ...     observation_config=obs_config
        ... )
        >>>
        >>> # Use optimized configuration for training
        >>> train_config = EnvConfig.get_optimized_train()
    """

    data_config: DataConfig
    obs_window_size: int
    initial_balance: float
    trade_cost_percent: float
    trade_cost_fixed: float
    reward_function_config: BaseRewardConfig
    observation_config: ObservationConfig
    final_reward: float = 0.0
    verbosity: Optional[int] = None
    log_dir: Optional[str] = DEFAULT_LOG_DIR
    config_name: Optional[str] = "DEFAULT_CONFIG"
    risk_free_rate: Optional[float] = 0
    rebalancing_period: Optional[int] = (
        1  # Number of steps between rebalancing (1=daily, None=no rebalancing)
    )
    datatype: str = (
        "float16"  # New datatype configuration: "float32", "float16", "bfloat16", "int8"
    )

    # Performance optimization settings
    enable_tensor_preallocation: bool = True
    enable_memory_pooling: bool = True
    enable_batch_processing: bool = True
    enable_caching: bool = True
    performance_monitoring: bool = False
    use_detailed_logging: bool = (
        False  # Controls the logging of the episode. If True every step of the episode is logged to tensorboard. This is slower but more detailed. If false then only mean max min values are logged.
    )

    def __post_init__(self):
        """
        Validate configuration parameters after initialization.

        Performs comprehensive validation of all configuration parameters to ensure
        they meet the required constraints and types. Sets default values for
        optional parameters if not provided.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        errors = []

        if not isinstance(self.data_config, DataConfig):
            errors.append("data_config must be a DataConfig object.")
        if not isinstance(self.obs_window_size, int) or self.obs_window_size <= 0:
            errors.append("obs_window_size must be a positive integer.")
        if (
            not isinstance(self.initial_balance, (int, float))
            or self.initial_balance <= 0
        ):
            errors.append("initial_balance must be a positive number.")
        if (
            not isinstance(self.trade_cost_percent, (int, float))
            or self.trade_cost_percent < 0
        ):
            errors.append("trade_cost_percent must be >= 0.")
        if (
            not isinstance(self.trade_cost_fixed, (int, float))
            or self.trade_cost_fixed < 0
        ):
            errors.append("trade_cost_fixed must be >= 0.")
        if not isinstance(self.reward_function_config, BaseRewardConfig):
            errors.append("reward_function_config must be a BaseRewardConfig object.")
        if not isinstance(self.observation_config, ObservationConfig):
            errors.append("observation_config must be a ObservationConfig object.")
        if self.rebalancing_period is not None and (
            not isinstance(self.rebalancing_period, int) or self.rebalancing_period <= 0
        ):
            errors.append("rebalancing_period must be None or a positive integer.")
        if self.verbosity is None:
            logger.warning("verbosity not set. Using default 0.")
            self.verbosity = VERBOSITY
        elif not isinstance(self.verbosity, int) or self.verbosity < 0:
            errors.append("verbosity must be a non-negative integer.")
        if self.datatype not in ["float32", "float16", "bfloat16", "int8"]:
            errors.append(
                "datatype must be one of: 'float32', 'float16', 'bfloat16', 'int8'"
            )

        if (
            self.log_dir is None
            or not isinstance(self.log_dir, str)
            or not self.log_dir.strip()
        ):
            logger.warning("log_dir not set or empty. Using Defaults")
            self.log_dir = DEFAULT_LOG_DIR

        if (
            self.config_name is None
            or not isinstance(self.config_name, str)
            or not self.config_name.strip()
        ):
            logger.warning("config_name not set or empty. Using default 'default'.")
            self.config_name = "default"

        if errors:
            for error in errors:
                logger.error(error)
            logger.error("Invalid EnvConfig. See usage signature.")
            raise ValueError("Invalid EnvConfig configuration.")

    def get_torch_dtype(self) -> torch.dtype:
        """
        Get the PyTorch dtype corresponding to the datatype string.

        Returns:
            torch.dtype: The corresponding PyTorch data type.

        Example:
            >>> config = EnvConfig.get_default_train()
            >>> dtype = config.get_torch_dtype()
            >>> print(dtype)  # torch.float32
        """
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
        }
        return dtype_map.get(self.datatype, torch.float32)

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for serialization.

        Returns:
            dict: Dictionary representation of the configuration.

        Example:
            >>> config = EnvConfig.get_default_train()
            >>> config_dict = config.to_dict()
            >>> print(config_dict['initial_balance'])  # 1000000
        """
        return asdict(self)

    def copy(self) -> "EnvConfig":
        """
        Create a copy of the environment configuration.

        This method creates a deep copy by properly handling nested objects
        that are dataclass instances, ensuring they are copied correctly
        rather than converted to dictionaries.
        """
        # Create a copy of the data_config
        data_config_copy = self.data_config.copy()

        # Create a copy of the observation_config
        observation_config_copy = ObservationConfig(
            include_actions=self.observation_config.include_actions,
            include_portfolio=self.observation_config.include_portfolio,
            include_cash=self.observation_config.include_cash,
        )

        # Create a copy of the reward_function_config
        reward_config_copy = type(self.reward_function_config)(
            **asdict(self.reward_function_config)
        )

        # Create the new EnvConfig with copied nested objects
        return EnvConfig(
            data_config=data_config_copy,
            obs_window_size=self.obs_window_size,
            initial_balance=self.initial_balance,
            trade_cost_percent=self.trade_cost_percent,
            trade_cost_fixed=self.trade_cost_fixed,
            reward_function_config=reward_config_copy,
            observation_config=observation_config_copy,
            final_reward=self.final_reward,
            verbosity=self.verbosity,
            log_dir=self.log_dir,
            config_name=self.config_name,
            risk_free_rate=self.risk_free_rate,
            rebalancing_period=self.rebalancing_period,
            datatype=self.datatype,
            enable_tensor_preallocation=self.enable_tensor_preallocation,
            enable_memory_pooling=self.enable_memory_pooling,
            enable_batch_processing=self.enable_batch_processing,
            enable_caching=self.enable_caching,
            performance_monitoring=self.performance_monitoring,
            use_detailed_logging=self.use_detailed_logging,
        )

    @classmethod
    def get_default_train(cls, **overrides) -> "EnvConfig":
        """
        Get default configuration optimized for training.

        Returns a configuration with settings suitable for training reinforcement
        learning agents, including appropriate data ranges, daily rebalancing,
        and float32 precision for numerical stability.

        Args:
            **overrides: Keyword arguments to override default values.
                Common overrides include:
                - initial_balance: Starting portfolio value
                - trade_cost_percent: Proportional trading cost
                - obs_window_size: Observation window size
                - datatype: Tensor datatype ("float32", "float16", "bfloat16", "int8")

        Returns:
            EnvConfig: Default training configuration with optional overrides.

        Example:
            >>> # Use default training config
            >>> train_config = EnvConfig.get_default_train()
            >>> print(train_config.config_name)  # "DEFAULT_TRAIN_CONFIG"

            >>> # Override specific values
            >>> custom_config = EnvConfig.get_default_train(
            ...     initial_balance=500_000,
            ...     trade_cost_percent=0.005,
            ...     obs_window_size=30
            ... )
            >>> print(custom_config.initial_balance)  # 500000
        """
        default_config = cls(
            data_config=DataConfig.get_default_train_config(),
            obs_window_size=60,
            initial_balance=1_000_000,
            trade_cost_percent=0.01,
            trade_cost_fixed=1,
            reward_function_config=get_default_config(),
            observation_config=ObservationConfig(
                include_actions=True, include_portfolio=True, include_cash=True
            ),
            verbosity=VERBOSITY,
            log_dir=DEFAULT_LOG_DIR,
            config_name="DEFAULT_TRAIN_CONFIG",
            rebalancing_period=1,  # Daily rebalancing for training
            datatype="float32",  # Default to float32 for training
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(default_config, key):
                setattr(default_config, key, value)
            else:
                logger.warning(f"Unknown override key: {key}")

        return default_config

    @classmethod
    def get_default_eval(cls, **overrides) -> "EnvConfig":
        """
        Get default configuration optimized for evaluation.

        Returns a configuration with settings suitable for evaluating trained
        agents, including appropriate data ranges and daily rebalancing.

        Args:
            **overrides: Keyword arguments to override default values.
                Common overrides include:
                - initial_balance: Starting portfolio value
                - trade_cost_percent: Proportional trading cost
                - obs_window_size: Observation window size
                - datatype: Tensor datatype ("float32", "float16", "bfloat16", "int8")

        Returns:
            EnvConfig: Default evaluation configuration with optional overrides.

        Example:
            >>> # Use default evaluation config
            >>> eval_config = EnvConfig.get_default_eval()
            >>> print(eval_config.config_name)  # "DEFAULT_EVAL_CONFIG"

            >>> # Override specific values
            >>> custom_config = EnvConfig.get_default_eval(
            ...     initial_balance=2_000_000,
            ...     trade_cost_percent=0.02
            ... )
            >>> print(custom_config.initial_balance)  # 2000000
        """
        default_config = cls(
            data_config=DataConfig.get_default_eval_config(),
            obs_window_size=60,
            initial_balance=1_000_000,
            trade_cost_percent=0.01,
            trade_cost_fixed=1,
            reward_function_config=get_default_config(),
            observation_config=ObservationConfig(
                include_actions=True, include_portfolio=True, include_cash=True
            ),
            verbosity=VERBOSITY,
            log_dir=DEFAULT_LOG_DIR,
            config_name="DEFAULT_EVAL_CONFIG",
            rebalancing_period=1,  # Daily rebalancing for evaluation
            datatype="float32",  # Default to float32 for evaluation
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(default_config, key):
                setattr(default_config, key, value)
            else:
                logger.warning(f"Unknown override key: {key}")

        return default_config

    @classmethod
    def get_optimized_train(cls, **overrides) -> "EnvConfig":
        """
        Get optimized training configuration with float16 for better performance.

        Returns a configuration optimized for faster training with float16
        precision and all performance optimizations enabled.

        Args:
            **overrides: Keyword arguments to override default values.
                Common overrides include:
                - initial_balance: Starting portfolio value
                - trade_cost_percent: Proportional trading cost
                - obs_window_size: Observation window size
                - datatype: Tensor datatype (defaults to "float16" for optimization)

        Returns:
            EnvConfig: Optimized training configuration with optional overrides.

        Example:
            >>> # Use optimized training config
            >>> fast_config = EnvConfig.get_optimized_train()
            >>> print(fast_config.datatype)  # "float16"
            >>> print(fast_config.enable_tensor_preallocation)  # True

            >>> # Override specific values
            >>> custom_config = EnvConfig.get_optimized_train(
            ...     initial_balance=500_000,
            ...     datatype="float32"  # Override the default float16
            ... )
            >>> print(custom_config.datatype)  # "float32"
        """
        config = cls.get_default_train(**overrides)
        config.datatype = overrides.get(
            "datatype", "float16"
        )  # Allow override of datatype
        config.enable_tensor_preallocation = True
        config.enable_memory_pooling = True
        config.enable_batch_processing = True
        config.enable_caching = True
        config.performance_monitoring = True
        return config

    @classmethod
    def get_ultra_fast_train(cls, **overrides) -> "EnvConfig":
        """
        Get ultra-fast training configuration with int8 for maximum performance.

        Returns a configuration optimized for maximum speed with int8 precision.
        This configuration trades numerical precision for speed and should be
        used carefully as it may affect training stability.

        Args:
            **overrides: Keyword arguments to override default values.
                Common overrides include:
                - initial_balance: Starting portfolio value
                - trade_cost_percent: Proportional trading cost
                - obs_window_size: Observation window size
                - datatype: Tensor datatype (defaults to "int8" for ultra-fast)

        Returns:
            EnvConfig: Ultra-fast training configuration with optional overrides.

        Example:
            >>> # Use ultra-fast training config
            >>> ultra_config = EnvConfig.get_ultra_fast_train()
            >>> print(ultra_config.datatype)  # "int8"
            >>> print(ultra_config.performance_monitoring)  # True

            >>> # Override specific values
            >>> custom_config = EnvConfig.get_ultra_fast_train(
            ...     initial_balance=1_500_000,
            ...     datatype="float16"  # Override the default int8
            ... )
            >>> print(custom_config.datatype)  # "float16"
        """
        config = cls.get_default_train(**overrides)
        config.datatype = overrides.get(
            "datatype", "int8"
        )  # Allow override of datatype
        config.enable_tensor_preallocation = True
        config.enable_memory_pooling = True
        config.enable_batch_processing = True
        config.enable_caching = True
        config.performance_monitoring = True
        return config

    @classmethod
    def from_defaults(cls, preset: str = "train", **overrides) -> "EnvConfig":
        """
        Create configuration from a preset with optional overrides.

        This is a convenience method that allows you to easily create configurations
        from predefined presets while overriding specific values.

        Args:
            preset (str): The preset to use ("train", "eval", "optimized", "ultra_fast").
            **overrides: Keyword arguments to override default values.

        Returns:
            EnvConfig: Configuration based on the preset with optional overrides.

        Example:
            >>> # Create training config with custom balance
            >>> config = EnvConfig.from_defaults("train", initial_balance=500_000)
            >>> print(config.initial_balance)  # 500000
            >>> print(config.config_name)  # "DEFAULT_TRAIN_CONFIG"

            >>> # Create optimized config with custom datatype
            >>> config = EnvConfig.from_defaults("optimized", datatype="float32")
            >>> print(config.datatype)  # "float32"
            >>> print(config.enable_tensor_preallocation)  # True

            >>> # Create evaluation config with multiple overrides
            >>> config = EnvConfig.from_defaults(
            ...     "eval",
            ...     initial_balance=2_000_000,
            ...     trade_cost_percent=0.02,
            ...     obs_window_size=30
            ... )
        """
        preset_methods = {
            "train": cls.get_default_train,
            "eval": cls.get_default_eval,
            "optimized": cls.get_optimized_train,
            "ultra_fast": cls.get_ultra_fast_train,
        }

        if preset not in preset_methods:
            raise ValueError(
                f"Unknown preset '{preset}'. Available presets: {list(preset_methods.keys())}"
            )

        return preset_methods[preset](**overrides)
