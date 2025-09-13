"""
Reward Function Configuration Module.

This module provides various reward function configuration classes for
defining reward functions in the QuantForce framework. It supports
different types of reward functions including linear returns, log returns,
absolute returns, and various Sharpe ratio-based rewards.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseRewardConfig(ABC):
    """
    Base configuration for all reward functions.

    This class provides the foundation for all reward function configurations,
    defining common parameters like reward scaling, bad reward values,
    and log return preferences that are shared across different reward types.

    Attributes:
        type (str): Reward function type identifier.
        reward_scaling (float): Scaling factor for the reward.
            Used to adjust the magnitude of rewards. Default: 1.
        bad_reward (float): Reward value for bad/terminal states.
            Used when the agent reaches an undesirable state. Default: 0.
        use_log_returns (Optional[bool]): Whether to use log returns.
            Some reward functions can work with both linear and log returns.
            Default: False.

    Example:
        >>> from qf.envs.reward_functions.config.reward_function_config import LinearRateOfReturnConfig
        >>>
        >>> # Create a basic reward configuration
        >>> config = LinearRateOfReturnConfig(
        ...     reward_scaling=1.0,
        ...     bad_reward=0.0
        ... )
    """

    type: str
    reward_scaling: float = 1
    bad_reward: float = 0
    use_log_returns: Optional[bool] = False


@dataclass
class LinearRateOfReturnConfig(BaseRewardConfig):
    """
    Configuration for Linear Rate of Return reward function.

    This reward function calculates the linear rate of return for the
    portfolio, which is the simple percentage change in portfolio value.

    Attributes:
        type (str): Reward function type identifier, set to "linear_rate_of_return".

    Example:
        >>> from qf.envs.reward_functions.config.reward_function_config import LinearRateOfReturnConfig
        >>>
        >>> # Create a linear rate of return configuration
        >>> config = LinearRateOfReturnConfig(
        ...     reward_scaling=100,  # Scale rewards by 100
        ...     bad_reward=-1.0     # Negative reward for bad states
        ... )
        >>>
        >>> # Create a configuration with log returns
        >>> log_config = LinearRateOfReturnConfig(
        ...     use_log_returns=True,
        ...     reward_scaling=1.0
        ... )
    """

    type: str = "linear_rate_of_return"


@dataclass
class LogReturnConfig(BaseRewardConfig):
    """
    Configuration for Log Return reward function.

    This reward function calculates the logarithmic rate of return for the
    portfolio, which is the natural logarithm of the portfolio value ratio.
    Log returns are additive over time and are commonly used in finance.

    Attributes:
        type (str): Reward function type identifier, set to "log_return".

    Example:
        >>> from qf.envs.reward_functions.config.reward_function_config import LogReturnConfig
        >>>
        >>> # Create a log return configuration
        >>> config = LogReturnConfig(
        ...     reward_scaling=1.0,
        ...     bad_reward=-10.0  # Large negative reward for bad states
        ... )
    """

    type: str = "log_return"


@dataclass
class AbsoluteReturnConfig(BaseRewardConfig):
    """
    Configuration for Absolute Return reward function.

    This reward function calculates the absolute return (profit/loss) for the
    portfolio in currency units, rather than as a percentage.

    Attributes:
        type (str): Reward function type identifier, set to "absolute_return".

    Example:
        >>> from qf.envs.reward_functions.config.reward_function_config import AbsoluteReturnConfig
        >>>
        >>> # Create an absolute return configuration
        >>> config = AbsoluteReturnConfig(
        ...     reward_scaling=0.001,  # Scale to thousands of dollars
        ...     bad_reward=-1000.0     # Large negative reward for losses
        ... )
    """

    type: str = "absolute_return"


@dataclass
class WindowedRewardConfig(BaseRewardConfig):
    """
    Configuration for Windowed Reward function.

    This reward function calculates rewards over a specified time window,
    allowing for more sophisticated reward calculations that consider
    historical performance.

    Attributes:
        type (str): Reward function type identifier, set to "windowed_reward".
        past_window (int): Number of past time steps to consider.
            Used for calculating historical context. Default: 0.
        future_window (int): Number of future time steps to consider.
            Used for calculating forward-looking rewards. Default: 60.

    Example:
        >>> from qf.envs.reward_functions.config.reward_function_config import WindowedRewardConfig
        >>>
        >>> # Create a windowed reward configuration
        >>> config = WindowedRewardConfig(
        ...     past_window=30,    # Consider 30 past days
        ...     future_window=60,  # Look ahead 60 days
        ...     reward_scaling=1.0
        ... )
    """

    type: str = "windowed_reward"
    past_window: int = 0
    future_window: int = 60


@dataclass
class DifferentialSharpeRatioConfig(WindowedRewardConfig):
    """
    Configuration for Differential Sharpe Ratio reward function.

    This reward function calculates the differential Sharpe ratio, which
    measures the change in risk-adjusted returns over time. It provides
    a more sophisticated reward signal that considers both return and risk.

    Attributes:
        type (str): Reward function type identifier, set to "differential_sharpe_ratio".
        eta (float): Learning rate parameter for the differential Sharpe ratio.
            Controls how quickly the Sharpe ratio estimate adapts.
            Default: 0.01.

    Example:
        >>> from qf.envs.reward_functions.config.reward_function_config import DifferentialSharpeRatioConfig
        >>>
        >>> # Create a differential Sharpe ratio configuration
        >>> config = DifferentialSharpeRatioConfig(
        ...     eta=0.01,           # Learning rate
        ...     past_window=30,     # Consider 30 past days
        ...     future_window=60,   # Look ahead 60 days
        ...     reward_scaling=1.0
        ... )
    """

    type: str = "differential_sharpe_ratio"
    eta: float = 0.01


@dataclass
class CostAdjustedSharpeRatioConfig(WindowedRewardConfig):
    """
    Configuration for Cost Adjusted Sharpe Ratio reward function.

    This reward function calculates the Sharpe ratio while accounting for
    transaction costs, providing a more realistic measure of risk-adjusted
    returns that considers the impact of trading.

    Attributes:
        type (str): Reward function type identifier, set to "cost_adjusted_sharpe_ratio".
        use_log_returns (bool): Whether to use log returns for calculations.
            Default: False.

    Example:
        >>> from qf.envs.reward_functions.config.reward_function_config import CostAdjustedSharpeRatioConfig
        >>>
        >>> # Create a cost-adjusted Sharpe ratio configuration
        >>> config = CostAdjustedSharpeRatioConfig(
        ...     past_window=30,     # Consider 30 past days
        ...     future_window=60,   # Look ahead 60 days
        ...     use_log_returns=True,
        ...     reward_scaling=1.0
        ... )
    """

    type: str = "cost_adjusted_sharpe_ratio"
    use_log_returns: bool = False


def get_default_config() -> "CostAdjustedSharpeRatioConfig":
    """
    Get default reward function configuration.

    Returns a configuration with sensible defaults for the cost-adjusted
    Sharpe ratio reward function, which is often a good choice for
    portfolio optimization tasks.

    Returns:
        CostAdjustedSharpeRatioConfig: Default reward function configuration.

    Example:
        >>> from qf.envs.reward_functions.config.reward_function_config import get_default_config
        >>>
        >>> # Get default configuration
        >>> config = get_default_config()
        >>> print(config.type)  # "cost_adjusted_sharpe_ratio"
        >>> print(config.past_window)  # 0
        >>> print(config.future_window)  # 60
    """
    return CostAdjustedSharpeRatioConfig(
        type="cost_adjusted_sharpe_ratio",
        past_window=0,
        future_window=60,
        use_log_returns=False,
    )
