"""
Random Agent Configuration Module.

This module provides the RandomAgentConfig class which defines the
configuration for random portfolio agents. These agents implement a
simple random allocation strategy, often used as a baseline or for
exploration purposes in portfolio optimization research.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import optuna

from qf.agents.config.classic_agents.classic_agent_config import ClassicAgentConfig


@dataclass
class RandomAgentConfig(ClassicAgentConfig):
    """
    Configuration for Random Portfolio Agent.

    This class defines the configuration for random portfolio agents
    that implement a simple random allocation strategy. Random agents
    select portfolio weights randomly from the action space, making
    them useful for baseline comparisons and exploration studies.

    Random agents are particularly valuable in portfolio optimization
    research as they provide a lower bound for performance evaluation
    and can help identify whether more sophisticated strategies are
    actually adding value beyond random allocation.

    Attributes:
        type (str): Agent type identifier, set to "random_portfolio".
        rebalancing_period (Optional[int]): Number of steps between
            portfolio rebalancing. np.inf = buy-and-hold strategy,
            1 = daily rebalancing, 7 = weekly rebalancing, etc.
            Default is np.inf (no rebalancing).

    Example:
        >>> from qf.agents.config.classic_agents.random_agent_config import RandomAgentConfig
        >>>
        >>> # Create a basic random portfolio configuration
        >>> config = RandomAgentConfig()
        >>>

        >>> # Create with daily rebalancing
        >>> daily_config = RandomAgentConfig(
        ...     rebalancing_period=1  # Daily rebalancing
        ... )
        >>>
        >>> # Create buy-and-hold random portfolio
        >>> buy_hold_config = RandomAgentConfig(
        ...     rebalancing_period=np.inf  # No rebalancing
        ... )
    """

    type: str = "random_portfolio"

    @staticmethod
    def get_default_config() -> "RandomAgentConfig":
        """
        Get default configuration for Random Portfolio Agent.

        Returns a configuration with standard settings for random
        portfolio allocation.

        Returns:
            RandomAgentConfig: Default configuration.

        Example:
            >>> config = RandomAgentConfig.get_default_config()
            >>> print(config.type)  # "random_portfolio"
        """
        return RandomAgentConfig()

    @staticmethod
    def get_hyperparameter_space(trial: optuna.Trial) -> "RandomAgentConfig":
        """
        Get hyperparameter space for Optuna optimization.

        For the random strategy, the main hyperparameter that could be
        optimized is the rebalancing period.

        Args:
            trial (optuna.Trial): Optuna trial object for parameter suggestions.

        Returns:
            RandomAgentConfig: Configuration with suggested parameters.

        Example:
            >>> import optuna
            >>>
            >>> def objective(trial):
            ...     config = RandomAgentConfig.get_hyperparameter_space(trial)
            ...     # Use config for evaluation
            ...     return performance_score
            >>>
            >>> study = optuna.create_study(direction="maximize")
            >>> study.optimize(objective, n_trials=10)
        """
        return RandomAgentConfig()
