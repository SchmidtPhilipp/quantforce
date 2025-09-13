"""
One-Over-N Portfolio Agent Configuration Module.

This module provides the OneOverNPortfolioAgentConfig class which defines the
configuration for equal-weight portfolio agents. These agents implement the
simple but effective strategy of equally weighting all assets in the portfolio.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import optuna

from qf.agents.config.classic_agents.classic_agent_config import ClassicAgentConfig


@dataclass
class OneOverNPortfolioAgentConfig(ClassicAgentConfig):
    """
    Configuration for One-Over-N Portfolio Agent.

    This class defines the configuration for equal-weight portfolio agents
    that implement the 1/N strategy. This is a simple but often effective
    portfolio strategy that allocates equal weights to all available assets.

    The 1/N strategy has been shown to perform competitively with more
    sophisticated approaches in many scenarios, particularly when transaction
    costs are considered. It serves as a useful benchmark for more complex
    portfolio optimization strategies.

    Attributes:
        type (str): Agent type identifier, set to "one_over_n_portfolio".

    Example:
        >>> from qf.agents.config.classic_agents.one_over_n_portfolio_agent_config import OneOverNPortfolioAgentConfig
        >>>
        >>> # Create a basic 1/N portfolio configuration
        >>> config = OneOverNPortfolioAgentConfig()
        >>>
        >>> # Create with custom rebalancing period
        >>> weekly_config = OneOverNPortfolioAgentConfig(
        ...     rebalancing_period=7  # Weekly rebalancing
        ... )
        >>>
        >>> # Create buy-and-hold 1/N portfolio
        >>> buy_hold_config = OneOverNPortfolioAgentConfig(
        ...     rebalancing_period=np.inf  # No rebalancing
        ... )
    """

    type: str = "one_over_n_portfolio"

    @staticmethod
    def get_default_config() -> "OneOverNPortfolioAgentConfig":
        """
        Get default configuration for One-Over-N Portfolio Agent.

        Returns a configuration with standard settings for equal-weight
        portfolio allocation.

        Returns:
            OneOverNPortfolioAgentConfig: Default configuration.

        Example:
            >>> config = OneOverNPortfolioAgentConfig.get_default_config()
            >>> print(config.type)  # "one_over_n_portfolio"
        """
        return OneOverNPortfolioAgentConfig()

    @staticmethod
    def get_hyperparameter_space(trial: optuna.Trial) -> "OneOverNPortfolioAgentConfig":
        """
        Get hyperparameter space for Optuna optimization.

        For the 1/N strategy, there are typically no hyperparameters to optimize
        since the strategy is deterministic. However, this method is provided for
        consistency with other agent configurations and potential future extensions.

        Args:
            trial (optuna.Trial): Optuna trial object for parameter suggestions.

        Returns:
            OneOverNPortfolioAgentConfig: Configuration with suggested parameters.

        Example:
            >>> import optuna
            >>>
            >>> def objective(trial):
            ...     config = OneOverNPortfolioAgentConfig.get_hyperparameter_space(trial)
            ...     # Use config for evaluation
            ...     return performance_score
            >>>
            >>> study = optuna.create_study(direction="maximize")
            >>> study.optimize(objective, n_trials=10)
        """
        return OneOverNPortfolioAgentConfig()
