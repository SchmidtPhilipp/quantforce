"""
Classic Agent Configuration Module.

This module provides the ClassicAgentConfig class which serves as the base
configuration for classic portfolio optimization agents. These agents implement
traditional financial strategies like Markowitz portfolio optimization and
equal-weight portfolios.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from qf.agents.config.base_agent_config import BaseAgentConfig


@dataclass
class ClassicAgentConfig(BaseAgentConfig):
    """
    Base configuration for classic portfolio optimization agents.

    This class extends BaseAgentConfig to include parameters specific to
    classic portfolio optimization strategies. Classic agents implement
    traditional financial approaches like Markowitz portfolio optimization,
    equal-weight portfolios, and other rule-based strategies.

    The rebalancing_period parameter is particularly important for classic
    agents as it determines how frequently the portfolio is rebalanced,
    which can significantly impact performance and transaction costs.

    Attributes:
        rebalancing_period (Optional[int]): Number of steps between portfolio
            rebalancing. np.inf = buy-and-hold strategy, 1 = daily rebalancing,
            7 = weekly rebalancing, etc. Default is np.inf (no rebalancing).

    Example:
        >>> from qf.agents.config.classic_agents.classic_agent_config import ClassicAgentConfig
        >>>
        >>> # Create a daily rebalancing configuration
        >>> config = ClassicAgentConfig(
        ...     type="classic_agent",
        ...     rebalancing_period=1  # Daily rebalancing
        ... )
        >>>
        >>> # Create a buy-and-hold configuration
        >>> buy_hold_config = ClassicAgentConfig(
        ...     type="classic_agent",
        ...     rebalancing_period=np.inf  # No rebalancing
        ... )
        >>>
        >>> # Create a weekly rebalancing configuration
        >>> weekly_config = ClassicAgentConfig(
        ...     type="classic_agent",
        ...     rebalancing_period=7  # Weekly rebalancing
        ... )
    """

    rebalancing_period: Optional[int] = np.inf
