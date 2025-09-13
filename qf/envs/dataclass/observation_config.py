"""
Observation Configuration Module.

This module provides the ObservationConfig class which defines the configuration
for observation spaces in the QuantForce framework. It controls what information
is included in the agent's observation vector.
"""

from dataclasses import dataclass


@dataclass
class ObservationConfig:
    """
    Configuration for observation space in portfolio trading environments.

    This class defines what information is included in the agent's observation
    vector. The observation space can include various components like market
    data, portfolio state, action history, and cash positions.

    The observation configuration is crucial for the agent's ability to make
    informed decisions, as it determines what information the agent has access
    to about the current state of the market and portfolio.

    Attributes:
        include_actions (bool): Whether to include previous actions in the observation.
            This allows the agent to see its own trading history.
            Default: True.
        include_portfolio (bool): Whether to include current portfolio weights in the observation.
            This provides the agent with information about its current positions.
            Default: True.
        include_cash (bool): Whether to include cash position in the observation.
            This allows the agent to see how much cash it has available.
            Default: True.

    Example:
        >>> from qf.envs.dataclass.observation_config import ObservationConfig
        >>>
        >>> # Create a basic observation configuration
        >>> config = ObservationConfig(
        ...     include_actions=True,
        ...     include_portfolio=True,
        ...     include_cash=True
        ... )
        >>>
        >>> # Create a minimal observation configuration
        >>> minimal_config = ObservationConfig(
        ...     include_actions=False,    # No action history
        ...     include_portfolio=True,   # Include portfolio weights
        ...     include_cash=False       # No cash information
        ... )
        >>>
        >>> # Create a comprehensive observation configuration
        >>> comprehensive_config = ObservationConfig(
        ...     include_actions=True,     # Full action history
        ...     include_portfolio=True,   # Full portfolio state
        ...     include_cash=True        # Cash position
        ... )
    """

    include_actions: bool = True
    include_portfolio: bool = True
    include_cash: bool = True
