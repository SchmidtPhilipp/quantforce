"""
Reinforcement Learning Agent Configuration Module.

This module provides the RLAgentConfig class which serves as the base
configuration for all reinforcement learning agents in the QuantForce framework.
It defines common parameters and methods that are shared across different RL
agent types, including learning rates, discount factors, and buffer settings.
"""

from dataclasses import dataclass, field

from qf.agents.config.base_agent_config import BaseAgentConfig


@dataclass
class RLAgentConfig(BaseAgentConfig):
    """
    Base configuration for all reinforcement learning agents.

    This class extends BaseAgentConfig to include parameters specific to
    reinforcement learning algorithms. It provides common settings for
    learning rates, discount factors, experience replay buffers, and
    batch processing that are shared across all RL agent types.

    The RLAgentConfig serves as the foundation for more specific RL agent
    configurations like actor-critic methods, value-based methods, and
    policy gradient methods.

    Attributes:
        learning_rate (float): Learning rate for gradient-based updates.
            Controls the step size in parameter updates. Default: 3e-4.
        gamma (float): Discount factor for future rewards.
            Determines how much future rewards are valued relative to
            immediate rewards. Range: [0, 1]. Default: 0.9.
        buffer_size (int): Size of the experience replay buffer.
            Larger buffers provide more stable training but use more memory.
            Default: 50_000.
        batch_size (int): Number of samples per training batch.
            Larger batches provide more stable gradients but use more memory.
            Default: 64.
        _n_agents (int): Number of agents (internal field, not settable via __init__).
            Used for multi-agent scenarios. Default: 1.

    Example:
        >>> from qf.agents.config.rl_agent_config.rl_agent_config import RLAgentConfig
        >>>
        >>> # Create a basic RL configuration
        >>> config = RLAgentConfig(
        ...     type="test_rl_agent",
        ...     learning_rate=0.001,
        ...     gamma=0.95,
        ...     buffer_size=10000,
        ...     batch_size=32
        ... )
        >>>
        >>> # Create a configuration for fast learning
        >>> fast_config = RLAgentConfig(
        ...     type="fast_rl_agent",
        ...     learning_rate=0.01,  # Higher learning rate
        ...     gamma=0.8,           # Lower discount factor
        ...     buffer_size=5000     # Smaller buffer
        ... )
        >>>
        >>> # Create a configuration for stable learning
        >>> stable_config = RLAgentConfig(
        ...     type="stable_rl_agent",
        ...     learning_rate=0.0001,  # Lower learning rate
        ...     gamma=0.99,            # Higher discount factor
        ...     buffer_size=100000     # Larger buffer
        ... )
    """

    learning_rate: float = 3e-4
    gamma: float = 0.9
    buffer_size: int = 50_000
    batch_size: int = 64
    # n_agents is not settable via __init__ by default
    _n_agents: int = field(default=1, init=False, repr=False)

    @property
    def n_agents(self) -> int:
        """
        Get the number of agents.

        Returns:
            int: Number of agents in the configuration.

        Example:
            >>> config = RLAgentConfig(type="test_agent")
            >>> print(config.n_agents)  # 1
        """
        return self._n_agents

    def _set_n_agents(self, value: int) -> None:
        """
        Set the number of agents (internal method).

        This method is used by subconfig that want to allow setting
        the number of agents. It's not part of the public interface
        but can be overridden by subclasses.

        Args:
            value (int): Number of agents to set.

        Example:
            >>> config = RLAgentConfig(type="test_agent")
            >>> config._set_n_agents(3)
            >>> print(config.n_agents)  # 3
        """
        self._n_agents = value
