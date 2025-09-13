"""
Critic Agent Configuration Module.

This module provides the CriticAgentConfig class which defines the configuration
for value-based (critic-only) reinforcement learning agents like DQN and SPQL.
These agents learn value functions to estimate the expected return of actions.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

from qf.agents.config.rl_agent_config.rl_agent_config import RLAgentConfig
from qf.networks.default_networks import DefaultNetworks
from qf.networks.network_config import NetworkConfig


@dataclass
class CriticAgentConfig(RLAgentConfig):
    """
    Configuration for value-based (critic-only) RL agents like DQN, SPQL.

    This class extends RLAgentConfig to include parameters specific to
    value-based reinforcement learning algorithms. These agents learn
    value functions (critics) to estimate the expected return of actions
    or state-action pairs.

    Value-based methods are particularly effective for discrete action
    spaces and can handle complex state representations through deep
    neural networks.

    Attributes:
        epsilon_start (float): Initial exploration rate for epsilon-greedy
            exploration. Higher values encourage more exploration initially.
            Default: 0.4.
        tau (Optional[float]): Soft update parameter for target networks.
            Controls how much the target network is updated towards the
            main network. Range: [0, 1]. Default: 0.01.
        critic_config (Optional[Union[NetworkConfig, dict]]): Configuration
            for the critic network architecture. If None, uses default
            standard value critic network.
        feature_extractor_config (Optional[Union[NetworkConfig, dict]]): Configuration
            for the feature extractor network. If None, uses default
            feature extractor network.

    Example:
        >>> from qf.agents.config.rl_agent_config.critic_agent_config.critic_agent_config import CriticAgentConfig
        >>> from qf.networks.network_config import NetworkConfig
        >>>
        >>> # Create a basic critic configuration
        >>> config = CriticAgentConfig(
        ...     type="dqn",
        ...     epsilon_start=0.5,
        ...     tau=0.01
        ... )
        >>>
        >>> # Create with custom network configurations
        >>> custom_config = CriticAgentConfig(
        ...     type="spql",
        ...     epsilon_start=0.3,
        ...     tau=0.005,
        ...     critic_config=NetworkConfig(layers=[...]),
        ...     feature_extractor_config=NetworkConfig(layers=[...])
        ... )
        >>>
        >>> # Create a configuration for high exploration
        >>> high_exploration_config = CriticAgentConfig(
        ...     type="dqn",
        ...     epsilon_start=0.8,  # High initial exploration
        ...     tau=0.02            # Faster target updates
        ... )
    """

    epsilon_start: float = 0.4
    tau: Optional[float] = 0.01  # For soft target updates

    critic_config: Optional[Union[NetworkConfig, dict]] = field(
        default_factory=lambda: DefaultNetworks.get_standard_value_critic()
    )
    feature_extractor_config: Optional[Union[NetworkConfig, dict]] = field(
        default_factory=lambda: DefaultNetworks.get_default_feature_extractor()
    )

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for serialization.

        Handles the conversion of NetworkConfig objects to dictionaries
        for proper JSON serialization.

        Returns:
            dict: Dictionary representation of the configuration.

        Example:
            >>> config = CriticAgentConfig(type="dqn")
            >>> config_dict = config.to_dict()
            >>> print("critic_config" in config_dict)  # True
        """
        base = super().to_dict()
        if isinstance(self.critic_config, NetworkConfig):
            base["critic_config"] = self.critic_config.to_dict()
        if isinstance(self.feature_extractor_config, NetworkConfig):
            base["feature_extractor_config"] = self.feature_extractor_config.to_dict()
        return base

    @classmethod
    def from_dict(cls, config_dict: dict) -> "CriticAgentConfig":
        """
        Create configuration instance from dictionary.

        Handles the conversion of network configuration dictionaries back
        to NetworkConfig objects for proper deserialization.

        Args:
            config_dict (dict): Dictionary containing configuration parameters.

        Returns:
            CriticAgentConfig: New configuration instance.

        Example:
            >>> config_dict = {
            ...     "type": "dqn",
            ...     "epsilon_start": 0.5,
            ...     "critic_config": {"layers": [...]}
            ... }
            >>> config = CriticAgentConfig.from_dict(config_dict)
            >>> print(config.type)  # "dqn"
        """
        # Filter out fields that are not part of __init__
        init_fields = {
            field.name for field in cls.__dataclass_fields__.values() if field.init
        }
        filtered_dict = {k: v for k, v in config_dict.items() if k in init_fields}

        critic = filtered_dict.get("critic_config")
        features = filtered_dict.get("feature_extractor_config")

        # Convert network config back to objects
        if isinstance(critic, dict):
            filtered_dict["critic_config"] = NetworkConfig.from_dict(critic)
        if isinstance(features, dict):
            filtered_dict["feature_extractor_config"] = NetworkConfig.from_dict(
                features
            )

        return cls(**filtered_dict)
