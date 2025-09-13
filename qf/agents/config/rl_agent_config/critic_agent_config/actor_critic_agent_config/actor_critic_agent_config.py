"""
Actor-Critic Agent Configuration Module.

This module provides the ActorCriticAgentConfig class which defines the configuration
for actor-critic reinforcement learning agents like A2C, PPO, SAC, DDPG, and TD3.
These agents learn both policy (actor) and value (critic) functions.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

from qf.agents.config.rl_agent_config.critic_agent_config.critic_agent_config import (
    CriticAgentConfig,
)
from qf.networks.default_networks import DefaultNetworks
from qf.networks.network_config import NetworkConfig


@dataclass
class ActorCriticAgentConfig(CriticAgentConfig):
    """
    Configuration for actor-critic RL agents like A2C, PPO, SAC, DDPG, TD3.

    This class extends CriticAgentConfig to include parameters specific to
    actor-critic reinforcement learning algorithms. These agents learn both
    a policy function (actor) that maps states to actions and a value function
    (critic) that estimates the expected return.

    Actor-critic methods are particularly effective for continuous action
    spaces and can handle both discrete and continuous state spaces through
    deep neural networks.

    Attributes:
        ent_coef (Optional[float]): Entropy coefficient for exploration.
            Higher values encourage more exploration by increasing the
            entropy of the policy. If None, uses algorithm-specific defaults.
            Default: None.
        vf_coef (Optional[float]): Value function coefficient for loss weighting.
            Controls the relative weight of the value function loss in the
            total loss. If None, uses algorithm-specific defaults.
            Default: None.
        actor_config (Optional[Union[NetworkConfig, dict]]): Configuration
            for the actor network architecture. If None, uses default
            standard actor network.

    Example:
        >>> from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.actor_critic_agent_config import ActorCriticAgentConfig
        >>> from qf.networks.network_config import NetworkConfig
        >>>
        >>> # Create a basic actor-critic configuration
        >>> config = ActorCriticAgentConfig(
        ...     type="ppo",
        ...     ent_coef=0.01,
        ...     vf_coef=0.5
        ... )
        >>>
        >>> # Create with custom network configurations
        >>> custom_config = ActorCriticAgentConfig(
        ...     type="sac",
        ...     ent_coef=0.1,
        ...     vf_coef=0.25,
        ...     actor_config=NetworkConfig(layers=[...]),
        ...     critic_config=NetworkConfig(layers=[...])
        ... )
        >>>
        >>> # Create a configuration for high exploration
        >>> high_exploration_config = ActorCriticAgentConfig(
        ...     type="a2c",
        ...     ent_coef=0.2,  # High entropy coefficient
        ...     vf_coef=0.5
        ... )
        >>>
        >>> # Create a configuration for stable learning
        >>> stable_config = ActorCriticAgentConfig(
        ...     type="ppo",
        ...     ent_coef=0.001,  # Low entropy coefficient
        ...     vf_coef=1.0      # High value function weight
        ... )
    """

    ent_coef: Optional[float] = None  # Entropy coefficient
    vf_coef: Optional[float] = None  # Value function coefficient

    actor_config: Optional[Union[NetworkConfig, dict]] = field(
        default_factory=lambda: DefaultNetworks.get_standard_actor()
    )

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for serialization.

        Handles the conversion of NetworkConfig objects to dictionaries
        for proper JSON serialization, including the actor network configuration.

        Returns:
            dict: Dictionary representation of the configuration.

        Example:
            >>> config = ActorCriticAgentConfig(type="ppo")
            >>> config_dict = config.to_dict()
            >>> print("actor_config" in config_dict)  # True
        """
        base = super().to_dict()
        if isinstance(self.actor_config, NetworkConfig):
            base["actor_config"] = self.actor_config.to_dict()
        return base

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ActorCriticAgentConfig":
        """
        Create configuration instance from dictionary.

        Handles the conversion of network configuration dictionaries back
        to NetworkConfig objects for proper deserialization, including
        the actor network configuration.

        Args:
            config_dict (dict): Dictionary containing configuration parameters.

        Returns:
            ActorCriticAgentConfig: New configuration instance.

        Example:
            >>> config_dict = {
            ...     "type": "ppo",
            ...     "ent_coef": 0.01,
            ...     "actor_config": {"layers": [...]}
            ... }
            >>> config = ActorCriticAgentConfig.from_dict(config_dict)
            >>> print(config.type)  # "ppo"
        """
        # Filter out fields that are not part of __init__
        init_fields = {
            field.name for field in cls.__dataclass_fields__.values() if field.init
        }
        filtered_dict = {k: v for k, v in config_dict.items() if k in init_fields}

        actor = filtered_dict.get("actor_config")

        # Convert network config back to objects
        if isinstance(actor, dict):
            filtered_dict["actor_config"] = NetworkConfig.from_dict(actor)

        # Call parent class from_dict to handle critic_config and feature_extractor_config
        return super().from_dict(filtered_dict)
