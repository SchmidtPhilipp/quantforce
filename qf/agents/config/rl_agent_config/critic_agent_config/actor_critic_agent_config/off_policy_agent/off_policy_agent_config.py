"""
Off-Policy Agent Configuration Module.

This module provides the OffPolicyAgentConfig class which defines the configuration
for off-policy actor-critic reinforcement learning agents like SAC, DDPG, and TD3.
These agents can learn from experience replay buffers and are generally more
sample-efficient than on-policy methods.
"""

from dataclasses import dataclass

from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.actor_critic_agent_config import (
    ActorCriticAgentConfig,
)


@dataclass
class OffPolicyAgentConfig(ActorCriticAgentConfig):
    """
    Configuration for off-policy actor-critic agents like SAC, DDPG, TD3.

    This class extends ActorCriticAgentConfig to include parameters specific to
    off-policy reinforcement learning algorithms. Off-policy methods can learn
    from experience replay buffers, making them more sample-efficient than
    on-policy methods.

    Off-policy methods are particularly effective for continuous action spaces
    and can handle complex environments with high-dimensional state and action
    spaces. They are generally more stable and sample-efficient than on-policy
    methods but may be more sensitive to hyperparameter tuning.

    Attributes:
        train_freq (int): Frequency of training updates in environment steps.
            Higher values reduce computational overhead but may slow learning.
            Default: 10 (train every 10 steps).
        gradient_steps (int): Number of gradient steps per training update.
            Higher values can improve learning but increase computational cost.
            Default: 1 (single gradient step).
        tau (float): Soft update parameter for target networks.
            Controls how much the target network is updated towards the
            main network. Range: [0, 1]. Default: 0.01.

    Example:
        >>> from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.off_policy_agent.off_policy_agent_config import OffPolicyAgentConfig
        >>>
        >>> # Create a basic off-policy configuration
        >>> config = OffPolicyAgentConfig(
        ...     type="sac",
        ...     train_freq=10,
        ...     gradient_steps=1
        ... )
        >>>
        >>> # Create a configuration for frequent training
        >>> frequent_config = OffPolicyAgentConfig(
        ...     type="ddpg",
        ...     train_freq=1,      # Train every step
        ...     gradient_steps=4   # Multiple gradient steps
        ... )
        >>>
        >>> # Create a configuration for efficient training
        >>> efficient_config = OffPolicyAgentConfig(
        ...     type="td3",
        ...     train_freq=20,     # Train less frequently
        ...     gradient_steps=1   # Single gradient step
        ... )
        >>>
        >>> # Create a configuration for stable learning
        >>> stable_config = OffPolicyAgentConfig(
        ...     type="sac",
        ...     train_freq=8,      # Moderate training frequency
        ...     gradient_steps=2,  # Multiple gradient steps
        ...     tau=0.005          # Slower target updates
        ... )
    """

    train_freq: int = (
        10  # Train every 10 steps for better decorrelation and performance
    )
    gradient_steps: int = 10  # Attention for SAC train_freq/gradient_steps \approx 1

    tau: float = 0.01  # Override with default value for off-policy methods
