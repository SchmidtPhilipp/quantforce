"""
On-Policy Agent Configuration Module.

This module provides the OnPolicyAgentConfig class which defines the configuration
for on-policy actor-critic reinforcement learning agents like A2C and PPO.
These agents learn from the current policy's experience and are generally
more stable but less sample-efficient than off-policy methods.
"""

from dataclasses import dataclass

from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.actor_critic_agent_config import (
    ActorCriticAgentConfig,
)


@dataclass
class OnPolicyAgentConfig(ActorCriticAgentConfig):
    """
    Configuration for on-policy actor-critic agents like A2C, PPO.

    This class extends ActorCriticAgentConfig to include parameters specific to
    on-policy reinforcement learning algorithms. On-policy methods learn from
    the current policy's experience, making them more stable but generally
    less sample-efficient than off-policy methods.

    On-policy methods are particularly effective for environments where
    exploration is important and the policy needs to be updated frequently.
    They are generally more stable and easier to tune than off-policy methods
    but may require more environment interactions to achieve good performance.

    Attributes:
        n_steps (int): Number of steps to run for each environment per update.
            Higher values provide more stable updates but may slow learning.
            Default: 2048.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
            Controls the trade-off between bias and variance in advantage
            estimation. Range: [0, 1]. Default: 0.95.
        normalize_advantage (bool): Whether to normalize advantages.
            Normalization can improve training stability. Default: True.
        max_grad_norm (float): Maximum gradient norm for gradient clipping.
            Prevents exploding gradients and improves training stability.
            Default: 0.5.

    Example:
        >>> from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.on_policy_agent.on_policy_agent_config import OnPolicyAgentConfig
        >>>
        >>> # Create a basic on-policy configuration
        >>> config = OnPolicyAgentConfig(
        ...     type="ppo",
        ...     n_steps=2048,
        ...     gae_lambda=0.95
        ... )
        >>>
        >>> # Create a configuration for frequent updates
        >>> frequent_config = OnPolicyAgentConfig(
        ...     type="a2c",
        ...     n_steps=1024,        # Fewer steps per update
        ...     gae_lambda=0.9       # Lower lambda for faster updates
        ... )
        >>>
        >>> # Create a configuration for stable learning
        >>> stable_config = OnPolicyAgentConfig(
        ...     type="ppo",
        ...     n_steps=4096,        # More steps per update
        ...     gae_lambda=0.99,     # Higher lambda for stability
        ...     max_grad_norm=0.3    # Lower gradient clipping
        ... )
        >>>
        >>> # Create a configuration for exploration
        >>> exploration_config = OnPolicyAgentConfig(
        ...     type="a2c",
        ...     n_steps=512,         # Very frequent updates
        ...     gae_lambda=0.8,      # Lower lambda for exploration
        ...     normalize_advantage=False  # No advantage normalization
        ... )
    """

    n_steps: int = 2048
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    max_grad_norm: float = 0.5
