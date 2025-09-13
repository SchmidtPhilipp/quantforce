"""
Proximal Policy Optimization (PPO) Agent Configuration Module.

This module provides the PPOConfig class which defines the configuration
for Proximal Policy Optimization reinforcement learning agents. PPO is an
on-policy algorithm that uses a clipped objective to ensure stable updates.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import optuna

from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.on_policy_agent.on_policy_agent_config import (
    OnPolicyAgentConfig,
)


@dataclass
class PPOConfig(OnPolicyAgentConfig):
    """
    Configuration for Proximal Policy Optimization (PPO) reinforcement learning agents.

    PPO is an on-policy actor-critic algorithm that uses a clipped objective
    to ensure stable policy updates. It is one of the most popular and reliable
    RL algorithms, known for its stability and ease of tuning.

    PPO uses a trust region approach by clipping the policy ratio, preventing
    too large policy updates that could destabilize training. It also supports
    multiple epochs of training on the same batch of data.

    Attributes:
        type (Literal["ppo"]): Agent type identifier.
        actor_config (Optional[dict]): Configuration for the actor network.
            If None, uses default actor network configuration.
        clip_range (float): Clipping parameter for the PPO objective.
            Controls the maximum allowed change in the policy ratio.
            Range: [0, 1]. Default: 0.2.
        n_epochs (int): Number of epochs for PPO training on the same batch.
            Higher values can improve sample efficiency but increase
            computational cost. Default: 10.
        learning_rate (float): Learning rate for all networks (actor, critic).
            Default: 3e-4.

    Example:
        >>> from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.on_policy_agent.ppo_config import PPOConfig
        >>>
        >>> # Create a basic PPO configuration
        >>> config = PPOConfig(
        ...     clip_range=0.2,
        ...     n_epochs=10
        ... )
        >>>
        >>> # Create a configuration for conservative updates
        >>> conservative_config = PPOConfig(
        ...     clip_range=0.1,      # Smaller clipping range
        ...     n_epochs=20,         # More epochs
        ...     learning_rate=1e-4   # Lower learning rate
        ... )
        >>>
        >>> # Create a configuration for aggressive updates
        >>> aggressive_config = PPOConfig(
        ...     clip_range=0.3,      # Larger clipping range
        ...     n_epochs=5,          # Fewer epochs
        ...     learning_rate=1e-3   # Higher learning rate
        ... )
        >>>
        >>> # Create a configuration for stable learning
        >>> stable_config = PPOConfig(
        ...     clip_range=0.15,     # Moderate clipping
        ...     n_epochs=15,         # Moderate epochs
        ...     learning_rate=3e-4,  # Standard learning rate
        ...     n_steps=4096,        # More steps per update
        ...     gae_lambda=0.99      # High lambda for stability
        ... )
    """

    type: Literal["ppo"] = "ppo"
    actor_config: Optional[dict] = None
    clip_range: float = 0.2
    n_epochs: int = 10  # Number of epochs for PPO training

    # Override defaults specific to PPO
    learning_rate: float = 3e-4  # lr in original
    # n_steps and other properties inherited from OnPolicyAgentConfig

    @staticmethod
    def get_default_config() -> "PPOConfig":
        """
        Get default configuration for PPO agent.

        Returns a configuration with sensible defaults for PPO training,
        including standard clipping range and epoch settings.

        Returns:
            PPOConfig: Default configuration.

        Example:
            >>> config = PPOConfig.get_default_config()
            >>> print(config.type)  # "ppo"
            >>> print(config.clip_range)  # 0.2
        """
        return PPOConfig()

    @staticmethod
    def get_hyperparameter_space(trial: optuna.Trial) -> "PPOConfig":
        """
        Get hyperparameter space for Optuna optimization.

        Defines the search space for hyperparameter optimization using Optuna.
        This method is used by the hyperparameter optimization framework to
        automatically search for optimal parameter combinations.

        Args:
            trial (optuna.Trial): Optuna trial object for parameter suggestions.

        Returns:
            PPOConfig: Configuration with suggested parameters.

        Example:
            >>> import optuna
            >>>
            >>> def objective(trial):
            ...     config = PPOConfig.get_hyperparameter_space(trial)
            ...     # Use config for training/evaluation
            ...     return performance_score
            >>>
            >>> study = optuna.create_study(direction="maximize")
            >>> study.optimize(objective, n_trials=100)
        """
        return PPOConfig(
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            gamma=trial.suggest_float("gamma", 0.95, 0.999),
            clip_range=trial.suggest_float("clip_range", 0.1, 0.4),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            gae_lambda=trial.suggest_float("gae_lambda", 0.8, 0.99),
            n_steps=trial.suggest_int("n_steps", 512, 2048),
            n_epochs=trial.suggest_int("n_epochs", 4, 20),
        )
