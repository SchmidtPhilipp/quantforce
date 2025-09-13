"""
Soft Actor-Critic (SAC) Agent Configuration Module.

This module provides the SACConfig class which defines the configuration
for Soft Actor-Critic reinforcement learning agents. SAC is an off-policy
actor-critic algorithm that uses entropy regularization for exploration.
"""

from dataclasses import dataclass
from typing import Literal

import optuna
from qf.utils.logging_config import get_logger

logger = get_logger(__name__)
from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.off_policy_agent.off_policy_agent_config import (
    OffPolicyAgentConfig,
)


@dataclass
class SACConfig(OffPolicyAgentConfig):
    """
    Configuration for Soft Actor-Critic (SAC) reinforcement learning agents.

    SAC is an off-policy actor-critic algorithm that uses entropy regularization
    to encourage exploration. It is particularly effective for continuous action
    spaces and can achieve high sample efficiency while maintaining stability.

    SAC automatically adjusts the entropy coefficient to maintain a target
    entropy level, making it less sensitive to hyperparameter tuning compared
    to other RL algorithms.

    Attributes:
        type (Literal["sac"]): Agent type identifier.
        alpha (float): Initial entropy coefficient for automatic entropy tuning.
            Controls the trade-off between exploration and exploitation.
            Default: 1.0.
        ent_coef (str): Entropy coefficient mode. Options include:
            - "auto": Automatic entropy tuning
            - "auto_0.01", "auto_0.1", "auto_1", "auto_10", "auto_100":
              Automatic tuning with different initial values
            - Numeric values: Fixed entropy coefficient
            Default: "auto_1".
        action_noise (bool): Whether to add noise to actions for exploration.
            Default: False.
        action_noise_sigma_init (float): Initial standard deviation for action noise.
            Default: 0.2.
        action_noise_sigma_final (float): Final standard deviation for action noise.
            Default: 0.001.
        action_noise_decay_steps (int): Number of steps to decay action noise.
            Default: 100_000.
        learning_rate (float): Learning rate for all networks (actor, critic, entropy).
            Default: 0.0001.
        batch_size (int): Number of samples per training batch.
            Default: 64.
        buffer_size (int): Size of the experience replay buffer.
            Default: 10_000.
        tau (float): Soft update parameter for target networks.
            Default: 0.01.

    Example:
        >>> from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.off_policy_agent.sac_config import SACConfig
        >>>
        >>> # Create a basic SAC configuration
        >>> config = SACConfig(
        ...     alpha=1.0,
        ...     ent_coef="auto_1"
        ... )
        >>>
        >>> # Create a configuration for high exploration
        >>> high_exploration_config = SACConfig(
        ...     alpha=2.0,                    # Higher entropy coefficient
        ...     ent_coef="auto_10",           # Higher initial entropy
        ...     action_noise=True,             # Add action noise
        ...     action_noise_sigma_init=0.3   # Higher initial noise
        ... )
        >>>
        >>> # Create a configuration for stable learning
        >>> stable_config = SACConfig(
        ...     alpha=0.5,                    # Lower entropy coefficient
        ...     ent_coef="auto_0.1",          # Lower initial entropy
        ...     learning_rate=0.00005,        # Lower learning rate
        ...     batch_size=128,               # Larger batch size
        ...     buffer_size=50_000            # Larger buffer
        ... )
        >>>
        >>> # Create a configuration for fast learning
        >>> fast_config = SACConfig(
        ...     alpha=1.0,
        ...     ent_coef="auto_1",
        ...     learning_rate=0.001,          # Higher learning rate
        ...     batch_size=32,                # Smaller batch size
        ...     train_freq=1                  # Train every step
        ... )
    """

    type: Literal["sac"] = "sac"
    alpha: float = 1.0
    ent_coef: str = "auto_1"

    action_noise: bool = False
    action_noise_sigma_init: float = 0.2
    action_noise_sigma_final: float = 0.001
    action_noise_decay_steps: int = 100_000

    # Optimized defaults for better performance
    learning_rate: float = 0.0001
    batch_size: int = 64
    buffer_size: int = 50_000
    tau: float = 0.01

    # After init
    def __post_init__(self):
        UTD = self.gradient_steps / self.train_freq
        if UTD > 1:
            logger.warning(
                f"UTD = {UTD} > 1. Attention you are overfitting to the data. This might lead to instability."
            )
        if UTD < 1:
            logger.warning(
                f"UTD = {UTD} < 1. Network is going to be undertrained. Might lead to divergence."
            )

    @staticmethod
    def get_default_config() -> "SACConfig":
        """
        Get default configuration for SAC agent.

        Returns a configuration with sensible defaults for SAC training,
        including automatic entropy tuning and optimized hyperparameters.

        Returns:
            SACConfig: Default configuration.

        Example:
            >>> config = SACConfig.get_default_config()
            >>> print(config.type)  # "sac"
            >>> print(config.ent_coef)  # "auto_1"
        """
        return SACConfig()

    @staticmethod
    def get_hyperparameter_space(trial: optuna.Trial) -> "SACConfig":
        """
        Get hyperparameter space for Optuna optimization.

        Defines the search space for hyperparameter optimization using Optuna.
        This method is used by the hyperparameter optimization framework to
        automatically search for optimal parameter combinations.

        Args:
            trial (optuna.Trial): Optuna trial object for parameter suggestions.

        Returns:
            SACConfig: Configuration with suggested parameters.

        Example:
            >>> import optuna
            >>>
            >>> def objective(trial):
            ...     config = SACConfig.get_hyperparameter_space(trial)
            ...     # Use config for training/evaluation
            ...     return performance_score
            >>>
            >>> study = optuna.create_study(direction="maximize")
            >>> study.optimize(objective, n_trials=100)
        """
        return SACConfig(
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            gamma=trial.suggest_float("gamma", 0.8, 0.99),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            buffer_size=trial.suggest_categorical(
                "buffer_size", [int(1e4), int(1e5), int(1e6)]
            ),
            alpha=trial.suggest_float("alpha", 0.0, 1.0),
            tau=trial.suggest_float("tau", 0.001, 0.2),
            train_freq=trial.suggest_categorical("train_freq", [1, 2, 4, 8, 16]),
            gradient_steps=trial.suggest_categorical(
                "gradient_steps", [1, 2, 4, 8, 16]
            ),
            ent_coef=trial.suggest_categorical(
                "ent_coef",
                [
                    "auto",
                    "auto_0.01",
                    "auto_0.1",
                    "auto_1",
                    "auto_10",
                    "auto_100",
                    0.01,
                    0.1,
                    1,
                    10,
                    100,
                ],
            ),
        )
