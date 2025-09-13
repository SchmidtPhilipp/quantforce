from dataclasses import dataclass
from typing import Literal

import optuna

from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.off_policy_agent.off_policy_agent_config import (
    OffPolicyAgentConfig,
)


@dataclass
class DDPGConfig(OffPolicyAgentConfig):
    type: Literal["ddpg"] = "ddpg"
    action_noise: bool = False
    action_noise_sigma: float = 0.2

    # Override defaults specific to DDPG
    learning_rate: float = 0.001

    # Performance optimizations (inherited from OffPolicyAgentConfig)
    train_freq: int = 10  # Train every 10 steps (larger is better for decorrelation)
    gradient_steps: int = 1  # Single gradient step to avoid excessive computation

    @staticmethod
    def get_default_config() -> "DDPGConfig":
        return DDPGConfig()

    @staticmethod
    def get_hyperparameter_space(trial: optuna.Trial) -> "DDPGConfig":
        return DDPGConfig(
            learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
            gamma=trial.suggest_float("gamma", 0.8, 0.99),
            tau=trial.suggest_float("tau", 0.001, 0.01),
            buffer_size=trial.suggest_categorical(
                "buffer_size", [int(1e4), int(1e5), int(1e6)]
            ),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            action_noise=trial.suggest_categorical("action_noise", [True, False]),
            action_noise_sigma=trial.suggest_float("action_noise_sigma", 0.1, 0.3),
        )
