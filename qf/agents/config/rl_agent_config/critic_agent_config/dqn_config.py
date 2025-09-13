from dataclasses import dataclass
from typing import Literal, Optional

import optuna

from qf.agents.config.rl_agent_config.critic_agent_config.critic_agent_config import (
    CriticAgentConfig,
)


@dataclass
class DQNConfig(CriticAgentConfig):
    type: Literal["dqn"] = "dqn"
    actor_config: Optional[dict] = None
    target_mode: Literal["soft-bellman", "hard-bellman"] = "soft-bellman"

    # Override defaults specific to DQN
    learning_rate: float = 1e-3
    batch_size: int = 32

    @staticmethod
    def get_default_config() -> "DQNConfig":
        return DQNConfig()

    @staticmethod
    def get_hyperparameter_space(trial: optuna.Trial) -> "DQNConfig":
        return DQNConfig(
            learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            gamma=trial.suggest_float("gamma", 0.90, 0.999),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            epsilon_start=trial.suggest_float("epsilon_start", 0.1, 1.0),
            target_mode=trial.suggest_categorical(
                "target_mode", ["soft-bellman", "hard-bellman"]
            ),
            buffer_size=trial.suggest_categorical(
                "buffer_size", [int(1e4), int(1e5), int(1e6)]
            ),
        )
