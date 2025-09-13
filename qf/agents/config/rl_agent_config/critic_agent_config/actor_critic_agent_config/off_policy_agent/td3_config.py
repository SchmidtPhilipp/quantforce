from dataclasses import dataclass
from typing import Literal

import optuna
import torch.nn as nn

from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.off_policy_agent.off_policy_agent_config import (
    OffPolicyAgentConfig,
)


@dataclass
class TD3Config(OffPolicyAgentConfig):
    type: Literal["td3"] = "td3"
    policy: str = "MlpPolicy"

    noise_std: float = 0.2
    noise_clip: float = 0.5

    # Override defaults specific to TD3
    learning_rate: float = 0.0003
    batch_size: int = 100
    gradient_steps: int = -1  # -1 means auto

    @staticmethod
    def get_default_config() -> "TD3Config":
        return TD3Config()

    @staticmethod
    def get_hyperparameter_space(trial: optuna.Trial) -> "TD3Config":
        return TD3Config(
            learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
            gamma=trial.suggest_float("gamma", 0.8, 0.99),
            tau=trial.suggest_float("tau", 0.001, 0.01),
            train_freq=trial.suggest_int("train_freq", 1, 5),
            gradient_steps=trial.suggest_int("gradient_steps", -1, 5),
            noise_std=trial.suggest_float("noise_std", 0.1, 0.3),
            noise_clip=trial.suggest_float("noise_clip", 0.3, 0.5),
            net_arch=trial.suggest_categorical(
                "net_arch",
                [
                    [64, 64],
                    [128, 128],
                    [256, 256],
                    [400, 300],
                    [64, 64, 64],
                ],
            ),
            activation_fn=trial.suggest_categorical(
                "activation_fn", [nn.ReLU, nn.Tanh, nn.ELU]
            ),
        )
