from dataclasses import dataclass
from typing import Literal

import optuna

from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.on_policy_agent.on_policy_agent_config import (
    OnPolicyAgentConfig,
)


@dataclass
class A2CConfig(OnPolicyAgentConfig):
    type: Literal["a2c"] = "a2c"

    # Override defaults specific to A2C
    learning_rate: float = 0.0007
    n_steps: int = 5
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    gae_lambda: float = 1.0

    @staticmethod
    def get_default_config() -> "A2CConfig":
        return A2CConfig()

    @staticmethod
    def get_hyperparameter_space(trial: optuna.Trial) -> "A2CConfig":
        return A2CConfig(
            learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
            gamma=trial.suggest_float("gamma", 0.95, 0.999),
            ent_coef=trial.suggest_float("ent_coef", 0.001, 0.1),
            vf_coef=trial.suggest_float("vf_coef", 0.1, 1.0),
            n_steps=trial.suggest_int("n_steps", 5, 20),
            gae_lambda=trial.suggest_float("gae_lambda", 0.9, 1.0),
            normalize_advantage=trial.suggest_categorical(
                "normalize_advantage", [True, False]
            ),
            max_grad_norm=trial.suggest_float("max_grad_norm", 0.3, 0.7),
        )
