from dataclasses import dataclass
from typing import Literal

import optuna

from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.actor_critic_agent_config import (
    ActorCriticAgentConfig,
)


@dataclass
class MADDPGConfig(ActorCriticAgentConfig):
    type: Literal["maddpg"] = "maddpg"
    loss_fn: Literal["mse", "weighted_correlation_loss"] = "mse"
    lambda_: float = 0.95
    ou_mu: float = 0.0
    ou_theta: float = 0.15
    ou_sigma: float = 0.2
    ou_dt: float = 1e-2

    # Override defaults specific to MADDPG
    learning_rate: float = 0.0001

    # Allow n_agents to be settable via __init__
    n_agents: int = 1

    @staticmethod
    def get_default_config() -> "MADDPGConfig":
        return MADDPGConfig()

    @staticmethod
    def get_hyperparameter_space(trial: optuna.Trial) -> "MADDPGConfig":
        return MADDPGConfig(
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-3, step=1e-4),
            lambda_=trial.suggest_float("lambda_", 0.9, 0.95, step=0.01),
            loss_fn=trial.suggest_categorical(
                "loss_fn", ["mse", "weighted_correlation_loss"]
            ),
            tau=trial.suggest_float("tau", 0.001, 0.01, step=1e-4),
            gamma=trial.suggest_float("gamma", 0.8, 0.99, step=0.01),
            ou_mu=trial.suggest_float("ou_mu", -0.1, 0.1, step=0.01),
            ou_theta=trial.suggest_float("ou_theta", 0.1, 0.2, step=0.01),
            ou_sigma=trial.suggest_float("ou_sigma", 0.1, 0.2, step=0.01),
            ou_dt=trial.suggest_float("ou_dt", 1e-3, 1e-2, step=1e-3),
        )
