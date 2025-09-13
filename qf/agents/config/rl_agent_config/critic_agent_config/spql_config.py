from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import optuna

from qf.agents.config.rl_agent_config.critic_agent_config.critic_agent_config import (
    CriticAgentConfig,
)
from qf.networks.default_networks import DefaultNetworks
from qf.networks.network_config import NetworkConfig

#######################################################################################################


@dataclass
class SPQLConfig(CriticAgentConfig):
    type: Literal["spql"] = "spql"
    critic_config: Optional[dict] = None  # Do not configure
    temperature: float = 1.0  # Temperature parameter for soft updates
    backup_mode: Literal["soft-bellman", "hard-bellman"] = "soft-bellman"

    # Feature extractor config
    feature_extractor_config: Optional[Union[NetworkConfig, dict]] = field(
        default_factory=lambda: DefaultNetworks.get_cnn_feature_extractor()
    )

    # Actor config
    actor_config: Optional[Union[NetworkConfig, dict]] = field(
        default_factory=lambda: DefaultNetworks.get_deep_actor()
    )

    # Override defaults specific to SPQL
    learning_rate: float = 0.005511452523855128
    batch_size: int = 32

    @staticmethod
    def get_default_config() -> "SPQLConfig":
        return SPQLConfig()

    @staticmethod
    def get_optimized_config() -> "SPQLConfig":
        return SPQLConfig(
            learning_rate=0.005511452523855128,
            gamma=0.8923961400174644,
            batch_size=64,
            epsilon_start=0.4140285477731602,
            temperature=0.0022636670850494545,
            backup_mode="soft-bellman",
        )

    @staticmethod
    def get_hyperparameter_space(trial: optuna.Trial) -> "SPQLConfig":
        return SPQLConfig(
            learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            gamma=trial.suggest_float("gamma", 0.90, 0.999),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            epsilon_start=trial.suggest_float("epsilon_start", 0.1, 1.0),
            buffer_size=trial.suggest_categorical(
                "buffer_size", [int(1e4), int(1e5), int(1e6)]
            ),
            tau=trial.suggest_float("tau", 0.001, 0.05),
            temperature=trial.suggest_float("temperature", 0.001, 1),
        )
