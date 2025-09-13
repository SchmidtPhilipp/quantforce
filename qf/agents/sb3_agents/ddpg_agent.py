from typing import Optional

import optuna
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.td3.policies import TD3Policy

from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.off_policy_agent.ddpg_config import (
    DDPGConfig,
)
from qf.agents.sb3_agents.policy.custom_actor_citric_policy import (
    CustomFeaturesExtractor,
)
from qf.agents.sb3_agents.sb3_agent import SB3Agent


class DDPGAgent(SB3Agent):
    def __init__(self, env, config: Optional[DDPGConfig] = None):
        """
        Initializes the DDPG agent with the given environment and configuration.
        Parameters:
            env: The environment in which the agent will operate.
            config (dict): Configuration dictionary for the DDPG agent.
        """
        self.config = config or DDPGConfig.get_default_config()
        super().__init__(env, config=self.config)

        n_actions = self.env.action_space.shape[0]
        action_noise = (
            NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=self.config.action_noise_sigma * np.ones(n_actions),
            )
            if self.config.action_noise
            else None
        )

        # Create custom feature extractor
        custom_features_extractor = CustomFeaturesExtractor(
            observation_space=env.observation_space,
            network_config=self.config.feature_extractor_config,
        )

        policy_kwargs = dict(
            features_extractor_class=type(custom_features_extractor),
            features_extractor_kwargs=dict(
                network_config=self.config.feature_extractor_config
            ),
        )

        # Initialize DDPG model
        self.model = DDPG(
            policy=TD3Policy,
            env=self.env,
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            batch_size=self.config.batch_size,
            tau=self.config.tau,
            gamma=self.config.gamma,
            train_freq=self.config.train_freq,
            gradient_steps=self.config.gradient_steps,
            verbose=self.config.verbosity,
            device=self.config.device,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
        )

        from .td3_agent import train_TD3_with_TD_error_logging

        # self.model.train = lambda gradient_steps, batch_size=64: train_TD3_with_TD_error_logging(self.model, gradient_steps, batch_size)

    @staticmethod
    def get_default_config():
        return DDPGConfig.get_default_config()

    @staticmethod
    def get_hyperparameter_space(trial: optuna.Trial):
        return DDPGConfig.get_hyperparameter_space(trial)
