from typing import Optional

import optuna
from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy

import qf
from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.on_policy_agent.a2c_config import (
    A2CConfig,
)
from qf.agents.sb3_agents.policy.custom_actor_citric_policy import (
    CustomFeaturesExtractor,
)
from qf.agents.sb3_agents.sb3_agent import SB3Agent


class A2CAgent(SB3Agent):
    def __init__(self, env, config: Optional[A2CConfig] = None):
        """
        Initializes the A2C agent with the given environment and configuration.
        Parameters:
            env: The environment in which the agent will operate.
            config (dict): Configuration dictionary for the A2C agent.
        """
        self.config = config or A2CConfig.get_default_config()
        super().__init__(env, config=self.config)

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

        # Initialize A2C model
        self.model = A2C(
            policy=ActorCriticPolicy,
            env=self.env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            verbose=self.config.verbosity,
            device=self.config.device,
            policy_kwargs=policy_kwargs,
        )

    @staticmethod
    def get_default_config():
        return A2CConfig.get_default_config()

    @staticmethod
    def get_hyperparameter_space(trial: optuna.Trial):
        return A2CConfig.get_hyperparameter_space(trial)
