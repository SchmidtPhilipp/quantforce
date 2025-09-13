from typing import Optional

import optuna
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.td3.policies import TD3Policy

import qf as qf
from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.off_policy_agent.td3_config import (
    TD3Config,
)
from qf.agents.sb3_agents.policy.custom_actor_citric_policy import (
    CustomFeaturesExtractor,
)
from qf.agents.sb3_agents.sb3_agent import SB3Agent


class TD3Agent(SB3Agent):
    def __init__(self, env, config: Optional[TD3Config] = None):
        """
        Initializes the TD3 agent with the given environment and configuration.
        Parameters:
            env: The environment in which the agent will operate.
            config (dict): Configuration dictionary for the TD3 agent.
        """
        self.config = config or TD3Config.get_default_config()
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

        # Initialize TD3 model
        self.model = TD3(
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
            policy_kwargs=policy_kwargs,
            # tensorboard_log=self.env.get_save_dir()  # Use the environment's save directory for TensorBoard logging
        )

        # Override the train method
        self.model.train = (
            lambda gradient_steps, batch_size=64: train_TD3_with_TD_error_logging(
                self.model, gradient_steps, batch_size
            )
        )

    @staticmethod
    def get_default_config():
        return TD3Config.get_default_config()

    @staticmethod
    def get_hyperparameter_space(trial: optuna.Trial):
        return TD3Config.get_hyperparameter_space(trial)


# We are overriding the train method to be able to calculate the TD error to log it in TensorBoard.
# This is basically a copy of the original train method from stable-baselines3, but with added logging for TD error.
def train_TD3_with_TD_error_logging(
    self, gradient_steps: int, batch_size: int = 100
) -> None:
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)

    # Update learning rate according to lr schedule
    self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

    actor_losses, critic_losses = [], []
    for _ in range(gradient_steps):
        self._n_updates += 1
        # Sample replay buffer
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

        with th.no_grad():
            # Select action according to policy and add clipped noise
            noise = replay_data.actions.clone().data.normal_(
                0, self.target_policy_noise
            )
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (
                self.actor_target(replay_data.next_observations) + noise
            ).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = th.cat(
                self.critic_target(replay_data.next_observations, next_actions), dim=1
            )
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            target_q_values = (
                replay_data.rewards
                + (1 - replay_data.dones) * self.gamma * next_q_values
            )

        # Get current Q-values estimates for each critic network
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        # Compute TD error
        td_error = (
            th.abs(target_q_values - current_q_values[0]).mean().item()
        )  # This line is added

        # Compute critic loss
        critic_loss = sum(
            F.mse_loss(current_q, target_q_values) for current_q in current_q_values
        )
        assert isinstance(critic_loss, th.Tensor)
        critic_losses.append(critic_loss.item())

        # Optimize the critics
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1_forward(
                replay_data.observations, self.actor(replay_data.observations)
            ).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(
                self.critic.parameters(), self.critic_target.parameters(), self.tau
            )
            polyak_update(
                self.actor.parameters(), self.actor_target.parameters(), self.tau
            )
            # Copy running stats, see GH issue #996
            polyak_update(
                self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0
            )
            polyak_update(
                self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0
            )

    # And the lines underneath are added to log the TD error and losses in TensorBoard.
    self.env.envs[0].env.env.experiment_logger.log_scalar(
        "TRAIN_model_loss/10*log(TD_Error)",
        10 * np.log10(td_error),
        step=self._n_updates,
    )
    if len(actor_losses) > 0:
        self.env.envs[0].env.env.experiment_logger.log_scalar(
            "TRAIN_model_loss/actor_loss", np.mean(actor_losses), step=self._n_updates
        )
    self.env.envs[0].env.env.experiment_logger.log_scalar(
        "TRAIN_model_loss/critic_loss", np.mean(critic_losses), step=self._n_updates
    )
