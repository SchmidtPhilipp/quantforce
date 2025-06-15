from typing import Any, Dict, Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import polyak_update

import qf
from qf.agents.sb3_agents.sb3_agent import SB3Agent
from qf.envs.multi_agent_portfolio_env import MultiAgentPortfolioEnv


class DecayingNormalActionNoise(ActionNoise):
    def __init__(
        self, mean: np.ndarray, sigma_init: float, sigma_final: float, decay_steps: int
    ):
        """
        Gaussian action noise with linear decay over time.

        :param mean: Mean of the noise (array of zeros matching action space shape)
        :param sigma_init: Initial standard deviation
        :param sigma_final: Final standard deviation
        :param decay_steps: Number of steps over which to decay sigma
        """
        self.mean = mean
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.decay_steps = decay_steps
        self.n_steps = 0

    def __call__(self) -> np.ndarray[Any, np.dtype[np.float32]]:

        # Linearly decaying sigma
        decay_ratio = min(self.n_steps / self.decay_steps, 1.0)
        sigma = self.sigma_init * (1 - decay_ratio) + self.sigma_final * decay_ratio
        self.n_steps += 1
        return np.array(np.random.normal(self.mean, sigma))

    def reset(self):
        self.n_steps = 0


class SACAgent(SB3Agent):
    def __init__(
        self, env: MultiAgentPortfolioEnv, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initializes the SAC agent with the given environment and configuration.
        Parameters:
            env: The environment in which the agent will operate.
            config (dict): Configuration dictionary for the SAC agent.
        """
        super().__init__(env)

        # Default configuration
        default_config = {
            "policy": qf.DEFAULT_SAC_POLICY,  # Default policy architecture
            "learning_rate": qf.DEFAULT_SAC_LR,
            "buffer_size": qf.DEFAULT_SAC_BUFFER_MAX_SIZE,
            "batch_size": qf.DEFAULT_SAC_BATCH_SIZE,
            "tau": qf.DEFAULT_SAC_TAU,  # Target network update rate
            "gamma": qf.DEFAULT_SAC_GAMMA,
            "train_freq": qf.DEFAULT_SAC_TRAIN_FREQ,  # Frequency of training steps
            "gradient_steps": qf.DEFAULT_SAC_GRADIENT_STEPS,  # Number of gradient steps per training iteration
            "device": qf.DEFAULT_DEVICE,  # Device to run the computations on
            "ent_coef": qf.DEFAULT_SAC_ENT_COEF,  # Automatic entropy coefficient adjustment
            "verbose": qf.DEFAULT_SAC_VERBOSITY,  # Verbosity level for logging
            "action_noise": qf.DEFAULT_SAC_ACTION_NOISE,  # Action noise for exploration
            "action_noise_sigma_init": qf.DEFAULT_SAC_ACTION_NOISE_SIGMA_INIT,  # Initial action noise sigma
            "action_noise_sigma_final": qf.DEFAULT_SAC_ACTION_NOISE_SIGMA_FINAL,  # Final action noise sigma
            "action_noise_decay_steps": qf.DEFAULT_SAC_ACTION_NOISE_DECAY_STEPS,  # Decay steps for action noise
        }

        # Merge default config with provided config
        self.config = {**default_config, **(config or {})}

        # Initialize SAC model
        self.model = SAC(
            policy=self.config["policy"],
            env=self.env,
            learning_rate=self.config["learning_rate"],
            buffer_size=self.config["buffer_size"],
            batch_size=self.config["batch_size"],
            tau=self.config["tau"],
            gamma=self.config["gamma"],
            train_freq=self.config["train_freq"],
            gradient_steps=self.config["gradient_steps"],
            verbose=self.config["verbose"],
            device=self.config["device"],
            ent_coef=self.config["ent_coef"],
            action_noise=(
                DecayingNormalActionNoise(
                    mean=np.zeros(self.env.action_space.shape[0]),
                    sigma_init=self.config["action_noise_sigma_init"],
                    sigma_final=self.config["action_noise_sigma_final"],
                    decay_steps=self.config["action_noise_decay_steps"],
                )
                if "action_noise" in self.config
                else None
            ),
        )

        # Override the train method
        self.model.train = (
            lambda gradient_steps, batch_size=64: train_SAC_with_TD_error_logging(
                self.model, gradient_steps, batch_size
            )
        )

    @staticmethod
    def get_default_config():
        return qf.DEFAULT_SACAGENT_CONFIG

    @staticmethod
    def get_hyperparameter_space():
        return qf.DEFAULT_SACAGENT_HYPERPARAMETER_SPACE


# We are overriding the train method to be able to calculate the TD error to log it in TensorBoard.
# This is basically a copy of the original train method from stable-baselines3, but with added logging for TD error.
def train_SAC_with_TD_error_logging(
    self, gradient_steps: int, batch_size: int = 64
) -> None:
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update optimizers learning rate
    optimizers = [self.actor.optimizer, self.critic.optimizer]
    if self.ent_coef_optimizer is not None:
        optimizers += [self.ent_coef_optimizer]

    # Update learning rate according to lr schedule
    self._update_learning_rate(optimizers)

    ent_coef_losses, ent_coefs = [], []
    actor_losses, critic_losses = [], []

    for gradient_step in range(gradient_steps):
        # Sample replay buffer
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

        # We need to sample because `log_std` may have changed between two gradient steps
        if self.use_sde:
            self.actor.reset_noise()

        # Action by the current actor for the sampled state
        actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = th.exp(self.log_ent_coef.detach())
            assert isinstance(self.target_entropy, float)
            ent_coef_loss = -(
                self.log_ent_coef * (log_prob + self.target_entropy).detach()
            ).mean()
            ent_coef_losses.append(ent_coef_loss.item())
        else:
            ent_coef = self.ent_coef_tensor

        ent_coefs.append(ent_coef.item())

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(
                replay_data.next_observations
            )
            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(
                self.critic_target(replay_data.next_observations, next_actions), dim=1
            )
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = (
                replay_data.rewards
                + (1 - replay_data.dones) * self.gamma * next_q_values
            )

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        # Compute TD error
        sac_td_error = (
            th.abs(target_q_values - current_q_values[0]).mean().item()
        )  # This line is added

        with th.no_grad():
            next_q_values = th.cat(
                self.critic_target(replay_data.next_observations, next_actions), dim=1
            )
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            # 👉 KEIN Entropie-Term hier:
            corrected_target_q_values = (
                replay_data.rewards
                + (1 - replay_data.dones) * self.gamma * next_q_values
            )

        # TD Error "wie bei Q-Learning"
        td_error = th.abs(corrected_target_q_values - current_q_values[0]).mean().item()

        # Compute critic loss
        critic_loss = 0.5 * sum(
            F.mse_loss(current_q, target_q_values) for current_q in current_q_values
        )
        assert isinstance(critic_loss, th.Tensor)  # for type checker
        critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Min over all critic networks
        q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        actor_losses.append(actor_loss.item())

        # Optimize the actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        if gradient_step % self.target_update_interval == 0:
            polyak_update(
                self.critic.parameters(), self.critic_target.parameters(), self.tau
            )
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

    self._n_updates += gradient_steps

    # And the lines underneath are added to log the TD error and losses in TensorBoard.
    if len(ent_coef_losses) > 0:
        self.env.envs[0].env.env.logger.record(
            "TRAIN_model_loss/ent_coef_loss", np.mean(ent_coef_losses)
        )
    self.env.envs[0].env.env.logger.record(
        "TRAIN_model_loss/10*log(TD_Error)",
        10 * np.log10(td_error),
        step=self._n_updates,
    )
    self.env.envs[0].env.env.logger.record(
        "TRAIN_model_loss/10*log(SAC_TD_Error)",
        10 * np.log10(sac_td_error),
        step=self._n_updates,
    )
    if len(actor_losses) > 0:
        self.env.envs[0].env.env.logger.record(
            "TRAIN_model_loss/actor_loss", np.mean(actor_losses), step=self._n_updates
        )
    self.env.envs[0].env.env.logger.record(
        "TRAIN_model_loss/critic_loss", np.mean(critic_losses), step=self._n_updates
    )
    self.env.envs[0].env.env.logger.record(
        "TRAIN_model_loss/ent_coef", np.mean(ent_coefs)
    )
