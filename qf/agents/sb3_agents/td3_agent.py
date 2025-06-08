import torch as th
from stable_baselines3 import TD3
from qf.agents.sb3_agents.sb3_agent import SB3Agent
import qf as qf
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update
import numpy as np


class TD3Agent(SB3Agent):
    def __init__(self, env, config=None):
        """
        Initializes the TD3 agent with the given environment and configuration.
        Parameters:
            env: The environment in which the agent will operate.
            config (dict): Configuration dictionary for the TD3 agent.
        """
        super().__init__(env)

        # Default configuration
        default_config = {
            "policy": qf.DEFAULT_TD3_POLICY,  # Default policy architecture
            "learning_rate": qf.DEFAULT_TD3_LR,
            "buffer_size": qf.DEFAULT_TD3_BUFFER_MAX_SIZE,
            "batch_size": qf.DEFAULT_TD3_BATCH_SIZE,
            "tau": qf.DEFAULT_TD3_TAU,  # Target network update rate
            "gamma": qf.DEFAULT_TD3_GAMMA,
            "train_freq": qf.DEFAULT_TD3_TRAIN_FREQ,  # Frequency of training steps
            "gradient_steps": qf.DEFAULT_TD3_GRADIENT_STEPS,  # Number of gradient steps per training iteration
            "device": qf.DEFAULT_DEVICE  # Device to run the computations on
        }

        # Merge default config with provided config
        self.config = {**default_config, **(config or {})}

        # Initialize TD3 model
        self.model = TD3(
            policy=self.config["policy"],
            env=self.env,
            learning_rate=self.config["learning_rate"],
            buffer_size=self.config["buffer_size"],
            batch_size=self.config["batch_size"],
            tau=self.config["tau"],
            gamma=self.config["gamma"],
            train_freq=self.config["train_freq"],
            gradient_steps=self.config["gradient_steps"],
            verbose=1,
            device=self.config["device"],
            #tensorboard_log=self.env.get_save_dir()  # Use the environment's save directory for TensorBoard logging
        )

        # Override the train method
        self.model.train = lambda gradient_steps, batch_size=64: train_TD3_with_TD_error_logging(self.model, gradient_steps, batch_size)


    @staticmethod
    def get_default_config():
        return qf.DEFAULT_TD3AGENT_CONFIG
    
    @staticmethod
    def get_hyperparameter_space():
        return qf.DEFAULT_TD3AGENT_HYPERPARAMETER_SPACE
    

# We are overriding the train method to be able to calculate the TD error to log it in TensorBoard.
# This is basically a copy of the original train method from stable-baselines3, but with added logging for TD error.
def train_TD3_with_TD_error_logging(self, gradient_steps: int, batch_size: int = 100) -> None:
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
            noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        # Compute TD error
        td_error = th.abs(target_q_values - current_q_values[0]).mean().item() # This line is added

        # Compute critic loss
        critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        assert isinstance(critic_loss, th.Tensor)
        critic_losses.append(critic_loss.item())

        # Optimize the critics
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
            polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

    
    # And the lines underneath are added to log the TD error and losses in TensorBoard.
    self.env.envs[0].env.env.logger.record("TRAIN_model_loss/10*log(TD_Error)", 10*np.log10(td_error), step=self._n_updates)
    if len(actor_losses) > 0:
        self.env.envs[0].env.env.logger.record("TRAIN_model_loss/actor_loss", np.mean(actor_losses), step=self._n_updates)
    self.env.envs[0].env.env.logger.record("TRAIN_model_loss/critic_loss", np.mean(critic_losses), step=self._n_updates)