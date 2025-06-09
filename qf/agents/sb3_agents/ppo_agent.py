from stable_baselines3 import PPO
from qf.agents.sb3_agents.sb3_agent import SB3Agent
import qf
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.utils import explained_variance
import numpy as np
from gymnasium import spaces

class PPOAgent(SB3Agent):
    def __init__(self, env, config=None):
        """
        Initializes the PPO agent with the given environment and configuration.
        Parameters:
            env: The environment in which the agent will operate.
            config (dict): Configuration dictionary for the PPO agent.
        """
        super().__init__(env)

        # Default configuration
        default_config = {
            "policy": qf.DEFAULT_PPO_POLICY,
            "learning_rate": qf.DEFAULT_PPO_LR,
            "n_steps": qf.DEFAULT_PPO_N_STEPS,
            "batch_size": qf.DEFAULT_PPO_BATCH_SIZE,
            "gamma": qf.DEFAULT_PPO_GAMMA,
            "gae_lambda": qf.DEFAULT_PPO_GAE_LAMBDA,
            "clip_range": qf.DEFAULT_PPO_CLIP_RANGE,
            "device": qf.DEFAULT_DEVICE,
            "verbose": qf.DEFAULT_PPO_VERBOSITY
        }

        # Merge default config with provided config
        self.config = {**default_config, **(config or {})}

        # Initialize PPO model
        self.model = PPO(
            policy=self.config["policy"],
            env=self.env,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_range"],
            verbose=self.config["verbose"],
            device=self.config["device"]
            #tensorboard_log=self.env.get_save_dir()  # Use the environment's save directory for TensorBoard logging
        )
        # Override the train method
        # Override the train method
        self.model.train = lambda : train_PPO_with_TD_error_logging(self.model)


    @staticmethod
    def get_default_config():
        return qf.DEFAULT_PPOAGENT_CONFIG
    
    @staticmethod
    def get_hyperparameter_space():
        return qf.DEFAULT_PPOAGENT_HYPERPARAMETER_SPACE
    

def train_PPO_with_TD_error_logging(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        td_errors = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                
                # Calculate TD error
                td_error = th.abs(rollout_data.returns - values).mean().item()
                td_errors.append(td_error)
                
                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.env.envs[0].env.env.logger.record("TRAIN_model_loss/entropy_loss", np.mean(entropy_losses), step=self._n_updates)
        self.env.envs[0].env.env.logger.record("TRAIN_model_loss/policy_gradient_loss", np.mean(pg_losses), step=self._n_updates)
        self.env.envs[0].env.env.logger.record("TRAIN_model_loss/value_loss", np.mean(value_losses), step=self._n_updates)
        self.env.envs[0].env.env.logger.record("TRAIN_model_loss/10*log(TD_Error)", 10 * np.log10(np.mean(td_errors)), step=self._n_updates*self.n_epochs)
        self.env.envs[0].env.env.logger.record("TRAIN_model_loss/approx_kl", np.mean(approx_kl_divs), step=self._n_updates)
        self.env.envs[0].env.env.logger.record("TRAIN_model_loss/clip_fraction", np.mean(clip_fractions), step=self._n_updates)
        self.env.envs[0].env.env.logger.record("TRAIN_model_loss/loss", loss.item(), step=self._n_updates)
        self.env.envs[0].env.env.logger.record("TRAIN_model_loss/explained_variance", explained_var, step=self._n_updates)
        if hasattr(self.policy, "log_std"):
            self.env.envs[0].env.env.logger.record("TRAIN_model_loss/std", th.exp(self.policy.log_std).mean().item(), step=self._n_updates)

        self.env.envs[0].env.env.logger.record("TRAIN_model_loss/n_updates", self._n_updates, step=self._n_updates)
        self.env.envs[0].env.env.logger.record("TRAIN_model_loss/clip_range", clip_range, step=self._n_updates)
        if self.clip_range_vf is not None:
            self.env.envs[0].env.env.logger.record("TRAIN_model_loss/clip_range_vf", clip_range_vf, step=self._n_updates)