import os
import random
import sys
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm

import qf as qf
from qf.agents.agent import Agent
from qf.agents.buffers.replay_buffer import ReplayBuffer
from qf.agents.config.rl_agent_config.critic_agent_config.spql_config import SPQLConfig
from qf.envs.dataclass.observation import Observation
from qf.networks.build_network import build_network_with_features
from qf.utils.logging_config import get_logger

logger = get_logger(__name__)


class SPQLAgent(Agent):
    def __init__(self, env, config: Optional[SPQLConfig] = None):
        self.config = config or SPQLConfig.get_default_config()
        super().__init__(env=env, config=self.config)

        self.n_agents = 1
        if hasattr(env, "n_agents") and env.n_agents != self.n_agents:
            raise ValueError(
                f"Environment is configured for {env.n_agents} agents, "
                f"but SPQLAgent is set up for {self.n_agents} agent(s)."
            )

        self.device = self.config.device
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # In spql we do not need the critic config because we use a linear critic network.
        # Only the actor is modeled using a neural network.

        # Set output_dim if not already set in config
        actor_config = self.config.actor_config
        if actor_config.output_dim is None:
            actor_config.output_dim = self.act_dim

        self.model = build_network_with_features(self.obs_dim, actor_config).to(
            self.device
        )
        self.target_model = build_network_with_features(self.obs_dim, actor_config).to(
            self.device
        )

        # Optimizer & other training parameters
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.gamma = self.config.gamma
        self.batch_size = self.config.batch_size
        self.buffer_max_size = self.config.buffer_size
        self.epsilon_start = self.config.epsilon_start
        self.loss_fn = torch.nn.MSELoss()
        self.tau = self.config.tau
        self.temperature = self.config.temperature
        self.backup_mode = self.config.backup_mode
        self.global_step = 0

        # Initialize replay buffer
        self.memory = ReplayBuffer(capacity=self.buffer_max_size)

    def act(self, state: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        if isinstance(state, Observation):
            state = state.as_tensor()
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)

        state = state.unsqueeze(0)
        with torch.no_grad():
            logits = self.model(state).squeeze(0) / self.temperature
            soft_probs = torch.softmax(logits, dim=-1)

            uniform_probs = torch.ones_like(soft_probs) / soft_probs.shape[0]
            probs = (1 - epsilon) * soft_probs + epsilon * uniform_probs

            probs = probs / probs.sum(dim=-1, keepdim=True)

        return probs  # shape: [act_dim]

    def store(self, transition):
        self.memory.store(transition)

    def _train(
        self, total_timesteps=500_000, use_tqdm=True, patience=10_000, min_delta=1e-3
    ):
        """
        Trains the agent for a specified number of timesteps.
        Parameters:
            total_timesteps (int): Total number of timesteps to train the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print training summaries.
            patience (int): Number of steps without improvement before stopping.
            min_delta (float): Minimum improvement required to reset patience counter.
        """
        progress = (
            tqdm(range(total_timesteps), desc="Training SPQLAgent", file=sys.__stderr__)
            if use_tqdm
            else range(total_timesteps)
        )

        state, info = self.env.reset()
        total_reward = 0
        epsilon = self.epsilon_start  # Initial epsilon for exploration
        timestep = 0

        best_td_error = float("inf")  # Best TD error observed
        no_improvement_steps = 0  # Counter for steps without improvement

        for _ in progress:
            # Linear epsilon decay
            epsilon = max(0.1, epsilon - (1 / self.total_timesteps))

            # Select action using epsilon-greedy policy
            action = self.act(state, epsilon)
            next_state, reward, done, info = self.env.step(action)

            # Store transition in replay buffer
            self.memory.store(
                (state.as_tensor(), action, reward, next_state.as_tensor(), done)
            )
            state = next_state
            total_reward += reward

            # Perform training step and calculate TD error
            td_error = self._train_step()

            # Reset environment if done
            if done:
                state, info = self.env.reset()
                total_reward = 0

            # Report the td_error in the env logger
            if td_error is not None:

                # Check for early stopping based on TD error
                if td_error < best_td_error - min_delta:
                    best_td_error = td_error
                    no_improvement_steps = 0  # Reset patience counter
                else:
                    no_improvement_steps += 1

                # Log TD error for each agent
                for i in range(self.n_agents):
                    self.env.experiment_logger.log_scalar(
                        "TRAIN_TD_Error/10*log(TD_Error)",
                        10 * np.log10(np.abs(td_error)),
                        self.global_step,
                    )

                td_error = np.mean(td_error) if td_error is not None else "N/A"
                text = f"Timestep: {timestep:010d}, Last Reward: {reward.mean().item():+015.2f}, TD Error: {td_error:+015.2f}"
            else:
                text = f"Timestep: {timestep:010d}, Last Reward: {reward.mean().item():+015.2f}, TD Error: N/A"

            if use_tqdm:
                progress.set_postfix(text=text)
            else:
                logger.info(
                    f"Timestep: {timestep}, Last Reward: {reward} TD Error: {td_error if td_error else 'N/A'}"
                )

            # Increment step counters
            timestep += 1
            self.global_step += 1

        # Send the final tqdm progress bar to the logger
        if use_tqdm:
            progress.close()
            logger.info(f"Training SPQLAgent completed after {timestep} timesteps")
            logger.info(text)

        # End of training save data
        self.env.save_data()

        return True

    def _train_step(self):
        """
        Perform a single training step and calculate TD error.

        Returns:
            float: TD error for the current training step.
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample a batch from memory
        states, actions, rewards, next_states, _ = self.memory.sample(self.batch_size)

        # Ensure states, actions, rewards, and next_states are tensors
        if isinstance(states[0], torch.Tensor):
            states = torch.stack(states).to(self.device)
            actions = torch.stack(actions).to(self.device)
            rewards = torch.stack(rewards).to(self.device)
            next_states = torch.stack(next_states).to(self.device)
        else:
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(np.array(actions)).to(self.device)
            rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

        # Compute Q-values for current states and actions
        q_values = self.model(states)  # shape: [batch_size, act_dim]
        q_values_selected = torch.sum(q_values * actions, dim=-1)  # shape: [batch_size]

        if self.backup_mode == "LSE":
            with torch.no_grad():
                # Compute target Q-values using the target model
                next_q_values = self.target_model(
                    next_states
                )  # shape: [batch_size, act_dim]
                next_q_values_selected = torch.sum(
                    next_q_values * actions, dim=-1
                )  # shape: [batch_size]
                target = (
                    rewards + self.gamma * next_q_values_selected
                )  # shape: [batch_size]

        else:  # Use soft Bellman update
            with torch.no_grad():
                # Compute target Q-values using the target model
                next_q = self.target_model(next_states)  # shape: [batch_size, act_dim]
                logsumexp_next_q = self.temperature * torch.logsumexp(
                    next_q / self.temperature, dim=-1
                )  # shape: [batch_size]
                target = rewards + self.gamma * logsumexp_next_q  # shape: [batch_size]

        # Compute TD error
        td_error = torch.abs(target - q_values_selected).mean().item()

        # Compute loss and optimize the model
        loss = self.loss_fn(q_values_selected, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=10.0
        )  # Max norm 5-10
        self.optimizer.step()

        self.soft_update()

        return td_error

    def soft_update(self):
        for target_param, param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def _save_impl(self, path):
        """
        Implementation-specific save method for SPQL agent.
        Parameters:
            path (str): Path to save the agent's state.
        """
        # Save model state
        model_path = os.path.join(path, "model.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
            },
            model_path,
        )

    def _load_impl(self, path):
        """
        Implementation-specific load method for SPQL agent.
        Parameters:
            path (str): Path to load the agent's state from.
        """
        # Load model state
        model_path = os.path.join(path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Load model states
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])

        # Move models to device
        self.model.to(self.device)
        self.target_model.to(self.device)

    @staticmethod
    def get_hyperparameter_space(trial):
        """
        Returns the hyperparameter space for the SPQL agent.
        Returns:
            dict: Hyperparameter space for the SPQL agent.
        """
        return SPQLConfig.get_hyperparameter_space(trial)

    @staticmethod
    def get_default_config():
        """
        Returns the default configuration for the SPQL agent.
        Returns:
            dict: Default configuration for the SPQL agent.
        """
        return SPQLConfig.get_default_config()
