import os
import random
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm

import qf as qf
from qf.agents.agent import Agent
from qf.agents.buffers.replay_buffer import ReplayBuffer
from qf.agents.config.rl_agent_config.critic_agent_config.dqn_config import DQNConfig
from qf.envs.dataclass.observation import Observation
from qf.networks.build_network import build_network_with_features


class DQNAgent(Agent):
    def __init__(self, env, config: Optional[DQNConfig] = None) -> None:
        self.config = config or DQNConfig.get_default_config()
        super().__init__(env=env, config=self.config)

        self.n_agents = 1
        if hasattr(env, "n_agents") and env.n_agents != self.n_agents:
            raise ValueError(
                f"Environment expects {env.n_agents} agents, but DQNAgent supports only 1."
            )

        self.device = self.config.device
        self.gamma = self.config.gamma
        self.batch_size = self.config.batch_size
        self.buffer_size = self.config.buffer_size
        self.epsilon_start = self.config.epsilon_start
        self.loss_fn = torch.nn.MSELoss()
        self.target_mode = self.config.target_mode

        # Netzwerke mit optionalem Feature Extractor
        critic_config = self.config.critic_config
        if critic_config.output_dim is None:
            critic_config.output_dim = self.act_dim

        self.model = build_network_with_features(self.obs_dim, critic_config).to(
            self.device
        )
        self.target_model = build_network_with_features(self.obs_dim, critic_config).to(
            self.device
        )

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.memory = ReplayBuffer(capacity=self.buffer_size)

    def act(self, state: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        if isinstance(state, Observation):
            state = state.as_tensor()
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(state)  # [1, act_dim]

            if random.random() < epsilon:
                # Uniforme Zufallsverteilung
                probs = torch.rand_like(logits)
                probs = probs / probs.sum(dim=1, keepdim=True)
            else:
                probs = torch.softmax(logits, dim=1)  # Softmax Q → Verteilung

        return probs  # .squeeze(0)  # [act_dim]

    def store(self, transition):
        self.memory.store(transition)

    def train(self, total_timesteps=5000, use_tqdm=True):
        """
        Trains the agent for a specified number of timesteps.
        Parameters:
            total_timesteps (int): Total number of timesteps to train the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print training summaries.
        """
        progress = (
            tqdm(range(total_timesteps), desc="Training DQNAgent")
            if use_tqdm
            else range(total_timesteps)
        )

        state, info = self.env.reset()
        total_reward = 0
        epsilon = self.epsilon_start  # Initial epsilon for exploration
        timestep = 0

        for _ in progress:
            # Linear epsilon decay
            epsilon = max(0.1, epsilon - (1 / total_timesteps))

            # Select action using epsilon-greedy policy
            action = self.act(state, epsilon)
            next_state, reward, done, info = self.env.step(action)

            # Store transition in replay buffer
            self.memory.store(
                (state.as_tensor(), action, reward, next_state.as_tensor(), done)
            )
            state = next_state
            total_reward += reward
            timestep += 1

            # Perform training step
            self._train_step()

            # Reset environment if done
            if done:
                state, info = self.env.reset()
                total_reward = 0

            # Update tqdm progress bar
            if use_tqdm:
                progress.set_postfix(
                    {
                        "Epsilon": epsilon,
                        "Timestep": timestep,
                        "Last Reward": f"{reward.item():.2f}",
                    }
                )

    def _train_step(self):
        if len(self.memory) < self.batch_size:
            return

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

        q_values = self.model(states)  # shape: [B, 1, n_actions]
        q_values_selected = torch.sum(q_values * actions, dim=-1)  # shape: [B, 1]

        if self.target_mode == "soft-bellman":
            with torch.no_grad():
                next_q = self.target_model(next_states)  # shape: [B, 1, n_actions]
                temperature = 1.0
                logsumexp_next_q = temperature * torch.logsumexp(
                    next_q / temperature, dim=-1
                )  # shape: [B, 1]
                target = rewards + self.gamma * logsumexp_next_q

        elif self.target_mode == "hard-bellman":
            with torch.no_grad():
                next_q = self.target_model(next_states)  # shape: [B, 1, n_actions]
                max_next_q = next_q.max(dim=-1).values  # shape: [B, 1]
                target = rewards + self.gamma * max_next_q

        loss = self.loss_fn(q_values_selected, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _save_impl(self, path):
        """
        Implementation-specific save method for DQN agent.
        Parameters:
            path (str): Path to save the agent's state.
        """
        # Save model state
        model_path = os.path.join(path, "model.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "memory": self.memory,
                "epsilon": self.epsilon_start,
                "gamma": self.gamma,
                "batch_size": self.batch_size,
                "buffer_max_size": self.buffer_size,
                "target_mode": self.target_mode,
            },
            model_path,
        )

    def _load_impl(self, path):
        """
        Implementation-specific load method for DQN agent.
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
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load other parameters
        self.memory = checkpoint["memory"]
        self.epsilon_start = checkpoint["epsilon"]
        self.gamma = checkpoint["gamma"]
        self.batch_size = checkpoint["batch_size"]
        self.buffer_max_size = checkpoint["buffer_max_size"]
        self.target_mode = checkpoint["target_mode"]

        # Move models to device
        self.model.to(self.device)
        self.target_model.to(self.device)

    @staticmethod
    def get_hyperparameter_space():
        """
        Returns the hyperparameter space for the DQN agent.
        Returns:
            dict: Hyperparameter space for the DQN agent.
        """
        return qf.DEFAULT_DQNAGENT_HYPERPARAMETER_SPACE

    @staticmethod
    def get_default_config():
        """
        Returns the default configuration for the DQN agent.
        Returns:
            dict: Default configuration for the DQN agent.
        """
        return qf.DEFAULT_DQNAGENT_CONFIG
