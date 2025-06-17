import os
import random

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import qf as qf
from qf.agents.agent import Agent
from qf.agents.buffers.replay_buffer import ReplayBuffer
from qf.agents.utils.model_builder import ModelBuilder


# Attention this is Soft-SPQL
class SPQLAgent(Agent):
    def __init__(self, env, config=None):
        super().__init__(env=env)

        default_config = {
            "learning_rate": qf.DEFAULT_SPQL_LR,
            "gamma": qf.DEFAULT_SPQL_GAMMA,
            "batch_size": qf.DEFAULT_SPQL_BATCH_SIZE,
            "buffer_max_size": qf.DEFAULT_SPQL_BUFFER_MAX_SIZE,
            "device": qf.DEFAULT_DEVICE,
            "epsilon_start": qf.DEFAULT_SPQL_EPSILON_START,
            "tau": qf.DEFAULT_SPQL_TAU,
            "temperature": qf.DEFAULT_SPQL_TEMPERATURE,
        }

        # Merge default config with provided config
        self.config = {**default_config, **(config or {})}

        # Use the provided network architecture or a default one
        default_architecture = [
            {
                "type": "Linear",
                "params": {"in_features": self.obs_dim, "out_features": 256},
                "activation": "ReLU",
            },
            {
                "type": "Linear",
                "params": {"in_features": 256, "out_features": 256},
                "activation": "ReLU",
            },
            {
                "type": "Linear",
                "params": {"in_features": 256, "out_features": 512},
                "activation": "ReLU",
            },
            {
                "type": "Linear",
                "params": {"in_features": 512, "out_features": 256},
                "activation": "ReLU",
            },
            {
                "type": "Linear",
                "params": {"in_features": 256, "out_features": 128},
                "activation": "ReLU",
            },
            {
                "type": "Linear",
                "params": {"in_features": 128, "out_features": self.act_dim},
            },
        ]
        # actor_config = actor_config or default_architecture
        actor_config = default_architecture

        # Single-agent environment setup
        self.n_agents = 1
        # Check if the environment agent settings are compatible
        if hasattr(env, "n_agents") and env.n_agents != self.n_agents:
            raise ValueError(
                f"Environment is configured for {env.n_agents} agents, but SPQLAgent is set up for {self.n_agents} agents."
            )

        self.device = self.config["device"]
        # Use ModelBuilder to create the models
        self.model = ModelBuilder(actor_config).build().to(self.device)
        self.target_model = ModelBuilder(actor_config).build().to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )
        self.gamma = self.config["gamma"]
        self.batch_size = self.config["batch_size"]
        self.buffer_max_size = self.config["buffer_max_size"]
        self.epsilon_start = self.config["epsilon_start"]
        self.loss_fn = torch.nn.MSELoss()
        self.memory = ReplayBuffer(
            capacity=self.buffer_max_size
        )  # Initialize the replay buffer
        self.tau = self.config["tau"]
        self.temperature = self.config["temperature"]

        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        state = torch.as_tensor(state, device=self.device).unsqueeze(0)
        T = self.temperature
        with torch.no_grad():
            logits = self.model(state).squeeze(0) / T
            soft_probs = torch.softmax(logits, dim=-1)

            uniform_probs = torch.ones_like(soft_probs) / soft_probs.shape[0]
            probs = (1 - epsilon) * soft_probs + epsilon * uniform_probs

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
        """
        progress = (
            tqdm(range(total_timesteps), desc="Training SPQLAgent")
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
            epsilon = max(0.1, epsilon - (1 / total_timesteps))

            # Select action using epsilon-greedy policy
            action = self.act(state, epsilon)
            next_state, reward, done, info = self.env.step(action)

            # Store transition in replay buffer
            self.memory.store((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            timestep += 1

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
                    self.env.logger.log_scalar(
                        "TRAIN_model_loss/10*log(TD_Error)",
                        10 * np.log10(np.clip(td_error, min=1e-5)),
                        timestep,
                    )

                td_error = np.mean(td_error) if td_error is not None else "N/A"
                text = f"Timestep: {timestep:010d}, Last Reward: {reward.mean().item():+015.2f}, TD Error: {td_error:+015.2f}"
            else:
                text = f"Timestep: {timestep:010d}, Last Reward: {reward.mean().item():+015.2f}, TD Error: N/A"

            if use_tqdm:
                progress.set_postfix(text=text)
            else:
                print(
                    f"Timestep: {timestep}, Last Reward: {reward} TD Error: {td_error if td_error else 'N/A'}"
                )

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
                # "optimizer_state_dict": self.optimizer.state_dict(),
                # "memory": self.memory,
                # "epsilon": self.epsilon_start,
                # "gamma": self.gamma,
                # "batch_size": self.batch_size,
                # "buffer_max_size": self.buffer_max_size,
                # "tau": self.tau,
                # "temperature": self.temperature,
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
        # self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load other parameters
        # self.memory = checkpoint["memory"]
        # self.epsilon_start = checkpoint["epsilon"]
        # self.gamma = checkpoint["gamma"]
        # self.batch_size = checkpoint["batch_size"]
        # self.buffer_max_size = checkpoint["buffer_max_size"]
        # self.tau = checkpoint["tau"]
        # self.temperature = checkpoint["temperature"]

        # Move models to device
        self.model.to(self.device)
        self.target_model.to(self.device)

    @staticmethod
    def get_hyperparameter_space():
        """
        Returns the hyperparameter space for the SPQL agent.
        Returns:
            dict: Hyperparameter space for the SPQL agent.
        """
        return qf.DEFAULT_SPQLAGENT_HYPERPARAMETER_SPACE

    @staticmethod
    def get_default_config():
        """
        Returns the default configuration for the SPQL agent.
        Returns:
            dict: Default configuration for the SPQL agent.
        """
        return qf.DEFAULT_SPQLAGENT_CONFIG
