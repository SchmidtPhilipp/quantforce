import os
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from qf.agents.agent import Agent
from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.off_policy_agent.maddpg_config import (
    MADDPGConfig,
)
from qf.agents.noise.ornstein_uhlenbeck_noise import OrnsteinUhlenbeckNoise
from qf.envs.dataclass.observation import Observation
from qf.networks.build_network import build_network
from qf.utils.logging_config import get_logger
from qf.utils.loss_functions.loss_functions import weighted_mse_correlation_loss
from qf.utils.math import correlation

logger = get_logger(__name__)


class MADDPGAgent(Agent):
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG) agent with support for custom loss functions.
    """

    def __init__(self, env, config: Optional[MADDPGConfig] = None):
        """
        Initialize the MADDPG agent.

        Parameters:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            n_agents (int): Number of agents.
            lr (float): Learning rate for the optimizers.
            gamma (float): Discount factor for future rewards.
            tau (float): Soft update parameter for target networks.
            verbosity (int): Verbosity level for logging.
            batch_size (int): Batch size for training.
            loss_function (callable): Custom loss function for training the critic. Defaults to nn.MSELoss.
        """

        self.config = config or MADDPGConfig.get_default_config()
        super().__init__(env, config=self.config)

        # Extract environment parameters
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        n_agents = env.n_agents

        self.batch_size = self.config.batch_size
        self.n_agents = n_agents
        self.gamma = self.config.gamma
        self.tau = self.config.tau
        self.verbosity = self.verbosity
        self.lambda_ = self.config.lambda_
        self.buffer_max_size = self.config.buffer_size
        self.lr = self.config.learning_rate
        self.global_step = 0

        # Use the provided loss function or default to MSELoss
        self.loss_function = self.config.loss_fn

        if self.loss_function == "weighted_mse_correlation":
            self.loss_function = weighted_mse_correlation_loss
        elif self.loss_function == "mse":
            self.loss_function = nn.MSELoss()
        elif not callable(self.loss_function):
            raise ValueError(
                f"Unsupported loss function: {self.loss_function}. Must be a callable or 'weighted_mse_correlation' or 'mse'."
            )

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        n_agents = env.n_agents

        # Set output_dim falls nicht gesetzt
        if self.config.actor_config.output_dim is None:
            self.config.actor_config.output_dim = act_dim

        if self.config.critic_config.output_dim is None:
            self.config.critic_config.output_dim = 1

        # Optional: critic input dim anpassen (state + actions für alle Agenten)
        critic_input_dim = obs_dim * n_agents + act_dim * n_agents

        # Netzwerke bauen
        self.actors = [
            build_network(obs_dim, self.config.actor_config).to(self.config.device)
            for _ in range(n_agents)
        ]

        self.critics = [
            build_network(critic_input_dim, self.config.critic_config).to(
                self.config.device
            )
            for _ in range(n_agents)
        ]

        self.target_actors = [
            build_network(obs_dim, self.config.actor_config).to(self.config.device)
            for _ in range(n_agents)
        ]

        self.target_critics = [
            build_network(critic_input_dim, self.config.critic_config).to(
                self.config.device
            )
            for _ in range(n_agents)
        ]

        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=self.lr) for actor in self.actors
        ]
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=self.lr) for critic in self.critics
        ]

        # Initialize replay memory using deque for efficient operations
        self.memory = deque(maxlen=self.buffer_max_size)

        # Initialize target networks with the same weights as the original networks
        for i in range(n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        # Initialize Ornstein-Uhlenbeck noise for each agent
        self.ou_noises = [
            OrnsteinUhlenbeckNoise(
                size=act_dim,
                mu=self.config.ou_mu,
                theta=self.config.ou_theta,
                sigma=self.config.ou_sigma,
                dt=self.config.ou_dt,
            )
            for _ in range(n_agents)
        ]

        logger.info(
            f"MADDPGAgent initialized with {self.n_agents} agents and replay memory size {self.buffer_max_size}."
        )

    def act(
        self, states: torch.Tensor, epsilon: float = 0.0, use_ou_noise=False
    ) -> torch.Tensor:
        """
        Select actions for each agent based on the current policy (actor network) with optional OU noise.

        Parameters:
            states (torch.Tensor): A tensor of states for all agents (shape: [n_agents, obs_dim]).
            epsilon (float): Probability of selecting a random action.
            use_ou_noise (bool): Whether to use Ornstein-Uhlenbeck noise for exploration.

        Returns:
            actions (torch.Tensor): A tensor of normalized actions for all agents (shape: [n_agents, act_dim]).
        """
        if isinstance(states, Observation):
            states = states.as_tensor()
        else:
            states = torch.tensor(states, dtype=torch.float32, device=self.device)

        actions = []

        for i, actor in enumerate(self.actors):
            state = states[i]  # (shape: [obs_dim])

            # Convert numpy array to tensor if necessary
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)

            with torch.no_grad():
                logits = actor(state)  # (shape: [act_dim])

                # Add Ornstein-Uhlenbeck noise or epsilon-decaying noise
                if use_ou_noise:
                    noise = torch.FloatTensor(self.ou_noises[i].sample())  # OU noise
                else:
                    act_dim = logits.shape[0]
                    noise = torch.FloatTensor(act_dim).normal_(
                        -epsilon, epsilon
                    )  # Epsilon-decaying noise

                noisy_logits = logits + noise
                action = F.softmax(noisy_logits, dim=-1)

                actions.append(action)

            logger.debug(f"Agent {i} action (normalized): {logits}")

        # Stack actions into a single tensor (shape: [n_agents, act_dim])
        return torch.stack(actions)  # shape: [n_agents, act_dim]

    def store(self, transition):
        """
        Store a transition in the replay memory.

        Parameters:
            transition (tuple): A tuple containing (states, actions, rewards, next_states, dones).
        """
        self.memory.append(transition)  # Efficiently adds to the deque
        logger.debug(f"Stored transition. Memory size: {len(self.memory)}")

    def _train(
        self,
        total_timesteps=5000,
        use_tqdm=True,
    ):
        """
        Train the MADDPG agent for a specified number of timesteps.

        Parameters:
            total_timesteps (int): Total number of timesteps to train the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print training summaries.
            eval_env: Optional environment for evaluation during training.
            eval_every_n_steps (int): Number of training steps between evaluations. If None, no evaluation is performed.
            save_best (bool): If True, saves the best performing agent based on evaluation.
        """
        progress = (
            tqdm(range(total_timesteps), desc="Training MADDPGAgent")
            if use_tqdm
            else range(total_timesteps)
        )

        state, _ = self.env.reset()
        self.reset_noise()
        total_reward = 0
        timestep = 0

        for _ in progress:
            actions = self.act(state, use_ou_noise=True)
            next_state, rewards, done, _ = self.env.step(actions)
            self.store(
                (state.as_tensor(), actions, rewards, next_state.as_tensor(), done)
            )
            state = next_state
            total_reward += sum(rewards)

            td_error = self._train_step()

            if done:
                state, _ = self.env.reset()
                self.reset_noise()
                total_reward = 0

                if use_tqdm:
                    progress.set_postfix(
                        {
                            "Reward": f"{total_reward:.2f}",
                        }
                    )

            if td_error is not None:
                for i in range(self.n_agents):
                    self.env.experiment_logger.log_scalar(
                        "TRAIN_model_loss/10*log(TD_Error)",
                        10 * np.log10(np.clip(td_error[i], a_min=1e-5, a_max=1e5)),
                        step=self.global_step,
                    )

                td_error = np.mean(td_error) if td_error is not None else "N/A"
                text = f"Timestep: {timestep:010d}, Last Reward: {total_reward:.2f}, TD Error: {td_error:+015.2f}"
            else:
                text = f"Timestep: {timestep:010d}, Last Reward: {total_reward:.2f}, TD Error: N/A"

            if use_tqdm:
                progress.set_postfix(text=text)
            else:
                logger.info(
                    f"Timestep: {timestep}, Last Reward: {total_reward} TD Error: {td_error if td_error else 'N/A'}"
                )

            timestep += 1
            self.global_step += 1

    def _train_step(self):
        """
        Train the agents by sampling a batch of transitions from the replay memory.
        Returns:
            list: TD errors for each agent.
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample a batch of transitions from the replay memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(
            np.array(states)
        )  # shape: [batch_size, n_agents, obs_dim]
        actions = torch.FloatTensor(
            np.array(actions)
        )  # shape: [batch_size, n_agents, act_dim]
        rewards = torch.FloatTensor(np.array(rewards))  # shape: [batch_size, n_agents]
        next_states = torch.FloatTensor(
            np.array(next_states)
        )  # shape: [batch_size, n_agents, obs_dim]

        td_errors = []  # List to store TD errors for each agent

        with torch.no_grad():
            # Get the actions for the next states from the target actors
            next_actions = [
                self.target_actors[j](next_states[:, j, :])
                for j in range(self.n_agents)
            ]  # shape: [batch_size, n_agents, act_dim]

            # Normalize each agent's actions
            next_actions = [F.softmax(action, dim=-1) for action in next_actions]

            # Concatenate normalized actions -> Important removes the list
            next_actions = torch.cat(
                next_actions, dim=-1
            )  # shape: [batch_size, n_agents * act_dim]

            # Concatenate next_states and next_actions
            next_inputs = torch.cat(
                [next_states.view(self.batch_size, -1), next_actions], dim=-1
            )  # shape: [batch_size, n_agents * obs_dim + n_agents * act_dim]

        for i in range(self.n_agents):
            # Update critic
            with torch.no_grad():
                # Compute the target Q value
                target_q = (
                    rewards[:, i]
                    + self.gamma * self.target_critics[i](next_inputs).squeeze()
                )  # shape: [batch_size]

            # Concatenate states and actions for the current Q value
            current_inputs = torch.cat(
                [states.view(self.batch_size, -1), actions.view(self.batch_size, -1)],
                dim=-1,
            )  # shape: [batch_size, n_agents * obs_dim + n_agents * act_dim]

            # Compute the current Q value
            current_q = self.critics[i](current_inputs).squeeze()  # shape: [batch_size]

            # Compute the correlation penalty
            with torch.no_grad():
                correlation_penality = 0.0
                for j in range(self.n_agents):
                    if j != i:
                        correlation_penality += correlation(
                            actions[:, i], actions[:, j]
                        )

            # Compute the critic loss using the custom loss function
            critic_loss = (
                self.lambda_ * self.loss_function(current_q, target_q)
                + (1 - self.lambda_) * correlation_penality
            )

            # Optimize the critic
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            logger.debug(f"Agent {i} critic loss: {critic_loss.item()}")

            # Compute TD error for the agent
            td_error = torch.abs(target_q - current_q).mean().item()
            td_errors.append(td_error)

            # Log TD error to tensorboard
            self.env.experiment_logger.log_scalar(
                "TRAIN_model_loss/10*log(TD_Error)",
                10 * np.log10(np.clip(td_error, a_min=1e-5, a_max=1e5)),
                step=self.global_step,
            )

            # Update actor
            current_actions = [
                self.actors[j](states[:, j, :]) if j == i else actions[:, j, :]
                for j in range(self.n_agents)
            ]
            current_actions = torch.cat(current_actions, dim=1)

            # Concatenate states and current actions
            actor_inputs = torch.cat(
                [states.view(self.batch_size, -1), current_actions], dim=-1
            )

            # Compute the actor loss
            actor_loss = -self.critics[i](actor_inputs).mean()

            # Optimize the actor
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            logger.debug(f"Agent {i} actor loss: {actor_loss.item()}")

            # Update target networks
            for target_param, param in zip(
                self.target_actors[i].parameters(), self.actors[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for target_param, param in zip(
                self.target_critics[i].parameters(), self.critics[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            logger.debug(f"Agent {i} target networks updated.")

        return np.array(td_errors)

    def _save_impl(self, path):
        """
        Implementation-specific save method for MADDPG agent.
        Parameters:
            path (str): Path to save the agent's state.
        """
        import os

        import torch

        # Save model state
        model_path = os.path.join(path, "model.pt")
        save_data = {
            "actors": [actor.state_dict() for actor in self.actors],
            "critics": [critic.state_dict() for critic in self.critics],
            "target_actors": [
                target_actor.state_dict() for target_actor in self.target_actors
            ],
            "target_critics": [
                target_critic.state_dict() for target_critic in self.target_critics
            ],
            "actor_optimizers": [
                optimizer.state_dict() for optimizer in self.actor_optimizers
            ],
            "critic_optimizers": [
                optimizer.state_dict() for optimizer in self.critic_optimizers
            ],
            "config": self.config,
            "verbosity": self.verbosity,
            "tau": self.tau,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "loss_fn": (
                self.loss_function.__name__
                if hasattr(self.loss_function, "__name__")
                else str(self.loss_function)
            ),
        }
        torch.save(save_data, model_path)

    def _load_impl(self, path):
        """
        Implementation-specific load method for MADDPG agent.
        Parameters:
            path (str): Path to load the agent's state from.
        """
        import os

        import torch

        # Load model state
        model_path = os.path.join(path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        checkpoint = torch.load(model_path, weights_only=False)

        # Load model states
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint["actors"][i])
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(checkpoint["critics"][i])
        for i, target_actor in enumerate(self.target_actors):
            target_actor.load_state_dict(checkpoint["target_actors"][i])
        for i, target_critic in enumerate(self.target_critics):
            target_critic.load_state_dict(checkpoint["target_critics"][i])

        # Load optimizers
        for i, optimizer in enumerate(self.actor_optimizers):
            optimizer.load_state_dict(checkpoint["actor_optimizers"][i])
        for i, optimizer in enumerate(self.critic_optimizers):
            optimizer.load_state_dict(checkpoint["critic_optimizers"][i])

        # Load other parameters
        self.config = checkpoint["config"]
        self.verbosity = checkpoint["verbosity"]
        self.tau = checkpoint["tau"]
        self.gamma = checkpoint["gamma"]
        self.batch_size = checkpoint["batch_size"]

        # Restore loss function
        loss_fn_name = checkpoint["loss_fn"]
        if loss_fn_name == "weighted_mse_correlation_loss":
            self.loss_function = weighted_mse_correlation_loss
        else:
            self.loss_function = torch.nn.MSELoss()

        # Move models to device
        for actor in self.actors:
            actor.to(self.device)
        for critic in self.critics:
            critic.to(self.device)
        for target_actor in self.target_actors:
            target_actor.to(self.device)
        for target_critic in self.target_critics:
            target_critic.to(self.device)

    def reset_noise(self):
        """
        Reset the Ornstein-Uhlenbeck noise for all agents.
        """
        for noise in self.ou_noises:
            noise.reset()

    @staticmethod
    def get_default_config():
        return MADDPGConfig.get_default_config()

    @staticmethod
    def get_hyperparameter_space():
        return MADDPGConfig.get_hyperparameter_space()
