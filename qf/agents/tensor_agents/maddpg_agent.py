import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ..utils.model_builder import ModelBuilder
from qf.utils.loss_functions.loss_functions import weighted_mse_correlation_loss
from qf.utils.correlation import compute_correlation

import qf 
from qf.agents.agent import Agent

from tqdm import tqdm

class MADDPGAgent(Agent):
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG) agent with support for custom loss functions.
    """
    def __init__(self, env, config=None):
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
        super().__init__(env)

        # Extract environment parameters
        obs_dim = env.get_observation_space().shape[0]
        act_dim = env.get_action_space().shape[0]
        n_agents = env.n_agents

        # Default configuration parameters
        default_config = {
            "learning_rate": qf.DEFAULT_MADDPG_LR,
            "gamma": qf.DEFAULT_MADDPG_GAMMA,
            "tau": qf.DEFAULT_MADDPG_TAU,
            "verbosity": qf.DEFAULT_MADDPG_VERBOSITY,
            "batch_size": qf.DEFAULT_MADDPG_BATCH_SIZE,
            "loss_function": qf.DEFAULT_MADDPG_LOSS_FN,  # Default to None, will use nn.MSELoss if not provided
            "lambda_": qf.DEFAULT_MADDPG_LAMBDA,  # Custom weighting factor for the loss function
            "buffer_max_size": qf.DEFAULT_MADDPG_BUFFER_MAX_SIZE
        }

        self.config = {**default_config, **(config or {})}

        self.batch_size = self.config["batch_size"]
        self.n_agents = n_agents
        self.gamma = self.config["gamma"]
        self.tau = self.config["tau"]
        self.verbosity = self.config["verbosity"]
        self.lambda_ = self.config["lambda_"]
        self.buffer_max_size = self.config["buffer_max_size"]
        self.lr = self.config["learning_rate"]

        # Use the provided loss function or default to MSELoss
        self.loss_function = self.config["loss_function"]

        if self.loss_function == "weighted_mse_correlation":
            self.loss_function = weighted_mse_correlation_loss
        elif self.loss_function == "mse":
            self.loss_function = nn.MSELoss()
        elif not callable(self.loss_function):
            raise ValueError(f"Unsupported loss function: {self.loss_function}. Must be a callable or 'weighted_mse_correlation' or 'mse'.")

        # Dynamically generate the default actor and critic configs based on obs_dim and act_dim
        default_actor_config = [
            {"type": "Linear", "params": {"in_features": obs_dim, "out_features": 128}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 128, "out_features": 64}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 64, "out_features": act_dim}}
        ]

        default_critic_config = [
            {"type": "Linear", "params": {"in_features": obs_dim * n_agents + act_dim * n_agents, "out_features": 128}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 128, "out_features": 64}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 64, "out_features": 1}}
        ]

        # Use the default configs if no custom configs are provided
        actor_config = default_actor_config
        critic_config = default_critic_config

        # We need to append a cliping and a softmax layer to the actor config
        # This is done to ensure that the actions are in the range [0, 1]
        #actor_config.append({"type": "clip", "params": {"min": -10, "max": 10}})
        #actor_config.append({"type": "softmax", "params": {}})

        # Dynamically replace placeholders in the network architecture
        actor_config = self._replace_placeholders(actor_config, obs_dim, act_dim, n_agents)
        critic_config = self._replace_placeholders(critic_config, obs_dim, act_dim, n_agents)

        # Create actor and critic networks
        self.actors = [ModelBuilder(actor_config).build() for _ in range(n_agents)]
        self.critics = [ModelBuilder(critic_config).build() for _ in range(n_agents)]

        # Create target networks
        self.target_actors = [ModelBuilder(actor_config).build() for _ in range(n_agents)]
        self.target_critics = [ModelBuilder(critic_config).build() for _ in range(n_agents)]

        # Optimizers
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=self.lr) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=self.lr) for critic in self.critics]

        # Initialize replay memory
        self.memory = []

        # Initialize target networks with the same weights as the original networks
        for i in range(n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        if self.verbosity > 0:
            print(f"MADDPGAgent initialized with {self.n_agents} agents.")

    def set_env_mode(self):
        return "gym"

    def _replace_placeholders(self, config, obs_dim, act_dim, n_agents):
        """
        Replace placeholders in the network architecture with actual dimensions.

        Parameters:
            config (list): Network architecture configuration.
            obs_dim (int): Observation space dimension.
            act_dim (int): Action space dimension.
            n_agents (int): Number of agents.

        Returns:
            list: Updated network architecture configuration.
        """

        return config

    def act(self, states: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        """
        Select actions for each agent based on the current policy (actor network) with epsilon-greedy exploration.

        Parameters:
            states (torch.Tensor): A tensor of states for all agents (shape: [n_agents, obs_dim]).
            epsilon (float): Probability of selecting a random action.

        Returns:
            actions (torch.Tensor): A tensor of normalized actions for all agents (shape: [n_agents, act_dim]).
        """
        actions = []

        for i, actor in enumerate(self.actors):
            state = states[i] # Add batch dimension (shape: [1, obs_dim])

            with torch.no_grad():
                logits = actor(state)  # Remove batch dimension (shape: [act_dim])

            # add simple epsilon decaing noise
            act_dim = logits.shape[0]
            action = logits + torch.FloatTensor(act_dim).normal_(-epsilon, epsilon)

            logits = F.softmax(logits, dim=-1)  # Normalize the logits to [0, 1]
            actions.append(logits)

            if self.verbosity > 1:
                print(f"Agent {i} action (normalized): {logits}")

        # Stack actions into a single tensor (shape: [n_agents, act_dim])
        return torch.stack(actions)

    def store(self, transition):
        """
        Store a transition in the replay memory.

        Parameters:
            transition (tuple): A tuple containing (states, actions, rewards, next_states).
        """
        self.memory.append(transition)
        if len(self.memory) > self.buffer_max_size:
            self.memory.pop(0)
        if self.verbosity > 1:
            print(f"Stored transition. Memory size: {len(self.memory)}")

    def train(self, total_timesteps=5000, use_tqdm=True):
        """
        Train the MADDPG agent for a specified number of timesteps.

        Parameters:
            total_timesteps (int): Total number of timesteps to train the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print training summaries.
        """
        progress = tqdm(range(total_timesteps), desc="Training MADDPGAgent") if use_tqdm else range(total_timesteps)

        state, _ = self.env.reset()
        total_reward = 0
        timestep = 0

        for _ in progress:
            # Select actions for each agent
            actions = self.act(state)

            # Step the environment
            next_state, rewards, done, _, _ = self.env.step(actions)

            # Store transition in replay memory
            self.store((state, actions, rewards, next_state, done))

            state = next_state
            total_reward += sum(rewards)
            timestep += 1

            # Train the agents
            self._train()

            # Reset environment if done
            if done:
                state, _ = self.env.reset()
                total_reward = 0

            # Update tqdm progress bar
            if use_tqdm:
                progress.set_postfix({"Timestep": timestep, "Last Reward": f"{rewards}"})
            else:
                print(f"Timestep: {timestep}, Last Reward: {rewards}")


    def _train(self):
        """
        Train the agents by sampling a batch of transitions from the replay memory.
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions from the replay memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))

        with torch.no_grad():
            # Get the actions for the next states from the target actors
            next_actions = [self.target_actors[j](next_states[:, j, :]) for j in range(self.n_agents)]

            # Normalize each agent's actions
            next_actions = [F.softmax(action, dim=-1) for action in next_actions]

            # Concatenate normalized actions -> Important removes the list
            next_actions = torch.cat(next_actions, dim=-1)

            # Concatenate next_states and next_actions
            next_inputs = torch.cat([next_states.view(self.batch_size, -1), next_actions], dim=-1)

        for i in range(self.n_agents):
            # Update critic

            with torch.no_grad():
                # Compute the target Q value
                target_q = rewards[:, i] + self.gamma * self.target_critics[i](next_inputs).squeeze()


            # Concatenate states and actions for the current Q value 
            # We view here to flatten the tensor because every critic sees the whole state-action space of all agents
            current_inputs = torch.cat([states.view(self.batch_size, -1), actions.view(self.batch_size, -1)], dim=-1)

            # Compute the current Q value
            current_q = self.critics[i](current_inputs).squeeze()

            # Compute the correlation penalty
            with torch.no_grad():
                correlation_penality = 0.0
                for j in range(self.n_agents):
                    if j != i:
                        correlation_penality += compute_correlation(actions[:, i], actions[:, j])

            # Compute the critic loss using the custom loss function
            critic_loss = self.lambda_*self.loss_function(current_q, target_q) #+ (1-self.lambda_)*per_sample_correlation_penalty(actions, i)

            # Optimize the critic
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            if self.verbosity > 0:
                print(f"Agent {i} critic loss: {critic_loss.item()}")

            # Update actor
            current_actions = [self.actors[j](states[:, j, :]) if j == i else actions[:, j, :] for j in range(self.n_agents)]
            current_actions = torch.cat(current_actions, dim=1)

            # Concatenate states and current actions
            actor_inputs = torch.cat([states.view(self.batch_size, -1), current_actions], dim=-1)

            # Compute the actor loss
            actor_loss = -self.critics[i](actor_inputs).mean()

            # Optimize the actor
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            if self.verbosity > 0:
                print(f"Agent {i} actor loss: {actor_loss.item()}")

            # Update target networks
            for target_param, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.verbosity > 0:
                print(f"Agent {i} target networks updated.")

    def save(self, save_path):
        """
        Save the MADDPG agent's actor and critic networks to the specified path.

        Parameters:
            save_path (str): Path to save the model files.
        """
        save_data = {
            "actors": [actor.state_dict() for actor in self.actors],
            "critics": [critic.state_dict() for critic in self.critics],
            "target_actors": [target_actor.state_dict() for target_actor in self.target_actors],
            "target_critics": [target_critic.state_dict() for target_critic in self.target_critics],
            "actor_optimizers": [optimizer.state_dict() for optimizer in self.actor_optimizers],
            "critic_optimizers": [optimizer.state_dict() for optimizer in self.critic_optimizers],
        }
        torch.save(save_data, save_path)
        if self.verbosity > 0:
            print(f"MADDPG agent saved to {save_path}")

    def load(self, load_path):
        """
        Load the MADDPG agent's actor and critic networks from the specified path.

        Parameters:
            load_path (str): Path to the saved model file.
        """
        checkpoint = torch.load(load_path)

        # Load actor and critic networks
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint["actors"][i])
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(checkpoint["critics"][i])

        # Load target actor and critic networks
        for i, target_actor in enumerate(self.target_actors):
            target_actor.load_state_dict(checkpoint["target_actors"][i])
        for i, target_critic in enumerate(self.target_critics):
            target_critic.load_state_dict(checkpoint["target_critics"][i])

        # Load optimizers
        for i, optimizer in enumerate(self.actor_optimizers):
            optimizer.load_state_dict(checkpoint["actor_optimizers"][i])
        for i, optimizer in enumerate(self.critic_optimizers):
            optimizer.load_state_dict(checkpoint["critic_optimizers"][i])

        if self.verbosity > 0:
            print(f"MADDPG agent loaded from {load_path}")


