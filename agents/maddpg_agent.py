import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from agents.model_builder import ModelBuilder

class MADDPGAgent:
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG) agent.
    This class implements the MADDPG algorithm for multiple agents.

    Attributes:
        n_agents (int): Number of agents.
        gamma (float): Discount factor for future rewards.
        tau (float): Soft update parameter for target networks.
        verbosity (int): Verbosity level for logging.
        actors (list): List of actor networks for each agent.
        critics (list): List of critic networks for each agent.
        target_actors (list): List of target actor networks for each agent.
        target_critics (list): List of target critic networks for each agent.
        actor_optimizers (list): List of optimizers for actor networks.
        critic_optimizers (list): List of optimizers for critic networks.
        memory (list): Replay memory for storing transitions.
    """
    def __init__(self, obs_dim, act_dim, n_agents, actor_config=None, critic_config=None, lr=1e-3, gamma=0.99, tau=0.01, verbosity=0):
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
        """
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.verbosity = verbosity

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
        actor_config = actor_config or default_actor_config
        critic_config = critic_config or default_critic_config

        # Create actor and critic networks
        self.actors = [ModelBuilder(actor_config).build() for _ in range(n_agents)]
        self.critics = [ModelBuilder(critic_config).build() for _ in range(n_agents)]

        # Create target networks
        self.target_actors = [ModelBuilder(actor_config).build() for _ in range(n_agents)]
        self.target_critics = [ModelBuilder(critic_config).build() for _ in range(n_agents)]

        # Optimizers
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=lr) for critic in self.critics]

        # Initialize replay memory
        self.memory = []

        # Initialize target networks with the same weights as the original networks
        for i in range(n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        if self.verbosity > 0:
            print(f"MADDPGAgent initialized with {self.n_agents} agents.")

    def act(self, states):
        """
        Select actions for each agent based on the current policy (actor network).

        Parameters:
            states (list): List of states for each agent.

        Returns:
            actions (list): List of normalized actions for each agent.
        """
        actions = []
        for i, actor in enumerate(self.actors):
            state = states[i]
            if not isinstance(state, (list, np.ndarray)):
                state = [state]  # Ensure state is a sequence
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = actor(state).squeeze().numpy()
            
            # Normalize the action using Softmax
            action = F.softmax(torch.FloatTensor(action), dim=0).numpy()
            
            actions.append(action)
            if self.verbosity > 0:
                print(f"Agent {i} action (normalized): {action}")
        return actions

    def store(self, transition):
        """
        Store a transition in the replay memory.

        Parameters:
            transition (tuple): A tuple containing (states, actions, rewards, next_states).
        """
        self.memory.append(transition)
        if len(self.memory) > 10000:
            self.memory.pop(0)
        if self.verbosity > 0:
            print(f"Stored transition. Memory size: {len(self.memory)}")

    def train(self):
        """
        Train the agents by sampling a batch of transitions from the replay memory.
        """
        if len(self.memory) < 32:
            return

        # Sample a batch of transitions from the replay memory
        batch = random.sample(self.memory, 32)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))

        for i in range(self.n_agents):
            # Update critic
            with torch.no_grad():
                # Get the actions for the next states from the target actors
                next_actions = [self.target_actors[j](next_states[:, j, :]) for j in range(self.n_agents)]
                
                # Normalize each agent's actions
                next_actions = [F.softmax(action, dim=1) for action in next_actions]
                
                # Concatenate normalized actions
                next_actions = torch.cat(next_actions, dim=1)

                # Concatenate next_states and next_actions
                next_inputs = torch.cat([next_states.view(32, -1), next_actions], dim=-1)

                # Compute the target Q value
                target_q = rewards[:, i] + self.gamma * self.target_critics[i](next_inputs).squeeze()

            # Concatenate states and actions for the current Q value
            current_inputs = torch.cat([states.view(32, -1), actions.view(32, -1)], dim=-1)

            # Compute the current Q value
            current_q = self.critics[i](current_inputs).squeeze()

            # Compute the critic loss
            critic_loss = nn.MSELoss()(current_q, target_q)

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
            actor_inputs = torch.cat([states.view(32, -1), current_actions], dim=-1)

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