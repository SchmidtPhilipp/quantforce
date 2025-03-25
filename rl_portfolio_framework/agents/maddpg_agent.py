import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class Actor(nn.Module):
    """
    Actor network for the MADDPG agent.
    This network takes the state as input and outputs the action.
    """
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, act_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    """
    Critic network for the MADDPG agent.
    This network takes the state and action as input and outputs the Q-value.
    """
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x, a):
        x = self.relu(self.fc1(torch.cat([x, a], dim=1)))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
    def __init__(self, obs_dim, act_dim, n_agents, lr=1e-3, gamma=0.99, tau=0.01, verbosity=0):
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

        # Initialize actors and critics for each agent
        self.actors = [Actor(obs_dim, act_dim) for _ in range(n_agents)]
        self.critics = [Critic(obs_dim * n_agents, act_dim * n_agents) for _ in range(n_agents)]
        self.target_actors = [Actor(obs_dim, act_dim) for _ in range(n_agents)]
        self.target_critics = [Critic(obs_dim * n_agents, act_dim * n_agents) for _ in range(n_agents)]

        # Initialize optimizers for actors and critics
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
            actions (list): List of actions for each agent.
        """
        actions = []
        for i, actor in enumerate(self.actors):
            state = states[i]
            if not isinstance(state, (list, np.ndarray)):
                state = [state]  # Ensure state is a sequence
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = actor(state).squeeze().numpy()
            actions.append(action)
            if self.verbosity > 0:
                print(f"Agent {i} action: {action}")
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
                next_actions = torch.cat(next_actions, dim=1)
                # Compute the target Q value
                target_q = rewards[:, i] + self.gamma * self.target_critics[i](next_states.view(32, -1), next_actions).squeeze()

            # Compute the current Q value
            current_q = self.critics[i](states.view(32, -1), actions.view(32, -1)).squeeze()
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
            # Compute the actor loss
            actor_loss = -self.critics[i](states.view(32, -1), current_actions).mean()

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