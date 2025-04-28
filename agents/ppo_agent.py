from agents.base_agent import BaseAgent
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from agents.model_builder import ModelBuilder


class PPOAgent(BaseAgent):
    def __init__(self, obs_dim, act_dim, model_config=None, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        """
        Initializes the PPO Agent.

        Parameters:
            obs_dim (int): Dimension of the state space.
            act_dim (int): Dimension of the action space.
            model_config (list[dict]): Configuration for the neural network.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for rewards.
            eps_clip (float): Clipping parameter for PPO.
            k_epochs (int): Number of epochs for updating the policy.
        """
        super().__init__()

        # Default model configuration
        default_config = [
            {"type": "Linear", "params": {"in_features": obs_dim, "out_features": 128}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 128, "out_features": 128}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 128, "out_features": act_dim}, "activation": "Softmax"}
        ]

        # Use the default config if no custom config is provided
        model_config = model_config or default_config

        # Build the actor and critic networks using ModelBuilder
        self.actor = ModelBuilder(model_config).build()
        critic_config = [
            {"type": "Linear", "params": {"in_features": obs_dim, "out_features": 128}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 128, "out_features": 128}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 128, "out_features": 1}}
        ]
        self.critic = ModelBuilder(critic_config).build()

        # Optimizers
        self.optimizer = optim.Adam([
            {"params": self.actor.parameters(), "lr": lr},
            {"params": self.critic.parameters(), "lr": lr}
        ])

        # PPO parameters
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # Loss function
        self.mse_loss = nn.MSELoss()

        # Memory for storing trajectories
        self.memory = []

    def act(self, state):
        """
        Selects an action based on the current policy.

        Parameters:
            state (np.ndarray): The current state.

        Returns:
            action (int): The selected action.
            log_prob (torch.Tensor): The log probability of the selected action.
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def store(self, transition):
        """
        Stores a transition in memory.

        Parameters:
            transition (tuple): A tuple containing (state, action, log_prob, reward, next_state, done).
        """
        self.memory.append(transition)

    def train(self):
        """
        Updates the policy using the PPO algorithm.
        """
        if len(self.memory) == 0:
            return

        # Convert memory to tensors
        states, actions, log_probs, rewards, next_states, dones = zip(*self.memory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        log_probs = torch.cat(log_probs)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Compute discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                cumulative_reward = 0
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        # PPO update
        for _ in range(self.k_epochs):
            # Get values and action probabilities from the current policy
            action_probs = self.actor(states)
            state_values = self.critic(states).squeeze()
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions.squeeze())

            # Compute ratios
            ratios = torch.exp(new_log_probs - log_probs)

            # Compute surrogate loss
            advantages = discounted_rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = self.mse_loss(state_values, discounted_rewards)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss

            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear memory
        self.memory = []