from agents.base_agent import BaseAgent
import torch
import torch.optim as optim
import numpy as np
import random
from agents.model_builder import ModelBuilder

class DQNAgent(BaseAgent):
    def __init__(self, obs_dim, act_dim, actor_config=None, lr=1e-3, gamma=0.99, batch_size=32, buffer_max_size=100000):
        super().__init__()

        # Use the provided network architecture or a default one
        default_architecture = [
            {"type": "Linear", "params": {"in_features": obs_dim, "out_features": 128}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 128, "out_features": 128}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 128, "out_features": 64}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 64, "out_features": act_dim}}
        ]
        actor_config = actor_config or default_architecture

        # Use ModelBuilder to create the models
        self.model = ModelBuilder(actor_config).build()
        self.target_model = ModelBuilder(actor_config).build()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.loss_fn = torch.nn.MSELoss()
        self.memory = []

        self.buffer_max_size = buffer_max_size

    def act(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Selects an action based on the current state and epsilon-greedy policy.
        Args:
            state (np.ndarray): The current state of the environment.
            epsilon (float): Probability of selecting a random action.
        Returns:
            action (int): The selected action.
        """
        with torch.no_grad():
            logits = self.model(state)

            if random.random() < epsilon:
                logits = torch.rand(logits.shape)

            probs = torch.softmax(logits, dim=1)

            return probs / probs.sum()

    def store(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.buffer_max_size:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))

        pred = self.model(states)
        with torch.no_grad():
            target = rewards + self.gamma * self.target_model(next_states).max(dim=1).values

        loss = self.loss_fn(pred.max(dim=1).values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
