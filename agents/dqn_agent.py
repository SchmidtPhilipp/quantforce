from agents.base_agent import BaseAgent
import torch
import torch.optim as optim
import numpy as np
import random
from agents.model_builder import ModelBuilder

class DQNAgent(BaseAgent):
    def __init__(self, obs_dim, act_dim, model_config=None, lr=1e-3):
        super().__init__()

        # Dynamically generate the default config based on obs_dim and act_dim
        default_config = [
            {"type": "Linear", "params": {"in_features": obs_dim, "out_features": 128}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 128, "out_features": 128}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 128, "out_features": 64}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 64, "out_features": act_dim}}
        ]

        # Use the default config if no custom config is provided
        model_config = model_config or default_config

        # Use ModelBuilder to create the models
        self.model = ModelBuilder(model_config).build()
        self.target_model = ModelBuilder(model_config).build()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = 0.99
        self.batch_size = 32
        self.loss_fn = torch.nn.MSELoss()
        self.memory = []

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(state).squeeze()
            probs = torch.softmax(logits, dim=0).numpy()
            return probs / np.sum(probs)

    def store(self, transition):
        self.memory.append(transition)
        if len(self.memory) > 10000:
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
