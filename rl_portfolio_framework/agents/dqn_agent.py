from agents.base_agent import BaseAgent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DQNAgent(BaseAgent):
    def __init__(self, obs_dim, act_dim):  # act_dim = n_assets + 1
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

        self.target_model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.batch_size = 32
        self.loss_fn = nn.MSELoss()
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
