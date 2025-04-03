import numpy as np
from agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, act_dim):
        self.act_dim = act_dim  # includes cash dimension

    def act(self, state):
        # Sample random allocation vector
        weights = np.random.rand(self.act_dim)
        weights /= np.sum(weights) + 1e-8
        return weights

    def store(self, transition):
        pass  # No memory for random agent

    def train(self):
        pass  # No training

    def load(self, path):
        pass

    def save(self, path):
        pass
