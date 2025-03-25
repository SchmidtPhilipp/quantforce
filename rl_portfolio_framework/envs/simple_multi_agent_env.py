import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleMultiAgentEnv(gym.Env):
    def __init__(self, n_agents=2, obs_dim=4, act_dim=2):
        super(SimpleMultiAgentEnv, self).__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_agents, obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_agents, act_dim), dtype=np.float32)

    def reset(self):
        self.state = np.random.randn(self.n_agents, self.obs_dim)
        return self.state

    def step(self, actions):
        rewards = np.random.randn(self.n_agents)
        next_state = np.random.randn(self.n_agents, self.obs_dim)
        done = False
        return next_state, rewards, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass