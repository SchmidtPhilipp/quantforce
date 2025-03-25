import unittest
import numpy as np
from agents.maddpg_agent import MADDPGAgent
from envs.simple_multi_agent_env import SimpleMultiAgentEnv

class TestMADDPGAgent(unittest.TestCase):
    def setUp(self):
        self.n_agents = 2
        self.obs_dim = 4
        self.act_dim = 2
        self.env = SimpleMultiAgentEnv(n_agents=self.n_agents, obs_dim=self.obs_dim, act_dim=self.act_dim)
        self.agent = MADDPGAgent(obs_dim=self.obs_dim, act_dim=self.act_dim, n_agents=self.n_agents)

    def test_act(self):
        state = self.env.reset()
        actions = self.agent.act(state)
        self.assertEqual(len(actions), self.n_agents)
        for action in actions:
            self.assertEqual(len(action), self.act_dim)

    def test_store_and_train(self):
        state = self.env.reset()
        actions = self.agent.act(state)
        next_state, rewards, done, _ = self.env.step(actions)
        transition = (state, actions, rewards, next_state)
        self.agent.store(transition)
        self.assertEqual(len(self.agent.memory), 1)
        self.agent.train()
        self.assertTrue(len(self.agent.memory) <= 10000)

if __name__ == '__main__':
    unittest.main()