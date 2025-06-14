import unittest
import numpy as np
import pandas as pd
from envs.portfolio_agent_generator import create_portfolio_env

class TestPortfolioEnv(unittest.TestCase):
    def setUp(self):
        # Beispiel-Daten (MultiIndex: 2 Assets, jeweils Close-Preise)
        index = pd.date_range("2020-01-01", periods=5)
        columns = pd.MultiIndex.from_product([["AAPL", "MSFT"], ["Open", "High", "Low", "Close", "Volume"]], names=["Ticker", "Feature"])
        data = pd.DataFrame(
            [
                [100, 200, 100, 200, 1000, 100, 200, 100, 200, 1000],
                [101, 202, 101, 202, 1000, 101, 202, 101, 202, 1000],
                [102, 203, 102, 203, 1000, 102, 203, 102, 203, 1000],
                [103, 204, 103, 204, 1000, 103, 204, 103, 204, 1000],
                [104, 206, 104, 206, 1000, 104, 206, 104, 206, 1000],
            ],
            index=index,
            columns=columns,
        )
        self.data = data

    def test_single_agent(self):
        env = create_portfolio_env(self.data, n_agents=1)
        obs = env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape[0], self.data.shape[1])

        action = np.array([0.4, 0.4, 0.2])  # 2 Assets + Cash
        obs, reward, done, _ = env.step(action)
        self.assertIsInstance(obs, np.ndarray)
        self.assertTrue(np.isscalar(reward))
        self.assertIsInstance(done, bool)
        self.assertEqual(obs.shape[0], self.data.shape[1])

    def test_multi_agent_shared_action(self):
        env = create_portfolio_env(self.data, n_agents=2, shared_action=True)
        obs = env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape[0], self.data.shape[1])

        actions = [np.array([0.4, 0.4, 0.2]), np.array([0.5, 0.3, 0.2])]  # 2 Agents, 2 Assets + Cash
        obs, rewards, done, _ = env.step(actions)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(len(rewards), env.n_agents)
        self.assertIsInstance(rewards[0], float)
        self.assertIsInstance(done, bool)
        self.assertEqual(obs.shape[0], self.data.shape[1])

    def test_multi_agent_individual_action(self):
        env = create_portfolio_env(self.data, n_agents=2, shared_action=False)
        obs = env.reset()
        self.assertIsInstance(obs, list)
        self.assertEqual(len(obs), env.n_agents)
        self.assertEqual(obs[0].shape[0], self.data.shape[1] // env.n_agents)

        actions = [np.array([0.4, 0.6]), np.array([0.5, 0.5])]  # Each agent has 1 Asset + Cash
        obs, rewards, done, _ = env.step(actions)
        self.assertIsInstance(obs, list)
        self.assertEqual(len(rewards), env.n_agents)
        self.assertIsInstance(rewards[0], float)
        self.assertIsInstance(done, bool)
        self.assertEqual(obs[0].shape[0], self.data.shape[1] // env.n_agents)

if __name__ == "__main__":
    unittest.main()
