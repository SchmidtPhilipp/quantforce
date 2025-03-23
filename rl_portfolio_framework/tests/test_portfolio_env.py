import unittest
import numpy as np
import pandas as pd
from envs.portfolio_env import PortfolioEnv  # Passe diesen Pfad ggf. an

class TestPortfolioEnv(unittest.TestCase):
    def setUp(self):
        # Beispiel-Daten (MultiIndex: 2 Assets, jeweils Close-Preise)
        index = pd.date_range("2020-01-01", periods=5)
        columns = pd.MultiIndex.from_product([["AAPL", "MSFT"], ["Close"]], names=["Ticker", "Feature"])
        data = pd.DataFrame(
            [
                [100, 200],
                [101, 202],
                [102, 203],
                [103, 204],
                [104, 206],
            ],
            index=index,
            columns=columns,
        )
        self.env = PortfolioEnv(data)

    def test_reset_returns_observation(self):
        obs = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape[0], len(self.env.assets))

    def test_step_returns_expected_outputs(self):
        self.env.reset()
        action = np.array([0.4, 0.4, 0.2])  # 2 Assets + Cash
        obs, reward, done, _ = self.env.step(action)

        self.assertIsInstance(obs, np.ndarray)
        self.assertTrue(np.isscalar(reward))
        self.assertIsInstance(done, bool)
        self.assertEqual(obs.shape[0], len(self.env.assets))

    def test_done_flag(self):
        self.env.reset()
        for _ in range(len(self.env.data)):
            action = np.array([0.3, 0.3, 0.4])
            obs, reward, done, _ = self.env.step(action)
        self.assertTrue(done)

    def test_action_normalization(self):
        self.env.reset()
        raw_action = np.array([5.0, 3.0, 2.0])  # Should be normalized internally
        obs, reward, done, _ = self.env.step(raw_action)
        self.assertFalse(np.isnan(reward))
        self.assertEqual(obs.shape[0], len(self.env.assets))

if __name__ == "__main__":
    unittest.main()
