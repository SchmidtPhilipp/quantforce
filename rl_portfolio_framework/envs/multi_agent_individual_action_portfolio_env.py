from gymnasium import spaces
import numpy as np
from envs.base_portfolio_env import BasePortfolioEnv



class MultiAgentIndividualActionPortfolioEnv(BasePortfolioEnv):
    def __init__(self, data, initial_balance=1_000, verbosity=0, n_agents=2):
        super().__init__(data, initial_balance, verbosity, n_agents)
        assert n_agents == len(data.columns.get_level_values(0).unique()), "Number of agents must equal number of assets"
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.data.shape[1] // self.n_agents,),
            dtype=np.float32
        )

    def _get_observation(self):
        obs = self.data.iloc[self.current_step].values.astype(np.float32)
        return np.array_split(obs, self.n_agents)

    def step(self, actions):
        rewards = []
        for i in range(self.n_agents):
            final_action = np.clip(actions[i], 0, 1)
            final_action /= np.sum(final_action) + 1e-8

            cash_weight = final_action[-1]
            asset_weight = final_action[0]

            old_prices = self.data.xs("Close", axis=1, level=1).iloc[self.current_step].values
            self.current_step += 1
            done = self.current_step >= len(self.data)

            if done:
                reward = [0.0] * self.n_agents
                obs = np.zeros(self.data.shape[1], dtype=np.float32)
                if self.verbosity > 0:
                    print("Episode finished!")
                return obs, reward, done, {}

            new_prices = self.data.xs("Close", axis=1, level=1).iloc[self.current_step].values
            asset_return = new_prices[i] / (old_prices[i] + 1e-15) - 1
            portfolio_return = cash_weight * 1.0 + asset_weight * asset_return

            self.balance *= (1 + portfolio_return)
            reward = np.log(1 + portfolio_return)
            rewards.append(reward)

            obs = self.data.iloc[self.current_step].values.astype(np.float32)
            obs = np.array_split(obs, self.n_agents)
            if self.verbosity > 0:
                print(f"Step: {self.current_step} | Reward: {reward:.4f} | Balance: {self.balance:.2f}")
                print(f"Action: {actions[i]} | Final Action: {final_action} | Prices: {new_prices}")
                print(f"Portfolio return: {portfolio_return:.4f} | Asset return: {asset_return}")
                print(f"Obs: {obs}")
                print(f"Observation shape: {obs[i].shape}")

        return obs, rewards, done, {}