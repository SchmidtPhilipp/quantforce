import numpy as np

class UniversalPortfolioAgent:
    def __init__(self, n_assets, n_crps=1000, seed=42):
        """
        Implements Cover's Universal Portfolio (UP) strategy for a multi-agent environment.

        Parameters:
        - n_assets: Number of assets in the portfolio (excluding cash).
        - n_crps: Number of constant rebalanced portfolios to sample.
        - seed: Random seed for reproducibility.
        """
        np.random.seed(seed)
        self.n_assets = n_assets
        self.n_crps = n_crps
        self.crp_candidates = self._sample_crps()
        self.performance = np.ones(n_crps)

    def _sample_crps(self):
        """
        Uniformly sample CRPs from the n-dimensional simplex.
        """
        return np.random.dirichlet(np.ones(self.n_assets + 1), self.n_crps)

    def _update_performance(self, relative_price_vector):
        """
        Update the wealth of each CRP based on new price relatives.

        Parameters:
        - relative_price_vector: np.array of asset price relatives for the current timestep.
        """
        gains = np.dot(self.crp_candidates, relative_price_vector)
        self.performance *= gains

    def _compute_universal_portfolio(self):
        """
        Compute the weighted average portfolio over all CRPs.

        Returns:
        - portfolio: np.array of portfolio weights (including cash).
        """
        weights = self.performance / np.sum(self.performance)
        portfolio = np.average(self.crp_candidates, axis=0, weights=weights)
        return portfolio

    def act(self, observation):
        """
        Compute the action (portfolio weights) based on the current observation
        and update the performance (training).

        Parameters:
            - observation: np.array of the current observation (price relatives).

        Returns:
            - action: np.array of portfolio weights (including cash).
        """
        # Normalize prices relative to cash
        relative_price_vector = observation / observation[0]
        
        # Compute the action (portfolio weights)
        action = self._compute_universal_portfolio()
        
        # Update performance (train)
        self._update_performance(relative_price_vector)
        
        return action

    def train(self):
        """
        Placeholder for training logic. Currently unused as training is integrated into `act`.
        """
        pass

