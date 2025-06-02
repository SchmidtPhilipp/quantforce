import numpy as np
import torch

import matplotlib.pyplot as plt
from pypfopt.plotting import plot_efficient_frontier
from pypfopt.efficient_frontier import EfficientFrontier

from scipy.optimize import minimize

import qf as qf
from qf.agents.agent import Agent


class TangencyAgent(Agent):
    def __init__(self, env, config=None):
        """
        Initializes the Tangency agent with the given environment.
        """
        super().__init__(env=env)

        default_config = {
            "method": "default",  # Method to estimate tangency portfolio weights
            "risk_free_rate": qf.DEFAULT_TANGENCY_RISK_FREE_RATE,  # Risk-free rate for the tangency portfolio
            "log_returns": True  # Whether to use log returns for calculations
        }

        # Merge default config with provided config
        self.config = {**default_config, **(config or {})}

        self.historical_data = None
        self.method=self.config["method"]
        self.risk_free_rate = self.config["risk_free_rate"]
        self.log_returns = self.config["log_returns"]
        self.expected_returns = None
        self.cov_matrix = None
        self.weights = None  # torch.Tensor of shape (n_assets + 1,)

    def train(self, episodes=0, total_timesteps=0, use_tqdm=True): #episodes and use_tqdm for compatibility with Agent interface
        """
        Trains the agent by calculating the tangency portfolio weights with cash (constrained to [0,1]^{n+1}).
        """
        dataset = self.env.get_dataset()
        self.historical_data = dataset.get_data()
        self.historical_data = self.historical_data.xs('Close', level=1, axis=1)

        self.weights, self.expected_returns, self.cov_matrix = estimate_tangency_portfolio_weights(self.historical_data, 
                                                           risk_free_rate=self.risk_free_rate, 
                                                           log_returns=self.log_returns, 
                                                           method=self.method)

    def act(self, state):
        """
        Returns the full action vector (including cash weight).
        """
        if self.weights is None:
            raise ValueError("Agent has not been trained yet. Call `train()` first.")
        return self.weights.unsqueeze(0)

    def evaluate(self, eval_env, episodes=1, use_tqdm=True):
        """
        Evaluates the static tangency portfolio agent.
        """

        if eval_env is None:
            eval_env = self.eval_env

        for _ in range(episodes):

            done = False
            total_reward = 0
            state, _ = eval_env.reset()

            while not done:
                action = self.act(state)
                next_state, reward, done, _ = eval_env.step(action)
                total_reward += reward
                state = next_state

        eval_env.print_metrics()

        print(f"Total reward over evaluation: {total_reward}")
        return total_reward
    
    def visualize(self):
        """
        Visualizes the mean-variance diagram (Efficient Frontier).
        """
        from pypfopt import expected_returns, risk_models
        import os

        if self.weights is None:
            raise ValueError("Agent has not been trained yet. Call `train()` first.")

        mu = expected_returns.mean_historical_return(self.historical_data, log_returns=self.log_returns)
        S = risk_models.sample_cov(self.historical_data, log_returns=self.log_returns)

        weight_bounds = (0, 1)
        ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

        # Calculate key portfolios
        # Max Sharpe Ratio Portfolio
        ef_max_sharpe = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        max_sharpe_weights = ef_max_sharpe.max_sharpe()
        ret_ms, std_ms, sharpe_ms = ef_max_sharpe.portfolio_performance()

        # Min Volatility Portfolio
        ef_min_vol = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        min_vol_weights = ef_min_vol.min_volatility()
        ret_mv, std_mv, _ = ef_min_vol.portfolio_performance()

        # Plot Efficient Frontier
        fig, ax = plt.subplots(figsize=(20, 8))

        # Plot individual asset points
        for i, ticker in enumerate(self.historical_data.columns):
            asset_return = mu[ticker]
            asset_vol = np.sqrt(S[ticker][ticker])
            ax.scatter(asset_vol, asset_return, marker=".", s=70, color="0.2", label=ticker)

        # Plot the efficient frontier
        plot_efficient_frontier(ef, ax=ax, show_assets=False, color="1", linewidth=2)

        # Step 7: Monte Carlo Simulation - Generate random portfolios
        n_portfolios = 2000  # Number of random portfolios
        mc_returns = []  # List to store portfolio returns
        mc_vols = []  # List to store portfolio volatilities
        mc_sharpes = []  # List to store Sharpe ratios

        for _ in range(n_portfolios):
            # Generate random weights
            weights = np.random.dirichlet(np.ones(len(mu)), size=1)[0]
            port_return = np.dot(weights, mu)  # Portfolio return
            port_vol = np.sqrt(np.dot(weights.T, np.dot(S, weights)))  # Portfolio volatility
            sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else np.nan  # Sharpe ratio
            mc_returns.append(port_return)
            mc_vols.append(port_vol)
            mc_sharpes.append(sharpe)

        # Plot Monte Carlo portfolios, colored by Sharpe Ratio
        cmap = "hsv"  # Colormap for Sharpe Ratio
        sharpe_normalized = (mc_sharpes - np.nanmin(mc_sharpes)) / (np.nanmax(mc_sharpes) - np.nanmin(mc_sharpes))
        sc = ax.scatter(mc_vols, mc_returns, c=mc_sharpes, cmap=cmap, s=5, alpha=0.6, label="Monte Carlo Portfolios")
        cbar = plt.colorbar(sc, ax=ax, label="Sharpe Ratio", orientation='horizontal', location="bottom", shrink=1, aspect=40, anchor=(0.5, 0.5))

        # Highlight optimal portfolios
        ax.scatter(std_ms, ret_ms, marker="*", s=100, color="red", label="Max Sharpe Ratio")
        ax.scatter(std_mv, ret_mv, marker=5, s=100, color="black", label="Min Volatility")

        # Visualize the tangency portfolio
        tangency_weights = self.weights[:-1]
        tangency_return = np.dot(tangency_weights, mu)
        tangency_vol = np.sqrt(np.dot(tangency_weights.T, np.dot(S, tangency_weights)))
        ax.scatter(tangency_vol, tangency_return, marker="o", s=100, color="blue", label="Tangency Portfolio")

        # Step 8: Customize plot
        ax.set_xlabel("$\sigma_p = \sqrt{\mathbb{V}ar[r_p]}$ (Portfolio risk)")  # X-axis label
        ax.set_ylabel("$\mathbb{E}[r_p]$ (Expected rate of return)")  # Y-axis label

        # Legend above the plot
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="center right", bbox_to_anchor=(1.15, 0.5))
        plt.grid(True, color="0.85")
        plt.tight_layout(rect=[0, 0, 1, 1])
        # Step 9: Save plot
        plt.savefig(os.path.join(self.env.log_dir, "efficient_frontier.png"), dpi=300, bbox_inches='tight')

    def save(self, path):
        """
        Saves the Tangency agent's model to a file.
        Parameters:
            path (str): Path to save the model.
        """
        name = self.__class__.__name__
        if not path.endswith('.pt'):
            path = f"{path}/{name}.pt"
        else:
            path = path.replace('.pt', f'_{name}.pt')

        print(f"Saving Tangency agent to {path}")

        # Save the weights, expected returns, covariance matrix, and config
        torch.save({
            'weights': self.weights,
            'expected_returns': self.expected_returns,
            'cov_matrix': self.cov_matrix,
            'config': self.config
        }, path)

    def load(self, path):
        """
        Loads the Tangency agent's model from a file.
        Parameters:
            path (str): Path to load the model from.
        """
        print(f"Loading Tangency agent from {path}")
        checkpoint = torch.load(path)
        self.weights = checkpoint['weights']
        self.expected_returns = checkpoint['expected_returns']
        self.cov_matrix = checkpoint['cov_matrix']
        self.config = checkpoint['config']

def estimate_tangency_portfolio_weights(historical_data, risk_free_rate=0.0001, log_returns=True, method='default'):

    if method == 'default':
        # Convert to tensor and compute returns
        historical_data = torch.tensor(historical_data.values, dtype=torch.float32)
        returns = (historical_data[1:] / historical_data[:-1]) - 1

        if log_returns:
            returns = torch.clamp(returns + 1e-8, min=1e-8)
            returns = torch.log(returns)

        # Convert to NumPy for optimization
        returns_np = returns.numpy()
        mu = np.mean(returns_np, axis=0)

        # insert cash as the last asset
        mu = np.append(mu, risk_free_rate)
        C = np.zeros((len(mu), len(mu)))
        C[:-1, :-1] = np.cov(returns_np.T)

        # check if C is positive semi definite
        if np.linalg.det(C) < 0:
            raise ValueError("Covariance matrix is not positive semi-definite. Cannot compute tangency portfolio weights.")
        
        # Define Sharpe ratio (to maximize) as a minimization
        def neg_sharpe(w):
            port_return = np.dot(w, mu)
            port_vol = np.sqrt(np.dot(w, np.dot(C, w)))
            if port_vol == 0:
                return np.inf
            return -port_return / port_vol

        n = len(mu)
        # Initial guess and constraints
        w0 = np.ones(n) / (n)
        bounds = [(0, 1)] * (n)
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        weights = torch.tensor(result.x, dtype=torch.float32)  # save as torch.Tensor
        return weights, mu, C
    
    elif method == 'pyportfolioopt':
        from pypfopt.efficient_frontier import EfficientFrontier
        from pypfopt import expected_returns
        from pypfopt import risk_models


        weight_bounds = (0, 1)
        mu = expected_returns.mean_historical_return(historical_data, log_returns=log_returns)  # Expected returns
        C= risk_models.sample_cov(historical_data, log_returns=log_returns)


        # check if covariance matrix is positive semi-definite
        if not np.all(np.linalg.eigvals(C) >= 0):
            raise ValueError("Covariance matrix is not positive semi-definite. Cannot compute tangency portfolio weights.")

        ef_max_sharpe = EfficientFrontier(mu, C, weight_bounds=weight_bounds)
        max_sharpe_weights = ef_max_sharpe.max_sharpe()
        weights = torch.tensor(list(max_sharpe_weights.values()), dtype=torch.float32)
        # Add cash with the risk-free rate as a constant series
        weights = torch.cat((weights, torch.tensor([1 - weights.sum()], dtype=torch.float32)))

        return weights, mu, C

    else:   
        raise ValueError(f"Unknown method: {method}. Use 'default' or 'pyportfolioopt'.")



if __name__ == "__main__":
    qf.start_tensorboard()

    # Tangency Agent configuration
    DEFAULT_TANGENCY_TICKERS =["NVDA", "BLDR", "UBER", "WBD"]  # Example tickers from different sectors
    DEFAULT_TANGENCY_TRAIN_START = "2020-01-01"  # Start date for historical data
    DEFAULT_TANGENCY_TRAIN_END = "2025-01-01"  # End date for historical data

    CONFIG = qf.DEFAULT_TRAIN_ENV_CONFIG
    CONFIG['tickers'] = DEFAULT_TANGENCY_TICKERS
    CONFIG['start_date'] = DEFAULT_TANGENCY_TRAIN_START
    CONFIG['end_date'] = DEFAULT_TANGENCY_TRAIN_END

    env = qf.MultiAgentPortfolioEnv(**CONFIG)
    agent = qf.TangencyAgent(env)
    agent.train(episodes=1)

    CONFIG = qf.DEFAULT_EVAL_ENV_CONFIG
    CONFIG['tickers'] = qf.DEFAULT_TANGENCY_TICKERS
    eval_env = qf.MultiAgentPortfolioEnv(**qf.DEFAULT_EVAL_ENV_CONFIG)
    agent.evaluate(eval_env)
    agent.visualize()


    