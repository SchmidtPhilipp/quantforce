import numpy as np
import torch

from qf.train.logger import Logger
import matplotlib.pyplot as plt
from pypfopt.plotting import plot_efficient_frontier
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return

from scipy.optimize import minimize

from qf.agents.agent import Agent

from qf.utils.metrics import Metrics

class TangencyAgent(Agent):
    def __init__(self, env, 
                method="default", risk_free_rate=0.02, log_returns=True):
        """
        Initializes the Tangency agent with the given environment.
        """
        super().__init__(env=env)
        self.method=method
        self.risk_free_rate = risk_free_rate
        self.log_returns = log_returns
        self.expected_returns = None
        self.cov_matrix = None
        self.weights = None  # torch.Tensor of shape (n_assets + 1,)

    def train(self):
        """
        Trains the agent by calculating the tangency portfolio weights with cash (constrained to [0,1]^{n+1}).
        """

        dataset = self.env.get_dataset()
        historical_data = dataset.get_data()
        historical_data = historical_data.xs('Close', level=1, axis=1)

        self.weights = estimate_tangency_portfolio_weights(historical_data, 
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

    def evaluate(self, env, episodes=10, use_tqdm=True):
        """
        Evaluates the static tangency portfolio agent.
        """

        if env is None:
            env = self.env

        for _ in range(episodes):

            done = False
            total_reward = 0
            state = env.reset().to(env.device)

            while not done:
                action = self.act(state)
                #print("Action (weights incl. cash):", action)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state

        env.print_metrics()

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

        # Convert historical data to NumPy
        dataset = self.env.get_dataset()
        historical_data = dataset.get_data()
        historical_data = historical_data.xs('Close', level=1, axis=1)

        # Calculate expected returns and covariance matrix using PyPortfolioOpt
        mu = mean_historical_return(historical_data, log_returns=True)
        Sigma = CovarianceShrinkage(historical_data).ledoit_wolf()

        # Step 3: Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(historical_data)  # Expected returns
        S = risk_models.sample_cov(historical_data)  # Covariance matrix

        weight_bounds = (0, 1)
        # Step 4: Create an efficient frontier object
        ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

        # Step 5: Calculate key portfolios
        # Max Sharpe Ratio Portfolio
        ef_max_sharpe = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        max_sharpe_weights = ef_max_sharpe.max_sharpe()
        ret_ms, std_ms, sharpe_ms = ef_max_sharpe.portfolio_performance()

        # Min Volatility Portfolio
        ef_min_vol = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        min_vol_weights = ef_min_vol.min_volatility()
        ret_mv, std_mv, _ = ef_min_vol.portfolio_performance()

        # Step 6: Plot Efficient Frontier
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot individual asset points in grayscale
        for i, ticker in enumerate(df.columns):
            asset_return = mu[ticker]
            asset_vol = np.sqrt(S.loc[ticker, ticker])
            ax.scatter(asset_vol, asset_return, marker=".", s=70, color="0.2", label=ticker)
            ax.annotate(ticker, (asset_vol, asset_return),
                        textcoords="offset points", xytext=(5, -2.5), ha="left", fontsize=9, color="0.2")

        # Plot the efficient frontier
        plot_efficient_frontier(ef, ax=ax, show_assets=False, color="1", linewidth=2)

        # Step 7: Monte Carlo Simulation - Generate random portfolios
        n_portfolios = 2000  # Number of random portfolios
        mc_returns = []  # List to store portfolio returns
        mc_vols = []  # List to store portfolio volatilities
        mc_sharpes = []  # List to store Sharpe ratios

        risk_free_rate = 0.0  # Risk-free rate (adjust as needed)

        for _ in range(n_portfolios):
            # Generate random weights
            weights = np.random.dirichlet(np.ones(len(self.env.tickers)), size=1)[0]
            port_return = np.dot(weights, mu)  # Portfolio return
            port_vol = np.sqrt(np.dot(weights.T, np.dot(S, weights)))  # Portfolio volatility
            sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else np.nan  # Sharpe ratio
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
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=4, frameon=True)
        #ax.legend(loc="right", bbox_to_anchor=(0.5, 0.5, 0.1, 0.5), frameon=True)
        # Grid and layout adjustments
        plt.grid(True, color="0.85")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.xlim((0.3, 0.475))  # Adjust x-axis limits
        plt.ylim((-0.2, 0.6))  # Adjust y-axis limits

        # Step 9: Save plot
        plt.savefig(os.path.join(self.env.log_dir, "efficient_frontier.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.env.log_dir, "efficient_frontier.pgf"), bbox_inches='tight')

        # Show plot
        plt.show()


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
        Sigma = np.cov(returns_np.T)
        n = len(mu)

        # Extend mu and Sigma to include cash
        mu_ext = np.append(mu, risk_free_rate)
        C = np.zeros((n + 1, n + 1)) # increase size for cash
        C[:n, :n] = Sigma

        # check if C is positive semi definite
        if np.linalg.det(C) < 0:
            raise ValueError("Covariance matrix is not positive semi-definite. Cannot compute tangency portfolio weights.")
        
        # Define Sharpe ratio (to maximize) as a minimization
        def neg_sharpe(w):
            port_return = np.dot(w, mu_ext)
            port_vol = np.sqrt(np.dot(w, np.dot(C, w)))
            if port_vol == 0:
                return np.inf
            return -port_return / port_vol

        # Initial guess and constraints
        w0 = np.ones(n + 1) / (n + 1)
        bounds = [(0, 1)] * (n + 1)
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        weights = torch.tensor(result.x, dtype=torch.float32)  # save as torch.Tensor
        return weights
    
    elif method == 'pyportfolioopt':
        from pypfopt.efficient_frontier import EfficientFrontier
        from pypfopt import expected_returns
        from pypfopt import risk_models


        weight_bounds = (0, 1)
        df = historical_data
        mu = expected_returns.mean_historical_return(df)  # Expected returns
        S = risk_models.sample_cov(df)  # Covariance matrix
        # check if covariance matrix is positive semi-definite
        if not np.all(np.linalg.eigvals(S) >= 0):
            raise ValueError("Covariance matrix is not positive semi-definite. Cannot compute tangency portfolio weights.")

        ef_max_sharpe = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        max_sharpe_weights = ef_max_sharpe.max_sharpe()
        weights = torch.tensor(list(max_sharpe_weights.values()), dtype=torch.float32)
        return weights

    else:   
        raise ValueError(f"Unknown method: {method}. Use 'default' or 'pyportfolioopt'.")


        




if __name__ == "__main__":
    # Example usage


    from qf import TimeBasedDataset, MultiAgentPortfolioEnv, start_tensorboard
    from qf.data import DOWJONES

    start_tensorboard()

    env = MultiAgentPortfolioEnv()
    agent = TangencyAgent(env)
    agent.train()




    eval_env = MultiAgentPortfolioEnv()
    agent.evaluate(env=eval_env)

    #agent.visualize()