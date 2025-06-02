import numpy as np
import torch
from qf.agents.agent import Agent
from qf.utils.correlation import compute_correlation


class MinVarianceAgent(Agent):
    def __init__(self, env, config=None):
        """
        Initializes the MinVarianceAgent with the given environment.

        Parameters:
            env: The environment in which the agent operates.
            config (dict, optional): Configuration dictionary for the agent.
        """
        super().__init__(env=env)

        default_config = {
            "log_returns": True,  # Whether to use log returns for calculations
            "risk_free_rate": 0.0001,  # Risk-free rate for portfolio optimization
        }

        # Merge default config with provided config
        self.config = {**default_config, **(config or {})}

        self.historical_data = None
        self.log_returns = self.config["log_returns"]
        self.weights = None  # torch.Tensor of shape (n_assets + 1,)

    def train(self, episodes=0, total_timesteps=0, use_tqdm=True):
        """
        Trains the agent by calculating the minimum variance portfolio weights.
        """
        dataset = self.env.get_dataset()
        self.historical_data = dataset.get_data()
        self.historical_data = self.historical_data.xs('Close', level=1, axis=1)

        # Calculate returns
        historical_data = torch.tensor(self.historical_data.values, dtype=torch.float32)
        returns = (historical_data[1:] / historical_data[:-1]) - 1

        if self.log_returns:
            returns = torch.clamp(returns + 1e-8, min=1e-8)
            returns = torch.log(returns)

        # Compute covariance matrix
        covariance_matrix = torch.cov(returns.T)

        # Solve for minimum variance portfolio weights
        n_assets = covariance_matrix.shape[0]
        ones = torch.ones(n_assets, dtype=torch.float32)
        inv_cov = torch.inverse(covariance_matrix)

        weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)

        # Add cash weight (set to 0 for minimum variance portfolio)
        weights_with_cash = torch.cat([weights, torch.tensor([0.0])])

        self.weights = weights_with_cash

    def act(self, state):
        """
        Returns the full action vector (including cash weight).

        Parameters:
            state: The current state of the environment.

        Returns:
            torch.Tensor: Action vector.
        """
        if self.weights is None:
            raise ValueError("Agent has not been trained yet. Call `train()` first.")
        return self.weights.unsqueeze(0)

    def evaluate(self, eval_env, episodes=1, use_tqdm=True):
        """
        Evaluates the static minimum variance portfolio agent.

        Parameters:
            eval_env: The evaluation environment.
            episodes (int): Number of episodes to evaluate.
            use_tqdm (bool): Whether to use tqdm for progress tracking.

        Returns:
            float: Total reward over evaluation.
        """
        eval_env.set_environment_mode(self.set_env_mode())

        if eval_env is None:
            eval_env = self.eval_env

        total_reward = 0
        for _ in range(episodes):
            done = False
            state, _ = eval_env.reset()

            while not done:
                action = self.act(state)
                next_state, reward, done, _, _ = eval_env.step(action)
                total_reward += reward
                state = next_state

        eval_env.print_metrics()
        print(f"Total reward over evaluation: {total_reward}")
        return total_reward

    def save(self, path):
        """
        Saves the MinVarianceAgent's model to a file.

        Parameters:
            path (str): Path to save the model.
        """
        name = self.__class__.__name__
        if not path.endswith('.pt'):
            path = f"{path}/{name}.pt"
        else:
            path = path.replace('.pt', f'_{name}.pt')

        print(f"Saving MinVarianceAgent to {path}")

        # Save the weights and config
        torch.save({
            'weights': self.weights,
            'config': self.config
        }, path)

    def load(self, path):
        """
        Loads the MinVarianceAgent's model from a file.

        Parameters:
            path (str): Path to load the model from.
        """
        print(f"Loading MinVarianceAgent from {path}")
        checkpoint = torch.load(path)
        self.weights = checkpoint['weights']
        self.config = checkpoint['config']