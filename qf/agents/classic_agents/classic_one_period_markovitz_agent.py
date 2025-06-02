import numpy as np
import torch
from qf.agents.agent import Agent
from pypfopt.risk_models import risk_matrix
from pypfopt.efficient_frontier import EfficientFrontier


class ClassicOnePeriodMarkovitzAgent(Agent):
    def __init__(self, env, config=None):
        """
        Initializes the ClassicOnePeriodMarkovitzAgent with the given environment.

        Parameters:
            env: The environment in which the agent operates.
            config (dict, optional): Configuration dictionary for the agent.
        """
        super().__init__(env=env)

        default_config = {
            "log_returns": True,  # Whether to use log returns for calculations
            "target": "Tangency",  # Optimization target: Tangency, MaxExpReturn, MinVariance
            "risk_model": "sample_cov",  # Risk model: sample_cov, semicovariance, exp_cov, etc.
            "risk_free_rate": 0.01,  # Risk-free rate for Tangency optimization
        }

        # Merge default config with provided config
        self.config = {**default_config, **(config or {})}

        self.historical_data = None
        self.weights = None  # torch.Tensor of shape (n_assets + 1,)

    def train(self, episodes=0, total_timesteps=0, use_tqdm=True):
        """
        Trains the agent by calculating portfolio weights based on the selected target and risk model.
        """
        dataset = self.env.get_dataset()
        self.historical_data = dataset.get_data()
        self.historical_data = self.historical_data.xs('Close', level=1, axis=1)

        # Convert historical data to numpy array
        historical_data = self.historical_data.values

        # Calculate returns
        if self.config["log_returns"]:
            returns = np.log(historical_data[1:] / historical_data[:-1])
        else:
            returns = (historical_data[1:] / historical_data[:-1]) - 1

        # Compute covariance matrix using the selected risk model
        try:
            cov_matrix = risk_matrix(
                prices=self.historical_data,
                method=self.config["risk_model"],
                returns_data=True  # We are passing returns instead of prices
            )
        except ValueError:
            raise ValueError(f"Unsupported risk model: {self.config['risk_model']}")

        # Select optimization target
        n_assets = cov_matrix.shape[0]
        if self.config["target"] == "Tangency":
            mean_returns = np.mean(returns, axis=0)
            ef = EfficientFrontier(mean_returns, cov_matrix)
            weights = ef.max_sharpe(risk_free_rate=self.config["risk_free_rate"])
        elif self.config["target"] == "MaxExpReturn":
            mean_returns = np.mean(returns, axis=0)
            weights = np.zeros(n_assets)
            weights[np.argmax(mean_returns)] = 1.0
        elif self.config["target"] == "MinVariance":
            ef = EfficientFrontier(None, cov_matrix)
            weights = ef.min_volatility()
        else:
            raise ValueError(f"Unsupported target: {self.config['target']}")

        # Convert weights to torch.Tensor and add cash weight
        weights = torch.tensor(list(weights.values()), dtype=torch.float32)
        weights_with_cash = torch.cat([weights, torch.tensor([0.0])])  # Add cash weight (set to 0)

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
        Evaluates the static portfolio agent.

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
        Saves the ClassicOnePeriodMarkovitzAgent's model to a file.

        Parameters:
            path (str): Path to save the model.
        """
        name = self.__class__.__name__
        if not path.endswith('.pt'):
            path = f"{path}/{name}.pt"
        else:
            path = path.replace('.pt', f'_{name}.pt')

        print(f"Saving ClassicOnePeriodMarkovitzAgent to {path}")

        # Save the weights and config
        torch.save({
            'weights': self.weights,
            'config': self.config
        }, path)

    def load(self, path):
        """
        Loads the ClassicOnePeriodMarkovitzAgent's model from a file.

        Parameters:
            path (str): Path to load the model from.
        """
        print(f"Loading ClassicOnePeriodMarkovitzAgent from {path}")
        checkpoint = torch.load(path)
        self.weights = checkpoint['weights']
        self.config = checkpoint['config']