import numpy as np
import torch
from qf.agents.agent import Agent


class MaxExpReturnAgent(Agent):
    def __init__(self, env, config=None):
        """
        Initializes the GreedyAgent with the given environment.

        Parameters:
            env: The environment in which the agent operates.
            config (dict, optional): Configuration dictionary for the agent.
        """
        super().__init__(env=env)

        default_config = {
            "log_returns": True,  # Whether to use log returns for calculations
        }

        # Merge default config with provided config
        self.config = {**default_config, **(config or {})}

        self.historical_data = None
        self.log_returns = self.config["log_returns"]
        self.weights = None  # torch.Tensor of shape (n_assets + 1,)

    def train(self, episodes=0, total_timesteps=0, use_tqdm=True):
        """
        Trains the agent by calculating the greedy portfolio weights (all on the highest expected return asset).
        """
        dataset = self.env.get_dataset()
        self.historical_data = dataset.get_data()
        self.historical_data = self.historical_data.xs('Close', level=1, axis=1)

        # Calculate expected returns
        historical_data = torch.tensor(self.historical_data.values, dtype=torch.float32)
        returns = (historical_data[1:] / historical_data[:-1]) - 1

        if self.log_returns:
            returns = torch.clamp(returns + 1e-8, min=1e-8)
            returns = torch.log(returns)

        # Compute mean returns
        mean_returns = torch.mean(returns, dim=0)

        # Find the asset with the highest expected return
        max_return_idx = torch.argmax(mean_returns).item()

        # Allocate all weights to the asset with the highest return
        n_assets = len(mean_returns)
        weights = torch.zeros(n_assets + 1, dtype=torch.float32)  # Include cash
        weights[max_return_idx] = 1.0  # Allocate all to the highest return asset

        self.weights = weights

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
        Evaluates the static greedy portfolio agent.

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
        Saves the GreedyAgent's model to a file.

        Parameters:
            path (str): Path to save the model.
        """
        name = self.__class__.__name__
        if not path.endswith('.pt'):
            path = f"{path}/{name}.pt"
        else:
            path = path.replace('.pt', f'_{name}.pt')

        print(f"Saving GreedyAgent to {path}")

        # Save the weights and config
        torch.save({
            'weights': self.weights,
            'config': self.config
        }, path)

    def load(self, path):
        """
        Loads the GreedyAgent's model from a file.

        Parameters:
            path (str): Path to load the model from.
        """
        print(f"Loading GreedyAgent from {path}")
        checkpoint = torch.load(path)
        self.weights = checkpoint['weights']
        self.config = checkpoint['config']