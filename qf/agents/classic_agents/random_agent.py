from typing import Optional

import numpy as np
import torch

from qf.agents.agent import Agent
from qf.agents.config.base_agent_config import BaseAgentConfig
from qf.agents.config.classic_agents.classic_agent_config import ClassicAgentConfig
from qf.agents.config.classic_agents.random_agent_config import RandomAgentConfig


class RandomAgent(Agent):
    def __init__(
        self,
        env,
        config: Optional[ClassicAgentConfig] = None,
        weights: np.ndarray = None,
    ):
        """
        Initializes the RandomAgent with the given environment.
        Parameters:
            env: The environment in which the agent will operate.
        """
        super().__init__(env, config=config)
        self.weights = weights
        if weights is None:
            self.weights = self.env.action_space.sample()

    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Selects a random action from the action space of the environment.
        Parameters:
            state (np.ndarray): The current state of the environment.
        Returns:
            np.ndarray: A random action from the action space.
        """
        return self.weights

    def train(
        self,
        total_timesteps=100000,
        use_tqdm=True,
        save_best=True,
        eval_env=None,
        eval_every_n_steps=None,
        n_eval_episodes=1,
        print_eval_metrics=False,
    ):
        """
        Trains the RandomAgent for a specified number of timesteps.
        Parameters:
            total_timesteps (int): Total number of timesteps to train the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print training summaries.
        """
        # RandomAgent does not require training, but we need to save the model if requested
        if save_best:
            import os

            best_model_path = os.path.join(self.env.save_dir, "best_model")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            self.save(best_model_path)

    def _save_impl(self, path):
        """
        Saves the RandomAgent's model to a file.

        Parameters:
            path (str): Path to save the model.
        """
        name = self.__class__.__name__
        if not path.endswith(".pt"):
            path = f"{path}/{name}.pt"
        else:
            path = path.replace(".pt", f"_{name}.pt")

        print(f"Saving RandomAgent to {path}")

        # Save the weights and config
        torch.save({"weights": self.weights, "config": self.config}, path)

    def _load_impl(self, path):
        """
        Loads the RandomAgent's model from a file.

        Parameters:
            path (str): Path to load the model from.
        """
        # print(f"Loading RandomAgent from {path}")
        name = self.__class__.__name__
        if not path.endswith(".pt"):
            path = f"{path}/{name}.pt"
        else:
            path = path.replace(".pt", f"_{name}.pt")
        checkpoint = torch.load(path, weights_only=False)
        self.weights = checkpoint["weights"]
        self.config = checkpoint["config"]

    @staticmethod
    def get_hyperparameter_space():
        """
        Returns the hyperparameters of the RandomAgent.

        Returns:
            dict: Hyperparameter space for the agent.
        """
        return RandomAgentConfig.get_hyperparameter_space()

    @staticmethod
    def get_default_config():
        """
        Returns the default configuration for the RandomAgent.

        Returns:
            dict: Default configuration for the agent.
        """
        return RandomAgentConfig.get_default_config()
