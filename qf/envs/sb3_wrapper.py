import gymnasium as gym
import numpy as np
import torch

from qf.envs.tensor_env import TensorEnv


class SB3Wrapper(gym.Env):
    def __init__(self, env: TensorEnv):
        self.env: TensorEnv = env
        self.observation_space = env.get_observation_space()
        self.action_space = env.get_action_space()
        self.device: str = env.device
        self.config = env.config

    def step(self, actions: np.ndarray) -> tuple:
        """
        Executes a step in the environment with actions provided as a NumPy array.

        Parameters:
            actions (np.ndarray): Actions to take in the environment. Shape is (n_agents, action_dim).

        Returns:
            obs (np.ndarray): Next observation(s). Shape is (n_agents, obs_dim).
            rewards (np.ndarray): Rewards received. Shape is (n_agents,).
            terminateds (np.ndarray): Done flags indicating if the episode has ended. Shape is (n_agents,).
            truncateds (np.ndarray): Flags indicating if the episode was truncated. Shape is (n_agents,).
            info (dict): Additional information from the environment.
        """
        # Konvertiere die Aktionen in einen Tensor
        tensor_actions: torch.Tensor = torch.tensor(actions, dtype=torch.float32)

        # If missing the agent dimension, add it
        if tensor_actions.ndim == 1:
            tensor_actions = tensor_actions.unsqueeze(0)

        obs, rewards, done, info = self.env.step(tensor_actions)
        # Entferne die Agentendimension für SB3

        obs = obs.squeeze(0).numpy()
        rewards = rewards.squeeze(0).numpy()
        terminateds = done.squeeze(0).numpy()
        truncateds = done.squeeze(0).numpy()
        info = info

        return obs, rewards, terminateds, truncateds, info

    def reset(self, *, seed: int = None, options: dict = None) -> np.ndarray:
        obs, _ = self.env.reset()
        return obs.squeeze(0).numpy(), {}

    def print_metrics(self):
        """
        Prints the metrics of the environment.
        This method can be overridden by subclasses to provide specific metrics.
        """
        self.env.print_metrics()

    def log_metrics(self, logger=None, run_type=None):
        """
        Logs the metrics of the environment at a specific step.
        This method can be overridden by subclasses to provide specific logging functionality.
        """
        self.env.log_metrics(logger=logger, run_type=run_type)

    def get_save_dir(self):
        """
        Returns the save directory of the environment.
        """
        return self.env.save_dir

    def save_data(self):
        """
        Saves the environment data to the specified path.
        """
        self.env.save_data()

    def get_logger(self):
        """
        Returns the logger associated with the environment.
        """
        return self.env.logger
