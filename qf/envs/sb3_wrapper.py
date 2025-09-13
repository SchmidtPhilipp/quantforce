from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch

from qf.envs.tensor_env import TensorEnv


class SB3Wrapper(gym.Env):
    def __init__(self, env: TensorEnv):
        self.env: TensorEnv = env
        # Force CPU for optimal performance - neural networks can use GPU separately
        self.device: str = "cpu"
        self.config = getattr(env, "config", {})
        self.env_config = getattr(env, "env_config", {})
        self.n_agents = getattr(env, "n_agents", 1)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def save_dir(self):
        return self.env.save_dir

    @property
    def save_data(self):
        return self.env.save_data

    @property
    def print_metrics(self):
        return self.env.print_metrics

    @property
    def log_metrics(self):
        return self.env.log_metrics

    @property
    def experiment_logger(self):
        return self.env.experiment_logger

    @property
    def environment_name(self):
        return self.env.environment_name

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        # Optimized: Keep actions on CPU to avoid transfers
        if isinstance(actions, np.ndarray):
            tensor_actions = torch.from_numpy(actions).float()
        else:
            tensor_actions = torch.tensor(actions, dtype=torch.float32)

        if tensor_actions.ndim == 1:
            tensor_actions = tensor_actions.unsqueeze(0)

        obs, reward, done, info = self.env.step(tensor_actions)

        # Optimized: Convert to numpy efficiently without GPU transfers
        if hasattr(obs, "as_tensor"):
            obs_tensor = obs.as_tensor()
        else:
            obs_tensor = obs

        if obs_tensor.device.type != "cpu":
            obs_numpy = obs_tensor.squeeze(0).cpu().numpy()
        else:
            obs_numpy = obs_tensor.squeeze(0).numpy()

        if reward.device.type != "cpu":
            reward_numpy = reward.squeeze(0).cpu().numpy()
        else:
            reward_numpy = reward.squeeze(0).numpy()

        if done.device.type != "cpu":
            done_numpy = done.squeeze(0).cpu().numpy()
        else:
            done_numpy = done.squeeze(0).numpy()

        info_dict = info

        return (
            obs_numpy,
            reward_numpy,
            done_numpy,
            np.zeros_like(done_numpy, dtype=bool),
            info_dict,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset()

        # Optimized: Convert to numpy efficiently
        if hasattr(obs, "as_tensor"):
            obs_tensor = obs.as_tensor()
        else:
            obs_tensor = obs

        if obs_tensor.device.type != "cpu":
            obs_numpy = obs_tensor.squeeze(0).cpu().numpy()
        else:
            obs_numpy = obs_tensor.squeeze(0).numpy()

        return obs_numpy, info
