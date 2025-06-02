import torch
import gymnasium as gym

class TensorEnv(gym.Env):
    """
    Base class for tensor-based environments.

    This class provides a structure for environments that use tensors for actions, observations, and rewards.
    Subclasses should implement the `_get_observation`, `_calculate_rewards`, and `_check_done` methods.
    """
    def __init__(self, device="cpu"):
        super(TensorEnv, self).__init__()
        self.device = device

    from typing import Tuple

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Executes a step in the environment.

        Parameters:
            actions (torch.Tensor): Tensor of actions. Shape is (n_agents, action_dim).

        Returns:
            obs (torch.Tensor): Next observation(s). Shape is (n_agents, obs_dim).
            rewards (torch.Tensor): Rewards. Shape is (n_agents,).
            done (torch.Tensor): Done flags. Shape is (n_agents,).
            info (dict): Additional information. Shape is (n_agents,).
        """
        raise NotImplementedError("Subclasses must implement the step method.")

    def reset(self):
        """
        Resets the environment and returns the initial observation.

        Returns:
            obs (torch.Tensor): Initial observation(s).
        """
        raise NotImplementedError("Subclasses must implement the reset method.")



