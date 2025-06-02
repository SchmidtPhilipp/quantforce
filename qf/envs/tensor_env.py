import torch

class TensorEnv(gym.Env):
    def step(self, actions: torch.Tensor):
        # Kernlogik der Umgebung
        obs = self._get_observation()
        rewards = self._calculate_rewards()
        done = self._check_done()
        info = {}
        return obs, rewards, done, info

    def reset(self):
        # Reset-Logik
        obs = self._get_observation()
        return obs

