from typing import Any

import numpy as np
from stable_baselines3.common.noise import ActionNoise


class DecayingNormalActionNoise(ActionNoise):
    def __init__(
        self, mean: np.ndarray, sigma_init: float, sigma_final: float, decay_steps: int
    ):
        """
        Gaussian action noise with linear decay over time.

        :param mean: Mean of the noise (array of zeros matching action space shape)
        :param sigma_init: Initial standard deviation
        :param sigma_final: Final standard deviation
        :param decay_steps: Number of steps over which to decay sigma
        """
        self.mean = mean
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.decay_steps = decay_steps
        self.n_steps = 0

    def __call__(self) -> np.ndarray[Any, np.dtype[np.float32]]:

        # Linearly decaying sigma
        decay_ratio = min(self.n_steps / self.decay_steps, 1.0)
        sigma = self.sigma_init * (1 - decay_ratio) + self.sigma_final * decay_ratio
        self.n_steps += 1
        return np.array(np.random.normal(self.mean, sigma))

    def reset(self):
        self.n_steps = 0
