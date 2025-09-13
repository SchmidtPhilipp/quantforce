import numpy as np


class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2, x0=None):
        """
        Initialize the Ornstein-Uhlenbeck noise process.

        Parameters:
            size (int): Dimension of the noise.
            mu (float): Mean of the noise.
            theta (float): Speed of mean reversion.
            sigma (float): Volatility of the noise.
            dt (float): Time step.
            x0 (float or None): Initial value of the noise.
        """
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        """Reset the noise to its initial state."""
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

    def sample(self):
        """Generate a sample of OU noise."""
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        )
        self.x_prev = x
        return x
