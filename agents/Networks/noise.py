import numpy as np

class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        """
        Parameters:
            size (int): Dimension of the noise.
            mu (float): Long-running mean.
            theta (float): Speed of mean reversion.
            sigma (float): Volatility parameter.
            dt (float): Time step size.
        """
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        self.x_prev = np.zeros(self.size)

    def sample(self):
        """Generate next noise value."""
        dx = self.theta * (self.mu - self.x_prev) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        self.x_prev += dx
        return self.x_prev
