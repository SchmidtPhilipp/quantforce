class EpsilonScheduler:
    def __init__(self, epsilon_start=1.0, epsilon_min=0.01):
        """
        Base class for epsilon schedulers.

        Parameters:
            epsilon_start (float): Initial epsilon value.
            epsilon_min (float): Minimum epsilon value.
        """
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min

    def step(self, episode, total_episodes):
        """
        Updates epsilon based on the current episode.

        Parameters:
            episode (int): Current episode number.
            total_episodes (int): Total number of episodes.

        Returns:
            float: Updated epsilon value.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class ExponentialEpsilonScheduler(EpsilonScheduler):
    def __init__(self, epsilon_start=1.0, epsilon_min=0.01, decay_rate=0.995):
        super().__init__(epsilon_start, epsilon_min)
        self.decay_rate = decay_rate

    def step(self, episode, total_episodes):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)
        return self.epsilon

class LinearEpsilonScheduler(EpsilonScheduler):
    def __init__(self, epsilon_start=1.0, epsilon_min=0.01):
        super().__init__(epsilon_start, epsilon_min)

    def step(self, episode, total_episodes):
        decay_rate = (self.epsilon - self.epsilon_min) / total_episodes
        self.epsilon = max(self.epsilon_min, self.epsilon - decay_rate)
        return self.epsilon

class InverseSigmoidEpsilonScheduler(EpsilonScheduler):
    def __init__(self, epsilon_start=1.0, epsilon_min=0.01, k=10):
        super().__init__(epsilon_start, epsilon_min)
        self.k = k

    def step(self, episode, total_episodes):
        self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) / (1 + self.k * (episode / total_episodes))
        return self.epsilon