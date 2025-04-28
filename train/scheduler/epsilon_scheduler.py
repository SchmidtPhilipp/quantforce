import math

class EpsilonScheduler:
    def __init__(self, epsilon_start=1.0, epsilon_min=0.01):
        """
        Base class for epsilon schedulers.

        Parameters:
            epsilon_start (float): Initial epsilon value.
            epsilon_min (float): Minimum epsilon value.
        """
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.current_step = 0  # Internal counter to track steps

    def step(self):
        """
        Updates epsilon based on the current step.

        Returns:
            float: Updated epsilon value.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def reset(self):
        """
        Resets the scheduler to its initial state.
        """
        self.epsilon = self.epsilon_start
        self.current_step = 0

class ExponentialEpsilonScheduler(EpsilonScheduler):
    def __init__(self, epsilon_start=1.0, epsilon_min=0.01, decay_rate=0.995):
        super().__init__(epsilon_start, epsilon_min)
        self.decay_rate = decay_rate

    def step(self):
        """
        Updates epsilon using exponential decay.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)
        self.current_step += 1
        return self.epsilon

class LinearEpsilonScheduler(EpsilonScheduler):
    def __init__(self, epsilon_start=1.0, epsilon_min=0.01, total_steps=1000):
        """
        Linear epsilon scheduler.

        Parameters:
            total_steps (int): Total number of steps for epsilon to decay linearly.
        """
        super().__init__(epsilon_start, epsilon_min)
        self.total_steps = total_steps
        self.decay_rate = (self.epsilon_start - self.epsilon_min) / total_steps

    def step(self):
        """
        Updates epsilon using linear decay.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon - self.decay_rate)
        self.current_step += 1
        return self.epsilon

class InverseSigmoidEpsilonScheduler(EpsilonScheduler):
    def __init__(self, epsilon_start=1.0, epsilon_min=0.01, k=10):
        """
        Inverse sigmoid epsilon scheduler.

        Parameters:
            k (float): Controls the steepness of the sigmoid curve.
        """
        super().__init__(epsilon_start, epsilon_min)
        self.k = k

    def step(self):
        """
        Updates epsilon using an inverse sigmoid function.
        """
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) / (1 + self.k * self.current_step)
        self.current_step += 1
        return self.epsilon

class ExponentialRestartEpsilonScheduler(EpsilonScheduler):
    def __init__(self, epsilon_start=1.0, epsilon_min=0.01, decay_rate=0.995, period=100):
        """
        Exponential epsilon scheduler with periodic restarts.

        Parameters:
            decay_rate (float): Exponential decay rate.
            period (int): Number of steps before restarting epsilon.
        """
        super().__init__(epsilon_start, epsilon_min)
        self.decay_rate = decay_rate
        self.period = period

    def step(self):
        """
        Updates epsilon using exponential decay with periodic restarts.
        """
        if self.current_step % self.period == 0:
            self.epsilon = self.epsilon_start
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)
        self.current_step += 1
        return self.epsilon

class PeriodicEpsilonScheduler(EpsilonScheduler):
    def __init__(self, epsilon_start=1.0, epsilon_min=0.01, period=100, function="cos"):
        """
        Periodic epsilon scheduler based on cos² or sin².

        Parameters:
            period (int): Number of steps for one full oscillation.
            function (str): Periodic function to use ("cos" or "sin").
        """
        super().__init__(epsilon_start, epsilon_min)
        self.period = period
        self.function = function.lower()
        assert self.function in ["cos", "sin"], "Function must be 'cos' or 'sin'."

    def step(self):
        """
        Updates epsilon using a periodic function.
        """
        # Calculate the phase of the periodic function
        phase = (2 * math.pi * self.current_step) / self.period

        # Compute the periodic value (cos² or sin²)
        if self.function == "cos":
            periodic_value = math.cos(phase) ** 2
        elif self.function == "sin":
            periodic_value = math.sin(phase) ** 2

        # Normalize and scale epsilon
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * periodic_value
        self.current_step += 1
        return self.epsilon