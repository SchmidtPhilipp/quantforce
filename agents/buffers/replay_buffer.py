from collections import deque
import random
import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    



class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        A replay buffer that prioritizes transitions based on their rewards.

        Parameters:
            capacity (int): Maximum number of transitions to store.
            alpha (float): Degree of prioritization (0 = uniform sampling, 1 = full prioritization).
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def store(self, transition):
        """
        Stores a transition in the buffer.

        Parameters:
            transition (tuple): A tuple (state, action, reward, next_state, done).
        """
        _, _, reward, _, _ = transition
        priority = abs(reward).mean()  # Use the absolute reward as the priority
    
        self.buffer.append(transition)
        self.priorities.append(priority)

    def sample(self, batch_size):
        """
        Samples a batch of transitions based on their priorities.

        Parameters:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: Separate lists for states, actions, rewards, next_states, and dones.
        """
        if len(self.buffer) == 0:
            raise ValueError("The replay buffer is empty!")

        # Compute probabilities based on priorities
        scaled_priorities = np.array(self.priorities) ** self.alpha
        probabilities = scaled_priorities / scaled_priorities.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)

        # Extract sampled transitions
        batch = [self.buffer[idx] for idx in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)