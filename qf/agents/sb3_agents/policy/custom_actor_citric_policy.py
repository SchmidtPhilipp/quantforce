import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from qf.networks.build_network import build_network
from qf.networks.network_config import NetworkConfig


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """Custom features extractor that uses our network building system."""

    def __init__(self, observation_space: gym.Space, network_config: NetworkConfig):
        # Calculate features dimension by building the network
        input_dim = observation_space.shape[0]

        # Call super().__init__() first
        # We need to calculate features_dim first
        temp_network = build_network(input_dim, network_config)
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim)
            output = temp_network(dummy_input)
            features_dim = output.shape[-1]

        super().__init__(observation_space, features_dim)

        # Now assign the network
        self.network = build_network(input_dim, network_config)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)
