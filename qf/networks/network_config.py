"""
Network Configuration Module.

This module provides the NetworkConfig class which defines the configuration
for neural network architectures in the QuantForce framework. It supports
various layer types and provides serialization capabilities for saving
and loading network configurations.
"""

from dataclasses import dataclass
from typing import List, Optional

from qf.networks.layer_config import LayerConfig


@dataclass
class NetworkConfig:
    """
    Configuration for neural network architecture.

    This class defines the configuration for neural network architectures
    used in reinforcement learning agents. It supports various layer types
    including linear, convolutional, recurrent, and transformer layers.

    The NetworkConfig provides a flexible way to define complex network
    architectures while maintaining serialization capabilities for saving
    and loading configurations.

    Attributes:
        layers (List[LayerConfig]): List of layer configurations defining
            the network architecture. Each layer can be of different types
            (linear, conv, lstm, gru, transformer, etc.).
        output_dim (Optional[int]): Output dimension for the final layer.
            If None, no output layer is added. Otherwise, specifies the
            output dimension. Default: None.

    Example:
        >>> from qf.networks.network_config import NetworkConfig
        >>> from qf.networks.layer_config import LinearLayerConfig, LSTMLayerConfig
        >>>
        >>> # Create a simple feedforward network
        >>> layers = [
        ...     LinearLayerConfig(out_features=128, activation="relu"),
        ...     LinearLayerConfig(out_features=64, activation="relu"),
        ...     LinearLayerConfig(out_features=32, activation="relu")
        ... ]
        >>> config = NetworkConfig(layers=layers, output_dim=10)
        >>>
        >>> # Create a recurrent network
        >>> layers = [
        ...     LSTMLayerConfig(hidden_size=128, num_layers=2),
        ...     LinearLayerConfig(out_features=64, activation="relu"),
        ...     LinearLayerConfig(out_features=32, activation="relu")
        ... ]
        >>> rnn_config = NetworkConfig(layers=layers, output_dim=5)
        >>>
        >>> # Create a network without output layer
        >>> feature_config = NetworkConfig(layers=layers, output_dim=None)
    """

    layers: List[LayerConfig]
    output_dim: Optional[int] = (
        None  # None means no output layer, otherwise specifies output dimension
    )

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for serialization.

        Converts the network configuration to a dictionary format suitable
        for JSON serialization, including all layer configurations.

        Returns:
            dict: Dictionary representation of the network configuration.

        Example:
            >>> config = NetworkConfig(layers=[], output_dim=10)
            >>> config_dict = config.to_dict()
            >>> print("layers" in config_dict)  # True
            >>> print("output_dim" in config_dict)  # True
        """
        return {
            "layers": [layer.to_dict() for layer in self.layers],
            "output_dim": self.output_dim,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "NetworkConfig":
        """
        Create network configuration from dictionary.

        Reconstructs a NetworkConfig object from a dictionary representation,
        including proper reconstruction of all layer configurations.

        Args:
            config_dict (dict): Dictionary containing network configuration.
                Must contain "layers" key with list of layer dictionaries.

        Returns:
            NetworkConfig: New network configuration instance.

        Example:
            >>> config_dict = {
            ...     "layers": [
            ...         {"type": "linear", "out_features": 128, "activation": "relu"}
            ...     ],
            ...     "output_dim": 10
            ... }
            >>> config = NetworkConfig.from_dict(config_dict)
            >>> print(len(config.layers))  # 1
        """
        from qf.networks.layer_factory import layer_config_from_dict

        layers = [
            layer_config_from_dict(layer_dict) for layer_dict in config_dict["layers"]
        ]
        return cls(layers=layers, output_dim=config_dict.get("output_dim"))
