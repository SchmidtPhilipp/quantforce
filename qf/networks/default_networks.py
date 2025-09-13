"""
Default network configurations for different agent types.

This module provides pre-configured network architectures that work well
for different reinforcement learning algorithms and problem types.
"""

from typing import Any, Dict

from qf.networks.layer_config import (
    ConvLayerConfig,
    FlattenLayerConfig,
    GRULayerConfig,
    LinearLayerConfig,
    LSTMLayerConfig,
    TransformerLayerConfig,
)
from qf.networks.network_config import NetworkConfig


class DefaultNetworks:
    """Collection of default network configurations for different agent types."""

    # Standard Actor Network (for all actor-critic methods)
    @staticmethod
    def get_standard_actor() -> NetworkConfig:
        """Standard actor network for all actor-critic agents (SAC, PPO, DDPG, A2C)."""
        return NetworkConfig(
            layers=[
                LinearLayerConfig(
                    out_features=64, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=64, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=None,  # Will be set based on action space
        )

    # Standard Critic Network (for all actor-critic methods)
    @staticmethod
    def get_standard_critic() -> NetworkConfig:
        """Standard critic network for all actor-critic agents (SAC, PPO, DDPG, A2C)."""
        return NetworkConfig(
            layers=[
                LinearLayerConfig(
                    out_features=64, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=64, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=1,  # Q-value or value function
        )

    # Standard Critic Network for Value-Based Methods (DQN, SPQL)
    @staticmethod
    def get_standard_value_critic() -> NetworkConfig:
        """Standard critic network for value-based agents (DQN, SPQL)."""
        return NetworkConfig(
            layers=[
                LinearLayerConfig(
                    out_features=64, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=64, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=None,  # Will be set based on action space
        )

    # Feature Extractors for different input types
    @staticmethod
    def get_default_feature_extractor() -> NetworkConfig:
        """Default (flat) feature extractor for all agents."""
        return NetworkConfig(
            layers=[FlattenLayerConfig()],
            output_dim=None,  # No output layer for feature extractors
        )

    @staticmethod
    def get_linear_feature_extractor() -> NetworkConfig:
        """Linear feature extractor for tabular/vector data."""
        return NetworkConfig(
            layers=[
                LinearLayerConfig(
                    out_features=128, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=64, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=None,  # No output layer for feature extractors
        )

    @staticmethod
    def get_small_linear_feature_extractor() -> NetworkConfig:
        """Small linear feature extractor for tabular/vector data."""
        return NetworkConfig(
            layers=[
                LinearLayerConfig(
                    out_features=256, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=None,  # No output layer for feature extractors
        )

    @staticmethod
    def get_medium_linear_feature_extractor() -> NetworkConfig:
        """Medium linear feature extractor for tabular/vector data."""
        return NetworkConfig(
            layers=[
                LinearLayerConfig(
                    out_features=256, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=512, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=None,  # No output layer for feature extractors
        )

    @staticmethod
    def get_large_linear_feature_extractor() -> NetworkConfig:
        """Large linear feature extractor for tabular/vector data."""
        return NetworkConfig(
            layers=[
                LinearLayerConfig(
                    out_features=512, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=1024, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=2048, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=None,  # No output layer for feature extractors
        )

    @staticmethod
    def get_lstm_feature_extractor() -> NetworkConfig:
        """LSTM feature extractor for sequential data."""
        return NetworkConfig(
            layers=[
                LSTMLayerConfig(
                    hidden_size=64,
                    num_layers=1,
                    bidirectional=True,
                    dropout=0.1,  # Keep dropout for RNN layers
                ),
                LinearLayerConfig(
                    out_features=64, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=None,  # No output layer for feature extractors
        )

    @staticmethod
    def get_transformer_feature_extractor() -> NetworkConfig:
        """Transformer feature extractor for complex sequences."""
        return NetworkConfig(
            layers=[
                TransformerLayerConfig(
                    d_model=128,
                    nhead=8,
                    dim_feedforward=512,
                    dropout=0.1,  # Keep dropout for transformer layers
                    activation="gelu",
                ),
                LinearLayerConfig(
                    out_features=64, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=None,  # No output layer for feature extractors
        )

    @staticmethod
    def get_small_transformer_feature_extractor() -> NetworkConfig:
        """Small transformer feature extractor (~50K parameters)."""
        return NetworkConfig(
            layers=[
                TransformerLayerConfig(
                    d_model=64,
                    nhead=4,
                    dim_feedforward=256,
                    dropout=0.1,
                    activation="gelu",
                ),
                LinearLayerConfig(
                    out_features=32, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=None,  # No output layer for feature extractors
        )

    @staticmethod
    def get_medium_transformer_feature_extractor() -> NetworkConfig:
        """Medium transformer feature extractor (~200K parameters)."""
        return NetworkConfig(
            layers=[
                TransformerLayerConfig(
                    d_model=128,
                    nhead=8,
                    dim_feedforward=512,
                    dropout=0.1,
                    activation="gelu",
                ),
                TransformerLayerConfig(
                    d_model=128,
                    nhead=8,
                    dim_feedforward=512,
                    dropout=0.1,
                    activation="gelu",
                ),
                LinearLayerConfig(
                    out_features=64, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=None,  # No output layer for feature extractors
        )

    @staticmethod
    def get_large_transformer_feature_extractor() -> NetworkConfig:
        """Large transformer feature extractor (~800K parameters)."""
        return NetworkConfig(
            layers=[
                TransformerLayerConfig(
                    d_model=256,
                    nhead=8,
                    dim_feedforward=1024,
                    dropout=0.1,
                    activation="gelu",
                ),
                TransformerLayerConfig(
                    d_model=256,
                    nhead=8,
                    dim_feedforward=1024,
                    dropout=0.1,
                    activation="gelu",
                ),
                TransformerLayerConfig(
                    d_model=256,
                    nhead=8,
                    dim_feedforward=1024,
                    dropout=0.1,
                    activation="gelu",
                ),
                LinearLayerConfig(
                    out_features=128, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=64, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=None,  # No output layer for feature extractors
        )

    @staticmethod
    def get_cnn_feature_extractor() -> NetworkConfig:
        """Convolutional feature extractor for spatial patterns."""
        return NetworkConfig(
            layers=[
                ConvLayerConfig(
                    out_channels=32,
                    kernel_size=3,
                    type="conv1d",
                    activation="relu",
                    normalization="batchnorm",  # Keep BatchNorm for Conv layers
                ),
                ConvLayerConfig(
                    out_channels=64,
                    kernel_size=3,
                    type="conv1d",
                    activation="relu",
                    normalization="batchnorm",  # Keep BatchNorm for Conv layers
                ),
                LinearLayerConfig(
                    out_features=128, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=64, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=None,  # No output layer for feature extractors
        )

    # Advanced architectures for specific use cases

    @staticmethod
    def get_deep_actor() -> NetworkConfig:
        """Deep actor network for complex environments."""
        return NetworkConfig(
            layers=[
                LinearLayerConfig(
                    out_features=512, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=256, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=256, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=128, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=64, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=None,
        )

    @staticmethod
    def get_deep_critic() -> NetworkConfig:
        """Deep critic network for complex environments."""
        return NetworkConfig(
            layers=[
                LinearLayerConfig(
                    out_features=512, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=256, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=256, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=128, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=64, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=1,
        )

    @staticmethod
    def get_wide_actor() -> NetworkConfig:
        """Wide actor network for high-dimensional inputs."""
        return NetworkConfig(
            layers=[
                LinearLayerConfig(
                    out_features=1024, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=512, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=256, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=None,
        )

    @staticmethod
    def get_wide_critic() -> NetworkConfig:
        """Wide critic network for high-dimensional inputs."""
        return NetworkConfig(
            layers=[
                LinearLayerConfig(
                    out_features=1024, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=512, activation="relu", normalization="layernorm"
                ),
                LinearLayerConfig(
                    out_features=256, activation="relu", normalization="layernorm"
                ),
            ],
            output_dim=1,
        )

    # Utility methods

    @staticmethod
    def get_feature_extractors() -> Dict[str, NetworkConfig]:
        """Get all available feature extractor configurations."""
        return {
            "linear": DefaultNetworks.get_linear_feature_extractor(),
            "lstm": DefaultNetworks.get_lstm_feature_extractor(),
            "transformer": DefaultNetworks.get_transformer_feature_extractor(),
            "transformer_small": DefaultNetworks.get_small_transformer_feature_extractor(),
            "transformer_medium": DefaultNetworks.get_medium_transformer_feature_extractor(),
            "transformer_large": DefaultNetworks.get_large_transformer_feature_extractor(),
            "cnn": DefaultNetworks.get_cnn_feature_extractor(),
        }

    @staticmethod
    def get_actor_networks() -> Dict[str, NetworkConfig]:
        """Get all available actor network configurations."""
        return {
            "standard": DefaultNetworks.get_standard_actor(),
            "deep": DefaultNetworks.get_deep_actor(),
            "wide": DefaultNetworks.get_wide_actor(),
        }

    @staticmethod
    def get_critic_networks() -> Dict[str, NetworkConfig]:
        """Get all available critic network configurations."""
        return {
            "standard": DefaultNetworks.get_standard_critic(),
            "value_based": DefaultNetworks.get_standard_value_critic(),
            "deep": DefaultNetworks.get_deep_critic(),
            "wide": DefaultNetworks.get_wide_critic(),
        }

    @staticmethod
    def get_all_defaults() -> Dict[str, Dict[str, NetworkConfig]]:
        """Get all default network configurations organized by type."""
        return {
            "actors": DefaultNetworks.get_actor_networks(),
            "critics": DefaultNetworks.get_critic_networks(),
            "feature_extractors": DefaultNetworks.get_feature_extractors(),
        }
