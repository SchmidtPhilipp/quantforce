from typing import Optional

import torch
import torch.nn as nn

from qf.networks.layer_config import (
    ConvLayerConfig,
    FlattenLayerConfig,
    GRULayerConfig,
    LinearLayerConfig,
    LSTMLayerConfig,
    TransformerLayerConfig,
)
from qf.networks.network_config import NetworkConfig


def build_network_with_features(input_dim, config) -> nn.Module:
    feature_extractor = None
    if hasattr(config, "feature_extractor_config") and config.feature_extractor_config:
        feature_extractor = build_feature_extractor(
            input_dim, config.feature_extractor_config
        )
        input_dim = config.feature_extractor_config.output_dim
    mlp = build_network(input_dim, config)
    if feature_extractor:
        return nn.Sequential(feature_extractor, mlp)
    return mlp


def build_network(input_dim: int, config: NetworkConfig) -> nn.Module:
    """
    Build a neural network from configuration.

    Args:
        input_dim: Input dimension
        config: Network configuration

    Returns:
        PyTorch neural network module
    """
    layers = []
    current_dim = input_dim

    # Build layers from config
    for layer_cfg in config.layers:
        # FLATTEN
        if isinstance(layer_cfg, FlattenLayerConfig):
            layers.append(nn.Flatten())
            current_dim = None
            continue

        # LINEAR
        if isinstance(layer_cfg, LinearLayerConfig):
            layers.append(nn.Linear(current_dim, layer_cfg.out_features))
            current_dim = layer_cfg.out_features

            # Add activation
            if layer_cfg.activation == "relu":
                layers.append(nn.ReLU())
            elif layer_cfg.activation == "tanh":
                layers.append(nn.Tanh())
            elif layer_cfg.activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif layer_cfg.activation == "gelu":
                layers.append(nn.GELU())

            # Add normalization
            if layer_cfg.normalization == "layernorm":
                layers.append(nn.LayerNorm(current_dim))
            elif layer_cfg.normalization == "batchnorm":
                layers.append(nn.BatchNorm1d(current_dim))

            # Add dropout
            if layer_cfg.dropout is not None:
                layers.append(nn.Dropout(layer_cfg.dropout))

        # CONV1D
        elif isinstance(layer_cfg, ConvLayerConfig) and layer_cfg.type == "conv1d":
            layers.append(
                nn.Conv1d(
                    in_channels=current_dim,
                    out_channels=layer_cfg.out_channels,
                    kernel_size=layer_cfg.kernel_size,
                    padding=layer_cfg.kernel_size // 2,
                )
            )
            current_dim = layer_cfg.out_channels

            # Add activation
            if layer_cfg.activation == "relu":
                layers.append(nn.ReLU())
            elif layer_cfg.activation == "tanh":
                layers.append(nn.Tanh())
            elif layer_cfg.activation == "gelu":
                layers.append(nn.GELU())

            # Add normalization
            if layer_cfg.normalization == "batchnorm":
                layers.append(nn.BatchNorm1d(current_dim))
            elif layer_cfg.normalization == "layernorm":
                layers.append(nn.LayerNorm(current_dim))

            # Add dropout
            if layer_cfg.dropout is not None:
                layers.append(nn.Dropout(layer_cfg.dropout))

        # CONV2D
        elif isinstance(layer_cfg, ConvLayerConfig) and layer_cfg.type == "conv2d":
            layers.append(
                nn.Conv2d(
                    in_channels=current_dim,
                    out_channels=layer_cfg.out_channels,
                    kernel_size=layer_cfg.kernel_size,
                    padding=layer_cfg.kernel_size // 2,
                )
            )
            current_dim = layer_cfg.out_channels

            # Add activation
            if layer_cfg.activation == "relu":
                layers.append(nn.ReLU())
            elif layer_cfg.activation == "tanh":
                layers.append(nn.Tanh())
            elif layer_cfg.activation == "gelu":
                layers.append(nn.GELU())

            # Add normalization
            if layer_cfg.normalization == "batchnorm":
                layers.append(nn.BatchNorm2d(current_dim))

            # Add dropout
            if layer_cfg.dropout is not None:
                layers.append(nn.Dropout(layer_cfg.dropout))

        # LSTM
        elif isinstance(layer_cfg, LSTMLayerConfig):
            rnn = nn.LSTM(
                input_size=current_dim,
                hidden_size=layer_cfg.hidden_size,
                num_layers=layer_cfg.num_layers,
                batch_first=True,
                bidirectional=layer_cfg.bidirectional,
            )

            # Create a wrapper to handle the tuple output from LSTM
            class LSTMWrapper(nn.Module):
                def __init__(self, lstm):
                    super().__init__()
                    self.lstm = lstm

                def forward(self, x):
                    output, _ = self.lstm(x)
                    return output

            layers.append(LSTMWrapper(rnn))
            current_dim = layer_cfg.hidden_size * (2 if layer_cfg.bidirectional else 1)

        # GRU
        elif isinstance(layer_cfg, GRULayerConfig):
            rnn = nn.GRU(
                input_size=current_dim,
                hidden_size=layer_cfg.hidden_size,
                num_layers=layer_cfg.num_layers,
                batch_first=True,
                bidirectional=layer_cfg.bidirectional,
            )

            # Create a wrapper to handle the tuple output from GRU
            class GRUWrapper(nn.Module):
                def __init__(self, gru):
                    super().__init__()
                    self.gru = gru

                def forward(self, x):
                    output, _ = self.gru(x)
                    return output

            layers.append(GRUWrapper(rnn))
            current_dim = layer_cfg.hidden_size * (2 if layer_cfg.bidirectional else 1)

        # TRANSFORMER
        elif isinstance(layer_cfg, TransformerLayerConfig):
            # Add projection layer if input dimension doesn't match d_model
            if current_dim != layer_cfg.d_model:
                layers.append(nn.Linear(current_dim, layer_cfg.d_model))
                current_dim = layer_cfg.d_model

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=layer_cfg.d_model,
                nhead=layer_cfg.nhead,
                dim_feedforward=layer_cfg.dim_feedforward,
                dropout=layer_cfg.dropout,
                activation=layer_cfg.activation,
                batch_first=layer_cfg.batch_first,
            )
            layers.append(nn.TransformerEncoder(encoder_layer, num_layers=1))
            current_dim = layer_cfg.d_model

    # Add output layer if specified
    if config.output_dim is not None:
        layers.append(nn.Linear(current_dim, config.output_dim))
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)


# Aliases for backward compatibility
def build_feature_extractor(input_dim: int, config: NetworkConfig) -> nn.Module:
    """Alias for build_network."""
    return build_network(input_dim, config)


def build_actor_network(input_dim: int, config: NetworkConfig) -> nn.Module:
    """Alias for build_network."""
    return build_network(input_dim, config)


def build_critic_network(input_dim: int, config: NetworkConfig) -> nn.Module:
    """Alias for build_network."""
    return build_network(input_dim, config)
