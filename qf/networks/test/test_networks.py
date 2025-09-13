import unittest
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from qf.networks.build_network import build_network
from qf.networks.layer_config import (
    ConvLayerConfig,
    GRULayerConfig,
    LinearLayerConfig,
    LSTMLayerConfig,
    TransformerLayerConfig,
)
from qf.networks.network_config import NetworkConfig


class TestNetworkConfig(unittest.TestCase):
    """Test cases for NetworkConfig and network building functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 10
        self.output_dim = 5

    def test_linear_network_config(self):
        """Test creating a simple linear network configuration."""
        layers = [
            LinearLayerConfig(out_features=64, activation="relu", dropout=0.1),
            LinearLayerConfig(
                out_features=32, activation="tanh", normalization="layernorm"
            ),
            LinearLayerConfig(out_features=16, activation="sigmoid"),
        ]

        config = NetworkConfig(
            layers=layers, add_output_layer=True, output_dim=self.output_dim
        )

        # Test basic properties
        self.assertEqual(len(config.layers), 3)
        self.assertTrue(config.add_output_layer)
        self.assertEqual(config.output_dim, self.output_dim)

        # Test serialization
        config_dict = config.to_dict()
        self.assertIn("layers", config_dict)
        self.assertIn("add_output_layer", config_dict)
        self.assertIn("output_dim", config_dict)

        # Test deserialization
        reconstructed_config = NetworkConfig.from_dict(config_dict)
        self.assertEqual(len(reconstructed_config.layers), 3)
        self.assertTrue(reconstructed_config.add_output_layer)
        self.assertEqual(reconstructed_config.output_dim, self.output_dim)

    def test_conv_network_config(self):
        """Test creating a convolutional network configuration."""
        layers = [
            ConvLayerConfig(
                out_channels=32,
                kernel_size=3,
                type="conv1d",
                activation="relu",
                normalization="batchnorm",
            ),
            ConvLayerConfig(
                out_channels=64,
                kernel_size=5,
                type="conv2d",
                activation="gelu",
                dropout=0.2,
            ),
        ]

        config = NetworkConfig(layers=layers)

        # Test basic properties
        self.assertEqual(len(config.layers), 2)
        self.assertFalse(config.add_output_layer)
        self.assertIsNone(config.output_dim)

        # Test layer types
        self.assertEqual(config.layers[0].type, "conv1d")
        self.assertEqual(config.layers[1].type, "conv2d")

    def test_rnn_network_config(self):
        """Test creating RNN network configurations."""
        lstm_layer = LSTMLayerConfig(
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            activation="tanh",
            dropout=0.1,
        )

        gru_layer = GRULayerConfig(
            hidden_size=64, num_layers=1, bidirectional=False, activation="relu"
        )

        config = NetworkConfig(layers=[lstm_layer, gru_layer])

        # Test basic properties
        self.assertEqual(len(config.layers), 2)
        self.assertEqual(config.layers[0].type, "lstm")
        self.assertEqual(config.layers[1].type, "gru")
        self.assertTrue(config.layers[0].bidirectional)
        self.assertFalse(config.layers[1].bidirectional)

    def test_transformer_network_config(self):
        """Test creating transformer network configuration."""
        transformer_layer = TransformerLayerConfig(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        config = NetworkConfig(layers=[transformer_layer])

        # Test basic properties
        self.assertEqual(len(config.layers), 1)
        self.assertEqual(config.layers[0].type, "transformer")
        self.assertEqual(config.layers[0].d_model, 256)
        self.assertEqual(config.layers[0].nhead, 8)

    def test_network_building_linear(self):
        """Test building a simple linear network."""
        layers = [
            LinearLayerConfig(out_features=64, activation="relu"),
            LinearLayerConfig(out_features=32, activation="tanh"),
            LinearLayerConfig(out_features=16, activation="sigmoid"),
        ]

        config = NetworkConfig(
            layers=layers, add_output_layer=True, output_dim=self.output_dim
        )

        network = build_network(self.input_dim, config)

        # Test network structure
        self.assertIsInstance(network, nn.Sequential)

        # Test forward pass
        x = torch.randn(2, self.input_dim)
        output = network(x)

        self.assertEqual(output.shape, (2, self.output_dim))

    def test_network_building_with_normalization(self):
        """Test building network with normalization layers."""
        layers = [
            LinearLayerConfig(
                out_features=64, activation="relu", normalization="layernorm"
            ),
            LinearLayerConfig(
                out_features=32, activation="tanh", normalization="batchnorm"
            ),
        ]

        config = NetworkConfig(layers=layers)
        network = build_network(self.input_dim, config)

        # Test forward pass
        x = torch.randn(2, self.input_dim)
        output = network(x)

        self.assertEqual(output.shape, (2, 32))

    def test_network_building_with_dropout(self):
        """Test building network with dropout layers."""
        layers = [
            LinearLayerConfig(out_features=64, activation="relu", dropout=0.2),
            LinearLayerConfig(out_features=32, activation="gelu", dropout=0.1),
        ]

        config = NetworkConfig(layers=layers)
        network = build_network(self.input_dim, config)

        # Test forward pass
        x = torch.randn(2, self.input_dim)
        output = network(x)

        self.assertEqual(output.shape, (2, 32))

    def test_network_building_conv1d(self):
        """Test building a 1D convolutional network."""
        layers = [
            ConvLayerConfig(
                out_channels=32,
                kernel_size=3,
                type="conv1d",
                activation="relu",
                normalization="batchnorm",
            ),
            ConvLayerConfig(
                out_channels=64, kernel_size=5, type="conv1d", activation="tanh"
            ),
        ]

        config = NetworkConfig(layers=layers)
        network = build_network(self.input_dim, config)

        # Test forward pass with appropriate input shape for conv1d
        x = torch.randn(2, self.input_dim, 100)  # (batch, channels, length)
        output = network(x)

        self.assertEqual(output.shape[0], 2)  # batch size
        self.assertEqual(output.shape[1], 64)  # output channels

    def test_network_building_lstm(self):
        """Test building an LSTM network."""
        layers = [
            LSTMLayerConfig(
                hidden_size=128,
                num_layers=2,
                bidirectional=True,
                activation="tanh",
                dropout=0.1,
            )
        ]

        config = NetworkConfig(layers=layers)
        network = build_network(self.input_dim, config)

        # Test forward pass with appropriate input shape for LSTM
        x = torch.randn(2, 10, self.input_dim)  # (batch, seq_len, features)
        output = network(x)

        # LSTM output shape should be (batch, seq_len, hidden_size * 2) for bidirectional
        self.assertEqual(output.shape[0], 2)  # batch size
        self.assertEqual(output.shape[1], 10)  # sequence length
        self.assertEqual(output.shape[2], 256)  # hidden_size * 2 for bidirectional

    def test_network_building_transformer(self):
        """Test building a transformer network."""
        layers = [
            TransformerLayerConfig(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
        ]

        config = NetworkConfig(layers=layers)
        network = build_network(self.input_dim, config)

        # Test forward pass with appropriate input shape for transformer
        x = torch.randn(2, 10, self.input_dim)  # (batch, seq_len, features)
        output = network(x)

        self.assertEqual(output.shape[0], 2)  # batch size
        self.assertEqual(output.shape[1], 10)  # sequence length
        self.assertEqual(output.shape[2], 256)  # d_model

    def test_invalid_output_layer_config(self):
        """Test that ValueError is raised when add_output_layer is True but output_dim is None."""
        layers = [LinearLayerConfig(out_features=64)]
        config = NetworkConfig(layers=layers, add_output_layer=True, output_dim=None)

        with self.assertRaises(ValueError):
            build_network(self.input_dim, config)

    def test_layer_config_serialization(self):
        """Test that layer configurations can be properly serialized and deserialized."""
        layer = LinearLayerConfig(
            out_features=64, activation="relu", normalization="layernorm", dropout=0.1
        )

        # Test serialization
        layer_dict = layer.to_dict()
        self.assertIn("out_features", layer_dict)
        self.assertIn("activation", layer_dict)
        self.assertIn("normalization", layer_dict)
        self.assertIn("dropout", layer_dict)
        self.assertIn("type", layer_dict)

        # Test deserialization
        reconstructed_layer = LinearLayerConfig.from_dict(layer_dict)
        self.assertEqual(reconstructed_layer.out_features, 64)
        self.assertEqual(reconstructed_layer.activation, "relu")
        self.assertEqual(reconstructed_layer.normalization, "layernorm")
        self.assertEqual(reconstructed_layer.dropout, 0.1)
        self.assertEqual(reconstructed_layer.type, "linear")


if __name__ == "__main__":
    unittest.main()
