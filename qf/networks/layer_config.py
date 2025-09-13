"""
Layer Configuration Module.

This module provides various layer configuration classes for defining
neural network architectures in the QuantForce framework. It supports
linear, convolutional, recurrent, and transformer layers with various
activation functions, normalization, and dropout options.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Literal, Optional


@dataclass
class LayerConfig(ABC):
    """
    Abstract base class for all layer configurations.

    This class provides the foundation for all layer configurations,
    defining common parameters like activation functions, normalization,
    and dropout that are shared across different layer types.

    Attributes:
        activation (Optional[str]): Activation function to apply after the layer.
            Common options: "relu", "tanh", "sigmoid", "leaky_relu", "elu".
            Default: None (no activation).
        normalization (Optional[str]): Normalization method to apply.
            Common options: "batch_norm", "layer_norm", "instance_norm".
            Default: None (no normalization).
        dropout (Optional[float]): Dropout probability for regularization.
            Range: [0, 1]. Default: None (no dropout).

    Example:
        >>> from qf.networks.layer_config import LinearLayerConfig
        >>>
        >>> # Create a linear layer with activation and dropout
        >>> layer = LinearLayerConfig(
        ...     out_features=128,
        ...     activation="relu",
        ...     dropout=0.1
        ... )
    """

    activation: Optional[str] = None
    normalization: Optional[str] = None
    dropout: Optional[float] = None

    def to_dict(self) -> dict:
        """
        Convert layer configuration to dictionary.

        Returns:
            dict: Dictionary representation of the layer configuration.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LayerConfig":
        """
        Create layer configuration from dictionary.

        Args:
            config_dict (dict): Dictionary containing layer configuration.

        Returns:
            LayerConfig: New layer configuration instance.
        """
        from .layer_factory import layer_config_from_dict

        return layer_config_from_dict(config_dict)

    @property
    @abstractmethod
    def type(self) -> str:
        """
        Get the layer type identifier.

        Returns:
            str: Layer type identifier.
        """
        pass


@dataclass
class FlattenLayerConfig(LayerConfig):
    """
    Configuration for flatten layers.

    Flatten layers are used to reshape multi-dimensional tensors
    into 2D tensors for processing by fully connected layers.

    Attributes:
        type (Literal["flatten"]): Layer type identifier.

    Example:
        >>> from qf.networks.layer_config import FlattenLayerConfig
        >>>
        >>> # Create a flatten layer
        >>> flatten = FlattenLayerConfig()
        >>> print(flatten.type)  # "flatten"
    """

    type: Literal["flatten"] = "flatten"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "FlattenLayerConfig":
        """Create from dictionary."""
        from .layer_factory import layer_config_from_dict

        return layer_config_from_dict(config_dict)


@dataclass
class LinearLayerConfig:
    """
    Configuration for linear (fully connected) layers.

    Linear layers perform affine transformations on input data.
    They are the most basic building blocks of neural networks.

    Attributes:
        out_features (int): Number of output features.
        activation (Optional[str]): Activation function to apply.
            Default: None.
        normalization (Optional[str]): Normalization method to apply.
            Default: None.
        dropout (Optional[float]): Dropout probability.
            Default: None.
        type (Literal["linear"]): Layer type identifier.

    Example:
        >>> from qf.networks.layer_config import LinearLayerConfig
        >>>
        >>> # Create a linear layer with ReLU activation
        >>> linear = LinearLayerConfig(
        ...     out_features=128,
        ...     activation="relu",
        ...     dropout=0.1
        ... )
        >>>
        >>> # Create a linear layer with batch normalization
        >>> linear_bn = LinearLayerConfig(
        ...     out_features=64,
        ...     activation="relu",
        ...     normalization="batch_norm"
        ... )
    """

    out_features: int
    activation: Optional[str] = None
    normalization: Optional[str] = None
    dropout: Optional[float] = None
    type: Literal["linear"] = "linear"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LinearLayerConfig":
        """Create from dictionary."""
        from .layer_factory import layer_config_from_dict

        return layer_config_from_dict(config_dict)


@dataclass
class ConvLayerConfig:
    """
    Configuration for convolutional layers.

    Convolutional layers are used for processing spatial data and
    can be 1D or 2D depending on the application.

    Attributes:
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution operation.
            Default: 1.
        padding (int): Padding size for the convolution.
            Default: 0.
        activation (Optional[str]): Activation function to apply.
            Default: None.
        normalization (Optional[str]): Normalization method to apply.
            Default: None.
        dropout (Optional[float]): Dropout probability.
            Default: None.
        type (Literal["conv1d", "conv2d"]): Layer type identifier.
            Default: "conv1d".

    Example:
        >>> from qf.networks.layer_config import ConvLayerConfig
        >>>
        >>> # Create a 1D convolutional layer
        >>> conv1d = ConvLayerConfig(
        ...     out_channels=64,
        ...     kernel_size=3,
        ...     activation="relu"
        ... )
        >>>
        >>> # Create a 2D convolutional layer
        >>> conv2d = ConvLayerConfig(
        ...     out_channels=32,
        ...     kernel_size=5,
        ...     type="conv2d",
        ...     activation="relu"
        ... )
    """

    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    activation: Optional[str] = None
    normalization: Optional[str] = None
    dropout: Optional[float] = None
    type: Literal["conv1d", "conv2d"] = "conv1d"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ConvLayerConfig":
        """Create from dictionary."""
        from .layer_factory import layer_config_from_dict

        return layer_config_from_dict(config_dict)


@dataclass
class RNNLayerConfig(ABC):
    """
    Abstract base class for recurrent neural network layers.

    This class provides the foundation for RNN layer configurations,
    defining common parameters like hidden size, number of layers,
    and bidirectional processing.

    Attributes:
        hidden_size (int): Number of hidden units in the RNN.
        num_layers (int): Number of RNN layers to stack.
            Default: 1.
        bidirectional (bool): Whether to use bidirectional processing.
            Default: False.
        activation (Optional[str]): Activation function to apply.
            Default: None.
        normalization (Optional[str]): Normalization method to apply.
            Default: None.
        dropout (Optional[float]): Dropout probability.
            Default: None.

    Example:
        >>> from qf.networks.layer_config import LSTMLayerConfig
        >>>
        >>> # Create an LSTM layer
        >>> lstm = LSTMLayerConfig(
        ...     hidden_size=128,
        ...     num_layers=2,
        ...     bidirectional=True
        ... )
    """

    hidden_size: int
    num_layers: int = 1
    bidirectional: bool = False
    activation: Optional[str] = None
    normalization: Optional[str] = None
    dropout: Optional[float] = None

    @property
    @abstractmethod
    def type(self) -> str:
        """
        Get the RNN layer type identifier.

        Returns:
            str: RNN layer type identifier.
        """
        pass

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RNNLayerConfig":
        """Create from dictionary."""
        from .layer_factory import layer_config_from_dict

        return layer_config_from_dict(config_dict)


@dataclass
class LSTMLayerConfig(RNNLayerConfig):
    """
    Configuration for Long Short-Term Memory (LSTM) layers.

    LSTM layers are a type of recurrent neural network that can
    learn long-term dependencies in sequential data.

    Attributes:
        type (Literal["lstm"]): Layer type identifier.

    Example:
        >>> from qf.networks.layer_config import LSTMLayerConfig
        >>>
        >>> # Create a simple LSTM layer
        >>> lstm = LSTMLayerConfig(hidden_size=128)
        >>>
        >>> # Create a bidirectional LSTM layer
        >>> bi_lstm = LSTMLayerConfig(
        ...     hidden_size=256,
        ...     num_layers=3,
        ...     bidirectional=True,
        ...     dropout=0.2
        ... )
    """

    type: Literal["lstm"] = "lstm"


@dataclass
class GRULayerConfig(RNNLayerConfig):
    """
    Configuration for Gated Recurrent Unit (GRU) layers.

    GRU layers are a simplified version of LSTM that can also
    learn long-term dependencies but with fewer parameters.

    Attributes:
        type (Literal["gru"]): Layer type identifier.

    Example:
        >>> from qf.networks.layer_config import GRULayerConfig
        >>>
        >>> # Create a simple GRU layer
        >>> gru = GRULayerConfig(hidden_size=128)
        >>>
        >>> # Create a bidirectional GRU layer
        >>> bi_gru = GRULayerConfig(
        ...     hidden_size=256,
        ...     num_layers=2,
        ...     bidirectional=True,
        ...     dropout=0.1
        ... )
    """

    type: Literal["gru"] = "gru"


@dataclass
class TransformerLayerConfig:
    """
    Configuration for Transformer layers.

    Transformer layers implement the attention mechanism and are
    particularly effective for processing sequential data.

    Attributes:
        d_model (int): Embedding dimension (model dimension).
        nhead (int): Number of attention heads.
        dim_feedforward (int): Size of the feedforward network within the block.
            Default: 2048.
        dropout (float): Dropout probability.
            Default: 0.1.
        batch_first (bool): Whether the batch dimension comes first.
            Important for compatibility with (B, T, D) format.
            Default: True.
        activation (Literal["relu", "gelu"]): Activation function for feedforward.
            Default: "relu".
        normalization (Optional[str]): Normalization method to apply.
            Default: None.
        type (Literal["transformer"]): Layer type identifier.

    Example:
        >>> from qf.networks.layer_config import TransformerLayerConfig
        >>>
        >>> # Create a basic transformer layer
        >>> transformer = TransformerLayerConfig(
        ...     d_model=512,
        ...     nhead=8
        ... )
        >>>
        >>> # Create a transformer layer with custom settings
        >>> custom_transformer = TransformerLayerConfig(
        ...     d_model=256,
        ...     nhead=4,
        ...     dim_feedforward=1024,
        ...     dropout=0.2,
        ...     activation="gelu"
        ... )
    """

    d_model: int  # Embedding-Dimension
    nhead: int  # Anzahl der Attention-Heads
    dim_feedforward: int = 2048  # Größe des FFN innerhalb des Blocks
    dropout: float = 0.1  # Dropout-Wahrscheinlichkeit
    batch_first: bool = True  # Wichtig für Kompatibilität mit (B, T, D)
    activation: Literal["relu", "gelu"] = "relu"
    normalization: Optional[str] = None
    type: Literal["transformer"] = "transformer"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TransformerLayerConfig":
        """Create from dictionary."""
        from .layer_factory import layer_config_from_dict

        return layer_config_from_dict(config_dict)
