from typing import Type

from qf.networks.layer_config import (
    ConvLayerConfig,
    FlattenLayerConfig,
    GRULayerConfig,
    LayerConfig,
    LinearLayerConfig,
    LSTMLayerConfig,
    TransformerLayerConfig,
)

LAYER_TYPE_MAP: dict[str, Type[LayerConfig]] = {
    "flatten": FlattenLayerConfig,
    "linear": LinearLayerConfig,
    "conv1d": ConvLayerConfig,
    "conv2d": ConvLayerConfig,
    "lstm": LSTMLayerConfig,
    "gru": GRULayerConfig,
    "transformer": TransformerLayerConfig,
}


def layer_config_from_dict(d: dict) -> LayerConfig:
    layer_type = d.get("type")
    cls = LAYER_TYPE_MAP.get(layer_type)
    if cls is None:
        raise ValueError(f"Unbekannter Layer-Typ '{layer_type}'")
    return cls(**d)
