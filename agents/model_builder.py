import torch.nn as nn

class ModelBuilder:
    """
    A utility class to dynamically build neural network models based on a dictionary configuration.

    Parameters:
        config (list[dict]): A list of dictionaries where each dictionary defines a layer.
                             Each dictionary should have the following keys:
                             - 'type' (str): The type of the layer (e.g., 'Linear', 'Dropout', 'BatchNorm1d').
                             - 'params' (dict): The parameters for the layer (e.g., {'in_features': 64, 'out_features': 128}).
                             - 'activation' (str, optional): The activation function to apply after the layer (e.g., 'ReLU').
    """
    def __init__(self, config):
        self.config = config

    def build(self):
        """
        Builds a sequential neural network model based on the configuration.

        Returns:
            nn.Sequential: The constructed PyTorch model.
        """
        layers = []

        for layer_config in self.config:
            # Add the layer
            layer_type = layer_config['type']
            layer_params = layer_config['params']
            layers.append(getattr(nn, layer_type)(**layer_params))

            # Add the activation function if specified
            if 'activation' in layer_config:
                layers.append(getattr(nn, layer_config['activation'])())

        return nn.Sequential(*layers)