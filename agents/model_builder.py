import torch.nn as nn
import torch.nn.init as init

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

        model = nn.Sequential(*layers)
        self._initialize_weights(model)
        return model

    def _initialize_weights(self, model):
        """
        Initializes the weights of the model using a specific initialization strategy.

        Parameters:
            model (nn.Module): The PyTorch model whose weights will be initialized.
        """
        for layer in model:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Conv2d):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.BatchNorm2d):
                init.constant_(layer.weight, 1)
                init.constant_(layer.bias, 0)