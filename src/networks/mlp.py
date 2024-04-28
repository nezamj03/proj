import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    A configurable Multi-Layer Perceptron (MLP) that can be customized via a config dictionary.

    Parameters:
    - input_size (int): The size of the input features.
    - output_size (int): The size of the output layer.
    - config (dict): Configuration for the network architecture, including:
        - 'hidden_layers' (list of int): List specifying the size of each hidden layer.
        - 'activation' (str): The name of the activation function to use ('relu', 'tanh', etc.).
    """
    def __init__(self, input_size, output_size, config):
        super(MLP, self).__init__()
        self.config = config

        layer_sizes = [input_size] + self.config['hidden_layers'] + [output_size]
        self.layers = nn.ModuleList()

        # Create layers based on the configuration
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        # Set the activation function
        if self.config['activation'] == 'relu':
            self.activation = F.relu
        elif self.config['activation'] == 'tanh':
            self.activation = F.tanh
        elif self.config['activation'] == 'sigmoid':
            self.activation = F.sigmoid
        else:
            self.activation = F.relu  # Default to ReLU if unspecified or unrecognized

    def forward(self, x):
        """
        Defines the forward pass of the MLP.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor after passing through the MLP.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation function to all but the output layer
                x = self.activation(x)
        return x
