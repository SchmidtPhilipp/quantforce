import torch
import torch.nn as nn

class ClipLayer(nn.Module):
    """
    A custom PyTorch layer that applies torch.clip to its input.
    """
    def __init__(self, min, max):
        super(ClipLayer, self).__init__()
        self.min_val = min
        self.max_val = max

    def forward(self, x):
        return torch.clip(x, min=self.min_val, max=self.max_val)