from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


class TensorView:
    """
    Wraps a tensor to support chained operations like .mean(), .std(), etc.
    with NaN-safe reductions and custom plotting.
    """

    def __init__(self, data: torch.Tensor):
        self.data = data

    def mean(self, *args, **kwargs):
        return TensorView(torch.nanmean(self.data, *args, **kwargs))

    def std(self, *args, **kwargs):
        # Use torch.std with nan handling for older PyTorch versions
        if hasattr(torch, "nanstd"):
            return TensorView(torch.nanstd(self.data, *args, **kwargs))
        else:
            # Fallback for older PyTorch versions
            mask = ~torch.isnan(self.data)
            if mask.any():
                return TensorView(torch.std(self.data[mask], *args, **kwargs))
            else:
                return TensorView(torch.tensor(float("nan")))

    def sum(self, *args, **kwargs):
        return TensorView(torch.nansum(self.data, *args, **kwargs))

    def max(self, *args, **kwargs):
        result = torch.max(self.data, *args, **kwargs)
        if isinstance(result, tuple):  # Handle case where max returns (values, indices)
            return TensorView(result.values)
        return TensorView(result)

    def min(self, *args, **kwargs):
        result = torch.min(self.data, *args, **kwargs)
        if isinstance(result, tuple):  # Handle case where min returns (values, indices)
            return TensorView(result.values)
        return TensorView(result)

    def __getitem__(self, key):
        return TensorView(self.data[key])

    def __repr__(self):
        return f"TensorView({self.data})"

    def tensor(self):
        return self.data
