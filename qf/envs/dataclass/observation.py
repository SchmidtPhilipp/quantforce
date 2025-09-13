from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch


@dataclass
class Observation:
    date: datetime
    observations: torch.Tensor  # shape: (n_agents, obs_dim)
    actions: Optional[torch.Tensor] = None  # shape: (n_agents, n_assets+1)
    portfolio: Optional[torch.Tensor] = None  # shape: (n_agents, n_assets)
    cash: Optional[torch.Tensor] = None  # shape: (n_agents, 1)

    def as_tensor(self) -> torch.Tensor:
        parts = [self.observations]
        if self.actions is not None:
            parts.append(self.actions)
        if self.portfolio is not None:
            parts.append(self.portfolio)
        if self.cash is not None:
            parts.append(self.cash)
        return torch.cat(parts, dim=1)
