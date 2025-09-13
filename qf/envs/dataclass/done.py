from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch


@dataclass
class Done:
    flags: torch.Tensor = field(repr=False)

    def __init__(
        self,
        flags: Union[torch.Tensor, np.ndarray, bool, float, int],
        n_agents: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if isinstance(flags, torch.Tensor):
            self.flags = flags
        elif isinstance(flags, np.ndarray):
            self.flags = torch.from_numpy(flags)
        elif isinstance(flags, (bool, float, int)):
            if n_agents is None:
                raise ValueError(
                    "n_agents must be specified when initializing Done with a scalar."
                )
            self.flags = torch.full(
                (n_agents,),
                float(flags),
                dtype=torch.float32,
                device=device if device is not None else "cpu",
            )
        else:
            raise TypeError(f"Unsupported type for flags: {type(flags)}")

    def __bool__(self):
        # True if any agent is done
        return bool(torch.any(self.flags > 0))

    def __setattr__(self, name, value):
        # Allow normal attribute setting except for 'flags' with bool/float/int/np.ndarray/torch.Tensor
        if name == "flags":
            object.__setattr__(self, name, value)
        elif name == "done":
            # Support setting 'done' as an alias for flags
            self._set_flags_from_scalar(value)
        else:
            object.__setattr__(self, name, value)

    def __setitem__(self, key, value):
        # Allow item assignment to flags
        self.flags[key] = value

    def __eq__(self, other):
        # Allow comparison with bool, int, float, np.ndarray, torch.Tensor, or Done
        if isinstance(other, Done):
            return torch.equal(self.flags, other.flags)
        elif isinstance(other, (bool, int, float)):
            return bool(self) == bool(other)
        elif isinstance(other, (np.ndarray, torch.Tensor)):
            return np.allclose(self.to_multi_agent_numpy(), np.array(other))
        return False

    def _set_flags_from_scalar(self, value):
        # Helper to set all flags from a scalar
        if isinstance(value, (bool, float, int)):
            self.flags.fill_(float(value))
        elif isinstance(value, np.ndarray):
            self.flags = torch.from_numpy(value)
        elif isinstance(value, torch.Tensor):
            self.flags = value
        else:
            raise TypeError(f"Unsupported type for setting flags: {type(value)}")

    def __assign__(self, value):
        # Support assignment: done = True/False/array/tensor
        self._set_flags_from_scalar(value)

    def __call__(self, value):
        # Support done(True) or done(False)
        self._set_flags_from_scalar(value)

    def __ior__(self, value):
        # Support done |= True/False
        self._set_flags_from_scalar(value)
        return self

    def __set__(self, instance, value):
        # For dataclass compatibility
        self._set_flags_from_scalar(value)

    def to_stable_baselines3_numpy(self) -> np.ndarray:
        return self.flags.squeeze(0).cpu().numpy()

    def to_single_agent(self, agent_id: int) -> bool:
        return bool(self.flags[agent_id].item())

    def as_tensor(self) -> torch.Tensor:
        return self.flags
