"""
Base Agent Configuration Module.

This module provides the BaseAgentConfig class which serves as the foundation
for all agent configurations in the QuantForce framework. It defines common
parameters and methods that are shared across all agent types, including
serialization capabilities and device management.
"""

import json
from dataclasses import asdict, dataclass
from typing import Optional

from qf import DEFAULT_DEVICE, VERBOSITY


@dataclass
class BaseAgentConfig:
    """
    Base configuration class for all agents in the QuantForce framework.

    This class provides the foundation for all agent configurations, defining
    common parameters and methods that are shared across different agent types.
    It includes serialization capabilities for saving and loading configurations,
    as well as device management for GPU/CPU operations.

    Attributes:
        type (str): The type identifier for the agent (e.g., "sac", "ppo", "dqn").
        device (str): Computing device for agent operations ("cpu", "cuda", etc.).
        verbosity (int): Logging verbosity level for the agent.

    Example:
        >>> from qf.agents.config.base_agent_config import BaseAgentConfig
        >>>
        >>> # Create a basic configuration
        >>> config = BaseAgentConfig(type="test_agent", device="cpu")
        >>>
        >>> # Convert to JSON for storage
        >>> json_str = config.to_json()
        >>>
        >>> # Load from JSON
        >>> loaded_config = BaseAgentConfig.from_json(json_str)
    """

    type: str
    device: str = DEFAULT_DEVICE
    verbosity: int = VERBOSITY
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for JSON serialization.

        Returns:
            dict: Dictionary representation of the configuration.

        Example:
            >>> config = BaseAgentConfig(type="test_agent")
            >>> config_dict = config.to_dict()
            >>> print(config_dict['type'])  # "test_agent"
        """
        return asdict(self)

    def to_json(self) -> str:
        """
        Convert configuration to JSON string for storage or transmission.

        Returns:
            str: JSON string representation of the configuration.

        Example:
            >>> config = BaseAgentConfig(type="test_agent", device="cpu")
            >>> json_str = config.to_json()
            >>> print(json_str)  # '{"type": "test_agent", "device": "cpu", ...}'
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BaseAgentConfig":
        """
        Create configuration instance from dictionary.

        Args:
            config_dict (dict): Dictionary containing configuration parameters.

        Returns:
            BaseAgentConfig: New configuration instance.

        Example:
            >>> config_dict = {"type": "test_agent", "device": "cpu"}
            >>> config = BaseAgentConfig.from_dict(config_dict)
            >>> print(config.type)  # "test_agent"
        """
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_str: str) -> "BaseAgentConfig":
        """
        Create configuration instance from JSON string.

        Args:
            json_str (str): JSON string containing configuration parameters.

        Returns:
            BaseAgentConfig: New configuration instance.

        Example:
            >>> json_str = '{"type": "test_agent", "device": "cpu"}'
            >>> config = BaseAgentConfig.from_json(json_str)
            >>> print(config.type)  # "test_agent"
        """
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    def copy(self) -> "BaseAgentConfig":
        """
        Create a deep copy of the configuration.

        Returns a new configuration instance with the same parameters.
        This method filters out fields that are not part of the __init__
        method to ensure proper copying.

        Returns:
            BaseAgentConfig: New configuration instance with identical parameters.

        Example:
            >>> original = BaseAgentConfig(type="test_agent", device="cpu")
            >>> copied = original.copy()
            >>> print(copied.type)  # "test_agent"
            >>> print(copied is original)  # False
        """
        config_dict = self.to_dict()
        # Remove fields that are not part of __init__
        init_fields = {
            field.name
            for field in self.__class__.__dataclass_fields__.values()
            if field.init
        }
        filtered_dict = {k: v for k, v in config_dict.items() if k in init_fields}
        return self.from_dict(filtered_dict)
