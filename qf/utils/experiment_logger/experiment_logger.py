"""
Optimized experiment logger with TensorBoard and Weights & Biases support.
"""

import os
from typing import Any, Dict, Optional

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from qf.utils.logging_config import get_logger

logger = get_logger(__name__)


class ExperimentLogger:
    """Unified logger for TensorBoard and Weights & Biases."""

    def __init__(
        self,
        run_name: str = "default",
        log_dir: str = "runs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        self.run_name = run_name
        self.step = 0
        self.config = config or {}
        self.use_wandb = False
        self.wandb_run = None

        # Initialize TensorBoard
        self.tb_writer = None
        if use_tensorboard and HAS_TENSORBOARD:
            self.run_path = os.path.join(log_dir, run_name)
            os.makedirs(self.run_path, exist_ok=True)
            self.tb_writer = SummaryWriter(self.run_path)
            logger.info(f"TensorBoard logging to {self.run_path}")
        elif use_tensorboard and not HAS_TENSORBOARD:
            logger.warning(
                "TensorBoard requested but not available. Install with: pip install tensorboard"
            )

        # Initialize Weights & Biases
        if use_wandb and HAS_WANDB:
            try:
                self.wandb_run = wandb.init(
                    project=wandb_project or "quantforce",
                    entity=wandb_entity,
                    name=run_name,
                    config=self.config,
                    reinit=True,  # Allow multiple runs in same process
                )
                self.use_wandb = True
                logger.info(
                    f"W&B logging initialized for project: {wandb_project or 'quantforce'}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize W&B: {e}")
                logger.info("Continuing without W&B logging")
                self.use_wandb = False
        elif use_wandb and not HAS_WANDB:
            logger.warning(
                "W&B requested but not available. Install with: pip install wandb"
            )

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log scalar value to all enabled backends."""
        if step is None:
            step = self.step

        # Log to TensorBoard
        if self.tb_writer:
            try:
                self.tb_writer.add_scalar(name, value, step)
            except Exception as e:
                logger.warning(f"Failed to log to TensorBoard: {e}")

        # Log to W&B
        if self.use_wandb and self.wandb_run:
            try:
                wandb.log({name: value}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {e}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log multiple metrics at once with optimized batching."""
        if step is None:
            step = self.step

        # Prepare metrics for batch logging
        tb_metrics = {}
        wandb_metrics = {}

        for name, value in metrics.items():
            if isinstance(value, (list, np.ndarray)):
                # Log mean and std for arrays
                arr = np.array(value)
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr))

                tb_metrics[f"{name}_mean"] = mean_val
                tb_metrics[f"{name}_std"] = std_val
                wandb_metrics[f"{name}_mean"] = mean_val
                wandb_metrics[f"{name}_std"] = std_val
            else:
                # Convert to float to ensure serializable
                float_val = float(value) if not isinstance(value, str) else value
                tb_metrics[name] = float_val
                wandb_metrics[name] = float_val

        # Batch log to TensorBoard
        if self.tb_writer and tb_metrics:
            try:
                for name, value in tb_metrics.items():
                    self.tb_writer.add_scalar(name, value, step)
            except Exception as e:
                logger.warning(f"Failed to log metrics to TensorBoard: {e}")

        # Batch log to W&B
        if self.use_wandb and self.wandb_run and wandb_metrics:
            try:
                wandb.log(wandb_metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metrics to W&B: {e}")

    def next_step(self) -> None:
        """Increment the step counter."""
        self.step += 1

    def close(self) -> None:
        """Close all logging backends."""
        if self.tb_writer:
            try:
                self.tb_writer.close()
                logger.debug("TensorBoard writer closed")
            except Exception as e:
                logger.warning(f"Error closing TensorBoard writer: {e}")

        if self.use_wandb and self.wandb_run:
            try:
                wandb.finish()
                logger.debug("W&B run finished")
            except Exception as e:
                logger.warning(f"Error finishing W&B run: {e}")
