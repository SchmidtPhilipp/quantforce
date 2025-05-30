from torch.utils.tensorboard import SummaryWriter
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

DEFAULT_LOG_DIR = "runs"

class Logger:
    def __init__(self, run_name=None, log_dir="runs"):
        run_name = run_name or "default"
        self.run_name = run_name
        self.step = 0
        self.run_path = os.path.join(log_dir, f"{run_name}")
        os.makedirs(self.run_path, exist_ok=True)
        self.writer = SummaryWriter(self.run_path)

        # Optional accumulators
        self.balances = []
        self.weights = []
        self.asset_holdings = []
        self.metrics = {}
        self.config = {}

    def log_scalar(self, name, value, step=None):

        if step is None:
            step = self.step
        self.writer.add_scalar(name, value, step)

    def next_step(self):
        self.step += 1

    def log_metrics(self, metrics_dict):
        for key, values in metrics_dict.items():
            arr = np.array(values)
            self.log_scalar(f"04_metrics_scalar/{key}_mean", np.mean(arr))
            self.log_scalar(f"04_metrics_scalar/{key}_std", np.std(arr))
            self.metrics[key] = values

    def log_emulated_histogram(self, name, values, bins=10, step=None):
        pass

    def add_run_data(self, balances, weights, asset_holdings):
        self.balances.append(balances)
        self.weights.append(weights)
        self.asset_holdings.append(asset_holdings)

    def close(self):
        self.writer.close()


