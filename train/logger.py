from torch.utils.tensorboard import SummaryWriter
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

class Logger:
    def __init__(self, run_name=None, log_dir="runs"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = run_name or "default"
        self.run_name = run_name
        self.step = 0
        self.run_path = os.path.join(log_dir, f"{run_name}_{timestamp}")
        os.makedirs(self.run_path, exist_ok=True)
        self.writer = SummaryWriter(self.run_path)

        # Optional accumulators
        self.balances = []
        self.weights = []
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

    def add_run_data(self, balances, weights):
        self.balances.append(balances)
        self.weights.append(weights)

    def save_evaluation_data(self, config=None):
        # Save balances
        balances_array = np.array(self.balances)
        pd.DataFrame(balances_array.T).to_csv(os.path.join(self.run_path, "balances.csv"), index=False)
        np.save(os.path.join(self.run_path, "balances.npy"), balances_array)

        # Save weights
        weights_array = np.array(self.weights)
        np.save(os.path.join(self.run_path, "weights.npy"), weights_array)

        # Save metrics
        summary = {
            "mean": {k: float(np.mean(v)) for k, v in self.metrics.items()},
            "std": {k: float(np.std(v)) for k, v in self.metrics.items()}
        }
        with open(os.path.join(self.run_path, "metrics_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        pd.DataFrame(self.metrics).to_csv(os.path.join(self.run_path, "metrics_all.csv"), index=False)

        # Save config
        if config:
            self.config = config
        if self.config:
            with open(os.path.join(self.run_path, "config.json"), "w") as f:
                json.dump(self.config, f, indent=2)

    def close(self):
        self.writer.close()
