import torch
import os

class BaseAgent:
    def __init__(self):
        self.model = None  # Must be set in the subclass

    def save(self, path):
        if self.model is None:
            raise ValueError("No model defined in agent.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"✅ Agent saved to: {path}")

    def load(self, path):
        if self.model is None:
            raise ValueError("No model defined in agent.")
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
