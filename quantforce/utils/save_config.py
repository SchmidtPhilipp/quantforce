import os
import json
from datetime import datetime

def save_config(config, run_name=None):
    os.makedirs("configs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = run_name or "config"
    path = os.path.join("configs", f"{name}_{timestamp}.json")

    with open(path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"📝 Saved config to: {path}")