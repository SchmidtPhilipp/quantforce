import json
import os
from datetime import datetime

from qf.utils.logging_config import get_logger

logger = get_logger(__name__)


def save_config(config, run_name=None):
    os.makedirs("config", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = run_name or "config"
    path = os.path.join("config", f"{name}_{timestamp}.json")

    with open(path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"📝 Saved config to: {path}")
