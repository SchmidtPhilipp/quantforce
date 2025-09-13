import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from itertools import chain, combinations

import torch

from qf.data.tickers.tickers import (
    DOWJONES,
    MSCIACWI,
    MSCIEU,
    MSCIWORLD,
    NASDAQ100,
    SNP500,
)

from qf.settings import *

# Initialize logging configuration automatically
from qf.utils.logging_config import setup_logging


USE_GPU_ACCELERATION = (
    False  # We do not use GPU because we are doing a lot of small tensor operations.
)
if torch.cuda.is_available() and USE_GPU_ACCELERATION:
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available() and USE_GPU_ACCELERATION:
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"


from qf.agents import *

# Data
from qf.data import *

from qf.envs.config.env_config import EnvConfig

# Environments
from qf.envs.multi_agent_portfolio_env import MultiAgentPortfolioEnv

# Networks
from qf.networks.default_networks import DefaultNetworks

# Optimizers
from qf.optim.grid_search_optimizer import GridSearchOptimizer
from qf.results.episode import Episode
from qf.results.plotframe import PlotFrame
from qf.results.result import Result

# Results
from qf.results.run import Run
from qf.results.tensorview import TensorView

# Unsupervised learning
from qf.unsupervised import PCA

# Utils
from qf.utils.experiment_logger import ExperimentLogger

# Logging
from qf.utils.logging_config import get_logger, setup_logging
from qf.utils.metrics import Metrics

# Plotting
from qf.utils.plot import *

# TensorBoard utilities
from qf.utils.tensorboard.start_tensorboard import start_tensorboard
