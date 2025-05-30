from qf.data import DOWJONES, NASDAQ100, SNP500

### DEFAULLTS
VERBOSITY = 0

DEFAULT_LOG_DIR = 'runs'

DEFAULT_INITIAL_BALANCE = 1000000

# Default data configuration
DEFAULT_TICKERS = DOWJONES
DEFAULT_TRAIN_START = "2010-01-01"
DEFAULT_TRAIN_END = "2020-01-01"
DEFAULT_EVAL_START = "2020-01-02"
DEFAULT_EVAL_END = "2021-01-01"


# Default parameters for TimeBasedDataset
DEFAULT_INTERVAL = "1d"
DEFAULT_WINDOW_SIZE = 60
DEFAULT_INDICATORS = ["rsi"]
DEFAULT_CACHE_DIR = "../cache"


# Default environment configuration
DEFAULT_N_AGENTS = 1
DEFAULT_TRADE_COST_PERCENT = 0.001
DEFAULT_TRADE_COST_FIXED = 0.0
DEFAULT_REWARD_FUNCTION = "linear_rate_of_return" # Options: "linear_rate_of_return", "sharpe_ratio"

DEFUALT_CONFIG_NAME = "DEFAULT_CONFIG" # eg MADDOG, DQN, MARKOVITZ, Tangency, etc.

DEFAULT_DEVICE = "cpu"  # Default device for PyTorch

# Default environment configuration
DEFAULT_ENV_CONFIG = {
    "initial_balance": DEFAULT_INITIAL_BALANCE,
    "n_agents": DEFAULT_N_AGENTS,
    "start_date": DEFAULT_TRAIN_START,
    "end_date": DEFAULT_TRAIN_END,
    "interval": DEFAULT_INTERVAL,
    "indicators": DEFAULT_INDICATORS,
    "cache_dir": DEFAULT_CACHE_DIR,
    "tickers": DEFAULT_TICKERS,
    "window_size": DEFAULT_WINDOW_SIZE
}

DEFAULT_TRAIN_ENV_CONFIG = {
    "tensorboard_prefix": "TRAIN",
    "tickers": DEFAULT_TICKERS,
    "start_date": DEFAULT_TRAIN_START,
    "end_date": DEFAULT_TRAIN_END,
    "window_size": DEFAULT_WINDOW_SIZE,
    "interval": DEFAULT_INTERVAL,
    "indicators": DEFAULT_INDICATORS,
    "cache_dir": DEFAULT_CACHE_DIR,

    "initial_balance": DEFAULT_INITIAL_BALANCE,
    "n_agents": DEFAULT_N_AGENTS,
    "trade_cost_percent": DEFAULT_TRADE_COST_PERCENT,
    "trade_cost_fixed": DEFAULT_TRADE_COST_FIXED,

    "reward_function": DEFAULT_REWARD_FUNCTION,

    "device": DEFAULT_DEVICE,
    "verbosity": VERBOSITY,
    "log_dir": DEFAULT_LOG_DIR,
    "config_name": DEFUALT_CONFIG_NAME
}

DEFAULT_EVAL_ENV_CONFIG = {
    "tensorboard_prefix": "EVAL",
    "tickers": DEFAULT_TICKERS,
    "start_date": DEFAULT_EVAL_START,
    "end_date": DEFAULT_EVAL_END,
    "window_size": DEFAULT_WINDOW_SIZE,
    "interval": DEFAULT_INTERVAL,
    "indicators": DEFAULT_INDICATORS,
    "cache_dir": DEFAULT_CACHE_DIR,

    "initial_balance": DEFAULT_INITIAL_BALANCE,
    "n_agents": DEFAULT_N_AGENTS,
    "trade_cost_percent": DEFAULT_TRADE_COST_PERCENT,
    "trade_cost_fixed": DEFAULT_TRADE_COST_FIXED,

    "reward_function": DEFAULT_REWARD_FUNCTION,

    "device": DEFAULT_DEVICE,
    "verbosity": VERBOSITY,
    "log_dir": DEFAULT_LOG_DIR,
    "config_name": DEFUALT_CONFIG_NAME
}

# Environments
from qf.envs.portfolio_agent_generator import create_portfolio_env
from qf.envs.multi_agent_portfolio_env import MultiAgentPortfolioEnv

# Data
from qf.data import TimeBasedDataset
from qf.data import load_data
from qf.data import add_technical_indicators
from qf.data import get_data

# Agents
from qf.agents.base_agent import BaseAgent
from qf.agents.dqn_agent import DQNAgent
from qf.agents.maddpg_agent import MADDPGAgent
from qf.agents.tangency_agent import TangencyAgent

# Agents utilities
from qf.agents import ModelBuilder

# Config processing
from qf.train.process import process_config
from qf.utils.config.config import Config
from qf.train.run_agent import run_agent

# General utilities
from qf.utils.tensorboard.start_tensorboard import start_tensorboard
from qf.utils.tensorboard.safari import focus_tensorboard_tab, refresh_current_safari_window

# Helper functions
from qf.utils.helper_functions import generate_random_name
from qf.utils.metrics import Metrics


