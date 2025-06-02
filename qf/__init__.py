from qf.data import DOWJONES, NASDAQ100, SNP500

### DEFAULTS
VERBOSITY = 0

DEFAULT_LOG_DIR = 'runs'

DEFAULT_INITIAL_BALANCE = 1000000

# Default data configuration
DEFAULT_TICKERS = DOWJONES
DEFAULT_TRAIN_START = "2000-01-01"
DEFAULT_TRAIN_END = "2020-01-01"
DEFAULT_EVAL_START = "2020-01-02"
DEFAULT_EVAL_END = "2025-01-01"


# Default parameters for TimeBasedDataset
DEFAULT_INTERVAL = "1d"
DEFAULT_WINDOW_SIZE = 1
DEFAULT_INDICATORS = ["rsi", "sma", "macd", "atr"]
DEFAULT_CACHE_DIR = "../cache"


# Default environment configuration
DEFAULT_N_AGENTS = 1
DEFAULT_TRADE_COST_PERCENT = 0.0
DEFAULT_TRADE_COST_FIXED = 0
DEFAULT_REWARD_FUNCTION = "log_return" # Options: "linear_rate_of_return", "sharpe_ratio", "log_return"
DEFAULT_REWARD_SCALING = 100  # Scaling factor for the reward function

DEFUALT_CONFIG_NAME = "DEFAULT_CONFIG"

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
    "reward_scaling": DEFAULT_REWARD_SCALING,

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
    "reward_scaling": DEFAULT_REWARD_SCALING,

    "device": DEFAULT_DEVICE,
    "verbosity": VERBOSITY,
    "log_dir": DEFAULT_LOG_DIR,
    "config_name": DEFUALT_CONFIG_NAME
}


##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

#### DEFAULT AGENT CONFIGS ####

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# Classic Agent configurations

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# Tangency Agent configurations
DEFAULT_TANGENCY_LOG_RETURNS = True  # Use log returns for calculations
DEFAULT_TANGENCY_RISK_FREE_RATE = 0.01  # Example risk-free rate for tangency portfolio calculations
DEFAULT_TANGENCY_METHOD = "default"  # Optimization method for the tangency portfolio
DEFAULT_TANGENCYAGENT_CONFIG = {
    "risk_free_rate": DEFAULT_TANGENCY_RISK_FREE_RATE,  # Example risk-free rate
    "method": DEFAULT_TANGENCY_METHOD,  # Optimization method for the tangency portfolio
    "log_returns": DEFAULT_TANGENCY_LOG_RETURNS,  # Use log returns for calculations
}

DEFAULT_TANGENCY_HYPERPARAMETER_SPACE = {
    # TODO implement different modes
    #"risk_free_rate": [0.01, 0.02],  # Example risk-free rates for hyperparameter search
    #"method": ["default", "alternative_method"],  # Different optimization methods for the tangency portfolio
}

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# Classic One Period Markovitz Agent configurations
DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZAGENT_LOG_RETURNS = True  # Use log returns for calculations
DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZAGENT_TARGET = "Tangency"  # Optimization target: Tangency, MaxExpReturn, MinVariance
DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZAGENT_RISK_MODEL = "sample_cov"  # Risk model: sample, exp_weighted
DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZAGENT_RISK_FREE_RATE = 0.0  # Risk-free rate for Tangency optimization

DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZAGENT_CONFIG = {
    "target": DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZAGENT_TARGET,  # Optimization target
    "risk_model": DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZAGENT_RISK_MODEL,  # Risk model
    "risk_free_rate": DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZAGENT_RISK_FREE_RATE,  # Risk-free rate for Tangency optimization
    "log_returns": DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZAGENT_LOG_RETURNS,  # Use log returns for calculations
}

DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZ_HYPERPARAMETER_SPACE = {
    "target": ["Tangency", "MaxExpReturn", "MinVariance"],
    "risk_model": ["sample_cov", 
                   "exp_cov", 
                   "ledoit_wolf", 
                   "ledoit_wolf_constant_variance", 
                   "ledoit_wolf_single_factor", 
                   "ledoit_wolf_constant_correlation",
                   "oracle_approximating"],
    "risk_free_rate": [0.0],
}

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# Signle-Agent configurations

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# DQN Agent configuration
DEFAULT_DQN_LR = 1e-3
DEFAULT_DQN_GAMMA = 0.99
DEFAULT_DQN_BATCH_SIZE = 32
DEFAULT_DQN_BUFFER_MAX_SIZE = 100000
DEFAULT_DQN_EPSILON_START = 0.4
DEFAULT_DQNAGENT_CONFIG = {
    "actor_config": None,  # Use default architecture
    "lr": DEFAULT_DQN_LR,
    "gamma": DEFAULT_DQN_GAMMA,
    "batch_size": DEFAULT_DQN_BATCH_SIZE,
    "buffer_max_size": DEFAULT_DQN_BUFFER_MAX_SIZE,
    "device": DEFAULT_DEVICE,
    "epsilon_start": DEFAULT_DQN_EPSILON_START
}

DEFAULT_DQNAGENT_HYPERPARAMETER_SPACE = {
    "learning_rate": [1e-3, 1e-4],
    "epsilon_start": [0.1, 0.4, 1],
}

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# SAC Agent configuration
DEFAULT_SAC_LR = 0.0001
DEFAULT_SAC_GAMMA = 0.99
DEFAULT_SAC_BATCH_SIZE = 128
DEFAULT_SAC_BUFFER_MAX_SIZE = 100000
DEFAULT_SAC_POLICY = "MlpPolicy"  # Default policy architecture for SAC
DEFAULT_SAC_TAU = 0.005  # Target network update rate for SAC
DEFAULT_SAC_GRADIENT_STEPS = 1  # Number of gradient steps per training iteration for SAC
DEFAULT_SAC_TRAIN_FREQ = 1  # Frequency of training steps for SAC
DEFAULT_SAC_ENT_COEF = "auto_0.1"  # Automatic entropy coefficient adjustment for SAC
DEFAULT_SAC_VERBOSITY = 1  # Verbosity level for logging

DEFAULT_SACAGENT_CONFIG = {
    "policy": "MlpPolicy",  # Default policy architecture
    "learning_rate": DEFAULT_SAC_LR,
    "buffer_size": DEFAULT_SAC_BUFFER_MAX_SIZE,
    "batch_size": DEFAULT_SAC_BATCH_SIZE,
    "tau": DEFAULT_SAC_TAU,  # Target network update rate
    "gamma": DEFAULT_SAC_GAMMA,
    "train_freq": DEFAULT_SAC_TRAIN_FREQ,  # Frequency of training steps
    "gradient_steps": DEFAULT_SAC_GRADIENT_STEPS,  # Number of gradient steps per training iteration
    "device": DEFAULT_DEVICE,  # Device to run the computations on
    "ent_coef": DEFAULT_SAC_ENT_COEF,  # Automatic entropy coefficient adjustment
    "verbose": DEFAULT_SAC_VERBOSITY  # Verbosity level for logging
}

DEFAULT_SACAGENT_HYPERPARAMETER_SPACE = {
    "learning_rate": [3e-4, 1e-4],
    "buffer_size": [100000, 500000],
    "batch_size": [64, 128],
    "ent_coef": ["auto", "auto_0.1"],
}

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# TD3 Agent configuration
DEFAULT_TD3_LR = 0.0003  # Learning rate
DEFAULT_TD3_GAMMA = 0.99  # Discount factor
DEFAULT_TD3_BATCH_SIZE = 100  # Batch size
DEFAULT_TD3_BUFFER_MAX_SIZE = 1000000  # Replay buffer size
DEFAULT_TD3_POLICY = "MlpPolicy"  # Default policy architecture for TD3
DEFAULT_TD3_TAU = 0.005  # Target network update rate
DEFAULT_TD3_GRADIENT_STEPS = -1  # Number of gradient steps per training iteration (-1 means auto)
DEFAULT_TD3_TRAIN_FREQ = 1  # Frequency of training steps
DEFAULT_TD3_NOISE_STD = 0.2  # Standard deviation of noise added to actions
DEFAULT_TD3_NOISE_CLIP = 0.5  # Clipping range for noise

DEFAULT_TD3AGENT_CONFIG = {
    "policy": DEFAULT_TD3_POLICY,  # Default policy architecture
    "learning_rate": DEFAULT_TD3_LR,
    "buffer_size": DEFAULT_TD3_BUFFER_MAX_SIZE,
    "batch_size": DEFAULT_TD3_BATCH_SIZE,
    "tau": DEFAULT_TD3_TAU,  # Target network update rate
    "gamma": DEFAULT_TD3_GAMMA,
    "train_freq": DEFAULT_TD3_TRAIN_FREQ,  # Frequency of training steps
    "gradient_steps": DEFAULT_TD3_GRADIENT_STEPS,  # Number of gradient steps per training iteration
    "device": DEFAULT_DEVICE,  # Device to run the computations on
    "noise_std": DEFAULT_TD3_NOISE_STD,  # Standard deviation of noise added to actions
    "noise_clip": DEFAULT_TD3_NOISE_CLIP  # Clipping range for noise
}

DEFAULT_TD3_HYPERPARAMETER_SPACE = {
    "learning_rate": [3e-4, 1e-4],
    "noise_std": [0.1, 0.2],
    "noise_clip": [0.3, 0.5],
}

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# DDPG Agent configuration
DEFAULT_DDPG_LR = 0.001  # Learning rate
DEFAULT_DDPG_GAMMA = 0.99
DEFAULT_DDPG_BATCH_SIZE = 64  # Batch size
DEFAULT_DDPG_BUFFER_MAX_SIZE = 1000000  # Replay buffer size
DEFAULT_DDPG_POLICY = "MlpPolicy"  # Default policy architecture for DDPG
DEFAULT_DDPG_TAU = 0.005  # Target network update rate
DEFAULT_DDPG_TRAIN_FREQ = 1  # Frequency of training steps
DEFAULT_DDPG_GRADIENT_STEPS = -1  # Number of gradient steps per training iteration (-1 means auto)
DEFAULT_DDPG_VERBOSITY = 1  # Verbosity level for logging

DEFAULT_DDPGAGENT_CONFIG = {
    "policy": DEFAULT_DDPG_POLICY,  # Default policy architecture
    "learning_rate": DEFAULT_DDPG_LR,
    "buffer_size": DEFAULT_DDPG_BUFFER_MAX_SIZE,
    "batch_size": DEFAULT_DDPG_BATCH_SIZE,
    "tau": DEFAULT_DDPG_TAU,  # Target network update rate
    "gamma": DEFAULT_DDPG_GAMMA,
    "train_freq": DEFAULT_DDPG_TRAIN_FREQ,  # Frequency of training steps
    "gradient_steps": DEFAULT_DDPG_GRADIENT_STEPS,  # Number of gradient steps per training iteration
    "device": DEFAULT_DEVICE,  # Device to run the computations on
    "verbose": DEFAULT_DDPG_VERBOSITY  # Verbosity level for logging
}

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# PPO Agent configuration
DEFAULT_PPO_POLICY = "MlpPolicy"  # Default policy architecture for PPO
DEFAULT_PPO_LR = 3e-4  # Learning rate
DEFAULT_PPO_N_STEPS = 2048  # Number of steps to run for each environment per update
DEFAULT_PPO_BATCH_SIZE = 64  # Mini-batch size for PPO
DEFAULT_PPO_GAMMA = 0.99  # Discount factor
DEFAULT_PPO_GAE_LAMBDA = 0.95  # GAE lambda parameter
DEFAULT_PPO_CLIP_RANGE = 0.2  # Clipping range for PPO
DEFAULT_PPO_VERBOSITY = 1  # Verbosity level for logging

DEFAULT_PPOAGENT_CONFIG = {
    "policy": DEFAULT_PPO_POLICY,  # Default policy architecture
    "learning_rate": DEFAULT_PPO_LR,
    "n_steps": DEFAULT_PPO_N_STEPS,  # Number of steps to run for each environment per update
    "batch_size": DEFAULT_PPO_BATCH_SIZE,  # Mini-batch size for PPO
    "gamma": DEFAULT_PPO_GAMMA,
    "gae_lambda": DEFAULT_PPO_GAE_LAMBDA,  # GAE lambda parameter
    "clip_range": DEFAULT_PPO_CLIP_RANGE,  # Clipping range for PPO
    "device": DEFAULT_DEVICE,  # Device to run the computations on
    "verbose": DEFAULT_PPO_VERBOSITY  # Verbosity level for logging
}

DEFAULT_PPO_HYPERPARAMETER_SPACE = {
    "learning_rate": [3e-4, 1e-4],
    "clip_range": [0.1, 0.2],
    "gae_lambda": [0.9, 0.95],
}

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# Default Multi-Agent configurations 

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# MADDPG Agent configuration
DEFAULT_MADDPG_LR = 0.001  # Learning rate
DEFAULT_MADDPG_GAMMA = 0.99  # Discount factor
DEFAULT_MADDPG_BATCH_SIZE = 64  # Batch size
DEFAULT_MADDPG_BUFFER_MAX_SIZE = 1000000  # Replay buffer size
DEFAULT_MADDPG_TAU = 0.005  # Target network update rate
DEFAULT_MADDPG_VERBOSITY = 0  # Verbosity level for logging
DEFAULT_MADDPG_LAMBDA = 0.95  # lambda parameter for weighting the loss function. 
DEFAULT_MADDPG_LOSS_FN = "mse"  # "MSE" or "weighted_correlation_loss"

DEFAULT_MADDPGAGENT_CONFIG = {
    "learning_rate": DEFAULT_MADDPG_LR,
    "buffer_max_size": DEFAULT_MADDPG_BUFFER_MAX_SIZE,
    "batch_size": DEFAULT_MADDPG_BATCH_SIZE,
    "tau": DEFAULT_MADDPG_TAU,  # Target network update rate
    "gamma": DEFAULT_MADDPG_GAMMA,
    "lambda_": DEFAULT_MADDPG_LAMBDA,  # GAE lambda parameter
    "device": DEFAULT_DEVICE,  # Device to run the computations on
    "verbose": DEFAULT_MADDPG_VERBOSITY,  # Verbosity level for logging
    "losloss_functions_fn": DEFAULT_MADDPG_LOSS_FN  # Loss function for MADDPG, can be "mse" or "huber"
}

DEFAULT_MADDPG_HYPERPARAMETER_SPACE = {
    "learning_rate": [1e-3, 1e-4],
    "lambda_": [0.9, 0.95],
    "loss_fn": ["mse", "weighted_correlation_loss"],
}


##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################



# Hyperparameter search configuration
from qf.utils.hyperparameter_search import hyperparameter_search

# Environments
from qf.envs.multi_agent_portfolio_env import MultiAgentPortfolioEnv

# Data
from qf.data import TimeBasedDataset
from qf.data import load_data
from qf.data import add_technical_indicators
from qf.data import get_data

# Custom Agents
from qf.agents.tensor_agents.dqn_agent import DQNAgent

# Classic Agents
from qf.agents.classic_agents.classic_one_period_markovitz_agent import ClassicOnePeriodMarkovitzAgent

# Stable Baselines3 Agents
from qf.agents.sb3_agents.sac_agent import SACAgent
from qf.agents.sb3_agents.td3_agent import TD3Agent
from qf.agents.sb3_agents.ddpg_agent import DDPGAgent
from qf.agents.sb3_agents.ppo_agent import PPOAgent
from qf.agents.buffers.a2c_agent import A2CAgent

# Multi-Agent Agents
from qf.agents.tensor_agents.maddpg_agent import MADDPGAgent



# General utilities
from qf.utils.tensorboard.start_tensorboard import start_tensorboard
from qf.utils.tensorboard.safari import focus_tensorboard_tab, refresh_current_safari_window

# Helper functions
from qf.utils.helper_functions import generate_random_name
from qf.utils.metrics import Metrics


# Visualization
from qf.utils.plot import setup_pgf, reset_pgf