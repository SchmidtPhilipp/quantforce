from utils.config.config import Config
from envs.portfolio_agent_generator import create_portfolio_env
from utils.tensorboard.start_tensorboard import start_tensorboard
from utils.tensorboard.safari import focus_tensorboard_tab
from data.downloader import get_data
from agents.create_agent import create_agent
from utils.tensorboard.safari import refresh_current_safari_window
from train.run_agent2 import run_agent2  # Import the updated run_agent2 function
from data.dataset import TimeBasedDataset
from torch.utils.data import DataLoader

def process_config(config_path):
    """
    Processes a single configuration file and runs training and evaluation.

    Parameters:
        config_path (str): Path to the configuration file.
    """
    print(f"Processing configuration: {config_path}")
    config = Config(config_path)

    if config.get("enable_tensorboard"):
        start_tensorboard(mode="safari", port=6005)
        focus_tensorboard_tab()
        refresh_current_safari_window()

    ##################################
    # Training setup
    train_dataset = TimeBasedDataset(
        config["tickers"], 
        config["train_start"], 
        config["train_end"], 
        "1d", 
        config["time_window_size"], 
        indicators=config["indicators"]
    )
    
    # Wrap with DataLoader
    train_dataloader = DataLoader(train_dataset, shuffle=False)  # Shuffle should never be true for time series data

    train_env = create_portfolio_env(
        data=train_dataloader,
        initial_balance=config["initial_balance"],
        verbosity=config["verbosity"],
        n_agents=config["n_agents"],
        trade_cost_percent=config["trade_cost_percent"],
        trade_cost_fixed=config["trade_cost_fixed"]
    )

    # Create an Agent
    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    train_agent_instance = config.load_agent(obs_dim, act_dim)

    # Create an epsilon scheduler
    epsilon_scheduler = config.load_scheduler()

    ##################################
    # Evaluation setup
    eval_dataset = TimeBasedDataset(
        config["tickers"], 
        config["eval_start"], 
        config["eval_end"], 
        "1d", 
        config["time_window_size"], 
        indicators=config["indicators"]
    )
    # Wrap with DataLoader
    eval_dataloader = DataLoader(eval_dataset, shuffle=False)  # Shuffle should never be true for time series data

    eval_env = create_portfolio_env(
        data=eval_dataloader,
        initial_balance=config["initial_balance"],
        verbosity=config["verbosity"],
        n_agents=config["n_agents"],
        trade_cost_percent=config["trade_cost_percent"],
        trade_cost_fixed=config["trade_cost_fixed"]
    )

    # Run training
    run_agent2(
        env=train_env,
        agent=train_agent_instance,
        config=config,
        max_episodes=config["max_train_episodes"],
        early_stopping_patience=config["early_stopping_patience"],
        epsilon_scheduler=epsilon_scheduler,
        run_name=config.run_name,
        mode="TRAIN",
        use_tqdm=True,
        eval_env=eval_env,  
        eval_interval=config["eval_interval"]
    )
    ##################################
    # Validation setup

    val_dataset = TimeBasedDataset(
        config["tickers"], 
        config["val_start"], 
        config["val_end"], 
        "1d", 
        config["time_window_size"], 
        indicators=config["indicators"]
    )

    val_dataloader = DataLoader(val_dataset, shuffle=False)  # Shuffle should never be true for time series data

    val_env = create_portfolio_env(
        data=val_dataloader,
        initial_balance=config["initial_balance"],
        verbosity=config["verbosity"],
        n_agents=config["n_agents"],
        trade_cost_percent=config["trade_cost_percent"],
        trade_cost_fixed=config["trade_cost_fixed"]
    )

    # Use the same agent for evaluation
    val_agent_instance = train_agent_instance

    # Run evaluation
    run_agent2(
        env=val_env, 
        agent=val_agent_instance, 
        config=config, 
        run_name=config.run_name,
        max_episodes=config["val_episodes"], 
        early_stopping_patience=config["val_episodes"],
        epsilon_scheduler=None,  # No epsilon scheduler during evaluation
        mode="VAL",
        use_tqdm=True
    )
    ##################################

    # END
    if config.get("enable_tensorboard"):
        focus_tensorboard_tab()
        refresh_current_safari_window()
