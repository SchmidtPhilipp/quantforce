from utils.config.config import Config
from envs.portfolio_agent_generator import create_portfolio_env
from utils.tensorboard.start_tensorboard import start_tensorboard
from utils.tensorboard.safari import focus_tensorboard_tab
from data.downloader import get_data
from agents.create_agent import create_agent
from utils.tensorboard.safari import refresh_current_safari_window
from train.run_agent import run_agent
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
    train_dataset = TimeBasedDataset(config["tickers"], config["train_start"], config["train_end"], "1d", config["time_window_size"], indicators=config["indicators"])
    
    # Wrap with DataLoader
    train_dataloader = DataLoader(train_dataset, shuffle=False) # Shuffle should never be true for time series data # Batch size of 1 for time series data!

    train_env = create_portfolio_env(
        data=train_dataloader,
        initial_balance=config["initial_balance"],
        verbosity=config["verbosity"],
        n_agents=config["n_agents"],
        trade_cost_percent=config["trade_cost_percent"],
        trade_cost_fixed=config["trade_cost_fixed"]
    )

    # Create an Agent
    #train_agent_instance = create_agent(config["agent"], train_env, config.data)
    # Create an Agent
    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    train_agent_instance = config.load_agent(obs_dim, act_dim)
    # Create an epsilon scheduler
    epsilon_scheduler = config.load_scheduler()

    load_path = run_agent(
        env=train_env,
        agent=train_agent_instance,
        config=config,
        n_episodes=config["train_episodes"],
        run_name=config.run_name,
        epsilon_scheduler=epsilon_scheduler,
        train = True,
    )
    ##################################

    ##################################
    # Evaluation setup
    eval_dataset = TimeBasedDataset(config["tickers"], config["eval_start"], config["eval_end"], "1d", config["time_window_size"], indicators=config["indicators"])
    
    # Wrap with DataLoader
    eval_dataloader = DataLoader(eval_dataset, shuffle=False) # Shuffle should never be true for time series data # Batch size of 1 for time series data!


    eval_env = create_portfolio_env(
        data=eval_dataloader,
        initial_balance=config["initial_balance"],
        verbosity=config["verbosity"],
        n_agents=config["n_agents"],
        trade_cost_percent=config["trade_cost_percent"],
        trade_cost_fixed=config["trade_cost_fixed"]
    )

    eval_agent_instance = train_agent_instance

    #eval_agent_instance.load(load_path)

    run_agent(
        env=eval_env, 
        agent=eval_agent_instance, 
        config=config, 
        run_name=config.run_name,
        n_episodes=config["eval_episodes"], 
        epsilon_scheduler=None,
        train=False)
    ##################################

    # END
    if config.get("enable_tensorboard"):
        focus_tensorboard_tab()
        refresh_current_safari_window()
        