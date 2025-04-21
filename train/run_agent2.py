from tqdm import tqdm
import os
import numpy as np
from train.logger import Logger
from utils.metrics import Metrics
from train.scheduler.epsilon_scheduler import EpsilonScheduler
from utils.tracker.assettracker import AssetTracker
import psutil
import GPUtil
from utils.tracker.tracker import Tracker


def run_agent2(env, agent, config, save_path=None, n_episodes=10, epsilon_scheduler=None, mode="TRAIN", use_tqdm=True, eval_env=None, eval_interval=5):
    """
    Runs the agent in the given environment for training or validation.

    Parameters:
        env: The environment to train or validate in.
        agent: The agent to train or validate.
        config (dict): Configuration dictionary.
        save_path (str): Path to save the trained agent (only used in training).
        n_episodes (int): Number of episodes to run.
        epsilon_scheduler (EpsilonScheduler): Scheduler for epsilon decay (only used in training).
        mode (str): Mode of operation: "TRAIN" or "VAL".
        use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print episode summaries.
        eval_env: Optional evaluation environment for training mode.
        eval_interval (int): Interval (in episodes) at which to evaluate the agent on the evaluation environment.

    Returns:
        dict: Metrics collected during the run (for validation).
    """
    assert mode in ["TRAIN", "VAL"], "Invalid mode. Choose from 'TRAIN' or 'VAL'."

    # Initialize logger based on mode
    logger = Logger(run_name=f"{mode}_run")
    metrics = Metrics()

    # Initialize Tracker for training or validation
    tracker = Tracker(timesteps=env.get_timesteps(), tensorboard_prefix=f"{mode}")
    tracker.register_value("rewards", (env.n_agents,), "Rewards per agent", ["timesteps", "agents"])
    tracker.register_value("balances", (env.n_agents,), "Balances per agent", ["timesteps", "agents"])
    tracker.register_value("actions", (env.n_agents, env.n_assets + 1), "Actions per agent", ["timesteps", "agents", "assets"])
    tracker.register_value("asset_holdings", (env.n_agents, env.n_assets), "Asset holdings per agent", ["timesteps", "agents", "assets"])
    tracker.register_value("resource_usage", (4,), "Resource usage (CPU, memory, GPU)", ["timesteps", "resources"])

    # Initialize evaluation tracker and logger if in TRAIN mode and eval_env is provided
    eval_tracker = None
    eval_logger = None
    if mode == "TRAIN" and eval_env:
        eval_logger = Logger(run_name="EVAL_run")
        eval_tracker = Tracker(timesteps=eval_env.get_timesteps(), tensorboard_prefix="EVAL")
        eval_tracker.register_value("rewards", (eval_env.n_agents,), "Rewards per agent", ["timesteps", "agents"])
        eval_tracker.register_value("balances", (eval_env.n_agents,), "Balances per agent", ["timesteps", "agents"])
        eval_tracker.register_value("actions", (eval_env.n_agents, eval_env.n_assets + 1), "Actions per agent", ["timesteps", "agents", "assets"])
        eval_tracker.register_value("asset_holdings", (eval_env.n_agents, eval_env.n_assets), "Asset holdings per agent", ["timesteps", "agents", "assets"])
        eval_tracker.register_value("resource_usage", (4,), "Resource usage (CPU, memory, GPU)", ["timesteps", "resources"])

    if mode == "TRAIN" and epsilon_scheduler is None:
        from train.scheduler.epsilon_scheduler import LinearEpsilonScheduler
        epsilon_scheduler = LinearEpsilonScheduler(epsilon_start=1.0, epsilon_min=0.01)

    # Keep track of the best agent
    best_reward = -np.inf
    best_episode = 0
    best_agent = None
    episodes_no_improvement = 0

    # Main loop
    for ep in range(n_episodes):
        # Run a single training or validation episode
        train_reward = episode(
            env=env,
            agent=agent,
            tracker=tracker,
            logger=logger,
            epsilon_scheduler=epsilon_scheduler if mode == "TRAIN" else None,
            train=(mode == "TRAIN"),
            ep=ep,
            use_tqdm=use_tqdm
        )

        # Evaluate the agent at specified intervals if in TRAIN mode and eval_env is provided
        if mode == "TRAIN" and eval_env and ep % eval_interval == 0:
            eval_reward = episode(
                env=eval_env,
                agent=agent,
                tracker=eval_tracker,
                logger=eval_logger,
                epsilon_scheduler=None,  # No epsilon scheduler for evaluation
                train=False,
                ep=ep,
                use_tqdm=use_tqdm
            )

            # Update the best agent based on evaluation reward
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_episode = ep
                best_agent = agent
                episodes_no_improvement = 0
            else:
                episodes_no_improvement += 1
        elif mode == "TRAIN":
            # Fallback: Update the best agent based on training reward if no eval_env is provided
            if train_reward > best_reward:
                best_reward = train_reward
                best_episode = ep
                best_agent = agent
                episodes_no_improvement = 0
            else:
                episodes_no_improvement += 1

        # Early stopping if no improvement for 10% of episodes
        if mode == "TRAIN" and episodes_no_improvement > n_episodes * 0.1:
            print(f"Early stopping at episode {ep} due to no improvement.")
            break

    # Save the best agent (only during training)
    if mode == "TRAIN":
        agent.save(os.path.join(logger.run_path, "last_agent.pt"))
        if best_agent is not None and best_episode != n_episodes - 1:
            best_agent.save(os.path.join(logger.run_path, f"best_agent_{best_episode}.pt"))
            print(f"Best agent saved at episode {best_episode} with reward {best_reward}")

    # Close the loggers
    logger.close()
    if eval_logger:
        eval_logger.close()

    # Save tracker data
    tracker.save(logger.run_path)
    if eval_tracker:
        eval_tracker.save(eval_logger.run_path)

    # Save config data
    config.save(os.path.join(logger.run_path, "config.json"))

    # Return metrics for validation
    if mode == "VAL":
        return metrics.metrics


def episode(env, agent, tracker, logger, epsilon_scheduler, train, ep, use_tqdm):
    """
    Runs a single episode (training or evaluation).

    Parameters:
        env: The environment to train or evaluate in.
        agent: The agent to train or evaluate.
        tracker: Tracker instance for tracking metrics.
        logger: Logger instance for logging metrics.
        epsilon_scheduler: Scheduler for epsilon decay (only used in training).
        train (bool): If True, runs training; if False, runs evaluation.
        ep (int): Current episode number.
        use_tqdm (bool): If True, use tqdm for progress tracking.

    Returns:
        float: Total reward for the episode.
    """
    state = env.reset(validation=not train)  # Use validation flag to specify mode
    total_reward = 0
    episode_done = False
    progress_bar = tqdm(total=env.get_timesteps(), desc=f"{'TRAIN' if train else 'VAL'} Episode {ep+1}", unit="step", ncols=80) if use_tqdm else None

    step = 0
    while not episode_done:
        # Use epsilon-greedy policy for exploration during training
        epsilon = epsilon_scheduler.epsilon if train and epsilon_scheduler else 0.0
        action = agent.act(state, epsilon=epsilon)

        next_state, reward, episode_done, _ = env.step(action)

        # Record actions, asset holdings, balances, and resource usage
        tracker.record_step(
            rewards=reward,
            balances=env.actor_balance,
            actions=action,
            asset_holdings=env.actor_asset_holdings,
            resource_usage=get_resource_usage()
        )

        if train:
            agent.store((state, action, reward, next_state))
            agent.train()

        state = next_state
        total_reward += reward
        step += 1

        if use_tqdm:
            progress_bar.update(1)

    tracker.end_episode()
    tracker.log(logger)

    if use_tqdm:
        progress_bar.close()

    # Update epsilon using the scheduler (only during training)
    if train and epsilon_scheduler:
        epsilon_scheduler.step(ep + 1, n_episodes)
        logger.log_scalar(f"TRAIN_epsilon/epsilon", epsilon_scheduler.epsilon, step=ep)

    return total_reward


def get_resource_usage():
    """
    Returns the current resource usage (CPU, memory, GPU).

    Returns:
        list: A list containing CPU usage (%), memory usage (GB), GPU usage (%), and GPU memory usage (GB).
    """
    # CPU and memory usage
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    memory_used = memory.used / (1024 ** 3)  # Convert to GB

    # GPU usage (if applicable)
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assuming a single GPU
        gpu_util = gpu.load * 100  # GPU utilization in percentage
        gpu_memory_used = gpu.memoryUsed / 1024  # Convert to GB
    else:
        gpu_util = gpu_memory_used = 0.0

    return [cpu_percent, memory_used, gpu_util, gpu_memory_used]