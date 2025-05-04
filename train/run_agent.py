from tqdm import tqdm
import os
import numpy as np
from train.logger import Logger
from utils.metrics import Metrics
from train.scheduler.epsilon_scheduler import PeriodicEpsilonScheduler
from utils.tracker.tracker import Tracker
from utils.risk_management.risk_management import risk_limiting_strategy, TimeVariantActionFilter

def run_agent(env, agent, config, save_path=None, run_name=None, max_episodes=100, early_stopping_patience=10, epsilon_scheduler=None, mode="TRAIN", use_tqdm=True, eval_env=None, eval_interval=5):
    """
    Runs the agent in the given environment for training or validation.

    Parameters:
        env: The environment to train or validate in.
        agent: The agent to train or validate.
        config (dict): Configuration dictionary.
        save_path (str): Path to save the trained agent (only used in training).
        max_episodes (int): Maximum number of episodes to run.
        early_stopping_patience (int): Number of episodes without improvement before stopping early.
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
    logger = Logger(run_name=f"{mode}_" + (run_name or "default"))
    metrics = Metrics()
    eval_metrics = Metrics()

    # Initialize Tracker for training or validation
    tracker = Tracker(timesteps=env.get_timesteps(), tensorboard_prefix=f"{mode}")
    env.register_tracker(tracker) # registers the values to be tracked

    # Initialize evaluation tracker and logger if in TRAIN mode and eval_env is provided
    eval_tracker = None
    eval_logger = None
    if mode == "TRAIN" and eval_env:
        eval_logger = Logger(run_name="EVAL_" + run_name)
        eval_tracker = Tracker(timesteps=eval_env.get_timesteps(), tensorboard_prefix="EVAL")
        eval_env.register_tracker(eval_tracker)

    # Use a sin²-based periodic epsilon scheduler if none is provided
    if mode == "TRAIN" and epsilon_scheduler is None:
        epsilon_scheduler = PeriodicEpsilonScheduler(epsilon_start=1.0, epsilon_min=0.01, period=100, function="cos")

    # Keep track of the best agent
    best_reward = -np.inf
    best_episode = 0
    best_agent = None
    episodes_no_improvement = 0

    # Main loop
    for ep in range(max_episodes):
        # Run a single training or validation episode
        train_reward = episode(env=env, agent=agent, tracker=tracker, logger=logger, epsilon_scheduler=epsilon_scheduler if mode == "TRAIN" else None, mode=mode, ep=ep, use_tqdm=use_tqdm)

        # Append metrics for this episode
        metrics.append(tracker.get_episode_data("balance", ep))

        # Evaluate the agent at specified intervals if in TRAIN mode and eval_env is provided
        if mode == "TRAIN" and eval_env and (ep+1) % eval_interval == 0:
            eval_reward = episode(
                env=eval_env,
                agent=agent,
                tracker=eval_tracker,
                logger=eval_logger,
                epsilon_scheduler=None,  # No epsilon scheduler for evaluation
                mode="EVAL",
                ep=ep,
                use_tqdm=use_tqdm
            )

            eval_metrics.append(eval_tracker.get_episode_data("balance", ep))
            eval_metrics.print_report()
            print("-" * 50)
            eval_metrics.reset()
            if env.device == "mps":
                eval_reward = eval_reward.cpu().numpy()
            # Update the best agent based on evaluation reward
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_episode = ep
                best_agent = agent
                episodes_no_improvement = 0
            else:
                episodes_no_improvement += 1
        elif mode == "TRAIN" and eval_env is None:
            # Fallback: Update the best agent based on training reward if no eval_env is provided
            if np.array(train_reward).sum() > best_reward:
                best_reward = np.array(train_reward).sum()
                best_episode = ep
                best_agent = agent
                episodes_no_improvement = 0
            else:
                episodes_no_improvement += 1

        # Early stopping if no improvement for `early_stopping_patience` episodes
        if mode == "TRAIN" and episodes_no_improvement >= early_stopping_patience:
            print(f"Early stopping at episode {ep} due to no improvement.")
            break


    # Save the best agent (only during training)
    if mode == "TRAIN":
        agent.save(os.path.join(logger.run_path, "last_agent.pt"))
        if best_agent is not None and best_episode != max_episodes - 1:
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
        tracker.log_statistics(logger, values_to_log=["balance", "actions", "rewards"]) # only makes sense for evaluation
        metrics.print_report()
        metrics.log_metrics(logger, run_type=mode)
        return metrics.metrics


def episode(env, agent, tracker, logger, epsilon_scheduler, mode, ep, use_tqdm):
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
    state = env.reset().to(env.device)  # Use validation flag to specify mode
    total_reward = 0
    episode_done = False
    progress_bar = tqdm(total=env.get_timesteps(), desc=f"{mode} Episode {ep+1}", unit="step", ncols=80) if use_tqdm else None
    train = mode == "TRAIN"
    step = 0
    while not episode_done:
        # Use epsilon-greedy policy for exploration during training
        epsilon = epsilon_scheduler.epsilon if train and epsilon_scheduler else 0.0
        action = agent.act(state, epsilon=epsilon)

        # Modify the action according to the risk management strategy
        #action = risk_limiting_strategy(action, env.last_actions, env.n_agents)
        
        next_state, reward, episode_done, _ = env.step(action)
        reward = reward.to(env.device)

        # Record actions, asset holdings, balances, environment balance, and resource usage
        tracker.record_step(
            rewards=reward,
            actor_balance=env.portfolio_value,
            actions=action,
            asset_holdings=env.portfolio_matrix,
            balance=env.get_portfolio_value().unsqueeze(0),
        )

        if train:
            agent.store((state, action, reward, next_state, episode_done))

        #if train and step % agent.batch_size == 0:
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
    if train:
        epsilon_scheduler.step()
        logger.log_scalar(f"TRAIN_epsilon/epsilon", epsilon_scheduler.epsilon, step=ep)

    # Log final balance
    logger.log_scalar(f"{mode}_final_balance/final_balance", env.get_portfolio_value().cpu().numpy(), step=ep)
    
    tracker.print_summary(run_type=mode, episode=ep)

    return total_reward