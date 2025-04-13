from .logger import Logger
from utils.metrics import Metrics
import os
import numpy as np
from utils.AssetTracker import AssetTracker


def run_agent(env, agent, config, save_path=None, n_episodes=10, run_name=None, epsilon_scheduler=None, train=True):
    """
    Runs the agent in the given environment for training or evaluation.

    Parameters:
        env: The environment to train or evaluate in.
        agent: The agent to train or evaluate.
        config (dict): Configuration dictionary.
        save_path (str): Path to save the trained agent (only used in training).
        n_episodes (int): Number of episodes to run.
        run_name (str): Name of the run for logging.
        epsilon_scheduler (EpsilonScheduler): Scheduler for epsilon decay (only used in training).
        train (bool): If True, runs training; if False, runs evaluation.

    Returns:
        dict: Metrics collected during the run (for evaluation).
    """
    run_type = "TRAIN" if train else "EVAL"
    run_name = f"{run_type}_" + (run_name or "default")
    logger = Logger(run_name=run_name)
    metrics = Metrics()

    # Initialize AssetTracker with initial values
    asset_tracker = AssetTracker(
        n_agents=env.n_agents,
        n_assets=env.n_assets,
        n_episodes=n_episodes,
        timesteps=env.get_timesteps(),
        tickers=config["tickers"],
        tensorboard_prefix=f"{run_type}"
    )

    if train and epsilon_scheduler is None:
        from train.scheduler.epsilon_scheduler import LinearEpsilonScheduler
        epsilon_scheduler = LinearEpsilonScheduler(epsilon_start=1.0, epsilon_min=0.01)

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0 # Note: this might not necessarily be a scalar.
        done = False
        steps = 0

        while not done:
            # Use epsilon-greedy policy for exploration during training
            epsilon = epsilon_scheduler.epsilon if train else 0.0
            action = agent.act(state, epsilon=epsilon)

            next_state, reward, done, _ = env.step(action)

            # Record actions, asset holdings, and balances
            asset_tracker.record_step(action, env.actor_asset_holdings, env.actor_balance, env.balance, reward)

            if train:
                agent.store((state, action, reward, next_state))
                agent.train()
                        
            state = next_state
            total_reward += reward
            steps += 1

        asset_tracker.log_episode(logger)
        asset_tracker.print_episode_summary(run_type=run_type, episode=ep)

        # append metrics for this episode
        metrics.append(asset_tracker.get_episode_data("balances", ep))

        # Update epsilon using the scheduler (only during training)
        if train:
            epsilon_scheduler.step(ep + 1, n_episodes)
            logger.log_scalar(f"{run_type}_epsilon/epsilon", epsilon_scheduler.epsilon, step=ep)

            metrics.print_report()
            print("-" * 50)

            # Reset the metrics for the next episode
            metrics.reset()

        # End the episode
        asset_tracker.end_episode()

    # Log portfolio mean and std over time
    if not train:
        asset_tracker.log_statistics(logger) # only makes sense for evaluation

        metrics.print_report()
        metrics.log_metrics(logger, run_type=run_type)
        print("-" * 50)

    # Save model (only during training)
    if train:
        agent.save(os.path.join(logger.run_path, "agent.pt"))
        
    # Close the logger
    logger.close()
    
    # Save asset tracker data
    asset_tracker.save(os.path.join(logger.run_path, run_type))

    # Save config data
    config.save(os.path.join(logger.run_path, "config.json"))

    # Return metrics for evaluation
    if not train:
        return metrics.metrics