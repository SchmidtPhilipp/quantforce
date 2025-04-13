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
    tickers = config["tickers"]

    if train and epsilon_scheduler is None:
        from train.scheduler.epsilon_scheduler import LinearEpsilonScheduler
        epsilon_scheduler = LinearEpsilonScheduler(epsilon_start=1.0, epsilon_min=0.01)

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        episode_balances = []  # Track portfolio balances for the episode

        # Initialize AssetTracker with initial values
        asset_tracker = AssetTracker(
            n_agents=env.n_agents,
            n_assets=env.n_assets,
            initial_actor_balance=env.actor_balance,
            initial_balance=env.balance,
            tensorboard_name=f"02_{run_type.lower()}_assets"
        )

        while not done:
            # Use epsilon-greedy policy for exploration during training
            epsilon = epsilon_scheduler.epsilon if train else 0.0
            action = agent.act(state, epsilon=epsilon)

            next_state, reward, done, _ = env.step(action)

            # Record actions, asset holdings, and balances
            asset_tracker.record_step(action, env.actor_asset_holdings, env.actor_balance, env.balance)

            if train:
                agent.store((state, action, reward, next_state))
                agent.train()

            # Log per-step metrics
            if isinstance(reward, (list, np.ndarray)):  # Multi-agent rewards
                for agent_idx, agent_reward in enumerate(reward):
                    logger.log_scalar(f"01_{run_type.lower()}/agent_{agent_idx}_reward", agent_reward)
            else:  # Single-agent reward
                logger.log_scalar(f"01_{run_type.lower()}/agent_0_reward", reward)

            logger.log_scalar(f"01_{run_type.lower()}/portfolio_value", env.balance)

        
            asset_tracker.log(logger, tickers)


            logger.next_step()

            # Track portfolio balance
            episode_balances.append(env.balance)

            state = next_state
            total_reward += reward
            steps += 1

        # Update epsilon using the scheduler (only during training)
        if train:
            epsilon_scheduler.step(ep + 1, n_episodes)
            logger.log_scalar(f"01_{run_type.lower()}/epsilon", epsilon_scheduler.epsilon)

        # Store data
        logger.add_run_data(asset_tracker.balances, asset_tracker.actions, asset_tracker.asset_holdings)

        #########################################
        #### Log episode summary
        #########################################
        if isinstance(total_reward, (list, np.ndarray)):  # Multi-agent rewards
            for agent_idx, agent_reward in enumerate(total_reward):
                logger.log_scalar(f"01_{run_type.lower()}/agent_{agent_idx}_total_reward_of_episode", agent_reward)
        else:  # Single-agent reward
            logger.log_scalar(f"01_{run_type.lower()}/agent_0_total_reward_of_episode", total_reward)

        logger.next_step()

        # Print episode summary
        if isinstance(total_reward, (list, np.ndarray)):  # Multi-agent rewards
            agent_rewards_str = " -> ".join([f"Agent {i}: {agent_reward:.4f}" for i, agent_reward in enumerate(total_reward)])
        else:  # Single-agent reward
            agent_rewards_str = f"Agent 0: {total_reward:.4f}"

        print(f"[{run_type}] Episode {ep+1:>3} | Steps: {steps} | Rewards: {agent_rewards_str}")
        print(f"Portfolio Value: {env.balance:.2f}")
        print(f"Total Reward: {np.sum(total_reward):.4f}")
        print(f"Asset Holdings: {env.asset_holdings}")

        # Calculate and print metrics for the episode
        metrics.calculate(episode_balances)
        metrics.print_report()
        print("-" * 50)

        ## Log actions, asset holdings, and balances
        #for step in range(len(asset_tracker.actions)):
            #asset_tracker.log(logger, tickers, step)

    # Log portfolio mean and std over time
    logger.log_portfolio_statistics()

    # Save model (only during training)
    if train:
        if save_path is None:
            save_path = os.path.join(logger.run_path, "agent.pt")
        agent.save(save_path)
        print(f"✅ Agent saved to: {save_path}")

    # Save evaluation data (only during evaluation)
    if not train:
        logger.save_evaluation_data(config=config)

    logger.close()

    # Return metrics for evaluation
    if not train:
        return metrics.metrics