from .logger import Logger
from utils.metrics import Metrics
import os
from utils.tracker.assettracker import AssetTracker
from tqdm import tqdm  # Import tqdm for progress tracking
import numpy as np
import torch

def risk_limiting_strategy(actions, last_actions, n_agents):
    # Constrain actions to an epsilon region around the previous actions
    epsilon = 0.01  # Initial epsilon value
    max_epsilon = 0.5  # Maximum epsilon value
    epsilon_growth_rate = 0.001  # How much epsilon grows per step

    constrained_actions = []
    for i in range(n_agents):
        # Calculate the allowed epsilon region
        lower_bound = last_actions[i] - epsilon
        upper_bound = last_actions[i] + epsilon

        # Clip the actions to the epsilon region
        constrained_action = np.clip(actions[i], lower_bound, upper_bound)
        constrained_actions.append(constrained_action)

        # Update epsilon based on the deviation
        deviation = np.abs(constrained_action - last_actions[i]).mean()
        epsilon = max(0.1, epsilon + epsilon_growth_rate - deviation * 0.05)
        epsilon = min(epsilon, max_epsilon)

    actions = np.array(constrained_actions)
    return actions




class TimeVariantActionFilter:
    def __init__(self, n_agents, n_assets, epsilon=0.2, alpha=0.01):
        self.n_agents = n_agents
        self.epsilon_max = epsilon # maximum epsilon value
        self.epsilon = epsilon # Epsilon schlauch
        self.alpha = alpha # acceleration of the filter
        self.n_assets = n_assets

        self.epsilon = np.ones((n_agents, n_assets)) * epsilon # Epsilon schlauch for each agent
    
    def filter(self, actions: torch.Tensor, last_actions: torch.Tensor, n_agents, epsilon=0.1):

        for i in range(n_agents):
            for a in range(len(actions[i])):
                last_action = last_actions[i][a]
                action = actions[i][a]
                if action > last_action:
                    # we map the distance between the last action and 1 to the distance between the last action + epsilon and 1
                    # which will map the current action into the range [last_actionm, last_action + epsilon]
                    actions[i][a] = last_action + epsilon * (action - last_action) / (1 - last_action)
                else: # action < last_action:
                    # we map the distance between the last action and 0 to the distance between the last action - epsilon and 0
                    # which will map the current action into the range [last_action - epsilon, last_action]
                    actions[i][a] = last_action - epsilon * (last_action - action) / last_action

                distance = actions[i][a] - last_action

                # if the distance is larger than half epsilon then we decrease epsilon for the next step
                if abs(distance) > epsilon / 2:
                    self.epsilon = max(0.1, self.epsilon - self.alpha * abs(distance))
                # if the distance is smaller than half epsilon then we increase epsilon for the next step
                else:
                    self.epsilon = min(self.epsilon_max, self.epsilon + self.alpha * abs(distance))

            # nomalize the actions
            actions[i] = actions[i] / actions[i].sum()
        return actions

# Implement CPPI risk limiting strategy
#def cppi_risk_limiting_strategy(actions, last_actions, n_agents, cushion=0.1):


class CPPIActionFilter:
    def __init__(self, n_agents, cushion=0.1):
        self.n_agents = n_agents
        self.cushion = cushion

    def filter(self, actions, last_actions, n_agents):
        for i in range(n_agents):
            # Calculate the cushion
            cushion = np.maximum(0, last_actions[i] - self.cushion)
            # Apply the CPPI strategy
            actions[i] = actions[i] * (1 + cushion)
            # Normalize the actions
            actions[i] = actions[i] / np.sum(actions[i])
        return actions



def run_agent(env, agent, config, save_path=None, n_episodes=10, run_name=None, epsilon_scheduler=None, train=True, use_tqdm=True):
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
        use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print episode summaries.

    Returns:
        dict: Metrics collected during the run (for evaluation).
    """
    run_type = "TRAIN" if train else "EVAL"
    run_name = f"{run_type}_" + (run_name or "default")
    logger = Logger(run_name=run_name)
    metrics = Metrics()
    #action_filter = TimeVariantActionFilter(env.n_agents, epsilon=0.2, alpha=0.01)
   
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

    # Keep track of the best reward
    best_reward = -np.inf
    best_episode = 0
    best_agent = None
    epsiodes_no_improvement = 0 
    ep = 0
    done = False

    while not done:
        state = env.reset()
        total_reward = 0
        episode_done = False
        steps = 0
        last_action = np.zeros((env.n_agents, env.n_assets+1))  # Initialize last actions for each agent
        # Initialize tqdm progress bar if enabled
        progress_bar = tqdm(total=env.get_timesteps(), desc=f"{run_type} Episode {ep+1}", unit="step", ncols=80) if use_tqdm else None
        while not episode_done:
            # Use epsilon-greedy policy for exploration during training
            epsilon = epsilon_scheduler.epsilon if train else 0.0
            action = agent.act(state, epsilon=epsilon)

            #action = action_filter.filter(action, last_action, env.n_agents)  # Apply action filter
            action = risk_limiting_strategy(action, last_action, env.n_agents)  # Apply risk-limiting strategy


            # TODO: Apply risk-limiting strategy

            next_state, reward, episode_done, _ = env.step(action)

            # Record actions, asset holdings, and balances
            asset_tracker.record_step(action, env.actor_asset_holdings, env.actor_balance, env.balance, reward)

            if train:
                agent.store((state, action, reward, next_state))
                agent.train()

            state = next_state
            total_reward += reward
            steps += 1

            # Update the unified progress bar if enabled
            if use_tqdm:
                progress_bar.update(1)

            last_action = action  # Update last actions for the next step

        asset_tracker.log_episode(logger)
        
        # Close tqdm progress bar if enabled
        if use_tqdm:
            progress_bar.close()


        # Append metrics for this episode
        metrics.append(asset_tracker.get_episode_data("balances", ep))

        # Update epsilon using the scheduler (only during training)
        if train:
            epsilon_scheduler.step(ep + 1, n_episodes)
            logger.log_scalar(f"{run_type}_epsilon/epsilon", epsilon_scheduler.epsilon, step=ep)
            
            asset_tracker.print_episode_summary(run_type=run_type, episode=ep)
            metrics.print_report()
            print("-" * 50)
            metrics.reset()

        # End the episode
        asset_tracker.end_episode() # Last task of the episode !!

        # Save the best agent based on total reward
        if train and total_reward.sum() > best_reward:
            best_reward = total_reward.sum()
            best_episode = ep
            best_agent = agent
            epsiodes_no_improvement = 0
        else:
            epsiodes_no_improvement += 1
            if epsiodes_no_improvement > n_episodes:
                # Early stopping if no improvement for 10% of the episodes
                print(f"Early stopping at episode {ep} due to no improvement.")
                done = True

        ep += 1
                

    # Log portfolio mean and std over time
    if not train:
        asset_tracker.log_statistics(logger)  # Only makes sense for evaluation
        metrics.print_report()
        metrics.log_metrics(logger, run_type=run_type)

    # Save model (only during training)
    if train:
        agent.save(os.path.join(logger.run_path, "last_agent.pt"))
        # save the best agent if it is not the last one
        if best_agent is not None and best_episode != n_episodes - 1:
            best_agent.save(os.path.join(logger.run_path, f"best_agent_{best_episode}.pt"))
            print(f"Best agent saved at episode {best_episode} with reward {best_reward}")

    # Close the logger
    logger.close()

    # Save asset tracker data
    asset_tracker.save(logger.run_path)

    # Save config data
    config.save(os.path.join(logger.run_path, "config.json"))

    # Return metrics for evaluation
    if not train:
        return metrics.metrics
    


