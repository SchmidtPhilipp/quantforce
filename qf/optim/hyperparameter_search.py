import itertools
import qf as qf
from pprint import pprint


def hyperparameter_search(env_config, agent_classes_with_param_grids, eval_env_config, max_timesteps=5000, episodes=10, use_tqdm=True):
    """
    Conducts a hyperparameter search for multiple agent classes with different hyperparameter spaces.
    Parameters:
        env_config (dict): Configuration for the training environment.
        agent_classes_with_param_grids (dict): Dictionary mapping agent classes to their specific hyperparameter grids.
        eval_env_config (dict): Configuration for the evaluation environment.
        max_timesteps (int): Number of timesteps for training.
        episodes (int): Number of episodes for evaluation.
    Returns:
        dict: Best agent class, best hyperparameter configuration, and its corresponding reward.
    """
    best_config = None
    best_reward = float('-inf')
    best_agent_class = None

    for agent_class, param_grid in agent_classes_with_param_grids.items():
        # Generate all combinations of hyperparameters for the current agent class
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())

        for params in param_combinations:
            # Create a configuration dictionary for the current combination
            config = {param_names[i]: params[i] for i in range(len(params))}

            # Merge with default configuration
            default_config = getattr(qf, f"DEFAULT_{agent_class.__name__.upper()}_CONFIG", {})
            merged_config = {**default_config, **config}

            # Set the environment's config_name to reflect the current hyperparameter sweep
            env_config["config_name"] = f"{agent_class.__name__}{'_'.join([f'{k}_{v}' for k, v in config.items()])}"

            print("#"*50)
            print(f"Testing {agent_class.__name__} configuration:")
            pprint(merged_config)
            print(f"Environment: {env_config['config_name']}")

            # Initialize the environment and agent
            env = qf.MultiAgentPortfolioEnv(tensorboard_prefix="TRAIN", config=env_config)
            agent = agent_class(env, config=merged_config)

            # Train the agent
            agent.train(total_timesteps=max_timesteps, use_tqdm=use_tqdm)

            # Save the data
            env.save_and_close()
            save_path = env.get_save_path()
            agent.save(save_path)

            # Evaluate the agent
            eval_env_config["config_name"] = env_config["config_name"]
            eval_env = qf.MultiAgentPortfolioEnv(tensorboard_prefix="EVAL", config=eval_env_config)
            avg_reward = agent.evaluate(eval_env, episodes=episodes, use_tqdm=use_tqdm)

            # Save the data
            env.save_and_close()
            save_path = env.get_save_path()
            agent.save(save_path)

            print("-"*50)
            print(f"Average reward: {avg_reward} for configuration:")
            pprint(merged_config)
            print("#"*50)

            # Update the best configuration if the current one is better
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_config = merged_config
                best_agent_class = agent_class

    return {"best_agent_class": best_agent_class, "best_config": best_config, "best_reward": best_reward}