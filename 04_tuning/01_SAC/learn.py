# Import the root folder of this folder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qf as qf

def main():

    log_dir = "runs_base_config"
    config_name = "SACAgent_default_config"

    qf.start_tensorboard(logdir=log_dir, port=6041)

    qf.DEFAULT_EVAL_ENV_CONFIG["log_dir"] = log_dir
    qf.DEFAULT_TRAIN_ENV_CONFIG["log_dir"] = log_dir
    qf.DEFAULT_TRAIN_ENV_CONFIG["config_name"] = config_name 
    qf.DEFAULT_EVAL_ENV_CONFIG["config_name"] = config_name

    # For SAC, we need to scale the reward.
    qf.DEFAULT_TRAIN_ENV_CONFIG["reward_scaling"] = 100
    qf.DEFAULT_EVAL_ENV_CONFIG["reward_scaling"] = 100



    config = {
        "policy": "MlpPolicy",  # Default policy architecture
        "learning_rate": qf.DEFAULT_SAC_LR,
        "buffer_size": qf.DEFAULT_SAC_BUFFER_MAX_SIZE,
        "batch_size": qf.DEFAULT_SAC_BATCH_SIZE,
        "tau": qf.DEFAULT_SAC_TAU,  # Target network update rate
        "gamma": qf.DEFAULT_SAC_GAMMA,
        "train_freq": qf.DEFAULT_SAC_TRAIN_FREQ,  # Frequency of training steps
        "gradient_steps": 4,  # Number of gradient steps per training iteration
        "device": qf.DEFAULT_DEVICE,  # Device to run the computations on
        "ent_coef": "0.2",  # Entropy coefficient for exploration
        "verbose": qf.DEFAULT_SAC_VERBOSITY  # Verbosity level for logging
    }

    env = qf.MultiAgentPortfolioEnv(tensorboard_prefix="TRAIN", config=qf.DEFAULT_TRAIN_ENV_CONFIG)

    agent = qf.SACAgent(
        env=env,
        config=config,
    )

    # approximate the training steps
    agent.train(total_timesteps=500_000, use_tqdm=True)

    eval_env = qf.MultiAgentPortfolioEnv(tensorboard_prefix="EVAL", config=qf.DEFAULT_EVAL_ENV_CONFIG)
    agent.evaluate(eval_env, episodes=1)
    agent.visualize()




if __name__ == "__main__":
    main()