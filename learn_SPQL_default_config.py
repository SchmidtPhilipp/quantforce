import qf as qf


def main():

    log_dir = "runs_base_config"
    config_name = "SPQLAgent_default_config"

    qf.start_tensorboard(logdir=log_dir, port=6009)

    qf.DEFAULT_EVAL_ENV_CONFIG["log_dir"] = log_dir
    qf.DEFAULT_TRAIN_ENV_CONFIG["log_dir"] = log_dir
    qf.DEFAULT_TRAIN_ENV_CONFIG["config_name"] = config_name 
    qf.DEFAULT_EVAL_ENV_CONFIG["config_name"] = qf.DEFAULT_TRAIN_ENV_CONFIG["config_name"]

    env = qf.MultiAgentPortfolioEnv(tensorboard_prefix="TRAIN", config=qf.DEFAULT_TRAIN_ENV_CONFIG)

    agent = qf.SPQLAgent(
        env=env,
        config=qf.DEFAULT_SPQLAGENT_CONFIG,
    )

    # approximate the training steps
    agent.train(total_timesteps=100_000, use_tqdm=True)

    eval_env = qf.MultiAgentPortfolioEnv(tensorboard_prefix="EVAL", config=qf.DEFAULT_EVAL_ENV_CONFIG)
    agent.evaluate(eval_env, episodes=1)
    agent.visualize()




if __name__ == "__main__":
    main()