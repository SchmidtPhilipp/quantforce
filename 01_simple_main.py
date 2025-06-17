# Minimal example
import qf


def main():

    qf.start_tensorboard(port=6007, logdir="runs")

    # Set the reward function
    reward_function = "sharpe_ratio_w10_10"
    # reward_function = "differential_sharpe_ratio"
    # Edit the config
    config = {
        "reward_function": reward_function,
        "verbosity": 1,
    }

    # Initialize the environment and agent
    train_env = qf.MultiAgentPortfolioEnv(
        tensorboard_prefix="TRAIN",
        config=config,
    )  # Note optional config
    eval_env = qf.MultiAgentPortfolioEnv(
        tensorboard_prefix="EVAL",
        config=config,
    )  # Note optional config
    agent = qf.SACAgent(env=train_env)  # Note optional config
    # Alternatively one of the other agents can be used:
    # agent = qf.SPQL(env=train_env)
    agent = qf.ClassicOnePeriodMarkovitzAgent(env=train_env)

    # Train the agent
    agent.train(total_timesteps=500_000, use_tqdm=True)

    # Evaluate the agent
    agent.evaluate(eval_env, episodes=1, use_tqdm=True)


if __name__ == "__main__":
    main()
