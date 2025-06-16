# Minimal example
import qf


def main():

    # Initialize the environment and agent
    train_env = qf.MultiAgentPortfolioEnv(
        tensorboard_prefix="TRAIN"
    )  # Note optional config
    eval_env = qf.MultiAgentPortfolioEnv(
        tensorboard_prefix="EVAL"
    )  # Note optional config
    agent = qf.SACAgent(env=train_env)  # Note optional config
    # Alternatively one of the other agents can be used:
    # agent = qf.SPQL(env=train_env)
    agent = qf.ClassicOnePeriodMarkovitzAgent(env=train_env)

    # Train the agent
    agent.train(total_timesteps=10000, use_tqdm=True)

    # Evaluate the agent
    agent.evaluate(eval_env, episodes=1, use_tqdm=True)


if __name__ == "__main__":
    main()
