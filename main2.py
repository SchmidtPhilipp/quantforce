import qf as qf


def main():

    print(qf.DEFAULT_TICKERS)
    env = qf.MultiAgentPortfolioEnv(**qf.DEFAULT_TRAIN_ENV_CONFIG)

    agent = qf.TangencyAgent(env)

    agent.train()

    eval_env = qf.MultiAgentPortfolioEnv(**qf.DEFAULT_EVAL_ENV_CONFIG)
    agent.evaluate(eval_env)
    #agent.visualize()

    eval_env


if __name__ == "__main__":
    qf.start_tensorboard()
    main()