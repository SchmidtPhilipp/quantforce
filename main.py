import qf as qf


def main():

    env = qf.MultiAgentPortfolioEnv(config=qf.DEFAULT_TRAIN_ENV_CONFIG)

    #agent = qf.TangencyAgent(env)
    agent = qf.DQNAgent(env, **qf.DEFAULT_DQN_AGENT_CONFIG)
    #agent = qf.SACAgent(env, config=qf.DEFAULT_SAC_AGENT_CONFIG)
    #agent = qf.TD3Agent(env, config=qf.DEFAULT_TD3_AGENT_CONFIG)

    agent.train(total_timesteps=5000, use_tqdm=True)

    eval_env = qf.MultiAgentPortfolioEnv(config=qf.DEFAULT_EVAL_ENV_CONFIG)
    agent.evaluate(eval_env)
    agent.visualize()

    eval_env


if __name__ == "__main__":
    qf.start_tensorboard()
    main()