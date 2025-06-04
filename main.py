import qf as qf


def main():

    env = qf.MultiAgentPortfolioEnv(tensorboard_prefix="TRAIN", config=qf.DEFAULT_TRAIN_ENV_CONFIG)

    agent = qf.ClassicOnePeriodMarkovitzAgent(env, config=qf.DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZAGENT_CONFIG)
    #agent = qf.DQNAgent(env, config=qf.DEFAULT_DQNAGENT_CONFIG)
    #agent = qf.SACAgent(env, config=qf.DEFAULT_SACAGENT_CONFIG)
    #agent = qf.TD3Agent(env, config=qf.DEFAULT_TD3AGENT_CONFIG)
    #agent = qf.MADDPGAgent(env, config=qf.DEFAULT_MADDPGAGENT_CONFIG)

    agent.train(total_timesteps=10, use_tqdm=True)

    eval_env = qf.MultiAgentPortfolioEnv(tensorboard_prefix="EVAL", config=qf.DEFAULT_EVAL_ENV_CONFIG)
    agent.evaluate(eval_env, episodes=2)
    agent.visualize()




if __name__ == "__main__":
    qf.start_tensorboard()
    main()