import qf as qf


def main():

    env = qf.MultiAgentPortfolioEnv(tensorboard_prefix="TRAIN", config=qf.DEFAULT_TRAIN_ENV_CONFIG)

    #agent = qf.TangencyAgent(env)
    #agent = qf.DQNAgent(env, config=qf.DEFAULT_DQNAGENT_CONFIG)
    #agent = qf.SACAgent(env, config=qf.DEFAULT_SACAGENT_CONFIG)
    #agent = qf.TD3Agent(env, config=qf.DEFAULT_TD3AGENT_CONFIG)
    agent = qf.MADDPGAgent(env, config=qf.DEFAULT_MADDPGAGENT_CONFIG)

    agent.train(total_timesteps=5000, use_tqdm=True)

    eval_env = qf.MultiAgentPortfolioEnv(tensorboard_prefix="EVAL", config=qf.DEFAULT_EVAL_ENV_CONFIG)
    agent.evaluate(eval_env)
    agent.visualize()

    eval_env


if __name__ == "__main__":
    qf.start_tensorboard()
    main()