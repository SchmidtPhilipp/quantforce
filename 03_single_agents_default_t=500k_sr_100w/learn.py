# Import the root folder of this folder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import qf as qf


def main():

    log_dir = "03_single_agents_default_t=500k_sr_100w/runs"
    config = "_default_config_t=500k_sr_100w"

    qf.start_tensorboard(logdir=log_dir, port=6006)
    

    agents = [qf.DDPGAgent, 
            qf.SACAgent, 
            qf.TD3Agent, 
            qf.PPOAgent, 
            qf.SPQLAgent, 
            qf.ClassicOnePeriodMarkovitzAgent]
    
    for agent_class in agents:
        config_name = agent_class.__name__ + config
        qf.DEFAULT_EVAL_ENV_CONFIG["log_dir"] = log_dir
        qf.DEFAULT_TRAIN_ENV_CONFIG["log_dir"] = log_dir
        qf.DEFAULT_TRAIN_ENV_CONFIG["config_name"] = config_name 
        qf.DEFAULT_EVAL_ENV_CONFIG["config_name"] = qf.DEFAULT_TRAIN_ENV_CONFIG["config_name"]

        qf.DEFAULT_TRAIN_ENV_CONFIG["reward_function"] = "sharpe_ratio_w100"
        qf.DEFAULT_EVAL_ENV_CONFIG["reward_function"] = "sharpe_ratio_w100"

        if agent_class == qf.SACAgent:
            qf.DEFAULT_TRAIN_ENV_CONFIG["reward_scaling"] = 100
            qf.DEFAULT_EVAL_ENV_CONFIG["reward_scaling"] = 100

        env = qf.MultiAgentPortfolioEnv(tensorboard_prefix="TRAIN", config=qf.DEFAULT_TRAIN_ENV_CONFIG)

        agent = agent_class(env=env)

        # approximate the training steps
        agent.train(total_timesteps=500_000, use_tqdm=True)

        eval_env = qf.MultiAgentPortfolioEnv(tensorboard_prefix="EVAL", config=qf.DEFAULT_EVAL_ENV_CONFIG)
        agent.evaluate(eval_env, episodes=1)#


    





if __name__ == "__main__":
    main()