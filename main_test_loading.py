import os

import qf.agents

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

import qf
from qf.envs.sb3_wrapper import SB3Wrapper


def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create environment
    env_config = qf.DEFAULT_ENV_CONFIG

    # Create environment
    env = qf.MultiAgentPortfolioEnv(
        tensorboard_prefix="test_loading", config=env_config
    )

    # Create and train agent
    print("Creating and training initial agent...")
    # agent = qf.SPQLAgent(env)
    # agent = qf.TD3Agent(env)
    # agent = qf.SACAgent(env)
    # agent = qf.PPOAgent(env)
    # agent = qf.DDPGAgent(env)
    # agent = qf.MADDPGAgent(env)
    # agent = qf.ClassicOnePeriodMarkovitzAgent(env)
    agent = qf.A2CAgent(env)

    agent.train(
        total_timesteps=1,
        use_tqdm=True,
        save_best=True,
        # eval_env=env,
        # n_eval_steps=1,
    )

    # Save agent
    save_dir = os.path.join(env.get_save_dir())
    print(f"Saving agent to {save_dir}")
    agent.env.save_data()
    agent.save(save_dir)

    # Load agent
    print("Loading agent...")
    loaded_agent = qf.Agent.load_agent(save_dir)

    # Continue training
    print("Continuing training with loaded agent...")
    loaded_agent.train(total_timesteps=10, use_tqdm=True)

    print("Test completed successfully!")


if __name__ == "__main__":
    main()
