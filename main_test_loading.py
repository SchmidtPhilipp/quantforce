import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

import qf
from qf.agents.sb3_agents.sac_agent import SACAgent
from qf.envs.multi_agent_portfolio_env import MultiAgentPortfolioEnv
from qf.envs.sb3_wrapper import SB3Wrapper


def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create environment
    env_config = qf.DEFAULT_ENV_CONFIG

    # Create environment
    env = MultiAgentPortfolioEnv(tensorboard_prefix="test_loading", config=env_config)

    # Create and train agent
    print("Creating and training initial agent...")
    agent = SACAgent(env)
    agent.train(total_timesteps=10, use_tqdm=True)

    # Save agent
    save_dir = os.path.join(env.get_save_dir(), "checkpoint")
    print(f"Saving agent to {save_dir}")
    agent.save(save_dir)

    # Load agent
    print("Loading agent...")
    loaded_agent = SACAgent.load_agent(save_dir, env=env)

    # Continue training
    print("Continuing training with loaded agent...")
    loaded_agent.train(total_timesteps=10, use_tqdm=True)

    print("Test completed successfully!")


if __name__ == "__main__":
    main()
