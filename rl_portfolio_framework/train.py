from trainer.logger import Logger
import os

def train_agent(env, agent, save_path=None, n_episodes=10, run_name=None):
    run_name = "TRAIN_" + (run_name or "default")
    logger = Logger(run_name=run_name)

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            # Überprüfen Sie die Form der Zustände
            if state.shape != next_state.shape:
                raise ValueError(f"Inconsistent state shapes: {state.shape} vs {next_state.shape}")

            agent.store((state, action, reward, next_state))
            agent.train()

            # Log per-step metrics
            for agent_idx, agent_reward in enumerate(reward):
                logger.log_scalar(f"01_train/agent_{agent_idx}_reward", agent_reward)
            logger.log_scalar("01_train/portfolio_value", env.balance)
            logger.next_step()

            state = next_state
            total_reward += reward
            steps += 1

        # Log episode summary
        for agent_idx, agent_reward in enumerate(total_reward):
            logger.log_scalar(f"01_train/agent_{agent_idx}_total_reward_of_episode", agent_reward)

        logger.next_step()

        # Print episode summary for multiple agents
        agent_rewards_str = " -> ".join([f"Agent {i}: {agent_reward:.4f}" for i, agent_reward in enumerate(total_reward)])
        print(f"[Train] Episode {ep+1:>3} | Steps: {steps} | Rewards: {agent_rewards_str}")
        
    # ✅ Save model into run folder
    if save_path is None:
        save_path = os.path.join(logger.run_path, "agent.pt")

    logger.close()
    agent.save(save_path)
    print(f"✅ Agent saved to: {save_path}")

    return save_path