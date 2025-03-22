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

            agent.store((state, action, reward, next_state))
            agent.train()

            # Log per-step metrics
            logger.log_scalar("01_train/step_reward", reward)
            logger.log_scalar("01_train/portfolio_value", env.balance)
            logger.next_step()

            state = next_state
            total_reward += reward
            steps += 1

        # Log episode summary
        logger.log_scalar("01_train/total_reward_of_episode", total_reward)
        logger.next_step()

        print(f"[Train] Episode {ep+1:>3} | Total Reward: {total_reward:.4f} | Steps: {steps}")

    # ✅ Save model into run folder
    if save_path is None:
        save_path = os.path.join(logger.run_path, "agent.pt")


    logger.close()
    agent.save(save_path)
    print(f"✅ Agent saved to: {save_path}")

    return save_path
