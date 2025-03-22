from agents.dqn_agent import DQNAgent
from envs.live_env import LiveTradingEnv
import time

def run_live_loop(agent, tickers=["AAPL", "MSFT"], interval_sec=60):
    env = LiveTradingEnv(tickers=tickers)
    obs = env.reset()

    while True:
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        # Optional online learning:
        # agent.store((old_obs, action, reward, obs))
        # agent.train()

        time.sleep(interval_sec)  # Wait for new data
