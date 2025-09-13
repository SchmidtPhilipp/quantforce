import qf

# Create environment
train_env = qf.MultiAgentPortfolioEnv(
    "TRAIN", n_agents=1, env_config=qf.EnvConfig.get_default_train()
)

# Create agent
agent = qf.ClassicOnePeriodMarkowitzAgent(train_env)

# Train agent
agent.train(total_timesteps=2000)

# Evaluate agent
eval_env = qf.MultiAgentPortfolioEnv(
    "EVAL", n_agents=1, env_config=qf.EnvConfig.get_default_eval()
)

agent.evaluate(episodes=1, eval_env=eval_env)

run = eval_env.data_collector
run.rename("Tangency")

# Visualize results
frame = run.get_frame()

# Plot the results
frame.plot_balance()
frame.plot_rewards()
frame.plot_actions()
frame.plot_asset_holdings()
frame.plot_cash()

# Plot results with confidence intervals (here meaningless because we only have one run)
qf.PlotFrame(frame).plot_confidence_actions([frame], mean_of_level="run")
qf.PlotFrame(frame).plot_confidence_asset_holdings([frame], mean_of_level="run")
qf.PlotFrame(frame).plot_confidence_balance([frame], mean_of_level="run")
qf.PlotFrame(frame).plot_confidence_rewards([frame], mean_of_level="run")
qf.PlotFrame(frame).plot_confidence_cash([frame], mean_of_level="run")
