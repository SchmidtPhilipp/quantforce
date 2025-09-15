import pandas as pd

import qf

# Create environment
train_env = qf.MultiAgentPortfolioEnv(
    "TRAIN", n_agents=1, env_config=qf.EnvConfig.get_default_train()
)

eval_env = qf.MultiAgentPortfolioEnv(
    "EVAL", n_agents=1, env_config=qf.EnvConfig.get_default_eval()
)

# Define different agent configurations to compare
# Each tuple contains: (name, agent_class, config)
agents_to_compare = [
    # Classic Agents
    (
        "Tangency_Sample",
        qf.ClassicOnePeriodMarkowitzAgent,
        qf.agents.ClassicOnePeriodMarkowitzAgentConfig(
            target="Tangency",
            risk_model="sample_cov",
        ),
    ),
    (
        "Tangency_ML",
        qf.ClassicOnePeriodMarkowitzAgent,
        qf.agents.ClassicOnePeriodMarkowitzAgentConfig(
            target="Tangency",
            risk_model="ML_brownian_motion_logreturn",
        ),
    ),
    ("1/N", qf.OneOverNPortfolioAgent, qf.agents.OneOverNPortfolioAgentConfig()),
    ("Merton", qf.HJBPortfolioAgent, qf.agents.HJBPortfolioAgentConfig()),
    ("SAC", qf.SACAgent, qf.agents.SACConfig()),
    ("PPO", qf.PPOAgent, qf.agents.PPOConfig()),
    ("DDPG", qf.DDPGAgent, qf.agents.DDPGConfig()),
    ("MADDPG", qf.MADDPGAgent, qf.agents.MADDPGConfig()),
]

# Train and evaluate each agent
runs = []
for name, agent_class, config in agents_to_compare:
    print(f"\nTraining {name}...")

    try:
        # Create agent with specific configuration
        agent = agent_class(train_env, config=config)

        # Train agent (shorter for RL agents)
        if agent_class in [
            qf.DQNAgent,
            qf.SACAgent,
            qf.PPOAgent,
            qf.A2CAgent,
            qf.DDPGAgent,
            qf.TD3Agent,
        ]:
            # RL agents need more training
            agent.train(total_timesteps=5000)
        else:
            # Classic and modern agents train quickly or do not need training
            agent.train(total_timesteps=1)

        # Hard reset the environment (important to reset the data collector)
        eval_env = qf.MultiAgentPortfolioEnv(
            "EVAL", n_agents=1, env_config=qf.EnvConfig.get_default_eval()
        )

        # Evaluate agent
        agent.evaluate(episodes=1, eval_env=eval_env)

        # Collect results
        run = eval_env.data_collector
        run.rename(name)
        runs.append(run)

        print(f"✓ {name} completed successfully")

    except Exception as e:
        print(f"✗ {name} failed: {str(e)}")
        continue

# Visualize results
if runs:
    print("\nPlotting results...")

    # Plot comparison with confidence intervals
    frame = [run.get_frame() for run in runs]

    frames = qf.PlotFrame(pd.concat(frame, ignore_index=False, axis=1))
    frames.plot_balance()
    frames.plot_rewards()
    frames.plot_actions()
    frames.plot_asset_holdings()
    frames.metrics_table_comparison(frames, tranposed=False)

    print(
        f"\nComparison complete! Successfully trained and evaluated {len(runs)} different agent configurations."
    )
else:
    print("\nNo agents were successfully trained. Check the error messages above.")
