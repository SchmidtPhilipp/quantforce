import qf

qf.start_tensorboard(logdir="runs", port=1003)

# Total Time Steps
total_timesteps = 200_000
eval_every_n_steps = 10_000

########################################################
########################################################
# Create environment

train_env_config = qf.EnvConfig.get_default_train(
    trade_cost_percent=0, trade_cost_fixed=0
)

eval_env_config = qf.EnvConfig.get_default_train(
    trade_cost_percent=0, trade_cost_fixed=0
)

val_env_config = qf.EnvConfig.get_default_train(
    trade_cost_percent=0, trade_cost_fixed=0
)


train_env = qf.MultiAgentPortfolioEnv("TRAIN", n_agents=1, env_config=train_env_config)

N_random_portfolios = 100

# Define different agent configurations to compare
# Each tuple contains: (name, agent_class, config)
agents_to_compare = [
    # Classic Agents
    (
        "Tangency Portfolio",
        qf.ClassicOnePeriodMarkowitzAgent,
        qf.agents.ClassicOnePeriodMarkowitzAgentConfig(
            target="Tangency",
            risk_model="sample_cov",
        ),
        [1],
    ),
    (
        "Random Portfolio",
        qf.RandomAgent,
        qf.agents.RandomAgentConfig.get_default_config(),
        list(range(1, N_random_portfolios)),
    ),
]

for name, agent_class, config, seeds in agents_to_compare:
    print(f"\nTraining {name}...")

    # Create agent with specific configuration
    agent = agent_class(train_env, config=config)

    # RL agents need more training
    agents, train_runs, train_eval_runs, eval_runs = agent.multi_seeded_run(
        total_timesteps=total_timesteps,
        eval_env_config=eval_env_config,
        eval_every_n_steps=eval_every_n_steps,
        n_eval_episodes=1,
        val_env_config=val_env_config,
        val_episodes=1,
        seeds=seeds,
    )

    print(f"✓ {name} completed successfully")
