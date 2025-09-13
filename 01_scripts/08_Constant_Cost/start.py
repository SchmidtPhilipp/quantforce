import qf

qf.start_tensorboard(logdir="runs", port=6008)

# Total Time Steps
total_timesteps = 200_000
eval_every_n_steps = 10_000

########################################################
########################################################
# Create environment

train_env_config = qf.EnvConfig.get_default_train(
    trade_cost_percent=0, trade_cost_fixed=1
)

eval_env_config = qf.EnvConfig.get_default_eval(
    trade_cost_percent=0, trade_cost_fixed=1
)

val_env_config = qf.EnvConfig.get_default_eval(trade_cost_percent=0, trade_cost_fixed=1)


train_env = qf.MultiAgentPortfolioEnv("TRAIN", n_agents=1, env_config=train_env_config)


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
    ("SAC", qf.SACAgent, qf.SACConfig.get_default_config()),
    ("PPO", qf.PPOAgent, qf.PPOConfig.get_default_config()),
    ("DDPG", qf.DDPGAgent, qf.DDPGConfig.get_default_config()),
    ("MADDPG", qf.MADDPGAgent, qf.MADDPGConfig.get_default_config()),
]

for name, agent_class, config in agents_to_compare:
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
        seeds=[1, 2, 3],
    )

    print(f"✓ {name} completed successfully")
