from qf.optim.robustness_tester import RobustnessTester
import qf

def main():
    # Define configurations
    train_env_config = qf.DEFAULT_TRAIN_ENV_CONFIG
    eval_env_config = qf.DEFAULT_EVAL_ENV_CONFIG
    agent_config = qf.DEFAULT_SACAGENT_CONFIG

    # Adjust environment settings
    env_adjustments = {
        "reward_function": "sharpe_ratio_w50",
        "trade_costs": 1,
        "trade_costs_percent": 0.01,
    }
    train_env_config.update(env_adjustments)
    eval_env_config.update(env_adjustments)

    # Initialize RobustnessTester
    tester = RobustnessTester(
        agent_class=qf.SACAgent,
        env_class=qf.MultiAgentPortfolioEnv,
        train_env_config=train_env_config,
        eval_env_config=eval_env_config,
        agent_config=agent_config,
        n_trials=10,    # Number of trials for robustness testing; Agent will be traint n_trials times
        n_evaluations=5 # Number of evaluations during training => evaluation_interval = total_timesteps // n_evaluations
    )

    # Perform training progress testing
    results = tester.test_progress(total_timesteps=50000, episodes=10) 
    # total_timesteps is the total number of timesteps for training
    # episodes is the number of episodes for evaluation after every evaluation_interval steps

    # Print results
    print("Balances Matrix Shape:", results["shape"])

    # Visualize portfolio balance
    tester.plot_portfolio_balance()

if __name__ == "__main__":
    main()