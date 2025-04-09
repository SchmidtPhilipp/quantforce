from train.logger import Logger
from utils.metrics import (
    calculate_returns, sharpe_ratio, sortino_ratio,
    max_drawdown, volatility, cumulative_return,
    annualized_return, calmar_ratio
)
from utils.log_weights import log_weights
import numpy as np


def runs_single_evaluation(env, agent, config):
    state = env.reset()
    done = False

    # Initialize balances
    balances = [env.balance]

    # Initialize weights over time
    actions = []
    initial_weigths = np.zeros((env.n_agents, env.n_assets + 1))
    initial_weigths[:,-1] = np.ones((env.n_agents))
    actions.append(initial_weigths)

    # Initialize asset holdings
    asset_holdings = []
    initial_asset_holdings = np.zeros((env.n_assets))
    asset_holdings.append(initial_asset_holdings)

    # Run one evaluation loop
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        balances.append(env.balance)
        state = next_state

        actions.append(action) # Note have dim (runs, n_agents, n_assets + 1)
        asset_holdings.append(env.asset_holdings)

    if config["verbosity"] > 0:
        print(f"📈 Evaluation Summary:")
        print(f"Final Portfolio Value: {balances[-1]:.2f}")
        print(f"Final Asset Holdings: {env.asset_holdings}")
        print("-" * 50)

    return balances, actions, asset_holdings


def evaluate_agent(env, agent, config, run_name=None, n_runs=100):
    tickers = config["tickers"]
    run_name = "EVAL_" + (run_name or "default")
    logger = Logger(run_name=run_name)

    metrics = {
        "sharpe": [],
        "sortino": [],
        "drawdown": [],
        "volatility": [],
        "cumulative": [],
        "annualized": [],
        "calmar": [],
    }

    for run in range(n_runs):
        balances, weights_over_time, asset_holdings = runs_single_evaluation(env, agent, config)

        # Store data
        logger.add_run_data(balances, weights_over_time, asset_holdings)
        logger.next_step()
    
        # Compute metrics
        returns = calculate_returns(balances)
        drawdown = max_drawdown(balances)
        annual_ret = annualized_return(balances)

        metrics["sharpe"].append(sharpe_ratio(returns))
        metrics["sortino"].append(sortino_ratio(returns))
        metrics["drawdown"].append(drawdown)
        metrics["volatility"].append(volatility(returns))
        metrics["cumulative"].append(cumulative_return(balances))
        metrics["annualized"].append(annual_ret)
        metrics["calmar"].append(calmar_ratio(annual_ret, drawdown))

    # Log portfolio mean and std over time
    balances_array = np.array(logger.balances)  # shape: (n_runs, n_steps)
    mean_balances = np.mean(balances_array, axis=0)
    std_balances = np.std(balances_array, axis=0)

    # calculate the mean and std of the weights
    actions = np.array(logger.weights)
    mean_actions = np.mean(actions, axis=0)
    std_actions = np.std(actions, axis=0)

    # calculate mean and stad of the asset holdings
    

    logger.step = 0

    for t in range(len(mean_balances)):
        logger.log_scalar("02_eval/portfolio_value_mean", mean_balances[t])
        logger.log_scalar("02_eval/portfolio_value_std", std_balances[t])
        log_weights(logger, tickers, mean_actions[t,:,:], t)
        logger.next_step()

    logger.step = 0
    
    # Log metrics as scalars + emulated histograms
    logger.log_metrics(metrics)
    for name, values in metrics.items():
        logger.log_emulated_histogram(name, values, bins=10)

    # Save everything to disk
    logger.save_evaluation_data(config=config)
    logger.close()

    # Print summary
    print(f"\n📊 Evaluation Summary ({n_runs} runs):")
    for name, values in metrics.items():
        if name in ["sharpe", "sortino", "calmar"]:  # Ratios
            print(f"{name.capitalize():<20s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        elif name in ["drawdown", "cumulative", "annualized"]:  # Percentages
            print(f"{name.capitalize():<20s}: {np.mean(values) * 100:.2f}% ± {np.std(values) * 100:.2f}%")
        elif name == "volatility":  # Volatility as percentage
            print(f"{name.capitalize():<20s}: {np.mean(values) * 100:.2f}% ± {np.std(values) * 100:.2f}%")
        else:
            print(f"{name.capitalize():<20s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print("✅ Evaluation completed, logged, and saved.")
    return metrics
