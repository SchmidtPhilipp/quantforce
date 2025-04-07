from train.logger import Logger
from utils.metrics import (
    calculate_returns, sharpe_ratio, sortino_ratio,
    max_drawdown, volatility, cumulative_return,
    annualized_return, calmar_ratio
)
from utils.log_weights import log_weights
import numpy as np
from envs.multi_agent_individual_action_portfolio_env import calculate_actions_from_individual_actions


def runs_single_evaluation(env, agent, config):
    state = env.reset()
    done = False

    # Initialize balances
    balances = [env.balance]

    # Initialize weights over time
    weights_over_time = []
    initial_weigths = np.zeros((env.n_assets + 1))
    initial_weigths[-1] = 1
    weights_over_time.append(initial_weigths)

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
        
        # Here i have to check if the action is shared or not
        if not config["shared_action"]:
            asset_weight, cash_weight = calculate_actions_from_individual_actions(action, env.n_agents)
            action = np.concatenate((asset_weight, [cash_weight]), axis=0)
        elif not config["shared_obs"]:
            action = np.mean(action, axis=0)
        
        weights_over_time.append(action)
        asset_holdings.append(env.asset_holdings)

    if config["verbosity"] > 0:
        print(f"📈 Evaluation Summary:")
        print(f"Final Portfolio Value: {balances[-1]:.2f}")
        print(f"Final Asset Holdings: {env.asset_holdings}")
        print("-" * 50)

    return balances, weights_over_time, asset_holdings


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
    all_weights = np.array(logger.weights)
    mean_weights = np.mean(all_weights, axis=0)
    std_weights = np.std(all_weights, axis=0)

    # calculate mean and stad of the asset holdings
    

    logger.step = 0

    for t in range(len(mean_balances)):
        logger.log_scalar("02_eval/portfolio_value_mean", mean_balances[t])
        logger.log_scalar("02_eval/portfolio_value_std", std_balances[t])
        log_weights(logger, tickers, mean_weights[t,:], step=t)
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
