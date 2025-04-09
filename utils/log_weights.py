import numpy as np

def log_weights(logger, tickers, actions, step):
    """
    Logs the portfolio weights for each ticker and cash, including total distributions.

    Parameters:
        logger (Logger): The logger instance.
        tickers (list[str]): List of ticker symbols.
        actions (np.ndarray): Portfolio weights with shape (n_agents, n_assets + 1).
        step (int): The current step in the evaluation.
    """
    # Normalize weights to ensure they sum to 1 for each agent
    actions = np.clip(actions, 0, 1)
    actions /= np.sum(actions, axis=1, keepdims=True) + 1e-8  # Normalize along the last dimension

    # Separate cash weights and asset weights
    cash_weights = actions[:, -1]  # Shape: (n_agents,)
    asset_weights = actions[:, :-1]  # Shape: (n_agents, n_assets)

    # Log individual agent weights
    for agent_idx in range(actions.shape[0]):  # Iterate over agents
        # Log cash weight for the agent
        logger.log_scalar(f"03_weights/agent_{agent_idx}/cash", cash_weights[agent_idx], step=step)

        # Log asset weights for the agent
        for asset_idx, ticker in enumerate(tickers):
            logger.log_scalar(f"03_weights/agent_{agent_idx}/{ticker}", asset_weights[agent_idx, asset_idx], step=step)

    # Calculate and log total ticker distribution across all agents
    total_asset_weights = np.sum(asset_weights, axis=0)  # Shape: (n_assets,)
    for asset_idx, ticker in enumerate(tickers):
        logger.log_scalar(f"03_weights/total/{ticker}", total_asset_weights[asset_idx], step=step)

    # Calculate and log total cash distribution across all agents
    total_cash_weight = np.sum(cash_weights)  # Scalar
    logger.log_scalar(f"03_weights/total/cash", total_cash_weight, step=step)


