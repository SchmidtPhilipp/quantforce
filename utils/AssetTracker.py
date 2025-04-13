import numpy as np

class AssetTracker:
    def __init__(self, n_agents, n_assets, initial_actor_balance, initial_balance, tensorboard_name="04_eval_assets"):
        """
        Initializes the AssetTracker.

        Parameters:
            n_agents (int): Number of agents.
            n_assets (int): Number of assets.
            initial_balance (float): Initial balance for each agent.
        """
        self.tensorboard_name = tensorboard_name
        self.n_agents = n_agents
        self.n_assets = n_assets
        self.actions = []  # List to store actions over time
        self.asset_holdings = []  # List to store asset holdings over time
        self.actor_balances = []  # List to store portfolio balances over time
        self.balances = []  # List to store portfolio balances over time

        # Initialize actions with 100% cash
        initial_weights = np.zeros((n_agents, n_assets + 1))
        initial_weights[:, -1] = 1.0  
        self.actions.append(initial_weights)

        # Initialize asset holdings with zeros
        initial_asset_holdings = np.zeros((n_agents, n_assets))
        self.asset_holdings.append(initial_asset_holdings)

        # Initialize balances with the initial balance
        initial_actor_balance = np.full((n_agents,), initial_actor_balance)
        self.actor_balances.append(initial_actor_balance)

        # Initialize portfolio balance with the initial balance
        self.balances.append(initial_balance)

    def record_step(self, actions, asset_holdings, actor_balance, balance):
        """
        Records the actions, asset holdings, and balances for a single step.

        Parameters:
            actions (np.ndarray): Actions taken by agents (shape: [n_agents, n_assets + 1]).
            asset_holdings (np.ndarray): Asset holdings of agents (shape: [n_agents, n_assets]).
            balance (np.ndarray): Portfolio balances of agents (shape: [n_agents]).
        """
        # Actions always come as a list of arrays, we need to ensure they are 2D
        if isinstance(actions, list):
            actions = np.array(actions)
        if isinstance(asset_holdings, list):
            asset_holdings = np.array(asset_holdings)
        if isinstance(actor_balance, list):
            actor_balance = np.array(actor_balance)

        # Ensure actions is 2D
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)


        self.actions.append(actions)
        self.asset_holdings.append(asset_holdings)
        self.balances.append(balance)
        self.actor_balances.append(actor_balance)

    def log(self, logger, tickers, step=None):
        """
        Logs the actions, asset holdings, and balances for the current step.

        Parameters:
            logger (Logger): Logger instance for logging.
            tickers (list[str]): List of ticker symbols.
            step (int): Current step in the evaluation.
        """
        # Log portfolio balance
        for agent_idx in range(self.n_agents):
            # Log balances
            logger.log_scalar(f"{self.tensorboard_name}/agent_{agent_idx}/balance", self.actor_balances[-1][agent_idx], step=step)

            # Log actions (weights)
            for asset_idx, ticker in enumerate(tickers):
                logger.log_scalar(f"{self.tensorboard_name}/agent_{agent_idx}/{ticker}_weight", self.actions[-1][agent_idx, asset_idx], step=step)
            logger.log_scalar(f"{self.tensorboard_name}/agent_{agent_idx}/cash_weight", self.actions[-1][agent_idx, -1], step=step)

            # Log asset holdings
            for asset_idx, ticker in enumerate(tickers):
                logger.log_scalar(f"{self.tensorboard_name}/agent_{agent_idx}/{ticker}_holding", self.asset_holdings[-1][agent_idx, asset_idx], step=step)

    def print_summary(self):
        """
        Prints a summary of the actions, asset holdings, and balances over the episode.
        """
        print("\n📊 Asset Tracker Summary:")
        for step, (actions, holdings, actor_balance) in enumerate(zip(self.actions, self.asset_holdings, self.actor_balance)):
            print(f"Step {step}:")
            for agent_idx in range(self.n_agents):
                print(f"  Agent {agent_idx}:")
                print(f"    Balance: {actor_balance[agent_idx]:.2f}")
                print(f"    Actions (Weights): {actions[agent_idx]}")
                print(f"    Asset Holdings: {holdings[agent_idx]}")
        print("-" * 50)