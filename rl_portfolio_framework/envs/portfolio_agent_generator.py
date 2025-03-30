from .single_agent_portfolio_env import SingleAgentPortfolioEnv
from .multi_agent_shared_action_portfolio_env import MultiAgentSharedActionPortfolioEnv
from .multi_agent_individual_action_portfolio_env import MultiAgentIndividualActionPortfolioEnv



def create_portfolio_env(data, initial_balance=1_000, verbosity=0, n_agents=1, shared_obs=True, shared_action=True, trade_cost_percent=0.0, trade_cost_fixed=0.0):
    if n_agents == 1:
        return SingleAgentPortfolioEnv(data, initial_balance, verbosity, trade_cost_percent, trade_cost_fixed)
    elif shared_action:
        return MultiAgentSharedActionPortfolioEnv(data, initial_balance, verbosity, n_agents)
    else:
        return MultiAgentIndividualActionPortfolioEnv(data, initial_balance, verbosity, n_agents)