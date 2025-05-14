from .multi_agent_portfolio_env import MultiAgentPortfolioEnv

def create_portfolio_env(data, 
                         initial_balance=1_000, 
                         verbosity=0, 
                         n_agents=1, 
                         trade_cost_percent=0.0, 
                         trade_cost_fixed=0.0, 
                         reward_function=None,
                         device="cpu"):

    
    print("📈 Creating Multi Agent Portfolio Environment")
    print("-" * 50)
    return MultiAgentPortfolioEnv(data, initial_balance, verbosity, n_agents, trade_cost_percent, trade_cost_fixed, device=device, reward_function=reward_function)
