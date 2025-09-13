# Classic Agents
from qf.agents.classic_agents.classic_one_period_markowitz_agent import (
    ClassicOnePeriodMarkowitzAgent,
)
from qf.agents.classic_agents.one_over_n_portfolio_agent import OneOverNPortfolioAgent


# Classic Agents config
from qf.agents.config.classic_agents.classic_one_period_markowitz_agent_config import (
    ClassicOnePeriodMarkowitzAgentConfig,
)
from qf.agents.config.classic_agents.one_over_n_portfolio_agent_config import (
    OneOverNPortfolioAgentConfig,
)

from qf.agents.config.classic_agents.random_agent_config import RandomAgentConfig

# Modern Agents config
from qf.agents.config.modern_agents.hjb_portfolio_agent_config import (
    HJBPortfolioAgentConfig,
)
from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.off_policy_agent.ddpg_config import (
    DDPGConfig,
)
from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.off_policy_agent.maddpg_config import (
    MADDPGConfig,
)
from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.off_policy_agent.sac_config import (
    SACConfig,
)
from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.off_policy_agent.td3_config import (
    TD3Config,
)

# SB3 Agents config
from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.on_policy_agent.a2c_config import (
    A2CConfig,
)
from qf.agents.config.rl_agent_config.critic_agent_config.actor_critic_agent_config.on_policy_agent.ppo_config import (
    PPOConfig,
)

from qf.agents.agent import Agent

# Tensor Agents config
from qf.agents.config.rl_agent_config.critic_agent_config.dqn_config import DQNConfig
from qf.agents.config.rl_agent_config.critic_agent_config.spql_config import SPQLConfig
from qf.agents.modern_agents.hjb_portfolio_agent import HJBPortfolioAgent

# Modern Agents
from qf.agents.modern_agents.hjb_portfolio_agent_with_costs import (
    HJBPortfolioAgentWithCosts,
)

# SB3 Agents
from qf.agents.sb3_agents.a2c_agent import A2CAgent
from qf.agents.sb3_agents.ddpg_agent import DDPGAgent
from qf.agents.sb3_agents.ppo_agent import PPOAgent
from qf.agents.sb3_agents.sac_agent import SACAgent
from qf.agents.sb3_agents.td3_agent import TD3Agent

# Tensor Agents
from qf.agents.tensor_agents.dqn_agent import DQNAgent
from qf.agents.tensor_agents.maddpg_agent import MADDPGAgent
from qf.agents.tensor_agents.spql_agent import SPQLAgent


# Other Agents
from qf.agents.classic_agents.random_agent import RandomAgent
