from typing import Optional

import torch

from qf import VERBOSITY
from qf.agents.agent import Agent
from qf.agents.config.classic_agents.one_over_n_portfolio_agent_config import (
    OneOverNPortfolioAgentConfig,
)


class OneOverNPortfolioAgent(Agent):
    def __init__(self, env, config: Optional[OneOverNPortfolioAgentConfig] = None):
        self.config = config or OneOverNPortfolioAgentConfig.get_default_config()
        super().__init__(env, config=self.config)

    def act(self, state):
        weights = (
            torch.ones(self.env.action_space.shape[0]) / self.env.action_space.shape[0]
        )
        weights = weights.unsqueeze(0)
        return weights

    def train(
        self,
        episodes=0,
        total_timesteps=0,
        use_tqdm=True,
        save_best=True,
        eval_env=None,
        eval_every_n_steps=None,
        n_eval_episodes=1,
        print_eval_metrics=False,
    ):
        pass

    def _save_impl(self, path):
        pass

    def _load_impl(self, path):
        pass
