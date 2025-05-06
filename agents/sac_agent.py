import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent
from agents.model_builder import ModelBuilder
from agents.buffers.replay_buffer import ReplayBuffer


class SACAgent(BaseAgent):
    def __init__(self, obs_dim, act_dim, actor_config, critic_config, lr=0.0001, gamma=0.99, tau=0.005, alpha=0.2, batch_size=64, buffer_max_size=100000, device="cpu", learning_starts=1000, ent_coef="auto"):
        super().__init__()

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.total_steps = 0  # Zählt die Gesamtanzahl der Schritte
        self.ent_coef = ent_coef

        # Zielentropie basierend auf der Aktionsdimension
        self.target_entropy = -torch.prod(torch.tensor(act_dim, dtype=torch.float32)).item()

        # Replay Buffer
        self.memory = ReplayBuffer(capacity=buffer_max_size)

        # Actor Network
        self.actor = ModelBuilder(actor_config).build().to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic Networks (Q1 and Q2)
        self.critic_1 = ModelBuilder(critic_config).build().to(self.device)
        self.critic_2 = ModelBuilder(critic_config).build().to(self.device)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

        # Target Critic Networks
        self.target_critic_1 = ModelBuilder(critic_config).build().to(self.device)
        self.target_critic_2 = ModelBuilder(critic_config).build().to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Entropy Coefficient
        if isinstance(ent_coef, str) and ent_coef.startswith("auto"):
            # Automatische Anpassung des Entropie-Koeffizienten
            init_value = float(ent_coef.split("_")[1]) if "_" in ent_coef else 1.0
            self.log_alpha = torch.tensor(
                np.log(init_value), requires_grad=True, device=self.device
            )
            self.log_alpha = nn.Parameter(self.log_alpha)  # Als trainierbarer Parameter definieren
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            # Fester Entropie-Koeffizient
            self.log_alpha = None

        # Loss function
        self.critic_loss_fn = nn.MSELoss()

    def act(self, state, deterministic=False, epsilon=0.0):
        """
        Selects an action based on the current state.
        """
        with torch.no_grad():
            action = self.actor(state)
            if not deterministic:
                action += torch.randn_like(action) * self.alpha  # Add noise for exploration

        probs = torch.softmax(action, dim=1)
        return probs / probs.sum(dim=1, keepdim=True)

    def store(self, transition):
        """
        Stores a transition in the replay buffer.
        """
        self.memory.store(transition)
        self.total_steps += 1  # Schrittzähler erhöhen

    def train(self):
        """
        Trains the SAC agent using a batch of transitions from the replay buffer.
        """
        # Warte, bis genügend Schritte gesammelt wurden
        if self.total_steps < self.learning_starts or len(self.memory) < self.batch_size:
            return

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors and ensure the agent dimension is preserved
        states = torch.stack(states).to(self.device)  # (batch_size, n_agents, state_dim)
        actions = torch.stack(actions).to(self.device)  # (batch_size, n_agents, action_dim)
        rewards = torch.stack(rewards).to(self.device)  # (batch_size, n_agents)
        dones = torch.stack(dones).to(self.device)  # (batch_size, n_agents)
        next_states = torch.stack(next_states).to(self.device)  # (batch_size, n_agents, state_dim)

        # Update Critic Networks
        with torch.no_grad():
            next_actions = self.actor(next_states)  # (batch_size, n_agents, action_dim)
            next_q1 = self.target_critic_1(torch.cat([next_states, next_actions], dim=-1))  # (batch_size, n_agents, 1)
            next_q2 = self.target_critic_2(torch.cat([next_states, next_actions], dim=-1))  # (batch_size, n_agents, 1)
            next_q = torch.min(next_q1, next_q2)  # (batch_size, n_agents, 1)
            target_q = rewards.unsqueeze(-1) + (1 - dones) * self.gamma * next_q  # (batch_size, n_agents, 1)

        # Berechnung der aktuellen Q-Werte
        critic_input = torch.cat([states, actions], dim=-1)  # (batch_size, n_agents, state_dim + action_dim)
        current_q1 = self.critic_1(critic_input)  # (batch_size, n_agents, 1)
        current_q2 = self.critic_2(critic_input)  # (batch_size, n_agents, 1)

        # Berechnung der Verluste
        critic_1_loss = self.critic_loss_fn(current_q1, target_q)
        critic_2_loss = self.critic_loss_fn(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update Actor Network
        actions_pred = self.actor(states)  # (batch_size, n_agents, action_dim)
        actor_loss = (self.alpha * actions_pred.log() - self.critic_1(torch.cat([states, actions_pred], dim=-1))).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Entropy Coefficient (if auto)
        if self.log_alpha is not None:
            with torch.no_grad():
                actions_pred = self.actor(states)
            entropy_loss = -(self.log_alpha * (actions_pred.log() + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            entropy_loss.backward()
            self.alpha_optimizer.step()

            # Update alpha
            self.alpha = self.log_alpha.exp().item()

        # Soft update of target networks
        self._soft_update(self.critic_1, self.target_critic_1)
        self._soft_update(self.critic_2, self.target_critic_2)

    def _soft_update(self, source, target):
        """
        Performs a soft update of the target network parameters.
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)