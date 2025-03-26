from gymnasium import spaces
import numpy as np
from envs.base_portfolio_env import BasePortfolioEnv


class SingleAgentPortfolioEnv(BasePortfolioEnv):
    def __init__(self, data, initial_balance=1_000, verbosity=0, trade_cost_percent=0.0, trade_cost_fixed=0.0):
        """
        Single-Agent Portfolio Environment mit Handelskosten.

        Parameters:
            data (pd.DataFrame): Historische Daten für die Assets.
            initial_balance (float): Startkapital.
            verbosity (int): Verbositätslevel für Debugging-Ausgaben.
            trade_cost_percent (float): Prozentuale Kosten pro Trade (z. B. 0.001 für 0.1%).
            trade_cost_fixed (float): Feste Kosten pro Trade (z. B. 1.0 für 1 Einheit der Währung).
        """
        super().__init__(data, initial_balance, verbosity, n_agents=1)
        self.trade_cost_percent = trade_cost_percent
        self.trade_cost_fixed = trade_cost_fixed
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.data.shape[1],),
            dtype=np.float32
        )
        self.previous_weights = np.zeros(self.n_assets + 1)  # Start mit 100% Cash

    def _get_observation(self):
        """
        Liefert die aktuelle Beobachtung basierend auf den Daten.
        """
        return self.data.iloc[self.current_step].values.astype(np.float32)

    def step(self, action):
        """
        Führt einen Schritt im Environment aus.

        Parameters:
            action (np.ndarray): Die vom Agenten gewählten Gewichte.

        Returns:
            obs (np.ndarray): Die nächste Beobachtung.
            reward (float): Die Belohnung für den aktuellen Schritt.
            done (bool): Ob die Episode beendet ist.
            info (dict): Zusätzliche Informationen.
        """
        # Aktion validieren und normalisieren
        weights = np.clip(action, 0, 1)
        weights /= np.sum(weights) + 1e-8

        cash_weight = weights[-1]
        asset_weights = weights[:-1]

        # Preise des aktuellen Schritts abrufen
        old_prices = self.data.xs("Close", axis=1, level=1).iloc[self.current_step].values
        self.current_step += 1
        done = self.current_step >= len(self.data)

        if done:
            reward = 0.0
            obs = np.zeros(self.data.shape[1], dtype=np.float32)
            if self.verbosity > 0:
                print("Episode finished!")
            return obs, reward, done, {}

        # Preise des nächsten Schritts abrufen
        new_prices = self.data.xs("Close", axis=1, level=1).iloc[self.current_step].values

        # Robustheit: Überprüfen, ob Preise gültig sind
        if np.any(old_prices <= 0) or np.any(new_prices <= 0):
            raise ValueError("Ungültige Preise gefunden (<= 0). Überprüfen Sie die Eingabedaten.")

        # Prozentuale Rendite berechnen
        asset_returns = new_prices / old_prices - 1

        # Portfolio-Rendite berechnen
        portfolio_return = cash_weight * 1.0 + np.dot(asset_weights, asset_returns)

        # Handelskosten berechnen
        trade_amounts = np.abs(weights - self.previous_weights)  # Änderungen in den Gewichten
        trade_costs = np.sum(trade_amounts[:-1] * self.trade_cost_percent) + np.sum(trade_amounts > 0) * self.trade_cost_fixed

        # Portfolio-Rendite nach Abzug der Handelskosten
        portfolio_return -= trade_costs

        # Robustheit: Überprüfen, ob die Portfolio-Rendite gültig ist
        if portfolio_return < -1:
            portfolio_return = -1  # Verlust auf maximal 100% begrenzen

        # Balance aktualisieren
        self.balance *= (1 + portfolio_return)

        # Belohnung berechnen
        reward = portfolio_return

        # Aktualisiere vorherige Gewichte
        self.previous_weights = weights

        # Nächste Beobachtung abrufen
        obs = self.data.iloc[self.current_step].values.astype(np.float32)

        # Debugging-Ausgaben
        if self.verbosity > 0:
            print(f"Step: {self.current_step} | Reward: {reward:.4f} | Balance: {self.balance:.2f}")
            print(f"Action: {action} | Weights: {weights} | Prices: {new_prices}")
            print(f"Portfolio return: {portfolio_return:.4f} | Asset returns: {asset_returns}")
            print(f"Trade costs: {trade_costs:.4f} | Trade amounts: {trade_amounts}")
            print(f"Obs: {obs}")
            print(f"Observation shape: {obs.shape}")

        return obs, reward, done, {}