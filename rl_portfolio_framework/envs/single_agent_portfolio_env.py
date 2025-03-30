from gymnasium import spaces
import numpy as np
from envs.base_portfolio_env import BasePortfolioEnv


class SingleAgentPortfolioEnv(BasePortfolioEnv):
    def __init__(self, data, initial_balance=100_000, verbosity=1, trade_cost_percent=0.0, trade_cost_fixed=0.0):
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
        self.cash = initial_balance  # Startkapital in Cash
        self.asset_holdings = np.zeros(self.n_assets)  # Start mit 0 Assets
        self.verbosity = verbosity

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
        # Aktion validieren und normalisieren mit Softmax
        max_action = np.max(action)  # Für numerische Stabilität
        exp_action = np.exp(action - max_action)  # Subtrahiere max_action für Stabilität
        weights = exp_action / np.sum(exp_action)  # Softmax-Normalisierung

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

        # Berechnung des aktuellen Portfoliowerts
        current_portfolio_value = self.cash + np.sum(self.asset_holdings * new_prices)

        # Zielverteilung berechnen (basierend auf der Aktion)
        target_cash = current_portfolio_value * cash_weight
        target_asset_values = current_portfolio_value * asset_weights

        # Berechnung der Zielanzahl der Assets
        target_asset_numbers = np.floor(target_asset_values / new_prices)

        # Berechnung der Differenz zwischen aktueller und Zielverteilung
        asset_differences = target_asset_numbers - self.asset_holdings

        # Handelskosten berechnen
        buy_costs = np.sum(np.maximum(asset_differences, 0) * new_prices)  # Kosten für Käufe
        sell_proceeds = np.sum(np.maximum(-asset_differences, 0) * new_prices)  # Einnahmen aus Verkäufen
        trade_costs_percent = np.sum(np.abs(asset_differences) * new_prices * self.trade_cost_percent)
        trade_costs_fixed = np.sum(asset_differences != 0) * self.trade_cost_fixed
        total_trade_costs = trade_costs_percent + trade_costs_fixed

        # Aktualisierung von Cash und Asset-Holdings
        self.cash += sell_proceeds - buy_costs - total_trade_costs
        self.asset_holdings = target_asset_numbers

        # Portfolio-Wert berechnen
        portfolio_value = self.cash + np.sum(self.asset_holdings * new_prices)

        # Belohnung berechnen (absolute Veränderung des Portfoliowerts nach Abzug der Handelskosten)
        reward = portfolio_value - self.balance

        # Balance aktualisieren
        self.balance = portfolio_value

        # Nächste Beobachtung abrufen
        obs = self.data.iloc[self.current_step].values.astype(np.float32)

        # Debugging-Ausgaben
        if self.verbosity > 0:
            print(f"Step: {self.current_step} | Reward: {reward:.4f} | Balance: {self.balance:.2f}")
            print(f"Action: {action} | Weights: {weights} | Prices: {new_prices}")
            print(f"Target asset numbers: {target_asset_numbers} | Current asset holdings: {self.asset_holdings}")
            print(f"Remaining cash: {self.cash:.2f} | Trade costs: {total_trade_costs:.4f}")

        return obs, reward, done, {}