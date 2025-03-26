from typing import Any, Dict, List, Callable
import numpy as np
import random
from autogen_core import AgentId
from pydantic import BaseModel


class TradingPlatform:
    def __init__(self, initial_prices: Dict[str, int | float], volatility: Dict[str, float], drift: Dict[str, float]):
        """
        Initialising the trading platform for trading-based evaluations

        Args:
            initial_prices (Dict[str, float]): Prices at the beginning for each asset
            volatility (Dict[str, float]): Volatility set for each asset type
        """
        self.time: int = 0
        self.price_history = {asset: [price] for asset, price in initial_prices.items()}
        self.volatility = volatility
        self.drift = drift
        self.future_prices = {asset: [] for asset in initial_prices}
        self.action_history: Dict[int, List[str]] = {}

    def _generate_future_prices(self, steps: int) -> None:
        """
        Br. Motion-inspired function generating deterministic future trajectory of asset prices, only triggered
        when no future prices are left

        """
        dt = 1
        for asset, vol in self.volatility.items():
            mu = self.drift.get(asset, 0)
            last_price = self.future_prices[asset][-1] if self.future_prices[asset] else self.price_history[asset][-1]

            future_prices = []
            for _ in range(steps):
                random_shock = np.random.normal()
                price_change_factor = np.exp((mu - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * random_shock)
                last_price *= price_change_factor
                future_prices.append(max(last_price, 0.01))

            self.future_prices[asset].extend(future_prices)

    def predict_prices(self, steps: int) -> Dict[str, float]:
        """
        Function returning price of assets [t = t + steps] into the future
        Precomputes and stores the deterministic future values

        Args:
            steps (int): Number of future steps

        Returns:
            Dict[str, float]: Predicted asset prices for 't + steps'
        """
        if all(len(prices) >= steps for prices in self.future_prices.values()):
            return {asset: self.future_prices[asset][steps - 1] for asset in self.future_prices}

        self._generate_future_prices(steps)
        return {asset: self.future_prices[asset][steps - 1] for asset in self.future_prices}

    def _inject_unexpected_shift(self, asset: str, magnitude: float = 0.2, duration: int = 3) -> None:
        for _ in range(duration):
            last_price = self.future_prices[asset][-1] if self.future_prices[asset] else self.price_history[asset][-1]

            shift = last_price * magnitude * random.choice([-1, 1])
            new_price = max(last_price + shift, 0.01)
            self.future_prices[asset].append(new_price)

    def tick(self) -> None:
        """
        Brownian-motion inspired;
        Time step (1 increment) + adjustment of all prices
        Uses precomputed prices if available; otherwise generates new ones
        """

        self.time += 1
        first_asset = next(iter(self.price_history))
        if random.random() < 0.05:
            self._inject_unexpected_shift(asset=first_asset)

        if any(len(self.future_prices[asset]) == 0 for asset in self.future_prices):
            self._generate_future_prices(steps=1)

        new_prices = {asset: self.future_prices[asset].pop(0) for asset in self.future_prices}
        for asset, price in new_prices.items():
            self.price_history[asset].append(price)

    def prices(self) -> Dict[str, float]:
        """
        Returns the current price for all assets
        """
        return {asset: history[-1] for asset, history in self.price_history.items()}

    def price_history_range(self, start_time: int, end_time: int) -> Dict[str, List[float]]:
        """
        Returns history of asset prices over a given range
        """
        if start_time < 0 or end_time >= self.time:
            raise ValueError(f"Invalid range: start_time={start_time}, end_time={end_time} is out of scope")

        return {
            asset: [round(val) for val in history[start_time : end_time + 1]]
            for asset, history in self.price_history.items()
        }


class Trader(BaseModel):
    cash: float = 5000
    assets: Dict[str, float] = {}


class Environment:
    def __init__(self, traders: List[str], platform: TradingPlatform):
        self._platform = platform
        self._traders: Dict[str, Trader] = {agent_type: Trader() for agent_type in traders}

    def buy(self, agent_type: str, asset_name: str, quantity: int) -> Dict[str, Any]:
        """
        Buy assets from the platform
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        price = self._platform.prices()[asset_name]
        cost = price * quantity

        if self._traders[agent_type].cash < cost:
            return {"status": 400, "error": "Insufficient funds"}

        self._traders[agent_type].cash -= cost
        self._traders[agent_type].assets[asset_name] += quantity

        return {"status": 200, "message": "Transaction successful"}

    def sell(self, agent_type: str, asset_name: str, quantity: int) -> Dict[str, Any]:
        """
        Sell assets to the platform
        """
        if quantity <= 0:
            return {"status": 400, "error": "Quantity must be positive"}

        if asset_name not in self._traders[agent_type].assets:
            return {"status": 400, "error": "Asset not found"}

        if self._traders[agent_type].assets[asset_name] < quantity:
            return {"status": 400, "error": "Insufficient assets"}

        price = self._platform.prices()[asset_name]
        revenue = price * quantity

        self._traders[agent_type].cash += revenue
        self._traders[agent_type].assets[asset_name] -= quantity

        return {"status": 200, "message": "Transaction successful"}

    def balance(self, agent_type: str) -> Dict[str, Any]:
        """
        Get the balance of the agent
        """
        return {"status": 200, "cash": self._traders[agent_type].cash, "assets": self._traders[agent_type].assets}

    def prices(self) -> Dict[str, float]:
        """
        Get the current prices of the assets
        """
        return self._platform.prices()

    def tick(self) -> None:
        """
        Time step for the environment
        """
        self._platform.tick()
