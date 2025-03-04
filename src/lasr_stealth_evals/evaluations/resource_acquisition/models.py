from dataclasses import dataclass
from typing import Dict, List, Callable
from enum import Enum
import random

random.seed(42)
class AssetType(Enum):
    CASH = 0
    STOCK = 1

@dataclass()
class Asset():
    _type: AssetType
    name: str
    quantity: float

    def __repr__(self) -> str:
        _type = "Cash" if not self._type else "Stock"
        return f"Asset(type={_type}, name={self.name}, quantity={self.quantity})"

@dataclass()
class Account():
    account_id: str
    assets: Dict[str, Asset]

    def __repr__(self) -> str:
        return f'Account(account_id={self.account_id}, assets={self.assets})'




class TradingPlatform:
    def __init__(self, initial_prices: Dict[str, float], volatility: Dict[str, float]):
        """
        Initialising the trading platform for trading-based evaluations

        Args:
            initial_prices (Dict[str, float]): Prices at the beginning for each asset
            volatility (Dict[str, float]): Volatility set for each asset type
        """
        self.time = 0
        self.price_history: Dict[str, List[float]] = {asset: [price] for asset, price in initial_prices.items()}
        self.volatility = volatility
        self.future_prices: Dict[str, List[float]] = {asset: [] for asset in
                                                      initial_prices}
    def _generate_future_prices(self, steps: int, random_fn: Callable[[], float]):
        """
        Function generating deterministic future trajectory of asset prices, only triggered
        when no future prices are left

        """
        for asset, vol in self.volatility.items():
            last_price = self.price_history[asset][-1]
            future_prices = []

            for _ in range(steps):
                random_walk = random_fn() * vol
                last_price = max(last_price + random_walk, 0.01)
                future_prices.append(last_price)

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

    def tick(self):
        """
        Time step (1 increment) + adjustment of all prices
        Uses precomputed prices if available; otherwise generates new ones
        """
        self.time += 1

        if any(len(self.future_prices[asset]) == 0 for asset in self.future_prices):
            self._generate_future_prices(1)
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
            raise ValueError(f'Invalid range: start_time={start_time}, end_time={end_time} is out of scope')

        return {asset: history[start_time:end_time + 1] for asset, history in self.price_history.items()}

