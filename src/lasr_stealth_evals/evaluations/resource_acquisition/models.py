from dataclasses import dataclass
from typing import Dict
from enum import Enum

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
