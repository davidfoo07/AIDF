from dataclasses import dataclass, field
from typing import Dict, List

from .instruments import Instrument
from .payloads import PortfolioPayload


@dataclass
class Portfolio:
    portfolio_id: str
    name: str
    description: str
    payload: PortfolioPayload
    instruments: List[Instrument] = field(default_factory=list)
    resolved_portfolio_weights: Dict[str, float] = field(default_factory=dict)
    is_resolved: bool = False

    @classmethod
    def make(cls, payload: PortfolioPayload) -> "Portfolio":
        instruments = [
            Instrument.make({"symbol": symbol})
            for symbol in payload.portfolio_spec.raw_weights.keys()
        ]

        return cls(
            portfolio_id=payload.portfolio_id,
            name=payload.portfolio_name,
            description=payload.description,
            payload=payload,
            instruments=instruments,
        )

    def resolve_portfolio_inplace(self) -> None:
        weights = dict(self.payload.portfolio_spec.raw_weights)

        total_abs = sum(abs(v) for v in weights.values())
        if total_abs == 0:
            raise ValueError(f"Portfolio {self.portfolio_id} has zero total weight.")

        normalized = {k: v / total_abs for k, v in weights.items()}

        self.resolved_portfolio_weights = normalized
        self.is_resolved = True

    def tickers(self) -> List[str]:
        if self.is_resolved:
            return list(self.resolved_portfolio_weights.keys())
        return [inst.symbol for inst in self.instruments]
