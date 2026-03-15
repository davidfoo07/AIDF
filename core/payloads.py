from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PortfolioSpec:
    name: str
    raw_weights: Dict[str, float]

    @classmethod
    def make(cls, data: dict) -> "PortfolioSpec":
        return cls(
            name=data["name"],
            raw_weights=data["raw_weights"],
        )


@dataclass
class PortfolioPayload:
    portfolio_id: str
    portfolio_name: str
    description: str
    portfolio_spec: PortfolioSpec
    metadata: dict = field(default_factory=dict)

    @classmethod
    def make(cls, data: dict) -> "PortfolioPayload":
        return cls(
            portfolio_id=data["portfolio_id"],
            portfolio_name=data["portfolio_name"],
            description=data.get("description", ""),
            portfolio_spec=PortfolioSpec.make(data["portfolio_spec"]),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def make_many(cls, rows: List[dict]) -> List["PortfolioPayload"]:
        return [cls.make(row) for row in rows]
