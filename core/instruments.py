from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    asset_type: str = "equity"
    currency: str = "USD"

    @classmethod
    def make(cls, data: dict) -> "Instrument":
        return cls(
            symbol=data["symbol"],
            asset_type=data.get("asset_type", "equity"),
            currency=data.get("currency", "USD"),
        )
