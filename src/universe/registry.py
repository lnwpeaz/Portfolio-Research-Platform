"""Provider registry keeps strategy code independent of index vendors."""

from src.universe.fixed import FixedUniverseProvider
from src.universe.set50 import SET50UniverseProvider
from src.universe.sp500 import SP500UniverseProvider


def create_universe_provider(name: str, **kwargs):
    normalized = name.lower().replace("_", "").replace("-", "")
    if normalized in {"fixed20", "legacy20"}:
        return FixedUniverseProvider(**kwargs)
    if normalized in {"sp500", "s&p500"}:
        return SP500UniverseProvider(**kwargs)
    if normalized == "set50":
        return SET50UniverseProvider(**kwargs)
    raise ValueError(f"Unknown universe provider: {name}")
