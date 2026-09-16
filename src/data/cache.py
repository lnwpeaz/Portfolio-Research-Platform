"""Small, provenance-first cache helpers for downloaded datasets."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class CacheMetadata:
    source: str
    date_range: tuple[str | None, str | None]
    tickers_requested: tuple[str, ...]
    tickers_successful: tuple[str, ...]
    tickers_failed: tuple[str, ...]
    input_hash: str
    downloaded_at: str

    @property
    def complete(self):
        return not self.tickers_failed and set(self.tickers_requested) == set(self.tickers_successful)


def build_cache_metadata(source, start, end, requested, successful):
    requested = tuple(sorted(set(requested)))
    successful = tuple(sorted(set(successful)))
    failed = tuple(sorted(set(requested).difference(successful)))
    payload = {"source": source, "start": start, "end": end, "tickers": requested}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CacheMetadata(
        source=source,
        date_range=(start, end),
        tickers_requested=requested,
        tickers_successful=successful,
        tickers_failed=failed,
        input_hash=digest,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
    )


def write_cache_metadata(path, metadata: CacheMetadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n")
