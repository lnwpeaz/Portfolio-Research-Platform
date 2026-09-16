"""Immutable normalized versions plus a small discoverable manifest index."""

import json
from pathlib import Path
import pandas as pd

from src.data.institutional.metadata import panel_fingerprint
from src.data.institutional.normalization import normalize_panel


class InstitutionalDataCache:
    def __init__(self, root="data/institutional"):
        self.root = Path(root)
        for directory in ("raw", "normalized", "metadata", "manifests"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def manifests(self, provider):
        records = []
        for path in (self.root / "manifests").glob(f"{provider}_*.json"):
            try:
                records.append((path, json.loads(path.read_text())))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(records, key=lambda item: item[1].get("created_at", ""))

    def load_latest(self, provider):
        manifests = self.manifests(provider)
        if not manifests:
            return pd.DataFrame(), None
        manifest = manifests[-1][1]
        path = Path(manifest["normalized_file"])
        if not path.exists():
            return pd.DataFrame(), None
        return pd.read_parquet(path), manifest

    def write_version(self, provider, run_id, raw_parts, panel, manifest):
        raw_dir = self.root / "raw" / provider / run_id
        raw_dir.mkdir(parents=True, exist_ok=False)
        files = []
        for number, part in enumerate(raw_parts, start=1):
            path = raw_dir / f"response_{number:03d}.parquet"
            part.to_parquet(path, index=False)
            files.append(str(path.resolve()))
        normalized = normalize_panel(panel)
        fingerprint = panel_fingerprint(normalized)
        normalized_path = self.root / "normalized" / f"{provider}_{run_id}_{fingerprint[:12]}.parquet"
        normalized.to_parquet(normalized_path, index=False)
        manifest = dict(manifest)
        manifest.update({
            "raw_files": files,
            "normalized_file": str(normalized_path.resolve()),
            "normalized_fingerprint": fingerprint,
        })
        manifest_path = self.root / "manifests" / f"{provider}_{run_id}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
        return manifest, manifest_path


def merge_incremental(existing, additions):
    frames = [frame for frame in (existing, *additions) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(["canonical_ticker", "date"], keep="last")
    return normalize_panel(result)
