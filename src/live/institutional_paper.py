"""Isolated immutable institutional paper decisions."""

import json
from pathlib import Path


def write_institutional_paper_target(namespace, run_id, target, manifest, root="paper"):
    namespace_root = Path(root) / namespace
    run_dir = namespace_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    target.to_csv(run_dir / "target.csv", index=False)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    namespace_root.mkdir(parents=True, exist_ok=True)
    (namespace_root / "latest.json").write_text(
        json.dumps({"run_id": run_id, "run_directory": str(run_dir.resolve())}, indent=2) + "\n"
    )
    return run_dir
