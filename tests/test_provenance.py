import json

import pytest

from src.evaluation import provenance


def test_provenance_detects_input_and_uncommitted_source_changes(tmp_path):
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    data = tmp_path / "data/raw/prices.parquet"
    code = tmp_path / "src/example.py"
    data.write_bytes(b"input version one")
    code.write_text("x = 1\n")
    first = provenance.capture_provenance("main.py", tmp_path)
    assert first["git_commit"] is None
    assert first["source_hash"] == provenance.capture_provenance("main.py", tmp_path)["source_hash"]
    code.write_text("x = 2\n")
    data.write_bytes(b"input version two")
    second = provenance.capture_provenance("main.py", tmp_path)
    assert first["source_hash"] != second["source_hash"]
    assert first["input_files_sha256"] != second["input_files_sha256"]


@pytest.mark.parametrize("fail", [False, True])
def test_run_manifest_preserves_result_or_failure_and_hashes_outputs(tmp_path, monkeypatch, fail):
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    monkeypatch.setattr(provenance, "capture_provenance", lambda _: {})
    (tmp_path / "reports").mkdir()

    @provenance.record_research_run
    def run():
        (tmp_path / "reports/result.csv").write_text("value\n1\n")
        if fail:
            raise ValueError("failed calculation")
        return 42

    if fail:
        with pytest.raises(ValueError, match="failed calculation"):
            run()
    else:
        assert run() == 42
    manifests = list((tmp_path / "reports/provenance").glob("*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == ("failed" if fail else "completed")
    assert manifest["outputs_sha256"]["reports/result.csv"] == provenance.file_hash(tmp_path / "reports/result.csv")
