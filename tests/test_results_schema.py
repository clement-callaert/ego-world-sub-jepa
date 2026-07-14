"""Validate committed result artifacts against results/SCHEMA.md.

Two manifest schemas exist (see results/SCHEMA.md). For each artifact the
authoritative model config is:
  v1: manifest.config.model          (config replaced by the checkpoint's)
  v0: manifest.checkpoint_model_config
The recorded mode must match the checkpoint path: paths containing
"monolithic" are monolithic checkpoints, everything else is factored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "results"

ARTIFACTS = sorted(
    p
    for sub in ("probe", "eval", "archive")
    for p in (RESULTS / sub).glob("*.json")
    if (RESULTS / sub).exists()
)


def _authoritative_model_cfg(manifest: dict) -> dict:
    has_v1 = "config" in manifest
    has_v0 = "invocation_config" in manifest or "checkpoint_model_config" in manifest
    assert has_v1 != has_v0, "manifest must be exactly one of schema v0 or v1"
    if has_v1:
        model = manifest["config"].get("model")
        assert isinstance(model, dict), "v1 manifest.config.model missing"
        return model
    model = manifest.get("checkpoint_model_config")
    assert isinstance(model, dict), "v0 manifest.checkpoint_model_config missing"
    return model


@pytest.mark.parametrize("path", ARTIFACTS, ids=lambda p: str(p.relative_to(RESULTS)))
def test_recorded_mode_matches_checkpoint(path: Path) -> None:
    data = json.loads(path.read_text())
    assert "checkpoint" in data, "artifact must record its checkpoint path"
    assert "manifest" in data, "artifact must carry a manifest"

    model = _authoritative_model_cfg(data["manifest"])
    mode = model.get("mode")
    assert mode in ("factored", "monolithic")

    expected = "monolithic" if "monolithic" in data["checkpoint"] else "factored"
    assert mode == expected, (
        f"{path.name}: recorded mode {mode!r} does not match checkpoint "
        f"{data['checkpoint']!r} (expected {expected!r})"
    )


def test_artifacts_found() -> None:
    assert len(ARTIFACTS) >= 8
