# Result artifact schema

Every probe/eval JSON committed under `results/` carries a `manifest` block.
Two schema versions exist. Committed artifacts are never modified, so legacy
files keep their original schema; all NEW artifacts must use v1.

## v1 (current: written by `ewjepa.utils.build_run_manifest`)

```json
{
  "checkpoint": "outputs/<run>/model.pt",
  "...metric keys...": "...",
  "manifest": {
    "date": "YYYY-MM-DD",
    "git_sha": "…",
    "seed": 0,
    "torch_version": "…",
    "swm_version": "…",
    "config": { "...full invocation config...", "model": { "mode": "..." } }
  }
}
```

**Authoritative model config:** `manifest.config.model`. The writer scripts
(`scripts/probe.py`, `scripts/evaluate.py`, `scripts/diagnose.py`) replace
the Hydra invocation's `model` block with the config stored inside the
checkpoint before building the manifest, so `manifest.config.model.mode` is
guaranteed to describe the evaluated checkpoint.

## v0 (legacy: only in artifacts committed before 2026-07-13)

Legacy manifests have `invocation_config` instead of `config`, plus a
separate `checkpoint_model_config` key.

**Authoritative model config:** `manifest.checkpoint_model_config`.
`manifest.invocation_config.model` is the raw Hydra default of the command
line invocation and may NOT match the checkpoint. Known instance:
`results/probe/monolithic_seed0.json` has
`invocation_config.model.mode == "factored"` (the Hydra default) while the
probed checkpoint is monolithic (`checkpoint_model_config.mode ==
"monolithic"`, checkpoint `outputs/pusht_monolithic_seed0/model.pt`). Never
read the mode from `invocation_config`.

## Validation

`tests/test_results_schema.py` checks every probe/eval artifact under
`results/` (including `results/archive/`): exactly one schema version
applies, and the authoritative recorded `mode` is consistent with the
checkpoint path (paths containing `monolithic` must record
`mode == "monolithic"`, all others `mode == "factored"`).
