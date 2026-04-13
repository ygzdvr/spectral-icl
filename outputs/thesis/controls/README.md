# `outputs/thesis/controls/` — immutable archival control baseline

This directory holds the **frozen** Bordelon control-suite outputs referenced
by `EXPERIMENT_PLAN_FINAL.MD` §5.

## Freeze mode: **archive-by-copy of existing reproduced outputs**

Every per-control `metadata.json` (and the top-level `FROZEN.json`) records
`"freeze_mode": "archive_by_copy_of_existing_outputs"`. This is important:

- **The artifacts here were NOT produced by a fresh run invoked by the freezer.**
- The freezer (`scripts/thesis/freeze_bordelon_controls.py`) takes an existing
  successful reproduction of each Bordelon control under `outputs/<script_stem>/`
  at the recorded `git_commit`, copies the artifact tree here, computes SHA-256
  hashes for every copied file, and marks every copied file read-only (`0444`).
- The `reproduction_command` field in each `metadata.json` documents the
  canonical command that would regenerate these artifacts from scratch on an
  equivalent environment. Running that command is outside the freezer's scope.

## Contents

```
outputs/thesis/controls/
├── FROZEN.json                                       top-level index (read-only)
├── README.md                                         this file
├── run_isotropic_depth_vs_alpha/
│   ├── artifacts/ {figures, pdfs, npz, pt}/*         read-only artifacts
│   └── metadata.json                                 per-control manifest (read-only)
├── run_fixed_covariance/…
├── run_reduced_gamma_dynamics/…
├── run_compute_scaling_joint/…
├── run_linear_attention_dynamics/…
└── run_softmax_depth_sweep/…
```

## Per-control `metadata.json` fields

- `script_stem`, `plan_reference`, `source_path`
- `frozen_utc`, `git_commit`, `git_dirty_at_freeze`
- `env`: Python / torch / CUDA / platform / hostname fingerprint at freeze time
- `freeze_mode`: `"archive_by_copy_of_existing_outputs"`
- `archival_note`: explicit prose stating the archive-by-copy semantics
- `config_source`: pointer to frozen-dataclass defaults in `configs/` at
  `git_commit`; the experiment would be reproduced by re-running the listed
  `reproduction_command` from the same commit
- `reproduction_command`: canonical CLI invocation
- `artifacts[]`: per-file `{path, sha256, size_bytes}`
- `n_files`, `total_bytes`

## Immutability guarantees

- Every artifact file and every `metadata.json` is mode `0444`.
- `FROZEN.json` (the index) is also mode `0444`.
- Modification of any frozen file is detectable via SHA-256 hash mismatch
  against the per-control manifest.
- The freezer refuses to re-run if any destination subdirectory already
  exists; re-freezing requires explicit manual `chmod +w` and `rm -rf` of the
  specific subdirectory first (intentional friction).

## Usage

These artifacts are the **calibration package** for all new theorem-A / B / C
exact scripts and the conditional scaling-law layer. Every new thesis script
whose figure is a spectral analogue of a Bordelon figure (e.g., theorem-B
spectral-loss curves in the language of Bordelon §3.2) should visually and
numerically be comparable to the corresponding frozen baseline here.

**Do not overwrite, edit, or delete** anything under this directory during
theorem-A / B / C development. If a frozen file becomes actually obsolete
because of a spec change, the thesis archival convention is to keep the old
freeze in place and add a parallel `v2/` subdirectory rather than replace.
