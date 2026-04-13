# `scripts/thesis/` — thesis experimental namespace

This directory holds every new experimental script for the PhD thesis on
spectral scaling laws for in-context learning. The code is deliberately kept
separate from the top-level `scripts/` directory so that the original Bordelon
reproductions remain untouched and reproducible on demand.

The authoritative source for what lives here is
[`EXPERIMENT_PLAN_FINAL.MD`](../../EXPERIMENT_PLAN_FINAL.MD). This README is a
navigational aid only.

## Layout

```
scripts/thesis/
├── utils/           shared utilities (see list below)
├── theoremA/        Theorem A (bridge) exact experiments   — Section 8
├── theoremB/        Theorem B (circulant / stationary)     — Section 6
├── theoremC/        Theorem C (band-RRS / hybrid)          — Section 7
├── architectures/   architecture-aligned validation        — Section 9
└── scaling_laws/    conditional empirical scaling-law tier — Section 10
```

### `utils/` contents

| File | Purpose | Spec |
|---|---|---|
| `run_metadata.py` | `ThesisRunDir`, `RunContext`, env + git fingerprint | v4 §§2, 4.1 |
| `plotting.py` | canonical style, `save_both`, overlay / heatmap / frontier helpers | v4 §4.2 |
| `_test_scaffold.py` | end-to-end smoke test for the metadata + plotting scaffold | — |
| `fourier_ops.py` | unitary DFT (isolated complex), real-even symbol constructors, circulant diag, off-diagonal Fourier energy | v4 §4 |
| `partitions.py` | `BlockPartition`, `equal_blocks`, `dyadic_ladder`, `custom_ladder`, mass-preserving `mass_preserving_block_spectrum` / `mass_preserving_block_task` | v4 §5 |
| `commutants.py` | block-commutant projection, block-scalar extraction / reconstruction, violation metric, `refines` | v4 §6 *(not yet implemented)* |
| `metrics.py` | reduced-model err, (A,B)-perturbation bound, γ★ trajectory, oracle commutant loss, contraction overlay, OOD slope, frontier regret | v4 §7 *(not yet implemented)* |
| `cost_models.py` | canonical compute proxy and wall-clock calibration | v4 §8 *(not yet implemented)* |
| `fit_powerlaws.py` | log-log LSQ with fixed fit windows, bootstrap CI, held-out eval | v4 §9 *(not yet implemented)* |
| `data_generators.py` | GA, G1, G2 operator-only / sampled, G3 | v4 §10 *(not yet implemented)* |
| `_self_tests/` | hard-gate self-tests per v4 §12 | *(not yet implemented)* |

Companion directories:

- `configs/thesis/<phase>/` — YAML / JSON configs per experiment family.
- `experiments/thesis/<phase>/` — Slurm / shell launchers, one per script.
- `outputs/thesis/<phase>/` — populated at run time with per-run directories.

The `scaling_laws/` directory replaces the earlier `theoremD/` placeholder —
the previously-empty `theoremD/` directories under `configs/thesis/` and
`experiments/thesis/` have been removed so that all scaling-law artifacts land
in a single canonical location.

## Import convention

Thesis scripts reach both the thesis namespace and the Bordelon top-level code
via the project root on `sys.path` (as set by `starter.sh`). The canonical
prologue for a new thesis script is:

```python
import sys
from pathlib import Path
_PROJ = Path(__file__).resolve().parents[3]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from scripts.thesis.utils.run_metadata import RunContext, ThesisRunDir
from scripts.thesis.utils.plotting import apply_thesis_style, save_both
```

This avoids any collision with the top-level Bordelon `utils/` package.

## Output-directory contract

Every run writes to `outputs/thesis/<phase>/<script_stem>/<run_id>/`:

```
<run_dir>/
├── figures/         PNG figures (300 dpi)
├── pdfs/            vector PDF figures (Type-42 fonts, LaTeX-embeddable)
├── npz/             numpy archives (raw arrays)
├── pt/              torch tensors / checkpoints
├── config.json      exact config used, written on RunContext __enter__
├── metadata.json    run_id, seeds, git, env, timings, compute proxy, status
├── summary.txt      human-readable summary of fitted quantities
└── run.log          optional stdout/stderr log (launcher writes this)
```

`run_id = <script_stem>-<UTC-timestamp>-<8-hex>`. Both `config.json` and the
initial `metadata.json` (status `"started"`) are written **before** any
computation begins, so that a crashed run still leaves a full record of its
inputs on disk (Section 4.1 of the plan).

## Canonical minimal script skeleton

```python
from scripts.thesis.utils.run_metadata import RunContext, ThesisRunDir
from scripts.thesis.utils.plotting import apply_thesis_style, save_both

def main() -> None:
    apply_thesis_style()
    run = ThesisRunDir(__file__, phase="theoremB")
    with RunContext(run, config=cfg, seeds=cfg.seeds) as ctx:
        # ... actual experiment ...
        ctx.record_compute_proxy(cfg.t * cfg.L_S * (cfg.P * math.log(cfg.P) + cfg.P * cfg.r))
        ctx.record_step_time(elapsed_per_step)
        # ... save figures, npz, pt via run.png/pdf/npz_path/pt_path ...
        ctx.write_summary({"loss_final": ..., "beta_hat": ...})
```

## Plotting contract

- Call `apply_thesis_style()` exactly once at the start of the script (or wrap
  the plotting block in `with thesis_style(): ...`).
- Sequential sweeps (depth, rank, $P$, $t$) use `PALETTE_SEQUENTIAL = "rocket"`.
- Phase-diagram heatmaps use `PALETTE_PHASE = "mako"`.
- Diverging quantities (OOD, signed residuals) use `PALETTE_DIVERGING = "vlag"`.
- Categorical groups use `PALETTE_CATEGORICAL = "colorblind"`.
- Theory overlays use dashed black lines at `zorder=10` via `overlay_powerlaw`
  or `overlay_reference`.
- Save via `save_both(fig, run_dir, name)`; it emits both PNG (figures/) and
  PDF (pdfs/) with `pdf.fonttype=42` for LaTeX embedding.

## Smoke test

The scaffold can be verified end-to-end without any real experiment:

```bash
python -u scripts/thesis/utils/_test_scaffold.py
```

A passing run prints one confirmation line; any failure raises an
`AssertionError` and exits non-zero.

## Implementation status (per Section 13 of the plan)

- **Step 1a** ✅ — scaffolding + `run_metadata.py` + `plotting.py`.
- **Step 1b** in progress, per the v4 Step-1 Generator / Utility Specification:
  1. ✅ `fourier_ops.py` (§4)
  2. ✅ `partitions.py` (§5)
  3. ⏳ `commutants.py` (§6)
  4. ⏳ `metrics.py` (§7)
  5. ⏳ `cost_models.py` (§8)
  6. ⏳ `fit_powerlaws.py` (§9)
  7. ⏳ `data_generators.py` (§10)
  8. ⏳ `_self_tests/run_all.py` (§12)

Each file is implemented and audited one at a time, under the v4 spec as the
sole source of truth. The `_self_tests/` harness will gate every subsequent
step of the thesis plan (Phase I controls onward).
