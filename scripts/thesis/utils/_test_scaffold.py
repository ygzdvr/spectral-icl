"""End-to-end smoke test for the thesis scaffolding.

Exercises :mod:`scripts.thesis.utils.run_metadata` and
:mod:`scripts.thesis.utils.plotting` together: constructs a temporary run
directory, enters a ``RunContext`` with a dataclass-like config + seeds,
verifies initial metadata is on disk *before* any "work", emits a figure via
``save_both`` + ``overlay_powerlaw``, exits the context, and verifies the
final metadata + summary are consistent with the contract.

Run from the project root::

    python -u scripts/thesis/utils/_test_scaffold.py

Exit code 0 = all assertions passed. The script prints a one-line confirmation
with the environment fingerprint on success.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


# Make ``scripts.thesis.utils.*`` importable when the script is run directly.
_HERE = Path(__file__).resolve()
_PROJ = _HERE.parents[3]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from scripts.thesis.utils.run_metadata import (
    RunContext,
    ThesisRunDir,
    env_fingerprint,
    make_run_id,
)
from scripts.thesis.utils.plotting import (
    apply_thesis_style,
    overlay_powerlaw,
    save_both,
)


@dataclass(frozen=True)
class _DummyConfig:
    alpha: float = 1.5
    beta: float = 1.75
    lvals: tuple[int, ...] = (1, 2, 4)
    dtype: str = "float64"


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # Simulate: we are running `scripts/thesis/theoremB/run_dummy.py`.
        fake_script = td_path / "scripts" / "thesis" / "theoremB" / "run_dummy.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        fake_script.write_text("")  # existence is enough

        # run_id stability check: same stem should produce deterministically-prefixed ids.
        rid = make_run_id("run_dummy")
        _assert(rid.startswith("run_dummy-"), f"run_id prefix wrong: {rid}")

        # Construct the run directory with an explicit base to keep it inside tmpdir.
        run = ThesisRunDir(
            str(fake_script),
            phase="theoremB",
            base=td_path / "outputs" / "thesis",
        )
        _assert(run.root.parent.name == "run_dummy", "script stem dir misnamed")
        _assert(run.root.parent.parent.name == "theoremB", "phase dir misnamed")
        for sub in (run.figures, run.pdfs, run.npz, run.pt):
            _assert(sub.is_dir(), f"expected subdir {sub!s}")

        cfg = _DummyConfig()
        with RunContext(run, config=cfg, seeds=[0, 1, 2], notes="scaffolding smoke test") as ctx:
            # --- INITIAL metadata must be on disk BEFORE any "work". ---
            _assert(run.config_path.exists(), "config.json missing on enter")
            cfg_loaded = json.loads(run.config_path.read_text())
            _assert(cfg_loaded["alpha"] == cfg.alpha, "config alpha mismatch")
            _assert(cfg_loaded["lvals"] == list(cfg.lvals), "config lvals mismatch")

            _assert(run.metadata_path.exists(), "metadata.json missing on enter")
            meta0 = json.loads(run.metadata_path.read_text())
            _assert(meta0["status"] == "started", f"initial status {meta0['status']!r}")
            _assert(meta0["seeds"] == [0, 1, 2], "seeds not recorded")
            _assert("env" in meta0 and "python_version" in meta0["env"], "env fp missing")
            _assert("git_commit" in meta0, "git_commit key missing")  # value may be None
            _assert(meta0["phase"] == "theoremB", "phase not recorded")

            # --- Produce a figure via the canonical helpers. ---
            apply_thesis_style()
            t = np.logspace(0, 3, 100)
            y = 0.5 * t ** (-0.5)
            fig, ax = plt.subplots()
            ax.loglog(t, y, label="empirical")
            overlay_powerlaw(ax, t, coef=0.5, exponent=-0.5, label=r"$t^{-1/2}$")
            ax.set_xlabel("t")
            ax.set_ylabel(r"$\mathcal{L}$")
            ax.legend()
            png_path, pdf_path = save_both(fig, run, "dummy_powerlaw")
            plt.close(fig)
            _assert(Path(png_path).exists(), "PNG not saved")
            _assert(pdf_path is not None and Path(pdf_path).exists(), "PDF not saved")

            # --- Record runtime observables. ---
            ctx.record_compute_proxy(1.234e10)
            ctx.record_measured_compute(9.87e-3)
            ctx.record_step_time(0.010)
            ctx.record_step_time(0.012)
            ctx.record_step_time(0.011)
            ctx.record_extra("fit_window", [10, 1000])
            ctx.write_summary({"exp_hat": -0.5, "r2": 0.9999})

        # --- FINAL metadata state. ---
        meta1 = json.loads(run.metadata_path.read_text())
        _assert(meta1["status"] == "completed", f"final status {meta1['status']!r}")
        _assert(meta1["wall_clock_seconds"] is not None, "wall-clock missing")
        _assert(meta1["wall_clock_seconds"] >= 0.0, "wall-clock negative")
        _assert(meta1["compute_proxy"] == 1.234e10, "compute proxy lost")
        _assert(meta1["measured_compute"] == 9.87e-3, "measured compute lost")
        pstats = meta1["per_step_wall_clock"]
        _assert(pstats["n_steps"] == 3, f"n_steps = {pstats['n_steps']}")
        _assert(pstats["total"] is not None and 0.032 <= pstats["total"] <= 0.034,
                f"total step-time out of range: {pstats['total']}")
        _assert(meta1["extras"]["fit_window"] == [10, 1000], "extras lost")

        _assert(run.summary_path.exists(), "summary.txt missing")
        summary_text = run.summary_path.read_text()
        _assert("run_id:" in summary_text and "wall_clock_seconds:" in summary_text,
                "summary missing canonical header lines")
        _assert("exp_hat: -0.5" in summary_text, "summary missing user fields")

        # --- Failure path: exception inside the context should record status=failed. ---
        run_fail = ThesisRunDir(
            str(fake_script),
            phase="theoremB",
            base=td_path / "outputs" / "thesis",
            run_id="fail-case-001",
        )
        try:
            with RunContext(run_fail, config={"k": 1}, seeds=[]) as _:
                raise RuntimeError("intentional test failure")
        except RuntimeError:
            pass
        else:
            raise AssertionError("exception was suppressed - that is forbidden")
        meta_fail = json.loads(run_fail.metadata_path.read_text())
        _assert(meta_fail["status"] == "failed", f"failed status wrong: {meta_fail['status']}")
        _assert(meta_fail["error"] and "intentional test failure" in meta_fail["error"],
                "failure error not recorded")
        _assert(run_fail.summary_path.exists(), "summary missing on failure path")

    fp = env_fingerprint()
    print(f"[ok] scaffolding smoke test passed (python={fp['python_version']}, "
          f"torch={fp.get('torch_version', 'n/a')}, "
          f"cuda={fp.get('cuda_available', False)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
