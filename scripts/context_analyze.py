"""
context_analyze.py — Orchestrator for all post-run analyses of an
S-vs-P context experiment (exp08 / exp10 / exp11).

Under the hood, dispatches to one or more of the analysis helpers in
`scripts/_helpers/`. Each helper is a self-contained script and can
still be invoked directly if you need only that one output.

Tasks
-----
    descriptive          →  <exp-dir>/descriptive_plots/  (fig01–05 + summary CSV)
    mean-diff            →  <exp-dir>/{mean_diff_table.csv, fig06_*, fig07_*}
    summary              →  <exp-dir>/fig08_peace_vs_security_summary.{pdf,png}
    subgroup             →  <exp-dir>/subgroup_analysis/  (7 dims × 3 DVs; 26 plots + 2 CSVs)
    subgroup-means-peace →  <exp-dir>/subgroup_analysis/plots/subgroup_means_peace__*.{pdf,png}
    all (default)        →  all of the above in sequence

Usage
-----
    # every analysis for exp10 (default: --task all)
    python scripts/context_analyze.py --exp exp10_context_peacev3

    # just the mean-diff forest
    python scripts/context_analyze.py --exp exp10_context_peacev3 --task mean-diff

    # descriptive + summary only
    python scripts/context_analyze.py --exp exp11_context_peacev4 --task descriptive summary
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HELPERS = Path(__file__).resolve().parent / "_helpers"

VALID_EXPS = [
    "exp08_context_only",
    "exp10_context_peacev3",
    "exp11_context_peacev4",
]

# task-id → helper filename
TASKS: dict[str, str] = {
    "descriptive":          "context_descriptive_plots.py",
    "mean-diff":            "context_mean_diff.py",
    "summary":              "context_peace_vs_security_summary.py",
    "subgroup":             "context_subgroup.py",
    "subgroup-means-peace": "context_subgroup_means_peace.py",
}
DEFAULT_ORDER = list(TASKS)


def run_task(task: str, exp_id: str) -> None:
    helper = HELPERS / TASKS[task]
    print(f"\n{'='*72}")
    print(f"  ▶ {task:22s}  ({helper.name})")
    print(f"{'='*72}")
    subprocess.run([sys.executable, str(helper), "--exp", exp_id], check=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run any subset of context-experiment analyses.")
    p.add_argument("--exp", required=True, choices=VALID_EXPS,
                   help="Experiment id (folder under outputs/experiments/).")
    p.add_argument("--task", nargs="+", default=["all"],
                   choices=["all"] + list(TASKS),
                   help="Task(s) to run. Default: all.")
    args = p.parse_args()

    if "all" in args.task:
        tasks = DEFAULT_ORDER
    else:
        # preserve DEFAULT_ORDER but restrict to what user asked for
        tasks = [t for t in DEFAULT_ORDER if t in args.task]

    for t in tasks:
        run_task(t, args.exp)

    print(f"\n✓ Completed {len(tasks)} task(s) for {args.exp}.")


if __name__ == "__main__":
    main()
