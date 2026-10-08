"""
Multi-seed repetition of the full nested ablation chain (E1 -> E4).

Addresses review item M5: every number in the paper was previously a single-seed
point estimate, including the E4 interaction term gamma, which is a difference of
differences across four cells of ~900 test intents each and therefore carries
roughly +/-1 point of sampling noise per cell.

Each seed runs in a FRESH SUBPROCESS. This matters: xagentic_common reads
XAGENTIC_SEED at import time and the experiment modules bind SEED by value, so an
in-process loop would silently reuse seed 42 for everything after the first run.

Usage
-----
    python experiment/run_seeds.py                 # 10 seeds, full corpus
    python experiment/run_seeds.py --seeds 42 43   # explicit list
    python experiment/run_seeds.py --quick         # fast smoke test

Outputs
-------
    results/seeds/all_summary_seed<S>.json   raw per-seed summary
    results/seeds/e4_factorial_seed<S>.csv   raw per-seed E4 cells
    results/e1_seeds.csv                     per-seed E1 headline metrics
    results/e4_seeds.csv                     per-seed E4 cells + gamma
    results/results_macros_seeds.tex         mean/std/CI macros for paper.tex
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results")
SEEDS_DIR = os.path.join(OUT, "seeds")
DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
N_BOOT = 10000


def _boot_ci(x, n_boot=N_BOOT, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean. With ~10 seeds the normal
    approximation is unreliable, so we resample instead."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), \
        float(np.percentile(means, 100 * (1 - alpha / 2)))


def run_one(seed, quick=False, extra=()):
    env = dict(os.environ, XAGENTIC_SEED=str(seed))
    cmd = [sys.executable, os.path.join(ROOT, "run_all.py"), "--seed", str(seed)]
    if quick:
        cmd.append("--quick")
    cmd += list(extra)
    print("\n" + "=" * 72)
    print(f"SEED {seed}")
    print("=" * 72, flush=True)
    subprocess.run(cmd, check=True, env=env)

    os.makedirs(SEEDS_DIR, exist_ok=True)
    for name, dst in (("all_summary.json", f"all_summary_seed{seed}.json"),
                      ("e4_factorial.csv", f"e4_factorial_seed{seed}.csv"),
                      ("e1_models.csv", f"e1_models_seed{seed}.csv")):
        src = os.path.join(OUT, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(SEEDS_DIR, dst))


def collect(seeds):
    """Pull the headline metrics out of each seed's summary."""
    e1_rows, e4_rows = [], []
    for s in seeds:
        p = os.path.join(SEEDS_DIR, f"all_summary_seed{s}.json")
        if not os.path.exists(p):
            print(f"  !! missing summary for seed {s}; skipped")
            continue
        d = json.load(open(p))
        e1, e4 = d["e1"], d.get("e4")

        e1_rows.append(dict(
            seed=s,
            acc=100 * e1["acc_overall"],
            macro_f1=e1["macro_f1"],
            oov_gap=100 * e1["oov_gap"],
            acc_ambiguous=100 * e1["acc_ambiguous"],
            model=e1["model"],
            tau=e1.get("tau_selected", float("nan")),
            fid1=e1["fidelity"]["ratio1"],
            expl_ms=e1["fidelity"]["explanation_latency_ms"],
        ))

        if e4:
            # gamma as exp4_combined defines it: combined - (d_expl + d_prior),
            # all relative to the E1 cell. Recomputed here from the stored
            # effects so this script stays independent of that module.
            e4_rows.append(dict(
                seed=s,
                acc_base=e4["acc_base"] * 100,
                acc_e2_only=e4["acc_e2_only"] * 100,
                acc_e3_only=e4.get("acc_e3_only", float("nan")) * 100,
                acc_combined=e4["acc_combined"] * 100,
                d_expl=e4["effect_explainer_pts"],
                d_prior=e4["effect_prior_pts"],
                d_combined=e4["effect_combined_pts"],
                gamma=e4["interaction_pts"],
                oov_base=e4["oov_base_pts"],
                oov_combined=e4["oov_combined_pts"],
            ))
    return pd.DataFrame(e1_rows), pd.DataFrame(e4_rows)


def _stat(name, vals, macros, nd=1, seed=0):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return
    lo, hi = _boot_ci(v, seed=seed)
    macros[f"{name}Mean"] = f"{v.mean():+.{nd}f}" if name.startswith("Gamma") \
        else f"{v.mean():.{nd}f}"
    macros[f"{name}Std"] = f"{v.std(ddof=1):.{nd}f}" if len(v) > 1 else "0.0"
    macros[f"{name}CILo"] = f"{lo:+.{nd}f}" if name.startswith("Gamma") else f"{lo:.{nd}f}"
    macros[f"{name}CIHi"] = f"{hi:+.{nd}f}" if name.startswith("Gamma") else f"{hi:.{nd}f}"
    print(f"  {name:<18} {v.mean():+8.{nd}f}  std {v.std(ddof=1):6.{nd}f}  "
          f"95% CI [{lo:+.{nd}f}, {hi:+.{nd}f}]  (n={len(v)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--collect-only", action="store_true",
                    help="skip running; just re-aggregate existing seed outputs")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-embeddings", action="store_true")
    args = ap.parse_args()

    extra = []
    if args.no_llm:
        extra.append("--no-llm")
    if args.no_embeddings:
        extra.append("--no-embeddings")

    if not args.collect_only:
        for s in args.seeds:
            run_one(s, quick=args.quick, extra=extra)

    e1, e4 = collect(args.seeds)
    if len(e1):
        e1.to_csv(os.path.join(OUT, "e1_seeds.csv"), index=False)
        print(f"  -> {OUT}/e1_seeds.csv")
    if len(e4):
        e4.to_csv(os.path.join(OUT, "e4_seeds.csv"), index=False)
        print(f"  -> {OUT}/e4_seeds.csv")

    print("\n" + "=" * 72)
    print(f"MULTI-SEED SUMMARY  (n = {len(e1)} seeds)")
    print("=" * 72)

    macros = {"NSeeds": str(len(e1))}
    if len(e1):
        _stat("EOneAccS", e1["acc"], macros, seed=1)
        _stat("EOneOOVS", e1["oov_gap"], macros, seed=2)
        _stat("EOneFidS", e1["fid1"], macros, seed=3)
        sel = e1["model"].value_counts()
        macros["EOneModelAgree"] = f"{100 * sel.iloc[0] / len(e1):.0f}"
        macros["EOneModelMode"] = str(sel.index[0])
        print(f"  model selected    {sel.index[0]} in {sel.iloc[0]}/{len(e1)} seeds")
        if e1["tau"].notna().any():
            tv = e1["tau"].value_counts()
            macros["EOneTauMode"] = f"{float(tv.index[0]):.1f}"
            macros["EOneTauAgree"] = f"{100 * tv.iloc[0] / e1['tau'].notna().sum():.0f}"
            print(f"  tau selected      {tv.index[0]:.2f} in "
                  f"{tv.iloc[0]}/{e1['tau'].notna().sum()} seeds")

    if len(e4):
        _stat("EFourAccS", e4["acc_combined"], macros, seed=4)
        _stat("EFourOOVS", e4["oov_combined"], macros, seed=5)
        _stat("DeltaExpl", e4["d_expl"], macros, seed=6)
        _stat("DeltaPrior", e4["d_prior"], macros, seed=7)
        _stat("Gamma", e4["gamma"], macros, seed=8)

        g = e4["gamma"].values
        lo, hi = _boot_ci(g, seed=8)
        sig = hi < 0 or lo > 0
        macros["GammaSig"] = "excludes" if sig else "includes"
        print("\n  " + "-" * 68)
        print(f"  INTERACTION gamma = {g.mean():+.1f} +/- {g.std(ddof=1):.1f} pts, "
              f"95% CI [{lo:+.1f}, {hi:+.1f}]")
        print(f"  -> the interval {'EXCLUDES' if sig else 'INCLUDES'} zero: "
              + ("the gains demonstrably do not compose."
                 if sig else
                 "report as 'no evidence that the gains compose'."))
        print("  " + "-" * 68)

    path = os.path.join(OUT, "results_macros_seeds.tex")
    with open(path, "w") as fh:
        fh.write("% Auto-generated by experiment/run_seeds.py -- do not edit.\n")
        fh.write(f"% seeds: {args.seeds}\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"\n  -> {path}")
    print("[run_seeds] copy results_macros_seeds.tex next to paper.tex.")


if __name__ == "__main__":
    main()
