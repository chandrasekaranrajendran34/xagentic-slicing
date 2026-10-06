"""
Run every experiment and emit the cross-experiment comparison.

    E1  Post-hoc explanation (SHAP)        -- exp1_posthoc.py
    E2  In-hoc vs post-hoc explanation     -- exp2_inhoc.py
    E3  Semantic Intent Agent              -- exp3_semantic.py
    E4  Combined E2 + E3 (2x2 factorial)   -- exp4_combined.py
    E5  Agentic Intent Agent (opt-in)      -- exp5_agentic.py

All of them share one corpus instance, one seed and one train/test split, so the
cross-experiment comparison is controlled rather than merely co-reported. Each
experiment substitutes exactly one component of the E1 pipeline.

Outputs (results/)
    comparison.csv          cross-experiment summary table
    results_macros.tex      LaTeX \newcommand definitions for paper.tex
    fig_e2_latency.pdf      explanation latency vs faithfulness
    fig_e3_oov.pdf          open-vocabulary gap by Intent Agent variant
    fig_e4_combined.pdf     factorial accuracy/cost panels
    fig_e5_agentic.pdf      agent vs lexicon, and escalation quality
    all_summary.json        every metric from every experiment

Usage
    python run_all.py                       # E1-E4
    python run_all.py --no-llm              # skip the Together API variant
    python run_all.py --quick               # 1200-intent corpus, faster
    python run_all.py --with-agent          # add E5 (needs TOGETHER_API_KEY)
    python run_all.py --with-agent --agent-limit 60   # cheap E5 smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- seed bootstrap -------------------------------------------------------
# xagentic_common reads XAGENTIC_SEED at import time, and the experiment
# modules bind SEED by value via `from xagentic_common import SEED`. The seed
# must therefore be in the environment BEFORE the imports below, which is why
# this is handled here rather than inside main().
if "--seed" in sys.argv:
    os.environ["XAGENTIC_SEED"] = sys.argv[sys.argv.index("--seed") + 1]

from xagentic_common import OUT, SEED, synth_corpus, save_csv, save_json  # noqa: E402
import exp1_posthoc, exp2_inhoc, exp3_semantic, exp4_combined  # noqa: E402

# ---------------------------------------------------------------------------
# LaTeX macro emission
# ---------------------------------------------------------------------------

def _fmt(v, nd=1, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if pct:
        v = v * 100.0
    return f"{v:.{nd}f}"


def emit_macros(e1, e2, e3, path, e4=None, e5=None):
    """Write \newcommand definitions so paper.tex never hard-codes a number."""
    m = {}

    # ---- Experiment 1 ----
    m["EOneAcc"] = _fmt(e1["acc_overall"], 1, pct=True)
    if "baseline_rulelex_acc" in e1:
        m["EOneRuleAcc"] = _fmt(e1["baseline_rule_acc"], 1, pct=True)
        m["EOneRuleFone"] = _fmt(e1["baseline_rule_f1"], 3)
        m["EOneRuleLexAcc"] = _fmt(e1["baseline_rulelex_acc"], 1, pct=True)
        m["EOneRuleLexFone"] = _fmt(e1["baseline_rulelex_f1"], 3)
        m["EOneRuleLexValid"] = _fmt(e1["baseline_rulelex_valid"], 1, pct=True)
        m["EOneRuleLexLat"] = _fmt(e1["baseline_rulelex_lat_ms"] * 1000.0, 1)
        m["EOneRuleLexGain"] = _fmt(
            (e1["acc_overall"] - e1["baseline_rulelex_acc"]) * 100.0, 1)
    m["EOneFone"] = _fmt(e1["macro_f1"], 3)
    m["EOneSlotRecall"] = _fmt(e1["slot_recall"], 1, pct=True)
    m["EOneSlotFP"] = _fmt(e1["slot_false_positive"], 1, pct=True)
    m["EOneExplMs"] = _fmt(e1["fidelity"]["explanation_latency_ms"], 1)
    m["EOneInferMs"] = _fmt(e1["fidelity"]["inference_latency_ms"] * 1000.0, 1)
    m["EOneRatio"] = _fmt(e1["fidelity"]["latency_ratio"], 0)
    m["EOneFid"] = _fmt(e1["fidelity"]["ratio1"], 1)
    m["EOneTopOne"] = _fmt(e1["fidelity"]["top1"], 1, pct=True)
    m["EOneRandOne"] = _fmt(e1["fidelity"]["rand1"], 1, pct=True)
    if "low1" in e1["fidelity"]:
        m["EOneLowOne"] = _fmt(e1["fidelity"]["low1"], 1, pct=True)
        m["EOneFidLow"] = _fmt(e1["fidelity"]["ratiolow1"], 1)
    m["EOneOOV"] = _fmt(e1["oov_gap"], 1, pct=True)
    m["EOneAccKnown"] = _fmt(e1["acc_known"], 1, pct=True)
    m["EOneAccAmb"] = _fmt(e1["acc_ambiguous"], 1, pct=True)
    m["EOneAccClear"] = _fmt(e1["acc_clear"], 1, pct=True)
    m["EOneErrTotal"] = str(int(e1["err_total"]))
    m["EOneErrAmbPct"] = _fmt(e1["err_from_ambiguous"] / max(e1["err_total"], 1),
                              1, pct=True)
    _tau = float(e1.get("tau_selected", 0.80))
    esc80 = next(r for r in e1["escalation"] if abs(r["tau"] - _tau) < 1e-9)
    m["EOneTau"] = f"{_tau:.1f}"
    m["EOneEscPct"] = _fmt(esc80["escalated"], 1, pct=True)
    m["EOneEscRecall"] = _fmt(esc80["recall"], 1, pct=True)
    m["EOneAutoAcc"] = _fmt(esc80["autonomous_acc"], 1, pct=True)
    if e1.get("escalation_oof"):
        _o = next(r for r in e1["escalation_oof"] if abs(r["tau"] - _tau) < 1e-9)
        m["EOneEscPctOOF"] = _fmt(_o["escalated"], 1, pct=True)
        m["EOneEscRecallOOF"] = _fmt(_o["recall"], 1, pct=True)
        m["EOneAutoAccOOF"] = _fmt(_o["autonomous_acc"], 1, pct=True)
    if e1.get("escalation_signals"):
        _s = {r["signal"]: r for r in e1["escalation_signals"]}
        _cc = _s["Confidence + contract"]
        _all = _s["Confidence + contract + attribution"]
        m["EOneSigBudget"] = _fmt(_cc["escalated"], 1, pct=True)
        m["EOneSigConfRecall"] = _fmt(_s["Confidence only"]["recall"], 1, pct=True)
        m["EOneSigAttrRecall"] = _fmt(
            _s["Attribution concentration only"]["recall"], 1, pct=True)
        m["EOneSigBothRecall"] = _fmt(_cc["recall"], 1, pct=True)
        m["EOneSigAllRecall"] = _fmt(_all["recall"], 1, pct=True)
        _d = 100.0 * (_all["recall"] - _cc["recall"])
        m["EOneSigDelta"] = f"{_d:+.1f}"
        m["EOneAttrVerdict"] = "adds" if _d > 0.5 else "does not add"

    # ---- Experiment 2 ----
    m["ETwoBest"] = str(e2["best_inhoc"])
    m["ETwoBestAcc"] = _fmt(e2["best_inhoc_acc"], 1, pct=True)
    m["ETwoBestExplUs"] = _fmt(e2["best_inhoc_expl_ms"] * 1000.0, 1)
    m["ETwoBestFid"] = _fmt(e2["best_inhoc_fidelity"], 1)
    m["ETwoShapAcc"] = _fmt(e2["shap_acc"], 1, pct=True)
    m["ETwoShapMs"] = _fmt(e2["shap_expl_ms"], 1)
    m["ETwoShapFid"] = _fmt(e2["shap_fidelity"], 1)
    m["ETwoSpeedup"] = _fmt(e2["speedup"], 0)
    m["ETwoAccCost"] = _fmt(abs(e2["accuracy_cost"]), 1, pct=True)

    # ---- Experiment 3 ----
    m["EThreeBest"] = str(e3["best_variant"]).split(". ", 1)[-1]
    m["EThreeAcc"] = _fmt(e3["best_acc"], 1, pct=True)
    m["EThreeOOV"] = _fmt(e3["best_oov_gap"], 1, pct=True)
    m["EThreeBaseOOV"] = _fmt(e3["baseline_oov_gap"], 1, pct=True)
    m["EThreeGapClosed"] = _fmt(e3["gap_closed_pts"], 1)
    m["EThreeAccGain"] = _fmt(e3["acc_gain_pts"], 1)
    m["EThreePhrases"] = str(e3["n_phrases"])

    # ---- Full table bodies, so paper.tex never transcribes a row ----
    def _esc(s):
        return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")

    e2_rows = []
    for r in sorted(e2["table"], key=lambda r: (r["kind"] != "in-hoc",
                                                r["expl_ms"])):
        bold = (r["explainer"] == e2["best_inhoc"])
        b = (lambda s: f"\\textbf{{{s}}}") if bold else (lambda s: s)
        e2_rows.append(
            f"{b(_esc(r['explainer']))} & {r['kind']} & "
            f"{b(_fmt(r['acc'], 1, pct=True))} & "
            f"{b(_fmt(r['expl_ms'] * 1000.0, 1))} & "
            f"{b(_fmt(r['fid1'], 1))} \\\\")
    m["ETwoRows"] = "\n".join(e2_rows)

    e3_rows = []
    for r in e3["table"]:
        bold = (str(r["variant"]) == str(e3["best_variant"]))
        b = (lambda s: f"\\textbf{{{s}}}") if bold else (lambda s: s)
        e3_rows.append(
            f"{b(_esc(r['variant']))} & {b(_fmt(r['acc'], 1, pct=True))} & "
            f"{b(_fmt(r['acc_known'], 1, pct=True))} & "
            f"{b(_fmt(r['acc_unseen'], 1, pct=True))} & "
            f"{b(_fmt(r['acc_generic'], 1, pct=True))} & "
            f"{b(_fmt(r['oov_gap'], 1, pct=True))} \\\\")
    m["EThreeRows"] = "\n".join(e3_rows)

    # ---- Experiment 4: combined configuration (factorial) ----
    if e4:
        m["EFourAcc"] = _fmt(e4["acc_combined"], 1, pct=True)
        m["EFourOOV"] = _fmt(e4["oov_combined_pts"], 1)
        m["EFourExplUs"] = _fmt(e4["expl_ms_combined"] * 1000.0, 1)
        m["EFourSpeedup"] = _fmt(e4["speedup_combined"], 0)
        m["EFourFid"] = _fmt(e4["fid1_combined"], 1)
        m["EFourEffExpl"] = _fmt(e4["effect_explainer_pts"], 1)
        m["EFourEffPrior"] = _fmt(e4["effect_prior_pts"], 1)
        m["EFourEffBoth"] = _fmt(e4["effect_combined_pts"], 1)
        m["EFourInteract"] = _fmt(e4["interaction_pts"], 1)
        m["EFourAutoAcc"] = _fmt(e4["autonomous80_combined"], 1, pct=True)
        m["EFourEscPct"] = _fmt(e4["escalated80_combined"], 1, pct=True)
        e4_rows = []
        for r in e4["rows"]:
            bold = r["cell"].startswith("E4")
            w = (lambda s: f"\\textbf{{{s}}}") if bold else (lambda s: s)
            name, _, cfg = str(r["cell"]).partition("  ")
            e4_rows.append(
                f"{w(name)} & {w(_esc(cfg))} & {w(_fmt(r['acc'],1,pct=True))} & "
                f"{w(_fmt(r['expl_ms']*1000.0,1))} & {w(_fmt(r['fid1'],1))} & "
                f"{w(_fmt(r['oov_gap'],1,pct=True))} \\\\")
        m["EFourRows"] = "\n".join(e4_rows)

    # ---- Experiment 5: agentic Intent Agent -----------------------------
    if e5:
        m["EFiveN"] = str(int(e5["n_intents"]))
        m["EFiveModel"] = str(e5["model"]).split("/")[-1].replace("_", "\\_")
        m["EFiveAcc"] = _fmt(e5["acc_overall"], 1, pct=True)
        m["EFiveFone"] = _fmt(e5["macro_f1"], 3)
        m["EFiveOOV"] = _fmt(e5["oov_gap"], 1, pct=True)
        m["EFiveAccKnown"] = _fmt(e5["acc_known"], 1, pct=True)
        m["EFiveAccUnseen"] = _fmt(e5["acc_unseen"], 1, pct=True)
        m["EFiveAccGeneric"] = _fmt(e5["acc_generic"], 1, pct=True)
        m["EFiveBaseAcc"] = _fmt(e5["e1_acc_overall"], 1, pct=True)
        m["EFiveBaseOOV"] = _fmt(e5["e1_oov_gap"], 1, pct=True)
        m["EFiveAccGain"] = f"{e5['acc_gain_pts']:+.1f}"
        m["EFiveGapClosed"] = f"{e5['oov_closed_pts']:+.1f}"
        m["EFiveSlotRecall"] = _fmt(e5["slot_recall"], 1, pct=True)
        m["EFiveSlotFP"] = _fmt(e5["slot_false_positive"], 1, pct=True)
        m["EFiveRetryPct"] = _fmt(e5["retry_rate"], 1, pct=True)
        m["EFiveTools"] = _fmt(e5["tool_calls_per_intent"], 2)
        m["EFiveCalls"] = _fmt(e5["llm_calls_per_intent"], 2)
        m["EFiveLatency"] = _fmt(e5["latency_s_per_intent"], 2)
        m["EFiveCostK"] = _fmt(e5["cost_usd_per_intent"] * 1000.0, 2)
        m["EFiveEscPct"] = _fmt(e5["escalation"]["escalated_pct"], 1, pct=True)
        m["EFiveEscRecall"] = _fmt(e5["escalation"]["recall_agent"], 1, pct=True)
        m["EFiveEscConfRecall"] = _fmt(
            e5["escalation"]["recall_confidence"], 1, pct=True)
        m["EFiveEscDelta"] = f"{e5['escalation']['delta_pts']:+.1f}"
        m["EFiveCriticVerdict"] = (
            "adds" if e5["escalation"]["delta_pts"] > 0.5 else "does not add")
        m["EFiveVerdict"] = ("closes" if e5["oov_closed_pts"] > 0
                             else "does not close")
        # Cost of the agent relative to the deterministic parser it replaces.
        m["EFiveCostRatio"] = _fmt(
            e5["latency_s_per_intent"] * 1000.0
            / max(e1["fidelity"]["inference_latency_ms"], 1e-9), 0)

    with open(path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by experiment/run_all.py -- do not edit by hand.\n")
        f.write("% Regenerate with:  python experiment/run_all.py\n")
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"  -> {path}  ({len(m)} macros)")
    return m


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(e2, e3, e4_table=None):
    plt.rcParams.update({"font.size": 7, "figure.dpi": 300,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    # --- E1: global attribution + escalation sweep -----------------------
    # paper.tex includes both of these (Fig. 3a/3b). They are rebuilt here from
    # the CSVs exp1_posthoc.py just wrote, so a single `run_all.py` invocation
    # regenerates every figure the manuscript needs.
    try:
        gi = pd.read_csv(os.path.join(OUT, "e1_shap_global.csv"))
        gi = gi.sort_values("importance", ascending=True).tail(8)
        fig, ax = plt.subplots(figsize=(3.3, 2.1))
        ax.barh(gi["feature"], gi["importance"], color="#2b6cb0")
        ax.set_xlabel("mean |SHAP| (normalised)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_shap_global.pdf"))
        plt.close(fig)
        print(f"  -> {OUT}/fig_shap_global.pdf")
    except Exception as e:
        print(f"  !! fig_shap_global.pdf skipped ({type(e).__name__}: {e})")

    try:
        esc = pd.read_csv(os.path.join(OUT, "e1_escalation.csv"))
        fig, ax = plt.subplots(figsize=(3.3, 2.1))
        ax.plot(esc["escalated"] * 100, esc["autonomous_acc"] * 100,
                marker="o", color="#2b6cb0")
        for _, r in esc.iterrows():
            ax.annotate(rf"$\tau$={r['tau']:.1f}",
                        (r["escalated"] * 100, r["autonomous_acc"] * 100),
                        textcoords="offset points", xytext=(4, -7), fontsize=6)
        ax.set_xlabel("intents escalated to operator (%)")
        ax.set_ylabel("accuracy on autonomous\ndecisions (%)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_escalation.pdf"))
        plt.close(fig)
        print(f"  -> {OUT}/fig_escalation.pdf")
    except Exception as e:
        print(f"  !! fig_escalation.pdf skipped ({type(e).__name__}: {e})")

    # --- E2: explanation latency vs faithfulness -------------------------
    t2 = pd.DataFrame(e2["table"])
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for kind, mark, col in (("in-hoc", "o", "#2b6cb0"), ("post-hoc", "s", "#c05621")):
        sub = t2[t2["kind"] == kind]
        if sub.empty:
            continue
        ax.scatter(sub["expl_ms"].clip(lower=1e-5), sub["fid1"],
                   marker=mark, s=34, c=col, label=kind, zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(r["explainer"], (max(r["expl_ms"], 1e-5), r["fid1"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("explanation latency per intent (ms, log scale)")
    ax.set_ylabel(r"faithfulness (top-1 / random)")
    ax.grid(alpha=0.3, zorder=0)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_e2_latency.pdf"))
    plt.close(fig)
    print(f"  -> {OUT}/fig_e2_latency.pdf")

    # --- E3: open-vocabulary gap by variant ------------------------------
    t3 = pd.DataFrame(e3["table"])
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    idx = np.arange(len(t3))
    w = 0.26
    ax.bar(idx - w, t3["acc_known"] * 100, w, label="in-lexicon", color="#2b6cb0")
    ax.bar(idx, t3["acc_unseen"] * 100, w, label="unseen", color="#68a4d8")
    ax.bar(idx + w, t3["acc_generic"] * 100, w, label="generic", color="#b9d4ea")
    ax.set_xticks(idx)
    ax.set_xticklabels([v.split(". ", 1)[-1] for v in t3["variant"]],
                       rotation=12, ha="right")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(70, 102)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=3, fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_e3_oov.pdf"))
    plt.close(fig)
    print(f"  -> {OUT}/fig_e3_oov.pdf")

    # --- E4: combined configuration, 2x2 factorial -----------------------
    # Two panels: the accuracy/OOV axis (E3 effect) and the explanation-cost
    # axis (E2 effect), with all four cells on each so composition is visible.
    if e4_table is not None and len(e4_table):
        try:
            t4 = pd.DataFrame(e4_table)
            short = [c.split("  ", 1)[0] for c in t4["cell"]]
            colors = ["#b9d4ea", "#68a4d8", "#4a7fb5", "#1a4971"]
            fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))

            ax = axes[0]
            idx = np.arange(len(t4))
            w = 0.38
            ax.bar(idx - w / 2, t4["acc"] * 100, w, color=colors,
                   label="accuracy")
            ax.bar(idx + w / 2, t4["oov_gap"] * 100, w, color=colors,
                   alpha=0.45, hatch="///", label="$\\Delta_{OOV}$")
            ax.set_xticks(idx)
            ax.set_xticklabels(short)
            ax.set_ylabel("accuracy (%) / gap (pts)")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(frameon=False, fontsize=6, ncol=2, loc="center right")
            ax.set_title("(a) accuracy and open-vocabulary gap", fontsize=7)

            ax = axes[1]
            ax.bar(idx, t4["expl_ms"] * 1000.0, 0.55, color=colors)
            ax.set_yscale("log")
            ax.set_xticks(idx)
            ax.set_xticklabels(short)
            ax.set_ylabel("explanation cost ($\\mu$s/intent)")
            ax.grid(axis="y", alpha=0.3, which="both")
            ax.set_title("(b) explanation cost (log scale)", fontsize=7)

            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "fig_e4_combined.pdf"))
            plt.close(fig)
            print(f"  -> {OUT}/fig_e4_combined.pdf")
        except Exception as e:
            print(f"  !! fig_e4_combined.pdf skipped ({type(e).__name__}: {e})")


def make_e5_figure(e5):
    """E5 vs E1 on the open-vocabulary axis, and the price of the agent."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
        ax = axes[0]
        idx = np.arange(3)
        w = 0.38
        base = [e5["e1_acc_known"], e5["e1_acc_unseen"], e5["e1_acc_generic"]]
        ag = [e5["acc_known"], e5["acc_unseen"], e5["acc_generic"]]
        ax.bar(idx - w / 2, np.array(base) * 100, w, label="E1 lexicon",
               color="#b9d4ea")
        ax.bar(idx + w / 2, np.array(ag) * 100, w, label="E5 agent",
               color="#1a4971")
        ax.set_xticks(idx)
        ax.set_xticklabels(["in-lexicon", "unseen", "generic"])
        ax.set_ylabel("accuracy (%)")
        ax.set_ylim(60, 102)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False, fontsize=6, ncol=2, loc="lower left")
        ax.set_title("(a) accuracy by use-case vocabulary", fontsize=7)

        ax = axes[1]
        esc = e5["escalation"]
        ax.bar([0, 1], [esc["recall_confidence"] * 100, esc["recall_agent"] * 100],
               0.5, color=["#b9d4ea", "#1a4971"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["confidence only", "agent critique"])
        ax.set_ylabel("error recall at equal budget (%)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"(b) escalation quality "
                     f"({esc['escalated_pct']*100:.0f}% budget)", fontsize=7)

        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_e5_agentic.pdf"))
        plt.close(fig)
        print(f"  -> {OUT}/fig_e5_agentic.pdf")
    except Exception as e:
        print(f"  !! fig_e5_agentic.pdf skipped ({type(e).__name__}: {e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the Together API Intent Agent variant")
    ap.add_argument("--no-embeddings", action="store_true",
                    help="skip the sentence-transformer Intent Agent variant")
    ap.add_argument("--quick", action="store_true",
                    help="smaller corpus for a fast smoke run")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="corpus/split/model seed (handled at import time)")
    ap.add_argument("--with-agent", action="store_true",
                    help="run Experiment 5, the LLM agentic Intent Agent "
                         "(needs TOGETHER_API_KEY, or a populated llm_cache.json)")
    ap.add_argument("--agent-limit", type=int, default=None,
                    help="evaluate E5 on only N held-out intents (smoke run)")
    args = ap.parse_args()

    n = 1200 if args.quick else 3000
    print(f"[run_all] generating shared corpus (n={n}, seed={SEED}) ...")
    df = synth_corpus(n=n)
    save_csv(df, "corpus.csv")

    e1 = exp1_posthoc.run(df=df)
    e2 = exp2_inhoc.run(df=df)
    e3 = exp3_semantic.run(df=df,
                           use_llm=(not args.no_llm) and bool(os.environ.get("TOGETHER_API_KEY")),
                           use_embeddings=not args.no_embeddings)
    e4_table, e4 = exp4_combined.run(df=df)

    e5 = e5_rows = None
    if args.with_agent:
        import exp5_agentic
        print("\n[run_all] running E5 (agentic Intent Agent) ...")
        try:
            e5_rows, e5 = exp5_agentic.run(df=df, limit=args.agent_limit)
        except Exception as e:
            import traceback
            print(f"  !! E5 skipped ({type(e).__name__}: {e})")
            traceback.print_exc()
            e5 = e5_rows = None

    # ---- cross-experiment comparison -----------------------------------
    t2 = pd.DataFrame(e2["table"])
    best2 = t2[t2["kind"] == "in-hoc"].sort_values("f1", ascending=False).iloc[0]
    t3 = pd.DataFrame(e3["table"])
    best3 = t3.iloc[t3["oov_gap"].astype(float).values.argmin()]

    comp = pd.DataFrame([
        dict(experiment="E1", focus="Post-hoc explanation",
             intent_agent="Lexicon", explainer="SHAP (post-hoc)",
             acc=e1["acc_overall"], f1=e1["macro_f1"],
             expl_ms=e1["fidelity"]["explanation_latency_ms"],
             fidelity=e1["fidelity"]["ratio1"], oov_gap=e1["oov_gap"]),
        dict(experiment="E2", focus="In-hoc explanation",
             intent_agent="Lexicon", explainer=f"{best2['explainer']} (in-hoc)",
             acc=float(best2["acc"]), f1=float(best2["f1"]),
             expl_ms=float(best2["expl_ms"]), fidelity=float(best2["fid1"]),
             oov_gap=e2["details"][best2["explainer"]]["subgroup"]["oov_gap"]),
        dict(experiment="E3", focus="Semantic intent parsing",
             intent_agent=str(best3["variant"]).split(". ", 1)[-1],
             explainer="SHAP (post-hoc)",
             acc=float(best3["acc"]), f1=float(best3["f1"]),
             expl_ms=e1["fidelity"]["explanation_latency_ms"],
             fidelity=e1["fidelity"]["ratio1"], oov_gap=float(best3["oov_gap"])),
        dict(experiment="E4", focus="Combined (E2 + E3)",
             intent_agent="TF-IDF n-gram", explainer="Decision-Path (in-hoc)",
             acc=e4["acc_combined"], f1=float(
                 e4_table[(e4_table.prior == "tfidf")
                          & (e4_table.explainer == "dpath")]["f1"].iloc[0]),
             expl_ms=e4["expl_ms_combined"], fidelity=e4["fid1_combined"],
             oov_gap=e4["oov_combined_pts"] / 100.0),
    ])
    if e5:
        comp = pd.concat([comp, pd.DataFrame([
            dict(experiment="E5", focus="Agentic intent parsing",
                 intent_agent="LLM agent (tools + critic)",
                 explainer="SHAP (post-hoc)",
                 acc=e5["acc_overall"], f1=e5["macro_f1"],
                 expl_ms=e1["fidelity"]["explanation_latency_ms"],
                 fidelity=e1["fidelity"]["ratio1"], oov_gap=e5["oov_gap"])])],
            ignore_index=True)
    save_csv(comp, "comparison.csv")
    print("\n" + "=" * 72)
    print("CROSS-EXPERIMENT COMPARISON")
    print("=" * 72)
    print(comp.to_string(index=False))

    macros_path = os.path.join(OUT, "results_macros.tex")
    print("\n[run_all] writing LaTeX macros ...")
    emit_macros(e1, e2, e3, macros_path, e4=e4, e5=e5)

    print("[run_all] rendering figures ...")
    make_figures(e2, e3, e4_table)
    if e5 is not None:
        make_e5_figure(e5)

    save_json(dict(e1=e1, e2=e2, e3=e3, e4=e4, e5=e5,
                   comparison=comp.to_dict(orient="records")), "all_summary.json")

    print("\n[run_all] done. Copy these back into the paper directory:")
    for f in ("results_macros.tex", "fig_shap_global.pdf", "fig_escalation.pdf",
              "fig_e2_latency.pdf", "fig_e3_oov.pdf", "fig_e4_combined.pdf",
              "fig_e5_agentic.pdf", "comparison.csv", "e4_factorial.csv",
              "e5_agentic.csv", "e5_summary.json", "llm_cache.json",
              "all_summary.json"):
        p = os.path.join(OUT, f)
        print(f"  {'OK ' if os.path.exists(p) else '-- '}{p}")
    print("\n[run_all] tip: zip the whole results folder and download it:")
    print(f"  shutil.make_archive('xagentic_results', 'zip', '{OUT}')")


if __name__ == "__main__":
    main()
