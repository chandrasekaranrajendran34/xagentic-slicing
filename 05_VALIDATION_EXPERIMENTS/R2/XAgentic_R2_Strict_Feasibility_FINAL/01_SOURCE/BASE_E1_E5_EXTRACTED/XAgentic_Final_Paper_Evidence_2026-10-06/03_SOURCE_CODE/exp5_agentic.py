"""
Experiment 5 -- Agentic Intent Agent (LLM, tool-using, with a critic loop).

E5 substitutes **one** component of the E1 pipeline: the Intent Agent. The
deterministic regex slot-filler plus brittle substring lexicon is replaced by an
LLM agent that plans, calls typed tools, and is audited by a Validation Agent
that can reject its frame and send it back. Candidate screening, the Selection
Agent (gradient boosting), the Explanation Agent and the feasibility contract
are byte-identical to E1, so E5 extends the nested single-component ablation
chain rather than confounding it.

Pre-registered hypothesis
-------------------------
H5. Replacing the deterministic Intent Agent with an LLM agent that reasons
    semantically about novel phrasings will reduce the open-vocabulary gap
    Delta_OOV by more than the TF-IDF prior of E3, at materially higher
    per-intent cost.

Both outcomes are reportable. If E5 wins, the agentic framing is earned and the
paper gains a cost/benefit curve across E1 -> E3 -> E5. If E5 ties or loses,
that is a second negative composition result, consistent with E4, and the thesis
"we measure which substitutions pay" is reinforced.

Endpoints
---------
    overall accuracy                  comparability with E1-E4
    Delta_OOV                         the primary endpoint
    accuracy on the generic subgroup  the hardest ~30%
    slot recall / false-positive rate isolates agent quality from classifier
    retry rate + post-retry gain      evidence the negotiation loop does work
    tool-call distribution            evidence it is an agent, not a prompt
    latency, tokens, USD per intent   the honest price
    ESCALATE error recall vs conf.    closes gap G3 (reviewer item M7)

Usage
-----
    python exp5_agentic.py                 # full 900-intent test split
    python exp5_agentic.py --limit 60      # cheap smoke run
    XAGENTIC_LLM_OFFLINE=1 python exp5_agentic.py    # replay from cache only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from xagentic_common import (  # noqa: E402
    FEATURES, OUT, PRETTY, SEED, SLICES, SLICE_ID, SLOT_MAP, SLOT_TOL,
    build_features, contract_valid, save_csv, save_json, split,
    subgroup_report, synth_corpus,
)
from exp1_posthoc import build_models  # noqa: E402
import agents  # noqa: E402


# ---------------------------------------------------------------------------
# Frame -> feature vector (mirrors build_features for a single intent)
# ---------------------------------------------------------------------------

def frame_to_features(frame, median):
    """Project one agent slot frame onto the shared 14-dim feature space.

    The transformation is identical to build_features(); only the *source* of
    the five KPI slots and the three lexical-prior features differs. Missing
    slots are filled with the TRAIN-split medians passed in, exactly as
    split() does for E1-E4, so no held-out statistic is used.
    """
    row = {}
    for slot in SLOT_MAP:
        v = frame.get(slot)
        row[slot] = float(v) if v is not None and np.isfinite(float(v)) else np.nan

    hint = frame.get("slice_hint")
    conf = float(frame.get("confidence", 0.0) or 0.0)
    # The agent's semantic hint replaces the binary lexicon match. Weighting by
    # the agent's own confidence keeps the feature on the same [0,1] scale the
    # Selection Agent was trained on, so no retraining is required.
    row["lex_embb"] = conf if hint == "eMBB" else 0.0
    row["lex_urllc"] = conf if hint == "URLLC" else 0.0
    row["lex_mmtc"] = conf if hint == "mMTC" else 0.0

    for slot in SLOT_MAP:
        row["miss_" + slot.split("_")[0]] = int(not np.isfinite(row[slot]))
    row["log_latency"] = np.log10(np.clip(row["latency_ms"], 1e-3, None))
    row["log_throughput"] = np.log10(np.clip(row["throughput_mbps"], 1e-4, None))
    row["log_density"] = np.log10(np.clip(row["density_per_km2"], 1.0, None))

    x = pd.Series({k: row.get(k, np.nan) for k in FEATURES}, dtype=float)
    return x.fillna(median)


def score_agent_slots(frames, dte):
    """Slot recall and false-positive rate for the agent, comparable to E1."""
    hit = {k: 0 for k in SLOT_MAP}
    stated = {k: 0 for k in SLOT_MAP}
    fp = {k: 0 for k in SLOT_MAP}
    unstated = {k: 0 for k in SLOT_MAP}
    for fr, (_, r) in zip(frames, dte.iterrows()):
        for slot, col in SLOT_MAP.items():
            truth = r[col]
            got = fr.get(slot)
            got = float(got) if got is not None else np.nan
            if np.isfinite(truth):
                stated[slot] += 1
                tol = 0.02 * abs(truth) + SLOT_TOL[slot]
                if np.isfinite(got) and abs(got - truth) <= tol:
                    hit[slot] += 1
            else:
                unstated[slot] += 1
                if np.isfinite(got):
                    fp[slot] += 1
    recall = {k: hit[k] / max(stated[k], 1) for k in SLOT_MAP}
    fprate = {k: fp[k] / max(unstated[k], 1) for k in SLOT_MAP}
    return dict(recall=recall, false_positive=fprate,
                mean_slot_accuracy=float(np.mean(list(recall.values()))),
                mean_false_positive=float(np.mean(list(fprate.values()))))


def _escalation_recall(mask, errs):
    """Share of the pipeline's errors captured by an escalation mask."""
    mask = np.asarray(mask, dtype=bool)
    errs = np.asarray(errs, dtype=bool)
    if errs.sum() == 0:
        return float("nan")
    return float((mask & errs).sum() / errs.sum())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(df=None, limit=None, max_retry=1, verbose=True):
    if df is None:
        df = synth_corpus()

    # ---- shared pipeline up to the Selection Agent, trained exactly as E1 ----
    X = build_features(df)
    y = df["slice"].map(SLICE_ID).values
    Xtr, Xte, ytr, yte, dtr, dte = split(X, y, df)
    median = Xtr.median(numeric_only=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_name, best_cv, fitted = None, -np.inf, {}
    for name, m in build_models().items():
        cv = cross_val_score(m, Xtr, ytr, cv=skf, scoring="f1_macro", n_jobs=1).mean()
        m.fit(Xtr, ytr)
        fitted[name] = m
        if cv > best_cv:
            best_name, best_cv = name, cv
    model = fitted[best_name]
    if verbose:
        print(f"[E5] Selection Agent = {best_name} "
              f"(5-fold CV macro F1 = {best_cv:.3f}, train split only)")

    # E1 reference on the identical split -- the thing E5 is compared against.
    e1_pred = model.predict(Xte)
    e1_rep = subgroup_report(e1_pred, yte, dte)

    # ---- the two callbacks the orchestrator needs -------------------------
    # Mean |SHAP| is unavailable per-intent at agent speed, so the Explanation
    # Agent here uses the model's own impurity importances weighted by the
    # instance's feature deviation -- a cheap, deterministic local attribution
    # whose only job is to give the Validation Agent something to audit.
    try:
        gimp = np.asarray(model.feature_importances_, dtype=float)
    except AttributeError:
        gimp = np.ones(len(FEATURES))
    gimp = gimp / max(gimp.sum(), 1e-12)
    mad = (Xtr - median).abs().median().replace(0.0, 1.0).values

    def select_fn(frame):
        x = frame_to_features(frame, median)
        p = model.predict_proba(x.values.reshape(1, -1))[0]
        return SLICES[int(p.argmax())], p.tolist()

    def explain_fn(frame, k=3):
        x = frame_to_features(frame, median)
        contrib = gimp * np.abs((x.values - median.values) / mad)
        order = np.argsort(-contrib)[:k]
        tot = max(contrib.sum(), 1e-12)
        return [{"feature": PRETTY.get(FEATURES[i], FEATURES[i]),
                 "share": round(float(contrib[i] / tot), 3),
                 "value": round(float(x.values[i]), 3)} for i in order]

    # ---- run the agentic pipeline over the held-out intents ---------------
    idx = np.arange(len(dte))
    if limit:
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(idx, size=min(int(limit), len(idx)),
                                 replace=False))
    sub = dte.iloc[idx]
    y_sub = yte[idx]

    frames, preds, probas, escal, retries, verdicts, valid = [], [], [], [], [], [], []
    tools = Counter()
    t0 = time.perf_counter()
    # The cache is flushed in `finally` so that a crash at intent 400 of 900
    # never discards work that has already been paid for.
    try:
        for n, (_, r) in enumerate(sub.iterrows(), 1):
            bb = agents.run_agentic(r["text"], select_fn, explain_fn,
                                    max_retry=max_retry)
            frames.append(bb["slot_frame"])
            preds.append(SLICE_ID[bb["decision"]])
            probas.append(bb["proba"])
            escal.append(bool(bb["escalate"]))
            retries.append(int(bb["retries"]))
            verdicts.append(bb["verdict"]["verdict"])
            valid.append(bool(bb["contract_valid"]))
            for t in bb["trace"]:
                if "tool" in t:
                    tools[t["tool"]] += 1
            if verbose and (n % 25 == 0 or n == len(sub)):
                s = agents.agent_stats()
                print(f"  [E5] {n}/{len(sub)}  calls={s['calls']} "
                      f"cached={s['cache_hits']}  ${s['cost_usd']:.3f}")
    finally:
        agents.flush_cache()
    wall = time.perf_counter() - t0

    preds = np.asarray(preds)
    probas = np.asarray(probas, dtype=float)
    errs = (preds != y_sub)

    rep = subgroup_report(preds, y_sub, sub)
    # E1 restricted to the SAME intents, so the comparison is paired.
    e1_sub = subgroup_report(e1_pred[idx], y_sub, sub)

    # ---- M7 / gap G3: does the agent's critique beat confidence alone? ----
    conf = probas.max(axis=1)
    budget = int(np.sum(escal))
    conf_mask = np.zeros(len(conf), dtype=bool)
    if budget:
        conf_mask[np.argsort(conf)[:budget]] = True

    esc = dict(
        budget=budget,
        escalated_pct=float(budget / max(len(preds), 1)),
        recall_agent=_escalation_recall(escal, errs),
        recall_confidence=_escalation_recall(conf_mask, errs),
        autonomous_acc_agent=float(
            accuracy_score(y_sub[~np.asarray(escal)], preds[~np.asarray(escal)]))
        if (~np.asarray(escal)).any() else float("nan"),
    )
    esc["delta_pts"] = 100.0 * (esc["recall_agent"] - esc["recall_confidence"])

    slots = score_agent_slots(frames, sub)
    retries = np.asarray(retries)
    stats = agents.agent_stats()
    n = len(preds)

    e5 = dict(
        n_intents=int(n),
        selection_model=best_name,
        acc_overall=rep["acc_overall"],
        macro_f1=float(f1_score(y_sub, preds, average="macro")),
        contract_valid=float(np.mean(valid)),
        oov_gap=rep["oov_gap"],
        acc_known=rep["acc_known"], acc_unseen=rep["acc_unseen"],
        acc_generic=rep["acc_generic"], acc_ambiguous=rep["acc_ambiguous"],
        # paired E1 reference on the identical intents
        e1_acc_overall=e1_sub["acc_overall"], e1_oov_gap=e1_sub["oov_gap"],
        e1_acc_generic=e1_sub["acc_generic"], e1_acc_unseen=e1_sub["acc_unseen"],
        e1_acc_known=e1_sub["acc_known"],
        e1_acc_full_split=e1_rep["acc_overall"], e1_oov_full_split=e1_rep["oov_gap"],
        acc_gain_pts=100.0 * (rep["acc_overall"] - e1_sub["acc_overall"]),
        oov_closed_pts=100.0 * (e1_sub["oov_gap"] - rep["oov_gap"]),
        slot_recall=slots["mean_slot_accuracy"],
        slot_false_positive=slots["mean_false_positive"],
        slot_detail=slots,
        retry_rate=float(np.mean(retries > 0)),
        n_retried=int((retries > 0).sum()),
        acc_retried=float(accuracy_score(y_sub[retries > 0], preds[retries > 0]))
        if (retries > 0).any() else float("nan"),
        acc_not_retried=float(accuracy_score(y_sub[retries == 0],
                                             preds[retries == 0]))
        if (retries == 0).any() else float("nan"),
        verdicts={k: int(v) for k, v in Counter(verdicts).items()},
        tool_calls={k: int(v) for k, v in tools.items()},
        tool_calls_per_intent=float(sum(tools.values()) / max(n, 1)),
        escalation=esc,
        llm_calls_per_intent=float((stats["calls"] + stats["cache_hits"]) / max(n, 1)),
        latency_s_per_intent=float(wall / max(n, 1)),
        tokens_in=stats["tok_in"], tokens_out=stats["tok_out"],
        cost_usd_total=stats["cost_usd"],
        cost_usd_per_intent=float(stats["cost_usd"] / max(n, 1)),
        parse_failures=stats["parse_failures"],
        api_errors=stats["api_errors"],
        model=stats["model"], cache_hit_rate=stats["cache_hit_rate"],
        wall_clock_s=wall,
    )

    rows = pd.DataFrame([
        dict(experiment="E1", intent_agent="Lexicon (regex + substring)",
             acc=e1_sub["acc_overall"], acc_known=e1_sub["acc_known"],
             acc_unseen=e1_sub["acc_unseen"], acc_generic=e1_sub["acc_generic"],
             oov_gap=e1_sub["oov_gap"], lat_s=float("nan"), usd=0.0),
        dict(experiment="E5", intent_agent="LLM agent (tools + critic)",
             acc=rep["acc_overall"], acc_known=rep["acc_known"],
             acc_unseen=rep["acc_unseen"], acc_generic=rep["acc_generic"],
             oov_gap=rep["oov_gap"], lat_s=e5["latency_s_per_intent"],
             usd=e5["cost_usd_per_intent"]),
    ])

    save_csv(rows, "e5_agentic.csv")
    save_csv(pd.DataFrame([
        dict(intent=t, verdict=v, retries=int(rt), escalated=bool(e),
             pred=SLICES[p], truth=SLICES[g], uc_kind=k)
        for t, v, rt, e, p, g, k in zip(
            sub["text"].values, verdicts, retries, escal, preds, y_sub,
            sub["uc_kind"].values)]), "e5_decisions.csv")
    save_json(e5, "e5_summary.json")

    if verbose:
        print("\n" + "=" * 72)
        print(f"E5 AGENTIC INTENT AGENT  (n={n}, model={stats['model']})")
        print("=" * 72)
        print(rows.to_string(index=False))
        print(f"\n  accuracy      {e5['acc_overall']*100:.1f}%  "
              f"({e5['acc_gain_pts']:+.1f} pts vs E1 on the same intents)")
        print(f"  Delta_OOV     {e5['oov_gap']*100:.1f} pts  "
              f"({e5['oov_closed_pts']:+.1f} pts closed vs E1)")
        print(f"  retry rate    {e5['retry_rate']*100:.1f}%   "
              f"tool calls/intent {e5['tool_calls_per_intent']:.2f}")
        print(f"  escalation    agent {esc['recall_agent']*100:.1f}% vs "
              f"confidence {esc['recall_confidence']*100:.1f}% error recall "
              f"at a {esc['escalated_pct']*100:.1f}% budget")
        print(f"  cost          ${e5['cost_usd_per_intent']*1000:.3f} / 1000 "
              f"intents-equivalent unit; ${e5['cost_usd_total']:.2f} total")
        print(f"  tools         {dict(tools)}")

    return rows, e5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only N held-out intents (cheap smoke run)")
    ap.add_argument("--max-retry", type=int, default=1,
                    help="bound on Validation Agent RETRY edges (default 1)")
    ap.add_argument("--quick", action="store_true",
                    help="1200-intent corpus instead of 3000")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = synth_corpus(n=1200 if args.quick else 3000)
    run(df=df, limit=args.limit, max_retry=args.max_retry)


if __name__ == "__main__":
    main()
