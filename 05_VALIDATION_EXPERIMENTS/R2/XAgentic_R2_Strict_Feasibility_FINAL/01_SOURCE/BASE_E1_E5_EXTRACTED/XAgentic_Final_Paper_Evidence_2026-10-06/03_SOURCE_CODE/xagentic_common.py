"""
Shared components for the Explainable Agentic AI slice-selection experiments.

This module holds everything the three experiments have in common so that the
comparison across them is strictly controlled: identical corpus, identical
seed, identical train/test split, identical slice catalogue and safety contract.

  E1  Post-hoc explanation          (exp1_posthoc.py)
  E2  In-hoc vs post-hoc            (exp2_inhoc.py)
  E3  Semantic Intent Agent         (exp3_semantic.py)

Stages
  S1  Synthetic intent + KPI corpus grounded in 3GPP TS 22.261 / TS 23.501
  S2  Intent Agent      : NL intent -> structured intent vector (slot filling)
  S3  Candidate Agent   : SST/feasibility screening against a slice catalogue
  S4  Selection Agent   : supervised slice classifier
  S5  Validation Agent  : safety-contract check (3GPP KPI envelopes)
  S6  Explanation Agent : attribution (post-hoc or in-hoc, per experiment)
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

SEED = int(os.environ.get("XAGENTIC_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

SLICES = ["eMBB", "URLLC", "mMTC"]
SLICE_ID = {s: i for i, s in enumerate(SLICES)}

N_CORPUS = 3000
TEST_SIZE = 0.30

# ---------------------------------------------------------------------------
# S1. Slice catalogue + synthetic intent/KPI corpus
# ---------------------------------------------------------------------------

# 3GPP-informed KPI envelopes per standardised slice/service type (SST).
CATALOGUE = {
    "eMBB":  dict(lat=(10.0, 150.0), thr=(25.0, 1000.0), rel=(99.0, 99.9),
                  den=(1e2, 5e4), mob=(0.0, 250.0), sst=1),
    "URLLC": dict(lat=(0.5, 10.0), thr=(0.1, 100.0), rel=(99.99, 99.9999),
                  den=(1e2, 1e4), mob=(0.0, 500.0), sst=2),
    "mMTC":  dict(lat=(50.0, 10000.0), thr=(0.001, 2.0), rel=(95.0, 99.9),
                  den=(1e4, 1e6), mob=(0.0, 30.0), sst=3),
}

# Use cases the Intent Agent's design-time lexicon has seen.
USE_CASES_KNOWN = {
    "eMBB": ["4K video streaming", "AR/VR gaming", "live broadcast",
             "fixed wireless access", "stadium hotspot"],
    "URLLC": ["factory robot control", "remote surgery", "smart-grid protection",
              "drone fleet control", "haptic teleoperation"],
    "mMTC": ["smart water meters", "agricultural sensors", "asset trackers",
             "environmental monitoring", "pipeline telemetry"],
}

# Lexically *unseen* use cases: realistic open-vocabulary drift. They carry no
# exact keyword signal, so a lexicon-based agent must fall back to KPI slots.
USE_CASES_UNSEEN = {
    "eMBB": ["high-definition conferencing", "digital signage uplink",
             "newsroom contribution feed", "in-flight entertainment uplink"],
    "URLLC": ["port crane synchronisation", "substation differential tripping",
              "rail interlocking signalling", "surgical imaging feedback"],
    "mMTC": ["utility submetering fleet", "soil-moisture reporting nodes",
             "cold-chain condition reporting", "street luminaire telemetry"],
}

# Vertical-agnostic phrasings that give the agent no lexical prior at all.
GENERIC_USE_CASES = [
    "campus connectivity", "enterprise application", "smart city service",
    "public safety communications", "retail branch connectivity",
    "municipal network service", "event coverage service", "research testbed service",
]

# Templates differ in which KPI slots the operator actually states.
TEMPLATES = [
    "Deploy a slice for {uc} with latency under {lat} ms, at least {thr} Mbps "
    "throughput and {rel}% reliability for about {den} devices per square km.",
    "I need {uc} support: end-to-end delay must stay below {lat} ms, sustained rate "
    "{thr} Mbps, availability {rel}%, device density around {den} per km2, "
    "mobility up to {mob} km/h.",
    "Provision connectivity for {uc}. Target budget: {lat} ms latency, {rel}% "
    "reliability, {den} devices/km2, users moving at {mob} km/h.",
    "Can you set up {uc}? We require no more than {lat} ms one-way latency, a "
    "minimum of {thr} Mbps and {rel}% service availability.",
    "Create a network slice supporting {uc} with {thr} Mbps guaranteed bitrate, "
    "{rel} percent reliability, {den} terminals per km2 and {mob} km/h mobility.",
    "Set up {uc}. It should sustain {thr} Mbps for roughly {den} devices per km2.",
    "Please onboard {uc} with a {lat} ms delay bound and {rel}% availability.",
]


def _loguniform(lo, hi, rng):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _fmt(x):
    if x >= 1000:
        return f"{x:,.0f}"
    if x >= 10:
        return f"{x:.0f}"
    if x >= 1:
        return f"{x:.1f}"
    return f"{x:.3f}"


def synth_corpus(n=N_CORPUS, p_amb=0.30, seed=SEED):
    """Generate NL intents with latent ground-truth slice labels.

    Two difficulty dimensions are controlled independently:
      * KPI ambiguity   -- a fraction `p_amb` of requests have KPI targets
        blended towards a neighbouring slice envelope;
      * vocabulary drift -- only ~40% of intents use a use-case phrase the
        design-time lexicon covers; the rest are unseen or generic.

    The ground-truth label is the *pre-blending* latent slice type.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = SLICES[i % 3]
        c = CATALOGUE[label]
        lat = _loguniform(*c["lat"], rng)
        thr = _loguniform(*c["thr"], rng)
        rel = float(rng.uniform(*c["rel"]))
        den = _loguniform(*c["den"], rng)
        mob = float(rng.uniform(*c["mob"]))

        ambiguous = rng.random() < p_amb
        if ambiguous:
            other = SLICES[int(rng.integers(0, 3))]
            o = CATALOGUE[other]
            a = rng.uniform(0.40, 0.75)

            def blend(v, other_v):
                return float(np.exp((1 - a) * np.log(max(v, 1e-6))
                                    + a * np.log(max(other_v, 1e-6))))

            lat = blend(lat, _loguniform(*o["lat"], rng))
            thr = blend(thr, _loguniform(*o["thr"], rng))
            den = blend(den, _loguniform(*o["den"], rng))
            rel = (1 - a) * rel + a * float(rng.uniform(*o["rel"]))

        u = rng.random()
        if u < 0.40:
            pool, uc_kind = USE_CASES_KNOWN[label], "known"
        elif u < 0.70:
            pool, uc_kind = USE_CASES_UNSEEN[label], "unseen"
        else:
            pool, uc_kind = GENERIC_USE_CASES, "generic"
        uc = pool[int(rng.integers(0, len(pool)))]

        ti = int(rng.integers(0, len(TEMPLATES)))
        tpl = TEMPLATES[ti]
        text = tpl.format(uc=uc, lat=_fmt(lat), thr=_fmt(thr),
                          rel=f"{rel:.4f}".rstrip("0").rstrip("."),
                          den=_fmt(den), mob=_fmt(mob))

        rows.append(dict(
            id=i, text=text, use_case=uc, uc_kind=uc_kind, template=ti,
            slice=label, ambiguous=int(ambiguous),
            lat_true=lat if "{lat}" in tpl else np.nan,
            thr_true=thr if "{thr}" in tpl else np.nan,
            rel_true=rel if "{rel}" in tpl else np.nan,
            den_true=den if "{den}" in tpl else np.nan,
            mob_true=mob if "{mob}" in tpl else np.nan,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# S2. Intent Agent -- deterministic slot-filling NLU
# ---------------------------------------------------------------------------

NUM = r"([0-9][0-9,]*\.?[0-9]*)"

PATTERNS = {
    "latency_ms": [
        rf"(?:latency|delay)[^.;]*?(?:under|below|of|budget:?|bound)?\s*{NUM}\s*ms",
        rf"{NUM}\s*ms\s*(?:one-way\s*)?(?:latency|delay|budget)",
        rf"no more than\s*{NUM}\s*ms",
    ],
    "throughput_mbps": [
        rf"(?:at least|minimum of|sustained rate|guaranteed bitrate)?\s*{NUM}\s*Mbps",
    ],
    "reliability_pct": [
        rf"{NUM}\s*(?:%|percent)",
        rf"(?:reliability|availability)[^.;]*?{NUM}",
    ],
    "density_per_km2": [
        rf"{NUM}\s*(?:devices|terminals)\s*(?:per|/)\s*(?:square km|km2|km\^2)",
        rf"(?:capacity for|about|around)\s*{NUM}\s*(?:devices|terminals)",
    ],
    "mobility_kmh": [
        rf"{NUM}\s*km/h",
    ],
}

# Design-time domain lexicon: exact substring match only. This is deliberately
# brittle -- quantifying that brittleness is the point of Experiment 3.
LEXICON = {
    "URLLC": ["robot control", "surgery", "grid protection", "drone fleet", "haptic",
              "teleoperation"],
    "mMTC": ["water meter", "agricultural sensor", "asset tracker",
             "environmental monitoring", "pipeline telemetry"],
    "eMBB": ["4k video", "ar/vr", "live broadcast", "fixed wireless access",
             "stadium hotspot"],
}


@dataclass
class Intent:
    latency_ms: float = np.nan
    throughput_mbps: float = np.nan
    reliability_pct: float = np.nan
    density_per_km2: float = np.nan
    mobility_kmh: float = np.nan
    lex_embb: float = 0.0
    lex_urllc: float = 0.0
    lex_mmtc: float = 0.0


def parse_slots(text: str) -> Intent:
    """Slot-filling only (no lexical prior). Shared by every Intent Agent variant."""
    it = Intent()
    for slot, pats in PATTERNS.items():
        for p in pats:
            m = re.search(p, text, flags=re.I)
            if m:
                try:
                    setattr(it, slot, float(m.group(1).replace(",", "")))
                except ValueError:
                    continue
                break
    return it


def lexicon_scores(text: str):
    """E1/E2 lexical prior: binary exact substring match against the lexicon."""
    low = text.lower()
    return (
        float(any(k.lower() in low for k in LEXICON["eMBB"])),
        float(any(k.lower() in low for k in LEXICON["URLLC"])),
        float(any(k.lower() in low for k in LEXICON["mMTC"])),
    )


def parse_intent(text: str) -> Intent:
    """Full E1 Intent Agent: slot filling + binary lexicon prior."""
    it = parse_slots(text)
    it.lex_embb, it.lex_urllc, it.lex_mmtc = lexicon_scores(text)
    return it


FEATURES = ["latency_ms", "throughput_mbps", "reliability_pct",
            "density_per_km2", "mobility_kmh",
            "log_latency", "log_throughput", "log_density",
            "miss_latency", "miss_throughput", "miss_density",
            "lex_embb", "lex_urllc", "lex_mmtc"]

PRETTY = {
    "latency_ms": "latency (ms)", "throughput_mbps": "throughput (Mbps)",
    "reliability_pct": "reliability (%)", "density_per_km2": "density (dev/km2)",
    "mobility_kmh": "mobility (km/h)", "log_latency": "log latency",
    "log_throughput": "log throughput", "log_density": "log density",
    "miss_latency": "latency unstated", "miss_throughput": "throughput unstated",
    "miss_density": "density unstated",
    "lex_embb": "lex:eMBB", "lex_urllc": "lex:URLLC", "lex_mmtc": "lex:mMTC",
}

SLOT_MAP = {"latency_ms": "lat_true", "throughput_mbps": "thr_true",
            "reliability_pct": "rel_true", "density_per_km2": "den_true",
            "mobility_kmh": "mob_true"}
SLOT_TOL = {"latency_ms": 0.5, "throughput_mbps": 0.05, "reliability_pct": 0.0,
            "density_per_km2": 0.5, "mobility_kmh": 0.5}


def score_slots(df: pd.DataFrame):
    """Score slot-filling quality over stated slots only."""
    hit = {k: 0 for k in SLOT_MAP}
    stated = {k: 0 for k in SLOT_MAP}
    fp = {k: 0 for k in SLOT_MAP}
    unstated = {k: 0 for k in SLOT_MAP}

    t0 = time.perf_counter()
    for _, r in df.iterrows():
        it = parse_slots(r["text"])
        for slot, col in SLOT_MAP.items():
            truth = r[col]
            got = getattr(it, slot)
            if np.isfinite(truth):
                stated[slot] += 1
                tol = 0.02 * abs(truth) + SLOT_TOL[slot]
                if np.isfinite(got) and abs(got - truth) <= tol:
                    hit[slot] += 1
            else:
                unstated[slot] += 1
                if np.isfinite(got):
                    fp[slot] += 1
    parse_ms = (time.perf_counter() - t0) / len(df) * 1000.0

    recall = {k: hit[k] / max(stated[k], 1) for k in SLOT_MAP}
    fprate = {k: fp[k] / max(unstated[k], 1) for k in SLOT_MAP}
    return dict(
        recall=recall, false_positive=fprate, stated=stated, unstated=unstated,
        mean_slot_accuracy=float(np.mean(list(recall.values()))),
        mean_false_positive=float(np.mean(list(fprate.values()))),
        parse_latency_ms=parse_ms,
    )


def build_features(df: pd.DataFrame, lex_fn=None):
    """Assemble the 14-dim feature matrix.

    `lex_fn(text) -> (embb, urllc, mmtc)` supplies the lexical prior. Passing a
    different `lex_fn` is exactly how Experiment 3 swaps the Intent Agent while
    holding every other pipeline stage fixed.
    """
    if lex_fn is None:
        lex_fn = lexicon_scores

    recs = []
    for _, r in df.iterrows():
        it = parse_slots(r["text"])
        it.lex_embb, it.lex_urllc, it.lex_mmtc = lex_fn(r["text"])
        recs.append(asdict(it))

    X = pd.DataFrame(recs)
    for slot in SLOT_MAP:
        X["miss_" + slot.split("_")[0]] = (~np.isfinite(X[slot].astype(float))).astype(int)
    X["log_latency"] = np.log10(X["latency_ms"].astype(float).clip(lower=1e-3))
    X["log_throughput"] = np.log10(X["throughput_mbps"].astype(float).clip(lower=1e-4))
    X["log_density"] = np.log10(X["density_per_km2"].astype(float).clip(lower=1.0))
    X = X[FEATURES].astype(float)
    # NOTE: missing slots are deliberately left as NaN here. Imputation is a
    # *fitted* step and is performed inside split() using training-split medians
    # only, so no test-split statistic can leak into the feature matrix.
    return X


# ---------------------------------------------------------------------------
# S3/S5. Candidate screening + safety contract
# ---------------------------------------------------------------------------

def feasible_set(row):
    """One-sided, tolerant screening: exclude a slice only when a *stated*
    requirement provably exceeds its envelope. Unstated slots never exclude."""
    out = []
    for s, c in CATALOGUE.items():
        ok = True
        if np.isfinite(row["latency_ms"]) and row["latency_ms"] < c["lat"][0] * 0.5:
            ok = False
        if np.isfinite(row["throughput_mbps"]) and row["throughput_mbps"] > c["thr"][1] * 1.5:
            ok = False
        if np.isfinite(row["reliability_pct"]) and row["reliability_pct"] > c["rel"][1] + 1e-6:
            ok = False
        if np.isfinite(row["density_per_km2"]) and row["density_per_km2"] > c["den"][1] * 2.0:
            ok = False
        if ok:
            out.append(s)
    return out or list(CATALOGUE)


def contract_valid(pred_label, row):
    return pred_label in feasible_set(row)


# ---------------------------------------------------------------------------
# S4-baseline. Rule-based operator playbook
# ---------------------------------------------------------------------------

def rule_based(row):
    lat, rel = row["latency_ms"], row["reliability_pct"]
    den, thr = row["density_per_km2"], row["throughput_mbps"]
    if (np.isfinite(lat) and lat <= 10) or (np.isfinite(rel) and rel >= 99.99):
        return "URLLC"
    if (np.isfinite(den) and den >= 1e4) and (np.isfinite(thr) and thr <= 2.0):
        return "mMTC"
    if np.isfinite(thr) and thr >= 25:
        return "eMBB"
    if np.isfinite(den) and den >= 1e4:
        return "mMTC"
    return "eMBB"


def rule_based_lex(row):
    """Lexicon-augmented playbook: the *fair* non-learned baseline.

    `rule_based` sees only the five raw KPI slots, while every learned model also
    receives the three lexical-prior features. Comparing them is therefore not
    like-for-like and understates what hand-written rules can do. This variant
    consumes exactly the same 14-dim feature vector: it trusts the lexical prior
    whenever it fires with a unique winner, and falls back to the KPI playbook
    otherwise -- the obvious thing an engineer would write given both channels.
    """
    lex = np.array([row.get("lex_embb", 0.0), row.get("lex_urllc", 0.0),
                    row.get("lex_mmtc", 0.0)], dtype=float)
    lex = np.where(np.isfinite(lex), lex, 0.0)
    top = lex.max()
    if top > 0 and int((lex == top).sum()) == 1:
        return SLICES[int(lex.argmax())]
    return rule_based(row)


# ---------------------------------------------------------------------------
# Shared evaluation utilities
# ---------------------------------------------------------------------------

def ablation_fidelity(predict_fn, Xte, attributions, median, k_list=(1, 2, 3), seed=SEED):
    """Top-k ablation faithfulness against random-k AND lowest-k controls.

    `attributions` is (n_samples, n_features) signed or absolute attribution for
    the predicted class. A faithful attribution should flip the decision far
    more often than ablating random features -- but beating random only shows the
    ranking is non-degenerate. The lowest-k control is the stronger test: it
    ablates the SAME NUMBER of features from the same instances, chosen by the
    same attribution vector, so a large top-k/lowest-k separation cannot be
    explained by the amount of perturbation, only by where it was directed.

    Caveat: median replacement moves inputs off the data manifold, so flips may
    partly reflect distribution shift. The three measures share that bias, which
    is why the paper reads them as a relative comparison.
    """
    Xte = pd.DataFrame(Xte, columns=FEATURES).astype(float)
    base = predict_fn(Xte)
    order = np.argsort(-np.abs(np.asarray(attributions)), axis=1)
    rng = np.random.default_rng(seed)
    out = {}
    for k in k_list:
        Xa = Xte.to_numpy(dtype=float, copy=True)
        for i in range(len(Xa)):
            for j in order[i, :k]:
                Xa[i, j] = median.iloc[j]
        out[f"top{k}"] = float(np.mean(
            predict_fn(pd.DataFrame(Xa, columns=FEATURES)) != base))

        Xr = Xte.to_numpy(dtype=float, copy=True)
        for i in range(len(Xr)):
            for j in rng.choice(len(FEATURES), k, replace=False):
                Xr[i, j] = median.iloc[j]
        out[f"rand{k}"] = float(np.mean(
            predict_fn(pd.DataFrame(Xr, columns=FEATURES)) != base))

        # Lowest-attributed k: same budget, same attribution vector, opposite end.
        Xl = Xte.to_numpy(dtype=float, copy=True)
        for i in range(len(Xl)):
            for j in order[i, -k:]:
                Xl[i, j] = median.iloc[j]
        out[f"low{k}"] = float(np.mean(
            predict_fn(pd.DataFrame(Xl, columns=FEATURES)) != base))

        out[f"ratio{k}"] = out[f"top{k}"] / max(out[f"rand{k}"], 1e-9)
        out[f"ratiolow{k}"] = out[f"top{k}"] / max(out[f"low{k}"], 1e-9)
    return out


def escalation_table(proba, ypred, yte, Xte, taus=(0.50, 0.60, 0.70, 0.80, 0.90)):
    """Confidence + safety-contract escalation policy."""
    conf = np.asarray(proba).max(axis=1)
    infeasible = np.array([not contract_valid(SLICES[p], r)
                           for p, (_, r) in zip(ypred, Xte.iterrows())])
    wrong = (np.asarray(ypred) != np.asarray(yte))
    rows = []
    for tau in taus:
        flag = infeasible | (conf < tau)
        rows.append(dict(
            tau=tau,
            escalated=float(flag.mean()),
            precision=float(wrong[flag].mean()) if flag.any() else 0.0,
            recall=float(wrong[flag].sum() / max(wrong.sum(), 1)),
            autonomous_acc=float(1 - wrong[~flag].mean()) if (~flag).any() else 1.0,
        ))
    return pd.DataFrame(rows), conf, wrong


def attribution_concentration(attr):
    """Top-1 attribution mass: |phi_(1)| / sum_j |phi_j|, per instance.

    Low concentration means the explanation spreads blame across many features,
    i.e. the model had no single dominant reason -- a candidate error signal
    that is derived from the EXPLANATION rather than from the class posterior.
    """
    a = np.abs(np.asarray(attr, dtype=float))
    s = a.sum(axis=1)
    return a.max(axis=1) / np.where(s > 0, s, 1.0)


def _prank(v):
    """Percentile rank in [0,1], no scipy dependency."""
    v = np.asarray(v, dtype=float)
    o = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    return o / max(len(v) - 1, 1)


def _rank_escalate(score, budget, force=None):
    """Escalate the `budget` fraction with the LOWEST score.

    `force` entries are always escalated and consume part of the budget, so
    every policy in the comparison spends the same number of operator reviews.
    """
    score = np.asarray(score, dtype=float)
    n = len(score)
    k = int(round(budget * n))
    flag = np.zeros(n, dtype=bool)
    if force is not None:
        flag |= np.asarray(force, dtype=bool)
    rem = k - int(flag.sum())
    if rem > 0:
        cand = np.where(~flag)[0]
        order = cand[np.argsort(score[cand], kind="stable")]
        flag[order[:rem]] = True
    return flag


def escalation_signal_ablation(proba, ypred, yte, Xte, attr, tau):
    """Does the EXPLANATION add error-detection power over confidence alone?

    The paper's G3 claim is that explanation, not merely the posterior, drives
    the autonomy/accuracy trade-off. Testing that requires holding the operator
    budget fixed: a signal that escalates more intents will trivially recall
    more errors. Every policy below therefore escalates the SAME number of
    intents -- the number the confidence-only policy spends at `tau` -- and the
    only question is which ranking surfaces errors best.
    """
    conf = np.asarray(proba).max(axis=1)
    wrong = (np.asarray(ypred) != np.asarray(yte))
    infeasible = np.array([not contract_valid(SLICES[p], r)
                           for p, (_, r) in zip(ypred, Xte.iterrows())])
    mass = attribution_concentration(attr)
    budget = float((conf < tau).mean())

    def stats(flag, name):
        flag = np.asarray(flag, dtype=bool)
        return dict(
            signal=name,
            escalated=float(flag.mean()),
            precision=float(wrong[flag].mean()) if flag.any() else 0.0,
            recall=float(wrong[flag].sum() / max(wrong.sum(), 1)),
            autonomous_acc=(float(1 - wrong[~flag].mean())
                            if (~flag).any() else 1.0))

    combined = 0.5 * (_prank(conf) + _prank(mass))
    rows = [
        stats(_rank_escalate(conf, budget), "Confidence only"),
        stats(_rank_escalate(mass, budget), "Attribution concentration only"),
        stats(_rank_escalate(conf, budget, force=infeasible),
              "Confidence + contract"),
        stats(_rank_escalate(combined, budget, force=infeasible),
              "Confidence + contract + attribution"),
    ]
    out = pd.DataFrame(rows)
    out.insert(1, "budget", budget)
    return out


def select_tau(tbl, min_autonomous_acc=0.98):
    """Pre-registered threshold rule, applied to *selection* data only.

    Choose the smallest tau (i.e. the least escalation, hence the most autonomy)
    whose autonomous accuracy reaches the SLA target. If no sweep point reaches
    the target, fall back to the tau with the highest autonomous accuracy.

    The rule is fixed in advance and consumes no held-out label, so the operating
    point reported on the test split is not tuned on it.
    """
    ok = tbl[tbl["autonomous_acc"] >= min_autonomous_acc]
    if len(ok):
        return float(ok.sort_values("tau").iloc[0]["tau"])
    return float(tbl.sort_values("autonomous_acc", ascending=False).iloc[0]["tau"])


def subgroup_report(ypred, yte, dte):
    """Accuracy decomposition by KPI ambiguity and use-case vocabulary kind."""
    from sklearn.metrics import accuracy_score
    ypred, yte = np.asarray(ypred), np.asarray(yte)
    amb = dte["ambiguous"].values.astype(bool)

    def _acc(mask):
        return float(accuracy_score(yte[mask], ypred[mask])) if mask.any() \
            else float("nan")

    rep = dict(
        acc_overall=float(accuracy_score(yte, ypred)),
        acc_clear=_acc(~amb),
        acc_ambiguous=_acc(amb),
        n_clear=int((~amb).sum()), n_ambiguous=int(amb.sum()),
        err_total=int((ypred != yte).sum()),
        err_from_ambiguous=int(((ypred != yte) & amb).sum()),
    )
    for kind in ("known", "unseen", "generic"):
        m = (dte["uc_kind"].values == kind)
        rep[f"acc_{kind}"] = float(accuracy_score(yte[m], ypred[m])) if m.any() else float("nan")
        rep[f"n_{kind}"] = int(m.sum())
        rep[f"err_{kind}"] = int(((ypred != yte) & m).sum())
    # The headline open-vocabulary metric: in-lexicon minus out-of-lexicon.
    oov = [rep[f"acc_{k}"] for k in ("unseen", "generic")]
    rep["oov_gap"] = float(rep["acc_known"] - np.mean(oov))
    return rep


def save_json(obj, name):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  -> {path}")
    return path


def save_csv(df, name):
    path = os.path.join(OUT, name)
    pd.DataFrame(df).to_csv(path, index=False)
    print(f"  -> {path}")
    return path


def split(X, y, df):
    """Stratified 70/30 split, then median-impute using TRAIN statistics only.

    build_features() returns raw NaNs for unstated slots; the median is fitted
    here on the training split and applied to both splits, so the held-out data
    contributes nothing to the imputation statistics.
    """
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte, dtr, dte = train_test_split(
        X, y, df, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    med = Xtr.median(numeric_only=True)
    return Xtr.fillna(med), Xte.fillna(med), ytr, yte, dtr, dte
