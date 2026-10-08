"""
Explainable Agentic AI for Intent-Based Network Slice Selection
Proof-of-concept experiment pipeline (single comparative experiment).

Stages
  S1  Synthetic intent + KPI corpus grounded in 3GPP TS 22.261 / TS 23.501 slice KPIs
  S2  Intent Agent        : NL intent -> structured intent vector (slot filling)
  S3  Candidate Agent     : SST/feasibility screening against a slice catalogue
  S4  Selection Agent     : supervised slice classifier (5 models + rule baseline)
  S5  Validation Agent    : safety-contract check (3GPP KPI envelopes)
  S6  Explanation Agent   : SHAP global + local attributions, fidelity + latency

Outputs: results/*.json, results/*.csv, results/*.pdf (figures)
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

import shap

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

SLICES = ["eMBB", "URLLC", "mMTC"]
SLICE_ID = {s: i for i, s in enumerate(SLICES)}

# ----------------------------------------------------------------------------
# S1. Slice catalogue + synthetic intent/KPI corpus
# ----------------------------------------------------------------------------

# 3GPP-informed KPI envelopes per standardised slice/service type (SST).
CATALOGUE = {
    "eMBB":  dict(lat=(10.0, 150.0), thr=(25.0, 1000.0), rel=(99.0, 99.9),
                  den=(1e2, 5e4), mob=(0.0, 250.0), sst=1),
    "URLLC": dict(lat=(0.5, 10.0),   thr=(0.1, 100.0),  rel=(99.99, 99.9999),
                  den=(1e2, 1e4),   mob=(0.0, 500.0), sst=2),
    "mMTC":  dict(lat=(50.0, 10000.0), thr=(0.001, 2.0), rel=(95.0, 99.9),
                  den=(1e4, 1e6),   mob=(0.0, 30.0),  sst=3),
}

# Use cases the Intent Agent's domain lexicon has seen during design time.
USE_CASES_KNOWN = {
    "eMBB": ["4K video streaming", "AR/VR gaming", "live broadcast",
             "fixed wireless access", "stadium hotspot"],
    "URLLC": ["factory robot control", "remote surgery", "smart-grid protection",
              "drone fleet control", "haptic teleoperation"],
    "mMTC": ["smart water meters", "agricultural sensors", "asset trackers",
             "environmental monitoring", "pipeline telemetry"],
}

# Lexically *unseen* use cases: realistic open-vocabulary drift. They carry no
# keyword signal, so the Selection Agent must rely on the KPI slots alone.
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

# Templates differ in which KPI slots the operator actually states. Missing slots
# are the norm in real intent declarations, not the exception.
TEMPLATES = [
    # full specification
    "Deploy a slice for {uc} with latency under {lat} ms, at least {thr} Mbps "
    "throughput and {rel}% reliability for about {den} devices per square km.",
    "I need {uc} support: end-to-end delay must stay below {lat} ms, sustained rate "
    "{thr} Mbps, availability {rel}%, device density around {den} per km2, "
    "mobility up to {mob} km/h.",
    # throughput omitted
    "Provision connectivity for {uc}. Target budget: {lat} ms latency, {rel}% "
    "reliability, {den} devices/km2, users moving at {mob} km/h.",
    # density omitted
    "Can you set up {uc}? We require no more than {lat} ms one-way latency, a "
    "minimum of {thr} Mbps and {rel}% service availability.",
    # latency omitted
    "Create a network slice supporting {uc} with {thr} Mbps guaranteed bitrate, "
    "{rel} percent reliability, {den} terminals per km2 and {mob} km/h mobility.",
    # minimal: two slots only
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


def synth_corpus(n=3000, p_amb=0.30, seed=SEED):
    """Generate NL intents with latent ground-truth slice labels.

    Difficulty is controlled by two realistic effects:
      * KPI ambiguity  -- a fraction `p_amb` of requests have KPI targets blended
        towards a neighbouring slice envelope (under-specified or over-cautious
        operator requests that sit in the overlap region of two SSTs);
      * vocabulary drift -- only ~40% of intents use a use-case phrase the Intent
        Agent's design-time lexicon covers; the rest are unseen or generic.
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
            blend = lambda v, rng_o: float(np.exp((1 - a) * np.log(max(v, 1e-6))
                                                  + a * np.log(max(rng_o, 1e-6))))
            lat = blend(lat, _loguniform(*o["lat"], rng))
            thr = blend(thr, _loguniform(*o["thr"], rng))
            den = blend(den, _loguniform(*o["den"], rng))
            rel = (1 - a) * rel + a * float(rng.uniform(*o["rel"]))

        u = rng.random()
        if u < 0.40:
            pool = USE_CASES_KNOWN[label]
            uc_kind = "known"
        elif u < 0.70:
            pool = USE_CASES_UNSEEN[label]
            uc_kind = "unseen"
        else:
            pool = GENERIC_USE_CASES
            uc_kind = "generic"
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


# ----------------------------------------------------------------------------
# S2. Intent Agent -- deterministic slot-filling NLU
# ----------------------------------------------------------------------------

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
    lex_embb: int = 0
    lex_urllc: int = 0
    lex_mmtc: int = 0


def parse_intent(text: str) -> Intent:
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
    low = text.lower()
    it.lex_embb = int(any(k.lower() in low for k in LEXICON["eMBB"]))
    it.lex_urllc = int(any(k.lower() in low for k in LEXICON["URLLC"]))
    it.lex_mmtc = int(any(k.lower() in low for k in LEXICON["mMTC"]))
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


def featurise(df: pd.DataFrame):
    """Run the Intent Agent over the corpus and score slot-filling quality.

    Slot accuracy is measured only over slots the operator actually stated;
    spurious extractions for unstated slots are counted as false positives.
    """
    recs = []
    hit = {k: 0 for k in SLOT_MAP}
    stated = {k: 0 for k in SLOT_MAP}
    fp = {k: 0 for k in SLOT_MAP}
    unstated = {k: 0 for k in SLOT_MAP}

    t0 = time.perf_counter()
    for _, r in df.iterrows():
        it = parse_intent(r["text"])
        recs.append(asdict(it))
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

    X = pd.DataFrame(recs)
    # Explicit missingness indicators: the agent must know what it does not know.
    for slot in SLOT_MAP:
        X["miss_" + slot.split("_")[0]] = (~np.isfinite(X[slot].astype(float))).astype(int)
    X["log_latency"] = np.log10(X["latency_ms"].astype(float).clip(lower=1e-3))
    X["log_throughput"] = np.log10(X["throughput_mbps"].astype(float).clip(lower=1e-4))
    X["log_density"] = np.log10(X["density_per_km2"].astype(float).clip(lower=1.0))
    X = X[FEATURES].astype(float)
    X = X.fillna(X.median(numeric_only=True))

    slot = {k: hit[k] / max(stated[k], 1) for k in SLOT_MAP}
    slot_fp = {k: fp[k] / max(unstated[k], 1) for k in SLOT_MAP}
    out = dict(
        recall=slot, false_positive=slot_fp,
        stated=stated, unstated=unstated,
        mean_slot_accuracy=float(np.mean(list(slot.values()))),
        mean_false_positive=float(np.mean(list(slot_fp.values()))),
        parse_latency_ms=parse_ms,
    )
    return X, out


# ----------------------------------------------------------------------------
# S3/S5. Candidate screening + safety contract
# ----------------------------------------------------------------------------

def feasible_set(row):
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


# ----------------------------------------------------------------------------
# S4-baseline. Rule-based selector
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print("[S1] generating corpus ...")
    df = synth_corpus(3000)
    df.to_csv(os.path.join(OUT, "corpus.csv"), index=False)

    print("[S2] intent agent: slot filling ...")
    X, slot = featurise(df)
    y = df["slice"].map(SLICE_ID).values
    print("     slots:", json.dumps(slot, indent=2))

    Xtr, Xte, ytr, yte, dtr, dte = train_test_split(
        X, y, df, test_size=0.3, random_state=SEED, stratify=y)

    models = {
        "Logistic Regression": Pipeline([("sc", StandardScaler()),
                                         ("m", LogisticRegression(max_iter=2000, random_state=SEED))]),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
        "MLP (64,32)": Pipeline([("sc", StandardScaler()),
                                 ("m", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1200,
                                                     random_state=SEED))]),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }

    rows = []
    t0 = time.perf_counter()
    rb = np.array([SLICE_ID[rule_based(r)] for _, r in Xte.iterrows()])
    rb_lat = (time.perf_counter() - t0) / len(Xte) * 1000
    rows.append(dict(model="Rule-based playbook",
                     acc=accuracy_score(yte, rb),
                     f1=f1_score(yte, rb, average="macro"),
                     cv_mean=np.nan, cv_std=np.nan,
                     valid=float(np.mean([contract_valid(SLICES[p], r)
                                          for p, (_, r) in zip(rb, Xte.iterrows())])),
                     lat_ms=rb_lat))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fitted = {}
    for name, m in models.items():
        print(f"[S4] training {name} ...")
        cv = cross_val_score(m, Xtr, ytr, cv=skf, scoring="f1_macro", n_jobs=1)
        m.fit(Xtr, ytr)
        fitted[name] = m
        t0 = time.perf_counter()
        p = m.predict(Xte)
        lat = (time.perf_counter() - t0) / len(Xte) * 1000
        rows.append(dict(model=name, acc=accuracy_score(yte, p),
                         f1=f1_score(yte, p, average="macro"),
                         cv_mean=cv.mean(), cv_std=cv.std(),
                         valid=float(np.mean([contract_valid(SLICES[pi], r)
                                              for pi, (_, r) in zip(p, Xte.iterrows())])),
                         lat_ms=lat))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
    print(res.to_string(index=False))

    best_name = res.iloc[1:].sort_values("f1", ascending=False).iloc[0]["model"]
    best = fitted[best_name]
    ypred = best.predict(Xte)
    print(f"[S4] best = {best_name}")

    pr, rc, f1c, sup = precision_recall_fscore_support(yte, ypred, labels=[0, 1, 2])
    per_class = pd.DataFrame(dict(slice=SLICES, precision=pr, recall=rc, f1=f1c, support=sup))
    per_class.to_csv(os.path.join(OUT, "per_class.csv"), index=False)
    print(per_class.to_string(index=False))

    cm = confusion_matrix(yte, ypred, labels=[0, 1, 2])
    np.savetxt(os.path.join(OUT, "confusion.csv"), cm, fmt="%d", delimiter=",")

    amb = dte["ambiguous"].values.astype(bool)
    fail = dict(
        acc_clear=float(accuracy_score(yte[~amb], ypred[~amb])),
        acc_ambiguous=float(accuracy_score(yte[amb], ypred[amb])),
        n_clear=int((~amb).sum()), n_ambiguous=int(amb.sum()),
        err_total=int((ypred != yte).sum()),
        err_from_ambiguous=int(((ypred != yte) & amb).sum()),
    )
    for kind in ("known", "unseen", "generic"):
        m = (dte["uc_kind"].values == kind)
        fail[f"acc_{kind}"] = float(accuracy_score(yte[m], ypred[m]))
        fail[f"n_{kind}"] = int(m.sum())
        fail[f"err_{kind}"] = int(((ypred != yte) & m).sum())
    caught = sum(1 for pi, (_, r) in zip(ypred, Xte.iterrows())
                 if not contract_valid(SLICES[pi], r))
    caught_and_wrong = sum(1 for pi, yi, (_, r) in zip(ypred, yte, Xte.iterrows())
                           if not contract_valid(SLICES[pi], r) and pi != yi)
    fail["contract_flagged"] = int(caught)
    fail["contract_flagged_and_wrong"] = int(caught_and_wrong)
    fail["contract_precision"] = caught_and_wrong / max(caught, 1)
    fail["contract_recall"] = caught_and_wrong / max(fail["err_total"], 1)

    # Escalation policy: hand the intent to the operator when either the safety
    # contract is violated or the Selection Agent is not confident.
    proba = best.predict_proba(Xte)
    conf = proba.max(axis=1)
    infeasible = np.array([not contract_valid(SLICES[pi], r)
                           for pi, (_, r) in zip(ypred, Xte.iterrows())])
    wrong = (ypred != yte)
    esc = []
    for tau in (0.50, 0.60, 0.70, 0.80, 0.90):
        flag = infeasible | (conf < tau)
        esc.append(dict(tau=tau, escalated=float(flag.mean()),
                        precision=float(wrong[flag].mean()) if flag.any() else 0.0,
                        recall=float(wrong[flag].sum() / max(wrong.sum(), 1)),
                        autonomous_acc=float(1 - wrong[~flag].mean()) if (~flag).any() else 1.0))
    fail["escalation"] = esc
    fail["mean_confidence_correct"] = float(conf[~wrong].mean())
    fail["mean_confidence_wrong"] = float(conf[wrong].mean())
    print("[S5]", json.dumps(fail, indent=2))
    pd.DataFrame(esc).to_csv(os.path.join(OUT, "escalation.csv"), index=False)

    print("[S6] SHAP ...")
    bg = shap.sample(Xtr, 100, random_state=SEED)
    try:
        explainer = shap.TreeExplainer(best)
        t0 = time.perf_counter()
        sv = explainer.shap_values(Xte)
        shap_backend = "TreeExplainer (exact)"
    except Exception as e:
        print("     TreeExplainer unavailable ->", type(e).__name__, "; using PermutationExplainer")
        explainer = shap.PermutationExplainer(best.predict_proba, bg, seed=SEED)
        t0 = time.perf_counter()
        sv = explainer(Xte, silent=True).values
        shap_backend = "PermutationExplainer"
    shap_ms = (time.perf_counter() - t0) / len(Xte) * 1000

    sv = np.array(sv)
    if sv.ndim == 3 and sv.shape[0] == len(Xte):
        sv = np.transpose(sv, (2, 0, 1))
    imp = np.abs(sv).mean(axis=(0, 1))
    imp = imp / imp.sum()
    gi = pd.DataFrame(dict(feature=[PRETTY[f] for f in FEATURES], importance=imp)) \
        .sort_values("importance", ascending=False)
    gi.to_csv(os.path.join(OUT, "shap_global.csv"), index=False)
    print(gi.to_string(index=False))

    med = Xtr.median()
    fid = {}
    base = best.predict(Xte)
    order = np.argsort(-np.abs(sv).mean(axis=0), axis=1)
    rng = np.random.default_rng(SEED)
    for k in (1, 2, 3):
        Xa = Xte.to_numpy(dtype=float, copy=True)
        for i in range(len(Xa)):
            for j in order[i, :k]:
                Xa[i, j] = med.iloc[j]
        flip = float(np.mean(best.predict(pd.DataFrame(Xa, columns=FEATURES)) != base))
        Xr = Xte.to_numpy(dtype=float, copy=True)
        for i in range(len(Xr)):
            for j in rng.choice(len(FEATURES), k, replace=False):
                Xr[i, j] = med.iloc[j]
        rflip = float(np.mean(best.predict(pd.DataFrame(Xr, columns=FEATURES)) != base))
        fid[f"top{k}"] = flip
        fid[f"rand{k}"] = rflip
    fid["shap_latency_ms"] = shap_ms
    fid["backend"] = shap_backend
    print("[S6] fidelity:", json.dumps(fid, indent=2))

    local = []
    for c in range(3):
        idx = int(np.where((yte == c) & (ypred == c))[0][0])
        contrib = sv[c][idx]
        o = np.argsort(-np.abs(contrib))[:4]
        local.append(dict(slice=SLICES[c],
                          intent=dte.iloc[idx]["text"],
                          top=[(PRETTY[FEATURES[j]], float(contrib[j])) for j in o]))
    with open(os.path.join(OUT, "local_explanations.json"), "w") as f:
        json.dump(local, f, indent=2)
    for l in local:
        print(f"  {l['slice']}: " + ", ".join(f"{a}={b:+.3f}" for a, b in l["top"]))

    plt.rcParams.update({"font.size": 8, "figure.dpi": 300,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(SLICES)
    ax.set_yticks(range(3)); ax.set_yticklabels(SLICES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_confusion.pdf")); plt.close(fig)

    g = gi.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.barh(g["feature"], g["importance"], color="#2b6cb0")
    ax.set_xlabel("mean |SHAP| (normalised)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_shap_global.pdf")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ks = [1, 2, 3]; w = 0.38
    ax.bar([k - w / 2 for k in ks], [fid[f"top{k}"] for k in ks], w, label="SHAP top-$k$", color="#2b6cb0")
    ax.bar([k + w / 2 for k in ks], [fid[f"rand{k}"] for k in ks], w, label="random-$k$", color="#a0aec0")
    ax.set_xticks(ks); ax.set_xlabel("$k$ ablated features"); ax.set_ylabel("decision flip rate")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_fidelity.pdf")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    rr = res.sort_values("f1")
    ax.barh(rr["model"], rr["f1"], color="#2b6cb0")
    ax.set_xlabel("macro F1"); ax.set_xlim(0.5, 1.0)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_models.pdf")); plt.close(fig)

    e = pd.DataFrame(esc)
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(e["escalated"] * 100, e["autonomous_acc"] * 100, "o-", color="#2b6cb0")
    for _, r in e.iterrows():
        ax.annotate(f"$\\tau$={r['tau']:.1f}", (r["escalated"] * 100, r["autonomous_acc"] * 100),
                    textcoords="offset points", xytext=(4, -8), fontsize=6)
    ax.set_xlabel("intents escalated to operator (%)")
    ax.set_ylabel("accuracy on autonomous\ndecisions (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_escalation.pdf")); plt.close(fig)

    summary = dict(best_model=best_name, slot=slot,
                   models=res.to_dict("records"),
                   per_class=per_class.to_dict("records"),
                   confusion=cm.tolist(), failure=fail, fidelity=fid,
                   shap_global=gi.to_dict("records"),
                   escalation=esc,
                   n_total=len(df), n_train=len(Xtr), n_test=len(Xte),
                   report=classification_report(yte, ypred, target_names=SLICES))
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n[done] results ->", OUT)


if __name__ == "__main__":
    main()
