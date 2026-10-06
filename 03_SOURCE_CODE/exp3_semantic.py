"""
Experiment 3 -- Semantic Intent Agent vs the design-time lexicon.

Research gap addressed
    E1 shows accuracy is 100% on intents whose use-case phrase is in the Intent
    Agent's design-time lexicon but drops by ~11 points on lexically unseen and
    vertical-agnostic phrasings. The lexical prior is simultaneously the
    strongest single signal and the framework's principal brittleness. E1 names
    semantic coverage of the Intent Agent as the highest-value improvement.

Design
    Only the lexical prior changes. Slot filling, the feature set, the Candidate
    and Validation Agents, the Selection Agent, the corpus, the split and the
    seed are all held fixed, so any movement in the open-vocabulary gap is
    attributable to the Intent Agent's semantic coverage alone.

    A  Lexicon (E1 baseline) : binary exact substring match
    B  TF-IDF char n-gram    : cosine similarity to per-slice prototypes
    C  Sentence embeddings   : MiniLM cosine similarity (optional dependency)
    D  LLM zero-shot         : Together API slice-affinity scoring (optional)

    Prototypes are drawn *only* from the design-time known use cases, so the
    unseen and generic phrasings remain genuinely out-of-vocabulary for every
    variant. Scoring is cached per distinct use-case phrase.

Headline metric
    oov_gap = accuracy(in-lexicon) - mean(accuracy(unseen), accuracy(generic))

Outputs (results/)
    e3_variants.csv, e3_summary.json, e3_semantic_cache.json
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from xagentic_common import (
    SEED, SLICES, SLICE_ID, OUT,
    USE_CASES_KNOWN, synth_corpus, build_features, split,
    lexicon_scores, escalation_table, subgroup_report, save_json, save_csv,
)

CACHE_PATH = os.path.join(OUT, "e3_semantic_cache.json")

# The seven corpus templates place the use-case phrase in a recoverable span.
# Extracting it is ordinary slot filling -- the same job the Intent Agent
# already does for KPI values -- and lets every semantic variant be cached
# against the small set of distinct phrases rather than all 3000 intents.
UC_PATTERNS = [
    r"Deploy a slice for (.+?) with latency",
    r"I need (.+?) support:",
    r"Provision connectivity for (.+?)\.",
    r"Can you set up (.+?)\?",
    r"Create a network slice supporting (.+?) with",
    r"Set up (.+?)\. It should sustain",
    r"Please onboard (.+?) with a",
]


def extract_use_case(text: str) -> str:
    for p in UC_PATTERNS:
        m = re.search(p, text, flags=re.I)
        if m:
            return m.group(1).strip()
    return ""


def _load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _norm(scores):
    """Normalise three raw affinities to [0, 1] with max-scaling."""
    v = np.asarray(scores, dtype=float)
    v = np.clip(v, 0.0, None)
    return (v / v.max()).tolist() if v.max() > 0 else [0.0, 0.0, 0.0]


# ===========================================================================
# Variant B -- TF-IDF character n-gram similarity
# ===========================================================================

def build_tfidf_scorer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    protos = {s: USE_CASES_KNOWN[s] for s in SLICES}
    corpus = [p for s in SLICES for p in protos[s]]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(corpus)
    mats = {s: vec.transform(protos[s]) for s in SLICES}

    def score(phrase: str):
        if not phrase:
            return [0.0, 0.0, 0.0]
        q = vec.transform([phrase])
        return _norm([float(cosine_similarity(q, mats[s]).max()) for s in SLICES])

    return score


# ===========================================================================
# Variant C -- sentence-transformer embeddings
# ===========================================================================

def build_embedding_scorer(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    protos = {s: USE_CASES_KNOWN[s] for s in SLICES}
    emb = {s: model.encode(protos[s], normalize_embeddings=True) for s in SLICES}

    def score(phrase: str):
        if not phrase:
            return [0.0, 0.0, 0.0]
        q = model.encode([phrase], normalize_embeddings=True)[0]
        return _norm([float(np.max(emb[s] @ q)) for s in SLICES])

    return score


# ===========================================================================
# Variant D -- Together API zero-shot slice affinity
# ===========================================================================

TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
DEFAULT_LLM = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

LLM_SYSTEM = (
    "You are a 3GPP network-slicing expert. Given a service use case, rate how "
    "well it matches each standardised slice/service type.\n"
    "eMBB  = enhanced mobile broadband (high throughput, video, media)\n"
    "URLLC = ultra-reliable low-latency communication (control, safety, automation)\n"
    "mMTC  = massive machine-type communication (many low-rate sensors/meters)\n"
    'Reply with ONLY a JSON object: {"eMBB": <0-1>, "URLLC": <0-1>, "mMTC": <0-1>}'
)


def build_llm_scorer(model=None, api_key=None, max_workers=8):
    import requests

    api_key = api_key or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY is not set")
    model = model or os.environ.get("TOGETHER_MODEL", DEFAULT_LLM)
    session = requests.Session()

    def _one(phrase: str):
        body = {
            "model": model,
            "messages": [{"role": "system", "content": LLM_SYSTEM},
                         {"role": "user", "content": f"Use case: {phrase}"}],
            "temperature": 0.0, "max_tokens": 80,
        }
        for attempt in range(4):
            try:
                r = session.post(
                    TOGETHER_URL, json=body,
                    headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                txt = r.json()["choices"][0]["message"]["content"]
                m = re.search(r"\{.*\}", txt, flags=re.S)
                d = json.loads(m.group(0)) if m else {}
                return _norm([float(d.get(s, 0.0)) for s in SLICES])
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        print(f"     [warn] LLM scoring failed for {phrase!r}; falling back to zeros")
        return [0.0, 0.0, 0.0]

    def score_many(phrases):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_one, phrases))

    return _one, score_many, model


# ===========================================================================
# Runner
# ===========================================================================

def _cached_lex_fn(variant, phrases, scorer=None, score_many=None, cache=None):
    """Score every distinct use-case phrase once, then serve lookups."""
    cache = cache if cache is not None else {}
    bucket = cache.setdefault(variant, {})
    todo = [p for p in phrases if p not in bucket]
    if todo:
        print(f"     scoring {len(todo)} distinct phrases ({variant}) ...")
        if score_many is not None:
            for p, s in zip(todo, score_many(todo)):
                bucket[p] = s
        else:
            for p in todo:
                bucket[p] = scorer(p)

    def lex_fn(text: str):
        return tuple(bucket.get(extract_use_case(text), [0.0, 0.0, 0.0]))

    return lex_fn


def run(df=None, use_llm=None, use_embeddings=True, verbose=True):
    if verbose:
        print("=" * 72)
        print("EXPERIMENT 3 -- Semantic Intent Agent vs design-time lexicon")
        print("=" * 72)

    if df is None:
        df = synth_corpus()

    y = df["slice"].map(SLICE_ID).values
    phrases = sorted({extract_use_case(t) for t in df["text"]} - {""})
    print(f"[E3] {len(phrases)} distinct use-case phrases in the corpus")

    cache = _load_cache()
    variants = {"A. Lexicon (E1)": lexicon_scores}

    print("[E3] building TF-IDF char n-gram scorer ...")
    variants["B. TF-IDF n-gram"] = _cached_lex_fn(
        "tfidf", phrases, scorer=build_tfidf_scorer(), cache=cache)

    if use_embeddings:
        try:
            print("[E3] building sentence-embedding scorer ...")
            variants["C. Embeddings"] = _cached_lex_fn(
                "embed", phrases, scorer=build_embedding_scorer(), cache=cache)
        except Exception as e:
            print(f"[E3] embeddings unavailable ({type(e).__name__}: {e}) -- skipping. "
                  f"pip install sentence-transformers to enable")

    if use_llm is None:
        use_llm = bool(os.environ.get("TOGETHER_API_KEY"))
    if use_llm:
        try:
            print("[E3] building Together API LLM scorer ...")
            one, many, model_id = build_llm_scorer()
            variants[f"D. LLM zero-shot"] = _cached_lex_fn(
                "llm", phrases, scorer=one, score_many=many, cache=cache)
            print(f"     model = {model_id}")
        except Exception as e:
            print(f"[E3] LLM unavailable ({type(e).__name__}: {e}) -- skipping. "
                  f"Set TOGETHER_API_KEY to enable")
    else:
        print("[E3] TOGETHER_API_KEY not set -- skipping LLM variant")

    _save_cache(cache)

    rows, details = [], {}
    for name, lex_fn in variants.items():
        print(f"[E3] evaluating {name} ...")
        t0 = time.perf_counter()
        X = build_features(df, lex_fn=lex_fn)
        feat_ms = (time.perf_counter() - t0) / len(df) * 1000

        Xtr, Xte, ytr, yte, dtr, dte = split(X, y, df)
        clf = GradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)

        rep = subgroup_report(pred, yte, dte)
        esc, conf, wrong = escalation_table(proba, pred, yte, Xte)
        at80 = esc[np.isclose(esc["tau"], 0.80)].iloc[0]

        rows.append(dict(
            variant=name,
            acc=rep["acc_overall"],
            f1=float(f1_score(yte, pred, average="macro")),
            acc_known=rep["acc_known"], acc_unseen=rep["acc_unseen"],
            acc_generic=rep["acc_generic"], oov_gap=rep["oov_gap"],
            acc_ambiguous=rep["acc_ambiguous"],
            intent_ms=feat_ms,
            escalated80=float(at80["escalated"]),
            autonomous80=float(at80["autonomous_acc"]),
        ))
        details[name] = dict(subgroup=rep, escalation=esc.to_dict(orient="records"))
        print(f"  acc={rep['acc_overall']:.4f} known={rep['acc_known']:.4f} "
              f"unseen={rep['acc_unseen']:.4f} generic={rep['acc_generic']:.4f} "
              f"OOV gap={rep['oov_gap']*100:.1f} pts")

    res = pd.DataFrame(rows)
    save_csv(res, "e3_variants.csv")
    print()
    print(res.to_string(index=False))

    base = res[res["variant"].str.startswith("A.")].iloc[0]
    best = res.iloc[res["oov_gap"].astype(float).values.argmin()]
    summary = dict(
        experiment="E3", label="Semantic Intent Agent",
        table=res.to_dict(orient="records"), details=details,
        n_phrases=len(phrases),
        baseline_variant=str(base["variant"]),
        baseline_acc=float(base["acc"]), baseline_oov_gap=float(base["oov_gap"]),
        best_variant=str(best["variant"]),
        best_acc=float(best["acc"]), best_oov_gap=float(best["oov_gap"]),
        gap_closed_pts=float((base["oov_gap"] - best["oov_gap"]) * 100),
        acc_gain_pts=float((best["acc"] - base["acc"]) * 100),
    )
    save_json(summary, "e3_summary.json")
    print(f"\n[E3] {summary['best_variant']} closes the open-vocabulary gap by "
          f"{summary['gap_closed_pts']:.1f} pts "
          f"({summary['baseline_oov_gap']*100:.1f} -> {summary['best_oov_gap']*100:.1f}) "
          f"and lifts accuracy by {summary['acc_gain_pts']:.1f} pts")
    return summary


if __name__ == "__main__":
    run()
