"""
EXPERIMENT 4 -- Combined configuration (E2 x E3 factorial)
==========================================================

E2 and E3 each change one component of the E1 pipeline and each reports a gain
on a different axis:

    E2  explainer      : SHAP (post-hoc)  ->  Decision-Path (in-hoc)
                         moves explanation cost by ~4 orders of magnitude
    E3  lexical prior  : exact match      ->  TF-IDF char n-gram
                         moves the out-of-lexicon gap

The paper claims these two improvements *compose*. That claim is untested by E2
and E3 alone, because neither varies the other factor. E4 tests it directly with
a full 2x2 factorial over the same corpus, split, seed, catalogue and contract:

                       | lexical prior = exact | lexical prior = TF-IDF
    ------------------ + --------------------- + ----------------------
    explainer = SHAP   |  (E1 baseline)        |  (E3 best)
    explainer = D-Path |  (E2 best)            |  (E4 combined)

Because the design is factorial we can report the *interaction*: whether the
combined cell matches the sum of the two single-factor gains (additive, i.e.
the axes are orthogonal) or falls short of / exceeds it.

Outputs (results/)
    e4_factorial.csv, e4_summary.json
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from xagentic_common import (
    SEED, SLICE_ID, FEATURES, synth_corpus, lexicon_scores, build_features,
    ablation_fidelity, escalation_table, subgroup_report, split,
    save_csv, save_json,
)
from exp2_inhoc import decision_path_attribution, _timed_explanation
from exp3_semantic import (
    extract_use_case, build_tfidf_scorer, _cached_lex_fn, _load_cache, _save_cache,
)

# The two factors, named exactly as the paper names them.
PRIOR_BASE, PRIOR_BEST = "exact", "tfidf"
EXPL_BASE, EXPL_BEST = "shap", "dpath"

CELL_LABEL = {
    (PRIOR_BASE, EXPL_BASE): "E1  exact + SHAP",
    (PRIOR_BASE, EXPL_BEST): "E2  exact + Decision-Path",
    (PRIOR_BEST, EXPL_BASE): "E3  TF-IDF + SHAP",
    (PRIOR_BEST, EXPL_BEST): "E4  TF-IDF + Decision-Path",
}


def _shap_values(gb, Xtr, Xte):
    """Post-hoc SHAP attributions, matching exp2_inhoc's backend fallback."""
    import shap
    bg = shap.sample(Xtr, 100, random_state=SEED)
    try:
        sv = shap.TreeExplainer(gb).shap_values(Xte)
        backend = "TreeExplainer"
    except Exception:
        sv = shap.PermutationExplainer(gb.predict_proba, bg, seed=SEED)(
            Xte, max_evals=2 * len(FEATURES) + 1).values
        backend = "PermutationExplainer"
    sv = np.asarray(sv)
    # Normalise to (n_samples, n_features) by collapsing the class axis on the
    # predicted class, matching how the paper defines a per-decision attribution.
    if sv.ndim == 3:
        if sv.shape[0] == 3:            # (classes, n, features)
            sv = np.transpose(sv, (1, 2, 0))
        pred = gb.predict(Xte)
        sv = np.stack([sv[i, :, pred[i]] for i in range(len(pred))])
    return np.abs(sv), backend


def _evaluate_cell(prior, expl, df, lex_fns, verbose=True):
    """Train and score one factorial cell."""
    label = CELL_LABEL[(prior, expl)]
    if verbose:
        print(f"[E4] {label} ...")

    y = df["slice"].map(SLICE_ID).values
    X = build_features(df, lex_fn=lex_fns[prior])
    Xtr, Xte, ytr, yte, dtr, dte = split(X, y, df)
    med = Xtr.median()
    backend = None

    if expl == EXPL_BEST:
        model = DecisionTreeClassifier(max_depth=8, random_state=SEED).fit(Xtr, ytr)
        t0 = time.perf_counter()
        pred = model.predict(Xte)
        infer_ms = (time.perf_counter() - t0) / len(Xte) * 1000
        attr, expl_ms = _timed_explanation(
            lambda: decision_path_attribution(model, Xte, pred), len(Xte))
        model_name = "Decision Tree (d=8)"
    else:
        model = GradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
        t0 = time.perf_counter()
        pred = model.predict(Xte)
        infer_ms = (time.perf_counter() - t0) / len(Xte) * 1000
        t0 = time.perf_counter()
        attr, backend = _shap_values(model, Xtr, Xte)
        expl_ms = (time.perf_counter() - t0) / len(Xte) * 1000
        model_name = "Gradient Boosting"

    proba = model.predict_proba(Xte)
    fid = ablation_fidelity(model.predict, Xte, attr, med)
    rep = subgroup_report(pred, yte, dte)
    esc, _, _ = escalation_table(proba, pred, yte, Xte)
    at80 = esc[np.isclose(esc["tau"], 0.80)].iloc[0]

    row = dict(
        cell=label, prior=prior, explainer=expl, model=model_name,
        acc=float(accuracy_score(yte, pred)),
        f1=float(f1_score(yte, pred, average="macro")),
        infer_ms=infer_ms, expl_ms=expl_ms,
        ratio=expl_ms / max(infer_ms, 1e-12),
        fid1=fid["ratio1"], fid3=fid["ratio3"],
        acc_known=rep["acc_known"], acc_unseen=rep["acc_unseen"],
        acc_generic=rep["acc_generic"], oov_gap=rep["oov_gap"],
        escalated80=float(at80["escalated"]),
        autonomous80=float(at80["autonomous_acc"]),
        shap_backend=backend,
    )
    if verbose:
        print(f"     acc={row['acc']:.4f}  expl={expl_ms:9.4f} ms  "
              f"fid1={row['fid1']:.1f}x  oov_gap={row['oov_gap']*100:.1f} pts")
    return row


def run(df=None, verbose=True):
    if verbose:
        print("=" * 72)
        print("EXPERIMENT 4 -- Combined configuration (E2 x E3 factorial)")
        print("=" * 72)

    if df is None:
        df = synth_corpus()

    # Build both lexical priors once; TF-IDF is cached per distinct phrase.
    phrases = sorted({extract_use_case(t) for t in df["text"]} - {""})
    cache = _load_cache()
    lex_fns = {
        PRIOR_BASE: lexicon_scores,
        PRIOR_BEST: _cached_lex_fn("tfidf", phrases,
                                   scorer=build_tfidf_scorer(), cache=cache),
    }
    _save_cache(cache)
    if verbose:
        print(f"[E4] {len(phrases)} distinct use-case phrases; 4 cells to evaluate")

    rows = [_evaluate_cell(p, e, df, lex_fns, verbose)
            for p in (PRIOR_BASE, PRIOR_BEST)
            for e in (EXPL_BASE, EXPL_BEST)]
    out = pd.DataFrame(rows)
    save_csv(out, "e4_factorial.csv")

    def cell(p, e, col):
        return float(out[(out.prior == p) & (out.explainer == e)][col].iloc[0])

    # Main effects and interaction, measured on the two axes the paper claims
    # are orthogonal: accuracy / OOV robustness (E3 axis) and explanation cost
    # (E2 axis).
    base = cell(PRIOR_BASE, EXPL_BASE, "acc")
    d_expl = cell(PRIOR_BASE, EXPL_BEST, "acc") - base      # E2 effect alone
    d_prior = cell(PRIOR_BEST, EXPL_BASE, "acc") - base     # E3 effect alone
    combined = cell(PRIOR_BEST, EXPL_BEST, "acc") - base    # both together
    interaction = combined - (d_expl + d_prior)

    oov_base = cell(PRIOR_BASE, EXPL_BASE, "oov_gap")
    summary = dict(
        n_cells=len(rows),
        acc_base=base,
        acc_e2_only=cell(PRIOR_BASE, EXPL_BEST, "acc"),
        acc_e3_only=cell(PRIOR_BEST, EXPL_BASE, "acc"),
        acc_combined=cell(PRIOR_BEST, EXPL_BEST, "acc"),
        effect_explainer_pts=d_expl * 100,
        effect_prior_pts=d_prior * 100,
        effect_combined_pts=combined * 100,
        interaction_pts=interaction * 100,
        oov_base_pts=oov_base * 100,
        oov_combined_pts=cell(PRIOR_BEST, EXPL_BEST, "oov_gap") * 100,
        expl_ms_base=cell(PRIOR_BASE, EXPL_BASE, "expl_ms"),
        expl_ms_combined=cell(PRIOR_BEST, EXPL_BEST, "expl_ms"),
        speedup_combined=cell(PRIOR_BASE, EXPL_BASE, "expl_ms")
        / max(cell(PRIOR_BEST, EXPL_BEST, "expl_ms"), 1e-12),
        fid1_combined=cell(PRIOR_BEST, EXPL_BEST, "fid1"),
        autonomous80_combined=cell(PRIOR_BEST, EXPL_BEST, "autonomous80"),
        escalated80_combined=cell(PRIOR_BEST, EXPL_BEST, "escalated80"),
        rows=rows,
    )
    save_json(summary, "e4_summary.json")

    if verbose:
        print("\n" + "-" * 72)
        print(out[["cell", "acc", "expl_ms", "fid1", "oov_gap"]].to_string(index=False))
        print("-" * 72)
        print(f"[E4] accuracy effect  explainer only : {d_expl*100:+.1f} pts")
        print(f"[E4] accuracy effect  prior only     : {d_prior*100:+.1f} pts")
        print(f"[E4] accuracy effect  both combined  : {combined*100:+.1f} pts")
        print(f"[E4] interaction (combined - sum)    : {interaction*100:+.1f} pts")
        print(f"[E4] OOV gap  {oov_base*100:.1f} -> "
              f"{summary['oov_combined_pts']:.1f} pts")
        print(f"[E4] explanation cost speedup        : "
              f"{summary['speedup_combined']:.0f}x")
    return out, summary


if __name__ == "__main__":
    run()
