"""
Experiment 1 -- Post-hoc explanation (SHAP) for agentic slice selection.

This is the baseline experiment: a four-agent pipeline (Intent, Candidate,
Selection, Validation) plus a cross-cutting Explanation Agent that attributes
every decision with SHAP *after* the fact.

Research question
    Can a composed agentic pipeline select 3GPP slice types accurately, and are
    post-hoc SHAP attributions faithful enough to drive an escalation policy?

Outputs (results/)
    e1_models.csv, e1_per_class.csv, e1_confusion.csv, e1_escalation.csv,
    e1_shap_global.csv, e1_local_explanations.json, e1_summary.json
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
)

from xagentic_common import (
    SEED, SLICES, SLICE_ID, FEATURES, PRETTY, OUT,
    synth_corpus, build_features, score_slots, split,
    contract_valid, rule_based, rule_based_lex, ablation_fidelity, escalation_table,
    subgroup_report, save_json, save_csv, select_tau,
    escalation_signal_ablation,
)


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("sc", StandardScaler()),
            ("m", LogisticRegression(max_iter=2000, random_state=SEED))]),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
        "MLP (64,32)": Pipeline([
            ("sc", StandardScaler()),
            ("m", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1200,
                                random_state=SEED))]),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED,
                                                n_jobs=-1),
    }


def run(df=None, verbose=True):
    if verbose:
        print("=" * 72)
        print("EXPERIMENT 1 -- Post-hoc explanation (SHAP)")
        print("=" * 72)

    if df is None:
        print("[S1] generating corpus ...")
        df = synth_corpus()
        save_csv(df, "corpus.csv")

    print("[S2] Intent Agent: slot filling + binary lexicon ...")
    slots = score_slots(df)
    X = build_features(df)
    y = df["slice"].map(SLICE_ID).values
    print(f"     mean slot recall {slots['mean_slot_accuracy']:.3f} "
          f"| false-positive {slots['mean_false_positive']:.3f} "
          f"| {slots['parse_latency_ms']:.3f} ms/intent")

    Xtr, Xte, ytr, yte, dtr, dte = split(X, y, df)

    # ---- S4: Selection Agent model sweep -----------------------------------
    rows = []
    t0 = time.perf_counter()
    rb = np.array([SLICE_ID[rule_based(r)] for _, r in Xte.iterrows()])
    rb_lat = (time.perf_counter() - t0) / len(Xte) * 1000
    rows.append(dict(
        model="Rule-based playbook", acc=accuracy_score(yte, rb),
        f1=f1_score(yte, rb, average="macro"), cv_mean=np.nan, cv_std=np.nan,
        valid=float(np.mean([contract_valid(SLICES[p], r)
                             for p, (_, r) in zip(rb, Xte.iterrows())])),
        lat_ms=rb_lat))

    # M9: lexicon-augmented playbook -- same 14-dim input as the learned models,
    # so the comparison against them is like-for-like.
    t0 = time.perf_counter()
    rbl = np.array([SLICE_ID[rule_based_lex(r)] for _, r in Xte.iterrows()])
    rbl_lat = (time.perf_counter() - t0) / len(Xte) * 1000
    rows.append(dict(
        model="Rule-based + lexicon", acc=accuracy_score(yte, rbl),
        f1=f1_score(yte, rbl, average="macro"), cv_mean=np.nan, cv_std=np.nan,
        valid=float(np.mean([contract_valid(SLICES[p], r)
                             for p, (_, r) in zip(rbl, Xte.iterrows())])),
        lat_ms=rbl_lat))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fitted = {}
    for name, m in build_models().items():
        print(f"[S4] training {name} ...")
        cv = cross_val_score(m, Xtr, ytr, cv=skf, scoring="f1_macro", n_jobs=1)
        m.fit(Xtr, ytr)
        fitted[name] = m
        t0 = time.perf_counter()
        p = m.predict(Xte)
        lat = (time.perf_counter() - t0) / len(Xte) * 1000
        rows.append(dict(
            model=name, acc=accuracy_score(yte, p),
            f1=f1_score(yte, p, average="macro"),
            cv_mean=cv.mean(), cv_std=cv.std(),
            valid=float(np.mean([contract_valid(SLICES[pi], r)
                                 for pi, (_, r) in zip(p, Xte.iterrows())])),
            lat_ms=lat))

    res = pd.DataFrame(rows)
    save_csv(res, "e1_models.csv")
    print(res.to_string(index=False))

    # Model selection uses ONLY the cross-validated score computed on the
    # training split. The held-out split is never consulted for selection.
    best_name = res[res["cv_mean"].notna()].sort_values(
        "cv_mean", ascending=False).iloc[0]["model"]
    best = fitted[best_name]
    ypred = best.predict(Xte)
    infer_lat_ms = float(res.loc[res["model"] == best_name, "lat_ms"].iloc[0])
    print(f"[S4] best = {best_name} (selected on 5-fold CV macro F1, train split)")

    # ---- S4b: escalation threshold selected on out-of-fold TRAIN predictions --
    # cross_val_predict gives every training intent a prediction from a model
    # that never saw it, so the sweep below consumes no held-out label.
    oof_proba = cross_val_predict(clone(best), Xtr, ytr, cv=skf,
                                  method="predict_proba", n_jobs=1)
    oof_pred = oof_proba.argmax(axis=1)
    oof_esc, _, _ = escalation_table(oof_proba, oof_pred, ytr, Xtr)
    save_csv(oof_esc, "e1_escalation_oof.csv")
    TAU = select_tau(oof_esc, min_autonomous_acc=0.98)
    print(f"[S4b] tau = {TAU:.2f} selected on out-of-fold train predictions")
    print(oof_esc.to_string(index=False))

    pr, rc, f1c, sup = precision_recall_fscore_support(yte, ypred, labels=[0, 1, 2])
    save_csv(dict(slice=SLICES, precision=pr, recall=rc, f1=f1c, support=sup),
             "e1_per_class.csv")

    cm = confusion_matrix(yte, ypred, labels=[0, 1, 2])
    np.savetxt(f"{OUT}/e1_confusion.csv", cm, fmt="%d", delimiter=",")

    # ---- S6: Explanation Agent (post-hoc SHAP) -----------------------------
    print("[S6] computing SHAP attributions ...")
    import shap
    bg = shap.sample(Xtr, 100, random_state=SEED)
    try:
        explainer = shap.TreeExplainer(best)
        t0 = time.perf_counter()
        sv = explainer.shap_values(Xte)
        backend = "TreeExplainer (exact)"
    except Exception as e:  # pragma: no cover - backend availability varies
        print(f"     TreeExplainer unavailable ({type(e).__name__}); "
              f"falling back to PermutationExplainer")
        explainer = shap.PermutationExplainer(best.predict_proba, bg, seed=SEED)
        t0 = time.perf_counter()
        sv = explainer(Xte, silent=True).values
        backend = "PermutationExplainer"
    shap_ms = (time.perf_counter() - t0) / len(Xte) * 1000

    sv = np.array(sv)
    if sv.ndim == 3 and sv.shape[0] == len(Xte):
        sv = np.transpose(sv, (2, 0, 1))  # -> (classes, samples, features)

    imp = np.abs(sv).mean(axis=(0, 1))
    imp = imp / imp.sum()
    gi = pd.DataFrame(dict(feature=[PRETTY[f] for f in FEATURES],
                           importance=imp)).sort_values("importance", ascending=False)
    save_csv(gi, "e1_shap_global.csv")
    print(gi.to_string(index=False))

    # Attribution for the *predicted* class of each instance.
    attr_pred = np.stack([sv[ypred[i]][i] for i in range(len(ypred))])
    fid = ablation_fidelity(best.predict, Xte, attr_pred, Xtr.median())
    fid["explanation_latency_ms"] = shap_ms
    fid["inference_latency_ms"] = infer_lat_ms
    fid["latency_ratio"] = shap_ms / max(infer_lat_ms, 1e-12)
    fid["backend"] = backend
    print("[S6] fidelity:", json.dumps(fid, indent=2, default=float))

    # ---- S5: escalation policy ---------------------------------------------
    # The held-out split is evaluated once, at the frozen tau from S4b.
    proba = best.predict_proba(Xte)
    esc, conf, wrong = escalation_table(proba, ypred, yte, Xte)
    save_csv(esc, "e1_escalation.csv")
    print(esc.to_string(index=False))
    _o = oof_esc.loc[np.isclose(oof_esc["tau"], TAU)].iloc[0]
    _t = esc.loc[np.isclose(esc["tau"], TAU)].iloc[0]
    print(f"[S5] tau={TAU:.2f}  OOF(selection) esc {100*_o['escalated']:.1f}% "
          f"rec {100*_o['recall']:.1f}% auto {100*_o['autonomous_acc']:.1f}%  |  "
          f"TEST esc {100*_t['escalated']:.1f}% rec {100*_t['recall']:.1f}% "
          f"auto {100*_t['autonomous_acc']:.1f}%")

    # ---- S5b: is the ESCALATION signal explanation-derived? (review item M7) --
    # All four policies spend the same operator budget, so differences in error
    # recall are attributable to the ranking signal and not to escalating more.
    sig = escalation_signal_ablation(proba, ypred, yte, Xte, attr_pred, TAU)
    save_csv(sig, "e1_escalation_signals.csv")
    print("[S5b] escalation signal ablation at matched budget:")
    print(sig.to_string(index=False))

    rep = subgroup_report(ypred, yte, dte)
    rep.update(dict(
        experiment="E1", label="Post-hoc (SHAP)", model=best_name,
        model_selection="5-fold CV macro F1 on train split",
        tau_selected=float(TAU),
        tau_selection="smallest tau with >=98% out-of-fold autonomous accuracy",
        escalation_oof=oof_esc.to_dict(orient="records"),
        escalation_signals=sig.to_dict(orient="records"),
        macro_f1=float(f1_score(yte, ypred, average="macro")),
        slot_recall=slots["mean_slot_accuracy"],
        slot_false_positive=slots["mean_false_positive"],
        parse_latency_ms=slots["parse_latency_ms"],
        mean_confidence_correct=float(conf[~wrong].mean()),
        mean_confidence_wrong=float(conf[wrong].mean()),
        fidelity=fid,
        baseline_rule_acc=float(accuracy_score(yte, rb)),
        baseline_rule_f1=float(f1_score(yte, rb, average="macro")),
        baseline_rulelex_acc=float(accuracy_score(yte, rbl)),
        baseline_rulelex_f1=float(f1_score(yte, rbl, average="macro")),
        baseline_rulelex_valid=float(np.mean([contract_valid(SLICES[p], r)
                                              for p, (_, r) in zip(rbl, Xte.iterrows())])),
        baseline_rulelex_lat_ms=float(rbl_lat),
        escalation=esc.to_dict(orient="records"),
    ))

    local = []
    for c in range(3):
        idx_all = np.where((yte == c) & (ypred == c))[0]
        if not len(idx_all):
            continue
        idx = int(idx_all[0])
        contrib = sv[c][idx]
        o = np.argsort(-np.abs(contrib))[:4]
        local.append(dict(slice=SLICES[c], intent=dte.iloc[idx]["text"],
                          top=[(PRETTY[FEATURES[j]], float(contrib[j])) for j in o]))
    save_json(local, "e1_local_explanations.json")

    save_json(rep, "e1_summary.json")
    print(f"[E1] accuracy {rep['acc_overall']:.4f} | macro F1 {rep['macro_f1']:.4f} "
          f"| OOV gap {rep['oov_gap']*100:.1f} pts "
          f"| explanation {shap_ms:.2f} ms/intent")
    return rep


if __name__ == "__main__":
    run()
