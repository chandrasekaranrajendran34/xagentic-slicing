"""
Experiment 2 -- In-hoc (self-explaining) vs post-hoc explanation.

Research gap addressed
    E1 shows post-hoc SHAP costs ~3,100x the selection inference itself. That is
    tolerable for an admission-time decision on a human timescale but rules out
    near-real-time RIC control loops. The open question is whether explanations
    generated *during* inference (in-hoc) can match post-hoc faithfulness while
    removing the latency penalty.

Design
    Every explainer is evaluated on the identical corpus, split, seed and
    feature matrix as E1, so differences are attributable to the explanation
    mechanism alone.

    In-hoc (attribution emitted as part of the forward pass, zero extra cost)
      * Linear-Attribution  : logistic regression, a_j = w_jc * x_j
      * Decision-Path       : depth-8 tree, a_j = class-probability shift at
                              each split along the instance's decision path
      * SENN                : self-explaining neural network, logit_c =
                              sum_j Theta_jc(x) * x_j, a_j = Theta_jc(x) * x_j

    Post-hoc (attribution requires a second, separate computation)
      * SHAP                : TreeExplainer / PermutationExplainer on the GBM
      * LIME (optional)     : local surrogate, evaluated on a subsample

Metrics
    accuracy, macro F1, explanation latency, inference latency, latency ratio,
    top-k ablation faithfulness vs a random-feature control.

Outputs (results/)
    e2_explainers.csv, e2_summary.json
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from xagentic_common import (
    SEED, SLICES, SLICE_ID, FEATURES, OUT,
    synth_corpus, build_features, split,
    ablation_fidelity, subgroup_report, save_json, save_csv,
)


# ===========================================================================
# Self-Explaining Neural Network (pure NumPy, no deep-learning dependency)
# ===========================================================================

class SENN:
    """Self-explaining network with input-dependent, additive coefficients.

        h       = tanh(W1 x + b1)
        Theta   = reshape(W2 h + b2)          -> (n_features, n_classes)
        logit_c = sum_j Theta[j, c] * x_j + b0_c

    Because the logit is *by construction* an additive function of the inputs,
    the attribution a_j = Theta[j, c] * x_j is exact and is produced by the same
    forward pass that produces the prediction. There is no second computation,
    so explanation is effectively free -- which is the property under test.

    A stability penalty lambda * ||Theta||^2 discourages the coefficient network
    from varying wildly between neighbouring inputs.
    """

    def __init__(self, n_features, n_classes, hidden=32, lr=3e-3, epochs=300,
                 batch=128, lam=1e-4, seed=SEED):
        self.F, self.C, self.H = n_features, n_classes, hidden
        self.lr, self.epochs, self.batch, self.lam = lr, epochs, batch, lam
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_features), (hidden, n_features))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (n_features * n_classes, hidden))
        self.b2 = np.zeros(n_features * n_classes)
        self.b0 = np.zeros(n_classes)
        self._adam = {}
        self.scaler = StandardScaler()

    # -- internals ---------------------------------------------------------
    def _forward(self, Xs):
        h = np.tanh(Xs @ self.W1.T + self.b1)
        theta = (h @ self.W2.T + self.b2).reshape(-1, self.F, self.C)
        logits = np.einsum("nfc,nf->nc", theta, Xs) + self.b0
        return h, theta, logits

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _step(self, name, param, grad):
        st = self._adam.setdefault(name, {"m": np.zeros_like(param),
                                          "v": np.zeros_like(param), "t": 0})
        st["t"] += 1
        st["m"] = 0.9 * st["m"] + 0.1 * grad
        st["v"] = 0.999 * st["v"] + 0.001 * (grad ** 2)
        mhat = st["m"] / (1 - 0.9 ** st["t"])
        vhat = st["v"] / (1 - 0.999 ** st["t"])
        param -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)

    # -- public API --------------------------------------------------------
    def fit(self, X, y):
        Xs = self.scaler.fit_transform(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=int)
        Y = np.eye(self.C)[y]
        n = len(Xs)
        rng = np.random.default_rng(SEED)
        for _ in range(self.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, self.batch):
                b = idx[s:s + self.batch]
                xb, yb = Xs[b], Y[b]
                m = len(b)
                h, theta, logits = self._forward(xb)
                p = self._softmax(logits)

                dz = (p - yb) / m                                  # (m, C)
                dtheta = np.einsum("nc,nf->nfc", dz, xb)           # (m, F, C)
                dtheta += (2.0 * self.lam / m) * theta             # stability
                dflat = dtheta.reshape(m, -1)                      # (m, F*C)

                dW2 = dflat.T @ h
                db2 = dflat.sum(0)
                dh = dflat @ self.W2
                dpre = dh * (1 - h ** 2)
                dW1 = dpre.T @ xb
                db1 = dpre.sum(0)
                db0 = dz.sum(0)

                self._step("W1", self.W1, dW1)
                self._step("b1", self.b1, db1)
                self._step("W2", self.W2, dW2)
                self._step("b2", self.b2, db2)
                self._step("b0", self.b0, db0)
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        _, _, logits = self._forward(Xs)
        return self._softmax(logits)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_and_explain(self, X):
        """Single forward pass returning both the decision and its attribution."""
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        _, theta, logits = self._forward(Xs)
        p = self._softmax(logits)
        pred = p.argmax(axis=1)
        attr = np.stack([theta[i, :, pred[i]] * Xs[i] for i in range(len(Xs))])
        return pred, p, attr


# ===========================================================================
# In-hoc attribution for the classical models
# ===========================================================================

def linear_attribution(clf, scaler, X, pred):
    """a_j = w_jc * x_j for the predicted class -- exact for a linear logit."""
    Xs = scaler.transform(np.asarray(X, dtype=float))
    W = clf.coef_
    if W.shape[0] == 1:  # binary fallback
        W = np.vstack([-W[0], W[0]])
    return np.stack([W[pred[i]] * Xs[i] for i in range(len(Xs))])


def decision_path_attribution(tree, X, pred):
    """Attribute to each split feature the shift it causes in the predicted
    class probability along the instance's root-to-leaf decision path."""
    X = np.asarray(X, dtype=float)
    node_ind = tree.decision_path(X)
    feat = tree.tree_.feature
    value = tree.tree_.value  # (n_nodes, 1, n_classes)
    dist = value[:, 0, :] / np.clip(value[:, 0, :].sum(axis=1, keepdims=True), 1e-12, None)

    attr = np.zeros((len(X), X.shape[1]))
    indptr, indices = node_ind.indptr, node_ind.indices
    for i in range(len(X)):
        path = indices[indptr[i]:indptr[i + 1]]
        c = pred[i]
        for a, b in zip(path[:-1], path[1:]):
            f = feat[a]
            if f >= 0:
                attr[i, f] += dist[b, c] - dist[a, c]
    return attr


def _timed_explanation(fn, n):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) / n * 1000.0


def run(df=None, verbose=True):
    if verbose:
        print("=" * 72)
        print("EXPERIMENT 2 -- In-hoc vs post-hoc explanation")
        print("=" * 72)

    if df is None:
        df = synth_corpus()

    X = build_features(df)
    y = df["slice"].map(SLICE_ID).values
    Xtr, Xte, ytr, yte, dtr, dte = split(X, y, df)
    med = Xtr.median()
    rows, details = [], {}

    def record(label, kind, model_name, pred, proba, attr, expl_ms, infer_ms,
               predict_fn):
        fid = ablation_fidelity(predict_fn, Xte, attr, med)
        rows.append(dict(
            explainer=label, kind=kind, model=model_name,
            acc=float(accuracy_score(yte, pred)),
            f1=float(f1_score(yte, pred, average="macro")),
            infer_ms=infer_ms, expl_ms=expl_ms,
            ratio=expl_ms / max(infer_ms, 1e-12),
            top1=fid["top1"], rand1=fid["rand1"], fid1=fid["ratio1"],
            top3=fid["top3"], rand3=fid["rand3"], fid3=fid["ratio3"],
        ))
        details[label] = dict(fidelity=fid,
                              subgroup=subgroup_report(pred, yte, dte))
        print(f"  {label:<24} acc={rows[-1]['acc']:.4f} "
              f"expl={expl_ms:9.4f} ms  fidelity(top1/rand1)={fid['ratio1']:.1f}x")

    # ---------------- In-hoc 1: linear attribution --------------------------
    print("[E2] in-hoc: Linear-Attribution (logistic regression) ...")
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, random_state=SEED).fit(sc.transform(Xtr), ytr)
    lr_predict = lambda Z: lr.predict(sc.transform(np.asarray(Z, dtype=float)))
    t0 = time.perf_counter()
    p_lr = lr_predict(Xte)
    lr_infer = (time.perf_counter() - t0) / len(Xte) * 1000
    attr_lr, lr_expl = _timed_explanation(
        lambda: linear_attribution(lr, sc, Xte, p_lr), len(Xte))
    record("Linear-Attribution", "in-hoc", "Logistic Regression", p_lr,
           lr.predict_proba(sc.transform(Xte)), attr_lr, lr_expl, lr_infer, lr_predict)

    # ---------------- In-hoc 2: decision path -------------------------------
    print("[E2] in-hoc: Decision-Path (depth-8 tree) ...")
    dt = DecisionTreeClassifier(max_depth=8, random_state=SEED).fit(Xtr, ytr)
    t0 = time.perf_counter()
    p_dt = dt.predict(Xte)
    dt_infer = (time.perf_counter() - t0) / len(Xte) * 1000
    attr_dt, dt_expl = _timed_explanation(
        lambda: decision_path_attribution(dt, Xte, p_dt), len(Xte))
    record("Decision-Path", "in-hoc", "Decision Tree (d=8)", p_dt,
           dt.predict_proba(Xte), attr_dt, dt_expl, dt_infer, dt.predict)

    # ---------------- In-hoc 3: SENN ----------------------------------------
    print("[E2] in-hoc: SENN (self-explaining network) ...")
    senn = SENN(n_features=len(FEATURES), n_classes=3).fit(Xtr, ytr)
    t0 = time.perf_counter()
    _ = senn.predict(Xte)
    senn_infer = (time.perf_counter() - t0) / len(Xte) * 1000
    t0 = time.perf_counter()
    p_sn, pr_sn, attr_sn = senn.predict_and_explain(Xte)
    senn_joint = (time.perf_counter() - t0) / len(Xte) * 1000
    # Marginal cost of the explanation over the prediction it rides along with.
    senn_expl = max(senn_joint - senn_infer, 0.0)
    record("SENN", "in-hoc", "SENN (32 hidden)", p_sn, pr_sn, attr_sn,
           senn_expl, senn_infer, senn.predict)

    # ---------------- Post-hoc 1: SHAP --------------------------------------
    print("[E2] post-hoc: SHAP on gradient boosting ...")
    gb = GradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
    t0 = time.perf_counter()
    p_gb = gb.predict(Xte)
    gb_infer = (time.perf_counter() - t0) / len(Xte) * 1000

    import shap
    bg = shap.sample(Xtr, 100, random_state=SEED)

    def _shap():
        try:
            sv = shap.TreeExplainer(gb).shap_values(Xte)
        except Exception:
            sv = shap.PermutationExplainer(gb.predict_proba, bg, seed=SEED)(
                Xte, silent=True).values
        sv = np.array(sv)
        if sv.ndim == 3 and sv.shape[0] == len(Xte):
            sv = np.transpose(sv, (2, 0, 1))
        return np.stack([sv[p_gb[i]][i] for i in range(len(p_gb))])

    attr_shap, shap_expl = _timed_explanation(_shap, len(Xte))
    record("SHAP", "post-hoc", "Gradient Boosting", p_gb, gb.predict_proba(Xte),
           attr_shap, shap_expl, gb_infer, gb.predict)

    # ---------------- Post-hoc 2: LIME (optional) ---------------------------
    try:
        from lime.lime_tabular import LimeTabularExplainer
        n_lime = min(150, len(Xte))
        print(f"[E2] post-hoc: LIME on {n_lime} instances (subsampled; slow) ...")
        lex = LimeTabularExplainer(Xtr.to_numpy(dtype=float),
                                   feature_names=FEATURES,
                                   class_names=SLICES, discretize_continuous=True,
                                   random_state=SEED)
        sub = np.arange(n_lime)
        t0 = time.perf_counter()
        attr_lime = np.zeros((n_lime, len(FEATURES)))
        for i in sub:
            e = lex.explain_instance(Xte.to_numpy(dtype=float)[i], gb.predict_proba,
                                     num_features=len(FEATURES),
                                     labels=(int(p_gb[i]),))
            for j, w in e.as_map()[int(p_gb[i])]:
                attr_lime[i, j] = w
        lime_expl = (time.perf_counter() - t0) / n_lime * 1000
        fid_l = ablation_fidelity(gb.predict, Xte.iloc[sub], attr_lime, med)
        rows.append(dict(
            explainer="LIME", kind="post-hoc", model="Gradient Boosting",
            acc=float(accuracy_score(yte[sub], p_gb[sub])),
            f1=float(f1_score(yte[sub], p_gb[sub], average="macro")),
            infer_ms=gb_infer, expl_ms=lime_expl,
            ratio=lime_expl / max(gb_infer, 1e-12),
            top1=fid_l["top1"], rand1=fid_l["rand1"], fid1=fid_l["ratio1"],
            top3=fid_l["top3"], rand3=fid_l["rand3"], fid3=fid_l["ratio3"]))
        details["LIME"] = dict(fidelity=fid_l, n_evaluated=int(n_lime))
        print(f"  {'LIME':<24} expl={lime_expl:9.4f} ms  "
              f"fidelity(top1/rand1)={fid_l['ratio1']:.1f}x")
    except ImportError:
        print("[E2] LIME not installed -- skipping (pip install lime to enable)")

    res = pd.DataFrame(rows)
    save_csv(res, "e2_explainers.csv")
    print()
    print(res.to_string(index=False))

    posthoc = res[res["kind"] == "post-hoc"]
    inhoc = res[res["kind"] == "in-hoc"]
    best_inhoc = inhoc.sort_values("f1", ascending=False).iloc[0]
    shap_row = res[res["explainer"] == "SHAP"].iloc[0]

    summary = dict(
        experiment="E2", label="In-hoc vs post-hoc",
        table=res.to_dict(orient="records"), details=details,
        best_inhoc=str(best_inhoc["explainer"]),
        best_inhoc_acc=float(best_inhoc["acc"]),
        best_inhoc_expl_ms=float(best_inhoc["expl_ms"]),
        best_inhoc_fidelity=float(best_inhoc["fid1"]),
        shap_acc=float(shap_row["acc"]),
        shap_expl_ms=float(shap_row["expl_ms"]),
        shap_fidelity=float(shap_row["fid1"]),
        speedup=float(shap_row["expl_ms"] / max(best_inhoc["expl_ms"], 1e-12)),
        accuracy_cost=float(shap_row["acc"] - best_inhoc["acc"]),
        mean_inhoc_expl_ms=float(inhoc["expl_ms"].mean()),
        mean_posthoc_expl_ms=float(posthoc["expl_ms"].mean()),
    )
    save_json(summary, "e2_summary.json")
    print(f"\n[E2] best in-hoc = {summary['best_inhoc']}: "
          f"{summary['speedup']:.0f}x cheaper than SHAP, "
          f"accuracy cost {summary['accuracy_cost']*100:.1f} pts, "
          f"fidelity {summary['best_inhoc_fidelity']:.1f}x vs SHAP "
          f"{summary['shap_fidelity']:.1f}x")
    return summary


if __name__ == "__main__":
    run()
