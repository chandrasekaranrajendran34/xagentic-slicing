# R1 — Phrase Generalization and Lexical Ablation

## Status

**PASS — CLOSED AND FROZEN**

R1 must not be rerun, modified, or overwritten unless a deliberate
new experiment version is created.

---

## Purpose

R1 addresses the reviewer concern that classification performance
may be caused primarily by memorization of synthetic class-specific
phrases or lexical cues.

The experiment evaluates phrase-family generalization and lexical
ablation while preserving the frozen implementation used for the
revision experiments.

---

## Seeds

Five independent corpus seeds were evaluated:

- 42
- 43
- 44
- 45
- 46

Total evaluated condition runs:

**5 seeds × 4 conditions = 20 runs**

---

## Experimental Conditions

### L1 — Original

Original synthetic intent text evaluated using the frozen feature
pipeline and training-only model selection.

### L2 — Phrase-family-held-out

Strict class-specific phrase/use-case family holdout.

Train and test phrase families are disjoint.

Generic use cases are excluded from the main L2 evaluation.

This condition directly tests whether the classifier collapses when
evaluated on phrase families that were absent from training.

### L3 — Generic + KPI

Class-specific use-case phrases are replaced with the neutral phrase:

`network service`

Observable KPI information is preserved.

### L4 — KPI-only

Only KPI information observable to the frozen parser is retained.

No hidden latent KPI values are introduced.

---

## Final Five-Seed Results

| Condition | Accuracy mean ± SD | Macro-F1 mean ± SD |
|---|---:|---:|
| L1 Original | 0.9311 ± 0.0065 | 0.9311 ± 0.0065 |
| L2 Phrase-family-held-out | 0.9549 ± 0.0081 | 0.9548 ± 0.0081 |
| L3 Generic + KPI | 0.8920 ± 0.0080 | 0.8920 ± 0.0078 |
| L4 KPI-only | 0.8920 ± 0.0080 | 0.8920 ± 0.0078 |

---

## Lexical Ablation Result

Mean L1 -> L3/L4 accuracy reduction:

**0.0391 ± 0.0090**

Mean L1 -> L3/L4 macro-F1 reduction:

**0.0392 ± 0.0089**

This corresponds to approximately:

**3.92 percentage points macro-F1**

---

## Main Scientific Finding

Performance does not collapse when class-specific phrase families
are held out from training.

Removing lexical/use-case information causes a consistent performance
reduction across all five seeds.

Therefore:

**Lexical intent information contributes to classification, but the
classifier is not solely dependent on memorization of class-specific
phrases.**

---

## Important Interpretation Restrictions

### L2

Do NOT claim:

> L2 improves performance over L1.

L2 and L1 use different evaluation populations because generic
examples are excluded from L2.

L2 should instead be interpreted as evidence that held-out phrase
families do not cause performance collapse.

### L3 and L4

L3 and L4 are different text interventions but produce identical
frozen model feature representations.

Their identical results are therefore an architectural /
representation finding.

They must NOT be presented as two independent model-level pieces
of evidence.

### Ambiguous intents

Synthetic ground-truth slice labels are assigned before KPI blending.

Therefore, KPI-only disagreement for ambiguous samples should not
automatically be interpreted as conventional classification error.

---

## Model Selection

Model selection was performed using training data only.

The held-out test set was not used for model selection.

Across seeds, model selection was not artificially fixed.

Gradient Boosting was selected most frequently, while Decision Tree
or Random Forest was selected for some seed/condition combinations.

---

## Reproducibility Note

During R1 preparation, a discrepancy was identified between an older
archived E1 result and reproduction using the currently packaged
source.

Direct execution of the current packaged source reproduced the new
R1 baseline exactly.

Therefore:

**The current frozen source snapshot is the authoritative
implementation for R1.**

The historical archived metric was preserved as historical evidence
and was NOT forced into the R1 results.

---

## Evidence Structure

Important directories include:

- `data/`
- `splits/`
- `results/`
- `audit/`
- `manifests/`
- `frozen_reference/`
- `final_evidence/`

Final aggregate evidence:

`results/aggregate/`

Final frozen evidence:

`final_evidence/`

---

## Key Final Files

### Aggregate

- `r1_all_20_runs.csv`
- `r1_five_seed_summary.csv`
- `r1_model_selection_counts.csv`
- `r1_L1_vs_L3L4_paired.csv`
- `r1_L1_vs_L3L4_summary.csv`
- `r1_L2_descriptive_comparison.csv`
- `r1_L3_L4_equivalence.csv`

### Final evidence

- `R1_publication_table.csv`
- `R1_final_findings.txt`
- `R1_closure.json`
- `R1_SHA256_MANIFEST.csv`

---

## Final R1 Status

**R1 STATUS: PASS — CLOSED AND FROZEN**

Do not modify R1 evidence while conducting R2 or later experiments.

Create separate experiment directories for subsequent reviewer
experiments.

---

## Next Planned Experiment

R2 — Strict Feasibility / Validity Audit

Primary objective:

Evaluate whether the current feasibility/validity mechanism is too
permissive and quantify false-feasible decisions under stricter
network-slicing constraints.

R2 should be started as a separate experiment and should not modify
R1.
