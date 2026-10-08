# XAgentic-Slicing

**Explainable and Trustworthy Agentic AI for 5G/6G Network-Slicing Intent Interpretation**

A reproducible research project evaluating three-class network-slicing service recommendations (**eMBB**, **URLLC**, **mMTC**), explainability, lexical generalization, KPI feasibility, and selective autonomy.

> **Scope:** This is a controlled, synthetic-data research prototype. It predicts a requested service type; it does **not** perform real-network admission, resource reservation, slice deployment, or end-to-end orchestration. The tool-using LLM agent is evaluated separately in E5.

## Research roadmap

The repository has two phases:

1. **Original experiments (E1–E5):** Classification and post-hoc explanations; alternative explanation methods; lexical similarity prior; factorial combinations; and a tool-using LLM experiment.
2. **Reviewer-driven validation (R1–R5):** Phrase-held-out testing, strict five-KPI feasibility, candidate masking, uncertainty-based escalation, and semantic verification.

R1–R5 are **additional validation experiments**, not simple repetitions of E1–E5. Original and revised evidence are retained separately. The main research question is: *Can an intent interpretation pipeline make constraint-valid decisions and reduce incorrect autonomous actions, while transparently accounting for the cost of escalation?*

## Repository structure

```text
xagentic-slicing/
├── README.md
├── README_FIRST.md
├── LICENSE
├── CITATION.cff
├── NOTEBOOK_INVENTORY.csv
├── CLEAN_PACKAGE_SHA256SUMS.txt
├── 01_NOTEBOOKS/
│   └── REQUIRED_CHECKIN/
│       ├── E1_E4_Final_5Seed_Experiments.ipynb
│       └── E5_Final_Qwen37_N100_and_Final_Audit.ipynb
├── 02_RESULTS/
│   ├── FINAL/
│   │   ├── E1_E4_5SEEDS/
│   │   └── E5_QWEN37_N100_SEED52/
│   └── SUPPORTING/E5_MODEL_SCREENING/
├── 03_SOURCE_CODE/
│   ├── agents.py
│   ├── xagentic_common.py
│   ├── xagentic_slice.py
│   ├── exp1_posthoc.py
│   ├── exp2_inhoc.py
│   ├── exp3_semantic.py
│   ├── exp4_combined.py
│   ├── exp5_agentic.py
│   ├── run_all.py
│   └── run_seeds.py
├── 04_REPRODUCIBILITY/
│   ├── README.md
│   ├── requirements.txt
│   ├── FINAL_EVIDENCE_MANIFEST.txt
│   └── SHA256SUMS.txt
├── 05_VALIDATION_EXPERIMENTS/
│   ├── All/               # Consolidated manuscript and reviewer documents
│   ├── R1/                # Phrase-family holdout and KPI-only controls
│   ├── R2/                # Strict feasibility audit
│   ├── R3/                # Candidate masking
│   ├── R4/                # Uncertainty-based escalation
│   └── R5/                # Semantic verification
└── 06_XAGENTIC_SLICING_PAPER/  # Reserved for paper sources
```

This tree reflects the supplied repository inventory. Some validation directories contain recovered upstream data and archived evidence for auditability; these are **not independent new experiment runs**.

## Original experiments: E1–E5

| ID | Research focus | Where to find results |
|---|---|---|
| E1 | Classifier selection, post-hoc explanations, and decision behavior | `02_RESULTS/FINAL/E1_E4_5SEEDS/e1_models.csv`, `e1_summary.json` |
| E2 | In-model/alternative explainers and explanation timing | `e2_explainers.csv`, `e2_summary.json`, `fig_e2_latency.pdf` |
| E3 | Character n-gram TF-IDF lexical similarity prior and phrase-group evaluation | `e3_variants.csv`, `e3_summary.json`, `fig_e3_oov.pdf` |
| E4 | Factorial interactions of prior and explainer choices | `e4_factorial.csv`, `e4_seeds.csv`, `fig_e4_combined.pdf` |
| E5 | Tool-using Qwen3.7-Max agent on 100 requests | `02_RESULTS/FINAL/E5_QWEN37_N100_SEED52/e5_decisions.csv`, `e5_summary.json` |

The **E1–E4 final five-seed study** uses seeds **42–46**; some component analyses are single-seed and should be identified as such. The final **E5 study** uses **100 requests, seed 52**. The prior technical review reported **87% accuracy for the agent versus 93% for the paired deterministic comparator**; the agent should not be described as an accuracy improvement. Supporting model-screening runs are separate from the final E5 experiment.

The original generator used label-conditioned phrase inventories. Its *unseen* phrases are not equivalent to wholly novel semantic intents. Likewise, the original feasibility score represented a permissive implemented screen rather than full five-KPI feasibility.

## Reviewer-driven validation: R1–R5

| Stage | Question | Data / setup | Frozen finding |
|---|---|---|---|
| **R1** | Does the classifier memorize class-specific wording? | Five seeds; 3,000 synthetic requests per seed; four text interventions | L1 macro-F1 **0.9311 ± 0.0065**; KPI-only **0.8920 ± 0.0078** |
| **R2** | Is the original feasibility check too permissive? | Frozen 1,200-request corpus; separate 100-decision E5 audit | **229/1,200 (19.08%)** candidate-set disagreements; **3/100** E5 false-feasible selections |
| **R3** | Can strict masking prevent infeasible autonomous actions? | 840 development / 360 held-out | **341** autonomous; **12** wrong; **0** strictly infeasible autonomous selections |
| **R4** | Can uncertainty-aware escalation capture residual errors? | Same frozen 360 held-out | **306** autonomous; **5** wrong |
| **R5** | Can semantic verification catch confident residual errors? | Same frozen 360 held-out | **280** autonomous; **2** wrong |

### R1 — Phrase-family holdout and lexical ablation

- **L1 Original:** Original text and frozen feature pipeline.
- **L2 Phrase-family held-out:** Entire class-associated phrase families excluded from training; generic cases excluded from this condition.
- **L3 Generic + KPI:** Replace use-case phrase with neutral wording while retaining observable KPIs.
- **L4 KPI-only:** Keep only KPI information visible to the frozen parser.

| Condition | Accuracy (mean ± SD) | Macro-F1 (mean ± SD) |
|---|---:|---:|
| L1 Original | 0.9311 ± 0.0065 | 0.9311 ± 0.0065 |
| L2 Phrase-family held-out | 0.9549 ± 0.0081 | 0.9548 ± 0.0081 |
| L3 Generic + KPI | 0.8920 ± 0.0080 | 0.8920 ± 0.0078 |
| L4 KPI-only | 0.8920 ± 0.0080 | 0.8920 ± 0.0078 |

Removing lexical/use-case cues lowered macro-F1 by **0.0392 ± 0.0089** across five seeds. **Do not call L2 an improvement over L1**: its evaluation population differs. L3 and L4 map to **identical frozen downstream feature representations**, so their identical results are not independent replications.

Entry point: `05_VALIDATION_EXPERIMENTS/R1/R1_Phrase_Heldout_KPI_Validation.ipynb`. Final evidence: `R1/R1_phrase_generalization/final_evidence/` and `results/aggregate/`.

### R2 — Strict five-KPI feasibility audit

The authoritative **Strict V2** oracle checks **latency, throughput, reliability, device density, and mobility**. Unspecified KPIs are unconstrained, and an empty feasible set stays empty rather than reverting to the entire catalogue.

- **229/1,200 (19.08%)** requests had a legacy-versus-Strict-V2 candidate-set disagreement; all were legacy over-acceptances.
- **14/1,200 (1.17%)** had no strictly feasible catalogue slice.
- **3/100** frozen E5 selected decisions were false-feasible under Strict V2; **2/100** were autonomously accepted.
- The old **100% contract validity** claim applies only to the legacy implemented screen, **not** complete strict KPI compliance.

Entry point: `05_VALIDATION_EXPERIMENTS/R2/XAgentic_R2_Strict_Feasibility_Validity_Audit.ipynb`. Authoritative documentation: `R2/XAgentic_R2_Strict_Feasibility_FINAL/06_DOCUMENTATION/`. The earlier V1 oracle is archived as **superseded**.

### R3–R5 — Frozen selective-autonomy comparison

All three systems below were evaluated on the **same 360-request held-out set**, with development-only calibration and no held-out threshold retuning.

| Metric | R3 | R4 | R5 |
|---|---:|---:|---:|
| Autonomous requests | 341 | 306 | 280 |
| Escalated requests | 19 | 54 | 80 |
| Correct autonomous | 329 | 301 | 278 |
| Incorrect autonomous | 12 | 5 | 2 |
| Coverage | 94.72% | 85.00% | 77.78% |
| Accuracy among autonomous | 96.48% | 98.37% | 99.29% |
| Selective risk among autonomous | 3.52% | 1.63% | 0.71% |
| Correct autonomous / all 360 | 91.39% | 83.61% | 77.22% |

**Definitions:** Coverage = autonomous / total; autonomous accuracy = correct autonomous / autonomous; selective risk = incorrect autonomous / autonomous. **Escalation is deferral, not a correct classification.** Thus **99.29%** is R5 accuracy *among the 280 autonomous decisions*, **not** accuracy over all 360 requests.

**R3:** Apply Strict V2 feasibility masking; escalate empty feasible sets or low feasible-probability-mass cases. Frozen development-calibrated threshold: `0.5486315987262682`. The final policy had **zero strictly infeasible autonomous selections**. An unconditional hard mask could force low-support substitutes; R3's calibrated escalation avoids that failure mode.

**R4:** Retain R3 escalations, and additionally escalate low top-two probability-margin decisions. Frozen margin threshold: `0.7097637192002952`. R4 captured **7 of 12** residual R3 errors but also escalated **28 correct** R3 decisions.

**R5:** Retain R4 escalations and check its remaining autonomous decisions using development-fitted TF-IDF class prototypes. Frozen semantic-support threshold: `-0.0275677986595984`. R5 captured **3 of 5** R4 residual errors while escalating **23 correct** R4 decisions, mainly generic intents. The observed R4-to-R5 relative selective-risk reduction was **56.29%**, with a **7.22 percentage-point coverage cost**.

**Uncertainty:** A paired 10,000-resample bootstrap for the R4-to-R5 absolute risk reduction gave a 95% interval of approximately **−0.053 to +2.184 percentage points**. It includes zero; **do not claim statistically conclusive superiority** from these five residual errors.

### Validation entry points and evidence

| Stage | Primary notebook or frozen documentation |
|---|---|
| R1 | `R1/R1_Phrase_Heldout_KPI_Validation.ipynb` |
| R2 | `R2/XAgentic_R2_Strict_Feasibility_Validity_Audit.ipynb` |
| R3 | `R3/XAgentic_R3_Candidate_Masking_FINAL/05_DOCUMENTATION/R3_FINAL_INTERPRETATION.md` |
| R4 | `R4/XAgentic_R4_Selective_Escalation_plan.ipynb` |
| R5 | `R5/XAgentic_R5_High_Confidence_Error_Detection_1.ipynb` and `_2.ipynb` |

All paths in this table are relative to `05_VALIDATION_EXPERIMENTS/`. **The supplied tree does not show a dedicated R3 top-level rerun notebook**. R3's frozen outputs and documentation are present, but a complete one-click rerun should not be claimed until its executable reconstruction is independently verified. R5's `R5_work/` includes recovered R3/R4 inputs; these are provenance copies, not extra test sets.

## Setup and execution

### Clone and install

```bash
git clone <YOUR-REPOSITORY-URL>
cd xagentic-slicing
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r 04_REPRODUCIBILITY/requirements.txt
```

Replace `<YOUR-REPOSITORY-URL>` with your actual repository address. For Google Colab, open the notebooks in `01_NOTEBOOKS/REQUIRED_CHECKIN/` and follow their setup cells. E5 may require network access and a **separately configured API key** for the remote model.

### Reproduce the original experiments

1. Read `README_FIRST.md`, `04_REPRODUCIBILITY/README.md`, and `NOTEBOOK_INVENTORY.csv`.
2. Execute `01_NOTEBOOKS/REQUIRED_CHECKIN/E1_E4_Final_5Seed_Experiments.ipynb` for E1–E4.
3. Execute `01_NOTEBOOKS/REQUIRED_CHECKIN/E5_Final_Qwen37_N100_and_Final_Audit.ipynb` for E5, following its API configuration and run instructions.
4. Compare outputs with `02_RESULTS/FINAL/`, preserving the originals.

### Reproduce or audit R1–R5

1. Read the stage's notebook, findings and manifest **before running**.
2. Use the frozen data, split and policy files identified by that stage; do not substitute regenerated corpora or retune on held-out data.
3. Write new outputs to a **new rerun directory**, never over the frozen evidence.
4. Record Git commit, Python/package versions, seeds, input hashes, and result checksums.
5. Compare counts, metrics, and decision-level evidence with the stage's frozen manifest.

There is **no verified single command** in the supplied inventory that reproduces E1–E5 and R1–R5 end to end. `run_all.py` is part of the original source tree and should not be assumed to execute every validation stage.

### Verify integrity

```bash
find 02_RESULTS/FINAL -type f | sort
find 05_VALIDATION_EXPERIMENTS -type f -iname '*manifest*' | sort
```

For a manifest **in standard sha256sum format**, run `sha256sum -c` from the appropriate base directory. R-stage `*_SHA256_MANIFEST.csv` files may require their own CSV-based verifier; do **not** feed those directly to `sha256sum -c` without checking format. Moving archived files can invalidate path-based verification even when bytes are unchanged.

## Reproducibility and research-integrity policy

- **Frozen means frozen:** Do not edit or silently replace archived evidence, calibrated thresholds, source snapshots or final test decisions.
- **No test-set tuning:** R3/R4/R5 thresholds were selected using development/out-of-fold data. The 360 held-out cases must not be used to tune a new model and then represented as independent evaluation.
- **Separate populations:** E1–E4, E5, R1, R2 and R3–R5 do not all share the same sample design; avoid pooled accuracy claims.
- **Preserve negative results:** Report E5's paired accuracy disadvantage, R1's lexical reliance, and R5's false escalations alongside positive results.
- **Inspect before publication:** Check `llm_cache.json`, logs, notebooks and output files for credentials, tokens or sensitive information. Do not commit `.env` secrets.
- **Avoid large Git history:** Keep bulky immutable ZIP archives in Git LFS or release assets where appropriate; document their SHA256 checksums.
- **Use reproducible environments:** Remote LLM behavior and runtime measurements may vary with provider, model revision, hardware and API conditions.

## Technical-review response

The earlier **ICIITCEE 2027 technical peer review** recommended major revisions to the original E1–E5 paper. The reviewer-driven experiments address the following concerns:

| Review concern | Validation response |
|---|---|
| Class-conditioned vocabulary and overstated OOV claims | **R1:** Phrase-family holdout, generic+KPI, KPI-only controls |
| Permissive feasibility and empty-set fallback | **R2:** Strict five-KPI V2 oracle and decision audit |
| Candidate screen not enforced in actual selection | **R3:** Hard feasible-set masking and calibrated escalation |
| Residual decision errors | **R4:** Uncertainty-aware selective escalation |
| High-confidence wrong predictions | **R5:** Complementary semantic verification |

**Remaining manuscript obligations:** The original median-feature perturbation results measure **sensitivity**, not verified causal explanation faithfulness; E5's critic/retry claims must match observed execution; five-seed statistical intervals must be described as exploratory where appropriate; and remote LLM timings must not be equated with local batch inference latency. R1–R5 do **not** establish deployment readiness.

The consolidated documents and reviewer plan are in `05_VALIDATION_EXPERIMENTS/All/`. The supplied inventory shows `06_XAGENTIC_SLICING_PAPER/` as currently empty; add manuscript source files there when ready.

## Limitations and future work

The experiments use synthetic requests and a three-service catalogue. Labels are generator-defined, sometimes conflicting with strict KPI feasibility. Phrase holdout does not establish unrestricted natural-language generalization. The autonomous reliability results are conditional on **abstaining** for some requests, and the eventual correctness of escalated cases is not measured. R5 evaluates only five residual R4 errors, so confidence intervals are wide. The project does not measure live network admission, operator interventions, resource allocation, or production-scale SLA compliance.

Future directions include independent out-of-distribution/operator data, category-aware verification that reduces false generic escalations, larger independent error-detection evaluations, group-consistent explanation tests, and integration with a real policy-controlled slice manager. **R6 is deferred** and is not part of the frozen R1–R5 experimental scope.

## Citation and license

See [`CITATION.cff`](CITATION.cff) for citation metadata and [`LICENSE`](LICENSE) for license terms. Do not infer publication acceptance from the presence of a manuscript or citation file.

---

**Research principle:** A trustworthy autonomous network-slicing recommendation must be more than a confident label: it should satisfy the represented KPI constraints, survive appropriate checks, and allow escalation when the evidence is insufficient.
