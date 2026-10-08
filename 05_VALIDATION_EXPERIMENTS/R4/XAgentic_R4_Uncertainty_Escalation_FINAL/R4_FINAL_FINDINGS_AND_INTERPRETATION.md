# R4 — Uncertainty-Aware Selective Escalation

## 1. Objective

R4 evaluates whether uncertainty information from the frozen R3 selection
pipeline can be used to selectively escalate unreliable autonomous decisions.

The research question is:

> Can uncertainty-aware selective escalation reduce residual autonomous errors
> while preserving a useful level of autonomous coverage?

R4 is evaluated as an extension of the frozen R3 candidate-masking system.

---

## 2. Experimental Integrity

R4 was designed to preserve strict separation between development/calibration
data and held-out evaluation data.

- Development/calibration observations: 840
- Frozen held-out observations: 360
- Frozen R3 feasible-mass threshold: 0.5486315987262682
- R4 uncertainty signal: top-1 minus top-2 probability margin
- Lower margin indicates greater uncertainty.
- R4 threshold selection was performed using development/OOF evidence only.
- Held-out labels were not used for threshold selection.
- The R4 policy was frozen before held-out evaluation.
- No post-hoc threshold modification was performed after observing held-out results.

Original frozen R4 policy SHA256:

1b2d69794b1508f807227b12e78fdabce857683731cb91ee1879f7069029db42

Original frozen held-out decision SHA256:

8f7aab8ed6cdf66a5616d750f070ea02614c0fd1f3e0bf043f36d07f7527611a

Original frozen held-out summary SHA256:

9c4a1c98cf2d17da30045f60a5021b64bdd6deb410f3d7a310b207dff19ad412

---

## 3. Development-Set Uncertainty Analysis

The R4 development analysis examined several uncertainty signals:

1. 1 - top-1 probability
2. Predictive entropy
3. 1 - top-1/top-2 probability margin
4. 1 - feasible probability mass

Their development error-detection AUC values were:

| Signal | Error-detection AUC |
|---|---:|
| 1 - top-1 | 0.903454 |
| Entropy | 0.902956 |
| 1 - margin | 0.902169 |
| 1 - feasible mass | 0.765217 |

This showed that confidence-, entropy-, and margin-based uncertainty signals
were substantially more informative for detecting residual errors than feasible
probability mass alone.

At approximately 10% additional escalation:

- top-1 captured 35 of 62 development errors;
- margin captured 36 of 62;
- entropy captured 36 of 62;
- feasible mass captured 28 of 62.

---

## 4. Frozen R4 Operating Point

The selected R4 policy uses the top-1/top-2 probability margin.

The frozen margin threshold is:

0.7097637192002952

The rule is:

1. Apply the already-frozen R3 policy.
2. If R3 escalates, retain the escalation.
3. If R3 would automate, calculate the top-1 minus top-2 probability margin.
4. If margin < 0.7097637192002952, escalate the decision.
5. Otherwise retain autonomous execution.

The threshold was frozen from OOF development evidence before evaluating the
360-row held-out set.

Development performance at this operating point:

- Total development observations: 840
- Combined escalations: 136
- Autonomous observations: 704
- Coverage: 0.838095
- Autonomous errors: 13
- Autonomous accuracy: 0.981534

---

## 5. One-Shot Held-Out Evaluation

The frozen policy was applied once to the 360-row held-out evaluation set.

### Frozen R3 baseline

- Total observations: 360
- Escalated: 19
- Autonomous: 341
- Coverage: 0.947222
- Escalation rate: 0.052778
- Autonomous correct: 329
- Autonomous errors: 12
- Autonomous accuracy: 0.964809
- Selective risk: 0.035191
- Correct autonomous / total: 0.913889

### Frozen R4

- Total observations: 360
- Additional uncertainty escalations: 35
- Total escalated: 54
- Autonomous: 306
- Coverage: 0.850000
- Escalation rate: 0.150000
- Autonomous correct: 301
- Autonomous errors: 5
- Autonomous accuracy: 0.983660
- Selective risk: 0.016340
- Correct autonomous / total: 0.836111

---

## 6. R4 Improvement Over R3

R4 reduced the residual autonomous errors from:

12 -> 5

Therefore, 7 of the 12 residual R3 errors were successfully intercepted by
uncertainty-aware escalation.

Residual error capture:

7 / 12 = 58.33%

Autonomous accuracy improved from:

96.48% -> 98.37%

Selective risk decreased from:

3.52% -> 1.63%

Relative selective-risk reduction:

53.57%

However, this reliability improvement required a reduction in autonomous
coverage:

94.72% -> 85.00%

Coverage change:

-9.72 percentage points

Therefore, R4 should be interpreted as improving the safety/reliability of
autonomous operation rather than increasing autonomous throughput.

---

## 7. Escalation Efficiency

R4 added 35 uncertainty-driven escalations.

Among these:

- Residual R3 errors captured: 7
- Correct R3 decisions also escalated: 28

Added escalation precision:

7 / 35 = 20%

This demonstrates an important trade-off.

The uncertainty mechanism successfully removes more than half of the remaining
R3 errors, but it also escalates some correct cases.

Thus R4 implements conservative selective autonomy rather than perfect
error identification.

---

## 8. Error Capture by Generalization Category

Residual R3 errors consisted of:

### Generic cases

- R3 errors: 5
- Captured by R4: 4
- Missed by R4: 1
- Capture rate: 80%

### Unseen cases

- R3 errors: 7
- Captured by R4: 3
- Missed by R4: 4
- Capture rate: 42.86%

The uncertainty mechanism is therefore substantially more successful on generic
residual errors than on unseen residual errors.

---

## 9. High-Confidence Error Limitation

The R4 audit identified an important limitation.

Some residual errors were made with high model confidence and large
top-1/top-2 probability margins.

Captured errors had substantially lower margins, while the five missed errors
had high margins.

Approximate mean profiles observed during the completed R4 audit were:

### Captured residual errors

- n = 7
- mean top-1 confidence = 0.646939
- mean margin = 0.360555
- mean normalized entropy = 0.703867
- mean feasible mass = 0.957960

### Missed residual errors

- n = 5
- mean top-1 confidence = 0.945313
- mean margin = 0.908073
- mean normalized entropy = 0.209493
- mean feasible mass = 0.991204

This means conventional uncertainty measures cannot detect every model error.

In particular, confidently wrong predictions remain a significant challenge.

This limitation must not be addressed by post-hoc threshold tuning on the
held-out set because doing so would compromise evaluation integrity.

Instead, high-confidence errors motivate future work involving richer
distribution-shift detection, semantic consistency checks, ensemble
disagreement, calibrated out-of-distribution detection, or additional
verification mechanisms.

---

## 10. Main Finding

The central R4 finding is:

> A frozen uncertainty-aware selective escalation layer substantially improves
> the reliability of the R3 autonomous network-slicing decision pipeline,
> increasing autonomous accuracy from 96.48% to 98.37% and reducing selective
> risk by approximately 53.57%, while reducing autonomous coverage from
> 94.72% to 85.00%.

This represents a deliberate reliability-versus-coverage trade-off.

---

## 11. Research Interpretation

R3 addressed feasibility-aware candidate masking.

R4 adds a second trust mechanism:

- R3 asks whether the candidate decision is feasible.
- R4 additionally asks whether the model is sufficiently certain about the
  autonomous decision.

The resulting architecture therefore separates:

1. feasibility-based safety filtering; and
2. uncertainty-based selective autonomy.

This provides a stronger trust architecture than either mechanism alone.

---

## 12. Limitations

The following limitations should be explicitly reported.

1. R4 improves reliability at the cost of autonomous coverage.
2. Only 20% of additional R4 escalations corresponded to residual R3 errors.
3. Some correct decisions were conservatively escalated.
4. High-confidence incorrect predictions cannot reliably be detected using
   probability margin alone.
5. Unseen-case error capture (42.86%) was lower than generic-case error
   capture (80%).
6. The current experiment evaluates selective escalation rather than a complete
   production-scale distribution-shift detection framework.
7. Additional robustness testing should be treated as separate future
   experimentation rather than post-hoc modification of the frozen R4 result.

---

## 13. Runtime-Recovery Provenance

After completion of the R4 experiment, the Google Colab runtime reset and
removed the temporary /content workspace.

The experiment itself was not rerun.

The frozen R3 evidence package was subsequently restored from:

XAgentic_R3_Candidate_Masking_FINAL_2026-10-07.zip

with SHA256:

ba0f680c372ddda498eba2f8b84f5df9a599dc04810a679b53ce77af6fce0152

All 29 entries in the internal R3 SHA256 manifest were successfully verified.

The recovered evidence contained:

- 840 OOF development/calibration observations;
- 360 frozen held-out observations; and
- 360 frozen R3 held-out decisions.

The already-frozen R4 margin threshold was then reapplied to the SHA-verified
R3 held-out evidence.

The reconstruction reproduced the completed R4 counts exactly:

- R3 autonomous: 341
- R3 errors: 12
- R4 added escalations: 35
- R4 errors captured: 7
- R4 autonomous: 306
- R4 autonomous correct: 301
- R4 autonomous errors: 5

No R4 threshold was retuned and no held-out-driven policy modification was
performed during recovery.

Recovered files are explicitly named R4_RECOVERED_* to distinguish them from
the original byte-level artifacts.

---

## 14. Conclusion

R4 demonstrates that uncertainty-aware selective escalation can materially
improve the reliability of an already feasibility-aware autonomous
network-slicing pipeline.

Compared with frozen R3, the R4 mechanism reduced autonomous errors from 12 to
5 and reduced selective risk by approximately 53.57%.

The improvement comes with a measurable coverage cost: autonomous coverage
decreased from 94.72% to 85.00%.

Consequently, R4 supports the use of selective autonomy for trustworthy
agentic network-slicing systems where reliability is prioritized over maximum
automation.

The remaining high-confidence errors provide a clear motivation for subsequent
experiments focused on stronger verification, robustness, and detection of
confidently incorrect decisions.
