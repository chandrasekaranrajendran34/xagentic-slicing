# R5 - FINAL FINDINGS AND INTERPRETATION

## 1. Research objective

R5 evaluates a frozen semantic verification layer added after the
R4 selective escalation mechanism in an explainable agentic AI
pipeline for network slicing intent interpretation.

The objective is to identify residual autonomous classification
errors while preserving as many correct autonomous decisions as
possible.

## 2. Experimental protocol

- Development cases: 840
- Held-out evaluation cases: 360
- Held-out policy retuning: prohibited
- Frozen semantic support threshold: -0.0275677986595984
- Frozen policy SHA256: b8e5ef13e6895fcfb3656f7257c94727e3cea17d38f08563436d55b6cf0ba243

The semantic verifier was calibrated using development data.
Its policy was frozen before held-out evaluation.

## 3. Frozen system comparison

| Metric | R3 | R4 | R5 |
|---|---:|---:|---:|
| Autonomous decisions | 341 | 306 | 280 |
| Escalated decisions | 19 | 54 | 80 |
| Correct autonomous decisions | 329 | 301 | 278 |
| Autonomous errors | 12 | 5 | 2 |
| Coverage | 94.72% | 85.00% | 77.78% |
| Autonomous accuracy | 96.48% | 98.37% | 99.29% |
| Selective risk | 3.52% | 1.63% | 0.71% |

R5 reduces observed selective risk by 56.29% relative to R4
and 79.70% relative to R3.

The R4-to-R5 coverage cost is 7.22 percentage points.

## 4. Semantic verifier ablation

Without R5:
- Autonomous cases: 306
- Autonomous errors: 5
- Selective risk: 1.634%

With frozen R5:
- Autonomous cases: 280
- Autonomous errors: 2
- Selective risk: 0.714%

R5 additionally escalates 26 cases:
- Previously incorrect decisions captured: 3
- Previously correct decisions escalated: 23
- Residual-error capture rate: 60%

## 5. Use-case category findings

Generic:
- R4 autonomous cases: 88
- R4 errors: 1
- R5 captures: 1
- Correct decisions escalated: 23

Known:
- R4 autonomous cases: 123
- R4 errors: 0
- R5 escalations: 0

Unseen:
- R4 autonomous cases: 95
- R4 errors: 4
- R5 captures: 2
- Correct decisions escalated: 0

All 23 unnecessary R5 escalations occur in generic cases.

## 6. Residual error analysis

Captured errors:
- ID 1197: generic, eMBB predicted mMTC
- ID 240: unseen, eMBB predicted mMTC
- ID 769: unseen, URLLC predicted eMBB

Missed errors:
- ID 658: unseen, URLLC predicted mMTC
- ID 810: unseen, eMBB predicted mMTC

Both missed errors have semantic support margins above
the frozen escalation threshold.

No threshold adjustment was performed.

## 7. Statistical robustness

Method:
- Paired case-level bootstrap
- 10,000 resamples
- Random seed: 42
- Held-out sample size: 360

95% percentile bootstrap confidence intervals:

R4 selective risk:
- Observed: 1.634%
- CI: 0.328% to 3.226%

R5 selective risk:
- Observed: 0.714%
- CI: 0% to 1.812%

Absolute risk reduction:
- Observed: 0.920 percentage points
- CI: -0.053 to 2.184 percentage points

Coverage cost:
- Observed: 7.222 percentage points
- CI: 4.722 to 10.000 percentage points

The absolute risk-reduction confidence interval includes zero.
Consequently, the observed improvement should not be
described as statistically conclusive at the 95% level.

## 8. Novelty and contribution

R5 adds a development-calibrated semantic verification
mechanism to an existing feasibility-aware, selectively
escalated intent interpretation pipeline.

Its experimental contribution is the measured reduction
in residual autonomous errors under a frozen policy,
together with explicit quantification of coverage costs,
category-specific behavior, and statistical uncertainty.

The current experiment does not establish deployment-scale
reliability or universal generalization.

## 9. Limitations

1. Only five residual R4 errors exist in the held-out sample.
2. The confidence interval for risk reduction includes zero.
3. Twenty-three correct generic decisions are escalated.
4. Two unseen-intent errors remain undetected.
5. Evaluation is limited to the frozen benchmark.
6. Runtime deployment, operator feedback, and live
   network-slicing actuation are not established here.

## 10. Future research

Future work may investigate:
- Category-aware semantic verification
- Better treatment of generic intents
- Independent held-out datasets
- Larger residual-error samples
- Runtime overhead and end-to-end latency
- Integration with network slicing orchestration

Any modified policy requires development-only calibration
and evaluation on a fresh untouched test set.

## 11. Conclusion

The frozen R5 verifier reduces observed autonomous errors
from five to two after R4, increasing autonomous accuracy
from 98.366% to 99.286%.

This comes at the cost of reducing coverage from 85.00%
to 77.78%.

The results support the feasibility of semantic verification
as an additional assurance layer, while the uncertainty
analysis and generic-intent false escalations demonstrate
the need for further independent validation.
