# R3 — Candidate Masking: Final Experimental Interpretation

## Objective

R3 evaluates whether strict feasibility constraints can be incorporated
into Selection-Agent decision making so that autonomous network-slice
selection does not execute KPI-infeasible actions.

## Experimental Design

The experiment uses the exact frozen 1,200-request E5 corpus and the
packaged Selection-Agent implementation.

The exact held-out Selection-Agent test split contains 360 requests.

Strict candidate feasibility is supplied by the frozen R2-V2 oracle.

The final selective policy was calibrated independently using five-fold
out-of-fold predictions over the 840-row training split.

The feasible-probability-mass threshold was frozen before final held-out
evaluation.

Frozen threshold:

`tau = 0.548631598726`

Policy SHA256:

`9ebab1f743c71ee4fb3202976a6d9846b03391e256fbc095bd561c5adf831774`

## Final Policy

1. If the strict feasible set is empty, escalate.
2. Otherwise compute the Selection Agent's probability mass assigned to
   strict-feasible candidates.
3. If feasible probability mass is below the frozen OOF threshold,
   escalate.
4. Otherwise renormalize/select among strict-feasible candidates and
   execute the strict masked argmax.

## Final Held-Out Results

Total held-out requests: **360**

Autonomous requests: **341 / 360 (94.722%)**

Escalated requests: **19 / 360 (5.278%)**

- Strict-empty escalations: **4**
- Low-feasible-mass escalations: **15**

Autonomous strict validity: **100.000%**

Autonomous strict violations: **0**

Autonomous accuracy: **96.481%**

Correct autonomous decisions: **329 / 341**

Correct autonomous decisions over all requests:
**91.389%**

Final autonomous correct-to-wrong transitions: **0**

## System Comparison

### Original Selection Agent

- Coverage: 100%
- Accuracy: 95.833%
- Strict validity: 94.722%

The unconstrained Selection Agent provides high predictive accuracy but
can select slices that violate strict request-level KPI feasibility.

### Strict-V2 Hard Mask

- Coverage: 98.889%
- Autonomous accuracy: 92.697%
- Autonomous strict validity: 100%

Hard masking eliminates autonomous strict violations but can force the
classifier onto very-low-probability feasible classes when nominal labels
and strict KPI constraints conflict.

### Strict-V2 + Frozen OOF Feasible-Mass Escalation

- Coverage: 94.722%
- Autonomous accuracy: 96.481%
- Autonomous strict validity: 100%
- Escalation rate: 5.278%
- Autonomous correct-to-wrong transitions: 0

The OOF-calibrated selective policy preserves the hard safety guarantee
while avoiding autonomous forced decisions in low-feasible-mass cases.

## Failure-Mode Interpretation

The earlier hard-mask analysis showed that all 13 correct-to-wrong mask
changes occurred where the nominal ground-truth slice itself was
strictly infeasible under the request's KPI constraints.

Therefore these cases primarily expose a label/constraint conflict rather
than a failure of the masking implementation.

The final selective policy converts these uncertain safety conflicts into
escalations rather than autonomous substitutions.

## Main R3 Finding

R3 demonstrates a measurable safety-utility trade-off.

Strict candidate masking can guarantee constraint-valid autonomous
decisions, but unconditional hard masking may reduce classification
accuracy when the nominal class label conflicts with strict KPI
feasibility.

A feasible-probability-mass escalation mechanism calibrated using
out-of-fold training predictions resolves this failure mode by abstaining
from low-support constrained decisions.

On the frozen held-out test set, the final policy achieved:

- **100% autonomous strict validity**
- **96.481% autonomous accuracy**
- **94.722% autonomous coverage**
- **5.278% escalation**
- **zero autonomous correct-to-wrong transitions**

These results support selective constrained decision making rather than
unconditional post-classification hard masking.

## Scientific Limitation

The final system intentionally trades coverage for safety and selective
accuracy.

Escalated requests are not counted as autonomous classification
successes. Therefore autonomous accuracy must always be reported together
with coverage and escalation rate.

The result should not be described as improving overall 360-request
classification accuracy. Its primary contribution is safe selective
autonomy.
