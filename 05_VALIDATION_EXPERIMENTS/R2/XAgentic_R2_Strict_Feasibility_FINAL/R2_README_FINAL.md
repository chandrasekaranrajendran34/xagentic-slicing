# XAgentic R2 — Strict Feasibility and Validity Audit

## Status

FINAL / FROZEN

R2 evaluates the original feasibility contract used by the frozen
XAgentic E1-E5 experiments.

The original experiment evidence was not modified.

No model was retrained.

No E5 prediction was regenerated.


## Authoritative oracle

R2_strict_feasibility_oracle_v2.py


## Core results

### Request-level audit

Frozen intents audited: 1,200

Legacy/strict candidate-set disagreements:
229 / 1,200 = 19.08%

Strict-empty requests:
14 / 1,200 = 1.17%

Legacy returned a non-empty candidate set for all 14 strict-empty requests.

Mobility contributed to over-acceptance:
61 / 1,200 = 5.08%

Mobility-only disagreements:
37 / 1,200 = 3.08%


### E5 prediction-level audit

Frozen E5 decisions audited: 100

Legacy-valid predictions:
100 / 100 = 100%

Strict-V2-valid predictions:
97 / 100 = 97%

False-feasible predictions:
3 / 100 = 3%

False-feasible + autonomous ACCEPT:
2 / 100 = 2%

False-feasible + ESCALATE:
1 / 100 = 1%


## Important interpretation

Two false-feasible predictions were classification-correct but violated
strict KPI feasibility.

Therefore classification correctness does not by itself guarantee safe
slice orchestration.

All three false-feasible E5 decisions corresponded to requests with an
empty strict feasible set.

Such requests require escalation, rejection, or explicit infeasibility
handling rather than selection of another catalogue slice.


## V1 provenance

The earlier V1 strict oracle is preserved under:

07_ARCHIVE/R2_V1_LATENCY_ORACLE_SUPERSEDED/

V1 used an incorrect latency-direction interpretation.

V1 results are retained only for provenance and MUST NOT be used for final
publication claims.


## Primary documentation

06_DOCUMENTATION/R2_FINAL_FINDINGS.md
06_DOCUMENTATION/R2_METHODOLOGY.md
06_DOCUMENTATION/R2_CLOSURE.json


## Evidence integrity

Frozen source, frozen corpus, and frozen E5 decision artifacts remained
unchanged throughout R2.

Final integrity validation passed 35/35 checks.
