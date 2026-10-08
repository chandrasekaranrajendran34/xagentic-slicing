# R2 — Strict Feasibility and Validity Audit

## Status

R2 evaluates the feasibility and safety implications of the original
legacy contract used in the frozen E1-E5 experiments.

The original E1-E5 evidence was not modified or regenerated.
R2 is an independent post-hoc strict re-analysis of the frozen evidence.

The authoritative strict oracle is:

R2_strict_feasibility_oracle_v2.py

The earlier V1 strict audit is retained only for provenance and is
superseded by V2 because V2 applies direction-aware latency semantics.


## Strict feasibility semantics

Five KPI dimensions represented by the experiment are evaluated:

- latency
- throughput
- reliability
- device density
- mobility

Latency is interpreted as an upper-bound SLA. A slice is latency-feasible
when its minimum supported latency is less than or equal to the requested
maximum latency.

Throughput, reliability, density, and mobility are capability requirements.
A slice is feasible when the requested requirement does not exceed the
slice's maximum supported capability.

Missing KPI values are treated as unconstrained.

Unlike the legacy implementation, the strict oracle preserves an empty
feasible set when no catalogue slice satisfies the complete request.


## Request-level results

The strict V2 oracle was applied to all 1,200 frozen intents using KPI slots
reconstructed by the frozen parser.

Legacy and strict candidate sets disagreed for 229 of 1,200 intents
(19.08%).

All 229 disagreements involved legacy over-acceptance: the legacy contract
retained at least one candidate that failed the stricter five-KPI audit.

There were no cases in which the legacy contract rejected a candidate that
the strict V2 oracle considered feasible.

Fourteen intents (1.17%) had no feasible slice under the strict oracle.
Nevertheless, the legacy implementation returned a non-empty candidate set
for all fourteen because its empty-set fallback restores the full catalogue.

This demonstrates that the legacy contract can conceal infeasible requests
instead of explicitly representing infeasibility.


## KPI attribution

Density contributed to legacy over-acceptance in 103 of 1,200 intents
(8.58%), making it the largest observed contributor.

Latency contributed in 63 intents (5.25%).

Mobility contributed in 61 intents (5.08%).

Throughput contributed in 37 intents (3.08%).

Reliability produced no disagreement in this frozen corpus.

The attribution categories can overlap because a single request may violate
multiple KPI constraints.


## Mobility finding

Mobility is represented in the generated intents and catalogue but was not
checked by the legacy feasibility function.

The strict audit identified 61 intents (5.08%) in which mobility contributed
to legacy over-acceptance.

Thirty-seven intents (3.08%) were mobility-only disagreements, meaning the
legacy-extra candidate or candidates failed solely because of mobility.

This directly quantifies the impact of the missing mobility check.


## Prediction-level E5 audit

The 100 frozen E5 decisions were mapped back to the frozen 1,200-intent
corpus using exact intent text.

All 100 mappings were exact.

Truth labels matched for 100/100 rows and use-case categories matched for
100/100 rows.

The original predictions were not regenerated.

The original legacy contract classified all 100 E5 predictions as valid,
reproducing the frozen E5 contract-validity result of 100%.

Under strict V2 feasibility, 97 of the 100 predictions remained feasible.

Three predictions (3.0%) were legacy-valid but strict-invalid and are
therefore classified as false-feasible decisions under the strict audit.


## Operational safety finding

Of the three false-feasible E5 decisions:

- one was already escalated by the agent;
- two were autonomously ACCEPTed.

Thus, 2% of the audited E5 decisions were strict-infeasible yet autonomously
accepted under the original pipeline.

All three false-feasible predictions occurred on requests for which the
strict feasible set was empty.

Therefore these cases cannot be corrected merely by choosing another
catalogue slice. The safe outcome is escalation, rejection, or explicit
infeasibility handling.


## Classification accuracy versus feasibility

Two of the three false-feasible cases had predictions equal to their
ground-truth slice labels.

This demonstrates an important distinction between classification
correctness and operational feasibility.

A model may correctly predict the dataset's nominal slice label while the
complete KPI combination is infeasible under the catalogue constraints.

Consequently, classification accuracy and macro-F1 alone are insufficient
safety indicators for autonomous slice orchestration.


## Interpretation

R2 does not invalidate the original E1-E5 experimental results.

Instead, it identifies a limitation in the original contract-validity
definition and quantifies its effect using frozen experimental evidence.

The original results should therefore be reported as legacy-contract
results, while the R2 strict audit should be reported separately as a
safety-oriented re-analysis.

The results motivate explicit strict feasibility enforcement before
autonomous slice deployment.


## Scope limitation

Row-level predictions were preserved for the 100-intent E5 experiment,
allowing direct strict prediction-level re-analysis.

Equivalent row-level predictions were not preserved for every E1-E4
multi-seed experiment. Therefore prediction-level strict feasibility metrics
should not be retrospectively claimed for those runs without rerunning the
models.

The 1,200-intent request-level audit and the 100-intent E5 prediction-level
audit should therefore be reported as distinct analyses.


## Main quantitative findings

1. Legacy/strict candidate disagreement: 229/1,200 (19.08%).
2. Strict-empty requests: 14/1,200 (1.17%).
3. Legacy returned candidates for all 14 strict-empty requests.
4. Mobility contributed to 61/1,200 disagreements (5.08%).
5. Mobility-only disagreements: 37/1,200 (3.08%).
6. E5 legacy contract validity: 100/100 (100%).
7. E5 strict V2 prediction validity: 97/100 (97%).
8. E5 false-feasible predictions: 3/100 (3%).
9. False-feasible autonomous ACCEPT decisions: 2/100 (2%).
10. Two false-feasible decisions were classification-correct but
    operationally infeasible.
