# R2 Methodology — Strict Feasibility Re-analysis

## Objective

R2 audits whether the original network-slice feasibility contract can admit
candidate slices that violate the represented KPI requirements.

## Evidence preservation

The E1-E5 source code, corpus, predictions, and result artifacts were treated
as frozen evidence.

No model was retrained.

No E5 prediction was regenerated.

No original result was overwritten.

## Parser reconstruction

The frozen corpus stores natural-language intent text and generated KPI truth
values.

Because the original experiment did not persist all parsed KPI slots at
row level, R2 reconstructs those slots deterministically using the original
frozen parser.

Generated truth values are not substituted for parsed values because the
runtime feasibility contract operates on parsed intent values.

## Strict oracle

The independent V2 oracle evaluates all five represented KPI dimensions:
latency, throughput, reliability, density, and mobility.

Missing KPI values are treated as unconstrained.

Latency is evaluated as an upper-bound SLA:

    catalogue minimum latency <= requested maximum latency

Other KPI requirements are evaluated as required capabilities:

    requested requirement <= catalogue maximum capability

An empty feasible set is preserved.

## Legacy comparison

For each frozen intent, the candidate set returned by the original legacy
feasibility function is compared with the strict V2 candidate set.

Legacy-extra candidates are candidates accepted by the legacy function but
rejected by the strict oracle.

Legacy-missing candidates are candidates rejected by the legacy function but
accepted by the strict oracle.

## E5 prediction-level audit

The frozen E5 decision artifact contains the original intent text, prediction,
truth label, verdict, escalation state, and subgroup.

Each E5 decision is mapped to the frozen corpus by exact intent text.

Mapping integrity is verified against both truth label and subgroup.

The original prediction is then tested against:

1. the original legacy feasible set; and
2. the independent strict V2 feasible set.

A false-feasible decision is defined as:

    legacy prediction valid = True
    AND
    strict V2 prediction valid = False

No prediction is regenerated during this analysis.

## Reporting

Candidate-set disagreement and prediction-level false-feasibility are
reported separately.

Candidate-set disagreement must not be described as a false-feasible
prediction unless an actual selected prediction is available.

This distinction prevents overstatement of the R2 findings.
