R2 V1 STRICT ORACLE — SUPERSEDED EVIDENCE

STATUS
------

These files are retained only for methodological provenance.

They MUST NOT be used as the final R2 quantitative results.


REASON FOR SUPERSESSION
-----------------------

The V1 strict feasibility oracle evaluated latency using:

    requested_latency <= catalogue_max_latency

This does not correctly represent the directional semantics of a
maximum-latency SLA.

For a request of the form:

    latency <= X

the correct feasibility relationship is:

    catalogue_min_latency <= requested_max_latency


AUTHORITATIVE REPLACEMENT
-------------------------

R2 V2 corrects the latency direction while retaining strict evaluation of:

- latency
- throughput
- reliability
- density
- mobility

V2 also preserves an empty feasible set when no catalogue slice satisfies
the request.

The authoritative oracle is:

    R2_strict_feasibility_oracle_v2.py


PUBLICATION RULE
----------------

Do not report V1 R2-15 through R2-20 quantitative results.

Use only V2-controlled validation, the V2 1,200-intent audit, and the V2
100-decision E5 prediction-level audit for final R2 claims.

V1 is retained solely to document the complete experimental history.
