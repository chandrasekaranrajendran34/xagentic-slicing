def strict_slice_audit_v2(slice_name, row):

    c = frozen.CATALOGUE[slice_name]

    violations = []
    checks = {}

    for kpi, rule in KPI_RULES_V2.items():

        request_value = _finite_value(row, kpi)
        catalogue_key = rule["catalogue_key"]

        lower, upper = c[catalogue_key]

        if request_value is None:

            checks[kpi] = {
                "stated": False,
                "request": None,
                "catalogue_min": float(lower),
                "catalogue_max": float(upper),
                "pass": None,
            }

            continue

        # ----------------------------------------------------
        # Direction-aware feasibility
        # ----------------------------------------------------

        if kpi == "latency_ms":

            # Requested latency is an upper-bound SLA.
            # Slice must be capable of latency <= requested value.
            passed = float(lower) <= (
                float(request_value) + 1e-12
            )

        else:

            # Required capability must not exceed slice maximum.
            passed = float(request_value) <= (
                float(upper) + 1e-12
            )

        checks[kpi] = {
            "stated": True,
            "request": float(request_value),
            "catalogue_min": float(lower),
            "catalogue_max": float(upper),
            "pass": bool(passed),
        }

        if not passed:
            violations.append(kpi)

    return {
        "slice": slice_name,
        "feasible": len(violations) == 0,
        "violations": violations,
        "checks": checks,
    }


def strict_feasible_set_v2(row):

    feasible = []

    for slice_name in frozen.SLICES:

        audit = strict_slice_audit_v2(
            slice_name,
            row
        )

        if audit["feasible"]:
            feasible.append(slice_name)

    return feasible


def strict_contract_valid_v2(pred_label, row):

    return (
        pred_label
        in strict_feasible_set_v2(row)
    )
