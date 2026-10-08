def _finite_value(row, key):
    """Return finite float value or None for a missing constraint."""
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None

    return value if np.isfinite(value) else None


def strict_slice_audit(slice_name, row):
    """
    Independently audit one slice against every stated KPI.

    Returns:
        feasible: bool
        violations: list[str]
        checks: dict
    """

    c = frozen.CATALOGUE[slice_name]

    violations = []
    checks = {}

    for kpi, rule in KPI_RULES.items():

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

        # Capability requirement:
        # requested value must not exceed slice capability maximum.
        passed = request_value <= (float(upper) + 1e-12)

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


def strict_feasible_set(row):
    """
    Return only slices satisfying ALL stated KPI constraints.

    Critical R2 behavior:
        no feasible slices -> []
    """

    feasible = []

    for slice_name in frozen.SLICES:

        audit = strict_slice_audit(
            slice_name,
            row
        )

        if audit["feasible"]:
            feasible.append(slice_name)

    return feasible


def strict_contract_valid(pred_label, row):
    """
    Strict validity of a selected slice.
    """

    return pred_label in strict_feasible_set(row)
