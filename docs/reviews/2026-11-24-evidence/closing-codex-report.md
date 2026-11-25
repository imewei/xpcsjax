# Codex closing report
## Fix check
- R4-A1 — ACCEPTED. The shifted-axis solver probe reported 664 contributing observations, matching the intersection of the time and sigma masks.
- R4-A2 — ACCEPTED. The NPZ column-vector case returns a flat angle vector, and the row-vector rejection test still passes.
- R4-A4 — ACCEPTED. The real budget-exhaustion probe reports a failed convergence status and a false success flag.
- R4-C3 — ACCEPTED. The public fit produces matching chi-squared for the nonuniform singleton slab and its equivalent 2-D mask.
- R4-C1 — ACCEPTED. Replacing the measured-q twin releases the old one and preserves the evaluated surface. All seven selected tests passed at `6c7dd07`. The full suite was outside scope.

## Handoffs
- R4-A3 — ACCEPT HANDOFF. Both joint result builders need a common observation count that excludes zero-weight rows. The handoff covers positive-DOF uncertainty and goodness-of-fit reporting as well as zero-DOF placeholders.
- R4-A5 — ACCEPT HANDOFF. Equivalent shared and full weight layouts now follow the same engine policy. Whether to apply non-unit weights in the robust objective or route those fits to the weighted solver remains a maintainer decision.

## Regressions
None.

## Verdict
AGREE_NO_BUGS
END_OF_REPORT
