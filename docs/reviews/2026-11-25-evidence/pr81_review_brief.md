# Review request: xpcsjax PR #81 (draft)

Review and adversarially verify the changes in PR #81
(`fix/maintainer-decisions-2026-11-25`, 19 commits on top of merged `main` `fde41b4`). Claude will
reproduce every finding you report and fix the confirmed ones, so report only what you can back
with a concrete failing scenario.

## Where everything is (exact paths; do NOT search the filesystem)
- `$WT` = `/home/wei/Documents/GitHub/xpcsjax/.claude/worktrees/codex-review-loop` — the branch is
  checked out here. Your pane's own directory is a different checkout; read code only under `$WT`.
- The diff (source + tests): `$WT/.review/pr81.diff`
- Changed files, one per line: `$WT/.review/pr81_files.txt`
- The spec — one disposition per item, and what each fix is meant to do:
  `$WT/docs/reviews/2026-11-25-maintainer-decisions.md`
- New tests: `$WT/tests/review_r1/test_maintainer_decisions.py`
- Project rules and deliberate design decisions: `$WT/CLAUDE.md`

## Rules
- READ ONLY except your own report file. No git commands that change state.
- Do NOT run the full suite or `make`. You may run a single small test or a short
  `uv run --no-sync python -c ...` probe from `$WT` to prove a finding.
- Fitted parameters and chi-squared on default paths are pinned by `rtol=1e-10` goldens that pass
  on this branch. A finding that needs them to move is a design question, not a bug: mark it so.

## What to look for
1. **Bugs in the fixes**: wrong result, crash, wrong DOF/uncertainty, a path the fix missed, a
   caller broken by a changed signature or removed export, a silent fallback hiding a failure.
2. **Half-fixes**: the same defect still present on a sibling path (e.g. another per-angle mode,
   the >= 1 M stratified path, hybrid streaming, cold load vs cache hit).
3. **Default-path numeric changes not listed** in the spec's "Default-visible changes".
4. **Items closed as intended**: challenge any whose cited evidence does not actually justify
   leaving it (spec section "Closed as intended").
5. **Tests that cannot fail**: a new test that would pass without its fix.

Priority areas: the masked-fit DOF count (`heterodyne_data_prep.py`: `n_contributing_rows`,
`covariance_on_valid_rows`, and their callers in `heterodyne_core.py`,
`heterodyne_constant_mode.py`, `heterodyne_stratified_ls.py`,
`strategies/heterodyne_hybrid_streaming.py`); the heterodyne time-axis clamp
(`core/heterodyne_model_stateful.py`, `data/xpcs_loader.py`, `data/npz_cache.py`); the Layer 3
enable flag (`anti_degeneracy_controller.py`, `heterodyne_config.py`,
`strategies/hybrid_streaming_layers.py`); the averaged-mode covariance rescale (`wrapper.py`); the
adapter time-grid guard (`adapter.py`).

## Report format (your file is named in the message that sent you here)

# <codex|agy> review of PR #81
## Findings
### F1: <one-line claim>
- Must-fix: yes | no
- Location: path:line
- Scenario: <concrete input/state -> wrong output/crash>
- Evidence: <what you ran or read; include the command if you ran one>
- Suggested fix: <one or two sentences>
(repeat; write `None.` if you found nothing)
## Closed-as-intended items you dispute
<ID — reason, or `None.`>
## Verdict
Must-fix count: <n>
END_OF_REPORT
