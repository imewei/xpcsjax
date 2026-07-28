# xpcsjax

XPCS (X-ray Photon Correlation Spectroscopy) NLSQ analysis package. JAX-native
port of the homodyne/heterodyne fitting pipelines — NLSQ-only, no Bayesian
sampling.

## Language

**`__all__` member** vs **documented symbol**:
Two different sets that do *not* coincide in this codebase. A package's
`__init__.py` `__all__` marks names safe for `from xpcsjax.pkg import *` and
worth re-exporting from the package — it does **not** mean "worth a
dedicated prose/autodoc entry in the `.rst` page." Verified empirically on
pages with **zero** `automodule` directives (so no hidden coverage is
possible): `xpcsjax.optimization.__all__` has 19 names, only 8 appear in
`optimization.rst`; `core.rst` misses `ShearModel`/`make_model`; `data.rst`
misses `get_data_module_info`. Internal/implementation-detail exports
(`NLSQ_AVAILABLE`, `OPTIMIZATION_STATUS`, `create_angle_stratified_indices`)
are legitimately `__all__` members with no doc entry. (A literal-text grep
against `cli.rst` looks like a bigger gap — 2 of 12 names — but that's a
grep artifact: the other 10 are each defined in a submodule `cli.rst`
documents wholesale via `automodule`, just never named as literal text; see
`Structural doc coverage` below for why that mechanism matters.) Do not
treat `__all__` as a "doc-worthy public API" list — see
[ADR-0001](docs/adr/0001-automated-structural-doc-coverage-check.md) for the
plan this broke.
_Avoid_: "public surface" as a synonym for either set — ambiguous which one
is meant.

**Structural doc coverage**:
The mechanically-verifiable fact that a top-level `xpcsjax` submodule has a
corresponding `docs/source/api/{name}.rst` page — existence only, nothing
about the page's content. Enforced by an automated test — see
[ADR-0001](docs/adr/0001-automated-structural-doc-coverage-check.md).
_Avoid_: "doc completeness", "API coverage" (both wrongly imply content is
checked too)

**Content doc accuracy**:
Whether documented claims (architecture descriptions, removed-feature
warnings, command examples, config keys, *and* which symbols got a doc
entry) are actually true of and complete for the current implementation. Not
mechanically checkable at the granularity this codebase's `.rst` pages use
(hand-curated symbol subsets, not exhaustive `__all__` dumps) — verified by
manual/agent-driven drift audits only. See
[ADR-0001](docs/adr/0001-automated-structural-doc-coverage-check.md).
