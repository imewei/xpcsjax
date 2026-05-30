# Migrating from `homodyne` or `heterodyne` to `xpcsjax`

`xpcsjax` v0.1 consolidates the NLSQ fitting pipelines from the standalone
`homodyne` and `heterodyne` packages into a single JAX-native package.

This guide covers what changes for downstream users of the two source packages.

---

## Quickstart — both physics models

```python
from xpcsjax import load_xpcs_data, fit_nlsq

# Same call for both models — analysis_mode in the YAML config selects the physics
data   = load_xpcs_data("my_config.yaml")
result = fit_nlsq(data, "my_config.yaml")
print(result.parameters)
```

The YAML config's `analysis_mode` field picks the physics model:

| `analysis_mode` | Physics | Param count |
|---|---|---|
| `static_isotropic` | Homodyne equilibrium diffusion (D₀, α, D_offset), angle-collapsed | 3 |
| `static_anisotropic` | Same physics; angle-resolved data prep | 3 |
| `laminar_flow` | Homodyne diffusion + sinc-shear (D, γ̇, φ₀) | 7 |
| `two_component` | Heterodyne reference + sample + velocity + mixing | 14 |
| `heterodyne` | Accepted synonym for `two_component` (normalized at load) | 14 |

> **Deprecation (vs. the upstream `homodyne` package):** the bare value
> `analysis_mode: static` is **accepted but deprecated**. It was ambiguous
> between `static_isotropic` and `static_anisotropic`, so `ConfigManager`
> normalizes it to `static_anisotropic` (angle-resolved — the
> drop-in-compatible default) and emits a deprecation warning:
>
> ```text
> analysis_mode='static' is deprecated; mapping to 'static_anisotropic'
> (angle-resolved drop-in). Set 'static_anisotropic' or 'static_isotropic'
> explicitly to silence this warning.
> ```
>
> Bare `static` is **not** hard-failed: upstream `homodyne` configs and the
> characterization parity oracle legitimately use it. Set the variant
> explicitly to silence the warning.

### Command-line interface

A `xpcsjax` console entry point ships in v0.1 (the upstream packages had their
own runners). It dispatches on the same YAML `analysis_mode`:

```bash
# Run an NLSQ fit from a YAML config
xpcsjax --config analysis.yaml

# Override the output directory; run multistart with 16 restarts
xpcsjax --config analysis.yaml --output ./results --multistart --multistart-n 16
```

Companion entry points: `xpcsjax-config` (generate a starter YAML),
`xpcsjax-config-xla` (XLA flags), `xpcsjax-validate` (system check), and
`xpcsjax-post-install` / `xpcsjax-cleanup` (shell-completion installer).

---

## What changed

### Import paths

| Old | New |
|---|---|
| `from homodyne.data import load_xpcs_data` | `from xpcsjax.data import load_xpcs_data` |
| `from homodyne.optimization import fit_nlsq_jax` | `from xpcsjax.optimization.nlsq import fit_nlsq` (single entry) |
| `from heterodyne.data import load_xpcs_data` | `from xpcsjax.data import load_xpcs_data` |
| `from heterodyne.optimization import fit_nlsq_jax` | `from xpcsjax.optimization.nlsq import fit_nlsq` |
| `from homodyne.config import ConfigManager` | `from xpcsjax.config import ConfigManager` |
| `from homodyne.core import CombinedModel, DiffusionModel` | `from xpcsjax.core import CombinedModel, DiffusionModel, HomodyneModel` |

### Single fit entry point

Both packages previously had their own `fit_nlsq_jax` entry. xpcsjax exposes a single
`fit_nlsq(data, config)` that dispatches on `analysis_mode`:

```python
# Homodyne — was:
from homodyne.optimization import fit_nlsq_jax
result = fit_nlsq_jax(data, config)

# Heterodyne — was:
from heterodyne.optimization import fit_nlsq_multi_phi
result = fit_nlsq_multi_phi(model, c2_data, phi_angles, nlsq_config)

# xpcsjax — both:
from xpcsjax import fit_nlsq
result = fit_nlsq(data, config)  # internal dispatch picks the right path
```

### Heterodyne parameter renames

Two parameter names collide between homodyne and heterodyne. xpcsjax renames the
heterodyne versions to keep a flat registry:

| Heterodyne docs | xpcsjax registry | Reason |
|---|---|---|
| `beta` (velocity exponent) | `v_beta` | Homodyne's `beta` is the shear-rate exponent — different physics, different bounds |
| `phi0` (degrees, [-10, 10]) | `phi0_het` | Renamed only to avoid colliding with homodyne's `phi0`; same unit and bounds (degrees, [-10, 10]) |

YAML configs written against the upstream heterodyne package may need to rename
`beta` → `v_beta` and `phi0` → `phi0_het` in their `parameters` section. The
xpcsjax dispatch normalizer accepts the `heterodyne` synonym for `analysis_mode`
but does NOT rename parameter keys automatically.

### Result classes

| Old | New |
|---|---|
| `homodyne` returned `OptimizationResult` (numpy params array, chi_squared, convergence_status) | Same `OptimizationResult` in `xpcsjax.optimization.nlsq.results` |
| `heterodyne` returned `NLSQResult` (parameters, parameter_names, success, message, uncertainties, ...) | `NLSQResult` ported as `xpcsjax.optimization.nlsq.heterodyne_results.NLSQResult` |
| Heterodyne multi-angle fits returned `list[NLSQResult]` | Same — `fit_nlsq` returns `list[NLSQResult]` for `two_component` mode |

---

## What `xpcsjax` does NOT support — and never will

- **CMC / Bayesian sampling (NumPyro, BlackJAX, ArviZ).** xpcsjax is NLSQ-only
  by design. The strict CMC audit (`grep -rn -i "numpyro|blackjax|arviz" xpcsjax/`)
  must return zero matches. Users needing posterior sampling should continue to
  use the source `homodyne` or `heterodyne` packages directly.
- **`scipy.optimize.least_squares`.** The trust-region solver is `nlsq.CurveFit`
  (JAX-native, GPU/TPU-capable). xpcsjax never calls into scipy's LM step.
- **Heterodyne's separately-developed anti-degeneracy controller** — superseded
  by homodyne's 5-layer controller with `ShearSensitivityWeighting` (Layer 5)
  gated by `model.analysis_mode` so it's inert for heterodyne fits.
- **Heterodyne's soft-L1 robust loss.** xpcsjax uses chi-squared throughout.

---

## What is NOT in v0.1 (deferred to v0.2+)

- **Interactive** visualization (PyQtGraph) and the PyQt6 desktop UI
- NetworkDynamics, advanced HPC scheduling

> **Shipped since the original v0.1 merge plan** (no longer deferred):
> static-plot visualization (matplotlib + datashader) via
> `from xpcsjax.viz import plot_nlsq_fit, plot_residual_map,
> plot_simulated_data, generate_nlsq_plots`; the `xpcsjax` CLI (see
> [Command-line interface](#command-line-interface) above); and the
> shell-completion installer (`runtime/`, `post_install.py`).

---

## Bit-equivalence guarantee for homodyne fits

xpcsjax reproduces source `homodyne` NLSQ fits **bit-equivalently** (`rtol=1e-10`)
on the configs in `tests/characterization/fixtures/configs/`. This includes the
full CMA-ES escape, anti-degeneracy controller, and memory-aware strategy
routing.

Run the gate locally with:

```bash
XPCSJAX_RUN_CHARACTERIZATION=1 uv run pytest tests/characterization/ -v
```

## Multi-angle heterodyne agreement

xpcsjax reproduces source `heterodyne`'s joint multi-angle NLSQ fit χ²
**exactly** (7131.31 for the C044 reference dataset) and recovers the 14
physics parameters within a few percent (worst case 1.5% on `D_offset_ref`).
The `f0/f2` parameters are physically degenerate along `f0·exp(-f1·f2)`; the
invariant matches.

Run the gate locally with:

```bash
XPCSJAX_RUN_CHARACTERIZATION=1 uv run pytest tests/heterodyne/test_two_component_real_data.py -v
```

---

## Notes for power users

### Direct model access

```python
from xpcsjax import ConfigManager, HomodyneModel, HeterodyneModel

cfg   = ConfigManager("config.yaml")
model = cfg.get_model()      # routes to HomodyneModel or HeterodyneModel
```

The `make_model(config)` factory in `xpcsjax.core.models` is the lower-level dispatcher.

### Two HeterodyneModel classes

xpcsjax v0.1 has **two** classes named `HeterodyneModel` by design:

- `xpcsjax.core.heterodyne_model.HeterodyneModel` — a thin wrapper exposing the
  `PhysicsModelBase` contract. This is what `from xpcsjax import HeterodyneModel`
  resolves to. Used for standalone kernel access and per-angle smoke tests.
- `xpcsjax.core.heterodyne_model_stateful.HeterodyneModel` — the stateful dataclass
  ported from upstream heterodyne (with `from_config`, `t/q/dt`, `param_manager`,
  `scaling`, `set_params`, `sync_time_axis`). Used internally by the multi-phi
  fit pipeline.

These will be unified in v0.2 cleanup.

### Custom analysis_mode synonyms

`"heterodyne"`, `"two_component"`, and `"two-component"` (and case variants) all
resolve to the same 14-param heterodyne fit. The `_normalize_analysis_mode` step
in `ConfigManager` handles this consistently.

---

## See also

- [Design spec](superpowers/specs/2026-05-18-xpcsjax-nlsq-merge-design.md)
- [Implementation plan](superpowers/plans/2026-05-18-xpcsjax-nlsq-merge.md)
- README
