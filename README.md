# xpcsjax

JAX-native NLSQ fitting for X-ray Photon Correlation Spectroscopy (XPCS).

Consolidates the [`homodyne`](https://github.com/imewei/homodyne) and
[`heterodyne`](https://github.com/imewei/heterodyne) analysis pipelines into one
package with a shared engine and config-driven physics-model dispatch.

---

## Install

```bash
uv sync --extra dev
```

Python 3.12+ required. Runtime dependencies are managed via `pyproject.toml` and
mirror what the source `homodyne` package pins (versions of `jax`, `nlsq`,
`evosax`, `h5py`, `interpax`, `jaxopt`, `psutil`, `scikit-learn`, `tqdm`, etc.).

---

## Quickstart

```python
from xpcsjax import load_xpcs_data, fit_nlsq

data   = load_xpcs_data("config.yaml")
result = fit_nlsq(data, "config.yaml")
print(result.parameters)
```

The YAML config's `analysis_mode` field picks the physics:

| `analysis_mode` | Model | Parameters |
|---|---|---|
| `static_isotropic` | Homodyne equilibrium diffusion, angle-collapsed | 3 |
| `static_anisotropic` | Same physics; angle-resolved data prep | 3 |
| `laminar_flow` | Homodyne diffusion + sinc-shear | 7 |
| `two_component` (or `heterodyne`) | Heterodyne two-component (reference + sample + velocity + mixing) | 14 |

For heterodyne (`two_component`) fits, `fit_nlsq` returns a `list[NLSQResult]`
(one per phi angle, jointly fit). For homodyne modes, it returns a single
`OptimizationResult`.

---

## What's here in v0.1

- **Data loading** — verbatim port of `homodyne/data/`: HDF5 reader, diagonal
  correction (mandatory, three methods: basic / statistical / interpolation),
  multi-level cache (LRU + disk NPZ).
- **JAX-native NLSQ engine** — `nlsq.CurveFit` (trust-region reflective LM,
  end-to-end on device). **Never** calls `scipy.optimize.least_squares`.
- **5-layer anti-degeneracy controller** — `FourierReparameterizer`,
  `HierarchicalOptimizer`, `AdaptiveRegularizer`, `GradientCollapseMonitor`,
  `ShearSensitivityWeighting`. Layer 5 is gated by model lineage (active for
  homodyne modes, inert for `two_component`).
- **Memory-aware strategy routing** — `STANDARD` / `OUT_OF_CORE` /
  `HYBRID_STREAMING` selected adaptively from system RAM via `psutil`.
- **CMA-ES escape** — auto-triggers when bound `scale_ratio ≥ 1000`.
  Implementation: `nlsq.CMAESOptimizer` with `evosax` backend + BIPOP restart.
- **Multi-angle heterodyne** — full parity with source heterodyne's
  `fit_nlsq_multi_phi`: joint Fourier / independent / constant-averaged scaling
  modes, plus CMA-ES multi-angle path.

---

## What's coming in v0.2+

- Visualization (matplotlib + pyqtgraph + datashader).
- CLI (`xpcsjax fit ...`, `xpcsjax plot ...`, `xpcsjax convert ...`).
- PyQt6 desktop UI.
- Unification of the two `HeterodyneModel` classes (Task 27 wrapper vs Task 30
  stateful — see [MIGRATION.md](docs/MIGRATION.md)).

---

## What `xpcsjax` will NEVER add

- **CMC / Bayesian sampling** (NumPyro, BlackJAX, ArviZ). xpcsjax is NLSQ-only by
  design. Users needing posterior sampling should use the source `homodyne` or
  `heterodyne` packages.
- `scipy.optimize.least_squares`. The trust-region solver is `nlsq.CurveFit`
  end-to-end. JAX-first, GPU/TPU-capable, no per-iteration Host↔Device transfers.
- Heterodyne's soft-L1 loss (xpcsjax uses chi-squared throughout).

---

## Validation

xpcsjax reproduces source-package fits with strong guarantees:

| Gate | Tolerance | Verification |
|---|---|---|
| Homodyne static (`static_simon` fixture, 3 params) | `rtol=1e-10` | bit-equivalent |
| Homodyne laminar (`laminar_c020`, 53 params w/ CMA-ES path) | `rtol=1e-10` | bit-equivalent |
| Heterodyne joint multi-angle (`two_component_c044`, 14 physics params) | within a few percent; χ² exact; `f0/f2` degeneracy invariant matched | per-parameter |

Run the slow gates manually:

```bash
XPCSJAX_RUN_CHARACTERIZATION=1 uv run pytest tests/ -v
```

The fast suite (68 tests, ~2 s) runs by default:

```bash
uv run pytest tests/ --ignore=tests/characterization --ignore=tests/heterodyne/test_two_component_real_data.py -v
```

---

## See also

- [MIGRATION.md](docs/MIGRATION.md) — moving downstream code from `homodyne` /
  `heterodyne` to `xpcsjax`.
- [Design spec](docs/superpowers/specs/2026-05-18-xpcsjax-nlsq-merge-design.md)
- [Implementation plan](docs/superpowers/plans/2026-05-18-xpcsjax-nlsq-merge.md)

## License

MIT. See `LICENSE`.
