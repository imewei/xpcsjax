# xpcsjax

JAX-native NLSQ fitting for X-ray Photon Correlation Spectroscopy (XPCS).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)

xpcsjax consolidates the [`homodyne`](https://github.com/imewei/homodyne) and
[`heterodyne`](https://github.com/imewei/heterodyne) analysis pipelines — both now
deprecated in its favor — into one package with a shared engine and config-driven
physics-model dispatch. It implements the transport-coefficient framework of
[He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121) and
[He et al. PNAS 2025](https://doi.org/10.1073/pnas.2514216122) for characterizing
nonequilibrium dynamics in flowing soft-matter systems.

> [!NOTE]
> **xpcsjax is NLSQ-only by design.** Bayesian sampling — NumPyro, BlackJAX, ArviZ,
> Consensus Monte Carlo (CMC), NUTS, HMC, parallel tempering — is **permanently out of
> scope.** If you need Bayesian XPCS analysis, use the upstream `homodyne` / `heterodyne`
> packages instead.

---

## Install

```bash
uv sync --extra dev
```

Python 3.12+ required, CPU-only in v0.1 (GPU support is v0.2+). Runtime dependencies are
managed via `pyproject.toml` and mirror what the source `homodyne` package pins (versions
of `jax`, `nlsq`, `evosax`, `h5py`, `interpax`, `jaxopt`, `psutil`, `scikit-learn`,
`tqdm`, etc.).

---

## Quickstart

```python
from xpcsjax import load_xpcs_data, fit_nlsq

data   = load_xpcs_data("config.yaml")
result = fit_nlsq(data, "config.yaml")
print(result.parameters)
```

The YAML config's `analysis_mode` field selects the physics model and parameter set:

| `analysis_mode` | Lineage | Model | Physics params |
|---|---|---|---|
| `static_isotropic` | homodyne | Equilibrium diffusion, angle-collapsed | 3 |
| `static_anisotropic` | homodyne | Same physics; angle-resolved data prep | 3 |
| `laminar_flow` | homodyne | Diffusion + sinc-shear | 7 |
| `two_component` (or `heterodyne`) | heterodyne | Two-component: reference + sample + velocity + mixing | 14 |

For heterodyne (`two_component`) fits, `fit_nlsq` returns a `list[NLSQResult]` (one per
phi angle, jointly fit). For homodyne modes, it returns a single `OptimizationResult`.
Two per-angle scaling parameters (`contrast`, `offset`) are appended automatically for
every azimuthal angle in all modes.

### Data flow

```
YAML config --> XPCSDataLoader(HDF5) --> HomodyneModel / HeterodyneModel --> NLSQ engine --> Results (JSON + NPZ)
```

---

## Physics models

xpcsjax fits two-time intensity correlation functions $c_2(\vec{q}, t_1, t_2)$. All time
integrals are evaluated **numerically** via cumulative trapezoid on the discrete time grid
— no analytical antiderivatives — so the general power-law forms stay correct.

### Homodyne (`static_*`, `laminar_flow`)

Single-component scattering where correlation decay encodes diffusion and shear:

$$c_2(\vec{q}, t_1, t_2) = 1 + \beta \times [c_1(\vec{q}, t_1, t_2)]^2$$

$$c_1(\vec{q}, t_1, t_2) = \exp\left(-q^2 \int_{t_1}^{t_2} J(t')\,dt'\right) \times \mathrm{sinc}\left(\frac{qL\cos(\phi)}{2} \int_{t_1}^{t_2} \dot{\gamma}(t')\,dt'\right)$$

where $\beta$ is the optical contrast and $\phi$ is the angle between the scattering
vector and the flow direction. Transport and shear follow power-law forms:

$$J(t) = D_0\,t^{\alpha} + D_{\text{offset}} \qquad \dot{\gamma}(t) = \dot{\gamma}_0\,t^{\beta} + \dot{\gamma}_{\text{offset}}$$

| Group | Parameter | Description | Default | Units |
|---|---|---|---|---|
| Diffusion | `D0` | Diffusion prefactor | 1e4 | Å²/s |
| | `alpha` | Transport exponent (0 = Wiener, 1 = ballistic) | 0.0 | — |
| | `D_offset` | Transport rate offset | 0.0 | Å²/s |
| Shear (`laminar_flow` only) | `gamma_dot_0` | Shear-rate prefactor | 1e-3 | s⁻¹ |
| | `beta` | Shear-rate exponent (0 = constant shear) | 0.0 | — |
| | `gamma_dot_offset` | Shear-rate offset | 0.0 | s⁻¹ |
| Flow angle | `phi0` | Flow angle offset relative to q-vector | 0.0 | degrees |
| Per-angle scaling | `contrast` | Optical (speckle) contrast | 0.5 | — |
| | `offset` | Baseline offset | 1.0 | — |

The static modes use the 3 diffusion parameters; `laminar_flow` adds the 3 shear
parameters and the flow angle (7 physics parameters total).

### Heterodyne (`two_component`)

Two-component scattering (PNAS 2025 SI Eqs. S-77–S-98): light from a moving **sample**
interferes with a static **reference**, and the cross-term oscillates at a frequency set
by the sample velocity. The two-time correlation (Eq. S-95) is

$$c_2(\vec{q}, t_1, t_2) = 1 + \frac{\beta}{f^2}\left[C_{\text{ref}} + C_{\text{sample}} + C_{\text{cross}}\right]$$

$$C_{\text{ref}} = [x_r(t_1)x_r(t_2)]^2 \exp\left(-q^2\int_{t_1}^{t_2} J_r\,dt'\right) \qquad C_{\text{sample}} = [x_s(t_1)x_s(t_2)]^2 \exp\left(-q^2\int_{t_1}^{t_2} J_s\,dt'\right)$$

$$C_{\text{cross}} = 2\,x_r(t_1)x_r(t_2)x_s(t_1)x_s(t_2)\,\exp\left(-\tfrac{1}{2}q^2\int_{t_1}^{t_2}[J_s + J_r]\,dt'\right)\cos\left[q\cos(\varphi)\int_{t_1}^{t_2}\mathbb{E}[v]\,dt'\right]$$

where $x_s(t)$ is the sample fraction, $x_r = 1 - x_s$ the reference fraction, $\varphi$
the angle between velocity and $\vec{q}$, and
$f^2 = [x_s(t_1)^2 + x_r(t_1)^2][x_s(t_2)^2 + x_r(t_2)^2]$ normalizes so that
$c_2(t, t) = 1 + \beta$ on the diagonal. The fit wraps the correlation with per-angle
scaling, $c_2^{\text{model}} = \text{offset} + \text{contrast}\times(C_{\text{ref}} + C_{\text{sample}} + C_{\text{cross}})/f^2$.

Each transport coefficient and the velocity follow power laws, and the sample fraction is
time-dependent — **14 physics parameters** in five groups:

| Group | Parameters | Rate function | Defaults | Units |
|---|---|---|---|---|
| Reference transport (3) | `D0_ref`, `alpha_ref`, `D_offset_ref` | $J_r(t) = D_{0,r}\,t^{\alpha_r} + D_{\text{offset},r}$ | 1e4, 0, 0 | Å²/s^(α+1), —, Å²/s |
| Sample transport (3) | `D0_sample`, `alpha_sample`, `D_offset_sample` | $J_s(t) = D_{0,s}\,t^{\alpha_s} + D_{\text{offset},s}$ | 1e4, 0, 0 | Å²/s^(α+1), —, Å²/s |
| Velocity (3) | `v0`, `beta`, `v_offset` | $v(t) = v_0\,t^{\beta} + v_{\text{offset}}$ | 1e3, 0, 0 | Å/s^(β+1), —, Å/s |
| Sample fraction (4) | `f0`, `f1`, `f2`, `f3` | $f_s(t) = f_0\,\exp\!\big(f_1(t - f_2)\big) + f_3$ | 0.5, 0, 0, 0 | —, s⁻¹, s, — |
| Flow angle (1) | `phi0` | — | 0 | degrees |

As with homodyne, 2 per-angle scaling parameters (`contrast`, `offset`) are tracked per
azimuthal angle but live outside the 14-element physics vector. Per-angle scaling modes
`constant` / `averaged` / `individual` (resolved from `auto`) reach full parity with the
source heterodyne package's `fit_nlsq_multi_phi`.

---

## What's here in v0.1

- **Data loading** — verbatim port of `homodyne/data/`: HDF5 reader, diagonal
  correction (mandatory, three methods: basic / statistical / interpolation),
  multi-level cache (LRU + disk NPZ).
- **JAX-native NLSQ engine** — `nlsq.CurveFit` (trust-region reflective LM,
  end-to-end on device). **Never** calls `scipy.optimize.least_squares`.
- **5-layer anti-degeneracy controller** — `PerAngleScalingPlan`
  (per-angle reparameterization), `HierarchicalOptimizer`,
  `AdaptiveRegularizer`, `GradientCollapseMonitor`,
  `ShearSensitivityWeighting`. Layer 5 is gated by model lineage (active for
  homodyne modes, inert for `two_component`).
- **Memory-aware strategy routing** — `STANDARD` / `OUT_OF_CORE` /
  `HYBRID_STREAMING` selected adaptively from system RAM via `psutil`.
- **CMA-ES escape** — auto-triggers when bound `scale_ratio ≥ 1000`.
  Implementation: `nlsq.CMAESOptimizer` with `evosax` backend + BIPOP restart.
- **Multi-angle heterodyne** — full parity with source heterodyne's
  `fit_nlsq_multi_phi`: per-angle scaling modes `constant` / `averaged` /
  `individual` (resolved from `auto`), plus CMA-ES multi-angle path.
- **Visualization** (`xpcsjax.viz`) — fit-comparison, residual-map, and
  simulated-data plots with an optional Datashader fast path and parallel
  multi-process rendering (`pip install 'xpcsjax[viz-fast]'`).
- **Command-line interface** — console scripts for flag-driven fits, config
  generation/validation, and plotting (see the table below).

### CLI commands

| Command | Purpose |
|---|---|
| `xpcsjax` / `xj` | Run an XPCS NLSQ fit (and standalone QC / simulation plots) |
| `xpcsjax-config` | Generate and validate config templates |
| `xpcsjax-validate` | Validate a config without running a fit |
| `xjexp` / `xjsim` | Experimental-data / simulated-data plotting shortcuts |
| `xpcsjax-post-install` | Install shell completion + XLA activation |
| `xpcsjax-cleanup` | Remove shell completion files |

See the [CLI guide](docs/source/user_guide/cli.rst) for the full reference.

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

---

## Citation

xpcsjax implements the transport-coefficient framework introduced in:

```bibtex
@article{He2024,
  author  = {He, Hongrui and Liang, Hao and Chu, Miaoqi and Jiang, Zhang and
             de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh
             and Chen, Wei},
  title   = {Transport coefficient approach for characterizing nonequilibrium
             dynamics in soft matter},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {121},
  number  = {31},
  year    = {2024},
  doi     = {10.1073/pnas.2401162121}
}

@article{He2025,
  author  = {He, Hongrui and Liang, Heyi and Chu, Miaoqi and Jiang, Zhang and
             de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh
             and Chen, Wei},
  title   = {Bridging microscopic dynamics and rheology in the yielding
             of charged colloidal suspensions},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {122},
  number  = {42},
  year    = {2025},
  doi     = {10.1073/pnas.2514216122}
}
```

## License

MIT. See [`LICENSE`](LICENSE).

## Authors

- Wei Chen (weichen@anl.gov) — Argonne National Laboratory
