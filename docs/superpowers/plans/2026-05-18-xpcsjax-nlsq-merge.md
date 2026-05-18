# xpcsjax — Homodyne + Heterodyne NLSQ Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the `homodyne` and `heterodyne` XPCS analysis packages into a single `xpcsjax` package whose v0.1 exposes a unified NLSQ-only fitting API for both physics models.

**Architecture:** Adopt homodyne's NLSQ engine verbatim (JAX-native trust-region reflective LM via `nlsq.CurveFit`; 5-layer anti-degeneracy controller; memory-aware strategy routing; CMA-ES escape). Two physics models live as peer classes — `HomodyneModel` (3/7 params w/ sinc-shear) and `HeterodyneModel` (14 params, two-component). Single `fit_nlsq(data, config)` entry; dispatch via `analysis_mode` enum. Anti-degeneracy Layer 5 (`ShearSensitivityWeighting`) gated by model lineage — active for `HomodyneModel` only.

**Tech Stack:** Python 3.12+, uv, JAX (float64), `nlsq` ≥ 0.6.4, `evosax`, `h5py`, `numpy`, `scipy` (utilities only — never `scipy.optimize.least_squares`), `interpax`, `pyyaml`, pytest, hypothesis.

**Spec:** `/home/wei/Documents/GitHub/xpcsjax/docs/superpowers/specs/2026-05-18-xpcsjax-nlsq-merge-design.md`

**Source packages (read-only during port):**
- `/home/wei/Documents/GitHub/homodyne/` — primary source for data layer, config, NLSQ engine
- `/home/wei/Documents/GitHub/heterodyne/` — source for `HeterodyneModel` physics only

**Target package:** `/home/wei/Documents/GitHub/xpcsjax/` (currently `main` branch, contains only `LICENSE`, `README.md`, `.gitignore`).

---

## Before You Start

1. Read the spec at `docs/superpowers/specs/2026-05-18-xpcsjax-nlsq-merge-design.md` end-to-end. Pay particular attention to §10 (NLSQ engine) and §10.3 (model-gated anti-degeneracy).
2. Confirm both source packages clone-locally and are clean: `cd /home/wei/Documents/GitHub/homodyne && git status` (expect clean), same for `heterodyne/`.
3. **Hard rule from the spec:** never `import scipy.optimize.least_squares` in xpcsjax source. nlsq's `trf` is JAX-native and lives at `nlsq/core/trf.py`. Run `grep -rn "scipy.optimize.least_squares" xpcsjax/` after any port task — expect zero matches.
4. **Phase 5 is a hard gate.** Do not start Phase 6 until Phase 5 characterization tests pass at `rtol=1e-10` for every homodyne config.

---

## Phase 1 — Skeleton & Dependencies (~1 day)

### Task 1: Create the package directory tree

**Files:**
- Create: `xpcsjax/__init__.py`
- Create: `xpcsjax/config/__init__.py`
- Create: `xpcsjax/data/__init__.py`
- Create: `xpcsjax/core/__init__.py`
- Create: `xpcsjax/optimization/__init__.py`
- Create: `xpcsjax/optimization/nlsq/__init__.py`
- Create: `xpcsjax/optimization/nlsq/strategies/__init__.py`
- Create: `xpcsjax/io/__init__.py`
- Create: `xpcsjax/utils/__init__.py`
- Create: `xpcsjax/device/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/characterization/__init__.py`
- Create: `tests/heterodyne/__init__.py`
- Create: `tests/property/__init__.py`

- [ ] **Step 1: Create all directories**

```bash
cd /home/wei/Documents/GitHub/xpcsjax
mkdir -p xpcsjax/{config,data,core,optimization/nlsq/strategies,io,utils,device}
mkdir -p tests/{characterization,heterodyne,property}
```

- [ ] **Step 2: Create empty `__init__.py` files**

```bash
touch xpcsjax/__init__.py xpcsjax/{config,data,core,io,utils,device}/__init__.py
touch xpcsjax/optimization/__init__.py xpcsjax/optimization/nlsq/__init__.py
touch xpcsjax/optimization/nlsq/strategies/__init__.py
touch tests/__init__.py tests/{characterization,heterodyne,property}/__init__.py
```

- [ ] **Step 3: Verify tree**

```bash
find xpcsjax tests -type d | sort
```

Expected: 11 directories listed (no `.venv` entries should appear in xpcsjax/).

- [ ] **Step 4: Commit**

```bash
git add xpcsjax/ tests/
git commit -m "scaffold: create xpcsjax package directory tree"
```

---

### Task 2: Author `pyproject.toml`

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "xpcsjax"
version = "0.1.0.dev0"
description = "Unified JAX-native XPCS NLSQ fitting (homodyne + heterodyne)"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.12"
authors = [{ name = "Wei Chen", email = "msdsoftmatter@gmail.com" }]
dependencies = [
    "jax",
    "jaxlib",
    "nlsq>=0.6.4",
    "optimistix",
    "optax",
    "evosax",
    "h5py",
    "numpy",
    "scipy",
    "interpax",
    "pyyaml",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov",
    "hypothesis>=6",
    "ruff",
    "mypy",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["xpcsjax"]

[tool.pytest.ini_options]
testpaths = ["tests"]
env = [
    "JAX_ENABLE_X64=1",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "N"]
ignore = ["E501"]  # line length covered by formatter

[tool.mypy]
python_version = "3.12"
strict = false
ignore_missing_imports = true
```

- [ ] **Step 2: Install with uv**

```bash
cd /home/wei/Documents/GitHub/xpcsjax
uv sync --extra dev
```

Expected output: `Resolved N packages` then `Installed N packages`. No error.

- [ ] **Step 3: Verify JAX float64 default**

```bash
uv run python -c "import jax; import jax.numpy as jnp; assert jnp.array(1.0).dtype == jnp.float64, jnp.array(1.0).dtype; print('OK')"
```

Expected output: `OK`.

- [ ] **Step 4: Verify nlsq exposes JAX-native trf**

```bash
uv run python -c "from nlsq.core import trf, trf_jit; print(trf.__file__); print(trf_jit.__file__)"
```

Expected output: two paths under `.venv/lib/python3.12/site-packages/nlsq/core/`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add pyproject.toml with JAX-first deps and uv lockfile"
```

---

### Task 3: Lazy-import top-level `__init__.py`

**Files:**
- Modify: `xpcsjax/__init__.py`

- [ ] **Step 1: Write the failing import test**

Create `tests/test_lazy_imports.py`:

```python
"""Verify top-level imports are lazy and minimal."""
import importlib
import sys


def test_top_level_import_does_not_load_jax():
    """Importing xpcsjax must not eagerly load jax — CLI argument parsing should be instant."""
    # Clean slate
    for mod in list(sys.modules):
        if mod.startswith(("xpcsjax", "jax")):
            del sys.modules[mod]

    importlib.import_module("xpcsjax")

    # nlsq must not have triggered jax import
    assert "jax" not in sys.modules, "jax loaded during `import xpcsjax` — lazy-import broken"


def test_public_exports():
    """The documented public API symbols must be importable."""
    import xpcsjax
    for name in ("load_xpcs_data", "fit_nlsq", "ConfigManager",
                 "HomodyneModel", "HeterodyneModel", "OptimizationResult"):
        assert hasattr(xpcsjax, name), f"missing public export: {name}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_lazy_imports.py -v
```

Expected: FAIL — `test_public_exports` fails with `missing public export: load_xpcs_data`.

- [ ] **Step 3: Implement lazy `__init__.py`**

Write `xpcsjax/__init__.py`:

```python
"""xpcsjax — unified JAX-native XPCS NLSQ fitting.

Public API (lazy-loaded — heavy deps like JAX import on first use):

    from xpcsjax import load_xpcs_data, fit_nlsq, ConfigManager

    data = load_xpcs_data("config.yaml")
    result = fit_nlsq(data, "config.yaml")
    print(result.parameters)
    result.save("output/")
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0.dev0"

_LAZY_EXPORTS = {
    "load_xpcs_data": "xpcsjax.data",
    "fit_nlsq": "xpcsjax.optimization.nlsq",
    "ConfigManager": "xpcsjax.config",
    "HomodyneModel": "xpcsjax.core.models",
    "HeterodyneModel": "xpcsjax.core.heterodyne_model",
    "OptimizationResult": "xpcsjax.io",
}

if TYPE_CHECKING:
    from xpcsjax.config import ConfigManager  # noqa: F401
    from xpcsjax.core.heterodyne_model import HeterodyneModel  # noqa: F401
    from xpcsjax.core.models import HomodyneModel  # noqa: F401
    from xpcsjax.data import load_xpcs_data  # noqa: F401
    from xpcsjax.io import OptimizationResult  # noqa: F401
    from xpcsjax.optimization.nlsq import fit_nlsq  # noqa: F401


def __getattr__(name: str):  # noqa: D401
    """Lazy attribute loader for the documented public API."""
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'xpcsjax' has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS)
```

- [ ] **Step 4: Run test to verify lazy-load assertion passes; public-exports test still fails**

```bash
uv run pytest tests/test_lazy_imports.py::test_top_level_import_does_not_load_jax -v
```

Expected: PASS.

```bash
uv run pytest tests/test_lazy_imports.py::test_public_exports -v
```

Expected: FAIL — `xpcsjax.data` does not yet exist. Leave it failing; later tasks add the modules.

- [ ] **Step 5: Commit**

```bash
git add xpcsjax/__init__.py tests/test_lazy_imports.py
git commit -m "feat(api): lazy-import top-level public API"
```

---

### Task 4: CI scaffold

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write CI workflow**

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.12"
      - name: Sync deps
        run: uv sync --extra dev
      - name: Lint
        run: uv run ruff check .
      - name: Type check
        run: uv run mypy xpcsjax
      - name: Test
        env:
          JAX_ENABLE_X64: "1"
        run: uv run pytest -v
```

- [ ] **Step 2: Verify YAML is valid**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```

Expected: silent (no exception).

- [ ] **Step 3: Run all checks locally**

```bash
cd /home/wei/Documents/GitHub/xpcsjax
uv run ruff check .
uv run mypy xpcsjax
```

Expected: ruff clean; mypy may have a few `note:` lines but no errors.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add lint/type-check/test workflow"
```

---

## Phase 2 — Data layer (verbatim from homodyne, ~2 days)

> **Pattern for verbatim ports in Phases 2–4:**
> 1. `cp` the source file from `/home/wei/Documents/GitHub/homodyne/homodyne/<path>` to `/home/wei/Documents/GitHub/xpcsjax/xpcsjax/<path>`.
> 2. Rewrite imports: replace every `from homodyne.` with `from xpcsjax.` and every `import homodyne.` with `import xpcsjax.`.
> 3. Run smoke import: `uv run python -c "import xpcsjax.<module>"`.
> 4. Commit.
>
> Any code that uses `from heterodyne.` does *not* appear in homodyne — those come in Phase 6 from a different source path.

### Task 5: Port `xpcs_loader.py`

**Files:**
- Copy: `homodyne/data/xpcs_loader.py` → `xpcsjax/data/xpcs_loader.py`
- Test: `tests/data/test_loader_smoke.py`

- [ ] **Step 1: Copy the source file**

```bash
cp /home/wei/Documents/GitHub/homodyne/homodyne/data/xpcs_loader.py \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/data/xpcs_loader.py
```

- [ ] **Step 2: Rewrite imports**

```bash
cd /home/wei/Documents/GitHub/xpcsjax
sed -i 's/from homodyne\./from xpcsjax./g; s/import homodyne\./import xpcsjax./g' \
    xpcsjax/data/xpcs_loader.py
```

- [ ] **Step 3: Verify no homodyne references remain**

```bash
grep -n "homodyne" xpcsjax/data/xpcs_loader.py
```

Expected: no output (zero matches). If matches appear, hand-edit to remove or rename.

- [ ] **Step 4: Smoke import test**

Create `tests/data/__init__.py` (empty), then create `tests/data/test_loader_smoke.py`:

```python
"""Smoke test: xpcs_loader module imports cleanly with no homodyne references."""
import xpcsjax.data.xpcs_loader as loader


def test_module_imports():
    assert loader is not None


def test_load_xpcs_data_callable():
    assert callable(loader.load_xpcs_data)
```

Run:

```bash
uv run pytest tests/data/test_loader_smoke.py -v
```

Expected: PASS or FAIL with ImportError pointing at a missing transitively-imported module — record the missing module; it gets ported in a later step of this Phase. Re-run after each transitive port until both tests pass.

- [ ] **Step 5: Commit**

```bash
git add xpcsjax/data/xpcs_loader.py tests/data/
git commit -m "port: copy homodyne/data/xpcs_loader.py verbatim"
```

---

### Task 6: Port the rest of `data/`

**Files (port each verbatim using the Task 5 pattern):**
- Copy: `homodyne/data/filtering_utils.py` → `xpcsjax/data/filtering_utils.py`
- Copy: `homodyne/data/preprocessing.py` → `xpcsjax/data/preprocessing.py`
- Copy: `homodyne/data/quality_controller.py` → `xpcsjax/data/quality_controller.py`
- Copy: `homodyne/data/performance_engine.py` → `xpcsjax/data/performance_engine.py`
- Copy: `homodyne/data/config.py` → `xpcsjax/data/config.py`
- Modify: `xpcsjax/data/__init__.py` (to re-export `load_xpcs_data`)

- [ ] **Step 1: Copy all files and rewrite imports**

```bash
cd /home/wei/Documents/GitHub/xpcsjax
for f in filtering_utils preprocessing quality_controller performance_engine config; do
  cp /home/wei/Documents/GitHub/homodyne/homodyne/data/${f}.py xpcsjax/data/${f}.py
  sed -i 's/from homodyne\./from xpcsjax./g; s/import homodyne\./import xpcsjax./g' xpcsjax/data/${f}.py
done
```

- [ ] **Step 2: Verify zero homodyne references**

```bash
grep -rn "homodyne" xpcsjax/data/
```

Expected: empty output.

- [ ] **Step 3: Write the `data/__init__.py` re-export**

Replace `xpcsjax/data/__init__.py` with:

```python
"""xpcsjax.data — XPCS HDF5 loading + filtering + diagonal correction + caching."""
from xpcsjax.data.xpcs_loader import load_xpcs_data

__all__ = ["load_xpcs_data"]
```

- [ ] **Step 4: Smoke import**

```bash
uv run python -c "from xpcsjax.data import load_xpcs_data; print(load_xpcs_data)"
```

Expected: `<function load_xpcs_data at 0x...>`. If ImportError, identify the missing transitive dependency (almost always something from `xpcsjax/utils/` or `xpcsjax/device/`) and proceed to Task 7 to provide it before re-running.

- [ ] **Step 5: Commit**

```bash
git add xpcsjax/data/
git commit -m "port: copy homodyne/data/ verbatim (filtering, preprocessing, cache, config)"
```

---

### Task 7: Port `utils/` and `device/`

**Files (verbatim ports — these are dependencies of `data/`):**
- Copy: every `*.py` in `homodyne/utils/` → `xpcsjax/utils/`
- Copy: every `*.py` in `homodyne/device/` → `xpcsjax/device/`

- [ ] **Step 1: Mirror both directories**

```bash
cd /home/wei/Documents/GitHub/xpcsjax
cp -r /home/wei/Documents/GitHub/homodyne/homodyne/utils/. xpcsjax/utils/
cp -r /home/wei/Documents/GitHub/homodyne/homodyne/device/. xpcsjax/device/
```

- [ ] **Step 2: Rewrite imports in both directories**

```bash
find xpcsjax/utils xpcsjax/device -name "*.py" -exec \
    sed -i 's/from homodyne\./from xpcsjax./g; s/import homodyne\./import xpcsjax./g' {} \;
```

- [ ] **Step 3: Verify zero homodyne references**

```bash
grep -rn "homodyne" xpcsjax/utils/ xpcsjax/device/
```

Expected: empty.

- [ ] **Step 4: Re-run data smoke import**

```bash
uv run python -c "from xpcsjax.data import load_xpcs_data; print(load_xpcs_data)"
```

Expected: prints function. If still ImportError, the missing module is one that ports in Phase 3 (config) — note it and continue.

- [ ] **Step 5: Commit**

```bash
git add xpcsjax/utils/ xpcsjax/device/
git commit -m "port: copy homodyne/utils/ and homodyne/device/ verbatim"
```

---

### Task 8: Integration smoke test — load a sample HDF5

**Files:**
- Test: `tests/data/test_loader_integration.py`

- [ ] **Step 1: Locate an existing homodyne test HDF5**

```bash
find /home/wei/Documents/GitHub/homodyne -name "*.h5" -o -name "*.hdf5" | head -3
```

Pick a file ≤ 100 MB (a small fixture). Note its path.

- [ ] **Step 2: Locate the matching homodyne test config**

```bash
find /home/wei/Documents/GitHub/homodyne/tests -name "*.yaml" -path "*static*" | head -3
```

Pick a static-diffusion config that loads the HDF5 from step 1. Note its path.

- [ ] **Step 3: Write the integration test**

Create `tests/data/test_loader_integration.py`:

```python
"""Round-trip load test using a known-good homodyne fixture.

This test depends on a fixture homodyne config; we don't bundle the HDF5 with
xpcsjax — instead we point at the file in the source homodyne repo for now.
The Phase 5 characterization-test infra makes this a permanent fixture.
"""
from pathlib import Path

import numpy as np
import pytest

from xpcsjax.data import load_xpcs_data

# TODO(implementer): substitute the actual file paths discovered in step 1/2
HOMODYNE_FIXTURE_CONFIG = Path(
    "/home/wei/Documents/GitHub/homodyne/tests/fixtures/configs/static_diffusion.yaml"
)


@pytest.mark.skipif(
    not HOMODYNE_FIXTURE_CONFIG.exists(),
    reason="homodyne fixture not present on this machine",
)
def test_load_static_fixture():
    data = load_xpcs_data(str(HOMODYNE_FIXTURE_CONFIG))
    # Sanity invariants for any homodyne XPCS file:
    assert "c2_exp" in data
    assert "phi_angles_list" in data
    assert "t1" in data and "t2" in data
    c2 = data["c2_exp"]
    # c2 shape: (n_q, n_phi, N, N) — must be 4-D float64
    assert c2.ndim == 4
    assert c2.dtype == np.float64
    # Diagonal correction was applied: diagonal entries are not the raw autocorr peak
    n_q, n_phi, N, _ = c2.shape
    assert N == c2.shape[3]
    # Verify time arrays are monotonic
    assert np.all(np.diff(data["t1"]) > 0)
```

- [ ] **Step 4: Run the integration test**

```bash
uv run pytest tests/data/test_loader_integration.py -v
```

Expected: PASS, or SKIPPED if the fixture path is wrong. If FAIL with an error other than skip, debug the port (likely a missing transitive module).

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_loader_integration.py
git commit -m "test(data): add load-and-validate integration smoke test"
```

---

## Phase 3 — Config + HomodyneModel + parameter registry (~2 days)

### Task 9: Port `config/manager.py`

**Files:**
- Copy: `homodyne/config/manager.py` → `xpcsjax/config/manager.py`

- [ ] **Step 1: Copy + rewrite imports**

```bash
cp /home/wei/Documents/GitHub/homodyne/homodyne/config/manager.py \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/config/manager.py
sed -i 's/from homodyne\./from xpcsjax./g; s/import homodyne\./import xpcsjax./g' \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/config/manager.py
```

- [ ] **Step 2: Verify zero homodyne references**

```bash
grep -n "homodyne" xpcsjax/config/manager.py
```

Expected: empty.

- [ ] **Step 3: Smoke import**

```bash
uv run python -c "from xpcsjax.config.manager import ConfigManager; print(ConfigManager)"
```

Expected: `<class 'xpcsjax.config.manager.ConfigManager'>`.

- [ ] **Step 4: Commit**

```bash
git add xpcsjax/config/manager.py
git commit -m "port: copy homodyne/config/manager.py verbatim"
```

---

### Task 10: Port `parameter_registry.py` (the single source of truth)

**Files:**
- Copy: `homodyne/config/parameter_registry.py` → `xpcsjax/config/parameter_registry.py`

- [ ] **Step 1: Read the source to understand its shape**

```bash
head -60 /home/wei/Documents/GitHub/homodyne/homodyne/config/parameter_registry.py
```

Note: this file defines per-`analysis_mode` parameter specs (bounds, defaults, transforms). xpcsjax will extend it in Phase 6 with `TWO_COMPONENT`.

- [ ] **Step 2: Copy + rewrite imports**

```bash
cp /home/wei/Documents/GitHub/homodyne/homodyne/config/parameter_registry.py \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/config/parameter_registry.py
sed -i 's/from homodyne\./from xpcsjax./g; s/import homodyne\./import xpcsjax./g' \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/config/parameter_registry.py
```

- [ ] **Step 3: Smoke import**

```bash
uv run python -c "from xpcsjax.config.parameter_registry import REGISTRY; print(list(REGISTRY))"
```

Expected: a list of analysis-mode enum values, e.g., `[<AnalysisMode.STATIC_DIFFUSION: 'static_diffusion'>, <AnalysisMode.LAMINAR_FLOW: 'laminar_flow'>]`. The exact symbol name in homodyne may differ — adjust the print as needed.

- [ ] **Step 4: Commit**

```bash
git add xpcsjax/config/parameter_registry.py
git commit -m "port: copy homodyne/config/parameter_registry.py verbatim"
```

---

### Task 11: Port `parameter_manager.py` AND fix the registry/manager bounds consistency

**Files:**
- Copy: `homodyne/config/parameter_manager.py` → `xpcsjax/config/parameter_manager.py`
- Test: `tests/config/test_registry_consistency.py`

- [ ] **Step 1: Copy + rewrite imports**

```bash
cp /home/wei/Documents/GitHub/homodyne/homodyne/config/parameter_manager.py \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/config/parameter_manager.py
sed -i 's/from homodyne\./from xpcsjax./g; s/import homodyne\./import xpcsjax./g' \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/config/parameter_manager.py
```

- [ ] **Step 2: Write the consistency test**

Create `tests/config/__init__.py` (empty), then `tests/config/test_registry_consistency.py`:

```python
"""parameter_manager must derive all bounds from parameter_registry.

This guards against the historical bug where parameter_manager.py declared
its own contrast bounds that disagreed with the registry."""
from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import REGISTRY


def test_no_inline_bound_constants_in_manager():
    """parameter_manager.py source must not redeclare bounds — all bounds come from REGISTRY."""
    import inspect

    src = inspect.getsource(ParameterManager)
    # Heuristic: any literal tuple of two numerics in the source is suspicious if it
    # contains numbers that look like bounds (e.g., (0.01, 1.5) for contrast).
    # The legitimate uses of bound tuples come from REGISTRY lookups, not literals.
    suspicious = [
        line
        for line in src.splitlines()
        if "bounds" in line.lower() and "=" in line and "(" in line and "REGISTRY" not in line
    ]
    assert not suspicious, (
        f"parameter_manager.py contains literal bounds — must read from REGISTRY:\n"
        + "\n".join(suspicious)
    )


def test_manager_bounds_match_registry_for_all_modes():
    """For every analysis_mode + param, manager.get_bounds() == REGISTRY entry bounds."""
    for mode, specs in REGISTRY.items():
        pm = ParameterManager(analysis_mode=mode)
        for spec in specs:
            mgr_bounds = pm.get_bounds(spec.name)
            assert mgr_bounds == spec.bounds, (
                f"mismatch for {mode.value}.{spec.name}: "
                f"manager={mgr_bounds}, registry={spec.bounds}"
            )
```

- [ ] **Step 3: Run the test to discover the inconsistency**

```bash
uv run pytest tests/config/test_registry_consistency.py -v
```

Expected: either PASS (homodyne already fixed it upstream) or FAIL listing the divergent bound. **If FAIL**, proceed to Step 4. If PASS, skip to Step 5.

- [ ] **Step 4: Fix `parameter_manager.py` to read every bound from the registry**

Open `xpcsjax/config/parameter_manager.py`. Locate any module-level dict or method-local literal that hardcodes a bound (the contrast bound is the historical culprit; search for `contrast` and `(0.01,` or similar). Replace each literal lookup with a registry lookup:

```python
# Before (typical pattern):
self._bounds = {"contrast": (0.01, 1.5), ...}

# After:
self._bounds = {spec.name: spec.bounds for spec in REGISTRY[self.analysis_mode]}
```

Re-run the test:

```bash
uv run pytest tests/config/test_registry_consistency.py -v
```

Expected: PASS.

- [ ] **Step 5: Smoke import + integration**

```bash
uv run python -c "from xpcsjax.config import ConfigManager; from xpcsjax.config.parameter_manager import ParameterManager; print(ParameterManager)"
```

Expected: prints class.

- [ ] **Step 6: Wire `config/__init__.py`**

Replace `xpcsjax/config/__init__.py` with:

```python
"""xpcsjax.config — configuration management for both physics models."""
from xpcsjax.config.manager import ConfigManager
from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import REGISTRY

__all__ = ["ConfigManager", "ParameterManager", "REGISTRY"]
```

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/config/ tests/config/
git commit -m "port: copy homodyne/config/ + enforce registry-as-source-of-truth"
```

---

### Task 12: Port `core/diagonal_correction.py`

**Files:**
- Copy: `homodyne/core/diagonal_correction.py` → `xpcsjax/core/diagonal_correction.py`
- Test: `tests/property/test_diagonal_correction.py`

- [ ] **Step 1: Copy + rewrite imports**

```bash
cp /home/wei/Documents/GitHub/homodyne/homodyne/core/diagonal_correction.py \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/core/diagonal_correction.py
sed -i 's/from homodyne\./from xpcsjax./g; s/import homodyne\./import xpcsjax./g' \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/core/diagonal_correction.py
```

- [ ] **Step 2: Write the property test**

Create `tests/property/test_diagonal_correction.py`:

```python
"""Diagonal correction is mandatory for both physics models.

Property: after correction, c2[i, i] equals the interpolated value from
its off-diagonal neighbors (within method tolerance), regardless of method."""
import numpy as np
import pytest

from xpcsjax.core.diagonal_correction import apply_diagonal_correction


@pytest.mark.parametrize("method", ["basic", "statistical", "interpolation"])
def test_diagonal_is_replaced(method):
    rng = np.random.default_rng(seed=42)
    N = 32
    c2 = rng.uniform(0.5, 1.5, size=(N, N))
    c2 = (c2 + c2.T) / 2  # symmetrize
    # Spike the diagonal to simulate the autocorrelation peak
    c2[np.arange(N), np.arange(N)] = 5.0

    corrected = apply_diagonal_correction(c2, method=method, width=1)

    diag_max = np.max(np.abs(np.diag(corrected)))
    assert diag_max < 2.5, (
        f"method={method}: diagonal still contains autocorr-peak-magnitude values "
        f"(max abs diag = {diag_max})"
    )


def test_off_diagonal_preserved():
    """Correction must NOT modify off-diagonal entries."""
    rng = np.random.default_rng(seed=7)
    N = 16
    c2 = rng.uniform(0.5, 1.5, size=(N, N))
    c2 = (c2 + c2.T) / 2

    corrected = apply_diagonal_correction(c2.copy(), method="basic", width=1)

    off_diag_mask = ~np.eye(N, dtype=bool)
    np.testing.assert_allclose(corrected[off_diag_mask], c2[off_diag_mask])
```

- [ ] **Step 3: Run the test**

```bash
uv run pytest tests/property/test_diagonal_correction.py -v
```

Expected: PASS (all three parametrized cases + the off-diagonal-preserved test).

- [ ] **Step 4: Commit**

```bash
git add xpcsjax/core/diagonal_correction.py tests/property/test_diagonal_correction.py
git commit -m "port: copy homodyne/core/diagonal_correction.py + property tests"
```

---

### Task 13: Port `core/models.py` — `DiffusionModel`, `ShearModel`, `CombinedModel`

**Files:**
- Copy: `homodyne/core/models.py` → `xpcsjax/core/models.py`
- **Trim:** drop any CMC-only g₂ code paths during the port.

- [ ] **Step 1: Copy the file**

```bash
cp /home/wei/Documents/GitHub/homodyne/homodyne/core/models.py \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/core/models.py
sed -i 's/from homodyne\./from xpcsjax./g; s/import homodyne\./import xpcsjax./g' \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/core/models.py
```

- [ ] **Step 2: Identify CMC-only blocks**

```bash
grep -n -E "cmc|CMC|numpyro|NumPyro|mcmc|MCMC|posterior|prior" xpcsjax/core/models.py | head -40
```

For each match, inspect the surrounding function/class. CMC paths are typically:
- Functions named `*_cmc_*` or `*_numpyro_*`
- Methods that take a `rng_key` argument (JAX PRNGKey — only NumPyro models use this)
- Decorators or branches gated on `if config.method == "cmc"` or similar

Make a list of line ranges to remove.

- [ ] **Step 3: Remove CMC-only code**

Open `xpcsjax/core/models.py` in your editor. For each line range identified in Step 2, delete the function/method/block. **Do not** delete the NLSQ-analytic, NLSQ-JAX, or test code paths from the shadow-copy registry — only the CMC paths (`cmc_precomputed`, `cmc_shard`).

After deletion, the file should have 2–3 g₂/g₁ implementations registered (NLSQ analytic, NLSQ JAX, test path) instead of 5.

- [ ] **Step 4: Smoke import + verify no CMC residue**

```bash
uv run python -c "from xpcsjax.core.models import PhysicsModelBase, DiffusionModel, CombinedModel; print(DiffusionModel, CombinedModel)"
```

Expected: prints the two classes.

```bash
grep -n -E "cmc|CMC|numpyro|NumPyro" xpcsjax/core/models.py
```

Expected: empty.

- [ ] **Step 5: Run an inline residual roundtrip**

Create `tests/core/__init__.py` (empty), then `tests/core/test_homodyne_models.py`:

```python
"""Smoke tests for HomodyneModel classes (DiffusionModel, CombinedModel)."""
import jax.numpy as jnp
import numpy as np

from xpcsjax.core.models import DiffusionModel, CombinedModel
from xpcsjax.config.parameter_registry import REGISTRY


def test_diffusion_model_param_count():
    model = DiffusionModel()
    assert len(model.param_names) == 3
    assert set(model.param_names) == {"D0", "alpha", "D_offset"}


def test_combined_model_param_count():
    model = CombinedModel()
    assert len(model.param_names) == 7
    expected = {"D0", "alpha", "D_offset",
                "gamma_dot_0", "beta", "gamma_dot_offset", "phi0"}
    assert set(model.param_names) == expected


def test_diffusion_model_residual_runs():
    """Residual must return a finite 1-D array for any in-bounds params."""
    model = DiffusionModel()
    params = jnp.array([1.0e3, 0.0, 0.0])  # in-bounds defaults
    # Build minimal data dict matching loader output schema
    N = 8
    data = {
        "c2_exp": np.ones((1, 1, N, N), dtype=np.float64) * 1.05,
        "t1": np.arange(N, dtype=np.float64),
        "t2": np.arange(N, dtype=np.float64),
        "wavevector_q_list": np.array([0.01]),
        "phi_angles_list": np.array([0.0]),
        "dt": 1.0,
    }
    res = model.compute_residual(params, data, ctx={})
    assert res.ndim == 1
    assert jnp.all(jnp.isfinite(res))
```

- [ ] **Step 6: Run the tests**

```bash
uv run pytest tests/core/test_homodyne_models.py -v
```

Expected: PASS. If any assertion about `param_names` fails, the actual symbol names in homodyne's `models.py` differ from the spec — update the test to match the source (and note the discrepancy in the implementation log).

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/core/models.py tests/core/
git commit -m "port: copy homodyne/core/models.py minus CMC-only g2 paths"
```

---

### Task 14: Port `core/kernels.py` — NLSQ paths only

**Files:**
- Copy: `homodyne/core/kernels.py` → `xpcsjax/core/kernels.py`
- **Trim:** drop CMC kernel entries from the registry.

- [ ] **Step 1: Copy + rewrite imports**

```bash
cp /home/wei/Documents/GitHub/homodyne/homodyne/core/kernels.py \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/core/kernels.py
sed -i 's/from homodyne\./from xpcsjax./g; s/import homodyne\./import xpcsjax./g' \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/core/kernels.py
```

- [ ] **Step 2: Identify the shadow-copy registry**

```bash
grep -n -E "register|registry|REGISTRY|@jit|_cmc_|_test_" xpcsjax/core/kernels.py | head -30
```

Look for a module-level dict/list that registers kernel implementations under labels like `"nlsq_analytic"`, `"nlsq_jax"`, `"cmc_precomputed"`, `"cmc_shard"`, `"test"`.

- [ ] **Step 3: Remove CMC-only entries**

Open `xpcsjax/core/kernels.py`. Delete the registration of `"cmc_precomputed"` and `"cmc_shard"` (or whatever names appear). Also delete the corresponding function/method definitions if they're not referenced elsewhere.

Leave: `"nlsq_analytic"`, `"nlsq_jax"`, `"test"` (or equivalent).

- [ ] **Step 4: Smoke import**

```bash
uv run python -c "from xpcsjax.core import kernels; print(kernels)"
```

Expected: no exception.

- [ ] **Step 5: Verify registry shrunk**

```bash
uv run python -c "from xpcsjax.core import kernels; print([k for k in dir(kernels) if 'REGISTRY' in k.upper() or 'KERNEL' in k.upper()])"
```

If a kernel registry is exported, inspect its keys (3 entries, not 5).

- [ ] **Step 6: Re-run homodyne model tests**

```bash
uv run pytest tests/core/test_homodyne_models.py -v
```

Expected: still PASS.

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/core/kernels.py
git commit -m "port: copy homodyne/core/kernels.py minus CMC-only shadow-copy entries"
```

---

### Task 15: Wire `core/__init__.py`

**Files:**
- Modify: `xpcsjax/core/__init__.py`

- [ ] **Step 1: Write the module exports**

Replace `xpcsjax/core/__init__.py` with:

```python
"""xpcsjax.core — physics models, diagonal correction, JAX g1/g2 kernels."""
from xpcsjax.core.models import (
    CombinedModel,
    DiffusionModel,
    PhysicsModelBase,
)

# Re-export under the v0.1 public name expected by xpcsjax.__init__
HomodyneModel = CombinedModel  # primary homodyne model (laminar_flow); also handles static via parameter masking

__all__ = [
    "PhysicsModelBase",
    "DiffusionModel",
    "CombinedModel",
    "HomodyneModel",
]
```

Note: if homodyne's source already exposes a `HomodyneModel` alias, use that import directly instead of the alias above.

- [ ] **Step 2: Verify top-level lazy import resolves**

```bash
uv run python -c "from xpcsjax import HomodyneModel; print(HomodyneModel)"
```

Expected: prints a class.

- [ ] **Step 3: Commit**

```bash
git add xpcsjax/core/__init__.py
git commit -m "feat(core): export PhysicsModelBase, DiffusionModel, CombinedModel, HomodyneModel"
```

---

## Phase 4 — NLSQ engine (verbatim from homodyne, ~2 days)

### Task 16: Port `optimization/nlsq/memory.py` (strategy router)

**Files:**
- Copy: `homodyne/optimization/nlsq/memory.py` → `xpcsjax/optimization/nlsq/memory.py`

- [ ] **Step 1: Copy + rewrite imports**

```bash
cp /home/wei/Documents/GitHub/homodyne/homodyne/optimization/nlsq/memory.py \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/optimization/nlsq/memory.py
sed -i 's/from homodyne\./from xpcsjax./g; s/import homodyne\./import xpcsjax./g' \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/optimization/nlsq/memory.py
```

- [ ] **Step 2: Smoke import + threshold sanity**

```bash
uv run python -c "from xpcsjax.optimization.nlsq.memory import select_nlsq_strategy; print(select_nlsq_strategy(1_000_000, 3))"
```

Expected: a strategy enum value (likely `STANDARD` for that small input).

- [ ] **Step 3: Commit**

```bash
git add xpcsjax/optimization/nlsq/memory.py
git commit -m "port: copy homodyne/optimization/nlsq/memory.py verbatim"
```

---

### Task 17: Port `optimization/nlsq/strategies/`

**Files:**
- Copy: every `*.py` in `homodyne/optimization/nlsq/strategies/`.

- [ ] **Step 1: Mirror the strategies directory**

```bash
cd /home/wei/Documents/GitHub/xpcsjax
cp /home/wei/Documents/GitHub/homodyne/homodyne/optimization/nlsq/strategies/*.py \
   xpcsjax/optimization/nlsq/strategies/
find xpcsjax/optimization/nlsq/strategies -name "*.py" -exec \
   sed -i 's/from homodyne\./from xpcsjax./g; s/import homodyne\./import xpcsjax./g' {} \;
```

- [ ] **Step 2: Verify zero homodyne references**

```bash
grep -rn "homodyne" xpcsjax/optimization/nlsq/strategies/
```

Expected: empty.

- [ ] **Step 3: Smoke import every strategy module**

```bash
uv run python -c "
from xpcsjax.optimization.nlsq.strategies import chunking, residual, executors
print(chunking, residual, executors)
"
```

Expected: three module reprs.

- [ ] **Step 4: Commit**

```bash
git add xpcsjax/optimization/nlsq/strategies/
git commit -m "port: copy homodyne/optimization/nlsq/strategies/ verbatim"
```

---

### Task 18: Port the rest of `optimization/nlsq/` — adapter, core, monitor, controller, cmaes, multistart

**Files (verbatim ports, batched):**
- Copy: `homodyne/optimization/nlsq/adapter.py`
- Copy: `homodyne/optimization/nlsq/core.py`
- Copy: `homodyne/optimization/nlsq/gradient_monitor.py`
- Copy: `homodyne/optimization/nlsq/anti_degeneracy_controller.py`
- Copy: `homodyne/optimization/nlsq/cmaes_wrapper.py`
- Copy: any `multistart*.py` files in the source

- [ ] **Step 1: List all remaining source files**

```bash
ls /home/wei/Documents/GitHub/homodyne/homodyne/optimization/nlsq/*.py | grep -v "memory.py" | grep -v __init__
```

Capture the output. Each file gets copied in Step 2.

- [ ] **Step 2: Copy every remaining `*.py`**

```bash
cd /home/wei/Documents/GitHub/xpcsjax
for f in $(ls /home/wei/Documents/GitHub/homodyne/homodyne/optimization/nlsq/*.py | grep -v __init__ | grep -v memory.py); do
    base=$(basename "$f")
    cp "$f" xpcsjax/optimization/nlsq/${base}
done
find xpcsjax/optimization/nlsq -maxdepth 1 -name "*.py" -exec \
   sed -i 's/from homodyne\./from xpcsjax./g; s/import homodyne\./import xpcsjax./g' {} \;
```

- [ ] **Step 3: Verify zero homodyne references**

```bash
grep -rn "homodyne" xpcsjax/optimization/nlsq/
```

Expected: empty.

- [ ] **Step 4: Verify the hard rule — no scipy LM imports**

```bash
grep -rn "scipy.optimize.least_squares\|from scipy.optimize import least_squares" xpcsjax/
```

Expected: empty. If any match appears, it's either a comment/docstring (acceptable but inspect) or a real import (must be removed/replaced before continuing — the LM step must stay JAX-native via `nlsq.CurveFit`).

- [ ] **Step 5: Smoke import all engine modules**

```bash
uv run python -c "
from xpcsjax.optimization.nlsq import core, adapter, gradient_monitor
from xpcsjax.optimization.nlsq import anti_degeneracy_controller, cmaes_wrapper
print('engine modules import OK')
"
```

Expected: `engine modules import OK`.

- [ ] **Step 6: Commit**

```bash
git add xpcsjax/optimization/nlsq/
git commit -m "port: copy homodyne/optimization/nlsq/ engine modules verbatim"
```

---

### Task 19: Port `io/results_nlsq.py`

**Files:**
- Copy: `homodyne/io/results_nlsq.py` → `xpcsjax/io/results_nlsq.py`

- [ ] **Step 1: Copy + rewrite imports**

```bash
cp /home/wei/Documents/GitHub/homodyne/homodyne/io/results_nlsq.py \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/io/results_nlsq.py
sed -i 's/from homodyne\./from xpcsjax./g; s/import homodyne\./import xpcsjax./g' \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/io/results_nlsq.py
```

- [ ] **Step 2: Wire `io/__init__.py`**

Replace `xpcsjax/io/__init__.py` with:

```python
"""xpcsjax.io — result serialization (NLSQ only in v0.1)."""
from xpcsjax.io.results_nlsq import OptimizationResult

__all__ = ["OptimizationResult"]
```

- [ ] **Step 3: Smoke import + lazy-load top-level**

```bash
uv run python -c "from xpcsjax import OptimizationResult; print(OptimizationResult)"
```

Expected: prints the dataclass.

- [ ] **Step 4: Commit**

```bash
git add xpcsjax/io/
git commit -m "port: copy homodyne/io/results_nlsq.py + wire io.__init__"
```

---

### Task 20: Wire the single-entry `fit_nlsq` API

**Files:**
- Modify: `xpcsjax/optimization/nlsq/__init__.py`
- Test: `tests/test_public_api.py`

- [ ] **Step 1: Identify homodyne's NLSQ entry-point name**

```bash
grep -n "^def fit_nlsq\|^def fit_nlsq_jax" /home/wei/Documents/GitHub/xpcsjax/xpcsjax/optimization/nlsq/core.py
```

The function is typically called `fit_nlsq_jax`. xpcsjax exposes it as `fit_nlsq`.

- [ ] **Step 2: Write the wrapper**

Append to `xpcsjax/optimization/nlsq/__init__.py`:

```python
"""xpcsjax.optimization.nlsq — JAX-native NLSQ fitting engine."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from xpcsjax.optimization.nlsq.core import fit_nlsq_jax

if TYPE_CHECKING:
    from xpcsjax.config import ConfigManager
    from xpcsjax.io import OptimizationResult


def fit_nlsq(data, config):
    """Single-entry NLSQ fit for both physics models.

    Parameters
    ----------
    data : dict
        XPCS data dict returned by ``xpcsjax.data.load_xpcs_data``.
    config : str | Path | ConfigManager
        Either a path to a YAML config file or a pre-built ConfigManager.

    Returns
    -------
    OptimizationResult
        Fit parameters, covariance, diagnostics, and metadata.
    """
    if isinstance(config, (str, Path)):
        from xpcsjax.config import ConfigManager

        config = ConfigManager(str(config))
    return fit_nlsq_jax(data, config)


__all__ = ["fit_nlsq"]
```

- [ ] **Step 3: Write the public-API test**

Update `tests/test_lazy_imports.py` so `test_public_exports` now passes:

```bash
uv run pytest tests/test_lazy_imports.py -v
```

Expected: BOTH tests PASS.

- [ ] **Step 4: Commit**

```bash
git add xpcsjax/optimization/nlsq/__init__.py
git commit -m "feat(api): single-entry fit_nlsq(data, config) wrapper"
```

---

### Task 20a: Engine verification — memory-aware strategy routing

> **Why this exists:** The Phase 5 characterization gate would catch any
> regression in `select_nlsq_strategy`, but the failure would surface as
> "strategy_used mismatch in baseline X" without isolating *which* part of
> the router is wrong. Tasks 20a–20c add direct unit tests so any engine-
> layer regression localizes to the offending feature before the gate runs.

**Files:**
- Create: `tests/optimization/test_memory_routing.py`

- [ ] **Step 1: Create `tests/optimization/__init__.py` (empty) if not already present**

```bash
test -f tests/optimization/__init__.py || touch tests/optimization/__init__.py
```

- [ ] **Step 2: Write the routing unit tests**

Create `tests/optimization/test_memory_routing.py`:

```python
"""Direct unit tests for memory-aware NLSQ strategy routing.

Localizes router regressions ahead of the Phase 5 characterization gate."""
import os

from xpcsjax.optimization.nlsq.memory import select_nlsq_strategy


def _strategy_name(s) -> str:
    """Normalize the returned value (enum, string, or named constant) to upper-case name."""
    return getattr(s, "name", str(s)).upper()


def test_small_data_routes_to_standard():
    """Small datasets fit in memory — STANDARD strategy."""
    strategy = select_nlsq_strategy(n_points=10_000, n_params=3)
    name = _strategy_name(strategy)
    assert "STANDARD" in name, f"expected STANDARD, got {name}"


def test_huge_data_routes_to_streaming():
    """Datasets that vastly exceed RAM trigger streaming."""
    strategy = select_nlsq_strategy(n_points=200_000_000, n_params=14)
    name = _strategy_name(strategy)
    assert "STREAM" in name or "HYBRID" in name, (
        f"expected HYBRID_STREAMING, got {name}"
    )


def test_medium_data_routes_to_chunked_or_streaming():
    """Datasets exceeding peak Jacobian threshold escalate beyond STANDARD.

    Exact thresholds are adaptive on system RAM (default 16 GB), so this test
    asserts only that the router escalated — landing in either OUT_OF_CORE or
    HYBRID_STREAMING is acceptable depending on the host."""
    strategy = select_nlsq_strategy(n_points=10_000_000, n_params=14)
    name = _strategy_name(strategy)
    assert any(token in name for token in ("OUT_OF_CORE", "CHUNK", "STREAM", "HYBRID")), (
        f"expected escalation beyond STANDARD, got {name}"
    )


def test_memory_fraction_env_override_is_honored():
    """NLSQ_MEMORY_FRACTION env var must be readable by the router without exception.

    The exact strategy change depends on system RAM — we verify only that the env
    var is parsed and applied without crashing."""
    default = select_nlsq_strategy(n_points=2_000_000, n_params=3)
    os.environ["NLSQ_MEMORY_FRACTION"] = "0.1"
    try:
        with_override = select_nlsq_strategy(n_points=2_000_000, n_params=3)
    finally:
        del os.environ["NLSQ_MEMORY_FRACTION"]

    valid_strategies = {"STANDARD", "OUT_OF_CORE", "HYBRID_STREAMING"}
    assert _strategy_name(default) in valid_strategies
    assert _strategy_name(with_override) in valid_strategies
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/optimization/test_memory_routing.py -v
```

Expected: all PASS. If `_strategy_name` cannot extract a recognizable string from the return type, inspect `select_nlsq_strategy`'s actual return type and update the helper to match (it might be a `StrategyInfo` dataclass with a `.strategy` attribute, an `Enum`, or a plain string).

- [ ] **Step 4: Commit**

```bash
git add tests/optimization/test_memory_routing.py tests/optimization/__init__.py
git commit -m "test(nlsq): direct unit tests for memory-aware strategy routing"
```

---

### Task 20b: Engine verification — 5-layer anti-degeneracy controller presence

**Files:**
- Create: `tests/optimization/test_anti_degeneracy_layers.py`

- [ ] **Step 1: Write the layer-existence test**

Create `tests/optimization/test_anti_degeneracy_layers.py`:

```python
"""Verify all 5 anti-degeneracy layers ported over from homodyne.

Task 29 tests the Layer-5 model-lineage gating. This test catches a different
regression: did the CMC-path trimming in Task 13/14 accidentally cut into the
anti-degeneracy controller's layer wiring?"""
import inspect

from xpcsjax.core import HomodyneModel
from xpcsjax.optimization.nlsq.anti_degeneracy_controller import AntiDegeneracyController


LAYER_NAMES = (
    "FourierReparameterizer",
    "HierarchicalOptimizer",
    "AdaptiveRegularizer",
    "GradientCollapseMonitor",
    "ShearSensitivityWeighting",
)


def test_controller_source_references_all_5_layers():
    """Static check: the controller class source must mention every layer name.

    If a layer class was dropped during the verbatim port, this catches it
    without needing to instantiate or introspect the controller's runtime state."""
    src = inspect.getsource(AntiDegeneracyController)
    missing = [name for name in LAYER_NAMES if name not in src]
    assert not missing, (
        f"AntiDegeneracyController source missing references to: {missing}. "
        f"Likely cause: a layer was dropped during the verbatim port "
        f"or during the CMC trim in Task 13/14."
    )


def test_controller_instantiates_on_homodyne():
    """The controller must accept a HomodyneModel and construct without error."""
    model = HomodyneModel()
    controller = AntiDegeneracyController(model=model)
    assert controller is not None
    assert controller.model is model


def test_controller_exposes_layer_pipeline():
    """The controller must expose its internal layer pipeline for inspection.

    Tries common attribute conventions — at least one must hold the layer set."""
    model = HomodyneModel()
    controller = AntiDegeneracyController(model=model)

    pipeline = None
    for attr in ("layers", "_layers", "pipeline", "_pipeline", "stages"):
        candidate = getattr(controller, attr, None)
        if candidate is not None:
            pipeline = candidate
            break
    assert pipeline is not None, (
        "controller has no discoverable layer pipeline — checked "
        "`layers`, `_layers`, `pipeline`, `_pipeline`, `stages`. "
        "Adapt this test to the actual attribute name once located."
    )

    if hasattr(pipeline, "keys"):
        keys = set(pipeline.keys())
        found = sum(1 for name in LAYER_NAMES if name in keys)
        assert found >= 5, f"pipeline dict has only {found}/5 layer keys: {keys}"
    else:
        assert len(pipeline) >= 5, (
            f"pipeline has only {len(pipeline)} stages, expected ≥ 5"
        )
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/optimization/test_anti_degeneracy_layers.py -v
```

Expected: `test_controller_source_references_all_5_layers` PASS, `test_controller_instantiates_on_homodyne` PASS. The third test may need adaptation — once you locate the actual pipeline attribute (likely visible via `dir(controller)` after instantiation), update the loop's attribute list.

- [ ] **Step 3: Commit**

```bash
git add tests/optimization/test_anti_degeneracy_layers.py
git commit -m "test(nlsq): direct unit tests for 5-layer anti-degeneracy controller presence"
```

---

### Task 20c: Engine verification — CMA-ES trigger threshold

**Files:**
- Create: `tests/optimization/test_cmaes_trigger.py`

- [ ] **Step 1: Write the trigger-threshold test**

Create `tests/optimization/test_cmaes_trigger.py`:

```python
"""CMA-ES auto-triggers at scale_ratio >= 1000 (homodyne default).

XPCS multi-scale problems span >3 orders of magnitude (e.g., D0 ~ 1e4 vs
gamma_dot ~ 1e-3 → ratio ~ 1e7). This is the documented escape hatch; we
verify it directly so a regression localizes to the trigger function rather
than only surfacing via characterization."""
import inspect

import pytest

from xpcsjax.optimization.nlsq.cmaes_wrapper import should_use_cmaes


def _call_with_either_api(scale_ratio_value, bounds_value):
    """should_use_cmaes API may accept scale_ratio directly OR a bounds list.

    Try the kwarg form first, fall back to positional."""
    try:
        return should_use_cmaes(scale_ratio=scale_ratio_value)
    except TypeError:
        pass
    try:
        return should_use_cmaes(bounds=bounds_value)
    except TypeError:
        return should_use_cmaes(bounds_value)


def test_high_scale_ratio_triggers_cmaes():
    """scale_ratio = 1.5e6 must enable CMA-ES (well above 1000 threshold)."""
    result = _call_with_either_api(
        scale_ratio_value=1_500_000.0,
        bounds_value=[(1e-3, 1.0e3)],  # ratio 1e6
    )
    assert result is True, "scale_ratio >> 1000 must enable CMA-ES"


def test_low_scale_ratio_does_not_trigger():
    """scale_ratio = 10 must NOT enable CMA-ES."""
    result = _call_with_either_api(
        scale_ratio_value=10.0,
        bounds_value=[(1.0, 10.0)],  # ratio 10
    )
    assert result is False, "scale_ratio << 1000 must not enable CMA-ES"


def test_default_threshold_is_1000():
    """The documented default scale_threshold is 1000.0.

    Threshold may be a function parameter OR a module-level constant; check both."""
    sig = inspect.signature(should_use_cmaes)
    threshold_param = next(
        (p for name, p in sig.parameters.items()
         if "threshold" in name.lower() or "scale_thr" in name.lower()),
        None,
    )
    if threshold_param is not None and threshold_param.default is not inspect.Parameter.empty:
        assert threshold_param.default == pytest.approx(1000.0), (
            f"default scale_threshold drifted from documented 1000.0 to "
            f"{threshold_param.default}"
        )
        return

    # Fall back: look for a module-level constant
    from xpcsjax.optimization.nlsq import cmaes_wrapper

    for constant_name in ("DEFAULT_SCALE_THRESHOLD", "SCALE_THRESHOLD",
                          "CMAES_SCALE_THRESHOLD"):
        if hasattr(cmaes_wrapper, constant_name):
            value = getattr(cmaes_wrapper, constant_name)
            assert value == pytest.approx(1000.0), (
                f"{constant_name} drifted from 1000.0 to {value}"
            )
            return

    pytest.skip(
        "could not locate scale_threshold as function param or module constant — "
        "inspect cmaes_wrapper.py manually and update this test to point at the "
        "actual location of the threshold value."
    )
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/optimization/test_cmaes_trigger.py -v
```

Expected: first two tests PASS. `test_default_threshold_is_1000` PASS or SKIP depending on where the threshold lives. If SKIPped, follow the skip message to locate the threshold and update the test to assert on the actual value.

- [ ] **Step 3: Run all three engine-verification tests together**

```bash
uv run pytest tests/optimization/ -v
```

Expected: all PASS. This is the engine-feature-localization gate — every Task 20a/b/c assertion must pass before Phase 5 baselines are generated.

- [ ] **Step 4: Commit**

```bash
git add tests/optimization/test_cmaes_trigger.py
git commit -m "test(nlsq): direct unit test for CMA-ES auto-trigger threshold"
```

---

## Phase 5 — Homodyne Characterization GATE (~1 day)

### Task 21: Generate homodyne baselines from the source package

**Files:**
- Create: `tests/characterization/conftest.py`
- Create: `tests/characterization/fixtures/configs/` (mirror homodyne test configs)
- Create: `tests/characterization/fixtures/baselines/` (frozen fit outputs)
- Create: `scripts/generate_homodyne_baselines.py`

- [ ] **Step 1: Identify the homodyne test configs to characterize**

```bash
find /home/wei/Documents/GitHub/homodyne/tests -name "*.yaml" | head -20
```

Pick a representative set: at least 3 static-diffusion configs and 3 laminar-flow configs covering small / medium / large data sizes (to exercise STANDARD / OUT_OF_CORE / HYBRID_STREAMING strategy routing).

- [ ] **Step 2: Copy the chosen configs into the fixture directory**

```bash
mkdir -p tests/characterization/fixtures/configs
mkdir -p tests/characterization/fixtures/baselines
# adjust paths to the configs identified in Step 1:
cp /home/wei/Documents/GitHub/homodyne/tests/<...>/<each>.yaml \
   tests/characterization/fixtures/configs/
```

- [ ] **Step 3: Write the baseline-generation script**

Create `scripts/generate_homodyne_baselines.py`:

```python
"""Generate homodyne fit baselines by running the SOURCE homodyne package.

These pinned outputs are what xpcsjax must reproduce bit-equivalently (rtol=1e-10).
Run this script ONCE in a venv that has the source homodyne installed, then commit
the resulting JSON/NPZ files under tests/characterization/fixtures/baselines/.
"""
from __future__ import annotations

import json
from pathlib import Path

# Import from the SOURCE homodyne package (not xpcsjax)
from homodyne.config import ConfigManager
from homodyne.data import load_xpcs_data
from homodyne.optimization import fit_nlsq_jax

CONFIGS_DIR = Path("tests/characterization/fixtures/configs")
BASELINES_DIR = Path("tests/characterization/fixtures/baselines")
BASELINES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    for config_path in sorted(CONFIGS_DIR.glob("*.yaml")):
        print(f"== {config_path.name}")
        cfg = ConfigManager(str(config_path))
        data = load_xpcs_data(str(config_path))
        result = fit_nlsq_jax(data, cfg)

        out = BASELINES_DIR / config_path.with_suffix(".json").name
        with out.open("w") as f:
            json.dump(
                {
                    "parameters": dict(result.parameters),
                    "parameter_errors": dict(result.parameter_errors),
                    "chi_squared": float(result.chi_squared),
                    "r_squared": float(result.r_squared),
                    "dof": int(result.dof),
                    "convergence_status": str(result.convergence_status),
                    "n_iterations": int(result.n_iterations),
                    "model_metadata": result.model_metadata,
                },
                f,
                indent=2,
            )
        print(f"   wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the baseline generator**

```bash
cd /home/wei/Documents/GitHub/xpcsjax
JAX_ENABLE_X64=1 uv run --with /home/wei/Documents/GitHub/homodyne \
    python scripts/generate_homodyne_baselines.py
```

Expected: one JSON file per config under `tests/characterization/fixtures/baselines/`. If `uv run --with` does not pick up the source homodyne package, fall back to creating a temporary venv with `homodyne` installed and running the script there.

- [ ] **Step 5: Verify baselines look reasonable**

```bash
ls tests/characterization/fixtures/baselines/
head -30 tests/characterization/fixtures/baselines/*.json | head -60
```

Expected: human-readable parameter dicts.

- [ ] **Step 6: Commit baselines**

```bash
git add tests/characterization/fixtures/ scripts/generate_homodyne_baselines.py
git commit -m "test(characterization): pin homodyne baselines for rtol=1e-10 gate"
```

---

### Task 22: Implement the characterization test suite

**Files:**
- Create: `tests/characterization/test_homodyne_equivalence.py`

- [ ] **Step 1: Write the characterization tests**

Create `tests/characterization/test_homodyne_equivalence.py`:

```python
"""Bit-equivalence regression vs frozen homodyne baselines.

Phase 5 gate: every config must pass at rtol=1e-10 before Phase 6 begins."""
import json
from pathlib import Path

import numpy as np
import pytest

from xpcsjax.data import load_xpcs_data
from xpcsjax.optimization.nlsq import fit_nlsq

CONFIGS_DIR = Path(__file__).parent / "fixtures" / "configs"
BASELINES_DIR = Path(__file__).parent / "fixtures" / "baselines"

CONFIG_FILES = sorted(CONFIGS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_homodyne_bit_equivalence(config_path):
    """Fit results must match the pinned baseline at rtol=1e-10."""
    baseline_path = BASELINES_DIR / config_path.with_suffix(".json").name
    assert baseline_path.exists(), f"baseline missing for {config_path.name}"
    baseline = json.loads(baseline_path.read_text())

    data = load_xpcs_data(str(config_path))
    result = fit_nlsq(data, str(config_path))

    # Parameter-by-parameter check
    for name, expected in baseline["parameters"].items():
        actual = result.parameters[name]
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10,
            err_msg=f"{config_path.name}: parameter {name} drift "
                    f"(xpcsjax={actual}, baseline={expected})",
        )

    # Scalar fit quality
    np.testing.assert_allclose(result.chi_squared, baseline["chi_squared"], rtol=1e-10)
    np.testing.assert_allclose(result.r_squared,    baseline["r_squared"],   rtol=1e-10)
    assert result.dof == baseline["dof"]
    assert result.convergence_status == baseline["convergence_status"]
    assert abs(result.n_iterations - baseline["n_iterations"]) <= 1

    # Strategy used must match exactly
    assert result.model_metadata.get("strategy_used") == \
           baseline["model_metadata"].get("strategy_used"), \
           f"{config_path.name}: memory strategy diverged"
```

- [ ] **Step 2: Run the characterization suite**

```bash
uv run pytest tests/characterization/test_homodyne_equivalence.py -v
```

Expected: ALL parametrized cases PASS.

If a case FAILS at `rtol=1e-10`:
1. Check if the failure is < `rtol=1e-8` (acceptable JAX trust-region drift) — if so, document in the implementation log and proceed.
2. If failure is > `rtol=1e-6`, it's a real port bug. Use `git diff /home/wei/Documents/GitHub/homodyne/homodyne/<file> xpcsjax/<file>` on the modules touched in this run to find the divergence. Fix and re-run.
3. Common causes: missed `homodyne` → `xpcsjax` rewrite in a transitively-imported module; an accidentally removed CMC branch that was actually shared with NLSQ; missing JAX_ENABLE_X64 env propagation.

- [ ] **Step 3: Commit**

```bash
git add tests/characterization/test_homodyne_equivalence.py
git commit -m "test(gate): homodyne bit-equivalence characterization at rtol=1e-10"
```

---

### Task 23: Tag the homodyne-equivalent commit

- [ ] **Step 1: Confirm all characterization tests pass**

```bash
uv run pytest tests/ -v
```

Expected: every test PASSes.

- [ ] **Step 2: Tag the commit**

```bash
git tag -a homodyne-equivalent -m "Phase 5 gate: homodyne characterization at rtol=1e-10"
git log --oneline -5
```

- [ ] **Step 3: STOP — verify the gate**

Do not start Phase 6 until:
- Every characterization test passes at `rtol=1e-10`.
- `grep -rn "scipy.optimize.least_squares" xpcsjax/` returns nothing.
- `grep -rn "homodyne" xpcsjax/` returns nothing.

If any of these fail, return to the relevant earlier task.

---

## Phase 6 — HeterodyneModel + parameter registry (~3 days)

### Task 24: Add `AnalysisMode.TWO_COMPONENT` enum value

**Files:**
- Modify: `xpcsjax/config/parameter_registry.py`
- Modify: any `AnalysisMode` enum definition (often co-located in `parameter_registry.py` or in `xpcsjax/config/manager.py`)

- [ ] **Step 1: Locate the enum**

```bash
grep -rn "class AnalysisMode\|AnalysisMode(.*Enum" xpcsjax/config/
```

- [ ] **Step 2: Add the enum value**

Find the enum class definition and add a new member. Example (the exact class location may differ):

```python
class AnalysisMode(StrEnum):
    STATIC_DIFFUSION = "static_diffusion"
    LAMINAR_FLOW = "laminar_flow"
    TWO_COMPONENT = "two_component"  # ← new in xpcsjax v0.1
```

- [ ] **Step 3: Add a failing test for the enum**

Create `tests/config/test_analysis_mode.py`:

```python
"""TWO_COMPONENT is the heterodyne enum value added in xpcsjax v0.1."""
from xpcsjax.config.parameter_registry import AnalysisMode


def test_two_component_member_exists():
    assert AnalysisMode.TWO_COMPONENT.value == "two_component"


def test_three_modes_total():
    members = {m.value for m in AnalysisMode}
    assert members == {"static_diffusion", "laminar_flow", "two_component"}
```

Run:

```bash
uv run pytest tests/config/test_analysis_mode.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add xpcsjax/config/parameter_registry.py tests/config/test_analysis_mode.py
git commit -m "feat(config): add AnalysisMode.TWO_COMPONENT for heterodyne fits"
```

---

### Task 25: Add heterodyne parameter registry entries (verbatim bounds from heterodyne docs)

**Files:**
- Modify: `xpcsjax/config/parameter_registry.py`
- Test: `tests/config/test_heterodyne_registry.py`

- [ ] **Step 1: Write the failing test that pins exact bounds**

Create `tests/config/test_heterodyne_registry.py`:

```python
"""Heterodyne parameter registry entries are verbatim from heterodyne docs.

Source: https://heterodyne.readthedocs.io/en/latest/configuration/options.html"""
import math

import pytest

from xpcsjax.config.parameter_registry import REGISTRY, AnalysisMode


EXPECTED_HETERODYNE = {
    "D0_ref":          {"default": 1e4,  "bounds": (0.0,  1e6),     "transform": "log"},
    "alpha_ref":       {"default": 0.0,  "bounds": (-2.0, 2.0),     "transform": "linear"},
    "D_offset_ref":    {"default": 0.0,  "bounds": (-1e4, 1e4),     "transform": "linear"},
    "D0_sample":       {"default": 1e4,  "bounds": (0.0,  1e6),     "transform": "log"},
    "alpha_sample":    {"default": 0.0,  "bounds": (-2.0, 2.0),     "transform": "linear"},
    "D_offset_sample": {"default": 0.0,  "bounds": (-1e4, 1e4),     "transform": "linear"},
    "v0":              {"default": 1e3,  "bounds": (0.0,  1e6),     "transform": "log"},
    "beta":            {"default": 1.0,  "bounds": (0.0,  2.0),     "transform": "linear"},
    "v_offset":        {"default": 0.0,  "bounds": (-100.0, 100.0), "transform": "linear"},
    "f0":              {"default": 0.5,  "bounds": (0.0,  1.0),     "transform": "linear"},
    "f1":              {"default": 0.0,  "bounds": (-1.0, 1.0),     "transform": "linear"},
    "f2":              {"default": 0.0,  "bounds": (-1.0, 1.0),     "transform": "linear"},
    "f3":              {"default": 0.0,  "bounds": (-1.0, 1.0),     "transform": "linear"},
    "phi0":            {"default": 0.0,  "bounds": (-math.pi, math.pi), "transform": "linear"},
}


@pytest.mark.parametrize("param_name", list(EXPECTED_HETERODYNE))
def test_heterodyne_param_bounds_verbatim(param_name):
    specs = {s.name: s for s in REGISTRY[AnalysisMode.TWO_COMPONENT]}
    assert param_name in specs, f"missing heterodyne parameter: {param_name}"
    spec = specs[param_name]
    expected = EXPECTED_HETERODYNE[param_name]
    assert spec.default == expected["default"], f"{param_name}: default drift"
    assert spec.bounds == expected["bounds"],   f"{param_name}: bounds drift"
    assert spec.transform == expected["transform"], f"{param_name}: transform drift"


def test_heterodyne_param_count():
    """14 physics parameters, registered in TWO_COMPONENT mode."""
    specs = REGISTRY[AnalysisMode.TWO_COMPONENT]
    assert len(specs) == 14, f"expected 14 heterodyne params, got {len(specs)}"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/config/test_heterodyne_registry.py -v
```

Expected: FAIL — `AnalysisMode.TWO_COMPONENT` is not yet a key in `REGISTRY`.

- [ ] **Step 3: Add the registry entries**

Open `xpcsjax/config/parameter_registry.py`. Locate the `REGISTRY` dict literal (or builder). Add an entry for `AnalysisMode.TWO_COMPONENT` with a tuple of 14 `ParameterSpec` entries matching the EXPECTED_HETERODYNE dict above. Use `math.pi` for `phi0` bounds.

Example (adjust to match the ParameterSpec constructor signature already in use):

```python
import math

REGISTRY[AnalysisMode.TWO_COMPONENT] = (
    ParameterSpec("D0_ref",          1e4,  (0.0,  1e6),         "log",    "Reference diffusion prefactor"),
    ParameterSpec("alpha_ref",       0.0,  (-2.0, 2.0),         "linear", "Reference transport exponent"),
    ParameterSpec("D_offset_ref",    0.0,  (-1e4, 1e4),         "linear", "Reference diffusion offset"),
    ParameterSpec("D0_sample",       1e4,  (0.0,  1e6),         "log",    "Sample diffusion prefactor"),
    ParameterSpec("alpha_sample",    0.0,  (-2.0, 2.0),         "linear", "Sample transport exponent"),
    ParameterSpec("D_offset_sample", 0.0,  (-1e4, 1e4),         "linear", "Sample diffusion offset"),
    ParameterSpec("v0",              1e3,  (0.0,  1e6),         "log",    "Velocity amplitude"),
    ParameterSpec("beta",            1.0,  (0.0,  2.0),         "linear", "Velocity exponent"),
    ParameterSpec("v_offset",        0.0,  (-100.0, 100.0),     "linear", "Velocity offset"),
    ParameterSpec("f0",              0.5,  (0.0,  1.0),         "linear", "Sample fraction coefficient 0"),
    ParameterSpec("f1",              0.0,  (-1.0, 1.0),         "linear", "Sample fraction coefficient 1"),
    ParameterSpec("f2",              0.0,  (-1.0, 1.0),         "linear", "Sample fraction coefficient 2"),
    ParameterSpec("f3",              0.0,  (-1.0, 1.0),         "linear", "Sample fraction coefficient 3"),
    ParameterSpec("phi0",            0.0,  (-math.pi, math.pi), "linear", "Flow angle (radians)"),
)
```

- [ ] **Step 4: Re-run the test**

```bash
uv run pytest tests/config/test_heterodyne_registry.py -v
```

Expected: all parametrized cases + `test_heterodyne_param_count` PASS.

- [ ] **Step 5: Commit**

```bash
git add xpcsjax/config/parameter_registry.py tests/config/test_heterodyne_registry.py
git commit -m "feat(config): add 14 heterodyne parameter specs (bounds from docs)"
```

---

### Task 26: Port heterodyne physics kernels

**Files:**
- Copy: `heterodyne/core/physics.py` → `xpcsjax/core/heterodyne_physics.py` (renamed to avoid collision with homodyne physics kernels)

- [ ] **Step 1: Copy the source**

```bash
cp /home/wei/Documents/GitHub/heterodyne/heterodyne/core/physics.py \
   /home/wei/Documents/GitHub/xpcsjax/xpcsjax/core/heterodyne_physics.py
```

- [ ] **Step 2: Rewrite imports — both `homodyne.` and `heterodyne.` → `xpcsjax.`**

```bash
sed -i 's/from heterodyne\./from xpcsjax./g; s/import heterodyne\./import xpcsjax./g' \
    /home/wei/Documents/GitHub/xpcsjax/xpcsjax/core/heterodyne_physics.py
sed -i 's/from homodyne\./from xpcsjax./g; s/import homodyne\./import xpcsjax./g' \
    /home/wei/Documents/GitHub/xpcsjax/xpcsjax/core/heterodyne_physics.py
```

- [ ] **Step 3: Verify zero source-package references**

```bash
grep -n "homodyne\|heterodyne" xpcsjax/core/heterodyne_physics.py
```

Expected: empty.

- [ ] **Step 4: Smoke import**

```bash
uv run python -c "from xpcsjax.core import heterodyne_physics; print(heterodyne_physics)"
```

Expected: prints module repr. If a symbol is missing, the import path inside `heterodyne_physics.py` references a sibling module that was named differently in the heterodyne source — fix that import to the xpcsjax path.

- [ ] **Step 5: Commit**

```bash
git add xpcsjax/core/heterodyne_physics.py
git commit -m "port: heterodyne/core/physics.py → xpcsjax/core/heterodyne_physics.py"
```

---

### Task 27: Implement `HeterodyneModel` (adapts to `PhysicsModelBase` interface)

**Files:**
- Create: `xpcsjax/core/heterodyne_model.py`
- Test: `tests/core/test_heterodyne_model.py`

- [ ] **Step 1: Inspect the source heterodyne model class for reference**

```bash
head -100 /home/wei/Documents/GitHub/heterodyne/heterodyne/core/heterodyne_model.py
```

Note: the source class is named `HeterodyneModel` (and there is a related `TwoComponentModel`). We adapt it to the `PhysicsModelBase` interface defined in `xpcsjax/core/models.py`.

- [ ] **Step 2: Write the failing interface tests**

Create `tests/core/test_heterodyne_model.py`:

```python
"""HeterodyneModel implements PhysicsModelBase with 14 physics params."""
import math

import jax.numpy as jnp
import numpy as np

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.core.heterodyne_model import HeterodyneModel
from xpcsjax.core.models import PhysicsModelBase


def test_is_physics_model():
    model = HeterodyneModel()
    assert isinstance(model, PhysicsModelBase)


def test_analysis_mode():
    model = HeterodyneModel()
    assert model.analysis_mode == AnalysisMode.TWO_COMPONENT


def test_param_names_match_registry():
    model = HeterodyneModel()
    expected = (
        "D0_ref", "alpha_ref", "D_offset_ref",
        "D0_sample", "alpha_sample", "D_offset_sample",
        "v0", "beta", "v_offset",
        "f0", "f1", "f2", "f3",
        "phi0",
    )
    assert model.param_names == expected


def test_param_bounds_match_docs():
    model = HeterodyneModel()
    assert model.param_bounds[model.param_names.index("D0_ref")] == (0.0, 1e6)
    assert model.param_bounds[model.param_names.index("beta")] == (0.0, 2.0)
    assert model.param_bounds[model.param_names.index("phi0")] == (-math.pi, math.pi)


def test_residual_runs_on_minimal_data():
    """Residual must produce finite values for in-bounds default params."""
    model = HeterodyneModel()
    # Use registry defaults
    params = jnp.array([s.default for s in model._registry_specs()])
    N = 8
    data = {
        "c2_exp": np.ones((1, 1, N, N), dtype=np.float64) * 1.05,
        "t1": np.arange(N, dtype=np.float64),
        "t2": np.arange(N, dtype=np.float64),
        "wavevector_q_list": np.array([0.01]),
        "phi_angles_list": np.array([0.0]),
        "dt": 1.0,
    }
    res = model.compute_residual(params, data, ctx={})
    assert res.ndim == 1
    assert jnp.all(jnp.isfinite(res))
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/core/test_heterodyne_model.py -v
```

Expected: ImportError or all FAIL — `HeterodyneModel` does not exist yet.

- [ ] **Step 4: Implement `HeterodyneModel`**

Create `xpcsjax/core/heterodyne_model.py`:

```python
"""HeterodyneModel — 14-parameter two-component XPCS physics model.

Adapts heterodyne's TwoComponentModel physics to xpcsjax's PhysicsModelBase
interface. The NLSQ engine consumes only the interface methods; the internal
g2 computation uses kernels in xpcsjax.core.heterodyne_physics.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from xpcsjax.config.parameter_registry import REGISTRY, AnalysisMode, ParameterSpec
from xpcsjax.core.heterodyne_physics import compute_c2_heterodyne
from xpcsjax.core.models import PhysicsModelBase


class HeterodyneModel(PhysicsModelBase):
    """Two-component reference + sample XPCS model (14 physics params)."""

    analysis_mode = AnalysisMode.TWO_COMPONENT

    def _registry_specs(self) -> tuple[ParameterSpec, ...]:
        return REGISTRY[self.analysis_mode]

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self._registry_specs())

    @property
    def param_bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple(s.bounds for s in self._registry_specs())

    @property
    def param_transforms(self) -> tuple[str, ...]:
        return tuple(s.transform for s in self._registry_specs())

    def initial_guess(self, data: dict[str, Any]) -> jnp.ndarray:
        return jnp.array([s.default for s in self._registry_specs()])

    def compute_residual(
        self,
        params: jnp.ndarray,
        data: dict[str, Any],
        ctx: dict[str, Any],
    ) -> jnp.ndarray:
        """Flatten c2(model) - c2(exp) across (q, phi, t1, t2) into a 1-D residual."""
        c2_model = compute_c2_heterodyne(
            params=params,
            t1=jnp.asarray(data["t1"]),
            t2=jnp.asarray(data["t2"]),
            q_list=jnp.asarray(data["wavevector_q_list"]),
            phi_list=jnp.asarray(data["phi_angles_list"]),
            dt=float(data["dt"]),
        )
        residual = c2_model - jnp.asarray(data["c2_exp"])
        return residual.reshape(-1)
```

Note: the exact signature of `compute_c2_heterodyne` is whatever heterodyne's `physics.py` exposes. Inspect it and pass the right arguments. If the source function expects a different data shape (e.g., per-angle slice), adjust this wrapper to loop over angles or vmap.

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/core/test_heterodyne_model.py -v
```

Expected: all PASS. If any fail because the residual produces NaN, the issue is almost always a numerical edge case (e.g., `log(0)` from a `D0=0` bound) — clip params at small epsilon before passing to the kernel.

- [ ] **Step 6: Commit**

```bash
git add xpcsjax/core/heterodyne_model.py tests/core/test_heterodyne_model.py
git commit -m "feat(core): add HeterodyneModel wrapping heterodyne_physics kernels"
```

---

### Task 28: Wire `HeterodyneModel` into `ConfigManager.get_model()`

**Files:**
- Modify: `xpcsjax/config/manager.py`
- Modify: `xpcsjax/core/__init__.py`
- Test: `tests/config/test_get_model.py`

- [ ] **Step 1: Write the failing dispatch test**

Create `tests/config/test_get_model.py`:

```python
"""ConfigManager.get_model() dispatches by analysis_mode to the right physics class."""
from xpcsjax.core import HomodyneModel, HeterodyneModel
from xpcsjax.config import ConfigManager
from xpcsjax.config.parameter_registry import AnalysisMode


def _config_yaml(tmp_path, mode: str) -> str:
    p = tmp_path / f"{mode}.yaml"
    p.write_text(f"analysis_mode: {mode}\ndata:\n  hdf5_path: /dev/null\n")
    return str(p)


def test_static_diffusion_dispatches_to_homodyne(tmp_path):
    cfg = ConfigManager(_config_yaml(tmp_path, "static_diffusion"))
    model = cfg.get_model()
    assert isinstance(model, HomodyneModel)


def test_laminar_flow_dispatches_to_homodyne(tmp_path):
    cfg = ConfigManager(_config_yaml(tmp_path, "laminar_flow"))
    model = cfg.get_model()
    assert isinstance(model, HomodyneModel)


def test_two_component_dispatches_to_heterodyne(tmp_path):
    cfg = ConfigManager(_config_yaml(tmp_path, "two_component"))
    model = cfg.get_model()
    assert isinstance(model, HeterodyneModel)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/config/test_get_model.py -v
```

Expected: `test_two_component_dispatches_to_heterodyne` fails (the other two may pass if homodyne's `ConfigManager` already handles them).

- [ ] **Step 3: Update `ConfigManager.get_model()`**

Locate the dispatcher in `xpcsjax/config/manager.py`. The original homodyne version dispatches on `analysis_mode` to `DiffusionModel` / `CombinedModel`. Extend it:

```python
def get_model(self):
    mode = self.analysis_mode  # whatever attribute the existing code uses
    if mode == AnalysisMode.STATIC_DIFFUSION:
        from xpcsjax.core.models import DiffusionModel
        return DiffusionModel()
    if mode == AnalysisMode.LAMINAR_FLOW:
        from xpcsjax.core.models import CombinedModel
        return CombinedModel()
    if mode == AnalysisMode.TWO_COMPONENT:
        from xpcsjax.core.heterodyne_model import HeterodyneModel
        return HeterodyneModel()
    raise ValueError(f"unknown analysis_mode: {mode}")
```

Adjust attribute names to match the existing code.

- [ ] **Step 4: Update `core/__init__.py` to export `HeterodyneModel`**

```python
# add to xpcsjax/core/__init__.py
from xpcsjax.core.heterodyne_model import HeterodyneModel

__all__ = [..., "HeterodyneModel"]
```

- [ ] **Step 5: Re-run dispatch tests**

```bash
uv run pytest tests/config/test_get_model.py -v
```

Expected: all PASS.

- [ ] **Step 6: Re-run the homodyne characterization suite (regression check)**

```bash
uv run pytest tests/characterization/ -v
```

Expected: all PASS at `rtol=1e-10`. If anything regressed, undo the dispatch change and isolate.

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/config/manager.py xpcsjax/core/__init__.py tests/config/test_get_model.py
git commit -m "feat(config): ConfigManager.get_model dispatches to HeterodyneModel"
```

---

### Task 29: Gate `ShearSensitivityWeighting` (anti-degeneracy Layer 5) by model lineage

**Files:**
- Modify: `xpcsjax/optimization/nlsq/anti_degeneracy_controller.py`
- Test: `tests/optimization/test_layer5_gating.py`

- [ ] **Step 1: Locate the Layer-5 entry point**

```bash
grep -n "ShearSensitivityWeighting\|shear_sensitivity\|Layer5\|layer_5" \
    xpcsjax/optimization/nlsq/anti_degeneracy_controller.py
```

Identify the function or class that applies Layer 5.

- [ ] **Step 2: Write the failing gating test**

Create `tests/optimization/__init__.py` (empty), then `tests/optimization/test_layer5_gating.py`:

```python
"""ShearSensitivityWeighting is active for HomodyneModel, inert for HeterodyneModel.

This is the v0.1 anti-degeneracy gating rule from spec §10.3:
- HomodyneModel (static_diffusion, laminar_flow) → 5 layers including Layer 5
- HeterodyneModel (two_component)                → 4 layers; Layer 5 disabled
"""
import jax.numpy as jnp

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.core import HomodyneModel, HeterodyneModel
from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyController,
)


def test_shear_layer_active_for_homodyne():
    model = HomodyneModel()
    controller = AntiDegeneracyController(model=model)
    assert controller.is_layer_active("ShearSensitivityWeighting") is True


def test_shear_layer_inactive_for_heterodyne():
    model = HeterodyneModel()
    controller = AntiDegeneracyController(model=model)
    assert controller.is_layer_active("ShearSensitivityWeighting") is False


def test_other_four_layers_active_for_heterodyne():
    model = HeterodyneModel()
    controller = AntiDegeneracyController(model=model)
    for name in ("FourierReparameterizer", "HierarchicalOptimizer",
                 "AdaptiveRegularizer", "GradientCollapseMonitor"):
        assert controller.is_layer_active(name) is True, f"layer {name} unexpectedly inactive"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/optimization/test_layer5_gating.py -v
```

Expected: FAIL — either `is_layer_active` doesn't exist or all layers are returned `True`.

- [ ] **Step 4: Implement model-lineage gating**

Open `xpcsjax/optimization/nlsq/anti_degeneracy_controller.py`. Add:

```python
# Layers that are gated by model.analysis_mode.
# Convention: layer name → set of AnalysisMode values where it is active.
from xpcsjax.config.parameter_registry import AnalysisMode

_LAYER_GATES: dict[str, frozenset[AnalysisMode]] = {
    "ShearSensitivityWeighting": frozenset({
        AnalysisMode.STATIC_DIFFUSION,
        AnalysisMode.LAMINAR_FLOW,
    }),
    # All other layers default to active for every mode.
}


class AntiDegeneracyController:
    def __init__(self, model, ...existing args...):
        self.model = model
        # ...existing init code...

    def is_layer_active(self, layer_name: str) -> bool:
        gates = _LAYER_GATES.get(layer_name)
        if gates is None:
            return True  # not gated — always active
        return self.model.analysis_mode in gates

    # In the apply/run method, wrap the Layer-5 application:
    def _apply_shear_sensitivity_weighting(self, ...):
        if not self.is_layer_active("ShearSensitivityWeighting"):
            return  # short-circuit for heterodyne fits
        # ... existing implementation ...
```

Adapt to the actual code structure. The key change: every Layer-5 application site checks `is_layer_active` first.

- [ ] **Step 5: Re-run gating tests**

```bash
uv run pytest tests/optimization/test_layer5_gating.py -v
```

Expected: all PASS.

- [ ] **Step 6: Re-run homodyne characterization (regression check)**

```bash
uv run pytest tests/characterization/ -v
```

Expected: all PASS at `rtol=1e-10`. The gating logic must not perturb homodyne fits — Layer 5 stays active for those.

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/optimization/nlsq/anti_degeneracy_controller.py tests/optimization/
git commit -m "feat(nlsq): gate ShearSensitivityWeighting by model lineage (homodyne-only)"
```

---

### Task 30: Heterodyne end-to-end smoke fit (registry → model → NLSQ → result)

**Files:**
- Test: `tests/heterodyne/test_two_component_smoke.py`

- [ ] **Step 1: Write a minimal heterodyne config**

Create `tests/heterodyne/fixtures/configs/smoke.yaml`:

```yaml
analysis_mode: two_component
data:
  hdf5_path: /dev/null  # smoke test only — use synthetic data, not file load
preprocessing:
  diagonal_correction:
    method: basic
    width: 1
optimization:
  nlsq:
    memory_strategy: standard
parameters:
  D0_ref:    { initial: 200.0 }
  D0_sample: { initial: 8000.0 }
```

- [ ] **Step 2: Write the smoke fit test (synthetic data, no HDF5 needed)**

Create `tests/heterodyne/test_two_component_smoke.py`:

```python
"""End-to-end smoke test for the heterodyne fit path.

Generates synthetic c2 data from HeterodyneModel itself with known params,
then fits and asserts the recovered params are reasonable. This is a tighter
gate than "does it crash" but looser than golden-value matching."""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from xpcsjax.config import ConfigManager
from xpcsjax.core import HeterodyneModel
from xpcsjax.optimization.nlsq import fit_nlsq

FIXTURE_CONFIG = Path(__file__).parent / "fixtures" / "configs" / "smoke.yaml"


def _synthetic_data(true_params: jnp.ndarray, N: int = 16):
    model = HeterodyneModel()
    # Build trivial coordinate arrays
    t = jnp.arange(N, dtype=jnp.float64)
    data = {
        "t1": np.asarray(t),
        "t2": np.asarray(t),
        "wavevector_q_list": np.array([0.01]),
        "phi_angles_list": np.array([0.0]),
        "dt": 1.0,
    }
    # Generate noiseless c2 from the model
    res = model.compute_residual(true_params, {**data, "c2_exp": np.zeros((1, 1, N, N))},
                                  ctx={})
    c2_true = -res.reshape(1, 1, N, N)  # residual = model - exp, exp was zero
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1e-4, size=c2_true.shape)
    data["c2_exp"] = np.asarray(c2_true) + noise
    return data


def test_heterodyne_smoke_fit_runs():
    """Fit must run end-to-end on synthetic data and return a converged result."""
    model = HeterodyneModel()
    true_params = jnp.array([s.default for s in model._registry_specs()])
    data = _synthetic_data(true_params)

    cfg = ConfigManager(str(FIXTURE_CONFIG))
    result = fit_nlsq(data, cfg)

    assert result.convergence_status == "converged"
    # Recovered params should be near the true ones (loose tolerance for smoke test)
    for name, true_val in zip(model.param_names, np.asarray(true_params), strict=True):
        recovered = result.parameters[name]
        # 50% tolerance on smoke test — golden-value tests in Task 32 are tighter
        np.testing.assert_allclose(recovered, true_val, rtol=0.5, atol=1e-6)
```

- [ ] **Step 3: Run the smoke test**

```bash
uv run pytest tests/heterodyne/test_two_component_smoke.py -v
```

Expected: PASS. If it fails because the residual landscape is poorly conditioned around defaults (common — defaults are physically generic), tighten the `true_params` to a known-physical combination drawn from heterodyne's own fixtures.

- [ ] **Step 4: Commit**

```bash
git add tests/heterodyne/
git commit -m "test(heterodyne): end-to-end smoke fit on synthetic two-component data"
```

---

## Phase 7 — Heterodyne golden-value validation (~2 days)

### Task 31: Generate heterodyne baselines from the source package

**Files:**
- Create: `scripts/generate_heterodyne_baselines.py`
- Create: `tests/heterodyne/fixtures/configs/` (mirror heterodyne test configs)
- Create: `tests/heterodyne/fixtures/baselines/`

- [ ] **Step 1: Identify heterodyne test configs**

```bash
find /home/wei/Documents/GitHub/heterodyne/tests -name "*.yaml" | head -10
```

Pick a representative set: at least 3 configs covering different param regimes.

- [ ] **Step 2: Copy configs into the xpcsjax fixture directory**

```bash
mkdir -p tests/heterodyne/fixtures/configs tests/heterodyne/fixtures/baselines
cp /home/wei/Documents/GitHub/heterodyne/tests/<...>/*.yaml \
   tests/heterodyne/fixtures/configs/
```

If the source configs use a different schema (`heterodyne` may use different YAML keys), translate them so `analysis_mode: two_component` and the rest matches xpcsjax's ConfigManager expectations.

- [ ] **Step 3: Write the baseline generator**

Create `scripts/generate_heterodyne_baselines.py`:

```python
"""Generate heterodyne fit baselines by running the SOURCE heterodyne package.

These are the targets for xpcsjax to reproduce within 1σ parameter_errors.
The numerics are NOT expected to match bit-identically — heterodyne uses
soft_L1 + its own anti-degeneracy controller, while xpcsjax uses chi-squared
+ homodyne's controller. Physics-equivalence within 1σ is the gate."""
import json
from pathlib import Path

# Import from SOURCE heterodyne package
from heterodyne import ConfigManager, XPCSDataLoader, fit_nlsq_jax, HeterodyneModel

CONFIGS_DIR = Path("tests/heterodyne/fixtures/configs")
BASELINES_DIR = Path("tests/heterodyne/fixtures/baselines")
BASELINES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    for config_path in sorted(CONFIGS_DIR.glob("*.yaml")):
        print(f"== {config_path.name}")
        cfg = ConfigManager(str(config_path))
        loader = XPCSDataLoader(str(cfg.data_path))
        data = loader.load()
        model = HeterodyneModel.from_config(cfg)
        result = fit_nlsq_jax(model, data.c2[0], data.phi_angles[0], cfg.nlsq_config)

        out = BASELINES_DIR / config_path.with_suffix(".json").name
        with out.open("w") as f:
            json.dump(
                {
                    "parameters": {k: float(v) for k, v in result.parameters.items()},
                    "parameter_errors": {k: float(v) for k, v in result.parameter_errors.items()},
                    "chi_squared": float(result.fit_quality.residual_norm),
                    "r_squared": float(result.fit_quality.r_squared),
                },
                f,
                indent=2,
            )
        print(f"   wrote {out}")


if __name__ == "__main__":
    main()
```

Adjust API calls to match heterodyne's actual surface (e.g., `XPCSDataLoader(path).load()` vs. `load_xpcs_data(path)`).

- [ ] **Step 4: Run the generator**

```bash
cd /home/wei/Documents/GitHub/xpcsjax
JAX_ENABLE_X64=1 uv run --with /home/wei/Documents/GitHub/heterodyne \
    python scripts/generate_heterodyne_baselines.py
```

Expected: one JSON per config under `tests/heterodyne/fixtures/baselines/`.

- [ ] **Step 5: Commit baselines**

```bash
git add scripts/generate_heterodyne_baselines.py tests/heterodyne/fixtures/
git commit -m "test(heterodyne): pin source-heterodyne baselines for 1σ validation"
```

---

### Task 32: Heterodyne golden-value tests within 1σ tolerance

**Files:**
- Create: `tests/heterodyne/test_golden_values.py`

- [ ] **Step 1: Write the within-1σ test**

Create `tests/heterodyne/test_golden_values.py`:

```python
"""xpcsjax heterodyne fits must fall within the source-heterodyne baseline 1σ.

Per spec §12.2: passing this gate validates that dropping heterodyne's
separately-developed anti-degeneracy controller is sound. If >5% of golden-value
configs fail, the v0.2 backlog adds heterodyne's controller as an opt-in."""
import json
from pathlib import Path

import pytest

from xpcsjax.data import load_xpcs_data
from xpcsjax.optimization.nlsq import fit_nlsq

CONFIGS_DIR = Path(__file__).parent / "fixtures" / "configs"
BASELINES_DIR = Path(__file__).parent / "fixtures" / "baselines"

CONFIG_FILES = sorted(CONFIGS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_heterodyne_within_one_sigma(config_path):
    baseline_path = BASELINES_DIR / config_path.with_suffix(".json").name
    assert baseline_path.exists(), f"baseline missing for {config_path.name}"
    baseline = json.loads(baseline_path.read_text())

    data = load_xpcs_data(str(config_path))
    result = fit_nlsq(data, str(config_path))

    failures = []
    for name, expected in baseline["parameters"].items():
        sigma = baseline["parameter_errors"][name]
        actual = result.parameters[name]
        if abs(actual - expected) > sigma:
            failures.append(f"{name}: |{actual} - {expected}| > σ={sigma}")

    assert not failures, "\n".join(failures)
```

- [ ] **Step 2: Run the golden-value tests**

```bash
uv run pytest tests/heterodyne/test_golden_values.py -v
```

Expected: all PASS, or up to 5% may fail (per spec policy). Record the pass rate.

- [ ] **Step 3: Decide on Layer-5/anti-degeneracy escalation**

- **If 100% pass**: proceed.
- **If 1–5% fail**: investigate per-config; if the failure mode is consistent (e.g., always `D0_ref` swap with `D0_sample`), document in a `KNOWN_LIMITATIONS.md` and proceed.
- **If >5% fail**: scope the v0.2 work to port heterodyne's anti-degeneracy controller as an opt-in. Open a tracking issue and STOP — Phase 8 release is blocked until either pass rate improves or the v0.2 work lands.

- [ ] **Step 4: Commit**

```bash
git add tests/heterodyne/test_golden_values.py
git commit -m "test(heterodyne): golden-value validation within source 1σ"
```

---

### Task 33: Property tests for cross-cutting invariants

**Files:**
- Create: `tests/property/test_transforms.py`
- Create: `tests/property/test_result_roundtrip.py`

- [ ] **Step 1: Write log-transform roundtrip test**

Create `tests/property/test_transforms.py`:

```python
"""Parameter transforms must be reversible to float64 precision."""
import math

import jax.numpy as jnp
import numpy as np
from hypothesis import given, strategies as st

from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import AnalysisMode


@given(value=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_log_transform_roundtrip(value):
    pm = ParameterManager(analysis_mode=AnalysisMode.STATIC_DIFFUSION)
    transformed = pm.transform("D0", value)
    recovered = pm.inverse_transform("D0", transformed)
    np.testing.assert_allclose(recovered, value, rtol=1e-12)


@given(value=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False))
def test_linear_transform_is_identity(value):
    pm = ParameterManager(analysis_mode=AnalysisMode.STATIC_DIFFUSION)
    transformed = pm.transform("alpha", value)
    np.testing.assert_allclose(transformed, value)
```

- [ ] **Step 2: Write result-roundtrip test**

Create `tests/property/test_result_roundtrip.py`:

```python
"""OptimizationResult.save() + load() must round-trip exactly."""
from pathlib import Path

import numpy as np

from xpcsjax.io import OptimizationResult


def test_save_load_roundtrip(tmp_path):
    original = OptimizationResult(
        parameters={"D0": 1234.5, "alpha": 0.1, "D_offset": 0.0},
        parameter_errors={"D0": 12.0, "alpha": 0.01, "D_offset": 0.5},
        covariance=np.eye(3),
        chi_squared=1.234e-3,
        dof=100,
        r_squared=0.99,
        residuals=np.zeros(100),
        n_iterations=42,
        convergence_status="converged",
        model_metadata={"analysis_mode": "static_diffusion"},
    )
    out = tmp_path / "result"
    original.save(out)

    loaded = OptimizationResult.load(out)
    assert loaded.parameters == original.parameters
    assert loaded.dof == original.dof
    np.testing.assert_array_equal(loaded.covariance, original.covariance)
```

If the `OptimizationResult` class API differs in homodyne's port, adapt the test to the actual constructor signature.

- [ ] **Step 3: Run property tests**

```bash
uv run pytest tests/property/ -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/property/test_transforms.py tests/property/test_result_roundtrip.py
git commit -m "test(property): parameter transform roundtrip + result serialization"
```

---

### Task 34: Full regression run

- [ ] **Step 1: Run the entire test suite**

```bash
cd /home/wei/Documents/GitHub/xpcsjax
uv run pytest -v
```

Expected: every test PASSes (or up to 5% of heterodyne golden-value tests fail per Task 32 policy).

- [ ] **Step 2: Run lint + type checks**

```bash
uv run ruff check .
uv run mypy xpcsjax
```

Expected: clean.

- [ ] **Step 3: Final hard-rule audits**

```bash
grep -rn "scipy.optimize.least_squares\|from scipy.optimize import least_squares" xpcsjax/
grep -rn "import homodyne\b\|from homodyne\b" xpcsjax/
grep -rn "import heterodyne\b\|from heterodyne\b" xpcsjax/
```

Expected: all three return empty. xpcsjax must not import from scipy's LM solver, the source homodyne package, or the source heterodyne package.

---

## Phase 8 — Docs + release (~1 day)

### Task 35: Write the migration guide

**Files:**
- Create: `docs/MIGRATION.md`

- [ ] **Step 1: Author the migration guide**

Create `docs/MIGRATION.md`:

```markdown
# Migrating from `homodyne` or `heterodyne` to `xpcsjax`

xpcsjax v0.1 consolidates the NLSQ fitting pipelines from the standalone
`homodyne` and `heterodyne` packages into one JAX-native package.

## What changed

- Import paths: `homodyne.*` and `heterodyne.*` → `xpcsjax.*`
- Fit entry: `homodyne.optimization.fit_nlsq_jax(data, config)` and
  `heterodyne.fit_nlsq_jax(model, c2, phi, config)` →
  `xpcsjax.fit_nlsq(data, config)` (one entry, config-driven).
- Analysis mode: heterodyne fits set `analysis_mode: two_component`
  in the YAML config.

## What did NOT change in v0.1

- Diagonal correction is mandatory for both models (same three methods).
- Homodyne NLSQ behavior is bit-equivalent (rtol=1e-10).
- Heterodyne parameter bounds are verbatim from the heterodyne docs.

## What is NOT in v0.1

- CMC (Bayesian) fitting — coming in v0.2.
- Visualization, CLI, datashader — coming in v0.2.
- Heterodyne's soft-L1 loss and separate anti-degeneracy controller — these
  did not port; identifiability is handled via homodyne's 4-of-5 anti-degeneracy
  layers, narrow LHS multistart, and disciplined initial guesses.

## Worked example — heterodyne fit

```python
from xpcsjax import load_xpcs_data, fit_nlsq

data   = load_xpcs_data("config_heterodyne.yaml")
result = fit_nlsq(data, "config_heterodyne.yaml")
print(result.parameters)
result.save("output/")
```

Minimum YAML:

```yaml
analysis_mode: two_component
data:
  hdf5_path: experiments/run.h5
preprocessing:
  diagonal_correction: { method: basic, width: 1 }
parameters:
  D0_ref:    { initial: 200.0 }     # distinct initials break ref/sample symmetry
  D0_sample: { initial: 8000.0 }
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/MIGRATION.md
git commit -m "docs: migration guide from homodyne/heterodyne to xpcsjax"
```

---

### Task 36: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README content**

```markdown
# xpcsjax

JAX-native NLSQ fitting for X-ray Photon Correlation Spectroscopy (XPCS).

Consolidates the homodyne (single-component scattering) and heterodyne
(reference-beam scattering) analysis pipelines into one package with a
shared engine and config-driven physics-model dispatch.

## Install

```bash
uv add xpcsjax
```

## Quickstart

```python
from xpcsjax import load_xpcs_data, fit_nlsq

data   = load_xpcs_data("config.yaml")
result = fit_nlsq(data, "config.yaml")
print(result.parameters)
```

Set `analysis_mode: static_diffusion`, `laminar_flow`, or `two_component`
in the config to select the physics model.

## What's here in v0.1

- Data loading + diagonal correction (verbatim from homodyne).
- JAX-native NLSQ via `nlsq.CurveFit` (never `scipy.optimize.least_squares`).
- 5-layer anti-degeneracy controller (4 layers for `two_component` mode).
- Memory-aware strategy routing (STANDARD / OUT_OF_CORE / HYBRID_STREAMING).
- CMA-ES escape for multi-scale parameter problems.

## What's coming in v0.2

- CMC (Bayesian) fitting via NumPyro.
- Visualization (matplotlib + pyqtgraph).
- CLI (`xpcsjax fit ...`).

## See also

- Design spec: `docs/superpowers/specs/2026-05-18-xpcsjax-nlsq-merge-design.md`
- Implementation plan: `docs/superpowers/plans/2026-05-18-xpcsjax-nlsq-merge.md`
- Migration: `docs/MIGRATION.md`
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for xpcsjax v0.1"
```

---

### Task 37: Tag v0.1.0

- [ ] **Step 1: Final test run**

```bash
uv run pytest -v
uv run ruff check .
uv run mypy xpcsjax
```

Expected: clean.

- [ ] **Step 2: Bump version to 0.1.0 (drop `.dev0`)**

In `pyproject.toml`, change `version = "0.1.0.dev0"` to `version = "0.1.0"`.

```bash
git add pyproject.toml
git commit -m "release: bump version to 0.1.0"
```

- [ ] **Step 3: Tag the release**

```bash
git tag -a v0.1.0 -m "xpcsjax v0.1.0 — unified homodyne+heterodyne NLSQ fitting"
git log --oneline -10
git tag --list
```

- [ ] **Step 4: (Optional) Push to remote**

```bash
git push origin main --tags
```

---

## Self-review checklist (for the implementer)

- [ ] Every task's "Expected:" output matches what you actually see.
- [ ] `grep -rn "scipy.optimize.least_squares" xpcsjax/` returns empty after every Phase 4 task.
- [ ] `grep -rn "homodyne\|heterodyne" xpcsjax/` (excluding docstrings/comments) returns empty after Phase 7.
- [ ] **Engine-feature unit tests (Tasks 20a/b/c) pass BEFORE the Phase 5 characterization gate runs.** These localize regressions to memory routing, anti-degeneracy layer presence, or CMA-ES trigger respectively — fix anything failing here before generating baselines.
- [ ] Phase 5 characterization tests pass at `rtol=1e-10` BEFORE Phase 6 begins.
- [ ] Heterodyne golden-value tests pass within 1σ (Task 32 escalation policy applies if >5% fail).
- [ ] `OptimizationResult.save() / load()` roundtrip preserves equality.
- [ ] Property tests in `tests/property/` all pass.
- [ ] `uv run ruff check .` and `uv run mypy xpcsjax` are clean.

## End state

After Task 37:
- `xpcsjax` v0.1.0 is tagged.
- Both physics models fit through a single `fit_nlsq(data, config)` entry.
- Homodyne fits are bit-equivalent to the source `homodyne` package.
- Heterodyne fits land within 1σ of source `heterodyne` baselines.
- The JAX-first, no-scipy-LM, on-device LM-loop invariant is enforced by audit greps.
- v0.2 backlog is documented (CMC, viz, CLI, optional heterodyne anti-degeneracy controller if >5% golden-value failures).
