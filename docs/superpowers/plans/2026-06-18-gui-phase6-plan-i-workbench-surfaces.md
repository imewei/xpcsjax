# GUI Plan I — Workbench Surfaces (Config Editor · Data Preview · Fit Summary · Inspector) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the four workbench surfaces the holistic review (spec §14) deferred from the six-phase happy path: the **Config** tab (form editor over `parameter_registry` + mode templates, live validation, raw-YAML toggle), the **Data** tab (HDF5 metadata + two-time C₂ preview), the **Fit** tab (resolved-settings summary), and the right-hand **Inspector** dock (parameter values + uncertainties + the anti-degeneracy diagnostics block).

**Architecture:** All four are **JAX-free** in the GUI process. (1) Extend `service/config.py` with JAX-free `validate_config` / template helpers over the (JAX-free) `parameter_registry` + the mode YAML templates. (2) A JAX-free `gui/data_inspect.py` reads HDF5 structure/attrs and a single downsampled C₂ slice via **h5py directly** (never `xpcsjax.data`, which pulls JAX). (3) Qt widgets `gui/views/{config_editor,data_panel,fit_panel,inspector}.py` consume those + the Plan-G `rasterize`/`TwoTimeMapView` and the Plan-D `ResultSummary` (extended with `diagnostics`). (4) `MainWindow` mounts the three center tabs + the Inspector dock.

**Tech Stack:** Python 3.12+, `uv`, `PySide6`, `pyqtgraph`, `h5py`, `pyyaml`, `numpy`, `pytest-qt`. Consumes Plans 1A (JAX-free boundary), B2 (`service/config.py`), C/D (`MainWindow`, `ResultSummary`), F (`Project`/`FitRun`, `ProjectSidebar.runs_selected`), G (`rasterize`, `TwoTimeMapView`).

## Prerequisite (hard dependency)

**Plans 1A, 1B, B2, C, D, F, G must be implemented first** (G provides the preview widget + rasterizer; **F provides the project/run model (`Project`/`FitRun`), the `ProjectSidebar.runs_selected` signal, and `FitRun.summary` that Task 5 wires `InspectorDock.show_summary` to**; **1A must make `xpcsjax.config.parameter_registry` + `xpcsjax.config.types` importable JAX-free** — see the Global Constraints note; this is a hard gate). Verify the files exist **and** the config imports are JAX-free:

```bash
test -f xpcsjax/gui/views/plots_view.py && test -f xpcsjax/gui/result_loader.py \
  && test -f xpcsjax/service/config.py && test -f xpcsjax/config/parameter_registry.py \
  && test -f xpcsjax/gui/project/model.py && test -f xpcsjax/gui/views/project_panel.py \
  && test -f xpcsjax/gui/controllers/fit_queue.py \
  && echo "B2+D+F+G present" || echo "RUN PLANS B2/D/F/G FIRST"
# 1A gate — these MUST print 'jax-free' (else build/extend Plan 1A's config/__init__ fix first):
python -c "import sys,importlib; importlib.import_module('xpcsjax.config.parameter_registry'); print('registry', 'JAX' if 'jax' in sys.modules else 'jax-free')"
python -c "import sys,importlib; importlib.import_module('xpcsjax.config.types'); print('types', 'JAX' if 'jax' in sys.modules else 'jax-free')"
```

## Global Constraints

- **Python ≥ 3.12**; **uv-first**.
- **GUI process never imports JAX — and this plan has a hard dependency that Plan 1A must satisfy first.** An empirical probe (2026-06-18) shows that on the current tree, importing `xpcsjax.config.parameter_registry` **or** `xpcsjax.config.types` **already loads `jax`**, because `xpcsjax/config/__init__.py:33` eagerly imports `ParameterManager`, which imports `xpcsjax.core.physics` (`parameter_manager.py:21`). The `parameter_registry` *module* imports only `xpcsjax.utils.logging`, but the **package `__init__` runs on any submodule import** and pulls JAX. So Plan 1A's job is to make **`import xpcsjax.config.parameter_registry` and `import xpcsjax.config.types` JAX-free**, proven by Task 5's import-graph guard — the enforceable requirement is the closed boundary, not a specific mechanism. **(Resolved as built, 2026-06-19 audit:** deferring the transitive `core.physics` import (`parameter_manager.py:21`) was *sufficient* on its own — the `config/__init__.py` re-exports stay eager but no longer pull JAX, and the import-graph guard confirms it. The earlier "1A must *also* make `config/__init__.py` lazy/minimal" wording over-specified a second mechanism that turned out unnecessary.**)** Until the boundary holds, in-process validation is impossible JAX-free and **must run in a short-lived worker** (spec §3/§8). The Data reader uses **h5py directly** — never `xpcsjax.data` (which pulls JAX). Task 5's import-graph guard is the gate that proves the boundary; if it fails, the worker fallback applies.
- **Display-only downsampling.** The Data-tab C₂ preview uses the Plan-G `rasterize` block-mean decimation, explicitly labeled display rendering — never analysis downsampling; no fit consumes it.
- **View layer logic-free.** Validation, template loading, and HDF5 reading are testable non-Qt units (`service/config.py`, `gui/data_inspect.py`); widgets only render.
- **GUI tests headless** (`QT_QPA_PLATFORM=offscreen`); pyqtgraph/h5py via `importorskip`. `make verify` green.
- **Lint/style.** ruff line-length 100, rules `E,F,W,I,B,UP,N`, NumPy docstrings (`D`) on `xpcsjax/**`.

---

### Task 1: JAX-free config validation + template helpers (`service/config.py`)

**Files:**
- Modify: `xpcsjax/service/config.py` (add `validate_config`, `available_modes`, `template_dict` + `ValidationReport`)
- Test: `tests/service/test_config_validate.py`

**Interfaces:**
- Consumes: `xpcsjax.config.parameter_registry.ParameterRegistry` (`get_bounds(name) -> tuple`, `get_all_param_names(mode, include_scaling=) -> list[str]`), `xpcsjax.config.types.AnalysisMode`. **Templates are read via `importlib.resources`, NOT `xpcsjax.cli.config_generator`** (which imports `ConfigManager` → JAX).
- Produces:
  - `ValidationReport` (frozen dataclass): `ok: bool`, `errors: list[str]`, `warnings: list[str]`.
  - `validate_config(config: dict) -> ValidationReport` — JAX-free; honors the real template schema `initial_parameters.{parameter_names, values}`: unknown `analysis_mode` → error; a `parameter_names` entry not used by the mode (`get_all_param_names`) → warning; `values`/`parameter_names` length mismatch → error; a numeric value outside the registry's per-name `get_bounds` → error.
  - `available_modes() -> list[str]` — the four mode strings.
  - `template_dict(mode: str) -> dict` — load the packaged mode YAML template via `importlib.resources.files("xpcsjax.config") / "templates" / <file>` (a mode→filename map), `yaml.safe_load`.

- [ ] **Step 1: Write the failing test**

Create `tests/service/test_config_validate.py`:

```python
"""JAX-free config validation + template loading."""

import pytest

from xpcsjax.service.config import (
    ValidationReport,
    available_modes,
    template_dict,
    validate_config,
)


def test_available_modes_are_the_four_known():
    modes = set(available_modes())
    assert modes == {"static_isotropic", "static_anisotropic", "laminar_flow", "two_component"}


# Configs use the template schema: initial_parameters.{parameter_names, values}.
def _ip(names, values):
    return {"initial_parameters": {"parameter_names": names, "values": values}}


def test_valid_config_passes():
    cfg = {"analysis_mode": "static_isotropic", **_ip(["D0", "alpha", "D_offset"], [1000.0, -1.2, 0.0])}
    rep = validate_config(cfg)
    assert isinstance(rep, ValidationReport)
    assert rep.ok and rep.errors == []


def test_unknown_mode_is_an_error():
    rep = validate_config({"analysis_mode": "nope", **_ip([], None)})
    assert not rep.ok
    assert any("mode" in e.lower() for e in rep.errors)


def test_out_of_bounds_value_is_an_error():
    # D0 (diffusion) must be positive; a negative value is out of registry bounds.
    cfg = {"analysis_mode": "static_isotropic", **_ip(["D0", "alpha", "D_offset"], [-5.0, -1.2, 0.0])}
    rep = validate_config(cfg)
    assert not rep.ok
    assert any("D0" in e for e in rep.errors)


def test_parameter_not_used_by_mode_is_warned():
    # two_component uses v_beta, not beta -> a warning (not a hard error).
    rep = validate_config({"analysis_mode": "two_component", **_ip(["beta"], [1.0])})
    assert any("beta" in w for w in rep.warnings)


def test_values_length_mismatch_is_an_error():
    cfg = {"analysis_mode": "static_isotropic", **_ip(["D0", "alpha", "D_offset"], [1.0])}
    assert not validate_config(cfg).ok


def test_template_dict_loads_a_mode_template():
    tpl = template_dict("laminar_flow")
    assert isinstance(tpl, dict)
    assert "initial_parameters" in tpl
```

- [ ] **Step 2: Run → fail; Step 3: implement in `service/config.py`**

> **Additive — merge into the existing B2 module.** `service/config.py` already exists from Plan B2 (with its module docstring + `load_config`). Add the imports below to the module's existing import block (don't duplicate `from __future__ import annotations`) and append the new symbols; do **not** replace the file.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files

import yaml

from xpcsjax.config.parameter_registry import ParameterRegistry
from xpcsjax.config.types import AnalysisMode

# Mode -> packaged template filename. Read directly via importlib.resources so we
# NEVER import xpcsjax.cli.config_generator (which imports ConfigManager -> JAX).
_TEMPLATE_FILES = {
    "static_isotropic": "xpcsjax_static_isotropic.yaml",
    "static_anisotropic": "xpcsjax_static_anisotropic.yaml",
    "laminar_flow": "xpcsjax_laminar_flow.yaml",
    "two_component": "xpcsjax_two_component.yaml",
}


@dataclass(frozen=True)
class ValidationReport:
    """JAX-free outcome of validating a config dict."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def available_modes() -> list[str]:
    """Return the four known analysis-mode strings."""
    return [m.value for m in AnalysisMode]


def template_dict(mode: str) -> dict:
    """Load the packaged YAML template for ``mode`` (JAX-free; raises on unknown mode)."""
    if mode not in _TEMPLATE_FILES:
        raise ValueError(f"unknown mode: {mode!r} (expected one of {available_modes()})")
    path = files("xpcsjax.config") / "templates" / _TEMPLATE_FILES[mode]
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def validate_config(config: dict) -> ValidationReport:
    """Validate a config dict JAX-free against the registry.

    Honors the template schema ``initial_parameters.{parameter_names, values}``.
    Bounds use the registry's per-name global bounds (``get_bounds``); config-level
    ``parameter_space.bounds`` overrides are the fit-time authority, so this is a
    lightweight editor sanity check, not the final bounds resolution.
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        mode_enum = AnalysisMode(config.get("analysis_mode"))
    except ValueError:
        return ValidationReport(
            ok=False,
            errors=[
                f"unknown analysis_mode: {config.get('analysis_mode')!r} "
                f"(expected one of {available_modes()})"
            ],
        )

    registry = ParameterRegistry()
    expected = set(registry.get_all_param_names(mode_enum, include_scaling=False))

    ip = config.get("initial_parameters") or {}
    names = list(ip.get("parameter_names", []) or [])
    values = ip.get("values")

    for name in names:
        if name not in expected:  # mode-specific membership (e.g. v_beta vs beta)
            warnings.append(f"parameter {name!r} is not used by mode {mode_enum.value}")

    if isinstance(values, list):
        if len(values) != len(names):
            errors.append(
                f"initial_parameters.values has {len(values)} entries "
                f"but parameter_names has {len(names)}"
            )
        for name, value in zip(names, values, strict=False):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                errors.append(f"{name}: value {value!r} is not numeric")
                continue
            if name in expected:
                lo, hi = registry.get_bounds(name)
                if not (lo <= numeric <= hi):
                    errors.append(f"{name}={numeric} is outside bounds ({lo}, {hi})")

    return ValidationReport(ok=not errors, errors=errors, warnings=warnings)
```

- [ ] **Step 4: Run + lint + commit**

Run: `uv run pytest tests/service/test_config_validate.py -q` → PASS.

```bash
git add xpcsjax/service/config.py tests/service/test_config_validate.py
git commit -m "feat(service): JAX-free config validation + template helpers"
```

---

### Task 2: Config tab — form editor + raw-YAML toggle + live validation

**Files:**
- Create: `xpcsjax/gui/views/config_editor.py`
- Test: `tests/gui/test_config_editor.py`

**Interfaces:**
- Produces: `ConfigEditor(QWidget)`:
  - `set_mode(mode: str)` — loads `template_dict(mode)` and builds one numeric field per `initial_parameters.parameter_names` entry (seeded from `values` when non-null), keeping the rest of the template intact (mode dropdown drives this).
  - `current_config() -> dict` — assembles the **real schema**: the loaded template with `analysis_mode` = the dropdown and `initial_parameters.values` = the field values **aligned to `parameter_names`** (or parses the raw-YAML pane when in raw mode). This is what gets written to the temp YAML the Plan-C worker loads, so it must round-trip into a valid config.
  - `validate()` — runs `validate_config(current_config())`, renders `errors`/`warnings` into a status label; returns the `ValidationReport`.
  - `toggle_raw(on: bool)` — swaps a `QPlainTextEdit` raw-YAML view (round-trips `yaml.safe_dump`/`safe_load`) with the form.
  - Signal `config_ready = Signal(dict)` — emitted when a valid config is confirmed (toolbar Run consumes it / writes a temp YAML for the worker).

- [ ] **Step 1: Write the failing test** (`tests/gui/test_config_editor.py`)

```python
import pytest

pytest.importorskip("PySide6")

from xpcsjax.gui.views.config_editor import ConfigEditor  # noqa: E402


def test_set_mode_populates_form(qtbot):
    w = ConfigEditor()
    qtbot.addWidget(w)
    w.set_mode("static_isotropic")
    cfg = w.current_config()
    assert cfg["analysis_mode"] == "static_isotropic"


def test_validate_flags_bad_value(qtbot):
    w = ConfigEditor()
    qtbot.addWidget(w)
    w.set_mode("static_isotropic")
    w.set_parameter("D0", -5.0)  # out of bounds
    rep = w.validate()
    assert not rep.ok


def test_raw_yaml_round_trips(qtbot):
    w = ConfigEditor()
    qtbot.addWidget(w)
    w.set_mode("laminar_flow")
    w.toggle_raw(True)
    text = w.raw_text()
    assert "analysis_mode" in text
    w.toggle_raw(False)  # parse back without error
    assert w.current_config()["analysis_mode"] == "laminar_flow"
```

- [ ] **Step 2: Run → fail; Step 3: implement `config_editor.py`**

Build a `QWidget` with: a mode `QComboBox` (`available_modes()`); a `QFormLayout` with **one row per `initial_parameters.parameter_names` entry** (a `QLineEdit` — **not** a bounds-clamped `QDoubleSpinBox` — seeded from the template's `values[i]` when present; show `registry.get_bounds(name)` as a **tooltip only**, do **not** call `setRange(*get_bounds(name))`, so an out-of-bounds value reaches `validate_config` instead of being silently clamped to the nearest bound and passing validation spuriously); a "Raw YAML" `QCheckBox` swapping a `QPlainTextEdit` (via `QStackedWidget`); a "Validate" button → `validate()` writing `rep.errors`/`rep.warnings` into a `QLabel`; and `set_parameter(name, value)` (updates the field for that name) / `raw_text()` test helpers. `current_config()` rebuilds the template dict with `analysis_mode` = dropdown and `initial_parameters.values` = the fields **in `parameter_names` order** (or `yaml.safe_load(raw_text())` in raw mode); `config_ready` fires on a passing validate. Keep all numeric/validation logic delegated to Task 1 — the widget only renders + reshapes.

- [ ] **Step 4: Run + lint + commit**

Run: `uv run pytest tests/gui/test_config_editor.py -q` → PASS.

```bash
git add xpcsjax/gui/views/config_editor.py tests/gui/test_config_editor.py
git commit -m "feat(gui): Config tab — form editor + raw-YAML + live validation"
```

---

### Task 3: Data tab — JAX-free HDF5 metadata + two-time C₂ preview

**Files:**
- Create: `xpcsjax/gui/data_inspect.py` (JAX-free, h5py)
- Create: `xpcsjax/gui/views/data_panel.py`
- Test: `tests/gui/test_data_inspect.py`, `tests/gui/test_data_panel.py`

**Interfaces:**
- Produces (`data_inspect.py`, JAX-free — **h5py only, never `xpcsjax.data`**):
  - `read_hdf5_metadata(path) -> list[DatasetInfo]` where `DatasetInfo(name: str, shape: tuple, dtype: str)` — walks the file listing datasets + shapes + dtypes, **without loading array data**.
  - `read_c2_preview(path, dataset: str, *, data_type: str | None = None, phi_index: int = 0, max_dim: int = 512) -> np.ndarray | None` — for a known `data_type`, opens the per-format C₂ **group** (named by the **shared layout descriptor** `_C2_PREVIEW_LAYOUTS`, which **mirrors `xpcsjax.data`'s real layout** — both formats store C₂ as a group of per-angle 2-D **half**-matrices, spec §6), selects the `phi_index`-th key (creation order for APS-old, sorted for APS-U), reads it via a **bounded strided hyperslab**, and reconstructs the full matrix (`c2_half + c2_half.T`, diagonal halved — symmetric reconstruction commutes with the strided read). Returns `None` ("preview unavailable") when the group is absent/empty — it never renders a guessed layout. With `data_type=None` a best-effort raw 3-D-slice / 2-D heuristic (no reconstruction) applies. Then applies the Plan-G `rasterize`.
- Produces (`data_panel.py`): `DataPanel(QWidget)` — a file path field, a metadata `QTreeWidget` (one row per `DatasetInfo`), a dataset selector, and a Plan-G `TwoTimeMapView` showing `read_c2_preview(...)`; `load(path)` populates everything. `load(path, data_type=None)` forwards the active config's `data_type` (when one is loaded) to `read_c2_preview` so the angle axis comes from the shared layout descriptor; with no config it falls back to the heuristic and shows "preview unavailable" on an unrecognized layout.

- [ ] **Step 1: Write the failing test** (`tests/gui/test_data_inspect.py`)

```python
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from xpcsjax.gui.data_inspect import DatasetInfo, read_c2_preview, read_hdf5_metadata  # noqa: E402


def _make_h5(path):
    with h5py.File(path, "w") as f:
        f.attrs["detector"] = "test"
        f.create_dataset("c2", data=np.random.default_rng(0).random((2, 64, 64)))
        f.create_dataset("phi", data=np.array([0.0, 45.0]))


def test_metadata_lists_datasets_without_loading(tmp_path):
    p = tmp_path / "d.h5"
    _make_h5(p)
    infos = read_hdf5_metadata(p)
    names = {i.name.split("/")[-1]: i for i in infos}
    assert isinstance(infos[0], DatasetInfo)
    assert names["c2"].shape == (2, 64, 64)
    assert names["phi"].shape == (2,)


def test_c2_preview_slices_and_downsamples(tmp_path):
    p = tmp_path / "d.h5"
    _make_h5(p)
    img = read_c2_preview(p, "c2", phi_index=1, max_dim=32)
    assert img is not None and img.ndim == 2
    assert max(img.shape) <= 32


def test_c2_preview_missing_dataset_returns_none(tmp_path):
    p = tmp_path / "d.h5"
    _make_h5(p)
    assert read_c2_preview(p, "does_not_exist") is None


def test_c2_preview_missing_group_returns_none(tmp_path):
    # Spec §6: a known data_type whose C₂ group is absent yields "preview
    # unavailable" (None), never a guessed render.
    p = tmp_path / "d.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("c2", data=np.zeros((64, 64)))  # no exchange/C2T_all group
    assert read_c2_preview(p, "c2", data_type="aps_old") is None


def test_c2_preview_reconstructs_group_half_matrix(tmp_path):
    # Real formats store C₂ as a GROUP of per-angle 2-D half-matrices; the preview
    # reconstructs c2_half + c2_half.T (diag halved), mirroring the loader.
    p = tmp_path / "u.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("xpcs/twotime/correlation_map")
        for k in ("c2_00001", "c2_00002"):
            g.create_dataset(k, data=np.tril(np.ones((48, 48))))
    img = read_c2_preview(p, "unused", data_type="aps_u", phi_index=1, max_dim=32)
    assert img is not None and img.ndim == 2 and max(img.shape) <= 32
```

- [ ] **Step 2: Run → fail; Step 3: implement `data_inspect.py`**

```python
"""JAX-free HDF5 inspection for the Data tab (h5py only — no xpcsjax.data/JAX)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from xpcsjax.gui.views.raster import rasterize


@dataclass(frozen=True)
class DatasetInfo:
    """Shape/dtype metadata for one HDF5 dataset (no array data loaded)."""

    name: str
    shape: tuple
    dtype: str


def read_hdf5_metadata(path: str | Path) -> list[DatasetInfo]:
    """List every dataset in the file with shape + dtype, loading no array data."""
    infos: list[DatasetInfo] = []

    def _visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            infos.append(DatasetInfo(name=name, shape=tuple(obj.shape), dtype=str(obj.dtype)))

    with h5py.File(path, "r") as f:
        f.visititems(_visit)
    return infos


# Shared C₂ layout descriptor (spec §6) — keyed by data_type. The GUI preview reads
# HDF5 directly (JAX-free, bypassing xpcsjax.data), so it must NOT re-derive the
# layout ad hoc: this table is the single source of truth and MUST mirror the loader.
# On disk BOTH real formats store C₂ as an HDF5 **group of per-angle 2-D
# half-matrices** (NOT a 3-D dataset); the full matrix is reconstructed as
# ``c2_half + c2_half.T`` with the diagonal halved, mirroring
# ``xpcsjax/data/xpcs_loader.py::_reconstruct_full_matrix`` (verified at
# xpcs_loader.py:1283/1356-1357, :1553/1578-1580, :1695-1708). ``key_order`` matches
# the loader: APS-old iterates keys in HDF5 creation order, APS-U sorts them.
_C2_PREVIEW_LAYOUTS: dict[str, dict] = {
    "aps_old": {"group": "exchange/C2T_all", "key_order": "creation"},          # integer keys, creation order
    "aps_u": {"group": "xpcs/twotime/correlation_map", "key_order": "sorted"},  # zero-padded c2_* keys
}


def _reconstruct_c2(half: np.ndarray) -> np.ndarray:
    """Full two-time matrix from a stored half: ``c2_half + c2_half.T``, diagonal halved.

    Mirrors ``xpcsjax/data/xpcs_loader.py::_reconstruct_full_matrix``. It commutes
    with equal-stride decimation — ``_reconstruct_c2(half[::s, ::s])`` equals
    ``_reconstruct_c2(half)[::s, ::s]`` — so a bounded strided read stays correct.
    """
    full = half + half.T
    full[np.diag_indices(full.shape[0])] /= 2
    return full


def read_c2_preview(
    path: str | Path,
    dataset: str,
    *,
    data_type: str | None = None,
    phi_index: int = 0,
    max_dim: int = 512,
) -> np.ndarray | None:
    """Read one per-angle two-time C₂ matrix and block-mean downsample it for display.

    For a known ``data_type`` the angle's matrix lives in the HDF5 group named by
    ``_C2_PREVIEW_LAYOUTS`` as a 2-D **half**-matrix; we pick the ``phi_index``-th
    key (creation order for APS-old, sorted for APS-U — mirroring the loader), read it
    with a bounded strided hyperslab, and reconstruct it via :func:`_reconstruct_c2`.
    Returns ``None`` ("preview unavailable") when the C₂ group is absent or empty — it
    never renders a guessed layout. When ``data_type`` is ``None`` a best-effort
    heuristic reads ``dataset`` directly (raw 2-D matrix, or one slice of a 3-D array)
    without reconstruction. ``dataset`` is ignored for a known format.
    """
    with h5py.File(path, "r") as f:
        layout = _C2_PREVIEW_LAYOUTS.get(data_type) if data_type else None
        if layout is not None:
            grp = f.get(layout["group"])
            if not isinstance(grp, h5py.Group) or len(grp) == 0:
                return None  # C₂ group absent/empty → preview unavailable
            keys = sorted(grp.keys()) if layout["key_order"] == "sorted" else list(grp.keys())
            half_dset = grp[keys[max(0, min(int(phi_index), len(keys) - 1))]]
            if half_dset.ndim != 2 or half_dset.shape[0] != half_dset.shape[1]:
                return None
            step = max(1, int(np.ceil(max(half_dset.shape) / max_dim)))
            arr = _reconstruct_c2(np.asarray(half_dset[::step, ::step]))  # bounded + symmetric
        elif dataset not in f:
            return None
        else:  # data_type unknown: best-effort raw read, no reconstruction
            dset = f[dataset]
            if dset.ndim == 3:  # heuristic: assume (n_phi, t, t)
                idx = max(0, min(int(phi_index), dset.shape[0] - 1))
                step = max(1, int(np.ceil(max(dset.shape[1:]) / max_dim)))
                arr = np.asarray(dset[idx, ::step, ::step])
            elif dset.ndim == 2:
                step = max(1, int(np.ceil(max(dset.shape) / max_dim)))
                arr = np.asarray(dset[::step, ::step])
            else:
                return None
    return rasterize(arr, max_dim=max_dim)
```

- [ ] **Step 4: Build `DataPanel` (TDD)**

`tests/gui/test_data_panel.py` (under `importorskip("PySide6")`+`importorskip("pyqtgraph")`+`importorskip("h5py")`): build a tiny `.h5`, `DataPanel().load(path)`, assert the metadata tree has a row per dataset and the preview shows an image. Implement `DataPanel`: a read-only path field + "Browse" (`QFileDialog.getOpenFileName`), a `QTreeWidget` filled from `read_hdf5_metadata`, a dataset `QComboBox`, and an embedded Plan-G `TwoTimeMapView` driven by `read_c2_preview`.

- [ ] **Step 5: Run + lint + commit**

Run: `uv run pytest tests/gui/test_data_inspect.py tests/gui/test_data_panel.py -q` → PASS.

```bash
git add xpcsjax/gui/data_inspect.py xpcsjax/gui/views/data_panel.py tests/gui/test_data_inspect.py tests/gui/test_data_panel.py
git commit -m "feat(gui): Data tab — JAX-free HDF5 metadata + two-time preview"
```

---

### Task 4: Inspector dock (params + uncertainties + diagnostics) + Fit-settings summary

**Files:**
- Modify: `xpcsjax/gui/result_loader.py` (add a `diagnostics` field to `ResultSummary`, populated from `meta["nlsq_diagnostics"]`)
- Create: `xpcsjax/gui/views/inspector.py`, `xpcsjax/gui/views/fit_panel.py`
- Test: `tests/gui/test_inspector.py`, `tests/gui/test_result_loader.py` (extend)

**Interfaces:**
- Modify: `ResultSummary` gains `diagnostics: dict = field(default_factory=dict)`; `load_result_summary` sets it to `meta.get("nlsq_diagnostics", {})` (the block `result_saving.py:123` writes).
- Produces:
  - `InspectorDock(QWidget)`: `show_summary(summary: ResultSummary | None)` — a parameters table (name | value | ± uncertainty, joining `summary.parameters` + `summary.uncertainties`) and a diagnostics `QTreeWidget` rendering `summary.diagnostics`. Empty/None → cleared.
  - `FitPanel(QWidget)`: `show_settings(config: dict, overrides: dict | None)` — a read-only resolved-settings summary (mode, key parameters, applied overrides). Logic-free rendering.

- [ ] **Step 1: Extend `ResultSummary.diagnostics` (TDD)**

Extend `tests/gui/test_result_loader.py`: write an `nlsq_result.json` whose `metadata` includes `"nlsq_diagnostics": {"hierarchical_active": true}`, assert `load_result_summary(dir).diagnostics == {"hierarchical_active": True}`. Implement: add `diagnostics: dict = field(default_factory=dict)` to `ResultSummary` (defaulted, so existing constructions stay valid — same pattern as `uncertainties`) and `diagnostics=meta.get("nlsq_diagnostics", {})` in `load_result_summary`.

- [ ] **Step 2: Write the failing Inspector test** (`tests/gui/test_inspector.py`)

```python
import pytest

pytest.importorskip("PySide6")

from pathlib import Path  # noqa: E402

from xpcsjax.gui.result_loader import ResultSummary  # noqa: E402
from xpcsjax.gui.views.inspector import InspectorDock  # noqa: E402


def _summary():
    return ResultSummary(
        result_dir=Path("."),
        success=True,
        convergence_status="converged",
        chi_squared=1.0,
        reduced_chi_squared=0.9,
        quality_flag="good",
        parameters={"D0": 1234.5},
        uncertainties={"D0": 12.0},
        diagnostics={"hierarchical_active": True},
    )


def test_inspector_renders_params_and_diagnostics(qtbot):
    w = InspectorDock()
    qtbot.addWidget(w)
    w.show_summary(_summary())
    assert w.param_row_count() == 1
    assert w.diagnostics_row_count() >= 1


def test_inspector_clears_on_none(qtbot):
    w = InspectorDock()
    qtbot.addWidget(w)
    w.show_summary(_summary())
    w.show_summary(None)
    assert w.param_row_count() == 0
```

- [ ] **Step 3: Implement `inspector.py` + `fit_panel.py`**

`InspectorDock`: a `QTableWidget` (columns name/value/uncertainty, one row per `summary.parameters` joined with `summary.uncertainties.get(name)`) + a `QTreeWidget` populated by recursively walking `summary.diagnostics`; expose `param_row_count()` / `diagnostics_row_count()`; `show_summary(None)` clears both. `FitPanel`: a `QPlainTextEdit`(read-only) rendering `mode`, the initial parameters, and any `overrides` as a human-readable summary.

- [ ] **Step 4: Run + lint + commit**

Run: `uv run pytest tests/gui/test_inspector.py tests/gui/test_result_loader.py -q` → PASS.

```bash
git add xpcsjax/gui/result_loader.py xpcsjax/gui/views/inspector.py xpcsjax/gui/views/fit_panel.py tests/gui/test_inspector.py tests/gui/test_result_loader.py
git commit -m "feat(gui): Inspector dock (params/uncertainties/diagnostics) + Fit summary"
```

---

### Task 5: Mount the surfaces in `MainWindow` + JAX-free guard

**Files:**
- Modify: `xpcsjax/gui/views/main_window.py` (Config/Data/Fit center tabs + Inspector right dock; route `config_ready` → toolbar Run; selecting a finished run → `InspectorDock.show_summary`)
- Modify: `tests/gui/test_gui_jax_free.py`
- Test: `tests/gui/test_workbench_surfaces.py`

**Interfaces:** `MainWindow` gains a center `QTabWidget` (Data / Config / Fit / Results) and an Inspector `QDockWidget` (right). `ConfigEditor.config_ready` feeds the run path (write the dict to a temp YAML the worker loads, or pass through `service`); on run-selection, `InspectorDock.show_summary(run.summary)`; `FitPanel.show_settings` mirrors the resolved config.

- [ ] **Step 1–3: Wire + integration test (TDD)**

`tests/gui/test_workbench_surfaces.py` (`importorskip` PySide6/pyqtgraph): `window, _ = build_workbench()`; assert the center tab widget has the Data/Config/Fit/Results tabs and the Inspector dock exists; drive `window.show_inspector(summary)` and assert the dock renders. Wire `MainWindow`: add the tabs + dock in `__init__`; connect `ConfigEditor.config_ready`; on `runs_selected`/finish, call `self._inspector.show_summary(run.summary)`.

- [ ] **Step 4: Extend the JAX-free guard + suite + gate**

Extend `tests/gui/test_gui_jax_free.py` to probe `xpcsjax.service.config`, `xpcsjax.gui.data_inspect` (both must import **without** `jax`/`jaxlib` in `sys.modules`) and — under `importorskip` — `xpcsjax.gui.views.{config_editor,data_panel,fit_panel,inspector}`. This is the keystone check: it proves the Config validator + HDF5 reader stay on the JAX-free side (catching any regression that re-introduces a `core.physics`/`xpcsjax.data` edge).

Run: `uv run pytest tests/gui/ tests/service/test_config_validate.py -q` → PASS. `make verify` → green.

```bash
git add xpcsjax/gui/views/main_window.py tests/gui/test_gui_jax_free.py tests/gui/test_workbench_surfaces.py
git commit -m "feat(gui): mount Config/Data/Fit tabs + Inspector dock"
```

---

## Self-Review

**1. Spec coverage (the §14-deferred surfaces):** Config tab form editor + live validation + raw-YAML toggle (Tasks 1–2), Data tab HDF5 metadata + two-time preview (Task 3), Fit tab resolved-settings (Task 4), Inspector dock params+uncertainties+diagnostics (Task 4), all mounted (Task 5). Every surface §5 promised now has a task. ✔

**2. Placeholder scan:** No TBD. The registry API (`get_bounds`/`get_all_param_names`), template schema (`parameter_names`/`values`), result-JSON diagnostics key (`nlsq_diagnostics`, `result_saving.py:123`), and the JAX-free import probes are all **empirically verified**, not assumed (see Review provenance).

**3. Type consistency:** `ValidationReport`/`validate_config`/`template_dict`/`available_modes` (Task 1) feed `ConfigEditor` (Task 2); `DatasetInfo`/`read_hdf5_metadata`/`read_c2_preview` (Task 3) feed `DataPanel` + reuse Plan-G `rasterize`/`TwoTimeMapView`; `ResultSummary` (+ `uncertainties` from Plan D, + new `diagnostics`) feeds `InspectorDock` (Task 4); `MainWindow` (Task 5) consumes `ConfigEditor.config_ready` + `InspectorDock.show_summary`.

**4. JAX-free boundary:** in-process validation rests on Plan 1A making `xpcsjax.config.parameter_registry`/`types` JAX-free. A re-probe found `config/__init__` has *multiple* eager JAX edges (`manager`/`parameter_manager`/`physics_validators`/`parameter_space`), so 1A may need to lazy-ify `config/__init__` (strengthened there) — Plan I's prereq is a **hard gate** with a documented worker-validation fallback if unmet. The Data reader uses h5py directly. Both pinned by Task 5's import-graph guard.

---

## Review provenance (2026-06-18)

Reviewed by codex (→ NEEDS-REWORK, 10 findings; agy empty) + a Claude verification pass that **empirically probed every claim**. All fixes applied:
- **CRITICAL — config imports not JAX-free:** a probe confirmed `import xpcsjax.config.parameter_registry`/`types` loads JAX *today* via `config/__init__.py`, which eagerly imports `manager`/`parameter_manager`/`physics_validators`/`parameter_space` (all currently JAX-loading). Plan 1A's single-edge fix may be insufficient → **strengthened Plan 1A** (Step 7 + a new test asserting `parameter_registry`/`types` are JAX-free; lazy-ify `config/__init__` if independent edges survive) and made Plan I's prereq a **hard gate** with a worker-validation fallback.
- **HIGH — registry API mismatch:** `ParameterInfo` has no `.bounds`/`.is_within_bounds`; switched to `registry.get_bounds(name) -> tuple` (verified at `parameter_registry.py:636`).
- **HIGH — mode membership:** validation now checks `parameter_names` against `get_all_param_names(mode, include_scaling=False)` (so `beta` is flagged in `two_component`, which uses `v_beta`).
- **HIGH — template schema:** `validate_config` + `ConfigEditor` now honor the real `initial_parameters.{parameter_names, values}` schema (verified in the templates), not `{name: value}`.
- **HIGH — `config_generator` pulls JAX:** `template_dict` now reads templates via `importlib.resources` + a mode→filename map, never importing the CLI module.
- **MEDIUM — huge whole-read:** `read_c2_preview` uses a bounded strided hyperslab (`dset[::step, ::step]`), never `dset[()]`.
- **MEDIUM — C₂ preview layout contract (spec §6):** the C₂ **group path + key order** is resolved via the shared `_C2_PREVIEW_LAYOUTS` descriptor (keyed by `data_type`, mirroring `xpcsjax.data` — both formats store C₂ as a group of per-angle 2-D half-matrices, NOT a 3-D dataset), and the preview reconstructs the full matrix exactly as the loader does (`c2_half + c2_half.T`, diag halved); an absent group returns `None` ("preview unavailable") instead of guessing (tested by `test_c2_preview_missing_group_returns_none` and `test_c2_preview_reconstructs_group_half_matrix`). Removes the duplicated-layout-knowledge hazard.
- **LOW — metadata attrs:** dropped the unkept "root attrs" promise (datasets/shape/dtype only).
- codex **confirmed correct:** the `nlsq_diagnostics` key (`result_saving.py:123`), the defaulted `diagnostics`/`uncertainties` `ResultSummary` additions, and the `AnalysisMode` StrEnum contract.

## Completes the roadmap

With Plan I, every surface in spec §5 has an implementing plan: design spec → 1A · 1B · B2 (service) → C (IPC) → D (first fit) → E · E2 (diagnostics) → F (project/queue/compare) → G (interactive plots) → H (packaging) → **I (workbench surfaces)**.
