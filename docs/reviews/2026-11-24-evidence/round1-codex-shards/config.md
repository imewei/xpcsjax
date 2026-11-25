# Codex review — shard config

## Files read

xpcsjax/config/__init__.py — 92 — full  
xpcsjax/config/heterodyne_parameter_manager.py — 831 — full  
xpcsjax/config/heterodyne_parameter_names.py — 118 — full  
xpcsjax/config/heterodyne_parameter_space.py — 887 — full  
xpcsjax/config/heterodyne_physics_validators.py — 537 — full  
xpcsjax/config/manager.py — 1523 — full  
xpcsjax/config/parameter_manager.py — 776 — full  
xpcsjax/config/parameter_registry.py — 943 — full  
xpcsjax/config/parameter_space.py — 354 — full  
xpcsjax/config/physics_validation_base.py — 117 — full  
xpcsjax/config/physics_validators.py — 321 — full  
xpcsjax/config/templates/xpcsjax_laminar_flow.yaml — 866 — full  
xpcsjax/config/templates/xpcsjax_static_anisotropic.yaml — 1172 — full  
xpcsjax/config/templates/xpcsjax_static_isotropic.yaml — 655 — full  
xpcsjax/config/templates/xpcsjax_two_component.yaml — 983 — full  
xpcsjax/config/types.py — 467 — full

## Bugs

### config-B1: Template legacy fields override the explicitly selected dataset

- Location: xpcsjax/config/manager.py:1475
- Severity: Required
- Failure scenario: Set `experimental_data.file_path` to `/data/chosen.h5` in the static anisotropic template and leave its legacy folder and filename populated. The normalized config replaces the chosen file with the legacy file. If the legacy file exists, the fit can analyze the wrong experiment.
- Root cause: `_normalize_experimental_data()` rebuilds `file_path` whenever both legacy fields are present. It never checks for an explicit `file_path` or synchronizes the fields consumed by the loader.
- Checked: The static anisotropic template explicitly promises `file_path` precedence. `service.data.load_dataset()` passes the normalized config to `XPCSDataLoader`, whose validation and load paths use the legacy pair. The relative-path tests supply matching filenames or omit the legacy filename.
- Red test:
  ```python
  from pathlib import Path
  from xpcsjax.config import ConfigManager

  cfg = ConfigManager(config_override={"experimental_data": {
      "file_path": "/data/chosen.h5",
      "data_folder_path": "/data",
      "data_file_name": "sample.hdf",
  }})
  exp = cfg.config["experimental_data"]
  assert exp["file_path"] == "/data/chosen.h5"
  assert Path(exp["data_folder_path"]) / exp["data_file_name"] == Path(exp["file_path"])
  ```
- Confidence: high

### config-B2: Equal heterodyne bounds leave a fixed coordinate in the optimizer’s free vector

- Location: xpcsjax/config/heterodyne_parameter_space.py:413
- Severity: Required
- Failure scenario: Configure `v0` with `min: 500` and `max: 500`. The feasible value is fixed at 500, yet `v0` still counts as varying and reaches the solver with equal bounds. Fit paths that produce a result count it as an estimated parameter in reduced chi-squared.
- Root cause: Bounds overlays store equality as a valid interval. Only vary flags, ties, and explicit fixed entries determine the free vector. No final step reconciles equal bounds with the value or vary flag.
- Checked: `test_equal_bounds_allowed` explicitly describes equality as fixing the parameter but only checks the stored tuple. `HeterodyneParameterManager.get_bounds()` slices by vary flags, and the engine and direct local paths derive the parameter count from that vector.
- Red test:
  ```python
  from xpcsjax.config.heterodyne_parameter_manager import HeterodyneParameterManager

  pm = HeterodyneParameterManager.from_config({
      "parameters": {"velocity": {"v0": {"min": 500, "max": 500}}},
  })
  assert "v0" not in pm.varying_names
  assert pm.get_parameter_dict()["v0"] == 500
  ```
- Confidence: high

### config-B3: Misspelled heterodyne constraints change the fitted model instead of stopping the fit

- Location: xpcsjax/config/heterodyne_parameter_space.py:509
- Severity: Required
- Failure scenario: `active_parameters: [D0_ref, D0_sampel]` silently freezes `D0_sample` while fitting `D0_ref`. Independently, `fixed_parameters: {v_offest: 0}` leaves `v_offset` free after logging a warning.
- Root cause: The active whitelist intersects with known names without validating unmatched entries. `_apply_fixed_parameters()` likewise logs and skips unknown names at line 548. The manager’s zero-varying guard only rejects configurations that leave no coordinates varying.
- Checked: The public fitting dispatcher constructs `HeterodyneModel` from the raw config, and the model uses `HeterodyneParameterManager.from_config()`. The configuration guide requires a `ValueError` for either typo. Existing heterodyne fixed-parameter tests check known names, alias collisions, and an empty whitelist. The homodyne typo guard is on a separate path.
- Red test:
  ```python
  import pytest
  from xpcsjax.config.heterodyne_parameter_manager import HeterodyneParameterManager

  @pytest.mark.parametrize("initial, typo", [
      ({"active_parameters": ["D0_ref", "D0_sampel"]}, "D0_sampel"),
      ({"fixed_parameters": {"v_offest": 0}}, "v_offest"),
  ])
  def test_reject_unknown_heterodyne_constraints(initial, typo):
      with pytest.raises(ValueError, match=typo):
          HeterodyneParameterManager.from_config({"initial_parameters": initial})
  ```
- Confidence: high

### config-B4: Quoted heterodyne `vary: "false"` frees the parameter

- Location: xpcsjax/config/heterodyne_parameter_space.py:399
- Severity: Required
- Failure scenario: A generated YAML config contains `parameters.velocity.v0: {value: 500, vary: "false"}`. The fit varies `v0` and includes it in the optimizer’s parameter count. The same failure occurs with `vary: "false"` in a `parameter_space.bounds` entry.
- Root cause: Both overlays apply Python truthiness directly to the parsed scalar. A nonempty string evaluates to `True`.
- Checked: `HeterodyneParameterManager.from_config()` retains the parsed flag, and its varying names and indices feed the fit. `test_heterodyne_grouped_coercion.py` covers quoted numeric values and a native `True` vary flag.
- Red test:
  ```python
  import pytest
  from xpcsjax.config.heterodyne_parameter_manager import HeterodyneParameterManager

  @pytest.mark.parametrize("cfg", [
      {"parameters": {"velocity": {"v0": {"value": 500, "vary": "false"}}}},
      {"parameter_space": {"bounds": [{"name": "v0", "vary": "false"}]}},
  ])
  def test_quoted_false_never_frees_v0(cfg):
      try:
          pm = HeterodyneParameterManager.from_config(cfg)
      except ValueError:
          return  # Explicit rejection of a string flag also avoids the wrong fit.
      assert "v0" not in pm.varying_names
  ```
- Confidence: high

### config-B5: Grouped heterodyne initial values are lost in config-driven previews

- Location: xpcsjax/config/manager.py:812
- Severity: Required
- Failure scenario: A `two_component` config with `parameters.velocity.v0.value: 42` and no flat `initial_parameters` starts the fitter at 42, but the standalone simulated plot evaluates the model at `v0=500000`, changing the cross-term oscillations.
- Root cause: `get_initial_parameters()` treats the missing flat section as a request for registry-bound midpoints. It never applies the grouped heterodyne overlay.
- Checked: `_plot_simulated_from_config()` takes the accessor’s values directly into the model evaluation. `HeterodyneModel.from_config()` uses the heterodyne parameter manager for the fit. The simulated-grid regression captures the time grid without asserting the parameter vector.
- Red test:
  ```python
  from xpcsjax.config import ConfigManager
  from xpcsjax.config.heterodyne_parameter_manager import HeterodyneParameterManager

  cfg = {"analysis_mode": "two_component",
         "parameters": {"velocity": {"v0": {"value": 42}}}}
  fit_values = HeterodyneParameterManager.from_config(cfg).get_parameter_dict()
  assert ConfigManager(config_override=cfg).get_initial_parameters()["v0"] == fit_values["v0"]
  ```
- Confidence: high

### config-B6: Blank optional heterodyne parameter groups crash model construction

- Location: xpcsjax/config/heterodyne_parameter_space.py:307
- Severity: Required
- Failure scenario: A valid flat two-component config with a blank `parameters:` section crashes with `TypeError`. A blank nested group such as `parameters: {velocity: null}` instead crashes with `AttributeError`, even when no grouped override is needed.
- Root cause: The optional overlay uses `.get(..., {})`, which preserves an explicit YAML null. The root is then passed to `set()` and each group to `.items()` at line 327.
- Checked: The public dispatcher passes the parsed mapping directly to `HeterodyneModel.from_config()`. ConfigManager leaves the optional grouped section untouched. Grouped coercion tests use mapping-valued sections.
- Red test:
  ```python
  import pytest
  from xpcsjax.config.heterodyne_parameter_manager import HeterodyneParameterManager

  @pytest.mark.parametrize("grouped", [None, {"velocity": None}])
  def test_null_group_preserves_flat_values(grouped):
      pm = HeterodyneParameterManager.from_config({
          "parameters": grouped,
          "initial_parameters": {"parameter_names": ["v0"], "values": [42]},
      })
      assert pm.get_parameter_dict()["v0"] == 42
  ```
- Confidence: high

### config-B7: Public homodyne `ParameterSpace` rejects a null bound that its manager accepts

- Location: xpcsjax/config/parameter_space.py:179
- Severity: Required
- Failure scenario: A `D0` bound with `min: null` and `max: 900` resolves to `(100.0, 900.0)` through `ParameterManager`. `ParameterSpace.from_config()` raises `TypeError` on the same config.
- Root cause: The manager removes null endpoints before merging with registry defaults. `ParameterSpace` then reparses the original entry and coerces both endpoints whenever both keys exist.
- Checked: `ParameterSpace.from_config()` constructs the manager successfully before taking the failing branch. `test_null_bound_falls_back_to_default` exercises only the manager. The public parameter-space tests check NaN and infinity rejection.
- Red test:
  ```python
  from xpcsjax.config import ParameterSpace

  cfg = {"parameter_space": {"bounds": [{"name": "D0", "min": None, "max": 900}]}}
  assert ParameterSpace.from_config(cfg).get_bounds("D0") == (100.0, 900.0)
  ```
- Confidence: high

### config-B8: `ConfigManager.load_config()` leaves cached parameters from the previous file

- Location: xpcsjax/config/manager.py:284
- Severity: Required
- Failure scenario: Read active parameters from a laminar manager, point it at the static isotropic template, and reload. `analysis_mode` becomes static isotropic, but the active-parameter accessor still returns the seven laminar parameters. A simulated preview can then fail when it sends that vector to the static model.
- Root cause: Reload replaces `self.config` without clearing `_cached_param_manager`. The cache survives with the previous config and mode. `update_config()` has its own invalidation, which reload bypasses.
- Checked: Both shipped templates reproduce the stale cache. `_get_parameter_manager()` returns its cached object whenever one exists. The public load tests cover single construction and validation errors, without warming the parameter cache before reload.
- Red test:
  ```python
  from xpcsjax.config import ConfigManager

  base = "xpcsjax/config/templates/xpcsjax_"
  cfg = ConfigManager(base + "laminar_flow.yaml")
  assert "gamma_dot_t0" in cfg.get_active_parameters()
  cfg.config_file = base + "static_isotropic.yaml"
  cfg.load_config()
  assert cfg.get_active_parameters() == ["D0", "alpha", "D_offset"]
  ```
- Confidence: high

### config-B9: Midpoint initialization drops explicitly configured per-angle scaling

- Location: xpcsjax/config/manager.py:835
- Severity: Minor
- Failure scenario: Leave `initial_parameters.values: null` as shipped and set `per_angle_scaling.contrast: [0.1, 0.2]`. The public initial-parameter accessor returns the physical midpoint values and loses both contrast entries.
- Root cause: The null-values branch returns before the independent per-angle scaling block is read. A missing `parameter_names` list takes another early return with the same effect.
- Checked: The static anisotropic template presents physical values and per-angle scaling as independent settings. The config regression for preserving per-angle scaling supplies explicit physical values. The primary solver has separate scaling parsing, so this finding concerns the config accessor.
- Red test:
  ```python
  from xpcsjax.config import ConfigManager

  cfg = ConfigManager(config_override={"initial_parameters": {
      "parameter_names": ["D0"],
      "values": None,
      "per_angle_scaling": {"contrast": [0.1, 0.2]},
  }})
  params = cfg.get_initial_parameters()
  assert [params["contrast_0"], params["contrast_1"]] == [0.1, 0.2]
  ```
- Confidence: high

### config-B10: A rejected heterodyne update mutates values and leaves the cached array stale

- Location: xpcsjax/config/heterodyne_parameter_space.py:204
- Severity: Minor
- Failure scenario: After reading `get_full_values()`, call `update_values({"D0_ref": 5.0, "unknown": 0.0})`. The call raises, but the dictionary reports `D0_ref=5.0` while the cached physics array still contains `10000.0`.
- Root cause: `update_from_dict()` validates and writes one entry at a time. `HeterodyneParameterManager.update_values()` clears the cache only after the whole update returns successfully.
- Checked: The public stateful model exposes this path through `set_params()`, with separate array and dictionary getters. Existing tied-manager tests exercise expansion from valid values and do not cover a mixed valid/invalid update.
- Red test:
  ```python
  import pytest
  from xpcsjax.config.heterodyne_parameter_manager import HeterodyneParameterManager

  pm = HeterodyneParameterManager()
  before = pm.get_full_values()[0]
  with pytest.raises(ValueError, match="unknown"):
      pm.update_values({"D0_ref": 5.0, "unknown": 0.0})
  assert pm.get_parameter_dict()["D0_ref"] == pm.get_full_values()[0] == before
  ```
- Confidence: high

### config-B11: Heterodyne correlation validation accepts an infinite time coordinate

- Location: xpcsjax/config/heterodyne_physics_validators.py:527
- Severity: Minor
- Failure scenario: A finite `2×2` correlation matrix with `t1=[0, inf]` and `t2=[0, 1]` returns `is_valid=True`.
- Root cause: Finiteness checks apply only to `c2_data`. The time axes are checked solely via `np.diff(axis) > 0`, and a positive infinite difference passes.
- Checked: The standalone validator promises finiteness checks. Its tests cover a matrix NaN, incompatible shapes, and non-monotone finite axes. The production loader uses separate input validation.
- Red test:
  ```python
  import numpy as np
  from xpcsjax.config.heterodyne_physics_validators import validate_correlation_inputs

  result = validate_correlation_inputs(
      np.array([0.0, np.inf]), np.array([0.0, 1.0]), np.ones((2, 2))
  )
  assert not result.is_valid
  ```
- Confidence: high

## Non-blocking notes

- Verification used static caller tracing and read-only Python probes. The test suite was skipped as requested.

## Verdict

BUGS_FOUND  
END_OF_REPORT