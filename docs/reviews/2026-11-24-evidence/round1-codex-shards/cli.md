# Codex review — shard cli
## Files read

xpcsjax/__init__.py — 175 — full  
xpcsjax/cli/__init__.py — 78 — full  
xpcsjax/cli/args_parser.py — 426 — full  
xpcsjax/cli/commands.py — 219 — full  
xpcsjax/cli/config_generator.py — 218 — full  
xpcsjax/cli/config_handling.py — 316 — full  
xpcsjax/cli/config_template.py — 372 — full  
xpcsjax/cli/data_pipeline.py — 49 — full  
xpcsjax/cli/main.py — 202 — full  
xpcsjax/cli/optimization_runner.py — 324 — full  
xpcsjax/cli/plot_backend.py — 82 — full  
xpcsjax/cli/plot_dispatch.py — 181 — full  
xpcsjax/cli/plot_families/__init__.py — 1 — full  
xpcsjax/cli/plot_families/experimental.py — 73 — full  
xpcsjax/cli/plot_families/postfit.py — 180 — full  
xpcsjax/cli/plot_families/simulated.py — 279 — full  
xpcsjax/cli/result_saving.py — 20 — full  
xpcsjax/cli/xla_config.py — 192 — full  
xpcsjax/device/__init__.py — 42 — full  
xpcsjax/device/cpu.py — 576 — full  
xpcsjax/runtime/__init__.py — 34 — full  
xpcsjax/runtime/shell/__init__.py — 49 — full  
xpcsjax/runtime/shell/activation/__init__.py — 1 — full  
xpcsjax/runtime/shell/activation/xla_config.bash — 126 — full  
xpcsjax/runtime/shell/activation/xla_config.fish — 119 — full  
xpcsjax/runtime/shell/completion.sh — 248 — full  
xpcsjax/runtime/shell/completion_spec.py — 99 — full  
xpcsjax/runtime/shell/generate_completion.py — 266 — full  
xpcsjax/runtime/utils/__init__.py — 17 — full  
xpcsjax/runtime/utils/system_validator.py — 669 — full  

## Bugs

### cli-B1: Generated configs can direct an analysis to the wrong dataset

- Location: xpcsjax/cli/config_template.py:152
- Severity: Critical
- Failure scenario: Generate a static-anisotropic config with `--data /data/experiment.h5`. A subsequent CLI fit uses the template’s `data/sample.hdf`. If that file exists, the fit can succeed on the wrong dataset. The interactive generator has the complementary failure at line 357. It emits only `file_path`, so the loader rejects the config for missing `data_folder_path`.
- Root cause: The template generator substitutes only `file_path` and leaves the template’s sample directory and filename in place. `ConfigManager._normalize_experimental_data()` then rebuilds `file_path` from those legacy fields. `XPCSDataLoader._validate_configuration()` requires the legacy pair and constructs the input path from it.
- Checked: `config_generator.main`, `commands.dispatch_command`, `service.data.load_dataset`, and the loader’s normalization and validation. `test_overrides_produce_parseable_yaml` checks the raw YAML `file_path` before either consumer reads it.
- Red test:
  ```python
  def test_generated_config_loads_requested_dataset(tmp_path):
      from pathlib import Path
      from xpcsjax.cli.config_template import generate_config
      from xpcsjax.config import ConfigManager

      requested = tmp_path / "experiment.h5"
      path = generate_config("static_anisotropic", tmp_path / "cfg.yaml",
                             data_path=str(requested))
      data = ConfigManager(str(path)).config["experimental_data"]
      assert Path(data["data_folder_path"]) / data["data_file_name"] == requested
  ```
- Confidence: high

### cli-B2: A JSON-only rerun can write current C₂ arrays into a previous fit’s NPZ

- Location: xpcsjax/cli/commands.py:188
- Severity: Critical
- Failure scenario: Run fit A with `--output-format npz --no-plot`. Run fit B in the same directory with `--output-format json` and default plotting. The existing `nlsq_result.npz` can end up with A’s parameters and fit statistics alongside B’s C₂ surfaces and residuals.
- Root cause: `_dispatch_fit` runs the NPZ merge whenever plotting is enabled and an output directory exists, even when `save_results` skipped writing a primary NPZ. The merge accepts a sidecar newer than the target and fills missing keys, so B’s new sidecar passes the freshness guard against A’s old file.
- Checked: Parser output-format choices, `service.persist.save_results`, `merge_fitted_c2`, and the service tests for missing inputs and older/newer sidecars. The newer-sidecar test accepts the timestamp ordering in this scenario. A read-only probe confirmed that the JSON-only dispatch calls the merge.
- Red test:
  ```python
  def test_json_only_fit_preserves_prior_npz(tmp_path, monkeypatch):
      from types import SimpleNamespace
      import numpy as np
      from xpcsjax.cli import commands as c

      old = tmp_path / "nlsq_result.npz"
      np.savez(old, parameters=[1.0])
      before = old.read_bytes()

      def plot(*args, **kwargs):
          sidecar = tmp_path / "plots/simulated_data/c2_fitted_data.npz"
          sidecar.parent.mkdir(parents=True)
          np.savez(sidecar, c2_exp=[[[2.0]]])
          return 0

      monkeypatch.setattr(c, "load_and_validate_data", lambda *a: {})
      monkeypatch.setattr(c, "resolve_phi_angles", lambda *a: None)
      monkeypatch.setattr(c, "run_nlsq", lambda *a: SimpleNamespace(success=True))
      monkeypatch.setattr(c, "save_results", lambda *a, **k: None)
      monkeypatch.setattr(c, "dispatch_plots", plot)
      args = SimpleNamespace(output=tmp_path, output_format="json", plot=True)
      assert c._dispatch_fit(args, None) == 0
      assert old.read_bytes() == before
  ```
- Confidence: high

### cli-B3: `--threads 2` aborts JAX backend initialization

- Location: xpcsjax/cli/main.py:71
- Severity: Required
- Failure scenario: Run the CLI with documented `--threads 2` on the installed, supported `jaxlib` 0.12.1. The process terminates at backend initialization with `Unknown flag in XLA_FLAGS: --intra_op_parallelism_threads=2`. The same startup without `--threads` enumerates CPU devices successfully.
- Root cause: `_bootstrap_xla_env` emits `--intra_op_parallelism_threads` as an XLA flag. The installed XLA rejects that flag during native initialization. The library helper and configuration preview emit the same spelling at `xpcsjax/cli/xla_config.py:62` and `:174`.
- Checked: `main` runs the bootstrap before importing the dispatch pipeline, argparse accepts the positive integer, and `validate_args` leaves it untouched. The parser and completion tests exercise flag recognition without initializing JAX under the resulting environment.
- Red test: `python -B -c 'import resource; resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); from xpcsjax.cli.main import _bootstrap_xla_env; _bootstrap_xla_env(["--threads", "2"]); import jax; assert jax.devices("cpu")'`
- Confidence: high

### cli-B4: Standalone heterodyne simulation ignores the grouped wavevector

- Location: xpcsjax/cli/plot_families/simulated.py:114
- Severity: Required
- Failure scenario: Simulate a two-component config with `temporal: {dt: 0.1, time_length: 3}` and top-level `scattering: {wavevector_q: 0.0237}`. The CLI passes `q=0.01` to the kernel, changing the diffusion decay and velocity-dependent fringe phase.
- Root cause: The CLI reads grouped timing fields but reads `q` exclusively from `analyzer_parameters.scattering`. `HeterodyneModel.from_config()` supports the top-level scattering block, and `ConfigManager._normalize_schema()` leaves it there.
- Checked: `plot_dispatch.dispatch_plots` forwards the manager, `get_model()` returns a stateless adapter, and its `compute_g1()` uses the supplied `q`. `test_simulated_grid_uses_elapsed_time` uses nested scattering. A read-only probe captured `q=0.01` for the grouped input.
- Red test:
  ```python
  def test_simulation_uses_grouped_wavevector(tmp_path, monkeypatch):
      import numpy as np
      import xpcsjax.viz as viz
      from xpcsjax.cli.plot_families import simulated as s
      from xpcsjax.config import ConfigManager

      cm = ConfigManager(config_override={
          "analysis_mode": "two_component",
          "temporal": {"dt": 0.1, "time_length": 3},
          "scattering": {"wavevector_q": 0.0237},
      })
      seen = []
      def evaluate(*args, **kwargs):
          seen.append(kwargs["q"])
          return np.ones((3, 3))
      monkeypatch.setattr(s, "_evaluate_model_c2", evaluate)
      monkeypatch.setattr(viz, "plot_simulated_data", lambda *a, **k: None)
      s._plot_simulated_from_config(cm, 0.3, 1.0, "0", tmp_path, None)
      assert seen == [0.0237]
  ```
- Confidence: high

### cli-B5: Standalone simulation drops fixed parameters before assembling the model vector

- Location: xpcsjax/cli/plot_families/simulated.py:148
- Severity: Required
- Failure scenario: Simulate a static-isotropic config with names `[D0, alpha, D_offset]`, values `[1000.0, 1.0, 0.0]`, and `fixed_parameters: {D_offset: 0.0}`. Vector assembly raises `KeyError: 'D_offset'`, the helper produces zero plots, and the standalone command returns 0.
- Root cause: `get_initial_parameters()` drops fixed entries from explicit values, while `get_active_parameters()` can still return those names. The CLI indexes the free initial-value mapping directly. The model needs a full physical vector in canonical order.
- Checked: `ParameterManager.get_active_parameters`, `ConfigManager._drop_fixed_parameters`, `CombinedModel.compute_g2`, and the fit path’s `resolve_optimized_physical_parameters`. The static-isotropic template documents fixing `D_offset`. The simulated grid and filename tests do not exercise this case.
- Red test:
  ```python
  def test_simulation_includes_fixed_physics(tmp_path, monkeypatch):
      import numpy as np
      import xpcsjax.viz as viz
      from xpcsjax.cli.plot_families import simulated as s
      from xpcsjax.config import ConfigManager

      cm = ConfigManager(config_override={
          "analysis_mode": "static_isotropic",
          "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 3},
          "initial_parameters": {
              "parameter_names": ["D0", "alpha", "D_offset"],
              "values": [1000.0, 1.0, 0.0],
              "fixed_parameters": {"D_offset": 0.0},
          },
      })
      seen = []
      def evaluate(model, params, *args, **kwargs):
          seen.append(params.tolist())
          return np.ones((3, 3))
      monkeypatch.setattr(s, "_evaluate_model_c2", evaluate)
      monkeypatch.setattr(viz, "plot_simulated_data", lambda *a, **k: None)
      s._plot_simulated_from_config(cm, 0.3, 1.0, "0", tmp_path, None)
      assert seen == [[1000.0, 1.0, 0.0]]
  ```
- Confidence: high

### cli-B6: Config validation reports empty and scalar YAML documents as valid

- Location: xpcsjax/cli/config_template.py:238
- Severity: Required
- Failure scenario: Run `xpcsjax-config --validate --output empty.yaml` on a zero-byte file. It prints `Result: VALID` and exits 0. A document containing only `42` also passes.
- Root cause: `validate_config_file` discards the parsed YAML value and treats successful construction as proof of validity. `ConfigManager.load_config()` deliberately substitutes defaults for a non-mapping document and returns normally. The same discarded-value preflight appears in `xpcsjax/cli/config_handling.py:117`.
- Checked: `config_generator.main` exits according to `report.ok`. `ConfigManager.load_config` explicitly handles the non-mapping fallback. `test_load_failure_names_the_file` covers invalid YAML syntax, which fails earlier. A read-only probe using `/dev/null` returned `ok=True`.
- Red test:
  ```python
  def test_config_validation_rejects_empty_yaml(tmp_path):
      from xpcsjax.cli.config_template import validate_config_file

      path = tmp_path / "empty.yaml"
      path.write_text("")
      assert not validate_config_file(path).ok
  ```
- Confidence: high

### cli-B7: A plot-directory error changes a converged fit into a failed command

- Location: xpcsjax/cli/commands.py:189
- Severity: Required
- Failure scenario: A valid output directory already contains a regular file named `plots`. A fit converges and saves its result files. Plot setup raises `FileExistsError`, and the CLI ultimately returns 1.
- Root cause: `dispatch_plots` calls `resolve_plots_dir`, which creates the directory. `_dispatch_fit` catches the first error as a best-effort plotting failure, then repeats the same directory creation outside the exception handler before the merge.
- Checked: `plot_backend.resolve_plots_dir`, the ordering of save and plot in `_dispatch_fit`, and `main`’s exception-to-exit-code mapping. `test_dispatch_fit_is_non_fatal_on_malformed_phi_angles` disables plotting and cannot reach the second call.
- Red test:
  ```python
  def test_plot_directory_conflict_preserves_fit_exit_code(tmp_path, monkeypatch):
      from types import SimpleNamespace
      from xpcsjax.cli import commands as c

      (tmp_path / "plots").touch()
      monkeypatch.setattr(c, "load_and_validate_data", lambda *a: {})
      monkeypatch.setattr(c, "resolve_phi_angles", lambda *a: None)
      monkeypatch.setattr(c, "run_nlsq", lambda *a: SimpleNamespace(success=True))
      monkeypatch.setattr(c, "save_results", lambda *a, **k: None)
      args = SimpleNamespace(output=tmp_path, output_format="both", plot=True)
      assert c._dispatch_fit(args, None) == 0
  ```
- Confidence: high

### cli-B8: Config generation emits broken YAML for a data path containing a newline

- Location: xpcsjax/cli/config_template.py:149
- Severity: Minor
- Failure scenario: On POSIX, pass a filename such as `run\n2026.h5` containing a literal newline to `--data`. The generator reports success and writes an unterminated YAML scalar.
- Root cause: PyYAML emits newline-containing strings as multiline quoted scalars. `_scalar()` keeps only the first physical line to remove the optional document-end marker. Increasing the wrap width preserves explicit newlines, so the slice removes part of the quoted value.
- Checked: The config parser accepts the path as a plain string, all shipped templates have the matching placeholder, and `test_overrides_produce_parseable_yaml` uses a single-line path. A read-only PyYAML probe reproduced the truncation and parser error.
- Red test:
  ```python
  def test_generated_config_round_trips_multiline_path(tmp_path):
      import yaml
      from xpcsjax.cli.config_template import generate_config

      path = "/data/run\n2026.h5"
      out = generate_config("static_isotropic", tmp_path / "cfg.yaml", data_path=path)
      assert yaml.safe_load(out.read_text())["experimental_data"]["file_path"] == path
  ```
- Confidence: high

### cli-B9: A previous shell activation overrides the newly activated environment’s XLA mode

- Location: xpcsjax/runtime/shell/activation/xla_config.bash:73
- Severity: Minor
- Failure scenario: Activate an environment configured for `auto`, then activate one configured for `nlsq` in the same shell. On a host with four or more available cores, the second environment still starts XLA with four CPU devices.
- Root cause: Activation preserves the existing device-count token whenever it is present, including when `XPCSJAX_PRESERVE_XLA_FLAGS` is unset. The second activation loads its own mode file but cannot replace the value exported by the first. The fish script has the same condition at line 68.
- Checked: The post-install activation hook sources the script for the active environment. `xpcsjax/__init__.py` preserves an existing device-count flag. `test_xla_config_bash_no_arg_uses_saved_mode` starts each source operation with an isolated environment. A read-only Bash probe reproduced the stale value.
- Red test:
  ```python
  def test_activation_switches_to_new_environment_mode(tmp_path):
      import os
      import subprocess
      from xpcsjax.runtime.shell import XLA_CONFIG_BASH

      for name, mode in [("first", "4"), ("second", "nlsq")]:
          folder = tmp_path / name / "etc/xpcsjax"
          folder.mkdir(parents=True)
          (folder / "xla_mode").write_text(mode)
      env = {**os.environ, "HOME": str(tmp_path)}
      env.pop("XLA_FLAGS", None)
      env.pop("XPCSJAX_PRESERVE_XLA_FLAGS", None)
      out = subprocess.check_output(
          ["bash", "-c",
           'VIRTUAL_ENV="$2/first"; source "$1"; '
           'VIRTUAL_ENV="$2/second"; source "$1"; printenv XLA_FLAGS',
           "bash", str(XLA_CONFIG_BASH), str(tmp_path)],
          env=env, text=True,
      )
      assert "--xla_force_host_platform_device_count=1" in out.split()
  ```
- Confidence: high

### cli-B10: The public `xpcsjax.cli.main` export changes type with import order

- Location: xpcsjax/cli/__init__.py:31
- Severity: Minor
- Failure scenario: An application imports the `xpcsjax.cli.main` submodule before another component executes `from xpcsjax.cli import main`. The second component receives a module and gets `TypeError` when it calls the expected CLI function.
- Root cause: The lazy callable export shares its name with a child module. Importing the child binds it to the parent’s `main` attribute. Since the attribute now exists, the parent’s `__getattr__` cannot resolve it to the function.
- Checked: The type-checking import and `__all__` advertise the callable, and `pyproject.toml` uses the colliding submodule for the console entry points. The root lazy-import tests do not constrain this import order. A fresh-process probe reproduced the module binding.
- Red test: `python -B -c 'import xpcsjax.cli.main; from xpcsjax.cli import main; assert callable(main), main'`
- Confidence: high

## Non-blocking notes

- Reviewed tracked source at `bd0a549`. Verification used source tracing and targeted read-only probes. The test suite and Fish activation behavior were untested.

## Verdict

BUGS_FOUND  
END_OF_REPORT