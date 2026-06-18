"""Regression guard: importing xpcsjax.config must not load JAX (F1)."""

import subprocess
import sys
import textwrap

_IMPORT_CLEAN = 0  # module imported, "jax" NOT in sys.modules
_IMPORT_LOADS_JAX = 1  # module imported, but "jax" IS in sys.modules (the leak)
_IMPORT_ERROR = 2  # the module itself failed to import


def _probe_import(module: str) -> int:
    """Import ``module`` in a fresh interpreter; return 0/1/2 (see contract above)."""
    code = textwrap.dedent(
        f"""
        import importlib
        import sys

        try:
            importlib.import_module({module!r})
        except BaseException:
            sys.exit(2)
        sys.exit(1 if "jax" in sys.modules else 0)
        """
    )
    completed = subprocess.run([sys.executable, "-c", code], check=False)
    return completed.returncode


def _imports_jax(module: str) -> bool:
    """Return True iff importing ``module`` cleanly loads jax.

    Raises AssertionError if the module fails to import, so a missing module or
    unrelated import failure is reported as a real test failure rather than
    being silently conflated with "jax was loaded" (both used to return rc=1).
    """
    rc = _probe_import(module)
    assert rc != _IMPORT_ERROR, f"{module!r} failed to import in a fresh interpreter"
    assert rc in (_IMPORT_CLEAN, _IMPORT_LOADS_JAX), (
        f"unexpected probe exit code {rc} for {module!r}"
    )
    return rc == _IMPORT_LOADS_JAX


def test_importing_config_does_not_load_jax():
    # The GUI process imports config-validation directly; it must stay JAX-free.
    assert not _imports_jax("xpcsjax.config")


def test_importing_parameter_manager_does_not_load_jax():
    assert not _imports_jax("xpcsjax.config.parameter_manager")


def test_importing_registry_and_types_does_not_load_jax():
    # Plan I's in-process config validation imports exactly these two directly;
    # importing a submodule runs config/__init__, so these pin the whole package.
    assert not _imports_jax("xpcsjax.config.parameter_registry")
    assert not _imports_jax("xpcsjax.config.types")


def test_parameter_validation_still_works_after_lazy_import():
    # Behavior preservation: the lazily-imported validators must still run.
    import numpy as np

    from xpcsjax.config.parameter_manager import ParameterManager

    pm = ParameterManager()
    names = pm.get_all_parameter_names()
    params = np.array([pm.get_parameter_bounds([n])[0]["min"] for n in names])
    result = pm.validate_parameters(params, names)
    assert result.valid  # at the lower bound, all parameters are in range

    phys = pm.validate_physical_constraints({names[0]: float(params[0])})
    assert hasattr(phys, "valid")
