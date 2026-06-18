"""Persist service: import-path equivalence + JAX-free guard."""

import subprocess
import sys
import textwrap


def _probe_import(module: str) -> int:
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
    return subprocess.run([sys.executable, "-c", code], check=False).returncode


def test_persist_service_is_jax_free():
    # The GUI process may import the persist service in-process; keep it JAX-free.
    assert _probe_import("xpcsjax.service.persist") == 0


def test_cli_shim_reexports_same_callable():
    from xpcsjax.cli.result_saving import save_results as cli_save
    from xpcsjax.service.persist import save_results as svc_save

    assert cli_save is svc_save


def test_save_results_rejects_bad_format(tmp_path):
    from unittest.mock import MagicMock

    import pytest

    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.service.persist import save_results

    with pytest.raises(ValueError, match="Unknown output_format"):
        save_results(MagicMock(spec=OptimizationResult), tmp_path, "xml", None, None)
