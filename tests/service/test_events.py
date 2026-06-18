"""Tests for the JAX-free FitEvent schema (xpcsjax/service/events.py)."""

import pickle
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


def test_events_module_does_not_import_jax():
    assert not _imports_jax("xpcsjax.service.events")


def test_event_subclasses_carry_run_id_and_seq():
    from xpcsjax.service.events import Finished, Iteration

    it = Iteration(run_id="r1", seq=3, n=10, ssr=1.5, chi2=0.9)
    assert (it.run_id, it.seq, it.n, it.ssr, it.chi2) == ("r1", 3, 10, 1.5, 0.9)

    fin = Finished(run_id="r1", seq=99, result_path="/tmp/out.npz")
    assert fin.result_path == "/tmp/out.npz"


def test_events_are_picklable_round_trip():
    # Spawn-based IPC requires every event to survive pickle.
    from xpcsjax.service.events import Banner, BannerKind, Died

    b = Banner(run_id="r1", seq=1, text="CMA-ES escape", kind=BannerKind.CMAES_ESCAPE)
    assert pickle.loads(pickle.dumps(b)) == b

    d = Died(run_id="r1", seq=42, exit_code=None, signal=9)
    assert pickle.loads(pickle.dumps(d)) == d


def test_terminal_events_set():
    from xpcsjax.service.events import TERMINAL_EVENTS, Died, Failed, Finished

    assert set(TERMINAL_EVENTS) == {Finished, Failed, Died}
