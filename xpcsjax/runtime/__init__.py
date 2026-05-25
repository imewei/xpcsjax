"""Runtime utilities for the xpcsjax package.

Provides:
    * :mod:`xpcsjax.runtime.utils` — system validation (CPU, RAM, JAX, deps)
    * :mod:`xpcsjax.runtime.shell` — bash/zsh/fish completion + XLA activation

Example::

    from xpcsjax.runtime import run_validation
    results = run_validation(verbose=True)
    ok = all(r.success for r in results)
"""

from __future__ import annotations

from xpcsjax.runtime.shell import (
    get_completion_script,
    get_xla_config_script,
)
from xpcsjax.runtime.utils import (
    Severity,
    SystemValidator,
    ValidationResult,
    run_validation,
)

__all__ = [
    "SystemValidator",
    "ValidationResult",
    "Severity",
    "run_validation",
    "get_completion_script",
    "get_xla_config_script",
]
