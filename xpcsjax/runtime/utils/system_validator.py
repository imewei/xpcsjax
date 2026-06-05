"""System validation utilities for xpcsjax installation.

This module provides comprehensive validation of the xpcsjax installation,
including environment detection, dependency verification, JAX configuration
testing, and template / public-API integrity checks.

xpcsjax is NLSQ-only by design — this validator deliberately does *not*
check for NumPyro / BlackJAX / ArviZ or any Bayesian / MCMC dependency.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = [
    "Severity",
    "SystemValidator",
    "ValidationResult",
    "main",
    "run_validation",
]


# ---------------------------------------------------------------------------
# Required runtime dependencies (mirrors pyproject.toml)
# ---------------------------------------------------------------------------

# Each entry: (distribution_name, minimum_version, import_name)
REQUIRED_DEPENDENCIES: tuple[tuple[str, str, str], ...] = (
    ("numpy", "2.3", "numpy"),
    ("scipy", "1.17", "scipy"),
    ("jax", "0.8.2", "jax"),
    ("jaxlib", "0.8.2", "jaxlib"),
    ("jaxopt", "0.8.3", "jaxopt"),
    ("interpax", "0.3.12", "interpax"),
    ("nlsq", "0.6.10", "nlsq"),
    ("evosax", "0.2.0", "evosax"),
    ("h5py", "3.15", "h5py"),
    ("pyyaml", "6.0.3", "yaml"),
    ("psutil", "7.2", "psutil"),
    ("cloudpickle", "3.1", "cloudpickle"),
    ("tqdm", "4.67.1", "tqdm"),
    ("scikit-learn", "1.6", "sklearn"),
)

# Optional viz-fast extras — informational only.
OPTIONAL_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("datashader", "datashader"),
    ("xarray", "xarray"),
    ("colorcet", "colorcet"),
)

# Required config templates under xpcsjax/config/templates/
REQUIRED_TEMPLATES: tuple[str, ...] = (
    "xpcsjax_static_anisotropic.yaml",
    "xpcsjax_static_isotropic.yaml",
    "xpcsjax_laminar_flow.yaml",
    "xpcsjax_two_component.yaml",
)

# Public API symbols (resolved via xpcsjax.__getattr__ lazy loader)
PUBLIC_API_SYMBOLS: tuple[str, ...] = (
    "fit_nlsq",
    "ConfigManager",
    "load_xpcs_data",
    "HomodyneModel",
    "HeterodyneModel",
    "OptimizationResult",
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class Severity(Enum):
    """Severity level for validation results."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Result of a single validation test.

    Attributes
    ----------
    success : bool
        Whether the test passed.
    severity : Severity
        Severity level (INFO / WARNING / ERROR).
    message : str
        One-line description of the result.
    name : str
        Human-readable test name.
    details : str or None
        Optional multi-line details / remediation string.
    """

    success: bool
    severity: Severity
    message: str
    name: str
    details: str | None = field(default=None)


# ---------------------------------------------------------------------------
# Version parsing helpers
# ---------------------------------------------------------------------------


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a PEP 440-ish version string into an int tuple for comparison.

    Strips pre-release / dev / local suffixes (e.g. ``1.2.3rc1+abc`` ->
    ``(1, 2, 3)``). Non-numeric trailing components are dropped to keep ordering
    total.

    Parameters
    ----------
    version : str
        Version string to parse.

    Returns
    -------
    tuple of int
        Numeric release components.
    """
    cleaned = re.split(r"[+\-]", version)[0]
    parts: list[int] = []
    for chunk in cleaned.split("."):
        m = re.match(r"(\d+)", chunk)
        if m is None:
            break
        parts.append(int(m.group(1)))
    return tuple(parts)


def _version_at_least(actual: str, minimum: str) -> bool:
    """Return whether ``actual`` is at least ``minimum``.

    Both versions are normalized via :func:`_parse_version` first.

    Parameters
    ----------
    actual : str
        Installed version string.
    minimum : str
        Required minimum version string.

    Returns
    -------
    bool
        ``True`` if ``actual >= minimum``.
    """
    return _parse_version(actual) >= _parse_version(minimum)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class SystemValidator:
    """Comprehensive system validator for xpcsjax installation.

    Runs a series of self-contained validation tests. Each test returns a
    :class:`ValidationResult`; uncaught exceptions are converted to ERROR
    results so a single broken probe never aborts the run.

    Examples
    --------
    >>> validator = SystemValidator(verbose=True)
    >>> results = validator.validate()
    >>> for r in results:
    ...     print(f"{r.name}: {'PASS' if r.success else 'FAIL'}")
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the validator.

        Parameters
        ----------
        verbose : bool, optional
            If ``True``, print per-test status as validation proceeds.
        """
        self.verbose = verbose
        self._tests: list[Callable[[], ValidationResult]] = [
            self.test_python_version,
            self.test_dependency_versions,
            self.test_jax_installation,
            self.test_xpcsjax_import,
            self.test_config_templates,
            self.test_cpu_info,
            self.test_xla_config,
        ]

    # -- public entry ------------------------------------------------------

    def validate(self) -> list[ValidationResult]:
        """Run all validation tests.

        Each test is isolated: an uncaught exception is converted into an ERROR
        :class:`ValidationResult` so one broken probe never aborts the run.

        Returns
        -------
        list of ValidationResult
            One result per test, in test order.

        Examples
        --------
        >>> results = SystemValidator().validate()
        >>> all(isinstance(r, ValidationResult) for r in results)
        True
        """
        results: list[ValidationResult] = []
        for test in self._tests:
            try:
                result = test()
            except Exception as exc:  # noqa: BLE001 - we want any failure mapped to ERROR
                result = ValidationResult(
                    success=False,
                    severity=Severity.ERROR,
                    message=f"Test raised exception: {exc}",
                    name=test.__name__.replace("test_", "").replace("_", " ").title(),
                )
            results.append(result)
            if self.verbose:
                status = "PASS" if result.success else "FAIL"
                print(f"[{status}] {result.name}: {result.message}")
        return results

    # -- individual probes -------------------------------------------------

    def test_python_version(self) -> ValidationResult:
        """Check that the running Python is >= 3.12.

        Returns
        -------
        ValidationResult
            INFO on success, ERROR if the interpreter is too old.
        """
        v = sys.version_info
        version_str = f"{v.major}.{v.minor}.{v.micro}"
        impl = platform.python_implementation()
        if v >= (3, 12):
            return ValidationResult(
                success=True,
                severity=Severity.INFO,
                message=f"Python {version_str} ({impl}) meets requirement >= 3.12",
                name="Python Version",
            )
        return ValidationResult(
            success=False,
            severity=Severity.ERROR,
            message=f"Python {version_str} is below the required 3.12",
            name="Python Version",
            details="Install Python 3.12 or later (e.g. `uv python install 3.12`).",
        )

    def test_dependency_versions(self) -> ValidationResult:
        """Verify every required runtime dependency meets its minimum version.

        Optional viz-fast extras are reported informationally and never cause a
        failure.

        Returns
        -------
        ValidationResult
            INFO when all requirements are satisfied, ERROR listing any missing
            or outdated distributions otherwise.
        """
        missing: list[str] = []
        outdated: list[str] = []
        present: list[str] = []

        for dist_name, min_version, _import_name in REQUIRED_DEPENDENCIES:
            try:
                actual = importlib_metadata.version(dist_name)
            except importlib_metadata.PackageNotFoundError:
                missing.append(f"{dist_name} (need >= {min_version})")
                continue
            if _version_at_least(actual, min_version):
                present.append(f"{dist_name}=={actual}")
            else:
                outdated.append(f"{dist_name}=={actual} (need >= {min_version})")

        # Optional viz-fast extras — informational
        optional_present: list[str] = []
        for dist_name, _import_name in OPTIONAL_DEPENDENCIES:
            try:
                actual = importlib_metadata.version(dist_name)
            except importlib_metadata.PackageNotFoundError:
                continue
            optional_present.append(f"{dist_name}=={actual}")

        if not missing and not outdated:
            extra = (
                f"\n  Optional viz-fast extras present: {', '.join(optional_present)}"
                if optional_present
                else "\n  No optional viz-fast extras detected (install xpcsjax[viz-fast] to enable)."
            )
            return ValidationResult(
                success=True,
                severity=Severity.INFO,
                message=f"All {len(present)} required dependencies satisfied",
                name="Dependency Versions",
                details="Installed: " + ", ".join(present) + extra,
            )

        problems: list[str] = []
        if missing:
            problems.append("Missing: " + ", ".join(missing))
        if outdated:
            problems.append("Outdated: " + ", ".join(outdated))
        return ValidationResult(
            success=False,
            severity=Severity.ERROR,
            message=f"{len(missing) + len(outdated)} dependency issue(s) detected",
            name="Dependency Versions",
            details="\n  ".join(problems) + "\n  Run `uv sync` to resolve.",
        )

    def test_jax_installation(self) -> ValidationResult:
        """Verify JAX imports, exposes devices, and has x64 precision enabled.

        Returns
        -------
        ValidationResult
            INFO when JAX loads with x64 enabled, ERROR on import failure, a
            device-enumeration error, or x64 being disabled.
        """
        try:
            import jax  # noqa: PLC0415
        except ImportError as exc:
            return ValidationResult(
                success=False,
                severity=Severity.ERROR,
                message=f"JAX import failed: {exc}",
                name="JAX Installation",
                details="Run `uv sync` to install dependencies.",
            )

        try:
            devices = jax.devices()
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(
                success=False,
                severity=Severity.ERROR,
                message=f"jax.devices() raised: {exc}",
                name="JAX Installation",
            )

        x64_enabled = bool(jax.config.read("jax_enable_x64"))
        platform_name = devices[0].platform if devices else "unknown"
        version = getattr(jax, "__version__", "unknown")

        if not x64_enabled:
            return ValidationResult(
                success=False,
                severity=Severity.ERROR,
                message=(
                    f"JAX {version} loaded ({len(devices)} {platform_name} device(s)) "
                    "but x64 precision is NOT enabled"
                ),
                name="JAX Installation",
                details=(
                    "xpcsjax requires float64 (parameters span 6+ orders of magnitude). "
                    "`xpcsjax/__init__.py` sets JAX_ENABLE_X64=1 before importing jax; "
                    "if you bypassed that, set the env var manually before any jax import."
                ),
            )

        return ValidationResult(
            success=True,
            severity=Severity.INFO,
            message=(
                f"JAX {version} loaded with {len(devices)} {platform_name} device(s); "
                "x64 precision enabled"
            ),
            name="JAX Installation",
        )

    def test_xpcsjax_import(self) -> ValidationResult:
        """Import xpcsjax and resolve every public lazy-loaded symbol.

        Returns
        -------
        ValidationResult
            INFO when the import and all public symbols resolve, ERROR on import
            failure or any unresolved symbol.
        """
        try:
            xpcsjax = importlib.import_module("xpcsjax")
        except ImportError as exc:
            return ValidationResult(
                success=False,
                severity=Severity.ERROR,
                message=f"Failed to import xpcsjax: {exc}",
                name="xpcsjax Import",
                details="Run `uv pip install -e .` from the repo root.",
            )

        version = getattr(xpcsjax, "__version__", "unknown")
        missing: list[str] = []
        for symbol in PUBLIC_API_SYMBOLS:
            try:
                getattr(xpcsjax, symbol)
            except (AttributeError, ImportError) as exc:
                missing.append(f"{symbol} ({exc.__class__.__name__}: {exc})")

        if missing:
            return ValidationResult(
                success=False,
                severity=Severity.ERROR,
                message=f"{len(missing)} public API symbol(s) failed to resolve",
                name="xpcsjax Import",
                details="Unresolved: " + "; ".join(missing),
            )

        return ValidationResult(
            success=True,
            severity=Severity.INFO,
            message=(
                f"xpcsjax {version} import succeeded; all "
                f"{len(PUBLIC_API_SYMBOLS)} public symbols accessible"
            ),
            name="xpcsjax Import",
        )

    def test_config_templates(self) -> ValidationResult:
        """Verify every shipped YAML config template is present on disk.

        Returns
        -------
        ValidationResult
            INFO when all templates exist, ERROR if any are missing or the
            package cannot be located.
        """
        try:
            import xpcsjax  # noqa: PLC0415
        except ImportError as exc:
            return ValidationResult(
                success=False,
                severity=Severity.ERROR,
                message=f"Cannot locate templates - xpcsjax import failed: {exc}",
                name="Config Templates",
            )

        pkg_root = Path(xpcsjax.__file__).resolve().parent
        templates_dir = pkg_root / "config" / "templates"
        missing = [name for name in REQUIRED_TEMPLATES if not (templates_dir / name).is_file()]

        if missing:
            return ValidationResult(
                success=False,
                severity=Severity.ERROR,
                message=f"{len(missing)} of {len(REQUIRED_TEMPLATES)} templates missing",
                name="Config Templates",
                details=(f"Templates directory: {templates_dir}\nMissing: " + ", ".join(missing)),
            )

        return ValidationResult(
            success=True,
            severity=Severity.INFO,
            message=f"All {len(REQUIRED_TEMPLATES)} config templates present",
            name="Config Templates",
            details=f"Templates directory: {templates_dir}",
        )

    def test_cpu_info(self) -> ValidationResult:
        """Report CPU core count and system RAM (informational only).

        Returns
        -------
        ValidationResult
            INFO with the core/RAM summary, or WARNING if :mod:`psutil` is
            unavailable.
        """
        try:
            import psutil  # noqa: PLC0415
        except ImportError as exc:
            return ValidationResult(
                success=False,
                severity=Severity.WARNING,
                message=f"psutil unavailable: {exc}",
                name="CPU Info",
            )

        physical = psutil.cpu_count(logical=False) or 0
        logical = psutil.cpu_count(logical=True) or 0
        ram_gb = psutil.virtual_memory().total / (1024**3)
        return ValidationResult(
            success=True,
            severity=Severity.INFO,
            message=(f"{physical} physical / {logical} logical cores, {ram_gb:.1f} GiB RAM"),
            name="CPU Info",
            details=f"Platform: {platform.system()} {platform.release()} ({platform.machine()})",
        )

    def test_xla_config(self) -> ValidationResult:
        """Check that xpcsjax's ``XLA_FLAGS`` configuration was applied.

        Returns
        -------
        ValidationResult
            INFO either way; ``success`` is ``True`` when the host-device-count
            marker is present in ``XLA_FLAGS`` and ``False`` otherwise.
        """
        xla_flags = os.environ.get("XLA_FLAGS", "")
        marker = "xla_force_host_platform_device_count"
        if marker in xla_flags:
            return ValidationResult(
                success=True,
                severity=Severity.INFO,
                message="XLA_FLAGS set by xpcsjax (parallel CPU paths enabled)",
                name="XLA Configuration",
                details=f"XLA_FLAGS={xla_flags}",
            )
        return ValidationResult(
            success=False,
            severity=Severity.INFO,
            message=f"XLA_FLAGS does not contain `{marker}`",
            name="XLA Configuration",
            details=(
                "xpcsjax/__init__.py sets this before importing jax. "
                "If you imported jax before xpcsjax, the flag won't take effect."
            ),
        )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _result_to_dict(r: ValidationResult) -> dict[str, object]:
    """Serialize a :class:`ValidationResult` into a JSON-friendly dict.

    Parameters
    ----------
    r : ValidationResult
        Result to serialize.

    Returns
    -------
    dict
        Mapping with ``name``, ``success``, ``severity``, ``message``, and
        ``details`` keys.
    """
    return {
        "name": r.name,
        "success": r.success,
        "severity": r.severity.value,
        "message": r.message,
        "details": r.details,
    }


def _print_report(results: list[ValidationResult]) -> None:
    """Print a human-readable report (boxed header, per-test lines, summary).

    Parameters
    ----------
    results : list of ValidationResult
        Results to render to stdout.
    """
    title = "xpcsjax System Validator"
    bar = "=" * max(60, len(title) + 4)
    print(bar)
    print(f"  {title}")
    print(bar)

    for r in results:
        if r.success:
            tag = "[ OK   ]"
        elif r.severity is Severity.ERROR:
            tag = "[ FAIL ]"
        elif r.severity is Severity.WARNING:
            tag = "[ WARN ]"
        else:
            tag = "[ INFO ]"
        print(f"{tag} {r.name}: {r.message}")
        if r.details:
            for line in r.details.splitlines():
                print(f"         {line}")

    total = len(results)
    passed = sum(1 for r in results if r.success)
    errors = sum(1 for r in results if not r.success and r.severity is Severity.ERROR)
    warnings = sum(1 for r in results if not r.success and r.severity is Severity.WARNING)

    print(bar)
    print(f"  Summary: {passed}/{total} passed, {errors} error(s), {warnings} warning(s)")
    print(bar)


def run_validation(verbose: bool = False, as_json: bool = False) -> list[ValidationResult]:
    """Run all validation tests and emit a report.

    Parameters
    ----------
    verbose : bool, optional
        If ``True``, print per-test status while validating (suppressed when
        ``as_json`` is set).
    as_json : bool, optional
        If ``True``, emit a JSON array of results to stdout instead of the
        human-readable report.

    Returns
    -------
    list of ValidationResult
        The results produced by :meth:`SystemValidator.validate`.
    """
    validator = SystemValidator(verbose=verbose and not as_json)
    results = validator.validate()

    if as_json:
        import json

        print(json.dumps([_result_to_dict(r) for r in results], indent=2))
    else:
        _print_report(results)

    return results


def main() -> int:
    """Run the ``xpcsjax-validate`` CLI command.

    Returns
    -------
    int
        Process exit code: ``1`` if any ERROR-severity test failed, else ``0``.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="xpcsjax-validate",
        description="Validate the xpcsjax installation and configuration.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-test status while validating.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as a JSON array on stdout (suppresses the human report).",
    )
    args = parser.parse_args()

    results = run_validation(verbose=args.verbose, as_json=args.json)
    has_errors = any(not r.success and r.severity is Severity.ERROR for r in results)
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
