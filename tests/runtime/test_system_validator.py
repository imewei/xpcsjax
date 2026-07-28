"""Tests for xpcsjax.runtime.utils.system_validator.

Covers the version-parsing helpers, every individual probe, the
exception-to-ERROR mapping in ``validate()``, the report/JSON emitters, and the
``main`` CLI entry point.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from xpcsjax.runtime.utils import system_validator as sv
from xpcsjax.runtime.utils.system_validator import (
    PUBLIC_API_SYMBOLS,
    REQUIRED_TEMPLATES,
    Severity,
    SystemValidator,
    ValidationResult,
    _parse_version,
    _print_report,
    _result_to_dict,
    _version_at_least,
    run_validation,
)

# --- version helpers (pure) -------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.2.3", (1, 2, 3)),
        ("1.2.3rc1+abc", (1, 2, 3)),
        ("0.8.2", (0, 8, 2)),
        ("2", (2,)),
        ("1.2.dev0", (1, 2)),  # non-numeric chunk halts parsing
        ("1.2-3", (1,)),  # split on '-' keeps only leading "1.2"... -> (1, 2)
    ],
)
def test_parse_version(text: str, expected: tuple[int, ...]) -> None:
    # "1.2-3": re.split on '-' yields "1.2" -> (1, 2)
    if text == "1.2-3":
        assert _parse_version(text) == (1, 2)
    else:
        assert _parse_version(text) == expected


@pytest.mark.parametrize(
    ("actual", "minimum", "ok"),
    [
        ("2.3.0", "2.3", True),
        # Semantically-equal versions of differing length must compare equal:
        # "2.3" is treated as "2.3.0" (zero-padded) so it satisfies the minimum.
        # A raw tuple compare would give (2,3) < (2,3,0) == False (the bug).
        ("2.3", "2.3.0", True),
        ("1.17", "1.17.0", True),
        ("1.9", "2.0", False),
        ("0.8.2", "0.8.2", True),
        ("0.8.1", "0.8.2", False),
        # A pre-release / dev build of the required minimum must NOT satisfy
        # it: _parse_version drops the non-numeric suffix, so the numeric
        # tuples alone are equal (1, 2, 3) == (1, 2, 3) -- _version_at_least
        # must break that tie using _is_final_release, not silently pass.
        ("1.2.3rc1", "1.2.3", False),
        ("0.8.2.dev20250101", "0.8.2", False),
        ("1.2.3", "1.2.3rc1", True),
    ],
)
def test_version_at_least(actual: str, minimum: str, ok: bool) -> None:
    assert _version_at_least(actual, minimum) is ok


# --- result dataclass -------------------------------------------------------


def test_validation_result_defaults() -> None:
    r = ValidationResult(True, Severity.INFO, "ok", "Name")
    assert r.details is None
    assert r.severity is Severity.INFO


def test_severity_values() -> None:
    assert {s.value for s in Severity} == {"info", "warning", "error"}


# --- individual probes (real environment) -----------------------------------


def test_python_version_probe_passes() -> None:
    # The test suite itself requires >= 3.12.
    r = SystemValidator().test_python_version()
    assert r.success is True
    assert r.severity is Severity.INFO


def test_dependency_versions_probe_passes() -> None:
    r = SystemValidator().test_dependency_versions()
    assert r.success is True, r.details
    assert "required dependencies satisfied" in r.message


def test_jax_installation_probe_x64_enabled() -> None:
    # pytest config sets JAX_ENABLE_X64=1.
    r = SystemValidator().test_jax_installation()
    assert r.success is True, r.details
    assert "x64 precision enabled" in r.message


def test_xpcsjax_import_probe_resolves_public_symbols() -> None:
    r = SystemValidator().test_xpcsjax_import()
    assert r.success is True, r.details
    assert str(len(PUBLIC_API_SYMBOLS)) in r.message


def test_required_dependencies_mirror_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    declared = {}
    for dep in tomllib.loads(pyproject.read_text())["project"]["dependencies"]:
        req = Requirement(dep)
        # Only the >= floor is mirrored; upper caps (nlsq<1.0, h5py<4.0) are not probed.
        floors = [s.version for s in req.specifier if s.operator == ">="]
        declared[canonicalize_name(req.name)] = floors[0]

    probed = {canonicalize_name(dist): floor for dist, floor, _ in sv.REQUIRED_DEPENDENCIES}
    assert probed == declared


def test_public_api_symbols_cover_xpcsjax_all() -> None:
    import xpcsjax

    assert set(PUBLIC_API_SYMBOLS) == set(xpcsjax.__all__)


def test_config_templates_probe_finds_all() -> None:
    r = SystemValidator().test_config_templates()
    assert r.success is True, r.details
    assert str(len(REQUIRED_TEMPLATES)) in r.message


def test_cpu_info_probe_reports() -> None:
    r = SystemValidator().test_cpu_info()
    assert r.success is True
    assert "cores" in r.message


def test_xla_config_probe_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=4 --foo=bar")
    r = SystemValidator().test_xla_config()
    assert r.success is True
    assert "parallel CPU paths" in r.message


def test_xla_config_probe_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLA_FLAGS", "--something_else=1")
    r = SystemValidator().test_xla_config()
    assert r.success is False
    assert r.severity is Severity.INFO  # absence is informational, not an error


# --- validate() orchestration ----------------------------------------------


def test_validate_runs_all_probes() -> None:
    results = SystemValidator().validate()
    assert len(results) == 7
    assert all(isinstance(r, ValidationResult) for r in results)


def test_validate_verbose_prints(capsys: pytest.CaptureFixture[str]) -> None:
    SystemValidator(verbose=True).validate()
    out = capsys.readouterr().out
    assert "[PASS]" in out or "[FAIL]" in out


def test_validate_maps_probe_exception_to_error() -> None:
    validator = SystemValidator()

    def boom() -> ValidationResult:
        raise RuntimeError("synthetic probe failure")

    boom.__name__ = "test_boom_probe"
    validator._tests = [boom]  # type: ignore[attr-defined]
    results = validator.validate()
    assert len(results) == 1
    r = results[0]
    assert r.success is False
    assert r.severity is Severity.ERROR
    assert "synthetic probe failure" in r.message
    assert r.name == "Boom Probe"


# --- reporting + entry points ----------------------------------------------


def test_result_to_dict_roundtrip() -> None:
    r = ValidationResult(False, Severity.WARNING, "msg", "Name", details="d")
    d = _result_to_dict(r)
    assert d == {
        "name": "Name",
        "success": False,
        "severity": "warning",
        "message": "msg",
        "details": "d",
    }


def test_print_report_renders_all_tags(capsys: pytest.CaptureFixture[str]) -> None:
    results = [
        ValidationResult(True, Severity.INFO, "ok", "OkTest"),
        ValidationResult(False, Severity.ERROR, "bad", "ErrTest", details="line1\nline2"),
        ValidationResult(False, Severity.WARNING, "meh", "WarnTest"),
        ValidationResult(False, Severity.INFO, "info-fail", "InfoTest"),
    ]
    _print_report(results)
    out = capsys.readouterr().out
    assert "[ OK   ]" in out
    assert "[ FAIL ]" in out
    assert "[ WARN ]" in out
    assert "[ INFO ]" in out
    assert "line1" in out and "line2" in out  # details lines are indented
    assert "1/4 passed" in out


def test_run_validation_human_report(capsys: pytest.CaptureFixture[str]) -> None:
    results = run_validation(verbose=False, as_json=False)
    out = capsys.readouterr().out
    assert "xpcsjax System Validator" in out
    assert len(results) == 7


def test_run_validation_json(capsys: pytest.CaptureFixture[str]) -> None:
    results = run_validation(verbose=True, as_json=True)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert len(parsed) == len(results)
    assert {"name", "success", "severity", "message", "details"} == set(parsed[0])


def test_main_returns_zero_when_no_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sv.sys, "argv", ["xpcsjax-validate"])
    # Ensure the XLA probe (INFO-only on failure) and others don't produce ERROR.
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    assert sv.main() == 0


def test_main_returns_one_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sv.sys, "argv", ["xpcsjax-validate", "--json"])
    error_results = [ValidationResult(False, Severity.ERROR, "boom", "X")]
    monkeypatch.setattr(sv, "run_validation", lambda verbose, as_json: error_results)
    assert sv.main() == 1
