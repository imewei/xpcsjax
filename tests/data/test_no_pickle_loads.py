"""Regression guard: no unsafe ``np.load`` calls inside ``xpcsjax/``.

The NPZ cache loader treats its inputs as untrusted (cache paths are
config-controlled). This test AST-scans every package source file and asserts
that any ``np.load`` / ``numpy.load`` / bare ``load`` call has its
``allow_pickle`` argument either omitted, set to a literal ``False``, or set
to a literal ``None``. A literal ``True`` or a non-literal value fails the
test — the latter blocks smuggling via a variable.

If a call really must permit object deserialization (trusted, version-pinned
input only), add its path to ``ALLOWED_FILES`` with a justification.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "xpcsjax"

ALLOWED_FILES: frozenset[str] = frozenset()


def _iter_source_files() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _is_np_load(node: ast.expr) -> bool:
    """Return True if ``node`` is ``np.load``, ``numpy.load``, or bare ``load``."""
    if isinstance(node, ast.Attribute) and node.attr == "load":
        if isinstance(node.value, ast.Name) and node.value.id in {"np", "numpy"}:
            return True
    if isinstance(node, ast.Name) and node.id == "load":
        return True
    return False


def _find_violations(path: Path) -> list[tuple[int, str]]:
    """Return ``(line, reason)`` for offending ``np.load`` calls in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_np_load(node.func):
            continue
        for kw in node.keywords:
            if kw.arg != "allow_pickle":
                continue
            value = kw.value
            if isinstance(value, ast.Constant):
                if value.value is True:
                    violations.append((node.lineno, "allow_pickle=True literal"))
            else:
                violations.append((node.lineno, f"allow_pickle=<non-literal: {ast.dump(value)}>"))
    return violations


def test_no_unsafe_np_load_calls():
    """No file in ``xpcsjax/`` may set ``allow_pickle`` to True (or a variable)."""
    failures: dict[str, list[tuple[int, str]]] = {}
    for source in _iter_source_files():
        rel = source.relative_to(PACKAGE_ROOT).as_posix()
        if rel in ALLOWED_FILES:
            continue
        violations = _find_violations(source)
        if violations:
            failures[rel] = violations

    if failures:
        lines = ["Unsafe np.load(allow_pickle=...) calls detected:"]
        for rel, viols in failures.items():
            for line_no, reason in viols:
                lines.append(f"  {rel}:{line_no}  {reason}")
        lines.append(
            "If a call is intentionally safe (trusted, version-pinned input), "
            "add its path to ALLOWED_FILES in this test with a justification."
        )
        pytest.fail("\n".join(lines))


def test_guard_can_detect_a_violation(tmp_path: Path):
    """Sanity: the AST walker actually flags ``allow_pickle=True``."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import numpy as np\ndef f(p):\n    return np.load(p, allow_pickle=True)\n",
        encoding="utf-8",
    )
    violations = _find_violations(sample)
    assert violations == [(3, "allow_pickle=True literal")]


def test_guard_passes_on_safe_call(tmp_path: Path):
    """Sanity: ``allow_pickle=False`` (and omission) do not trip the guard."""
    sample = tmp_path / "safe.py"
    sample.write_text(
        "import numpy as np\n"
        "def f(p):\n"
        "    a = np.load(p, allow_pickle=False)\n"
        "    b = np.load(p)\n"
        "    return a, b\n",
        encoding="utf-8",
    )
    assert _find_violations(sample) == []


def test_guard_flags_nonliteral_allow_pickle(tmp_path: Path):
    """Sanity: variable-smuggled allow_pickle is flagged."""
    sample = tmp_path / "sneaky.py"
    sample.write_text(
        "import numpy as np\ndef f(p, flag):\n    return np.load(p, allow_pickle=flag)\n",
        encoding="utf-8",
    )
    violations = _find_violations(sample)
    assert len(violations) == 1
    assert violations[0][0] == 3
    assert "non-literal" in violations[0][1]
