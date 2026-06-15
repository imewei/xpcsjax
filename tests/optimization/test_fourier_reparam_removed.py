"""Phase 7: fourier_reparam.py is deleted; its survivor behaviors moved to
PerAngleScalingPlan (Phase 0). Importing it must now ModuleNotFoundError, and no
package module may import it anymore.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Text-source suffixes scanned for forbidden tokens. Pure-Python scanning is used
# instead of an `rg`/`grep` subprocess so the guard runs on every CI runner
# (Windows/macOS lack ripgrep) without an external-tool dependency.
_SCAN_SUFFIXES = {".py", ".pyi", ".yaml", ".yml", ".rst", ".md", ".txt", ".toml", ".cfg", ".json"}


def test_fourier_reparam_module_is_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("xpcsjax.optimization.nlsq.fourier_reparam")


def test_no_package_module_imports_fourier_reparam() -> None:
    pattern = re.compile(r"fourier_reparam|FourierReparameterizer|FourierReparamConfig")
    offenders: list[str] = []
    for path in sorted((_REPO_ROOT / "xpcsjax").rglob("*")):
        if not path.is_file() or path.suffix not in _SCAN_SUFFIXES:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "fourier_reparam still referenced inside the package:\n" + "\n".join(
        offenders
    )
