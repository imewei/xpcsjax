"""Phase 7: fourier_reparam.py is deleted; its survivor behaviors moved to
PerAngleScalingPlan (Phase 0). Importing it must now ModuleNotFoundError, and no
package module may import it anymore.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_fourier_reparam_module_is_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("xpcsjax.optimization.nlsq.fourier_reparam")


def test_no_package_module_imports_fourier_reparam() -> None:
    proc = subprocess.run(
        [
            "rg",
            "-n",
            "fourier_reparam|FourierReparameterizer|FourierReparamConfig",
            "xpcsjax/",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1 and proc.stdout.strip() == "", (
        "fourier_reparam still referenced inside the package:\n" + proc.stdout
    )
