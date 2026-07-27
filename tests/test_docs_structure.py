"""Every top-level xpcsjax submodule needs a Sphinx API page (ADR-0001).

Structural check only — page existence, not content. Symbol-level coverage
and content accuracy are deliberately not automated; see ADR-0001 for why.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
XPCSJAX_ROOT = REPO_ROOT / "xpcsjax"
API_DOCS_DIR = REPO_ROOT / "docs" / "source" / "api"

# Submodules with no docs/source/api/{name}.rst page, by design — see ADR-0001.
EXCLUDED_PACKAGES = {"gui"}


def _top_level_submodules() -> list[str]:
    """Every importable top-level xpcsjax submodule (has __init__.py), sorted."""
    return sorted(
        p.name for p in XPCSJAX_ROOT.iterdir() if p.is_dir() and (p / "__init__.py").is_file()
    )


def test_every_submodule_has_an_api_page():
    """Each top-level xpcsjax submodule needs docs/source/api/{name}.rst (ADR-0001)."""
    missing = [
        name
        for name in _top_level_submodules()
        if name not in EXCLUDED_PACKAGES and not (API_DOCS_DIR / f"{name}.rst").is_file()
    ]
    assert not missing, (
        f"xpcsjax submodule(s) missing docs/source/api/*.rst page: {missing}. "
        f"Add docs/source/api/{{name}}.rst (see docs/adr/0001-automated-structural-"
        f"doc-coverage-check.md), or add to EXCLUDED_PACKAGES in this file if "
        f"intentionally undocumented (e.g. a non-library-surface app like the GUI)."
    )
