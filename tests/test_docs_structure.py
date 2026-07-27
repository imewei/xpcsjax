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


def _missing_api_pages(
    submodules: list[str], excluded: set[str], existing_page_names: set[str]
) -> list[str]:
    """Submodules (minus `excluded`) absent from `existing_page_names` — pure, no I/O."""
    return [name for name in submodules if name not in excluded and name not in existing_page_names]


def test_every_submodule_has_an_api_page():
    """Each top-level xpcsjax submodule needs docs/source/api/{name}.rst (ADR-0001)."""
    existing_pages = {p.stem for p in API_DOCS_DIR.glob("*.rst")}
    missing = _missing_api_pages(_top_level_submodules(), EXCLUDED_PACKAGES, existing_pages)
    assert not missing, (
        f"xpcsjax submodule(s) missing docs/source/api/*.rst page: {missing}. "
        f"Add docs/source/api/{{name}}.rst (see docs/adr/0001-automated-structural-"
        f"doc-coverage-check.md), or add to EXCLUDED_PACKAGES in this file if "
        f"intentionally undocumented (e.g. a non-library-surface app like the GUI)."
    )


def test_missing_api_pages_detects_a_gap():
    """Regression proof the check actually fires — synthetic inputs, no filesystem.

    Guards against the check silently going inert (e.g. a future edit to the
    comprehension logic) without relying on a one-off manual verification.
    """
    missing = _missing_api_pages(
        submodules=["device", "gui", "io", "utils"],
        excluded={"gui"},
        existing_page_names={"io", "utils"},
    )
    assert missing == ["device"]


def test_excluded_packages_are_real_submodules():
    """A stale EXCLUDED_PACKAGES entry (e.g. a renamed/removed package) should be
    visible, not silently no-op."""
    stale = EXCLUDED_PACKAGES - set(_top_level_submodules())
    assert not stale, f"EXCLUDED_PACKAGES contains non-existent submodule(s): {stale}"
