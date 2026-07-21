"""The GUI entry must be freeze-safe and resolvable as a console script."""

import importlib.metadata
import inspect
import re


def _extract_collect_all_names(spec_src: str) -> set[str]:
    """Lowercase package names from the spec's ``collect_all()`` for-loop tuple.

    Strips ``#``-comments line-by-line first so comment prose — which may
    contain quoted words or a ``"):"``-shaped sequence — can't be mis-parsed
    as tuple content (a real bug PR #10's review caught: an added comment's
    punctuation truncated the match and leaked a comment-only word into the
    result).
    """
    clean = "\n".join(line.split("#", 1)[0] for line in spec_src.splitlines())
    match = re.search(r"for pkg in \((.*?)\):", clean, re.S)
    assert match is not None, "collect_all() for-loop tuple not found in spec"
    return {p.lower() for p in re.findall(r'"([A-Za-z0-9_]+)"', match.group(1))}


def test_main_calls_freeze_support_before_qapplication():
    # Weak-but-useful smoke: confirm the freeze_support() CALL (not a bare mention
    # in a comment) precedes constructing the QApplication. A frozen spawn worker
    # that skips it re-runs the GUI. (True frozen-spawn behavior is only provable
    # by an actual packaged build — a manual/CI step, see packaging/README.md.)
    from xpcsjax.gui import app

    src = inspect.getsource(app.main)
    assert "multiprocessing.freeze_support()" in src
    assert src.index("multiprocessing.freeze_support()") < src.index("QApplication")


def test_console_script_registered():
    scripts = importlib.metadata.entry_points(group="console_scripts")
    assert any(ep.name == "xpcsjax-gui" for ep in scripts)


def test_pyinstaller_spec_covers_runtime_deps():
    # Drift guard (spec §10/§12): every declared runtime dependency must be EITHER
    # explicitly bundled in the .spec collect_all() list OR classified as covered
    # (pure-python / collected transitively). A NEW pyproject dep that is neither
    # fails here — forcing a conscious "bundle it or allowlist it" decision instead
    # of a silent runtime-only break in the frozen app.
    import pathlib
    import tomllib

    root = pathlib.Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    spec_src = (root / "packaging" / "xpcsjax-gui.spec").read_text(encoding="utf-8")

    # dist-name -> import-name for the few that differ
    ALIAS = {"scikit-learn": "sklearn", "pyyaml": "yaml", "pillow": "pil"}
    # covered without an explicit collect_all entry (pure-python, or pulled in
    # transitively by an already-listed compiled package like scipy/jax)
    COVERED = {"numpy", "yaml", "psutil", "cloudpickle", "tqdm"}
    # test/build-only deps that live in a runtime extra but must NEVER be frozen
    # into the binary: `pytest-qt` is in the `gui` extra (Plan C co-locates it with
    # PySide6 so a headless `.[dev]` install doesn't crash pytest collection). It is
    # excluded from the runtime-bundle drift guard by design (it ships nothing).
    TEST_ONLY = {"pytest-qt"}

    def import_name(dep: str) -> str:
        name = re.split(r"[<>=!~ \[]", dep, maxsplit=1)[0].strip().lower()
        return ALIAS.get(name, name)

    listed = _extract_collect_all_names(spec_src)

    proj = pyproject["project"]
    deps = list(proj.get("dependencies", []))
    for grp in ("gui", "viz-fast"):
        deps += proj.get("optional-dependencies", {}).get(grp, [])

    missing = sorted(
        n
        for n in (import_name(d) for d in deps)
        if n not in listed and n not in COVERED and n not in TEST_ONLY
    )
    assert not missing, (
        f"runtime deps neither bundled in xpcsjax-gui.spec nor allowlisted: {missing}. "
        "Add each to the spec's collect_all() list (if it ships extensions/data) "
        "or to COVERED (if pure-python / transitively collected)."
    )


def test_collect_all_extraction_ignores_comment_text():
    # Regression guard (PR #10 review): a `#` comment inside the collect_all()
    # tuple must not corrupt extraction — neither a quoted word that only
    # appears in comment prose ("pytz" below, never actually bundled) should
    # leak into the result, nor should a "):"-shaped sequence inside a
    # comment truncate the match before the tuple's real closing paren.
    synthetic_spec = """
for pkg in (
    "matplotlib",  # also handles "pytz" (a real dep, not bundled here):
    "numpy",
):
"""
    assert _extract_collect_all_names(synthetic_spec) == {"matplotlib", "numpy"}
