"""The GUI entry must be freeze-safe and resolvable as a console script."""

import importlib.metadata
import inspect


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
    import re
    import tomllib

    root = pathlib.Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    spec_src = (root / "packaging" / "xpcsjax-gui.spec").read_text(encoding="utf-8")

    # dist-name -> import-name for the few that differ
    ALIAS = {"scikit-learn": "sklearn", "pyyaml": "yaml"}
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

    tuple_src = re.search(r"for pkg in \((.*?)\):", spec_src, re.S).group(1)
    listed = {p.lower() for p in re.findall(r'"([A-Za-z0-9_]+)"', tuple_src)}

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
