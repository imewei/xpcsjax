import shutil
import subprocess
import sys

import pytest

from xpcsjax.runtime.shell.generate_completion import COMPLETION_SH_PATH


def _generate_via_subprocess() -> str:
    """Run generate() in a CLEAN interpreter (spec hermeticity).

    Keeps any import-time side effects of the parser modules out of the pytest
    process, and matches exactly what `python -m ...generate_completion` writes.
    """
    code = (
        "import sys\n"
        "from xpcsjax.runtime.shell.generate_completion import generate\n"
        "sys.stdout.write(generate())\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return res.stdout


@pytest.fixture(scope="module")
def generated() -> str:
    return _generate_via_subprocess()


def test_committed_completion_matches_generator(generated: str):
    committed = COMPLETION_SH_PATH.read_text(encoding="utf-8")
    if generated != committed:
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                committed.splitlines(), generated.splitlines(),
                "committed completion.sh", "generated", lineterm="",
            )
        )
        pytest.fail(f"completion.sh is stale — run `make completion`.\n{diff}")


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "completion.sh is a POSIX (bash/zsh) artifact; on Windows runners "
        "shutil.which('bash') resolves to the System32 WSL launcher stub "
        "(C:\\Windows\\System32\\bash.exe), which exits 1 with a "
        "'no installed distributions' message rather than syntax-checking the "
        "script. Bash-syntax validity is platform-independent and is covered by "
        "the Linux/macOS jobs."
    ),
)
def test_generated_script_is_valid_bash():
    assert shutil.which("bash"), "bash required"
    r = subprocess.run(
        ["bash", "-n", str(COMPLETION_SH_PATH)], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr


def test_three_drift_defects_resolved(generated: str):
    # Phantom flags gone:
    assert "--initial-contrast" not in generated
    assert "--initial-offset" not in generated
    # Previously-missing flags present:
    assert "--initial-beta" in generated
    assert "--no-multistart" in generated


def test_config_data_is_file_completed(generated: str):
    # --data/-d carries a `file` hint -> a value case arm with _filedir.
    assert "--data|-d)" in generated


def test_config_validate_is_flag_not_file(generated: str):
    # --validate/-V is store_true: it must appear in the option list but NEVER as
    # a value-completion case arm (today's hand-written completion file-completes
    # it — the bug this fixes). Check all arm spellings.
    assert "--validate" in generated and "-V" in generated  # present as options
    assert "--validate)" not in generated
    assert "--validate|-V)" not in generated
    assert "-V)" not in generated
