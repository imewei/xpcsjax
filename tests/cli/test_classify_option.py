import argparse

from xpcsjax.runtime.shell.generate_completion import Completion, classify_option


def _action(*flags, **kw):
    p = argparse.ArgumentParser()
    return p.add_argument(*flags, **kw)


def test_store_true_is_flag_returns_none():
    assert classify_option(_action("--plot", action="store_true"), {}) is None


def test_store_false_is_flag_returns_none():
    # --no-plot is store_false: zero-arg, must NOT raise (the BLOCKER fix).
    assert classify_option(_action("--no-plot", action="store_false"), {}) is None


def test_count_action_is_flag_returns_none():
    assert classify_option(_action("-v", "--verbose", action="count"), {}) is None


def test_choices_become_choices_completion():
    c = classify_option(_action("--mode", choices=["a", "b"]), {})
    assert c == Completion(kind="choices", payload="a b")


def test_path_type_defaults_to_file():
    from pathlib import Path

    c = classify_option(_action("--out", type=Path), {})
    assert c == Completion(kind="file")


def test_dir_hint_overrides_path():
    from pathlib import Path

    c = classify_option(_action("--out", type=Path), {"--out": "dir"})
    assert c == Completion(kind="dir")


def test_str_path_needs_explicit_file_hint():
    # type=str with no hint -> plain value, no completion.
    assert classify_option(_action("--data", type=str), {}) == Completion(kind="none")
    # with file hint -> file completion.
    assert classify_option(_action("--data", type=str), {"--data": "file"}) == Completion(
        kind="file"
    )


def test_literal_word_hint():
    c = classify_option(_action("--xla-mode", type=str), {"--xla-mode": ("auto", "nlsq")})
    assert c == Completion(kind="words", payload="auto nlsq")


def test_threads_hint():
    c = classify_option(_action("--threads", type=int), {"--threads": "threads"})
    assert c.kind == "threads"


def test_plain_value_no_hint_no_choices_is_none_kind():
    assert classify_option(_action("--tol", type=float), {}) == Completion(kind="none")
