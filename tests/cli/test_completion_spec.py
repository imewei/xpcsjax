from xpcsjax.runtime.shell.completion_spec import COMMAND_SPECS


def test_every_console_script_is_covered():
    names = {n for spec in COMMAND_SPECS for n in spec.command_names}
    expected = {
        "xpcsjax",
        "xj",
        "xjexp",
        "xjsim",
        "xpcsjax-config",
        "xj-config",
        "xpcsjax-config-xla",
        "xj-config-xla",
        "xpcsjax-post-install",
        "xj-post-install",
        "xpcsjax-cleanup",
        "xj-cleanup",
        "xpcsjax-validate",
        "xj-validate",
        "xpcsjax-gui",
        "xj-gui",
    }
    assert names == expected


def test_factories_are_callable_and_unique_funcs():
    funcs = [s.completion_func for s in COMMAND_SPECS]
    assert len(funcs) == len(set(funcs))  # no duplicate function names
    for spec in COMMAND_SPECS:
        parser = spec.parser_factory()
        assert parser is not None


def test_hint_flags_exist_on_their_parser():
    for spec in COMMAND_SPECS:
        opts = {s for a in spec.parser_factory()._actions for s in a.option_strings}
        for flag in spec.dynamic_hints:
            assert flag in opts, f"{spec.completion_func}: hint flag {flag} not in parser"
