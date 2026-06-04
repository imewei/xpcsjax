"""Quality-gate finding #7: the validation helpers narrow their except clause to
``(ValueError, TypeError, KeyError, IndexError, RuntimeError)`` but routinely touch
``.shape`` / ``.ndim`` / arithmetic on loaded values. An ``AttributeError`` (a
non-array value where an array was expected) or ``ArithmeticError`` therefore
escapes uncaught — the validation crashes silently instead of recording a
data-format issue. These tests pin those two exception types into the catch.
"""

from __future__ import annotations

from xpcsjax.data import validation as val


def _report():
    return val.DataQualityReport(is_valid=True, validation_level="basic", total_issues=0)


def test_validate_array_shapes_records_issue_on_attributeerror(monkeypatch):
    report = _report()

    def _raise(*_a, **_k):
        raise AttributeError("simulated non-array value (no .shape)")

    monkeypatch.setattr(val.np, "asarray", _raise)

    # Must NOT propagate — it must be caught and recorded as a crash issue.
    val._validate_array_shapes({"t1": [1, 2]}, report)
    assert any("validation crashed" in i.message for i in report.errors), (
        "an AttributeError inside the validator must be recorded, not propagated"
    )


def test_validate_array_shapes_records_issue_on_arithmeticerror(monkeypatch):
    report = _report()

    def _raise(*_a, **_k):
        raise OverflowError("simulated arithmetic overflow")  # ArithmeticError subclass

    monkeypatch.setattr(val.np, "asarray", _raise)

    val._validate_array_shapes({"t1": [1, 2]}, report)
    assert any("validation crashed" in i.message for i in report.errors)
