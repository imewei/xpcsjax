"""Pure side-by-side run-comparison table formatting (no Qt).

Extracted from ``ComparisonView.show_runs`` (a ``QWidget`` method) so the
row-building / diff-marking / column-layout logic is a plain, unit-testable
function. ``ComparisonView`` (``gui/views/project_panel.py``) is now a
one-line wrapper: ``self._text.setPlainText(format_comparison(summaries))``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xpcsjax.gui.result_loader import ResultSummary


def format_comparison(summaries: list[tuple[str, ResultSummary | None]]) -> str:
    """Render ``[(label, ResultSummary|None), ...]`` as a side-by-side text table.

    Runs are columns, not stacked text blocks, so values line up for direct
    comparison. Any field row where the runs' values disagree is prefixed
    with ``≠`` — a "side-by-side" view that never marks what differs was not
    actually doing the job of a comparison tool.

    Parameters
    ----------
    summaries : list of (str, ResultSummary or None)
        One entry per compared run: its display label and its summary (or
        ``None`` if that run has no result yet).

    Returns
    -------
    str
        The rendered comparison table, or ``""`` for an empty input.
    """
    if not summaries:
        return ""

    labels = [label for label, _ in summaries]
    col_w = max(12, max(len(label) for label in labels) + 2)
    label_w = 16

    def row(field_label: str, values: list[str | None]) -> str:
        rendered = [v if v is not None else "—" for v in values]
        present = {v for v in rendered if v != "—"}
        prefix = "≠ " if len(present) > 1 else "  "
        cells = "".join(v.ljust(col_w) for v in rendered)
        return f"{prefix}{field_label.ljust(label_w - 2)}{cells}"

    def fmt(x: float | None) -> str | None:
        """Format a possibly-``None`` numeric field.

        chi_squared/reduced_chi_squared can themselves be ``None`` on an
        incomplete/older result — ``f"{None:.6g}"`` raises, so the
        summary-presence check above isn't enough on its own.
        """
        return f"{x:.6g}" if x is not None else None

    def _param_cell(value: float | None) -> str:
        """Format a present parameter's value, distinguishing NaN from absent.

        A ``None`` here means the parameter WAS reported (present in the run's
        ``parameters`` dict) but is non-finite -- render "NaN" so it reads
        differently from the "—" sentinel `row()` uses for a run that never
        reported this parameter at all. Routing this through the plain
        ``fmt()`` above (which also returns ``None`` for ``None``) would
        collapse both cases to the same "—" cell, silently hiding a diverged
        run's NaN parameter and defeating the ``≠`` diff marker.
        """
        return f"{value:.6g}" if value is not None else "NaN"

    lines = [" " * label_w + "".join(label.ljust(col_w) for label in labels)]
    lines.append("-" * len(lines[0]))
    lines.append(row("status", [s.convergence_status if s else None for _, s in summaries]))
    lines.append(row("chi^2", [fmt(s.chi_squared) if s else None for _, s in summaries]))
    lines.append(
        row(
            "reduced chi^2",
            [fmt(s.reduced_chi_squared) if s else None for _, s in summaries],
        )
    )
    lines.append(row("quality", [s.quality_flag if s else None for _, s in summaries]))

    # Union of parameter names across present summaries, first-seen order.
    param_names: list[str] = []
    seen: set[str] = set()
    for _, summary in summaries:
        if summary is None:
            continue
        for name in summary.parameters:
            if name not in seen:
                seen.add(name)
                param_names.append(name)
    if param_names:
        lines.append("")
        lines.append("parameters:")
        for name in param_names:
            values = [
                _param_cell(s.parameters[name]) if s is not None and name in s.parameters else None
                for _, s in summaries
            ]
            lines.append(row(name, values))

    missing = [label for label, s in summaries if s is None]
    if missing:
        lines.append("")
        lines.extend(f"{label}: no result" for label in missing)

    return "\n".join(lines)
