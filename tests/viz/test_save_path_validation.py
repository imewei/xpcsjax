"""Security regression tests: plot save paths must be validated.

Quality-gate finding #1: ``_save_fig`` and the Datashader save sinks wrote
attacker-influenceable paths verbatim (``Path(save_path)`` -> ``mkdir`` ->
``savefig``) with no traversal / extension / null-byte check, while the
``validate_plot_save_path`` guard sat unused (dead code). These tests pin the
guard to the save sinks and the ``base_dir`` containment tightening in
``validate_save_path``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from xpcsjax.utils.path_validation import PathValidationError, validate_save_path
from xpcsjax.viz.nlsq_plots import _save_fig


def _fresh_fig():
    fig = plt.figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    return fig


def test_save_fig_rejects_parent_traversal(tmp_path):
    fig = _fresh_fig()
    with pytest.raises(PathValidationError):
        _save_fig(fig, str(tmp_path / ".." / ".." / "evil.png"))


def test_save_fig_rejects_non_image_extension(tmp_path):
    fig = _fresh_fig()
    with pytest.raises(PathValidationError):
        _save_fig(fig, str(tmp_path / "payload.sh"))


def test_save_fig_rejects_null_byte(tmp_path):
    fig = _fresh_fig()
    with pytest.raises(PathValidationError):
        _save_fig(fig, str(tmp_path / "ok.png") + "\x00.sh")


def test_save_fig_accepts_valid_path_and_creates_parent(tmp_path):
    """Regression: legitimate saves into a not-yet-existing subdir still work."""
    fig = _fresh_fig()
    target = tmp_path / "new_subdir" / "trace.png"
    _save_fig(fig, str(target))
    assert target.exists()


def test_validate_save_path_enforces_containment_for_absolute_with_base_dir(tmp_path):
    """When an explicit base_dir is given, an absolute path outside it is rejected
    even though allow_absolute=True (closes the allow_absolute footgun)."""
    base = tmp_path / "allowed"
    base.mkdir()
    outside = tmp_path / "elsewhere" / "x.png"
    with pytest.raises(PathValidationError):
        validate_save_path(
            str(outside),
            allow_absolute=True,
            base_dir=base,
            require_parent_exists=False,
        )


def test_validate_save_path_allows_absolute_inside_base_dir(tmp_path):
    base = tmp_path / "allowed"
    base.mkdir()
    inside = base / "x.png"
    result = validate_save_path(
        str(inside),
        allow_absolute=True,
        base_dir=base,
        require_parent_exists=False,
    )
    assert result == inside.resolve()


datashader = pytest.importorskip("datashader")


def test_datashader_heatmap_rejects_traversal(tmp_path):
    import numpy as np

    from xpcsjax.viz.datashader_backend import plot_c2_heatmap_fast

    data = np.random.rand(8, 8)
    t1 = np.arange(8.0)
    t2 = np.arange(8.0)
    with pytest.raises(PathValidationError):
        plot_c2_heatmap_fast(
            data,
            t1,
            t2,
            output_path=tmp_path / ".." / ".." / "evil.png",
            title="t",
        )
