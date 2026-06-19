"""Live-diagnostics view widgets: SSR curve, L1–L5 chips, banner list.

Logic-free views — they render data pushed by the controller. The SSR curve
consumes Iteration events (produced once the Plan-E2 engine seam lands); the
chips/banners consume LayerStatus/Banner events available now.
"""

from __future__ import annotations

import math

import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QVBoxLayout, QWidget

_LAYER_ORDER = ("L1", "L2", "L3", "L4", "L5")


class SSRCurveWidget(pg.PlotWidget):
    """A convergence plot of SSR vs iteration."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # An explicit title signals the curve is expected-empty until a fit
        # streams per-iteration data (the engine source lands in Plan E2), so a
        # blank plot does not read as broken.
        self.setTitle("Convergence (SSR) — live per-iteration data arrives in Plan E2")
        self.setLabel("left", "SSR")
        self.setLabel("bottom", "iteration")
        self.setLogMode(y=True)
        self._xs: list[int] = []
        self._ys: list[float] = []
        self._curve = self.plot(self._xs, self._ys)

    def add_point(self, n: int, ssr: float) -> None:
        """Append one (iteration, SSR) sample and redraw.

        Non-positive or non-finite SSR is skipped — the y-axis is log-scaled, so
        0 / negative / NaN have no valid position on it.
        """
        ssr = float(ssr)
        if not (ssr > 0.0 and math.isfinite(ssr)):
            return
        self._xs.append(int(n))
        self._ys.append(ssr)
        self._curve.setData(self._xs, self._ys)

    def reset(self) -> None:
        """Clear the curve (called when a new fit starts)."""
        self._xs.clear()
        self._ys.clear()
        self._curve.setData(self._xs, self._ys)

    def point_count(self) -> int:
        """Return the number of plotted samples (inspection helper)."""
        return len(self._xs)


class LayerStatusChips(QWidget):
    """Five on/off chips for anti-degeneracy layers L1–L5."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        self._chips: dict[str, QLabel] = {}
        for name in _LAYER_ORDER:
            chip = QLabel(name)
            chip.setObjectName(f"chip_{name}")
            self._style_chip(chip, active=False)
            layout.addWidget(chip)
            self._chips[name] = chip

    @staticmethod
    def _style_chip(chip: QLabel, *, active: bool) -> None:
        chip.setProperty("active", active)
        chip.setStyleSheet(
            "QLabel { padding: 2px 8px; border-radius: 8px; "
            + ("background:#2e7d32; color:white; }" if active else "background:#bdbdbd; color:#444; }")
        )

    def set_layers(self, layers: dict[str, bool]) -> None:
        """Update the chips from an L1–L5 active map."""
        for name, chip in self._chips.items():
            self._style_chip(chip, active=bool(layers.get(name, False)))

    def active_layers(self) -> set[str]:
        """Return the set of currently-active layer names (inspection helper)."""
        return {name for name, chip in self._chips.items() if bool(chip.property("active"))}


class BannerList(QWidget):
    """A scrolling list of engine banners (anti-degeneracy / escape / collapse)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._list = QListWidget()
        layout.addWidget(self._list)

    def add_banner(self, text: str, kind: str) -> None:
        """Append one banner row."""
        self._list.addItem(f"[{kind}] {text}")

    def count(self) -> int:
        """Return the number of banner rows (inspection helper)."""
        return self._list.count()

    def clear(self) -> None:
        """Remove all banner rows."""
        self._list.clear()
