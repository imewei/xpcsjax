"""Application theme — a system-aware "precision instrument" palette and global QSS.

The workbench was previously un-styled: every colour was an ad-hoc inline
``setStyleSheet`` that ignored the OS colour scheme, so the UI read as broken the
moment the system flipped to dark mode. This module is the single source of truth
for colour, typography, and component styling. It is applied once in
:func:`xpcsjax.gui.app.main` and consumed elsewhere through *dynamic properties*
(e.g. ``layer_chip[active="true"]``) rather than per-widget inline CSS.

Aesthetic direction: a dark-first "instrument console" — graphite surfaces, a
single luminous cyan signal-accent, and a monospace family for every numeric
readout (parameters, χ², SSR, logs) so scientific output lines up like a
spectrometer display. A matching light palette is selected automatically when the
OS reports a light colour scheme (CLAUDE.md §4 "system-aware Light/Dark").

JAX-free by construction — PySide6 only. ``pyqtgraph`` is imported lazily and
best-effort so an unusual install never blocks theming.
"""

from __future__ import annotations

from dataclasses import dataclass

# Font stacks — distinctive, engineering-flavoured families with graceful
# fallbacks. IBM Plex is intentionally chosen over the generic Inter/Roboto/Arial
# defaults; the stack degrades to whatever the host actually ships.
FONT_UI = '"IBM Plex Sans", "Cantarell", "Segoe UI", "DejaVu Sans", sans-serif'
FONT_MONO = '"IBM Plex Mono", "JetBrains Mono", "DejaVu Sans Mono", monospace'


@dataclass(frozen=True)
class Palette:
    """A complete colour token set for one colour scheme.

    Every visible colour in the GUI resolves to one of these tokens, so the dark
    and light schemes stay in lock-step and no widget hard-codes a hex value.

    Attributes
    ----------
    name:
        ``"dark"`` or ``"light"`` — used for detection round-trips and tests.
    bg, surface, surface_alt:
        Window background, panel background, and raised/header background.
    border:
        Hairline separators and control outlines.
    text, text_muted:
        Primary and de-emphasised foreground.
    accent, accent_hover, accent_text:
        The signal colour, its hover state, and legible text drawn on top of it.
    success, danger, warning:
        Semantic state colours (active layers, errors, cautions).
    selection:
        Selected-row / highlighted-range fill.
    """

    name: str
    bg: str
    surface: str
    surface_alt: str
    border: str
    text: str
    text_muted: str
    accent: str
    accent_hover: str
    accent_text: str
    success: str
    danger: str
    warning: str
    selection: str


DARK = Palette(
    name="dark",
    bg="#0f1216",
    surface="#161a20",
    surface_alt="#1e242c",
    border="#2a313b",
    text="#e7ecf3",
    text_muted="#8a94a3",
    accent="#22d3ee",
    accent_hover="#4ee0f5",
    accent_text="#06222a",
    success="#34d399",
    danger="#f87171",
    warning="#fbbf24",
    selection="#1f3a40",
)

LIGHT = Palette(
    name="light",
    bg="#eef1f5",
    surface="#ffffff",
    surface_alt="#e6eaf0",
    border="#cdd5df",
    text="#161a20",
    text_muted="#5a6573",
    accent="#0e8f9e",
    accent_hover="#0bb0c1",
    accent_text="#ffffff",
    success="#15803d",
    danger="#c0392b",
    warning="#b45309",
    selection="#cdeef2",
)


def detect_scheme(app: object) -> Palette:
    """Return the :class:`Palette` matching the OS colour scheme.

    Prefers Qt 6.5+'s ``QStyleHints.colorScheme()``; falls back to the window
    background lightness; defaults to :data:`DARK` when undetermined (the
    instrument-console direction is dark-first).

    Parameters
    ----------
    app:
        The live ``QApplication`` instance.

    Returns
    -------
    Palette
        :data:`DARK` or :data:`LIGHT`.
    """
    try:
        from PySide6.QtCore import Qt

        scheme = app.styleHints().colorScheme()  # type: ignore[attr-defined]
        if scheme == Qt.ColorScheme.Dark:
            return DARK
        if scheme == Qt.ColorScheme.Light:
            return LIGHT
    except Exception:  # pragma: no cover — older Qt / headless quirks
        pass
    try:
        from PySide6.QtGui import QPalette

        win = app.palette().color(QPalette.ColorRole.Window)  # type: ignore[attr-defined]
        return DARK if win.lightness() < 128 else LIGHT
    except Exception:  # pragma: no cover — defensive only
        return DARK


def _qpalette(p: Palette) -> object:
    """Build a ``QPalette`` from *p* so native chrome (menus, tooltips) matches."""
    from PySide6.QtGui import QColor, QPalette

    qp = QPalette()
    role = QPalette.ColorRole
    qp.setColor(role.Window, QColor(p.bg))
    qp.setColor(role.WindowText, QColor(p.text))
    qp.setColor(role.Base, QColor(p.surface))
    qp.setColor(role.AlternateBase, QColor(p.surface_alt))
    qp.setColor(role.Text, QColor(p.text))
    qp.setColor(role.Button, QColor(p.surface_alt))
    qp.setColor(role.ButtonText, QColor(p.text))
    qp.setColor(role.ToolTipBase, QColor(p.surface_alt))
    qp.setColor(role.ToolTipText, QColor(p.text))
    qp.setColor(role.Highlight, QColor(p.accent))
    qp.setColor(role.HighlightedText, QColor(p.accent_text))
    qp.setColor(role.PlaceholderText, QColor(p.text_muted))
    qp.setColor(role.Link, QColor(p.accent))
    return qp


def stylesheet(p: Palette) -> str:
    """Return the global Qt stylesheet string for palette *p*.

    Rules are scoped to concrete classes / ``objectName`` / dynamic-property
    selectors rather than a blanket ``QWidget {}`` rule, so the stylesheet never
    bleeds into ``pyqtgraph`` plot internals.

    Parameters
    ----------
    p:
        The active palette.

    Returns
    -------
    str
        A QSS string suitable for ``QApplication.setStyleSheet``.
    """
    return f"""
    QMainWindow, QDialog {{ background: {p.bg}; }}
    QWidget {{ color: {p.text}; font-family: {FONT_UI}; font-size: 10pt; }}

    /* --- Toolbar ---------------------------------------------------------- */
    QToolBar {{
        background: {p.surface};
        border: none;
        border-bottom: 1px solid {p.border};
        padding: 4px 6px;
        spacing: 4px;
    }}
    QToolBar::separator {{
        background: {p.border};
        width: 1px;
        margin: 4px 6px;
    }}
    QToolButton {{
        color: {p.text};
        background: transparent;
        border: 1px solid transparent;
        border-radius: 6px;
        padding: 5px 12px;
        font-weight: 600;
    }}
    QToolButton:hover {{ background: {p.surface_alt}; border-color: {p.border}; }}
    QToolButton:pressed {{ background: {p.selection}; }}
    QToolButton:disabled {{ color: {p.text_muted}; }}

    /* --- Menu bar --------------------------------------------------------- */
    QMenuBar {{ background: {p.surface}; border-bottom: 1px solid {p.border}; }}
    QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
    QMenuBar::item:selected {{ background: {p.surface_alt}; border-radius: 4px; }}
    QMenu {{ background: {p.surface_alt}; border: 1px solid {p.border}; padding: 4px; }}
    QMenu::item {{ padding: 5px 22px; border-radius: 4px; }}
    QMenu::item:selected {{ background: {p.accent}; color: {p.accent_text}; }}

    /* --- Dock widgets ----------------------------------------------------- */
    QDockWidget {{ color: {p.text}; }}
    QDockWidget::title {{
        background: {p.surface_alt};
        color: {p.text_muted};
        padding: 6px 10px;
        border-bottom: 1px solid {p.border};
        font-size: 8.5pt;
        font-weight: 700;
        letter-spacing: 1px;
    }}

    /* --- Tabs ------------------------------------------------------------- */
    QTabWidget::pane {{ border: 1px solid {p.border}; background: {p.surface}; top: -1px; }}
    QTabBar::tab {{
        background: {p.bg};
        color: {p.text_muted};
        padding: 7px 16px;
        border: 1px solid transparent;
        border-bottom: 2px solid transparent;
        margin-right: 2px;
    }}
    QTabBar::tab:hover {{ color: {p.text}; }}
    QTabBar::tab:selected {{
        background: {p.surface};
        color: {p.accent};
        border-bottom: 2px solid {p.accent};
    }}

    /* --- Text / data surfaces (monospace readouts) ----------------------- */
    QPlainTextEdit, QTextEdit {{
        background: {p.surface};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: 6px;
        selection-background-color: {p.accent};
        selection-color: {p.accent_text};
        font-family: {FONT_MONO};
        font-size: 9.5pt;
        padding: 4px;
    }}
    QLineEdit {{
        background: {p.surface};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: 6px;
        padding: 5px 8px;
        selection-background-color: {p.accent};
        selection-color: {p.accent_text};
    }}
    QLineEdit:focus, QPlainTextEdit:focus {{ border-color: {p.accent}; }}

    /* --- Trees / tables / lists ------------------------------------------ */
    QTreeWidget, QTreeView, QTableWidget, QListWidget {{
        background: {p.surface};
        alternate-background-color: {p.surface_alt};
        border: 1px solid {p.border};
        border-radius: 6px;
        outline: none;
    }}
    QTreeWidget::item, QTreeView::item, QListWidget::item {{ padding: 3px 4px; }}
    QTreeView::item:selected, QTreeWidget::item:selected,
    QTableWidget::item:selected, QListWidget::item:selected {{
        background: {p.selection};
        color: {p.text};
    }}
    QHeaderView::section {{
        background: {p.surface_alt};
        color: {p.text_muted};
        border: none;
        border-right: 1px solid {p.border};
        border-bottom: 1px solid {p.border};
        padding: 5px 8px;
        font-weight: 600;
    }}
    QTableWidget {{ gridline-color: {p.border}; }}

    /* --- Buttons / combos ------------------------------------------------ */
    QPushButton {{
        background: {p.surface_alt};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: 6px;
        padding: 5px 14px;
        font-weight: 600;
    }}
    QPushButton:hover {{ border-color: {p.accent}; color: {p.accent}; }}
    QPushButton:pressed {{ background: {p.selection}; }}
    QPushButton:disabled {{ color: {p.text_muted}; border-color: {p.border}; }}
    QComboBox {{
        background: {p.surface};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: 6px;
        padding: 4px 8px;
    }}
    QComboBox:hover {{ border-color: {p.accent}; }}
    QComboBox QAbstractItemView {{
        background: {p.surface_alt};
        border: 1px solid {p.border};
        selection-background-color: {p.accent};
        selection-color: {p.accent_text};
    }}

    /* --- Scrollbars ------------------------------------------------------- */
    QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
    QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 0; }}
    QScrollBar::handle {{ background: {p.border}; border-radius: 5px; min-height: 28px; min-width: 28px; }}
    QScrollBar::handle:hover {{ background: {p.text_muted}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    /* --- Status pill (live run state) ------------------------------------ */
    QLabel#status_pill {{
        background: {p.surface_alt};
        color: {p.text_muted};
        border: 1px solid {p.border};
        border-radius: 10px;
        padding: 4px 12px;
        font-family: {FONT_MONO};
        font-size: 9pt;
    }}
    QLabel#status_pill[state="running"] {{
        color: {p.accent}; border-color: {p.accent};
    }}
    QLabel#status_pill[state="finished"] {{
        color: {p.success}; border-color: {p.success};
    }}
    QLabel#status_pill[state="failed"] {{
        color: {p.danger}; border-color: {p.danger};
    }}

    /* --- Anti-degeneracy layer chips (L1–L5) ----------------------------- */
    /* Targeted by the ``layer_chip`` dynamic property so the per-chip
       objectName (chip_L1 … chip_L5) stays free for findChild() lookups. */
    QLabel[layer_chip="true"] {{
        padding: 3px 11px;
        border-radius: 9px;
        font-family: {FONT_MONO};
        font-weight: 700;
        font-size: 9pt;
        background: {p.surface_alt};
        color: {p.text_muted};
        border: 1px solid {p.border};
    }}
    QLabel[layer_chip="true"][active="true"] {{
        background: {p.success};
        color: {p.accent_text};
        border-color: {p.success};
    }}

    /* --- Inline status / error labels ------------------------------------ */
    QLabel#config_status[status="ok"] {{ color: {p.success}; font-weight: 600; }}
    QLabel#config_status[status="error"] {{ color: {p.danger}; font-weight: 600; }}
    QLabel#data_error {{ color: {p.danger}; }}
    """


def apply_theme(app: object, palette: Palette | None = None) -> Palette:
    """Apply the full theme (style, font, palette, QSS, plot colours) to *app*.

    Parameters
    ----------
    app:
        The live ``QApplication``.
    palette:
        An explicit :class:`Palette` to force; ``None`` auto-detects via
        :func:`detect_scheme`.

    Returns
    -------
    Palette
        The palette that was applied (handy for tests / callers).
    """
    from PySide6.QtGui import QFont

    p = palette or detect_scheme(app)

    # Fusion gives a consistent, palette-driven base across platforms.
    try:
        app.setStyle("Fusion")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover — defensive only
        pass

    font = QFont()
    font.setFamilies(["IBM Plex Sans", "Cantarell", "Segoe UI", "DejaVu Sans"])
    font.setPointSize(10)
    app.setFont(font)  # type: ignore[attr-defined]

    app.setPalette(_qpalette(p))  # type: ignore[attr-defined]
    app.setStyleSheet(stylesheet(p))  # type: ignore[attr-defined]

    # Best-effort: match pyqtgraph plot backgrounds to the theme so the SSR
    # curve and C₂ previews sit inside the console rather than glowing white.
    try:  # pragma: no cover — exercised only with a real pyqtgraph install
        import pyqtgraph as pg

        pg.setConfigOption("background", p.surface)
        pg.setConfigOption("foreground", p.text_muted)
    except Exception:
        pass  # pyqtgraph theming is best-effort (missing install, broken install, or a malformed palette); never block app startup over it

    global _active_palette
    _active_palette = p
    return p


def current_palette() -> Palette:
    """Return the palette last applied by :func:`apply_theme` (``DARK`` before any apply).

    Widgets that need a resolved hex value for rich text (log/result severity
    coloring, where QSS attribute selectors can't reach into free-form appended
    HTML) read it from here instead of hardcoding a color.
    """
    return _active_palette


_active_palette: Palette = DARK


def app_icon(palette: Palette | None = None) -> object:
    """Build a window/taskbar icon from the active palette (no bundled asset).

    No icon file exists anywhere in the repo, so without this the app shows
    the generic Python interpreter icon in the taskbar/dock/alt-tab. A
    monogram on the accent color keeps the icon in lock-step with the
    dark/light theme instead of shipping a static asset that could drift.

    Parameters
    ----------
    palette:
        An explicit :class:`Palette`; ``None`` uses :func:`current_palette`.

    Returns
    -------
    QIcon
        A generated icon (an "X" monogram on an accent-colored rounded tile).
    """
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap

    p = palette or current_palette()
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(p.accent))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(QRectF(0, 0, size, size), size * 0.2, size * 0.2)
    painter.setPen(QColor(p.accent_text))
    font = QFont("IBM Plex Mono")
    font.setBold(True)
    font.setPixelSize(int(size * 0.6))
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "X")
    painter.end()
    return QIcon(pixmap)


def repolish(widget: object) -> None:
    """Re-evaluate dynamic-property selectors after a property change.

    Qt only re-applies attribute-selector QSS (e.g. ``[active="true"]``) once the
    widget is unpolished/repolished. Call this after ``setProperty``.

    Parameters
    ----------
    widget:
        The widget whose style should be recomputed.
    """
    style = widget.style()  # type: ignore[attr-defined]
    style.unpolish(widget)
    style.polish(widget)
    widget.update()  # type: ignore[attr-defined]
