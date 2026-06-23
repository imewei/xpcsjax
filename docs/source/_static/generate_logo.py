"""Generate the xpcsjax logo: a rendered two-time correlation surface c2(t1, t2).

The logo depicts exactly what xpcsjax computes -- the two-time intensity
correlation function ``c2(t1, t2)`` -- as a 3D surface viewed in a rotated
"diamond" orientation, with the ``t1`` / ``t2`` axes as the lower diagonals.

The surface is a separable damped-cosine field ``h(t1) * h(t2)`` (a Gaussian
envelope modulating a cosine), which produces the ripples running parallel to
the diamond edges and the sharp central ridge. Height is colour-mapped from a
dark teal floor through the body to a hot-pink / near-white crest, on a dark
charcoal background, matching the package's visual identity.

Run:
    uv run python docs/source/_static/generate_logo.py

Writes ``xpcsjax_logo.jpg`` (1024x1024) next to this script. Re-run after
tweaking any of the tunables in ``main()``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Palette (matches docs/source/conf.py furo brand colours + the magenta "jax")
# ---------------------------------------------------------------------------
BG = "#2b2d33"          # dark charcoal background
TEAL_LO = "#0e3a39"     # deep teal floor
TEAL = "#1f7a74"        # body teal
CYAN = "#39b0a6"        # bright teal highlight
MAUVE = "#b06a98"       # transition into the ridge
PINK = "#ff3d9a"        # hot-pink ridge ("jax" colour)
PINK_HI = "#ffd6ea"     # near-white crest
ARROW = "#a9c7e0"       # soft blue axis arrows
TEXT = "#f3f4f6"        # off-white labels
MAGENTA = "#ff2d9b"     # "jax" wordmark


def _cmap() -> LinearSegmentedColormap:
    """Teal floor -> teal body -> mauve -> hot pink -> near-white crest."""
    stops = [
        (0.00, TEAL_LO),
        (0.30, TEAL),
        (0.48, CYAN),
        (0.62, MAUVE),
        (0.78, PINK),
        (1.00, PINK_HI),
    ]
    return LinearSegmentedColormap.from_list("xpcsjax", stops, N=512)


def _surface(n: int, extent: float, sigma: float, k: float) -> tuple[np.ndarray, ...]:
    """Two-time field: smooth radial ripples + an elongated diagonal ridge.

    * Radial sombrero ``exp(-(r/sigma)**2) * cos(k*r)`` gives the smooth
      concentric waves decaying outward (cleaner than a separable lattice).
    * A narrow Gaussian ridge elongated along the ``t1 == t2`` diagonal builds
      the sharp central pink crest -- which, viewed at low elevation, reads as
      the vertical ridge in the reference.
    """
    t = np.linspace(-extent, extent, n)
    t1, t2 = np.meshgrid(t, t)

    def h(x: np.ndarray) -> np.ndarray:
        # Damped cosine: bright centre, lobes shrinking and reaching the edges.
        return np.exp(-((x / sigma) ** 2)) * np.cos(k * x)

    # Separable field -> lobes arranged parallel to the diamond edges
    # (the diffraction-like speckle pattern), filling to sharp corners.
    base = h(t1) * h(t2)

    # Rotated coords: d across the t1==t2 diagonal, s along it. A narrow spike
    # elongated along the diagonal extends the bright pink central ridge.
    d = (t1 - t2) / np.sqrt(2.0)
    s = (t1 + t2) / np.sqrt(2.0)
    ridge = np.exp(-((d / 0.9) ** 2)) * np.exp(-((s / (0.7 * sigma)) ** 2))

    z = base + 0.55 * ridge
    return t1, t2, z


def main() -> Path:
    out = Path(__file__).resolve().parent / "xpcsjax_logo.jpg"

    # --- tunables -----------------------------------------------------------
    n = 700          # grid resolution
    extent = 6.4     # half-width of the (t1, t2) domain
    sigma = 3.2      # lobe envelope width
    k = 1.9          # lobe frequency (a few lobes per axis, reaching the edges)
    elev, azim = 31, -45   # camera: diamond orientation, peak toward viewer
    px = 1024        # output is px x px

    t1, t2, z = _surface(n, extent, sigma, k)
    cmap = _cmap()

    # Height-normalised colours with a directional light for relief.
    norm = (z - z.min()) / (z.max() - z.min())
    ls = LightSource(azdeg=315, altdeg=50)
    rgb = ls.shade(
        norm, cmap=cmap, blend_mode="soft", vert_exag=2.3,
        dx=t1[0, 1] - t1[0, 0], dy=t2[1, 0] - t2[0, 0],
    )

    dpi = 200
    fig = plt.figure(figsize=(px / dpi, px / dpi), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    ax.set_position((-0.06, -0.10, 1.12, 1.18))  # bleed the surface to fill frame

    ax.plot_surface(
        t1, t2, z, facecolors=rgb, rstride=2, cstride=2,
        linewidth=0, antialiased=True, shade=False, zorder=2,
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.72))
    try:
        ax.set_proj_type("persp", focal_length=0.9)
    except TypeError:  # older matplotlib
        ax.set_proj_type("persp")

    # --- overlaid labels (figure coords so placement is exact) --------------
    fig.text(0.70, 0.80, r"$c_2(t_1,\,t_2)$", color=TEXT, fontsize=30,
             ha="center", va="center", style="italic")

    # Axis arrows fanning down-left (t1) and down-right (t2) from the base.
    arrow_kw = dict(arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=3.2,
                                    mutation_scale=24), xycoords="figure fraction")
    fig.text(0.205, 0.135, r"$t_1$", color=TEXT, fontsize=30, ha="center", va="center")
    fig.text(0.795, 0.135, r"$t_2$", color=TEXT, fontsize=30, ha="center", va="center")
    ax.annotate("", xy=(0.085, 0.20), xytext=(0.45, 0.085), **arrow_kw)
    ax.annotate("", xy=(0.915, 0.20), xytext=(0.55, 0.085), **arrow_kw)

    # Wordmark: "xpcs" (white) + "jax" (magenta), centred at the base.
    fig.text(0.415, 0.058, "xpcs", color=TEXT, fontsize=33, ha="right", va="center",
             fontweight="bold", family="DejaVu Sans")
    fig.text(0.415, 0.058, "jax", color=MAGENTA, fontsize=33, ha="left", va="center",
             fontweight="bold", family="DejaVu Sans")

    fig.savefig(out, dpi=dpi, facecolor=BG, pad_inches=0)
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = main()
    print(f"wrote {path}")
