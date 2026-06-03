"""Sphinx configuration for xpcsjax documentation.

Builds against the installed xpcsjax package. The lazy public-API loader in
``xpcsjax/__init__.py`` keeps JAX off the import path until first use, so
``autodoc`` can introspect the package without paying the JAX warm-up cost on
every build.
"""

from __future__ import annotations

import importlib.metadata as _md
import os
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Path setup
# ----------------------------------------------------------------------------
# Make ``xpcsjax`` importable from a source checkout without ``pip install -e``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# ----------------------------------------------------------------------------
# Project information
# ----------------------------------------------------------------------------
project = "xpcsjax"
author = "Wei Chen"
copyright = "2026, Wei Chen (Argonne National Laboratory)"

try:
    _full_version = _md.version("xpcsjax")
except _md.PackageNotFoundError:
    _full_version = "0.1.0"
version = _full_version.split(".post")[0].split("+")[0]
release = _full_version

# ----------------------------------------------------------------------------
# General configuration
# ----------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

# Exclude the "All modules" page from LaTeX builds. It exists only so HTML
# :mod: cross-references resolve; LaTeX emits two labels per py:module
# directive, which makes the LaTeX run report ~100 "Label multiply defined"
# warnings. Skipping it in LaTeX is harmless because the PDF audience does
# not click :mod: links.
import sys as _sys  # noqa: E402  (Sphinx conf: intentional mid-file import near use)

if any(a in ("latex", "latexpdf") for a in _sys.argv):
    exclude_patterns.append("api/modules.rst")

# Ignore unresolved cross-refs for third-party types that would resolve under
# intersphinx but are noise in offline builds. Keep this list tight — it
# should only cover stdlib, numpy, jax, scipy, h5py, evosax surface.
_THIRD_PARTY_PKGS = r"(jax|jaxlib|numpy|np|scipy|h5py|evosax|nlsq|interpax|psutil|cloud[a-z]+|sklearn|jaxopt|optax|argparse|matplotlib|PIL|os)"
nitpick_ignore_regex = [
    (r"py:(class|data|obj|func|attr|exc)", _THIRD_PARTY_PKGS + r"\..*"),
    (r"py:(class|data|obj)", r"typing\..*"),
    (r"py:(class|data|obj)", r"(np|jnp|sp)\..*"),
    (r"py:(class|data|obj)", r"collections\..*"),
    (r"py:(class|data|obj)", r"pathlib\..*"),
    (r"py:(class|data|obj)", r"(int|float|str|bool|bytes|list|tuple|dict|set|frozenset|type|object|None|NoneType|Any|Optional|optional|Union|Callable|Iterable|Iterator|Mapping|Sequence|Generator|Ellipsis|NDArray|Path|ArrayLike|DType|DTypeLike)"),
    # Private autodoc protocol references that leak from class hierarchies.
    (r"py:class", r".*\._[A-Z][A-Za-z0-9_]*Protocol"),
    # Bare class refs that appear inside autodoc'd docstrings of xpcsjax
    # internal modules. The classes are documented at their canonical paths
    # under :doc:`/api/core`, but the docstrings reference them without the
    # full path. Fixing these would mean editing every docstring; suppress
    # the bare-name lookups instead.
    (r"py:class", r"PhysicsModelBase|DiffusionModel|CombinedModel|PhysicsFactors"),
    (r"py:meth", r"from_config|HomodyneModel\.compute_c2|HeterodyneModel\.compute_g1|compute_g1|compute_c2_single_angle"),
    (r"py:func", r"xpcsjax\.core\.models\.make_model"),
    # Bare attribute names referenced from inside autodoc'd docstrings.
    (r"py:attr", r"parameter_names|param_names"),
    # Bare class/func names referenced from CLI/internal docstrings. Same
    # rationale as the bare-name block above: the symbols are documented at
    # their canonical paths (or are deliberately internal), but the docstrings
    # reference them unqualified. Suppress the lookups instead of rewriting
    # every docstring.
    (r"py:class", r"ConfigManager|OptimizationResult|GradientCollapseMonitor"),
    (r"py:func", r"fit_nlsq_jax"),
    (r"py:func", r"xpcsjax\.(fit_nlsq|load_xpcs_data)"),
    # Internal enums/classes referenced fully-qualified from type hints and
    # authored prose but intentionally not autodoc'd (members are suppressed
    # on the dedicated API pages).
    (r"py:class", r"xpcsjax\.config\.parameter_registry\.AnalysisMode"),
    (r"py:class", r"xpcsjax\.config\.physics_validators\.ConstraintSeverity"),
    (r"py:class", r"xpcsjax\.uninstall_scripts\.CleanupTarget"),
    (r"py:func", r"xpcsjax\.optimization\.nlsq\.cmaes_wrapper\.fit_with_cmaes"),
    (
        r"py:func",
        r"xpcsjax\.optimization\.nlsq\.anti_degeneracy_diagnostics\."
        r"assemble_anti_degeneracy_diagnostics",
    ),
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
language = "en"
nitpicky = False

# Surface but don't fail on common autodoc noise.
suppress_warnings = [
    "myst.xref_missing",
    "autosummary.import_cycle",
    "ref.python",
    # The six public lazy-loaded symbols are intentionally documented twice:
    # once at the user-facing ``xpcsjax.<name>`` path (api/public.rst) and
    # once at the canonical backing-module path (api/core.rst etc.). Sphinx
    # 8+ emits `python.duplicate_object` for each pair; suppress it.
    "duplicate_object_description",
    "docutils",
    # docutils-level "Definition list ends without a blank line" warnings
    # come from a malformed docstring in
    # xpcsjax/core/heterodyne_jax_backend.py that's outside the docs build's
    # control. Suppress at parser level.
    "docutils.parsers.rst",
]

# ----------------------------------------------------------------------------
# autodoc / autosummary
# ----------------------------------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
autodoc_class_signature = "mixed"

# Don't prepend the canonical module path to documented names — keep the
# directive's currentmodule as the authoritative location.
add_module_names = False
autodoc_default_options = {
    "undoc-members": False,
}

# Heavy or optional deps that may not be installed when autodoc runs in CI.
# The ``viz-fast`` datashader backend (xpcsjax.viz.datashader_backend) imports
# datashader/xarray/Pillow only when that extra is installed; mock them so
# autodoc can introspect the backend without the optional stack (otherwise the
# import fails and the strict ``-W`` build errors on the autodoc warning).
autodoc_mock_imports: list[str] = ["datashader", "xarray", "PIL"]

# Napoleon (NumPy-style docstrings, like the upstream homodyne / heterodyne
# packages this code was ported from).
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_param = True
napoleon_use_rtype = True

typehints_fully_qualified = False
typehints_document_rtype = True
always_document_param_types = True

# Napoleon turns NumPy-style ``Attributes`` sections into autodoc-style
# attribute directives. When combined with autoclass `:members:`, the same
# attribute is documented twice — once by Napoleon, once by autodoc.
napoleon_attr_annotations = False
napoleon_use_ivar = True

# ----------------------------------------------------------------------------
# Intersphinx
# ----------------------------------------------------------------------------
# Default to offline-friendly builds; set SPHINX_OFFLINE=0 to actually fetch
# the upstream inventories.
_offline = os.environ.get("SPHINX_OFFLINE", "1") == "1"
intersphinx_mapping = (
    {}
    if _offline
    else {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "scipy": ("https://docs.scipy.org/doc/scipy/", None),
        "jax": ("https://docs.jax.dev/en/latest/", None),
    }
)
intersphinx_timeout = 15

# ----------------------------------------------------------------------------
# MyST
# ----------------------------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# ----------------------------------------------------------------------------
# HTML output
# ----------------------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = True
html_show_sphinx = False
html_copy_source = False
html_title = f"xpcsjax {version}"
html_short_title = "xpcsjax"

# ----------------------------------------------------------------------------
# LaTeX / PDF output — use xelatex so Unicode physics symbols (χ, α, Å, ²,
# ≈, ⁻¹) survive without a giant fontenc preamble.
# ----------------------------------------------------------------------------
latex_engine = "xelatex"
latex_documents = [
    ("index", "xpcsjax.tex", "xpcsjax Documentation", author, "manual"),
]
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    # cmap is a pdftex-only package; under xelatex it just prints a warning
    # at start and exits. Skip it entirely.
    "cmappkg": "",
    "preamble": r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{unicode-math}
% Help hyperref render Unicode/math characters in PDF bookmarks. Without this,
% xelatex crashes on section headings that contain math like ``:math:`\chi^2```.
\PassOptionsToPackage{unicode,pdfencoding=auto,psdextra}{hyperref}
% Sphinx-rendered autodoc can produce deeply nested itemize blocks (NumPy-style
% docstrings → field lists → bullet lists → ...). LaTeX caps nesting at 4 by
% default; bump it so xelatex can render them.
\usepackage{enumitem}
\setlistdepth{12}
\renewlist{itemize}{itemize}{12}
\setlist[itemize]{label=\textbullet}
\renewlist{enumerate}{enumerate}{12}
% Long Python qualified names (e.g.
% ``xpcsjax.optimization.nlsq.strategies.hybrid_streaming``) frequently don't
% fit cleanly into a column, producing hundreds of cosmetic "Underfull \hbox"
% and "Overfull \hbox" warnings. Raise the badness threshold above the maximum
% (10000) so these layout-only complaints stop firing — they're noise here,
% not a correctness signal.
\hbadness=10001
\vbadness=10001
\hfuzz=200pt
\vfuzz=200pt
% The document-level \hfuzz/\hbadness above do not reach inside table cells:
% LaTeX runs \@arrayparboxrestore at the start of every tabulary/varwidth cell,
% which resets paragraph parameters. xpcsjax's API/theory tables hold long
% inline code literals (e.g. ``static_anisotropic``, ``compute_regularization_jax``)
% that slightly overflow their auto-sized columns, producing cosmetic
% "Overfull \hbox" warnings detected during the cell's varwidth packing. Re-apply
% the tolerances inside the cell restore hook so the suppression actually covers
% table cells too.
\makeatletter
\g@addto@macro\@arrayparboxrestore{\hfuzz=200pt\hbadness=10001\relax}
\makeatother
""",
    "figure_align": "H",
}

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "https://github.com/imewei/xpcsjax",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/imewei/xpcsjax",
            "html": "",
            "class": "fa-brands fa-github",
        },
    ],
    "light_css_variables": {
        "color-brand-primary": "#0a6cb8",
        "color-brand-content": "#0a6cb8",
        "color-admonition-background": "transparent",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5bb0ee",
        "color-brand-content": "#5bb0ee",
    },
}

# ----------------------------------------------------------------------------
# Other
# ----------------------------------------------------------------------------
todo_include_todos = False

# Copy button rules — exclude the leading prompt characters from copy.
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: |\s*\.\.\.: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False
