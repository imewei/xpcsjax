Lazy Public API
===============

xpcsjax exposes six public symbols, all resolved lazily through the
``_LAZY_EXPORTS`` table and a module-level ``__getattr__`` in
:mod:`xpcsjax`. The point of the indirection is to keep
``import xpcsjax`` cheap: JAX, NumPyro-free NLSQ, ``optimistix``, and
the rest of the heavyweight scientific stack only get imported when a
user actually touches one of the six names.

The three pieces of the pattern
-------------------------------

1. The lazy table
~~~~~~~~~~~~~~~~~

.. code-block:: python

    _LAZY_EXPORTS = {
        "load_xpcs_data":     "xpcsjax.data",
        "fit_nlsq":           "xpcsjax.optimization.nlsq",
        "ConfigManager":      "xpcsjax.config",
        "HomodyneModel":      "xpcsjax.core",
        "HeterodyneModel":    "xpcsjax.core",
        "OptimizationResult": "xpcsjax.optimization.nlsq.results",
    }

Each key is a public symbol name; each value is the dotted module
path that owns the symbol's real definition.

2. The literal ``__all__``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    __all__ = [
        "ConfigManager",
        "HeterodyneModel",
        "HomodyneModel",
        "OptimizationResult",
        "fit_nlsq",
        "load_xpcs_data",
    ]

This must be a **literal list of string literals**, not a derivation
from ``_LAZY_EXPORTS.keys()``. Pyright's
``reportUnsupportedDunderAll`` rejects dynamic ``__all__`` because
static analysers cannot follow comprehensions, ``list()``
conversions, or any other runtime expression. The literal form is
the contract that ``from xpcsjax import *`` (discouraged but legal)
imports exactly these six names and nothing else.

3. The ``__getattr__`` hook
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def __getattr__(name: str) -> Any:
        if name not in _LAZY_EXPORTS:
            raise AttributeError(
                f"module 'xpcsjax' has no attribute {name!r}"
            )
        module = importlib.import_module(_LAZY_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value  # cache on the module dict
        return value

Module-level ``__getattr__`` (PEP 562) fires when a normal attribute
lookup misses. The first ``xpcsjax.fit_nlsq`` access triggers
``importlib.import_module("xpcsjax.optimization.nlsq")``, pulls the
symbol off the resulting module object, caches it on
``xpcsjax``'s globals, and returns it. Every subsequent access goes
through the normal ``__dict__`` lookup, so the lazy cost is paid
exactly once per name.

Why it matters
--------------

A naive eager import — for example::

    from xpcsjax.optimization.nlsq import fit_nlsq    # eager
    from xpcsjax.core import HomodyneModel, HeterodyneModel  # eager

would import JAX and the entire NLSQ engine the moment any user
typed ``import xpcsjax``. Even small CLI helpers and notebooks would
pay the ~1.5 s JAX import cost before doing anything else. The lazy
form keeps the cost local: a script that only calls
:func:`~xpcsjax.optimization.nlsq.fit_nlsq` does pay the price, but a script that only
calls :class:`~xpcsjax.config.ConfigManager` does not.

This matters most for:

- Test suites that import many modules in a loop.
- Sphinx documentation builds (this very page does not need JAX to
  render).
- Editor language servers that ``import xpcsjax`` repeatedly for
  introspection.

Extending the API
-----------------

Adding a seventh public symbol requires touching three places:

1. **Add an entry to** ``_LAZY_EXPORTS`` with the dotted module path.
2. **Add the name to** ``__all__`` (literal-list form — alphabetical
   order, no comprehension).
3. **Expose the symbol from the target submodule.** That is,
   ``getattr(<target_module>, <name>)`` must succeed once the lazy
   import runs.

A runtime ``assert`` near the bottom of :mod:`xpcsjax`
checks that ``set(_LAZY_EXPORTS) == set(__all__)``. Step (3) is
**not** caught by that assert — it is checked at first access, which
manifests as an ``AttributeError`` from the target submodule. The
fix is always to make sure the new name is in the submodule's own
``__all__`` (or otherwise importable from it).

Common drift to avoid
~~~~~~~~~~~~~~~~~~~~~

- Adding to ``__all__`` but forgetting ``_LAZY_EXPORTS`` → assert
  fails at import time.
- Adding to ``_LAZY_EXPORTS`` but forgetting ``__all__`` → assert
  fails at import time, also breaks ``from xpcsjax import *``.
- Adding both but forgetting the submodule re-export → first access
  raises ``AttributeError`` with a confusing message. Always run
  ``python -c "import xpcsjax; xpcsjax.<NewName>"`` after a change.

.. important::

   Public lazy symbols are an API surface. Adding one is a
   breaking-change-shaped commitment: removing or renaming requires
   a deprecation cycle. Resist the urge to add symbols here for
   convenience; submodule-level imports are fine for internal use.

Cross-references
----------------

- :doc:`jax_environment` explains why keeping JAX off the import path
  matters in concrete numbers.
- :doc:`architecture` shows the call graph that runs once the lazy
  symbols are touched.
