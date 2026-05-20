Contributing
============

This page describes how to set up a development environment for xpcsjax,
the code-style rules that the linter and type checker enforce, and the
pre-push workflow that mirrors CI. New contributors should read this
page in full before opening their first pull request.

Setting up a development environment
-------------------------------------

xpcsjax standardises on `uv <https://docs.astral.sh/uv/>`_ for package
management. The repository ships a ``uv.lock`` file; ``pip`` and
``conda`` are only kept in the Makefile for portability and should not
be used for day-to-day work.

Clone the repository, create a Python 3.12+ virtual environment, and
install the development extras:

.. code-block:: shell

   git clone https://github.com/<your-fork>/xpcsjax.git
   cd xpcsjax
   uv sync
   make dev

``make dev`` resolves to ``uv pip install -e ".[dev]"`` and pulls in
``pytest``, ``pytest-cov``, ``pytest-xdist``, ``hypothesis``, ``ruff``,
``mypy``, and the type stubs. To work on the documentation as well, add
the ``docs`` extra:

.. code-block:: shell

   uv pip install -e ".[dev,docs]"

This installs Sphinx, the Furo theme, ``sphinx-copybutton``,
``sphinx-autodoc-typehints``, ``sphinx-design``, and ``myst-parser``.

.. important::

   ``uv.lock`` is the single source of truth for dependency versions.
   Never run a bare ``pip install`` in the project virtual environment.
   If you need a new dependency, edit :file:`pyproject.toml` and re-run
   ``uv sync`` so that the lock file is updated atomically.

Verifying the install
~~~~~~~~~~~~~~~~~~~~~

Once dependencies are installed, confirm that the NLSQ integration
imports cleanly:

.. code-block:: shell

   make verify-nlsq

This runs three import-level checks: :func:`xpcsjax.optimization.nlsq.fit_nlsq`
is reachable, the upstream ``nlsq`` package is importable with a
version stamp, and ``evosax`` (the CMA-ES backend) is available.

Code style
----------

The project pins a single linter and a single formatter, both backed by
`ruff <https://docs.astral.sh/ruff/>`_, and a non-strict ``mypy`` pass
for catching the easy class of typing mistakes at API boundaries.

Ruff configuration
~~~~~~~~~~~~~~~~~~

The rule set is declared in :file:`pyproject.toml`:

.. code-block:: toml

   [tool.ruff]
   line-length = 100
   target-version = "py312"

   [tool.ruff.lint]
   select = ["E", "F", "W", "I", "B", "UP", "N"]
   ignore = ["E501", "N806", "N803"]

Notes:

- Line length is 100. ``E501`` is ignored because the formatter handles
  wrapping; the linter would otherwise double-flag every long string.
- ``N806`` and ``N803`` are off because scientific code uses
  uppercase math-convention names (``D0``, ``JTJ``, ``L``, ``P``, ``Q``,
  ``R``, ``X``, ``Y``) in function and argument scope. Enforcing
  ``snake_case`` here would degrade readability against the physics
  literature.
- ``F`` (Pyflakes) blocks ``from module import *``. This matches the
  user-level convention in :file:`CLAUDE.md`.
- ``UP`` enforces modern Python 3.12 idioms (``X | Y`` unions, ``list``
  / ``dict`` generics over ``typing.List``).

Run the linter and formatter together with:

.. code-block:: shell

   make format    # ruff format + ruff check --fix
   make lint      # ruff check (no fix)

mypy configuration
~~~~~~~~~~~~~~~~~~

``mypy`` is intentionally non-strict:

.. code-block:: toml

   [tool.mypy]
   python_version = "3.12"
   strict = false
   ignore_missing_imports = true

Several NLSQ-engine and heterodyne modules have per-module disable
lists because JAX's ``Array`` and NumPy's ``ndarray`` are
interchangeable at runtime but their type stubs do not model the
relationship. Run mypy with:

.. code-block:: shell

   make type-check

In the pre-push gate (``make verify``) mypy runs in **advisory** mode
(``| tail -1 || true``), so type findings do not block the push. Treat
type-check output as guidance, not a gate, until the JAX stub story
improves upstream.

Commits and pull requests
-------------------------

The repository uses `Conventional Commits
<https://www.conventionalcommits.org/>`_. Recent history is dominated
by these prefixes:

.. code-block:: text

   docs(graphify): update semantic knowledge graph and architecture reports
   refact(types): resolve static type analysis issues across modules
   fix(optimization): resolve heterodyne CMA-ES signature mismatch
   sec(data): harden trusted cache loader with pre-deserialization gates
   build(env): add bandit security scans and wall-clock benchmarks
   feat(nlsq): support unwrapped optimization config and gated layers
   test(property): cross-cutting parameter-registry invariants

Use the same shape for new commits. Suggested prefixes:

- ``feat`` — user-visible new functionality.
- ``fix`` — bug fix with no API change.
- ``refact`` — internal refactor, no behaviour change.
- ``test`` — test-only changes.
- ``docs`` — documentation only.
- ``build`` — packaging, dependencies, lockfile.
- ``ci`` — workflow and CI config.
- ``sec`` — security-sensitive changes.

Pre-push checklist
~~~~~~~~~~~~~~~~~~

Before pushing a branch or opening a pull request, run:

.. code-block:: shell

   make verify

``make verify`` runs three steps in order, mirroring CI:

1. Linting (``ruff check`` on ``xpcsjax/`` and ``tests/``).
2. Type checking (advisory, output truncated to the summary line).
3. Smoke tests (``pytest tests -n auto -v --tb=short -x -q``).

If only static checks are needed, use:

.. code-block:: shell

   make verify-fast

Pull request expectations:

- Branch off ``main`` and rebase before requesting review.
- Reference the relevant issue in the PR description.
- If you added or changed a public symbol, update the docs in
  :doc:`/api/index`.
- If you added a new test category, update :doc:`testing`.
- If the change touches porting from upstream homodyne or heterodyne,
  update :doc:`porting_notes` and regenerate baselines as described
  there.

The "don't touch" list
----------------------

These rules are non-negotiable for contributors. They exist because
breaking them silently destroys the parity contract or re-introduces
removed upstream APIs.

.. warning::

   **Do not loosen characterisation tolerances.**
   :file:`tests/characterization/test_homodyne_equivalence.py` asserts
   bit-comparable output against the upstream ``homodyne`` package at
   ``rtol=1e-10``. If a regression makes this test fail, fix the code,
   do not loosen the tolerance. The only legitimate reason to
   regenerate the baseline is that the upstream ``homodyne`` package
   itself changed; see :doc:`porting_notes` for the procedure.

.. warning::

   **Do not call removed NLSQ APIs.**
   The upstream ``nlsq>=0.6.10`` package removed
   ``WorkflowSelector`` in v0.6.0. xpcsjax also does not call
   ``MemoryBudgetSelector`` because :func:`xpcsjax.optimization.nlsq.select_nlsq_strategy`
   routes memory itself. Both names are tripwires; new code must not
   reintroduce them. See :doc:`nlsq_integration` for the full
   ownership split.

.. warning::

   **Do not add Bayesian sampling code paths.**
   xpcsjax is NLSQ-only by design for v0.1. Stale references to
   ``get_cmc_config``, ``_get_default_cmc_config``, and the
   ``"mcmc"`` config block survive from the homodyne port and are
   scheduled for removal. Do not add new call sites and do not write
   tests that exercise them. Users needing Bayesian XPCS analysis
   should use the upstream ``homodyne`` or ``heterodyne`` packages.

.. warning::

   **Do not use ``jax.numpy.interp`` inside JIT-compiled paths.**
   Use ``interpax`` instead. This is enforced by convention, not by
   the linter, so reviewers will catch it manually.

.. note::

   Float64 is mandatory throughout the package. ``JAX_ENABLE_X64=1`` is
   set at package import (see :mod:`xpcsjax`) and in the
   pytest environment block of :file:`pyproject.toml`. Parameters span
   six or more orders of magnitude in typical XPCS fits; ``float32``
   silently loses the small components.

Where to ask questions
----------------------

- For an architectural question that spans multiple modules, read
  :doc:`/advanced/architecture` first, then file an issue.
- For a porting question (homodyne or heterodyne parity), read
  :doc:`porting_notes` and inspect
  :file:`scripts/generate_homodyne_baselines.py`.
- For an NLSQ-engine question, read :doc:`nlsq_integration` and
  inspect :mod:`xpcsjax.optimization.nlsq`.
