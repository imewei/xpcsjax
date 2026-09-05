Releasing
=========

How a new xpcsjax version is cut, tagged, and published to PyPI. The project
follows `Semantic Versioning <https://semver.org/>`_ and `Keep a Changelog
<https://keepachangelog.com/>`_; the current line is **v0.1.7** (CPU-only;
GPU support is planned for v0.2+).

Publishing is automated by the :file:`.github/workflows/release.yml` workflow,
which uploads to PyPI via **Trusted Publishing** (OIDC) — there is no API-token
secret to manage.

Version consistency
-------------------

The package version is declared in two source-of-truth files and surfaced in
several others. A release **must** keep all of these in agreement:

============================================  ===================================================
Site                                          What it holds
============================================  ===================================================
:file:`pyproject.toml` ``[project].version``  Canonical build version (read by the wheel/sdist).
:file:`xpcsjax/__init__.py` ``__version__``   Runtime constant; kept equal to the build version.
:file:`docs/source/conf.py`                   Reads the installed metadata; ``0.1.7`` fallback.
:file:`docs/source/installation.rst`          Doctest line ``>>> xpcsjax.__version__``.
:file:`docs/source/changelog.rst`             User-facing milestone heading.
:file:`CHANGELOG.md`                          Authoritative dated changelog entry.
:file:`README.md`                             Citation block ``version = {…}``.
============================================  ===================================================

At runtime everything else resolves dynamically: :func:`importlib.metadata.version`
reads the installed distribution, and both the CLI (``xpcsjax --version``) and
``conf.py`` defer to it, falling back to the hardcoded string only when the
package is not installed.

.. note::

   **Three distinct version concepts live in the tree — do not unify them.**

   * The **package version** (``0.1.7``) — the release identity described above.
   * The **data-layer provenance version** (``xpcsjax/data/__init__.py``
     ``__version__ = "2.23.1"``) — tracks the ported upstream ``xpcs_loader``
     and is surfaced by :func:`xpcsjax.data.get_data_module_info`. It is
     intentionally decoupled from the package version.
   * The **config-schema version** (``config_version: "0.1.7"`` in the config
     templates) — versions the YAML config format. ``ConfigManager`` only warns
     on a **major.minor** mismatch against the package version
     (:meth:`ConfigManager._validate_config_version`), so it does not have to
     track the package's patch version exactly — but it is kept in sync as a
     matter of release hygiene.

Release steps
-------------

#. **Bump the version.** Edit ``[project].version`` in :file:`pyproject.toml`
   and ``__version__`` in :file:`xpcsjax/__init__.py` to the new ``X.Y.Z``.

#. **Update the changelog.** Move the ``[Unreleased]`` notes into a new dated
   ``[X.Y.Z]`` section in :file:`CHANGELOG.md`, and mirror the milestone in
   :file:`docs/source/changelog.rst`.

#. **Verify.** Run the full gate and confirm the docs still build clean:

   .. code-block:: shell

      make verify            # lint + advisory mypy + smoke (-x -n auto)
      make test-all-parallel # full suite incl. heavy-serial nodes
      uv run sphinx-build -W -b html docs/source docs/_build/html

#. **Confirm consistency.** A quick sanity check that the resolved version
   agrees with the tag you are about to cut:

   .. code-block:: shell

      uv run python -c "import importlib.metadata as m; print(m.version('xpcsjax'))"
      uv run xpcsjax --version

#. **Commit and tag.** Commit the bump, then create an annotated tag and push
   both the branch and the tag:

   .. code-block:: shell

      git commit -am "release: bump version to X.Y.Z"
      git tag -a vX.Y.Z -m "xpcsjax vX.Y.Z"
      git push origin main
      git push origin vX.Y.Z

#. **Publish the GitHub Release.** Create a GitHub Release for tag ``vX.Y.Z``.
   Publishing it triggers :file:`release.yml`, which builds and uploads to PyPI.

The publishing workflow
-----------------------

:file:`.github/workflows/release.yml` runs on a published GitHub Release (with a
``workflow_dispatch`` fallback) as two jobs:

* **build** — runs with **no credentials**. It runs ``uv build`` (sdist +
  wheel), asserts the :file:`pyproject.toml` version equals the release tag
  (skipped on manual dispatch), runs ``twine check --strict``, and uploads the
  ``dist/`` artifact.
* **publish** — runs in the protected ``pypi`` environment with the minimal
  ``id-token: write`` permission. It downloads the artifact and uploads it with
  ``pypa/gh-action-pypi-publish`` using OIDC. No long-lived secret is involved.

All actions are pinned to commit SHAs (with a version comment), matching the
convention in :file:`.github/workflows/ci.yml`.

One-time PyPI setup
~~~~~~~~~~~~~~~~~~~~

Trusted Publishing requires a publisher registered on PyPI **before** the first
run. At https://pypi.org/manage/account/publishing/ add a pending publisher:

============================  ===============
Field                         Value
============================  ===============
PyPI Project Name             ``xpcsjax``
Owner                         ``imewei``
Repository name               ``xpcsjax``
Workflow name                 ``release.yml``
Environment name              ``pypi``
============================  ===============

Once the project exists on PyPI, this is replaced by a project-scoped publisher
under the project's *Publishing* settings. The ``pypi`` environment is also where
required reviewers or branch filters can gate a release.
