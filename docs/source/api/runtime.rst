xpcsjax.runtime
===============

Runtime support utilities: installation validation and the shell integration
assets (completion + XLA activation) that the post-install helper deploys.

.. currentmodule:: xpcsjax.runtime

Package surface
---------------

.. automodule:: xpcsjax.runtime
   :members:

System validator
----------------

``xpcsjax-validate`` (alias ``xj-validate``) checks the installation:
environment detection, dependency verification, JAX configuration, and
template / public-API integrity. It is **NLSQ-only by design** — it
deliberately does *not* probe for NumPyro / BlackJAX / ArviZ or any
Bayesian / MCMC dependency.

.. autofunction:: xpcsjax.runtime.utils.system_validator.run_validation

.. autofunction:: xpcsjax.runtime.utils.system_validator.main

Shell integration
-----------------

:mod:`xpcsjax.runtime.shell` ships the static assets installed into a user's
shell by ``xpcsjax-post-install``:

* ``completion.sh`` — bash/zsh completion definitions for the console scripts.
* ``activation/xla_config.bash`` and ``activation/xla_config.fish`` — sourced
  on virtual-environment activation to export the tuned ``XLA_FLAGS`` for the
  CPU backend.

These are data files rather than importable Python APIs; see
:doc:`/user_guide/cli` for how they are installed and removed.
