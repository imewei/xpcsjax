xpcsjax.service
===============

The headless **core-service layer**: the argparse-free, Qt-free orchestration
seam shared by the :doc:`command-line interface </user_guide/cli>` and the
:doc:`GUI workbench </user_guide/gui>` worker. Each function here is the pure
core of a CLI sub-command, callable without parsing arguments or touching Qt.

.. currentmodule:: xpcsjax.service

Import discipline
-----------------

The modules split along the GUI's JAX-free boundary (see
:doc:`/user_guide/gui`):

* **JAX-free** — importable in the GUI process itself: :mod:`~xpcsjax.service.events`
  (the streamed fit-progress schema) and :mod:`~xpcsjax.service.config`
  (config loading + validation + templates).
* **Worker-side** — import JAX and must only be loaded inside the spawn worker
  or the CLI process: :mod:`~xpcsjax.service.data`, :mod:`~xpcsjax.service.fit`,
  and :mod:`~xpcsjax.service.plots`.

Config service (JAX-free)
-------------------------

.. automodule:: xpcsjax.service.config
   :members:

Events (JAX-free)
-----------------

The structured progress events a worker streams back to the GUI. Dependency-light
by design — standard library only — so the GUI process can import the schema
without pulling JAX in.

.. automodule:: xpcsjax.service.events
   :members:

Data service (worker-side)
--------------------------

.. automodule:: xpcsjax.service.data
   :members:

Fit service (worker-side)
-------------------------

.. automodule:: xpcsjax.service.fit
   :members:

Plot service (worker-side)
--------------------------

.. automodule:: xpcsjax.service.plots
   :members:

Result persistence
------------------

.. automodule:: xpcsjax.service.persist
   :members:
