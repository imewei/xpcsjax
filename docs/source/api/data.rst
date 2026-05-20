xpcsjax.data
============

Data loading, validation, and angle filtering. The public entry point is
:func:`xpcsjax.data.xpcs_loader.load_xpcs_data` (re-exported on the top-level :mod:`xpcsjax`
namespace).

.. currentmodule:: xpcsjax.data

Loader entry points
-------------------

.. autofunction:: xpcsjax.data.xpcs_loader.load_xpcs_data

.. autofunction:: xpcsjax.data.xpcs_loader.load_xpcs_config

Loader class
------------

.. autoclass:: xpcsjax.data.xpcs_loader.XPCSDataLoader



Exceptions
----------

.. autoexception:: xpcsjax.data.xpcs_loader.XPCSDataFormatError


.. autoexception:: xpcsjax.data.xpcs_loader.XPCSDependencyError


.. autoexception:: xpcsjax.data.xpcs_loader.XPCSConfigurationError


Returned dictionary contract
----------------------------

:func:`xpcsjax.data.xpcs_loader.load_xpcs_data` returns ``dict[str, Any]`` with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Key
     - Type
     - Meaning
   * - ``wavevector_q_list``
     - ``np.ndarray``
     - 1-D array of detector q-values (Å⁻¹).
   * - ``phi_angles_list``
     - ``np.ndarray``
     - 1-D array of detector φ angles (degrees).
   * - ``t1``, ``t2``
     - ``np.ndarray``
     - Time arrays for the correlation matrices, length ``n_time``.
   * - ``c2_exp``
     - ``np.ndarray``
     - Experimental ``g2`` correlation. Shape ``(n_phi, n_time, n_time)`` for
       multi-angle datasets; ``(n_time, n_time)`` for a single angle.

The xpcsjax loader is the homodyne-style format. The heterodyne fit path
also accepts a dict with ``c2`` / ``phi`` (heterodyne-cache layout) — see
the dispatch logic in :func:`xpcsjax.optimization.nlsq.fit_nlsq`.

Related modules
---------------

The data subpackage also contains internal validators, an angle-filtering
utility, and a memory-aware preprocessing engine. These are not part of the
public API; consult the source if you need to extend them.
