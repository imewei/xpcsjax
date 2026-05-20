Data loading
============

.. currentmodule:: xpcsjax


All xpcsjax analyses begin by calling :func:`xpcsjax.data.xpcs_loader.load_xpcs_data`.
This is a thin convenience wrapper over the :class:`xpcsjax.data.xpcs_loader.XPCSDataLoader`
class in :mod:`xpcsjax.data.xpcs_loader`; for v0.1 you should not need
to instantiate the loader yourself.

The function signature
----------------------

.. code-block:: python

   xpcsjax.load_xpcs_data(
       config_path: str | dict | None = None,
       config_dict: dict | None = None,
   ) -> dict[str, Any]

Both arguments are positionally compatible with the legacy upstream
``homodyne`` call style — passing a configuration ``dict`` as
``config_path`` is silently re-routed to ``config_dict`` for backward
compatibility. Passing both forms raises ``ValueError``.

Three invocation patterns are supported.

From a YAML or JSON file
~~~~~~~~~~~~~~~~~~~~~~~~

The most common pattern. The file format is auto-detected by extension.

.. code-block:: python

   import xpcsjax

   data = xpcsjax.load_xpcs_data("xpcs_config.yaml")

From a configuration ``dict`` (positional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Useful when the configuration was constructed in memory:

.. code-block:: python

   config = {
       "analysis_mode": "static_isotropic",
       "experimental_data": {"data_file_name": "experiment.h5"},
       # ...
   }
   data = xpcsjax.load_xpcs_data(config)

From a configuration ``dict`` (keyword)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The explicit form, recommended when the call is part of a longer
pipeline:

.. code-block:: python

   data = xpcsjax.load_xpcs_data(config_dict=config)

The returned dictionary
-----------------------

Regardless of how the loader is invoked, the returned object is a
plain Python ``dict`` with exactly five keys:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Contents
   * - ``wavevector_q_list``
     - Per-angle scattering wave-vector magnitudes :math:`q` in
       inverse Ångström, as a 1-D array of length :math:`n_\phi`.
   * - ``phi_angles_list``
     - Azimuthal angles :math:`\phi` (degrees) corresponding to each
       ``q`` entry; same length as ``wavevector_q_list``.
   * - ``t1``
     - First time axis (frame indices or seconds depending on
       configuration); the row index of the two-time correlation
       function.
   * - ``t2``
     - Second time axis; the column index of the two-time
       correlation function.
   * - ``c2_exp``
     - Experimental two-time correlation tensor with shape
       ``(n_phi, n_t1, n_t2)``. When JAX is available the array is a
       ``jax.numpy.ndarray`` with ``dtype=float64``; otherwise a NumPy
       array of the same shape and dtype.

The contract is shape-strict. Any analysis downstream of the loader
assumes ``c2_exp.shape[0] == len(phi_angles_list) == len(wavevector_q_list)``
and that the time axes are monotone increasing.

HDF5 expectations
-----------------

The loader transparently consumes two HDF5 layouts that arise in
practice at synchrotron beamlines:

* **APS old format.** Per-angle datasets are indexed by phi group.
* **APS-U new format.** Datasets are stored under the standardized
  NeXus-flavoured layout produced by current APS-U beamlines.

For homodyne datasets the canonical correlation array is stored under
the dataset path documented by the originating beamline. The loader
applies the configured frame range
(``analyzer_parameters.temporal.start_frame`` and ``end_frame``) when
slicing into the array, so the returned ``c2_exp`` already reflects the
analysis window — there is no separate cropping step downstream.

For heterodyne / two-component datasets, the fit-side code accepts
either ``c2`` or ``g2`` as the experimental correlation key, and either
``phi`` or ``phi_angles`` for the angle list. The loader resolves
these aliases before returning, so the five-key dictionary layout is
identical for homodyne and heterodyne data.

Phi-angle filtering
-------------------

XPCS datasets often contain phi angles that contribute little signal
(near the beam-stop, behind absorbers, in noisy regions of the
detector). xpcsjax ships two related utilities to drop those angles
before fitting:

:mod:`xpcsjax.data.phi_filtering`
    Functions to construct isotropic or anisotropic phi-range tables
    and to map a phi list through them, returning the indices of the
    surviving angles. Used at data-prep time to subset the loaded
    arrays.

:mod:`xpcsjax.data.angle_filtering`
    Lower-level normalisation utilities (e.g. mapping angles into the
    canonical symmetric range, point-in-range queries). Used by the
    higher-level phi-filtering layer and inside the optimisation
    pipeline when the configured analysis is anisotropic and only a
    subset of phi-angle bins drive the fit.

Concretely, when the configuration declares an anisotropic analysis
mode (see :doc:`/user_guide/analysis_modes`), the optimisation
pipeline consults the configured ``target_angle_ranges`` to subset the
loaded data; you do not need to apply the filter by hand. The filter
modules are exposed so that you can preview which angles will survive
the cut.

.. note::

   Phi-angle filtering only removes angles; it does not interpolate or
   rebin. The ``c2_exp`` tensor returned by the loader is full-fidelity
   experimental data, even if downstream code chooses to fit only a
   subset of it.

Validation hooks
----------------

The loader runs a series of validation checks at I/O boundaries:

* Shape and dtype consistency across ``c2_exp``, ``t1``, ``t2``, and
  the phi/q lists.
* Detection of NaN or non-finite entries in the correlation tensor.
* Monotonicity of the time axes.
* Plausibility of phi-angle and q-vector ranges.

A validation failure raises one of the loader-specific exceptions
(:class:`xpcsjax.data.xpcs_loader.XPCSDataFormatError`, :class:`xpcsjax.data.xpcs_loader.XPCSConfigurationError`,
:class:`xpcsjax.data.xpcs_loader.XPCSDependencyError`), all of which inherit from the standard
``Exception`` base. Catch the specific class when you want to
distinguish a malformed HDF5 file from a misconfigured YAML schema.

A complete example
------------------

.. code-block:: python

   import xpcsjax

   data = xpcsjax.load_xpcs_data("xpcs_config.yaml")

   print(f"phi angles: {len(data['phi_angles_list'])}")
   print(f"c2 shape:   {data['c2_exp'].shape}")
   print(f"q range:    {data['wavevector_q_list'].min():.4f} "
         f"to {data['wavevector_q_list'].max():.4f} 1/A")

   # Hand the data dict directly to fit_nlsq
   result = xpcsjax.fit_nlsq(data, "xpcs_config.yaml")

The next step is to understand the YAML schema that ``"xpcs_config.yaml"``
must conform to — covered in :doc:`/user_guide/configuration`.
