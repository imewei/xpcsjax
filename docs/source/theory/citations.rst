.. _theory_citations:

References and Citations
========================

This page collects full bibliographic information for all works cited in the
Theory section, with BibTeX entries for inclusion in LaTeX documents. The
``[Author Year]_`` style references used throughout the theory pages resolve
to the bibliography entries on this page.

Primary references
------------------

.. [He2024]
   He, H., Liang, H., Chu, M., Jiang, Z., de Pablo, J. J., Tirrell, M. V.,
   Narayanan, S., and Chen, W. "Transport coefficient approach for
   characterizing nonequilibrium dynamics in soft matter." *Proceedings of
   the National Academy of Sciences*, **121**\ (31), e2401162121 (2024).
   `doi:10.1073/pnas.2401162121
   <https://doi.org/10.1073/pnas.2401162121>`_.

   Introduces the transport-coefficient framework :math:`J(t)` and the
   two-time correlation function :math:`c_2(\mathbf{q}, t_1, t_2)` as the
   central quantities for XPCS analysis of non-stationary systems. Derives
   the homodyne scattering formula for laminar flow and the connection to
   the generalised Green--Kubo relation.

   .. code-block:: bibtex

      @article{He2024PNAS,
        author  = {He, Hongrui and Liang, Hao and Chu, Miaoqi and Jiang, Zhang and
                   de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh
                   and Chen, Wei},
        title   = {Transport coefficient approach for characterizing nonequilibrium
                   dynamics in soft matter},
        journal = {Proceedings of the National Academy of Sciences},
        year    = {2024},
        volume  = {121},
        number  = {31},
        pages   = {e2401162121},
        doi     = {10.1073/pnas.2401162121},
      }

.. [He2025]
   He, H., Liang, H., Chu, M., Jiang, Z., de Pablo, J. J., Tirrell, M. V.,
   Narayanan, S., and Chen, W. "Bridging microscopic dynamics and rheology
   in the yielding of charged colloidal suspensions." *Proceedings of the
   National Academy of Sciences*, **122**\ (42), e2514216122 (2025).
   `doi:10.1073/pnas.2514216122
   <https://doi.org/10.1073/pnas.2514216122>`_.

   Applies the :math:`J(t)` framework to the yielding transition in
   repulsive (Andrade creep) and attractive (shear banding) colloidal
   suspensions. Introduces the multi-component heterodyne formula and the
   non-Gaussian displacement analysis used by xpcsjax's
   :class:`~xpcsjax.core.HeterodyneModel`.

   .. code-block:: bibtex

      @article{He2025PNAS,
        author  = {He, Hongrui and Liang, Heyi and Chu, Miaoqi and Jiang, Zhang and
                   de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh
                   and Chen, Wei},
        title   = {Bridging microscopic dynamics and rheology in the yielding
                   of charged colloidal suspensions},
        journal = {Proceedings of the National Academy of Sciences},
        year    = {2025},
        volume  = {122},
        number  = {42},
        pages   = {e2514216122},
        doi     = {10.1073/pnas.2514216122},
      }

XPCS methodology
----------------

.. [Sutton2008]
   Sutton, M. "A review of X-ray intensity fluctuation spectroscopy."
   *Comptes Rendus Physique*, **9**\ (5--6), 657--667 (2008).
   `doi:10.1016/j.crhy.2007.04.008
   <https://doi.org/10.1016/j.crhy.2007.04.008>`_.

   Review of the XPCS technique, coherence requirements, and connection to
   dynamic light scattering. Contains the canonical derivation of the
   Siegert relation in the synchrotron context.

   .. code-block:: bibtex

      @article{Sutton2008,
        author  = {Sutton, Mark},
        title   = {A review of X-ray intensity fluctuation spectroscopy},
        journal = {Comptes Rendus Physique},
        year    = {2008},
        volume  = {9},
        pages   = {657--667},
        doi     = {10.1016/j.crhy.2007.04.008},
      }

.. [Lumma2000]
   Lumma, D., Lurio, L. B., Mochrie, S. G. J., and Sutton, M. "Area detector
   based photon correlation in the regime of short data batches: Data
   reduction for dynamic X-ray scattering." *Review of Scientific
   Instruments*, **71**\ (9), 3274--3289 (2000).
   `doi:10.1063/1.1287334 <https://doi.org/10.1063/1.1287334>`_.

   Introduces the two-time correlation matrix as a practical estimator for
   non-stationary XPCS, and defines the diagonal-averaging approximation
   that recovers the equilibrium :math:`g_2(q, \tau)`.

   .. code-block:: bibtex

      @article{Lumma2000,
        author  = {Lumma, D. and Lurio, L. B. and Mochrie, S. G. J. and Sutton, M.},
        title   = {Area detector based photon correlation in the regime
                   of short data batches},
        journal = {Review of Scientific Instruments},
        year    = {2000},
        volume  = {71},
        pages   = {3274--3289},
        doi     = {10.1063/1.1287334},
      }

.. [Duri2005]
   Duri, A., Bissig, H., Trappe, V., and Cipelletti, L.
   "Time-resolved-correlation: A new tool for studying temporally
   heterogeneous dynamics." *Journal of Physics: Condensed Matter*,
   **17**, S3455 (2005).
   `doi:10.1088/0953-8984/17/31/003
   <https://doi.org/10.1088/0953-8984/17/31/003>`_.

   Establishes the two-time correlation function as a diagnostic for
   temporally heterogeneous ("aging") dynamics.

   .. code-block:: bibtex

      @article{Duri2005,
        author  = {Duri, A. and Bissig, H. and Trappe, V. and Cipelletti, L.},
        title   = {Time-resolved-correlation: A new tool for studying temporally
                   heterogeneous dynamics},
        journal = {Journal of Physics: Condensed Matter},
        year    = {2005},
        volume  = {17},
        pages   = {S3455},
        doi     = {10.1088/0953-8984/17/31/003},
      }

Stochastic processes and fluctuation--dissipation
-------------------------------------------------

.. [UhlenbeckOrnstein1930]
   Uhlenbeck, G. E. and Ornstein, L. S. "On the theory of the Brownian
   motion." *Physical Review*, **36**\ (5), 823--841 (1930).
   `doi:10.1103/PhysRev.36.823
   <https://doi.org/10.1103/PhysRev.36.823>`_.

   Original derivation of the Ornstein--Uhlenbeck process.

   .. code-block:: bibtex

      @article{UhlenbeckOrnstein1930,
        author  = {Uhlenbeck, G. E. and Ornstein, L. S.},
        title   = {On the theory of the {B}rownian motion},
        journal = {Physical Review},
        year    = {1930},
        volume  = {36},
        pages   = {823--841},
        doi     = {10.1103/PhysRev.36.823},
      }

.. [Kubo1966]
   Kubo, R. "The fluctuation-dissipation theorem." *Reports on Progress in
   Physics*, **29**\ (1), 255--284 (1966).
   `doi:10.1088/0034-4885/29/1/306
   <https://doi.org/10.1088/0034-4885/29/1/306>`_.

   Establishes the Green--Kubo relation connecting transport coefficients
   to velocity autocorrelation functions; the foundation of the
   :math:`J(t)` definition in :doc:`transport_coefficient`.

   .. code-block:: bibtex

      @article{Kubo1966,
        author  = {Kubo, Ryogo},
        title   = {The fluctuation-dissipation theorem},
        journal = {Reports on Progress in Physics},
        year    = {1966},
        volume  = {29},
        pages   = {255--284},
        doi     = {10.1088/0034-4885/29/1/306},
      }

Optimisation
------------

.. [More1977]
   Moré, J. J. "The Levenberg-Marquardt algorithm: Implementation and
   theory." In *Numerical Analysis* (G. A. Watson, ed.), Lecture Notes in
   Mathematics, vol. 630, pp. 105--116. Springer, Berlin, 1977.
   `doi:10.1007/BFb0067700 <https://doi.org/10.1007/BFb0067700>`_.

   Canonical reference for the trust-region Levenberg--Marquardt algorithm
   used by the upstream NLSQ library that xpcsjax wraps.

   .. code-block:: bibtex

      @incollection{More1977,
        author    = {Mor{\'e}, J. J.},
        title     = {The {L}evenberg-{M}arquardt algorithm: Implementation
                     and theory},
        booktitle = {Numerical Analysis},
        editor    = {Watson, G. A.},
        series    = {Lecture Notes in Mathematics},
        volume    = {630},
        pages     = {105--116},
        publisher = {Springer},
        year      = {1977},
        doi       = {10.1007/BFb0067700},
      }

.. [Hansen2016]
   Hansen, N. "The CMA evolution strategy: A tutorial." *arXiv:1604.00772*
   (2016). `arXiv:1604.00772 <https://arxiv.org/abs/1604.00772>`_.

   Tutorial on CMA-ES, the global optimiser used by the CMA-ES escape path
   in xpcsjax when the local NLSQ trust-region step stalls in a
   degenerate region.

   .. code-block:: bibtex

      @article{Hansen2016CMAes,
        author  = {Hansen, Nikolaus},
        title   = {The {CMA} evolution strategy: A tutorial},
        journal = {arXiv preprint arXiv:1604.00772},
        year    = {2016},
      }

JAX and scientific stack
------------------------

.. [Bradbury2018]
   Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C.,
   Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne,
   S., and Zhang, Q. "JAX: composable transformations of Python+NumPy
   programs." 2018. https://github.com/jax-ml/jax.

   The JAX library providing JIT compilation, automatic differentiation,
   and ``vmap`` used throughout :mod:`xpcsjax.core` and
   :mod:`xpcsjax.optimization.nlsq`.

   .. code-block:: bibtex

      @software{jax2018github,
        author  = {Bradbury, James and Frostig, Roy and Hawkins, Peter and
                   Johnson, Matthew James and Leary, Chris and Maclaurin, Dougal and
                   Necula, George and Paszke, Adam and VanderPlas, Jake and
                   Wanderman-Milne, Skye and Zhang, Qiao},
        title   = {{JAX}: composable transformations of {Python+NumPy} programs},
        url     = {https://github.com/jax-ml/jax},
        year    = {2018},
      }

Rheology
--------

.. [Andrade1910]
   Andrade, E. N. da C. "On the viscous flow in metals, and allied
   phenomena." *Proceedings of the Royal Society A*, **84**, 1--12 (1910).
   `doi:10.1098/rspa.1910.0050
   <https://doi.org/10.1098/rspa.1910.0050>`_.

   Original characterisation of the :math:`\gamma \sim t^{1/3}` creep law
   that appears in repulsive colloidal suspensions under constant stress
   and motivates the power-law shear-rate parameterisation in
   :doc:`homodyne_model`.

   .. code-block:: bibtex

      @article{Andrade1910,
        author  = {Andrade, E. N. da~C.},
        title   = {On the viscous flow in metals, and allied phenomena},
        journal = {Proceedings of the Royal Society A},
        year    = {1910},
        volume  = {84},
        pages   = {1--12},
        doi     = {10.1098/rspa.1910.0050},
      }

Citing xpcsjax
--------------

If you use xpcsjax in your research, please cite the primary PNAS papers
[He2024]_ and [He2025]_. You may additionally cite the software package:

.. code-block:: bibtex

   @software{xpcsjax,
     title       = {xpcsjax: JAX-native NLSQ analysis of XPCS two-time correlations},
     author      = {Chen, Wei and He, Hongrui},
     year        = {2025},
     url         = {https://github.com/imewei/xpcsjax},
     institution = {Argonne National Laboratory},
   }

**Example citation text** for a Methods section:

   "XPCS data were analysed using the transport-coefficient approach
   [He2024]_ as implemented in the xpcsjax software package. The analysis
   used the laminar-flow mode with quantile-based per-angle scaling and the
   five-layer anti-degeneracy defence to prevent the parameter absorption
   degeneracy that arises when fitting many azimuthal sectors jointly."

Further reading
---------------

Additional references that informed the theoretical framework and
computational design of xpcsjax: [Lumma2000]_, [Duri2005]_,
[More1977]_, [Hansen2016]_, and [Bradbury2018]_.

Acknowledgements
----------------

If you use xpcsjax in your research, please consider acknowledging:

* U.S. Department of Energy, Office of Science, Basic Energy Sciences;
* Advanced Photon Source User Facility at Argonne National Laboratory.

Contact
-------

* **Principal Investigator:** Wei Chen (wchen@anl.gov), Argonne National
  Laboratory.
* **Issues and support:** `GitHub Issues
  <https://github.com/imewei/xpcsjax/issues>`_.
