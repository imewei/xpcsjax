"""Typed container for loaded XPCS experimental data.

``XpcsDataset`` is the named schema exchanged between :func:`load_xpcs_data`
and :func:`~xpcsjax.optimization.nlsq.fit_nlsq`. It subclasses ``dict`` so that
every existing key-indexed access (``data["c2_exp"]``, ``data["phi_angles_list"]``)
keeps working unchanged — the type is purely additive — while giving callers
typed, alias-resolving accessors (:attr:`c2`, :attr:`phi`, :attr:`t1`,
:attr:`t2`) and a documented set of canonical keys.

The loader and the fitter historically accepted several spellings of the same
field (``c2`` vs ``c2_exp``; ``phi`` / ``phi_angles`` / ``phi_angles_list``).
The accessors resolve those aliases in one place so consumers no longer guess.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Canonical → alias key spellings accepted at the load/fit boundary, in
# resolution order (first present wins). Mirrors the alias handling in
# ``fit_nlsq``.
_C2_KEYS = ("c2_exp", "c2")
_PHI_KEYS = ("phi_angles_list", "phi_angles", "phi")
_T1_KEYS = ("t1",)
_T2_KEYS = ("t2",)


class XpcsDataset(dict):
    """A loaded XPCS dataset: a ``dict`` with typed, alias-resolving accessors."""

    def _first(self, keys: tuple[str, ...], what: str) -> NDArray[Any]:
        """Return the first present alias key as a NumPy array.

        Parameters
        ----------
        keys : tuple of str
            Candidate key spellings in resolution order; the first one present
            in the dataset wins.
        what : str
            Human-readable name of the field, used only in the error message.

        Returns
        -------
        numpy.ndarray
            The value stored under the first matching key, coerced via
            :func:`numpy.asarray`.

        Raises
        ------
        KeyError
            If none of ``keys`` are present in the dataset.
        """
        for key in keys:
            if key in self:
                return np.asarray(self[key])
        raise KeyError(
            f"XpcsDataset is missing the {what} array (looked for any of "
            f"{keys!r}; present keys: {sorted(self)!r})."
        )

    @property
    def c2(self) -> NDArray[Any]:
        """Experimental correlation data (``c2_exp`` / ``c2``)."""
        return self._first(_C2_KEYS, "correlation")

    @property
    def phi(self) -> NDArray[Any]:
        """Phi angles (``phi_angles_list`` / ``phi_angles`` / ``phi``)."""
        return self._first(_PHI_KEYS, "phi-angle")

    @property
    def t1(self) -> NDArray[Any]:
        """First time axis."""
        return self._first(_T1_KEYS, "t1")

    @property
    def t2(self) -> NDArray[Any]:
        """Second time axis."""
        return self._first(_T2_KEYS, "t2")
