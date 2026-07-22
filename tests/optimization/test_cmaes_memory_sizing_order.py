"""Regression: CMA-ES auto-memory sizing must use the FINAL (post-adaptive-
scaling) popsize, not the pre-scaling value.

Root cause (P2, basin-risk): ``fit()`` used to call ``self._configure_memory``
BEFORE the ``if scale_ratio > 1e3:`` block that multiplies ``popsize`` up
3x-20x (the typical laminar_flow case). The memory-sizing decision
(``pop_batch``/``data_chunk``) was derived from the pre-scaling popsize
(~9-11), so a small population that fits the memory budget produced
``(None, None)`` ("no batching needed") -- and that stale decision was then
unconditionally applied to the now-much-larger scaled popsize, silently
defeating ``memory_limit_gb``.

Fix: ``_configure_memory`` now accepts an explicit ``popsize`` override, and
``fit()`` calls it AFTER the scale-ratio block with ``cmaes_config.popsize``
(the final value).
"""

from __future__ import annotations

import inspect

from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapper, CMAESWrapperConfig


def test_configure_memory_respects_explicit_popsize_override():
    """At a memory budget where the small (pre-scaling) popsize fits without
    batching but the large (scaled) popsize does not, the explicit ``popsize``
    argument must be the one that decides the batch/chunk sizing."""
    wrapper = CMAESWrapper(CMAESWrapperConfig(auto_memory=True, memory_limit_gb=1.0))

    n_data = 200_000
    n_params = 7

    # Pre-scaling popsize (NLSQ default for n_params=7, ~9-11): fits in budget.
    pop_batch_small, _ = wrapper._configure_memory(n_data, n_params)
    assert pop_batch_small is None, (
        "test assumption broken: the small/default popsize no longer fits the "
        "1.0GB budget without batching -- adjust the fixture numbers"
    )

    # Scaled popsize (e.g. the 20x multiplier for scale_ratio > 1e6): needs
    # real batching -- must NOT silently inherit the small-popsize (None, None)
    # decision.
    scaled_popsize = 220
    pop_batch_scaled, _ = wrapper._configure_memory(n_data, n_params, popsize=scaled_popsize)
    assert pop_batch_scaled is not None, (
        "memory sizing for the scaled popsize must produce a real batch size, "
        "not silently reuse the pre-scaling 'no batching needed' decision"
    )
    assert pop_batch_scaled < scaled_popsize


def test_fit_calls_configure_memory_after_scale_ratio_block_with_final_popsize():
    """Wiring: ``_configure_memory`` must be called AFTER the adaptive
    scale_ratio > 1e3 block (which may rewrite ``cmaes_config.popsize``), and
    with that final value -- not before it."""
    src = inspect.getsource(CMAESWrapper.fit)

    scale_block_idx = src.index("if scale_ratio > 1e3:")
    # The last statement of the adaptive-scaling block, present only inside it.
    adaptive_end_idx = src.index("cmaes_config.max_generations = adaptive_gen")
    configure_memory_idx = src.index("self._configure_memory(")

    assert scale_block_idx < adaptive_end_idx < configure_memory_idx, (
        "_configure_memory must be called after the scale_ratio > 1e3 block "
        "finishes rewriting cmaes_config.popsize, not before it"
    )
    # Must pass the FINAL popsize explicitly, not silently re-derive the
    # pre-scaling value from self.config.popsize.
    call_site = src[configure_memory_idx : configure_memory_idx + 200]
    assert "popsize=cmaes_config.popsize" in call_site
