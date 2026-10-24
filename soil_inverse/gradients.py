"""Gradient calculators for the soil thermal inverse problem."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .forward import HeatGrid, forward_temperature
from .objective import TikhonovRegularization, evaluate_cost


def finite_difference_gradient(func, params, eps: float = 1.0e-4, min_scale: float = 1.0e-6):
    """Centered finite-difference gradient for scalar or vector parameters."""
    p = np.atleast_1d(np.asarray(params, dtype=float))
    grad = np.zeros_like(p)
    for i in range(len(p)):
        step = eps * max(min_scale, abs(p[i]))
        p_plus = p.copy()
        p_minus = p.copy()
        p_plus[i] += step
        p_minus[i] -= step
        grad[i] = (func(p_plus) - func(p_minus)) / (2.0 * step)
    return grad


def adjoint_gradient_scalar_alpha(
    alpha: float,
    obs_window: np.ndarray,
    grid: HeatGrid,
    space_scheme: str = "central",
):
    """Return ``dJ/dalpha`` for the L2 term using the refined implicit sensitivity.

    The YB implementation computes the exact tangent-linear/discrete-adjoint
    scalar gradient inside :meth:`HeatGrid.simulate`.  ``lam`` is retained only
    as a compatibility placeholder for older notebooks that unpack three values.
    """
    if space_scheme != "central":
        raise ValueError(
            f"space_scheme={space_scheme!r} is not supported by the refined implicit solver; "
            "use 'central'."
        )
    pred = forward_temperature(alpha, obs_window, grid, space_scheme=space_scheme)
    _, grad = evaluate_cost(grid, alpha, obs_window, reg=None, with_grad=True)
    lam = np.zeros_like(pred)
    return float(grad), pred, lam


def total_gradient_adjoint(
    alpha: float,
    obs_window: np.ndarray,
    grid: HeatGrid,
    reg: Optional[TikhonovRegularization] = None,
    space_scheme: str = "central",
):
    """Adjoint data gradient plus optional regularization gradient."""
    if space_scheme != "central":
        raise ValueError(
            f"space_scheme={space_scheme!r} is not supported by the refined implicit solver; "
            "use 'central'."
        )
    cost, grad = evaluate_cost(grid, alpha, obs_window, reg=reg, with_grad=True)
    pred = forward_temperature(alpha, obs_window, grid, space_scheme=space_scheme)
    lam = np.zeros_like(pred)
    return float(grad), pred, lam
