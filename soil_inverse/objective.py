"""Misfit functional and Tikhonov regularisation for the alpha inverse problem.

The data misfit is the L2 distance between the modelled and observed interior
temperatures over a window::

    J_data(alpha) = 1/2 * sum_{steps, interior sensors} (T_pred - T_obs)^2

An optional Tikhonov term penalises departure from a prior diffusivity::

    J_reg(alpha) = 1/2 * beta * (alpha - prior)^2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .forward import HeatGrid
from .forward import active_node_indices, forward_temperature


@dataclass(frozen=True)
class TikhonovRegularization:
    """Quadratic penalty pulling ``alpha`` toward ``prior`` with weight ``beta``."""

    beta: float
    prior: float

    def value(self, alpha: float) -> float:
        d = alpha - self.prior
        return 0.5 * self.beta * d * d

    def gradient(self, alpha: float) -> float:
        return self.beta * (alpha - self.prior)


def evaluate_cost(
    grid: HeatGrid,
    alpha: float,
    temps: np.ndarray,
    reg: Optional[TikhonovRegularization] = None,
    *,
    with_grad: bool = False,
) -> Tuple[float, Optional[float]]:
    """Return ``(cost, grad)`` for a single window.

    ``grad`` is ``None`` when ``with_grad`` is ``False``.  The data-misfit
    gradient uses the exact tangent-linear sensitivity from
    :meth:`HeatGrid.simulate`.
    """
    temps = np.asarray(temps, dtype=float)
    obs = temps[:, list(grid.obs_columns)]

    sim = grid.simulate(alpha, temps, with_sensitivity=with_grad)
    residual = sim["prediction"] - obs
    cost = 0.5 * float(np.sum(residual * residual))
    if reg is not None:
        cost += reg.value(alpha)

    if not with_grad:
        return cost, None

    grad = float(np.sum(residual * sim["sensitivity"]))
    if reg is not None:
        grad += reg.gradient(alpha)
    return cost, grad


def interior_residual(pred: np.ndarray, obs: np.ndarray, space_scheme: str = "central") -> np.ndarray:
    """Compatibility residual over interior sensor columns."""
    active = active_node_indices(pred.shape[1], space_scheme)
    return np.asarray(pred)[:, active] - np.asarray(obs)[:, active]


def l2_cost(
    alpha: float,
    obs_window: np.ndarray,
    grid: HeatGrid,
    space_scheme: str = "central",
) -> float:
    """Compatibility wrapper for the L2 data-misfit cost."""
    if space_scheme != "central":
        raise ValueError(
            f"space_scheme={space_scheme!r} is not supported by the refined implicit solver; "
            "use 'central'."
        )
    cost, _ = evaluate_cost(grid, alpha, obs_window, reg=None, with_grad=False)
    return float(cost)


def objective(
    alpha: float,
    obs_window: np.ndarray,
    grid: HeatGrid,
    reg: Optional[TikhonovRegularization] = None,
    space_scheme: str = "central",
) -> float:
    """Compatibility wrapper for the total inverse objective."""
    if space_scheme != "central":
        raise ValueError(
            f"space_scheme={space_scheme!r} is not supported by the refined implicit solver; "
            "use 'central'."
        )
    cost, _ = evaluate_cost(grid, alpha, obs_window, reg=reg, with_grad=False)
    return float(cost)


def rmse_for_solution(
    alpha: float,
    obs_window: np.ndarray,
    grid: HeatGrid,
    space_scheme: str = "central",
) -> float:
    """Interior-sensor RMSE for a fitted alpha."""
    pred = forward_temperature(alpha, obs_window, grid, space_scheme=space_scheme)
    res = interior_residual(pred, obs_window, space_scheme=space_scheme)
    return float(np.sqrt(np.mean(res * res)))
