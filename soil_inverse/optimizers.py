"""Scalar ``alpha`` estimators for a single window.

The windowed misfit ``J(alpha)`` is **not globally convex** -- for some windows it
has a spurious local minimum pinned at the lower ``alpha`` bound in addition to
the true (lower-cost) interior minimum.  A naive local solver started from a
fixed ``alpha`` can therefore return an ``alpha`` that is off by orders of
magnitude.  Both solvers below guard against this with the same two-stage
strategy:

1. a cheap **coarse scan** in ``log(alpha)`` to bracket the *global* basin, then
2. a **local refinement** inside that basin.

They differ only in the refinement: :func:`solve_l2_bounded` uses derivative-free
Brent; :func:`solve_adjoint_gradient_descent` uses gradient descent driven by the
exact tangent-linear/adjoint gradient.  Everything runs in ``log(alpha)`` space so
the optimisation is well conditioned despite ``alpha ~ 1e-6``.

Each solver returns ``{"alpha", "cost", "nit", "nfev", ...}``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .forward import HeatGrid, forward_temperature
from .objective import TikhonovRegularization, evaluate_cost


def _check_scheme(space_scheme: str) -> None:
    if space_scheme != "central":
        raise ValueError(
            f"space_scheme={space_scheme!r} is not supported; the diffusion "
            "operator uses a central second difference ('central')."
        )


def _coarse_scan(
    cost_of_logalpha: Callable[[float], float],
    log_lo: float,
    log_hi: float,
    n_scan: int,
) -> Tuple[float, float, float, float]:
    """Evaluate the cost on a log-uniform grid and bracket the global minimum.

    Returns ``(m_best, cost_best, bracket_lo, bracket_hi)`` where the bracket is
    the neighbouring grid points around the best sample (clamped to the bounds).
    """
    grid_m = np.linspace(log_lo, log_hi, n_scan)
    costs = np.fromiter((cost_of_logalpha(m) for m in grid_m), dtype=float, count=n_scan)
    j = int(np.argmin(costs))
    lo_b = grid_m[max(0, j - 1)]
    hi_b = grid_m[min(n_scan - 1, j + 1)]
    return float(grid_m[j]), float(costs[j]), float(lo_b), float(hi_b)


def solve_l2_bounded(
    temps: np.ndarray,
    grid: HeatGrid,
    *,
    bounds: Tuple[float, float] = (1.0e-8, 1.0e-5),
    reg: Optional[TikhonovRegularization] = None,
    space_scheme: str = "central",
    n_scan: int = 16,
) -> Dict[str, object]:
    """Globally-robust bounded minimisation of the windowed cost over ``alpha``.

    A coarse log-scan brackets the global basin; Brent then refines inside it.
    """
    _check_scheme(space_scheme)
    lo, hi = float(bounds[0]), float(bounds[1])
    log_lo, log_hi = math.log(lo), math.log(hi)

    evals = {"n": 0}

    def cost(log_alpha: float) -> float:
        evals["n"] += 1
        c, _ = evaluate_cost(grid, math.exp(log_alpha), temps, reg)
        return c

    m_best, c_best, lo_b, hi_b = _coarse_scan(cost, log_lo, log_hi, n_scan)
    res = minimize_scalar(cost, bounds=(lo_b, hi_b), method="bounded")

    if res.fun <= c_best:
        m_star, c_star = float(res.x), float(res.fun)
    else:  # refinement never does worse than the scan minimum
        m_star, c_star = m_best, c_best

    return {
        "alpha": float(math.exp(m_star)),
        "cost": c_star,
        "nit": int(getattr(res, "nit", 0)),
        "nfev": evals["n"],
        "success": bool(getattr(res, "success", True)),
        "optimizer": "l2_bounded",
        "pred": forward_temperature(float(math.exp(m_star)), temps, grid),
    }


def solve_adjoint_gradient_descent(
    temps: np.ndarray,
    grid: HeatGrid,
    *,
    alpha_initial: float = 1.0e-6,
    bounds: Tuple[float, float] = (1.0e-8, 1.0e-5),
    reg: Optional[TikhonovRegularization] = None,
    space_scheme: str = "central",
    n_scan: int = 16,
    gtol: float = 1e-8,
) -> Dict[str, object]:
    """Globally-robust ``alpha`` solve driven by the exact adjoint gradient.

    A coarse log-scan selects the global basin (the better of the scan minimum
    and ``alpha_initial``).  A quasi-Newton descent (L-BFGS-B) then refines inside
    that basin using the exact tangent-linear/adjoint gradient -- which converges
    in a handful of curvature-aware steps where plain steepest descent would
    zig-zag for dozens.
    """
    _check_scheme(space_scheme)
    lo, hi = float(bounds[0]), float(bounds[1])
    log_lo, log_hi = math.log(lo), math.log(hi)

    evals = {"n": 0, "grad": 0}
    history = []

    def cost_only(log_alpha: float) -> float:
        evals["n"] += 1
        c, _ = evaluate_cost(grid, math.exp(log_alpha), temps, reg)
        return c

    def cost_grad(m_vec: np.ndarray):
        evals["n"] += 1
        evals["grad"] += 1
        alpha = math.exp(m_vec[0])
        c, g_alpha = evaluate_cost(grid, alpha, temps, reg, with_grad=True)
        history.append((evals["grad"], alpha, c, g_alpha))
        # chain rule: dJ/dm = dJ/d(alpha) * alpha
        return c, np.array([g_alpha * alpha])

    # Global bracketing: start the local descent from whichever of the scan
    # minimum and the requested initial guess has the lower cost.
    m_scan, c_scan, _, _ = _coarse_scan(cost_only, log_lo, log_hi, n_scan)
    m_init = min(max(math.log(float(alpha_initial)), log_lo), log_hi)
    c_init = cost_only(m_init)
    m_start = m_scan if c_scan <= c_init else m_init

    res = minimize(
        cost_grad,
        x0=np.array([m_start]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(log_lo, log_hi)],
        options={"gtol": gtol, "ftol": 1e-12, "maxiter": 60},
    )

    # Never report something worse than the coarse-scan minimum.
    if res.fun <= min(c_scan, c_init):
        m_star, c_star = float(res.x[0]), float(res.fun)
    else:
        m_star, c_star = m_scan, c_scan

    return {
        "alpha": float(math.exp(m_star)),
        "cost": c_star,
        "nit": int(res.nit),
        "nfev": evals["n"],
        "n_grad_eval": evals["grad"],
        "grad_logspace": float(np.ravel(res.jac)[0]) if res.jac is not None else float("nan"),
        "success": bool(res.success),
        "optimizer": "adjoint_gradient_descent",
        "pred": forward_temperature(float(math.exp(m_star)), temps, grid),
        "history": np.array(history, dtype=float),
    }
