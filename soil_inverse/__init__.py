"""Soil thermal inverse problem: estimate effective thermal diffusivity ``alpha``.

A small, fast 1-D heat-diffusion inverse framework:

* :mod:`soil_inverse.data` -- load and resample the raw sensor CSV.
* :mod:`soil_inverse.forward` -- :class:`HeatGrid`, an arbitrary-resolution
  implicit (Crank-Nicolson) finite-difference forward model + exact sensitivity.
* :mod:`soil_inverse.objective` -- L2 misfit and Tikhonov regularisation.
* :mod:`soil_inverse.optimizers` -- bounded and gradient-descent ``alpha`` solvers.
* :mod:`soil_inverse.windowing` -- sliding-window index generation.
"""

from __future__ import annotations

from .data import load_and_resample_soil_csv, load_temperature_csv, median_time_step_seconds, parse_time
from .forward import HeatGrid, active_node_indices, forward_temperature
from .gradients import adjoint_gradient_scalar_alpha, finite_difference_gradient
from .objective import (
    TikhonovRegularization,
    evaluate_cost,
    interior_residual,
    l2_cost,
    objective,
    rmse_for_solution,
)
from .optimizers import solve_adjoint_gradient_descent, solve_l2_bounded
from .timevarying import (
    ControlParameterization,
    TimeDependentResult,
    build_parameterization,
    solve_time_dependent_alpha,
)
from .windowing import make_window_indices, select_windows_by_time

__all__ = [
    "load_and_resample_soil_csv",
    "load_temperature_csv",
    "median_time_step_seconds",
    "parse_time",
    "HeatGrid",
    "active_node_indices",
    "forward_temperature",
    "TikhonovRegularization",
    "evaluate_cost",
    "interior_residual",
    "l2_cost",
    "objective",
    "rmse_for_solution",
    "adjoint_gradient_scalar_alpha",
    "finite_difference_gradient",
    "solve_adjoint_gradient_descent",
    "solve_l2_bounded",
    "make_window_indices",
    "select_windows_by_time",
    # time- (and layer-) dependent inversion
    "solve_time_dependent_alpha",
    "build_parameterization",
    "ControlParameterization",
    "TimeDependentResult",
]
