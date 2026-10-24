r"""1-D heat-diffusion forward model on an arbitrary-resolution vertical grid.

Physics
-------
We solve the 1-D heat equation for temperature ``T(z, t)`` with a single scalar
thermal diffusivity ``alpha``::

    dT/dt = alpha * d^2T/dz^2

over the depth interval spanned by the temperature sensors.  The shallowest and
deepest sensors act as **time-varying Dirichlet boundaries**; the interior of the
column is integrated by the solver and compared against the interior sensors.

Why this replaces the old "4 cells" scheme
-------------------------------------------
The previous implementation placed grid nodes *exactly* on the four sensors
(``dz = 0.25 m``, 2 interior unknowns) and marched with an **explicit** update.
That is both spatially crude and -- for any finer grid -- unstable, because an
explicit scheme requires ``alpha*dt/dz^2 <= 0.5``.

This module instead

* builds a **uniform refined grid** with an arbitrary number of cells
  (``cells_per_gap`` sub-cells between adjacent sensors), and
* marches with the **theta-method** (``theta = 0.5`` -> Crank-Nicolson), which is
  *unconditionally stable*, so ``dt`` is free of the explicit stability limit.

Performance
-----------
For a fixed ``alpha`` the implicit system matrix ``A`` is constant in time.  We
factor it **once per solve** into a dense propagator ``P = A^{-1} B`` plus two
boundary-response columns ``A^{-1} e_top`` / ``A^{-1} e_bot``.  Each time step is
then a single BLAS matrix-vector product (``u_{n+1} = P u_n + ...``) rather than a
Python-level tridiagonal sweep, which keeps the inner time loop almost entirely
in compiled code.

The exact gradient ``d(prediction)/d(alpha)`` is produced by a **tangent-linear**
(forward-sensitivity) march that reuses ``P`` and ``A^{-1}``.  For a single scalar
parameter this is mathematically identical to the discrete adjoint gradient, at
the same cost, but is simpler and allocation-light.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class HeatGrid:
    """Vertical finite-difference grid + implicit forward model for ``alpha``.

    Parameters
    ----------
    depths_m
        Sensor depths in metres, ascending (e.g. ``[0.25, 0.5, 0.75, 1.0]``).
        The first and last are treated as Dirichlet boundaries.
    dt_seconds
        Marching time step (typically the resampled data cadence).
    cells_per_gap
        Number of finite-difference cells inserted between each pair of adjacent
        sensors.  ``1`` reproduces the legacy node-on-sensor grid; the default of
        ``5`` gives a 5x finer column.  Total cells = ``cells_per_gap *
        (n_sensors - 1)``.
    theta
        Time-integration weight: ``0.5`` = Crank-Nicolson (2nd-order, default),
        ``1.0`` = backward Euler (1st-order, maximally damped).
    """

    def __init__(
        self,
        depths_m,
        dt_seconds: float,
        *,
        cells_per_gap: int = 5,
        theta: float = 0.5,
    ) -> None:
        depths = np.asarray(depths_m, dtype=float)
        if depths.ndim != 1 or depths.size < 2:
            raise ValueError("depths_m must be a 1-D array of at least 2 sensors")
        if not np.all(np.diff(depths) > 0):
            raise ValueError("depths_m must be strictly ascending")
        if cells_per_gap < 1:
            raise ValueError("cells_per_gap must be >= 1")
        if not (0.0 <= theta <= 1.0):
            raise ValueError("theta must lie in [0, 1]")

        self.depths_m = depths
        self.dt_seconds = float(dt_seconds)
        self.cells_per_gap = int(cells_per_gap)
        self.theta = float(theta)

        n_gaps = depths.size - 1
        self.n_cells = self.cells_per_gap * n_gaps
        z0, z_end = depths[0], depths[-1]
        self.dz_m = (z_end - z0) / self.n_cells
        self.nodes_m = z0 + self.dz_m * np.arange(self.n_cells + 1)

        # Map each sensor onto its nearest grid node.  For uniformly spaced
        # sensors this is exact; otherwise we fail loudly rather than silently
        # mis-place an observation.
        sensor_nodes = np.rint((depths - z0) / self.dz_m).astype(int)
        placement_err = np.max(np.abs(self.nodes_m[sensor_nodes] - depths))
        if placement_err > 0.51 * self.dz_m:
            raise ValueError(
                "sensors do not align with the refined grid; increase "
                "cells_per_gap or use uniformly spaced sensors"
            )
        self.sensor_node_index = sensor_nodes  # node index of each sensor

        # Column bookkeeping: temperature inputs are ordered as the sensors.
        self.bc_columns = (0, depths.size - 1)               # top / bottom sensors
        self.obs_columns = tuple(range(1, depths.size - 1))  # interior sensors
        # Interior unknowns are nodes 1 .. n_cells-1 (length n_cells-1).  An
        # interior sensor at node ``k`` lives at local index ``k - 1``.
        self.n_unknown = self.n_cells - 1
        if self.n_unknown < 1:
            raise ValueError("refined grid has no interior nodes")
        self._obs_local = np.array(
            [self.sensor_node_index[c] - 1 for c in self.obs_columns], dtype=int
        )

        # Constant interior Laplacian L0 (zero-boundary) as a dense matrix; it is
        # small (n_unknown^2) and lets us assemble the implicit operators with a
        # couple of BLAS calls per solve.
        n = self.n_unknown
        self._L0 = (
            np.diag(np.full(n, -2.0))
            + np.diag(np.ones(n - 1), 1)
            + np.diag(np.ones(n - 1), -1)
        )
        self._eye = np.eye(n)

    # ----------------------------------------------------------------- #
    #  Reporting helpers                                                  #
    # ----------------------------------------------------------------- #
    def r(self, alpha: float) -> float:
        """Diffusion number ``r = alpha * dt / dz^2`` (reported, not a CFL limit)."""
        return float(alpha) * self.dt_seconds / self.dz_m**2

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"HeatGrid(n_cells={self.n_cells}, dz_m={self.dz_m:.4f}, "
            f"dt_seconds={self.dt_seconds:g}, theta={self.theta:g})"
        )

    # ----------------------------------------------------------------- #
    #  Implicit operators for a given alpha (factored once per solve)     #
    # ----------------------------------------------------------------- #
    def _operators(self, alpha: float):
        """Return ``(P, a_inv, g_top, g_bot, kappa)`` for the theta-scheme.

        ``A = I - theta*kappa*L0``, ``B = I + (1-theta)*kappa*L0`` and
        ``P = A^{-1} B`` is the one-step propagator.  ``g_top``/``g_bot`` are the
        boundary-response columns ``A^{-1} e_0`` / ``A^{-1} e_{-1}``.
        """
        kappa = float(alpha) * self.dt_seconds / self.dz_m**2
        theta = self.theta
        a_mat = self._eye - (theta * kappa) * self._L0
        b_mat = self._eye + ((1.0 - theta) * kappa) * self._L0
        a_inv = np.linalg.inv(a_mat)
        prop = a_inv @ b_mat
        return prop, a_inv, a_inv[:, 0].copy(), a_inv[:, -1].copy(), kappa

    def _laplacian_full(self, u: np.ndarray, top: float, bot: float) -> np.ndarray:
        """Discrete Laplacian on interior nodes including the boundary values."""
        lap = self._L0 @ u
        lap[0] += top
        lap[-1] += bot
        return lap

    # ----------------------------------------------------------------- #
    #  Forward solve (+ optional tangent-linear sensitivity)              #
    # ----------------------------------------------------------------- #
    def simulate(
        self,
        alpha: float,
        temps: np.ndarray,
        *,
        with_sensitivity: bool = False,
    ) -> Dict[str, np.ndarray]:
        """March the column and return interior-sensor predictions.

        Parameters
        ----------
        alpha
            Thermal diffusivity (m^2/s).
        temps
            Observed sensor temperatures, shape ``(n_steps, n_sensors)``; columns
            follow ``depths_m``.  The first/last columns drive the Dirichlet
            boundaries; interior columns are the prediction targets.
        with_sensitivity
            If ``True`` also return ``d(prediction)/d(alpha)``.

        Returns
        -------
        dict with keys ``"prediction"`` (and ``"sensitivity"`` if requested),
        each of shape ``(n_steps, n_obs_sensors)``.
        """
        temps = np.asarray(temps, dtype=float)
        n_steps = temps.shape[0]
        top = temps[:, self.bc_columns[0]]
        bot = temps[:, self.bc_columns[1]]

        prop, a_inv, g_top, g_bot, kappa = self._operators(alpha)
        theta = self.theta
        one_m_theta = 1.0 - theta
        dkappa = self.dt_seconds / self.dz_m**2  # d(kappa)/d(alpha)
        obs_local = self._obs_local

        # Initial interior profile: linear interpolation through the sensors.
        u = np.interp(self.nodes_m[1:-1], self.depths_m, temps[0])

        pred = np.empty((n_steps, obs_local.size))
        pred[0] = u[obs_local]

        # Boundary contribution to the RHS at step n is
        #   c_n = [ (1-theta)*kappa*top_n + theta*kappa*top_{n+1} ] at node 0
        #       + [ (1-theta)*kappa*bot_n + theta*kappa*bot_{n+1} ] at node -1
        top_coef = kappa * (one_m_theta * top[:-1] + theta * top[1:])
        bot_coef = kappa * (one_m_theta * bot[:-1] + theta * bot[1:])

        if not with_sensitivity:
            for n in range(n_steps - 1):
                u = prop @ u + top_coef[n] * g_top + bot_coef[n] * g_bot
                pred[n + 1] = u[obs_local]
            return {"prediction": pred}

        # Tangent-linear march for d u / d alpha (zero initial condition).
        v = np.zeros_like(u)
        sens = np.empty_like(pred)
        sens[0] = 0.0
        lap_u_curr = self._laplacian_full(u, top[0], bot[0])
        for n in range(n_steps - 1):
            u_next = prop @ u + top_coef[n] * g_top + bot_coef[n] * g_bot
            lap_u_next = self._laplacian_full(u_next, top[n + 1], bot[n + 1])
            # A v^{n+1} = B v^n + dkappa*(theta*Lap u^{n+1} + (1-theta)*Lap u^n)
            source = dkappa * (theta * lap_u_next + one_m_theta * lap_u_curr)
            v = prop @ v + a_inv @ source
            u = u_next
            lap_u_curr = lap_u_next
            pred[n + 1] = u[obs_local]
            sens[n + 1] = v[obs_local]

        return {"prediction": pred, "sensitivity": sens}


# --------------------------------------------------------------------------- #
#  Backward-compatible helpers for notebooks written against the explicit API  #
# --------------------------------------------------------------------------- #
def active_node_indices(nz: int, space_scheme: str = "central") -> np.ndarray:
    """Return sensor-column indices fitted by the inverse problem.

    The refined implicit solver uses the first and last sensor columns as
    time-varying Dirichlet boundaries.  All interior sensor columns are active
    data locations.  ``space_scheme`` is accepted for compatibility; the refined
    solver uses a central finite-difference diffusion operator internally.
    """
    if space_scheme != "central":
        raise ValueError(
            f"space_scheme={space_scheme!r} is not supported by the refined implicit solver; "
            "use 'central'."
        )
    return np.arange(1, int(nz) - 1)


def forward_temperature(
    alpha: float,
    obs_window: np.ndarray,
    grid: HeatGrid,
    space_scheme: str = "central",
) -> np.ndarray:
    """Return full sensor-column predictions for compatibility.

    Internally this calls :meth:`HeatGrid.simulate`, which solves on the refined
    implicit grid.  The returned array has the same shape as ``obs_window``:
    boundary columns are prescribed observations and interior columns are model
    predictions at the sensor depths.
    """
    if space_scheme != "central":
        raise ValueError(
            f"space_scheme={space_scheme!r} is not supported by the refined implicit solver; "
            "use 'central'."
        )
    temps = np.asarray(obs_window, dtype=float)
    pred = temps.copy()
    sim = grid.simulate(alpha, temps, with_sensitivity=False)
    pred[:, list(grid.obs_columns)] = sim["prediction"]
    return pred


def laplacian_coefficients(j: int, space_scheme: str = "central"):
    """Legacy dimensionless central-difference stencil coefficients."""
    if space_scheme != "central":
        raise ValueError(
            f"space_scheme={space_scheme!r} is not supported by the refined implicit solver; "
            "use 'central'."
        )
    return [(j - 1, 1.0), (j, -2.0), (j + 1, 1.0)]


def laplacian_dimensionless(values: np.ndarray, j: int, space_scheme: str = "central") -> float:
    """Legacy central second-difference helper retained for old diagnostics."""
    return float(sum(coeff * values[k] for k, coeff in laplacian_coefficients(j, space_scheme)))
