r"""Time-dependent (and layer-ready) thermal-diffusivity inversion.

Where the scalar/windowed estimator returns one ``alpha`` per window, this module
solves a **single** inverse problem over a whole span for a diffusivity that
varies in time, ``alpha(t)`` (and, by the same machinery, in depth -- ``alpha(z,t)``).

Model
-----
Implicit theta-method discretisation of the (possibly heterogeneous) heat
equation in conservative form::

    dT/dt = d/dz ( alpha(z, t) dT/dz )

The shallowest / deepest sensors are time-varying Dirichlet boundaries; interior
sensors are the data.  The diffusivity used to advance step ``n -> n+1`` is a
per-face vector ``alpha_face[n, :]`` (one value per cell interface), so spatially
heterogeneous (multi-layer) diffusivity is supported by construction.  For a
single layer every face shares the step's scalar ``alpha[n]`` and the operator
reduces exactly to ``alpha * d^2T/dz^2``.

Parameterisation
----------------
The unknown is ``m`` = ``log(alpha)`` on a coarse **control grid** (knots in time,
optionally in depth/layers).  ``log alpha`` at each step/face is a linear
interpolation of ``m`` (matrix ``M_time`` in time, layer-grouping in depth).  Log
space keeps ``alpha > 0`` and well scaled; the control grid both reduces the
number of unknowns and is the natural place to impose smoothness.

Gradient
--------
A **discrete adjoint**: one forward sweep (storing the state history) plus one
backward sweep yields ``dJ/d(alpha_face)`` at *every* step and face at once -- the
whole gradient for the cost of one extra solve, independent of the number of
control knots.  This is why a genuinely time-resolved inversion is feasible where
per-knot finite differencing would not be.

Regularisation
--------------
Time-dependent parameter estimation is ill-posed, so the objective adds a
temporal-smoothness penalty (first differences of ``m`` in time) and an optional
prior pull, both quadratic.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .forward import HeatGrid


# --------------------------------------------------------------------------- #
#  Tridiagonal solve (general, per-step coefficients vary so we rebuild)        #
# --------------------------------------------------------------------------- #
def _thomas(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system. ``lower[0]`` and ``upper[-1]`` are unused."""
    n = diag.size
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i] * cp[i - 1]
        cp[i] = upper[i] / denom if i < n - 1 else 0.0
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / denom
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# --------------------------------------------------------------------------- #
#  Control-grid parameterisation (time knots x layer groups)                    #
# --------------------------------------------------------------------------- #
@dataclass
class ControlParameterization:
    """Maps a control vector ``m`` (log-alpha) to a per-step, per-face field.

    Attributes
    ----------
    M_time : (n_trans, n_time_knots)
        Linear-interpolation weights from time knots to step transitions.
    face_layer : (n_faces,)
        Layer-group index of each cell interface.
    n_layers : int
        Number of independent diffusivity layers (1 = single layer / pure time).
    knot_times : (n_time_knots,)
        Timestamps of the control knots (for reporting / plotting).
    """

    M_time: np.ndarray
    face_layer: np.ndarray
    n_layers: int
    knot_times_s: np.ndarray

    @property
    def n_time_knots(self) -> int:
        return self.M_time.shape[1]

    @property
    def n_params(self) -> int:
        return self.n_layers * self.n_time_knots

    def reshape(self, m_flat: np.ndarray) -> np.ndarray:
        """``(n_params,)`` -> ``(n_layers, n_time_knots)``."""
        return np.asarray(m_flat, dtype=float).reshape(self.n_layers, self.n_time_knots)

    def log_alpha_face_steps(self, m_flat: np.ndarray) -> np.ndarray:
        """Return ``log alpha`` of shape ``(n_trans, n_faces)``."""
        m = self.reshape(m_flat)
        # log-alpha per (layer, transition) = M_time @ m_layer
        log_per_layer = self.M_time @ m.T              # (n_trans, n_layers)
        return log_per_layer[:, self.face_layer]       # (n_trans, n_faces)

    def alpha_face_steps(self, m_flat: np.ndarray) -> np.ndarray:
        return np.exp(self.log_alpha_face_steps(m_flat))

    def contract_grad(self, grad_log_face: np.ndarray) -> np.ndarray:
        """Map ``dJ/d(log alpha_face)`` of shape ``(n_trans, n_faces)`` to ``dJ/dm``.

        Sums faces within each layer group, then applies ``M_time^T`` in time.
        """
        n_trans = grad_log_face.shape[0]
        per_layer = np.zeros((n_trans, self.n_layers))
        # accumulate face gradients into their layer group
        np.add.at(per_layer.T, self.face_layer, grad_log_face.T)
        grad_m = (self.M_time.T @ per_layer).T          # (n_layers, n_time_knots)
        return grad_m.reshape(-1)


def _interp_matrix(sample_times_s: np.ndarray, knot_times_s: np.ndarray) -> np.ndarray:
    """Linear-interpolation weight matrix: values at samples = W @ values at knots."""
    n_s = sample_times_s.size
    n_k = knot_times_s.size
    W = np.zeros((n_s, n_k))
    if n_k == 1:
        W[:, 0] = 1.0
        return W
    idx = np.clip(np.searchsorted(knot_times_s, sample_times_s, side="right") - 1, 0, n_k - 2)
    left = knot_times_s[idx]
    right = knot_times_s[idx + 1]
    w = (sample_times_s - left) / (right - left)
    w = np.clip(w, 0.0, 1.0)
    rows = np.arange(n_s)
    W[rows, idx] = 1.0 - w
    W[rows, idx + 1] = w
    return W


def build_parameterization(
    grid: HeatGrid,
    step_times_s: np.ndarray,
    *,
    control_hours: float = 3.0,
    n_layers: int = 1,
    layer_face_edges: Optional[Sequence[int]] = None,
) -> ControlParameterization:
    """Construct a time (and optional layer) control parameterisation.

    Parameters
    ----------
    step_times_s
        Elapsed seconds of each time *sample* (length ``n_steps``).  Control knots
        are placed every ``control_hours`` across this span; transitions are
        interpolated at the midpoints between consecutive samples.
    n_layers, layer_face_edges
        For multi-layer use: ``n_layers`` groups of faces.  ``layer_face_edges``
        gives the face indices that start each layer (defaults to an even split).
    """
    n_steps = step_times_s.size
    n_trans = n_steps - 1
    trans_times = 0.5 * (step_times_s[:-1] + step_times_s[1:])

    span = step_times_s[-1] - step_times_s[0]
    n_knots = max(2, int(round(span / (control_hours * 3600.0))) + 1)
    knot_times_s = np.linspace(step_times_s[0], step_times_s[-1], n_knots)
    M_time = _interp_matrix(trans_times, knot_times_s)

    n_faces = grid.n_cells
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if n_layers > n_faces:
        raise ValueError(
            f"n_layers={n_layers} exceeds the {n_faces} cell faces; refine the grid "
            "(raise cells_per_gap) or use fewer layers."
        )
    if n_layers == 1:
        face_layer = np.zeros(n_faces, dtype=int)
    elif layer_face_edges is None:
        face_layer = np.minimum((np.arange(n_faces) * n_layers) // n_faces, n_layers - 1)
    else:
        edges = np.asarray(layer_face_edges, dtype=int)
        if edges.size != n_layers or edges[0] != 0 or not np.all(np.diff(edges) > 0) \
                or edges[-1] >= n_faces:
            raise ValueError(
                "layer_face_edges must be strictly ascending, length n_layers, start at 0, "
                f"and stay below n_faces={n_faces}; got {edges.tolist()}"
            )
        face_layer = np.searchsorted(edges, np.arange(n_faces), side="right") - 1

    # No layer may be empty: an unconstrained free parameter would corrupt the fit.
    counts = np.bincount(face_layer, minlength=n_layers)
    empty = np.flatnonzero(counts == 0)
    if empty.size:
        raise ValueError(f"layer(s) {empty.tolist()} contain no cell faces (data-unconstrained)")

    return ControlParameterization(M_time=M_time, face_layer=face_layer,
                                   n_layers=n_layers, knot_times_s=knot_times_s)


# --------------------------------------------------------------------------- #
#  Forward + adjoint engine (per-face alpha, theta-method)                       #
# --------------------------------------------------------------------------- #
def _forward(grid: HeatGrid, alpha_face_steps: np.ndarray, top: np.ndarray, bot: np.ndarray,
             u0: np.ndarray, theta: float) -> np.ndarray:
    """March and return the interior state history ``T`` of shape ``(n_steps, n)``."""
    n = grid.n_unknown
    n_steps = top.size
    scale = grid.dt_seconds / grid.dz_m**2
    T = np.empty((n_steps, n))
    T[0] = u0
    u = u0
    for nstep in range(n_trans_of(n_steps)):
        kf = alpha_face_steps[nstep] * scale          # (n_faces,)
        kl, kc, ku = _ab_arrays(kf, theta)            # A subdiag/diag/superdiag
        bl, bd, bu = _b_arrays(kf, theta)
        rhs = _tri_matvec(bl, bd, bu, u)
        rhs[0] += theta * kf[0] * top[nstep + 1] + (1.0 - theta) * kf[0] * top[nstep]
        rhs[-1] += theta * kf[-1] * bot[nstep + 1] + (1.0 - theta) * kf[-1] * bot[nstep]
        u = _thomas(kl, kc, ku, rhs)
        T[nstep + 1] = u
    return T


def n_trans_of(n_steps: int) -> int:
    return n_steps - 1


def _ab_arrays(kf: np.ndarray, theta: float):
    """A = I - theta*G ; return (sub, diag, sup) for interior nodes."""
    left = kf[:-1]        # face to the left of interior node li  (kf[li])
    right = kf[1:]        # face to the right (kf[li+1])
    diag = 1.0 + theta * (left + right)
    sub = -theta * left
    sup = -theta * right
    return sub, diag, sup


def _b_arrays(kf: np.ndarray, theta: float):
    """B = I + (1-theta)*G ; return (sub, diag, sup)."""
    left = kf[:-1]
    right = kf[1:]
    omt = 1.0 - theta
    diag = 1.0 - omt * (left + right)
    sub = omt * left
    sup = omt * right
    return sub, diag, sup


def _tri_matvec(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = diag * x
    y[1:] += sub[1:] * x[:-1]
    y[:-1] += sup[:-1] * x[1:]
    return y


def _data_and_adjoint(grid: HeatGrid, alpha_face_steps: np.ndarray, top: np.ndarray, bot: np.ndarray,
                      u0: np.ndarray, obs: np.ndarray, theta: float, want_grad: bool):
    """Return ``(cost_data, pred, grad_alpha_face)``.

    ``grad_alpha_face`` (shape ``(n_trans, n_faces)``) is ``dJ_data/d(alpha_face)``
    via the discrete adjoint, or ``None`` if ``want_grad`` is False.
    """
    n = grid.n_unknown
    n_steps = top.size
    n_trans = n_steps - 1
    scale = grid.dt_seconds / grid.dz_m**2
    obs_local = grid._obs_local

    T = _forward(grid, alpha_face_steps, top, bot, u0, theta)
    pred = T[:, obs_local]
    resid = pred - obs                                  # (n_steps, n_obs)
    cost_data = 0.5 * float(np.sum(resid * resid))
    if not want_grad:
        return cost_data, pred, None

    # scattered data residual g^n (interior vector), zeros except obs nodes
    g = np.zeros((n_steps, n))
    g[:, obs_local] = resid

    # backward adjoint sweep for mu^0 .. mu^{n_trans-1}
    mu = np.zeros((n_trans, n))
    kf_last = alpha_face_steps[n_trans - 1] * scale
    sub, diag, sup = _ab_arrays(kf_last, theta)
    mu[n_trans - 1] = _thomas(sub, diag, sup, -g[n_steps - 1])
    for k in range(n_trans - 1, 0, -1):
        kf_k = alpha_face_steps[k] * scale
        bsub, bdiag, bsup = _b_arrays(kf_k, theta)
        rhs = _tri_matvec(bsub, bdiag, bsup, mu[k]) - g[k]
        kf_km1 = alpha_face_steps[k - 1] * scale
        asub, adiag, asup = _ab_arrays(kf_km1, theta)
        mu[k - 1] = _thomas(asub, adiag, asup, rhs)

    # full profiles (with Dirichlet boundaries) for the parameter gradient
    grad_alpha = np.empty((n_trans, grid.n_cells))
    for nstep in range(n_trans):
        Tn_full = np.empty(n + 2)
        Tn_full[0] = top[nstep]; Tn_full[-1] = bot[nstep]; Tn_full[1:-1] = T[nstep]
        Tnp_full = np.empty(n + 2)
        Tnp_full[0] = top[nstep + 1]; Tnp_full[-1] = bot[nstep + 1]; Tnp_full[1:-1] = T[nstep + 1]
        mu_full = np.zeros(n + 2)
        mu_full[1:-1] = mu[nstep]
        dmu = mu_full[1:] - mu_full[:-1]                # (n_faces,)
        dTn = Tn_full[1:] - Tn_full[:-1]
        dTnp = Tnp_full[1:] - Tnp_full[:-1]
        dJ_dkappa = dmu * (theta * dTnp + (1.0 - theta) * dTn)
        grad_alpha[nstep] = dJ_dkappa * scale           # d/d(alpha_face) = d/d(kappa) * scale
    return cost_data, pred, grad_alpha


# --------------------------------------------------------------------------- #
#  Public solver                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class TimeDependentResult:
    alpha_t: np.ndarray            # (n_steps,) alpha at each time sample
    alpha_control: np.ndarray      # (n_layers, n_time_knots) estimated control field
    knot_times: np.ndarray         # (n_time_knots,) control knot timestamps
    step_times: np.ndarray         # (n_steps,) sample timestamps
    prediction: np.ndarray         # (n_steps, n_obs) fitted interior temperatures
    cost: float
    cost_data: float
    cost_reg: float
    n_iter: int
    n_grad_eval: int
    success: bool
    m: np.ndarray                  # raw optimal control vector (log-alpha)
    param: ControlParameterization


def solve_time_dependent_alpha(
    temps: np.ndarray,
    grid: HeatGrid,
    times,
    *,
    control_hours: float = 3.0,
    beta_smooth: float = 1.0,
    beta_prior: float = 0.0,
    alpha_prior: float = 1.0e-6,
    alpha_initial: Optional[float] = None,
    bounds: Tuple[float, float] = (1.0e-8, 1.0e-5),
    theta: float = 0.5,
    n_layers: int = 1,
    layer_face_edges: Optional[Sequence[int]] = None,
    max_iter: int = 300,
    m0: Optional[np.ndarray] = None,
) -> TimeDependentResult:
    """Estimate a time- (and optionally layer-) varying ``alpha`` over the span.

    Parameters
    ----------
    temps
        Sensor temperatures ``(n_steps, n_sensors)``; first/last columns are the
        Dirichlet boundaries, interior columns are fitted.
    grid
        :class:`HeatGrid` providing the vertical discretisation.
    times
        Timestamps for the ``n_steps`` samples (used for the control grid).
    control_hours
        Spacing of the time control knots (smaller = finer ``alpha(t)``).
    beta_smooth
        Weight of the temporal smoothness penalty on ``log alpha`` (penalises the
        first difference between adjacent knots).  Normalised by the knot spacing,
        so its meaning is independent of ``control_hours``.  Runtime scales with the
        number of timesteps (a ~3-day span is order-of-magnitude slower than a
        ~12 h window), since each step solves its own tridiagonal system.
    beta_prior, alpha_prior
        Optional pull of ``log alpha`` toward ``log(alpha_prior)``.
    alpha_initial
        Constant starting guess (defaults to ``alpha_prior``).
    n_layers, layer_face_edges
        Enable a depth-varying ``alpha(z, t)`` with ``n_layers`` layers.
    """
    temps = np.asarray(temps, dtype=float)
    if not grid.obs_columns:
        raise ValueError("grid has no interior sensors to fit (need >= 1 between the boundaries)")
    top = temps[:, grid.bc_columns[0]]
    bot = temps[:, grid.bc_columns[1]]
    obs = temps[:, list(grid.obs_columns)]
    u0 = np.interp(grid.nodes_m[1:-1], grid.depths_m, temps[0])

    step_times_s = _elapsed_seconds(times)
    param = build_parameterization(grid, step_times_s, control_hours=control_hours,
                                   n_layers=n_layers, layer_face_edges=layer_face_edges)

    lo, hi = float(bounds[0]), float(bounds[1])
    log_lo, log_hi = np.log(lo), np.log(hi)
    a_init = float(alpha_initial if alpha_initial is not None else alpha_prior)
    if m0 is None:
        m0 = np.full(param.n_params, np.log(a_init))
    m_prior = np.log(float(alpha_prior))

    # Temporal first-difference operator on the time-knot axis (per layer).  The
    # penalty is normalised by the knot spacing so that ``beta_smooth`` has a
    # resolution-independent meaning (a penalty on d(log alpha)/dt) rather than
    # scaling with ``control_hours``.  ``n_time_knots >= 2`` always holds.
    nk = param.n_time_knots
    Dt = np.zeros((nk - 1, nk))
    rows = np.arange(nk - 1)
    Dt[rows, rows] = -1.0
    Dt[rows, rows + 1] = 1.0
    DtTDt = Dt.T @ Dt
    dt_knot_h = (param.knot_times_s[-1] - param.knot_times_s[0]) / (3600.0 * (nk - 1))
    smooth_w = beta_smooth / max(dt_knot_h, 1e-9)

    evals = {"n": 0}

    def cost_grad(m_flat: np.ndarray):
        evals["n"] += 1
        alpha_face_steps = param.alpha_face_steps(m_flat)
        cost_data, pred, grad_alpha = _data_and_adjoint(
            grid, alpha_face_steps, top, bot, u0, obs, theta, want_grad=True)
        # data gradient wrt m: chain through log-alpha (dJ/d log a_face = dJ/da_face * a_face)
        grad_log_face = grad_alpha * alpha_face_steps
        g_data = param.contract_grad(grad_log_face)

        m_mat = param.reshape(m_flat)
        # smoothness (knot-spacing normalised) + prior, per layer on the time-knot axis
        reg_smooth = 0.5 * smooth_w * float(np.sum((m_mat @ Dt.T) ** 2))
        diff_prior = m_mat - m_prior
        reg_prior = 0.5 * beta_prior * float(np.sum(diff_prior ** 2))
        g_smooth = smooth_w * (m_mat @ DtTDt)
        g_prior = beta_prior * diff_prior
        g_reg = (g_smooth + g_prior).reshape(-1)

        return cost_data + reg_smooth + reg_prior, g_data + g_reg

    res = minimize(cost_grad, m0, jac=True, method="L-BFGS-B",
                   bounds=[(log_lo, log_hi)] * param.n_params,
                   options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8})
    if not res.success:
        warnings.warn(f"solve_time_dependent_alpha did not converge: {res.message}", RuntimeWarning)

    # Recompute the diagnostics cleanly at the optimum.
    m_opt = res.x
    alpha_face_steps = param.alpha_face_steps(m_opt)
    cost_data, pred, _ = _data_and_adjoint(grid, alpha_face_steps, top, bot, u0, obs, theta, want_grad=False)
    m_mat = param.reshape(m_opt)
    reg_smooth = 0.5 * smooth_w * float(np.sum((m_mat @ Dt.T) ** 2))
    reg_prior = 0.5 * beta_prior * float(np.sum((m_mat - m_prior) ** 2))

    # alpha at each time *sample*: per-layer field interpolated back to samples, then
    # collapsed to a single reported series by a FACE-WEIGHTED mean (each layer weighted
    # by how many cell faces it spans).  Single layer -> identical to that layer's alpha;
    # the authoritative per-layer field is returned as ``alpha_control``.
    layer_weight = np.bincount(param.face_layer, minlength=param.n_layers) / param.face_layer.size
    alpha_trans = np.exp(param.M_time @ m_mat.T)          # (n_trans, n_layers)
    alpha_trans_mean = alpha_trans @ layer_weight         # (n_trans,)
    alpha_t = np.empty(temps.shape[0])
    alpha_t[:-1] = alpha_trans_mean
    alpha_t[-1] = alpha_trans_mean[-1]

    times_arr = np.asarray(times)
    knot_times = _knot_timestamps(times_arr, step_times_s, param.knot_times_s)

    return TimeDependentResult(
        alpha_t=alpha_t,
        alpha_control=np.exp(m_mat),
        knot_times=knot_times,
        step_times=times_arr,
        prediction=pred,
        cost=float(cost_data + reg_smooth + reg_prior),
        cost_data=float(cost_data),
        cost_reg=float(reg_smooth + reg_prior),
        n_iter=int(res.nit),
        n_grad_eval=int(evals["n"]),
        success=bool(res.success),
        m=m_opt,
        param=param,
    )


# --------------------------------------------------------------------------- #
#  Time helpers                                                                 #
# --------------------------------------------------------------------------- #
def _elapsed_seconds(times) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(times)))
    ns = idx.to_numpy().astype("datetime64[ns]").astype("int64")
    return (ns - ns[0]) / 1e9


def _knot_timestamps(times_arr: np.ndarray, step_times_s: np.ndarray, knot_times_s: np.ndarray) -> np.ndarray:
    base = pd.Timestamp(pd.to_datetime(times_arr[0]))
    return np.array([base + pd.Timedelta(seconds=float(s)) for s in knot_times_s], dtype=object)
