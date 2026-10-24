# soil_inverse

A small, fast 1-D heat-diffusion inverse framework for estimating an effective
soil thermal diffusivity `alpha` from a vertical stack of temperature sensors.
Used by `alpha_water_content_analysis.ipynb`.

## Model

We solve the 1-D heat equation on the depth interval spanned by the sensors,

```
dT/dt = alpha * d^2T/dz^2
```

with the shallowest and deepest sensors imposed as **time-varying Dirichlet
boundaries** and the interior sensors used as the data the model must match.
Each sliding window yields one scalar `alpha`.

## What changed (the "4 cells" fix)

The previous finite-difference solver placed grid nodes *exactly on the four
sensors* (`dz = 0.25 m`, only 2 interior unknowns) and marched **explicitly**.
That is spatially crude, and any attempt to refine it explodes because an
explicit scheme needs `alpha*dt/dz^2 <= 0.5`.

`HeatGrid` now supports an **arbitrary vertical resolution** and an
**unconditionally stable implicit** time integrator:

| | old | new |
|---|---|---|
| vertical cells | fixed at 3 (4 nodes) | `cells_per_gap * (n_sensors-1)`, any size |
| `dz` (default data) | 0.25 m | 0.05 m (`cells_per_gap=5`), tunable |
| time scheme | explicit (CFL-limited) | theta-method (`0.5`=Crank–Nicolson), unconditionally stable |
| gradient | — | exact tangent-linear / adjoint sensitivity |
| `alpha` solve | local only | coarse global scan + local refine (avoids spurious bound minima) |

Refining the grid matters: on this dataset the old 3-cell grid over-estimates
`alpha` (~1.2e-6); refinement converges toward ~6.6e-7.

## Performance

* For a fixed `alpha`, the implicit matrix is constant in time, so it is factored
  **once per solve** into a dense propagator `P = A^{-1} B`; each time step is a
  single BLAS matrix-vector product (no per-step factorisation, no Python sweep).
* The gradient comes from a tangent-linear march that reuses the same factor.
* The optimisers run a cheap coarse `log(alpha)` scan to find the *global* basin,
  then refine — bounded Brent (`solve_l2_bounded`) or quasi-Newton on the adjoint
  gradient (`solve_adjoint_gradient_descent`, ~5 iterations).

The full analysis notebook (all sweeps + figures) runs in ~75 s; a single 321-
window adjoint sweep at `cells_per_gap=5` is ~2 s (~7 ms/window).

## Controlling vertical resolution

```python
from soil_inverse.forward import HeatGrid

# 5 cells between each adjacent sensor pair -> dz = 0.05 m (15 cells total)
grid = HeatGrid(depths_m=[0.25, 0.5, 0.75, 1.0], dt_seconds=600.0, cells_per_gap=5)

grid.n_cells   # 15
grid.dz_m      # 0.05
grid.r(1e-6)   # diffusion number alpha*dt/dz^2 (reported, no longer a stability limit)

# Finer column (dz = 0.0167 m); still stable, slightly slower:
fine = HeatGrid(depths_m=[0.25, 0.5, 0.75, 1.0], dt_seconds=600.0, cells_per_gap=15)

# Backward-compatible legacy grid (nodes on sensors, dz = 0.25 m):
legacy = HeatGrid(depths_m=[0.25, 0.5, 0.75, 1.0], dt_seconds=600.0, cells_per_gap=1)
```

`cells_per_gap=5` is the default. Use `theta=1.0` for backward Euler (extra
numerical damping) if a window is noisy.

## Time-dependent diffusivity: `alpha(t)` (and `alpha(z, t)`)

The scalar/windowed estimators above return one `alpha` per window. `timevarying.py`
instead solves a **single** inversion over a whole span for a diffusivity that
varies in time — and, by the same machinery, in depth (multi-layer).

```python
from soil_inverse.timevarying import solve_time_dependent_alpha

res = solve_time_dependent_alpha(
    temps, grid, times,
    control_hours=3.0,     # temporal resolution of alpha(t): a knot every 3 h
    beta_smooth=5.0,       # temporal smoothness penalty on log-alpha (raise = smoother)
    beta_prior=1.0, alpha_prior=1e-6,
    bounds=(1e-8, 1e-5),
)
res.alpha_t          # alpha at every timestep
res.alpha_control    # estimated values at the control knots, shape (n_layers, n_knots)
res.knot_times       # knot timestamps
res.cost_data, res.cost_reg, res.n_iter

# depth + time varying alpha(z, t): just ask for layers
res2 = solve_time_dependent_alpha(temps, grid, times, n_layers=2, control_hours=6.0,
                                  beta_smooth=5.0, beta_prior=1.0)
res2.alpha_control   # shape (2, n_knots): one diffusivity history per layer
```

How it works:

* **Implicit theta-method** forward model in conservative form
  `dT/dt = d/dz(alpha(z,t) dT/dz)`, with a per-face diffusivity so spatial
  (multi-layer) heterogeneity is built in; reduces exactly to `alpha d²T/dz²` for
  one layer.
* **Discrete adjoint** gradient: one forward + one backward sweep yields the
  gradient w.r.t. *every* control knot at once (verified against finite
  differences to ~4e-8). This is what makes a time-resolved inversion feasible —
  per-knot finite differencing would cost one forward solve per knot.
* **Control-grid parameterisation** in `log(alpha)` (knots interpolated to each
  step) + **temporal-smoothness regularisation**, since time-varying parameter
  estimation is otherwise ill-posed.

Verification: recovers a synthetic `alpha(t)` to **corr 0.99, ~1% median error**
(robust to 1% noise); on real event-period data the time-resolved `alpha(t)` cuts
the data misfit ~55% vs the best constant `alpha`. A whole 72 h span solves in ~1 s.

## Modules

| module | contents |
|---|---|
| `data.py` | `load_and_resample_soil_csv`, `median_time_step_seconds` |
| `forward.py` | `HeatGrid` — refined implicit FD forward model + exact sensitivity |
| `objective.py` | `TikhonovRegularization`, `evaluate_cost` (cost + adjoint gradient) |
| `optimizers.py` | `solve_l2_bounded`, `solve_adjoint_gradient_descent` (scalar α per window) |
| `timevarying.py` | `solve_time_dependent_alpha` — α(t)/α(z,t) via implicit forward + discrete adjoint |
| `windowing.py` | `make_window_indices` |
```
