# Soil Thermal Diffusivity Inverse Framework (`soil_inverse`)

A 1-D heat-diffusion inverse modeling framework for estimating effective soil thermal diffusivity ($\alpha$) from vertical soil temperature profiles, applied to field observations during Typhoon Lupit (July–August 2021) and synthetic verification benchmarks.

---

## Overview

### Governing Physics & Forward Model
We solve the 1-D heat conduction equation over the depth interval spanned by a vertical array of soil temperature sensors ($z \in [0.25, 1.00]\text{ m}$):

$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial z^2}$$

and, for time- and depth-dependent inversions, its conservative heterogeneous form:

$$\frac{\partial T}{\partial t} = \frac{\partial}{\partial z}\left(\alpha(z, t) \frac{\partial T}{\partial z}\right)$$

- **Boundary Conditions**: The shallowest ($0.25\text{ m}$, `T1`) and deepest ($1.00\text{ m}$, `T4`) sensors act as time-varying Dirichlet boundaries.
- **Observation Targets**: Interior sensors ($0.50\text{ m}$, `T2` and $0.75\text{ m}$, `T3`) serve as the observations fitted by the inverse problem.
- **Spatial & Temporal Discretization**: `HeatGrid` subdivides each inter-sensor gap into `cells_per_gap` uniform finite-difference cells (default `cells_per_gap = 5`, yielding $\Delta z = 0.05\text{ m}$ across 15 cells) and integrates in time using the unconditionally stable implicit $\theta$-method (default $\theta = 0.5$, Crank–Nicolson, at $\Delta t = 600\text{ s}$). For constant $\alpha$ within a window, the implicit matrix is factored once per solve into a dense step propagator.

### Inverse Formulation & Optimization
Over a sliding time window (or full time span), the objective functional combines the interior L2 temperature misfit with optional Tikhonov regularization:

$$J(\alpha) = \frac{1}{2} \sum_{n, k \in \text{int}} \left(T^{\text{pred}}_{n,k}(\alpha) - T^{\text{obs}}_{n,k}\right)^2 + \frac{1}{2} \beta (\alpha - \alpha_{\text{prior}})^2$$

1. **Exact Gradients**: Computed via a tangent-linear sensitivity march for scalar $\alpha$ (`HeatGrid.simulate(..., with_sensitivity=True)`) or a discrete adjoint backward sweep for time/layer-varying $\alpha(z, t)$ (`soil_inverse.timevarying`).
2. **Globally Robust Two-Stage Solvers**: Because the windowed objective $J(\alpha)$ is not globally convex and can exhibit spurious local minima near the lower bound, the scalar solvers (`solve_adjoint_gradient_descent` and `solve_l2_bounded`) optimize in $\log(\alpha)$ space using:
   - A 16-point coarse scan across $\log(\alpha)$ bounds to bracket the global basin.
   - Local refinement via L-BFGS-B (using exact adjoint/tangent-linear gradients) or bounded Brent minimization.

---

## Repository Structure

```text
├── soil_inverse/                                     # Core Python package
│   ├── __init__.py                                   # Public package exports
│   ├── data.py                                       # CSV loading, 10-min resampling, linear interpolation
│   ├── forward.py                                    # HeatGrid implicit solver & tangent-linear sensitivity
│   ├── objective.py                                  # L2 data misfit & TikhonovRegularization
│   ├── gradients.py                                  # Exact adjoint/tangent-linear & finite-difference gradients
│   ├── optimizers.py                                 # Two-stage bounded Brent & L-BFGS-B scalar solvers
│   ├── timevarying.py                                # Time- & layer-dependent alpha(z, t) discrete-adjoint solver
│   ├── windowing.py                                  # Sliding-window index generation & time filtering
│   └── README.md                                     # Package-level technical notes
├── 01_alpha_estimation.ipynb                         # Grid-resolution & method comparison (dz = 0.05 m vs 0.01 m)
├── 02_window_length_long_short_comparison.ipynb      # Window-length sensitivity (3 hr, 6 hr, 12 hr)
├── 03_mean_m_alpha_empirical_updated.ipynb           # Inverted alpha vs soil water content & empirical curves
├── 04_synthetic_recovery_time_dependent_alpha.ipynb  # Synthetic cyclic alpha(t) recovery verification
├── 05_soil_forward_validation_manual_alpha.ipynb     # Forward temperature & residual contour validation
├── 06_comment_1_3_initial_value_sensitivity.ipynb    # Initial-value sensitivity analysis (321 windows × 6 starts)
├── Lupit_20210729_0812.csv                           # Field sensor dataset (M1–M4 water content, T1–T4 temperature)
├── C0U940_20210728-0812/                             # Hourly weather station precipitation records (2021-07-29 to 08-12)
└── 00_output_figures/                                # Generated figures (.jpg/.pdf) and summary tables (.csv/.md)
```

---

## Analysis Notebooks & Outputs

| Notebook | Description | Output Directory |
| :--- | :--- | :--- |
| **`01_alpha_estimation.ipynb`** | Estimates sliding-window (3 hr window, 1 hr stride) thermal diffusivity over 2021-07-29 to 2021-08-12 (`321` windows), comparing implicit Crank–Nicolson vertical grid resolutions ($\Delta z = 0.05\text{ m}$ vs. $0.01\text{ m}$) with and without Tikhonov regularization. | `00_output_figures/01_alpha_estimation/` |
| **`02_window_length_long_short_comparison.ipynb`** | Compares 3 hr, 6 hr, and 12 hr sliding windows (1 hr overlap) across the full period (`2021-07-29` to `2021-08-12`) and the Typhoon Lupit event period (`2021-08-06` to `2021-08-09`). | `00_output_figures/02_window_length/` |
| **`03_mean_m_alpha_empirical_updated.ipynb`** | Examines the relationship between inverted $\alpha$ and mean volumetric soil water content (`mean_M`) during rainy windows, benchmarked against empirical loam curves. | `00_output_figures/03_mean_m_alpha_empirical_updated/` |
| **`04_synthetic_recovery_time_dependent_alpha.ipynb`** | Verifies `solve_time_dependent_alpha` on a 192-hour synthetic benchmark with a cyclic $\alpha_{\text{true}}(t)$ and 1% observation noise (achieving $r = 0.998$ and $0.62\%$ median relative error). | `00_output_figures/04_synthetic_recovery_time_dependent_alpha/` |
| **`05_soil_forward_validation_manual_alpha.ipynb`** | Runs the forward heat-conduction model using a prescribed $\alpha$ over selected dry/wet intervals to compare observed, simulated, and residual temperature contours. | `00_output_figures/05_soil_forward/` |
| **`06_comment_1_3_initial_value_sensitivity.ipynb`** | Evaluates starting-value sensitivity across all 321 field windows and 6 initial guesses ($\alpha_0 \in [10^{-8}, 10^{-5}]\text{ m}^2/\text{s}$), comparing **Coarse scan + L-BFGS-B** against **Direct L-BFGS-B**. | `00_output_figures/06_comment_1_3_initial_value_sensitivity/` |

---

## Getting Started

### Dependencies
The framework requires Python 3.8+ with standard scientific computing libraries:

```bash
pip install numpy scipy pandas matplotlib jupyter
```

### Quick Usage Example

#### 1. Sliding-Window Scalar $\alpha$ Inversion
```python
from pathlib import Path
import numpy as np
from soil_inverse import (
    HeatGrid,
    TikhonovRegularization,
    load_and_resample_soil_csv,
    median_time_step_seconds,
    make_window_indices,
    solve_adjoint_gradient_descent,
)

# Load and resample sensor observations to a regular 10-minute grid
times, temps, water_content, df = load_and_resample_soil_csv(
    Path("Lupit_20210729_0812.csv"),
    temp_columns=["T1", "T2", "T3", "T4"],
    water_content_columns=["M1", "M2", "M3", "M4"],
    resample_interval="10min",
)

# Build refined implicit Crank-Nicolson grid (dz = 0.05 m, theta = 0.5)
dt_seconds = median_time_step_seconds(times)
grid = HeatGrid(depths_m=[0.25, 0.50, 0.75, 1.00], dt_seconds=dt_seconds, cells_per_gap=5, theta=0.5)

# Generate 3-hour windows with 1-hour stride and solve
windows = make_window_indices(times, window_hours=3.0, stride_hours=1.0, min_points=4)
reg = TikhonovRegularization(beta=1.0e12, prior=1.0e-6)

result = solve_adjoint_gradient_descent(
    temps[windows[0]],
    grid,
    alpha_initial=1.0e-6,
    bounds=(1.0e-8, 1.0e-5),
    reg=reg,
)
print(f"Estimated alpha = {result['alpha']:.3e} m^2/s (cost = {result['cost']:.4f})")
```

#### 2. Continuous Time-Dependent $\alpha(t)$ Inversion
```python
from soil_inverse.timevarying import solve_time_dependent_alpha

res = solve_time_dependent_alpha(
    temps,
    grid,
    times,
    control_hours=3.0,
    beta_smooth=1.0,
    beta_prior=0.1,
    alpha_prior=1.0e-6,
    bounds=(1.0e-8, 1.0e-5),
)
# res.alpha_t contains the time-resolved diffusivity at each sample step
```
