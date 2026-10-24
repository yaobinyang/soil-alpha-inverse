# Comment 1-3 starting-value test results

Executed 3852 solves across 321 field windows and six starting values.

Fixed beta = 1e12 K^2 s^2 m^-4; prior = 1e-6 m^2/s.

## Coarse scan + L-BFGS-B
- Windows with >1% between-start spread: 0/321.
- Maximum relative spread: 2.44676e-05%.
- Local optimizer unsuccessful: 78/1926 solves, involving 18 windows.
- Lower/upper bound solutions: 6/0.
- Solutions with higher cost than the best observed: 0/1926.
- Specified starting value replaced: 1778/1926.

## Direct L-BFGS-B
- Windows with >1% between-start spread: 2/321.
- Maximum relative spread: 99.3228%.
- Local optimizer unsuccessful: 65/1926 solves, involving 60 windows.
- Lower/upper bound solutions: 10/0.
- Solutions with higher cost than the best observed: 8/1926.
- Specified starting value replaced: 0/1926.

## Interpretation
In this executed comparison, use the counts above to assess robustness. Do not describe every solve as converged when unsuccessful flags are present.
Agreement in A may result from coarse-scan initialization. Compare B before claiming starting-value independence.
Solver success alone does not establish a global minimum. Inspect bound solutions and objective profiles.
The best-observed solution is a numerical comparison reference, not independently known truth.
These tests hold beta and prior fixed; their suitability and sensitivity require separate assessment.
A uses the original solver including its fallback behavior; its success flag refers to the local optimizer.
All relative-spread flags use a descriptive 1% threshold, not a statistical significance test.
