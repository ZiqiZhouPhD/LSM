# Evaluation of Longstaff-Schwartz Monte Carlo Methods for American Option Pricing

This repository contains a numerical study of Longstaff-Schwartz Monte Carlo (LSM) methods for pricing American-style options based on the memory-efficient variant proposed by Gustafsson (2015). Then, we further studied possible improvements to the algorithm using several computational and mathematical techniques.

The project was originally developed in an academic setting and has been restructured here as a self-contained numerical experiment and reference implementation. The emphasis is on algorithmic behavior, numerical accuracy, and diagnostic analysis rather than production-ready pricing infrastructure.

---

## Scope and Objectives

The goals of this project are to:
- Implement Gustafsson’s Brownian bridge variant of the LSM algorithm
- Compare it against a standard LSM implementation
- Study numerical behavior with respect to:
  - time discretization,
  - Monte Carlo sampling error,
  - regression instability near the exercise boundary
- Analyze the structure and stability of simulated exercise boundaries
- Improving the performance of LSM with
  - alternative regression schemes (e.g. isotonic regression)
  - control variates

The project focuses primarily on American put options under the Black–Scholes framework, with additional experiments on American calls.

---

## Methods

The following components are implemented and evaluated:

- **Path simulation**
  - Geometric Brownian motion under the risk-neutral measure
  - Backward path construction (Brownian bridge–style) for memory efficiency

- **Longstaff-Schwartz Monte Carlo (LSM)**
  - Regression on in-the-money paths
  - Polynomial / Laguerre-style basis functions
  - Comparison between pre-generated paths and backward-generated paths

- **Exercise boundary diagnostics**
  - Empirical extraction of exercise boundaries from simulated decisions
  - Visualization of boundary behavior across time
  - Analysis of instability near expiration and far in-the-money regions

- **Alternative regression strategies (exploratory)**
  - Centered isotonic regression
  - Dynamic exercise boundary estimation
  - Controlled variate using European option prices

---

## Key Findings

- Gustafsson’s LSM achieves **memory usage independent of the number of time steps**, while producing prices consistent with standard LSM implementations.
- Numerical variation in estimated prices is dominated by **Monte Carlo sampling error**, rather than regression-induced bias.
- Standard regression-based LSM may exhibit **unphysical multiple crossings** between continuation and exercise values, leading to unstable exercise decisions at the path level.
- Alternative regression approaches (e.g., isotonic regression) improve the qualitative structure of the exercise boundary but do **not materially improve pricing accuracy**.
- For American calls under Black–Scholes assumptions, LSM recovers prices consistent with analytic results, while still exhibiting regression artifacts in the simulated exercise boundary.

The `report/` directory contains a [PDF report](report/LSM_report.pdf) with full derivations, figures, and discussion of results.

---

## Repository Structure

```

src/            Core LSM implementations
experiments/    Scripts producing figures and numerical comparisons
report/         Project report detailing methodology and results

```

---

## Notes

* This project is intended for **numerical analysis and research demonstration**, not for live trading or production pricing.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.