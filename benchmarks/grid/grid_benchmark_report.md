# AC Optimal Power Flow Benchmark Report

This report describes a suite of 4 AC Optimal Power Flow (ACOPF) benchmark problems from the PGLib-OPF library and evaluates the performance of **ripopt** against **Ipopt**.

## Source

All test cases come from the **Power Grid Library for Benchmarking AC Optimal Power Flow Algorithms** (PGLib-OPF):

- **Repository**: https://github.com/power-grid-lib/pglib-opf
- **Paper**: S. Babaeinejadsarookolaee et al., "The Power Grid Library for Benchmarking AC Optimal Power Flow Algorithms," arXiv:1908.02788, 2019.
- **License**: Creative Commons Attribution
- **Reference solutions**: Computed with Ipopt (via PowerModels.jl) and published at https://lanl-ansi.github.io/PowerModels.jl/stable/experiment-results/

PGLib-OPF is the standard benchmark for ACOPF solver comparison. It provides 74 base cases (3 to 78,484 buses) in MATPOWER format, each with three operating condition variants (typical, congested, small-angle-difference), totaling 222 instances with known Ipopt solutions.

## Mathematical Formulation

The ACOPF minimizes total generation cost subject to AC power flow physics and engineering limits. The polar-coordinate formulation used here is:

### Variables

For a network with $N$ buses, $G$ generators, and $L$ branches:

$$\mathbf{x} = [V_1, \ldots, V_N, \theta_1, \ldots, \theta_N, P_{g_1}, \ldots, P_{g_G}, Q_{g_1}, \ldots, Q_{g_G}]$$

- $V_i$: voltage magnitude at bus $i$ (per unit)
- $\theta_i$: voltage angle at bus $i$ (radians)
- $P_{g_k}$: real power output of generator $k$ (per unit)
- $Q_{g_k}$: reactive power output of generator $k$ (per unit)

Total: $n = 2N + 2G$ variables.

### Objective

Minimize total generation cost (quadratic):

$$\min \sum_{k=1}^{G} \left( c_{2k} \cdot (S_b P_{g_k})^2 + c_{1k} \cdot S_b P_{g_k} + c_{0k} \right)$$

where $S_b$ is the base MVA and $c_{2k}, c_{1k}, c_{0k}$ are polynomial cost coefficients in \$/MW$^2$h, \$/MWh, and \$/h respectively.

### Equality Constraints

Power balance at each bus $i$ (2$N$ equations):

**Real power:**

$$\sum_{k \in \mathcal{G}_i} P_{g_k} - P_{d_i} = V_i \sum_{j=1}^{N} V_j \left( G_{ij} \cos(\theta_i - \theta_j) + B_{ij} \sin(\theta_i - \theta_j) \right)$$

**Reactive power:**

$$\sum_{k \in \mathcal{G}_i} Q_{g_k} - Q_{d_i} = V_i \sum_{j=1}^{N} V_j \left( G_{ij} \sin(\theta_i - \theta_j) - B_{ij} \cos(\theta_i - \theta_j) \right)$$

where $G_{ij} + jB_{ij}$ are elements of the bus admittance matrix $\mathbf{Y}$, $P_{d_i}$ and $Q_{d_i}$ are loads, and $\mathcal{G}_i$ is the set of generators at bus $i$.

### Inequality Constraints

**Branch apparent power flow limits** (from both ends of each line):

$$P_{ft}^2 + Q_{ft}^2 \leq S_{\max}^2, \quad P_{tf}^2 + Q_{tf}^2 \leq S_{\max}^2$$

where $P_{ft}$ and $Q_{ft}$ are the real and reactive power flows computed from the $\pi$-equivalent branch model, accounting for series impedance, line charging, and transformer tap ratios.

### Variable Bounds

$$V_i^{\min} \leq V_i \leq V_i^{\max}, \quad P_{g_k}^{\min} \leq P_{g_k} \leq P_{g_k}^{\max}, \quad Q_{g_k}^{\min} \leq Q_{g_k} \leq Q_{g_k}^{\max}$$

The reference bus angle is fixed: $\theta_{\text{ref}} = 0$.

### Admittance Matrix

The bus admittance matrix $\mathbf{Y} = \mathbf{G} + j\mathbf{B}$ is constructed from branch data. For each branch from bus $f$ to bus $t$ with series impedance $r + jx$, line charging susceptance $b_c$, and transformer tap ratio $\tau$:

$$y_s = \frac{1}{r + jx}, \quad Y_{ff} \mathrel{+}= \frac{y_s + jb_c/2}{\tau^2}, \quad Y_{tt} \mathrel{+}= y_s + jb_c/2, \quad Y_{ft} = Y_{tf} \mathrel{+}= \frac{-y_s}{\tau}$$

Bus shunt admittances $(G_{s_i} + jB_{s_i})/S_b$ are added to the diagonal.

### Key Properties

- **Nonconvex**: The power flow equations contain bilinear and trigonometric terms, making ACOPF NP-hard in general. Multiple local optima may exist.
- **Sparse**: The constraint Jacobian sparsity follows the network topology.
- **Well-scaled**: Per-unit normalization keeps most quantities $O(1)$.

## Test Cases

| Case        | Buses | Generators | Branches | Variables | Constraints | Jacobian nnz | Known Optimal (\$/h) |
|-------------|-------|------------|----------|-----------|-------------|--------------|----------------------|
| case3_lmbd  | 3     | 3          | 3        | 12        | 12          | 66           | 5,812.64             |
| case5_pjm   | 5     | 5          | 6        | 20        | 22          | 126          | 17,551.89            |
| case14_ieee | 14    | 5          | 20       | 38        | 68          | 386          | 2,178.08             |
| case30_ieee | 30    | 6          | 41       | 72        | 142         | 788          | 8,081.52             |

### case3_lmbd (3-bus Lavaei-Low)

The smallest PGLib-OPF case. Three generators serve 315 MW of load across 3 buses connected in a triangle. The branch from bus 3 to bus 2 has a tight 50 MVA flow limit that makes the problem nontrivial despite its small size. Generator cost functions are quadratic with $c_2 = 0.11$ and $0.085$ \$/MW$^2$h. Generator 3 has zero cost and zero capacity (synchronous condenser).

### case5_pjm (5-bus PJM)

Five buses with 5 generators serving 1000 MW of total load. All generator cost functions are linear (zero $c_2$). Multiple branches have binding flow limits (400, 426, and 240 MVA). This tests the solver's handling of active inequality constraints.

### case14_ieee (IEEE 14-bus)

The classic IEEE 14-bus test system. Two real generators (buses 1 and 2) plus three synchronous condensers (buses 3, 6, 8) that provide only reactive power ($P_{\max} = 0$). Three transformer branches with non-unity tap ratios (0.978, 0.969, 0.932). Bus 9 has a 19 MVAr shunt capacitor. The cost function is linear in $P_g$.

### case30_ieee (IEEE 30-bus)

The IEEE 30-bus system with 6 generators (2 real, 4 synchronous condensers) and 41 branches including 7 transformers. The network spans three voltage levels (132, 33, and 11 kV). Two buses have shunt compensation (bus 10: 19 MVAr, bus 24: 4.3 MVAr). Many branches have tight flow limits (20--30 MVA), creating a heavily constrained problem. Known to have multiple local optima.

## Benchmark Results

Both solvers run at tolerance $10^{-6}$ with a maximum of 3000 iterations, in release mode on Apple M3 Max.

```
AC Optimal Power Flow Benchmark: ripopt vs ipopt
================================================

Problem                 n    m  nnz |   ripopt obj  iter  time(s) |    ipopt obj  iter  time(s)
-----------------------------------------------------------------------------------------------
case3_lmbd             12   12   66 |      5812.64    10   0.0021 |      5812.64    10   0.0053
case5_pjm              20   22  126 |     17551.89    20   0.0031 |     17551.89    15   0.0066
case14_ieee            38   68  386 |      2178.08    14   0.0144 |      2178.08    11   0.0110
case30_ieee            72  142  788 |      8208.52    19   0.0378 |      8208.52    14   0.0522
-----------------------------------------------------------------------------------------------
```

## Performance Comparison

### Accuracy

| Case        | Known Optimal | ripopt    | Gap    | Ipopt     | Gap    |
|-------------|---------------|-----------|--------|-----------|--------|
| case3_lmbd  | 5,812.64      | 5,812.64  | 0.000% | 5,812.64  | 0.000% |
| case5_pjm   | 17,551.89     | 17,551.89 | 0.000% | 17,551.89 | 0.000% |
| case14_ieee | 2,178.08      | 2,178.08  | 0.000% | 2,178.08  | 0.000% |
| case30_ieee | 8,081.52      | 8,208.52  | 1.57%  | 8,208.52  | 1.57%  |

On cases 3, 5, and 14, both solvers reach the published global optimum to within solver tolerance. case30_ieee is a well-known multi-local-optimum problem in PGLib-OPF: at v0.8.1 both solvers converge to the same local optimum at 8,208.52 \$/h, 1.57% above the best-known optimum (8,081.52 \$/h obtained by PowerModels.jl with a different formulation and solver configuration). The v0.7.0 `MaxIterations` regression on case30_ieee — caused by removing the `n+m ≥ 100` shortcut in `factor_with_inertia_correction` (commit 66bce53) — has been resolved by the v0.8 IPM-alignment work.

### Iterations

| Case        | ripopt | Ipopt |
|-------------|--------|-------|
| case3_lmbd  | 10     | 10    |
| case5_pjm   | 20     | 15    |
| case14_ieee | 14     | 11    |
| case30_ieee | 19     | 14    |

ripopt converges in a comparable iteration count on all four cases (within 5 iterations of Ipopt). On case3_lmbd both solvers take 10 iterations; on the other three ripopt uses a handful more.

### Wall Time

| Case        | ripopt  | Ipopt   | Ratio       |
|-------------|---------|---------|-------------|
| case3_lmbd  | 2.1 ms  | 5.3 ms  | 2.5x faster |
| case5_pjm   | 3.1 ms  | 6.6 ms  | 2.1x faster |
| case14_ieee | 14.4 ms | 11.0 ms | 0.8x        |
| case30_ieee | 37.8 ms | 52.2 ms | 1.4x faster |

ripopt is 1.4–2.5x faster on three of four cases; case14_ieee is the lone Ipopt win (0.8x). Geometric mean across all four is 1.5x faster (median 2.1x).

### Convergence Status

At v0.8.1, ripopt reaches strict `Optimal` on all four cases (4/4, 100%), matching Ipopt. The v0.7.0 `MaxIterations` regression on case30_ieee is resolved.

## Implementation Notes

- **Derivatives**: The power balance constraint Hessian is computed analytically. The branch flow limit constraint Hessian uses numerical finite differences on the analytical Jacobian (each flow constraint depends on only 4 variables, so the numerical Hessian is a 4$\times$4 block).
- **Sparsity**: The Jacobian uses a precomputed sparse structure matching the network topology. The Hessian uses dense lower-triangle storage, which is adequate for these small systems but would need to be replaced with sparse storage for cases beyond ~100 buses.
- **Initial point**: Flat start ($V_i = 1.0$ p.u., $\theta_i = 0$, generators at midpoint of bounds).
- **Per-unit system**: All power quantities are normalized by $S_b = 100$ MVA. The cost function accounts for this scaling.

## Conclusions

1. **ripopt solves all 4 ACOPF benchmarks at v0.8.1** and matches the published global optimum on cases 3, 5, and 14. On case30_ieee both solvers converge to the same local optimum (8,208.52 \$/h, 1.57% above the best-known). The v0.7.0 `MaxIterations` regression on case30_ieee is resolved by the v0.8 IPM-alignment work.

2. **Nonconvexity matters**: case30_ieee is a known multi-local-optimum problem. ripopt at v0.6.2 found a different local optimum (8,609.66 \$/h, 4.6% gap); at v0.7.0 the stricter factorization acceptance exhausted the iteration budget; at v0.8.1 ripopt converges to the same local optimum as Ipopt.

3. **Per-iteration overhead favors ripopt on small problems** (cases 3, 5, 30), while Ipopt edges out ripopt on case14_ieee.

4. **The ACOPF problem structure** --- sparse Jacobian/Hessian, trigonometric nonlinearities, binding inequality constraints --- exercises different solver capabilities than the electrolyte problems and provides a complementary benchmark.

5. **Scaling to larger systems** (118+ buses) would require sparse Hessian storage and is a natural next step. The PGLib-OPF library provides cases up to 78,484 buses for stress testing.
