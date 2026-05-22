# ripopt

[![Tests](https://github.com/jkitchin/ripopt/actions/workflows/test.yml/badge.svg)](https://github.com/jkitchin/ripopt/actions/workflows/test.yml)

A memory-safe interior point optimizer written in Rust, inspired by [Ipopt](https://github.com/coin-or/Ipopt).

## What is ripopt?

ripopt solves nonlinear programming (NLP) problems of the form:

```
min  f(x)
s.t. g_l <= g(x) <= g_u
     x_l <= x    <= x_u
```

It implements a primal-dual interior point method (IPM) with a logarithmic barrier formulation. The solver is written entirely in Rust (~21,700 lines) with no external C/Fortran dependencies.

## At a glance

| Property | Value |
|---|---|
| Algorithm | Primal-dual IPM with Mehrotra predictor-corrector |
| Linear solver | Dense Bunch-Kaufman LDLᵀ (small) / `feral` multifrontal (large) |
| HS benchmark (retired) | **118/120** (98.3%, historical) — surpassed Ipopt's 116/120 |
| CUTEst benchmark | **551/727** (75.8%, strict-Optimal) — Ipopt edges by 5 (556/727) |
| Speed vs Ipopt | **8.1x** geo mean on CUTEst (median 10.5x) on 529 commonly-Optimal problems |
| Language | Rust (no unsafe FFI) |
| Interfaces | Rust API, C API, Pyomo/AMPL, GAMS, Julia/JuMP |

## Key features

- **Primal-dual IPM** with logarithmic barrier and fraction-to-boundary rule
- **Mehrotra predictor-corrector** with Gondzio centrality corrections (enabled by default)
- **Filter line search** with switching condition and second-order corrections
- **Two-phase restoration**: fast Gauss-Newton + full NLP restoration subproblem
- **Multi-solver fallback**: L-BFGS → Augmented Lagrangian → SQP → slack reformulation
- **Adaptive and monotone barrier strategies** with automatic mode switching
- **Implicit-slack KKT formulation** that accepts NE (nonlinear-equation) systems Ipopt's CUTEst wrapper rejects (12 of 22 ripopt-only CUTEst wins)
- **Parametric sensitivity analysis** (sIPOPT-style)
- **Preprocessing**: fixed variable elimination, redundant constraint removal, bound tightening
- **C API** mirroring Ipopt's interface; Pyomo, GAMS, and Julia/JuMP wrappers

## Quick example

```rust
use ripopt::{NlpProblem, SolveStatus, SolverOptions};

struct Hs071;

impl NlpProblem for Hs071 {
    // ... (see API Guide for full implementation)
}

fn main() {
    let result = ripopt::solve(&Hs071, &SolverOptions::default());
    assert_eq!(result.status, SolveStatus::Optimal);
    println!("f* = {:.6}", result.objective);  // 17.014017
}
```

See the [API Guide](api-guide.md) for a complete walkthrough.
