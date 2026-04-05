//! # ripopt — Rust Interior Point Optimizer
//!
//! ripopt is a primal-dual interior point solver for nonlinear programming (NLP) problems:
//!
//! ```text
//! min  f(x)
//!  x
//! s.t. g_l ≤ g(x) ≤ g_u
//!      x_l ≤  x   ≤ x_u
//! ```
//!
//! It closely follows the algorithm of [Ipopt](https://coin-or.github.io/Ipopt/) and achieves
//! comparable or better solve rates on standard benchmark suites (HS120, CUTEst).
//!
//! ## Quick Start
//!
//! Implement [`NlpProblem`] for your problem, then call [`solve`]:
//!
//! ```rust,no_run
//! use ripopt::{NlpProblem, SolveResult, SolverOptions, SolveStatus, solve};
//!
//! struct MyProblem;
//!
//! impl NlpProblem for MyProblem {
//!     fn num_variables(&self) -> usize { 2 }
//!     fn num_constraints(&self) -> usize { 0 }
//!     fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
//!         x_l.fill(f64::NEG_INFINITY);
//!         x_u.fill(f64::INFINITY);
//!     }
//!     fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
//!     fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.5; x0[1] = 0.5; }
//!     fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
//!         *obj = (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
//!         true
//!     }
//!     fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
//!         grad[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2));
//!         grad[1] = 200.0 * (x[1] - x[0].powi(2));
//!         true
//!     }
//!     fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
//!     fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
//!     fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
//!     fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
//!         (vec![0, 1, 1], vec![0, 0, 1])
//!     }
//!     fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
//!         vals[0] = obj_factor * (2.0 - 400.0 * (x[1] - x[0].powi(2)) + 800.0 * x[0].powi(2));
//!         vals[1] = obj_factor * (-400.0 * x[0]);
//!         vals[2] = obj_factor * 200.0;
//!         true
//!     }
//! }
//!
//! let result = solve(&MyProblem, &SolverOptions::default());
//! assert_eq!(result.status, SolveStatus::Optimal);
//! ```
//!
//! ## Key Types
//!
//! - [`NlpProblem`] — trait to implement for your problem
//! - [`SolverOptions`] — all solver tuning parameters
//! - [`SolveResult`] — solution, status, and diagnostics
//! - [`SolveStatus`] — outcome (`Optimal`, `LocalInfeasibility`, etc.)
//!
//! ## Algorithm
//!
//! The solver implements:
//! - Mehrotra predictor-corrector IPM with Gondzio corrections
//! - Filter line search with second-order corrections
//! - Gauss-Newton and NLP restoration phases
//! - Fallback cascade: IPM → L-BFGS → Augmented Lagrangian → SQP → slack reformulation
//! - Sparse (multifrontal LDL^T) and dense (Bunch-Kaufman LDL^T) linear solvers
//! - Parametric sensitivity analysis ([`solve_with_sensitivity`])

pub mod augmented_lagrangian;
pub mod intermediate;
pub(crate) mod logging;
pub mod c_api;
pub mod convergence;
pub mod filter;
pub mod ipm;
pub mod kkt;
pub mod lbfgs;
pub mod linearity;
pub mod linear_solver;
pub mod nl;
pub mod options;
pub mod preprocessing;
pub mod problem;
pub mod restoration;
pub mod restoration_nlp;
pub mod result;
pub mod sensitivity;
pub mod slack_formulation;
pub mod sqp;
pub mod warmstart;

pub use options::{SolverOptions, LinearSolverChoice};
pub use problem::NlpProblem;
pub use result::{SolveResult, SolverDiagnostics, SolveStatus};
pub use sensitivity::{ParametricNlpProblem, SensitivityContext, SensitivityResult};

/// Solve a nonlinear programming problem using the interior point method.
pub fn solve<P: NlpProblem>(problem: &P, options: &SolverOptions) -> SolveResult {
    ipm::solve(problem, options)
}

/// Solve and retain factored KKT for parametric sensitivity analysis.
pub fn solve_with_sensitivity<P: ParametricNlpProblem>(
    problem: &P,
    options: &SolverOptions,
) -> SensitivityContext {
    sensitivity::solve_with_sensitivity(problem, options)
}
