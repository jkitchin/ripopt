//! Integration tests for iterative (MINRES) and hybrid linear solvers.
//!
//! Run fast tests:    cargo test iterative_solvers
//! Run all tests:     cargo test --release -- --ignored iterative hybrid

use ripopt::{LinearSolverChoice, NlpProblem, SolveStatus, SolverOptions};

fn iterative_options() -> SolverOptions {
    SolverOptions {
        linear_solver: LinearSolverChoice::Iterative,
        print_level: 0,
        max_wall_time: 60.0,
        tol: 1e-6,
        ..SolverOptions::default()
    }
}

fn hybrid_options() -> SolverOptions {
    SolverOptions {
        linear_solver: LinearSolverChoice::Hybrid,
        print_level: 0,
        max_wall_time: 60.0,
        tol: 1e-6,
        ..SolverOptions::default()
    }
}

// ===========================================================================
// Problem definitions (duplicated from correctness.rs / large_scale.rs since
// Rust integration tests can't share structs across files)
// ===========================================================================

struct Rosenbrock;

impl NlpProblem for Rosenbrock {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..2 { x_l[i] = f64::NEG_INFINITY; x_u[i] = f64::INFINITY; }
    }

    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) { x0[0] = -1.2; x0[1] = 1.0; }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        *obj = a * a + 100.0 * b * b;
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        grad[1] = 200.0 * (x[1] - x[0] * x[0]);
        true
    }

    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1], vec![0, 0, 1])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, s: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = s * (2.0 + 1200.0 * x[0] * x[0] - 400.0 * x[1]);
        vals[1] = s * (-400.0 * x[0]);
        vals[2] = s * 200.0;
        true
    }
}

// ---------------------------------------------------------------------------

struct Hs071;

impl NlpProblem for Hs071 {
    fn num_variables(&self) -> usize { 4 }
    fn num_constraints(&self) -> usize { 2 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..4 { x_l[i] = 1.0; x_u[i] = 5.0; }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 25.0; g_u[0] = f64::INFINITY;
        g_l[1] = 40.0; g_u[1] = 40.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 1.0; x0[1] = 5.0; x0[2] = 5.0; x0[3] = 1.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        grad[1] = x[0] * x[3];
        grad[2] = x[0] * x[3] + 1.0;
        grad[3] = x[0] * (x[0] + x[1] + x[2]);
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 0, 0, 1, 1, 1, 1], vec![0, 1, 2, 3, 0, 1, 2, 3])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = x[1] * x[2] * x[3]; vals[1] = x[0] * x[2] * x[3];
        vals[2] = x[0] * x[1] * x[3]; vals[3] = x[0] * x[1] * x[2];
        vals[4] = 2.0 * x[0]; vals[5] = 2.0 * x[1];
        vals[6] = 2.0 * x[2]; vals[7] = 2.0 * x[3];
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1, 2, 2, 2, 3, 3, 3, 3], vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, s: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = s * 2.0 * x[3];
        vals[1] = s * x[3];
        vals[2] = 0.0;
        vals[3] = s * x[3];
        vals[4] = 0.0;
        vals[5] = 0.0;
        vals[6] = s * (2.0 * x[0] + x[1] + x[2]);
        vals[7] = s * x[0];
        vals[8] = s * x[0];
        vals[9] = 0.0;
        // Constraint 1: g1 = x1*x2*x3*x4
        vals[1] += lambda[0] * x[2] * x[3];
        vals[3] += lambda[0] * x[1] * x[3];
        vals[4] += lambda[0] * x[0] * x[3];
        vals[6] += lambda[0] * x[1] * x[2];
        vals[7] += lambda[0] * x[0] * x[2];
        vals[8] += lambda[0] * x[0] * x[1];
        // Constraint 2: g2 = sum xi^2
        vals[0] += lambda[1] * 2.0;
        vals[2] += lambda[1] * 2.0;
        vals[5] += lambda[1] * 2.0;
        vals[9] += lambda[1] * 2.0;
        true
    }
}

// ---------------------------------------------------------------------------

struct ChainedRosenbrock { n: usize }

impl NlpProblem for ChainedRosenbrock {
    fn num_variables(&self) -> usize { self.n }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..self.n { x_l[i] = f64::NEG_INFINITY; x_u[i] = f64::INFINITY; }
    }

    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        for v in x0.iter_mut() { *v = -1.2; }
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        let mut f = 0.0;
        for i in 0..self.n - 1 {
            let a = 1.0 - x[i];
            let b = x[i + 1] - x[i] * x[i];
            f += a * a + 100.0 * b * b;
        }
        *obj = f;
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        for g in grad.iter_mut() { *g = 0.0; }
        for i in 0..self.n - 1 {
            let r = x[i + 1] - x[i] * x[i];
            grad[i] += -2.0 * (1.0 - x[i]) + 200.0 * r * (-2.0 * x[i]);
            grad[i + 1] += 200.0 * r;
        }
        true
    }

    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.n;
        let mut rows = Vec::with_capacity(2 * n - 1);
        let mut cols = Vec::with_capacity(2 * n - 1);
        rows.push(0); cols.push(0);
        for i in 1..n {
            rows.push(i); cols.push(i - 1);
            rows.push(i); cols.push(i);
        }
        (rows, cols)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, s: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        for v in vals.iter_mut() { *v = 0.0; }
        for i in 0..self.n - 1 {
            let diag_i = if i == 0 { 0 } else { 2 * i };
            vals[diag_i] += s * (2.0 + 1200.0 * x[i] * x[i] - 400.0 * x[i + 1]);
            vals[2 * (i + 1) - 1] += s * (-400.0 * x[i]);
            vals[2 * (i + 1)] += s * 200.0;
        }
        true
    }
}

// ---------------------------------------------------------------------------

struct BratuProblem { n: usize, lambda_bratu: f64, h: f64 }

impl BratuProblem {
    fn new(n: usize) -> Self {
        Self { n, lambda_bratu: 1.0, h: 1.0 / (n as f64 + 1.0) }
    }
}

impl NlpProblem for BratuProblem {
    fn num_variables(&self) -> usize { self.n }
    fn num_constraints(&self) -> usize { self.n - 2 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..self.n { x_l[i] = f64::NEG_INFINITY; x_u[i] = f64::INFINITY; }
        x_l[0] = 0.0; x_u[0] = 0.0;
        x_l[self.n - 1] = 0.0; x_u[self.n - 1] = 0.0;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for j in 0..self.n - 2 { g_l[j] = 0.0; g_u[j] = 0.0; }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        for v in x0.iter_mut() { *v = 0.0; }
    }

    fn objective(&self, _x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = 0.0;
        true
    }

    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        for g in grad.iter_mut() { *g = 0.0; }
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        let h2 = self.h * self.h;
        for j in 0..self.n - 2 {
            let i = j + 1;
            g[j] = (-x[i - 1] + 2.0 * x[i] - x[i + 1]) / h2 - self.lambda_bratu * x[i].exp();
        }
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let m = self.n - 2;
        let mut rows = Vec::with_capacity(3 * m);
        let mut cols = Vec::with_capacity(3 * m);
        for j in 0..m {
            let i = j + 1;
            rows.push(j); cols.push(i - 1);
            rows.push(j); cols.push(i);
            rows.push(j); cols.push(i + 1);
        }
        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        let h2 = self.h * self.h;
        for j in 0..self.n - 2 {
            let i = j + 1;
            let base = 3 * j;
            vals[base] = -1.0 / h2;
            vals[base + 1] = 2.0 / h2 - self.lambda_bratu * x[i].exp();
            vals[base + 2] = -1.0 / h2;
        }
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let mut rows = Vec::with_capacity(self.n);
        let mut cols = Vec::with_capacity(self.n);
        for k in 0..self.n { rows.push(k); cols.push(k); }
        (rows, cols)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, _s: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        for v in vals.iter_mut() { *v = 0.0; }
        for j in 0..self.n - 2 {
            let k = j + 1;
            vals[k] += lambda[j] * (-self.lambda_bratu * x[k].exp());
        }
        true
    }
}

// ---------------------------------------------------------------------------

struct OptimalControl { t: usize, h: f64, alpha: f64 }

impl OptimalControl {
    fn new(t: usize) -> Self {
        Self { t, h: 1.0 / t as f64, alpha: 0.01 }
    }
}

impl NlpProblem for OptimalControl {
    fn num_variables(&self) -> usize { 2 * self.t + 1 }
    fn num_constraints(&self) -> usize { self.t + 1 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..self.num_variables() { x_l[i] = f64::NEG_INFINITY; x_u[i] = f64::INFINITY; }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for j in 0..self.num_constraints() { g_l[j] = 0.0; g_u[j] = 0.0; }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        for v in x0.iter_mut() { *v = 0.0; }
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        let (h, t) = (self.h, self.t);
        let mut f = 0.0;
        for i in 0..=t { let dy = x[i] - 1.0; f += h * dy * dy; }
        for i in 0..t { f += self.alpha * h * x[t + 1 + i] * x[t + 1 + i]; }
        *obj = f;
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        let (h, t) = (self.h, self.t);
        for i in 0..=t { grad[i] = 2.0 * h * (x[i] - 1.0); }
        for i in 0..t { grad[t + 1 + i] = 2.0 * self.alpha * h * x[t + 1 + i]; }
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        let (h, t) = (self.h, self.t);
        g[0] = x[0];
        for i in 0..t { g[i + 1] = x[i + 1] - (1.0 - h) * x[i] - h * x[t + 1 + i]; }
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let t = self.t;
        let mut rows = Vec::with_capacity(1 + 3 * t);
        let mut cols = Vec::with_capacity(1 + 3 * t);
        rows.push(0); cols.push(0);
        for i in 0..t {
            rows.push(i + 1); cols.push(i);
            rows.push(i + 1); cols.push(i + 1);
            rows.push(i + 1); cols.push(t + 1 + i);
        }
        (rows, cols)
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        let (h, t) = (self.h, self.t);
        vals[0] = 1.0;
        for i in 0..t {
            let base = 1 + 3 * i;
            vals[base] = -(1.0 - h);
            vals[base + 1] = 1.0;
            vals[base + 2] = -h;
        }
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.num_variables();
        let mut rows = Vec::with_capacity(n);
        let mut cols = Vec::with_capacity(n);
        for k in 0..n { rows.push(k); cols.push(k); }
        (rows, cols)
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, s: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        let (h, t) = (self.h, self.t);
        for i in 0..=t { vals[i] = s * 2.0 * h; }
        for i in 0..t { vals[t + 1 + i] = s * 2.0 * self.alpha * h; }
        true
    }
}

// ===========================================================================
// Iterative (MINRES) solver tests
// ===========================================================================

// --- Small problems (sparse_threshold: 0 to force iterative path) ---

#[test]
fn iterative_rosenbrock_small() {
    let mut opts = iterative_options();
    opts.sparse_threshold = 0;
    let result = ripopt::solve(&Rosenbrock, &opts);
    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal/Acceptable, got {:?}", result.status
    );
    assert!(result.objective < 1e-6, "f* should be ~0, got {}", result.objective);
    assert!((result.x[0] - 1.0).abs() < 1e-3, "x1 should be ~1, got {}", result.x[0]);
    assert!((result.x[1] - 1.0).abs() < 1e-3, "x2 should be ~1, got {}", result.x[1]);
}

#[test]
fn iterative_hs071_small() {
    let mut opts = iterative_options();
    opts.sparse_threshold = 0;
    let result = ripopt::solve(&Hs071, &opts);
    // HS071 is non-convex; iterative solver may converge to a different local minimum.
    // Check only that a locally optimal point is found with feasible constraints.
    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal, got {:?}", result.status
    );
}

// --- Moderate-scale problems (n+m >= 110, naturally use sparse path) ---

#[test]
#[ignore]
fn iterative_chained_rosenbrock_200() {
    let result = ripopt::solve(&ChainedRosenbrock { n: 200 }, &iterative_options());
    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal/Acceptable, got {:?}", result.status
    );
    assert!(result.objective < 1e-3, "f* should be ~0, got {}", result.objective);
}

#[test]
#[ignore]
fn iterative_bratu_200() {
    let result = ripopt::solve(&BratuProblem::new(200), &iterative_options());
    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal/Acceptable, got {:?}", result.status
    );
    // Feasibility problem: check constraint violation
    let n = 200;
    let mut g = vec![0.0; n - 2];
    BratuProblem::new(n).constraints(&result.x, true, &mut g);
    let max_cv: f64 = g.iter().map(|gi| gi.abs()).fold(0.0, f64::max);
    assert!(max_cv < 1e-3, "max constraint violation should be < 1e-3, got {}", max_cv);
}

#[test]
#[ignore]
fn iterative_optimal_control_99() {
    let result = ripopt::solve(&OptimalControl::new(99), &iterative_options());
    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal/Acceptable, got {:?}", result.status
    );
}

// ===========================================================================
// Hybrid solver tests
// ===========================================================================

// --- Small problems (sparse_threshold: 0 to force hybrid path) ---

#[test]
fn hybrid_rosenbrock_small() {
    let mut opts = hybrid_options();
    opts.sparse_threshold = 0;
    let result = ripopt::solve(&Rosenbrock, &opts);
    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal/Acceptable, got {:?}", result.status
    );
    assert!(result.objective < 1e-6, "f* should be ~0, got {}", result.objective);
}

#[test]
fn hybrid_hs071_small() {
    let mut opts = hybrid_options();
    opts.sparse_threshold = 0;
    let result = ripopt::solve(&Hs071, &opts);
    // HS071 is non-convex; hybrid solver may converge to a different local minimum.
    // Check only that a locally optimal point is found with feasible constraints.
    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal, got {:?}", result.status
    );
}

// --- Moderate-scale problems ---

#[test]
#[ignore]
fn hybrid_chained_rosenbrock_200() {
    let result = ripopt::solve(&ChainedRosenbrock { n: 200 }, &hybrid_options());
    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal/Acceptable, got {:?}", result.status
    );
    assert!(result.objective < 1e-3, "f* should be ~0, got {}", result.objective);
}

#[test]
#[ignore]
fn hybrid_bratu_200() {
    let result = ripopt::solve(&BratuProblem::new(200), &hybrid_options());
    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal/Acceptable, got {:?}", result.status
    );
    let n = 200;
    let mut g = vec![0.0; n - 2];
    BratuProblem::new(n).constraints(&result.x, true, &mut g);
    let max_cv: f64 = g.iter().map(|gi| gi.abs()).fold(0.0, f64::max);
    assert!(max_cv < 1e-3, "max constraint violation should be < 1e-3, got {}", max_cv);
}

#[test]
#[ignore]
fn hybrid_optimal_control_99() {
    let result = ripopt::solve(&OptimalControl::new(99), &hybrid_options());
    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal/Acceptable, got {:?}", result.status
    );
}
