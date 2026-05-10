//! Honest probe: does the ℓ₁ penalty-barrier wrapper actually solve
//! degenerate problems that the unwrapped path fails on?
//!
//! These tests do NOT assert success of either configuration. They
//! print iterations, status, and final residuals so we can read off
//! whether the wrapper helps. If both configurations succeed, the
//! degenerate test isn't degenerate enough. If only the wrapper
//! succeeds, that's the evidence we want.

use ripopt::{solve, NlpProblem, SolveStatus, SolverOptions};

/// REDUNDANT EQUALITIES — Jacobian is rank-deficient by construction.
/// min x0² + x1²
/// s.t. x0 + x1 = 1
///      2(x0 + x1) = 2     (= the same constraint, scaled)
struct RedundantEq;
impl NlpProblem for RedundantEq {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 2 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0; g_u[0] = 1.0;
        g_l[1] = 2.0; g_u[1] = 2.0;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; x0[1] = 0.0; }
    fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool {
        *o = x[0]*x[0] + x[1]*x[1]; true
    }
    fn gradient(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = 2.0*x[0]; g[1] = 2.0*x[1]; true
    }
    fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] + x[1];
        g[1] = 2.0 * (x[0] + x[1]);
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 1, 1], vec![0, 1, 0, 1])
    }
    fn jacobian_values(&self, _: &[f64], _: bool, v: &mut [f64]) -> bool {
        v[0] = 1.0; v[1] = 1.0;
        v[2] = 2.0; v[3] = 2.0;
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }
    fn hessian_values(&self, _: &[f64], _: bool, of: f64, _: &[f64], v: &mut [f64]) -> bool {
        v[0] = 2.0*of; v[1] = 2.0*of; true
    }
}

/// INCONSISTENT EQUALITIES — feasible only in the ℓ₁ sense.
/// min x²
/// s.t. x = 1
///      x = 2
/// No feasible point exists. Vanilla ripopt should declare local
/// infeasibility / restoration failure. The ℓ₁ reformulation should
/// converge to the ℓ₁-optimum: x = 1 or x = 2 (whichever minimizes
/// the penalty). Penalty contribution at x=1: ρ·1 (one slack ≈ 1).
/// At x=2: ρ·1 (other slack ≈ 1). Plus f(x) = 1 vs 4. So ℓ₁-opt is x=1.
struct InconsistentEq;
impl NlpProblem for InconsistentEq {
    fn num_variables(&self) -> usize { 1 }
    fn num_constraints(&self) -> usize { 2 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0; g_u[0] = 1.0;
        g_l[1] = 2.0; g_u[1] = 2.0;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; }
    fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool { *o = x[0]*x[0]; true }
    fn gradient(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool { g[0] = 2.0*x[0]; true }
    fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = x[0]; g[1] = x[0]; true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 0])
    }
    fn jacobian_values(&self, _: &[f64], _: bool, v: &mut [f64]) -> bool {
        v[0] = 1.0; v[1] = 1.0; true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
    fn hessian_values(&self, _: &[f64], _: bool, of: f64, _: &[f64], v: &mut [f64]) -> bool {
        v[0] = 2.0*of; true
    }
}

/// MPCC-LIKE — complementarity by equality.
/// min (x - 1)² + (y - 1)²
/// s.t. x*y = 0
///      x ≥ 0, y ≥ 0
/// Optima: (1, 0) and (0, 1) with f = 1. LICQ fails at every feasible
/// point along the x=0 or y=0 axis (gradient of x*y is (y, x), which
/// is parallel to the active bound's gradient).
struct MpccLike;
impl NlpProblem for MpccLike {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 0.0; x_u[0] = f64::INFINITY;
        x_l[1] = 0.0; x_u[1] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0; g_u[0] = 0.0;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.5; x0[1] = 0.5; }
    fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool {
        *o = (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2); true
    }
    fn gradient(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = 2.0*(x[0] - 1.0); g[1] = 2.0*(x[1] - 1.0); true
    }
    fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] * x[1]; true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0], vec![0, 1])
    }
    fn jacobian_values(&self, x: &[f64], _: bool, v: &mut [f64]) -> bool {
        v[0] = x[1]; v[1] = x[0]; true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // ∇²f diagonal entries + ∇²(x*y) off-diagonal.
        (vec![0, 1, 1], vec![0, 0, 1])
    }
    fn hessian_values(&self, _x: &[f64], _: bool, of: f64, lam: &[f64], v: &mut [f64]) -> bool {
        v[0] = 2.0*of;
        v[1] = lam[0]; // ∂²/∂x∂y of (x*y) = 1
        v[2] = 2.0*of;
        true
    }
}

fn run<P: NlpProblem>(prob: &P, label: &str, l1_on: bool) {
    let mut opts = SolverOptions::default();
    opts.print_level = 0;
    opts.max_iter = 200;
    opts.l1_exact_penalty_barrier = l1_on;
    opts.l1_penalty_init = 1000.0;

    let r = solve(prob, &opts);
    let status_tag = match r.status {
        SolveStatus::Optimal => "Optimal",
        SolveStatus::Acceptable => "Acceptable",
        SolveStatus::Infeasible => "Infeasible",
        SolveStatus::LocalInfeasibility => "LocalInfeas",
        SolveStatus::MaxIterations => "MaxIter",
        SolveStatus::MaxTimeExceeded => "MaxTime",
        SolveStatus::NumericalError => "NumErr",
        SolveStatus::DivergingIterates => "Diverging",
        SolveStatus::RestorationFailed => "RestoFail",
        SolveStatus::EvaluationError => "EvalErr",
        SolveStatus::UserRequestedStop => "UserStop",
        SolveStatus::StopAtTinyStep => "TinyStep",
        SolveStatus::InternalError => "InternalErr",
    };
    let x_str: Vec<String> = r.x.iter().map(|v| format!("{:.4}", v)).collect();
    println!(
        "{:<28} l1={:<5} status={:<11} iters={:>3} f={:>10.4e} x=[{}]",
        label, l1_on, status_tag, r.iterations, r.objective, x_str.join(", ")
    );
}

/// TANGENT INTERSECTION — two equalities tangent at the unique feasible
/// point. Jacobian is rank-deficient at the optimum.
/// min x² + y²
/// s.t. y − x² = 0      (parabola)
///      y = 0           (x-axis)
/// Unique feasible point: (0, 0). Optimum f* = 0.
/// At (0, 0): ∇c1 = (−2x, 1) = (0, 1), ∇c2 = (0, 1) — PARALLEL.
struct TangentEq;
impl NlpProblem for TangentEq {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 2 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0; g_u[0] = 0.0;
        g_l[1] = 0.0; g_u[1] = 0.0;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.5; x0[1] = 0.5; }
    fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool {
        *o = x[0]*x[0] + x[1]*x[1]; true
    }
    fn gradient(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = 2.0*x[0]; g[1] = 2.0*x[1]; true
    }
    fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = x[1] - x[0]*x[0];
        g[1] = x[1];
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // c1: dx0=-2x0, dx1=1; c2: dx1=1
        (vec![0, 0, 1], vec![0, 1, 1])
    }
    fn jacobian_values(&self, x: &[f64], _: bool, v: &mut [f64]) -> bool {
        v[0] = -2.0*x[0]; v[1] = 1.0;
        v[2] = 1.0;
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }
    fn hessian_values(&self, _: &[f64], _: bool, of: f64, lam: &[f64], v: &mut [f64]) -> bool {
        // ∇²f diag = 2; ∇²c1 has -2 on (0,0) from -x0².
        v[0] = 2.0*of + lam[0]*(-2.0);
        v[1] = 2.0*of;
        true
    }
}

/// COMPLEMENTARITY EQUALITY ON BOTH AXES — MPCC-style with the
/// complementarity x·y = 0 and *both* x and y free (no nonneg bounds).
/// Optimum at (0, 1) with f = 0; the constraint Jacobian gradient at
/// the optimum is (y, x) = (1, 0); this is full rank in isolation but
/// the Hessian is indefinite at every feasible point along an axis,
/// which historically traps vanilla in restoration.
/// min x² + (y - 1)²
/// s.t. x · y = 0
///      x, y free
struct AxesComplementarity;
impl NlpProblem for AxesComplementarity {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0; g_u[0] = 0.0;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.5; x0[1] = 0.5; }
    fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool {
        *o = x[0]*x[0] + (x[1] - 1.0).powi(2); true
    }
    fn gradient(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = 2.0*x[0]; g[1] = 2.0*(x[1] - 1.0); true
    }
    fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = x[0]*x[1]; true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0], vec![0, 1])
    }
    fn jacobian_values(&self, x: &[f64], _: bool, v: &mut [f64]) -> bool {
        v[0] = x[1]; v[1] = x[0]; true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1], vec![0, 0, 1])
    }
    fn hessian_values(&self, _x: &[f64], _: bool, of: f64, lam: &[f64], v: &mut [f64]) -> bool {
        v[0] = 2.0*of;
        v[1] = lam[0];
        v[2] = 2.0*of;
        true
    }
}

/// CIRCLE + TANGENT LINE — equalities meeting tangentially.
/// min x² + (y − 1)²
/// s.t. x² + y² − 1 = 0    (unit circle)
///      y − 1 = 0           (horizontal line)
/// Unique feasible point: (0, 1). f* = 0.
/// At (0, 1): ∇c1 = (2x, 2y) = (0, 2), ∇c2 = (0, 1) — PARALLEL.
struct CircleTangent;
impl NlpProblem for CircleTangent {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 2 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0; g_u[0] = 0.0;
        g_l[1] = 0.0; g_u[1] = 0.0;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.7; x0[1] = 0.7; }
    fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool {
        *o = x[0]*x[0] + (x[1] - 1.0).powi(2); true
    }
    fn gradient(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = 2.0*x[0]; g[1] = 2.0*(x[1] - 1.0); true
    }
    fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = x[0]*x[0] + x[1]*x[1] - 1.0;
        g[1] = x[1] - 1.0;
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 1], vec![0, 1, 1])
    }
    fn jacobian_values(&self, x: &[f64], _: bool, v: &mut [f64]) -> bool {
        v[0] = 2.0*x[0]; v[1] = 2.0*x[1];
        v[2] = 1.0;
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }
    fn hessian_values(&self, _x: &[f64], _: bool, of: f64, lam: &[f64], v: &mut [f64]) -> bool {
        v[0] = 2.0*of + 2.0*lam[0];
        v[1] = 2.0*of + 2.0*lam[0];
        true
    }
}

/// WÄCHTER-BIEGLER counterexample (Wächter-Biegler 2000) — the
/// canonical example where filter line search can fail to converge.
/// min x₁
/// s.t. x₁² − x₂ − 1 = 0
///      x₁ − x₃ − 0.5 = 0
///      x₂, x₃ ≥ 0,  x₁ free
/// Starting from x = (−2, 1, 1) the filter LS gets stuck because every
/// step that reduces f increases θ and vice versa — the classic filter
/// stall. The unique optimum is x = ((1+√5)/2, x₁²−1, x₁−0.5) with
/// f* = (1+√5)/2 ≈ 1.618.
struct WachterBiegler;
impl NlpProblem for WachterBiegler {
    fn num_variables(&self) -> usize { 3 }
    fn num_constraints(&self) -> usize { 2 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = 0.0; x_u[1] = f64::INFINITY;
        x_l[2] = 0.0; x_u[2] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0; g_u[0] = 0.0;
        g_l[1] = 0.0; g_u[1] = 0.0;
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = -2.0; x0[1] = 1.0; x0[2] = 1.0;
    }
    fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool {
        *o = x[0]; true
    }
    fn gradient(&self, _x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = 1.0; g[1] = 0.0; g[2] = 0.0; true
    }
    fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = x[0]*x[0] - x[1] - 1.0;
        g[1] = x[0] - x[2] - 0.5;
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // c1: dx0=2x0, dx1=-1; c2: dx0=1, dx2=-1
        (vec![0, 0, 1, 1], vec![0, 1, 0, 2])
    }
    fn jacobian_values(&self, x: &[f64], _: bool, v: &mut [f64]) -> bool {
        v[0] = 2.0*x[0]; v[1] = -1.0;
        v[2] = 1.0;       v[3] = -1.0;
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // ∇²(x₀²) contributes 2 on (0,0); rest is zero (linear).
        (vec![0], vec![0])
    }
    fn hessian_values(&self, _x: &[f64], _: bool, _of: f64, lam: &[f64], v: &mut [f64]) -> bool {
        v[0] = 2.0 * lam[0]; true
    }
}

/// Highly-redundant equalities — 8 copies of the same constraint with
/// random scaling. Makes the Jacobian rank 1 with 8 rows, forcing the
/// perturbation handler to add large δ_c. Vanilla solves but should
/// take more iters than the 1-constraint version.
struct ManyRedundantEqs;
impl NlpProblem for ManyRedundantEqs {
    fn num_variables(&self) -> usize { 3 }
    fn num_constraints(&self) -> usize { 8 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..3 { x_l[i] = f64::NEG_INFINITY; x_u[i] = f64::INFINITY; }
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        // Row k: k * (x0 + x1 + x2) = k for k = 1..=8.
        for k in 0..8 {
            g_l[k] = (k + 1) as f64;
            g_u[k] = (k + 1) as f64;
        }
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0; x0[1] = 0.0; x0[2] = 0.0;
    }
    fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool {
        *o = x[0]*x[0] + x[1]*x[1] + x[2]*x[2]; true
    }
    fn gradient(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        for i in 0..3 { g[i] = 2.0*x[i]; }
        true
    }
    fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        let s = x[0] + x[1] + x[2];
        for k in 0..8 {
            g[k] = (k + 1) as f64 * s;
        }
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let mut r = Vec::new();
        let mut c = Vec::new();
        for k in 0..8 {
            for j in 0..3 { r.push(k); c.push(j); }
        }
        (r, c)
    }
    fn jacobian_values(&self, _x: &[f64], _: bool, v: &mut [f64]) -> bool {
        for k in 0..8 {
            for j in 0..3 { v[k*3 + j] = (k + 1) as f64; }
        }
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 2], vec![0, 1, 2])
    }
    fn hessian_values(&self, _: &[f64], _: bool, of: f64, _: &[f64], v: &mut [f64]) -> bool {
        for i in 0..3 { v[i] = 2.0*of; }
        true
    }
}

/// BURKE-HAN-style degeneracy: equality constraint with **zero**
/// gradient at the unique feasible point. Extreme LICQ failure.
/// min  x₁
/// s.t. x₁² + x₂² = 0
/// Unique feasible point: (0, 0). f* = 0.
/// At (0, 0): ∇c = (2x₁, 2x₂) = (0, 0) — Jacobian is the zero row.
struct BurkeHan;
impl NlpProblem for BurkeHan {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0; g_u[0] = 0.0;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 1.0; x0[1] = 1.0; }
    fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool { *o = x[0]; true }
    fn gradient(&self, _x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = 1.0; g[1] = 0.0; true
    }
    fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = x[0]*x[0] + x[1]*x[1]; true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0], vec![0, 1])
    }
    fn jacobian_values(&self, x: &[f64], _: bool, v: &mut [f64]) -> bool {
        v[0] = 2.0*x[0]; v[1] = 2.0*x[1]; true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }
    fn hessian_values(&self, _: &[f64], _: bool, _of: f64, lam: &[f64], v: &mut [f64]) -> bool {
        v[0] = 2.0 * lam[0]; v[1] = 2.0 * lam[0]; true
    }
}

/// MULTI-ROW BURKE-HAN — same zero-gradient pathology, multiple rows.
/// min  Σ x_i
/// s.t. Σ x_j² = 0   (×3 redundant rows with different scales)
struct MultiBurkeHan;
impl NlpProblem for MultiBurkeHan {
    fn num_variables(&self) -> usize { 4 }
    fn num_constraints(&self) -> usize { 3 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..4 { x_l[i] = f64::NEG_INFINITY; x_u[i] = f64::INFINITY; }
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for k in 0..3 { g_l[k] = 0.0; g_u[k] = 0.0; }
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 1.0; x0[1] = 1.0; x0[2] = 1.0; x0[3] = 1.0;
    }
    fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool {
        *o = x[0] + x[1] + x[2] + x[3]; true
    }
    fn gradient(&self, _x: &[f64], _: bool, g: &mut [f64]) -> bool {
        for i in 0..4 { g[i] = 1.0; } true
    }
    fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        let s = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];
        g[0] = s;
        g[1] = 2.0 * s;
        g[2] = 0.5 * s;
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let mut r = Vec::new();
        let mut c = Vec::new();
        for k in 0..3 { for j in 0..4 { r.push(k); c.push(j); } }
        (r, c)
    }
    fn jacobian_values(&self, x: &[f64], _: bool, v: &mut [f64]) -> bool {
        let scales = [1.0, 2.0, 0.5];
        for k in 0..3 {
            for j in 0..4 {
                v[k*4 + j] = scales[k] * 2.0 * x[j];
            }
        }
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 2, 3], vec![0, 1, 2, 3])
    }
    fn hessian_values(&self, _: &[f64], _: bool, _of: f64, lam: &[f64], v: &mut [f64]) -> bool {
        let scale = 2.0 * (lam[0] + 2.0*lam[1] + 0.5*lam[2]);
        for i in 0..4 { v[i] = scale; }
        true
    }
}

#[test]
fn probe_multi_burke_han() {
    println!("\n--- MultiBurkeHan: 3 redundant zero-gradient equalities ---");
    run(&MultiBurkeHan, "MultiBurkeHan", false);
    run(&MultiBurkeHan, "MultiBurkeHan", true);
}

#[test]
fn probe_burke_han() {
    println!("\n--- BurkeHan: x₁²+x₂²=0, ∇c=0 at unique feasible (0,0) ---");
    run(&BurkeHan, "BurkeHan", false);
    run(&BurkeHan, "BurkeHan", true);
    // Manual high-rho run to see whether higher penalty drives x → 0.
    let mut opts = SolverOptions::default();
    opts.print_level = 0;
    opts.max_iter = 200;
    opts.l1_exact_penalty_barrier = true;
    opts.l1_penalty_init = 1e8;
    opts.l1_penalty_max = 1e12;
    let r = solve(&BurkeHan, &opts);
    println!(
        "BurkeHan @ rho_init=1e8         status={:?} iters={} f={:.4e} x=[{:.4e}, {:.4e}]",
        r.status, r.iterations, r.objective, r.x[0], r.x[1]
    );
}

#[test]
fn probe_many_redundant_eqs() {
    println!("\n--- ManyRedundantEqs: 8 redundant equality rows, rank-1 Jacobian ---");
    run(&ManyRedundantEqs, "ManyRedundantEqs", false);
    run(&ManyRedundantEqs, "ManyRedundantEqs", true);
}

#[test]
fn probe_wachter_biegler() {
    println!("\n--- WachterBiegler: filter-LS canonical failure mode ---");
    run(&WachterBiegler, "WachterBiegler", false);
    run(&WachterBiegler, "WachterBiegler", true);
}

#[test]
fn probe_tangent_equalities() {
    println!("\n--- TangentEq: y=x² and y=0 (tangent at (0,0), LICQ fails) ---");
    run(&TangentEq, "TangentEq", false);
    run(&TangentEq, "TangentEq", true);
}

#[test]
fn probe_axes_complementarity() {
    println!("\n--- AxesComplementarity: x·y=0, free x,y ---");
    run(&AxesComplementarity, "AxesComplementarity", false);
    run(&AxesComplementarity, "AxesComplementarity", true);
}

#[test]
fn probe_circle_tangent() {
    println!("\n--- CircleTangent: x²+y²=1 and y=1 (tangent at (0,1)) ---");
    run(&CircleTangent, "CircleTangent", false);
    run(&CircleTangent, "CircleTangent", true);
}

#[test]
fn probe_redundant_equalities() {
    println!("\n--- RedundantEq: 2 eq rows = same constraint scaled (LICQ fails) ---");
    run(&RedundantEq, "RedundantEq", false);
    run(&RedundantEq, "RedundantEq", true);
}

#[test]
fn probe_inconsistent_equalities() {
    println!("\n--- InconsistentEq: x=1 AND x=2 (no feasible point) ---");
    run(&InconsistentEq, "InconsistentEq", false);
    run(&InconsistentEq, "InconsistentEq", true);
}

#[test]
fn probe_mpcc_like() {
    println!("\n--- MpccLike: x*y=0, x≥0, y≥0 (LICQ fails on axes) ---");
    run(&MpccLike, "MpccLike", false);
    run(&MpccLike, "MpccLike", true);
}
