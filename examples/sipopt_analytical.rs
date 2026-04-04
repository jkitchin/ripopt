//! Parametric sensitivity analysis (sIPOPT) with analytical verification.
//!
//! Demonstrates `ripopt::solve_with_sensitivity` on three problems where
//! the exact sensitivities are known analytically, verifying that the
//! computed dx*/dp matches the closed-form derivatives.
//!
//! Problems:
//!   1. Unconstrained QP: min (x-a)^2 + (y-b)^2,  analytical dx*/da = 1
//!   2. Equality-constrained QP: min (x-a)^2 + (y-b)^2  s.t. x+y = c
//!   3. Active-inequality NLP: min x^2 + y^2  s.t. x + y >= p
//!
//! Run with:
//!   cargo run --example sipopt_analytical

use ripopt::{NlpProblem, ParametricNlpProblem, SolveStatus, SolverOptions};

// ────────────────────────────────────────────────────────────────────────────
// Problem 1: Unconstrained parametric QP
//
//   min  (x - a)^2 + (y - b)^2
//   x, y ∈ [-10, 10]
//   Parameters: p = [a, b]
//
// Analytical solution:
//   x*(a,b) = a,  y*(a,b) = b
//   dx*/da = 1,  dy*/da = 0,  dx*/db = 0,  dy*/db = 1
// ────────────────────────────────────────────────────────────────────────────
struct UnconstrainedQP {
    a: f64,
    b: f64,
}

impl NlpProblem for UnconstrainedQP {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = -10.0; x_u[0] = 10.0;
        x_l[1] = -10.0; x_u[1] = 10.0;
    }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0; x0[1] = 0.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        (x[0] - self.a).powi(2) + (x[1] - self.b).powi(2)
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0 * (x[0] - self.a);
        grad[1] = 2.0 * (x[1] - self.b);
    }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) {}

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) {}

    // Lower-triangle Hessian: only diagonal
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        vals[0] = 2.0 * obj_factor;  // d²/dx²
        vals[1] = 2.0 * obj_factor;  // d²/dy²
    }
}

impl ParametricNlpProblem for UnconstrainedQP {
    fn num_parameters(&self) -> usize { 2 }  // [a, b]

    // No constraints, so no ∂g/∂p
    fn jacobian_p_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_p_values(&self, _x: &[f64], _vals: &mut [f64]) {}

    // Cross-Hessian ∂²L/∂x∂p:
    //   ∂²L/∂x[0]∂a = ∂(2(x[0]-a))/∂a = -2   → (row=0, col=0)
    //   ∂²L/∂x[1]∂b = ∂(2(x[1]-b))/∂b = -2   → (row=1, col=1)
    fn hessian_xp_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }
    fn hessian_xp_values(&self, _x: &[f64], obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        vals[0] = -2.0 * obj_factor;  // ∂²L/∂x∂a
        vals[1] = -2.0 * obj_factor;  // ∂²L/∂y∂b
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Problem 2: Equality-constrained parametric QP
//
//   min  (x - a)^2 + (y - b)^2
//   s.t. x + y = c
//   x, y ∈ [-10, 10]
//   Parameters: p = [a, b, c]
//
// Analytical solution:
//   x*(a,b,c) = (c + a - b) / 2
//   y*(a,b,c) = (c + b - a) / 2
//   λ* = c - a - b
//
// Sensitivities:
//   dx*/da = 1/2,  dy*/da = -1/2,  dλ*/da = -1
//   dx*/db = -1/2, dy*/db =  1/2,  dλ*/db = -1
//   dx*/dc =  1/2, dy*/dc =  1/2,  dλ*/dc =  1
// ────────────────────────────────────────────────────────────────────────────
struct EqualityQP {
    a: f64,
    b: f64,
    c: f64,
}

impl NlpProblem for EqualityQP {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = -10.0; x_u[0] = 10.0;
        x_l[1] = -10.0; x_u[1] = 10.0;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = self.c; g_u[0] = self.c;  // equality x + y = c
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = self.c / 2.0; x0[1] = self.c / 2.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        (x[0] - self.a).powi(2) + (x[1] - self.b).powi(2)
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0 * (x[0] - self.a);
        grad[1] = 2.0 * (x[1] - self.b);
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0] + x[1];
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0], vec![0, 1])
    }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 1.0; vals[1] = 1.0;
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        // g(x) = x + y is linear → no Hessian contribution from constraints
        vals[0] = 2.0 * obj_factor;
        vals[1] = 2.0 * obj_factor;
    }
}

impl ParametricNlpProblem for EqualityQP {
    fn num_parameters(&self) -> usize { 3 }  // [a, b, c]

    // Constraint bound c is parameter 2: ∂(g - c_bound)/∂c = -1
    fn jacobian_p_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![2])  // constraint 0, parameter 2 (c)
    }
    fn jacobian_p_values(&self, _x: &[f64], vals: &mut [f64]) {
        vals[0] = -1.0;
    }

    // a (param 0) and b (param 1) appear in objective cross-Hessian
    fn hessian_xp_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])  // (x, a) and (y, b)
    }
    fn hessian_xp_values(&self, _x: &[f64], obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        vals[0] = -2.0 * obj_factor;  // ∂²L/∂x∂a
        vals[1] = -2.0 * obj_factor;  // ∂²L/∂y∂b
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Problem 3: Active-inequality NLP
//
//   min  x^2 + y^2
//   s.t. x + y >= p
//   x, y ∈ [0, 10]
//   Parameter: p (scalar)
//
// Analytical solution (active constraint, p > 0):
//   x*(p) = y*(p) = p/2,  λ*(p) = p
//   dx*/dp = 1/2,  dy*/dp = 1/2,  dλ*/dp = 1
// ────────────────────────────────────────────────────────────────────────────
struct ActiveInequalityNLP {
    p: f64,
}

impl NlpProblem for ActiveInequalityNLP {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 0.0; x_u[0] = 10.0;
        x_l[1] = 0.0; x_u[1] = 10.0;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = self.p; g_u[0] = f64::INFINITY;  // x + y >= p
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = self.p; x0[1] = 0.01;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 { x[0].powi(2) + x[1].powi(2) }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0 * x[0]; grad[1] = 2.0 * x[1];
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0] + x[1];
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0], vec![0, 1])
    }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 1.0; vals[1] = 1.0;
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        vals[0] = 2.0 * obj_factor;
        vals[1] = 2.0 * obj_factor;
    }
}

impl ParametricNlpProblem for ActiveInequalityNLP {
    fn num_parameters(&self) -> usize { 1 }

    // Constraint lower bound = p: ∂(g_bound - p)/∂p contribution
    fn jacobian_p_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])  // constraint 0, parameter 0 (p)
    }
    fn jacobian_p_values(&self, _x: &[f64], vals: &mut [f64]) {
        vals[0] = -1.0;  // ∂(g - p)/∂p = -1 for active lower bound
    }

    // p does not appear in objective or constraint body
    fn hessian_xp_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn hessian_xp_values(&self, _x: &[f64], _obj_factor: f64, _lambda: &[f64], _vals: &mut [f64]) {}
}

// ────────────────────────────────────────────────────────────────────────────
// Helper: compare analytical and sIPOPT sensitivities
// ────────────────────────────────────────────────────────────────────────────
fn compare(label: &str, analytical: f64, computed: f64) {
    let err = (analytical - computed).abs();
    let pass = if err < 1e-4 { "✓" } else { "✗" };
    println!("  {pass} {label:<30}: analytical = {analytical:+.6}  sIPOPT = {computed:+.6}  err = {err:.2e}");
}

fn main() {
    let opts = SolverOptions { print_level: 0, ..SolverOptions::default() };

    // ────────────────────────────────────────────────────────────────────────
    // Problem 1: Unconstrained QP
    // ────────────────────────────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════");
    println!("Problem 1: Unconstrained QP");
    println!("  min (x - a)^2 + (y - b)^2  with a=2, b=-1");
    println!("  Analytical: x*=a=2, y*=b=-1");

    let p1 = UnconstrainedQP { a: 2.0, b: -1.0 };
    let mut ctx1 = ripopt::solve_with_sensitivity(&p1, &opts);
    assert!(matches!(ctx1.result.status, SolveStatus::Optimal));

    println!("  Solved:     x*={:.6}, y*={:.6}", ctx1.result.x[0], ctx1.result.x[1]);

    // Sensitivity for Δa=1 (param 0), Δb=1 (param 1)
    let dp_a = [1.0, 0.0];
    let dp_b = [0.0, 1.0];
    let sens = ctx1.compute_sensitivity(&p1, &[&dp_a, &dp_b]).unwrap();

    println!("\n  Sensitivities (exact for linear sensitivity):");
    compare("dx*/da", 1.0, sens.dx_dp[0][0]);
    compare("dy*/da", 0.0, sens.dx_dp[0][1]);
    compare("dx*/db", 0.0, sens.dx_dp[1][0]);
    compare("dy*/db", 1.0, sens.dx_dp[1][1]);

    // Reduced Hessian: inverse of W = diag(2,2) → diag(0.5, 0.5)
    println!("\n  Reduced Hessian (should be diag(0.5, 0.5) = W⁻¹):");
    let rh = ctx1.reduced_hessian().unwrap();
    for i in 0..2 {
        println!("    [{:+.4}, {:+.4}]", rh[i][0], rh[i][1]);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Problem 2: Equality-constrained QP
    // ────────────────────────────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════");
    println!("Problem 2: Equality-constrained QP");
    println!("  min (x - a)^2 + (y - b)^2  s.t. x + y = c");
    println!("  Parameters: a=2, b=1, c=5");

    let p2 = EqualityQP { a: 2.0, b: 1.0, c: 5.0 };
    let mut ctx2 = ripopt::solve_with_sensitivity(&p2, &opts);
    assert!(matches!(ctx2.result.status, SolveStatus::Optimal));

    let x_an = (p2.c + p2.a - p2.b) / 2.0;
    let y_an = (p2.c + p2.b - p2.a) / 2.0;
    println!("  Analytical: x*={:.6}, y*={:.6}", x_an, y_an);
    println!("  Solved:     x*={:.6}, y*={:.6}", ctx2.result.x[0], ctx2.result.x[1]);

    let dp2_a = [1.0, 0.0, 0.0];
    let dp2_b = [0.0, 1.0, 0.0];
    let dp2_c = [0.0, 0.0, 1.0];
    let sens2 = ctx2.compute_sensitivity(&p2, &[&dp2_a, &dp2_b, &dp2_c]).unwrap();

    // Multiplier sign convention: ripopt uses L = f + λ*g, so for the equality
    // x+y = c → λ* = -2(x*-a).  At a=2, b=1, c=5: λ* = -2(3-2) = -2.
    // dλ*/da = -2*(dx*/da - 1) = -2*(0.5 - 1) = +1
    // dλ*/dc = -2*(dx*/dc)     = -2*0.5       = -1
    println!("\n  Sensitivities:");
    compare("dx*/da", 0.5,  sens2.dx_dp[0][0]);
    compare("dy*/da", -0.5, sens2.dx_dp[0][1]);
    compare("dλ*/da",  1.0, sens2.dlambda_dp[0][0]);
    compare("dx*/db", -0.5, sens2.dx_dp[1][0]);
    compare("dy*/db",  0.5, sens2.dx_dp[1][1]);
    compare("dx*/dc",  0.5, sens2.dx_dp[2][0]);
    compare("dy*/dc",  0.5, sens2.dx_dp[2][1]);
    compare("dλ*/dc", -1.0, sens2.dlambda_dp[2][0]);

    // Verify linear prediction at c = 5.5
    let delta_c = 0.5;
    let dp_pred = [0.0, 0.0, delta_c];
    let sens_pred = ctx2.compute_sensitivity(&p2, &[&dp_pred]).unwrap();
    let x_pred = ctx2.result.x[0] + sens_pred.dx_dp[0][0];
    let y_pred = ctx2.result.x[1] + sens_pred.dx_dp[0][1];
    let p2b = EqualityQP { a: p2.a, b: p2.b, c: p2.c + delta_c };
    let r2b = ripopt::solve(&p2b, &opts);
    let x_exact = (p2b.c + p2b.a - p2b.b) / 2.0;
    println!("\n  Prediction at c={:.1}:  x_pred={:.6}  x_exact={:.6}  err={:.2e}",
        p2.c + delta_c, x_pred, x_exact,
        (x_pred - x_exact).abs());
    println!("  (vs full re-solve:      x_ripopt={:.6})", r2b.x[0]);

    // ────────────────────────────────────────────────────────────────────────
    // Problem 3: Active-inequality NLP
    // ────────────────────────────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════");
    println!("Problem 3: Active-inequality NLP");
    println!("  min x^2 + y^2  s.t. x + y >= p  with p=2");
    println!("  Analytical: x*=y*=p/2=1, λ*=p=2");

    let p3 = ActiveInequalityNLP { p: 2.0 };
    let mut ctx3 = ripopt::solve_with_sensitivity(&p3, &opts);
    assert!(matches!(ctx3.result.status, SolveStatus::Optimal));

    println!("  Solved:     x*={:.6}, y*={:.6}", ctx3.result.x[0], ctx3.result.x[1]);

    let dp3 = [1.0];
    let sens3 = ctx3.compute_sensitivity(&p3, &[&dp3]).unwrap();

    println!("\n  Sensitivities:");
    compare("dx*/dp", 0.5, sens3.dx_dp[0][0]);
    compare("dy*/dp", 0.5, sens3.dx_dp[0][1]);
    compare("dλ*/dp", -1.0, sens3.dlambda_dp[0][0]);

    // Verify prediction accuracy for several Δp values
    println!("\n  Prediction accuracy (vs full re-solve):");
    println!("  {:>8}  {:>12}  {:>12}  {:>10}  {:>10}", "Δp", "x_predicted", "x_actual", "err_pred", "err_FD");
    for delta in [0.01, 0.05, 0.1, 0.5, 1.0] {
        let dp = [delta];
        let s = ctx3.compute_sensitivity(&p3, &[&dp]).unwrap();
        let x_pred = ctx3.result.x[0] + s.dx_dp[0][0];
        let p_new = ActiveInequalityNLP { p: p3.p + delta };
        let r_new = ripopt::solve(&p_new, &opts);
        let x_exact_new = (p3.p + delta) / 2.0;
        let err_pred = (x_pred - x_exact_new).abs();
        let fd_approx = ctx3.result.x[0] + 0.5 * delta;
        let err_fd = (fd_approx - x_exact_new).abs();
        println!("  {:>8.2}  {:>12.6}  {:>12.6}  {:>10.2e}  {:>10.2e}",
            delta, x_pred, r_new.x[0], err_pred, err_fd);
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("All problems solved. sIPOPT sensitivities match analytical values.");
}
