use ripopt::{NlpProblem, ParametricNlpProblem, SolveStatus, SolverOptions};

// ---------------------------------------------------------------------------
// HS071-like problem with parameter in a constraint:
//   min  x1*x4*(x1+x2+x3) + x3
//   s.t. x1*x2*x3*x4 >= 25
//        x1^2 + x2^2 + x3^2 + x4^2 = p   (parameter, nominally 40)
//        1 <= x_i <= 5
// ---------------------------------------------------------------------------

struct HS071Parametric {
    p: f64,
}

impl NlpProblem for HS071Parametric {
    fn num_variables(&self) -> usize {
        4
    }
    fn num_constraints(&self) -> usize {
        2
    }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..4 {
            x_l[i] = 1.0;
            x_u[i] = 5.0;
        }
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 25.0;
        g_u[0] = f64::INFINITY;
        g_l[1] = self.p;
        g_u[1] = self.p;
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 1.0;
        x0[1] = 5.0;
        x0[2] = 5.0;
        x0[3] = 1.0;
    }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = x[3] * (x[0] + x[1] + x[2]) + x[0] * x[3];
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
        (
            vec![0, 0, 0, 0, 1, 1, 1, 1],
            vec![0, 1, 2, 3, 0, 1, 2, 3],
        )
    }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = x[1] * x[2] * x[3];
        vals[1] = x[0] * x[2] * x[3];
        vals[2] = x[0] * x[1] * x[3];
        vals[3] = x[0] * x[1] * x[2];
        vals[4] = 2.0 * x[0];
        vals[5] = 2.0 * x[1];
        vals[6] = 2.0 * x[2];
        vals[7] = 2.0 * x[3];
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Lower triangle entries
        (
            vec![0, 1, 2, 3, 1, 2, 3, 2, 3, 3],
            vec![0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
        )
    }
    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        // Objective Hessian
        vals[0] = obj_factor * 2.0 * x[3]; // d²f/dx1² = 2*x4
        vals[1] = obj_factor * x[3]; // d²f/dx1dx2 = x4
        vals[2] = obj_factor * x[3]; // d²f/dx1dx3 = x4
        vals[3] = obj_factor * (2.0 * x[0] + x[1] + x[2]); // d²f/dx1dx4
        vals[4] = 0.0; // d²f/dx2²
        vals[5] = 0.0; // d²f/dx2dx3
        vals[6] = obj_factor * x[0]; // d²f/dx2dx4
        vals[7] = 0.0; // d²f/dx3²
        vals[8] = obj_factor * x[0]; // d²f/dx3dx4
        vals[9] = 0.0; // d²f/dx4²

        // Constraint 0 Hessian: g0 = x1*x2*x3*x4
        vals[0] += 0.0; // no x1²
        vals[1] += lambda[0] * x[2] * x[3];
        vals[2] += lambda[0] * x[1] * x[3];
        vals[3] += lambda[0] * x[1] * x[2];
        vals[4] += 0.0;
        vals[5] += lambda[0] * x[0] * x[3];
        vals[6] += lambda[0] * x[0] * x[2];
        vals[7] += 0.0;
        vals[8] += lambda[0] * x[0] * x[1];
        vals[9] += 0.0;

        // Constraint 1 Hessian: g1 = sum xi^2
        vals[0] += lambda[1] * 2.0;
        vals[4] += lambda[1] * 2.0;
        vals[7] += lambda[1] * 2.0;
        vals[9] += lambda[1] * 2.0;
        true
    }
}

impl ParametricNlpProblem for HS071Parametric {
    fn num_parameters(&self) -> usize {
        1
    }
    fn jacobian_p_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Constraint 1 has bound = p, so ∂(g1 - p)/∂p = -1
        (vec![1], vec![0])
    }
    fn jacobian_p_values(&self, _x: &[f64], vals: &mut [f64]) {
        vals[0] = -1.0;
    }
    fn hessian_xp_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }
    fn hessian_xp_values(
        &self,
        _x: &[f64],
        _obj_factor: f64,
        _lambda: &[f64],
        _vals: &mut [f64],
    ) {
    }
}

#[test]
fn test_hs071_sensitivity_vs_finite_differences() {
    let p0 = 40.0;
    let problem = HS071Parametric { p: p0 };
    let options = SolverOptions {
        print_level: 0,
        ..SolverOptions::default()
    };

    let mut ctx = ripopt::solve_with_sensitivity(&problem, &options);
    assert!(
        matches!(
            ctx.result.status,
            SolveStatus::Optimal
        ),
        "Expected converged, got {:?}",
        ctx.result.status
    );

    // Compute analytical sensitivity
    let dp = [1.0];
    let sens = ctx
        .compute_sensitivity(&problem, &[&dp])
        .expect("sensitivity should succeed");

    // Finite difference: solve at p0 + h
    let h = 1e-4;
    let problem2 = HS071Parametric { p: p0 + h };
    let result2 = ripopt::solve(&problem2, &options);

    for i in 0..4 {
        let fd = (result2.x[i] - ctx.result.x[i]) / h;
        let analytical = sens.dx_dp[0][i];
        let err = (fd - analytical).abs();
        assert!(
            err < 0.1,
            "x[{}]: FD dx/dp = {:.6}, analytical = {:.6}, error = {:.6}",
            i,
            fd,
            analytical,
            err
        );
    }
}

#[test]
fn test_sensitivity_linear_prediction_accuracy() {
    // Verify x(p + Δp) ≈ x(p) + (dx/dp)·Δp for small Δp
    let p0 = 40.0;
    let delta = 0.01;
    let problem = HS071Parametric { p: p0 };
    let options = SolverOptions {
        print_level: 0,
        ..SolverOptions::default()
    };

    let mut ctx = ripopt::solve_with_sensitivity(&problem, &options);
    let dp = [delta];
    let sens = ctx
        .compute_sensitivity(&problem, &[&dp])
        .expect("sensitivity should succeed");

    // Predicted solution
    let x_pred: Vec<f64> = ctx
        .result
        .x
        .iter()
        .zip(sens.dx_dp[0].iter())
        .map(|(x, dx)| x + dx)
        .collect();

    // Actual solution at perturbed p
    let problem2 = HS071Parametric { p: p0 + delta };
    let result2 = ripopt::solve(&problem2, &options);

    for i in 0..4 {
        let err = (x_pred[i] - result2.x[i]).abs();
        assert!(
            err < 1e-3,
            "x[{}]: predicted = {:.6}, actual = {:.6}, error = {:.6}",
            i,
            x_pred[i],
            result2.x[i],
            err
        );
    }
}
