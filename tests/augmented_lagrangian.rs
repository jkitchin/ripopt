use ripopt::{NlpProblem, SolveStatus, SolverOptions};

fn silent_opts() -> SolverOptions {
    SolverOptions {
        print_level: 0,
        ..SolverOptions::default()
    }
}

// ---------------------------------------------------------------------------
// 1. Equality constraint: min x0^2 + x1^2 s.t. x0 + x1 = 1
//    x* = (0.5, 0.5), f* = 0.5
// ---------------------------------------------------------------------------

struct EqualityQP;

impl NlpProblem for EqualityQP {
    fn num_variables(&self) -> usize {
        2
    }
    fn num_constraints(&self) -> usize {
        1
    }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_l[1] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
        x_u[1] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0;
        g_u[0] = 1.0;
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.5;
        x0[1] = 0.5;
    }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        x[0] * x[0] + x[1] * x[1]
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0 * x[0];
        grad[1] = 2.0 * x[1];
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0] + x[1];
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0], vec![0, 1])
    }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 1.0;
        vals[1] = 1.0;
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        vals[0] = 2.0 * obj_factor;
        vals[1] = 2.0 * obj_factor;
    }
}

#[test]
fn al_equality_convergence() {
    let result = ripopt::augmented_lagrangian::solve(&EqualityQP, &silent_opts());
    assert!(
        result.status == SolveStatus::Optimal,
        "status={:?}",
        result.status
    );
    assert!((result.objective - 0.5).abs() < 0.1, "obj={}", result.objective);
    assert!((result.x[0] - 0.5).abs() < 0.1, "x0={}", result.x[0]);
    assert!((result.x[1] - 0.5).abs() < 0.1, "x1={}", result.x[1]);
}

// ---------------------------------------------------------------------------
// 2. Inequality constraint: min (x0-2)^2 + (x1-2)^2 s.t. x0 + x1 >= 1
//    Unconstrained optimum (2,2) has x0+x1=4 >= 1, so constraint is inactive.
//    x* = (2, 2), f* = 0
// ---------------------------------------------------------------------------

struct InequalityQP;

impl NlpProblem for InequalityQP {
    fn num_variables(&self) -> usize {
        2
    }
    fn num_constraints(&self) -> usize {
        1
    }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_l[1] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
        x_u[1] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0;
        g_u[0] = f64::INFINITY;
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
        x0[1] = 0.0;
    }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2)
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0 * (x[0] - 2.0);
        grad[1] = 2.0 * (x[1] - 2.0);
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0] + x[1];
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0], vec![0, 1])
    }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 1.0;
        vals[1] = 1.0;
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        vals[0] = 2.0 * obj_factor;
        vals[1] = 2.0 * obj_factor;
    }
}

#[test]
fn al_inequality_convergence() {
    let result = ripopt::augmented_lagrangian::solve(&InequalityQP, &silent_opts());
    assert_eq!(result.status, SolveStatus::Optimal);
    assert!(result.objective < 0.1, "obj={}", result.objective);
    assert!((result.x[0] - 2.0).abs() < 0.1, "x0={}", result.x[0]);
    assert!((result.x[1] - 2.0).abs() < 0.1, "x1={}", result.x[1]);
}

// ---------------------------------------------------------------------------
// 3. Rho increase path: min x^2 s.t. x = 100, start x0 = 0
//    Large initial violation forces rho to increase.
//    x* = 100, f* = 10000
// ---------------------------------------------------------------------------

struct RhoIncrease;

impl NlpProblem for RhoIncrease {
    fn num_variables(&self) -> usize {
        1
    }
    fn num_constraints(&self) -> usize {
        1
    }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 100.0;
        g_u[0] = 100.0;
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
    }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        x[0] * x[0]
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0 * x[0];
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0];
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 1.0;
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        vals[0] = 2.0 * obj_factor;
    }
}

#[test]
fn al_rho_increase() {
    let result = ripopt::augmented_lagrangian::solve(&RhoIncrease, &silent_opts());
    assert!(
        result.status == SolveStatus::Optimal,
        "status={:?}",
        result.status
    );
    assert!((result.x[0] - 100.0).abs() < 0.1, "x={}", result.x[0]);
}

// ---------------------------------------------------------------------------
// 4. Near-tolerance convergence: min x^2 s.t. x^3 = 1000 (x* = 10)
//    Problem is well-posed; should reach Optimal or NumericalError.
// ---------------------------------------------------------------------------

struct AcceptableProblem;

impl NlpProblem for AcceptableProblem {
    fn num_variables(&self) -> usize {
        1
    }
    fn num_constraints(&self) -> usize {
        1
    }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1000.0;
        g_u[0] = 1000.0;
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 5.0;
    }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        x[0] * x[0]
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0 * x[0];
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0].powi(3);
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 3.0 * x[0].powi(2);
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }
    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = 2.0 * obj_factor + lambda[0] * 6.0 * x[0];
    }
}

#[test]
fn al_near_tolerance_convergence() {
    let opts = SolverOptions {
        print_level: 0,
        ..SolverOptions::default()
    };
    let result = ripopt::augmented_lagrangian::solve(&AcceptableProblem, &opts);
    assert!(
        result.status == SolveStatus::Optimal || result.status == SolveStatus::NumericalError,
        "Expected Optimal or NumericalError, got: status={:?}",
        result.status
    );
    assert!((result.x[0] - 10.0).abs() < 0.5, "x={}", result.x[0]);
}

// ---------------------------------------------------------------------------
// 5. Infeasible (contradictory): min x^2 s.t. x = 1 AND x = 2
//    No feasible point exists; expect MaxIterations.
// ---------------------------------------------------------------------------

struct Infeasible;

impl NlpProblem for Infeasible {
    fn num_variables(&self) -> usize {
        1
    }
    fn num_constraints(&self) -> usize {
        2
    }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0;
        g_u[0] = 1.0;
        g_l[1] = 2.0;
        g_u[1] = 2.0;
    }
    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
    }
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        x[0] * x[0]
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0 * x[0];
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0];
        g[1] = x[0];
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 0])
    }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 1.0;
        vals[1] = 1.0;
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        vals[0] = 2.0 * obj_factor;
    }
}

#[test]
fn al_max_iterations() {
    let result = ripopt::augmented_lagrangian::solve(&Infeasible, &silent_opts());
    assert_eq!(result.status, SolveStatus::MaxIterations);
}
