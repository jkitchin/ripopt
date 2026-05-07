use ripopt::{NlpProblem, SolveResult, SolveStatus, SolverOptions};
use std::cell::Cell;

// ---------------------------------------------------------------------------
// 1. NE-to-LS detection: f=0, grad=0, 3 equalities in 2 vars
//    x0+x1=1, x0-x1=0, 2*x0=1  =>  x*=(0.5, 0.5)
// ---------------------------------------------------------------------------

struct NeToLs;

impl NlpProblem for NeToLs {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 3 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0; g_u[0] = 1.0; // x0+x1=1
        g_l[1] = 0.0; g_u[1] = 0.0; // x0-x1=0
        g_l[2] = 1.0; g_u[2] = 1.0; // 2*x0=1
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
        x0[1] = 0.0;
    }

    fn objective(&self, _x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = 0.0;
        true
    }
    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 0.0;
        grad[1] = 0.0;
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] + x[1];
        g[1] = x[0] - x[1];
        g[2] = 2.0 * x[0];
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // (row, col): (0,0),(0,1),(1,0),(1,1),(2,0)
        (vec![0, 0, 1, 1, 2], vec![0, 1, 0, 1, 0])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0;  // dg0/dx0
        vals[1] = 1.0;  // dg0/dx1
        vals[2] = 1.0;  // dg1/dx0
        vals[3] = -1.0; // dg1/dx1
        vals[4] = 2.0;  // dg2/dx0;
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], _vals: &mut [f64]) -> bool { true }
}

#[test]
fn ipm_ne_to_ls_detection() {
    let problem = NeToLs;
    let options = SolverOptions {
        print_level: 0,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal or Acceptable, got {:?}",
        result.status
    );
    assert!((result.x[0] - 0.5).abs() < 1e-4, "x0={}, expected 0.5", result.x[0]);
    assert!((result.x[1] - 0.5).abs() < 1e-4, "x1={}, expected 0.5", result.x[1]);
}

// ---------------------------------------------------------------------------
// 2. Condensed KKT: m >= 2*n triggers condensed path
//    min x^2 s.t. x=1, x=1, x=1  (m=3, n=1)
// ---------------------------------------------------------------------------

struct CondensedKkt;

impl NlpProblem for CondensedKkt {
    fn num_variables(&self) -> usize { 1 }
    fn num_constraints(&self) -> usize { 3 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for i in 0..3 {
            g_l[i] = 1.0;
            g_u[i] = 1.0;
        }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.5;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[0] * x[0];
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 2.0 * x[0];
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0];
        g[1] = x[0];
        g[2] = x[0];
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 2], vec![0, 0, 0])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0;
        vals[1] = 1.0;
        vals[2] = 1.0;
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = 2.0 * obj_factor;
        true
    }
}

#[test]
fn ipm_condensed_kkt() {
    let problem = CondensedKkt;
    let options = SolverOptions {
        print_level: 0,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    assert_eq!(result.status, SolveStatus::Optimal, "Expected Optimal, got {:?}", result.status);
    assert!((result.x[0] - 1.0).abs() < 1e-6, "x={}, expected 1.0", result.x[0]);
}

// ---------------------------------------------------------------------------
// 2b. Dense condensed KKT with m >> n in "sparse" regime (n+m > sparse_threshold).
//     min sum(x_i^2) s.t. x_i = 1 for i=1..n, repeated M/n times each
//     n=5, m=200 => n+m=205 > sparse_threshold=110
//     Should use dense condensed (5x5) instead of sparse augmented (205x205).
// ---------------------------------------------------------------------------

struct TallNarrowCondensed;

impl NlpProblem for TallNarrowCondensed {
    fn num_variables(&self) -> usize { 5 }
    fn num_constraints(&self) -> usize { 200 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..5 {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for i in 0..200 {
            g_l[i] = 1.0;
            g_u[i] = 1.0;
        }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        for i in 0..5 { x0[i] = 0.5; }
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x.iter().map(|xi| xi * xi).sum();
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        for i in 0..5 { grad[i] = 2.0 * x[i]; }
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        // Each constraint pins one variable: g[j] = x[j % 5]
        for j in 0..200 { g[j] = x[j % 5]; }
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let mut rows = Vec::with_capacity(200);
        let mut cols = Vec::with_capacity(200);
        for j in 0..200 {
            rows.push(j);
            cols.push(j % 5);
        }
        (rows, cols)
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        for v in vals.iter_mut() { *v = 1.0; }
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 2, 3, 4], vec![0, 1, 2, 3, 4])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        for v in vals.iter_mut() { *v = 2.0 * obj_factor; }
        true
    }
}

#[test]
fn ipm_condensed_kkt_tall_narrow() {
    let problem = TallNarrowCondensed;
    let options = SolverOptions {
        print_level: 0,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal/Acceptable, got {:?}", result.status
    );
    for i in 0..5 {
        assert!((result.x[i] - 1.0).abs() < 1e-4, "x[{}]={}, expected 1.0", i, result.x[i]);
    }
    assert!((result.objective - 5.0).abs() < 1e-4, "obj={}, expected 5.0", result.objective);
}

// ---------------------------------------------------------------------------
// 3. Unbounded detection: min -x, no bounds, no constraints
// ---------------------------------------------------------------------------

struct UnboundedProblem;

impl NlpProblem for UnboundedProblem {
    fn num_variables(&self) -> usize { 1 }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
    }

    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = -x[0];
        true
    }

    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = -1.0;
        true
    }

    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], _vals: &mut [f64]) -> bool { true }
}

#[test]
fn ipm_unbounded_detection() {
    let problem = UnboundedProblem;
    let options = SolverOptions {
        print_level: 0,
        enable_lbfgs_fallback: false,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    // Without bounds, the solver may detect unboundedness, hit numerical issues,
    // reach max iterations, or declare acceptable at a large negative objective.
    assert!(
        result.status == SolveStatus::Unbounded
            || result.status == SolveStatus::NumericalError
            || result.status == SolveStatus::MaxIterations,
        "Expected Unbounded/NumericalError/MaxIterations, got {:?}",
        result.status
    );
}

// ---------------------------------------------------------------------------
// 4. Preprocessing: fixed variable x0=5, min x1^2+x2^2 s.t. x1+x2=1
//    x*=(5.0, 0.5, 0.5), obj*=0.5
// ---------------------------------------------------------------------------

struct PreprocessingProblem;

impl NlpProblem for PreprocessingProblem {
    fn num_variables(&self) -> usize { 3 }
    fn num_constraints(&self) -> usize { 1 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 5.0; x_u[0] = 5.0; // fixed
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
        x_l[2] = f64::NEG_INFINITY; x_u[2] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0;
        g_u[0] = 1.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 5.0;
        x0[1] = 0.0;
        x0[2] = 0.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[1] * x[1] + x[2] * x[2];
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 0.0;
        grad[1] = 2.0 * x[1];
        grad[2] = 2.0 * x[2];
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[1] + x[2];
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // dg0/dx1, dg0/dx2
        (vec![0, 0], vec![1, 2])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0;
        vals[1] = 1.0;
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Lower triangle: (1,1), (2,2)
        (vec![1, 2], vec![1, 2])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = 2.0 * obj_factor; // d2f/dx1^2
        vals[1] = 2.0 * obj_factor; // d2f/dx2^2;
        true
    }
}

struct AuxiliaryBasinGuardProblem;

impl NlpProblem for AuxiliaryBasinGuardProblem {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0;
        g_u[0] = 1.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = -0.8;
        x0[1] = 0.5;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[1] * x[1];
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 0.0;
        grad[1] = 2.0 * x[1];
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] * x[0];
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 2.0 * x[0];
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = 2.0 * lambda[0];
        vals[1] = 2.0 * obj_factor;
        true
    }
}

struct AuxiliaryReducedFallbackProblem;

impl NlpProblem for AuxiliaryReducedFallbackProblem {
    fn num_variables(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        2
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY;
        x_u[1] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = 0.0;
        g_l[1] = f64::NEG_INFINITY;
        g_u[1] = 10.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
        x0[1] = 3.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = -8.0 + (x[1] - 3.0) * (x[1] - 3.0);
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 0.0;
        grad[1] = 2.0 * (x[1] - 3.0);
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] - 2.0;
        g[1] = x[0] + x[1];
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1], vec![0, 0, 1])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0;
        vals[1] = 1.0;
        vals[2] = 1.0;
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![1], vec![1])
    }

    fn hessian_values(
        &self,
        _x: &[f64],
        _new_x: bool,
        obj_factor: f64,
        _lambda: &[f64],
        vals: &mut [f64],
    ) -> bool {
        vals[0] = 2.0 * obj_factor;
        true
    }
}

struct AuxiliaryFailureFallbackProblem {
    fail_next_constraint: Cell<bool>,
    failed_once: Cell<bool>,
}

impl NlpProblem for AuxiliaryFailureFallbackProblem {
    fn num_variables(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        1
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY;
        x_u[1] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = 0.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
        x0[1] = 0.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = (x[1] - 3.0) * (x[1] - 3.0);
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 0.0;
        grad[1] = 2.0 * (x[1] - 3.0);
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        if self.fail_next_constraint.replace(false) {
            self.failed_once.set(true);
            return false;
        }
        g[0] = x[0] - 2.0;
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0;
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }

    fn hessian_values(
        &self,
        _x: &[f64],
        _new_x: bool,
        obj_factor: f64,
        _lambda: &[f64],
        vals: &mut [f64],
    ) -> bool {
        vals[0] = 0.0;
        vals[1] = 2.0 * obj_factor;
        true
    }
}

struct AuxiliaryBranchObjectiveProblem;

impl NlpProblem for AuxiliaryBranchObjectiveProblem {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = 0.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
        x0[1] = -1.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = (x[0] - x[1]) * (x[0] - x[1]);
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 2.0 * (x[0] - x[1]);
        grad[1] = -2.0 * (x[0] - x[1]);
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[1] * x[1] - 4.0;
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![1])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 2.0 * x[1];
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1], vec![0, 0, 1])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = 2.0 * obj_factor;
        vals[1] = -2.0 * obj_factor;
        vals[2] = 2.0 * obj_factor + 2.0 * lambda[0];
        true
    }
}

struct AuxiliaryInequalityBranchProblem;

impl NlpProblem for AuxiliaryInequalityBranchProblem {
    fn num_variables(&self) -> usize { 1 }
    fn num_constraints(&self) -> usize { 2 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0;
        g_u[0] = 1.0;
        g_l[1] = 0.0;
        g_u[1] = f64::INFINITY;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = -1.0;
    }

    fn objective(&self, _x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = 0.0;
        true
    }

    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 0.0;
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] * x[0];
        g[1] = x[0];
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 0])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 2.0 * x[0];
        vals[1] = 1.0;
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }

    fn hessian_values(
        &self,
        _x: &[f64],
        _new_x: bool,
        _obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) -> bool {
        vals[0] = 2.0 * lambda[0];
        true
    }
}

struct AuxiliaryTriangularEndToEndProblem;

impl NlpProblem for AuxiliaryTriangularEndToEndProblem {
    fn num_variables(&self) -> usize { 3 }
    fn num_constraints(&self) -> usize { 2 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
        x_l[2] = f64::NEG_INFINITY; x_u[2] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = 0.0;
        g_l[1] = 0.0;
        g_u[1] = 0.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
        x0[1] = 1.0;
        x0[2] = 0.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = (x[0] - 1.0) * (x[0] - 1.0);
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 2.0 * (x[0] - 1.0);
        grad[1] = 0.0;
        grad[2] = 0.0;
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[1] * x[1] - 4.0;
        g[1] = x[2] - x[1] - 1.0;
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1], vec![1, 1, 2])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 2.0 * x[1];
        vals[1] = -1.0;
        vals[2] = 1.0;
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 2], vec![0, 1, 2])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = 2.0 * obj_factor;
        vals[1] = 2.0 * lambda[0];
        vals[2] = 0.0;
        true
    }
}

struct AuxiliaryReducedFailureFallbackProblem {
    fail_reduced_objective_once: Cell<bool>,
    reduced_failed_once: Cell<bool>,
}

impl NlpProblem for AuxiliaryReducedFailureFallbackProblem {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = 0.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
        x0[1] = 3.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        if (x[0] - 2.0).abs() < 1e-12 && self.fail_reduced_objective_once.replace(false) {
            self.reduced_failed_once.set(true);
            return false;
        }
        *obj = (x[1] - 3.0) * (x[1] - 3.0);
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 0.0;
        grad[1] = 2.0 * (x[1] - 3.0);
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] - 2.0;
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0;
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = 0.0;
        vals[1] = 2.0 * obj_factor;
        true
    }
}

struct PreprocessingDisabledBypassProblem {
    zero_obj_factor_hessian_seen: Cell<bool>,
}

impl NlpProblem for PreprocessingDisabledBypassProblem {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = 0.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
        x0[1] = 0.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = (x[0] - 3.0) * (x[0] - 3.0) + (x[1] - 2.0) * (x[1] - 2.0);
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 2.0 * (x[0] - 3.0);
        grad[1] = 2.0 * (x[1] - 2.0);
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[1] - 2.0;
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![1])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0;
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        if obj_factor.abs() < 1e-15 {
            self.zero_obj_factor_hessian_seen.set(true);
        }
        vals[0] = 2.0 * obj_factor;
        vals[1] = 2.0 * obj_factor;
        true
    }
}

#[test]
fn ipm_preprocessing_integration() {
    let problem = PreprocessingProblem;
    let options = SolverOptions {
        print_level: 0,
        enable_preprocessing: true,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    assert_eq!(result.status, SolveStatus::Optimal, "Expected Optimal, got {:?}", result.status);
    assert!((result.x[0] - 5.0).abs() < 1e-10, "x0={}, expected 5.0", result.x[0]);
    assert!((result.x[1] - 0.5).abs() < 1e-4, "x1={}, expected 0.5", result.x[1]);
    assert!((result.x[2] - 0.5).abs() < 1e-4, "x2={}, expected 0.5", result.x[2]);
    assert!((result.objective - 0.5).abs() < 1e-4, "obj={}, expected 0.5", result.objective);
}

#[test]
fn auxiliary_objective_coupled_branch_solves_full_space_objective() {
    let problem = AuxiliaryBranchObjectiveProblem;
    let options = SolverOptions {
        print_level: 0,
        enable_preprocessing: true,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        max_iter: 200,
        tol: 1e-8,
        ..SolverOptions::default()
    };

    let result = ripopt::solve(&problem, &options);

    assert_eq!(result.status, SolveStatus::Optimal, "Expected Optimal, got {:?}", result.status);
    assert!(
        (result.x[1] + 2.0).abs() < 1e-5,
        "objective-coupled equality should stay on the full-space branch selected by the initial point: x={:?}",
        result.x
    );
    assert!(
        (result.x[0] + 2.0).abs() < 1e-5,
        "full-space objective should align x0 with the selected equality branch: x={:?}",
        result.x
    );
    assert!(result.objective < 1e-10, "obj={}", result.objective);
}

#[test]
fn auxiliary_inequality_coupled_branch_solves_full_space_feasibly() {
    let problem = AuxiliaryInequalityBranchProblem;
    let options = SolverOptions {
        print_level: 0,
        enable_preprocessing: true,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        max_iter: 200,
        tol: 1e-8,
        ..SolverOptions::default()
    };

    let result = ripopt::solve(&problem, &options);

    assert_eq!(result.status, SolveStatus::Optimal, "Expected Optimal, got {:?}", result.status);
    assert!(
        (result.x[0] - 1.0).abs() < 1e-6,
        "inequality-coupled branch should solve to the feasible positive branch: x={:?}",
        result.x
    );
    assert!((result.constraint_values[0] - 1.0).abs() < 1e-8);
    assert!(result.constraint_values[1] >= -1e-8);
}

#[test]
fn auxiliary_triangular_system_solves_before_main_nlp() {
    let problem = AuxiliaryTriangularEndToEndProblem;
    let options = SolverOptions {
        print_level: 0,
        enable_preprocessing: true,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        max_iter: 200,
        tol: 1e-8,
        ..SolverOptions::default()
    };

    let result = ripopt::solve(&problem, &options);

    assert_eq!(result.status, SolveStatus::Optimal, "Expected Optimal, got {:?}", result.status);
    assert!((result.x[0] - 1.0).abs() < 1e-5, "x={:?}", result.x);
    assert!((result.x[1] - 2.0).abs() < 1e-5, "x={:?}", result.x);
    assert!((result.x[2] - 3.0).abs() < 1e-5, "x={:?}", result.x);
    assert!(result.constraint_values[0].abs() < 1e-8);
    assert!(result.constraint_values[1].abs() < 1e-8);
}

#[test]
fn auxiliary_preprocessing_integrates_without_fallback_tag() {
    let problem = AuxiliaryBasinGuardProblem;
    let options = SolverOptions {
        print_level: 0,
        enable_preprocessing: true,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        max_iter: 200,
        tol: 1e-8,
        ..SolverOptions::default()
    };

    let result = ripopt::solve(&problem, &options);

    assert_eq!(result.status, SolveStatus::Optimal, "Expected Optimal, got {:?}", result.status);
    assert!((result.x[0] + 1.0).abs() < 1e-4, "x0={}, expected -1.0", result.x[0]);
    assert!(result.x[1].abs() < 1e-4, "x1={}, expected 0.0", result.x[1]);
    assert!(
        result.diagnostics.fallback_used.as_deref() != Some("auxiliary_preprocessing"),
        "auxiliary preprocessing is part of preprocessing retry, not a failure fallback: {:?}",
        result.diagnostics.fallback_used
    );
}

#[test]
fn auxiliary_inequality_coupled_problem_solves_on_original_path() {
    let problem = AuxiliaryReducedFallbackProblem;
    let options = SolverOptions {
        print_level: 0,
        enable_preprocessing: true,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        user_g_scaling: Some(vec![1.0, 0.5]),
        ..SolverOptions::default()
    };

    let result = ripopt::solve(&problem, &options);

    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "Expected Optimal, got {:?}",
        result.status
    );
    assert_eq!(
        result.diagnostics.fallback_used.as_deref(),
        None,
        "auxiliary preprocessing should not be tagged as a post-failure fallback"
    );
    assert_eq!(result.x.len(), 2);
    assert!(
        (result.x[0] - 2.0).abs() < 1e-8,
        "x0={}, expected 2.0",
        result.x[0]
    );
    assert!(
        (result.x[1] - 3.0).abs() < 2e-2,
        "x1={}, expected 3.0",
        result.x[1]
    );
    assert!(
        (result.objective + 8.0).abs() < 1e-3,
        "obj={}, expected -8.0",
        result.objective
    );
    assert_eq!(result.constraint_values.len(), 2);
    assert!(result.constraint_values[0].abs() < 1e-8);
    assert!(result.constraint_values[1] <= 10.0 + 1e-8);
    assert_eq!(result.constraint_multipliers.len(), 2);
    assert!(result
        .constraint_multipliers
        .iter()
        .all(|value| value.is_finite()));
    assert_eq!(result.bound_multipliers_lower.len(), 2);
    assert_eq!(result.bound_multipliers_upper.len(), 2);
}

#[test]
fn auxiliary_failure_falls_back_to_original_nlp() {
    let problem = AuxiliaryFailureFallbackProblem {
        fail_next_constraint: Cell::new(true),
        failed_once: Cell::new(false),
    };
    let options = SolverOptions {
        print_level: 0,
        enable_preprocessing: true,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        max_iter: 200,
        tol: 1e-8,
        ..SolverOptions::default()
    };

    let result = ripopt::solve(&problem, &options);

    assert!(
        problem.failed_once.get(),
        "auxiliary preprocessing should have attempted the candidate solve"
    );
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "original NLP should solve after auxiliary failure, got {:?}",
        result.status
    );
    assert!(
        (result.x[0] - 2.0).abs() < 1e-6,
        "x0={}, expected 2.0",
        result.x[0]
    );
    assert!(
        (result.x[1] - 3.0).abs() < 1e-5,
        "x1={}, expected 3.0",
        result.x[1]
    );
    assert_ne!(
        result.diagnostics.fallback_used.as_deref(),
        Some("auxiliary_preprocessing")
    );
}

#[test]
fn auxiliary_reduced_solve_failure_falls_back_to_original_nlp() {
    let problem = AuxiliaryReducedFailureFallbackProblem {
        fail_reduced_objective_once: Cell::new(true),
        reduced_failed_once: Cell::new(false),
    };
    let options = SolverOptions {
        print_level: 0,
        enable_preprocessing: true,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        max_iter: 200,
        tol: 1e-8,
        ..SolverOptions::default()
    };

    let result = ripopt::solve(&problem, &options);

    assert!(
        problem.reduced_failed_once.get(),
        "auxiliary-reduced solve should have failed before the original retry"
    );
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "original NLP should solve after reduced-path failure, got {:?}",
        result.status
    );
    assert!((result.x[0] - 2.0).abs() < 1e-5, "x={:?}", result.x);
    assert!((result.x[1] - 3.0).abs() < 1e-5, "x={:?}", result.x);
    assert_ne!(
        result.diagnostics.fallback_used.as_deref(),
        Some("auxiliary_preprocessing")
    );
}

#[test]
fn preprocessing_disabled_bypasses_auxiliary_preprocessing_in_solve() {
    let problem = PreprocessingDisabledBypassProblem {
        zero_obj_factor_hessian_seen: Cell::new(false),
    };
    let options = SolverOptions {
        print_level: 0,
        enable_preprocessing: false,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        max_iter: 200,
        tol: 1e-8,
        ..SolverOptions::default()
    };

    let result = ripopt::solve(&problem, &options);

    assert_eq!(result.status, SolveStatus::Optimal, "Expected Optimal, got {:?}", result.status);
    assert!(
        !problem.zero_obj_factor_hessian_seen.get(),
        "auxiliary block solves call inner hessian_values with obj_factor=0"
    );
    assert!((result.x[0] - 3.0).abs() < 1e-5, "x={:?}", result.x);
    assert!((result.x[1] - 2.0).abs() < 1e-5, "x={:?}", result.x);
}

struct PostsolveRecoveryProblem {
    full_constraint_hessian_seen: Cell<bool>,
}

impl NlpProblem for PostsolveRecoveryProblem {
    fn num_variables(&self) -> usize { 3 }
    fn num_constraints(&self) -> usize { 1 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY;
        x_u[1] = f64::INFINITY;
        x_l[2] = 4.0;
        x_u[2] = 4.0;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = 0.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.5;
        x0[1] = 0.0;
        x0[2] = 4.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = (x[0] - 3.0) * (x[0] - 3.0) + (x[2] - 4.0) * (x[2] - 4.0);
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 2.0 * (x[0] - 3.0);
        grad[1] = 0.0;
        grad[2] = 2.0 * (x[2] - 4.0);
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[1] - x[0] * x[0];
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0], vec![0, 1])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = -2.0 * x[0];
        vals[1] = 1.0;
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 2], vec![0, 2])
    }

    fn hessian_values(
        &self,
        _x: &[f64],
        _new_x: bool,
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) -> bool {
        if lambda.first().copied().unwrap_or(0.0).abs() > 1e-12 {
            self.full_constraint_hessian_seen.set(true);
        }
        vals[0] = 2.0 * obj_factor - 2.0 * lambda.first().copied().unwrap_or(0.0);
        vals[1] = 2.0 * obj_factor;
        true
    }
}

#[test]
fn auxiliary_postsolve_recovers_equality_variable_after_reduced_solve() {
    let problem = PostsolveRecoveryProblem {
        full_constraint_hessian_seen: Cell::new(false),
    };
    let options = SolverOptions {
        print_level: 0,
        enable_preprocessing: true,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        max_iter: 200,
        tol: 1e-8,
        ..SolverOptions::default()
    };

    let result = ripopt::solve(&problem, &options);

    assert_eq!(result.status, SolveStatus::Optimal, "result={result:?}");
    assert!((result.x[0] - 3.0).abs() < 1e-6, "x={:?}", result.x);
    assert!((result.x[1] - 9.0).abs() < 1e-6, "x={:?}", result.x);
    assert!((result.x[2] - 4.0).abs() < 1e-12, "x={:?}", result.x);
    assert!(result.constraint_values[0].abs() < 1e-8);
    assert_eq!(result.constraint_multipliers.len(), 1);
    assert_eq!(result.bound_multipliers_lower.len(), 3);
    assert!(
        !problem.full_constraint_hessian_seen.get(),
        "postsolve path should remove the recovery row from the main reduced solve"
    );
}

#[derive(Debug)]
struct AuxiliaryGateMetrics {
    status: SolveStatus,
    objective: Option<f64>,
    constraint_violation: f64,
    iterations: usize,
}

fn auxiliary_gate_options(enable_preprocessing: bool) -> SolverOptions {
    SolverOptions {
        print_level: 0,
        enable_preprocessing,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        max_iter: 300,
        tol: 1e-8,
        early_stall_timeout: 0.0,
        ..SolverOptions::default()
    }
}

fn solve_auxiliary_gate_case<P: NlpProblem>(
    problem: &P,
    enable_preprocessing: bool,
) -> SolveResult {
    ripopt::solve(problem, &auxiliary_gate_options(enable_preprocessing))
}

fn auxiliary_gate_metrics<P: NlpProblem>(
    problem: &P,
    result: &SolveResult,
) -> AuxiliaryGateMetrics {
    let mut objective = 0.0;
    let objective = if problem.objective(&result.x, true, &mut objective) {
        Some(objective)
    } else {
        None
    };

    AuxiliaryGateMetrics {
        status: result.status,
        objective,
        constraint_violation: full_constraint_violation(problem, &result.x),
        iterations: result.iterations,
    }
}

fn full_constraint_violation<P: NlpProblem>(problem: &P, x: &[f64]) -> f64 {
    let m = problem.num_constraints();
    if m == 0 {
        return 0.0;
    }

    let mut g = vec![0.0; m];
    if !problem.constraints(x, true, &mut g) {
        return f64::INFINITY;
    }

    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    problem.constraint_bounds(&mut g_l, &mut g_u);

    let mut violation: f64 = 0.0;
    for i in 0..m {
        let row_violation = if !g[i].is_finite() {
            f64::INFINITY
        } else if g_l[i].is_finite()
            && g_u[i].is_finite()
            && (g_u[i] - g_l[i]).abs() <= 1e-12
        {
            (g[i] - 0.5 * (g_l[i] + g_u[i])).abs()
        } else {
            let lower = if g_l[i].is_finite() {
                (g_l[i] - g[i]).max(0.0)
            } else {
                0.0
            };
            let upper = if g_u[i].is_finite() {
                (g[i] - g_u[i]).max(0.0)
            } else {
                0.0
            };
            lower.max(upper)
        };
        violation = violation.max(row_violation);
    }
    violation
}

fn status_rank(status: SolveStatus) -> u8 {
    match status {
        SolveStatus::Optimal => 0,
        SolveStatus::Acceptable => 1,
        SolveStatus::LocalInfeasibility | SolveStatus::Infeasible | SolveStatus::Unbounded => 2,
        SolveStatus::MaxIterations => 3,
        SolveStatus::NumericalError
        | SolveStatus::RestorationFailed
        | SolveStatus::EvaluationError
        | SolveStatus::UserRequestedStop => 4,
        SolveStatus::InternalError => 5,
    }
}

fn is_solved(status: SolveStatus) -> bool {
    matches!(status, SolveStatus::Optimal | SolveStatus::Acceptable)
}

fn assert_auxiliary_gate_not_worse(
    name: &str,
    preprocessed: &AuxiliaryGateMetrics,
    fallback: &AuxiliaryGateMetrics,
) {
    assert!(
        status_rank(preprocessed.status) <= status_rank(fallback.status),
        "{name}: preprocessing status {:?} is worse than no-preprocessing status {:?}",
        preprocessed.status,
        fallback.status
    );

    if is_solved(preprocessed.status) && is_solved(fallback.status) {
        assert!(
            preprocessed.constraint_violation
                <= fallback.constraint_violation.max(1e-8) * 10.0 + 1e-8,
            "{name}: preprocessing full-space violation {} is worse than no-preprocessing {}",
            preprocessed.constraint_violation,
            fallback.constraint_violation
        );

        let pre_obj = preprocessed
            .objective
            .expect("solved preprocessed result should evaluate objective");
        let fallback_obj = fallback
            .objective
            .expect("solved fallback result should evaluate objective");
        let scale = pre_obj.abs().max(fallback_obj.abs()).max(1.0);
        assert!(
            pre_obj <= fallback_obj + 1e-6 * scale,
            "{name}: preprocessing objective {pre_obj} is worse than no-preprocessing {fallback_obj}"
        );
    }
}

fn assert_auxiliary_gate_iteration_budget(
    name: &str,
    preprocessed: &AuxiliaryGateMetrics,
    fallback: &AuxiliaryGateMetrics,
    allowed_extra: usize,
) {
    if is_solved(preprocessed.status) && is_solved(fallback.status) {
        assert!(
            preprocessed.iterations <= fallback.iterations + allowed_extra,
            "{name}: preprocessing iterations {} exceed no-preprocessing {} by more than {allowed_extra}",
            preprocessed.iterations,
            fallback.iterations
        );
    }
}

fn compare_auxiliary_gate_case<P: NlpProblem>(
    name: &str,
    pre_problem: &P,
    fallback_problem: &P,
) -> (AuxiliaryGateMetrics, AuxiliaryGateMetrics) {
    let preprocessed_result = solve_auxiliary_gate_case(pre_problem, true);
    let fallback_result = solve_auxiliary_gate_case(fallback_problem, false);
    let preprocessed = auxiliary_gate_metrics(pre_problem, &preprocessed_result);
    let fallback = auxiliary_gate_metrics(fallback_problem, &fallback_result);

    eprintln!("{name}: preprocessing={preprocessed:?}, no_preprocessing={fallback:?}");

    assert_auxiliary_gate_not_worse(name, &preprocessed, &fallback);
    assert_auxiliary_gate_iteration_budget(name, &preprocessed, &fallback, 5);
    (preprocessed, fallback)
}

#[test]
fn auxiliary_preprocessing_regression_gate_compares_reduced_and_fallback_paths() {
    compare_auxiliary_gate_case(
        "branch objective",
        &AuxiliaryBranchObjectiveProblem,
        &AuxiliaryBranchObjectiveProblem,
    );
    compare_auxiliary_gate_case(
        "triangular equality system",
        &AuxiliaryTriangularEndToEndProblem,
        &AuxiliaryTriangularEndToEndProblem,
    );
    compare_auxiliary_gate_case(
        "basin guard",
        &AuxiliaryBasinGuardProblem,
        &AuxiliaryBasinGuardProblem,
    );
}

#[test]
fn auxiliary_preprocessing_regression_gate_skips_inequality_coupled_path() {
    let (preprocessed, fallback) = compare_auxiliary_gate_case(
        "inequality-coupled auxiliary candidate",
        &AuxiliaryReducedFallbackProblem,
        &AuxiliaryReducedFallbackProblem,
    );
    assert!(
        is_solved(preprocessed.status),
        "inequality-coupled auxiliary candidate: preprocessing-enabled solve should still solve, got {:?}",
        preprocessed.status
    );
    assert!(
        is_solved(fallback.status),
        "inequality-coupled auxiliary candidate: no-preprocessing path should also solve, got {:?}",
        fallback.status
    );
}

// ---------------------------------------------------------------------------
// 5. Best-du restore at max_iter: min x^2 s.t. x^2=1
//    Two solutions: x=1 or x=-1. Start x0=2.0.
//    Use tight tol, expect Optimal if the problem is well-posed.
// ---------------------------------------------------------------------------

struct BestDuProblem;

impl NlpProblem for BestDuProblem {
    fn num_variables(&self) -> usize { 1 }
    fn num_constraints(&self) -> usize { 1 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0;
        g_u[0] = 1.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 2.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[0] * x[0];
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 2.0 * x[0];
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] * x[0];
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 2.0 * x[0];
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = 2.0 * obj_factor + 2.0 * lambda[0];
        true
    }
}

#[test]
fn ipm_best_du_at_maxiter() {
    let problem = BestDuProblem;
    let options = SolverOptions {
        print_level: 0,
        max_iter: 50,
        tol: 1e-8,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    assert!(
        result.status == SolveStatus::Optimal || result.status == SolveStatus::MaxIterations || result.status == SolveStatus::NumericalError,
        "Expected Optimal/MaxIterations/NumericalError, got {:?}",
        result.status
    );
    // x should be near 1.0 (the minimizer on x^2=1)
    assert!((result.x[0].abs() - 1.0).abs() < 1e-2, "x={}, expected |x|~1.0", result.x[0]);
}

// ---------------------------------------------------------------------------
// 6. L-BFGS fallback for unconstrained Rosenbrock
//    min (1-x0)^2 + 100*(x1-x0^2)^2, start (-1.2, 1.0)
// ---------------------------------------------------------------------------

struct RosenbrockLbfgs;

impl NlpProblem for RosenbrockLbfgs {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..2 {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }

    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = -1.2;
        x0[1] = 1.0;
    }

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

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1], vec![0, 0, 1])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = obj_factor * (2.0 - 400.0 * x[1] + 1200.0 * x[0] * x[0]);
        vals[1] = obj_factor * (-400.0 * x[0]);
        vals[2] = obj_factor * 200.0;
        true
    }
}

#[test]
fn ipm_lbfgs_fallback_unconstrained() {
    let problem = RosenbrockLbfgs;
    let options = SolverOptions {
        print_level: 0,
        enable_lbfgs_fallback: true,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    assert_eq!(result.status, SolveStatus::Optimal, "Expected Optimal, got {:?}", result.status);
    assert!((result.x[0] - 1.0).abs() < 1e-4, "x0={}, expected 1.0", result.x[0]);
    assert!((result.x[1] - 1.0).abs() < 1e-4, "x1={}, expected 1.0", result.x[1]);
    assert!(result.objective < 1e-6, "obj={}, expected ~0", result.objective);
}
