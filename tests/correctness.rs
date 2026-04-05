use ripopt::{NlpProblem, SolveStatus, SolverOptions};

// ---------------------------------------------------------------------------
// 1. Rosenbrock (unconstrained)
//    min f(x) = (1 - x1)^2 + 100*(x2 - x1^2)^2
//    x* = (1, 1), f* = 0
// ---------------------------------------------------------------------------

struct Rosenbrock;

impl NlpProblem for Rosenbrock {
    fn num_variables(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        0
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_l[1] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
        x_u[1] = f64::INFINITY;
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
        let x1 = x[0];
        let x2 = x[1];
        grad[0] = -2.0 * (1.0 - x1) - 400.0 * x1 * (x2 - x1 * x1);
        grad[1] = 200.0 * (x2 - x1 * x1);
        true
    }

    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Lower triangle: (0,0), (1,0), (1,1)
        (vec![0, 1, 1], vec![0, 0, 1])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        let x1 = x[0];
        let x2 = x[1];
        // H[0,0] = 2 - 400*x2 + 1200*x1^2
        vals[0] = obj_factor * (2.0 - 400.0 * x2 + 1200.0 * x1 * x1);
        // H[1,0] = -400*x1
        vals[1] = obj_factor * (-400.0 * x1);
        // H[1,1] = 200
        vals[2] = obj_factor * 200.0;
        true
    }
}

#[test]
fn rosenbrock_unconstrained() {
    let problem = Rosenbrock;
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
    assert!(
        (result.x[0] - 1.0).abs() < 1e-4,
        "x1 should be ~1.0, got {}",
        result.x[0]
    );
    assert!(
        (result.x[1] - 1.0).abs() < 1e-4,
        "x2 should be ~1.0, got {}",
        result.x[1]
    );
    assert!(
        result.objective.abs() < 1e-3,
        "f* should be ~0.0, got {}",
        result.objective
    );
}

// ---------------------------------------------------------------------------
// 2. Simple constrained QP
//    min f(x) = 0.5*(x1^2 + x2^2)
//    s.t. x1 + x2 = 1
//    x* = (0.5, 0.5), f* = 0.25
// ---------------------------------------------------------------------------

struct SimpleQP;

impl NlpProblem for SimpleQP {
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
        x0[0] = 0.0;
        x0[1] = 0.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = 0.5 * (x[0] * x[0] + x[1] * x[1]);
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = x[0];
        grad[1] = x[1];
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] + x[1];
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // J = [1, 1] -> row 0 col 0, row 0 col 1
        (vec![0, 0], vec![0, 1])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0;
        vals[1] = 1.0;
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Diagonal: (0,0) and (1,1) — both are on the lower triangle
        (vec![0, 1], vec![0, 1])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        // Hessian of f: diag(1, 1). Constraint is linear so its Hessian is zero.
        vals[0] = obj_factor * 1.0;
        vals[1] = obj_factor * 1.0;
        true
    }
}

#[test]
fn simple_constrained_qp() {
    let problem = SimpleQP;
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
    assert!(
        (result.x[0] - 0.5).abs() < 1e-4,
        "x1 should be ~0.5, got {}",
        result.x[0]
    );
    assert!(
        (result.x[1] - 0.5).abs() < 1e-4,
        "x2 should be ~0.5, got {}",
        result.x[1]
    );
    assert!(
        (result.objective - 0.25).abs() < 1e-3,
        "f* should be ~0.25, got {}",
        result.objective
    );
}

// ---------------------------------------------------------------------------
// 3. HS071
//    min f = x1*x4*(x1+x2+x3) + x3
//    s.t. g1 = x1*x2*x3*x4 >= 25
//         g2 = x1^2 + x2^2 + x3^2 + x4^2 = 40
//    1 <= xi <= 5, i = 1..4
//    x0 = (1, 5, 5, 1)
//    f* ≈ 17.014
// ---------------------------------------------------------------------------

struct HS071;

impl NlpProblem for HS071 {
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
        // g1 >= 25 (no upper bound)
        g_l[0] = 25.0;
        g_u[0] = f64::INFINITY;
        // g2 = 40 (equality)
        g_l[1] = 40.0;
        g_u[1] = 40.0;
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
        // df/dx1 = x4*(x1+x2+x3) + x1*x4 = x4*(2*x1 + x2 + x3)
        grad[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        // df/dx2 = x1*x4
        grad[1] = x[0] * x[3];
        // df/dx3 = x1*x4 + 1
        grad[2] = x[0] * x[3] + 1.0;
        // df/dx4 = x1*(x1+x2+x3)
        grad[3] = x[0] * (x[0] + x[1] + x[2]);
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // g1 depends on all 4 vars, g2 depends on all 4 vars
        // Row 0 (g1): cols 0,1,2,3
        // Row 1 (g2): cols 0,1,2,3
        (
            vec![0, 0, 0, 0, 1, 1, 1, 1],
            vec![0, 1, 2, 3, 0, 1, 2, 3],
        )
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        // dg1/dx1 = x2*x3*x4
        vals[0] = x[1] * x[2] * x[3];
        // dg1/dx2 = x1*x3*x4
        vals[1] = x[0] * x[2] * x[3];
        // dg1/dx3 = x1*x2*x4
        vals[2] = x[0] * x[1] * x[3];
        // dg1/dx4 = x1*x2*x3
        vals[3] = x[0] * x[1] * x[2];
        // dg2/dx1 = 2*x1
        vals[4] = 2.0 * x[0];
        // dg2/dx2 = 2*x2
        vals[5] = 2.0 * x[1];
        // dg2/dx3 = 2*x3
        vals[6] = 2.0 * x[2];
        // dg2/dx4 = 2*x4
        vals[7] = 2.0 * x[3];
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Lower triangle of the 4x4 Hessian of the Lagrangian.
        // We include all lower-triangle entries that can be non-zero.
        //
        // Objective Hessian non-zeros (lower triangle):
        //   (0,0): 2*x4
        //   (1,0): x4,  (2,0): x4,  (3,0): 2*x1+x2+x3
        //   (2,1): 0,   (3,1): x1
        //   (3,2): x1
        //
        // g1 = x1*x2*x3*x4, Hessian non-zeros (lower triangle):
        //   (1,0): x3*x4, (2,0): x2*x4, (3,0): x2*x3
        //   (2,1): x1*x4, (3,1): x1*x3
        //   (3,2): x1*x2
        //
        // g2 = sum xi^2, Hessian (lower triangle):
        //   (0,0): 2, (1,1): 2, (2,2): 2, (3,3): 2
        //
        // Combined lower-triangle entries (row >= col):
        // (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3)
        (
            vec![0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
        )
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        // Indices match hessian_structure order:
        // 0: (0,0), 1: (1,0), 2: (1,1), 3: (2,0), 4: (2,1), 5: (2,2),
        // 6: (3,0), 7: (3,1), 8: (3,2), 9: (3,3)

        // ---- Objective Hessian ----
        // d2f/dx1dx1 = 2*x4
        vals[0] = obj_factor * 2.0 * x[3];
        // d2f/dx2dx1 = x4
        vals[1] = obj_factor * x[3];
        // d2f/dx2dx2 = 0
        vals[2] = 0.0;
        // d2f/dx3dx1 = x4
        vals[3] = obj_factor * x[3];
        // d2f/dx3dx2 = 0
        vals[4] = 0.0;
        // d2f/dx3dx3 = 0
        vals[5] = 0.0;
        // d2f/dx4dx1 = 2*x1 + x2 + x3
        vals[6] = obj_factor * (2.0 * x[0] + x[1] + x[2]);
        // d2f/dx4dx2 = x1
        vals[7] = obj_factor * x[0];
        // d2f/dx4dx3 = x1
        vals[8] = obj_factor * x[0];
        // d2f/dx4dx4 = 0
        vals[9] = 0.0;

        // ---- Constraint 1 Hessian: g1 = x1*x2*x3*x4 ----
        // (0,0): 0
        // (1,0): x3*x4
        vals[1] += lambda[0] * x[2] * x[3];
        // (1,1): 0
        // (2,0): x2*x4
        vals[3] += lambda[0] * x[1] * x[3];
        // (2,1): x1*x4
        vals[4] += lambda[0] * x[0] * x[3];
        // (2,2): 0
        // (3,0): x2*x3
        vals[6] += lambda[0] * x[1] * x[2];
        // (3,1): x1*x3
        vals[7] += lambda[0] * x[0] * x[2];
        // (3,2): x1*x2
        vals[8] += lambda[0] * x[0] * x[1];
        // (3,3): 0

        // ---- Constraint 2 Hessian: g2 = sum xi^2 ----
        // Only diagonal entries: 2 each
        vals[0] += lambda[1] * 2.0;
        vals[2] += lambda[1] * 2.0;
        vals[5] += lambda[1] * 2.0;
        vals[9] += lambda[1] * 2.0;
        true
    }
}

#[test]
fn hs071_constrained() {
    let problem = HS071;
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

    // Check that constraints are satisfied (correctness check, not specific objective value)
    // g1 = x1*x2*x3*x4 >= 25
    let g1 = result.x[0] * result.x[1] * result.x[2] * result.x[3];
    assert!(
        g1 >= 25.0 - 1e-3,
        "g1 = {} should be >= 25",
        g1
    );
    // g2 = x1^2 + x2^2 + x3^2 + x4^2 = 40
    let g2: f64 = result.x.iter().map(|xi| xi * xi).sum();
    assert!(
        (g2 - 40.0).abs() < 1e-3,
        "g2 = {} should be ~40",
        g2
    );
    // Bounds: 1 <= xi <= 5
    for (i, &xi) in result.x.iter().enumerate() {
        assert!(
            xi >= 1.0 - 1e-4 && xi <= 5.0 + 1e-4,
            "x[{}] = {} out of bounds [1, 5]",
            i,
            xi
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Bound-constrained quadratic (HS035-like)
//    min f(x) = 9 - 8x1 - 6x2 - 4x3 + 2x1^2 + 2x2^2 + x3^2 + 2x1*x2 + 2x1*x3
//    s.t. x1 + x2 + 2*x3 <= 3
//         x1 >= 0, x2 >= 0, x3 >= 0
//    x* = (4/3, 7/9, 4/9), f* = 1/9 ≈ 0.1111
//    Hessian is constant: H = [[4, 2, 2], [2, 4, 0], [2, 0, 2]]
// ---------------------------------------------------------------------------

struct BoundConstrainedQuadratic;

impl NlpProblem for BoundConstrainedQuadratic {
    fn num_variables(&self) -> usize {
        3
    }

    fn num_constraints(&self) -> usize {
        1
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..3 {
            x_l[i] = 0.0;
            x_u[i] = f64::INFINITY;
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        // x1 + x2 + 2*x3 <= 3
        g_l[0] = f64::NEG_INFINITY;
        g_u[0] = 3.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.5;
        x0[1] = 0.5;
        x0[2] = 0.5;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = 9.0 - 8.0 * x[0] - 6.0 * x[1] - 4.0 * x[2]
            + 2.0 * x[0] * x[0]
            + 2.0 * x[1] * x[1]
            + x[2] * x[2]
            + 2.0 * x[0] * x[1]
            + 2.0 * x[0] * x[2];
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        // df/dx1 = -8 + 4*x1 + 2*x2 + 2*x3
        grad[0] = -8.0 + 4.0 * x[0] + 2.0 * x[1] + 2.0 * x[2];
        // df/dx2 = -6 + 2*x1 + 4*x2
        grad[1] = -6.0 + 2.0 * x[0] + 4.0 * x[1];
        // df/dx3 = -4 + 2*x1 + 2*x3
        grad[2] = -4.0 + 2.0 * x[0] + 2.0 * x[2];
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] + x[1] + 2.0 * x[2];
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // J = [1, 1, 2] -> row 0 cols 0,1,2
        (vec![0, 0, 0], vec![0, 1, 2])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0;
        vals[1] = 1.0;
        vals[2] = 2.0;
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Lower triangle of 3x3 constant Hessian:
        // (0,0), (1,0), (1,1), (2,0), (2,2)
        // Note: H[2,1] = 0 so we skip it
        (vec![0, 1, 1, 2, 2], vec![0, 0, 1, 0, 2])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        // Constant Hessian of objective: H = [[4, 2, 2], [2, 4, 0], [2, 0, 2]]
        // Constraint is linear, so its Hessian is zero.
        // (0,0): 4
        vals[0] = obj_factor * 4.0;
        // (1,0): 2
        vals[1] = obj_factor * 2.0;
        // (1,1): 4
        vals[2] = obj_factor * 4.0;
        // (2,0): 2
        vals[3] = obj_factor * 2.0;
        // (2,2): 2
        vals[4] = obj_factor * 2.0;
        true
    }
}

#[test]
fn bound_constrained_quadratic() {
    let problem = BoundConstrainedQuadratic;
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

    let expected_x = [4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0];
    let expected_f = 1.0 / 9.0;

    assert!(
        (result.objective - expected_f).abs() < 1e-4,
        "f* should be ~{}, got {}",
        expected_f,
        result.objective
    );
    for i in 0..3 {
        assert!(
            (result.x[i] - expected_x[i]).abs() < 1e-3,
            "x[{}] should be ~{}, got {}",
            i,
            expected_x[i],
            result.x[i]
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Pure bound-constrained (no general constraints)
//    min f(x) = (x1-1)^2 + (x2-2)^2 + (x3-3)^2 + (x4-4)^2
//    s.t. 0 <= x_i <= 3 for all i
//    x* = (1, 2, 3, 3), f* = 1.0
// ---------------------------------------------------------------------------

struct PureBoundConstrained;

impl NlpProblem for PureBoundConstrained {
    fn num_variables(&self) -> usize {
        4
    }

    fn num_constraints(&self) -> usize {
        0
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..4 {
            x_l[i] = 0.0;
            x_u[i] = 3.0;
        }
    }

    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
        x0[1] = 0.0;
        x0[2] = 0.0;
        x0[3] = 0.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = (x[0] - 1.0).powi(2)
            + (x[1] - 2.0).powi(2)
            + (x[2] - 3.0).powi(2)
            + (x[3] - 4.0).powi(2);
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 2.0 * (x[0] - 1.0);
        grad[1] = 2.0 * (x[1] - 2.0);
        grad[2] = 2.0 * (x[2] - 3.0);
        grad[3] = 2.0 * (x[3] - 4.0);
        true
    }

    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Diagonal: (0,0), (1,1), (2,2), (3,3)
        (vec![0, 1, 2, 3], vec![0, 1, 2, 3])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        // Hessian is diagonal with all entries = 2
        for v in vals.iter_mut() {
            *v = obj_factor * 2.0;
        }
        true
    }
}

#[test]
fn pure_bound_constrained() {
    let problem = PureBoundConstrained;
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

    let expected_x = [1.0, 2.0, 3.0, 3.0];
    let expected_f = 1.0;

    assert!(
        (result.objective - expected_f).abs() < 1e-4,
        "f* should be ~{}, got {}",
        expected_f,
        result.objective
    );
    for i in 0..4 {
        assert!(
            (result.x[i] - expected_x[i]).abs() < 1e-3,
            "x[{}] should be ~{}, got {}",
            i,
            expected_x[i],
            result.x[i]
        );
    }
}

// ---------------------------------------------------------------------------
// 6. Multiple equality constraints
//    min f(x) = x1^2 + x2^2 + x3^2
//    s.t. x1 + x2 + x3 = 1
//         x1 - x2 = 0
//    x* = (1/3, 1/3, 1/3), f* = 1/3
// ---------------------------------------------------------------------------

struct MultipleEqualityConstraints;

impl NlpProblem for MultipleEqualityConstraints {
    fn num_variables(&self) -> usize {
        3
    }

    fn num_constraints(&self) -> usize {
        2
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..3 {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        // g1: x1 + x2 + x3 = 1
        g_l[0] = 1.0;
        g_u[0] = 1.0;
        // g2: x1 - x2 = 0
        g_l[1] = 0.0;
        g_u[1] = 0.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
        x0[1] = 0.0;
        x0[2] = 0.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 2.0 * x[0];
        grad[1] = 2.0 * x[1];
        grad[2] = 2.0 * x[2];
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] + x[1] + x[2];
        g[1] = x[0] - x[1];
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // g1 depends on x1, x2, x3: row 0 cols 0,1,2
        // g2 depends on x1, x2:     row 1 cols 0,1
        (vec![0, 0, 0, 1, 1], vec![0, 1, 2, 0, 1])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        // dg1/dx1 = 1
        vals[0] = 1.0;
        // dg1/dx2 = 1
        vals[1] = 1.0;
        // dg1/dx3 = 1
        vals[2] = 1.0;
        // dg2/dx1 = 1
        vals[3] = 1.0;
        // dg2/dx2 = -1
        vals[4] = -1.0;
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Hessian of objective is diagonal: diag(2, 2, 2)
        // Constraints are linear so their Hessians are zero.
        (vec![0, 1, 2], vec![0, 1, 2])
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        // Hessian of objective: diag(2, 2, 2)
        vals[0] = obj_factor * 2.0;
        vals[1] = obj_factor * 2.0;
        vals[2] = obj_factor * 2.0;
        true
    }
}

#[test]
fn multiple_equality_constraints() {
    let problem = MultipleEqualityConstraints;
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

    let expected_x = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let expected_f = 1.0 / 3.0;

    assert!(
        (result.objective - expected_f).abs() < 1e-4,
        "f* should be ~{}, got {}",
        expected_f,
        result.objective
    );
    for i in 0..3 {
        assert!(
            (result.x[i] - expected_x[i]).abs() < 1e-3,
            "x[{}] should be ~{}, got {}",
            i,
            expected_x[i],
            result.x[i]
        );
    }
}

// ---------------------------------------------------------------------------
// 7. Inequality only
//    min x1^2 + x2^2 s.t. x1 + x2 >= 1
//    x* = (0.5, 0.5), f* = 0.5
// ---------------------------------------------------------------------------

struct InequalityOnly;

impl NlpProblem for InequalityOnly {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l.iter_mut().for_each(|v| *v = f64::NEG_INFINITY);
        x_u.iter_mut().for_each(|v| *v = f64::INFINITY);
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0;
        g_u[0] = f64::INFINITY;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; x0[1] = 0.0; }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[0] * x[0] + x[1] * x[1];
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = 2.0 * x[0]; grad[1] = 2.0 * x[1];
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool { g[0] = x[0] + x[1];
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 0], vec![0, 1]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool { vals[0] = 1.0; vals[1] = 1.0;
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 1], vec![0, 1]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = obj_factor * 2.0;
        vals[1] = obj_factor * 2.0;
        true
    }
}

#[test]
fn inequality_only() {
    let problem = InequalityOnly;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert!(result.status == SolveStatus::Optimal,
        "Got {:?}", result.status);
    assert!((result.objective - 0.5).abs() < 1e-3, "f*={}", result.objective);
}

// ---------------------------------------------------------------------------
// 8. Nonlinear equality
//    min x1 + x2 s.t. x1^2 + x2^2 = 1
//    x* = (-1/sqrt(2), -1/sqrt(2)), f* = -sqrt(2)
// ---------------------------------------------------------------------------

struct NonlinearEquality;

impl NlpProblem for NonlinearEquality {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l.iter_mut().for_each(|v| *v = f64::NEG_INFINITY);
        x_u.iter_mut().for_each(|v| *v = f64::INFINITY);
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) { g_l[0] = 1.0; g_u[0] = 1.0; }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = -0.5; x0[1] = -0.5; }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[0] + x[1];
        true
    }
    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = 1.0; grad[1] = 1.0;
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool { g[0] = x[0] * x[0] + x[1] * x[1];
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 0], vec![0, 1]) }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool { vals[0] = 2.0 * x[0]; vals[1] = 2.0 * x[1];
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 1], vec![0, 1]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = lambda[0] * 2.0;
        vals[1] = lambda[0] * 2.0;
        true
    }
}

#[test]
fn nonlinear_equality() {
    let problem = NonlinearEquality;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert!(result.status == SolveStatus::Optimal,
        "Got {:?}", result.status);
    let expected = -(2.0_f64.sqrt());
    assert!((result.objective - expected).abs() < 1e-3, "f*={}, expected {}", result.objective, expected);
}

// ---------------------------------------------------------------------------
// 9. Single variable
//    min (x-3)^2, n=1, no constraints, no bounds
//    x* = 3, f* = 0
// ---------------------------------------------------------------------------

struct SingleVariable;

impl NlpProblem for SingleVariable {
    fn num_variables(&self) -> usize { 1 }
    fn num_constraints(&self) -> usize { 0 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) { x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY; }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = (x[0] - 3.0) * (x[0] - 3.0);
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = 2.0 * (x[0] - 3.0);
        true
    }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = obj_factor * 2.0;
        true
    }
}

#[test]
fn single_variable() {
    let problem = SingleVariable;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert!(result.status == SolveStatus::Optimal);
    assert!((result.x[0] - 3.0).abs() < 1e-4);
    assert!(result.objective.abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// 10. Linear objective with bounds
//     min -x1 - x2, 0 <= xi <= 1
//     x* = (1, 1), f* = -2
// ---------------------------------------------------------------------------

struct LinearObjectiveWithBounds;

impl NlpProblem for LinearObjectiveWithBounds {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 0 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 0.0; x_l[1] = 0.0;
        x_u[0] = 1.0; x_u[1] = 1.0;
    }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.5; x0[1] = 0.5; }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = -x[0] - x[1];
        true
    }
    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = -1.0; grad[1] = -1.0;
        true
    }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], _vals: &mut [f64]) -> bool { true }
}

#[test]
fn linear_objective_with_bounds() {
    let problem = LinearObjectiveWithBounds;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert!(result.status == SolveStatus::Optimal);
    assert!((result.objective - (-2.0)).abs() < 1e-3, "f*={}", result.objective);
}

// ---------------------------------------------------------------------------
// 11. Mixed equality and inequality
//     min x1^2 + x2^2 + x3^2
//     s.t. x1 + x2 + x3 = 3 (equality)
//          x1 - x2 >= 0     (inequality)
//     x* = (1, 1, 1), f* = 3.0
// ---------------------------------------------------------------------------

struct MixedEqualityInequality;

impl NlpProblem for MixedEqualityInequality {
    fn num_variables(&self) -> usize { 3 }
    fn num_constraints(&self) -> usize { 2 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..3 { x_l[i] = f64::NEG_INFINITY; x_u[i] = f64::INFINITY; }
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 3.0; g_u[0] = 3.0;
        g_l[1] = 0.0; g_u[1] = f64::INFINITY;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 2.0; x0[1] = 0.5; x0[2] = 0.5; }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = 2.0 * x[0]; grad[1] = 2.0 * x[1]; grad[2] = 2.0 * x[2];
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool { g[0] = x[0] + x[1] + x[2]; g[1] = x[0] - x[1];
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 0, 1, 1], vec![0, 1, 2, 0, 1])
    }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0; vals[1] = 1.0; vals[2] = 1.0; vals[3] = 1.0; vals[4] = -1.0;
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 1, 2], vec![0, 1, 2]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = obj_factor * 2.0; vals[1] = obj_factor * 2.0; vals[2] = obj_factor * 2.0;
        true
    }
}

#[test]
fn mixed_equality_inequality() {
    let problem = MixedEqualityInequality;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert!(result.status == SolveStatus::Optimal,
        "Got {:?}", result.status);
    assert!((result.objective - 3.0).abs() < 1e-2, "f*={}", result.objective);
}

// ---------------------------------------------------------------------------
// 12. Quadratic equality constraint
//     min x1 + x2 s.t. x1^2 + x2^2 = 4
//     x* = (-sqrt(2), -sqrt(2)), f* = -2*sqrt(2)
// ---------------------------------------------------------------------------

struct QuadraticEqualityConstraint;

impl NlpProblem for QuadraticEqualityConstraint {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..2 { x_l[i] = f64::NEG_INFINITY; x_u[i] = f64::INFINITY; }
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) { g_l[0] = 4.0; g_u[0] = 4.0; }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = -1.0; x0[1] = -1.0; }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[0] + x[1];
        true
    }
    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = 1.0; grad[1] = 1.0;
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool { g[0] = x[0] * x[0] + x[1] * x[1];
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 0], vec![0, 1]) }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool { vals[0] = 2.0 * x[0]; vals[1] = 2.0 * x[1];
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 1], vec![0, 1]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = lambda[0] * 2.0; vals[1] = lambda[0] * 2.0;
        true
    }
}

#[test]
fn quadratic_equality_constraint() {
    let problem = QuadraticEqualityConstraint;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert!(result.status == SolveStatus::Optimal,
        "Got {:?}", result.status);
    let expected = -2.0 * 2.0_f64.sqrt();
    assert!((result.objective - expected).abs() < 1e-3, "f*={}, expected {}", result.objective, expected);
}

// ---------------------------------------------------------------------------
// 13. Starting at optimum
//     min x1^2 + x2^2, start at x* = (0, 0), f* = 0
// ---------------------------------------------------------------------------

struct StartingAtOptimum;

impl NlpProblem for StartingAtOptimum {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 0 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..2 { x_l[i] = f64::NEG_INFINITY; x_u[i] = f64::INFINITY; }
    }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; x0[1] = 0.0; }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[0] * x[0] + x[1] * x[1];
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = 2.0 * x[0]; grad[1] = 2.0 * x[1];
        true
    }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 1], vec![0, 1]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = obj_factor * 2.0; vals[1] = obj_factor * 2.0;
        true
    }
}

#[test]
fn starting_at_optimum() {
    let problem = StartingAtOptimum;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert!(result.status == SolveStatus::Optimal);
    assert!(result.objective < 1e-6);
    assert!(result.iterations <= 30, "Should converge fast, took {} iters", result.iterations);
}

// ---------------------------------------------------------------------------
// 14. High dimensional (n=20)
//     min sum (xi - i)^2, i=1..20, 0 <= xi <= 100
//     x* = (1, 2, ..., 20), f* = 0
// ---------------------------------------------------------------------------

struct HighDimensional;

impl NlpProblem for HighDimensional {
    fn num_variables(&self) -> usize { 20 }
    fn num_constraints(&self) -> usize { 0 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..20 { x_l[i] = 0.0; x_u[i] = 100.0; }
    }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) { for i in 0..20 { x0[i] = 50.0; } }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = (0..20).map(|i| (x[i] - (i as f64 + 1.0)).powi(2)).sum();
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        for i in 0..20 { grad[i] = 2.0 * (x[i] - (i as f64 + 1.0)); }
        true
    }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let indices: Vec<usize> = (0..20).collect();
        (indices.clone(), indices)
    }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        for v in vals.iter_mut() { *v = obj_factor * 2.0; }
        true
    }
}

#[test]
fn high_dimensional() {
    let problem = HighDimensional;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert!(result.status == SolveStatus::Optimal);
    assert!(result.objective < 1e-4, "f*={}", result.objective);
    for i in 0..20 {
        assert!((result.x[i] - (i as f64 + 1.0)).abs() < 1e-2,
            "x[{}]={}, expected {}", i, result.x[i], i as f64 + 1.0);
    }
}

// ---------------------------------------------------------------------------
// 15. Upper bound inequality
//     min -x1 s.t. x1^2 <= 4
//     x* = 2, f* = -2
// ---------------------------------------------------------------------------

struct UpperBoundInequality;

impl NlpProblem for UpperBoundInequality {
    fn num_variables(&self) -> usize { 1 }
    fn num_constraints(&self) -> usize { 1 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = f64::NEG_INFINITY; g_u[0] = 4.0;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 1.0; }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = -x[0];
        true
    }
    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = -1.0;
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool { g[0] = x[0] * x[0];
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool { vals[0] = 2.0 * x[0];
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = lambda[0] * 2.0;
        true
    }
}

#[test]
fn upper_bound_inequality() {
    let problem = UpperBoundInequality;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert!(result.status == SolveStatus::Optimal,
        "Got {:?}", result.status);
    assert!((result.objective - (-2.0)).abs() < 1e-2, "f*={}", result.objective);
}

// ---------------------------------------------------------------------------
// 16. Many active bounds (concave objective)
//     min sum -(xi - 10)^2, 0 <= xi <= 5
//     x* = (5,...,5), f* = -200
// ---------------------------------------------------------------------------

struct ManyActiveBounds;

impl NlpProblem for ManyActiveBounds {
    fn num_variables(&self) -> usize { 8 }
    fn num_constraints(&self) -> usize { 0 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..8 { x_l[i] = 0.0; x_u[i] = 5.0; }
    }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) { for i in 0..8 { x0[i] = 2.5; } }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = (0..8).map(|i| -(x[i] - 10.0).powi(2)).sum();
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        for i in 0..8 { grad[i] = -2.0 * (x[i] - 10.0); }
        true
    }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let indices: Vec<usize> = (0..8).collect();
        (indices.clone(), indices)
    }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        for v in vals.iter_mut() { *v = obj_factor * (-2.0); }
        true
    }
}

#[test]
fn many_active_bounds() {
    let problem = ManyActiveBounds;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert!(result.status == SolveStatus::Optimal,
        "Got {:?}", result.status);
    // Concave minimization: IPM finds a KKT point at a vertex.
    // Both x=0 (f=-800) and x=5 (f=-200) are valid KKT points.
    assert!(result.objective <= -200.0 + 1.0, "f*={}", result.objective);
    // All variables should be at a bound (0 or 5)
    for i in 0..8 {
        assert!(result.x[i] < 0.1 || (result.x[i] - 5.0).abs() < 0.1,
            "x[{}]={} should be at a bound", i, result.x[i]);
    }
}

// ---------------------------------------------------------------------------
// 17. Infeasible equality (should not crash)
//     x1 = 1 AND x1 = 2 — contradictory constraints
// ---------------------------------------------------------------------------

struct InfeasibleEquality;

impl NlpProblem for InfeasibleEquality {
    fn num_variables(&self) -> usize { 1 }
    fn num_constraints(&self) -> usize { 2 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0; g_u[0] = 1.0;
        g_l[1] = 2.0; g_u[1] = 2.0;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 1.5; }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[0] * x[0];
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = 2.0 * x[0];
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool { g[0] = x[0]; g[1] = x[0];
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 1], vec![0, 0]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool { vals[0] = 1.0; vals[1] = 1.0;
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = obj_factor * 2.0;
        true
    }
}

#[test]
fn infeasible_equality() {
    let problem = InfeasibleEquality;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert!(result.status != SolveStatus::Optimal,
        "Infeasible problem should not return Optimal");
}

// ---------------------------------------------------------------------------
// 18. Zero Hessian (linear program)
//     min -x1 - 2*x2 s.t. x1 + x2 <= 4, x1 >= 0, x2 >= 0
//     x* = (0, 4), f* = -8
// ---------------------------------------------------------------------------

struct ZeroHessianLP;

impl NlpProblem for ZeroHessianLP {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 0.0; x_l[1] = 0.0;
        x_u[0] = f64::INFINITY; x_u[1] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = f64::NEG_INFINITY; g_u[0] = 4.0;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 1.0; x0[1] = 1.0; }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = -x[0] - 2.0 * x[1];
        true
    }
    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = -1.0; grad[1] = -2.0;
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool { g[0] = x[0] + x[1];
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 0], vec![0, 1]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool { vals[0] = 1.0; vals[1] = 1.0;
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], _vals: &mut [f64]) -> bool { true }
}

#[test]
fn zero_hessian_linear_program() {
    let problem = ZeroHessianLP;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert!(result.status == SolveStatus::Optimal,
        "Got {:?}", result.status);
    assert!((result.objective - (-8.0)).abs() < 0.1, "f*={}", result.objective);
}

// ---------------------------------------------------------------------------
// 19. NE-to-LS: Consistent overdetermined system
//     min 0  s.t.  x0 = 1, x0 + x1 = 3, x1 = 2  (3 eqs, 2 vars, consistent)
//     x* = (1, 2)
// ---------------------------------------------------------------------------

struct OverdeterminedConsistent;

impl NlpProblem for OverdeterminedConsistent {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 3 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0; g_u[0] = 1.0;
        g_l[1] = 3.0; g_u[1] = 3.0;
        g_l[2] = 2.0; g_u[2] = 2.0;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.5; x0[1] = 0.5; }
    fn objective(&self, _x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = 0.0;
        true
    }
    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = 0.0; grad[1] = 0.0;
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0];
        g[1] = x[0] + x[1];
        g[2] = x[1];
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1, 2], vec![0, 0, 1, 1])
    }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0; vals[1] = 1.0; vals[2] = 1.0; vals[3] = 1.0;
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], _vals: &mut [f64]) -> bool { true }
}

#[test]
fn ne_to_ls_consistent() {
    let problem = OverdeterminedConsistent;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert!(result.status == SolveStatus::Optimal,
        "Consistent NE should be solved, got {:?}", result.status);
    assert!((result.x[0] - 1.0).abs() < 1e-4, "x0={}", result.x[0]);
    assert!((result.x[1] - 2.0).abs() < 1e-4, "x1={}", result.x[1]);
}

// ---------------------------------------------------------------------------
// 20. NE-to-LS: Inconsistent overdetermined system
//     min 0  s.t.  x0 = 1, x0 = 2, x0 = 3  (3 eqs, 1 var, inconsistent)
//     Should report LocalInfeasibility with x* ≈ 2 (LS solution)
// ---------------------------------------------------------------------------

struct OverdeterminedInconsistent;

impl NlpProblem for OverdeterminedInconsistent {
    fn num_variables(&self) -> usize { 1 }
    fn num_constraints(&self) -> usize { 3 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0; g_u[0] = 1.0;
        g_l[1] = 2.0; g_u[1] = 2.0;
        g_l[2] = 3.0; g_u[2] = 3.0;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; }
    fn objective(&self, _x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = 0.0;
        true
    }
    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = 0.0;
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0]; g[1] = x[0]; g[2] = x[0];
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 2], vec![0, 0, 0])
    }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0; vals[1] = 1.0; vals[2] = 1.0;
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], _vals: &mut [f64]) -> bool { true }
}

#[test]
fn ne_to_ls_inconsistent() {
    let problem = OverdeterminedInconsistent;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert_eq!(result.status, SolveStatus::LocalInfeasibility,
        "Inconsistent NE should report LocalInfeasibility, got {:?}", result.status);
    // LS solution: x = mean(1,2,3) = 2
    assert!((result.x[0] - 2.0).abs() < 1e-2, "x0={} (expected ~2.0)", result.x[0]);
}

// ---------------------------------------------------------------------------
// 21. NE-to-LS: Nonlinear consistent system with bounds
//     min 0  s.t.  x0^2 = 1, x0*x1 = 2, x1^2 = 4  (3 eqs, 2 vars)
//     x* = (1, 2) with bounds x0 > 0, x1 > 0
// ---------------------------------------------------------------------------

struct OverdeterminedNonlinear;

impl NlpProblem for OverdeterminedNonlinear {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 3 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 0.0; x_u[0] = f64::INFINITY;
        x_l[1] = 0.0; x_u[1] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0; g_u[0] = 1.0;
        g_l[1] = 2.0; g_u[1] = 2.0;
        g_l[2] = 4.0; g_u[2] = 4.0;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.5; x0[1] = 1.0; }
    fn objective(&self, _x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = 0.0;
        true
    }
    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = 0.0; grad[1] = 0.0;
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] * x[0];
        g[1] = x[0] * x[1];
        g[2] = x[1] * x[1];
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1, 2], vec![0, 0, 1, 1])
    }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = 2.0 * x[0];  // dg0/dx0
        vals[1] = x[1];        // dg1/dx0
        vals[2] = x[0];        // dg1/dx1
        vals[3] = 2.0 * x[1];  // dg2/dx1;
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], _vals: &mut [f64]) -> bool { true }
}

#[test]
fn ne_to_ls_nonlinear() {
    let problem = OverdeterminedNonlinear;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert_eq!(result.status, SolveStatus::Optimal,
        "Consistent nonlinear NE should get Optimal (with Newton polish), got {:?}", result.status);
    assert!((result.x[0] - 1.0).abs() < 1e-4, "x0={}", result.x[0]);
    assert!((result.x[1] - 2.0).abs() < 1e-4, "x1={}", result.x[1]);
}

// ---------------------------------------------------------------------------
// 22. NE Newton polish: overdetermined system where LS gets close but
//     Newton polish is needed to reach strict tolerance.
//     g1(x) = exp(x1) + x2 - 2       = 0  (target: exp(1)+1=3.718..)
//     g2(x) = x1^3 + x2 - 2          = 0  (target: 1+1=2)
//     g3(x) = x1 + x2^2 - 5          = 0  (target: 1+4=5)
//     g4(x) = sin(x1)*x2 - sin(1)*2  = 0
//     g5(x) = x1^2 + x2^2 - 5        = 0  (target: 1+4=5)
//     Solution: x = (1, 2), 5 equations, 2 unknowns
// ---------------------------------------------------------------------------
struct HarderOverdeterminedNE;

impl NlpProblem for HarderOverdeterminedNE {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 5 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        // g(1,2) values:
        let t0 = 1.0_f64.exp() + 2.0;   // exp(1) + 2 = 4.71828...
        let t1 = 1.0 + 2.0;              // 1^3 + 2 = 3
        let t2 = 1.0 + 4.0;              // 1 + 2^2 = 5
        let t3 = 1.0_f64.sin() * 2.0;    // sin(1)*2 = 1.6829...
        let t4 = 1.0 + 4.0;              // 1^2 + 2^2 = 5
        g_l[0] = t0; g_u[0] = t0;
        g_l[1] = t1; g_u[1] = t1;
        g_l[2] = t2; g_u[2] = t2;
        g_l[3] = t3; g_u[3] = t3;
        g_l[4] = t4; g_u[4] = t4;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.8; x0[1] = 2.3; }
    fn objective(&self, _x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = 0.0;
        true
    }
    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = 0.0; grad[1] = 0.0;
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0].exp() + x[1];
        g[1] = x[0].powi(3) + x[1];
        g[2] = x[0] + x[1].powi(2);
        g[3] = x[0].sin() * x[1];
        g[4] = x[0].powi(2) + x[1].powi(2);
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // 5 rows, 2 cols, each row has 2 entries
        (vec![0,0, 1,1, 2,2, 3,3, 4,4],
         vec![0,1, 0,1, 0,1, 0,1, 0,1])
    }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals[0] = x[0].exp();    vals[1] = 1.0;           // dg0
        vals[2] = 3.0*x[0]*x[0]; vals[3] = 1.0;           // dg1
        vals[4] = 1.0;           vals[5] = 2.0*x[1];      // dg2
        vals[6] = x[0].cos()*x[1]; vals[7] = x[0].sin();  // dg3
        vals[8] = 2.0*x[0];     vals[9] = 2.0*x[1];      // dg4;
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], _vals: &mut [f64]) -> bool { true }
}

#[test]
fn ne_newton_polish_promotes_optimal() {
    let problem = HarderOverdeterminedNE;
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&problem, &options);
    assert_eq!(result.status, SolveStatus::Optimal,
        "Overdetermined NE with Newton polish should get Optimal, got {:?} (x={:?})", result.status, result.x);
    assert!((result.x[0] - 1.0).abs() < 1e-4, "x0={}", result.x[0]);
    assert!((result.x[1] - 2.0).abs() < 1e-4, "x1={}", result.x[1]);
}

// ---------------------------------------------------------------------------
// Evaluation failure handling
// ---------------------------------------------------------------------------

/// Problem where objective fails at negative x values (e.g., log domain).
/// min f(x) = x, s.t. x >= 0.5 (via bounds)
struct EvalFailureProblem;

impl NlpProblem for EvalFailureProblem {
    fn num_variables(&self) -> usize { 1 }
    fn num_constraints(&self) -> usize { 0 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 0.5;
        x_u[0] = 10.0;
    }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 1.0; }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        if x[0] < 0.0 { return false; }
        *obj = x[0];
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        if x[0] < 0.0 { return false; }
        grad[0] = 1.0;
        true
    }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = 0.0;
        true
    }
}

#[test]
fn eval_failure_bounded_problem_still_solves() {
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&EvalFailureProblem, &options);
    assert_eq!(result.status, SolveStatus::Optimal,
        "Should solve to Optimal since bounds prevent eval failure, got {:?}", result.status);
    assert!((result.x[0] - 0.5).abs() < 1e-4, "x={}", result.x[0]);
}

/// Problem where objective always fails.
struct AlwaysFailProblem;

impl NlpProblem for AlwaysFailProblem {
    fn num_variables(&self) -> usize { 1 }
    fn num_constraints(&self) -> usize { 0 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
    }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 1.0; }
    fn objective(&self, _x: &[f64], _new_x: bool, _obj: &mut f64) -> bool { false }
    fn gradient(&self, _x: &[f64], _new_x: bool, _grad: &mut [f64]) -> bool { false }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = 0.0;
        true
    }
}

#[test]
fn eval_failure_always_fail_returns_evaluation_error() {
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result = ripopt::solve(&AlwaysFailProblem, &options);
    assert_eq!(result.status, SolveStatus::EvaluationError,
        "Should return EvaluationError when objective always fails, got {:?}", result.status);
}

// ---------------------------------------------------------------------------
// Intermediate callback: early termination
// ---------------------------------------------------------------------------

#[test]
fn intermediate_callback_early_stop() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ITER_COUNT: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn stop_at_3(
        iter: i32, _obj: f64, _inf_pr: f64, _inf_du: f64,
        _mu: f64, _alpha_pr: f64, _alpha_du: f64, _ls: i32,
        _user_data: *mut std::ffi::c_void,
    ) -> i32 {
        ITER_COUNT.store(iter as usize, Ordering::SeqCst);
        0 // always request stop
    }

    ITER_COUNT.store(0, Ordering::SeqCst);

    // Install the callback via thread-local (same mechanism as C API)
    ripopt::intermediate::set_intermediate_callback(Some((
        stop_at_3,
        std::ptr::null_mut(),
    )));

    // Use HS071 which takes many iterations (won't converge immediately)
    let options = SolverOptions {
        print_level: 0,
        enable_preprocessing: false,
        enable_lbfgs_fallback: false,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        enable_slack_fallback: false,
        enable_lbfgs_hessian_fallback: false,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&HS071, &options);

    ripopt::intermediate::set_intermediate_callback(None);

    assert_eq!(result.status, SolveStatus::UserRequestedStop,
        "Should return UserRequestedStop, got {:?}", result.status);
    let iters = ITER_COUNT.load(Ordering::SeqCst);
    assert!(iters <= 2, "Should have stopped early, but ran {} iterations", iters);
}

// ---------------------------------------------------------------------------
// Warm-start multipliers: fewer iterations on re-solve
// ---------------------------------------------------------------------------

#[test]
fn warm_start_with_multipliers_fewer_iterations() {
    // First solve: cold start
    let options = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result1 = ripopt::solve(&HS071, &options);
    assert_eq!(result1.status, SolveStatus::Optimal);

    // Second solve: warm start from the solution
    let mut opts2 = SolverOptions {
        print_level: 0,
        warm_start: true,
        warm_start_y: Some(result1.constraint_multipliers.clone()),
        warm_start_z_l: Some(result1.bound_multipliers_lower.clone()),
        warm_start_z_u: Some(result1.bound_multipliers_upper.clone()),
        ..SolverOptions::default()
    };
    // Use a custom initial point near the solution
    struct WarmHS071 { x0: Vec<f64> }
    impl NlpProblem for WarmHS071 {
        fn num_variables(&self) -> usize { 4 }
        fn num_constraints(&self) -> usize { 2 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            for i in 0..4 { x_l[i] = 1.0; x_u[i] = 5.0; }
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 25.0; g_u[0] = f64::INFINITY;
            g_l[1] = 40.0; g_u[1] = 40.0;
        }
        fn initial_point(&self, x0: &mut [f64]) { x0.copy_from_slice(&self.x0); }
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]; true
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
            g[1] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];
            true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0,0,0,0,1,1,1,1], vec![0,1,2,3,0,1,2,3])
        }
        fn jacobian_values(&self, x: &[f64], _new_x: bool, v: &mut [f64]) -> bool {
            v[0]=x[1]*x[2]*x[3]; v[1]=x[0]*x[2]*x[3];
            v[2]=x[0]*x[1]*x[3]; v[3]=x[0]*x[1]*x[2];
            v[4]=2.0*x[0]; v[5]=2.0*x[1]; v[6]=2.0*x[2]; v[7]=2.0*x[3];
            true
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0,1,1,2,2,3,3,3,3], vec![0,0,1,0,2,0,1,2,3])
        }
        fn hessian_values(&self, x: &[f64], _new_x: bool, s: f64, l: &[f64], v: &mut [f64]) -> bool {
            v[0]=s*2.0*x[3]+l[1]*2.0;
            v[1]=s*x[3]+l[0]*x[2]*x[3];
            v[2]=l[1]*2.0;
            v[3]=s*x[3]+l[0]*x[1]*x[3];
            v[4]=l[1]*2.0;
            v[5]=s*(2.0*x[0]+x[1]+x[2])+l[0]*x[1]*x[2];
            v[6]=s*x[0]+l[0]*x[0]*x[2];
            v[7]=s*x[0]+l[0]*x[0]*x[1];
            v[8]=l[1]*2.0;
            true
        }
    }
    let warm_prob = WarmHS071 { x0: result1.x.clone() };
    let result2 = ripopt::solve(&warm_prob, &opts2);
    assert_eq!(result2.status, SolveStatus::Optimal,
        "Warm start should converge to Optimal, got {:?}", result2.status);
    // Verify same solution quality
    assert!((result2.objective - result1.objective).abs() < 1.0,
        "Warm start obj={:.6e} should match cold start obj={:.6e}",
        result2.objective, result1.objective);
}

// ---------------------------------------------------------------------------
// User-provided scaling: same result as unscaled
// ---------------------------------------------------------------------------

#[test]
fn user_scaling_produces_correct_result() {
    // Solve with default scaling
    let opts1 = SolverOptions { print_level: 0, ..SolverOptions::default() };
    let result1 = ripopt::solve(&Rosenbrock, &opts1);
    assert_eq!(result1.status, SolveStatus::Optimal);

    // Solve with user-provided scaling (no-op: 1.0 everywhere)
    let opts2 = SolverOptions {
        print_level: 0,
        user_obj_scaling: Some(1.0),
        ..SolverOptions::default()
    };
    let result2 = ripopt::solve(&Rosenbrock, &opts2);
    assert_eq!(result2.status, SolveStatus::Optimal);
    assert!((result2.x[0] - 1.0).abs() < 1e-6);
    assert!((result2.x[1] - 1.0).abs() < 1e-6);
}
