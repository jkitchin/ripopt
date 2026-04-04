//! Near-linear constraint detection.
//!
//! Evaluates the Jacobian at two points to detect constraints whose Jacobian
//! entries don't change — these are linear constraints with zero Hessian
//! contribution (∇²g_i = 0).

use crate::problem::NlpProblem;

/// Detect which constraints are linear by comparing Jacobian values at two points.
///
/// Evaluates Jacobian at x0 and a perturbed x1, comparing entries per constraint.
/// If all entries for constraint i change by less than `tol` (relative), it is linear.
///
/// Returns a `Vec<bool>` of length m where `true` means the constraint is linear.
pub fn detect_linear_constraints<P: NlpProblem>(problem: &P, x0: &[f64]) -> Vec<bool> {
    let n = problem.num_variables();
    let m = problem.num_constraints();

    if m == 0 {
        return Vec::new();
    }

    let (jac_rows, jac_cols) = problem.jacobian_structure();
    let jac_nnz = jac_rows.len();

    if jac_nnz == 0 {
        return vec![true; m]; // No Jacobian entries => all constraints are constant (linear)
    }

    // Get bounds for clamping the perturbed point
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    problem.bounds(&mut x_l, &mut x_u);

    // Evaluate Jacobian at x0
    let mut jac0 = vec![0.0; jac_nnz];
    problem.jacobian_values(x0, true, &mut jac0);

    // Create perturbed point x1
    let mut x1 = vec![0.0; n];
    let mut well_perturbed = vec![false; n];
    let min_pert = 1e-4; // Minimum perturbation required for reliable detection
    for i in 0..n {
        let pert = 1.0 + 0.1 * x0[i].abs();
        x1[i] = x0[i] + pert;
        // Clamp to bounds
        if x_l[i].is_finite() {
            x1[i] = x1[i].max(x_l[i]);
        }
        if x_u[i].is_finite() {
            x1[i] = x1[i].min(x_u[i]);
        }
        // If clamped back to x0[i], try perturbing in the other direction
        if (x1[i] - x0[i]).abs() < min_pert {
            x1[i] = x0[i] - pert;
            if x_l[i].is_finite() {
                x1[i] = x1[i].max(x_l[i]);
            }
            if x_u[i].is_finite() {
                x1[i] = x1[i].min(x_u[i]);
            }
        }
        well_perturbed[i] = (x1[i] - x0[i]).abs() >= min_pert;
    }

    // Evaluate Jacobian at x1
    let mut jac1 = vec![0.0; jac_nnz];
    problem.jacobian_values(&x1, true, &mut jac1);

    // Identify which constraints involve poorly-perturbed variables
    // For safety, don't mark constraints as linear if we couldn't perturb all their variables
    let mut constraint_well_tested = vec![true; m];
    for k in 0..jac_nnz {
        let row = jac_rows[k];
        let col = jac_cols[k];
        if !well_perturbed[col] && jac0[k].abs() > 1e-20 {
            constraint_well_tested[row] = false;
        }
    }

    // Compare per constraint
    let tol = 1e-12;
    let mut is_linear = vec![true; m];

    for k in 0..jac_nnz {
        let row = jac_rows[k];
        if !is_linear[row] {
            continue;
        }
        let v0 = jac0[k];
        let v1 = jac1[k];
        let diff = (v0 - v1).abs();
        let scale = v0.abs().max(v1.abs()).max(1.0);
        if diff > tol * scale {
            is_linear[row] = false;
        }
    }

    // Don't trust linearity detection for constraints we couldn't test well
    for i in 0..m {
        if !constraint_well_tested[i] {
            is_linear[i] = false;
        }
    }

    is_linear
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Problem with mixed linear and nonlinear constraints:
    /// x0 + x1 = 2 (linear), x0^2 + x1^2 = 2 (nonlinear)
    struct MixedProblem;

    impl NlpProblem for MixedProblem {
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
            g_l[0] = 2.0;
            g_u[0] = 2.0;
            g_l[1] = 2.0;
            g_u[1] = 2.0;
        }
        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 1.0;
            x0[1] = 1.0;
        }
        fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
            x[0] + x[1]
        }
        fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) {
            grad[0] = 1.0;
            grad[1] = 1.0;
        }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
            g[0] = x[0] + x[1]; // linear
            g[1] = x[0] * x[0] + x[1] * x[1]; // nonlinear
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 0, 1, 1], vec![0, 1, 0, 1])
        }
        fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
            vals[0] = 1.0; // dg0/dx0
            vals[1] = 1.0; // dg0/dx1
            vals[2] = 2.0 * x[0]; // dg1/dx0
            vals[3] = 2.0 * x[1]; // dg1/dx1
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1], vec![0, 1])
        }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
            // Only constraint 1 has Hessian: 2*lambda[1]*I
            vals[0] = 2.0 * lambda[1];
            vals[1] = 2.0 * lambda[1];
        }
    }

    #[test]
    fn test_mixed_linear_detection() {
        let prob = MixedProblem;
        let x0 = vec![1.0, 1.0];
        let flags = detect_linear_constraints(&prob, &x0);
        assert_eq!(flags.len(), 2);
        assert!(flags[0]); // constraint 0 is linear
        assert!(!flags[1]); // constraint 1 is nonlinear
    }

    /// All-linear problem
    struct AllLinear;

    impl NlpProblem for AllLinear {
        fn num_variables(&self) -> usize {
            2
        }
        fn num_constraints(&self) -> usize {
            1
        }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = 0.0;
            x_u[0] = 10.0;
            x_l[1] = 0.0;
            x_u[1] = 10.0;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 1.0;
            g_u[0] = f64::INFINITY;
        }
        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 1.0;
            x0[1] = 1.0;
        }
        fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
            x[0] + x[1]
        }
        fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) {
            grad[0] = 1.0;
            grad[1] = 1.0;
        }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
            g[0] = 2.0 * x[0] + 3.0 * x[1];
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 0], vec![0, 1])
        }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) {
            vals[0] = 2.0;
            vals[1] = 3.0;
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![], vec![])
        }
        fn hessian_values(
            &self,
            _x: &[f64],
            _new_x: bool,
                _obj_factor: f64,
            _lambda: &[f64],
            _vals: &mut [f64],
        ) {
        }
    }

    #[test]
    fn test_all_linear() {
        let prob = AllLinear;
        let x0 = vec![1.0, 1.0];
        let flags = detect_linear_constraints(&prob, &x0);
        assert_eq!(flags.len(), 1);
        assert!(flags[0]);
    }

    #[test]
    fn test_no_constraints() {
        struct NoConstraints;
        impl NlpProblem for NoConstraints {
            fn num_variables(&self) -> usize {
                2
            }
            fn num_constraints(&self) -> usize {
                0
            }
            fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
                x_l[0] = f64::NEG_INFINITY;
                x_u[0] = f64::INFINITY;
                x_l[1] = f64::NEG_INFINITY;
                x_u[1] = f64::INFINITY;
            }
            fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
            fn initial_point(&self, x0: &mut [f64]) {
                x0[0] = 1.0;
                x0[1] = 1.0;
            }
            fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
                x[0] * x[0]
            }
            fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
                grad[0] = 2.0 * x[0];
                grad[1] = 0.0;
            }
            fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) {}
            fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
                (vec![], vec![])
            }
            fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) {}
            fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
                (vec![0], vec![0])
            }
            fn hessian_values(
                &self,
                _x: &[f64],
                _new_x: bool,
                obj_factor: f64,
                _lambda: &[f64],
                vals: &mut [f64],
            ) {
                vals[0] = 2.0 * obj_factor;
            }
        }

        let prob = NoConstraints;
        let x0 = vec![1.0, 1.0];
        let flags = detect_linear_constraints(&prob, &x0);
        assert!(flags.is_empty());
    }

    /// Problem with 1 variable with very tight bounds, 1 nonlinear constraint.
    /// Variable can't be perturbed enough, so constraint_well_tested = false.
    struct BoundedVarProblem;

    impl NlpProblem for BoundedVarProblem {
        fn num_variables(&self) -> usize { 1 }
        fn num_constraints(&self) -> usize { 1 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = 1.0;
            x_u[0] = 1.0 + 1e-6; // very tight bounds
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 0.0;
            g_u[0] = 10.0;
        }
        fn initial_point(&self, x0: &mut [f64]) { x0[0] = 1.0; }
        fn objective(&self, x: &[f64], _new_x: bool) -> f64 { x[0] }
        fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) { grad[0] = 1.0; }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
            g[0] = x[0] * x[0]; // nonlinear, but perturbation too small to detect
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0], vec![0])
        }
        fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
            vals[0] = 2.0 * x[0];
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0], vec![0])
        }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
            vals[0] = 2.0 * lambda[0];
        }
    }

    #[test]
    fn test_bounded_var_prevents_detection() {
        let prob = BoundedVarProblem;
        let x0 = vec![1.0];
        let flags = detect_linear_constraints(&prob, &x0);
        assert_eq!(flags.len(), 1);
        // Variable can't be perturbed well, so conservative: won't claim linear
        assert!(!flags[0], "expected false (conservative) due to tight bounds");
    }

    /// Problem: 2 variables, 1 linear constraint involving only x0.
    /// x1 has tight bounds but its Jacobian entry is 0, so it doesn't block detection.
    struct ZeroJacEntryProblem;

    impl NlpProblem for ZeroJacEntryProblem {
        fn num_variables(&self) -> usize { 2 }
        fn num_constraints(&self) -> usize { 1 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = 0.0;
            x_u[0] = 10.0;
            x_l[1] = 1.0;
            x_u[1] = 1.0 + 1e-6; // very tight bounds on x1
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 0.0;
            g_u[0] = 10.0;
        }
        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 1.0;
            x0[1] = 1.0;
        }
        fn objective(&self, x: &[f64], _new_x: bool) -> f64 { x[0] + x[1] }
        fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) {
            grad[0] = 1.0;
            grad[1] = 1.0;
        }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
            g[0] = 2.0 * x[0]; // linear, only depends on x0
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 0], vec![0, 1])
        }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) {
            vals[0] = 2.0; // dg/dx0
            vals[1] = 0.0; // dg/dx1 = 0
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![], vec![])
        }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, _lambda: &[f64], _vals: &mut [f64]) {}
    }

    #[test]
    fn test_zero_jac_entry_not_blocking() {
        let prob = ZeroJacEntryProblem;
        let x0 = vec![1.0, 1.0];
        let flags = detect_linear_constraints(&prob, &x0);
        assert_eq!(flags.len(), 1);
        // Zero Jacobian entry for x1 doesn't block linearity detection
        assert!(flags[0], "expected true: zero jac entry should not block detection");
    }
}
