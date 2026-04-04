use crate::problem::NlpProblem;

/// NLP problem wrapper that introduces explicit slack variables for inequality constraints.
///
/// Transforms:
///   Original:  min f(x)  s.t. g_l <= g(x) <= g_u,  x_l <= x <= x_u
///   Slack:     min f(x)  s.t. g(x) - s = 0,  x_l <= x <= x_u,  g_l <= s <= g_u
///
/// All constraints become equalities. Inequality bounds move to slack variable bounds,
/// handled by the existing z_l/z_u barrier machinery — no heuristic (2,2) block needed.
///
/// Variable layout: `[x(n), s(m)]`. Jacobian: `[J(m×n) | -I(m×m)]`.
/// Hessian: only x-x block (slacks are linear).
///
/// Uses dynamic dispatch (`&dyn NlpProblem`) to break infinite monomorphization
/// recursion when called from inside the generic `solve_ipm<P>`.
pub struct SlackFormulation<'a> {
    inner: &'a dyn NlpProblem,
    n_orig: usize,
    m_orig: usize,
    inner_jac_rows: Vec<usize>,
    inner_jac_cols: Vec<usize>,
    inner_hess_rows: Vec<usize>,
    inner_hess_cols: Vec<usize>,
    x_init: Vec<f64>,
    s_init: Vec<f64>,
}

impl<'a> SlackFormulation<'a> {
    /// Create a new slack formulation wrapper.
    ///
    /// - `inner`: the original NLP problem (via dynamic dispatch)
    /// - `x_warmstart`: primal variables from the failed solve (used as initial point)
    pub fn new(inner: &'a dyn NlpProblem, x_warmstart: &[f64]) -> Self {
        let n = inner.num_variables();
        let m = inner.num_constraints();

        let (inner_jac_rows, inner_jac_cols) = inner.jacobian_structure();
        let (inner_hess_rows, inner_hess_cols) = inner.hessian_structure();

        // Compute initial slack values: s = g(x) clamped to [g_l, g_u]
        let mut g_val = vec![0.0; m];
        if m > 0 {
            inner.constraints(x_warmstart, true, &mut g_val);
        }
        let mut g_l = vec![0.0; m];
        let mut g_u = vec![0.0; m];
        if m > 0 {
            inner.constraint_bounds(&mut g_l, &mut g_u);
        }

        let mut s_init = vec![0.0; m];
        for i in 0..m {
            let lo = if g_l[i].is_finite() { g_l[i] } else { f64::NEG_INFINITY };
            let hi = if g_u[i].is_finite() { g_u[i] } else { f64::INFINITY };
            debug_assert!(lo <= hi, "Inconsistent constraint bounds: g_l[{}]={} > g_u[{}]={}", i, lo, i, hi);
            s_init[i] = g_val[i].clamp(lo, hi);
        }

        SlackFormulation {
            inner,
            n_orig: n,
            m_orig: m,
            inner_jac_rows,
            inner_jac_cols,
            inner_hess_rows,
            inner_hess_cols,
            x_init: x_warmstart.to_vec(),
            s_init,
        }
    }
}

impl NlpProblem for SlackFormulation<'_> {
    fn num_variables(&self) -> usize {
        self.n_orig + self.m_orig
    }

    fn num_constraints(&self) -> usize {
        self.m_orig
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        let n = self.n_orig;

        // First n: original variable bounds (fill output slices directly)
        self.inner.bounds(&mut x_l[..n], &mut x_u[..n]);

        // Next m: slack bounds = original constraint bounds
        self.inner.constraint_bounds(&mut x_l[n..], &mut x_u[n..]);
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        // All constraints are equalities: g(x) - s = 0
        for i in 0..self.m_orig {
            g_l[i] = 0.0;
            g_u[i] = 0.0;
        }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        let n = self.n_orig;
        let m = self.m_orig;

        x0[..n].copy_from_slice(&self.x_init);
        x0[n..n + m].copy_from_slice(&self.s_init);
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        self.inner.objective(&x[..self.n_orig], _new_x)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let n = self.n_orig;
        let m = self.m_orig;

        self.inner.gradient(&x[..n], _new_x, &mut grad[..n]);
        // Slack gradient is zero (slacks don't appear in objective)
        for i in 0..m {
            grad[n + i] = 0.0;
        }
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let n = self.n_orig;
        let m = self.m_orig;

        // g_slack[i] = g_orig[i] - s[i]
        self.inner.constraints(&x[..n], _new_x, g);
        for i in 0..m {
            g[i] -= x[n + i];
        }
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.n_orig;
        let m = self.m_orig;

        let inner_nnz = self.inner_jac_rows.len();
        let total_nnz = inner_nnz + m;

        let mut rows = Vec::with_capacity(total_nnz);
        let mut cols = Vec::with_capacity(total_nnz);

        // Inner Jacobian entries (same rows/cols)
        rows.extend_from_slice(&self.inner_jac_rows);
        cols.extend_from_slice(&self.inner_jac_cols);

        // -I entries for slacks: (i, n+i) for i in 0..m
        for i in 0..m {
            rows.push(i);
            cols.push(n + i);
        }

        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        let n = self.n_orig;
        let m = self.m_orig;
        let inner_nnz = self.inner_jac_rows.len();

        // Inner Jacobian values
        self.inner.jacobian_values(&x[..n], _new_x, &mut vals[..inner_nnz]);

        // -1 for each slack diagonal
        for i in 0..m {
            vals[inner_nnz + i] = -1.0;
        }
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Only x-x block (slacks are linear, no Hessian contribution)
        (self.inner_hess_rows.clone(), self.inner_hess_cols.clone())
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        // Hessian only depends on x (slacks are linear)
        self.inner
            .hessian_values(&x[..self.n_orig], _new_x, obj_factor, lambda, vals);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple test problem: min x0^2 + x1^2, s.t. x0 + x1 >= 1, x0 >= 0, x1 >= 0
    struct SimpleInequality;

    impl NlpProblem for SimpleInequality {
        fn num_variables(&self) -> usize {
            2
        }
        fn num_constraints(&self) -> usize {
            1
        }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = 0.0;
            x_u[0] = f64::INFINITY;
            x_l[1] = 0.0;
            x_u[1] = f64::INFINITY;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 1.0;
            g_u[0] = f64::INFINITY;
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
    fn test_dimensions() {
        let prob = SimpleInequality;
        let x_r = vec![0.5, 0.5];
        let slack = SlackFormulation::new(&prob, &x_r);
        assert_eq!(slack.num_variables(), 3); // 2 orig + 1 slack
        assert_eq!(slack.num_constraints(), 1);
    }

    #[test]
    fn test_bounds() {
        let prob = SimpleInequality;
        let x_r = vec![0.5, 0.5];
        let slack = SlackFormulation::new(&prob, &x_r);
        let nv = slack.num_variables();
        let mut xl = vec![0.0; nv];
        let mut xu = vec![0.0; nv];
        slack.bounds(&mut xl, &mut xu);
        // Original bounds
        assert_eq!(xl[0], 0.0);
        assert_eq!(xu[0], f64::INFINITY);
        assert_eq!(xl[1], 0.0);
        assert_eq!(xu[1], f64::INFINITY);
        // Slack bounds = constraint bounds
        assert_eq!(xl[2], 1.0); // g_l
        assert_eq!(xu[2], f64::INFINITY); // g_u
    }

    #[test]
    fn test_constraint_bounds_are_equalities() {
        let prob = SimpleInequality;
        let x_r = vec![0.5, 0.5];
        let slack = SlackFormulation::new(&prob, &x_r);
        let m = slack.num_constraints();
        let mut gl = vec![0.0; m];
        let mut gu = vec![0.0; m];
        slack.constraint_bounds(&mut gl, &mut gu);
        assert_eq!(gl[0], 0.0);
        assert_eq!(gu[0], 0.0);
    }

    #[test]
    fn test_initial_point() {
        let prob = SimpleInequality;
        let x_r = vec![0.5, 0.5];
        let slack = SlackFormulation::new(&prob, &x_r);
        let nv = slack.num_variables();
        let mut x0 = vec![0.0; nv];
        slack.initial_point(&mut x0);
        assert_eq!(x0[0], 0.5);
        assert_eq!(x0[1], 0.5);
        // s = g(x) clamped to [g_l, g_u] = [1.0, inf] → g(0.5,0.5)=1.0 → s=1.0
        assert_eq!(x0[2], 1.0);
    }

    #[test]
    fn test_objective() {
        let prob = SimpleInequality;
        let x_r = vec![0.5, 0.5];
        let slack = SlackFormulation::new(&prob, &x_r);
        let x = vec![1.0, 2.0, 3.0]; // slack value doesn't affect objective
        assert!((slack.objective(&x, true) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient() {
        let prob = SimpleInequality;
        let x_r = vec![0.5, 0.5];
        let slack = SlackFormulation::new(&prob, &x_r);
        let nv = slack.num_variables();
        let x = vec![1.0, 2.0, 3.0];
        let mut grad = vec![0.0; nv];
        slack.gradient(&x, true, &mut grad);
        assert!((grad[0] - 2.0).abs() < 1e-10);
        assert!((grad[1] - 4.0).abs() < 1e-10);
        assert_eq!(grad[2], 0.0); // slack gradient = 0
    }

    #[test]
    fn test_constraints() {
        let prob = SimpleInequality;
        let x_r = vec![0.5, 0.5];
        let slack = SlackFormulation::new(&prob, &x_r);
        let m = slack.num_constraints();
        // g_slack = g(x) - s = (1+2) - 2.5 = 0.5
        let x = vec![1.0, 2.0, 2.5];
        let mut g = vec![0.0; m];
        slack.constraints(&x, true, &mut g);
        assert!((g[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_structure_and_values() {
        let prob = SimpleInequality;
        let x_r = vec![0.5, 0.5];
        let slack = SlackFormulation::new(&prob, &x_r);
        let (rows, cols) = slack.jacobian_structure();
        assert_eq!(rows.len(), 3); // 2 inner + 1 slack
        assert_eq!((rows[0], cols[0]), (0, 0)); // dg/dx0
        assert_eq!((rows[1], cols[1]), (0, 1)); // dg/dx1
        assert_eq!((rows[2], cols[2]), (0, 2)); // -1 for slack

        let mut vals = vec![0.0; 3];
        let x = vec![1.0, 2.0, 3.0];
        slack.jacobian_values(&x, true, &mut vals);
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
        assert!((vals[2] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_hessian() {
        let prob = SimpleInequality;
        let x_r = vec![0.5, 0.5];
        let slack = SlackFormulation::new(&prob, &x_r);
        let (hrows, hcols) = slack.hessian_structure();
        // Only x-x block, no slack entries
        assert_eq!(hrows.len(), 2);
        assert_eq!((hrows[0], hcols[0]), (0, 0));
        assert_eq!((hrows[1], hcols[1]), (1, 1));

        let mut vals = vec![0.0; 2];
        let x = vec![1.0, 2.0, 3.0];
        let lambda = vec![1.0];
        slack.hessian_values(&x, true, 1.0, &lambda, &mut vals);
        assert!((vals[0] - 2.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10);
    }
}
