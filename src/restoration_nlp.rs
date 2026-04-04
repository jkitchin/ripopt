use crate::problem::NlpProblem;

/// NLP problem wrapper for the restoration phase.
///
/// Decomposes constraint violations into positive/negative slack variables:
///
///   min   ρ·(Σp + Σn) + (η/2)·‖D_R(x − x_r)‖²
///   s.t.  g(x) − p + n = g_target   (same bounds as original constraints)
///         x_L ≤ x ≤ x_U             (original variable bounds)
///         p, n ≥ 0                   (slack non-negativity)
///
/// Variable layout: `[x(n), p(m), n(m)]` — total n + 2m variables, m constraints.
///
/// When p=n=0, all original constraints are satisfied → feasible.
///
/// Uses dynamic dispatch (`&dyn NlpProblem`) to break infinite monomorphization
/// recursion when called from inside the generic `solve_ipm<P>`.
pub struct RestorationNlp<'a> {
    inner: &'a dyn NlpProblem,
    n_orig: usize,
    m_orig: usize,
    x_r: Vec<f64>,
    d_r2: Vec<f64>,
    rho: f64,
    eta: f64,
    inner_jac_rows: Vec<usize>,
    inner_jac_cols: Vec<usize>,
    resto_hess_rows: Vec<usize>,
    resto_hess_cols: Vec<usize>,
    inner_hess_nnz: usize,
    /// For each original variable i, the index into resto_hess where D_R^2 should be added.
    diag_indices: Vec<usize>,
    /// Cached initial p values.
    p_init: Vec<f64>,
    /// Cached initial n values.
    n_init: Vec<f64>,
}

impl<'a> RestorationNlp<'a> {
    /// Create a new restoration NLP wrapper.
    ///
    /// - `inner`: the original NLP problem (via dynamic dispatch)
    /// - `x_r`: reference point (current x at restoration entry)
    /// - `mu_entry`: current barrier parameter at restoration entry
    /// - `rho`: penalty weight for slacks (default 1000.0)
    /// - `eta_f`: proximity factor (default 1.0); η = eta_f * sqrt(mu_entry)
    pub fn new(
        inner: &'a dyn NlpProblem,
        x_r: &[f64],
        mu_entry: f64,
        rho: f64,
        eta_f: f64,
    ) -> Self {
        let n = inner.num_variables();
        let m = inner.num_constraints();
        let eta = eta_f * mu_entry.sqrt();

        // D_R[i] = 1/max(1, |x_r[i]|), d_r2[i] = D_R[i]^2
        let d_r2: Vec<f64> = x_r
            .iter()
            .map(|&xi| {
                let d = 1.0 / xi.abs().max(1.0);
                d * d
            })
            .collect();

        let (inner_jac_rows, inner_jac_cols) = inner.jacobian_structure();
        let (inner_hess_rows, inner_hess_cols) = inner.hessian_structure();
        let inner_hess_nnz = inner_hess_rows.len();

        // Build restoration hessian structure:
        // Start with inner hessian entries (x block only).
        // Then add any missing diagonal entries for the D_R^2 proximity term.
        let mut resto_hess_rows = inner_hess_rows.clone();
        let mut resto_hess_cols = inner_hess_cols.clone();

        // For each variable, find if diagonal already exists in inner hessian
        let mut existing_diag = vec![None::<usize>; n];
        for (idx, (&r, &c)) in inner_hess_rows.iter().zip(inner_hess_cols.iter()).enumerate() {
            if r == c && r < n {
                existing_diag[r] = Some(idx);
            }
        }

        // For missing diagonals, append new entries. Record index for all diagonals.
        let mut diag_indices = vec![0usize; n];
        for i in 0..n {
            if let Some(idx) = existing_diag[i] {
                diag_indices[i] = idx;
            } else {
                diag_indices[i] = resto_hess_rows.len();
                resto_hess_rows.push(i);
                resto_hess_cols.push(i);
            }
        }

        // Compute initial p, n from constraint violations at x_r
        let mut g_r = vec![0.0; m];
        if m > 0 {
            inner.constraints(x_r, true, &mut g_r);
        }
        let mut g_l = vec![0.0; m];
        let mut g_u = vec![0.0; m];
        if m > 0 {
            inner.constraint_bounds(&mut g_l, &mut g_u);
        }

        // For each constraint: violation = g(x_r) - target
        // p_i = max(violation_i, 0) + mu_init_shift
        // n_i = max(-violation_i, 0) + mu_init_shift
        let mu_init_shift = mu_entry.max(1e-4);
        let mut p_init = vec![0.0; m];
        let mut n_init = vec![0.0; m];
        for i in 0..m {
            let target = if g_l[i].is_finite() && g_u[i].is_finite() {
                g_r[i].clamp(g_l[i], g_u[i])
            } else if g_l[i].is_finite() {
                g_l[i]
            } else if g_u[i].is_finite() {
                g_u[i]
            } else {
                g_r[i]
            };
            let viol = g_r[i] - target;
            p_init[i] = viol.max(0.0) + mu_init_shift;
            n_init[i] = (-viol).max(0.0) + mu_init_shift;
        }

        RestorationNlp {
            inner,
            n_orig: n,
            m_orig: m,
            x_r: x_r.to_vec(),
            d_r2,
            rho,
            eta,
            inner_jac_rows,
            inner_jac_cols,
            resto_hess_rows,
            resto_hess_cols,
            inner_hess_nnz,
            diag_indices,
            p_init,
            n_init,
        }
    }
}

impl NlpProblem for RestorationNlp<'_> {
    fn num_variables(&self) -> usize {
        self.n_orig + 2 * self.m_orig
    }

    fn num_constraints(&self) -> usize {
        self.m_orig
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        let n = self.n_orig;
        let m = self.m_orig;

        // First n: original variable bounds (fill output slices directly)
        self.inner.bounds(&mut x_l[..n], &mut x_u[..n]);

        // Next 2m: p and n slacks, lower=0, upper=+inf
        for i in 0..2 * m {
            x_l[n + i] = 0.0;
            x_u[n + i] = f64::INFINITY;
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        self.inner.constraint_bounds(g_l, g_u);
    }

    fn initial_point(&self, x0: &mut [f64]) {
        let n = self.n_orig;
        let m = self.m_orig;

        x0[..n].copy_from_slice(&self.x_r);
        x0[n..n + m].copy_from_slice(&self.p_init);
        x0[n + m..n + 2 * m].copy_from_slice(&self.n_init);
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let n = self.n_orig;
        let m = self.m_orig;

        // rho * (sum(p) + sum(n))
        let mut obj = 0.0;
        for i in 0..2 * m {
            obj += self.rho * x[n + i];
        }

        // (eta/2) * ||D_R(x - x_r)||^2
        for i in 0..n {
            let diff = x[i] - self.x_r[i];
            obj += 0.5 * self.eta * self.d_r2[i] * diff * diff;
        }

        obj
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let n = self.n_orig;
        let m = self.m_orig;

        // x part: eta * D_R^2 * (x - x_r)
        for i in 0..n {
            grad[i] = self.eta * self.d_r2[i] * (x[i] - self.x_r[i]);
        }

        // p and n parts: rho
        for i in 0..2 * m {
            grad[n + i] = self.rho;
        }
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let n = self.n_orig;
        let m = self.m_orig;

        // g_resto[i] = g_orig[i] - p[i] + n[i]
        self.inner.constraints(&x[..n], _new_x, g);
        for i in 0..m {
            g[i] = g[i] - x[n + i] + x[n + m + i];
        }
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.n_orig;
        let m = self.m_orig;

        let inner_nnz = self.inner_jac_rows.len();
        let total_nnz = inner_nnz + 2 * m;

        let mut rows = Vec::with_capacity(total_nnz);
        let mut cols = Vec::with_capacity(total_nnz);

        // Inner Jacobian entries (same rows/cols)
        rows.extend_from_slice(&self.inner_jac_rows);
        cols.extend_from_slice(&self.inner_jac_cols);

        // -1 entries for p: (i, n+i)
        for i in 0..m {
            rows.push(i);
            cols.push(n + i);
        }

        // +1 entries for n: (i, n+m+i)
        for i in 0..m {
            rows.push(i);
            cols.push(n + m + i);
        }

        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        let n = self.n_orig;
        let m = self.m_orig;
        let inner_nnz = self.inner_jac_rows.len();

        // Inner Jacobian values
        self.inner.jacobian_values(&x[..n], _new_x, &mut vals[..inner_nnz]);

        // -1 for p
        for i in 0..m {
            vals[inner_nnz + i] = -1.0;
        }

        // +1 for n
        for i in 0..m {
            vals[inner_nnz + m + i] = 1.0;
        }
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.resto_hess_rows.clone(), self.resto_hess_cols.clone())
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        let n = self.n_orig;

        // Inner hessian: obj_factor=0 (restoration doesn't optimize original objective),
        // lambda passed through (constraint curvature).
        let mut inner_vals = vec![0.0; self.inner_hess_nnz];
        self.inner
            .hessian_values(&x[..n], _new_x, 0.0, lambda, &mut inner_vals);

        // Copy inner values to output
        vals[..self.inner_hess_nnz].copy_from_slice(&inner_vals);

        // Zero out any appended entries (new diagonal slots)
        for v in vals[self.inner_hess_nnz..].iter_mut() {
            *v = 0.0;
        }

        // Add obj_factor * eta * d_r2[i] to each diagonal
        for i in 0..n {
            let idx = self.diag_indices[i];
            vals[idx] += obj_factor * self.eta * self.d_r2[i];
        }

        // p/n blocks have zero Hessian (barrier mu/s^2 is added by IPM automatically)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple test problem: min x0^2 + x1^2, s.t. x0 + x1 = 1
    struct SimpleConstrained;

    impl NlpProblem for SimpleConstrained {
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
            g_l[0] = 1.0;
            g_u[0] = 1.0;
        }
        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 0.0;
            x0[1] = 0.0;
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
        let prob = SimpleConstrained;
        let x_r = vec![0.0, 0.0];
        let resto = RestorationNlp::new(&prob, &x_r, 0.1, 1000.0, 1.0);
        assert_eq!(resto.num_variables(), 4); // 2 orig + 2*1 slacks
        assert_eq!(resto.num_constraints(), 1);
    }

    #[test]
    fn test_bounds() {
        let prob = SimpleConstrained;
        let x_r = vec![0.0, 0.0];
        let resto = RestorationNlp::new(&prob, &x_r, 0.1, 1000.0, 1.0);
        let nv = resto.num_variables();
        let mut xl = vec![0.0; nv];
        let mut xu = vec![0.0; nv];
        resto.bounds(&mut xl, &mut xu);
        assert_eq!(xl[0], f64::NEG_INFINITY);
        assert_eq!(xu[0], f64::INFINITY);
        assert_eq!(xl[2], 0.0);
        assert_eq!(xu[2], f64::INFINITY);
        assert_eq!(xl[3], 0.0);
        assert_eq!(xu[3], f64::INFINITY);
    }

    #[test]
    fn test_initial_point() {
        let prob = SimpleConstrained;
        let x_r = vec![0.0, 0.0];
        let resto = RestorationNlp::new(&prob, &x_r, 0.1, 1000.0, 1.0);
        let nv = resto.num_variables();
        let mut x0 = vec![0.0; nv];
        resto.initial_point(&mut x0);
        assert_eq!(x0[0], 0.0);
        assert_eq!(x0[1], 0.0);
        assert!(x0[2] > 0.0);
        assert!(x0[3] > 0.0);
    }

    #[test]
    fn test_objective_at_reference() {
        let prob = SimpleConstrained;
        let x_r = vec![1.0, 2.0];
        let resto = RestorationNlp::new(&prob, &x_r, 0.1, 1000.0, 1.0);
        // At x=x_r, p=1, n=2: obj = rho*(1+2) + 0
        let x = vec![1.0, 2.0, 1.0, 2.0];
        let obj = resto.objective(&x, true);
        assert!((obj - 3000.0).abs() < 1e-10, "obj = {}", obj);
    }

    #[test]
    fn test_gradient() {
        let prob = SimpleConstrained;
        let x_r = vec![1.0, 2.0];
        let resto = RestorationNlp::new(&prob, &x_r, 0.1, 1000.0, 1.0);
        let nv = resto.num_variables();
        let x = vec![1.0, 2.0, 0.5, 0.5];
        let mut grad = vec![0.0; nv];
        resto.gradient(&x, true, &mut grad);
        assert!(grad[0].abs() < 1e-10);
        assert!(grad[1].abs() < 1e-10);
        assert!((grad[2] - 1000.0).abs() < 1e-10);
        assert!((grad[3] - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_constraints_with_slacks() {
        let prob = SimpleConstrained;
        let x_r = vec![0.0, 0.0];
        let resto = RestorationNlp::new(&prob, &x_r, 0.1, 1000.0, 1.0);
        let m = resto.num_constraints();
        // g_resto = (0.3 + 0.5) - 0.2 + 0.1 = 0.7
        let x = vec![0.3, 0.5, 0.2, 0.1];
        let mut g = vec![0.0; m];
        resto.constraints(&x, true, &mut g);
        assert!((g[0] - 0.7).abs() < 1e-10, "g = {}", g[0]);
    }

    #[test]
    fn test_jacobian_structure_and_values() {
        let prob = SimpleConstrained;
        let x_r = vec![0.0, 0.0];
        let resto = RestorationNlp::new(&prob, &x_r, 0.1, 1000.0, 1.0);
        let (rows, cols) = resto.jacobian_structure();
        assert_eq!(rows.len(), 4);
        assert_eq!((rows[0], cols[0]), (0, 0));
        assert_eq!((rows[1], cols[1]), (0, 1));
        assert_eq!((rows[2], cols[2]), (0, 2));
        assert_eq!((rows[3], cols[3]), (0, 3));

        let mut vals = vec![0.0; 4];
        let x = vec![0.5, 0.5, 0.1, 0.1];
        resto.jacobian_values(&x, true, &mut vals);
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
        assert!((vals[2] - (-1.0)).abs() < 1e-10);
        assert!((vals[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hessian_values() {
        let prob = SimpleConstrained;
        let x_r = vec![1.0, 2.0];
        let resto = RestorationNlp::new(&prob, &x_r, 0.1, 1000.0, 1.0);
        let (hrows, _hcols) = resto.hessian_structure();
        let nnz = hrows.len();

        let mut vals = vec![0.0; nnz];
        let x = vec![1.0, 2.0, 0.1, 0.1];
        let lambda = vec![1.0];
        resto.hessian_values(&x, true, 1.0, &lambda, &mut vals);
        let eta = 0.1f64.sqrt();
        assert!((vals[0] - eta * 1.0).abs() < 1e-10, "vals[0] = {}", vals[0]);
        assert!(
            (vals[1] - eta * 0.25).abs() < 1e-10,
            "vals[1] = {}",
            vals[1]
        );
    }
}
