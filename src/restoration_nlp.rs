use std::cell::Cell;

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
    /// η is computed dynamically per-evaluation as
    /// `eta_factor * sqrt(current_mu)` (Ipopt
    /// `RestoIpoptNLP::Eta`, IpRestoIpoptNLP.cpp:759). The current
    /// μ is updated by the inner IPM via `notify_mu` so that as μ
    /// drops during the restoration solve, the proximity weight η
    /// tracks it instead of remaining frozen at the entry value.
    eta_factor: f64,
    /// Current barrier μ feeding the η computation. Interior
    /// mutability so `notify_mu(&self, ...)` can refresh it.
    current_mu: Cell<f64>,
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
        // η is recomputed each evaluation from current_mu (see
        // `eta()` below). At construction we seed current_mu with
        // mu_entry; the inner IPM refreshes it via `notify_mu`.

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
            let _ = inner.constraints(x_r, true, &mut g_r);
        }
        let mut g_l = vec![0.0; m];
        let mut g_u = vec![0.0; m];
        if m > 0 {
            inner.constraint_bounds(&mut g_l, &mut g_u);
        }

        // Ipopt's closed-form p/n init (IpRestoIterateInitializer.cpp:79-97):
        // solve simultaneously p_i*n_i=mu and p_i - n_i = c_i, giving
        //   a = mu/(2*rho) - c_i/2,  b = mu*c_i/(2*rho)
        //   n_i = a + sqrt(a^2 + b),  p_i = c_i + n_i
        // This guarantees p_i, n_i > 0 and the bound multipliers z_p=mu/p,
        // z_n=mu/n satisfy complementarity from iteration zero. Without this,
        // the inner IPM spends 50+ iters chasing a huge dual residual caused
        // by grad_f[p]=rho ≈ 1000 vs z_p ≈ 1 after generic bound_push.
        let mut p_init = vec![0.0; m];
        let mut n_init = vec![0.0; m];
        let safe_mu = mu_entry.max(1e-8);
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
            let c_i = g_r[i] - target;
            let a = safe_mu / (2.0 * rho) - c_i / 2.0;
            let b = safe_mu * c_i / (2.0 * rho);
            let disc = (a * a + b).max(0.0);
            let n_i = a + disc.sqrt();
            let p_i = c_i + n_i;
            // Numerical guard: keep strictly positive
            n_init[i] = n_i.max(safe_mu / rho);
            p_init[i] = p_i.max(safe_mu / rho);
        }

        RestorationNlp {
            inner,
            n_orig: n,
            m_orig: m,
            x_r: x_r.to_vec(),
            d_r2,
            rho,
            eta_factor: eta_f,
            current_mu: Cell::new(mu_entry.max(0.0)),
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

    /// Current η = η_factor · √μ, with μ tracked by `notify_mu`.
    /// Mirrors Ipopt `RestoIpoptNLP::Eta(mu)` (IpRestoIpoptNLP.cpp:759).
    fn eta(&self) -> f64 {
        self.eta_factor * self.current_mu.get().max(0.0).sqrt()
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

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        let n = self.n_orig;
        let m = self.m_orig;

        // rho * (sum(p) + sum(n))
        *obj = 0.0;
        for i in 0..2 * m {
            *obj += self.rho * x[n + i];
        }

        // (eta/2) * ||D_R(x - x_r)||^2
        let eta = self.eta();
        for i in 0..n {
            let diff = x[i] - self.x_r[i];
            *obj += 0.5 * eta * self.d_r2[i] * diff * diff;
        }

        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        let n = self.n_orig;
        let m = self.m_orig;

        // x part: eta * D_R^2 * (x - x_r)
        let eta = self.eta();
        for i in 0..n {
            grad[i] = eta * self.d_r2[i] * (x[i] - self.x_r[i]);
        }

        // p and n parts: rho
        for i in 0..2 * m {
            grad[n + i] = self.rho;
        }
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        let n = self.n_orig;
        let m = self.m_orig;

        // g_resto[i] = g_orig[i] - p[i] + n[i]
        if !self.inner.constraints(&x[..n], _new_x, g) {
            return false;
        }
        for i in 0..m {
            g[i] = g[i] - x[n + i] + x[n + m + i];
        }
        true
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

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        let n = self.n_orig;
        let m = self.m_orig;
        let inner_nnz = self.inner_jac_rows.len();

        // Inner Jacobian values
        if !self.inner.jacobian_values(&x[..n], _new_x, &mut vals[..inner_nnz]) {
            return false;
        }

        // -1 for p
        for i in 0..m {
            vals[inner_nnz + i] = -1.0;
        }

        // +1 for n
        for i in 0..m {
            vals[inner_nnz + m + i] = 1.0;
        }
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.resto_hess_rows.clone(), self.resto_hess_cols.clone())
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        let n = self.n_orig;

        // Inner hessian: obj_factor=0 (restoration doesn't optimize original objective),
        // lambda passed through (constraint curvature).
        let mut inner_vals = vec![0.0; self.inner_hess_nnz];
        if !self.inner
            .hessian_values(&x[..n], _new_x, 0.0, lambda, &mut inner_vals) {
            return false;
        }

        // Copy inner values to output
        vals[..self.inner_hess_nnz].copy_from_slice(&inner_vals);

        // Zero out any appended entries (new diagonal slots)
        for v in vals[self.inner_hess_nnz..].iter_mut() {
            *v = 0.0;
        }

        // Add obj_factor * eta * d_r2[i] to each diagonal
        let eta = self.eta();
        for i in 0..n {
            let idx = self.diag_indices[i];
            vals[idx] += obj_factor * eta * self.d_r2[i];
        }

        // p/n blocks have zero Hessian (barrier mu/s^2 is added by IPM automatically)
        true
    }

    /// Refresh `current_mu` so subsequent `objective` / `gradient` /
    /// `hessian_values` calls compute η = η_factor·√μ at the new μ
    /// instead of the entry-time value (Ipopt
    /// `RestoIpoptNLP::Eta(mu)` reads μ dynamically per evaluation).
    fn notify_mu(&self, mu: f64) {
        if mu.is_finite() && mu >= 0.0 {
            self.current_mu.set(mu);
        }
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
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool { *obj = x[0] * x[0] + x[1] * x[1]; true }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = 2.0 * x[0];
            grad[1] = 2.0 * x[1];
            true
        }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g[0] = x[0] + x[1];
            true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 0], vec![0, 1])
        }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals[0] = 1.0;
            vals[1] = 1.0;
            true
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1], vec![0, 1])
        }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
            vals[0] = 2.0 * obj_factor;
            vals[1] = 2.0 * obj_factor;
            true
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
        let mut obj = 0.0;
        resto.objective(&x, true, &mut obj);
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

    /// T0.10: η must NOT be frozen at restoration entry. After
    /// `notify_mu` the proximity term in `objective` and `gradient`
    /// must use the new μ (Ipopt RestoIpoptNLP::Eta reads μ
    /// dynamically per evaluation; IpRestoIpoptNLP.cpp:759).
    #[test]
    fn test_eta_tracks_notify_mu() {
        let prob = SimpleConstrained;
        // x_r = (1,2) so D_R^2 = (1, 1/4).
        let x_r = vec![1.0, 2.0];
        // mu_entry = 0.1, eta_factor = 1.0 → η_entry = sqrt(0.1).
        let resto = RestorationNlp::new(&prob, &x_r, 0.1, 1000.0, 1.0);

        // Evaluate objective at x = (2, 4, 0, 0): only proximity term contributes.
        // Proximity = 0.5 * η * (D_R^2[0] * (2-1)^2 + D_R^2[1] * (4-2)^2)
        //           = 0.5 * η * (1*1 + 0.25*4) = η.
        let x = vec![2.0, 4.0, 0.0, 0.0];
        let mut obj_entry = 0.0;
        resto.objective(&x, true, &mut obj_entry);
        let eta_entry = 0.1f64.sqrt();
        assert!(
            (obj_entry - eta_entry).abs() < 1e-12,
            "obj_entry = {}, expected {}",
            obj_entry, eta_entry
        );

        // Drop μ → η must shrink.
        resto.notify_mu(0.01);
        let mut obj_after = 0.0;
        resto.objective(&x, true, &mut obj_after);
        let eta_after = 0.01f64.sqrt();
        assert!(
            (obj_after - eta_after).abs() < 1e-12,
            "obj_after = {}, expected {} (η must track current μ, not entry μ)",
            obj_after, eta_after
        );
        assert!(
            obj_after < obj_entry,
            "lower μ must give smaller proximity term (η ∝ √μ)"
        );

        // Gradient at x = (1+ε, 2, 0, 0): grad[0] = η * D_R^2[0] * (x[0]-x_r[0]) = η.
        let x2 = vec![2.0, 2.0, 0.0, 0.0];
        let mut grad = vec![0.0; resto.num_variables()];
        resto.gradient(&x2, true, &mut grad);
        assert!(
            (grad[0] - eta_after).abs() < 1e-12,
            "grad[0] = {}, expected {} after notify_mu(0.01)",
            grad[0], eta_after
        );
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
