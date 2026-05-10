//! Thierry-Biegler ℓ₁-exact penalty-barrier NLP wrapper.
//!
//! Implements the NLP reformulation from
//! Thierry, D. & Biegler, L.T. (2020).
//! *"The ℓ₁ Exact Penalty-Barrier Phase for Degenerate Nonlinear
//! Programming Problems in Ipopt"*, IFAC-PapersOnLine.
//!
//! For every equality row of the user NLP (`g_l[i] == g_u[i]`), a slack
//! pair `(p_i, n_i) ≥ 0` is added and the row is rewritten as
//! `c_i(x) − p_i + n_i = g_target`. The objective is augmented with
//! `ρ · Σ(p + n)`. Inequality rows pass through unchanged. The
//! reformulated NLP automatically satisfies LICQ on the augmented
//! variables, which is the property the method exploits to handle
//! degenerate / MPCC-like cases the stock filter line search thrashes
//! on.
//!
//! **Phase 1 (this module):** ρ is held fixed at construction time.
//! A future phase will replace this with the Byrd-Nocedal-Waltz dynamic
//! update rule.
//!
//! This is structurally similar to [`crate::restoration_nlp::RestorationNlp`],
//! with three differences:
//! 1. The original objective `f(x)` is preserved (restoration drops it).
//! 2. No proximity term `(η/2)·‖D_R(x − x_R)‖²` is added.
//! 3. Slacks are only added for equality rows, not all rows.

use crate::problem::NlpProblem;

/// NLP wrapper implementing the ℓ₁-exact penalty-barrier reformulation.
///
/// Variable layout: `[x(n_orig), p(m_eq), n(m_eq)]` — total
/// `n_orig + 2·m_eq` variables. Constraint layout is unchanged from
/// the inner problem (same `m` rows in the same order).
///
/// Uses dynamic dispatch so a single wrapper type can adapt any user
/// NLP without polluting the IPM's monomorphization graph.
pub struct L1PenaltyBarrierNlp<'a> {
    inner: &'a dyn NlpProblem,
    n_orig: usize,
    m: usize,
    /// Number of equality rows (rows where `g_l[i] == g_u[i]`).
    m_eq: usize,
    /// Indices into the inner constraint vector that are equality rows,
    /// in ascending order. `eq_rows.len() == m_eq`.
    eq_rows: Vec<usize>,
    rho: f64,
    /// Cached inner Jacobian sparsity (immutable across the solve).
    inner_jac_rows: Vec<usize>,
    inner_jac_cols: Vec<usize>,
    /// Cached inner Hessian sparsity (immutable across the solve).
    inner_hess_rows: Vec<usize>,
    inner_hess_cols: Vec<usize>,
    /// Initial p values, one per equality row.
    p_init: Vec<f64>,
    /// Initial n values, one per equality row.
    n_init: Vec<f64>,
}

impl<'a> L1PenaltyBarrierNlp<'a> {
    /// Wrap `inner` with the ℓ₁ penalty-barrier reformulation using the
    /// fixed penalty weight `rho`.
    pub fn new(inner: &'a dyn NlpProblem, rho: f64) -> Self {
        let n = inner.num_variables();
        let m = inner.num_constraints();

        // Identify equality rows.
        let mut g_l = vec![0.0; m];
        let mut g_u = vec![0.0; m];
        if m > 0 {
            inner.constraint_bounds(&mut g_l, &mut g_u);
        }
        let mut eq_rows = Vec::new();
        for i in 0..m {
            if g_l[i] == g_u[i] && g_l[i].is_finite() {
                eq_rows.push(i);
            }
        }
        let m_eq = eq_rows.len();

        let (inner_jac_rows, inner_jac_cols) = inner.jacobian_structure();
        let (inner_hess_rows, inner_hess_cols) = inner.hessian_structure();

        // Seed p, n from the equality-row violation at the user's
        // initial point: c_i(x0) − g_target. Split the violation into
        // its positive and negative parts, both bumped by a small
        // strictly-positive floor so the IPM has a non-empty interior
        // on the slack bounds (z = μ/s requires s > 0).
        let mut x0 = vec![0.0; n];
        inner.initial_point(&mut x0);
        let mut g0 = vec![0.0; m];
        if m > 0 {
            let _ = inner.constraints(&x0, true, &mut g0);
        }
        let floor = 1e-4;
        let mut p_init = vec![floor; m_eq];
        let mut n_init = vec![floor; m_eq];
        for (k, &row) in eq_rows.iter().enumerate() {
            let viol = g0[row] - g_l[row];
            if viol > 0.0 {
                p_init[k] = viol + floor;
            } else if viol < 0.0 {
                n_init[k] = -viol + floor;
            }
        }

        L1PenaltyBarrierNlp {
            inner,
            n_orig: n,
            m,
            m_eq,
            eq_rows,
            rho,
            inner_jac_rows,
            inner_jac_cols,
            inner_hess_rows,
            inner_hess_cols,
            p_init,
            n_init,
        }
    }

    /// Number of original (un-augmented) variables — used by the solver
    /// driver to truncate the result back to user space.
    pub fn n_orig(&self) -> usize {
        self.n_orig
    }

    /// Indices (into the inner constraint vector) of the equality rows.
    /// The solver driver uses this to extract just the equality-row
    /// multipliers from the inner result for the BNW steering update.
    pub fn eq_rows(&self) -> &[usize] {
        &self.eq_rows
    }
}

impl NlpProblem for L1PenaltyBarrierNlp<'_> {
    fn num_variables(&self) -> usize {
        self.n_orig + 2 * self.m_eq
    }

    fn num_constraints(&self) -> usize {
        self.m
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        let n = self.n_orig;
        self.inner.bounds(&mut x_l[..n], &mut x_u[..n]);
        for k in 0..2 * self.m_eq {
            x_l[n + k] = 0.0;
            x_u[n + k] = f64::INFINITY;
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        self.inner.constraint_bounds(g_l, g_u);
    }

    fn initial_point(&self, x0: &mut [f64]) {
        let n = self.n_orig;
        let m_eq = self.m_eq;
        self.inner.initial_point(&mut x0[..n]);
        x0[n..n + m_eq].copy_from_slice(&self.p_init);
        x0[n + m_eq..n + 2 * m_eq].copy_from_slice(&self.n_init);
    }

    fn objective(&self, x: &[f64], new_x: bool, obj: &mut f64) -> bool {
        let n = self.n_orig;
        if !self.inner.objective(&x[..n], new_x, obj) {
            return false;
        }
        for k in 0..2 * self.m_eq {
            *obj += self.rho * x[n + k];
        }
        true
    }

    fn gradient(&self, x: &[f64], new_x: bool, grad: &mut [f64]) -> bool {
        let n = self.n_orig;
        if !self.inner.gradient(&x[..n], new_x, &mut grad[..n]) {
            return false;
        }
        for k in 0..2 * self.m_eq {
            grad[n + k] = self.rho;
        }
        true
    }

    fn constraints(&self, x: &[f64], new_x: bool, g: &mut [f64]) -> bool {
        let n = self.n_orig;
        let m_eq = self.m_eq;
        if !self.inner.constraints(&x[..n], new_x, g) {
            return false;
        }
        // For each equality row i with slot k: g[i] += -p_k + n_k.
        for (k, &row) in self.eq_rows.iter().enumerate() {
            g[row] = g[row] - x[n + k] + x[n + m_eq + k];
        }
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.n_orig;
        let m_eq = self.m_eq;
        let inner_nnz = self.inner_jac_rows.len();
        let total_nnz = inner_nnz + 2 * m_eq;

        let mut rows = Vec::with_capacity(total_nnz);
        let mut cols = Vec::with_capacity(total_nnz);
        rows.extend_from_slice(&self.inner_jac_rows);
        cols.extend_from_slice(&self.inner_jac_cols);

        // -1 entries for p_k at column n + k.
        for (k, &row) in self.eq_rows.iter().enumerate() {
            rows.push(row);
            cols.push(n + k);
        }
        // +1 entries for n_k at column n + m_eq + k.
        for (k, &row) in self.eq_rows.iter().enumerate() {
            rows.push(row);
            cols.push(n + m_eq + k);
        }
        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], new_x: bool, vals: &mut [f64]) -> bool {
        let n = self.n_orig;
        let m_eq = self.m_eq;
        let inner_nnz = self.inner_jac_rows.len();
        if !self
            .inner
            .jacobian_values(&x[..n], new_x, &mut vals[..inner_nnz])
        {
            return false;
        }
        for k in 0..m_eq {
            vals[inner_nnz + k] = -1.0;
            vals[inner_nnz + m_eq + k] = 1.0;
        }
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // The augmented objective contribution is linear in (p, n) and
        // the augmented constraint contribution is linear in (p, n), so
        // the Hessian is exactly the inner Hessian (over the original
        // n_orig variables). The (p, n) block of the augmented Hessian
        // is zero — the IPM contributes the barrier curvature `μ/s²`
        // there automatically.
        (self.inner_hess_rows.clone(), self.inner_hess_cols.clone())
    }

    fn hessian_values(
        &self,
        x: &[f64],
        new_x: bool,
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) -> bool {
        let n = self.n_orig;
        // Inner constraints `c_i(x) − p + n` have the same x-derivatives
        // as `c_i(x)`, so the same `lambda` flows through unchanged.
        // The original objective `f(x)` is preserved, so `obj_factor`
        // also flows through unchanged (contrast with the restoration
        // NLP which forces obj_factor=0 because it drops f).
        self.inner
            .hessian_values(&x[..n], new_x, obj_factor, lambda, vals)
    }

    fn notify_mu(&self, mu: f64) {
        self.inner.notify_mu(mu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny test problem: min x0² + x1², s.t. x0 + x1 = 1.
    /// One equality, no inequalities.
    struct EqOnly;
    impl NlpProblem for EqOnly {
        fn num_variables(&self) -> usize { 2 }
        fn num_constraints(&self) -> usize { 1 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
            x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 1.0; g_u[0] = 1.0;
        }
        fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; x0[1] = 0.0; }
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = x[0]*x[0] + x[1]*x[1]; true
        }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = 2.0*x[0]; grad[1] = 2.0*x[1]; true
        }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g[0] = x[0] + x[1]; true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 0], vec![0, 1])
        }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals[0] = 1.0; vals[1] = 1.0; true
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1], vec![0, 1])
        }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
            vals[0] = 2.0 * obj_factor; vals[1] = 2.0 * obj_factor; true
        }
    }

    /// Mixed problem: 1 equality + 1 inequality, n=2.
    /// min x0² + x1², s.t. x0+x1 = 1, x0 - x1 ≤ 0.5.
    struct EqAndIneq;
    impl NlpProblem for EqAndIneq {
        fn num_variables(&self) -> usize { 2 }
        fn num_constraints(&self) -> usize { 2 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
            x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 1.0; g_u[0] = 1.0;          // eq
            g_l[1] = f64::NEG_INFINITY; g_u[1] = 0.5;  // ineq
        }
        fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; x0[1] = 0.0; }
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = x[0]*x[0] + x[1]*x[1]; true
        }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = 2.0*x[0]; grad[1] = 2.0*x[1]; true
        }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g[0] = x[0] + x[1];
            g[1] = x[0] - x[1];
            true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 0, 1, 1], vec![0, 1, 0, 1])
        }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals[0] = 1.0; vals[1] = 1.0;
            vals[2] = 1.0; vals[3] = -1.0;
            true
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1], vec![0, 1])
        }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
            vals[0] = 2.0 * obj_factor; vals[1] = 2.0 * obj_factor; true
        }
    }

    #[test]
    fn dimensions_eq_only() {
        let prob = EqOnly;
        let w = L1PenaltyBarrierNlp::new(&prob, 1000.0);
        assert_eq!(w.num_variables(), 4); // 2 orig + 2*1 slacks for the single eq row
        assert_eq!(w.num_constraints(), 1);
        assert_eq!(w.n_orig(), 2);
    }

    #[test]
    fn only_equality_rows_get_slacks() {
        let prob = EqAndIneq;
        let w = L1PenaltyBarrierNlp::new(&prob, 1000.0);
        // m_eq = 1 (one equality), so augmented n = 2 + 2.
        assert_eq!(w.num_variables(), 4);
        assert_eq!(w.num_constraints(), 2);

        // Jacobian must add slack columns only for the equality row (row 0).
        let (rows, cols) = w.jacobian_structure();
        // 4 inner entries + 2 slack entries = 6.
        assert_eq!(rows.len(), 6);
        // The two slack entries are both attached to row 0 (the eq row).
        assert_eq!(rows[4], 0);
        assert_eq!(rows[5], 0);
        assert_eq!(cols[4], 2); // p column at n_orig
        assert_eq!(cols[5], 3); // n column at n_orig + m_eq
    }

    #[test]
    fn bounds_slacks_are_nonneg() {
        let prob = EqOnly;
        let w = L1PenaltyBarrierNlp::new(&prob, 1000.0);
        let nv = w.num_variables();
        let mut xl = vec![0.0; nv];
        let mut xu = vec![0.0; nv];
        w.bounds(&mut xl, &mut xu);
        assert_eq!(xl[0], f64::NEG_INFINITY);
        assert_eq!(xu[0], f64::INFINITY);
        assert_eq!(xl[2], 0.0);
        assert_eq!(xu[2], f64::INFINITY);
        assert_eq!(xl[3], 0.0);
        assert_eq!(xu[3], f64::INFINITY);
    }

    #[test]
    fn objective_adds_rho_slack_sum() {
        let prob = EqOnly;
        let rho = 1000.0;
        let w = L1PenaltyBarrierNlp::new(&prob, rho);
        // x = (1, 1, 0.5, 0.25): inner f = 1+1 = 2; slack contribution = ρ·(0.5+0.25) = 750.
        let x = vec![1.0, 1.0, 0.5, 0.25];
        let mut obj = 0.0;
        assert!(w.objective(&x, true, &mut obj));
        assert!((obj - (2.0 + rho * 0.75)).abs() < 1e-10, "obj = {}", obj);
    }

    #[test]
    fn gradient_passes_inner_x_block_and_rho_slack_block() {
        let prob = EqOnly;
        let rho = 1000.0;
        let w = L1PenaltyBarrierNlp::new(&prob, rho);
        let x = vec![1.5, 2.0, 0.1, 0.2];
        let mut grad = vec![0.0; 4];
        assert!(w.gradient(&x, true, &mut grad));
        assert!((grad[0] - 3.0).abs() < 1e-12); // 2*1.5
        assert!((grad[1] - 4.0).abs() < 1e-12); // 2*2.0
        assert!((grad[2] - rho).abs() < 1e-12);
        assert!((grad[3] - rho).abs() < 1e-12);
    }

    #[test]
    fn constraints_rewrites_eq_rows_with_slack_diff() {
        let prob = EqAndIneq;
        let w = L1PenaltyBarrierNlp::new(&prob, 1000.0);
        // x_orig = (0.3, 0.5); p_0 = 0.2; n_0 = 0.1.
        // Row 0 (eq): (0.3+0.5) - 0.2 + 0.1 = 0.7
        // Row 1 (ineq): 0.3 - 0.5 = -0.2 (unchanged, no slack added)
        let x = vec![0.3, 0.5, 0.2, 0.1];
        let mut g = vec![0.0; 2];
        assert!(w.constraints(&x, true, &mut g));
        assert!((g[0] - 0.7).abs() < 1e-12, "g[0]={}", g[0]);
        assert!((g[1] - (-0.2)).abs() < 1e-12, "g[1]={}", g[1]);
    }

    #[test]
    fn jacobian_values_inner_then_minus_one_plus_one() {
        let prob = EqOnly;
        let w = L1PenaltyBarrierNlp::new(&prob, 1000.0);
        let mut vals = vec![0.0; 4];
        let x = vec![0.5, 0.5, 0.1, 0.1];
        assert!(w.jacobian_values(&x, true, &mut vals));
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 1.0).abs() < 1e-12);
        assert!((vals[2] - (-1.0)).abs() < 1e-12); // p column
        assert!((vals[3] - 1.0).abs() < 1e-12);    // n column
    }

    #[test]
    fn hessian_preserves_obj_factor_and_inner_structure() {
        let prob = EqOnly;
        let w = L1PenaltyBarrierNlp::new(&prob, 1000.0);
        let (hr, hc) = w.hessian_structure();
        // Inner has 2 diagonal entries; wrapper Hessian should match exactly
        // (no slack rows/cols since the augmented Hessian over (p,n) is 0).
        assert_eq!(hr, vec![0, 1]);
        assert_eq!(hc, vec![0, 1]);
        let x = vec![1.0, 2.0, 0.1, 0.1];
        let lam = vec![3.0];
        let mut vals = vec![0.0; 2];
        // obj_factor = 1.0 → diagonal = 2.0 (preserves f curvature).
        assert!(w.hessian_values(&x, true, 1.0, &lam, &mut vals));
        assert!((vals[0] - 2.0).abs() < 1e-12);
        assert!((vals[1] - 2.0).abs() < 1e-12);
        // obj_factor = 0 → diagonal must collapse (proves the original
        // objective curvature does flow through with the user's factor,
        // unlike the restoration NLP which forces 0).
        let mut vals2 = vec![0.0; 2];
        assert!(w.hessian_values(&x, true, 0.0, &lam, &mut vals2));
        assert!(vals2[0].abs() < 1e-12);
        assert!(vals2[1].abs() < 1e-12);
    }

    #[test]
    fn initial_slacks_are_strictly_positive() {
        let prob = EqOnly;
        let w = L1PenaltyBarrierNlp::new(&prob, 1000.0);
        let mut x0 = vec![0.0; 4];
        w.initial_point(&mut x0);
        assert_eq!(x0[0], 0.0);
        assert_eq!(x0[1], 0.0);
        assert!(x0[2] > 0.0, "p_init must be > 0 for IPM interior");
        assert!(x0[3] > 0.0, "n_init must be > 0 for IPM interior");
    }

    #[test]
    fn initial_slacks_split_violation() {
        // Build a problem whose initial point has c(x0) − g_target = +0.7,
        // so p_init ≈ 0.7 + floor and n_init ≈ floor.
        struct OffStart;
        impl NlpProblem for OffStart {
            fn num_variables(&self) -> usize { 1 }
            fn num_constraints(&self) -> usize { 1 }
            fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
                x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
            }
            fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
                g_l[0] = 0.0; g_u[0] = 0.0;
            }
            fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.7; }
            fn objective(&self, x: &[f64], _: bool, obj: &mut f64) -> bool { *obj = x[0]*x[0]; true }
            fn gradient(&self, x: &[f64], _: bool, grad: &mut [f64]) -> bool { grad[0] = 2.0*x[0]; true }
            fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool { g[0] = x[0]; true }
            fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
            fn jacobian_values(&self, _: &[f64], _: bool, vals: &mut [f64]) -> bool { vals[0] = 1.0; true }
            fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
            fn hessian_values(&self, _: &[f64], _: bool, of: f64, _: &[f64], vals: &mut [f64]) -> bool { vals[0] = 2.0*of; true }
        }
        let prob = OffStart;
        let w = L1PenaltyBarrierNlp::new(&prob, 1000.0);
        let mut x0 = vec![0.0; 3];
        w.initial_point(&mut x0);
        assert!((x0[0] - 0.7).abs() < 1e-12);
        // p_init carries the +0.7 violation; n_init stays at the floor.
        assert!(x0[1] > 0.7, "p_init should absorb the +0.7 surplus, got {}", x0[1]);
        assert!(x0[2] > 0.0 && x0[2] < 0.1, "n_init should remain at the floor, got {}", x0[2]);
    }

    #[test]
    fn no_equality_rows_makes_wrapper_a_passthrough_in_dimensions() {
        struct IneqOnly;
        impl NlpProblem for IneqOnly {
            fn num_variables(&self) -> usize { 1 }
            fn num_constraints(&self) -> usize { 1 }
            fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) { x_l[0]=f64::NEG_INFINITY; x_u[0]=f64::INFINITY; }
            fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) { g_l[0]=f64::NEG_INFINITY; g_u[0]=1.0; }
            fn initial_point(&self, x0: &mut [f64]) { x0[0]=0.0; }
            fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool { *o = x[0]*x[0]; true }
            fn gradient(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool { g[0]=2.0*x[0]; true }
            fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool { g[0]=x[0]; true }
            fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
            fn jacobian_values(&self, _: &[f64], _: bool, v: &mut [f64]) -> bool { v[0]=1.0; true }
            fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
            fn hessian_values(&self, _: &[f64], _: bool, of: f64, _: &[f64], v: &mut [f64]) -> bool { v[0]=2.0*of; true }
        }
        let prob = IneqOnly;
        let w = L1PenaltyBarrierNlp::new(&prob, 1000.0);
        // No equality rows ⇒ no slacks added.
        assert_eq!(w.num_variables(), 1);
        assert_eq!(w.num_constraints(), 1);
        let (jr, _) = w.jacobian_structure();
        assert_eq!(jr.len(), 1, "no slack columns when there are no equality rows");
    }
}
