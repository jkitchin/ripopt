//! Internal TNLPAdapter-equivalent boundary.
//!
//! Mirrors Ipopt 3.14's `TNLPAdapter` (`IpTNLPAdapter.cpp`): the
//! user-facing `NlpProblem` trait keeps its combined m-form contract
//! (`g(x)`, `jac_g`, `lambda`) — that is the user's TNLP equivalent
//! — but every internal caller past this boundary sees only split-form
//! `c(x)` / `d(x)` and `Jac_c` / `Jac_d`. Combined m-vectors live only
//! inside per-call scratch buffers owned by [`SplitNlp`].
//!
//! Phase 10 status (post-10b.5): the IPM main driver in `ipm.rs`
//! routes every post-`SolverState::new` `NlpProblem` callback through
//! this adapter. The `NlpProblem` trait surface itself is unchanged
//! (Phase 10c) — that is the user's TNLP equivalent and the only
//! remaining user-facing combined-form surface, intentionally
//! mirroring Ipopt's TNLP user contract.
//!
//! The remaining direct `problem.X` calls in `ipm.rs` are *upstream*
//! of the IPM boundary (`SolverState::new`'s structural queries, the
//! scaling preflight, LS multiplier-init invoked from the constructor
//! itself). Diagnostic / preprocessing modules (`sensitivity.rs`,
//! `linearity.rs`) and the alternative L-BFGS path (`lbfgs.rs`) sit
//! outside the IPM main driver and have their own boundaries.

use std::cell::RefCell;

use crate::constraint_layout::ConstraintLayout;
use crate::problem::NlpProblem;

/// Internal split-form NLP adapter. Holds a borrow of the user-supplied
/// `NlpProblem` and the per-solve `ConstraintLayout`, plus reusable
/// scratch buffers for the combined-form callbacks.
#[allow(dead_code)] // Some methods exposed for completeness; not all are wired yet.
pub(crate) struct SplitNlp<'a, P: NlpProblem> {
    problem: &'a P,
    layout: &'a ConstraintLayout,
    /// Reused `g(x)` scratch (size `m`).
    g_scratch: RefCell<Vec<f64>>,
    /// Reused `jac_g` scratch (size `nnz_jac`).
    jac_scratch: RefCell<Vec<f64>>,
    /// Reused combined-form `lambda` scratch (size `m`) used to compose
    /// the Hessian multipliers from split `y_c` / `y_d`.
    lambda_scratch: RefCell<Vec<f64>>,
}

#[allow(dead_code)] // Some methods exposed for completeness; not all are wired yet.
impl<'a, P: NlpProblem> SplitNlp<'a, P> {
    /// Build an adapter from a borrow of the problem and the per-solve
    /// constraint layout. Pre-sizes the `g` and `lambda` scratch
    /// buffers; the Jacobian scratch grows lazily on first call.
    pub(crate) fn new(problem: &'a P, layout: &'a ConstraintLayout) -> Self {
        let m = problem.num_constraints();
        Self {
            problem,
            layout,
            g_scratch: RefCell::new(vec![0.0; m]),
            jac_scratch: RefCell::new(Vec::new()),
            lambda_scratch: RefCell::new(vec![0.0; m]),
        }
    }

    /// Borrow back the wrapped problem (for trait methods we don't
    /// surface here, e.g. `bounds`, `initial_point`).
    pub(crate) fn problem(&self) -> &'a P {
        self.problem
    }

    /// Borrow back the constraint layout.
    pub(crate) fn layout(&self) -> &'a ConstraintLayout {
        self.layout
    }

    /// Pass-through to `NlpProblem::objective`.
    pub(crate) fn objective(&self, x: &[f64], new_x: bool, obj: &mut f64) -> bool {
        self.problem.objective(x, new_x, obj)
    }

    /// Pass-through to `NlpProblem::gradient`.
    pub(crate) fn gradient(&self, x: &[f64], new_x: bool, grad: &mut [f64]) -> bool {
        self.problem.gradient(x, new_x, grad)
    }

    /// Evaluate the user constraints into split-form. `c_raw` receives
    /// the raw equality-row values `g[c_to_combined[k]]` (the caller
    /// subtracts `c_rhs` if it wants the residual `c(x) - c_rhs`);
    /// `d_out` receives the inequality-row values `g[d_to_combined[k]]`.
    /// The combined m-form lives only in the internal `g_scratch`.
    pub(crate) fn constraints_split(
        &self,
        x: &[f64],
        new_x: bool,
        c_raw: &mut [f64],
        d_out: &mut [f64],
    ) -> bool {
        debug_assert_eq!(c_raw.len(), self.layout.n_c);
        debug_assert_eq!(d_out.len(), self.layout.n_d);
        let mut g = self.g_scratch.borrow_mut();
        if !self.problem.constraints(x, new_x, &mut g) {
            return false;
        }
        for (k, &i) in self.layout.c_to_combined.iter().enumerate() {
            c_raw[k] = g[i];
        }
        for (k, &i) in self.layout.d_to_combined.iter().enumerate() {
            d_out[k] = g[i];
        }
        true
    }

    /// Evaluate the user constraints into a caller-owned combined
    /// m-buffer. For sites that need the m-form `g(x)` directly (e.g.
    /// l1-violation `θ` reductions, line-search residual norms).
    pub(crate) fn constraints_combined(
        &self,
        x: &[f64],
        new_x: bool,
        g_out: &mut [f64],
    ) -> bool {
        self.problem.constraints(x, new_x, g_out)
    }

    /// Pass-through to `NlpProblem::jacobian_structure`.
    pub(crate) fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        self.problem.jacobian_structure()
    }

    /// Evaluate the user Jacobian and route the values into the split
    /// `Jac_c` / `Jac_d` triplet vectors via the per-state combined-
    /// index maps. The combined m-form Jacobian lives only in
    /// `jac_scratch`.
    pub(crate) fn jacobian_split(
        &self,
        x: &[f64],
        new_x: bool,
        jac_c_combined_idx: &[usize],
        jac_d_combined_idx: &[usize],
        jac_c_vals: &mut [f64],
        jac_d_vals: &mut [f64],
    ) -> bool {
        let nnz = jac_c_combined_idx.len() + jac_d_combined_idx.len();
        let mut jac = self.jac_scratch.borrow_mut();
        if jac.len() != nnz {
            jac.resize(nnz, 0.0);
        }
        if !self.problem.jacobian_values(x, new_x, &mut jac) {
            return false;
        }
        for (k, &idx) in jac_c_combined_idx.iter().enumerate() {
            jac_c_vals[k] = jac[idx];
        }
        for (k, &idx) in jac_d_combined_idx.iter().enumerate() {
            jac_d_vals[k] = jac[idx];
        }
        true
    }

    /// Evaluate the user Jacobian into a caller-owned combined-nnz
    /// buffer. For sites that consume the full m-form Jacobian as
    /// `Jac_g` (e.g. residual matvecs that haven't been split-rewritten
    /// yet).
    pub(crate) fn jacobian_combined(
        &self,
        x: &[f64],
        new_x: bool,
        vals: &mut [f64],
    ) -> bool {
        self.problem.jacobian_values(x, new_x, vals)
    }

    /// Pass-through to `NlpProblem::hessian_structure`.
    pub(crate) fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        self.problem.hessian_structure()
    }

    /// Evaluate the user Lagrangian Hessian, composing the combined
    /// `lambda` from split `y_c` / `y_d` via the layout's row maps.
    /// The user callback still receives the m-form `lambda` because
    /// the user wrote it against the combined row order.
    pub(crate) fn hessian_from_split(
        &self,
        x: &[f64],
        new_x: bool,
        obj_factor: f64,
        y_c: &[f64],
        y_d: &[f64],
        vals: &mut [f64],
    ) -> bool {
        debug_assert_eq!(y_c.len(), self.layout.n_c);
        debug_assert_eq!(y_d.len(), self.layout.n_d);
        let mut lambda = self.lambda_scratch.borrow_mut();
        for (k, &i) in self.layout.c_to_combined.iter().enumerate() {
            lambda[i] = y_c[k];
        }
        for (k, &i) in self.layout.d_to_combined.iter().enumerate() {
            lambda[i] = y_d[k];
        }
        self.problem
            .hessian_values(x, new_x, obj_factor, &lambda, vals)
    }

    /// Evaluate the user Lagrangian Hessian with a pre-composed
    /// combined `lambda`. For sites (e.g. linear-constraint masking)
    /// that have already constructed the m-form multipliers.
    pub(crate) fn hessian_combined(
        &self,
        x: &[f64],
        new_x: bool,
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) -> bool {
        self.problem
            .hessian_values(x, new_x, obj_factor, lambda, vals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::NlpProblem;

    /// 2-var, 3-constraint fixture: row 0 inequality (g_l=0,g_u=∞),
    /// row 1 equality (g_l=g_u=2), row 2 inequality (g_l=-∞,g_u=5).
    /// `g(x) = [x0+x1, x0+x1, x0]`, Jac dense triplet:
    /// (0,0)=1 (0,1)=1 (1,0)=1 (1,1)=1 (2,0)=1.
    /// Hessian: only (0,0)=lambda[0]+lambda[1] (lower triangle).
    struct Fix;
    impl NlpProblem for Fix {
        fn num_variables(&self) -> usize {
            2
        }
        fn num_constraints(&self) -> usize {
            3
        }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l.fill(f64::NEG_INFINITY);
            x_u.fill(f64::INFINITY);
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 0.0;
            g_u[0] = f64::INFINITY;
            g_l[1] = 2.0;
            g_u[1] = 2.0;
            g_l[2] = f64::NEG_INFINITY;
            g_u[2] = 5.0;
        }
        fn initial_point(&self, x0: &mut [f64]) {
            x0.fill(1.0);
        }
        fn objective(&self, _x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = 0.0;
            true
        }
        fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad.fill(0.0);
            true
        }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g[0] = x[0] + x[1];
            g[1] = x[0] + x[1];
            g[2] = x[0];
            true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 0, 1, 1, 2], vec![0, 1, 0, 1, 0])
        }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals.copy_from_slice(&[1.0, 1.0, 1.0, 1.0, 1.0]);
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
            vals[0] = lambda[0] + lambda[1] + lambda[2];
            true
        }
    }

    #[test]
    fn test_constraints_split_routes_via_layout() {
        let prob = Fix;
        let mut g_l = vec![0.0; 3];
        let mut g_u = vec![0.0; 3];
        prob.constraint_bounds(&mut g_l, &mut g_u);
        let layout = ConstraintLayout::new(&g_l, &g_u);
        let nlp = SplitNlp::new(&prob, &layout);

        // Row 1 is the equality (n_c=1); rows 0, 2 are inequalities (n_d=2).
        assert_eq!(layout.n_c, 1);
        assert_eq!(layout.n_d, 2);

        let x = [3.0, 4.0];
        let mut c_raw = vec![0.0; layout.n_c];
        let mut d_out = vec![0.0; layout.n_d];
        assert!(nlp.constraints_split(&x, true, &mut c_raw, &mut d_out));
        // Equality row 1: g[1] = 7
        assert_eq!(c_raw[0], 7.0);
        // Inequality rows 0, 2: g[0]=7, g[2]=3 (order via d_to_combined)
        let i0 = layout.d_to_combined[0];
        let i1 = layout.d_to_combined[1];
        assert_eq!(d_out[0], if i0 == 0 { 7.0 } else { 3.0 });
        assert_eq!(d_out[1], if i1 == 0 { 7.0 } else { 3.0 });
    }

    #[test]
    fn test_jacobian_split_routes_via_combined_idx() {
        let prob = Fix;
        let mut g_l = vec![0.0; 3];
        let mut g_u = vec![0.0; 3];
        prob.constraint_bounds(&mut g_l, &mut g_u);
        let layout = ConstraintLayout::new(&g_l, &g_u);
        let nlp = SplitNlp::new(&prob, &layout);

        let (jac_rows, _jac_cols) = prob.jacobian_structure();
        let mut jac_c_combined_idx: Vec<usize> = Vec::new();
        let mut jac_d_combined_idx: Vec<usize> = Vec::new();
        for (idx, &row) in jac_rows.iter().enumerate() {
            if layout.eq_pos[row].is_some() {
                jac_c_combined_idx.push(idx);
            } else if layout.ineq_pos[row].is_some() {
                jac_d_combined_idx.push(idx);
            }
        }
        let mut jac_c_vals = vec![0.0; jac_c_combined_idx.len()];
        let mut jac_d_vals = vec![0.0; jac_d_combined_idx.len()];

        let x = [1.0, 1.0];
        assert!(nlp.jacobian_split(
            &x,
            true,
            &jac_c_combined_idx,
            &jac_d_combined_idx,
            &mut jac_c_vals,
            &mut jac_d_vals,
        ));
        // All 5 nonzeros are 1.0 in this fixture.
        assert!(jac_c_vals.iter().all(|&v| v == 1.0));
        assert!(jac_d_vals.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_hessian_from_split_composes_lambda() {
        let prob = Fix;
        let mut g_l = vec![0.0; 3];
        let mut g_u = vec![0.0; 3];
        prob.constraint_bounds(&mut g_l, &mut g_u);
        let layout = ConstraintLayout::new(&g_l, &g_u);
        let nlp = SplitNlp::new(&prob, &layout);

        // y_c is for the equality row (originally row 1); y_d is for
        // inequality rows (originally rows 0, 2).
        let y_c = vec![10.0]; // λ[1] = 10
        // d_to_combined[0]/[1] decides which value lands in λ[0] vs λ[2].
        let y_d = vec![20.0, 30.0];
        let mut hess = vec![0.0; 1];
        let x = [0.0, 0.0];
        assert!(nlp.hessian_from_split(&x, true, 1.0, &y_c, &y_d, &mut hess));
        // hessian_values returns lambda[0]+lambda[1]+lambda[2] = 60.
        assert_eq!(hess[0], 60.0);
    }

    #[test]
    fn test_constraints_combined_passthrough() {
        let prob = Fix;
        let mut g_l = vec![0.0; 3];
        let mut g_u = vec![0.0; 3];
        prob.constraint_bounds(&mut g_l, &mut g_u);
        let layout = ConstraintLayout::new(&g_l, &g_u);
        let nlp = SplitNlp::new(&prob, &layout);

        let x = [2.0, 3.0];
        let mut g = vec![0.0; 3];
        assert!(nlp.constraints_combined(&x, true, &mut g));
        assert_eq!(g, vec![5.0, 5.0, 2.0]);
    }
}
