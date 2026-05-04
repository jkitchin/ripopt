use std::cell::RefCell;
use std::time::{Duration, Instant};

use crate::convergence::{self, ConvergenceInfo, ConvergenceStatus};
use crate::filter::{self, Filter, FilterEntry};
use crate::kkt::{self, InertiaCorrectionParams};
use crate::linear_solver::dense::DenseLdl;
#[cfg(all(feature = "faer", not(any(feature = "feral", feature = "rmumps"))))]
use crate::linear_solver::sparse::SparseLdl;
#[cfg(feature = "feral")]
use crate::linear_solver::feral_direct::FeralLdl;
#[cfg(feature = "feral")]
use crate::linear_solver::feral_iterative::FeralIterativeMinres;
#[cfg(feature = "feral")]
use crate::linear_solver::feral_hybrid::FeralHybrid;
#[cfg(all(feature = "rmumps", not(feature = "feral")))]
use crate::linear_solver::multifrontal::MultifrontalLdl;
#[cfg(all(feature = "rmumps", not(feature = "feral")))]
use crate::linear_solver::iterative::IterativeMinres;
#[cfg(all(feature = "rmumps", not(feature = "feral")))]
use crate::linear_solver::hybrid::HybridSolver;
use crate::linear_solver::{KktMatrix, LinearSolver};
use crate::options::AlphaForY;
use crate::options::LinearSolverChoice;

/// Create a new sparse linear solver using the best available backend.
/// Prefers feral (multifrontal LDLᵀ, default), then rmumps, then faer (SparseLdl), then dense.
fn new_sparse_solver() -> Box<dyn LinearSolver> {
    new_sparse_solver_with_choice(LinearSolverChoice::Direct)
}

/// Create a sparse linear solver with the specified choice.
fn new_sparse_solver_with_choice(choice: LinearSolverChoice) -> Box<dyn LinearSolver> {
    match choice {
        LinearSolverChoice::Direct => {
            #[cfg(feature = "feral")]
            { return Box::new(FeralLdl::new()); }
            #[cfg(all(not(feature = "feral"), feature = "rmumps"))]
            { return Box::new(MultifrontalLdl::new()); }
            #[cfg(all(not(feature = "feral"), not(feature = "rmumps"), feature = "faer"))]
            { return Box::new(SparseLdl::new()); }
            #[cfg(not(any(feature = "feral", feature = "rmumps", feature = "faer")))]
            { return Box::new(DenseLdl::new()); }
        }
        LinearSolverChoice::Iterative => {
            #[cfg(feature = "feral")]
            { return Box::new(FeralIterativeMinres::new()); }
            #[cfg(all(not(feature = "feral"), feature = "rmumps"))]
            { return Box::new(IterativeMinres::new()); }
            #[cfg(all(not(feature = "feral"), not(feature = "rmumps")))]
            {
                log::warn!("Iterative solver requires feral or rmumps feature; falling back to direct");
                return new_sparse_solver_with_choice(LinearSolverChoice::Direct);
            }
        }
        LinearSolverChoice::Hybrid => {
            #[cfg(feature = "feral")]
            { return Box::new(FeralHybrid::new()); }
            #[cfg(all(not(feature = "feral"), feature = "rmumps"))]
            { return Box::new(HybridSolver::new()); }
            #[cfg(all(not(feature = "feral"), not(feature = "rmumps")))]
            {
                log::warn!("Hybrid solver requires feral or rmumps feature; falling back to direct");
                return new_sparse_solver_with_choice(LinearSolverChoice::Direct);
            }
        }
    }
}

/// Create the appropriate linear solver for a fallback KKT system.
/// Uses sparse solver when `use_sparse` is true, dense otherwise.
fn new_fallback_solver(use_sparse: bool) -> Box<dyn LinearSolver> {
    if use_sparse {
        new_sparse_solver()
    } else {
        Box::new(DenseLdl::new())
    }
}
use crate::options::BoundMultInitMethod;
use crate::options::FixedVariableTreatment;
use crate::options::SolverOptions;
use crate::problem::NlpProblem;
use crate::trace;
use crate::restoration_nlp::RestorationNlp;
use crate::result::{SolveResult, SolverDiagnostics, SolveStatus};
use crate::warmstart::WarmStartInitializer;
use crate::logging::rip_log;

/// NLP problem wrapper that rejects non-finite (NaN/Inf) values from
/// any evaluation. Mirrors Ipopt 3.14's `IpOrigIpoptNLP.cpp` per-call
/// `Eval_Error` checks (lines 498, 535, 580, 629). When the inner
/// implementation returns `true` but any output element is non-finite,
/// the wrapper converts that to a `false` return, which the IPM line
/// search treats as a trial-point rejection (and at the current
/// iterate, as `EvaluationError → NumericalBreakdown`).
///
/// Wrapped as the outermost layer in `solve_ipm` so every eval that
/// reaches the IPM core is guaranteed finite. The check covers
/// objective, gradient, constraints, Jacobian values, and Hessian
/// values; bounds / initial point / structure queries do not produce
/// numerical output and are not checked.
struct FiniteCheckedProblem<'a, P: NlpProblem> {
    inner: &'a P,
}

impl<'a, P: NlpProblem> FiniteCheckedProblem<'a, P> {
    fn new(inner: &'a P) -> Self {
        Self { inner }
    }
}

impl<P: NlpProblem> NlpProblem for FiniteCheckedProblem<'_, P> {
    fn num_variables(&self) -> usize { self.inner.num_variables() }
    fn num_constraints(&self) -> usize { self.inner.num_constraints() }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        self.inner.bounds(x_l, x_u);
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        self.inner.constraint_bounds(g_l, g_u);
    }
    fn initial_point(&self, x0: &mut [f64]) { self.inner.initial_point(x0); }
    fn initial_multipliers(
        &self,
        lam_g: &mut [f64],
        z_l: &mut [f64],
        z_u: &mut [f64],
    ) -> bool {
        self.inner.initial_multipliers(lam_g, z_l, z_u)
    }
    fn objective(&self, x: &[f64], new_x: bool, obj: &mut f64) -> bool {
        if !self.inner.objective(x, new_x, obj) { return false; }
        obj.is_finite()
    }
    fn gradient(&self, x: &[f64], new_x: bool, grad: &mut [f64]) -> bool {
        if !self.inner.gradient(x, new_x, grad) { return false; }
        grad.iter().all(|v| v.is_finite())
    }
    fn constraints(&self, x: &[f64], new_x: bool, g: &mut [f64]) -> bool {
        if !self.inner.constraints(x, new_x, g) { return false; }
        g.iter().all(|v| v.is_finite())
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        self.inner.jacobian_structure()
    }
    fn jacobian_values(&self, x: &[f64], new_x: bool, vals: &mut [f64]) -> bool {
        if !self.inner.jacobian_values(x, new_x, vals) { return false; }
        vals.iter().all(|v| v.is_finite())
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        self.inner.hessian_structure()
    }
    fn hessian_values(
        &self,
        x: &[f64],
        new_x: bool,
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) -> bool {
        if !self.inner.hessian_values(x, new_x, obj_factor, lambda, vals) {
            return false;
        }
        vals.iter().all(|v| v.is_finite())
    }
    fn notify_mu(&self, mu: f64) {
        self.inner.notify_mu(mu);
    }
    fn resto_early_exit(&self, x: &[f64]) -> bool {
        self.inner.resto_early_exit(x)
    }
}

/// NLP problem wrapper that applies gradient-based scaling.
///
/// Scales objective by `obj_scaling` and each constraint `i` by `g_scaling[i]`
/// so that the max gradient norm at the initial point is ≤ 100.
/// This matches Ipopt's `nlp_scaling_method = gradient-based`.
///
/// The constraint-bound path also applies `nlp_lower/upper_bound_inf`
/// sentinel mapping and `bound_relax_factor` IN RAW SPACE before
/// scaling, mirroring `IpOrigIpoptNLP::InitializeStructures`
/// (`ref/Ipopt/src/Algorithm/IpOrigIpoptNLP.cpp:343-374`): relax_bounds
/// runs on raw `d_L`/`d_U`, then `apply_vector_scaling_d_LU` scales the
/// already-relaxed bounds. Doing the order the other way (scale, then
/// relax in scaled space) gives `1e-8` padding for any constraint whose
/// raw bound is `0`, instead of Ipopt's `scale × 1e-8`. For rows with
/// `|raw_b| < 1/scale` this is an O(1/scale) effective-bound error.
struct ScaledProblem<'a, P: NlpProblem> {
    inner: &'a P,
    obj_scaling: f64,
    g_scaling: Vec<f64>,
    jac_rows: Vec<usize>,
    bound_relax_factor: f64,
    constr_viol_tol: f64,
    nlp_lower_bound_inf: f64,
    nlp_upper_bound_inf: f64,
}

impl<P: NlpProblem> NlpProblem for ScaledProblem<'_, P> {
    fn num_variables(&self) -> usize {
        self.inner.num_variables()
    }
    fn num_constraints(&self) -> usize {
        self.inner.num_constraints()
    }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        self.inner.bounds(x_l, x_u);
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        // Order matches Ipopt 3.14 IpOrigIpoptNLP.cpp:343-374:
        //   raw inner bounds → sentinel-to-infinity → relax_bounds (raw)
        //   → apply_vector_scaling_d_LU (scale)
        // The relaxation amount uses |raw_b| (not |scaled_b|), which
        // matters whenever |raw_b| < 1/scale — most importantly for
        // raw bounds at exactly 0 (e.g. inequalities `g(x) ≤ 0`).
        self.inner.constraint_bounds(g_l, g_u);
        for i in 0..g_l.len() {
            if g_l[i] <= self.nlp_lower_bound_inf {
                g_l[i] = f64::NEG_INFINITY;
            }
            if g_u[i] >= self.nlp_upper_bound_inf {
                g_u[i] = f64::INFINITY;
            }
        }
        if self.bound_relax_factor > 0.0 {
            for i in 0..g_l.len() {
                if g_l[i].is_finite() && g_u[i].is_finite() && g_l[i] == g_u[i] {
                    continue;
                }
                if g_l[i].is_finite() {
                    let delta = (self.bound_relax_factor * g_l[i].abs().max(1.0))
                        .min(self.constr_viol_tol);
                    g_l[i] -= delta;
                }
                if g_u[i].is_finite() {
                    let delta = (self.bound_relax_factor * g_u[i].abs().max(1.0))
                        .min(self.constr_viol_tol);
                    g_u[i] += delta;
                }
            }
        }
        for (i, &s) in self.g_scaling.iter().enumerate() {
            if g_l[i].is_finite() {
                g_l[i] *= s;
            }
            if g_u[i].is_finite() {
                g_u[i] *= s;
            }
        }
    }
    fn initial_point(&self, x0: &mut [f64]) {
        self.inner.initial_point(x0);
    }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        if !self.inner.objective(x, _new_x, obj) { return false; }
        *obj *= self.obj_scaling;
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        if !self.inner.gradient(x, _new_x, grad) { return false; }
        for g in grad.iter_mut() {
            *g *= self.obj_scaling;
        }
        true
    }
    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        if !self.inner.constraints(x, _new_x, g) { return false; }
        for (i, &s) in self.g_scaling.iter().enumerate() {
            g[i] *= s;
        }
        true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        self.inner.jacobian_structure()
    }
    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        if !self.inner.jacobian_values(x, _new_x, vals) { return false; }
        for (idx, &row) in self.jac_rows.iter().enumerate() {
            vals[idx] *= self.g_scaling[row];
        }
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        self.inner.hessian_structure()
    }
    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        let scaled_lambda: Vec<f64> = lambda
            .iter()
            .zip(self.g_scaling.iter())
            .map(|(l, s)| l * s)
            .collect();
        self.inner
            .hessian_values(x, _new_x, obj_factor * self.obj_scaling, &scaled_lambda, vals)
    }
    fn notify_mu(&self, mu: f64) {
        self.inner.notify_mu(mu);
    }
    fn resto_early_exit(&self, x: &[f64]) -> bool {
        self.inner.resto_early_exit(x)
    }
}

/// NLP problem wrapper that applies user-supplied primal scaling
/// (`options.user_x_scaling`).
///
/// The wrapper presents an *internal* variable `x' = D_x · x` to the
/// solver, where `D_x = diag(dx)` and `dx` is the user-provided
/// strictly-positive scaling vector. It mirrors Ipopt's
/// `IpScaledNLP` / `UserScaling` pipeline restricted to x-scaling
/// (objective and constraint scaling are left to `ScaledProblem`).
///
/// Per-quantity transformations (with `df = 1`, `dc = I`):
///   * `bounds`:           `x_L' = D_x · x_L`,  `x_U' = D_x · x_U`
///   * `initial_point`:    `x0'  = D_x · x0`
///   * `initial_mult`:     `z_L' = z_L / dx`,   `z_U' = z_U / dx`,
///                         `lam_g` unchanged
///   * `objective(x')`:    inner.objective(x' / dx)
///   * `gradient(x')`:     `∇f' = ∇f / dx` (componentwise)
///   * `constraints(x')`:  inner.constraints(x' / dx)
///   * `jacobian_values`:  `J'[i,j] = J[i,j] / dx[j]`
///   * `hessian_values`:   `H'[i,j] = H[i,j] / (dx[i] * dx[j])`
///
/// Sparsity structure is unchanged — only values are scaled.
///
/// On output the IPM driver must apply the inverse map:
///   `x_user      = x'      / dx`
///   `z_L_user    = dx      · z_L_internal`
///   `z_U_user    = dx      · z_U_internal`
///   `lam_g_user  = lam_g_internal`
///
/// References: Ipopt 3.14 `IpNLPScaling.cpp`,
/// `IpStandardScalingBase.cpp::apply_vector_scaling_x`,
/// `IpScaledMatrix.cpp` (post-multiply by `D_x^-1`),
/// `IpSymScaledMatrix.cpp` (`D_x^-1 H D_x^-1`).
struct XScaledProblem<'a, P: NlpProblem> {
    inner: &'a P,
    dx: Vec<f64>,
    inv_dx: Vec<f64>,
    jac_cols: Vec<usize>,
    hess_rows: Vec<usize>,
    hess_cols: Vec<usize>,
    scratch_x: RefCell<Vec<f64>>,
}

impl<'a, P: NlpProblem> XScaledProblem<'a, P> {
    fn new(inner: &'a P, dx: Vec<f64>) -> Self {
        let n = inner.num_variables();
        debug_assert_eq!(dx.len(), n);
        let inv_dx: Vec<f64> = dx.iter().map(|&d| 1.0 / d).collect();
        let (_, jac_cols) = inner.jacobian_structure();
        let (hess_rows, hess_cols) = inner.hessian_structure();
        Self {
            inner,
            dx,
            inv_dx,
            jac_cols,
            hess_rows,
            hess_cols,
            scratch_x: RefCell::new(vec![0.0; n]),
        }
    }

    /// Fill `scratch_x` with `x / dx` (unscale solver-side `x'`
    /// before passing to inner). Returns the borrow.
    fn unscale_x<'b>(&'b self, x: &[f64]) -> std::cell::Ref<'b, Vec<f64>> {
        {
            let mut buf = self.scratch_x.borrow_mut();
            for i in 0..x.len() {
                buf[i] = x[i] * self.inv_dx[i];
            }
        }
        self.scratch_x.borrow()
    }
}

impl<P: NlpProblem> NlpProblem for XScaledProblem<'_, P> {
    fn num_variables(&self) -> usize {
        self.inner.num_variables()
    }
    fn num_constraints(&self) -> usize {
        self.inner.num_constraints()
    }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        self.inner.bounds(x_l, x_u);
        for i in 0..x_l.len() {
            if x_l[i].is_finite() {
                x_l[i] *= self.dx[i];
            }
            if x_u[i].is_finite() {
                x_u[i] *= self.dx[i];
            }
        }
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        self.inner.constraint_bounds(g_l, g_u);
    }
    fn initial_point(&self, x0: &mut [f64]) {
        self.inner.initial_point(x0);
        for i in 0..x0.len() {
            x0[i] *= self.dx[i];
        }
    }
    fn initial_multipliers(
        &self,
        lam_g: &mut [f64],
        z_l: &mut [f64],
        z_u: &mut [f64],
    ) -> bool {
        if !self.inner.initial_multipliers(lam_g, z_l, z_u) {
            return false;
        }
        for i in 0..z_l.len() {
            z_l[i] *= self.inv_dx[i];
            z_u[i] *= self.inv_dx[i];
        }
        true
    }
    fn objective(&self, x: &[f64], new_x: bool, obj: &mut f64) -> bool {
        let x_user = self.unscale_x(x);
        self.inner.objective(&x_user, new_x, obj)
    }
    fn gradient(&self, x: &[f64], new_x: bool, grad: &mut [f64]) -> bool {
        let x_user = self.unscale_x(x);
        if !self.inner.gradient(&x_user, new_x, grad) {
            return false;
        }
        for i in 0..grad.len() {
            grad[i] *= self.inv_dx[i];
        }
        true
    }
    fn constraints(&self, x: &[f64], new_x: bool, g: &mut [f64]) -> bool {
        let x_user = self.unscale_x(x);
        self.inner.constraints(&x_user, new_x, g)
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        self.inner.jacobian_structure()
    }
    fn jacobian_values(&self, x: &[f64], new_x: bool, vals: &mut [f64]) -> bool {
        let x_user = self.unscale_x(x);
        if !self.inner.jacobian_values(&x_user, new_x, vals) {
            return false;
        }
        for (idx, &col) in self.jac_cols.iter().enumerate() {
            vals[idx] *= self.inv_dx[col];
        }
        true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        self.inner.hessian_structure()
    }
    fn hessian_values(
        &self,
        x: &[f64],
        new_x: bool,
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) -> bool {
        let x_user = self.unscale_x(x);
        if !self.inner.hessian_values(&x_user, new_x, obj_factor, lambda, vals) {
            return false;
        }
        for (idx, (&r, &c)) in self
            .hess_rows
            .iter()
            .zip(self.hess_cols.iter())
            .enumerate()
        {
            vals[idx] *= self.inv_dx[r] * self.inv_dx[c];
        }
        true
    }
    fn notify_mu(&self, mu: f64) {
        self.inner.notify_mu(mu);
    }
    fn resto_early_exit(&self, x: &[f64]) -> bool {
        self.inner.resto_early_exit(x)
    }
}

/// Snapshot of the most recent iterate that satisfied the acceptable-level
/// thresholds (Ipopt's `RestoreAcceptablePoint`). When restoration would
/// otherwise be triggered on a near-feasible iterate (where the resto NLP
/// is ill-defined), the IPM rolls back to this point and exits with
/// `SolveStatus::Acceptable`. Mirrors Ipopt's `STOP_AT_ACCEPTABLE_POINT`
/// path in `IpIpoptAlgorithm.cpp`.
struct IterateSnapshot {
    x: Vec<f64>,
    y_c: Vec<f64>,
    y_d: Vec<f64>,
    /// Native compressed bound multipliers (Phase 6d.6: combined
    /// storage dropped; this is the canonical form).
    z_l_compressed: Vec<f64>,
    z_u_compressed: Vec<f64>,
    /// Phase 8d: native compressed slack-bound multipliers (combined
    /// `v_l`/`v_u` storage dropped).
    v_l_compressed: Vec<f64>,
    v_u_compressed: Vec<f64>,
    s: Vec<f64>,
    mu: f64,
    obj: f64,
    c_x: Vec<f64>,
    d_x: Vec<f64>,
    grad_f: Vec<f64>,
    filter_entries: Vec<FilterEntry>,
    iteration: usize,
}

impl IterateSnapshot {
    fn capture(state: &SolverState, filter: &Filter, iteration: usize) -> Self {
        Self {
            x: state.x.clone(),
            y_c: state.y_c.clone(),
            y_d: state.y_d.clone(),
            z_l_compressed: state.z_l_compressed.clone(),
            z_u_compressed: state.z_u_compressed.clone(),
            v_l_compressed: state.v_l_compressed.clone(),
            v_u_compressed: state.v_u_compressed.clone(),
            s: state.s.clone(),
            mu: state.mu,
            obj: state.obj,
            c_x: state.c_x.clone(),
            d_x: state.d_x.clone(),
            grad_f: state.grad_f.clone(),
            filter_entries: filter.save_entries(),
            iteration,
        }
    }

    fn restore(&self, state: &mut SolverState, filter: &mut Filter) {
        state.x = self.x.clone();
        state.y_c = self.y_c.clone();
        state.y_d = self.y_d.clone();
        state.z_l_compressed = self.z_l_compressed.clone();
        state.z_u_compressed = self.z_u_compressed.clone();
        state.v_l_compressed = self.v_l_compressed.clone();
        state.v_u_compressed = self.v_u_compressed.clone();
        state.s = self.s.clone();
        state.mu = self.mu;
        state.obj = self.obj;
        state.c_x = self.c_x.clone();
        state.d_x = self.d_x.clone();
        state.grad_f = self.grad_f.clone();
        filter.restore_entries(self.filter_entries.clone());
    }
}

/// Saved state for the watchdog mechanism.
struct WatchdogSavedState {
    x: Vec<f64>,
    y_c: Vec<f64>,
    y_d: Vec<f64>,
    /// Phase 6d.2: native compressed bound multipliers.
    z_l_compressed: Vec<f64>,
    z_u_compressed: Vec<f64>,
    /// Phase 8d: native compressed slack-bound multipliers (combined
    /// `v_l`/`v_u` storage dropped).
    v_l_compressed: Vec<f64>,
    v_u_compressed: Vec<f64>,
    s: Vec<f64>,
    mu: f64,
    obj: f64,
    c_x: Vec<f64>,
    d_x: Vec<f64>,
    grad_f: Vec<f64>,
    filter_entries: Vec<FilterEntry>,
    theta: f64,
    phi: f64,
}

impl WatchdogSavedState {
    /// Snapshot the full iterate plus the filter entries and the
    /// (theta, phi) pair used as the watchdog's progress reference.
    fn snapshot(state: &SolverState, filter: &Filter, theta: f64, phi: f64) -> Self {
        Self {
            x: state.x.clone(),
            y_c: state.y_c.clone(),
            y_d: state.y_d.clone(),
            z_l_compressed: state.z_l_compressed.clone(),
            z_u_compressed: state.z_u_compressed.clone(),
            v_l_compressed: state.v_l_compressed.clone(),
            v_u_compressed: state.v_u_compressed.clone(),
            s: state.s.clone(),
            mu: state.mu,
            obj: state.obj,
            c_x: state.c_x.clone(),
            d_x: state.d_x.clone(),
            grad_f: state.grad_f.clone(),
            filter_entries: filter.save_entries(),
            theta,
            phi,
        }
    }

    /// Restore the snapshotted iterate fields into `state`. Filter
    /// entry restoration is left to the caller (the watchdog augments
    /// the filter after restore, which the helper would obscure).
    fn restore(&self, state: &mut SolverState) {
        state.x = self.x.clone();
        state.y_c = self.y_c.clone();
        state.y_d = self.y_d.clone();
        state.z_l_compressed = self.z_l_compressed.clone();
        state.z_u_compressed = self.z_u_compressed.clone();
        state.v_l_compressed = self.v_l_compressed.clone();
        state.v_u_compressed = self.v_u_compressed.clone();
        state.s = self.s.clone();
        state.mu = self.mu;
        state.obj = self.obj;
        state.c_x = self.c_x.clone();
        state.d_x = self.d_x.clone();
        state.grad_f = self.grad_f.clone();
        // T3.25 follow-up: watchdog revert mutates x/y/z/v outside the
        // line-search choke point that bumps atags. Bump them and drop
        // the cache so the next factor cannot replay a stale `(δ_w,
        // δ_c)` against a now-wrong matrix.
        state.bump_all_kkt_atags();
        state.factor_cache.invalidate();
    }
}

/// Central state struct for the IPM solver.
pub(crate) struct SolverState {
    /// Current primal variables.
    pub x: Vec<f64>,
    /// Equality-constraint multipliers `y_c` (size n_c). Mirrors Ipopt's
    /// `IteratesVector::y_c()` (`IpIteratesVector.hpp` slot 2). Phase 3b
    /// of the data-layout refactor: replaces the combined `y: Vec<f64>`
    /// (size m). User-facing combined multipliers are reconstructed in
    /// `unscale_solution_vectors` via the layout.
    pub y_c: Vec<f64>,
    /// Inequality-constraint multipliers `y_d` (size n_d). Mirrors Ipopt's
    /// `IteratesVector::y_d()` (slot 3).
    pub y_d: Vec<f64>,
    /// Search direction: primal.
    pub dx: Vec<f64>,
    /// Search direction: equality multipliers `dy_c` (size n_c). Phase 3c
    /// of the data-layout refactor: replaces the combined
    /// `dy: Vec<f64>` (size m). Mirrors Ipopt's IteratesVector
    /// search-direction slot for y_c.
    pub dy_c: Vec<f64>,
    /// Search direction: inequality multipliers `dy_d` (size n_d).
    pub dy_d: Vec<f64>,
    /// Explicit slack iterate (Ipopt's `s`, size `n_d`). Pushed to interior of
    /// `[d_L, d_U]` at init via slack_bound_push/slack_bound_frac, then advanced
    /// each iteration by `s ← s + α_p · ds`. Phase 3d of the data-layout refactor:
    /// dropped the equality-row sentinel slots; `s` is now d-block-only natively
    /// (matches Ipopt 3.14 `IpIpoptData.cpp:140` where `s` has dimension `n_d`).
    /// User-facing combined-indexed reads route through `state.s_at(i)`.
    /// Source: Ipopt 3.14 IpIteratesVector.hpp slot 1.
    pub s: Vec<f64>,
    /// Search direction for slack iterate (Ipopt's `delta_s`, size `n_d`).
    /// Computed by `recover_ds`: `ds[k] = (J_d·dx)[k] + (d[k] - s[k]) - δ_d·dy_d[k]`.
    /// Phase 3d: resized from m to n_d. Source: IpStdAugSystemSolver.cpp:431-465.
    pub ds: Vec<f64>,
    /// Barrier parameter.
    pub mu: f64,
    /// Primal step size.
    pub alpha_primal: f64,
    /// Dual step size.
    pub alpha_dual: f64,
    /// Iteration counter.
    pub iter: usize,
    // Phase 7c: combined-form `x_l`/`x_u` (size `n` with `±inf` sentinels)
    // dropped. The compressed mirrors `x_l_compressed` (size `n_x_l`) and
    // `x_u_compressed` (size `n_x_u`) are now the sole canonical storage,
    // matching Ipopt's `x_L_`/`x_U_` ExpansionMatrix-projected bounds
    // (`IpOrigIpoptNLP.hpp:226,253`). Use `state.x_l_at(i)` / `x_u_at(i)`
    // for per-element reads (returns `±inf` on unbounded sides) or
    // `state.x_l_combined()` / `x_u_combined()` to materialize a full-`n`
    // view at API boundaries.
    /// Equality-row target value (Ipopt's `c_rhs`, size `layout.n_c`):
    /// stores the equality-row target. Phase 3f-final: native split
    /// storage; the legacy combined `g_l`/`g_u` fields are gone.
    /// Materialise via `state.g_l_combined()` / `state.g_u_combined()`
    /// when an m-length slice is required.
    pub c_rhs: Vec<f64>,
    /// Number of variables.
    pub n: usize,
    /// Number of constraints.
    pub m: usize,
    /// Current objective value.
    pub obj: f64,
    /// Current gradient.
    pub grad_f: Vec<f64>,
    /// Phase 5f: equality residual `c(x)` (size `n_c`). Mirrors Ipopt's
    /// `OrigIpoptNLP::c()` (`IpIpoptNLP.hpp:117`), where the equality
    /// target is baked into the residual so `c(x) = 0` at feasibility.
    /// Written by `evaluate_with_linear` (and by the test-only
    /// `set_g_combined`).
    pub c_x: Vec<f64>,
    /// Phase 5f: inequality value `d(x)` (size `n_d`). Mirrors Ipopt's
    /// `OrigIpoptNLP::d()` (`IpIpoptNLP.hpp:118`).
    pub d_x: Vec<f64>,
    /// Jacobian sparsity pattern (combined m-row wire form). Kept as an
    /// immutable wire for the restoration NLP boundary, which still
    /// expects an m-form triplet pattern. Numerical values live in the
    /// split `jac_c_vals` / `jac_d_vals` storage; there is no combined
    /// values mirror. Read by test helpers (`rebuild_split_jac_structure`)
    /// that hand-craft a state from a combined triplet.
    #[allow(dead_code)]
    pub jac_rows: Vec<usize>,
    #[allow(dead_code)]
    pub jac_cols: Vec<usize>,
    /// Phase 4 split Jacobian — equality-row block (Ipopt's `Jac_c`,
    /// `IpOrigIpoptNLP.hpp:439`). Triplet form sized `jac_c_nnz`:
    ///   `jac_c_rows[k] ∈ 0..n_c`  (target row in `c`-block coordinates)
    ///   `jac_c_cols[k] ∈ 0..n`
    /// Populated at construction (structure) and refreshed each
    /// `problem.jacobian_values` call via [`refresh_split_jac_vals`].
    /// Phase 4a is additive — readers still consume the combined
    /// triplet; Phase 4b will migrate `kkt_aug` and other consumers.
    pub jac_c_rows: Vec<usize>,
    pub jac_c_cols: Vec<usize>,
    pub jac_c_vals: Vec<f64>,
    /// Phase 4 split Jacobian — inequality-row block (Ipopt's `Jac_d`,
    /// `IpOrigIpoptNLP.hpp:449`). Triplet form sized `jac_d_nnz`:
    ///   `jac_d_rows[k] ∈ 0..n_d`  (target row in `d`-block coordinates)
    ///   `jac_d_cols[k] ∈ 0..n`
    pub jac_d_rows: Vec<usize>,
    pub jac_d_cols: Vec<usize>,
    pub jac_d_vals: Vec<f64>,
    /// Phase 4 index maps: `jac_c_combined_idx[k]` is the position in
    /// the combined `jac_*` triplet that supplies `jac_c_vals[k]`. Same
    /// for `jac_d`. Built once at construction; used by
    /// `refresh_split_jac_vals` to copy values without re-checking the
    /// layout per triplet.
    pub jac_c_combined_idx: Vec<usize>,
    pub jac_d_combined_idx: Vec<usize>,
    /// Hessian structure and values.
    pub hess_rows: Vec<usize>,
    pub hess_cols: Vec<usize>,
    pub hess_vals: Vec<f64>,
    /// Consecutive acceptable iterations.
    pub consecutive_acceptable: usize,
    /// Objective scaling factor (for NLP scaling / result unscaling).
    pub obj_scaling: f64,
    /// Equality-block constraint scaling factors `dc` (size n_c). Mirrors
    /// Ipopt's `c_scaling` produced by `IpGradientScaling.cpp:144-185`.
    /// Phase 3 of the data-layout refactor: replaces the combined
    /// `g_scaling: Vec<f64>` (size m). User-facing combined scaling is
    /// reconstructed in `unscale_solution_vectors` via the layout.
    pub c_scaling: Vec<f64>,
    /// Inequality-block constraint scaling factors `dd` (size n_d). Mirrors
    /// Ipopt's `d_scaling` produced by `IpGradientScaling.cpp:191-232`.
    pub d_scaling: Vec<f64>,
    /// Equality / inequality constraint layout. Built once from `g_l`/`g_u`
    /// at construction; stable for the problem lifetime. Single source of
    /// truth for the c-block / d-block split (replaces per-assemble
    /// `ConstraintLayout::new` calls — Phase 1 of the data-layout refactor,
    /// see `docs/V0.8_DATA_LAYOUT_REFACTOR_PLAN.md`).
    pub layout: crate::constraint_layout::ConstraintLayout,
    /// Variable-bound layout. Built once from `x_l`/`x_u` at construction;
    /// stable for the problem lifetime. Single source of truth for the
    /// finite-bound compression that mirrors Ipopt's `Px_L_` / `Px_U_`
    /// ExpansionMatrix pair (`IpOrigIpoptNLP.hpp:197-219`). Phase 6 of
    /// the data-layout refactor — see
    /// `docs/V0.8_DATA_LAYOUT_REFACTOR_PLAN.md`.
    pub bound_layout: crate::bound_layout::BoundLayout,
    /// Compressed lower-bound multipliers (size `bound_layout.n_x_l`).
    /// Mirrors Ipopt's `z_L` (`IpIpoptData.cpp:140`). Phase 6b: populated
    /// as a projection of the combined `z_l` at every mutation site.
    /// Phase 6c will flip readers to consume this instead of the combined
    /// form; Phase 6d will drop the combined.
    pub z_l_compressed: Vec<f64>,
    /// Compressed upper-bound multipliers (size `bound_layout.n_x_u`).
    /// Mirrors Ipopt's `z_U`. See [`Self::z_l_compressed`].
    pub z_u_compressed: Vec<f64>,
    /// Compressed lower-bound multiplier search direction
    /// (size `bound_layout.n_x_l`). Phase 6b additive mirror of `dz_l`.
    pub dz_l_compressed: Vec<f64>,
    /// Compressed upper-bound multiplier search direction
    /// (size `bound_layout.n_x_u`). Phase 6b additive mirror of `dz_u`.
    pub dz_u_compressed: Vec<f64>,
    /// Phase 7a: compressed lower variable bounds (size `bound_layout.n_x_l`),
    /// holding only the finite lower-bound values (one entry per `k` in
    /// `0..n_x_l`, mapped via `bound_layout.x_l_to_full`). Mirrors Ipopt's
    /// `x_L_` (`IpOrigIpoptNLP.hpp:226`). Phase 7a is additive — readers
    /// still consume the combined `x_l` (size `n` with `-inf` sentinels);
    /// later sub-phases migrate readers and drop the combined form.
    pub x_l_compressed: Vec<f64>,
    /// Phase 7a: compressed upper variable bounds (size `bound_layout.n_x_u`).
    /// Mirrors Ipopt's `x_U_`. See [`Self::x_l_compressed`].
    pub x_u_compressed: Vec<f64>,
    /// Phase 8a: slack-bound expansion-matrix layout. Mirrors Ipopt's
    /// `Pd_L_`/`Pd_U_` ExpansionMatrix pair (`IpOrigIpoptNLP.hpp:241-263`).
    /// Indexes the n_d slack rows; classifies each as having a finite
    /// lower / upper slack bound. Built once per `SolverState` from
    /// `d_l`/`d_u`.
    pub d_bound_layout: crate::d_bound_layout::DBoundLayout,
    /// Phase 8b: compressed slack-lower-bound multipliers (size
    /// `d_bound_layout.n_d_l`). Mirrors Ipopt's `v_L`. Populated as a
    /// projection of the combined `v_l` at every mutation site; Phase 8c
    /// migrates readers and Phase 8d drops the combined storage.
    pub v_l_compressed: Vec<f64>,
    /// Phase 8b: compressed slack-upper-bound multipliers (size
    /// `d_bound_layout.n_d_u`). Mirrors Ipopt's `v_U`.
    pub v_u_compressed: Vec<f64>,
    /// Phase 8b: compressed slack-lower-bound multiplier search
    /// direction (size `d_bound_layout.n_d_l`). Mirrors Ipopt's `dv_L`.
    pub dv_l_compressed: Vec<f64>,
    /// Phase 8b: compressed slack-upper-bound multiplier search
    /// direction (size `d_bound_layout.n_d_u`).
    pub dv_u_compressed: Vec<f64>,
    /// Phase 9b: compressed slack-row lower bounds (size
    /// `d_bound_layout.n_d_l`). Mirrors Ipopt's `d_L_` in
    /// `IpOrigIpoptNLP.hpp:241-263`. Holds finite lower bounds only;
    /// the combined `d_l` array (size `n_d`) keeps `f64::NEG_INFINITY`
    /// in the unbounded slots.
    pub d_l_compressed: Vec<f64>,
    /// Phase 9b: compressed slack-row upper bounds (size
    /// `d_bound_layout.n_d_u`). Mirrors Ipopt's `d_U_`.
    pub d_u_compressed: Vec<f64>,
    /// Accumulated solver diagnostics.
    pub diagnostics: SolverDiagnostics,
    /// Last point at which evaluations were performed (for new_x tracking).
    /// Initialized to NaN so the first evaluation always gets new_x = true.
    x_last_eval: Vec<f64>,
    /// Cumulative count of slack adjustments performed by
    /// `apply_slack_move` (Ipopt 3.14's `slack_move`). Each undersized
    /// primal slack found after an accepted step contributes one
    /// increment; surfaced for diagnostics only.
    pub adjusted_slacks_count: usize,
    /// True when the problem is square in Ipopt's sense:
    /// `m == n` or the equality-constraint count equals `n`.
    /// Mirrors `IpoptCalculatedQuantities::IsSquareProblem`
    /// (IpIpoptCalculatedQuantities.cpp:3732). Set once at setup
    /// after bound preprocessing and read-only thereafter.
    /// Currently unread: kept for future restoration / convergence
    /// hooks that need square-problem detection (Ipopt branches in
    /// `IpAlgorithm::ComputeFeasibilityMultipliers`).
    #[allow(dead_code)]
    pub is_square: bool,
    /// Most recent iterate that met the acceptable-level thresholds.
    /// Overwritten every time the convergence helpers see acceptable
    /// quality; consumed by the restoration trigger to fall back to
    /// `Acceptable` instead of attempting a singular resto NLP on a
    /// near-feasible iterate.
    acceptable_iterate: Option<IterateSnapshot>,
    /// Objective value of the previous iterate, used by the
    /// acceptable-level relative-change gate
    /// (`acceptable_obj_change_tol`). `None` on iteration 0.
    pub last_obj_for_acceptable: Option<f64>,
    /// T3.25: monotone version counters for the upstream KKT inputs.
    /// Bumped at the coarse mutation events the IPM controls (line
    /// search step accepted, multiplier update, callback re-evaluation
    /// after a new x). Snapshotted into `KktSystem.input_atags` at
    /// assembly time and consulted by `FactorCache` to short-circuit
    /// redundant factorizations within a single iteration (QF mu oracle
    /// → main step, Mehrotra predictor → corrector, Gondzio MCC).
    pub kkt_atags: kkt::KktInputAtags,
    /// T3.25 follow-up: per-iteration factorization cache shared
    /// across the QF mu oracle, the main IPM step, and the condensed
    /// fallback retries. Initialised from
    /// `SolverOptions::factor_cache_enabled` in `SolverState::new`;
    /// when the option is `false` every call is a forced miss
    /// (`factor_with_inertia_correction_cached` still tracks
    /// `factor_calls` for diagnostics).
    pub factor_cache: kkt::FactorCache,
    /// B11: cumulative NLP-callback counts surfaced in the final
    /// summary. Mirror Ipopt's per-callback counters
    /// (`IpOrigIpoptNLP::FinalizeSolution` reports). ripopt's
    /// `constraints` and `jacobian_values` fill the joint c/d block in
    /// one call, so `n_constr_evals`/`n_jac_evals` count both the
    /// equality and the inequality side; the final summary prints the
    /// same value on both rows.
    pub n_obj_evals: usize,
    pub n_grad_evals: usize,
    pub n_constr_evals: usize,
    pub n_jac_evals: usize,
    pub n_hess_evals: usize,
}

impl SolverState {
    /// T3.25: invalidate all 11 KKT-input atags. Coarse hammer used
    /// after restoration handoffs, snapshot restores, and any other
    /// path where state mutates outside the line-search choke point.
    /// Cheap (11 increments); correctness over precision.
    #[inline]
    pub fn bump_all_kkt_atags(&mut self) {
        let a = &mut self.kkt_atags;
        a.w = a.w.wrapping_add(1);
        a.j_c = a.j_c.wrapping_add(1);
        a.j_d = a.j_d.wrapping_add(1);
        a.z_l = a.z_l.wrapping_add(1);
        a.z_u = a.z_u.wrapping_add(1);
        a.v_l = a.v_l.wrapping_add(1);
        a.v_u = a.v_u.wrapping_add(1);
        a.slacks_x = a.slacks_x.wrapping_add(1);
        a.slacks_s = a.slacks_s.wrapping_add(1);
        a.sigma_x = a.sigma_x.wrapping_add(1);
        a.sigma_s = a.sigma_s.wrapping_add(1);
    }

    /// T3.25: bump just the dual-multiplier atags (z_l, z_u, v_l, v_u),
    /// plus their derived `sigma_x` and `sigma_s` views. Used by paths
    /// that update bound multipliers without changing the primal x.
    #[inline]
    pub fn bump_dual_atags(&mut self) {
        let a = &mut self.kkt_atags;
        a.z_l = a.z_l.wrapping_add(1);
        a.z_u = a.z_u.wrapping_add(1);
        a.v_l = a.v_l.wrapping_add(1);
        a.v_u = a.v_u.wrapping_add(1);
        a.sigma_x = a.sigma_x.wrapping_add(1);
        a.sigma_s = a.sigma_s.wrapping_add(1);
    }
}

/// Barrier parameter mode (Ipopt's adaptive mu strategy).
#[derive(Debug, Clone, Copy, PartialEq)]
enum MuMode {
    /// Free mode: mu chosen by oracle each iteration. Can increase or decrease.
    Free,
    /// Fixed mode: monotone mu decrease. Subproblem solved to barrier_tol_factor * mu.
    Fixed,
}

/// State for the free/fixed mu update strategy.
struct MuState {
    mode: MuMode,
    /// Sliding window of KKT error values for progress tracking.
    ref_vals: Vec<f64>,
    /// Maximum reference values to keep.
    num_refs_max: usize,
    /// Required reduction factor (sufficient progress if error < refs_red_fact * any ref).
    refs_red_fact: f64,
    /// Flag for tiny step detection.
    tiny_step: bool,
    /// Flag to indicate first iteration after mode switch.
    first_iter_in_mode: bool,
    /// Count of consecutive restoration failures for giving up.
    consecutive_restoration_failures: usize,
    /// Count of consecutive insufficient-progress iterations in Free mode.
    consecutive_insufficient: usize,
    /// Number of consecutive iterations that have been accepted via the
    /// soft-restoration path (Ipopt's `soft_resto_counter_`,
    /// `IpBacktrackingLineSearch.cpp:442-444`). Capped at
    /// `max_soft_resto_iters = 10` before escalating to full
    /// restoration. Reset whenever the regular line search accepts a
    /// step or the full restoration cascade runs.
    consecutive_soft_restoration: usize,
    /// Sliding window of dual infeasibility values for stagnation detection.
    dual_inf_window: Vec<f64>,
    /// Snapshot of `avg_compl` at the *first* Free-mode μ-oracle call.
    /// Used to derive Ipopt's `mu_max = mu_max_fact * initial_avg_compl`
    /// upper bound on adaptive μ (`IpAdaptiveMuUpdate.cpp:267-273`).
    /// `None` until the first oracle call captures it.
    initial_avg_compl: Option<f64>,
    /// T3.11: snapshot of the last Free-mode iterate that satisfied
    /// `CheckSufficientProgress` (Ipopt `accepted_point_`,
    /// `IpAdaptiveMuUpdate.cpp:541-545`). Restored on the Free→Fixed
    /// switch when `adaptive_mu_restore_previous_iterate` is on. `None`
    /// until the option fires the first capture.
    accepted_iterate: Option<AcceptedIterateSnapshot>,
}

/// T3.11: full primal-dual iterate snapshot used by the
/// `adaptive_mu_restore_previous_iterate` rollback. Mirrors Ipopt's
/// `IteratesVector` payload (`IpAdaptiveMuUpdate.cpp:367-369`); slacks
/// are implicit in ripopt so only x/y/z_l/z_u/v_l/v_u are stored.
#[derive(Clone)]
struct AcceptedIterateSnapshot {
    x: Vec<f64>,
    y: Vec<f64>,
    /// Phase 6d.2: native compressed bound multipliers.
    z_l_compressed: Vec<f64>,
    z_u_compressed: Vec<f64>,
    v_l: Vec<f64>,
    v_u: Vec<f64>,
}

impl MuState {
    fn new() -> Self {
        Self {
            mode: MuMode::Free,
            ref_vals: Vec::with_capacity(8),
            num_refs_max: 4,
            refs_red_fact: 0.9999,
            tiny_step: false,
            first_iter_in_mode: true,
            consecutive_restoration_failures: 0,
            consecutive_insufficient: 0,
            consecutive_soft_restoration: 0,
            dual_inf_window: Vec::with_capacity(4),
            initial_avg_compl: None,
            accepted_iterate: None,
        }
    }

    /// Compute the Ipopt `mu_max = mu_max_fact * initial_avg_compl` cap,
    /// capturing `initial_avg_compl` lazily on the first Free-mode oracle
    /// call. Mirrors `IpAdaptiveMuUpdate.cpp:267-273`. Falls back to
    /// `1e10` until first capture (matches Ipopt before the first call).
    fn mu_max_cap(&mut self, options: &SolverOptions, avg_compl: f64) -> f64 {
        if self.initial_avg_compl.is_none() && avg_compl.is_finite() && avg_compl > 0.0 {
            self.initial_avg_compl = Some(avg_compl);
        }
        match self.initial_avg_compl {
            Some(init) => options.mu_max_fact * init,
            None => 1e10,
        }
    }

    /// Check if sufficient progress is being made (KKT error reference check).
    fn check_sufficient_progress(&self, kkt_error: f64) -> bool {
        if self.ref_vals.len() < self.num_refs_max {
            return true; // Not enough history yet
        }
        // Sufficient if current error < refs_red_fact * any reference
        self.ref_vals.iter().any(|&r| kkt_error <= self.refs_red_fact * r)
    }

    /// Remember an accepted KKT error value.
    fn remember_accepted(&mut self, kkt_error: f64) {
        if self.ref_vals.len() >= self.num_refs_max {
            self.ref_vals.remove(0);
        }
        self.ref_vals.push(kkt_error);
    }
}

/// L-BFGS Hessian approximation state for use inside the IPM loop.
///
/// Maintains curvature pairs (s_k, y_k) from Lagrangian gradient differences
/// and forms an explicit dense B_k matrix for the KKT system.
pub struct LbfgsIpmState {
    /// Number of primal variables.
    n: usize,
    /// Maximum number of stored pairs.
    m_max: usize,
    /// Stored s_k vectors (x_{k+1} - x_k).
    s_store: Vec<Vec<f64>>,
    /// Stored y_k vectors (∇L_{k+1} - ∇L_k, evaluated at new multipliers).
    y_store: Vec<Vec<f64>>,
    /// Previous iterate x_k.
    prev_x: Vec<f64>,
    /// Previous Lagrangian gradient ∇_x L(x_k, λ_{k+1}).
    prev_lag_grad: Vec<f64>,
    /// Whether we have a previous iterate (skip update on first call).
    has_prev: bool,
    /// Initial Hessian scaling factor gamma (H0 = gamma * I).
    gamma: f64,
}

impl LbfgsIpmState {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            m_max: 10,
            s_store: Vec::with_capacity(10),
            y_store: Vec::with_capacity(10),
            prev_x: vec![0.0; n],
            prev_lag_grad: vec![0.0; n],
            has_prev: false,
            gamma: 1.0,
        }
    }

    /// Compute ∇_x L = ∇f + Jac_c^T y_c + Jac_d^T y_d. Native split form.
    fn compute_lagrangian_gradient(
        grad_f: &[f64],
        jac_c_rows: &[usize],
        jac_c_cols: &[usize],
        jac_c_vals: &[f64],
        jac_d_rows: &[usize],
        jac_d_cols: &[usize],
        jac_d_vals: &[f64],
        y_c: &[f64],
        y_d: &[f64],
        n: usize,
    ) -> Vec<f64> {
        let mut lag_grad = grad_f.to_vec();
        for (idx, (&kc, &col)) in jac_c_rows.iter().zip(jac_c_cols.iter()).enumerate() {
            if col < n {
                lag_grad[col] += jac_c_vals[idx] * y_c[kc];
            }
        }
        for (idx, (&kd, &col)) in jac_d_rows.iter().zip(jac_d_cols.iter()).enumerate() {
            if col < n {
                lag_grad[col] += jac_d_vals[idx] * y_d[kd];
            }
        }
        lag_grad
    }

    /// Update L-BFGS pairs after a step has been accepted.
    /// Uses Powell damping to ensure positive curvature (s^T y > 0).
    pub fn update(
        &mut self,
        new_x: &[f64],
        new_lag_grad: &[f64],
    ) {
        if !self.has_prev {
            self.prev_x.copy_from_slice(new_x);
            self.prev_lag_grad.copy_from_slice(new_lag_grad);
            self.has_prev = true;
            return;
        }

        let n = self.n;
        let mut s_k = vec![0.0; n];
        let mut y_k = vec![0.0; n];
        for i in 0..n {
            s_k[i] = new_x[i] - self.prev_x[i];
            y_k[i] = new_lag_grad[i] - self.prev_lag_grad[i];
        }

        let ss: f64 = s_k.iter().map(|v| v * v).sum();
        if ss < 1e-30 {
            // Step too small, skip update
            self.prev_x.copy_from_slice(new_x);
            self.prev_lag_grad.copy_from_slice(new_lag_grad);
            return;
        }

        let sy: f64 = dot_product(&s_k, &y_k);

        // Compute B_k * s_k for Powell damping
        let bs = self.multiply_bk(&s_k);
        let sbs: f64 = dot_product(&s_k, &bs);

        // Powell damping: ensure s^T y >= 0.2 * s^T B s
        if sy >= 0.2 * sbs {
            // Use y_k as-is
        } else {
            let theta = if (sbs - sy).abs() < 1e-30 {
                1.0
            } else {
                0.8 * sbs / (sbs - sy)
            };
            for i in 0..n {
                y_k[i] = theta * y_k[i] + (1.0 - theta) * bs[i];
            }
        }

        // Verify positive curvature after damping
        let sy_damped: f64 = dot_product(&s_k, &y_k);
        if sy_damped <= 1e-20 {
            self.prev_x.copy_from_slice(new_x);
            self.prev_lag_grad.copy_from_slice(new_lag_grad);
            return;
        }

        // Update gamma = s^T y / y^T y
        let yy: f64 = y_k.iter().map(|v| v * v).sum();
        if yy > 1e-30 {
            self.gamma = sy_damped / yy;
        }

        // Store pair
        if self.s_store.len() == self.m_max {
            self.s_store.remove(0);
            self.y_store.remove(0);
        }
        self.s_store.push(s_k);
        self.y_store.push(y_k);

        self.prev_x.copy_from_slice(new_x);
        self.prev_lag_grad.copy_from_slice(new_lag_grad);
    }

    /// Compute B_k * v using the L-BFGS compact representation.
    /// B_k = gamma^{-1} I - ... (inverse Hessian formulation, then invert).
    /// Instead, we directly build B_k * v using the recursive formula.
    pub fn multiply_bk(&self, v: &[f64]) -> Vec<f64> {
        let n = self.n;
        let k = self.s_store.len();

        if k == 0 {
            // B_0 = (1/gamma) * I
            let scale = 1.0 / self.gamma.max(1e-12);
            return v.iter().map(|&vi| scale * vi).collect();
        }

        // Use the explicit B_k formation and multiply
        // B_k = (1/gamma) I + sum of rank-2 updates
        // It's easier to just form the full matrix and multiply for correctness
        let mut result = vec![0.0; n];
        let bk = self.form_dense_bk();
        for i in 0..n {
            for j in 0..n {
                let (r, c) = if i >= j { (i, j) } else { (j, i) };
                let idx = r * (r + 1) / 2 + c;
                result[i] += bk[idx] * v[j];
            }
        }
        result
    }

    /// Form explicit dense B_k matrix in lower-triangle format.
    /// Uses the L-BFGS compact representation:
    ///   B_k = B_0 - [B_0 S_k  Y_k] * M^{-1} * [S_k^T B_0; Y_k^T]
    /// where B_0 = (1/gamma) I.
    pub fn form_dense_bk(&self) -> Vec<f64> {
        let n = self.n;
        let k = self.s_store.len();
        let nnz = n * (n + 1) / 2;

        // Start with B_0 = (1/gamma) * I
        let b0_diag = 1.0 / self.gamma.max(1e-12);
        let mut bk = vec![0.0; nnz];
        for i in 0..n {
            let idx = i * (i + 1) / 2 + i;
            bk[idx] = b0_diag;
        }

        if k == 0 {
            return bk;
        }

        // Build B_k iteratively using rank-2 updates (BFGS formula):
        // B_{k+1} = B_k - (B_k s_k s_k^T B_k) / (s_k^T B_k s_k) + (y_k y_k^T) / (s_k^T y_k)
        for p in 0..k {
            let s = &self.s_store[p];
            let y = &self.y_store[p];

            // Compute B_k * s
            let mut bs = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    let (r, c) = if i >= j { (i, j) } else { (j, i) };
                    let idx = r * (r + 1) / 2 + c;
                    bs[i] += bk[idx] * s[j];
                }
            }

            let sbs: f64 = dot_product(s, &bs);
            let sy: f64 = dot_product(s, y);

            if sbs.abs() < 1e-30 || sy.abs() < 1e-30 {
                continue;
            }

            // Rank-2 update: B -= (bs bs^T) / sbs + (y y^T) / sy
            for i in 0..n {
                for j in 0..=i {
                    let idx = i * (i + 1) / 2 + j;
                    bk[idx] += -bs[i] * bs[j] / sbs + y[i] * y[j] / sy;
                }
            }
        }

        bk
    }

    /// Fill the hess_vals buffer with the dense B_k matrix.
    fn fill_hessian(&self, hess_vals: &mut [f64]) {
        let bk = self.form_dense_bk();
        hess_vals[..bk.len()].copy_from_slice(&bk);
    }
}

/// Generate dense lower-triangle sparsity pattern for n variables.
fn dense_lower_triangle_pattern(n: usize) -> (Vec<usize>, Vec<usize>) {
    let nnz = n * (n + 1) / 2;
    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);
    for i in 0..n {
        for j in 0..=i {
            rows.push(i);
            cols.push(j);
        }
    }
    (rows, cols)
}

impl SolverState {
    /// Initialize from an NLP problem.
    fn new<P: NlpProblem>(problem: &P, options: &SolverOptions) -> Self {
        let n = problem.num_variables();
        let m = problem.num_constraints();

        let mut x_l = vec![0.0; n];
        let mut x_u = vec![0.0; n];
        problem.bounds(&mut x_l, &mut x_u);

        let mut g_l = vec![0.0; m];
        let mut g_u = vec![0.0; m];
        problem.constraint_bounds(&mut g_l, &mut g_u);

        // Apply nlp_lower/upper_bound_inf: treat very large bounds as ±infinity.
        // This is the standard NLP convention (Ipopt uses the same option).
        // The GAMS/AMPL links map their "infinity" value to ±1e30 (which is finite
        // in f64). Without this conversion, z_l ≈ mu/1e30 and slack ≈ 1e30, giving
        // slack * z_l ≈ mu ≠ 0 as a spurious complementarity contribution that blocks
        // convergence detection even when the NLP is solved.
        //
        // For x bounds: applied here in the IPM layer (variables are not
        // scaled by ScaledProblem). For g bounds: ScaledProblem applies
        // sentinel + bound_relax_factor in raw space inside its
        // `constraint_bounds`, mirroring Ipopt's relax-then-scale order
        // (IpOrigIpoptNLP.cpp:343-374). See ScaledProblem doc comment.
        sentinel_bounds_to_infinity(&mut x_l, &mut x_u, options);

        // Ipopt's bound_relax_factor for variable bounds: widen every finite
        // bound outward by min(constr_viol_tol, factor·max(|b|,1)). Mirrors
        // IpOrigIpoptNLP.cpp:355-356. Must run AFTER infinity sentinels
        // (so we don't relax 1e30) and BEFORE bound_push / fixed-variable
        // handling. (Constraint-bound counterpart is in ScaledProblem.)
        apply_bound_relax_factor(
            &mut x_l, &mut x_u,
            options.bound_relax_factor, options.constr_viol_tol,
        );

        let mut x = vec![0.0; n];
        problem.initial_point(&mut x);

        relax_fixed_variable_bounds(&mut x_l, &mut x_u, options);

        push_initial_point_from_bounds(&mut x, &x_l, &x_u, options);

        // Initial barrier parameter: warm_start_target_mu overrides
        // mu_init when warm_start is enabled (Ipopt's
        // `warm_start_target_mu`, used to resume parametric/MPC sweeps
        // at the previous solve's final mu without re-centering).
        let initial_mu = match (options.warm_start, options.warm_start_target_mu) {
            (true, Some(mu)) if mu > 0.0 => mu,
            _ => options.mu_init,
        };
        let (mut z_l, mut z_u) = init_bound_multipliers(&x, &x_l, &x_u, initial_mu, options);

        let (jac_rows, jac_cols) = problem.jacobian_structure();
        let jac_nnz = jac_rows.len();
        let (hess_rows, hess_cols) = if options.hessian_approximation_lbfgs {
            dense_lower_triangle_pattern(n)
        } else {
            problem.hessian_structure()
        };
        let hess_nnz = hess_rows.len();

        let mut y = compute_initial_y_with_ls(
            problem, options, &x, &z_l, &z_u, &jac_rows, &jac_cols,
            &x_l, &x_u, &g_l, &g_u, n, m, jac_nnz,
        );

        if options.warm_start {
            apply_warm_start_multipliers(problem, &mut y, &mut z_l, &mut z_u);
        }

        let m_eq = (0..m).filter(|&i| g_l[i] == g_u[i]).count();
        let is_square = m == n || m_eq == n;
        let layout = crate::constraint_layout::ConstraintLayout::new(&g_l, &g_u);
        let bound_layout = crate::bound_layout::BoundLayout::new(&x_l, &x_u);
        // Phase 6b: native compressed bound-multiplier mirrors. At
        // construction, `z_l`/`z_u` already carry zero on unbounded sides,
        // so `project_l`/`project_u` extracts only the active components.
        let z_l_compressed = bound_layout.project_l(&z_l);
        let z_u_compressed = bound_layout.project_u(&z_u);
        // Phase 7a: compressed variable-bound storage (size n_x_l/n_x_u).
        // Mirrors Ipopt's `x_L_`/`x_U_` (`IpOrigIpoptNLP.hpp:226,253`).
        let x_l_compressed = bound_layout.project_l(&x_l);
        let x_u_compressed = bound_layout.project_u(&x_u);
        // Phase 8a/b: slack-bound expansion-matrix layout + compressed
        // multiplier mirrors. Mirrors Ipopt's `Pd_L_`/`Pd_U_` and
        // `v_L`/`v_U` (`IpOrigIpoptNLP.hpp:241-263`).
        let d_bound_layout =
            crate::d_bound_layout::DBoundLayout::new(&layout.project_d(&g_l), &layout.project_d(&g_u));
        let v_l_compressed = vec![0.0; d_bound_layout.n_d_l];
        let v_u_compressed = vec![0.0; d_bound_layout.n_d_u];
        let dv_l_compressed = vec![0.0; d_bound_layout.n_d_l];
        let dv_u_compressed = vec![0.0; d_bound_layout.n_d_u];
        // Phase 9b: compressed slack-bound storage (Ipopt's `d_L_`/`d_U_`).
        let d_l_compressed = d_bound_layout.project_l(&layout.project_d(&g_l));
        let d_u_compressed = d_bound_layout.project_u(&layout.project_d(&g_u));

        // Phase 3b: split y into y_c (n_c) and y_d (n_d). Combined y from
        // the LS init is projected via the layout; consumers needing the
        // combined view reconstruct it via `state.y_combined()`.
        let y_c = layout.project_c(&y);
        let y_d = layout.project_d(&y);

        // Phase 3f: split-form bound storage. c_rhs holds the equality
        // target (= g_l[eq] = g_u[eq]). Inequality bounds live in the
        // compressed `d_l_compressed` / `d_u_compressed` mirrors built
        // above (Phase 9b); the combined `d_l`/`d_u` storage was dropped
        // in Phase 9d.
        let c_rhs: Vec<f64> = layout.c_to_combined.iter().map(|&i| g_l[i]).collect();

        // Phase 4a: build the split Jacobian structure by walking the
        // combined triplet once and routing each entry to the c-block
        // (eq row) or d-block (ineq row) using the layout's row maps.
        // The combined-index map captures the per-split-entry source
        // position so values can be copied without re-checking the
        // partition each evaluate.
        let mut jac_c_rows: Vec<usize> = Vec::new();
        let mut jac_c_cols: Vec<usize> = Vec::new();
        let mut jac_c_combined_idx: Vec<usize> = Vec::new();
        let mut jac_d_rows: Vec<usize> = Vec::new();
        let mut jac_d_cols: Vec<usize> = Vec::new();
        let mut jac_d_combined_idx: Vec<usize> = Vec::new();
        for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
            if let Some(k_c) = layout.eq_pos[row] {
                jac_c_rows.push(k_c);
                jac_c_cols.push(col);
                jac_c_combined_idx.push(idx);
            } else if let Some(k_d) = layout.ineq_pos[row] {
                jac_d_rows.push(k_d);
                jac_d_cols.push(col);
                jac_d_combined_idx.push(idx);
            }
        }
        let jac_c_vals = vec![0.0; jac_c_rows.len()];
        let jac_d_vals = vec![0.0; jac_d_rows.len()];

        Self {
            x,
            y_c,
            y_d,
            dx: vec![0.0; n],
            dy_c: vec![0.0; layout.n_c],
            dy_d: vec![0.0; layout.n_d],
            // Slack iterate `s` and step `ds` (size n_d). At construction zeroed;
            // the proper push-to-interior init runs in `initialize_slack_iterate`
            // (B1.2). Phase 3d: native d-block sizing, no equality-row sentinels.
            s: vec![0.0; layout.n_d],
            ds: vec![0.0; layout.n_d],

            mu: initial_mu,
            alpha_primal: 0.0,
            alpha_dual: 0.0,
            iter: 0,
            c_rhs,
            n,
            m,
            obj: 0.0,
            grad_f: vec![0.0; n],
            c_x: vec![0.0; layout.n_c],
            d_x: vec![0.0; layout.n_d],
            jac_rows,
            jac_cols,
            jac_c_rows,
            jac_c_cols,
            jac_c_vals,
            jac_d_rows,
            jac_d_cols,
            jac_d_vals,
            jac_c_combined_idx,
            jac_d_combined_idx,
            hess_rows,
            hess_cols,
            hess_vals: vec![0.0; hess_nnz],
            consecutive_acceptable: 0,
            obj_scaling: 1.0,
            c_scaling: vec![1.0; layout.n_c],
            d_scaling: vec![1.0; layout.n_d],
            layout,
            z_l_compressed,
            z_u_compressed,
            dz_l_compressed: vec![0.0; bound_layout.n_x_l],
            dz_u_compressed: vec![0.0; bound_layout.n_x_u],
            x_l_compressed,
            x_u_compressed,
            bound_layout,
            d_bound_layout,
            v_l_compressed,
            v_u_compressed,
            dv_l_compressed,
            dv_u_compressed,
            d_l_compressed,
            d_u_compressed,
            diagnostics: SolverDiagnostics::default(),
            x_last_eval: vec![f64::NAN; n],
            adjusted_slacks_count: 0,
            is_square,
            acceptable_iterate: None,
            last_obj_for_acceptable: None,
            kkt_atags: kkt::KktInputAtags::default(),
            factor_cache: {
                let mut c = kkt::FactorCache::new();
                c.enabled = options.factor_cache_enabled;
                c
            },
            n_obj_evals: 0,
            n_grad_evals: 0,
            n_constr_evals: 0,
            n_jac_evals: 0,
            n_hess_evals: 0,
        }
    }

    // -------------------------------------------------------------------
    // Phase 1 split-layout accessors. Backed by combined storage today;
    // Phase 3 will swap the backing fields and these become trivial
    // pass-throughs. See docs/V0.8_DATA_LAYOUT_REFACTOR_PLAN.md.
    //
    // Naming mirrors Ipopt 3.14:
    //   c-block  ↔  equality constraints (size n_c)
    //   d-block  ↔  inequality constraints (size n_d), the only block
    //              with bounds and slacks
    // -------------------------------------------------------------------

    /// Equality-constraint residual `c(x) := g[i] - g_l[i]` projected onto
    /// the c-block (size n_c). At feasibility `c(x) = 0`. Mirrors Ipopt's
    /// `OrigIpoptNLP::c()` (`IpIpoptNLP.hpp:117`), where the equality
    /// target is baked into the residual at TNLPAdapter level
    /// (`IpTNLPAdapter.cpp:567-570`). Note the asymmetry with `g_d()`,
    /// which returns the raw inequality value (not a residual) because
    /// inequalities have a band, not a point target.
    pub fn g_c(&self) -> Vec<f64> {
        self.c_x.clone()
    }

    /// Inequality constraint values `d(x)` (size n_d). Mirrors Ipopt's
    /// `OrigIpoptNLP::d()` (`IpIpoptNLP.hpp:118`). At feasibility
    /// `d_L ≤ d(x) ≤ d_U`.
    pub fn g_d(&self) -> Vec<f64> {
        self.d_x.clone()
    }

    /// Inequality lower bounds `d_L` (size n_d, may contain -inf).
    /// Mirrors Ipopt's `OrigIpoptNLP::d_L()` (`IpIpoptNLP.hpp:153`).
    /// Phase 9c: expand from compressed storage (`f64::NEG_INFINITY`
    /// pad for d-rows without a finite lower bound).
    pub fn d_l(&self) -> Vec<f64> {
        self.d_bound_layout
            .expand_l(&self.d_l_compressed, f64::NEG_INFINITY)
    }

    /// Inequality upper bounds `d_U` (size n_d, may contain +inf).
    /// Mirrors Ipopt's `OrigIpoptNLP::d_U()` (`IpIpoptNLP.hpp:155`).
    /// Phase 9c: expand from compressed storage.
    pub fn d_u(&self) -> Vec<f64> {
        self.d_bound_layout
            .expand_u(&self.d_u_compressed, f64::INFINITY)
    }

    /// Inequality lower bound at d-block index `k`. Returns
    /// `f64::NEG_INFINITY` when the d-row has no finite lower bound.
    /// Phase 9c: routes through compressed storage.
    pub fn d_l_at(&self, k: usize) -> f64 {
        match self.d_bound_layout.full_to_d_l[k] {
            Some(kc) => self.d_l_compressed[kc],
            None => f64::NEG_INFINITY,
        }
    }

    /// Inequality upper bound at d-block index `k`. Returns
    /// `f64::INFINITY` when the d-row has no finite upper bound.
    /// Phase 9c: routes through compressed storage.
    pub fn d_u_at(&self, k: usize) -> f64 {
        match self.d_bound_layout.full_to_d_u[k] {
            Some(kc) => self.d_u_compressed[kc],
            None => f64::INFINITY,
        }
    }

    /// Combined-indexed read of the constraint lower bound. For eq rows
    /// returns `c_rhs[k]`; for ineq rows returns `d_l_at(k)`. Phase 9c:
    /// d-block read routes through compressed.
    pub fn g_l_at(&self, i: usize) -> f64 {
        if let Some(k) = self.layout.eq_pos[i] {
            self.c_rhs[k]
        } else if let Some(k) = self.layout.ineq_pos[i] {
            self.d_l_at(k)
        } else {
            unreachable!("constraint row {} is neither eq nor ineq", i)
        }
    }

    /// Combined-indexed read of the constraint upper bound. For eq rows
    /// returns `c_rhs[k]`; for ineq rows returns `d_u_at(k)`.
    pub fn g_u_at(&self, i: usize) -> f64 {
        if let Some(k) = self.layout.eq_pos[i] {
            self.c_rhs[k]
        } else if let Some(k) = self.layout.ineq_pos[i] {
            self.d_u_at(k)
        } else {
            unreachable!("constraint row {} is neither eq nor ineq", i)
        }
    }

    /// Materialise the m-form combined `g_l` for callers (kkt assembly,
    /// convergence helpers) that still take an m-length slice.
    pub fn g_l_combined(&self) -> Vec<f64> {
        (0..self.m).map(|i| self.g_l_at(i)).collect()
    }

    /// Materialise the m-form combined `g_u`.
    pub fn g_u_combined(&self) -> Vec<f64> {
        (0..self.m).map(|i| self.g_u_at(i)).collect()
    }

    /// Combined-indexed write of the constraint lower bound. Routes to
    /// `c_rhs` for eq rows or `d_l_compressed` for ineq rows with a
    /// finite lower bound. Phase 9d: writes the compressed mirror
    /// directly. The only runtime mutation site (`apply_slack_move`)
    /// only nudges already-finite bounds, so the layout is invariant.
    pub fn set_g_l_at(&mut self, i: usize, v: f64) {
        if let Some(k) = self.layout.eq_pos[i] {
            self.c_rhs[k] = v;
        } else if let Some(k) = self.layout.ineq_pos[i] {
            if let Some(kc) = self.d_bound_layout.full_to_d_l[k] {
                self.d_l_compressed[kc] = v;
            }
        } else {
            unreachable!("constraint row {} is neither eq nor ineq", i);
        }
    }

    /// Combined-indexed write of the constraint upper bound.
    pub fn set_g_u_at(&mut self, i: usize, v: f64) {
        if let Some(k) = self.layout.eq_pos[i] {
            self.c_rhs[k] = v;
        } else if let Some(k) = self.layout.ineq_pos[i] {
            if let Some(kc) = self.d_bound_layout.full_to_d_u[k] {
                self.d_u_compressed[kc] = v;
            }
        } else {
            unreachable!("constraint row {} is neither eq nor ineq", i);
        }
    }

    /// Reconstruct the combined m-length multiplier vector from the split
    /// storage. Allocates per call — for hot-path code, prefer reading
    /// `state.y_c` / `state.y_d` directly via the layout. User-facing
    /// (TNLPAdapter-style) sites and leaf-module signatures still take a
    /// combined `&[f64]`, so this helper bridges them.
    pub fn y_combined(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.m];
        for (k, &i) in self.layout.c_to_combined.iter().enumerate() {
            out[i] = self.y_c[k];
        }
        for (k, &i) in self.layout.d_to_combined.iter().enumerate() {
            out[i] = self.y_d[k];
        }
        out
    }

    /// Read the combined-indexed multiplier `y[i]` by routing through the
    /// layout (`y_c[k]` for equality rows, `y_d[k]` for inequality rows).
    pub fn y_at(&self, i: usize) -> f64 {
        if let Some(k) = self.layout.eq_pos[i] {
            self.y_c[k]
        } else {
            let k = self.layout.ineq_pos[i].expect("row is c or d");
            self.y_d[k]
        }
    }

    /// Set the combined-indexed multiplier `y[i]` by routing through the
    /// layout into the corresponding split storage slot.
    pub fn set_y_at(&mut self, i: usize, v: f64) {
        if let Some(k) = self.layout.eq_pos[i] {
            self.y_c[k] = v;
        } else {
            let k = self.layout.ineq_pos[i].expect("row is c or d");
            self.y_d[k] = v;
        }
    }

    /// Overwrite the multipliers from a combined m-length slice. Splits
    /// the input across y_c / y_d via the layout. Used by initial-y
    /// least-squares, dual recompute, and snapshot restore.
    pub fn set_y_combined(&mut self, y: &[f64]) {
        debug_assert_eq!(y.len(), self.m);
        for (k, &i) in self.layout.c_to_combined.iter().enumerate() {
            self.y_c[k] = y[i];
        }
        for (k, &i) in self.layout.d_to_combined.iter().enumerate() {
            self.y_d[k] = y[i];
        }
    }

    /// Slack iterate `s` (size n_d). Phase 3d flipped storage: this is now
    /// a direct clone of `self.s`. In Ipopt `s` has dimension `n_d`
    /// natively (`IpIpoptData.cpp:140`).
    pub fn s_d(&self) -> Vec<f64> {
        self.s.clone()
    }

    /// Reconstruct an m-length combined-indexed slack view. Equality rows
    /// get the sentinel `g_l[i]`; inequality rows get the real `s_d[k]`.
    /// Used by snapshots and diagnostics that still expect combined-indexed
    /// slack data. Allocates per call.
    pub fn s_combined(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.m];
        for i in 0..self.m {
            if let Some(k) = self.layout.ineq_pos[i] {
                out[i] = self.s[k];
            } else if let Some(k) = self.layout.eq_pos[i] {
                out[i] = self.c_rhs[k];
            }
        }
        out
    }

    /// Reconstruct an m-length combined-indexed g(x) view from split storage.
    /// Equality rows get `c_x[k] + c_rhs[k]` (the original constraint value);
    /// inequality rows get `d_x[k]`. Used by user-facing intermediate
    /// callbacks and diagnostics that still expect combined-indexed g.
    pub fn g_combined(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.m];
        for (k, &i) in self.layout.c_to_combined.iter().enumerate() {
            out[i] = self.c_x[k] + self.c_rhs[k];
        }
        for (k, &i) in self.layout.d_to_combined.iter().enumerate() {
            out[i] = self.d_x[k];
        }
        out
    }

    /// Combined-indexed slack read: returns `s_d[k]` for inequality rows
    /// and `c_rhs[k]` (equality target sentinel) for equality rows.
    pub fn s_at(&self, i: usize) -> f64 {
        if let Some(k) = self.layout.ineq_pos[i] {
            self.s[k]
        } else if let Some(k) = self.layout.eq_pos[i] {
            self.c_rhs[k]
        } else {
            unreachable!("constraint row {} is neither eq nor ineq", i)
        }
    }

    /// Combined-indexed slack write: stores into `s_d[k]` for inequality
    /// rows; no-op for equality rows (the sentinel is implicit, not stored).
    pub fn set_s_at(&mut self, i: usize, v: f64) {
        if let Some(k) = self.layout.ineq_pos[i] {
            self.s[k] = v;
        }
    }

    /// Reconstruct an m-length combined-indexed `ds` view. Equality rows
    /// get 0 (no slack step); inequality rows get the real `ds[k]`.
    pub fn ds_combined(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.m];
        for (k, &i) in self.layout.d_to_combined.iter().enumerate() {
            out[i] = self.ds[k];
        }
        out
    }

    /// Combined-indexed slack-step read: `ds_d[k]` for ineq rows, 0 for eq.
    pub fn ds_at(&self, i: usize) -> f64 {
        if let Some(k) = self.layout.ineq_pos[i] {
            self.ds[k]
        } else {
            0.0
        }
    }

    /// Inequality-block slack-bound multipliers `v_L` (size n_d).
    /// Phase 8d: combined `v_l` storage dropped; expand from compressed.
    pub fn v_l_d(&self) -> Vec<f64> {
        self.d_bound_layout.expand_l(&self.v_l_compressed, 0.0)
    }

    /// Inequality-block slack-bound multipliers `v_U` (size n_d).
    /// Phase 8d: combined `v_u` storage dropped; expand from compressed.
    pub fn v_u_d(&self) -> Vec<f64> {
        self.d_bound_layout.expand_u(&self.v_u_compressed, 0.0)
    }

    /// Reconstruct combined m-length `v_L` view (zero on equality rows
    /// and on inequality rows without a finite lower slack bound).
    /// Phase 8c: routes through the compressed mirror via
    /// `d_bound_layout.full_to_d_l`.
    pub fn v_l_combined(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.m];
        for (k, &i) in self.layout.d_to_combined.iter().enumerate() {
            if let Some(kc) = self.d_bound_layout.full_to_d_l[k] {
                out[i] = self.v_l_compressed[kc];
            }
        }
        out
    }

    /// Reconstruct combined m-length `v_U` view.
    pub fn v_u_combined(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.m];
        for (k, &i) in self.layout.d_to_combined.iter().enumerate() {
            if let Some(kc) = self.d_bound_layout.full_to_d_u[k] {
                out[i] = self.v_u_compressed[kc];
            }
        }
        out
    }

    /// Combined-indexed read: `v_l[k]` for ineq rows with finite lower
    /// slack bound, 0 otherwise. Phase 8c: routes through the compressed
    /// mirror.
    pub fn v_l_at(&self, i: usize) -> f64 {
        match self.layout.ineq_pos[i] {
            Some(k) => match self.d_bound_layout.full_to_d_l[k] {
                Some(kc) => self.v_l_compressed[kc],
                None => 0.0,
            },
            None => 0.0,
        }
    }

    /// Combined-indexed read: `v_u[k]` for ineq rows with finite upper
    /// slack bound, 0 otherwise. Phase 8c: compressed-routed.
    pub fn v_u_at(&self, i: usize) -> f64 {
        match self.layout.ineq_pos[i] {
            Some(k) => match self.d_bound_layout.full_to_d_u[k] {
                Some(kc) => self.v_u_compressed[kc],
                None => 0.0,
            },
            None => 0.0,
        }
    }

    /// Combined-indexed write into `v_L`; no-op for equality rows or
    /// d-rows without a finite lower slack bound. Phase 8d: writes
    /// straight into the compressed mirror.
    pub fn set_v_l_at(&mut self, i: usize, v: f64) {
        if let Some(k) = self.layout.ineq_pos[i] {
            if let Some(kc) = self.d_bound_layout.full_to_d_l[k] {
                self.v_l_compressed[kc] = v;
            }
        }
    }

    /// Combined-indexed write into `v_U`; no-op for equality rows or
    /// d-rows without a finite upper slack bound.
    pub fn set_v_u_at(&mut self, i: usize, v: f64) {
        if let Some(k) = self.layout.ineq_pos[i] {
            if let Some(kc) = self.d_bound_layout.full_to_d_u[k] {
                self.v_u_compressed[kc] = v;
            }
        }
    }

    /// Overwrite v_L from a combined m-length slice (drops eq rows).
    /// Phase 8d: rebuild the compressed mirror directly via Pd_L^T.
    pub fn set_v_l_combined(&mut self, v: &[f64]) {
        debug_assert_eq!(v.len(), self.m);
        let v_l_d: Vec<f64> = self
            .layout
            .d_to_combined
            .iter()
            .map(|&i| v[i])
            .collect();
        self.v_l_compressed = self.d_bound_layout.project_l(&v_l_d);
    }

    /// Overwrite v_U from a combined m-length slice (drops eq rows).
    pub fn set_v_u_combined(&mut self, v: &[f64]) {
        debug_assert_eq!(v.len(), self.m);
        let v_u_d: Vec<f64> = self
            .layout
            .d_to_combined
            .iter()
            .map(|&i| v[i])
            .collect();
        self.v_u_compressed = self.d_bound_layout.project_u(&v_u_d);
    }

    /// Overwrite the search direction from a combined m-length slice.
    /// Splits across `dy_c` / `dy_d` via the layout. Used by the KKT
    /// solver, gradient-descent fallback, and Gondzio correctors.
    #[cfg(test)]
    pub fn set_dy_combined(&mut self, dy: &[f64]) {
        debug_assert_eq!(dy.len(), self.m);
        for (k, &i) in self.layout.c_to_combined.iter().enumerate() {
            self.dy_c[k] = dy[i];
        }
        for (k, &i) in self.layout.d_to_combined.iter().enumerate() {
            self.dy_d[k] = dy[i];
        }
    }

    /// Phase 4a: split a combined m-form jacobian-values vector into the
    /// c-block and d-block native storage. Used by test fixtures that
    /// hand-craft a combined-form Jacobian.
    #[cfg(test)]
    pub fn set_jac_vals_combined(&mut self, vals: &[f64]) {
        for (k, &idx) in self.jac_c_combined_idx.iter().enumerate() {
            self.jac_c_vals[k] = vals[idx];
        }
        for (k, &idx) in self.jac_d_combined_idx.iter().enumerate() {
            self.jac_d_vals[k] = vals[idx];
        }
    }

    /// Split a combined m-form constraint vector into the c-block and
    /// d-block native storage. `c_x[k] = g[c_to_combined[k]] - c_rhs[k]`
    /// (Ipopt's `c(x)`, equality residual baked at TNLPAdapter level —
    /// `IpTNLPAdapter.cpp:567-570`) and `d_x[k] = g[d_to_combined[k]]`
    /// (Ipopt's raw `d(x)` value). Used by test fixtures that hand-craft
    /// a combined-form constraint vector.
    #[cfg(test)]
    pub fn set_g_combined(&mut self, g: &[f64]) {
        for (k, &i) in self.layout.c_to_combined.iter().enumerate() {
            self.c_x[k] = g[i] - self.c_rhs[k];
        }
        for (k, &i) in self.layout.d_to_combined.iter().enumerate() {
            self.d_x[k] = g[i];
        }
    }

    /// Reconstruct the combined m-length step vector from the split
    /// storage. Allocates per call.
    pub fn dy_combined(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.m];
        for (k, &i) in self.layout.c_to_combined.iter().enumerate() {
            out[i] = self.dy_c[k];
        }
        for (k, &i) in self.layout.d_to_combined.iter().enumerate() {
            out[i] = self.dy_d[k];
        }
        out
    }

    /// Read the combined-indexed step `dy[i]` by routing through the layout.
    pub fn dy_at(&self, i: usize) -> f64 {
        if let Some(k) = self.layout.eq_pos[i] {
            self.dy_c[k]
        } else {
            let k = self.layout.ineq_pos[i].expect("row is c or d");
            self.dy_d[k]
        }
    }

    /// Phase 6d.1: materialize a full-`n` view of `z_l` from the
    /// compressed mirror. Unbounded sides pad to 0 (the production
    /// invariant). Used at API boundaries (`unscale_solution_vectors`)
    /// and for kkt_aug callsites that still consume full-`n` slices.
    pub fn z_l_combined(&self) -> Vec<f64> {
        self.bound_layout.expand_l(&self.z_l_compressed, 0.0)
    }

    /// Phase 6d.1: materialize a full-`n` view of `z_u` from the
    /// compressed mirror. Unbounded sides pad to 0.
    pub fn z_u_combined(&self) -> Vec<f64> {
        self.bound_layout.expand_u(&self.z_u_compressed, 0.0)
    }

    /// Phase 6d.1: materialize a full-`n` view of `dz_l` from the
    /// compressed mirror. Unbounded sides pad to 0.
    pub fn dz_l_combined(&self) -> Vec<f64> {
        self.bound_layout.expand_l(&self.dz_l_compressed, 0.0)
    }

    /// Phase 6d.1: materialize a full-`n` view of `dz_u` from the
    /// compressed mirror. Unbounded sides pad to 0.
    pub fn dz_u_combined(&self) -> Vec<f64> {
        self.bound_layout.expand_u(&self.dz_u_compressed, 0.0)
    }

    /// Phase 7b: full-`n` view of the lower variable bound. Reads from
    /// the compressed mirror via `bound_layout.full_to_x_l[i]`; unbounded
    /// indices return `f64::NEG_INFINITY`. Bit-identical to the legacy
    /// `state.x_l_at(i)` read under the production invariant.
    pub fn x_l_at(&self, i: usize) -> f64 {
        match self.bound_layout.full_to_x_l[i] {
            Some(k) => self.x_l_compressed[k],
            None => f64::NEG_INFINITY,
        }
    }

    /// Phase 7b: full-`n` view of the upper variable bound. Reads from
    /// the compressed mirror via `bound_layout.full_to_x_u[i]`; unbounded
    /// indices return `f64::INFINITY`.
    pub fn x_u_at(&self, i: usize) -> f64 {
        match self.bound_layout.full_to_x_u[i] {
            Some(k) => self.x_u_compressed[k],
            None => f64::INFINITY,
        }
    }

    /// Phase 7b: materialize a full-`n` view of the lower variable bound
    /// from the compressed mirror. Unbounded sides pad to
    /// `f64::NEG_INFINITY`. Used for API boundaries that still expect
    /// the legacy `&[f64]` slice with `±inf` sentinels.
    pub fn x_l_combined(&self) -> Vec<f64> {
        self.bound_layout.expand_l(&self.x_l_compressed, f64::NEG_INFINITY)
    }

    /// Phase 7b: materialize a full-`n` view of the upper variable bound
    /// from the compressed mirror. Unbounded sides pad to `f64::INFINITY`.
    pub fn x_u_combined(&self) -> Vec<f64> {
        self.bound_layout.expand_u(&self.x_u_compressed, f64::INFINITY)
    }

    /// Evaluate all functions, zeroing Hessian lambda for linear constraints.
    /// When `skip_hessian` is true (L-BFGS mode), the Hessian evaluation is skipped.
    fn evaluate_with_linear<P: NlpProblem>(
        &mut self,
        problem: &P,
        obj_factor: f64,
        linear_constraints: Option<&[bool]>,
        skip_hessian: bool,
    ) -> bool {
        let new_x = self.x != self.x_last_eval;
        // Phase 10b.1: route through the SplitNlp adapter. The combined
        // m-form `g(x)` and `jac_g` live only in the adapter's internal
        // scratch; this function sees only split-form `c(x)` / `d(x)`
        // and `Jac_c` / `Jac_d`.
        let nlp = crate::split_nlp::SplitNlp::new(problem, &self.layout);
        self.n_obj_evals += 1;
        if !nlp.objective(&self.x, new_x, &mut self.obj) { return false; }
        if !self.obj.is_finite() { return false; }
        self.n_grad_evals += 1;
        if !nlp.gradient(&self.x, false, &mut self.grad_f) { return false; }
        // NB: 42f4015 added element-wise is_finite checks on grad_f and g
        // here. Benchmarking showed they caused regressions on CUTEst
        // problems (BIGGS6NE, CERI651*, HS84/89/92, MAKELA3, OPTCNTRL, ...)
        // where a transient NaN/Inf in an individual grad or constraint
        // element at a boundary previously propagated but did not prevent
        // convergence. Matches Ipopt: IpOrigIpoptNLP only checks Nrm2 of
        // grad_f and its default `check_derivatives_for_naninf = false`
        // means it does NOT check Jacobian element-wise either. Keep only
        // the obj finite check (essential for convergence checks) and the
        // user callback bool return.
        if self.m > 0 {
            self.n_constr_evals += 1;
            // Adapter projects user `g(x)` into split `c_raw` and `d`.
            // We then subtract `c_rhs` to land the equality residual.
            if !nlp.constraints_split(&self.x, false, &mut self.c_x, &mut self.d_x) {
                return false;
            }
            for k in 0..self.layout.n_c {
                self.c_x[k] -= self.c_rhs[k];
            }
            self.n_jac_evals += 1;
            if !nlp.jacobian_split(
                &self.x,
                false,
                &self.jac_c_combined_idx,
                &self.jac_d_combined_idx,
                &mut self.jac_c_vals,
                &mut self.jac_d_vals,
            ) {
                return false;
            }
        }
        self.x_last_eval.copy_from_slice(&self.x);
        if skip_hessian {
            return true;
        }
        self.n_hess_evals += 1;
        if let Some(flags) = linear_constraints {
            let mut lambda_for_hess = self.y_combined();
            for (i, &is_lin) in flags.iter().enumerate() {
                if is_lin {
                    lambda_for_hess[i] = 0.0;
                }
            }
            if !nlp.hessian_combined(&self.x, false, obj_factor, &lambda_for_hess, &mut self.hess_vals) { return false; }
        } else {
            // Compose the m-form `lambda` from split `y_c` / `y_d`
            // inside the adapter rather than materializing via
            // `y_combined()` first.
            if !nlp.hessian_from_split(&self.x, false, obj_factor, &self.y_c, &self.y_d, &mut self.hess_vals) { return false; }
        }
        true
    }

    /// Compute the barrier objective:
    /// f(x) - mu * sum(ln(x_i - x_l_i) + ln(x_u_i - x_i))
    /// Optionally includes constraint slack log-barriers when enabled.
    fn barrier_objective(&self, options: &SolverOptions) -> f64 {
        compute_barrier_phi(
            self.obj, &self.x, &self.s, self,
            self.n, self.m, options.constraint_slack_barrier,
            options.kappa_d,
        )
    }

    /// Compute constraint violation (theta).
    fn constraint_violation(&self) -> f64 {
        convergence::primal_infeasibility_split(
            &self.g_c(),
            &self.g_d(),
            &self.d_l(),
            &self.d_u(),
        )
    }

    /// Compute the directional derivative of the barrier objective along the search direction.
    ///
    /// ∇φ·dx = (∇f - μ/(x-x_l) + μ/(x_u-x))·dx
    /// Optionally includes constraint slack derivative terms when enabled.
    fn barrier_directional_derivative(&self, options: &SolverOptions) -> f64 {
        let mut grad_phi_dx = 0.0;
        let kappa_d = options.kappa_d;
        // RIPOPT_GBD_PROBE: split gBD into ∇f·dx, x-bound barrier · dx,
        // and (downstream) s-bound barrier · ds so we can pin down which
        // component diverges from Ipopt.
        let probe = std::env::var("RIPOPT_GBD_PROBE").is_ok() && self.iter <= 112;
        // RIPOPT_TRACK_VAR=533: print z_L, z_U, slacks, dx, dz for one
        // specific variable across iters.
        if let Ok(s) = std::env::var("RIPOPT_TRACK_VAR") {
            if let Ok(idx) = s.parse::<usize>() {
                if idx < self.n {
                    let z_l_full = self.bound_layout.expand_l(&self.z_l_compressed, 0.0);
                    let z_u_full = self.bound_layout.expand_u(&self.z_u_compressed, 0.0);
                    let dz_l_full = self.bound_layout.expand_l(&self.dz_l_compressed, 0.0);
                    let dz_u_full = self.bound_layout.expand_u(&self.dz_u_compressed, 0.0);
                    let xli = self.x_l_at(idx);
                    let xui = self.x_u_at(idx);
                    let s_l = if xli.is_finite() { self.x[idx] - xli } else { f64::NAN };
                    let s_u = if xui.is_finite() { xui - self.x[idx] } else { f64::NAN };
                    eprintln!(
                        "[track i={}] iter={} mu={:.3e} x={:.6e} dx={:.6e} s_L={:.3e} s_U={:.3e} z_L={:.3e} z_U={:.3e} dz_L={:.3e} dz_U={:.3e} z_L*s_L={:.3e} z_U*s_U={:.3e}",
                        idx, self.iter, self.mu, self.x[idx], self.dx[idx], s_l, s_u,
                        z_l_full[idx], z_u_full[idx], dz_l_full[idx], dz_u_full[idx],
                        z_l_full[idx]*s_l, z_u_full[idx]*s_u,
                    );
                }
            }
        }
        let mut g_f_dx = 0.0;
        let mut g_xb_dx = 0.0;
        let mut g_kd_dx = 0.0;
        for i in 0..self.n {
            let l_fin = self.x_l_at(i).is_finite();
            let u_fin = self.x_u_at(i).is_finite();
            let mut grad_phi_i = self.grad_f[i];
            if probe { g_f_dx += self.grad_f[i] * self.dx[i]; }
            if l_fin {
                let term = -self.mu / slack_xl(self, i);
                grad_phi_i += term;
                if probe { g_xb_dx += term * self.dx[i]; }
            }
            if u_fin {
                let term = self.mu / slack_xu(self, i);
                grad_phi_i += term;
                if probe { g_xb_dx += term * self.dx[i]; }
            }
            // kappa_d damping gradient: +kappa_d*mu if only x_l finite
            // (slack = x - x_l), -kappa_d*mu if only x_u finite
            // (slack = x_u - x).
            if kappa_d > 0.0 && (l_fin ^ u_fin) {
                let term = if l_fin { kappa_d * self.mu } else { -kappa_d * self.mu };
                grad_phi_i += term;
                if probe { g_kd_dx += term * self.dx[i]; }
            }
            grad_phi_dx += grad_phi_i * self.dx[i];
        }
        let mut g_sb_ds = 0.0;
        if options.constraint_slack_barrier && self.layout.n_d > 0 {
            // Ipopt CalcBarrierTermGradS: barrier on the explicit slack
            // variable s (not on d(x)), so the directional derivative is
            // (-μ/(s-d_l) + μ/(d_u-s)) · ds, where ds is the slack step.
            for k in 0..self.layout.n_d {
                let dl = self.d_l_at(k);
                let du = self.d_u_at(k);
                if dl.is_finite() {
                    let slack = self.s[k] - dl;
                    if slack > self.mu * 1e-2 {
                        let term = -self.mu * self.ds[k] / slack;
                        grad_phi_dx += term;
                        if probe { g_sb_ds += term; }
                    }
                }
                if du.is_finite() {
                    let slack = du - self.s[k];
                    if slack > self.mu * 1e-2 {
                        let term = self.mu * self.ds[k] / slack;
                        grad_phi_dx += term;
                        if probe { g_sb_ds += term; }
                    }
                }
            }
        }
        if probe {
            let dx_inf = self.dx.iter().fold(0.0_f64, |a,&b| a.max(b.abs()));
            let ds_inf = self.ds.iter().fold(0.0_f64, |a,&b| a.max(b.abs()));
            let gradf_inf = self.grad_f.iter().fold(0.0_f64, |a,&b| a.max(b.abs()));
            let mut s_minus_d_inf = 0.0_f64;
            for k in 0..self.layout.n_d {
                let v = (self.s[k] - self.d_x[k]).abs();
                if v > s_minus_d_inf { s_minus_d_inf = v; }
            }
            // Identify the dominant (xb)·dx contributor.
            let mut top_i: usize = usize::MAX;
            let mut top_term: f64 = 0.0;
            let mut top_slack: f64 = 0.0;
            let mut top_side: &'static str = "";
            for i in 0..self.n {
                let l_fin = self.x_l_at(i).is_finite();
                let u_fin = self.x_u_at(i).is_finite();
                if l_fin {
                    let s = slack_xl(self, i);
                    let term = -self.mu / s * self.dx[i];
                    if term.abs() > top_term.abs() {
                        top_term = term; top_slack = s; top_i = i; top_side = "L";
                    }
                }
                if u_fin {
                    let s = slack_xu(self, i);
                    let term = self.mu / s * self.dx[i];
                    if term.abs() > top_term.abs() {
                        top_term = term; top_slack = s; top_i = i; top_side = "U";
                    }
                }
            }
            let xi = if top_i < self.n { self.x[top_i] } else { f64::NAN };
            let dxi = if top_i < self.n { self.dx[top_i] } else { f64::NAN };
            let xli = if top_i < self.n { self.x_l_at(top_i) } else { f64::NAN };
            let xui = if top_i < self.n { self.x_u_at(top_i) } else { f64::NAN };
            eprintln!(
                "[gBD] iter={} mu={:.3e} ∇f·dx={:.6e} (xb)·dx={:.6e} (kd)·dx={:.6e} (sb)·ds={:.6e} TOTAL={:.6e}  |dx|={:.3e} |ds|={:.3e} |∇f|={:.3e} |s-d_x|={:.3e}",
                self.iter, self.mu, g_f_dx, g_xb_dx, g_kd_dx, g_sb_ds, grad_phi_dx,
                dx_inf, ds_inf, gradf_inf, s_minus_d_inf,
            );
            eprintln!(
                "[gBD-top] iter={} top_i={} side={} term={:.6e} slack={:.6e} x={:.6e} dx={:.6e} x_l={:.6e} x_u={:.6e}",
                self.iter, top_i, top_side, top_term, top_slack, xi, dxi, xli, xui,
            );
            // Bound-mult / complementarity diagnostic for the offending variable.
            if top_i < self.n {
                let z_l_full = self.bound_layout.expand_l(&self.z_l_compressed, 0.0);
                let z_u_full = self.bound_layout.expand_u(&self.z_u_compressed, 0.0);
                let dz_l_full = self.bound_layout.expand_l(&self.dz_l_compressed, 0.0);
                let dz_u_full = self.bound_layout.expand_u(&self.dz_u_compressed, 0.0);
                let z_l = z_l_full[top_i];
                let z_u = z_u_full[top_i];
                let dz_l = dz_l_full[top_i];
                let dz_u = dz_u_full[top_i];
                let mu_target = if top_side == "U" {
                    z_u * top_slack
                } else {
                    z_l * top_slack
                };
                let sigma = if top_side == "U" {
                    z_u / top_slack.max(1e-300)
                } else {
                    z_l / top_slack.max(1e-300)
                };
                eprintln!(
                    "[gBD-zmu] iter={} top_i={} z_L={:.3e} z_U={:.3e} dz_L={:.3e} dz_U={:.3e}  z*slack={:.3e} (mu={:.3e}) Σ={:.3e}",
                    self.iter, top_i, z_l, z_u, dz_l, dz_u, mu_target, self.mu, sigma,
                );
            }
        }
        grad_phi_dx
    }
}


/// Run preprocessing (fixed-variable and redundant-constraint elimination)
/// and, if it reduces the problem, recursively solve the smaller problem
/// then unmap the solution back to the user's variable space. Returns
/// `Some(result)` whenever preprocessing reduced the problem (regardless
/// of solve status); `None` if no reduction was possible, so the caller
/// solves the original problem directly.
///
/// This mirrors Ipopt 3.14's `TNLPAdapter` forward-pass behavior: when
/// the adapter eliminates fixed variables, it commits to the reduced
/// problem — there is no retry-on-failure with the unreduced problem.
fn try_preprocessed_solve<P: NlpProblem>(
    problem: &P,
    options: &SolverOptions,
) -> Option<SolveResult> {
    let make_parameter = matches!(
        options.fixed_variable_treatment,
        FixedVariableTreatment::MakeParameter
    );
    let prep = if options.enable_preprocessing {
        crate::preprocessing::PreprocessedProblem::new(
            problem as &dyn NlpProblem,
            options.bound_push,
        )
    } else if make_parameter {
        // `fixed_variable_treatment = make_parameter` activates fixed-var
        // elimination even when full preprocessing is disabled. Mirrors
        // Ipopt 3.14's `TNLPAdapter` behavior.
        crate::preprocessing::PreprocessedProblem::new_fixed_only(problem as &dyn NlpProblem)
    } else {
        return None;
    };
    if !prep.did_reduce() {
        return None;
    }
    if options.print_level >= 5 {
        rip_log!(
            "ripopt: Preprocessing reduced problem: {} fixed vars, {} redundant constraints ({}x{} -> {}x{})",
            prep.num_fixed(), prep.num_redundant(),
            problem.num_variables(), problem.num_constraints(),
            prep.num_variables(), prep.num_constraints(),
        );
    }
    let mut prep_opts = options.clone();
    prep_opts.enable_preprocessing = false;
    let reduced_result = solve(&prep, &prep_opts);
    Some(prep.unmap_solution(&reduced_result))
}

/// Compute the accepted step length and resulting θ for one Gauss–Newton
/// polish iterate. Applies a τ=0.995 fraction-to-boundary cap on variable
/// bounds, then halves α (up to 10×) until θ improves. Returns
/// `Some((alpha, theta))` if an improving step was found, `None` if the
/// constraint evaluation failed; the caller owns the "no improvement"
/// termination check by comparing the returned θ against the previous one.

/// Solve the NLP using the interior point method.
///
/// When `options.user_x_scaling` is `Some(non_empty)`, the user-supplied
/// strictly-positive `dx` vector is applied via [`XScaledProblem`]: the
/// IPM driver runs in the internal coordinate system `x' = D_x · x`, and
/// the returned [`SolveResult`] is unscaled back to the user's
/// coordinates (`x_user = x' / dx`, `z_L_user = dx · z_L_internal`,
/// `z_U_user = dx · z_U_internal`, multipliers and constraint values
/// unchanged). This mirrors Ipopt 3.14's `IpScaledNLP` /
/// `UserScaling` pipeline restricted to x-scaling.
pub fn solve<P: NlpProblem>(problem: &P, options: &SolverOptions) -> SolveResult {
    let solve_start = Instant::now();

    // Roadmap item #6: `options.user_x_scaling`. If a non-empty,
    // strictly-positive scaling vector is provided, wrap the NLP with
    // `XScaledProblem` and run the IPM in scaled coordinates, then
    // unscale the result. Invalid input (wrong length, non-positive,
    // non-finite entries) returns `InternalError` — same posture as
    // the previous "not yet implemented" guardrail.
    if let Some(ref xs) = options.user_x_scaling {
        if !xs.is_empty() {
            let n = problem.num_variables();
            if xs.len() != n || xs.iter().any(|&v| !v.is_finite() || v <= 0.0) {
                rip_log!(
                    "ripopt: user_x_scaling rejected (len={}, expected n={}, \
                     all entries must be strictly positive and finite); \
                     returning InternalError.",
                    xs.len(), n
                );
                return SolveResult {
                    x: vec![0.0; n],
                    objective: f64::NAN,
                    constraint_multipliers: vec![0.0; problem.num_constraints()],
                    bound_multipliers_lower: vec![0.0; n],
                    bound_multipliers_upper: vec![0.0; n],
                    constraint_values: vec![0.0; problem.num_constraints()],
                    status: SolveStatus::InternalError,
                    iterations: 0,
                    diagnostics: SolverDiagnostics::default(),
                };
            }
            let wrapped = XScaledProblem::new(problem, xs.clone());
            let mut inner_options = options.clone();
            inner_options.user_x_scaling = None;
            let mut result = solve_inner(&wrapped, &inner_options, solve_start);
            for i in 0..n {
                result.x[i] *= wrapped.inv_dx[i];
                result.bound_multipliers_lower[i] *= wrapped.dx[i];
                result.bound_multipliers_upper[i] *= wrapped.dx[i];
            }
            return result;
        }
    }

    solve_inner(problem, options, solve_start)
}

fn solve_inner<P: NlpProblem>(
    problem: &P,
    options: &SolverOptions,
    solve_start: Instant,
) -> SolveResult {
    if let Some(result) = try_preprocessed_solve(problem, options) {
        return result;
    }
    let mut result = solve_ipm(problem, options);
    result.diagnostics.wall_time_secs = solve_start.elapsed().as_secs_f64();
    if options.print_level >= 4 {
        print_final_summary(&result);
    }
    result
}

/// B11: Print Ipopt-style final summary block when the IPM has
/// terminated. Uses fields populated on `result.diagnostics` plus the
/// solve status. Mirrors Ipopt's
/// `IpoptApplication::FinalizeSolution` summary, except ripopt's
/// `constraints` / `jacobian_values` are joint callbacks so the
/// equality and inequality counters share a single value (Ipopt's
/// per-side split would require splitting the NLP trait, which is a
/// larger refactor — flagged as future work in the SolverDiagnostics
/// docstring).
///
/// Scaled vs unscaled: `obj_scaling` is applied at the IPM level, so
/// `result.objective` is already the unscaled value. ripopt does not
/// currently track a separate "scaled" objective for reporting; print
/// the same value on both rows. Same applies to inf_pr/inf_du/compl —
/// ripopt's diagnostics store the unscaled iteration values, and the
/// scaled gates use s_d/s_c on the fly. Print the diagnostics as
/// "(unscaled)" and the s_d-divided values as "(scaled)" to match
/// Ipopt's output ordering.
fn print_final_summary(result: &SolveResult) {
    let d = &result.diagnostics;
    let status_str = match result.status {
        SolveStatus::Optimal => "Optimal Solution Found.",
        SolveStatus::Acceptable => "Solved To Acceptable Level.",
        SolveStatus::Infeasible => "Converged to a point of local infeasibility. Problem may be infeasible.",
        SolveStatus::LocalInfeasibility => "Converged to a point of local infeasibility.",
        SolveStatus::MaxIterations => "Maximum Number of Iterations Exceeded.",
        SolveStatus::NumericalError => "Numerical Difficulties Encountered.",
        SolveStatus::DivergingIterates => "Diverging Iterates -- Problem May Be Unbounded.",
        SolveStatus::RestorationFailed => "Restoration Failed.",
        SolveStatus::EvaluationError => "Evaluation Error.",
        SolveStatus::UserRequestedStop => "User Requested Stop.",
        SolveStatus::StopAtTinyStep => "Search Direction Becomes Too Small.",
        SolveStatus::InternalError => "Internal Error.",
    };

    // s_d uses the dual-multiplier sum from the converged iterate; s_c
    // is recomputed from the dual_inf/compl ratio when meaningful.
    let s_d = d.final_s_d.max(1.0);
    let inf_du_scaled = d.final_dual_inf / s_d;
    // Ipopt's NLP-error formula: max(inf_du/s_d, inf_pr_user, compl/s_c).
    // ripopt does not currently track s_c on diagnostics; treat as 1.0
    // (matches the "no scaling" path Ipopt uses for problems where the
    // multiplier sums fall below the threshold).
    let s_c: f64 = 1.0;
    let compl_scaled = d.final_compl / s_c;
    let nlp_error = inf_du_scaled.max(d.final_primal_inf).max(compl_scaled);

    rip_log!("");
    rip_log!("Number of Iterations....: {}", result.iterations);
    rip_log!("");
    rip_log!("                                   (scaled)                 (unscaled)");
    rip_log!("Objective...............:  {:>22.16e}   {:>22.16e}", result.objective, result.objective);
    rip_log!("Dual infeasibility......:  {:>22.16e}   {:>22.16e}", inf_du_scaled, d.final_dual_inf);
    rip_log!("Constraint violation....:  {:>22.16e}   {:>22.16e}", d.final_primal_inf, d.final_primal_inf);
    rip_log!("Complementarity.........:  {:>22.16e}   {:>22.16e}", compl_scaled, d.final_compl);
    rip_log!("Overall NLP error.......:  {:>22.16e}   {:>22.16e}", nlp_error, nlp_error);
    rip_log!("");
    rip_log!("Number of objective function evaluations             = {}", d.n_obj_evals);
    rip_log!("Number of objective gradient evaluations             = {}", d.n_grad_evals);
    rip_log!("Number of equality constraint evaluations            = {}", d.n_constr_evals);
    rip_log!("Number of inequality constraint evaluations          = {}", d.n_constr_evals);
    rip_log!("Number of equality constraint Jacobian evaluations   = {}", d.n_jac_evals);
    rip_log!("Number of inequality constraint Jacobian evaluations = {}", d.n_jac_evals);
    rip_log!("Number of Lagrangian Hessian evaluations             = {}", d.n_hess_evals);
    rip_log!("Total seconds in ripopt                              = {:.3}", d.wall_time_secs);
    rip_log!("");
    rip_log!("EXIT: {}", status_str);
}

/// Accumulates wall-clock time spent in each phase of the IPM loop.
/// Printed as a summary table at the end of `solve_ipm` when `print_level >= 5`.
struct PhaseTimings {
    problem_eval: Duration,
    kkt_assembly: Duration,
    factorization: Duration,
    direction_solve: Duration,
    line_search: Duration,
}

impl PhaseTimings {
    fn new() -> Self {
        PhaseTimings {
            problem_eval: Duration::ZERO,
            kkt_assembly: Duration::ZERO,
            factorization: Duration::ZERO,
            direction_solve: Duration::ZERO,
            line_search: Duration::ZERO,
        }
    }

    fn print_summary(&self, iterations: usize, total: Duration) {
        let total_secs = total.as_secs_f64();
        let phases = [
            ("Problem eval", self.problem_eval),
            ("KKT assembly", self.kkt_assembly),
            ("Factorization", self.factorization),
            ("Direction solve", self.direction_solve),
            ("Line search", self.line_search),
        ];
        let accounted: Duration = phases.iter().map(|(_, d)| *d).sum();
        let other = total.saturating_sub(accounted);

        rip_log!("\nPhase breakdown ({} iterations):", iterations);
        for (name, dur) in &phases {
            let secs = dur.as_secs_f64();
            let pct = if total_secs > 0.0 { 100.0 * secs / total_secs } else { 0.0 };
            rip_log!("  {:<20} {:>8.3}s ({:>5.1}%)", name, secs, pct);
        }
        let other_secs = other.as_secs_f64();
        let other_pct = if total_secs > 0.0 { 100.0 * other_secs / total_secs } else { 0.0 };
        rip_log!("  {:<20} {:>8.3}s ({:>5.1}%)", "Other", other_secs, other_pct);
        rip_log!("  {:<20} {:>8.3}s", "Total", total_secs);
    }
}

/// B10: Print Ipopt-style problem statistics block at solve start.
///
/// Mirrors the block printed by `IpIpoptApplication::OptimizeNLP` after
/// the NLP is loaded and before iteration 0:
///
/// ```text
/// Number of nonzeros in equality constraint Jacobian.: NNNN
/// Number of nonzeros in inequality constraint Jacobian: NNNN
/// Number of nonzeros in Lagrangian Hessian.............: NNNN
///
/// Total number of variables............................: NNNN
///                      variables with only lower bounds: NNNN
///                 variables with lower and upper bounds: NNNN
///                      variables with only upper bounds: NNNN
/// Total number of equality constraints.................: NNNN
/// Total number of inequality constraints...............: NNNN
///         inequality constraints with only lower bounds: NNNN
///    inequality constraints with lower and upper bounds: NNNN
///         inequality constraints with only upper bounds: NNNN
/// ```
///
/// Equality vs inequality is the slack-reformulation classification
/// (`g_l[i] == g_u[i]` ⇔ equality). Hessian nnz counts the lower-triangle
/// entries actually returned by `hessian_structure` (Ipopt also reports
/// the lower triangle). Free variables (no finite bound) are not printed
/// as a separate row in Ipopt 3.14's standard output; they are implied by
/// `total − sum(only_lower, both, only_upper)`.
fn print_problem_header(state: &SolverState) {
    let n = state.n;
    let m = state.m;

    // Variable bound classification.
    let mut var_only_lower = 0usize;
    let mut var_both = 0usize;
    let mut var_only_upper = 0usize;
    for i in 0..n {
        let l_fin = state.x_l_at(i).is_finite();
        let u_fin = state.x_u_at(i).is_finite();
        match (l_fin, u_fin) {
            (true, true) => var_both += 1,
            (true, false) => var_only_lower += 1,
            (false, true) => var_only_upper += 1,
            _ => {}
        }
    }

    // Constraint classification + per-row equality flag.
    let mut n_eq = 0usize;
    let mut n_ineq_only_lower = 0usize;
    let mut n_ineq_both = 0usize;
    let mut n_ineq_only_upper = 0usize;
    let mut row_is_eq = vec![false; m];
    for i in 0..m {
        let l_fin = state.g_l_at(i).is_finite();
        let u_fin = state.g_u_at(i).is_finite();
        let is_eq = l_fin && u_fin && (state.g_l_at(i) - state.g_u_at(i)).abs() < 1e-14;
        if is_eq {
            n_eq += 1;
            row_is_eq[i] = true;
        } else {
            match (l_fin, u_fin) {
                (true, true) => n_ineq_both += 1,
                (true, false) => n_ineq_only_lower += 1,
                (false, true) => n_ineq_only_upper += 1,
                _ => {} // unbounded inequality (rare; not classified by Ipopt)
            }
        }
    }
    let n_ineq = n_ineq_only_lower + n_ineq_both + n_ineq_only_upper;

    // Jacobian nnz split by equality vs inequality row.
    let _ = m;
    let _ = row_is_eq;
    let jac_nnz_eq = state.jac_c_rows.len();
    let jac_nnz_ineq = state.jac_d_rows.len();
    let hess_nnz = state.hess_rows.len();

    rip_log!("");
    rip_log!("Number of nonzeros in equality constraint Jacobian...:    {:>8}", jac_nnz_eq);
    rip_log!("Number of nonzeros in inequality constraint Jacobian.:    {:>8}", jac_nnz_ineq);
    rip_log!("Number of nonzeros in Lagrangian Hessian.............:    {:>8}", hess_nnz);
    rip_log!("");
    rip_log!("Total number of variables............................:    {:>8}", n);
    rip_log!("                     variables with only lower bounds:    {:>8}", var_only_lower);
    rip_log!("                variables with lower and upper bounds:    {:>8}", var_both);
    rip_log!("                     variables with only upper bounds:    {:>8}", var_only_upper);
    rip_log!("Total number of equality constraints.................:    {:>8}", n_eq);
    rip_log!("Total number of inequality constraints...............:    {:>8}", n_ineq);
    rip_log!("        inequality constraints with only lower bounds:    {:>8}", n_ineq_only_lower);
    rip_log!("   inequality constraints with lower and upper bounds:    {:>8}", n_ineq_both);
    rip_log!("        inequality constraints with only upper bounds:    {:>8}", n_ineq_only_upper);
    rip_log!("");
}

/// Update the L-BFGS Hessian approximation (no-op when `lbfgs_state` is `None`).
///
/// Recomputes the Lagrangian gradient at the current iterate and feeds
/// `(x, grad_L)` into the L-BFGS memory, then fills `state.hess_vals` with
/// the dense approximate Hessian `B_k`. When exact Hessian is used
/// (`lbfgs_state == None`) this is a no-op — the Hessian was populated
/// directly by `state.evaluate_with_linear`.
///
/// Ipopt parallel: `HessianUpdater` strategy object
/// (`IpHessianUpdater.{hpp,cpp}` with `LimMemQuasiNewtonUpdater` /
/// `ExactHessianUpdater` subclasses).
///
/// Extracted from `solve_ipm` as part of the v0.8 main-loop decomposition
/// (pre-work step 2). Replaces ~16 copies of the same 6-line pattern.
fn update_lbfgs_hessian(
    lbfgs_state: &mut Option<LbfgsIpmState>,
    state: &mut SolverState,
) {
    if let Some(ref mut lbfgs) = lbfgs_state {
        let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
            &state.grad_f,
            &state.jac_c_rows, &state.jac_c_cols, &state.jac_c_vals,
            &state.jac_d_rows, &state.jac_d_cols, &state.jac_d_vals,
            &state.y_c, &state.y_d, state.n,
        );
        lbfgs.update(&state.x, &lag_grad);
        lbfgs.fill_hessian(&mut state.hess_vals);
    }
}

/// Re-evaluate the problem at the current `state.x` (Hessian, gradient,
/// constraints, Jacobian) and refresh the L-BFGS Hessian if L-BFGS mode
/// is on. The two-step "evaluate then update_lbfgs" pair appears at
/// over a dozen sites: every restoration recovery, every perturbation
/// retry, every saved-state restore. Returns the evaluator's success
/// flag for the few callers that branch on it.
fn evaluate_and_refresh_lbfgs<P: NlpProblem>(
    state: &mut SolverState,
    problem: &P,
    lbfgs_state: &mut Option<LbfgsIpmState>,
    linear_constraints: Option<&[bool]>,
    lbfgs_mode: bool,
) -> bool {
    let ok = state.evaluate_with_linear(problem, 1.0, linear_constraints, lbfgs_mode);
    update_lbfgs_hessian(lbfgs_state, state);
    ok
}

/// Outcome of the backtracking line-search trial loop.
enum LineSearchOutcome {
    StepAccepted,
    Rejected,
}

/// Run the Ipopt-style backtracking line search on the current direction.
///
/// Up to 40 α-halving trials starting from `alpha_primal_max`. Each trial:
///  1. form `x_trial = x + α·dx` (clamped strictly inside bounds),
///  2. evaluate f, g at the trial; reject NaN/Inf,
///  3. watchdog shortcut: accept the full step unconditionally,
///  4. compute φ_trial with the (optional) constraint-slack barrier,
///  5. run the filter acceptability test (switching + Armijo + sufficient
///     infeasibility reduction),
///  6. on failure, attempt a second-order correction (SOC) on the first
///     trial only (Maratos-effect guard),
///  7. otherwise backtrack.
///
/// Returns `Return(NumericalError)` if the early-iteration stall timeout
/// trips, `StepAccepted` on any accepted trial / SOC / watchdog full
/// step, `Rejected` if α falls below `min_alpha` / loop budget exhausts.
#[allow(clippy::too_many_arguments)]
/// Compute the barrier-augmented objective phi(x, g) = obj - mu*Σ ln(slack)
/// summed over (a) all finite variable bounds via x and (b) when
/// `constraint_slack_barrier` is on, all finite inequality-constraint
/// bounds via g whose slack exceeds mu*1e-2 (the small-slack guard
/// matches the line-search loop and the SOC routines).
///
/// When `kappa_d > 0`, also adds Ipopt's `kappa_d` damping term
/// `+ kappa_d * mu * Σ slack_oneside[i]` for each variable with exactly
/// one finite bound. Without this term the barrier is unbounded below
/// for one-sided-bound variables (Ipopt 3.14 default `kappa_d = 1e-5`).
fn compute_barrier_phi(
    obj: f64,
    x: &[f64],
    s: &[f64],
    state: &SolverState,
    n: usize,
    _m: usize,
    constraint_slack_barrier: bool,
    kappa_d: f64,
) -> f64 {
    let mut phi = obj;
    for i in 0..n {
        let l_fin = state.x_l_at(i).is_finite();
        let u_fin = state.x_u_at(i).is_finite();
        if l_fin {
            let slack = (x[i] - state.x_l_at(i)).max(1e-20);
            phi -= state.mu * slack.ln();
        }
        if u_fin {
            let slack = (state.x_u_at(i) - x[i]).max(1e-20);
            phi -= state.mu * slack.ln();
        }
        // kappa_d damping: penalize drift toward the open side for
        // variables with exactly one finite bound. Mirrors Ipopt 3.14
        // CalcBarrierTerm in IpIpoptCalculatedQuantities.cpp.
        if kappa_d > 0.0 && (l_fin ^ u_fin) {
            let s_oneside = if l_fin {
                (x[i] - state.x_l_at(i)).max(0.0)
            } else {
                (state.x_u_at(i) - x[i]).max(0.0)
            };
            phi += kappa_d * state.mu * s_oneside;
        }
    }
    if constraint_slack_barrier {
        // Ipopt 3.14 CalcBarrierTerm uses slack_s_L = s - d_l and
        // slack_s_U = d_u - s — the barrier acts on the explicit slack
        // iterate s, not on d(x). When the iterate is infeasible
        // (theta>0) these differ; using d(x) here flips the sign of
        // grad φ · dx and breaks filter alignment.
        for k in 0..state.layout.n_d {
            let dl = state.d_l_at(k);
            let du = state.d_u_at(k);
            if dl.is_finite() {
                let slack = s[k] - dl;
                if slack > state.mu * 1e-2 {
                    phi -= state.mu * slack.ln();
                }
            }
            if du.is_finite() {
                let slack = du - s[k];
                if slack > state.mu * 1e-2 {
                    phi -= state.mu * slack.ln();
                }
            }
        }
    }
    phi
}

/// Project the candidate iterate `x + alpha*dx` onto the open variable
/// box (1e-14 inset from finite x_l/x_u), then evaluate the objective
/// and constraints at the projected point. Returns
/// `Some((x_trial, obj_trial, g_trial, theta_trial))` when both
/// evaluations succeed and produced finite values; `None` when the
/// objective/constraints failed or returned NaN/Inf — caller halves
/// alpha and retries.
fn evaluate_trial_point<P: NlpProblem>(
    state: &mut SolverState,
    problem: &P,
    alpha: f64,
    m: usize,
) -> Option<(Vec<f64>, f64, Vec<f64>, f64)> {
    let x_trial = compute_clamped_trial_x(state, &state.dx, alpha);

    // Phase 10b.3: trial-point eval routes through SplitNlp.
    let nlp = crate::split_nlp::SplitNlp::new(problem, &state.layout);
    let mut obj_trial = f64::INFINITY;
    state.n_obj_evals += 1;
    let obj_ok = nlp.objective(&x_trial, true, &mut obj_trial);
    let mut g_trial = vec![0.0; m];
    let constr_ok = if m > 0 {
        state.n_constr_evals += 1;
        nlp.constraints_combined(&x_trial, true, &mut g_trial)
    } else {
        true
    };

    if !obj_ok || !constr_ok || obj_trial.is_nan() || obj_trial.is_infinite()
        || g_trial.iter().any(|v| v.is_nan() || v.is_infinite())
    {
        return None;
    }

    // Slack-coupled trial theta — A8.19. Trial slack `s + α·ds` is
    // feasible because `compute_alpha_max` already applied frac-to-bound
    // to the primal slack step.
    let s_trial = compute_trial_slack(state, alpha);
    let (c_trial, d_trial) = split_from_g(state, &g_trial);
    let theta_trial = theta_for_split_d_s(state, &c_trial, &d_trial, &s_trial);
    Some((x_trial, obj_trial, g_trial, theta_trial))
}

/// Print a diagnostic breakdown of why a line-search trial was
/// rejected: which of the four filter sub-tests (switching condition,
/// Armijo, sufficient θ-reduction, filter acceptability) passed and
/// which failed. Gated by `print_level >= 7` and the first 5 LS
/// rejections per iteration; isolates the verbose tracing from the
/// hot loop in `run_line_search_loop`.
fn log_line_search_rejection(
    filter: &Filter,
    theta_current: f64,
    phi_current: f64,
    theta_trial: f64,
    phi_trial: f64,
    grad_phi_step: f64,
    alpha: f64,
) {
    let sw = filter.switching_condition(theta_current, grad_phi_step, alpha);
    let ar = filter.armijo_condition(phi_current, phi_trial, grad_phi_step, alpha);
    let sr = filter.sufficient_infeasibility_reduction(theta_current, theta_trial);
    let fa = filter.is_acceptable(theta_trial, phi_trial);
    rip_log!(
        "  LS reject: alpha={:.2e} theta_t={:.2e} phi_t={:.2e} switch={} armijo={} suff_red={} filter_ok={}",
        alpha, theta_trial, phi_trial, sw, ar, sr, fa
    );
}

fn run_line_search_loop<P: NlpProblem>(
    state: &mut SolverState,
    problem: &P,
    options: &SolverOptions,
    filter: &mut Filter,
    inertia_params: &mut InertiaCorrectionParams,
    alpha_primal_max: f64,
    theta_current: f64,
    phi_current: f64,
    grad_phi_step: f64,
    min_alpha: f64,
    watchdog_active: bool,
    iteration: usize,
    n: usize,
    m: usize,
    trace_meta: &mut TraceMetadata,
    ls_steps: &mut usize,
    aug_solver: &mut dyn LinearSolver,
    aug_kkt: &crate::kkt_aug::AugKktSystem,
) -> LineSearchOutcome {
    let mut alpha = alpha_primal_max;
    let mut step_accepted = false;
    *ls_steps = 0;

    // DEV-35: no hard-coded line-search step cap. Ipopt
    // (IpFilterLSAcceptor.cpp::ComputeAlphaMin and the backtracking
    // loop in IpBacktrackingLineSearch::DoBacktrackingLineSearch)
    // terminates the line search on `alpha < alpha_min`, where
    // alpha_min is itself derived from filter parameters and the
    // current iterate. The previous `for _ls_iter in 0..40` cap was
    // ripopt-specific and could prematurely abandon a step that was
    // still on track to either accept or fall through to alpha_min.
    loop {
        if alpha < min_alpha {
            break;
        }

        let (x_trial, obj_trial, g_trial, theta_trial) =
            match evaluate_trial_point(state, problem, alpha, m) {
                Some(t) => t,
                None => {
                    alpha *= 0.5;
                    *ls_steps += 1;
                    continue;
                }
            };

        // Barrier objective at trial — uses the explicit-slack iterate
        // `s_trial = s + α·ds` (Ipopt's curr_slack_*-based phi).
        let s_trial = compute_trial_slack(state, alpha);
        let phi_trial = compute_barrier_phi(
            obj_trial, &x_trial, &s_trial, state, n, m, options.constraint_slack_barrier,
            options.kappa_d,
        );

        if std::env::var("RIPOPT_LS_PROBE").is_ok() && iteration <= 112 {
            let is_ft = filter.is_ftype(theta_current, grad_phi_step, alpha);
            let armijo = filter.armijo_condition(phi_current, phi_trial, grad_phi_step, alpha);
            let suf_theta = filter.sufficient_infeasibility_reduction(theta_current, theta_trial);
            let suf_phi_rhs = phi_current - filter.gamma_phi() * theta_current;
            let suf_phi = phi_trial <= suf_phi_rhs;
            let omi = filter.passes_obj_max_inc(phi_current, phi_trial, false);
            let in_filter = filter.is_acceptable(theta_trial, phi_trial);
            eprintln!(
                "[probe] iter={} ls={} alpha={:.3e} gBD={:.6e} theta_curr={:.10e} theta_tr={:.10e} phi_curr={:.10e} phi_tr={:.10e} dphi={:.3e} dtheta={:.3e} entries={} is_ft={} armijo={} suf_theta={} suf_phi={} (rhs={:.10e}) omi={} in_filter={}",
                iteration, *ls_steps, alpha, grad_phi_step,
                theta_current, theta_trial, phi_current, phi_trial,
                phi_trial - phi_current, theta_trial - theta_current,
                filter.len(), is_ft, armijo, suf_theta, suf_phi, suf_phi_rhs, omi, in_filter,
            );
        }

        // DEV-30/31: `augment_required` is the Ipopt
        // `UpdateForNextIteration` augmentation gate (`!IsFtype || !ArmijoHolds`,
        // IpFilterLSAcceptor.cpp:881-895). Pure IsFtype, *without* the
        // theta_min clause, so h-type accepts that happened to satisfy
        // IsFtype+Armijo at the accepted alpha do not over-augment.
        let (acceptable, augment_required) = filter.check_acceptability(
            theta_current,
            phi_current,
            theta_trial,
            phi_trial,
            grad_phi_step,
            alpha, false,
        );

        // Watchdog full step (T2.21, spec §4): filter check still runs (Ipopt
        // fidelity), but on accept we skip filter augmentation so transient
        // infeasibility is tolerated across the watchdog's trial window. See
        // `IpBacktrackingLineSearch.cpp::DoBacktrackingLineSearch`.
        if watchdog_active && alpha == alpha_primal_max && acceptable {
            commit_trial_point(state, x_trial, obj_trial, g_trial, alpha);
            step_accepted = true;
            // T3.10: watchdog-accept also runs the filter-reset
            // bookkeeping so consecutive trapped iterations advance
            // the counter even when filter augmentation is suppressed.
            filter.note_acceptance();
            break;
        }

        if !acceptable && options.print_level >= 7 && *ls_steps < 5 {
            log_line_search_rejection(
                filter, theta_current, phi_current, theta_trial, phi_trial,
                grad_phi_step, alpha,
            );
        }

        if acceptable {
            commit_trial_point(state, x_trial, obj_trial, g_trial, alpha);
            step_accepted = true;
            if augment_required {
                filter.add(theta_current, phi_current);
            }
            // T3.10: post-acceptance filter-reset trigger
            // (`IpFilterLSAcceptor.cpp:407-434`).
            let reset_fired = filter.note_acceptance();
            if reset_fired && options.print_level >= 5 {
                eprintln!(
                    "  [filter] iter={}: filter reset (resets={})",
                    state.iter, filter.n_filter_resets()
                );
            }
            break;
        }

        // SOC on the first trial only, if full step did not strictly decrease theta.
        // Mirrors `IpFilterLSAcceptor::TrySecondOrderCorrection` entry condition
        // (`IpFilterLSAcceptor.cpp` SOC dispatch): `theta_trial >= theta_current`.
        if theta_trial >= theta_current && options.max_soc > 0 && *ls_steps == 0 {
            let soc_accepted = attempt_soc_aug(
                state, problem, &g_trial, inertia_params, filter,
                theta_current, phi_current, grad_phi_step, alpha, options,
                aug_solver, aug_kkt,
            );
            if let Some((x_soc, obj_soc, g_soc, alpha_soc)) = soc_accepted {
                state.diagnostics.soc_corrections += 1;
                trace_meta.soc_accepted = true;
                commit_trial_point(state, x_soc, obj_soc, g_soc, alpha_soc);
                step_accepted = true;
                filter.add(theta_current, phi_current);
                // T3.10: SOC-accepted step counts as an accepted step
                // for the filter-reset heuristic.
                filter.note_acceptance();
                break;
            }
        }

        alpha *= 0.5;
        *ls_steps += 1;
    }

    if step_accepted {
        LineSearchOutcome::StepAccepted
    } else {
        LineSearchOutcome::Rejected
    }
}

/// Apply the accepted step to the dual variables (y, z_l, z_u) and
/// enforce Ipopt's kappa_sigma safeguard on bound multipliers.
///
/// - `y` (constraint multipliers) use `alpha_for_y = primal`: the
///   accepted primal step length (Ipopt default). Near convergence
///   (`consecutive_acceptable >= 1`), persistent sign changes in
///   `dy_i` across ≥3 iterations trigger 50% damping to suppress
///   oscillation.
/// - `z_l`, `z_u` use `alpha_d` (fraction-to-boundary on bound
///   multipliers). After the update, each is clamped to keep `z*s`
///   inside `[mu_ks/kappa_sigma, kappa_sigma*mu_ks]` (Ipopt
///   `IpIpoptCalculatedQuantities::ComputePDSystem`-style safeguard
///   with `kappa_sigma = 1e10`).
///
/// Writes `state.alpha_dual = alpha_d`. Returns `mu_ks`, which the
/// caller reuses when resetting slack-constraint multipliers `v_l`,
/// `v_u` after the post-step re-evaluation.
/// Apply the κ_σ=1e10 bound-multiplier reset (Wächter & Biegler 2006,
/// eq. (16)). Each `z_L[i]` and `z_U[i]` is taken `α_d` along the
/// Newton step, then clamped to `[μ_KS/(κ_σ·s), κ_σ·μ_KS/s]`. In Free
/// mu mode, μ_KS uses `max(avg_compl, μ)` (capped at 1e3) so the
/// clamp tracks actual centrality rather than the lagging μ. Returns
/// the μ_KS used (printed in the diagnostics row).
/// Advance bound multipliers to the trial step `z + alpha_d·dz`, with a
/// floor of `1e-20` to keep them strictly positive. Mirrors Ipopt's
/// trial-iterate construction (the line-search applies the same step
/// internally before AcceptTrialPoint commits). Split out from
/// `apply_kappa_sigma_bound_multiplier_reset` so that
/// `apply_slack_move` (which fires between this and the kappa_sigma
/// clamp, per Ipopt order) sees the trial z, not the previous-iter z.
fn advance_z_to_trial(state: &mut SolverState, alpha_d: f64) {
    // Phase 6d.6: compressed bound storage is canonical. Walk it
    // natively; the combined mirror is dropped.
    for k in 0..state.bound_layout.n_x_l {
        let new_z = (state.z_l_compressed[k] + alpha_d * state.dz_l_compressed[k]).max(1e-20);
        state.z_l_compressed[k] = new_z;
    }
    for k in 0..state.bound_layout.n_x_u {
        let new_z = (state.z_u_compressed[k] + alpha_d * state.dz_u_compressed[k]).max(1e-20);
        state.z_u_compressed[k] = new_z;
    }
    // Slack-bound multipliers v_L, v_U: same Newton update as z_L, z_U
    // (Ipopt's `IpIpoptAlg.cpp:652-770` advances all four blocks with the
    // shared α_dual). Phase 8d: walk compressed storage directly; the
    // d_bound_layout already excludes equality rows and d-rows without
    // a finite slack bound.
    for kc in 0..state.d_bound_layout.n_d_l {
        let new_v = (state.v_l_compressed[kc] + alpha_d * state.dv_l_compressed[kc]).max(1e-20);
        state.v_l_compressed[kc] = new_v;
    }
    for kc in 0..state.d_bound_layout.n_d_u {
        let new_v = (state.v_u_compressed[kc] + alpha_d * state.dv_u_compressed[kc]).max(1e-20);
        state.v_u_compressed[kc] = new_v;
    }
    // T3.25: bump dual atags so a downstream factor doesn't reuse a
    // stale fingerprint after this trial-step write.
    state.bump_dual_atags();
}

/// Apply the kappa_sigma reset to bound multipliers (Ipopt's
/// `correct_bound_multiplier` at `IpIpoptAlg.cpp:716-767`):
/// clamp each `z` into `[mu_ks/(kappa_sigma·s), kappa_sigma·mu_ks/s]`.
/// Assumes z has already been advanced to the trial step (see
/// `advance_z_to_trial`) and any pending slack_move has run.
fn apply_kappa_sigma_bound_multiplier_reset(
    state: &mut SolverState,
    mu_state: &MuState,
) -> f64 {
    let m = state.m;
    let kappa_sigma = 1e10;
    // T2.3: drop the ripopt-specific `.max(state.mu)` clamp on the Free-mode
    // mu_ks; Ipopt's `correct_bound_multiplier` uses the current barrier
    // parameter directly. Keep the `1e3` upper cap as a numerical safety
    // (avg_compl spikes can produce wildly wide z-clamps otherwise).
    let mu_ks = if mu_state.mode == MuMode::Free {
        compute_avg_complementarity(state).min(1e3)
    } else {
        state.mu
    };
    // Phase 6d.6: compressed bound storage is canonical.
    for k in 0..state.bound_layout.n_x_l {
        let i = state.bound_layout.x_l_to_full[k];
        let s_l = slack_xl(state, i);
        let z_lo = mu_ks / (kappa_sigma * s_l);
        let z_hi = kappa_sigma * mu_ks / s_l;
        state.z_l_compressed[k] = state.z_l_compressed[k].clamp(z_lo, z_hi);
    }
    for k in 0..state.bound_layout.n_x_u {
        let i = state.bound_layout.x_u_to_full[k];
        let s_u = slack_xu(state, i);
        let z_lo = mu_ks / (kappa_sigma * s_u);
        let z_hi = kappa_sigma * mu_ks / s_u;
        state.z_u_compressed[k] = state.z_u_compressed[k].clamp(z_lo, z_hi);
    }
    // Apply the same κ_σ band to the slack-bound multipliers v_L, v_U
    // (Ipopt's `correct_bound_multiplier` runs over ALL FOUR blocks,
    // `IpIpoptAlg.cpp:721-758`). Phase 8d: walk compressed storage
    // directly; only finite-bound entries exist.
    for kc in 0..state.d_bound_layout.n_d_l {
        let k = state.d_bound_layout.d_l_to_full[kc];
        let i = state.layout.d_to_combined[k];
        let s_l = slack_gl(state, i);
        let v_lo = mu_ks / (kappa_sigma * s_l);
        let v_hi = kappa_sigma * mu_ks / s_l;
        state.v_l_compressed[kc] = state.v_l_compressed[kc].clamp(v_lo, v_hi);
    }
    for kc in 0..state.d_bound_layout.n_d_u {
        let k = state.d_bound_layout.d_u_to_full[kc];
        let i = state.layout.d_to_combined[k];
        let s_u = slack_gu(state, i);
        let v_lo = mu_ks / (kappa_sigma * s_u);
        let v_hi = kappa_sigma * mu_ks / s_u;
        state.v_u_compressed[kc] = state.v_u_compressed[kc].clamp(v_lo, v_hi);
    }
    let _ = m;
    mu_ks
}

/// Apply Ipopt 3.14's `slack_move` runtime slack adjustment.
///
/// Mirrors `IpoptCalculatedQuantities::CalculateSafeSlack`
/// (`ref/Ipopt/src/Algorithm/IpIpoptCalculatedQuantities.cpp:455-537`).
///
/// Trigger: a primal slack `s_l = x[i] - x_l[i]` (or upper mirror
/// `s_u = x_u[i] - x[i]`) is "unsafe" when it falls below
///   `s_min = max(eps * min(1, mu), f64::MIN_POSITIVE)`.
/// At unsafe slacks, the corresponding *bound* (not the variable) is
/// nudged outward so the new slack equals
///   `new_s = min(max(mu / z, s_min),
///                slack_move * max(1.0, |bound|) + s_old)`.
/// When `z <= 0`, the `mu / z` term is treated as `+infinity`, so the
/// upper cap binds.
///
/// Per Ipopt: this fires BEFORE the kappa_sigma bound-multiplier reset,
/// so `apply_kappa_sigma_bound_multiplier_reset` sees the corrected
/// slacks when it clamps `z_l` / `z_u` against `mu / s`.
///
/// Returns the number of components adjusted on this call. The
/// cumulative count is also accumulated into
/// `state.adjusted_slacks_count`.
fn apply_slack_move(state: &mut SolverState, options: &SolverOptions) -> usize {
    let n = state.n;
    let m = state.m;
    let slack_move = options.slack_move;
    if !(slack_move > 0.0) {
        return 0;
    }
    let mu = state.mu;
    // Ipopt: s_min = eps * min(1, mu); fallback to MIN_POSITIVE if mu is
    // so small that s_min underflows. See IpIpoptCalculatedQuantities.cpp:469-477.
    let s_min = (f64::EPSILON * mu.min(1.0)).max(f64::MIN_POSITIVE);
    let mut adjusted = 0usize;
    // Phase 6d.3: walk compressed bound mirrors. The set
    // {i : x_l[i].is_finite()} matches `x_l_to_full[..n_x_l]` exactly.
    let _ = n;
    for k in 0..state.bound_layout.n_x_l {
        let i = state.bound_layout.x_l_to_full[k];
        let s_l = state.x[i] - state.x_l_compressed[k];
        if s_l < s_min {
            let z = state.z_l_compressed[k];
            let from_mu = if z > 0.0 { mu / z } else { f64::INFINITY };
            let cap = slack_move * state.x_l_compressed[k].abs().max(1.0) + s_l;
            let new_s = from_mu.max(s_min).min(cap);
            state.x_l_compressed[k] -= new_s - s_l;
            adjusted += 1;
        }
    }
    for k in 0..state.bound_layout.n_x_u {
        let i = state.bound_layout.x_u_to_full[k];
        let s_u = state.x_u_compressed[k] - state.x[i];
        if s_u < s_min {
            let z = state.z_u_compressed[k];
            let from_mu = if z > 0.0 { mu / z } else { f64::INFINITY };
            let cap = slack_move * state.x_u_compressed[k].abs().max(1.0) + s_u;
            let new_s = from_mu.max(s_min).min(cap);
            state.x_u_compressed[k] += new_s - s_u;
            adjusted += 1;
        }
    }
    // B-cross8: extend slack_move to the constraint slack iterate `s`
    // against `[g_l, g_u]` (Ipopt's CalculateSafeSlack runs over all
    // four slack blocks: x_L, x_U, s_L, s_U; see
    // IpIpoptCalculatedQuantities.cpp:455-537). Skip equality rows
    // (their s is held at the equality value as a sentinel).
    for i in 0..m {
        let l_fin = state.g_l_at(i).is_finite();
        let u_fin = state.g_u_at(i).is_finite();
        if l_fin && u_fin && (state.g_l_at(i) - state.g_u_at(i)).abs() < 1e-14 {
            continue;
        }
        let s_i = state.s_at(i);
        if l_fin {
            let g_l_i = state.g_l_at(i);
            let s_l = s_i - g_l_i;
            if s_l < s_min {
                let v = state.v_l_at(i);
                let from_mu = if v > 0.0 { mu / v } else { f64::INFINITY };
                let cap = slack_move * g_l_i.abs().max(1.0) + s_l;
                let new_s = from_mu.max(s_min).min(cap);
                state.set_g_l_at(i, g_l_i - (new_s - s_l));
                adjusted += 1;
            }
        }
        if u_fin {
            let g_u_i = state.g_u_at(i);
            let s_u = g_u_i - s_i;
            if s_u < s_min {
                let v = state.v_u_at(i);
                let from_mu = if v > 0.0 { mu / v } else { f64::INFINITY };
                let cap = slack_move * g_u_i.abs().max(1.0) + s_u;
                let new_s = from_mu.max(s_min).min(cap);
                state.set_g_u_at(i, g_u_i + (new_s - s_u));
                adjusted += 1;
            }
        }
    }
    state.adjusted_slacks_count += adjusted;
    adjusted
}

/// A8.9: Plain Ipopt y-update — `state.y[i] += alpha_y * dy_i`.
/// Mirrors `BacktrackingLineSearch::PerformDualStep`
/// (`IpBacktrackingLineSearch.cpp:919-1006`); Ipopt updates y_c,y_d
/// with the raw `α_y · dy` from the KKT solve and has no sign-flip
/// or oscillation damping. The previous ripopt-specific
/// `DyOscillationTracker` heuristic — halving dy when the same
/// component flipped sign 3 times near convergence — was a load-
/// bearing benchmark crutch with no analogue in Ipopt and is
/// removed here.
fn apply_y_update(state: &mut SolverState, alpha_y: f64) {
    for k in 0..state.layout.n_c {
        state.y_c[k] += alpha_y * state.dy_c[k];
    }
    for k in 0..state.layout.n_d {
        state.y_d[k] += alpha_y * state.dy_d[k];
    }
}

/// T3.32: closed-form 1D minimizer of the dual-infeasibility quadratic
/// `phi(α) = ||r_x + α·J^T·dy||² + ||r_s − α·dy_d||²` per Ipopt
/// `IpBacktrackingLineSearch.cpp:969-998`.
///
/// At call time `state.x` is the trial primal point (committed by the
/// line-search), but `state.grad_f` and `state.jac_vals` were
/// evaluated at the pre-step iterate. We re-evaluate both at the
/// trial point into local buffers so the formula uses the same
/// quantities Ipopt does (`grad_lag_x_trial` with current y/z, and
/// `J(x_trial)^T·dy`). Cost: one objective gradient + one Jacobian
/// values evaluation per accepted step. Paid only when this option
/// is selected (default `Primal` does no work here).
///
/// In ripopt's implicit-slack representation, the s-row residual is
///   `r_s_i = -y_i - v_l_i + v_u_i`     for inequality rows i
///   `r_s_i = 0`                          for equality rows i
/// and `dy_d` is the inequality-row slice of `state.dy` (zero on
/// equality rows). The closed-form minimum is `α* = -b/a` clipped to
/// the appropriate interval.
fn compute_min_dual_infeas_alpha<P: NlpProblem>(
    state: &SolverState,
    problem: &P,
    mode: AlphaForY,
    alpha_p: f64,
    alpha_d: f64,
) -> f64 {
    let n = state.n;
    let m = state.m;

    // Evaluate grad_f and Jacobian values at the trial primal point.
    // `state.x` already holds the trial coords (committed pre-call).
    // Phase 10b.4: route through SplitNlp; the adapter's
    // `jacobian_split` collapses the combined→split projection.
    let nlp = crate::split_nlp::SplitNlp::new(problem, &state.layout);
    let mut grad_f_trial = vec![0.0_f64; n];
    if !nlp.gradient(&state.x, true, &mut grad_f_trial) {
        return alpha_p; // graceful fallback
    }
    let mut jac_c_trial = vec![0.0_f64; state.jac_c_vals.len()];
    let mut jac_d_trial = vec![0.0_f64; state.jac_d_vals.len()];
    if !nlp.jacobian_split(
        &state.x,
        false,
        &state.jac_c_combined_idx,
        &state.jac_d_combined_idx,
        &mut jac_c_trial,
        &mut jac_d_trial,
    ) {
        return alpha_p;
    }

    // r_x = grad_lag_x(trial) = grad_f_trial + J(trial)^T · y_curr − z_L + z_U.
    // (The kappa_d damping is omitted here to match Ipopt's formula on
    // line 977 which uses the raw `grad_lag_x` without the damping
    // term — Ipopt resets y_c/y_d to current via `BackupCurrent` at
    // 975-980 then queries `curr_grad_lag_x_amax_func()`.)
    let mut r_x = grad_f_trial.clone();
    for (k, (&kc, &col)) in state.jac_c_rows.iter().zip(state.jac_c_cols.iter()).enumerate() {
        r_x[col] += jac_c_trial[k] * state.y_c[kc];
    }
    for (k, (&kd, &col)) in state.jac_d_rows.iter().zip(state.jac_d_cols.iter()).enumerate() {
        r_x[col] += jac_d_trial[k] * state.y_d[kd];
    }
    // Phase 6d.3: walk compressed bound mirrors. Unbounded sides
    // contribute 0, identical to the n-wide form.
    for k in 0..state.bound_layout.n_x_l {
        let i = state.bound_layout.x_l_to_full[k];
        r_x[i] -= state.z_l_compressed[k];
    }
    for k in 0..state.bound_layout.n_x_u {
        let i = state.bound_layout.x_u_to_full[k];
        r_x[i] += state.z_u_compressed[k];
    }

    // r_s and dy_d are d-block-only (Ipopt's slack rows don't exist on
    // equality constraints). Phase 5e: read directly from split storage.
    // Phase 8c.5: walk compressed v_l/v_u mirrors via Pd_L/Pd_U (the
    // unbounded zero-padded entries contribute nothing).
    let n_d = state.layout.n_d;
    let mut r_s = vec![0.0_f64; n_d];
    for k in 0..n_d {
        r_s[k] = -state.y_d[k];
    }
    for kc in 0..state.d_bound_layout.n_d_l {
        let k = state.d_bound_layout.d_l_to_full[kc];
        r_s[k] -= state.v_l_compressed[kc];
    }
    for kc in 0..state.d_bound_layout.n_d_u {
        let k = state.d_bound_layout.d_u_to_full[kc];
        r_s[k] += state.v_u_compressed[kc];
    }
    let _ = m;

    // Jt_dy = J(trial)^T · dy (split-form; equality contribution from
    // jac_c_trial · dy_c, inequality from jac_d_trial · dy_d).
    let mut jt_dy = vec![0.0_f64; n];
    for (k, (&kc, &col)) in state.jac_c_rows.iter().zip(state.jac_c_cols.iter()).enumerate() {
        jt_dy[col] += jac_c_trial[k] * state.dy_c[kc];
    }
    for (k, (&kd, &col)) in state.jac_d_rows.iter().zip(state.jac_d_cols.iter()).enumerate() {
        jt_dy[col] += jac_d_trial[k] * state.dy_d[kd];
    }

    // a = ||Jt_dy||² + ||dy_d||²
    let a: f64 = jt_dy.iter().map(|v| v * v).sum::<f64>()
        + state.dy_d.iter().map(|v| v * v).sum::<f64>();
    // b = r_x · Jt_dy − r_s · dy_d
    let b: f64 = r_x.iter().zip(jt_dy.iter()).map(|(rx, jd)| rx * jd).sum::<f64>()
        - r_s.iter().zip(state.dy_d.iter()).map(|(rs, dd)| rs * dd).sum::<f64>();

    // Closed-form minimum α* = -b/a. Guard a==0 (Δy entirely zero) by
    // falling back to alpha_p (matches Ipopt's behavior — α_y is
    // irrelevant when dy = 0).
    if a <= 0.0 {
        return alpha_p;
    }
    let alpha_star = -b / a;

    // Clip per mode.
    match mode {
        AlphaForY::MinDualInfeas => alpha_star.clamp(0.0, 1.0),
        AlphaForY::SaferMinDualInfeas => {
            let lo = alpha_p.min(alpha_d);
            let hi = alpha_p.max(alpha_d);
            alpha_star.clamp(lo, hi)
        }
        _ => unreachable!("compute_min_dual_infeas_alpha called with non-min-dual-infeas mode"),
    }
}

fn update_dual_variables<P: NlpProblem>(
    state: &mut SolverState,
    mu_state: &MuState,
    alpha_dual_max: f64,
    options: &SolverOptions,
    problem: &P,
) -> f64 {
    // T3.32: pick alpha_y per `alpha_for_y` option, matching Ipopt
    // IpBacktrackingLineSearch.cpp:84-104 (simple modes) and
    // :969-998 (closed-form 1D minimizer for MinDualInfeas variants).
    let alpha_p = state.alpha_primal;
    let alpha_d = alpha_dual_max;
    // DEV-32: PRIMAL_AND_FULL / DUAL_AND_FULL switch to the full step
    // when the *primal step infinity norm* is small, not when the
    // primal/dual step length is large. Per Ipopt
    // IpBacktrackingLineSearch.cpp:937-958 and the option
    // documentation at line 95-96/103-104:
    //   dxnorm = max(|delta_x|_inf, |delta_s|_inf)
    //   if dxnorm <= alpha_for_y_tol → alpha_y = 1
    let dxnorm = || -> f64 {
        let dx_inf = state.dx.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let ds_inf = state.ds.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        dx_inf.max(ds_inf)
    };
    let alpha_y = match options.alpha_for_y {
        AlphaForY::Primal => alpha_p,
        AlphaForY::BoundMult => alpha_d,
        AlphaForY::Min => alpha_p.min(alpha_d),
        AlphaForY::Max => alpha_p.max(alpha_d),
        AlphaForY::Full => 1.0,
        AlphaForY::PrimalAndFull => {
            if dxnorm() <= options.alpha_for_y_tol { 1.0 } else { alpha_p }
        }
        AlphaForY::DualAndFull => {
            if dxnorm() <= options.alpha_for_y_tol { 1.0 } else { alpha_d }
        }
        AlphaForY::MinDualInfeas | AlphaForY::SaferMinDualInfeas => {
            compute_min_dual_infeas_alpha(state, problem, options.alpha_for_y, alpha_p, alpha_d)
        }
    };

    apply_y_update(state, alpha_y);

    // Ipopt post-step order (IpIpoptAlg.cpp:652-770):
    //   1. Advance z to trial step.
    //   2. slack_move on trial (slack, z) pair (mutates x_l/x_u).
    //   3. kappa_sigma clamps z using the post-slack-move slacks.
    advance_z_to_trial(state, alpha_d);

    let n_adjusted = apply_slack_move(state, options);
    if n_adjusted > 0 && options.print_level >= 6 {
        eprintln!(
            "  [slack_move] iter={}: adjusted {} bound(s) (cumulative {})",
            state.iter, n_adjusted, state.adjusted_slacks_count
        );
    }

    let mu_ks = apply_kappa_sigma_bound_multiplier_reset(state, mu_state);

    state.alpha_dual = alpha_d;
    mu_ks
}

/// Watchdog control-flow decision returned after update.
enum WatchdogDecision {
    Proceed,
    Continue,
}

/// Watchdog (Chamberlain et al. 1982) state: shortened-step counter,
/// activation flag, trial counter, and the snapshot of the iterate
/// taken on activation. Threaded through try_activate_watchdog,
/// process_watchdog_trial, and update_watchdog.
struct Watchdog {
    consecutive_shortened: usize,
    active: bool,
    trial_count: usize,
    saved: Option<WatchdogSavedState>,
}

impl Watchdog {
    fn new() -> Self {
        Self {
            consecutive_shortened: 0,
            active: false,
            trial_count: 0,
            saved: None,
        }
    }

    fn deactivate(&mut self) {
        self.active = false;
        self.trial_count = 0;
        self.saved = None;
    }
}

/// Activate the watchdog if `consecutive_shortened` has hit the trigger
/// threshold. Snapshots the full iterate (x, multipliers, mu, obj, g,
/// grad_f, filter entries, theta, phi) into `wd.saved` so the trial
/// check can revert if no progress materializes.
fn try_activate_watchdog(
    state: &mut SolverState,
    options: &SolverOptions,
    iteration: usize,
    filter: &Filter,
    wd: &mut Watchdog,
) {
    if wd.active
        || wd.consecutive_shortened < options.watchdog_shortened_iter_trigger
    {
        return;
    }
    state.diagnostics.watchdog_activations += 1;
    wd.active = true;
    wd.trial_count = 0;
    let wd_theta = state.constraint_violation();
    let wd_phi = state.barrier_objective(options);
    wd.saved = Some(WatchdogSavedState::snapshot(state, filter, wd_theta, wd_phi));
    wd.consecutive_shortened = 0;
    log::debug!(
        "Watchdog activated at iteration {} (theta={:.2e}, phi={:.2e})",
        iteration, wd_theta, wd_phi
    );
}

/// Run one watchdog trial check after a step. If the saved point's
/// theta/phi has been improved (filter-acceptable, with strict relative
/// thresholds) the watchdog deactivates. If the trial budget
/// (`watchdog_trial_iter_max`) is exhausted, restore the saved iterate
/// and filter, augment the filter with the current point, and signal
/// `WatchdogDecision::Continue` so the main loop retries from the
/// reverted state.
fn process_watchdog_trial<P: NlpProblem>(
    state: &mut SolverState,
    problem: &P,
    options: &SolverOptions,
    filter: &mut Filter,
    lbfgs_state: &mut Option<LbfgsIpmState>,
    wd: &mut Watchdog,
    linear_constraints: Option<&[bool]>,
    lbfgs_mode: bool,
) -> Option<WatchdogDecision> {
    if !wd.active {
        return None;
    }
    wd.trial_count += 1;
    let saved = wd.saved.as_ref()?;
    let theta_now = state.constraint_violation();
    let phi_now = state.barrier_objective(options);
    // Ipopt-aligned watchdog progress check: the trial iterate must be
    // acceptable to the saved filter and provide sufficient decrease
    // against the saved (theta, phi) snapshot using the same
    // gamma_theta / gamma_phi margins the filter line search uses.
    // Reference: Ipopt's IpBacktrackingLineSearch only retains the
    // watchdog while DoBacktrackingLineSearch accepts; that acceptance
    // is the filter sufficient-progress test relative to the watchdog
    // reference iterate.
    let gamma_theta = filter.gamma_theta();
    let gamma_phi = filter.gamma_phi();
    let made_progress = filter.is_acceptable(theta_now, phi_now)
        && (theta_now <= (1.0 - gamma_theta) * saved.theta
            || phi_now <= saved.phi - gamma_phi * saved.theta);

    if made_progress {
        log::debug!(
            "Watchdog succeeded at trial {} (theta: {:.2e} -> {:.2e})",
            wd.trial_count, saved.theta, theta_now
        );
        wd.deactivate();
        return None;
    }
    if wd.trial_count >= options.watchdog_trial_iter_max {
        log::debug!(
            "Watchdog reverting after {} trials",
            wd.trial_count
        );
        let theta_now = state.constraint_violation();
        let phi_now = state.barrier_objective(options);
        filter.restore_entries(saved.filter_entries.clone());
        filter.add(theta_now, phi_now);

        saved.restore(state);
        let _ = evaluate_and_refresh_lbfgs(state, problem, lbfgs_state, linear_constraints, lbfgs_mode);

        wd.deactivate();
        return Some(WatchdogDecision::Continue);
    }
    None
}

/// Maintain the watchdog state after an accepted step.
///
/// - Track consecutive shortened steps
///   (`alpha_primal < 0.99 * alpha_primal_max`).
/// - Activate the watchdog and snapshot state when the shortened-step
///   counter hits `watchdog_shortened_iter_trigger`.
/// - If the watchdog is active, check for sufficient progress versus
///   the saved point; on success deactivate, on exhaustion
///   (`watchdog_trial_iter_max`) revert to the saved state and
///   request `Continue`.
///
/// Reference: Chamberlain et al. (1982); Ipopt's
/// `IpBacktrackingLineSearch::DoBacktrackingLineSearch` watchdog.
fn update_watchdog<P: NlpProblem>(
    state: &mut SolverState,
    problem: &P,
    options: &SolverOptions,
    iteration: usize,
    alpha_primal_max: f64,
    filter: &mut Filter,
    lbfgs_state: &mut Option<LbfgsIpmState>,
    wd: &mut Watchdog,
    linear_constraints: Option<&[bool]>,
    lbfgs_mode: bool,
) -> WatchdogDecision {
    if state.alpha_primal < alpha_primal_max * 0.99 {
        wd.consecutive_shortened += 1;
    } else {
        wd.consecutive_shortened = 0;
    }

    try_activate_watchdog(state, options, iteration, filter, wd);

    if let Some(decision) = process_watchdog_trial(
        state, problem, options, filter, lbfgs_state, wd,
        linear_constraints, lbfgs_mode,
    ) {
        return decision;
    }

    WatchdogDecision::Proceed
}

/// Post-step "acceptable" tracking. Mirrors the pre-step acceptable check
/// (see `track_consecutive_acceptable`) but is evaluated at the freshly
/// accepted iterate — catches cases where the step just taken pushes
/// the state into the acceptable region but the pre-step check at the
/// top of the loop missed it.
///
/// Increments `state.consecutive_acceptable` iff the accepted iterate
/// passes both the scaled (`options.tol`) and unscaled
/// (`constr_viol_tol` / `dual_inf_tol` / `compl_inf_tol`) Ipopt
/// acceptable thresholds. Never resets the counter here; pre-step
/// handling owns resets.
fn track_post_step_acceptable(state: &mut SolverState, options: &SolverOptions) {
    let n = state.n;
    let post_primal = state.constraint_violation();
    let (post_zl_opt, post_zu_opt) = {
        let mut gj = state.grad_f.clone();
        accumulate_jt_y(state, &mut gj);
        recover_active_set_z(state, &gj, n)
    };
    let post_du = dual_inf_with_z(state, &post_zl_opt, &post_zu_opt);
    let post_du_unsc = compute_dual_inf_unscaled_at_state(state);
    let post_compl = compute_compl_err_at_state(state);
    let post_compl_opt = compl_err_with_z(state, &post_zl_opt, &post_zu_opt);
    let post_compl_best = post_compl.min(post_compl_opt);
    let post_sd = compute_s_d_at_state(state);
    let post_near_scaled = post_primal <= 100.0 * options.tol
        && post_du <= 100.0 * options.tol * post_sd
        && post_compl_best <= 100.0 * options.tol * post_sd;
    let post_near_unscaled = post_primal <= 10.0 * options.constr_viol_tol
        && post_du_unsc <= 10.0 * options.dual_inf_tol
        && post_compl_best <= 10.0 * options.compl_inf_tol;
    if post_near_scaled && post_near_unscaled {
        state.consecutive_acceptable += 1;
    }
}

/// Outcome of the post-LS restoration cascade.
enum RestorationCascadeDecision {
    Continue,
    Return(SolveResult),
}

/// Attempt full NLP restoration on `fail_count ∈ {2, 4}` (skipped if
/// disabled or KKT dim > 50000). Returns true when restoration
/// succeeded and the cascade should short-circuit with `Continue`; false
/// when the caller should fall through to the recovery / max-attempts
/// classification.
#[allow(clippy::too_many_arguments)]
fn try_nlp_restoration_phase<P: NlpProblem>(
    state: &mut SolverState,
    problem: &P,
    options: &SolverOptions,
    filter: &mut Filter,
    mu_state: &mut MuState,
    lbfgs_state: &mut Option<LbfgsIpmState>,
    lbfgs_mode: bool,
    linear_constraints: Option<&[bool]>,
    _iteration: usize,
    fail_count: usize,
    n: usize,
    m: usize,
    start_time: Instant,
    theta_current: f64,
) -> bool {
    // Ipopt 3.14 invokes RestoPhase on the FIRST post-soft-resto
    // line-search failure (`IpBacktrackingLineSearch.cpp:558-623`).
    // There is no parity / fail_count gate in the reference. The
    // earlier `fail_count == 2 || 4` pattern was a ripopt-specific
    // heuristic that delayed restoration so mu-jitter recovery could
    // mask the failure — observed on arki0003 to keep the solver
    // running for 300+ iters past the point where Ipopt enters
    // restoration (iter ~110). Gate removed.
    let _ = fail_count;
    let kkt_dim = n + m;
    if options.disable_nlp_restoration || kkt_dim > 50000 {
        return false;
    }

    state.diagnostics.nlp_restoration_count += 1;
    let (x_nlp, resto_z, outcome) = attempt_nlp_restoration(
        problem, state, filter, options, theta_current, start_time,
    );
    match outcome {
        RestorationOutcome::Success => {
            // T0.9: apply_restoration_success now filter-gates the
            // restored iterate. If the filter rejects (theta_new,
            // phi_new), commit nothing and fall through to recovery
            // — matches Ipopt RestoFilterConvCheck::TestOrigProgress
            // returning CONTINUE instead of CONVERGED.
            apply_restoration_success(
                state, filter, mu_state, options, n, m,
                problem, &x_nlp, resto_z.as_ref(),
                linear_constraints, lbfgs_mode, lbfgs_state,
            )
        }
        RestorationOutcome::LocalInfeasibility | RestorationOutcome::Failed => {
            // Fall through to continue recovery. Don't immediately return
            // LocalInfeasibility — the fail_count > 6 stationarity check
            // is more reliable.
            false
        }
    }
}

/// Adjust mu / mu-mode and (on attempt 3+) jitter x to escape the current
/// basin after restoration failed but the cascade has not exhausted its
/// retry budget. fail_count == 1 switches Free → Fixed mode (or applies
/// the standard mu_linear/superlinear decrease in Fixed mode); subsequent
/// attempts cycle through the perturbation sequence ×10, ×0.1, ×100,
/// ×0.01, ×1000, ×0.001. The filter is reset and inertia δ_w cleared.
#[allow(clippy::too_many_arguments)]
fn apply_restoration_recovery_strategy<P: NlpProblem>(
    state: &mut SolverState,
    _problem: &P,
    options: &SolverOptions,
    filter: &mut Filter,
    mu_state: &mut MuState,
    inertia_params: &mut InertiaCorrectionParams,
    _lbfgs_state: &mut Option<LbfgsIpmState>,
    _lbfgs_mode: bool,
    _linear_constraints: Option<&[bool]>,
    fail_count: usize,
    _n: usize,
) {
    log::debug!("Restoration failed (attempt #{}), trying recovery", fail_count);
    let mu_factors: [f64; 6] = [10.0, 0.1, 100.0, 0.01, 1000.0, 0.001];

    match fail_count {
        1 => apply_first_restoration_failure_mu_update(state, mu_state, options),
        _ => {
            let factor = mu_factors[(fail_count - 2) % mu_factors.len()];
            state.mu = (state.mu * factor).max(options.mu_min).min(1e5);
        }
    }
    reset_filter_with_current_theta(state, filter);
    inertia_params.delta_w_last = 0.0;
    // T3.25 follow-up: μ change rescales sigma which feeds the Hessian
    // (1,1) block diagonal — the matrix differs from the cached one.
    // Belt-and-braces: bump atags AND drop the cache.
    state.bump_all_kkt_atags();
    state.factor_cache.invalidate();
}

/// First-restoration-failure μ update. In Free mode, switch to
/// Fixed and seed μ from `adaptive_mu_monotone_init_factor·avg_compl`
/// (clamped to `[μ_min, 1e5]`), or fall back to a linear decrease
/// when no active complementarity products exist. In Fixed mode,
/// drop μ by `min(linear, superlinear)` rate. Subsequent failures
/// use the cycling `mu_factors` table in the parent.
fn apply_first_restoration_failure_mu_update(
    state: &mut SolverState,
    mu_state: &mut MuState,
    options: &SolverOptions,
) {
    if mu_state.mode == MuMode::Free {
        switch_mu_mode(state, mu_state, MuMode::Fixed);
        let avg_compl = compute_avg_complementarity(state);
        if avg_compl > 0.0 {
            state.mu = (options.adaptive_mu_monotone_init_factor * avg_compl)
                .clamp(options.mu_min, 1e5);
        } else {
            state.mu = (options.mu_linear_decrease_factor * state.mu)
                .max(options.mu_min);
        }
    } else {
        let new_mu = (options.mu_linear_decrease_factor * state.mu)
            .min(state.mu.powf(options.mu_superlinear_decrease_power))
            .max(options.mu_min);
        state.mu = new_mu;
    }
}


/// Classify the terminal status when the restoration cascade has exhausted its
/// retry budget. Returns LocalInfeasibility when the constraint-violation
/// gradient is near-stationary at an infeasible point, Infeasible when theta
/// has stayed > 1% of its historical minimum well into the run, otherwise
/// RestorationFailed. Caller must only invoke this once `fail_count` exceeds
/// `max_restore_attempts`.
fn classify_exhausted_restoration_attempt(
    state: &SolverState,
    options: &SolverOptions,
    iteration: usize,
    fail_count: usize,
    feas: &FeasibilityTracker,
) -> SolveResult {
    log::warn!(
        "Restoration failed at iteration {} (attempt #{})",
        iteration, fail_count
    );
    let current_theta = state.constraint_violation();

    if current_theta > options.constr_viol_tol && !feas.ever_feasible {
        let grad_theta_norm = compute_grad_theta_norm(state);
        let stationarity_tol = 1e-4 * current_theta.max(1.0);
        if grad_theta_norm < stationarity_tol {
            log::info!(
                "Local infeasibility detected: theta={:.2e}, ||∇theta||={:.2e}",
                current_theta, grad_theta_norm
            );
            return make_result(state, SolveStatus::LocalInfeasibility);
        }
    }

    if !feas.ever_feasible && current_theta > 1e4 && iteration > 500
        && feas.history.len() >= feas.history_len
    {
        let min_theta = slice_min(&feas.history);
        if current_theta > 0.01 * min_theta {
            return make_result(state, SolveStatus::Infeasible);
        }
    }
    make_result(state, SolveStatus::RestorationFailed)
}

/// Invoke the restoration cascade after a failed line search. Runs the
/// fast Gauss–Newton restoration first; on failure, rotates through
/// (a) NLP restoration at `fail_count ∈ {2, 4}`,
/// (b) infeasibility / `RestorationFailed` / `MaxIterations` exits,
/// (c) mu-mode switching and mu perturbation,
/// (d) x-perturbation on `fail_count >= 3`.
///
/// Always ends with either `Continue` (the main loop should re-enter its
/// top with an updated iterate / recovery state) or `Return(SolveResult)`
/// (the solver is terminating).
#[allow(clippy::too_many_arguments)]
fn run_post_ls_restoration_cascade<P: NlpProblem>(
    state: &mut SolverState,
    problem: &P,
    options: &SolverOptions,
    filter: &mut Filter,
    mu_state: &mut MuState,
    inertia_params: &mut InertiaCorrectionParams,
    lbfgs_state: &mut Option<LbfgsIpmState>,
    lbfgs_mode: bool,
    linear_constraints: Option<&[bool]>,
    iteration: usize,
    n: usize,
    m: usize,
    start_time: Instant,
    _deadline: Option<Instant>,
    feas: &FeasibilityTracker,
    theta_current: f64,
) -> RestorationCascadeDecision {
    state.diagnostics.filter_rejects += 1;

    // Cascade is the escalation point past soft restoration; reset the soft
    // counter so a future post-cascade soft attempt starts fresh
    // (Ipopt's `IpBacktrackingLineSearch.cpp:442-444` resets on any
    // non-soft accept).
    mu_state.consecutive_soft_restoration = 0;

    // Filter augmentation happened at the line-search rejection site
    // (matching IpBacktrackingLineSearch.cpp:566 — PrepareRestoPhaseStart
    // augments before the almost-feasible guard, not inside the cascade).

    log::debug!("Line search failed at iteration {}, entering restoration", iteration);

    // Restoration NLP (L1-penalty Ipopt path). Recovery logic.
    // Bail out of recovery cascade if wall time is nearly exhausted.
    if options.max_wall_time > 0.0 {
        let remaining = options.max_wall_time - start_time.elapsed().as_secs_f64();
        if remaining < 1.0 {
            return RestorationCascadeDecision::Return(make_result(state, SolveStatus::MaxIterations));
        }
    }

    mu_state.consecutive_restoration_failures += 1;
    let fail_count = mu_state.consecutive_restoration_failures;

    if try_nlp_restoration_phase(
        state, problem, options, filter, mu_state, lbfgs_state, lbfgs_mode,
        linear_constraints, iteration, fail_count, n, m, start_time,
        theta_current,
    ) {
        return RestorationCascadeDecision::Continue;
    }

    // For large problems (no NLP restoration), give up sooner.
    let kkt_dim = n + m;
    let max_restore_attempts = if kkt_dim > 50000 { 3 } else { 6 };
    if fail_count > max_restore_attempts {
        return RestorationCascadeDecision::Return(classify_exhausted_restoration_attempt(
            state, options, iteration, fail_count, feas,
        ));
    }

    apply_restoration_recovery_strategy(
        state, problem, options, filter, mu_state, inertia_params,
        lbfgs_state, lbfgs_mode, linear_constraints, fail_count, n,
    );

    RestorationCascadeDecision::Continue
}

/// Recover from a post-step evaluation failure by halving α (and
/// α_dual) up to 5 times, retracting `state.x` toward `x_pre_step`
/// each time, and re-evaluating the problem. Returns `true` on the
/// first α at which evaluation succeeds (with `state.alpha_primal`,
/// `state.alpha_dual`, and the L-BFGS Hessian refreshed); `false`
/// when all 5 halvings fail. Mirrors Ipopt's
/// `IpBacktrackingLineSearch.cpp:776-784` post-step Eval_Error
/// backtrack.
fn try_alpha_halving_post_step_recovery<P: NlpProblem>(
    state: &mut SolverState,
    problem: &P,
    lbfgs_state: &mut Option<LbfgsIpmState>,
    x_pre_step: &[f64],
    linear_constraints: Option<&[bool]>,
    lbfgs_mode: bool,
    n: usize,
) -> bool {
    let mut retry_alpha = state.alpha_primal;
    let mut retry_alpha_dual = state.alpha_dual;
    for _ in 0..5 {
        retry_alpha *= 0.5;
        retry_alpha_dual *= 0.5;
        for i in 0..n {
            state.x[i] = x_pre_step[i] + retry_alpha * state.dx[i];
        }
        if state.evaluate_with_linear(problem, 1.0, linear_constraints, lbfgs_mode) {
            state.alpha_primal = retry_alpha;
            state.alpha_dual = retry_alpha_dual;
            update_lbfgs_hessian(lbfgs_state, state);
            return true;
        }
    }
    false
}

/// Outcome of the post-step re-evaluation with α-halving recovery.
enum PostStepEvalDecision {
    Proceed,
    Continue,
    Return(SolveResult),
}

/// Re-evaluate the problem at the accepted iterate and, on evaluation
/// failure, execute the Ipopt-style α-halving → restoration recovery
/// cascade.
///
/// Ipopt's `IpBacktrackingLineSearch.cpp:776-784` treats a post-step
/// `Eval_Error` as an α backtrack rather than a fatal. Mirror that by
/// halving α and α_dual (from the accepted point back toward
/// `x_pre_step`) up to 5 times; if that fails, return `NumericalError`.
/// (The L1-penalty restoration NLP is invoked separately from the
/// post-line-search cascade — Ipopt does NOT invoke restoration from
/// the post-step Eval_Error path either.)
///
/// On success (or successful recovery via α-halving) returns `Proceed`
/// / `Continue` for the main loop's control flow.
fn reevaluate_after_step<P: NlpProblem>(
    state: &mut SolverState,
    problem: &P,
    _options: &SolverOptions,
    lbfgs_state: &mut Option<LbfgsIpmState>,
    _filter: &mut Filter,
    timings: &mut PhaseTimings,
    x_pre_step: &[f64],
    linear_constraints: Option<&[bool]>,
    lbfgs_mode: bool,
    _deadline: Option<Instant>,
) -> PostStepEvalDecision {
    let n = state.n;
    let t_eval = Instant::now();
    let eval_ok = state.evaluate_with_linear(problem, 1.0, linear_constraints, lbfgs_mode);
    if eval_ok {
        update_lbfgs_hessian(lbfgs_state, state);
    }
    timings.problem_eval += t_eval.elapsed();

    if eval_ok {
        return PostStepEvalDecision::Proceed;
    }

    if try_alpha_halving_post_step_recovery(
        state, problem, lbfgs_state, x_pre_step,
        linear_constraints, lbfgs_mode, n,
    ) {
        return PostStepEvalDecision::Continue;
    }

    PostStepEvalDecision::Return(make_result(state, SolveStatus::NumericalError))
}

/// Compute the per-component "magic step" delta for an explicit slack
/// vector `s` against constraint values `d` and finite slack bounds
/// `[d_L, d_U]`. Mirrors Ipopt 3.14
/// `BacktrackingLineSearch::PerformMagicStep`
/// (`IpBacktrackingLineSearch.cpp:1013-1111`).
///
/// The magic step minimizes the constraint residual `d - s` along the
/// `s` coordinate while holding `x` (and therefore `d`) and all
/// multipliers fixed. For each component `i`:
///
/// - If `i` has only a lower bound (`d_L[i]` finite, `d_U[i]` not):
///   `delta_i = max(0, d_i - s_i)` — push `s` up if `d > s`.
/// - If `i` has only an upper bound (`d_U[i]` finite, `d_L[i]` not):
///   `delta_i = min(0, d_i - s_i)` — push `s` down if `d < s`.
/// - If `i` has both bounds: take the candidate `delta_i`, then
///   suppress (zero out) when the candidate would *not* reduce
///   `|d_L + d_U - 2 s|` (the symmetric centering measure used by Ipopt
///   to avoid pushing `s` against the opposite bound).
///
/// `delta` is written component-wise. `s_in` is read-only; the caller
/// applies `s_new = s_in + delta`. Returns the number of strictly
/// non-zero components in `delta`.
///
/// The helper is generic over slack representation: it accepts plain
/// slices of the slack value, constraint value, and bounds, so it can
/// be reused by both ripopt's `SlackFormulation` (where slacks are
/// appended to `x`) and any future explicit-slack path. ripopt's
/// standard implicit-slack mode has no `s` distinct from `x`, so the
/// helper is not invoked there. Marked `allow(dead_code)` outside test
/// builds because the only current caller is the unit tests; the
/// helper is intentionally retained as the wiring point for future
/// explicit-slack code paths (T2.24, spec §5.3).
#[cfg_attr(not(test), allow(dead_code))]
fn compute_magic_step_delta(
    s: &[f64],
    d: &[f64],
    d_l: &[f64],
    d_u: &[f64],
    delta: &mut [f64],
) -> usize {
    let m = s.len();
    debug_assert_eq!(d.len(), m);
    debug_assert_eq!(d_l.len(), m);
    debug_assert_eq!(d_u.len(), m);
    debug_assert_eq!(delta.len(), m);
    let mut nnz = 0usize;
    for i in 0..m {
        let has_l = d_l[i].is_finite();
        let has_u = d_u[i].is_finite();
        let r = d[i] - s[i]; // residual we'd like to drive to zero
        let cand = if has_l && has_u {
            // candidate is the unbounded magic step (push s toward d):
            //   max(0, r) along the lower side, min(0, r) along the upper side.
            // For doubly-bounded entries, Ipopt then suppresses the step
            // when |d_L + d_U - 2*(s + cand)| > |d_L + d_U - 2*s|.
            let lower_part = if r > 0.0 { r } else { 0.0 };
            let upper_part = if r < 0.0 { r } else { 0.0 };
            let c = lower_part + upper_part; // exactly one of these is non-zero
            let center_now = (d_l[i] + d_u[i] - 2.0 * s[i]).abs();
            let center_after = (d_l[i] + d_u[i] - 2.0 * (s[i] + c)).abs();
            if center_after <= center_now { c } else { 0.0 }
        } else if has_l {
            if r > 0.0 { r } else { 0.0 }
        } else if has_u {
            if r < 0.0 { r } else { 0.0 }
        } else {
            0.0
        };
        delta[i] = cand;
        if cand != 0.0 {
            nnz += 1;
        }
    }
    nnz
}

/// Apply Ipopt 3.14's magic step (spec §5.3,
/// `IpBacktrackingLineSearch.cpp:1013-1111`) to the explicit
/// inequality-constraint slack vector `s`, holding `x` and all
/// multipliers fixed.
///
/// **ripopt no-op.** ripopt uses an implicit-slack formulation (see
/// `.crucible/wiki/concepts/implicit-slack-formulation.org`): there is
/// no slack vector `s` in `SolverState` — the inequality side is
/// represented by `g(x)` directly with `v_l`, `v_u` carrying the
/// barrier multipliers. The magic step's degree of freedom (move `s`
/// while holding `x` fixed) does not exist in this representation, so
/// this function is a no-op on the standard solve path. The flag is
/// honored for spec compliance and future explicit-slack paths
/// (`slack_formulation.rs`); the closed-form delta is implemented in
/// [`compute_magic_step_delta`] and tested independently.
///
/// Returns the number of slack components updated (always 0 in the
/// standard implicit-slack path).
fn apply_magic_step(_state: &mut SolverState, options: &SolverOptions) -> usize {
    if !options.magic_step {
        return 0;
    }
    // ripopt has no explicit `s` to adjust here — the implicit-slack
    // formulation embeds the slack value in `g(x)`, which would change
    // `x` if mutated. The helper `compute_magic_step_delta` provides
    // the Ipopt-faithful per-component formula for callers that hold
    // an explicit slack representation (e.g. `SlackFormulation`).
    0
}

/// Cap on consecutive accepted soft-restoration iterates before forcing
/// the full GN/NLP restoration cascade. Matches Ipopt's
/// `max_soft_resto_iters = 10` (`IpBacktrackingLineSearch.cpp:442-444`).
const MAX_SOFT_RESTO_ITERS: usize = 10;

/// Snapshot of the iterate fields mutated by [`attempt_soft_restoration`]
/// so a rejected trial can be rolled back cleanly. Keeps allocations
/// off the success path that would otherwise dominate the helper.
struct SoftRestoSnapshot {
    x: Vec<f64>,
    y_c: Vec<f64>,
    y_d: Vec<f64>,
    /// Phase 6d.2: native compressed bound multipliers.
    z_l_compressed: Vec<f64>,
    z_u_compressed: Vec<f64>,
    s: Vec<f64>,
    obj: f64,
    c_x: Vec<f64>,
    d_x: Vec<f64>,
    grad_f: Vec<f64>,
    jac_c_vals: Vec<f64>,
    jac_d_vals: Vec<f64>,
    alpha_primal: f64,
}
impl SoftRestoSnapshot {
    fn take(state: &SolverState) -> Self {
        Self {
            x: state.x.clone(),
            y_c: state.y_c.clone(),
            y_d: state.y_d.clone(),
            z_l_compressed: state.z_l_compressed.clone(),
            z_u_compressed: state.z_u_compressed.clone(),
            s: state.s.clone(),
            obj: state.obj,
            c_x: state.c_x.clone(),
            d_x: state.d_x.clone(),
            grad_f: state.grad_f.clone(),
            jac_c_vals: state.jac_c_vals.clone(),
            jac_d_vals: state.jac_d_vals.clone(),
            alpha_primal: state.alpha_primal,
        }
    }
    fn restore(self, state: &mut SolverState) {
        state.x = self.x;
        state.y_c = self.y_c;
        state.y_d = self.y_d;
        state.z_l_compressed = self.z_l_compressed;
        state.z_u_compressed = self.z_u_compressed;
        state.s = self.s;
        state.obj = self.obj;
        state.c_x = self.c_x;
        state.d_x = self.d_x;
        state.grad_f = self.grad_f;
        state.jac_c_vals = self.jac_c_vals;
        state.jac_d_vals = self.jac_d_vals;
        state.alpha_primal = self.alpha_primal;
        // T3.25: snapshot restore touches every tracked input.
        state.bump_all_kkt_atags();
    }
}

/// Ipopt's `TrySoftRestoStep` (`IpBacktrackingLineSearch.cpp:1113-1217`):
/// before invoking the expensive GN/NLP restoration cascade, take the
/// computed search direction at `α = min(α_primal_max, α_dual_max)` for
/// primals AND duals, re-evaluate the NLP, and accept if either
///
///   - the parent filter accepts the trial (θ_trial, φ_trial), OR
///   - the averaged primal-dual barrier error drops:
///     `E_μ(trial) ≤ 0.9999 · E_μ(current)`
///
/// Up to `MAX_SOFT_RESTO_ITERS` consecutive soft accepts are allowed
/// before the cascade is forced. Returns `true` iff the step was taken
/// (in which case the iterate already reflects the trial and the
/// counter has been advanced).
#[allow(clippy::too_many_arguments)]
fn attempt_soft_restoration<P: NlpProblem>(
    state: &mut SolverState,
    problem: &P,
    options: &SolverOptions,
    filter: &mut Filter,
    mu_state: &mut MuState,
    linear_constraints: Option<&[bool]>,
    lbfgs_mode: bool,
    alpha_primal_max: f64,
    alpha_dual_max: f64,
    theta_current: f64,
    phi_current: f64,
) -> bool {
    if mu_state.consecutive_soft_restoration >= MAX_SOFT_RESTO_ITERS {
        return false;
    }

    let n = state.n;
    let m = state.m;
    let alpha_p = alpha_primal_max.min(alpha_dual_max);
    if !alpha_p.is_finite() || alpha_p <= 0.0 {
        return false;
    }
    let alpha_d = alpha_dual_max;

    let pderror_curr = compute_pderror_e_mu(state, state.mu);

    let snapshot = SoftRestoSnapshot::take(state);

    // Step primals (clamped to the open box) and duals together.
    let x_trial = compute_clamped_trial_x(state, &state.dx, alpha_p);
    state.x = x_trial;
    for i in 0..m {
        let v = state.y_at(i) + alpha_d * state.dy_at(i);
        state.set_y_at(i, v);
    }
    // Phase 6d.6: compressed bound storage is canonical. Note the
    // soft-resto floor is `0.0`, not `1e-20`.
    for k in 0..state.bound_layout.n_x_l {
        state.z_l_compressed[k] = (state.z_l_compressed[k]
            + alpha_d * state.dz_l_compressed[k]).max(0.0);
    }
    for k in 0..state.bound_layout.n_x_u {
        state.z_u_compressed[k] = (state.z_u_compressed[k]
            + alpha_d * state.dz_u_compressed[k]).max(0.0);
    }
    // T3.25: soft-resto trial step writes x and duals directly (does
    // not go through commit_trial_point), so bump atags ourselves.
    state.bump_all_kkt_atags();

    // Re-evaluate obj + grad + constraints + jac (skip Hessian — soft test
    // only inspects gradient-level info).
    let eval_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.evaluate_with_linear(problem, 1.0, linear_constraints, true)
    }));
    let evaluated = matches!(eval_ok, Ok(true));
    if !evaluated || !state.obj.is_finite() {
        snapshot.restore(state);
        return false;
    }

    // Soft restoration steps x and the duals but not the slack `s`,
    // so theta is measured against the unchanged `state.s` (A8.19
    // slack-coupled form).
    let theta_trial = theta_for_split_d_s(state, &state.c_x, &state.d_x, &state.s);
    let phi_trial = compute_barrier_phi(
        state.obj, &state.x, &state.s, state, n, m, options.constraint_slack_barrier,
        options.kappa_d,
    );

    // Ipopt's TrySoftRestoStep (IpBacktrackingLineSearch.cpp:1172) calls
    // `acceptor_->CheckAcceptabilityOfTrialPoint(0.)` — the FULL filter
    // check with gBD=0 forcing the h-type branch, which requires either
    // sufficient theta reduction or sufficient phi reduction. Using the
    // weaker `is_acceptable` (filter-domination test only) lets null
    // steps with theta_trial≈theta_current and phi_trial≈phi_current
    // pass, blocking the hard-restoration cascade on stalled iterates
    // (observed on arki0003 iters 110-115 with α≈1e-9).
    let (filter_ok, _) = filter.check_acceptability(
        theta_current, phi_current, theta_trial, phi_trial, 0.0, alpha_p, false,
    );
    let pderror_trial = compute_pderror_e_mu(state, state.mu);
    let pderror_ok = pderror_trial <= 0.9999 * pderror_curr;

    if filter_ok || pderror_ok {
        log::debug!(
            "Soft restoration accepted (filter={} pderror={}): theta {:.2e} -> {:.2e}, E_mu {:.2e} -> {:.2e}",
            filter_ok, pderror_ok, theta_current, theta_trial, pderror_curr, pderror_trial,
        );
        // Suppress lbfgs_mode warning when feature disabled.
        let _ = lbfgs_mode;
        state.alpha_primal = alpha_p;
        mu_state.consecutive_soft_restoration += 1;
        filter.add(theta_current, phi_current);
        true
    } else {
        snapshot.restore(state);
        false
    }
}


/// Fraction-to-boundary step limits for primal and dual.
///
/// `tau` = `max(1 - NLP_error, tau_min)` in Free mode or
/// `max(1 - mu, tau_min)` in Fixed mode. Primal bound:
/// `x + alpha*dx > x_l` and `< x_u` by a factor of `tau`.
/// Dual bound: `z + alpha*dz > 0` by a factor of `tau`.
fn compute_alpha_max(
    state: &SolverState,
    options: &SolverOptions,
    mu_state: &MuState,
    primal_inf: f64,
    dual_inf: f64,
    compl_inf: f64,
) -> (f64, f64, f64) {
    let tau = compute_tau(state, options, mu_state, primal_inf, dual_inf, compl_inf);

    let alpha_primal_max = fraction_to_boundary_primal_x(state, &state.dx, tau)
        .min(fraction_to_boundary_primal_s(state, &state.ds, tau))
        .clamp(0.0, 1.0);

    let alpha_dual_z = fraction_to_boundary_dual_z_min(state, &state.dz_l_compressed, &state.dz_u_compressed, tau);
    // Phase 8c.5: walk compressed v_l/v_u/dv_l/dv_u storage directly.
    let alpha_dual_v = filter::fraction_to_boundary(&state.v_l_compressed, &state.dv_l_compressed, tau)
        .min(filter::fraction_to_boundary(&state.v_u_compressed, &state.dv_u_compressed, tau));
    let alpha_dual_max = alpha_dual_z.min(alpha_dual_v);

    if std::env::var("RIPOPT_TRACE_STEP").is_ok() {
        let dvl_inf = state.dv_l_compressed.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let dvu_inf = state.dv_u_compressed.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let vl_min = state.v_l_compressed.iter().cloned().fold(f64::INFINITY, f64::min);
        let vu_min = state.v_u_compressed.iter().cloned().fold(f64::INFINITY, f64::min);
        eprintln!(
            "  dual-trace: alpha_d_z={:.3e} alpha_d_v={:.3e} | |dv_L|_inf={:.3e} |dv_U|_inf={:.3e} v_L_min={:.3e} v_U_min={:.3e}",
            alpha_dual_z, alpha_dual_v, dvl_inf, dvu_inf, vl_min, vu_min
        );
    }

    if std::env::var("RIPOPT_TRACE_STEP").is_ok() {
        // Identify the variable that limits alpha_primal_max.
        let mut lim_idx: usize = usize::MAX;
        let mut lim_alpha = 1.0f64;
        let mut lim_side = "";
        let mut lim_block = "x";
        for i in 0..state.n {
            if state.x_l_at(i).is_finite() && state.dx[i] < 0.0 {
                let slack = state.x[i] - state.x_l_at(i);
                let a = -tau * slack / state.dx[i];
                if a < lim_alpha {
                    lim_alpha = a;
                    lim_idx = i;
                    lim_side = "L";
                    lim_block = "x";
                }
            }
            if state.x_u_at(i).is_finite() && state.dx[i] > 0.0 {
                let slack = state.x_u_at(i) - state.x[i];
                let a = tau * slack / state.dx[i];
                if a < lim_alpha {
                    lim_alpha = a;
                    lim_idx = i;
                    lim_side = "U";
                    lim_block = "x";
                }
            }
        }
        for i in 0..state.m {
            let l_fin = state.g_l_at(i).is_finite();
            let u_fin = state.g_u_at(i).is_finite();
            if l_fin && u_fin && (state.g_l_at(i) - state.g_u_at(i)).abs() < 1e-14 {
                continue;
            }
            let ds_i = state.ds_at(i);
            let s_i = state.s_at(i);
            if l_fin && ds_i < 0.0 {
                let slack = (s_i - state.g_l_at(i)).max(0.0);
                let a = -tau * slack / ds_i;
                if a < lim_alpha {
                    lim_alpha = a;
                    lim_idx = i;
                    lim_side = "L";
                    lim_block = "s";
                }
            }
            if u_fin && ds_i > 0.0 {
                let slack = (state.g_u_at(i) - s_i).max(0.0);
                let a = tau * slack / ds_i;
                if a < lim_alpha {
                    lim_alpha = a;
                    lim_idx = i;
                    lim_side = "U";
                    lim_block = "s";
                }
            }
        }
        let dx_inf = state.dx.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let dy_inf = state
            .dy_c
            .iter()
            .chain(state.dy_d.iter())
            .fold(0.0f64, |a, &b| a.max(b.abs()));
        let dzl_inf = state.dz_l_compressed.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let dzu_inf = state.dz_u_compressed.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        if lim_idx != usize::MAX {
            let (xv, xb, dxv, slack) = if lim_block == "s" {
                let xv = state.s_at(lim_idx);
                let xb = if lim_side == "L" { state.g_l_at(lim_idx) } else { state.g_u_at(lim_idx) };
                (xv, xb, state.ds_at(lim_idx), (xv - xb).abs())
            } else {
                let xv = state.x[lim_idx];
                let xb = if lim_side == "L" { state.x_l_at(lim_idx) } else { state.x_u_at(lim_idx) };
                (xv, xb, state.dx[lim_idx], (xv - xb).abs())
            };
            eprintln!(
                "  step-trace: tau={:.4} alpha_p_max={:.3e} alpha_d_max={:.3e} | lim={}{}@row/var{} val={:.3e} bnd={:.3e} slack={:.3e} d={:.3e} | |dx|_inf={:.3e} |dy|_inf={:.3e} |dz_L|_inf={:.3e} |dz_U|_inf={:.3e}",
                tau, alpha_primal_max, alpha_dual_max, lim_block, lim_side, lim_idx,
                xv, xb, slack, dxv, dx_inf, dy_inf, dzl_inf, dzu_inf
            );
        } else {
            eprintln!(
                "  step-trace: tau={:.4} alpha_p_max={:.3e} alpha_d_max={:.3e} | no primal limiter | |dx|_inf={:.3e} |dy|_inf={:.3e} |dz_L|_inf={:.3e} |dz_U|_inf={:.3e}",
                tau, alpha_primal_max, alpha_dual_max, dx_inf, dy_inf, dzl_inf, dzu_inf
            );
        }
    }

    (tau, alpha_primal_max, alpha_dual_max)
}

/// Ipopt-style tiny-step detection.
///
/// Mirrors `IpBacktrackingLineSearch.cpp::DetectTinyStep` (Ipopt 3.14,
/// lines 1219-1278) plus the latch flow at lines 363-435.
///
/// Detection (returns true ⇔ all three hold):
/// - `max_i |Δx_i| / (1 + |x_i|) ≤ tiny_step_tol` (≈ 10·eps; line 1245)
/// - `max_i |Δs_i| / (1 + |s_i|) ≤ tiny_step_tol` (slack step, line 1261)
/// - `cviol ≤ 1e-4` (line 1270)
///
/// Detection does **not** include the dual step. That gate
/// (`tiny_step_y_tol`, default 1e-2) lives at line 421-424 and only
/// controls the **latch** `tiny_step_last_iter` set after a detection.
///
/// `mu_state.tiny_step` (≡ Ipopt's `tiny_step_flag`, line 410) fires
/// only when **the current iter detected AND the previous iter latched**.
/// `tiny_step_last_iter` is then refreshed for the next iter as
/// `detection && (‖Δy‖_∞ < tiny_step_y_tol)`.
///
/// The actual `STOP_AT_TINY_STEP` exit fires from
/// `update_barrier_parameter` when `tiny_step && new_μ == μ`
/// (`IpMonotoneMuUpdate.cpp:158-160`, `IpAdaptiveMuUpdate.cpp:330-332,377-379`).
/// The main loop consumes the resulting `pending_tiny_step_exit` flag
/// at the *top* of the next iteration, after `check_convergence` runs,
/// so KKT-clean tiny-step iterates still exit `Optimal` first.
///
/// Earlier ripopt versions conflated the dy gate with detection (using
/// it as an AND-condition for the counter increment) and omitted the
/// slack-step check. A8.12 restores Ipopt's separation-of-concerns.
fn detect_tiny_step(
    state: &mut SolverState,
    options: &SolverOptions,
    mu_state: &mut MuState,
    _filter: &mut Filter,
    tiny_step_last_iter: &mut bool,
    primal_inf: f64,
) {
    let n = state.n;
    let m = state.m;
    let tiny_tol = 10.0 * f64::EPSILON;

    // Relative x-step: IpBacktrackingLineSearch.cpp:1232-1248.
    let max_rel_dx: f64 = (0..n)
        .map(|i| state.dx[i].abs() / (state.x[i].abs() + 1.0))
        .fold(0.0f64, f64::max);

    // Relative s-step (slack vars for inequality constraints):
    // IpBacktrackingLineSearch.cpp:1250-1264. Ipopt requires both x
    // and s steps tiny; without this an iterate making real progress
    // only on slacks would be misclassified as tiny.
    let max_rel_ds: f64 = if state.s.is_empty() {
        0.0
    } else {
        state.s.iter()
            .zip(state.ds.iter())
            .map(|(&s_k, &ds_k)| ds_k.abs() / (s_k.abs() + 1.0))
            .fold(0.0f64, f64::max)
    };

    // Detection per IpBacktrackingLineSearch.cpp:1245,1261,1270.
    let detection_tiny =
        max_rel_dx <= tiny_tol && max_rel_ds <= tiny_tol && primal_inf <= 1e-4;

    // Latch gate for next iter: dy norm under tiny_step_y_tol
    // (IpBacktrackingLineSearch.cpp:421-424). Raw Amax, not relative.
    let dy_amax: f64 = if m == 0 {
        0.0
    } else {
        state
            .dy_c
            .iter()
            .chain(state.dy_d.iter())
            .map(|v| v.abs())
            .fold(0.0f64, f64::max)
    };

    // tiny_step_flag (= mu-update exit signal): current iter detection
    // AND previous iter's latch (IpBacktrackingLineSearch.cpp:407-411).
    mu_state.tiny_step = detection_tiny && *tiny_step_last_iter;

    // Refresh the latch for the next iter
    // (IpBacktrackingLineSearch.cpp:421-435).
    *tiny_step_last_iter = detection_tiny && (dy_amax < options.tiny_step_y_tol);
}

/// Update the barrier parameter μ (interior-point centering parameter) once
/// per iteration after the step has been accepted.
///
/// Handles three cases:
/// - No variable bounds: superlinear decrease `μ^mu_superlinear_decrease_power`.
/// - `MuMode::Free` (adaptive oracle): Loqo mu oracle when enabled, with
///   monotone floor to prevent single-step collapse; falls back to `avg_compl/kappa`
///   when quality-function oracle is off; switches to Fixed mode on repeated
///   insufficient progress or dual-infeasibility stagnation.
/// - `MuMode::Fixed` (monotone decrease): decrease μ only when the barrier
///   subproblem is approximately solved (Ipopt's `IpMonotoneMuUpdate`
///   kappa_eps gate), using the min of linear and superlinear rates; may
///   switch back to Free mode in adaptive strategy.
///
/// On every μ change the filter is reset and `theta_min` is recomputed
/// (Ipopt convention, `IpFilterLSAcceptor.cpp:524-532`).
///
/// Ipopt parallel: `IpMuUpdate::UpdateBarrierParameter`, with concrete
/// subclasses `MonotoneMuUpdate` / `AdaptiveMuUpdate` inline here and the
/// `MuOracle` role played by the embedded Loqo formula.
///
/// Extracted from `solve_ipm` as part of the v0.8 main-loop decomposition
/// (pre-work step 2). Does not return a value — all effects are via
/// `&mut` on state, mu_state, filter, last_mehrotra_sigma.
/// Centrality measure used by the Loqo μ oracle:
/// `ξ = min_i(z_i · s_i) / avg_compl`, clamped to `[0, 1]`. ξ near 1
/// indicates a well-centered iterate (uniform complementarity
/// products); ξ near 0 indicates one product is much smaller than the
/// average. Returns 1.0 when `avg_compl` is non-positive or no active
/// bound products exist (no centrality information available).
///
/// B-cross6: scans all four complementarity blocks
/// (`(z_L, x_L)`, `(z_U, x_U)`, `(v_L, s_L)`, `(v_U, s_U)`) per
/// `IpLoqoMuOracle::CalculateMu` ↔ `IpIpoptCalculatedQuantities::curr_compl_xi`.
fn compute_centrality_xi(state: &SolverState, avg_compl: f64) -> f64 {
    let mut min_compl = f64::INFINITY;
    // Phase 6c.2: walk compressed bound multipliers; the iteration set
    // is identical to {i : x_l[i].is_finite()} by BoundLayout
    // construction, so the n-wide+is_finite scan and the compressed
    // walk produce bit-identical output.
    for k in 0..state.bound_layout.n_x_l {
        let i = state.bound_layout.x_l_to_full[k];
        min_compl = min_compl.min(slack_xl(state, i) * state.z_l_compressed[k]);
    }
    for k in 0..state.bound_layout.n_x_u {
        let i = state.bound_layout.x_u_to_full[k];
        min_compl = min_compl.min(slack_xu(state, i) * state.z_u_compressed[k]);
    }
    for i in 0..state.m {
        let l_fin = state.g_l_at(i).is_finite();
        let u_fin = state.g_u_at(i).is_finite();
        if l_fin && u_fin && (state.g_l_at(i) - state.g_u_at(i)).abs() < 1e-14 {
            continue;
        }
        if l_fin {
            min_compl = min_compl.min(slack_gl(state, i) * state.v_l_at(i));
        }
        if u_fin {
            min_compl = min_compl.min(slack_gu(state, i) * state.v_u_at(i));
        }
    }
    if avg_compl > 0.0 && min_compl.is_finite() {
        (min_compl / avg_compl).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Loqo barrier-parameter oracle (IpLoqoMuOracle.cpp).
///
/// Centrality-driven μ update: ξ = min(z_i·s_i) / avg_compl, then
/// σ = 0.1 · min(0.05·(1-ξ)/ξ, 2)³, and μ_new = σ · avg_compl,
/// floored from below by Ipopt's monotone schedule
/// `min(κ_μ·μ, μ^sldp)` (so the Loqo-proposed μ can't undershoot
/// a gradual monotone schedule on a single step) and from above
/// by 1e5. The hard floor is `options.mu_min` (T2.3: removed the
/// ripopt-specific `μ/5` ramp that fired when the barrier subproblem
/// was not approximately solved; Ipopt's `IpLoqoMuOracle::CalculateMu`
/// has no such conditional).
fn compute_loqo_mu(
    state: &SolverState,
    options: &SolverOptions,
    mu_state: &mut MuState,
    avg_compl: f64,
) -> f64 {
    // T2.28 + T3.2: faithful mirror of Ipopt 3.14
    // `LoqoMuOracle::CalculateMu` (`IpLoqoMuOracle.cpp:34-66`).
    // No `monotone_floor`, no `mu^super` clamp — Ipopt's CalculateMu
    // is `Max(Min(mu_max, sigma*avg_compl), mu_min)`, where
    // `mu_max = mu_max_fact * initial_avg_compl` is captured lazily
    // by the adaptive μ-update on its first call
    // (`IpAdaptiveMuUpdate.cpp:267-273`).
    let xi = compute_centrality_xi(state, avg_compl);

    let ratio = if xi > 1e-20 {
        (0.05 * (1.0 - xi) / xi).min(2.0)
    } else {
        2.0
    };
    let sigma = 0.1 * ratio.powi(3);
    let loqo_mu = sigma * avg_compl;
    let mu_cap = mu_state.mu_max_cap(options, avg_compl);
    let new_mu = loqo_mu.clamp(options.mu_min, mu_cap);

    if options.print_level >= 5 {
        rip_log!("ripopt: mu loqo: xi={:.4} sigma={:.4} avg_compl={:.3e} mu_cap={:.3e} -> mu={:.3e}",
            xi, sigma, avg_compl, mu_cap, new_mu);
    }
    new_mu
}

/// Quality-function μ oracle (T2.23, spec §3.5,
/// `IpQualityFunctionMuOracle.cpp:154-485`).
///
/// Procedure:
/// 1. Assemble and factor the augmented KKT at the current iterate.
/// 2. Solve the affine-predictor RHS (μ=0 in centering rows) → `d_aff`.
/// 3. Solve the full RHS at `μ_cur` → `d_full`.  The centering direction
///    is `d_cen = d_full − d_aff`, so a candidate σ produces the trial
///    step `d(σ) = (1−σ)·d_aff + σ·d_full` (which by linearity solves
///    KKT at μ_target = σ·μ_cur).
/// 4. For each candidate σ, take the fraction-to-boundary step
///    (α_p, α_d) under τ=1 and evaluate
///    Q(σ) = (1−α_d)·dual_inf₀ / s_d
///          + (1−α_p)·primal_inf₀
///          + max_i (s+_i · z+_i) / s_c          (compl, target μ=0)
///          + (1−min(α_p,α_d)) · max(avg_compl, μ) / s_c   (balancing)
///          [+ 1/ξ at trial when `quality_function_centrality`]
///    using the linear-residual identity `‖res(α)‖ = (1−α)·‖res(0)‖`.
/// 5. Golden-section minimise over σ ∈ [1e-6, 1.0] with at most 8 steps
///    and relative tolerance 1e-2 on the bracket width.
/// 6. Return clamp(σ*·avg_compl, max=1e5, min=max(monotone_floor, μ_min)).
///
/// Returns `None` on factorisation/solve failure so the caller can fall
/// back to the Loqo oracle (this matches `IpQualityFunctionMuOracle`'s
/// behaviour — it skips the candidate and lets the algorithm fall
/// through). Re-factorises the KKT inside the oracle (option (b) in the
/// task spec) — slower than reusing the iteration's factor, but cleanly
/// scoped without plumbing the linear solver through five call frames.
///
/// Tech-debt notes versus Ipopt 3.14's QF reference:
///   * Ipopt's QF uses the **true nonlinear residual** at the trial
///     point (problem(x+α·dx)). ripopt uses the linearised
///     `(1−α)·current` identity, which is exact only when the trial step
///     stays inside the linear regime. Near optimum where α≈1 the
///     dual/primal terms collapse and Q is dominated by the compl term.
///     Without a balancing term the QF becomes degenerate (Q is flat
///     across σ). The balancing term above breaks that degeneracy and
///     restores meaningful σ selection.
///   * Spec §3.5 quotes σ_max=1e2 (matching Ipopt's full-balancing
///     mode). ripopt caps σ_max at 1.0 because the linearised Q above
///     has spurious local minima at σ≳1 in well-converged regimes.
///   * Compl target is μ=0 (optimality), not σ·μ_cur (centering). This
///     makes σ→0 strictly preferred whenever the affine direction
///     admits a long FTB stride, matching Mehrotra-like behaviour.
///   * To upgrade to the full Ipopt formulation we would need to plumb
///     `&P: NlpProblem` and the linear solver through `update_barrier_*`
///     so the oracle can call `problem.objective`/`gradient`/etc. at
///     each trial point. That is option (a) in the task spec.
fn compute_quality_function_mu(
    state: &SolverState,
    options: &SolverOptions,
    mu_state: &mut MuState,
    avg_compl: f64,
    use_sparse: bool,
    factor_cache: &mut kkt::FactorCache,
) -> Option<f64> {
    let n = state.n;
    let m = state.m;

    if avg_compl <= 0.0 {
        return None;
    }

    // 1) Assemble + factor KKT at the current iterate. T3.25 follow-up:
    // the QF oracle uses a *local* fallback solver instance, distinct
    // from the main-loop `lin_solver`. A cache hit on the shared cache
    // would replay `(δ_w, δ_c)` whose underlying factorization lives in
    // the main loop's solver — incorrect for this local solver. Use a
    // private per-call cache that mirrors the shared cache's `enabled`
    // flag, so the cached entry point is exercised on this path
    // (factor_calls bumps) but the shared cache remains valid for the
    // main loop's next factor.
    let sigma_vec = compute_sigma_from_state(state);
    let mut kkt = assemble_kkt_from_state(state, n, m, &sigma_vec, use_sparse, options.kappa_d);
    // Stamp the upstream atags onto the fresh KktSystem so the cached
    // entry point can fingerprint it (assemble_kkt_from_state predates
    // T3.25 and does not propagate atags).
    if kkt.input_atags.is_none() {
        kkt.input_atags = Some(state.kkt_atags);
    }
    let mut solver = new_fallback_solver(use_sparse);
    let mut inertia_params = InertiaCorrectionParams::default();
    let mut local_cache = kkt::FactorCache::new();
    local_cache.enabled = factor_cache.enabled;
    let factor_result = kkt::factor_with_inertia_correction_cached(
        &mut kkt, solver.as_mut(), &mut inertia_params, state.mu, &mut local_cache,
    );
    // Fold the local diagnostic counters into the shared cache so tests
    // can observe that the QF path exercised the cached entry point.
    factor_cache.factor_calls += local_cache.factor_calls;
    factor_cache.hits += local_cache.hits;
    factor_cache.misses += local_cache.misses;
    if factor_result.is_err() {
        return None;
    }

    // 2+3) Affine-predictor and full-step solves submitted as one batched
    // call. Both RHSes are known up front and use the same factor, so feral's
    // `solve_sparse_many` (F1.1) shares workspace and supernode traversal
    // across columns; the default trait impl loops single-RHS solves and
    // matches the prior behavior. T3.26: mu oracles use inexact backsolves
    // (allow_inexact=true, IpPDFullSpaceSolver.cpp:229-239).
    let x_l_full = state.x_l_combined();
    let x_u_full = state.x_u_combined();
    let rhs_aff = kkt::affine_predictor_rhs(
        &kkt.rhs, &state.x, &x_l_full, &x_u_full, state.mu, options.kappa_d,
    );
    let pairs = kkt::solve_with_custom_rhs_many(
        kkt.n, kkt.dim, solver.as_mut(), &[&rhs_aff, &kkt.rhs],
    ).ok()?;
    let (dx_aff, dy_aff) = pairs[0].clone();
    let (dx_full, dy_full) = pairs[1].clone();
    let (dz_l_aff, dz_u_aff) = recover_dz_from_state(state, &dx_aff, 0.0);
    let (dz_l_full, dz_u_full) = recover_dz_from_state(state, &dx_full, state.mu);
    // B-cross6: recover slack-side primal step `ds` and slack-bound
    // multiplier steps `dv_L, dv_U` so the QF oracle's centrality and
    // FTB scans cover all four bound blocks (matching Ipopt's
    // `IpQualityFunctionMuOracle::CalculateMu` which iterates over
    // x_L, x_U, s_L, s_U). Mu oracles run before PD perturbation, so
    // `ic_delta_c = 0` is correct for both predictor and full step
    // (`IpPDFullSpaceSolver.cpp:229-239`, `IpProbingMuOracle.cpp:71-72`).
    let ds_aff = recover_ds_from_state(state, &dx_aff, &dy_aff, 0.0);
    let ds_full = recover_ds_from_state(state, &dx_full, &dy_full, 0.0);
    let (dv_l_aff, dv_u_aff) = recover_dv_from_state(state, &ds_aff, 0.0);
    let (dv_l_full, dv_u_full) = recover_dv_from_state(state, &ds_full, state.mu);

    // 4) Pre-compute the residuals at the current iterate (needed for the
    //    `(1−α)·residual` linearised identity).
    let primal_inf0 = compute_primal_inf_max_at_state(state);
    let dual_inf0 = compute_dual_inf_at_state(state);
    let s_d = compute_residual_scaling(compute_multiplier_sum(state), compute_multiplier_count(state));
    let s_c = compute_residual_scaling(compute_bound_multiplier_sum(state), compute_bound_multiplier_count(state));

    // Quality-function evaluator at sigma. Captures the precomputed
    // affine + full directions; pure scalar arithmetic from here on.
    let q_eval = |sigma: f64| -> f64 {
        // Trial step d(σ) = (1−σ)·d_aff + σ·d_full
        let mut dx = vec![0.0; n];
        let mut dz_l = vec![0.0; n];
        let mut dz_u = vec![0.0; n];
        for i in 0..n {
            dx[i] = (1.0 - sigma) * dx_aff[i] + sigma * dx_full[i];
            dz_l[i] = (1.0 - sigma) * dz_l_aff[i] + sigma * dz_l_full[i];
            dz_u[i] = (1.0 - sigma) * dz_u_aff[i] + sigma * dz_u_full[i];
        }
        // B-cross6: σ-blended slack and slack-bound multiplier steps.
        let mut ds = vec![0.0; m];
        let mut dv_l = vec![0.0; m];
        let mut dv_u = vec![0.0; m];
        for i in 0..m {
            ds[i] = (1.0 - sigma) * ds_aff[i] + sigma * ds_full[i];
            dv_l[i] = (1.0 - sigma) * dv_l_aff[i] + sigma * dv_l_full[i];
            dv_u[i] = (1.0 - sigma) * dv_u_aff[i] + sigma * dv_u_full[i];
        }

        // Fraction-to-boundary step lengths under τ=1 (Ipopt QF probe uses
        // a full FTB scan on the candidate direction). B-cross6: include
        // s vs [g_l, g_u] and v_L/v_U non-negativity.
        let ds_d_qf = state.layout.project_d(&ds);
        let dv_l_d_qf = state.layout.project_d(&dv_l);
        let dv_u_d_qf = state.layout.project_d(&dv_u);
        let alpha_p = fraction_to_boundary_primal_x(state, &dx, 1.0)
            .min(fraction_to_boundary_primal_s(state, &ds_d_qf, 1.0))
            .clamp(0.0, 1.0);
        // Phase 6d.4: project σ-blended dz_l/dz_u to compressed form
        // for the FTB scan signature.
        let dz_l_c = state.bound_layout.project_l(&dz_l);
        let dz_u_c = state.bound_layout.project_u(&dz_u);
        let alpha_d = fraction_to_boundary_dual_z_min(state, &dz_l_c, &dz_u_c, 1.0)
            .min(fraction_to_boundary_dual_v_min(state, &dv_l_d_qf, &dv_u_d_qf, 1.0))
            .clamp(0.0, 1.0);

        // Linearised residual reduction: ‖r(α)‖ = (1−α)·‖r(0)‖.
        let dual_inf_trial = (1.0 - alpha_d) * dual_inf0;
        let primal_inf_trial = (1.0 - alpha_p) * primal_inf0;

        // Complementarity at the trial point, measured against the
        // *optimality* target μ=0 rather than the σ·μ_cur centering
        // target. This is the discriminator that Ipopt's QF effectively
        // uses (`IpQualityFunctionMuOracle.cpp` evaluates compl as
        // `slack·z` directly, not `slack·z − μ`): a smaller σ that drives
        // s·z toward zero is preferred whenever the affine step admits a
        // long FTB stride. The σ=σ_max degeneracy (where the σ·μ_cur
        // target is trivially met by the centered step) is broken.
        // B-cross6: scan all four blocks (z_L, z_U, v_L, v_U).
        let mut compl_max: f64 = 0.0;
        // Phase 6c.2: walk compressed bound mirrors. dz_l/dz_u are
        // full-`n` σ-blended directions, indexed via x_l_to_full[k].
        for k in 0..state.bound_layout.n_x_l {
            let i = state.bound_layout.x_l_to_full[k];
            let s_plus = (slack_xl(state, i) + alpha_p * dx[i]).max(1e-20);
            let z_plus = (state.z_l_compressed[k] + alpha_d * dz_l[i]).max(1e-20);
            compl_max = compl_max.max(s_plus * z_plus);
        }
        for k in 0..state.bound_layout.n_x_u {
            let i = state.bound_layout.x_u_to_full[k];
            let s_plus = (slack_xu(state, i) - alpha_p * dx[i]).max(1e-20);
            let z_plus = (state.z_u_compressed[k] + alpha_d * dz_u[i]).max(1e-20);
            compl_max = compl_max.max(s_plus * z_plus);
        }
        for i in 0..m {
            let l_fin = state.g_l_at(i).is_finite();
            let u_fin = state.g_u_at(i).is_finite();
            if l_fin && u_fin && (state.g_l_at(i) - state.g_u_at(i)).abs() < 1e-14 {
                continue;
            }
            if l_fin {
                let s_plus = (slack_gl(state, i) + alpha_p * ds[i]).max(1e-20);
                let v_plus = (state.v_l_at(i) + alpha_d * dv_l[i]).max(1e-20);
                compl_max = compl_max.max(s_plus * v_plus);
            }
            if u_fin {
                let s_plus = (slack_gu(state, i) - alpha_p * ds[i]).max(1e-20);
                let v_plus = (state.v_u_at(i) + alpha_d * dv_u[i]).max(1e-20);
                compl_max = compl_max.max(s_plus * v_plus);
            }
        }

        let mut q = dual_inf_trial / s_d + primal_inf_trial + compl_max / s_c;

        // Balancing term: penalise σ values whose linearised step admits
        // only a tiny fraction-to-boundary step. Without this, σ=0 (full
        // affine) and σ=1 (full centered) are both linearisation-optimal
        // (each kills its own residual), and the search lacks a
        // discriminator. Mirrors Ipopt's
        // `IpQualityFunctionMuOracle.cpp:625-660` balancing term, which
        // adds a contribution proportional to (1−min(α_p,α_d))·max_compl
        // to penalise short steps.
        let alpha_min = alpha_p.min(alpha_d).max(1e-12);
        // Use the current max compl as the scale so the balancing term is
        // commensurate with the rest of Q.
        let scale = avg_compl.max(state.mu);
        q += (1.0 - alpha_min) * scale / s_c;

        // Optional centrality penalty: 1/ξ where ξ = min(s·z) / avg(s·z)
        // at the trial point (Ipopt `centrality=reciprocal`,
        // `IpQualityFunctionMuOracle.cpp:622`). B-cross6: scan all
        // four bound blocks.
        if options.quality_function_centrality {
            let mut sum_sz = 0.0_f64;
            let mut min_sz = f64::INFINITY;
            let mut nb = 0usize;
            // Phase 6c.2: walk compressed bound mirrors.
            for k in 0..state.bound_layout.n_x_l {
                let i = state.bound_layout.x_l_to_full[k];
                let s_plus = (slack_xl(state, i) + alpha_p * dx[i]).max(1e-20);
                let z_plus = (state.z_l_compressed[k] + alpha_d * dz_l[i]).max(1e-20);
                let sz = s_plus * z_plus;
                sum_sz += sz;
                if sz < min_sz { min_sz = sz; }
                nb += 1;
            }
            for k in 0..state.bound_layout.n_x_u {
                let i = state.bound_layout.x_u_to_full[k];
                let s_plus = (slack_xu(state, i) - alpha_p * dx[i]).max(1e-20);
                let z_plus = (state.z_u_compressed[k] + alpha_d * dz_u[i]).max(1e-20);
                let sz = s_plus * z_plus;
                sum_sz += sz;
                if sz < min_sz { min_sz = sz; }
                nb += 1;
            }
            for i in 0..m {
                let l_fin = state.g_l_at(i).is_finite();
                let u_fin = state.g_u_at(i).is_finite();
                if l_fin && u_fin && (state.g_l_at(i) - state.g_u_at(i)).abs() < 1e-14 {
                    continue;
                }
                if l_fin {
                    let s_plus = (slack_gl(state, i) + alpha_p * ds[i]).max(1e-20);
                    let v_plus = (state.v_l_at(i) + alpha_d * dv_l[i]).max(1e-20);
                    let sv = s_plus * v_plus;
                    sum_sz += sv;
                    if sv < min_sz { min_sz = sv; }
                    nb += 1;
                }
                if u_fin {
                    let s_plus = (slack_gu(state, i) - alpha_p * ds[i]).max(1e-20);
                    let v_plus = (state.v_u_at(i) + alpha_d * dv_u[i]).max(1e-20);
                    let sv = s_plus * v_plus;
                    sum_sz += sv;
                    if sv < min_sz { min_sz = sv; }
                    nb += 1;
                }
            }
            if nb > 0 && sum_sz > 0.0 {
                let avg = sum_sz / nb as f64;
                let xi = (min_sz / avg).clamp(1e-20, 1.0);
                q += 1.0 / xi;
            }
        }
        q
    };

    // 5) Golden-section minimise Q over σ ∈ [σ_min, σ_max]. Spec §3.5
    //    quotes [1e-6, 1e2] from Ipopt's full QF with balancing terms;
    //    ripopt's simpler linearised Q has a degeneracy near σ=1 where
    //    both endpoints satisfy the linearised KKT, so we cap σ_max at
    //    1.0 (Ipopt's effective range when `quality_function_balancing_term`
    //    is `none`, which is the default). This avoids σ ≳ 1 picks that
    //    would pin μ at avg_compl indefinitely.
    let sigma_min = 1e-6;
    let sigma_max = 1.0;
    let sigma_star = golden_section_minimize(
        &q_eval,
        sigma_min,
        sigma_max,
        options.quality_function_max_section_steps,
        0.01, // relative tolerance: |σ_hi − σ_lo| < 0.01·σ_lo
    );

    // 6) Convert σ* to μ. Apply the same monotone floor and clamp as the
    //    Loqo oracle (Ipopt's `IpQualityFunctionMuOracle::CalculateMu`
    //    re-uses the Loqo clamp).
    let qf_mu = sigma_star * avg_compl;
    let monotone_floor =
        (options.mu_linear_decrease_factor * state.mu)
            .min(state.mu.powf(options.mu_superlinear_decrease_power));
    let mu_cap = mu_state.mu_max_cap(options, avg_compl);
    let new_mu = qf_mu.max(monotone_floor).clamp(options.mu_min, mu_cap);

    if options.print_level >= 5 {
        rip_log!("ripopt: mu QF: sigma*={:.4e} avg_compl={:.3e} floor={:.3e} -> mu={:.3e}",
            sigma_star, avg_compl, monotone_floor, new_mu);
    }
    Some(new_mu)
}

/// Golden-section search for the minimum of a unimodal `f` on `[lo, hi]`
/// (operating in log-space because the QF candidate range spans 8 orders
/// of magnitude). Stops after at most `max_steps` shrinks or once the
/// log-bracket width is below `rel_tol`. Returns the bracket centre.
///
/// Used by [`compute_quality_function_mu`] to minimise Q(σ) per
/// `IpQualityFunctionMuOracle.cpp:520-560` (Ipopt also operates in the
/// log scale and bounds `quality_function_max_section_steps` at 8 by
/// default).
fn golden_section_minimize<F: Fn(f64) -> f64>(
    f: &F,
    lo: f64,
    hi: f64,
    max_steps: usize,
    rel_tol: f64,
) -> f64 {
    debug_assert!(lo > 0.0 && hi > lo);
    let log_lo0 = lo.ln();
    let log_hi0 = hi.ln();
    let mut log_lo = log_lo0;
    let mut log_hi = log_hi0;
    // Golden-section split factor.
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0; // ~0.618
    let mut log_x1 = log_hi - phi * (log_hi - log_lo);
    let mut log_x2 = log_lo + phi * (log_hi - log_lo);
    let mut f1 = f(log_x1.exp());
    let mut f2 = f(log_x2.exp());
    for _ in 0..max_steps {
        if (log_hi - log_lo).abs() < rel_tol {
            break;
        }
        if f1 < f2 {
            log_hi = log_x2;
            log_x2 = log_x1;
            f2 = f1;
            log_x1 = log_hi - phi * (log_hi - log_lo);
            f1 = f(log_x1.exp());
        } else {
            log_lo = log_x1;
            log_x1 = log_x2;
            f1 = f2;
            log_x2 = log_lo + phi * (log_hi - log_lo);
            f2 = f(log_x2.exp());
        }
    }
    (0.5 * (log_lo + log_hi)).exp()
}

fn update_barrier_parameter(
    state: &mut SolverState,
    mu_state: &mut MuState,
    filter: &mut Filter,
    last_mehrotra_sigma: &mut Option<f64>,
    options: &SolverOptions,
    use_sparse: bool,
) {
    let n = state.n;
    // When the problem has neither variable bounds NOR inequality
    // constraints, mu serves no barrier purpose: there are no `s · v = μ`
    // or `(x − x_l) · z_l = μ` complementarity blocks. In that case mu is
    // only used for KKT regularization and the filter line search, and
    // we decrease it superlinearly to keep filter protection without
    // collapsing to mu_min instantly (the PENTAGON guard).
    //
    // Bug fix (qcqp1500-1c): the prior gate `!has_var_bounds → μ^1.5
    // unconditionally` was wrong when inequality constraints are
    // present, because the slack barrier `μ Σ log s_k` is then active
    // and requires sufficient-progress gating just like var bounds. On
    // qcqp1500-1c (0 var bounds, 10008 inequalities) it caused mu to
    // collapse to 1e-11 in 6 iterations while complementarity stayed
    // ~10⁵, racing the IPM into the floor with no barrier subproblem
    // ever solved.
    let has_var_bounds = (0..n).any(|i| state.x_l_at(i).is_finite() || state.x_u_at(i).is_finite());
    let has_slack_barrier = !state.s.is_empty();
    if !has_var_bounds && !has_slack_barrier {
        state.mu = state.mu.powf(options.mu_superlinear_decrease_power).max(options.mu_min);
        return;
    }

    let kkt_error = {
        let pi = state.constraint_violation();
        let di = compute_dual_inf_at_state(state);
        let ci = compute_compl_err_at_state(state);
        pi * pi + di * di + ci * ci
    };

    let sufficient = mu_state.check_sufficient_progress(kkt_error);

    // Track dual infeasibility for stagnation detection.
    // If du is not decreasing over 3 consecutive iterations in Free mode,
    // force switch to Fixed mode with mu = 0.8 * avg_compl.
    {
        let du_now = compute_dual_inf_at_state(state);
        if mu_state.dual_inf_window.len() >= 3 {
            mu_state.dual_inf_window.remove(0);
        }
        mu_state.dual_inf_window.push(du_now);
    }

    // A8.11: Ipopt's Free mode (IpAdaptiveMuUpdate.cpp:343-389) has no
    // barrier-subproblem-solved gate — that absolute test exists only in
    // Fixed mode (IpMonotoneMuUpdate.cpp). Free mode is a strict 2-way
    // split on `CheckSufficientProgress()` after the skipped-LS / tiny-step
    // override: sufficient → run oracle unconditionally (line 391-436),
    // not sufficient → switch to Fixed. Fixed mode does its own
    // barrier_err computation inside the decrement loop.
    match mu_state.mode {
        MuMode::Free => {
            update_barrier_parameter_free_mode(
                state, mu_state, filter, last_mehrotra_sigma, options,
                sufficient, kkt_error, use_sparse,
            );
        }
        MuMode::Fixed => {
            update_barrier_parameter_fixed_mode(
                state, mu_state, filter, options, sufficient, kkt_error,
            );
        }
    }
}

/// Record a mu-strategy mode change: bump the diagnostics counter,
/// flip `mu_state.mode`, and re-seed `first_iter_in_mode = true` so
/// the next iteration re-initialises the new mode's tracking state.
/// Used by every Free↔Fixed transition (insufficient-progress switch,
/// promotion-back-to-Free in `update_barrier_parameter_fixed_mode`,
/// and the post-restoration retry in `apply_restoration_recovery_strategy`).
fn switch_mu_mode(state: &mut SolverState, mu_state: &mut MuState, new_mode: MuMode) {
    state.diagnostics.mu_mode_switches += 1;
    mu_state.mode = new_mode;
    mu_state.first_iter_in_mode = true;
}

/// Switch the mu strategy from Free to Fixed and seed the new mu.
/// Triggered when Free mode shows insufficient progress for ≥2
/// iterations or dual-infeasibility stagnation. Resets
/// `consecutive_insufficient`, increments the diagnostics counter,
/// flips the mode and `first_iter_in_mode`, then sets
/// `μ = adaptive_mu_monotone_init·avg_compl` (clamped to
/// `[μ_min, 1e5]`) — falling back to a linear μ-decrease when no
/// active complementarity products exist. Finally resets the filter
/// at the new (μ, θ).
fn switch_to_fixed_mode_with_adaptive_init(
    state: &mut SolverState,
    mu_state: &mut MuState,
    filter: &mut Filter,
    options: &SolverOptions,
) {
    mu_state.consecutive_insufficient = 0;
    log::debug!("Switching to fixed mu mode (insufficient progress or tiny step)");
    switch_mu_mode(state, mu_state, MuMode::Fixed);
    // T3.11: optional rollback to the last Free-mode iterate that
    // satisfied `CheckSufficientProgress`. Mirrors Ipopt
    // `IpAdaptiveMuUpdate.cpp:362-370`. Only mu/tau are recomputed
    // afterwards; the snapshot is consumed on use so a subsequent
    // Fixed→Free→Fixed cycle re-captures.
    if options.adaptive_mu_restore_previous_iterate {
        if let Some(snap) = mu_state.accepted_iterate.take() {
            state.x = snap.x;
            state.set_y_combined(&snap.y);
            state.z_l_compressed = snap.z_l_compressed;
            state.z_u_compressed = snap.z_u_compressed;
            state.set_v_l_combined(&snap.v_l);
            state.set_v_u_combined(&snap.v_u);
            // T3.25: rollback touches every tracked KKT input.
            state.bump_all_kkt_atags();
            log::debug!("Free→Fixed rollback: restored accepted_point");
        }
    }
    let avg_compl = compute_avg_complementarity(state);
    if avg_compl > 0.0 {
        // A8.6: align switch-to-Fixed cap with Ipopt
        // (`IpAdaptiveMuUpdate.cpp:267-273`): cap by `mu_max_fact *
        // initial_avg_compl` rather than the hard `1e5` previously used.
        // Matches the cap already used by the four other Free-mode μ
        // update sites that all funnel through `mu_state.mu_max_cap`.
        let mu_cap = mu_state.mu_max_cap(options, avg_compl);
        state.mu = (options.adaptive_mu_monotone_init_factor * avg_compl)
            .clamp(options.mu_min, mu_cap);
    } else {
        state.mu = (options.mu_linear_decrease_factor * state.mu)
            .max(options.mu_min);
    }
    reset_filter_with_current_theta(state, filter);
}

/// Sufficient-progress branch of the Free-mode μ update: reset the
/// insufficient-progress counter, remember the accepted KKT error,
/// and select a new μ. Picks the Loqo oracle when
/// `mu_oracle_quality_function` is on, falls back to a rate-limited
/// `avg_compl/kappa` (with a `μ/5` floor when the barrier subproblem
/// is not approximately solved), or to `mu_linear_decrease_factor·μ`
/// when no active complementarity products exist. Resets the filter
/// at the new μ per Ipopt's `IpFilterLSAcceptor.cpp:524-532`.
fn apply_free_mode_sufficient_progress_update(
    state: &mut SolverState,
    mu_state: &mut MuState,
    filter: &mut Filter,
    options: &SolverOptions,
    kkt_error: f64,
    use_sparse: bool,
) {
    mu_state.consecutive_insufficient = 0;
    mu_state.remember_accepted(kkt_error);
    // T3.11: capture the accepted iterate so a later Free→Fixed switch
    // can roll back to it. Mirrors Ipopt
    // `RememberCurrentPointAsAccepted` (IpAdaptiveMuUpdate.cpp:541-545):
    // only enabled under `adaptive_mu_restore_previous_iterate`.
    if options.adaptive_mu_restore_previous_iterate {
        mu_state.accepted_iterate = Some(AcceptedIterateSnapshot {
            x: state.x.clone(),
            y: state.y_combined(),
            z_l_compressed: state.z_l_compressed.clone(),
            z_u_compressed: state.z_u_compressed.clone(),
            v_l: state.v_l_combined(),
            v_u: state.v_u_combined(),
        });
    }
    let avg_compl = compute_avg_complementarity(state);
    if options.mu_oracle_quality_function && avg_compl > 0.0 {
        // T2.23: try the Ipopt-style Quality Function oracle first
        // (spec §3.5, IpQualityFunctionMuOracle.cpp). Falls back to the
        // Loqo formula on aff/centering solve failure.
        //
        // T3.25 follow-up: move the factor cache out of `state`, run
        // the oracle with `&state` + the cache borrowed exclusively,
        // then put the cache back. This avoids a double mutable borrow
        // on `state` without resorting to raw pointers, and preserves
        // the diagnostic counters across the call.
        let mut cache = std::mem::take(&mut state.factor_cache);
        let qf_mu = compute_quality_function_mu(
            state, options, mu_state, avg_compl, use_sparse, &mut cache,
        );
        state.factor_cache = cache;
        state.mu = qf_mu
            .unwrap_or_else(|| compute_loqo_mu(state, options, mu_state, avg_compl));
    } else if avg_compl > 0.0 {
        // DEV-2: replace the ripopt-specific `avg_compl / options.kappa`
        // fallback with the Loqo oracle. Ipopt's `IpAdaptiveMuUpdate::DoUpdate`
        // (`IpAdaptiveMuUpdate.cpp:391-436`) always runs the configured
        // mu-oracle (loqo, quality-function, or probing) when sufficient
        // progress holds; it never uses an `avg_compl / kappa` formula.
        // Falling back to Loqo when the QF oracle is disabled gives the
        // closest Ipopt analog (`mu_oracle = loqo`).
        state.mu = compute_loqo_mu(state, options, mu_state, avg_compl);
    } else {
        state.mu = (options.mu_linear_decrease_factor * state.mu)
            .max(options.mu_min);
    }
    reset_filter_with_current_theta(state, filter);
}

/// Free-mode (adaptive) barrier-parameter update. Strict 2-way split
/// matching Ipopt's `IpAdaptiveMuUpdate::DoUpdate` (lines 343-389) and
/// the unconditional oracle call at lines 391-436:
/// 1) `sufficient && !tiny_step` → run the mu-oracle to pick a new μ,
///    remember the accepted point, and reset the filter.
/// 2) Otherwise → switch to Fixed mode with `μ = adaptive_mu_monotone_init *
///    avg_compl` and reset the filter.
///
/// A8.11: removed the previous `barrier_subproblem_solved` gate and
/// the `apply_free_mode_conservative_decrease` middle branch. Neither
/// has an analogue in Ipopt 3.14 — Free mode never tests
/// `barrier_err <= kappa_eps * mu`; that gate exists only in Fixed
/// mode (`IpMonotoneMuUpdate.cpp:135-194`). The "stay in Free with μ
/// unchanged" fall-through that the gate created is also non-Ipopt:
/// Free mode either runs the oracle or switches to Fixed.
fn update_barrier_parameter_free_mode(
    state: &mut SolverState,
    mu_state: &mut MuState,
    filter: &mut Filter,
    last_mehrotra_sigma: &mut Option<f64>,
    options: &SolverOptions,
    sufficient: bool,
    kkt_error: f64,
    use_sparse: bool,
) {
    // Consume Mehrotra sigma for use as quality function candidate
    let _sigma_mu = last_mehrotra_sigma.take();
    if sufficient && !mu_state.tiny_step {
        apply_free_mode_sufficient_progress_update(
            state, mu_state, filter, options, kkt_error, use_sparse,
        );
    } else {
        // A8.8: align Free→Fixed switch with Ipopt
        // (`IpAdaptiveMuUpdate.cpp:343-389`). Ipopt's only Free→Fixed
        // triggers are `!CheckSufficientProgress()`, `tiny_step_flag`,
        // and `CheckSkippedLineSearch()`. Critically,
        // `CheckSufficientProgress()` returns *true* whenever the
        // KKT-error reference window has fewer than `num_refs_max`
        // (default 4) entries (`IpAdaptiveMuUpdate.cpp:446-490`), so
        // the earliest possible switch is iter 4, not iter 1.
        mu_state.consecutive_insufficient += 1;
        switch_to_fixed_mode_with_adaptive_init(state, mu_state, filter, options);
    }
}

/// Fixed-mode (monotone) barrier-parameter update. Either switches back to
/// Free mode when adaptive strategy + sufficient progress is detected
/// (skipping the first iteration in Fixed mode), or — when the barrier
/// subproblem is approximately solved (barrier_err <= kappa_eps*mu) or a
/// tiny step was taken — decreases mu by the min of linear and superlinear
/// rates. Filter and theta_min are reset on every accepted decrease.
/// Mirrors Ipopt's IpMonotoneMuUpdate.cpp.
fn update_barrier_parameter_fixed_mode(
    state: &mut SolverState,
    mu_state: &mut MuState,
    filter: &mut Filter,
    options: &SolverOptions,
    sufficient: bool,
    kkt_error: f64,
) {
    if options.mu_strategy_adaptive && sufficient && !mu_state.tiny_step && !mu_state.first_iter_in_mode {
        // Switch back to free mode (only in adaptive strategy)
        log::debug!("Switching back to free mu mode (sufficient progress)");
        switch_mu_mode(state, mu_state, MuMode::Free);
        mu_state.remember_accepted(kkt_error);
    } else {
        mu_state.first_iter_in_mode = false;
        // Mirrors Ipopt's IpMonotoneMuUpdate.cpp:130-200: while the barrier
        // subproblem is solved at the current μ (or a tiny step was taken),
        // decrease μ. With `mu_allow_fast_monotone_decrease`, allow several
        // consecutive decreases per outer iteration; otherwise stop after one.
        //
        // DEV-3: floor mu at `min(tol, compl_inf_tol) / (barrier_tol_factor + 1)`
        // per IpMonotoneMuUpdate.cpp:215 — Ipopt won't drive mu below the
        // level the convergence test cannot benefit from, preventing the
        // algorithm from latching into pathological super-tight subproblems.
        // ripopt's `mu_min` (default 1e-11) is preserved as an absolute
        // hard-floor below the Ipopt formula.
        //
        // DEV-4: removed the `MAX_FAST_DECREASES = 4` cap; Ipopt loops
        // while-solved without bound (IpMonotoneMuUpdate.cpp:130-200).
        let mu_floor = options
            .tol
            .min(options.compl_inf_tol)
            / (options.barrier_tol_factor + 1.0);
        let mut decreases = 0usize;
        let mut tiny_step = mu_state.tiny_step;
        loop {
            let (barrier_err, du_e, co_e, pr_e) =
                compute_barrier_error_components(state);
            let solved = barrier_err <= options.barrier_tol_factor * state.mu;
            if std::env::var("RIPOPT_TRACE_MU").is_ok() {
                eprintln!(
                    "ripopt: mu-gate iter={} mu={:.3e} E_mu={:.3e} (du={:.3e} co={:.3e} pr={:.3e}) thr={:.3e} solved={}",
                    state.iter, state.mu, barrier_err, du_e, co_e, pr_e,
                    options.barrier_tol_factor * state.mu, solved,
                );
                if state.iter % 50 == 0 || state.iter >= 495 {
                    dump_compl_outliers(state);
                }
            }
            if !(solved || tiny_step) { break; }
            let new_mu = (options.mu_linear_decrease_factor * state.mu)
                .min(state.mu.powf(options.mu_superlinear_decrease_power))
                .max(mu_floor)
                .max(options.mu_min);
            let mu_changed = (new_mu - state.mu).abs() > 1e-20;
            if !mu_changed { break; }
            state.mu = new_mu;
            decreases += 1;
            tiny_step = false;
            log::debug!("Fixed mode: mu decreased to {:.2e}", state.mu);
            if !options.mu_allow_fast_monotone_decrease { break; }
        }
        if decreases > 0 {
            reset_filter_with_current_theta(state, filter);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn check_convergence_and_handle_promotions(
    state: &mut SolverState,
    options: &SolverOptions,
    primal_inf_max: f64,
    primal_inf_internal_max: f64,
    dual_inf: f64,
    dual_inf_unscaled: f64,
    compl_inf: f64,
    multiplier_sum: f64,
    multiplier_count: usize,
    bound_multiplier_sum: f64,
    bound_multiplier_count: usize,
    timings: &PhaseTimings,
    iteration: usize,
    ipm_start: Instant,
) -> Option<SolveResult> {
    let conv_info = ConvergenceInfo {
        primal_inf: primal_inf_max,
        primal_inf_internal: primal_inf_internal_max,
        dual_inf,
        dual_inf_unscaled,
        compl_inf,
        mu: state.mu,
        objective: state.obj,
        multiplier_sum,
        multiplier_count,
        bound_multiplier_sum,
        bound_multiplier_count,
        x_max_abs: linf_norm(&state.x),
    };

    match crate::convergence::check_convergence_with_last_obj(
        &conv_info, options, state.consecutive_acceptable, state.last_obj_for_acceptable,
    ) {
        ConvergenceStatus::Converged => {
            if options.print_level >= 5 {
                timings.print_summary(iteration + 1, ipm_start.elapsed());
            }
            Some(make_result(state, SolveStatus::Optimal))
        }
        ConvergenceStatus::Acceptable => {
            if options.print_level >= 5 {
                timings.print_summary(iteration + 1, ipm_start.elapsed());
            }
            // Matches Ipopt's Solved_To_Acceptable_Level.
            Some(make_result(state, SolveStatus::Acceptable))
        }
        ConvergenceStatus::Diverging => {
            Some(make_result(state, SolveStatus::DivergingIterates))
        }
        ConvergenceStatus::NotConverged => None,
    }
}

/// Check wall-clock time limit at the top of each iteration.
///
/// Returns `Some(MaxIterations)` if `max_wall_time` has been exceeded.
/// Wall-clock is polled every iteration during the first 10, then every 10
/// thereafter to keep overhead negligible on long runs.
fn check_time_limits(
    state: &SolverState,
    iteration: usize,
    start_time: Instant,
    options: &SolverOptions,
) -> Option<SolveResult> {
    if (iteration < 10 || iteration % 10 == 0) && options.max_wall_time > 0.0 {
        if start_time.elapsed().as_secs_f64() >= options.max_wall_time {
            return Some(make_result(state, SolveStatus::MaxIterations));
        }
    }
    None
}

/// Reset the filter (clear all entries). Standard "fresh-start" sequence
/// after μ changes, restoration, stall recovery, or watchdog promotions.
/// Mirrors Ipopt IpFilterLSAcceptor.cpp:524-532 (Reset()), which clears
/// the filter list but does NOT touch `theta_max`/`theta_min` — those
/// are seeded once from the initial iterate at IpFilterLSAcceptor.cpp:325-339
/// and remain fixed for the entire solve. T0.7: previously this helper
/// also called `set_theta_min_from_initial` on every μ change, letting
/// the filter envelope grow and admit iterates earlier filter entries
/// had rejected. The `set_theta_min_from_initial` method is now
/// one-shot, so even if it is called here it is a no-op after the first
/// solver-init seeding; the call is omitted for clarity.
fn reset_filter_with_current_theta(_state: &SolverState, filter: &mut Filter) {
    filter.reset();
}

/// Per-iteration KKT residuals used by the log row, filter, and
/// convergence check.
struct OptimalityMeasures {
    /// 1-norm constraint violation (filter/log).
    primal_inf: f64,
    /// Max-norm primal infeasibility (convergence gate, user-facing).
    primal_inf_max: f64,
    /// Max-norm slack-coupling residual `||c||_∞ ∪ ||d − s||_∞` for the
    /// scaled (barrier-level) convergence gate. Mirrors Ipopt's
    /// `curr_primal_infeasibility(NORM_MAX)`.
    primal_inf_internal_max: f64,
    /// Iterative-z dual infeasibility `||∇f + J^T y - z_L + z_U||_∞`.
    dual_inf: f64,
    /// Component-wise scaled dual infeasibility for the unscaled gate
    /// (divides each component by `1 + |∇f_i|`).
    dual_inf_unscaled: f64,
    /// Complementarity error `max_i {(x-x_L)_i z_{L,i}, (x_U-x)_i z_{U,i}}`
    /// against target 0 (unscaled).
    compl_inf: f64,
}

/// Compute the per-iteration optimality residuals.
///
/// Ipopt parallel: `IpIpoptCalculatedQuantities::curr_dual_infeasibility`
/// / `curr_primal_infeasibility` / `curr_complementarity`.
///
/// Extracted from `solve_ipm` as part of the v0.8 main-loop
/// decomposition. Pure function of `state`.
fn compute_optimality_measures(state: &SolverState) -> OptimalityMeasures {
    let primal_inf = state.constraint_violation();
    let primal_inf_max = compute_primal_inf_max_at_state(state);
    let primal_inf_internal_max = compute_primal_inf_internal_max_at_state(state);

    // Iterative-z dual infeasibility (matches Ipopt's curr_dual_infeasibility):
    // honest KKT residual. If iterative z is inconsistent with ∇f + J^T y the
    // residual stays large and iteration continues.
    let dual_inf = compute_dual_inf_at_state(state);
    let dual_inf_unscaled = compute_dual_inf_unscaled_at_state(state);
    let compl_inf = compute_compl_err_at_state(state);
    OptimalityMeasures {
        primal_inf,
        primal_inf_max,
        primal_inf_internal_max,
        dual_inf,
        dual_inf_unscaled,
        compl_inf,
    }
}

/// Compute the multiplier-based scaling factor `s_d` used by the
/// Ipopt acceptable-tolerance gate, then update
/// `state.consecutive_acceptable`.
///
/// Scaling matches `IpIpoptCalculatedQuantities::ComputeOptimalityErrorScaling`
/// with `s_max=100` and the 1e4 cap preserved for compatibility with
/// the rest of ripopt's tolerance pipeline. Acceptable thresholds
/// match `IpOptErrorConvCheck.cpp:70-121` defaults
/// (acceptable_tol=1e-6, acceptable_constr_viol_tol=1e-2,
/// acceptable_dual_inf_tol=1e10, acceptable_compl_inf_tol=1e-2).
fn track_consecutive_acceptable(
    state: &mut SolverState,
    options: &SolverOptions,
    primal_inf: f64,
    dual_inf: f64,
    dual_inf_unscaled: f64,
    compl_inf: f64,
    multiplier_sum: f64,
    bound_multiplier_sum: f64,
) -> f64 {
    let s_d_for_acc = compute_residual_scaling(multiplier_sum, compute_multiplier_count(state));
    let s_c_for_acc = compute_residual_scaling(bound_multiplier_sum, compute_bound_multiplier_count(state));
    let meets_acc_scaled = primal_inf <= 1e-6
        && dual_inf <= 1e-6 * s_d_for_acc
        && compl_inf <= 1e-6 * s_c_for_acc;
    let meets_acc_unscaled = primal_inf <= 1e-2
        && dual_inf_unscaled <= 1e10
        && compl_inf <= 1e-2;
    // Ipopt acceptable_obj_change_tol gate: |Δf| / max(1, |f|) ≤ tol.
    let obj_change_ok = match state.last_obj_for_acceptable {
        Some(prev) => {
            let denom = state.obj.abs().max(1.0);
            (state.obj - prev).abs() / denom <= options.acceptable_obj_change_tol
        }
        None => true,
    };
    if meets_acc_scaled && meets_acc_unscaled && obj_change_ok {
        state.consecutive_acceptable += 1;
    } else {
        state.consecutive_acceptable = 0;
    }
    s_d_for_acc
}

/// Capture an `IterateSnapshot` if the current iterate meets the
/// acceptable-level thresholds. Overwrites any previous snapshot so the
/// stored point is always the most recent acceptable one.
fn store_acceptable_iterate(
    state: &mut SolverState,
    filter: &Filter,
    iteration: usize,
    primal_inf: f64,
    primal_inf_internal: f64,
    dual_inf: f64,
    dual_inf_unscaled: f64,
    compl_inf: f64,
    multiplier_sum: f64,
    multiplier_count: usize,
    bound_multiplier_sum: f64,
    bound_multiplier_count: usize,
) {
    let info = ConvergenceInfo {
        primal_inf,
        primal_inf_internal,
        dual_inf,
        dual_inf_unscaled,
        compl_inf,
        mu: state.mu,
        objective: state.obj,
        multiplier_sum,
        multiplier_count,
        bound_multiplier_sum,
        bound_multiplier_count,
        x_max_abs: linf_norm(&state.x),
    };
    let s_max: f64 = 100.0;
    let s_d = if info.multiplier_count > 0 {
        s_max.max(info.multiplier_sum / info.multiplier_count as f64) / s_max
    } else {
        1.0
    };
    let s_c = if info.bound_multiplier_count > 0 {
        s_max.max(info.bound_multiplier_sum / info.bound_multiplier_count as f64) / s_max
    } else {
        1.0
    };
    // Pass None for last_obj: snapshot decisions skip the obj-change gate
    // (the gate is for the convergence-check exit, not for capturing an
    // acceptable iterate).
    let opts_for_snap = SolverOptions::default();
    if convergence::meets_acceptable_thresholds(&info, &opts_for_snap, s_d, s_c, None) {
        state.acceptable_iterate = Some(IterateSnapshot::capture(state, filter, iteration));
    }
}

/// Just before triggering full restoration, attempt to restore the most
/// recent acceptable iterate. Returns `Some(SolveResult)` with status
/// Almost-feasibility guard at restoration entry, mirroring Ipopt 3.14
/// `IpBacktrackingLineSearch.cpp:580-600`. Fires when both
///
/// ```text
/// curr_constraint_violation        <= 1e-2 * tol
/// unscaled_curr_constr_violation   <= 1e-1 * constr_viol_tol
/// ```
///
/// hold (the second criterion guards against very-large-tol settings).
/// When the guard fires, restoration is skipped entirely:
///
/// - If an acceptable iterate is cached, restore it and exit
///   `Acceptable` (Ipopt's `ACCEPTABLE_POINT_REACHED` throw at line 591).
/// - Otherwise abort with `NumericalError` (Ipopt's
///   `STEP_COMPUTATION_FAILED` throw at line 597 — "Abort in line
///   search due to no other fall back").
///
/// Returning `None` means the guard did not fire and the caller should
/// proceed to the restoration cascade.
fn check_almost_feasible_guard(
    state: &mut SolverState,
    options: &SolverOptions,
    filter: &mut Filter,
    primal_inf: f64,
    primal_inf_max: f64,
) -> Option<SolveResult> {
    let almost_feasible = primal_inf < 1e-2 * options.tol
        && primal_inf_max < 1e-1 * options.constr_viol_tol;
    if !almost_feasible {
        return None;
    }
    if let Some(snap) = state.acceptable_iterate.take() {
        if options.print_level >= 3 {
            rip_log!(
                "ripopt: almost-feasible guard restoring acceptable iterate from iter {} (pr={:.2e}, pr_max={:.2e}) -> Acceptable",
                snap.iteration, primal_inf, primal_inf_max
            );
        }
        snap.restore(state, filter);
        return Some(make_result(state, SolveStatus::Acceptable));
    }
    if options.print_level >= 3 {
        rip_log!(
            "ripopt: almost-feasible guard with no cached acceptable iterate (pr={:.2e}, pr_max={:.2e}) -> NumericalError",
            primal_inf, primal_inf_max
        );
    }
    Some(make_result(state, SolveStatus::NumericalError))
}

// DEV-7: removed `detect_unbounded`. The objective-magnitude heuristic
// (10 consecutive iters with `obj < -1e20` at feasibility) had no Ipopt
// analog and could mis-fire on legitimate large-objective problems.
// Ipopt 3.14's actual divergence detector is `‖x‖_∞ >
// diverging_iterates_tol` in IpOptErrorConvCheck (`IpOptErrorConvCheck.cpp:255`),
// already wired through `convergence::check_convergence` →
// `ConvergenceStatus::Diverging` → `SolveStatus::DivergingIterates`.

/// Primal-infeasibility divergence detector.
///
/// Tracks consecutive iterations where θ is strictly increasing (by
/// at least 1e-6 relative). After 8 such iterations *and* a cumulative
/// 20%+ growth from the start of the divergence run, sets the
/// force-restoration flag so the main loop re-enters restoration
/// rather than continuing with worsening feasibility.
///
/// Catches the pattern where after NLP restoration the filter accepts
/// steps via a slight φ decrease while θ grows steadily. Ipopt has no
/// direct analogue — this is a ripopt safety valve.
///
/// Extracted from `solve_ipm` as part of the v0.8 main-loop
/// decomposition. Returns `true` when the caller should force
/// restoration on the next step.
/// Constraint-violation history with auxiliary flags driving infeasibility
/// detection: `history` is a bounded ring of recent θ values, `ever_feasible`
/// is sticky once any θ falls below `constr_viol_tol`, and `stall_count`
/// counts consecutive iterations where θ has stagnated within the history
/// window. Owned by the main IPM loop.
struct FeasibilityTracker {
    history: Vec<f64>,
    history_len: usize,
    ever_feasible: bool,
    stall_count: usize,
}

impl FeasibilityTracker {
    fn new(history_len: usize) -> Self {
        Self {
            history: Vec::with_capacity(history_len),
            history_len,
            ever_feasible: false,
            stall_count: 0,
        }
    }
}

// T3.13–T3.22 (2026-04-30): retired the entire ripopt-specific stall machinery
// (`ProgressStallTracker`, `StallDecision`, `detect_and_handle_progress_stall`,
// `handle_near_tolerance_stall`, `classify_near_tolerance_stall_outcome`,
// `try_boost_mu_for_stall`, `try_force_mu_decrease_in_fixed_mode`,
// `update_stall_counters_and_check_limit`,
// `boost_mu_and_switch_to_fixed_with_stall_reset`,
// `reset_stall_counters_and_filter`, and the `stall_iter_limit` /
// `early_stall_timeout` options that gated them). No Ipopt analog.
// Stall handling in Ipopt 3.14 is owned by the filter line search, the
// watchdog reversal (`IpBacktrackingLineSearch.cpp:773`), the AcceptableLevel
// termination (`IpOptErrorConvCheck.cpp:328-330`), and the restoration phase
// — collectively those already cover the cases the retired machinery was
// catching, without ripopt-specific μ-boost flips that violated the monotone
// invariant.

/// Track constraint-violation history. Pushes `primal_inf` into the
/// θ-history ring, updates the sticky `ever_feasible` flag, and clears
/// the auxiliary stall counter when the iterate has been feasible.
///
/// T3.17: the proactive_infeasibility_detection branch was retired
/// (no Ipopt analog). LocalInfeasibility detection now flows only
/// through the restoration cascade (`classify_exhausted_restoration_attempt`)
/// and the MaxIter exit (`try_classify_max_iter_infeasibility`), both of
/// which test the same "stationary infeasible point" criterion.
fn track_feasibility_and_detect_infeasibility(
    _state: &SolverState,
    options: &SolverOptions,
    _iteration: usize,
    primal_inf: f64,
    feas: &mut FeasibilityTracker,
) -> Option<SolveResult> {
    if feas.history.len() >= feas.history_len {
        feas.history.remove(0);
    }
    feas.history.push(primal_inf);

    if primal_inf < options.constr_viol_tol {
        feas.ever_feasible = true;
        feas.stall_count = 0;
    }
    None
}

/// Emit one row of the per-iteration TSV trace (for the
/// direction-diff harness with Ipopt's IntermediateCallback).
///
/// Gated on `trace::is_enabled()` (RIP_TRACE_TSV env var). No-op in
/// production runs. Also resets the per-iteration scratch values
/// (`last_alpha_primal_max`, `last_tau_used`, `last_soc_accepted`)
/// so the next iteration starts clean.
///
/// Extracted from `solve_ipm` as part of the v0.8 main-loop
/// decomposition (pre-work step 2). Pure side-effect function.
#[allow(clippy::too_many_arguments)]
/// log10(max Σ_i / min Σ_i) where Σ_i = z_l_i/s_l_i + z_u_i/s_u_i over
/// variables with at least one active bound. High values (~10+) signal
/// ill-conditioning of the condensed KKT at low μ and are a suspect for
/// α=1/low-μ direction-quality issues. Returns NaN when no Σ_i > 0.
fn compute_sigma_condition(state: &SolverState) -> f64 {
    let mut mn = f64::INFINITY;
    let mut mx = 0.0_f64;
    // Phase 6c.2: accumulate Σ_i = z_L_i/s_L_i + z_U_i/s_U_i across the
    // union of variables with at least one finite bound, walking the
    // compressed bound mirrors. Build per-variable Σ via two passes
    // (L then U) keyed by full index to match the original semantics
    // exactly: a single Σ_i value per variable, whether it has 1 or 2
    // active bounds.
    let mut sigma_by_var = vec![0.0_f64; state.n];
    let mut active = vec![false; state.n];
    for k in 0..state.bound_layout.n_x_l {
        let i = state.bound_layout.x_l_to_full[k];
        sigma_by_var[i] += state.z_l_compressed[k] / slack_xl(state, i);
        active[i] = true;
    }
    for k in 0..state.bound_layout.n_x_u {
        let i = state.bound_layout.x_u_to_full[k];
        sigma_by_var[i] += state.z_u_compressed[k] / slack_xu(state, i);
        active[i] = true;
    }
    for i in 0..state.n {
        if !active[i] {
            continue;
        }
        let s_i = sigma_by_var[i];
        if s_i > 0.0 {
            mn = mn.min(s_i);
            mx = mx.max(s_i);
        }
    }
    if mn.is_finite() && mn > 0.0 && mx > 0.0 {
        (mx / mn).log10()
    } else {
        f64::NAN
    }
}

/// Per-iteration trace intermediates captured during the line-search /
/// direction-compute sub-phases, drained into the TSV at iteration-end.
/// The α_primal_max and τ values are set after compute_alpha_max; the
/// SOC-accepted flag is set inside the line-search loop. Reset to
/// defaults after each emission so a missing assignment shows as NaN
/// rather than stale data.
#[derive(Default)]
struct TraceMetadata {
    alpha_primal_max: Option<f64>,
    tau_used: Option<f64>,
    soc_accepted: bool,
}

fn emit_trace_row_if_enabled(
    state: &SolverState,
    iteration: usize,
    primal_inf: f64,
    dual_inf: f64,
    compl_inf: f64,
    ls_steps: usize,
    inertia_params: &InertiaCorrectionParams,
    last_mehrotra_sigma: Option<f64>,
    trace_meta: &mut TraceMetadata,
) {
    if !trace::is_enabled() {
        return;
    }
    let dx_inf = linf_norm(&state.dx);
    let dzl_inf = linf_norm(&state.dz_l_compressed);
    let dzu_inf = linf_norm(&state.dz_u_compressed);
    let sigma_cond = compute_sigma_condition(state);
    trace::emit(&trace::TraceRow {
        iter: iteration,
        obj: state.obj / state.obj_scaling,
        inf_pr: primal_inf,
        inf_du: dual_inf,
        compl: compl_inf,
        mu: state.mu,
        alpha_pr: state.alpha_primal,
        alpha_du: state.alpha_dual,
        alpha_aff_p: f64::NAN,
        alpha_aff_d: f64::NAN,
        mu_aff: f64::NAN,
        sigma: last_mehrotra_sigma.unwrap_or(f64::NAN),
        mu_pc: f64::NAN,
        delta_w: inertia_params.delta_w_last,
        delta_c: 0.0,
        dx_inf,
        dzl_inf,
        dzu_inf,
        mcc_iters: 0,
        ls: ls_steps as u32,
        accepted: true,
        alpha_primal_max: trace_meta.alpha_primal_max.unwrap_or(f64::NAN),
        tau_used: trace_meta.tau_used.unwrap_or(f64::NAN),
        sigma_cond,
        soc_accepted: trace_meta.soc_accepted,
    });
    trace_meta.alpha_primal_max = None;
    trace_meta.tau_used = None;
    trace_meta.soc_accepted = false;
}

/// Populate the per-iteration IterateSnapshot and invoke the user
/// intermediate callback.
///
/// Sets the global "current iterate" for GetCurrentIterate / GetViolations
/// access inside the callback, then runs it. If the callback returns false
/// the solver must terminate with `UserRequestedStop`; returns
/// `Some(SolveResult)` in that case. The snapshot is cleared before return
/// regardless.
///
/// Ipopt parallel: IpIpoptAlg intermediate-callback invocation (around
/// IpIpoptAlg.cpp:560–610 and IpOptionsList intermediate_callback wiring).
///
/// Extracted from `solve_ipm` as part of the v0.8 main-loop decomposition
/// (pre-work step 2).
/// Build the per-iteration `IterateSnapshot` exposed via
/// `GetCurrentIterate`/`GetCurrentViolations`. Computes per-bound
/// violations, complementarity products, and the Lagrangian gradient
/// `grad_f + J^T y - z_L + z_U`. Pure function of `state`.
fn build_iterate_snapshot(state: &SolverState) -> crate::intermediate::IterateSnapshot {
    use crate::intermediate::IterateSnapshot;
    let n = state.n;
    let m = state.m;

    let mut x_l_viol = vec![0.0; n];
    let mut x_u_viol = vec![0.0; n];
    let mut compl_xl = vec![0.0; n];
    let mut compl_xu = vec![0.0; n];
    // Phase 6d.3: bound-side reads consume the compressed mirror; the
    // n-wide infeasibility sweep over x_l/x_u still needs the n-wide
    // viol arrays so split the loop.
    for i in 0..n {
        if state.x_l_at(i).is_finite() {
            x_l_viol[i] = (state.x_l_at(i) - state.x[i]).max(0.0);
        }
        if state.x_u_at(i).is_finite() {
            x_u_viol[i] = (state.x[i] - state.x_u_at(i)).max(0.0);
        }
    }
    for k in 0..state.bound_layout.n_x_l {
        let i = state.bound_layout.x_l_to_full[k];
        compl_xl[i] = (state.x[i] - state.x_l_at(i)) * state.z_l_compressed[k];
    }
    for k in 0..state.bound_layout.n_x_u {
        let i = state.bound_layout.x_u_to_full[k];
        compl_xu[i] = (state.x_u_at(i) - state.x[i]) * state.z_u_compressed[k];
    }
    // grad_lag = grad_f + J^T y - z_l + z_u
    let mut grad_lag = state.grad_f.clone();
    accumulate_jt_y(state, &mut grad_lag);
    for k in 0..state.bound_layout.n_x_l {
        let i = state.bound_layout.x_l_to_full[k];
        grad_lag[i] -= state.z_l_compressed[k];
    }
    for k in 0..state.bound_layout.n_x_u {
        let i = state.bound_layout.x_u_to_full[k];
        grad_lag[i] += state.z_u_compressed[k];
    }
    // Materialise m-form g for the user-facing intermediate callback
    // (Ipopt's `IpoptCalculatedQuantities::curr_c`/`curr_d` projected
    // back to TNLP's wire format).
    let g_m: Vec<f64> = state.g_combined();
    let mut constr_viol = vec![0.0; m];
    let mut compl_g_vec = vec![0.0; m];
    for i in 0..m {
        if state.g_l_at(i).is_finite() && g_m[i] < state.g_l_at(i) {
            constr_viol[i] = state.g_l_at(i) - g_m[i];
        } else if state.g_u_at(i).is_finite() && g_m[i] > state.g_u_at(i) {
            constr_viol[i] = g_m[i] - state.g_u_at(i);
        }
        // Complementarity: lambda_i * c_i where c_i is the active constraint slack
        let yi = state.y_at(i);
        if state.g_l_at(i).is_finite() && state.g_u_at(i).is_finite() {
            // Equality or range: use min slack
            compl_g_vec[i] = yi * (g_m[i] - state.g_l_at(i)).min(state.g_u_at(i) - g_m[i]);
        } else if state.g_l_at(i).is_finite() {
            compl_g_vec[i] = yi * (g_m[i] - state.g_l_at(i));
        } else if state.g_u_at(i).is_finite() {
            compl_g_vec[i] = yi * (state.g_u_at(i) - g_m[i]);
        }
    }
    IterateSnapshot {
        x: state.x.clone(),
        // Phase 6d.2: materialize full-`n` z_l/z_u from compressed for
        // the public callback API.
        z_l: state.z_l_combined(),
        z_u: state.z_u_combined(),
        g: g_m,
        lambda: state.y_combined(),
        x_l_violation: x_l_viol,
        x_u_violation: x_u_viol,
        compl_x_l: compl_xl,
        compl_x_u: compl_xu,
        grad_lag_x: grad_lag,
        constraint_violation: constr_viol,
        compl_g: compl_g_vec,
    }
}

fn populate_snapshot_and_invoke_callback(
    state: &SolverState,
    iteration: usize,
    primal_inf: f64,
    dual_inf: f64,
    prev_ic_delta_w: f64,
    ls_steps: usize,
    options: &SolverOptions,
) -> Option<SolveResult> {
    crate::intermediate::set_current_iterate(Some(build_iterate_snapshot(state)));

    // Invoke intermediate callback (if registered)
    let d_norm = linf_norm(&state.dx);
    let continue_ok = crate::intermediate::invoke_intermediate(
        0, // alg_mod: 0 = regular mode (not in restoration)
        iteration,
        state.obj / state.obj_scaling,
        primal_inf,
        dual_inf,
        state.mu,
        d_norm,
        prev_ic_delta_w,
        state.alpha_dual,
        state.alpha_primal,
        ls_steps,
    );
    crate::intermediate::set_current_iterate(None);
    if !continue_ok {
        if options.print_level >= 5 {
            rip_log!("ripopt: User requested stop via intermediate callback");
        }
        return Some(make_result(state, SolveStatus::UserRequestedStop));
    }
    None
}

/// Emit a single iteration-row line to the solver log.
///
/// Ipopt parallel: `IpIpoptAlg::OutputIteration` (IpIpoptAlg.cpp:520–560).
/// Header is reprinted every 25 data rows.
///
/// Extracted from `solve_ipm` as part of the v0.8 main-loop decomposition
/// (pre-work step 2). Pure side-effect function — no solver state mutation
/// beyond incrementing `log_line_count`.
fn log_iteration_row(
    iteration: usize,
    state: &SolverState,
    primal_inf: f64,
    dual_inf: f64,
    compl_inf: f64,
    ls_steps: usize,
    log_line_count: &mut usize,
    options: &SolverOptions,
    inertia_params: &InertiaCorrectionParams,
) {
    if options.print_level < 3 {
        return;
    }
    // Reprint header every 25 data rows for readability
    if *log_line_count > 0 && *log_line_count % 25 == 0 {
        rip_log!(
            "{:>4}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}  {:>3}",
            "iter", "objective", "inf_pr", "inf_du", "compl", "lg(mu)", "alpha_pr", "alpha_du", "ls"
        );
    }
    let lg_mu = if state.mu > 0.0 { state.mu.log10() } else { f64::NEG_INFINITY };
    rip_log!(
        "{:>4}  {:>14.7e}  {:>10.2e}  {:>10.2e}  {:>10.2e}  {:>10.1}  {:>8.2e}  {:>8.2e}  {:>3}",
        iteration,
        state.obj / state.obj_scaling,
        primal_inf,
        dual_inf,
        compl_inf,
        lg_mu,
        state.alpha_primal,
        state.alpha_dual,
        ls_steps,
    );
    *log_line_count += 1;

    // Targeted dual-stuck probe (print_level >= 5). Surfaces:
    //   * |y_c|_inf, |y_d|_inf, |z_L|_inf, |z_U|_inf — has the multiplier
    //     vector escaped through repeated kappa_sigma clipping?
    //   * |dy_c|_inf, |dy_d|_inf, |dz_L|_inf, |dz_U|_inf — does the dual
    //     direction even have non-trivial magnitude?
    //   * α_du clamp source: which row of (z_L, z_U) hits the FTB cap, and
    //     by how much. Identifies a wedged multiplier row that the search
    //     direction never repairs.
    if options.print_level >= 5 {
        let yc_inf = linf_norm(&state.y_c);
        let yd_inf = linf_norm(&state.y_d);
        let zl_inf = linf_norm(&state.z_l_compressed);
        let zu_inf = linf_norm(&state.z_u_compressed);
        let dyc_inf = linf_norm(&state.dy_c);
        let dyd_inf = linf_norm(&state.dy_d);
        let dzl_inf = linf_norm(&state.dz_l_compressed);
        let dzu_inf = linf_norm(&state.dz_u_compressed);

        // Identify the (block, compressed-index, ratio) of the binding FTB
        // row on the dual side at the *current* iterate using the *current*
        // dz directions and the iteration's tau. This mirrors the inner
        // loop of `fraction_to_boundary_dual_z_min` but records the arg-min.
        let tau = (1.0 - state.mu).max(options.tau_min);
        let mut min_ratio = f64::INFINITY;
        let mut bind_block = "-";
        let mut bind_idx: usize = usize::MAX;
        for k in 0..state.bound_layout.n_x_l {
            let d = state.dz_l_compressed[k];
            if d < 0.0 {
                let r = -tau * state.z_l_compressed[k] / d;
                if r < min_ratio { min_ratio = r; bind_block = "zL"; bind_idx = k; }
            }
        }
        for k in 0..state.bound_layout.n_x_u {
            let d = state.dz_u_compressed[k];
            if d < 0.0 {
                let r = -tau * state.z_u_compressed[k] / d;
                if r < min_ratio { min_ratio = r; bind_block = "zU"; bind_idx = k; }
            }
        }
        let (bind_z, bind_dz) = if bind_idx == usize::MAX {
            (f64::NAN, f64::NAN)
        } else if bind_block == "zL" {
            (state.z_l_compressed[bind_idx], state.dz_l_compressed[bind_idx])
        } else {
            (state.z_u_compressed[bind_idx], state.dz_u_compressed[bind_idx])
        };
        let dw_last = inertia_params.delta_w_last;
        let dc_last = inertia_params.delta_c_last;
        rip_log!(
            "ripopt: iter{}-probe: |y_c|={:.2e} |y_d|={:.2e} |z_L|={:.2e} |z_U|={:.2e}  |dy_c|={:.2e} |dy_d|={:.2e} |dz_L|={:.2e} |dz_U|={:.2e}  ftb_du={}@{} z={:.2e} dz={:.2e} ratio={:.2e}  dw_last={:.2e} dc_last={:.2e}",
            iteration,
            yc_inf, yd_inf, zl_inf, zu_inf,
            dyc_inf, dyd_inf, dzl_inf, dzu_inf,
            bind_block,
            if bind_idx == usize::MAX { -1i64 } else { bind_idx as i64 },
            bind_z, bind_dz, min_ratio,
            dw_last, dc_last,
        );
    }
}

/// Detect constraints that are linear in `x` (constant Jacobian, zero
/// contribution to the Hessian). When `options.detect_linear_constraints`
/// is set, run `crate::linearity::detect_linear_constraints` on the
/// original unscaled problem. Returns `Some(flags)` only when at least one
/// linear constraint is found; `None` otherwise.
fn detect_linear_constraint_flags<P: NlpProblem>(
    problem: &P,
    options: &SolverOptions,
    x0: &[f64],
    m_sc: usize,
) -> Option<Vec<bool>> {
    if !(options.detect_linear_constraints && m_sc > 0) {
        return None;
    }
    let flags = crate::linearity::detect_linear_constraints(problem, x0);
    let n_linear = flags.iter().filter(|&&f| f).count();
    if n_linear == 0 {
        return None;
    }
    if options.print_level >= 5 {
        rip_log!(
            "ripopt: Detected {}/{} linear constraints (Hessian contribution skipped)",
            n_linear, m_sc
        );
    }
    Some(flags)
}

/// B1.2: push the explicit slack iterate `s` strictly into the interior
/// of `[g_l, g_u]` for inequality rows, mirroring `push_initial_point_from_bounds`
/// for variable bounds. Equality rows are left at the sentinel `s = g_l = g_u`.
///
/// Ipopt 3.14 (`IpDefaultIterateInitializer.cpp:526-599`, slack branch):
///   pL = min(κ1·max(|g_L|, 1), κ2·(g_U − g_L))   [two-sided]
///   pL = κ1·max(|g_L|, 1)                         [one-sided lower]
///   pU = κ1·max(|g_U|, 1)                         [one-sided upper]
/// where κ1 = `slack_bound_push` (defaults to `bound_push`) and κ2 =
/// `slack_bound_frac` (defaults to `bound_frac`). Initial s is the
/// projection of g(x) into the resulting open interval; this guarantees
/// strictly-positive `s_L = s − g_l` and `s_U = g_u − s` so the IPM's
/// log-barrier on the slack is well defined at iteration 0.
fn initialize_slack_iterate(state: &mut SolverState, m: usize, options: &SolverOptions) {
    // Phase 5f: walk d-block natively. `s[k] = clamp(d_x[k], d_l[k]+pL, d_u[k]-pU)`.
    let kappa1 = options.bound_push;
    let kappa2 = options.bound_frac;
    let _ = m;
    for k in 0..state.layout.n_d {
        let dl = state.d_l_at(k);
        let du = state.d_u_at(k);
        let l_fin = dl.is_finite();
        let u_fin = du.is_finite();
        let mut s_i = state.d_x[k];
        if l_fin && u_fin {
            let range = du - dl;
            let p_l = (kappa1 * dl.abs().max(1.0)).min(kappa2 * range);
            let p_u = (kappa1 * du.abs().max(1.0)).min(kappa2 * range);
            s_i = s_i.max(dl + p_l).min(du - p_u);
        } else if l_fin {
            let p_l = kappa1 * dl.abs().max(1.0);
            s_i = s_i.max(dl + p_l);
        } else if u_fin {
            let p_u = kappa1 * du.abs().max(1.0);
            s_i = s_i.min(du - p_u);
        }
        state.s[k] = s_i;
    }
}

/// Initialize constraint slack barrier multipliers `v_l`, `v_u` (Ipopt's
/// `v_L`, `v_U`). For each inequality constraint side,
/// `v = mu_init / max(slack, 1e-20)`. Equality rows (`g_l ≈ g_u`) are
/// skipped. Mirrors Ipopt's `IpDefaultIterateInitializer.cpp`: slack-bound
/// multipliers are initialized to `bound_mult_init_val` (default 1.0), the
/// same constant used for x-bound multipliers, NOT to `mu_init / slack`.
///
/// Mirrors Ipopt's `IpDefaultIterateInitializer.cpp`: slack-bound
/// multipliers are initialized to `bound_mult_init_val` (default 1.0),
/// the same constant used for x-bound multipliers. Equality rows
/// (`g_l ≈ g_u`) are skipped.
///
/// P6: Previously ripopt also set `y_d := v_U − v_L` when
/// `least_squares_mult_init` was OFF, which is not present in Ipopt
/// (Ipopt's `IpLeastSquareMults.cpp:53-81` chooses (y_c, y_d) jointly
/// from one 4-block LS, not piecewise). Removed for alignment.
fn initialize_constraint_slack_multipliers(state: &mut SolverState, m: usize, options: &SolverOptions) {
    let v_init = options.bound_mult_init_val;
    for i in 0..m {
        if constraint_is_equality(state, i) {
            continue;
        }
        if state.g_l_at(i).is_finite() {
            state.set_v_l_at(i, v_init);
        }
        if state.g_u_at(i).is_finite() {
            state.set_v_u_at(i, v_init);
        }
    }
}

/// Apply user-provided warm-start multipliers (`y`, `z_l`, `z_u`) and then
/// run `WarmStartInitializer::initialize`, which pushes `x` off the bounds
/// and rescales `(z_l, z_u)` if needed so (x − x_l)·z_l ≈ μ is well-posed.
/// No-op when `options.warm_start` is false.
fn apply_warm_start(state: &mut SolverState, options: &SolverOptions) {
    if !options.warm_start {
        return;
    }
    if let Some(ref init_y) = options.warm_start_y {
        let len = init_y.len().min(state.m);
        for i in 0..len {
            state.set_y_at(i, init_y[i]);
        }
    }
    if let Some(ref init_z_l) = options.warm_start_z_l {
        // Phase 6d.6: warm-start z is user-supplied full-`n`; project
        // onto the compressed bound storage (unbounded sides discarded).
        let proj = state.bound_layout.project_l(init_z_l);
        let len = proj.len().min(state.z_l_compressed.len());
        state.z_l_compressed[..len].copy_from_slice(&proj[..len]);
    }
    if let Some(ref init_z_u) = options.warm_start_z_u {
        let proj = state.bound_layout.project_u(init_z_u);
        let len = proj.len().min(state.z_u_compressed.len());
        state.z_u_compressed[..len].copy_from_slice(&proj[..len]);
    }
    // B9: warm-start the slack iterate `s` and its bound multipliers
    // `v_l`, `v_u`. The default slack-push initializer ran before us
    // (`initialize_slack_iterate`); we overwrite with user values, then
    // project s back into a strict interior to keep the barrier well-
    // defined (use the same κ1/κ2 push the cold-start path uses).
    if let Some(ref init_s) = options.warm_start_s {
        // init_s is combined-indexed (size m). Phase 3d: drop equality slots
        // and project onto the d-block storage.
        let m_in = init_s.len().min(state.m);
        let kappa1 = options.bound_push;
        let kappa2 = options.bound_frac;
        for i in 0..m_in {
            if constraint_is_equality(state, i) {
                continue;
            }
            let mut s_i = init_s[i];
            let l_fin = state.g_l_at(i).is_finite();
            let u_fin = state.g_u_at(i).is_finite();
            if l_fin && u_fin {
                let range = state.g_u_at(i) - state.g_l_at(i);
                let p_l = (kappa1 * state.g_l_at(i).abs().max(1.0)).min(kappa2 * range);
                let p_u = (kappa1 * state.g_u_at(i).abs().max(1.0)).min(kappa2 * range);
                s_i = s_i.max(state.g_l_at(i) + p_l).min(state.g_u_at(i) - p_u);
            } else if l_fin {
                let p_l = kappa1 * state.g_l_at(i).abs().max(1.0);
                s_i = s_i.max(state.g_l_at(i) + p_l);
            } else if u_fin {
                let p_u = kappa1 * state.g_u_at(i).abs().max(1.0);
                s_i = s_i.min(state.g_u_at(i) - p_u);
            }
            state.set_s_at(i, s_i);
        }
    }
    if let Some(ref init_v_l) = options.warm_start_v_l {
        // Phase 3e: user supplies combined m-form v_l; route through the
        // combined-indexed setter so values land in the d-block storage.
        let len = init_v_l.len().min(state.m);
        for i in 0..len {
            state.set_v_l_at(i, init_v_l[i]);
        }
    }
    if let Some(ref init_v_u) = options.warm_start_v_u {
        let len = init_v_u.len().min(state.m);
        for i in 0..len {
            state.set_v_u_at(i, init_v_u[i]);
        }
    }
    // Phase 6d.6: WarmStartInitializer indexes z by full var idx. Pass
    // a materialized full-`n` view, then project the result back into
    // compressed storage.
    let mut z_l_full = state.z_l_combined();
    let mut z_u_full = state.z_u_combined();
    let x_l_full = state.x_l_combined();
    let x_u_full = state.x_u_combined();
    state.mu = WarmStartInitializer::initialize(
        &mut state.x,
        &mut z_l_full,
        &mut z_u_full,
        &x_l_full,
        &x_u_full,
        options,
    );
    state.z_l_compressed = state.bound_layout.project_l(&z_l_full);
    state.z_u_compressed = state.bound_layout.project_u(&z_u_full);
}

/// Evaluate the NLP at the initial point. If the evaluation fails, produces
/// NaN/Inf in `obj` or `grad_f`, try up to three bound-push perturbations
/// (1%, 10%, 50% of each bound range) and retry. If every perturbation also
/// fails, return `Err(SolveResult)` with `SolveStatus::EvaluationError`.
fn initial_evaluate_with_recovery<P: NlpProblem>(
    state: &mut SolverState,
    problem: &P,
    lbfgs_state: &mut Option<LbfgsIpmState>,
    linear_constraints: Option<&[bool]>,
    lbfgs_mode: bool,
    n: usize,
    options: &SolverOptions,
) -> Result<(), SolveResult> {
    let init_eval_ok = evaluate_and_refresh_lbfgs(state, problem, lbfgs_state, linear_constraints, lbfgs_mode);

    if init_eval_ok && obj_and_grad_finite(state) {
        return Ok(());
    }

    let x_saved = state.x.clone();
    for &push_factor in &[1e-2, 1e-1, 0.5] {
        state.x.copy_from_slice(&x_saved);
        for i in 0..n {
            if state.x_l_at(i).is_finite() && state.x_u_at(i).is_finite() {
                let range = state.x_u_at(i) - state.x_l_at(i);
                let push = push_factor * range;
                if range > 2.0 * push {
                    state.x[i] = state.x[i].max(state.x_l_at(i) + push).min(state.x_u_at(i) - push);
                } else {
                    state.x[i] = 0.5 * (state.x_l_at(i) + state.x_u_at(i));
                }
            } else if state.x_l_at(i).is_finite() {
                let push = push_factor * state.x_l_at(i).abs().max(1.0);
                state.x[i] = state.x[i].max(state.x_l_at(i) + push);
            } else if state.x_u_at(i).is_finite() {
                let push = push_factor * state.x_u_at(i).abs().max(1.0);
                state.x[i] = state.x[i].min(state.x_u_at(i) - push);
            }
        }
        reseed_bound_multipliers_from_mu(state, options.mu_init);
        let perturb_ok = evaluate_and_refresh_lbfgs(state, problem, lbfgs_state, linear_constraints, lbfgs_mode);
        if perturb_ok && obj_and_grad_finite(state) {
            return Ok(());
        }
    }
    Err(make_result(state, SolveStatus::EvaluationError))
}

/// Compute gradient-based NLP scaling at the initial point `x0`, matching
/// Ipopt's `nlp_scaling_method = gradient-based`:
///
///   * Objective: if `||∇f(x0)||_∞ > 100`, set `obj_scaling = 100/||∇f||_∞`,
///     clamped below by `1e-8` (Ipopt's `nlp_scaling_min_value` default).
///   * Constraints: row-wise on `J(x0)`. If `max_j |J_{ij}| > 100`, set
///     `g_scaling[i] = 100/max_j |J_{ij}|`, clamped below by `1e-8`.
///
/// User-provided scalings (`options.user_obj_scaling`, `options.user_g_scaling`)
/// take priority — when either is set, skips automatic scaling entirely.
fn compute_nlp_scaling<P: NlpProblem>(
    problem: &P,
    options: &SolverOptions,
    x0: &[f64],
    jac_rows_sc: &[usize],
) -> (f64, Vec<f64>) {
    use crate::options::NlpScalingMethod;
    let n_sc = problem.num_variables();
    let m_sc = problem.num_constraints();

    // For backwards compatibility: setting any user_*_scaling field
    // short-circuits to user-supplied values, regardless of the method
    // option. This matches the pre-T-MIT-C contract.
    let user_supplied =
        options.user_obj_scaling.is_some() || options.user_g_scaling.is_some();

    let (mut os, gs) = match options.nlp_scaling_method {
        NlpScalingMethod::None => (1.0, vec![1.0; m_sc]),
        NlpScalingMethod::User => {
            let os = options.user_obj_scaling.unwrap_or(1.0);
            let gs = options.user_g_scaling.clone().unwrap_or_else(|| vec![1.0; m_sc]);
            (os, gs)
        }
        NlpScalingMethod::Gradient if user_supplied => {
            let os = options.user_obj_scaling.unwrap_or(1.0);
            let gs = options.user_g_scaling.clone().unwrap_or_else(|| vec![1.0; m_sc]);
            (os, gs)
        }
        NlpScalingMethod::Gradient => {
            let max_grad = options.nlp_scaling_max_gradient;
            let min_val = options.nlp_scaling_min_value;
            let obj_target = options.nlp_scaling_obj_target_gradient;

            let mut grad_f0 = vec![0.0; n_sc];
            let grad_ok = problem.gradient(x0, true, &mut grad_f0);
            let grad_amax = if grad_ok { linf_norm(&grad_f0) } else { 0.0 };

            // Mirrors GradientScaling at IpGradientScaling.cpp:101-128.
            let os = if obj_target > 0.0 {
                if grad_amax > 0.0 && grad_amax.is_finite() {
                    (obj_target / grad_amax).max(min_val)
                } else {
                    1.0_f64.max(min_val)
                }
            } else if grad_amax > max_grad && grad_amax.is_finite() {
                (max_grad / grad_amax).max(min_val)
            } else {
                1.0_f64.max(min_val)
            };

            let gs = compute_constraint_row_scaling(
                problem,
                x0,
                jac_rows_sc,
                m_sc,
                max_grad,
                min_val,
                options.nlp_scaling_constr_target_gradient,
            );
            (os, gs)
        }
    };

    // Apply obj_scaling_factor on top — matches StandardScalingBase
    // (`IpNLPScaling.cpp:276`: df_ *= obj_scaling_factor_). This is
    // applied for every method, including `None`.
    os *= options.obj_scaling_factor;
    (os, gs)
}

/// Compute the per-constraint row scaling for gradient-based NLP
/// scaling. Returns a length-`m_sc` vector of scale factors that map
/// each constraint row to a per-row Jacobian Linf bounded by
/// `max_gradient` (clamped below by `min_value`). Falls back to all
/// 1.0 when Jacobian evaluation fails. Mirrors Ipopt 3.14's
/// `GradientScaling::DetermineScalingParametersImpl`
/// (`IpGradientScaling.cpp:140-180`) — Ipopt scales unconditionally on
/// the initial point's Jacobian; an "init_cv >= 1e6" early-exit was
/// removed (no Ipopt analog and was suppressing scaling exactly on the
/// highly-infeasible Mittelmann starts where it is most needed).
fn compute_constraint_row_scaling<P: NlpProblem>(
    problem: &P,
    x0: &[f64],
    jac_rows_sc: &[usize],
    m_sc: usize,
    max_gradient: f64,
    min_value: f64,
    constr_target_gradient: f64,
) -> Vec<f64> {
    let mut gs = vec![1.0; m_sc];
    if m_sc == 0 {
        return gs;
    }
    let mut jac_vals0 = vec![0.0; jac_rows_sc.len()];
    if !problem.jacobian_values(x0, false, &mut jac_vals0) {
        return gs;
    }
    let mut row_max = vec![0.0f64; m_sc];
    for (idx, &row) in jac_rows_sc.iter().enumerate() {
        let v = jac_vals0[idx].abs();
        if v.is_finite() && v > row_max[row] {
            row_max[row] = v;
        }
    }
    if constr_target_gradient > 0.0 {
        // Ipopt's override path (`IpGradientScaling.cpp:171`): every row
        // gets the same scale `target / max(row_max)`. Skip if the
        // global max is zero (degenerate Jacobian).
        let global_max = row_max.iter().cloned().fold(0.0_f64, f64::max);
        if global_max > 0.0 && global_max.is_finite() {
            let s = (constr_target_gradient / global_max).max(min_value);
            for gi in gs.iter_mut() {
                *gi = s;
            }
        }
        return gs;
    }
    for i in 0..m_sc {
        if row_max[i] > max_gradient {
            gs[i] = (max_gradient / row_max[i]).max(min_value);
        }
    }
    gs
}

/// Core IPM solver implementation.
fn solve_ipm<P: NlpProblem>(problem: &P, options: &SolverOptions) -> SolveResult {
    let n_sc = problem.num_variables();
    let m_sc = problem.num_constraints();

    let mut x0 = vec![0.0; n_sc];
    problem.initial_point(&mut x0);

    let (jac_rows_sc, _) = problem.jacobian_structure();
    let (obj_scaling, g_scaling) = compute_nlp_scaling(problem, options, &x0, &jac_rows_sc);

    if options.print_level >= 5
        && (obj_scaling != 1.0 || g_scaling.iter().any(|&s| s != 1.0))
    {
        let n_scaled_g = g_scaling.iter().filter(|&&s| s != 1.0).count();
        rip_log!(
            "ripopt: NLP scaling: obj_scaling={:.4e}, {}/{} constraints scaled",
            obj_scaling, n_scaled_g, m_sc
        );
    }

    let linear_constraints = detect_linear_constraint_flags(problem, options, &x0, m_sc);

    let scaled = ScaledProblem {
        inner: problem,
        obj_scaling,
        g_scaling: g_scaling.clone(),
        jac_rows: jac_rows_sc,
        bound_relax_factor: options.bound_relax_factor,
        constr_viol_tol: options.constr_viol_tol,
        nlp_lower_bound_inf: options.nlp_lower_bound_inf,
        nlp_upper_bound_inf: options.nlp_upper_bound_inf,
    };
    // Mirrors Ipopt 3.14's IpOrigIpoptNLP per-call Eval_Error checks:
    // wrap the scaled problem with a NaN/Inf guard so every eval
    // reaching the IPM core is guaranteed finite. Non-finite outputs
    // are converted to `false` returns, treated as trial-point
    // rejections by the line search (and as EvaluationError →
    // NumericalBreakdown at the current iterate).
    let finite_checked = FiniteCheckedProblem::new(&scaled);
    let problem = &finite_checked; // shadow: all subsequent code uses the wrapped problem

    let mut state = SolverState::new(problem, options);
    state.obj_scaling = obj_scaling;
    // Project the combined per-row scaling onto the c-block / d-block
    // (Phase 3 split storage). The user's NlpProblem trait stays
    // combined-indexed, so g_scaling is still a length-m Vec at this
    // boundary; SolverState owns the split form.
    state.c_scaling = state.layout.project_c(&g_scaling);
    state.d_scaling = state.layout.project_d(&g_scaling);
    let n = state.n;
    let m = state.m;

    // L-BFGS-in-IPM mode
    let lbfgs_mode = options.hessian_approximation_lbfgs;
    let mut lbfgs_state = if lbfgs_mode {
        if options.print_level >= 5 {
            rip_log!("ripopt: Using L-BFGS Hessian approximation (limited-memory mode)");
        }
        Some(LbfgsIpmState::new(n))
    } else {
        None
    };

    apply_warm_start(&mut state, options);

    // After A7.6 the IPM uses only the augmented (4-block) KKT path,
    // which carries its own dense LDLᵀ solver — the global `lin_solver`
    // selection / sparse threshold / Schur-density gates are no longer
    // consulted, but `use_sparse` is still computed for downstream
    // diagnostics (NLP scaling threshold, large-scale tracing).
    let use_sparse = (n + m) >= options.sparse_threshold;
    let mut inertia_params = InertiaCorrectionParams::default();

    // Initialize filter
    let mut filter = Filter::new(1e4);
    filter.set_obj_max_inc(options.obj_max_inc);
    filter.set_alpha_min_frac(options.alpha_min_frac);
    filter.set_filter_reset_options(options.filter_reset_trigger, options.max_filter_resets);
    // DEV-36: plumb Ipopt `theta_min_fact` / `theta_max_fact` options.
    filter.set_theta_factors(options.theta_min_fact, options.theta_max_fact);

    // Mehrotra centering parameter from the last iteration's predictor step.
    // Used in the Free-mode mu update: when sigma is available, mu = sigma * mu_current
    // gives a more aggressive (and adaptive) decrease than the Loqo oracle.
    let mut last_mehrotra_sigma: Option<f64> = None;
    // Per-iteration trace intermediates captured during the line-search /
    // direction-compute sub-phases, drained into the TSV at iteration-end.
    let mut trace_meta = TraceMetadata::default();

    // Free/fixed mu mode state (replaces ad-hoc stall recovery)
    let mut mu_state = MuState::new();
    // Monotone mu strategy: start in Fixed mode and never switch to Free
    if !options.mu_strategy_adaptive {
        mu_state.mode = MuMode::Fixed;
    }

    // Wall-clock time limit
    let start_time = Instant::now();
    let deadline = if options.max_wall_time > 0.0 {
        Some(start_time + Duration::from_secs_f64(options.max_wall_time))
    } else {
        None
    };

    // Phase timing instrumentation
    let mut timings = PhaseTimings::new();
    let ipm_start = Instant::now();

    // Watchdog mechanism state
    let mut watchdog = Watchdog::new();

    // Constraint violation history + ever_feasible / stall_count flags
    // driving infeasibility detection.
    let mut feas = FeasibilityTracker::new(100);

    // Tiny-step latch (Ipopt: `tiny_step_last_iteration_`, set at end of
    // an iter whose Δx, Δs are tiny and Δy < tiny_step_y_tol; consumed
    // by the next iter's detection to fire `tiny_step_flag`).
    let mut tiny_step_last_iter: bool = false;

    // STOP_AT_TINY_STEP exit flag: set by `update_barrier_parameter` when
    // `tiny_step && new_μ == μ` (no-op mu update — Ipopt's
    // `IpMonotoneMuUpdate.cpp:158-160`, `IpAdaptiveMuUpdate.cpp:329,377`),
    // consumed at the top of the *next* iteration after `check_convergence`
    // so KKT-clean tiny-step iterates exit Optimal first.
    let mut pending_tiny_step_exit: bool = false;

    // Line-search backtrack count for the previous iteration (printed in table).
    let mut ls_steps: usize = 0;
    // Hessian regularization delta from previous iteration (for intermediate callback).
    let prev_ic_delta_w: f64 = 0.0;

    // DEV-7: removed `consecutive_unbounded` counter; Ipopt-aligned
    // divergence is checked via `ConvergenceStatus::Diverging` driven by
    // `info.x_max_abs > options.diverging_iterates_tol` in convergence.rs.

    // Initial evaluation with NaN/Inf recovery by bound-push perturbation.
    if let Err(result) = initial_evaluate_with_recovery(
        &mut state, problem, &mut lbfgs_state, linear_constraints.as_deref(), lbfgs_mode, n, options,
    ) {
        return result;
    }

    initialize_slack_iterate(&mut state, m, options);
    initialize_constraint_slack_multipliers(&mut state, m, options);

    // Set filter parameters based on initial constraint violation.
    // A8.19: use slack-coupled `||c||_1 + ||d − s||_1` so theta_min /
    // theta_max are on the same scale as the trial-point theta computed
    // during the line search (which now uses `theta_for_split_d_s`).
    let theta_init = theta_for_split_d_s(&state, &state.c_x, &state.d_x, &state.s);
    filter.set_theta_min_from_initial(theta_init);

    // B10: Ipopt-style problem statistics block (print_level >= 4
    // matches Ipopt's `print_level=4` default for the statistics block;
    // the iteration table is at level 3).
    if options.print_level >= 4 {
        print_problem_header(&state);
    }

    // Print iteration table header (shown at print_level >= 3, reprinted every 25 rows)
    let mut log_line_count: usize = 0;
    if options.print_level >= 3 {
        rip_log!(
            "{:>4}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}  {:>3}",
            "iter", "objective", "inf_pr", "inf_du", "compl", "mu", "alpha_pr", "alpha_du", "ls"
        );
    }

    if options.print_level >= 5 {
        rip_log!("ripopt: Starting main loop (n={}, m={})", n, m);
    }

    // A8.7: hoist `aug_solver` above the IPM loop so its symbolic
    // factorization / CSC pattern caches persist across iterations.
    // The augmented-KKT sparsity pattern is fixed by the Hessian and
    // Jacobian patterns (Σ values + δ perturbations only change the
    // numeric diagonal, not the structure), so feral's first-call
    // symbolic analysis only needs to run once per solve. Without
    // hoisting, every iter constructed a fresh `FeralLdl` and re-ran
    // METIS-style ordering — measured at ~1.5s/iter on Mittelmann
    // ex8_2_3 (90% of total wall time before this fix).
    let mut aug_solver: Box<dyn LinearSolver> = new_fallback_solver(use_sparse);

    // Main IPM loop
    for iteration in 0..options.max_iter {
        state.iter = iteration;

        // T0.14 (Ipopt 3.14 alignment): clear the once-per-outer-iter
        // pretend-singular flag at the top of each iteration so the
        // PD perturbation handler allows the trick exactly once per
        // iter (IpPDPerturbationHandler.cpp).
        inertia_params.reset_pretend_singular_for_new_iter();

        // T0.10: notify the problem of the current barrier μ. Used by
        // RestorationNlp so its η weight tracks the inner IPM's μ
        // instead of staying frozen at restoration entry. Default no-op
        // for normal NLPs (IpRestoIpoptNLP.cpp:759).
        problem.notify_mu(state.mu);

        // Ipopt `IpRestoFilterConvCheck::TestOrigProgress`: when the
        // problem is a restoration NLP and the parent's max-bound-
        // violation has already dropped below `kappa_resto · θ_entry`,
        // exit the inner solve immediately so the parent can resume.
        // No-op for non-resto problems (default trait impl returns
        // false). Skipped at iteration 0 to ensure we always evaluate
        // the trial step at least once.
        if iteration > 0 && problem.resto_early_exit(&state.x) {
            if options.print_level >= 5 {
                rip_log!(
                    "ripopt: resto early-exit at iter {} (parent θ target reached)",
                    iteration
                );
            }
            state.iter = iteration;
            return make_result(&state, SolveStatus::Optimal);
        }

        if let Some(result) = check_time_limits(&state, iteration, start_time, options) {
            return result;
        }

        let OptimalityMeasures {
            primal_inf,
            primal_inf_max,
            primal_inf_internal_max,
            dual_inf,
            dual_inf_unscaled,
            compl_inf,
        } = compute_optimality_measures(&state);

        log_iteration_row(
            iteration,
            &state,
            primal_inf_max,
            dual_inf,
            compl_inf,
            ls_steps,
            &mut log_line_count,
            options,
            &inertia_params,
        );

        if iteration == 0 && options.print_level >= 5 {
            let (gf_inf_idx, gf_inf) = state
                .grad_f
                .iter()
                .enumerate()
                .fold((0usize, 0.0f64), |(ai, av), (i, &v)| {
                    if v.abs() > av { (i, v.abs()) } else { (ai, av) }
                });
            let mut jty = vec![0.0; state.n];
            // Phase 5d: split-form J^T·y diagnostic.
            for (idx, (&kc, &col)) in state.jac_c_rows.iter().zip(state.jac_c_cols.iter()).enumerate() {
                jty[col] += state.jac_c_vals[idx] * state.y_c[kc];
            }
            for (idx, (&kd, &col)) in state.jac_d_rows.iter().zip(state.jac_d_cols.iter()).enumerate() {
                jty[col] += state.jac_d_vals[idx] * state.y_d[kd];
            }
            let jty_inf = jty.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            // Phase 6d.3: bound-side scalar diagnostics consume the
            // compressed mirror; zero-padded entries contribute nothing
            // to abs/max/sum.
            let zl_inf = state.z_l_compressed.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let zu_inf = state.z_u_compressed.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let y_inf = state
                .y_c
                .iter()
                .chain(state.y_d.iter())
                .fold(0.0f64, |a, &b| a.max(b.abs()));
            let y_sum: f64 = state
                .y_c
                .iter()
                .chain(state.y_d.iter())
                .map(|v| v.abs())
                .sum();
            let zl_sum: f64 = state.z_l_compressed.iter().map(|v| v.abs()).sum();
            let zu_sum: f64 = state.z_u_compressed.iter().map(|v| v.abs()).sum();
            let mut grad_lag = state.grad_f.clone();
            for i in 0..state.n {
                grad_lag[i] += jty[i];
            }
            for k in 0..state.bound_layout.n_x_l {
                let i = state.bound_layout.x_l_to_full[k];
                grad_lag[i] -= state.z_l_compressed[k];
            }
            for k in 0..state.bound_layout.n_x_u {
                let i = state.bound_layout.x_u_to_full[k];
                grad_lag[i] += state.z_u_compressed[k];
            }
            let (gl_idx, gl_inf) = grad_lag.iter().enumerate().fold(
                (0usize, 0.0f64),
                |(ai, av), (i, &v)| if v.abs() > av { (i, v.abs()) } else { (ai, av) },
            );
            let x_l_fin = state.x_l_at(gl_idx).is_finite();
            let x_u_fin = state.x_u_at(gl_idx).is_finite();
            let x_inf = state.x.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            rip_log!(
                "ripopt: iter0-probe: |grad_f|_inf={:.3e}@var{}, |J^T y|_inf={:.3e}, |y|_inf={:.3e}, |z_L|_inf={:.3e}, |z_U|_inf={:.3e}, sum|y|={:.3e}, sum|z_L|={:.3e}, sum|z_U|={:.3e}, |x|_inf={:.3e}, n={} m={}",
                gf_inf, gf_inf_idx, jty_inf, y_inf, zl_inf, zu_inf, y_sum, zl_sum, zu_sum, x_inf, state.n, state.m
            );
            // Phase 6d.3: gl_idx may or may not have a compressed slot.
            let zl_at_gl = state.bound_layout.full_to_x_l[gl_idx]
                .map(|k| state.z_l_compressed[k])
                .unwrap_or(0.0);
            let zu_at_gl = state.bound_layout.full_to_x_u[gl_idx]
                .map(|k| state.z_u_compressed[k])
                .unwrap_or(0.0);
            rip_log!(
                "ripopt: iter0-probe: |grad_lag|_inf={:.3e}@var{} (grad_f={:.3e}, J^T y={:.3e}, z_L={:.3e}, z_U={:.3e}, x_l_fin={}, x_u_fin={}, obj_scaling={:.3e})",
                gl_inf, gl_idx, state.grad_f[gl_idx], jty[gl_idx], zl_at_gl, zu_at_gl, x_l_fin, x_u_fin, state.obj_scaling
            );

            // Per-component multiplier dump for diffing against Ipopt's
            // file_print_level=8 output (curr_y_c / curr_y_d / curr_z_L /
            // curr_z_U). Gated on RIPOPT_ITER0_DUMP=<path>; format mirrors
            // Ipopt's annotated dump so a textual diff lines up by
            // {_scon[N]} / {_svar[N]} (1-indexed combined index).
            if let Ok(path) = std::env::var("RIPOPT_ITER0_DUMP") {
                use std::io::Write;
                if let Ok(mut f) = std::fs::File::create(&path) {
                    let _ = writeln!(f, "DenseVector \"curr_y_c\" with {} elements:", state.y_c.len());
                    for (k, &v) in state.y_c.iter().enumerate() {
                        let combined = state.layout.c_to_combined[k];
                        let _ = writeln!(f, "curr_y_c[{:5}]{{_scon[{}]}}={:24.16e}", k + 1, combined + 1, v);
                    }
                    let _ = writeln!(f, "DenseVector \"curr_y_d\" with {} elements:", state.y_d.len());
                    for (k, &v) in state.y_d.iter().enumerate() {
                        let combined = state.layout.d_to_combined[k];
                        let _ = writeln!(f, "curr_y_d[{:5}]{{_scon[{}]}}={:24.16e}", k + 1, combined + 1, v);
                    }
                    let _ = writeln!(f, "DenseVector \"curr_z_L\" with {} elements:", state.z_l_compressed.len());
                    for (k, &v) in state.z_l_compressed.iter().enumerate() {
                        let full = state.bound_layout.x_l_to_full[k];
                        let _ = writeln!(f, "curr_z_L[{:5}]{{_svar[{}]}}={:24.16e}", k + 1, full + 1, v);
                    }
                    let _ = writeln!(f, "DenseVector \"curr_z_U\" with {} elements:", state.z_u_compressed.len());
                    for (k, &v) in state.z_u_compressed.iter().enumerate() {
                        let full = state.bound_layout.x_u_to_full[k];
                        let _ = writeln!(f, "curr_z_U[{:5}]{{_svar[{}]}}={:24.16e}", k + 1, full + 1, v);
                    }
                    let _ = writeln!(f, "DenseVector \"curr_v_L\" with {} elements:", state.v_l_compressed.len());
                    for (k, &v) in state.v_l_compressed.iter().enumerate() {
                        let _ = writeln!(f, "curr_v_L[{:5}]={:24.16e}", k + 1, v);
                    }
                    let _ = writeln!(f, "DenseVector \"curr_v_U\" with {} elements:", state.v_u_compressed.len());
                    for (k, &v) in state.v_u_compressed.iter().enumerate() {
                        let _ = writeln!(f, "curr_v_U[{:5}]={:24.16e}", k + 1, v);
                    }
                    // Per-d-row Jacobian L_inf and nnz, for diagnosing why
                    // some y_d entries land at 0 in feral but ~0.8 in MA27.
                    let mut jd_row_max = vec![0.0_f64; state.layout.n_d];
                    let mut jd_row_nnz = vec![0usize; state.layout.n_d];
                    for (idx, &kd) in state.jac_d_rows.iter().enumerate() {
                        let v = state.jac_d_vals[idx].abs();
                        if v > jd_row_max[kd] { jd_row_max[kd] = v; }
                        if state.jac_d_vals[idx] != 0.0 { jd_row_nnz[kd] += 1; }
                    }
                    let _ = writeln!(f, "JacD row stats (kd, _scon, row_max, row_nnz):");
                    for kd in 0..state.layout.n_d {
                        let combined = state.layout.d_to_combined[kd];
                        let _ = writeln!(f, "jd_row[{:5}]{{_scon[{}]}} max={:.6e} nnz={}", kd + 1, combined + 1, jd_row_max[kd], jd_row_nnz[kd]);
                    }
                    rip_log!("ripopt: iter0-probe: dumped multipliers to {}", path);
                }
            }
        }

        emit_trace_row_if_enabled(
            &state,
            iteration,
            primal_inf,
            dual_inf,
            compl_inf,
            ls_steps,
            &inertia_params,
            last_mehrotra_sigma,
            &mut trace_meta,
        );

        if let Some(result) = populate_snapshot_and_invoke_callback(
            &state,
            iteration,
            primal_inf,
            dual_inf,
            prev_ic_delta_w,
            ls_steps,
            options,
        ) {
            return result;
        }

        // Compute multiplier scaling (also used by the consecutive-acceptable
        // tracker further below, so kept here rather than inside the helper).
        // Counts are finite-bound counts to match Ipopt
        // `IpIpoptCalculatedQuantities.cpp:3677-3699`.
        let multiplier_sum = compute_multiplier_sum(&state);
        let multiplier_count = compute_multiplier_count(&state);
        let bound_multiplier_sum = compute_bound_multiplier_sum(&state);
        let bound_multiplier_count = compute_bound_multiplier_count(&state);

        if let Some(result) = check_convergence_and_handle_promotions(
            &mut state,
            options,
            primal_inf_max,
            primal_inf_internal_max,
            dual_inf,
            dual_inf_unscaled,
            compl_inf,
            multiplier_sum,
            multiplier_count,
            bound_multiplier_sum,
            bound_multiplier_count,
            &timings,
            iteration,
            ipm_start,
        ) {
            return result;
        }

        // Tiny-step termination — consumed AFTER `check_convergence_and_handle_promotions`
        // so a KKT-clean tiny-step iterate exits Optimal first. The flag
        // itself was set by the previous iteration's `update_barrier_parameter`
        // when `tiny_step && new_μ == μ` (Ipopt's `IpMonotoneMuUpdate.cpp:158-160`,
        // `IpAdaptiveMuUpdate.cpp:329,377`). Mirrors `IpIpoptAlg.cpp:347-466`
        // ordering: AcceptTrialPoint → CheckConvergence (end of iter k) →
        // UpdateBarrierParameter throws (top of iter k+1).
        if pending_tiny_step_exit {
            log::debug!(
                "STOP_AT_TINY_STEP: tiny_step latched and mu update found nothing to change at mu={:.2e}",
                state.mu
            );
            return make_result(&state, SolveStatus::StopAtTinyStep);
        }

        let _s_d_for_acc = track_consecutive_acceptable(
            &mut state,
            options,
            primal_inf,
            dual_inf,
            dual_inf_unscaled,
            compl_inf,
            multiplier_sum,
            bound_multiplier_sum,
        );

        store_acceptable_iterate(
            &mut state,
            &filter,
            iteration,
            primal_inf,
            primal_inf_internal_max,
            dual_inf,
            dual_inf_unscaled,
            compl_inf,
            multiplier_sum,
            multiplier_count,
            bound_multiplier_sum,
            bound_multiplier_count,
        );

        // Record current objective so the next iteration's
        // acceptable_obj_change_tol gate has a previous f to diff against.
        state.last_obj_for_acceptable = Some(state.obj);

        if let Some(result) = track_feasibility_and_detect_infeasibility(
            &state,
            options,
            iteration,
            primal_inf,
            &mut feas,
        ) {
            return result;
        }

        // A7: primary Newton step via the 4-block augmented system
        // [x; s; y_c; y_d]. A7.5 ports the Ipopt Probing μ oracle
        // (`IpProbingMuOracle::CalculateMu`) and gates it on
        // `options.mehrotra_pc`. When the oracle runs, it returns the
        // chosen μ alongside the step; we install the new μ into
        // `state.mu` so downstream line-search/convergence sees it.
        // A7.7: keep `aug_solver` and the returned `aug_kkt` alive across
        // the line search so SOC can reuse the factorization.
        let t_dir = Instant::now();
        // A7.8: pick the linear solver per the (n+m, sparse_threshold)
        // sizing cutoff already used by the rest of the IPM. Aug matrix is
        // assembled in matching layout (sparse vs dense) below.
        // A8.7: `aug_solver` is hoisted above the loop so its symbolic
        // cache persists across iterations.
        let probing = options.mehrotra_pc;
        let (step, _dc, mu_new_opt, aug_kkt, _iter_dw, _iter_dc) = if probing {
            let avg_compl = compute_avg_complementarity(&state);
            let mu_max = mu_state.mu_max_cap(options, avg_compl);
            // Phase 6d.5: materialize compressed z to full-`n` for kkt_aug
            // consumers (kkt_aug.rs still indexes by full var idx).
            let z_l_full = state.z_l_combined();
            let z_u_full = state.z_u_combined();
            let x_l_full = state.x_l_combined();
            let x_u_full = state.x_u_combined();
            // Phase 8c.5: same for compressed v_l/v_u → length n_d.
            let v_l_full = state.d_bound_layout.expand_l(&state.v_l_compressed, 0.0);
            let v_u_full = state.d_bound_layout.expand_u(&state.v_u_compressed, 0.0);
            // Phase 9c.5: same for compressed d_l/d_u → length n_d
            // with ±∞ on unbounded sides.
            let d_l_full = state.d_l();
            let d_u_full = state.d_u();
            match crate::kkt_aug::aug_step_from_state_mehrotra(
                n, &state.grad_f,
                &state.hess_rows, &state.hess_cols, &state.hess_vals,
                &state.jac_c_rows, &state.jac_c_cols, &state.jac_c_vals,
                &state.jac_d_rows, &state.jac_d_cols, &state.jac_d_vals,
                &state.x, &x_l_full, &x_u_full, &z_l_full, &z_u_full,
                &state.s, &state.c_x, &state.d_x, &d_l_full, &d_u_full,
                &state.y_c, &state.y_d, &v_l_full, &v_u_full,
                state.mu, options.kappa_d,
                crate::kkt_aug::PROBING_SIGMA_MAX_DEFAULT,
                options.mu_min, mu_max,
                use_sparse,
                aug_solver.as_mut(),
                &mut inertia_params,
            ) {
                Ok((step, mu_new, dw, dc, aug)) => (step, dc, Some(mu_new), aug, dw, dc),
                Err(_e) => {
                    timings.direction_solve += t_dir.elapsed();
                    return make_result(&state, SolveStatus::NumericalError);
                }
            }
        } else {
            // Phase 6d.5: materialize compressed z to full-`n` for kkt_aug.
            let z_l_full = state.z_l_combined();
            let z_u_full = state.z_u_combined();
            let x_l_full = state.x_l_combined();
            let x_u_full = state.x_u_combined();
            // Phase 8c.5: same for compressed v_l/v_u → length n_d.
            let v_l_full = state.d_bound_layout.expand_l(&state.v_l_compressed, 0.0);
            let v_u_full = state.d_bound_layout.expand_u(&state.v_u_compressed, 0.0);
            // Phase 9c.5: same for compressed d_l/d_u → length n_d.
            let d_l_full = state.d_l();
            let d_u_full = state.d_u();

            // RIPOPT_RHS_PROBE=ITER,VAR: dump component breakdown of
            // the augmented-system x-row at `var` AT iter `ITER`,
            // before any linear-solver scaling. Mirrors the formulas
            // in src/kkt_aug.rs build_outer_rhs (lines 296-344) and
            // fold_aug_rhs (lines 435-468). Used to localize whether
            // the catastrophic dx[var] at iter 110 originates in the
            // RHS itself (assembly) or downstream in the matrix solve.
            if let Ok(spec) = std::env::var("RIPOPT_RHS_PROBE") {
                let mut parts = spec.split(',');
                let want_iter: usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
                let want_var:  usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                if state.iter == want_iter && want_var < n {
                    let i = want_var;
                    let xi = state.x[i];
                    let xli = x_l_full[i];
                    let xui = x_u_full[i];
                    let l_fin = xli.is_finite();
                    let u_fin = xui.is_finite();
                    let s_l = if l_fin { (xi - xli).max(1e-20) } else { f64::INFINITY };
                    let s_u = if u_fin { (xui - xi).max(1e-20) } else { f64::INFINITY };
                    let zli = if l_fin { z_l_full[i] } else { 0.0 };
                    let zui = if u_fin { z_u_full[i] } else { 0.0 };
                    // J^T y at column i.
                    let mut jty_i = 0.0_f64;
                    for k in 0..state.jac_c_rows.len() {
                        if state.jac_c_cols[k] == i {
                            jty_i += state.jac_c_vals[k] * state.y_c[state.jac_c_rows[k]];
                        }
                    }
                    for k in 0..state.jac_d_rows.len() {
                        if state.jac_d_cols[k] == i {
                            jty_i += state.jac_d_vals[k] * state.y_d[state.jac_d_rows[k]];
                        }
                    }
                    let kappa_d = options.kappa_d;
                    let kd_term = if kappa_d > 0.0 && (l_fin ^ u_fin) {
                        if l_fin { kappa_d * state.mu } else { -kappa_d * state.mu }
                    } else { 0.0 };
                    // build_outer_rhs convention: rhs_x = grad_f + J^T y - z_L + z_U + κ_d
                    let mut rhs_x_i = state.grad_f[i] + jty_i + kd_term;
                    if l_fin { rhs_x_i -= zli; }
                    if u_fin { rhs_x_i += zui; }
                    let rhs_z_l_i = if l_fin { zli * s_l - state.mu } else { 0.0 };
                    let rhs_z_u_i = if u_fin { zui * s_u - state.mu } else { 0.0 };
                    // fold_aug_rhs convention: aug[i] = -rhs_x - rhs_z_L/s_l + rhs_z_U/s_u.
                    let mut aug_i = -rhs_x_i;
                    if l_fin { aug_i -= rhs_z_l_i / s_l; }
                    if u_fin { aug_i += rhs_z_u_i / s_u; }
                    let mut sigma_x_i = 0.0_f64;
                    if l_fin { sigma_x_i += zli / s_l; }
                    if u_fin { sigma_x_i += zui / s_u; }
                    let mu_over_sl = if l_fin { state.mu / s_l } else { 0.0 };
                    let mu_over_su = if u_fin { state.mu / s_u } else { 0.0 };
                    eprintln!(
                        "[rhs-probe] iter={} var={} mu={:.3e}\n  \
                         x={:+.6e} x_l={:+.3e} x_u={:+.3e} s_l={:.3e} s_u={:.3e}\n  \
                         grad_f={:+.6e} J^T y={:+.6e} z_L={:.6e} z_U={:.6e} kd={:+.3e}\n  \
                         rhs_x = grad_f + Jty - z_L + z_U + kd = {:+.6e}\n  \
                         rhs_z_L = z_L*s_l - mu = {:+.6e}   rhs_z_U = z_U*s_u - mu = {:+.6e}\n  \
                         mu/s_L={:.3e} mu/s_U={:.3e}\n  \
                         aug_rhs[var] = -rhs_x - rhs_z_L/s_L + rhs_z_U/s_U = {:+.6e}\n  \
                         Sigma_x[var] = z_L/s_l + z_U/s_u = {:.6e}",
                        state.iter, i, state.mu,
                        xi, xli, xui, s_l, s_u,
                        state.grad_f[i], jty_i, zli, zui, kd_term,
                        rhs_x_i, rhs_z_l_i, rhs_z_u_i,
                        mu_over_sl, mu_over_su,
                        aug_i, sigma_x_i,
                    );
                }
            }

            match crate::kkt_aug::aug_step_from_state(
                n, &state.grad_f,
                &state.hess_rows, &state.hess_cols, &state.hess_vals,
                &state.jac_c_rows, &state.jac_c_cols, &state.jac_c_vals,
                &state.jac_d_rows, &state.jac_d_cols, &state.jac_d_vals,
                &state.x, &x_l_full, &x_u_full, &z_l_full, &z_u_full,
                &state.s, &state.c_x, &state.d_x, &d_l_full, &d_u_full,
                &state.y_c, &state.y_d, &v_l_full, &v_u_full,
                state.mu, options.kappa_d,
                use_sparse,
                aug_solver.as_mut(),
                &mut inertia_params,
            ) {
                Ok((step, dw, dc, aug)) => (step, dc, None, aug, dw, dc),
                Err(_e) => {
                    timings.direction_solve += t_dir.elapsed();
                    return make_result(&state, SolveStatus::NumericalError);
                }
            }
        };
        timings.direction_solve += t_dir.elapsed();
        if let Some(mu_new) = mu_new_opt {
            state.mu = mu_new;
        }
        install_step_directions(
            &mut state, step.dx, step.dy_c, step.dy_d, step.ds,
            step.dz_l, step.dz_u, step.dv_l, step.dv_u,
        );

        // A8.21: iter-0 step-direction probe. Dumps ||·||_inf and the
        // five largest-magnitude (signed value, index) pairs for each of
        // dx, ds, dy, dz_l, dz_u so we can compare element-by-element
        // against Ipopt's `print_level=12` iter-0 trace.
        if iteration == 0 && options.print_level >= 6 {
            fn top5_signed(v: &[f64]) -> Vec<(usize, f64)> {
                let mut idx: Vec<usize> = (0..v.len()).collect();
                idx.sort_by(|&a, &b| v[b].abs().partial_cmp(&v[a].abs()).unwrap());
                idx.iter().take(5).map(|&i| (i, v[i])).collect()
            }
            let dx_inf = state.dx.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let ds_inf = state.ds.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let ds_combined_iter0 = state.ds_combined();
            let dy_combined_iter0 = state.dy_combined();
            let dy_inf = dy_combined_iter0.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let dzl_inf = state.dz_l_compressed.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let dzu_inf = state.dz_u_compressed.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            rip_log!(
                "ripopt: iter0-step: ||dx||_inf={:.16e} ||ds||_inf={:.16e} ||dy||_inf={:.16e} ||dz_l||_inf={:.16e} ||dz_u||_inf={:.16e}",
                dx_inf, ds_inf, dy_inf, dzl_inf, dzu_inf
            );
            // A8.21: emit the perturbation δ_w/δ_c that landed for iter 0.
            // Ipopt's print_level=12 trace shows `delta_x=0.000000e+00
            // delta_s=0.000000e+00` at iter 0; if ripopt non-zero here,
            // that's a candidate root-cause for the dx gap.
            rip_log!(
                "ripopt: iter0-pert: delta_w_used={:.6e} delta_c_used={:.6e} delta_w_last={:.6e} delta_c_last={:.6e}",
                _iter_dw, _iter_dc,
                inertia_params.delta_w_last, inertia_params.delta_c_last
            );
            // Phase 6d.3: materialize dz_l/dz_u to full-n for the
            // top5 diagnostic so the printed var indices match
            // absolute variable ids.
            let dz_l_iter0 = state.dz_l_combined();
            let dz_u_iter0 = state.dz_u_combined();
            for (name, vec) in [
                ("dx", &state.dx[..]),
                ("ds", &ds_combined_iter0[..]),
                ("dy", &dy_combined_iter0[..]),
                ("dz_l", &dz_l_iter0[..]),
                ("dz_u", &dz_u_iter0[..]),
            ] {
                let top = top5_signed(vec);
                let mut s = format!("ripopt: iter0-step: top5 {}:", name);
                for (i, v) in top {
                    s.push_str(&format!(" [{}]={:.16e}", i, v));
                }
                rip_log!("{}", s);
            }
            // A8.21 deep-dive: dump the assembled-system inputs that feed
            // x-block row 1753 of the augmented system. This separates a
            // tape-evaluation discrepancy (grad_f, J, H) from a Σ-assembly
            // bug from a downstream issue.
            let probe_var: usize = 1753;
            if probe_var < state.n {
                // grad_f at probe_var
                let grad_f_pv = state.grad_f[probe_var];
                // Σ_x at probe_var (matches kkt_aug::aug_step_from_state formula)
                let xi = state.x[probe_var];
                let xli = state.x_l_at(probe_var);
                let xui = state.x_u_at(probe_var);
                let mut sigma_x_pv = 0.0_f64;
                // Phase 6d.3: index compressed mirrors via BoundLayout.
                if let Some(k) = state.bound_layout.full_to_x_l[probe_var] {
                    sigma_x_pv += state.z_l_compressed[k] / (xi - xli).max(1e-20);
                }
                if let Some(k) = state.bound_layout.full_to_x_u[probe_var] {
                    sigma_x_pv += state.z_u_compressed[k] / (xui - xi).max(1e-20);
                }
                // Hessian row probe_var (lower-triangle entries (probe_var, *)
                // and upper via (*, probe_var)).
                let mut h_row: Vec<(usize, f64)> = Vec::new();
                for k in 0..state.hess_rows.len() {
                    let r = state.hess_rows[k];
                    let c = state.hess_cols[k];
                    let v = state.hess_vals[k];
                    if v == 0.0 { continue; }
                    if r == probe_var { h_row.push((c, v)); }
                    else if c == probe_var { h_row.push((r, v)); }
                }
                // Jacobian column probe_var: rows of constraints that have a
                // structural entry at column probe_var. Rebuild combined
                // (rows, cols, vals) and m-form g once for the probe dump.
                let (jrows_m, jcols_m, jvals_m) = rebuild_combined_jac(&state);
                let g_m = state.g_combined();
                let mut j_col: Vec<(usize, f64)> = Vec::new();
                for k in 0..jrows_m.len() {
                    if jcols_m[k] == probe_var {
                        j_col.push((jrows_m[k], jvals_m[k]));
                    }
                }
                rip_log!(
                    "ripopt: iter0-row[{}]: grad_f={:.16e} sigma_x={:.16e} H_row_nnz={} J_col_nnz={}",
                    probe_var, grad_f_pv, sigma_x_pv, h_row.len(), j_col.len()
                );
                {
                    let mut s = format!("ripopt: iter0-row[{}]: H[{},*] (col,val):", probe_var, probe_var);
                    for (c, v) in h_row.iter().take(20) {
                        s.push_str(&format!(" ({},{:.6e})", c, v));
                    }
                    if h_row.len() > 20 { s.push_str(&format!(" ... [{}]", h_row.len())); }
                    rip_log!("{}", s);
                }
                {
                    let mut s = format!("ripopt: iter0-row[{}]: J[*,{}] (row,val):", probe_var, probe_var);
                    for (r, v) in j_col.iter().take(20) {
                        s.push_str(&format!(" ({},{:.6e})", r, v));
                    }
                    if j_col.len() > 20 { s.push_str(&format!(" ... [{}]", j_col.len())); }
                    rip_log!("{}", s);
                }
                // For each constraint row that touches probe_var, dump:
                //  g(x_0)[r], g_l[r], g_u[r], slack s[r], dy[r], plus the
                //  full row of J at row r (other variables coupled).
                for (r, _coeff) in j_col.iter().take(8) {
                    let r = *r;
                    let g_r = g_m[r];
                    let gl_r = state.g_l_at(r);
                    let gu_r = state.g_u_at(r);
                    let dy_r = state.dy_at(r);
                    let s_r = if r < state.m { state.s_at(r) } else { f64::NAN };
                    // collect J row r (cols, vals)
                    let mut j_row: Vec<(usize, f64)> = Vec::new();
                    for k in 0..jrows_m.len() {
                        if jrows_m[k] == r {
                            j_row.push((jcols_m[k], jvals_m[k]));
                        }
                    }
                    // Look up the per-row scaling via the layout (Phase 3
                    // split storage; user-facing index r is combined).
                    let scale_r = if let Some(k) = state.layout.eq_pos[r] {
                        state.c_scaling.get(k).copied().unwrap_or(1.0)
                    } else if let Some(k) = state.layout.ineq_pos[r] {
                        state.d_scaling.get(k).copied().unwrap_or(1.0)
                    } else {
                        1.0
                    };
                    rip_log!(
                        "ripopt: iter0-conrow[{}]: g={:.16e} g_l={:.16e} g_u={:.16e} s={:.16e} dy={:.16e} g_scale={:.16e} nnz={}",
                        r, g_r, gl_r, gu_r, s_r, dy_r, scale_r, j_row.len()
                    );
                    let mut s = format!("ripopt: iter0-conrow[{}]: J[{},*]:", r, r);
                    for (c, v) in j_row.iter().take(10) {
                        s.push_str(&format!(" ({},{:.6e})", c, v));
                    }
                    if j_row.len() > 10 { s.push_str(&format!(" ... [{}]", j_row.len())); }
                    rip_log!("{}", s);
                }
            }

            // A8.21: targeted dump of (x, x_l, z_l, dx, dz_l, x-x_l) at the
            // variable Ipopt reports as its delta_z_L max (absolute var
            // 1753, 0-indexed; AMPL name x1754). ripopt's max dz_l is at
            // 1753 too, so values at this index will reveal whether the
            // discrepancy comes from dx (linear solve precision) or from
            // the back-sub formula (x-x_l, z_l, μ inputs).
            for &probe_i in &[1753usize, 1801usize, 1871usize] {
                if probe_i < state.n {
                    let xi = state.x[probe_i];
                    let xli = state.x_l_at(probe_i);
                    let xui = state.x_u_at(probe_i);
                    // Phase 6d.3: index compressed mirrors via BoundLayout
                    // (unbounded sides report 0 — same as the legacy combined storage).
                    let zli = state.bound_layout.full_to_x_l[probe_i]
                        .map(|k| state.z_l_compressed[k]).unwrap_or(0.0);
                    let zui = state.bound_layout.full_to_x_u[probe_i]
                        .map(|k| state.z_u_compressed[k]).unwrap_or(0.0);
                    let dxi = state.dx[probe_i];
                    let dzli = state.bound_layout.full_to_x_l[probe_i]
                        .map(|k| state.dz_l_compressed[k]).unwrap_or(0.0);
                    let dzui = state.bound_layout.full_to_x_u[probe_i]
                        .map(|k| state.dz_u_compressed[k]).unwrap_or(0.0);
                    let slack_l = xi - xli;
                    let slack_u = xui - xi;
                    rip_log!(
                        "ripopt: iter0-probe[{}]: x={:.16e} x_l={:.16e} x_u={:.16e} (x-x_l)={:.16e} (x_u-x)={:.16e} z_l={:.16e} z_u={:.16e} dx={:.16e} dz_l={:.16e} dz_u={:.16e}",
                        probe_i, xi, xli, xui, slack_l, slack_u, zli, zui, dxi, dzli, dzui
                    );
                }
            }
        }

        // RIPOPT_IR_DUMP: emit the full iter-0 KKT-system snapshot as
        // structured JSON for cross-solver comparison via examples/arki_diff.
        // Schema: src/iter0_dump.rs::Iter0Dump. The OnceCell guards against
        // re-emission from restoration sub-IPM iter-0 (which would
        // overwrite the main-solve dump with the restoration NLP's larger
        // (n + 2m) state space).
        // Skip the dump when we're inside the restoration sub-IPM
        // (`configure_restoration_inner_options` flips this flag). The
        // OnceLock alone is insufficient because the restoration's iter-0
        // typically reaches this line before the outer main IPM does, and
        // we want the OUTER state captured, not the (n + 2m) restoration
        // state space.
        static IR_DUMP_DONE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        if iteration == 0
            && !options.disable_nlp_restoration
            && IR_DUMP_DONE.get().is_none()
        {
            if let Ok(dump_path) = std::env::var("RIPOPT_IR_DUMP") {
                let _ = IR_DUMP_DONE.set(());
                use crate::iter0_dump::Iter0Dump;
                let n = state.n;
                let m = state.m;
                let n_d = state.layout.n_d;

                // Materialize x_l, x_u to full-n with `None` at unbounded sides.
                let mut x_l_full: Vec<Option<f64>> = vec![None; n];
                let mut x_u_full: Vec<Option<f64>> = vec![None; n];
                for i in 0..n {
                    if state.bound_layout.full_to_x_l[i].is_some() {
                        x_l_full[i] = Some(state.x_l_at(i));
                    }
                    if state.bound_layout.full_to_x_u[i].is_some() {
                        x_u_full[i] = Some(state.x_u_at(i));
                    }
                }
                // Materialize d_l, d_u to length n_d with `None` at unbounded sides.
                let mut d_l_full: Vec<Option<f64>> = vec![None; n_d];
                let mut d_u_full: Vec<Option<f64>> = vec![None; n_d];
                for k in 0..n_d {
                    let dl = state.d_l_at(k);
                    let du = state.d_u_at(k);
                    if dl.is_finite() { d_l_full[k] = Some(dl); }
                    if du.is_finite() { d_u_full[k] = Some(du); }
                }

                // Materialize z_l, z_u, dz_l, dz_u, v_l, v_u, dv_l, dv_u.
                let mut z_l_n = vec![0.0; n];
                let mut z_u_n = vec![0.0; n];
                for i in 0..n {
                    if let Some(k) = state.bound_layout.full_to_x_l[i] {
                        z_l_n[i] = state.z_l_compressed[k];
                    }
                    if let Some(k) = state.bound_layout.full_to_x_u[i] {
                        z_u_n[i] = state.z_u_compressed[k];
                    }
                }
                let dz_l_n = state.dz_l_combined();
                let dz_u_n = state.dz_u_combined();

                // v_l/v_u, dv_l/dv_u as n_d-length arrays (zero at
                // unbounded sides). v_l_combined()/v_u_combined() return
                // length m; we want length n_d to match the schema.
                let mut v_l_n = vec![0.0; n_d];
                let mut v_u_n = vec![0.0; n_d];
                let mut dv_l_n = vec![0.0; n_d];
                let mut dv_u_n = vec![0.0; n_d];
                for k in 0..n_d {
                    if let Some(kc) = state.d_bound_layout.full_to_d_l[k] {
                        v_l_n[k] = state.v_l_compressed[kc];
                        dv_l_n[k] = state.dv_l_compressed[kc];
                    }
                    if let Some(kc) = state.d_bound_layout.full_to_d_u[k] {
                        v_u_n[k] = state.v_u_compressed[kc];
                        dv_u_n[k] = state.dv_u_compressed[kc];
                    }
                }

                // Σ_x diagonal at iter 0 (full-n).
                let mut sigma_x = vec![0.0; n];
                for i in 0..n {
                    let xi = state.x[i];
                    if let Some(k) = state.bound_layout.full_to_x_l[i] {
                        let xli = state.x_l_at(i);
                        sigma_x[i] += state.z_l_compressed[k] / (xi - xli).max(1e-20);
                    }
                    if let Some(k) = state.bound_layout.full_to_x_u[i] {
                        let xui = state.x_u_at(i);
                        sigma_x[i] += state.z_u_compressed[k] / (xui - xi).max(1e-20);
                    }
                }
                // Σ_s diagonal at iter 0 (length n_d).
                let mut sigma_s = vec![0.0; n_d];
                for k in 0..n_d {
                    let sk = state.s[k];
                    if let Some(kc) = state.d_bound_layout.full_to_d_l[k] {
                        let dl = state.d_l_compressed[kc];
                        sigma_s[k] += state.v_l_compressed[kc] / (sk - dl).max(1e-20);
                    }
                    if let Some(kc) = state.d_bound_layout.full_to_d_u[k] {
                        let du = state.d_u_compressed[kc];
                        sigma_s[k] += state.v_u_compressed[kc] / (du - sk).max(1e-20);
                    }
                }

                // Combined Jacobian and constraint values.
                let (jrows, jcols, jvals) = rebuild_combined_jac(&state);
                let g_combined = state.g_combined();

                // Per-variable scaling: ripopt has no full per-var scaling;
                // emit all-1.0 (matches the schema contract).
                let x_scaling = vec![1.0; n];
                // Combined per-constraint scaling (length m): merge split
                // c_scaling (eq) and d_scaling (ineq) via the layout.
                let mut c_scaling_combined = vec![1.0; m];
                for r in 0..m {
                    if let Some(k) = state.layout.eq_pos[r] {
                        c_scaling_combined[r] = state.c_scaling.get(k).copied().unwrap_or(1.0);
                    } else if let Some(k) = state.layout.ineq_pos[r] {
                        c_scaling_combined[r] = state.d_scaling.get(k).copied().unwrap_or(1.0);
                    }
                }

                let dump = Iter0Dump {
                    solver: "ripopt".to_string(),
                    note: format!(
                        "ripopt iter-0 dump @ mu={:.6e}, n={}, m={}, n_d={}",
                        state.mu, n, m, n_d
                    ),
                    n, m, n_d,
                    x: state.x.clone(),
                    x_l: x_l_full,
                    x_u: x_u_full,
                    s: state.s.clone(),
                    d_l: d_l_full,
                    d_u: d_u_full,
                    y_c: state.y_c.clone(),
                    y_d: state.y_d.clone(),
                    y_layout: "split".to_string(),
                    z_l: z_l_n,
                    z_u: z_u_n,
                    v_l: v_l_n,
                    v_u: v_u_n,
                    grad_f: state.grad_f.clone(),
                    g: g_combined,
                    jac_rows: jrows,
                    jac_cols: jcols,
                    jac_vals: jvals,
                    hess_rows: state.hess_rows.clone(),
                    hess_cols: state.hess_cols.clone(),
                    hess_vals: state.hess_vals.clone(),
                    obj_scaling: state.obj_scaling,
                    x_scaling,
                    c_scaling: c_scaling_combined,
                    sigma_x,
                    sigma_s,
                    aug_rhs: Vec::new(),
                    delta_w_used: _iter_dw,
                    delta_c_used: _iter_dc,
                    dx: state.dx.clone(),
                    ds: state.ds.clone(),
                    dy: state.dy_combined(),
                    dz_l: dz_l_n,
                    dz_u: dz_u_n,
                    dv_l: dv_l_n,
                    dv_u: dv_u_n,
                    alpha_pr: 0.0,
                    alpha_du: 0.0,
                    mu: state.mu,
                };
                dump.write(&dump_path);
                rip_log!("ripopt: iter0-dump: wrote JSON to {}", dump_path);
            }
        }

        let (tau, alpha_primal_max, alpha_dual_max) = compute_alpha_max(
            &state, options, &mu_state, primal_inf, dual_inf, compl_inf,
        );
        trace_meta.alpha_primal_max = Some(alpha_primal_max);
        trace_meta.tau_used = Some(tau);

        // Ipopt-style tiny-step detection (T2.21, spec §4): set the
        // `tiny_step` flag when the raw search direction is at machine-
        // precision noise AND the dual step is also small. The actual
        // STOP_AT_TINY_STEP exit fires from `update_barrier_parameter`
        // when `tiny_step && new_μ == μ` and is consumed at the top of
        // the next iteration (after `check_convergence`).
        // See `IpBacktrackingLineSearch.cpp:1219-1278,407-424`,
        // `IpMonotoneMuUpdate.cpp:158-160`,
        // `IpAdaptiveMuUpdate.cpp:329,377`.
        detect_tiny_step(
            &mut state,
            options,
            &mut mu_state,
            &mut filter,
            &mut tiny_step_last_iter,
            primal_inf,
        );

        // Line search
        let t_ls = Instant::now();
        // A8.19: filter theta is slack-coupled (`||c||_1 + ||d − s||_1`)
        // matching Ipopt's `IpCq::curr_constraint_violation`. The
        // box-violation `primal_inf` is preserved separately as a
        // diagnostic and as the input to other consumers (compute_tau,
        // detect_tiny_step, almost_feasible_guard, feasibility history).
        let theta_current = theta_for_split_d_s(&state, &state.c_x, &state.d_x, &state.s);
        let phi_current = state.barrier_objective(options);
        let grad_phi_step = state.barrier_directional_derivative(options);

        let mut step_accepted;
        let min_alpha = filter.compute_alpha_min(theta_current, grad_phi_step);

        // RIPOPT_LS_DECISION probe: print pre-LS step direction
        // norms + α_max + α_min + grad_phi_step. Pair with the
        // existing RIPOPT_LS_PROBE prints inside run_line_search_loop
        // to localise iter-N divergence vs Ipopt. Gated by env var,
        // disabled at print_level=0.
        if std::env::var("RIPOPT_LS_DECISION").is_ok() {
            let dx_inf = linf_norm(&state.dx);
            let ds_inf = linf_norm(&state.ds);
            let dyc_inf = linf_norm(&state.dy_c);
            let dyd_inf = linf_norm(&state.dy_d);
            let dzl_inf = linf_norm(&state.dz_l_compressed);
            let dzu_inf = linf_norm(&state.dz_u_compressed);
            eprintln!(
                "[ls-decision] iter={} alpha_p_max={:.3e} alpha_d_max={:.3e} alpha_min={:.3e} \
                 ||dx||={:.3e} ||ds||={:.3e} ||dy_c||={:.3e} ||dy_d||={:.3e} ||dz_L||={:.3e} ||dz_U||={:.3e} \
                 theta={:.6e} phi={:.6e} grad_phi_step={:.6e}",
                iteration, alpha_primal_max, alpha_dual_max, min_alpha,
                dx_inf, ds_inf, dyc_inf, dyd_inf, dzl_inf, dzu_inf,
                theta_current, phi_current, grad_phi_step,
            );
        }

        // Snapshot x before the line search so we can roll back + halve α if the
        // post-step gradient / Jacobian / Hessian eval trips the NaN/Inf guard
        // (Ipopt's line search catches Eval_Error from these with α-backtracking,
        // not a hard abort — IpBacktrackingLineSearch.cpp:776-784, 1158, 1193).
        let x_pre_step = state.x.clone();

        match run_line_search_loop(
            &mut state,
            problem,
            options,
            &mut filter,
            &mut inertia_params,
            alpha_primal_max,
            theta_current,
            phi_current,
            grad_phi_step,
            min_alpha,
            watchdog.active,
            iteration,
            n,
            m,
            &mut trace_meta,
            &mut ls_steps,
            aug_solver.as_mut(),
            &aug_kkt,
        ) {
            LineSearchOutcome::StepAccepted => {
                step_accepted = true;
            }
            LineSearchOutcome::Rejected => {
                step_accepted = false;
            }
        }

        let mut accepted_by_soft_resto = false;
        if !step_accepted {
            step_accepted = attempt_soft_restoration(
                &mut state,
                problem,
                options,
                &mut filter,
                &mut mu_state,
                linear_constraints.as_deref(),
                lbfgs_mode,
                alpha_primal_max,
                alpha_dual_max,
                theta_current,
                phi_current,
            );
            accepted_by_soft_resto = step_accepted;
        }

        if !step_accepted {
            // Ipopt's `PrepareRestoPhaseStart` augments the filter with
            // the (theta_current, phi_current) margin entry BEFORE the
            // almost-feasible guard fires (IpBacktrackingLineSearch.cpp:566
            // vs :580). Augmenting here means the filter is correctly
            // updated whether the guard exits or the restoration cascade
            // runs, and avoids a second augment inside the cascade.
            filter.augment_for_restoration(theta_current, phi_current);

            if let Some(result) = check_almost_feasible_guard(
                &mut state,
                options,
                &mut filter,
                primal_inf,
                primal_inf_max,
            ) {
                return result;
            }
            match run_post_ls_restoration_cascade(
                &mut state,
                problem,
                options,
                &mut filter,
                &mut mu_state,
                &mut inertia_params,
                &mut lbfgs_state,
                lbfgs_mode,
                linear_constraints.as_deref(),
                iteration,
                n,
                m,
                start_time,
                deadline,
                &feas,
                theta_current,
            ) {
                RestorationCascadeDecision::Continue => continue,
                RestorationCascadeDecision::Return(result) => return result,
            }
        }

        // Step was accepted — reset consecutive restoration failure counter.
        // Only the *regular* line search resets the soft-restoration counter;
        // a step taken via the soft path keeps it so the 10-iteration cap
        // (`MAX_SOFT_RESTO_ITERS`) bites after a sustained run of soft accepts
        // (Ipopt's `soft_resto_counter_`, `IpBacktrackingLineSearch.cpp:442-444`).
        mu_state.consecutive_restoration_failures = 0;
        if !accepted_by_soft_resto {
            mu_state.consecutive_soft_restoration = 0;
        }

        match update_watchdog(
            &mut state,
            problem,
            options,
            iteration,
            alpha_primal_max,
            &mut filter,
            &mut lbfgs_state,
            &mut watchdog,
            linear_constraints.as_deref(),
            lbfgs_mode,
        ) {
            WatchdogDecision::Proceed => {}
            WatchdogDecision::Continue => continue,
        }

        timings.line_search += t_ls.elapsed();

        let _ = update_dual_variables(
            &mut state,
            &mu_state,
            alpha_dual_max,
            options,
            problem,
        );

        match reevaluate_after_step(
            &mut state,
            problem,
            options,
            &mut lbfgs_state,
            &mut filter,
            &mut timings,
            &x_pre_step,
            linear_constraints.as_deref(),
            lbfgs_mode,
            deadline,
        ) {
            PostStepEvalDecision::Proceed => {}
            PostStepEvalDecision::Continue => continue,
            PostStepEvalDecision::Return(result) => return result,
        }

        // Magic step (spec §5.3, T2.24). Ipopt's
        // `BacktrackingLineSearch::PerformMagicStep` mutates the
        // explicit slack `s` to drive the residual `d(x) - s` toward
        // zero along the slack coordinate while holding `x` fixed
        // (`IpBacktrackingLineSearch.cpp:1013-1111`). ripopt's
        // implicit-slack formulation has no `s` distinct from `x`, so
        // `apply_magic_step` is a no-op on this path (see its docs);
        // the call is retained for spec-compliance and as the hook
        // for any future explicit-slack solve path.
        let _ = apply_magic_step(&mut state, options);
        maybe_recalc_y_post_step(&mut state, options, n, m, lbfgs_mode);

        // --- Barrier parameter update (free/fixed mode) ---
        // Save mu before so we can detect "mu update found nothing to
        // change" (Ipopt `!mu_changed`, IpMonotoneMuUpdate.cpp:158-160).
        let mu_before_update = state.mu;
        update_barrier_parameter(
            &mut state,
            &mut mu_state,
            &mut filter,
            &mut last_mehrotra_sigma,
            options,
            use_sparse,
        );
        // Set the STOP_AT_TINY_STEP latch when the tiny-step flag is
        // active and the mu update could not advance — exactly the
        // Ipopt throw condition (`!mu_changed && tiny_step_flag`,
        // IpMonotoneMuUpdate.cpp:158-160; IpAdaptiveMuUpdate.cpp:330,377).
        // `mu_state.tiny_step` already encodes "current iter detection
        // AND previous iter latched", so no extra counter gate is needed.
        // Consumed at the top of the next iteration AFTER check_convergence,
        // so a KKT-clean iterate still exits Optimal.
        if mu_state.tiny_step && state.mu == mu_before_update {
            pending_tiny_step_exit = true;
        }

        track_post_step_acceptable(&mut state, options);

        // RIPOPT_FILTER_DUMP=lo,hi: dump full filter contents at end of
        // each iter in [lo, hi]. Diagnostic for arki0003 iter-110
        // divergence — compares ripopt's filter set to Ipopt's
        // print_level=12 output to localize missing augmentations.
        if let Ok(spec) = std::env::var("RIPOPT_FILTER_DUMP") {
            let mut parts = spec.split(',');
            let lo: usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let hi: usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
            if state.iter >= lo && state.iter <= hi {
                let entries = filter.entries();
                eprintln!(
                    "[filter-dump] iter={} n_entries={} resets={}",
                    state.iter, entries.len(), filter.n_filter_resets(),
                );
                for (idx, e) in entries.iter().enumerate() {
                    eprintln!(
                        "  filter[{}] theta={:.10e} phi={:.10e}",
                        idx, e.theta, e.phi,
                    );
                }
            }
        }

        // A8.6+ dual-divergence trace: env-gated per-iter snapshot of
        // the dual state. Set RIPOPT_TRACE_DUAL=1 to log ‖y‖_∞,
        // worst-y_i index, α_pr/α_du, μ, and μ-mode at the end of
        // every accepted iteration. Used to identify the iter where a
        // diverging trajectory first deviates from the Ipopt log.
        if std::env::var("RIPOPT_TRACE_DUAL").is_ok() {
            let mut y_inf = 0.0_f64;
            let mut y_idx = usize::MAX;
            for i in 0..state.m {
                let yi = state.y_at(i);
                if yi.abs() > y_inf {
                    y_inf = yi.abs();
                    y_idx = i;
                }
            }
            let mode_str = match mu_state.mode {
                MuMode::Free => "Free",
                MuMode::Fixed => "Fixed",
            };
            // Step magnitude ranges: dx/dy/dz_l/dz_u L∞.
            let dx_inf = linf_norm(&state.dx);
            let dy_inf = state
                .dy_c
                .iter()
                .chain(state.dy_d.iter())
                .fold(0.0f64, |a, &b| a.max(b.abs()));
            let dzl_inf = linf_norm(&state.dz_l_compressed);
            let dzu_inf = linf_norm(&state.dz_u_compressed);
            // Worst (z·s)/μ ratio: should be bounded by κ_Σ (default 1e10)
            // per Ipopt's reset_slack_multipliers / IpIpoptCalculatedQuantities.
            // Ratios >> 1 indicate the κ_Σ clamp is not enforcing.
            let mut worst_zs_ratio = 0.0_f64;
            let mut worst_zs_idx = usize::MAX;
            let mut worst_zs_side = "";
            // Phase 6c.2: walk compressed z mirrors; iteration set is
            // identical to the n-wide+is_finite scan.
            for k in 0..state.bound_layout.n_x_l {
                let i = state.bound_layout.x_l_to_full[k];
                let s = state.x[i] - state.x_l_at(i);
                if s > 0.0 {
                    let r = (state.z_l_compressed[k] * s).abs() / state.mu.max(1e-300);
                    if r > worst_zs_ratio { worst_zs_ratio = r; worst_zs_idx = i; worst_zs_side = "L"; }
                }
            }
            for k in 0..state.bound_layout.n_x_u {
                let i = state.bound_layout.x_u_to_full[k];
                let s = state.x_u_at(i) - state.x[i];
                if s > 0.0 {
                    let r = (state.z_u_compressed[k] * s).abs() / state.mu.max(1e-300);
                    if r > worst_zs_ratio { worst_zs_ratio = r; worst_zs_idx = i; worst_zs_side = "U"; }
                }
            }
            eprintln!(
                "[dual] it={:4} ‖y‖∞={:.3e}@{} α_p={:.3e} α_d={:.3e} μ={:.3e} mode={} resto={} ‖dx‖={:.2e} ‖dy‖={:.2e} ‖dz_l‖={:.2e} ‖dz_u‖={:.2e} max(zs)/μ={:.2e}@{}{}",
                iteration, y_inf, y_idx,
                state.alpha_primal, state.alpha_dual, state.mu, mode_str,
                state.diagnostics.restoration_count,
                dx_inf, dy_inf, dzl_inf, dzu_inf,
                worst_zs_ratio, worst_zs_idx, worst_zs_side,
            );
        }
    }

    finalize_after_max_iter(
        &state, options, &feas, &timings, ipm_start,
    )
}

/// After the main IPM loop terminates at `max_iter`, decide the final
/// `SolveStatus`:
///
///   1. Log MaxIter diagnostics at print_level ≥ 5.
///   2. Infeasibility check (only if never feasible): `||∇θ|| ≈ 0`
///      → `LocalInfeasibility`; theta stagnation history → `Infeasible`.
///   3. Print phase timing summary.
///   4. Strict Optimal at final iterate (catches MGH10LS-class zero-residual
///      stalls where the counter reset).
///   5. Relaxed Acceptable at final iterate (Ipopt default thresholds).
///   6. Fallback: `MaxIterations`.
/// At MaxIterations exit, before falling through to MaxIterations,
/// check whether the iterate is stuck at a stationary infeasible point
/// (||J^T·violation||_∞ < 1e-4·max(theta, 1)) or at a persistently
/// large theta (> 1% of historical minimum, theta > 1e4). Returns
/// LocalInfeasibility / Infeasible when applicable, None otherwise.
/// Caller must only invoke when `!ever_feasible` and `theta > tol`.
fn try_classify_max_iter_infeasibility(
    state: &SolverState,
    final_theta: f64,
    feas: &FeasibilityTracker,
) -> Option<SolveResult> {
    let grad_theta_norm = compute_grad_theta_norm(state);
    let stationarity_tol = 1e-4 * final_theta.max(1.0);
    if grad_theta_norm < stationarity_tol {
        return Some(make_result(state, SolveStatus::LocalInfeasibility));
    }

    if final_theta > 1e4 && feas.history.len() >= feas.history_len {
        let min_theta = slice_min(&feas.history);
        if final_theta > 0.01 * min_theta {
            return Some(make_result(state, SolveStatus::Infeasible));
        }
    }
    None
}

/// At print_level >= 5, log a one-line MaxIter summary with the
/// scaled/unscaled dual infeasibility, complementarity, mu, dual
/// scaling factor s_d (Ipopt formula), and the acceptable-iter
/// counter. Pure side-effect helper — caller branches separately.
fn print_max_iter_diagnostics(
    state: &SolverState,
    options: &SolverOptions,
) {
    if options.print_level < 5 {
        return;
    }
    let final_primal_inf = compute_primal_inf_max_at_state(state);
    let final_dual_inf = compute_dual_inf_at_state(state);
    let final_dual_inf_unscaled = compute_dual_inf_unscaled_at_state(state);
    let final_compl = compute_compl_err_at_state(state);
    let s_d = compute_s_d_at_state(state);
    rip_log!(
        "ripopt: MaxIter diag: pr={:.2e} du={:.2e}(t={:.2e}) du_u={:.2e}(t={:.0e}) co={:.2e}(t={:.2e}) mu={:.2e} sd={:.1} ac={}",
        final_primal_inf,
        final_dual_inf, options.tol * s_d,
        final_dual_inf_unscaled, options.dual_inf_tol,
        final_compl, 10.0 * options.compl_inf_tol,
        state.mu, s_d, state.consecutive_acceptable,
    );
}

fn finalize_after_max_iter(
    state: &SolverState,
    options: &SolverOptions,
    feas: &FeasibilityTracker,
    timings: &PhaseTimings,
    ipm_start: Instant,
) -> SolveResult {
    print_max_iter_diagnostics(state, options);

    // Infeasibility detection (only when never feasible).
    let final_theta = state.constraint_violation();
    if !feas.ever_feasible && final_theta > options.constr_viol_tol {
        if let Some(result) = try_classify_max_iter_infeasibility(
            state, final_theta, feas,
        ) {
            return result;
        }
    }

    if options.print_level >= 5 {
        timings.print_summary(options.max_iter, ipm_start.elapsed());
    }

    let primal_inf = state.constraint_violation();
    let dual_inf = compute_dual_inf_at_state(state);
    let compl_inf = compute_compl_err_at_state(state);
    if primal_inf <= options.constr_viol_tol
        && dual_inf <= options.dual_inf_tol
        && compl_inf <= options.compl_inf_tol
        && primal_inf <= options.tol
        && dual_inf <= options.tol
        && compl_inf <= options.tol
    {
        return make_result(state, SolveStatus::Optimal);
    }
    if primal_inf <= 1e-2
        && dual_inf <= 1e10
        && compl_inf <= 1e-2
        && primal_inf <= 1e-6
        && dual_inf <= 1e-6
        && compl_inf <= 1e-6
    {
        return make_result(state, SolveStatus::Acceptable);
    }
    make_result(state, SolveStatus::MaxIterations)
}

/// Fraction-to-boundary on `dx_soc` against the variable bounds, then
/// build the bounded trial point `x_soc = x + α·dx_soc` (clamped strictly
/// inside finite bounds by 1e-14). Returns `(x_soc, alpha_primal_soc)`.
/// Shared across the three second-order-correction variants.
fn compute_soc_alpha_and_trial_x(
    state: &SolverState,
    dx_soc: &[f64],
    tau: f64,
) -> (Vec<f64>, f64) {
    let alpha_primal_soc =
        fraction_to_boundary_primal_x(state, dx_soc, tau).clamp(0.0, 1.0);

    let x_soc = compute_clamped_trial_x(state, dx_soc, alpha_primal_soc);
    (x_soc, alpha_primal_soc)
}

/// Outcome of one second-order-correction trial-point evaluation.
enum SocTrialOutcome {
    /// Filter accepted the trial — caller returns this from the SOC fn.
    Accepted { x_soc: Vec<f64>, obj_soc: f64, g_soc: Vec<f64> },
    /// Eval failed, NaN/Inf detected, or theta failed the κ_soc test —
    /// caller bails out (returns `None`).
    Abort,
    /// theta improved but filter rejected — caller continues the SOC
    /// loop using `g_soc` to refresh `latest_trial_c`.
    NotAccepted { g_soc: Vec<f64> },
}

/// Evaluate objective/constraints at `x_soc`, gate on the κ_soc theta
/// reduction test, then check filter acceptability of (theta_soc,
/// phi_soc) against (theta_current, phi_current). Mutates
/// `theta_prev_soc` when theta improved enough to keep iterating.
/// Mirrors Ipopt IpFilterLSAcceptor.cpp:617-640.
#[allow(clippy::too_many_arguments)]
fn evaluate_soc_trial_and_check<P: NlpProblem>(
    state: &SolverState,
    problem: &P,
    options: &SolverOptions,
    filter: &mut Filter,
    x_soc: Vec<f64>,
    s_soc: &[f64],
    n: usize,
    m: usize,
    theta_current: f64,
    phi_current: f64,
    grad_phi_step: f64,
    alpha: f64,
    kappa_soc: f64,
    theta_prev_soc: &mut f64,
) -> SocTrialOutcome {
    // Phase 10b.2: SOC trial eval routes through SplitNlp. The
    // m-form `g_soc` survives only inside the adapter's scratch and
    // the caller-owned mirror used to feed `commit_trial_point` /
    // `latest_trial_*` builders that still expect combined form.
    let nlp = crate::split_nlp::SplitNlp::new(problem, &state.layout);
    let mut obj_soc = f64::INFINITY;
    if !nlp.objective(&x_soc, true, &mut obj_soc) || !obj_soc.is_finite() {
        return SocTrialOutcome::Abort;
    }
    let mut g_soc = vec![0.0; m];
    if !nlp.constraints_combined(&x_soc, false, &mut g_soc) {
        return SocTrialOutcome::Abort;
    }
    if g_soc.iter().any(|v| !v.is_finite()) {
        return SocTrialOutcome::Abort;
    }

    // A8.19 slack-coupled SOC theta: s_soc = state.s + α_p_soc · ds_d_soc.
    let (c_soc_split, d_soc_split) = split_from_g(state, &g_soc);
    let theta_soc = theta_for_split_d_s(state, &c_soc_split, &d_soc_split, s_soc);
    if theta_soc >= kappa_soc * *theta_prev_soc {
        return SocTrialOutcome::Abort;
    }
    *theta_prev_soc = theta_soc;

    let phi_soc = compute_barrier_phi(
        obj_soc, &x_soc, s_soc, state, n, m, options.constraint_slack_barrier,
        options.kappa_d,
    );

    // Pass ORIGINAL alpha (alpha_primal_test), not alpha_primal_soc.
    // Matches Ipopt IpFilterLSAcceptor.cpp:629.
    let (acceptable, _) = filter.check_acceptability(
        theta_current,
        phi_current,
        theta_soc,
        phi_soc,
        grad_phi_step,
        alpha, false,
    );

    if acceptable {
        SocTrialOutcome::Accepted { x_soc, obj_soc, g_soc }
    } else {
        SocTrialOutcome::NotAccepted { g_soc }
    }
}

/// SOC using the 4-block augmented KKT system (A7.5b).
///
/// Ports `IpFilterLSAcceptor::TrySecondOrderCorrection` (`IpFilterLSAcceptor.cpp:550-640`)
/// for the augmented path. Two accumulators — `c_soc` (length `n_c`,
/// equality residual `g_eq − g_eq_target`) and `dms_soc` (length `n_d`,
/// inequality consistency residual `g_ineq − s_ineq`) — track the
/// constraint violation across SOC iterations. Each iteration accumulates
/// `α_p_soc · trial_*`, builds a fresh aug Newton RHS at the current
/// iterate, and overwrites the y_c / y_d slots with the accumulators
/// (`soc_method = 0`).
///
/// Re-factors per call; A7.7 (factor caching) will share the upstream
/// Newton step's factorization since the matrix W + Σ does not change.
#[allow(clippy::too_many_arguments)]
fn attempt_soc_aug<P: NlpProblem>(
    state: &SolverState,
    problem: &P,
    g_trial: &[f64],
    _inertia_params: &mut InertiaCorrectionParams,
    filter: &mut Filter,
    theta_current: f64,
    phi_current: f64,
    grad_phi_step: f64,
    alpha: f64,
    options: &SolverOptions,
    aug_solver: &mut dyn LinearSolver,
    aug_kkt: &crate::kkt_aug::AugKktSystem,
) -> Option<(Vec<f64>, f64, Vec<f64>, f64)> {
    let n = state.n;
    let m = state.m;
    if m == 0 {
        return None;
    }

    let g_l_combined_for_soc = state.g_l_combined();
    let g_u_combined_for_soc = state.g_u_combined();
    let partition = crate::constraint_layout::ConstraintLayout::new(&g_l_combined_for_soc, &g_u_combined_for_soc);
    let n_c = partition.n_c;
    let n_d = partition.n_d;

    // Phase 5d: read split storage directly. c_soc[k] = c(x)[k] (Ipopt's
    // curr_c, IpIpoptCalculatedQuantities); dms_soc[k] = d(x)[k] - s[k]
    // (curr_d_minus_s).
    let mut c_soc = vec![0.0; n_c];
    let mut dms_soc = vec![0.0; n_d];
    for k in 0..n_c {
        c_soc[k] = state.c_x[k];
    }
    for k in 0..n_d {
        dms_soc[k] = state.d_x[k] - state.s[k];
    }

    // First-iteration trial residuals: g_trial / s_trial = s + α·ds.
    let mut latest_trial_c = vec![0.0; n_c];
    let mut latest_trial_dms = vec![0.0; n_d];
    for i in 0..m {
        if let Some(k) = partition.eq_pos[i] {
            latest_trial_c[k] = g_trial[i] - state.g_l_at(i);
        } else if let Some(k) = partition.ineq_pos[i] {
            let s_trial_i = state.s[k] + alpha * state.ds[k];
            latest_trial_dms[k] = g_trial[i] - s_trial_i;
        }
    }

    let kappa_soc = 0.99;
    let tau = (1.0 - state.mu).max(options.tau_min);

    let mut alpha_primal_soc = alpha;
    // A8.19 slack-coupled theta: the SOC seed reuses the upstream
    // line-search trial slack `s + α·ds`, matching `latest_trial_dms`
    // initialised above.
    let s_trial_seed = compute_trial_slack(state, alpha);
    let (c_seed_soc, d_seed_soc) = split_from_g(state, g_trial);
    let mut theta_prev_soc = theta_for_split_d_s(state, &c_seed_soc, &d_seed_soc, &s_trial_seed);

    // Phase 6d.5: materialize compressed z once for the SOC inner loop's
    // kkt_aug calls (kkt_aug.rs still consumes full-`n` z slices).
    let z_l_full = state.z_l_combined();
    let z_u_full = state.z_u_combined();
    // Phase 7c: same for x_l/x_u.
    let x_l_full = state.x_l_combined();
    let x_u_full = state.x_u_combined();
    // Phase 8c.5: same for compressed v_l/v_u → length n_d.
    let v_l_full = state.d_bound_layout.expand_l(&state.v_l_compressed, 0.0);
    let v_u_full = state.d_bound_layout.expand_u(&state.v_u_compressed, 0.0);
    // Phase 9c.5: same for compressed d_l/d_u → length n_d.
    let d_l_full = state.d_l();
    let d_u_full = state.d_u();

    for _soc_iter in 0..options.max_soc {
        for k in 0..n_c {
            c_soc[k] += alpha_primal_soc * latest_trial_c[k];
        }
        for k in 0..n_d {
            dms_soc[k] += alpha_primal_soc * latest_trial_dms[k];
        }

        // A7.7: SOC reuses the upstream Newton step's factorization. The
        // aug matrix (W + Σ + perturbation) is identical to the one
        // factored at the top of the IPM iteration; only the y_c/y_d RHS
        // slots change.
        let (dx_soc, ds_d_soc) = match crate::kkt_aug::aug_soc_solve_dx_factored(
            n, &state.grad_f,
            &state.jac_c_rows, &state.jac_c_cols, &state.jac_c_vals,
            &state.jac_d_rows, &state.jac_d_cols, &state.jac_d_vals,
            &state.x, &x_l_full, &x_u_full, &z_l_full, &z_u_full,
            &state.s, &state.c_x, &state.d_x, &d_l_full, &d_u_full,
            &state.y_c, &state.y_d,
            &v_l_full, &v_u_full,
            state.mu, options.kappa_d,
            aug_solver,
            aug_kkt,
            &c_soc, &dms_soc,
        ) {
            Some(p) => p,
            None => return None,
        };

        let (x_soc, alpha_primal_soc_new) =
            compute_soc_alpha_and_trial_x(state, &dx_soc, tau);
        alpha_primal_soc = alpha_primal_soc_new;

        // A8.19: build s_soc = state.s + α_p_soc · ds_d_soc (d-form).
        // Phase 3d: state.s and ds_d_soc both index by k ∈ [0, n_d).
        let mut s_soc = state.s.clone();
        for k in 0..n_d {
            s_soc[k] = state.s[k] + alpha_primal_soc * ds_d_soc[k];
        }

        match evaluate_soc_trial_and_check(
            state, problem, options, filter, x_soc, &s_soc, n, m,
            theta_current, phi_current, grad_phi_step, alpha,
            kappa_soc, &mut theta_prev_soc,
        ) {
            SocTrialOutcome::Accepted { x_soc, obj_soc, g_soc } => {
                return Some((x_soc, obj_soc, g_soc, alpha_primal_soc));
            }
            SocTrialOutcome::Abort => return None,
            SocTrialOutcome::NotAccepted { g_soc } => {
                // Refresh latest_trial_* using the rejected SOC trial:
                //   s_soc[k] = state.s[k] + α_p_soc · ds_d_soc[k]   for k ∈ d-block.
                for i in 0..m {
                    if let Some(k) = partition.eq_pos[i] {
                        latest_trial_c[k] = g_soc[i] - state.g_l_at(i);
                    } else if let Some(k) = partition.ineq_pos[i] {
                        let s_soc_i = state.s[k] + alpha_primal_soc * ds_d_soc[k];
                        latest_trial_dms[k] = g_soc[i] - s_soc_i;
                    }
                }
            }
        }
    }

    None
}

/// Post-restoration bound-multiplier handoff aligned with Ipopt 3.14
/// `MinC_1NrmRestorationPhase::PerformRestoration`
/// (`IpRestoMinC_1Nrm.cpp:374-419`).
///
/// Treats the entire restoration progress in `(x, s)` as a single
/// primal-dual Newton step:
///
///   δ_z = (μ − z_curr · trial_slack) / curr_slack − z_curr   (per bound)
///
/// (Ipopt's `ComputeBoundMultiplierStep`, `IpRestoMinC_1Nrm.cpp:438-453`),
/// applied to all four blocks `(z_L, z_U, v_L, v_U)` using the parent's
/// **pre-restoration** multipliers and slacks (NOT the inner-resto NLP's
/// z's — those solve a different stationarity).
///
/// A single `α_dual` is computed via `dual_frac_to_the_bound` across all
/// four blocks (`IpRestoMinC_1Nrm.cpp:394-395`), then the step is applied
/// to all four. If any post-step multiplier exceeds
/// `bound_mult_reset_threshold` (Ipopt default 1e3, `IpRestoMinC_1Nrm.cpp:40`),
/// **all four blocks are reset to 1.0** (lines 402-419 — the "nuclear
/// reset"). Returns whether the nuclear reset fired (informational only;
/// caller no longer branches on it).
///
/// `x_cur` and `s_cur` are the pre-restoration primal iterate / slack;
/// `state.x` and `state.s` already hold the post-restoration trial at
/// call time.
fn update_bound_multipliers_after_restoration(
    state: &mut SolverState,
    options: &SolverOptions,
    x_cur: &[f64],
    s_cur: &[f64],
) -> bool {
    let mu = state.mu.max(1e-20);
    let bound_mult_reset_threshold = 1000.0;
    let tau = (1.0 - mu).max(options.tau_min);

    let nx_l = state.bound_layout.n_x_l;
    let nx_u = state.bound_layout.n_x_u;
    let nd_l = state.d_bound_layout.n_d_l;
    let nd_u = state.d_bound_layout.n_d_u;

    let mut delta_zl = vec![0.0; nx_l];
    let mut delta_zu = vec![0.0; nx_u];
    let mut delta_vl = vec![0.0; nd_l];
    let mut delta_vu = vec![0.0; nd_u];

    // Ipopt's `ComputeBoundMultiplierStep` (`IpRestoMinC_1Nrm.cpp:438-453`):
    //   delta_z = ((curr_slack - trial_slack)·curr_z + mu) / curr_slack - curr_z
    // expanding the division:
    //   = curr_z - curr_z·trial_slack/curr_slack + mu/curr_slack - curr_z
    //   = (mu - curr_z·trial_slack) / curr_slack
    // The two `curr_z` terms cancel; the closed form below matches.
    for k in 0..nx_l {
        let i = state.bound_layout.x_l_to_full[k];
        let s_curr = (x_cur[i] - state.x_l_at(i)).max(1e-12);
        let s_trial = (state.x[i] - state.x_l_at(i)).max(1e-12);
        let z_curr = state.z_l_compressed[k];
        delta_zl[k] = (mu - z_curr * s_trial) / s_curr;
    }
    for k in 0..nx_u {
        let i = state.bound_layout.x_u_to_full[k];
        let s_curr = (state.x_u_at(i) - x_cur[i]).max(1e-12);
        let s_trial = (state.x_u_at(i) - state.x[i]).max(1e-12);
        let z_curr = state.z_u_compressed[k];
        delta_zu[k] = (mu - z_curr * s_trial) / s_curr;
    }
    for kc in 0..nd_l {
        let k = state.d_bound_layout.d_l_to_full[kc];
        let dl = state.d_l_compressed[kc];
        let s_curr = (s_cur[k] - dl).max(1e-12);
        let s_trial = (state.s[k] - dl).max(1e-12);
        let v_curr = state.v_l_compressed[kc];
        delta_vl[kc] = (mu - v_curr * s_trial) / s_curr;
    }
    for kc in 0..nd_u {
        let k = state.d_bound_layout.d_u_to_full[kc];
        let du = state.d_u_compressed[kc];
        let s_curr = (du - s_cur[k]).max(1e-12);
        let s_trial = (du - state.s[k]).max(1e-12);
        let v_curr = state.v_u_compressed[kc];
        delta_vu[kc] = (mu - v_curr * s_trial) / s_curr;
    }

    // Single α_dual via fraction-to-the-boundary across all four blocks.
    let mut alpha_dual = 1.0_f64;
    for k in 0..nx_l {
        if delta_zl[k] < 0.0 {
            alpha_dual = alpha_dual.min(-tau * state.z_l_compressed[k] / delta_zl[k]);
        }
    }
    for k in 0..nx_u {
        if delta_zu[k] < 0.0 {
            alpha_dual = alpha_dual.min(-tau * state.z_u_compressed[k] / delta_zu[k]);
        }
    }
    for kc in 0..nd_l {
        if delta_vl[kc] < 0.0 {
            alpha_dual = alpha_dual.min(-tau * state.v_l_compressed[kc] / delta_vl[kc]);
        }
    }
    for kc in 0..nd_u {
        if delta_vu[kc] < 0.0 {
            alpha_dual = alpha_dual.min(-tau * state.v_u_compressed[kc] / delta_vu[kc]);
        }
    }
    alpha_dual = alpha_dual.clamp(0.0, 1.0);

    let mut max_mult: f64 = 0.0;
    for k in 0..nx_l {
        state.z_l_compressed[k] = (state.z_l_compressed[k] + alpha_dual * delta_zl[k]).max(0.0);
        max_mult = max_mult.max(state.z_l_compressed[k]);
    }
    for k in 0..nx_u {
        state.z_u_compressed[k] = (state.z_u_compressed[k] + alpha_dual * delta_zu[k]).max(0.0);
        max_mult = max_mult.max(state.z_u_compressed[k]);
    }
    for kc in 0..nd_l {
        state.v_l_compressed[kc] = (state.v_l_compressed[kc] + alpha_dual * delta_vl[kc]).max(0.0);
        max_mult = max_mult.max(state.v_l_compressed[kc]);
    }
    for kc in 0..nd_u {
        state.v_u_compressed[kc] = (state.v_u_compressed[kc] + alpha_dual * delta_vu[kc]).max(0.0);
        max_mult = max_mult.max(state.v_u_compressed[kc]);
    }

    let nuclear_reset = max_mult > bound_mult_reset_threshold;
    if nuclear_reset {
        for k in 0..nx_l { state.z_l_compressed[k] = 1.0; }
        for k in 0..nx_u { state.z_u_compressed[k] = 1.0; }
        for kc in 0..nd_l { state.v_l_compressed[kc] = 1.0; }
        for kc in 0..nd_u { state.v_u_compressed[kc] = 1.0; }
    }
    nuclear_reset
}

/// Post-restoration y handoff aligned with Ipopt 3.14
/// `MinC_1NrmRestorationPhase::PerformRestoration`
/// (`IpRestoMinC_1Nrm.cpp:421-422`), which calls
/// `DefaultIterateInitializer::least_square_mults(...,
/// constr_mult_reset_threshold_)`.
///
/// In `IpDefaultIterateInitializer.cpp:685-738`, the LS branch fires
/// only when the cap parameter (`constr_mult_reset_threshold` here) is
/// `> 0.0`. With Ipopt's default `0.0`, the function falls through to
/// `cpp:734-737` and **sets y_c = y_d = 0** unconditionally. That's the
/// principled handoff: the resto inner y's solve a different
/// stationarity (the L1 objective with p/n slacks), so they're
/// meaningless to the parent; and a non-zero LS y computed at a
/// poorly-scaled restored iterate biases the parent's first Newton
/// direction — observed on arki0003 as a post-restoration dual-residual
/// blow-up that re-triggers restoration.
///
/// With `threshold > 0`, we keep the LS estimate when
/// `‖y_LS‖_∞ ≤ threshold` (matching `cpp:722-727`); otherwise zero.
fn recompute_y_after_restoration(
    state: &mut SolverState,
    options: &SolverOptions,
    n: usize,
    m: usize,
) {
    if m == 0 {
        return;
    }
    let threshold = options.constr_mult_reset_threshold;
    if threshold <= 0.0 {
        // Ipopt-default path: y = 0.
        let y_zero = vec![0.0; m];
        state.set_y_combined(&y_zero);
        return;
    }
    let g_l_combined_for_ls = state.g_l_combined();
    let g_u_combined_for_ls = state.g_u_combined();
    // Phase 6d.5: materialize compressed z to full-`n` for LS estimator.
    let z_l_full = state.z_l_combined();
    let z_u_full = state.z_u_combined();
    let (jac_rows_m, jac_cols_m, jac_vals_m) = rebuild_combined_jac(state);
    let y_ls_result = compute_ls_multiplier_estimate_augmented(
        &state.grad_f, &jac_rows_m, &jac_cols_m, &jac_vals_m,
        &g_l_combined_for_ls, &g_u_combined_for_ls, n, m,
        Some(&z_l_full), Some(&z_u_full),
        None,
    );
    let y_zero;
    let y_to_set = match y_ls_result {
        Some(ref y_ls) if linf_norm(y_ls) <= threshold => y_ls.as_slice(),
        _ => {
            y_zero = vec![0.0; m];
            y_zero.as_slice()
        }
    };
    state.set_y_combined(y_to_set);
}

/// T3.30 (full): post-step `recalc_y` aligned with `IpIpoptAlg.cpp:782-816`.
///
/// Solves Ipopt's full 4-block LS system (eliminated to a (n+m) augmented
/// matrix with `−1` diagonals on inequality rows and `(v_L − v_U)` RHS
/// contributions). Unlike the restoration-entry path, no
/// `constr_mult_init_max` magnitude clamp — Ipopt accepts the LS y
/// unconditionally on solver success and skips the recalc on solver
/// failure. Bound multipliers `z_L`, `z_U`, `v_L`, `v_U` are held fixed
/// (matches `IpIpoptAlg.cpp:803-806` carry-over).
fn recompute_y_post_step_full_augmented(
    state: &mut SolverState,
    n: usize,
    m: usize,
) {
    if m == 0 {
        return;
    }
    let v_l_combined_for_ls = state.v_l_combined();
    let v_u_combined_for_ls = state.v_u_combined();
    let g_l_combined_for_ls = state.g_l_combined();
    let g_u_combined_for_ls = state.g_u_combined();
    // Phase 6d.5: materialize compressed z to full-`n` for LS estimator.
    let z_l_full = state.z_l_combined();
    let z_u_full = state.z_u_combined();
    let (jac_rows_m, jac_cols_m, jac_vals_m) = rebuild_combined_jac(state);
    let y_ls_result = compute_ls_multiplier_estimate_augmented(
        &state.grad_f, &jac_rows_m, &jac_cols_m, &jac_vals_m,
        &g_l_combined_for_ls, &g_u_combined_for_ls, n, m,
        Some(&z_l_full), Some(&z_u_full),
        Some((&v_l_combined_for_ls, &v_u_combined_for_ls)),
    );
    if let Some(y_ls) = y_ls_result {
        state.set_y_combined(&y_ls);
    }
}

/// Spec §5 step 5 (`IpIpoptAlg.cpp:652-819`, P27): once an accepted iterate
/// is sufficiently feasible, recompute y via least-squares to keep the
/// equality multipliers aligned with the current x and z. Default-on under
/// L-BFGS (where quasi-Newton multiplier estimates drift), opt-in otherwise.
///
/// The `recalc_y_feas_tol` gate is essential: at infeasible iterates, the
/// LS y absorbs constraint violation into the dual variables, biasing the
/// next Newton direction. Spec default tol is `1e-6`.
fn maybe_recalc_y_post_step(
    state: &mut SolverState,
    options: &SolverOptions,
    n: usize,
    m: usize,
    lbfgs_mode: bool,
) {
    if m == 0 {
        return;
    }
    let gate_on = options.recalc_y || lbfgs_mode;
    if !gate_on {
        return;
    }
    // T3.30: Ipopt gates recalc_y on `IpCq().curr_constraint_violation()`,
    // which defaults to NORM_MAX (constr_viol_normtype option). The 1-norm
    // variant `state.constraint_violation()` is used for filter decisions
    // and for many-constraint problems is much larger than the max-norm,
    // making the recalc_y_feas_tol gate spuriously tight.
    let theta_max = convergence::primal_infeasibility_max_split(
        &state.g_c(),
        &state.g_d(),
        &state.d_l(),
        &state.d_u(),
    );
    if theta_max >= options.recalc_y_feas_tol {
        return;
    }
    // T3.30: Ipopt-aligned post-step recalc uses the full 4-block LS system
    // (slack/v_L/v_U coupling) and accepts unconditionally on solver success.
    recompute_y_post_step_full_augmented(state, n, m);
}

/// Apply post-restoration success handling: update state, reset multipliers, and mu.
///
/// T0.9: the existing filter is NOT cleared (Ipopt
/// IpRestoFilterConvCheck::TestOrigProgress requires the trial point
/// to be acceptable to the *current* filter — including the entry
/// added at restoration entry by `Filter::augment_for_restoration`).
/// If the resto-returned (θ, φ) is not filter-acceptable, this
/// function returns `false` and does not commit any state changes;
/// the caller must treat that as a restoration failure. On
/// acceptance, the filter is left untouched and `true` is returned.
#[must_use]
fn apply_restoration_success<P: NlpProblem>(
    state: &mut SolverState,
    filter: &mut Filter,
    mu_state: &mut MuState,
    options: &SolverOptions,
    n: usize,
    m: usize,
    problem: &P,
    x_new: &[f64],
    resto_z: Option<&(Vec<f64>, Vec<f64>)>,
    linear_constraints: Option<&[bool]>,
    lbfgs_mode: bool,
    lbfgs_state: &mut Option<LbfgsIpmState>,
) -> bool {
    // T0.9 filter gate: evaluate (θ, φ) for the trial restored point
    // and verify it is acceptable to the existing filter BEFORE
    // mutating any state. The "feasibility recovery" exemption
    // (θ < constr_viol_tol) is preserved because Ipopt itself
    // bypasses the filter test on feasibility (the feasibility-
    // restoration check resets the filter implicitly when the
    // recovered point is feasible).
    {
        // Phase 10b.3: post-restoration filter gate eval via SplitNlp.
        let nlp = crate::split_nlp::SplitNlp::new(problem, &state.layout);
        let mut g_check = vec![0.0; m];
        let mut phi_check = f64::INFINITY;
        let g_ok = m == 0 || nlp.constraints_combined(x_new, true, &mut g_check);
        let phi_ok = nlp.objective(x_new, true, &mut phi_check) && phi_check.is_finite();
        if !g_ok || !phi_ok {
            return false;
        }
        // Slack-RESYNCED filter check: pretend s has been pushed to
        // d(x_new) (which we will do below via initialize_slack_iterate).
        // Using the stale state.s here would inflate theta_check by
        // exactly the constraint shift restoration just produced —
        // rejecting successful restorations whose x movement actually
        // improved feasibility (observed on arki0003: stale-s gave
        // theta_check ≈ 7.37e3 while the restored x's true 1-norm
        // residual is 1.59e2). Mirrors `classify_restoration_outcome`'s
        // metric (`theta_new = theta_for_split_d_s(c_new, d_new, d_new)`)
        // so the inner exit, post-resto Success classifier, and
        // pre-commit filter check all agree.
        let theta_check = if m == 0 {
            0.0
        } else {
            let (c_check, d_check) = split_from_g(state, &g_check);
            theta_for_split_d_s(state, &c_check, &d_check, &d_check)
        };
        let feasible = theta_check < options.constr_viol_tol;
        if !feasible && !filter.is_acceptable(theta_check, phi_check) {
            return false;
        }
    }

    // Capture pre-restoration x so the one-shot Newton z update
    // (Ipopt §7.8 / IpRestoMinC_1Nrm.cpp:378-399) can compute s_cur.
    let x_cur = state.x.clone();
    state.x.copy_from_slice(x_new);
    state.alpha_primal = 0.0;

    let _ = evaluate_and_refresh_lbfgs(state, problem, lbfgs_state, linear_constraints, lbfgs_mode);

    // Save the pre-restoration slack iterate so the bound-multiplier
    // synthetic Newton step below has a `curr_slack_s` to reference
    // (Ipopt's `IpCq().curr_slack_s_L/U`). After
    // `initialize_slack_iterate` overwrites `state.s` to the new
    // strictly-interior box around `d(x_new)`, the pre-resto slack is
    // gone — capture it now.
    let s_cur = state.s.clone();

    // Resync the inequality-slack iterate `s` to the post-restoration
    // `d(x_new)`, projected into the strictly-interior box
    // `(d_l + p_L, d_u − p_U)` (same `slack_bound_push` /
    // `slack_bound_frac` policy as `initialize_slack_iterate`).
    // Without this, `state.s` retains its pre-restoration value while
    // `state.d_x` is fresh, and `theta_for_split_d_s = ||c|| + ||d − s||`
    // inherits a spurious `||d_new − s_old||` term equal to the entire
    // constraint shift the restoration just produced. On arki0003 this
    // pinned theta back at the pre-resto value (7.37e3) immediately
    // after a Success classification (theta_new = 1.59e2), triggering
    // another identical restoration entry — an infinite cycle.
    // Ipopt's `IpRestoMinC_1Nrm::finalize_solution` performs the
    // analogous slack push when handing the parent the recovered point.
    initialize_slack_iterate(state, m, options);

    // Ipopt-aligned bound-multiplier handoff: synthetic Newton step on
    // (z_L, z_U, v_L, v_U) using the parent's pre-restoration multipliers
    // and slacks, then a single dual fraction-to-the-boundary, then a
    // nuclear reset to 1.0 if any multiplier exceeds 1e3. Mirrors
    // `MinC_1NrmRestorationPhase::PerformRestoration`
    // (`IpRestoMinC_1Nrm.cpp:374-419`). The previous κ_σ-clamp variant
    // (which used the inner-resto z's instead of parent z's, and applied
    // a κ_σ clamp Ipopt does not apply at this point) was responsible
    // for inflating multipliers across repeated restoration cycles on
    // arki0003. The `resto_z` parameter is now unused — Ipopt always
    // bridges via the parent's z, never the inner-resto z.
    let _ = resto_z;
    let _ = update_bound_multipliers_after_restoration(state, options, &x_cur, &s_cur);
    recompute_y_after_restoration(state, options, n, m);

    // T0.9: do NOT clear the filter — Ipopt keeps the existing entries
    // (including the augmentation added at resto entry) so future
    // iterations cannot revisit the pre-resto basin. Only the
    // consecutive_acceptable counter is reset.
    state.consecutive_acceptable = 0;

    // Per Ipopt `IpIpoptAlgorithm.cpp:842`, the parent inherits its
    // pre-restoration μ rather than recomputing one from post-reset
    // complementarity. The avg-compl recompute previously here was
    // inflating μ to ~2e4 on arki0003 (when the nuclear z=1 reset
    // fired and fake compl = 1·slack dominated the average), driving
    // the next outer iteration into an unbarriered Newton regime that
    // immediately blew up. Keep the pre-resto μ; let the regular
    // mu-update logic decrease it once the iterate stabilizes.
    state.mu = state.mu.max(options.mu_min);

    // Reset mu_state mode (restoration is a fresh start)
    // In monotone strategy, stay in Fixed mode
    if options.mu_strategy_adaptive {
        mu_state.mode = MuMode::Free;
    }
    mu_state.first_iter_in_mode = true;
    mu_state.ref_vals.clear();
    mu_state.consecutive_restoration_failures = 0;
    // T3.25 follow-up: restoration handoff replaces x/y/z/v wholesale.
    // The atag updates above (via the writes to state.x etc.) do NOT
    // bump kkt_atags by themselves, so drop the cache explicitly. This
    // is the coarse "invalidate at boundary" path the T3.25 report
    // recommended over fine-grained per-write atag plumbing.
    state.bump_all_kkt_atags();
    state.factor_cache.invalidate();
    true
}

/// Outcome of the NLP restoration attempt.
enum RestorationOutcome {
    /// Restoration found a point with improved feasibility.
    Success,
    /// Local infeasibility: restoration converged but constraints not improved.
    LocalInfeasibility,
    /// Restoration failed (inner solve did not converge).
    Failed,
}

/// Attempt full NLP restoration by solving a restoration subproblem with the IPM.
///
/// Formulates a restoration NLP with p/n slack variables and solves it using
/// the same IPM engine (with `disable_nlp_restoration=true` to prevent recursion).
/// Configure the inner SolverOptions for the restoration NLP solve. Caps
/// max_iter at restoration_max_iter (>=500), disables nested restoration to
/// prevent recursion, sets mu_init to resto_mu, and relaxes tol to 1e-7
/// (we want feasibility, not optimality). Propagates the remaining
/// wall-time budget so the inner solve can't outlive the outer fallback
/// cascade; returns None when the remaining budget is < 0.5s.
fn configure_restoration_inner_options(
    options: &SolverOptions,
    resto_mu: f64,
    _resto_dim: usize,
    start_time: Instant,
) -> Option<SolverOptions> {
    let mut inner_opts = options.clone();
    inner_opts.max_iter = options.restoration_max_iter.max(500);
    inner_opts.disable_nlp_restoration = true;
    inner_opts.print_level = if options.print_level >= 5 { 3 } else { 0 };
    inner_opts.mu_init = resto_mu;
    inner_opts.tol = 1e-7;

    if options.max_wall_time > 0.0 {
        let remaining = options.max_wall_time - start_time.elapsed().as_secs_f64();
        if remaining < 0.5 {
            return None;
        }
        inner_opts.max_wall_time = remaining;
    }
    Some(inner_opts)
}

fn attempt_nlp_restoration<P: NlpProblem>(
    problem: &P,
    state: &SolverState,
    filter: &Filter,
    options: &SolverOptions,
    theta_current: f64,
    start_time: Instant,
) -> (Vec<f64>, Option<(Vec<f64>, Vec<f64>)>, RestorationOutcome) {
    let n = state.n;
    let m = state.m;

    if options.print_level >= 5 {
        rip_log!(
            "ripopt: Entering NLP restoration (theta={:.2e}, mu={:.2e})",
            theta_current, state.mu
        );
    }

    // Ipopt's published default (IpRestoIpoptNLP.cpp:60): rho=1000 fixed.
    // The earlier adaptive rho was premature optimization — Ipopt's slack-penalty
    // analysis relies on rho being fixed and large enough that sum(p+n) is
    // monotone across inner iterations.
    let rho = 1000.0;

    // Ipopt's resto_mu = max(curr_mu, ||c(x_r)||_inf, ||d(x_r) - s||_inf)
    // per `IpRestoIterateInitializer.cpp:57-61`. The third term is the
    // slack-coupling residual — without it, problems with significant
    // inequality-constraint infeasibility (where d - s dominates over c)
    // get a too-small inner mu, the closed-form (p_d, n_d) init is wrong
    // by orders of magnitude, the FTB collapses α on the first inner
    // step, and the inner IPM diverges → repeated parent-side restoration
    // entries (the arki0003 cycling pattern). Use the internal-max form
    // (`||c||_∞ ∪ ||d − s||_∞`) which mirrors Ipopt's
    // `IpIpoptCalculatedQuantities::curr_primal_infeasibility(NORM_MAX)`.
    let c_inf = compute_primal_inf_internal_max_at_state(state);
    let resto_mu = state.mu.max(c_inf);

    // Build restoration NLP using the same resto_mu for p/n quadratic init.
    let mut resto_nlp = RestorationNlp::new(
        problem,
        &state.x,
        resto_mu,
        rho,
        options.resto_proximity_weight,
    );
    // T3.X: inject parent-violation target so the inner solve can short-
    // circuit on `IpRestoFilterConvCheck::TestOrigProgress`. Without
    // this, the inner solve_ipm runs to its own KKT optimum (often 499
    // iters with mu→0) even though the parent recovered feasibility long
    // ago — observed on arki0003 where iter ~48 of the inner solve
    // already had inf_pr ≤ 1e-9 of the resto NLP. The κ_resto gate uses
    // max-norm; the filter / sufficient-progress gates use 1-norm
    // (matching Ipopt's `IpFilterLSAcceptor.cpp:497-498`).
    let parent_theta_entry = compute_primal_inf_max_at_state(state);
    // 1-norm of bound violation at the parent iterate, computed from
    // state.c_x (equality residuals; equality rows have g_l == g_u so
    // |c| equals bound violation) and state.d_x against (d_l, d_u)
    // (inequality bound violations).
    let parent_theta_entry_l1 = {
        let mut sum = 0.0_f64;
        for &c in state.c_x.iter() {
            sum += c.abs();
        }
        let d_l = state.d_l();
        let d_u = state.d_u();
        for k in 0..state.layout.n_d {
            let dx = state.d_x[k];
            let lo = d_l[k];
            let hi = d_u[k];
            if dx < lo {
                sum += lo - dx;
            } else if dx > hi {
                sum += dx - hi;
            }
        }
        sum
    };
    // φ_entry = parent's barrier-augmented objective at restoration
    // entry: f(x_R) − μ · Σ ln(slack). D5 fix: parent filter entries
    // are stored as barrier-φ, so `phi_trial` in the resto progress
    // gate is also augmented (RestorationNlp::should_exit_for_parent);
    // φ_entry must use the same metric or the comparison is biased.
    let parent_x_l_full = state.x_l_combined();
    let parent_x_u_full = state.x_u_combined();
    let mu_entry = state.mu;
    let mut parent_phi_entry = f64::INFINITY;
    {
        let nlp = crate::split_nlp::SplitNlp::new(problem, &state.layout);
        let _ = nlp.objective(&state.x, false, &mut parent_phi_entry);
        if !parent_phi_entry.is_finite() {
            parent_phi_entry = 0.0;
        } else if mu_entry > 0.0 {
            for i in 0..n {
                let lo = parent_x_l_full[i];
                let hi = parent_x_u_full[i];
                if lo.is_finite() {
                    let s = state.x[i] - lo;
                    if s > 0.0 {
                        parent_phi_entry -= mu_entry * s.ln();
                    }
                }
                if hi.is_finite() {
                    let s = hi - state.x[i];
                    if s > 0.0 {
                        parent_phi_entry -= mu_entry * s.ln();
                    }
                }
            }
            if !parent_phi_entry.is_finite() {
                parent_phi_entry = 0.0;
            }
        }
    }
    let parent_small_threshold = options.tol.min(options.constr_viol_tol);
    resto_nlp.set_parent_target(
        parent_theta_entry,
        parent_theta_entry_l1,
        parent_phi_entry,
        options.kappa_resto,
        parent_small_threshold,
        filter.theta_max(),
        filter.gamma_theta(),
        filter.gamma_phi(),
        filter.save_entries(),
        mu_entry,
        parent_x_l_full,
        parent_x_u_full,
    );

    let inner_opts = match configure_restoration_inner_options(
        options, resto_mu, resto_nlp.num_variables() + resto_nlp.num_constraints(), start_time,
    ) {
        Some(opts) => opts,
        None => return (state.x[..n].to_vec(), None, RestorationOutcome::Failed),
    };

    // Solve the restoration NLP
    let result = solve_ipm(&resto_nlp, &inner_opts);

    // Extract x_orig from the restoration solution
    let x_nlp: Vec<f64> = result.x[..n].to_vec();
    // Extract the bound multipliers for the original-x block (T0.8). The
    // resto NLP variable layout is [x(n), p(m), n(m)]; the p/n slack
    // multipliers are not relevant to the parent problem. Validate that
    // the inner solve returned an n-block (it always should), else None.
    let resto_z: Option<(Vec<f64>, Vec<f64>)> = if result.bound_multipliers_lower.len() >= n
        && result.bound_multipliers_upper.len() >= n
    {
        let zl_x: Vec<f64> = result.bound_multipliers_lower[..n].to_vec();
        let zu_x: Vec<f64> = result.bound_multipliers_upper[..n].to_vec();
        let finite = zl_x.iter().chain(zu_x.iter()).all(|v| v.is_finite());
        if finite { Some((zl_x, zu_x)) } else { None }
    } else {
        None
    };

    // Phase 10b.3: post-restoration original-NLP eval via SplitNlp.
    let nlp = crate::split_nlp::SplitNlp::new(problem, &state.layout);
    // Evaluate original constraints at the restored point
    let mut g_new = vec![0.0; m];
    if !nlp.constraints_combined(&x_nlp, true, &mut g_new)
        || g_new.iter().any(|v| !v.is_finite())
    {
        return (x_nlp, resto_z, RestorationOutcome::Failed);
    }
    // Compute theta_new using the SLACK-RESYNCED metric: pretend s has
    // already been updated to match d_new (which apply_restoration_success
    // does shortly via update_bound_multipliers_after_restoration +
    // The pre-resto state.s reflects the OLD x_R's d_x, so leaving it stale
    // overstates ||d_new − s_old||_1 by exactly the constraint shift the
    // restoration just produced — penalising successful restorations whose
    // x movement actually improved feasibility (e.g. arki0003 where the
    // Ipopt-aligned `IpRestoFilterConvCheck::TestOrigProgress` early-exit
    // returns x_n with ||c_orig(x_n) − π[bounds]||_∞ ≤ kappa_resto · θ_R
    // but stale-s theta_new is huge). Using d_new for both d AND s reduces
    // the slack-coupled metric to ||c_new||_1, which is the metric the
    // parent will see post-resync.
    let theta_new = {
        let (c_new, d_new) = split_from_g(state, &g_new);
        theta_for_split_d_s(state, &c_new, &d_new, &d_new)
    };

    // Evaluate original objective at the restored point
    let mut phi_new = f64::INFINITY;
    if !nlp.objective(&x_nlp, false, &mut phi_new) || !phi_new.is_finite() {
        return (x_nlp, resto_z, RestorationOutcome::Failed);
    }

    if options.print_level >= 5 {
        // Log slack residuals to diagnose restoration effectiveness
        let sum_p: f64 = result.x[n..n + m].iter().sum();
        let sum_n: f64 = result.x[n + m..n + 2 * m].iter().sum();
        rip_log!(
            "ripopt: NLP restoration result: status={:?}, theta_new={:.2e} (was {:.2e}), phi_new={:.2e}, sum_p={:.2e}, sum_n={:.2e}, iters={}",
            result.status, theta_new, theta_current, phi_new, sum_p, sum_n, result.iterations
        );
    }

    let inner_converged = result.status == SolveStatus::Optimal;
    let outcome = classify_restoration_outcome(
        filter, options, theta_current, theta_new, phi_new, inner_converged,
    );
    (x_nlp, resto_z, outcome)
}

/// Check whether the restored `(theta_new, phi_new)` is acceptable
/// to the current filter. Mirrors the inline filter test but without
/// triggering an `add` — used by `classify_restoration_outcome` to
/// decide whether a 10–50% θ reduction qualifies as Success when
/// the new point is also filter-acceptable. NaN inputs and
/// `theta_new > theta_max` reject; otherwise rejects on any entry
/// where both `(1-γ_θ)·θ_e ≤ θ_new` and `φ_e − γ_φ·θ_e ≤ φ_new`.
fn filter_accepts_restored_iterate(
    filter: &Filter,
    theta_new: f64,
    phi_new: f64,
) -> bool {
    let theta_max = filter.theta_max();
    let gamma_theta = filter.gamma_theta();
    let gamma_phi = filter.gamma_phi();

    if theta_new.is_nan() || phi_new.is_nan() || theta_new > theta_max {
        return false;
    }
    for entry in filter.entries() {
        if theta_new >= (1.0 - gamma_theta) * entry.theta
            && phi_new >= entry.phi - gamma_phi * entry.theta
        {
            return false;
        }
    }
    true
}

/// Classify the outcome of a completed restoration solve. Mirrors Ipopt's
/// `IpRestoFilterConvCheck::CheckProgress` (`IpRestoConvCheck.cpp:71-248`,
/// spec §7.7). Decision tree:
/// 1. theta_new < min(tol, constr_viol_tol) → Success (achieved feasibility,
///    matches Ipopt's "small_threshold" gate).
/// 2. theta_new ≤ kappa_resto * theta_current AND filter-acceptable → Success
///    (Ipopt's primary `required_infeasibility_reduction` gate, default 0.9).
/// 3. inner_converged but no feasibility improvement → LocalInfeasibility
///    (the restoration NLP itself reached a stationary point of the
///    L1-feasibility objective with positive residual).
/// 4. Otherwise → Failed.
///
/// DEV-9: removed the lenient `theta_new ≤ 0.5 * theta_current` gate.
/// Ipopt's primary success criterion is the `kappa_resto` reduction
/// AND filter acceptance — never one without the other. The 50% gate
/// could accept restoration exits that the filter would reject, biasing
/// the next iterate.
fn classify_restoration_outcome(
    filter: &Filter,
    options: &SolverOptions,
    theta_current: f64,
    theta_new: f64,
    phi_new: f64,
    inner_converged: bool,
) -> RestorationOutcome {
    let small_threshold = options.tol.min(options.constr_viol_tol);
    if theta_new < small_threshold {
        return RestorationOutcome::Success;
    }
    if theta_new <= options.kappa_resto * theta_current
        && filter_accepts_restored_iterate(filter, theta_new, phi_new)
    {
        return RestorationOutcome::Success;
    }
    if inner_converged {
        return RestorationOutcome::LocalInfeasibility;
    }
    RestorationOutcome::Failed
}

/// Overwrite the default constraint and bound multipliers with values
/// supplied by the problem's `initial_multipliers` hook. Called only when
/// `options.warm_start` is true; if the hook returns false, the defaults
/// are kept. WarmStartInitializer::initialize (called later in solve())
/// will still floor z_l/z_u and recompute mu from complementarity.
fn apply_warm_start_multipliers<P: NlpProblem>(
    problem: &P,
    y: &mut [f64],
    z_l: &mut [f64],
    z_u: &mut [f64],
) {
    let mut ws_lam_g = vec![0.0; y.len()];
    let mut ws_z_l = vec![0.0; z_l.len()];
    let mut ws_z_u = vec![0.0; z_u.len()];
    if problem.initial_multipliers(&mut ws_lam_g, &mut ws_z_l, &mut ws_z_u) {
        if !y.is_empty() {
            y.copy_from_slice(&ws_lam_g);
        }
        z_l.copy_from_slice(&ws_z_l);
        z_u.copy_from_slice(&ws_z_u);
    }
}

/// Initialize bound multipliers `z_l`, `z_u`. The method is selected
/// by `options.bound_mult_init_method` (Ipopt 3.14
/// `IpDefaultIterateInitializer.cpp:254-288`):
///
/// - `Constant` (Ipopt default): every finite-bounded entry is set to
///   `options.bound_mult_init_val` (Ipopt default 1.0). Inactive
///   (infinite) bounds stay at 0.
/// - `MuBased`: `z_l = mu_init / (x − x_l)`, `z_u = mu_init / (x_u − x)`.
///   Slack is floored at 1e-20 to avoid division by zero.
fn init_bound_multipliers(
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    mu_init: f64,
    options: &SolverOptions,
) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut z_l = vec![0.0; n];
    let mut z_u = vec![0.0; n];
    match options.bound_mult_init_method {
        BoundMultInitMethod::Constant => {
            let v = options.bound_mult_init_val;
            for i in 0..n {
                if x_l[i].is_finite() {
                    z_l[i] = v;
                }
                if x_u[i].is_finite() {
                    z_u[i] = v;
                }
            }
        }
        BoundMultInitMethod::MuBased => {
            for i in 0..n {
                if x_l[i].is_finite() {
                    let slack = (x[i] - x_l[i]).max(1e-20);
                    z_l[i] = mu_init / slack;
                }
                if x_u[i].is_finite() {
                    let slack = (x_u[i] - x[i]).max(1e-20);
                    z_u[i] = mu_init / slack;
                }
            }
        }
    }
    (z_l, z_u)
}

/// Apply Ipopt 3.14's `bound_relax_factor` mechanism: relax every
/// finite bound outward by `min(constr_viol_tol, factor·max(|bound|, 1))`
/// — the cap is `constr_viol_tol`, not a machine-eps floor (matching
/// `IpOrigIpoptNLP.cpp:459-481`). Applied to both variable bounds
/// `x_l`/`x_u` and constraint bounds `g_l`/`g_u` (Ipopt relaxes
/// `d_L`/`d_U` the same way at `IpOrigIpoptNLP.cpp:355-358`).
///
/// Equality pairs (`lower[i] == upper[i]`) are left UNTOUCHED. Ipopt
/// represents equality constraints separately from inequality bounds and
/// never relaxes them; in ripopt they live in the same `g_l`/`g_u`
/// arrays, so we must guard explicitly. Same rule applies to fixed
/// variables (which `relax_fixed_variable_bounds` handles afterward).
///
/// `factor <= 0.0` is a no-op (option-disable path).
fn apply_bound_relax_factor(
    lower: &mut [f64],
    upper: &mut [f64],
    factor: f64,
    constr_viol_tol: f64,
) {
    if !(factor > 0.0) {
        return;
    }
    debug_assert_eq!(lower.len(), upper.len());
    for i in 0..lower.len() {
        if lower[i].is_finite() && upper[i].is_finite() && lower[i] == upper[i] {
            continue;
        }
        if lower[i].is_finite() {
            let delta = (factor * lower[i].abs().max(1.0)).min(constr_viol_tol);
            lower[i] -= delta;
        }
        if upper[i].is_finite() {
            let delta = (factor * upper[i].abs().max(1.0)).min(constr_viol_tol);
            upper[i] += delta;
        }
    }
}

/// Apply the configured `fixed_variable_treatment` to entries with
/// `x_l[i] == x_u[i]`.
///
/// `RelaxBounds` (current default): widen the bounds by ±1e-8·max(|c|, 1)
/// around the fixed value `c`. Mirrors Ipopt 3.14's `relax_bounds`.
///
/// `MakeParameter` (Ipopt default; not yet implemented): would substitute
/// out fixed variables. Falls back to `RelaxBounds` here so the option
/// is a no-op for callers — see TODO in `FixedVariableTreatment` doc.
fn relax_fixed_variable_bounds(
    x_l: &mut [f64],
    x_u: &mut [f64],
    options: &SolverOptions,
) {
    match options.fixed_variable_treatment {
        FixedVariableTreatment::RelaxBounds | FixedVariableTreatment::MakeParameter => {
            for i in 0..x_l.len() {
                if x_l[i].is_finite() && x_u[i].is_finite() && (x_u[i] - x_l[i]).abs() < 1e-10 {
                    let center = (x_l[i] + x_u[i]) / 2.0;
                    let relax = 1e-8 * center.abs().max(1.0);
                    x_l[i] = center - relax;
                    x_u[i] = center + relax;
                }
            }
        }
    }
}

/// Push the initial point strictly inside finite variable bounds.
///
/// Ipopt 3.14 (`IpDefaultIterateInitializer.cpp:526-599`):
///   pL = min(κ1·max(|x_L|, 1), κ2·(x_U − x_L))   [two-sided]
///   pL = κ1·max(|x_L|, 1)                         [one-sided lower]
///   pU = κ1·max(|x_U|, 1)                         [one-sided upper]
/// where κ1 = `bound_push` and κ2 = `bound_frac`. The `max(|x|, 1)`
/// scaling matters for one-sided bounds at large magnitude: without it,
/// `x ≥ 1e3` gets pushed by only 0.01 absolute and the slack-driven
/// initial multipliers `z = μ/slack` blow up the KKT factorization.
fn push_initial_point_from_bounds(
    x: &mut [f64],
    x_l: &[f64],
    x_u: &[f64],
    options: &SolverOptions,
) {
    let kappa1 = options.bound_push;
    let kappa2 = options.bound_frac;
    for i in 0..x.len() {
        let lower_finite = x_l[i].is_finite();
        let upper_finite = x_u[i].is_finite();
        if lower_finite && upper_finite {
            let range = x_u[i] - x_l[i];
            let p_l = (kappa1 * x_l[i].abs().max(1.0)).min(kappa2 * range);
            let p_u = (kappa1 * x_u[i].abs().max(1.0)).min(kappa2 * range);
            x[i] = x[i].max(x_l[i] + p_l).min(x_u[i] - p_u);
        } else if lower_finite {
            let p_l = kappa1 * x_l[i].abs().max(1.0);
            x[i] = x[i].max(x_l[i] + p_l);
        } else if upper_finite {
            let p_u = kappa1 * x_u[i].abs().max(1.0);
            x[i] = x[i].min(x_u[i] - p_u);
        }
    }
}

/// Compute initial constraint multipliers via least-squares estimate,
/// matching Ipopt 3.14 `IpLeastSquareMults::CalculateMultipliers`
/// (`IpLeastSquareMults.cpp:53-94`).
///
/// Solves the 4-block augmented saddle-point system
///
///   [ I        0      J_cᵀ   J_dᵀ ] [sol_x]   [grad_f − P_xL·z_L + P_xU·z_U]
///   [ 0        I       0     -I   ] [sol_s] = [P_dL·v_L − P_dU·v_U          ]
///   [ J_c      0       0      0   ] [y_c  ]   [0                             ]
///   [ J_d     -I       0      0   ] [y_d  ]   [0                             ]
///
/// with `δ_x = δ_s = 1` and `W = 0`. Eliminating the slack block yields a
/// sparse (n + m) symmetric system whose only difference from the reduced
/// 2-block form is the `-1` diagonal on inequality rows of (m,m) and the
/// `(v_L − v_U)` RHS contribution on those rows. At iter-0 cold start
/// `v_L = v_U = bound_mult_init_val`, so `b_s = 0`; the `-1` diagonal
/// is still essential — without it the saddle-point system is poorly
/// conditioned for inequality-heavy problems (e.g. arki0003: 2-block
/// `|y|_inf ≈ 5e3`, 4-block `|y_c|_inf ≈ 180`).
///
/// Returns `vec![0.0; m]` when LS init is disabled, `m == 0`, evaluation
/// fails, the LS solve fails, or `max(|y|_inf) > constr_mult_init_max`.
#[allow(clippy::too_many_arguments)]
fn compute_initial_y_with_ls<P: NlpProblem>(
    problem: &P,
    options: &SolverOptions,
    x: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    _x_l: &[f64],
    _x_u: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    n: usize,
    m: usize,
    jac_nnz: usize,
) -> Vec<f64> {
    if !(options.least_squares_mult_init && m > 0) {
        return vec![0.0; m];
    }
    // Square-problem branch: when m == n, Ipopt's
    // `IpDefaultIterateInitializer::least_square_mults` skips the LS solve and
    // initializes y = 0 directly (`IpDefaultIterateInitializer.cpp:685-690`).
    if m == n {
        return vec![0.0; m];
    }
    let mut grad_f_init = vec![0.0; n];
    let grad_ok = problem.gradient(x, true, &mut grad_f_init);
    let mut jac_vals_init = vec![0.0; jac_nnz];
    let jac_ok = problem.jacobian_values(x, false, &mut jac_vals_init);
    if !grad_ok || !jac_ok {
        return vec![0.0; m];
    }
    // Mirrors `LeastSquareMultipliers::CalculateMultipliers`
    // (`IpLeastSquareMults.cpp:30-95`), which is the default-path LSQ
    // multiplier estimator invoked from
    // `DefaultIterateInitializer::least_square_mults`
    // (IpDefaultIterateInitializer.cpp:669-743) when
    // `least_square_init_duals = no` (Ipopt 3.14 default). Solves the
    // 4-block augmented system
    //   ┌  I       0        J_c^T    J_d^T ┐ ┌sol_x┐   ┌ grad_f − z_L + z_U ┐
    //   │  0       I        0        -I    │ │sol_s│ = │      −v_L + v_U     │
    //   │ J_c      0        0         0    │ │ y_c │   │          0          │
    //   └ J_d     -I        0         0    ┘ └ y_d ┘   │          0          ┘
    // (delta_x = delta_s = 1.0, delta_c = delta_d = 0, W = 0).
    // The output y_c, y_d are NOT sign-flipped (unlike
    // `CalculateLeastSquareDuals` which does flip).
    let y_aug = compute_initial_ls_4block(
        &grad_f_init, jac_rows, jac_cols, &jac_vals_init,
        z_l, z_u, g_l, g_u, options.bound_mult_init_val, n, m,
    );
    match y_aug {
        Some(y) if linf_norm(&y) <= options.constr_mult_init_max => y,
        _ => vec![0.0; m],
    }
}

/// Solve Ipopt's default-path LSQ multiplier system per
/// `LeastSquareMultipliers::CalculateMultipliers`
/// (`IpLeastSquareMults.cpp:30-95`). Returns y_combined (length m) on
/// success, None on factorization/solve failure.
///
/// Layout of the (n + n_d + n_c + n_d) symmetric system:
///   slot 0..n             → sol_x   ((1,1) = +I)
///   slot n..n+n_d         → sol_s   ((2,2) = +I)
///   slot n+n_d..n+n_d+n_c → y_c     ((3,3) = 0)
///   slot n+n_d+n_c..end   → y_d     ((4,4) = 0)
///
/// Off-diagonals (upper-triangle):
///   J_c^T    at (col_x, n+n_d + row_c)
///   J_d^T    at (col_x, n+n_d+n_c + row_d)
///   -I       at (n+k,   n+n_d+n_c + k)        (slack/y_d coupling)
///
/// RHS:
///   rhs_x[i] = grad_f[i] − z_L[i] + z_U[i]   (length n; z_L/z_U
///                                              already zero on
///                                              unbounded sides)
///   rhs_s[k] = − v_L[k] + v_U[k]              (length n_d; both = bmiv
///                                              if d-bound is finite,
///                                              else 0)
///   rhs_c, rhs_d = 0
///
/// After solve, y_c[combined_row] = sol_yc[c_pos[r]],
/// y_d[combined_row] = sol_yd[d_pos[r]] — NO sign flip
/// (`IpLeastSquareMults.cpp:80-94`).
fn compute_initial_ls_4block(
    grad_f: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    bound_mult_init_val: f64,
    n: usize,
    m: usize,
) -> Option<Vec<f64>> {
    use crate::linear_solver::SparseSymmetricMatrix;

    let mut c_pos: Vec<Option<usize>> = vec![None; m];
    let mut d_pos: Vec<Option<usize>> = vec![None; m];
    let mut n_c = 0usize;
    let mut n_d = 0usize;
    for r in 0..m {
        let is_eq = (g_l[r] - g_u[r]).abs() < 1e-15
            && g_l[r].is_finite() && g_u[r].is_finite();
        if is_eq {
            c_pos[r] = Some(n_c);
            n_c += 1;
        } else {
            d_pos[r] = Some(n_d);
            n_d += 1;
        }
    }
    debug_assert_eq!(n_c + n_d, m);

    let dim = n + n_d + n_c + n_d;
    let cap = n + n_d + jac_rows.len() + n_d + n_c + n_d;
    let mut ssm = SparseSymmetricMatrix {
        n: dim,
        triplet_rows: Vec::with_capacity(cap),
        triplet_cols: Vec::with_capacity(cap),
        triplet_vals: Vec::with_capacity(cap),
    };

    // (1,1) = +I (delta_x = 1.0).
    for i in 0..n {
        ssm.triplet_rows.push(i);
        ssm.triplet_cols.push(i);
        ssm.triplet_vals.push(1.0);
    }

    // (2,2) = +I (delta_s = 1.0).
    for k in 0..n_d {
        ssm.triplet_rows.push(n + k);
        ssm.triplet_cols.push(n + k);
        ssm.triplet_vals.push(1.0);
    }

    // (3,3), (4,4) = 0 (delta_c = delta_d = 0). Explicit structural
    // zeros so the sparse pattern is non-singular for the solver.
    for off in 0..(n_c + n_d) {
        ssm.triplet_rows.push(n + n_d + off);
        ssm.triplet_cols.push(n + n_d + off);
        ssm.triplet_vals.push(0.0);
    }

    // (1,3) J_c^T and (1,4) J_d^T off-diagonals.
    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        let val = jac_vals[idx];
        if let Some(rc) = c_pos[row] {
            ssm.triplet_rows.push(col);
            ssm.triplet_cols.push(n + n_d + rc);
            ssm.triplet_vals.push(val);
        } else if let Some(rd) = d_pos[row] {
            ssm.triplet_rows.push(col);
            ssm.triplet_cols.push(n + n_d + n_c + rd);
            ssm.triplet_vals.push(val);
        }
    }

    // (2,4) -I slack/y_d coupling.
    for k in 0..n_d {
        ssm.triplet_rows.push(n + k);
        ssm.triplet_cols.push(n + n_d + n_c + k);
        ssm.triplet_vals.push(-1.0);
    }

    // RHS construction. Per `IpLeastSquareMults.cpp:53-66`:
    //   rhs_x = -grad_f + Px_L*z_L - Px_U*z_U
    //   rhs_s = +Pd_L*v_L - Pd_U*v_U
    // (Sign emerges from the `MultVector(α, x, β, y)` calls computing
    //  `y := α·Op·x + β·y`; rhs_x is initialized to grad_f then negated.)
    let mut rhs = vec![0.0_f64; dim];
    for i in 0..n {
        rhs[i] = -grad_f[i] + z_l[i] - z_u[i];
    }
    // rhs_s[k] = +v_L[k] - v_U[k] for k in 0..n_d. At iter-0 with
    // BoundMultInitMethod::Constant, v_L[k] = bmiv·has_d_lower(k),
    // v_U[k] = bmiv·has_d_upper(k); reconstruct from g_l/g_u.
    for r in 0..m {
        if let Some(k) = d_pos[r] {
            let v_l_k = if g_l[r].is_finite() { bound_mult_init_val } else { 0.0 };
            let v_u_k = if g_u[r].is_finite() { bound_mult_init_val } else { 0.0 };
            rhs[n + k] = v_l_k - v_u_k;
        }
    }
    // rhs_c, rhs_d already 0.

    let matrix = KktMatrix::Sparse(ssm);
    let mut solver = new_sparse_solver();
    if solver.factor(&matrix).is_err() {
        return None;
    }
    let mut sol = vec![0.0_f64; dim];
    if solver.solve(&rhs, &mut sol).is_err() {
        return None;
    }
    if sol.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // No sign flip. y_c[r] = sol[c-slot]; y_d[r] = sol[d-slot].
    let mut y_combined = vec![0.0_f64; m];
    for r in 0..m {
        if let Some(rc) = c_pos[r] {
            y_combined[r] = sol[n + n_d + rc];
        } else if let Some(rd) = d_pos[r] {
            y_combined[r] = sol[n + n_d + n_c + rd];
        }
    }
    Some(y_combined)
}

/// Compute least-squares multiplier estimate via the Ipopt-exact augmented
/// saddle-point system (matches IpLeastSquareMults::CalculateMultipliers with
/// W=0, δ=0, no inertia correction).
///
/// Solves the (n+m)×(n+m) system
///   [ I    J^T ] [ r ]   [ grad_f - z_L + z_U ]
///   [ J     0  ] [ y ] = [ 0                  ]
/// and returns the y block.
///
/// This is algebraically equivalent to the normal-equations form
/// (J·J^T)·y = J·(grad_f − z_L + z_U), but is numerically far better
/// conditioned when J is nearly rank-deficient. The normal-equations form
/// fails outright (J·J^T singular) for AC-OPF problems with gauge symmetry
/// at post-restoration iterates (case30_ieee), whereas the augmented form's
/// LDL^T handles the indefinite saddle point via Bunch-Kaufman pivoting
/// without explicitly forming J·J^T.
///
/// On factorization failure, returns None. Matches Ipopt behavior at
/// IpLeastSquareMults.cpp:82-87 — no δ_c retry, no Tikhonov regularization.
/// Callers apply the `constr_mult_init_max` cap externally.
/// Build the symmetric augmented matrix for the LS multiplier
/// estimate in upper-triangle triplet form:
///
///   ```text
///   [ I    J^T ]
///   [ J     0  ]
///   ```
///
/// Layout: identity diagonal on rows `0..n`, the Jacobian transpose
/// at `(col, n+row)` for each `(row, col)` triple (always in the
/// upper triangle since `col < n ≤ n+row`), and explicit
/// structural zeros on the lower `(2,2)` diagonal so the sparse
/// solver sees a non-singular pattern. Used by
/// `compute_ls_multiplier_estimate_augmented`.
/// Build the (n+m) × (n+m) symmetric augmented matrix for the LS
/// multiplier estimate.
///
/// `inequality_diag` toggles between Ipopt's two LS systems:
/// - `inequality_diag = None` → reduced 2-block system used for the
///   initial-iterate LS init (`IpDefaultIterateInitializer.cpp:382-409`)
///   and for restoration entry: lower-right (m,m) block is zero.
/// - `inequality_diag = Some(slice)` → 4-block-equivalent system used
///   for post-step `recalc_y` (`IpLeastSquareMults.cpp:80-94`): for
///   inequality rows i (where `slice[i] = true`), the (n+i, n+i) entry
///   is `-1.0`; for equality rows it stays at 0. Eliminating the slack
///   block from Ipopt's 4-block system gives this structure.
fn build_ls_augmented_matrix(
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    n: usize,
    m: usize,
    inequality_diag: Option<&[bool]>,
    delta_c: f64,
    delta_d_extra: f64,
) -> crate::linear_solver::SparseSymmetricMatrix {
    use crate::linear_solver::SparseSymmetricMatrix;
    let nnz_est = n + jac_rows.len() + m;
    let mut ssm = SparseSymmetricMatrix {
        n: n + m,
        triplet_rows: Vec::with_capacity(nnz_est),
        triplet_cols: Vec::with_capacity(nnz_est),
        triplet_vals: Vec::with_capacity(nnz_est),
    };
    for i in 0..n {
        ssm.triplet_rows.push(i);
        ssm.triplet_cols.push(i);
        ssm.triplet_vals.push(1.0);
    }
    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        ssm.triplet_rows.push(col);
        ssm.triplet_cols.push(n + row);
        ssm.triplet_vals.push(jac_vals[idx]);
    }
    // Bottom-block diagonal:
    //   equality row  → -delta_c   (mirrors Ipopt's PDFullSpaceSolver
    //                                 (m_c, m_c) entry; 0 in nominal LS,
    //                                 lifted to delta_c when retrying after
    //                                 SYMSOLVER_SINGULAR).
    //   inequality row → -1 - delta_d_extra
    //                                 (eliminated form's -1 plus optional
    //                                 lift to break null-space ties on
    //                                 rank-deficient `J_d`).
    for j in 0..m {
        ssm.triplet_rows.push(n + j);
        ssm.triplet_cols.push(n + j);
        let diag = match inequality_diag {
            Some(flags) if flags[j] => -1.0 - delta_d_extra,
            _ => -delta_c,
        };
        ssm.triplet_vals.push(diag);
    }
    ssm
}

fn compute_ls_multiplier_estimate_augmented(
    grad_f: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    n: usize,
    m: usize,
    z_l: Option<&[f64]>,
    z_u: Option<&[f64]>,
    slack_mults: Option<(&[f64], &[f64])>,
) -> Option<Vec<f64>> {
    if m == 0 {
        return None;
    }

    // T3.30: when slack-multiplier inputs `(v_L, v_U)` are passed, build the
    // 4-block-equivalent augmented system per Ipopt
    // `IpLeastSquareMults.cpp:80-94`. Eliminating the explicit-slack block
    // yields a (n+m) symmetric system whose only difference from the reduced
    // 2-block form is: (a) a `-1.0` diagonal on the (m,m) lower-right block
    // for inequality rows; (b) a `(v_L − v_U)` contribution to the bottom
    // half of the RHS on inequality rows. Equality rows match the reduced
    // form exactly. When `slack_mults = None` we fall back to the 2-block
    // form (initial-iterate / restoration entry path).
    let inequality_flags: Option<Vec<bool>> = slack_mults.map(|_| {
        (0..m)
            .map(|i| !(g_l[i].is_finite() && g_u[i].is_finite()
                       && (g_l[i] - g_u[i]).abs() < 1e-15))
            .collect()
    });

    // RHS: [grad_f − z_L + z_U; (v_L − v_U) per inequality row, else 0]
    let mut rhs = vec![0.0_f64; n + m];
    for i in 0..n {
        rhs[i] = grad_f[i];
        if let Some(zl) = z_l { rhs[i] -= zl[i]; }
        if let Some(zu) = z_u { rhs[i] += zu[i]; }
    }
    if let (Some((v_l, v_u)), Some(flags)) = (slack_mults, inequality_flags.as_ref()) {
        for j in 0..m {
            if flags[j] {
                rhs[n + j] = v_l[j] - v_u[j];
            }
        }
    }

    // Mirror Ipopt's `PDFullSpaceSolver` SYMSOLVER_SINGULAR fallback at
    // `IpPDFullSpaceSolver.cpp:282-305`: try the LS augmented system with
    // `delta_c = delta_d = 0` first, and on failure (or on detection of a
    // suspiciously zeroed multiplier vector — feral can return a valid
    // null-space solution where MA27 returns the minimum-norm one) retry
    // with a small `delta_c0 = 1e-8` (matching Ipopt's
    // `IpPDPerturbationHandler` initial perturbation; see
    // `IpPDPerturbationHandler.cpp:60`). The 4010×4010 LS matrix on
    // arki0003 has 1041 linear-dependent equality rows; without δ_c the
    // factorization picks an arbitrary null-space point that zeroes 58
    // inequality multipliers, even though the (m,m) block has `-1` on
    // those rows.
    let try_solve = |delta_c: f64, delta_d_extra: f64| -> Option<Vec<f64>> {
        let matrix = KktMatrix::Sparse(build_ls_augmented_matrix(
            jac_rows, jac_cols, jac_vals, n, m, inequality_flags.as_deref(),
            delta_c, delta_d_extra,
        ));
        // feral's default config (issue #2: bk.pivot_threshold = 1e-8 +
        // ScalingStrategy::InfNorm) already matches MA27's `cntl[1]` /
        // Ipopt's `ma27_pivtol`, so the LS init solve uses the same factory
        // as the rest of the IPM. The `δ_c0 = 1e-8` retry below is the
        // Ipopt-style perturbation, separate from the BK threshold.
        let mut solver = new_sparse_solver();
        if solver.factor(&matrix).is_err() {
            return None;
        }
        let mut sol = vec![0.0_f64; n + m];
        if solver.solve(&rhs, &mut sol).is_err() {
            return None;
        }
        if sol.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(sol)
    };
    let sol = match try_solve(0.0, 0.0) {
        Some(s) => s,
        None => match try_solve(1e-8, 1e-8) {
            Some(s) => s,
            None => return None,
        },
    };

    // Ipopt 3.14 `IpLeastSquareMults::CalculateMultipliers` does NOT
    // post-process the y solution to enforce sign conventions per
    // bound side; it returns whatever the augmented-system solve
    // produces. Earlier ripopt added a `fix_inequality_mult_signs`
    // pass that zeroed y on rows where the LS sign disagreed with
    // the bound side — this discards information from a near-feasible
    // primal estimate and biases the next Newton step toward the
    // wrong sign convention. Removed for alignment.
    let y_ls: Vec<f64> = sol[n..].to_vec();
    Some(y_ls)
}

/// Mirrors Ipopt's `ComputeOptimalityErrorScaling`
/// (IpIpoptCalculatedQuantities.cpp:3663-3700):
///   factor = max(s_max, sum / count) / s_max
/// with s_max = 100. The factor is clamped from below to 1 (so scaling
/// never *amplifies* a residual) but has no upper cap — large multiplier
/// magnitudes are trusted to mean the problem genuinely has loose
/// tolerances. Used for both s_d (with all multipliers / m+2n) and s_c
/// (with only bound multipliers / 2n).
fn compute_residual_scaling(sum: f64, count: usize) -> f64 {
    let s_max: f64 = 100.0;
    if count > 0 {
        s_max.max(sum / count as f64) / s_max
    } else {
        1.0
    }
}

/// Dual residual scaling s_d evaluated at the current iterate, using
/// the full multiplier sum (y, z_l, z_u) and the finite-bound multiplier
/// count (Ipopt `IpIpoptCalculatedQuantities.cpp:3689-3690`).
fn compute_s_d_at_state(state: &SolverState) -> f64 {
    compute_residual_scaling(compute_multiplier_sum(state), compute_multiplier_count(state))
}

/// Slack-coupled constraint violation θ evaluated at trial `(g, s)`
/// against the current state's `g_l`/`g_u` bounds.
///
/// Mirrors Ipopt's `IpCq::curr_constraint_violation` =
/// `||c||_1 + ||d − s||_1`
/// (`IpIpoptCalculatedQuantities.cpp:1468-1473, 2570-2610`):
/// - equality row (`g_l == g_u`): contributes `|g[i] − g_l[i]|`
/// - inequality row: contributes `|g[i] − s[i]|`
///
/// **Why slack-coupled and not box-violation**: the IPM iterates an
/// explicit slack `s` for inequality rows, and the Newton system
/// drives the residual `d(x) − s` to zero (not `g(x)` to `[g_l,
/// g_u]`). The filter line search must measure the same residual
/// the step is solving — see `docs/A8_FOLLOWUP_arki0003.md` §A8.19
/// for why the prior box-violation flavour made the h-type filter
/// test artificially permissive at high theta.
///
/// Phase 3d: `s` and `ds` are now d-block-native. Equality rows are
/// elided from the slack storage entirely, so the slack-coupled theta
/// reduces to `||c||_1 + ||d − s||_1` directly without sentinel
/// reconstruction.
fn theta_for_split_d_s(state: &SolverState, c_x: &[f64], d_x: &[f64], s_d: &[f64]) -> f64 {
    debug_assert_eq!(c_x.len(), state.layout.n_c);
    debug_assert_eq!(d_x.len(), state.layout.n_d);
    debug_assert_eq!(s_d.len(), state.layout.n_d);
    convergence::primal_infeasibility_internal_split(c_x, d_x, s_d)
}

/// Compute split (c, d) views from a fresh m-form `g_trial` so theta /
/// barrier helpers that already accept native split inputs can be
/// reached without round-tripping through `state.g`. Used by the
/// line-search / SOC / soft-resto trial paths whose user-side
/// `problem.constraints` callback emits combined m-form.
fn split_from_g(state: &SolverState, g: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let c_trial: Vec<f64> = state
        .layout
        .c_to_combined
        .iter()
        .enumerate()
        .map(|(k, &i)| g[i] - state.c_rhs[k])
        .collect();
    let d_trial: Vec<f64> = state.layout.project_d(g);
    (c_trial, d_trial)
}

/// Reconstruct the user-facing combined-m-form Jacobian (rows, cols, vals)
/// from the split storage. Used at the boundary between the IPM core
/// (which uses split storage natively) and helpers that take a single
/// combined triplet — restoration NLP, post-step recovery probes, the
/// LS-y multiplier estimate. The output order interleaves c-rows then
/// d-rows: a structurally distinct layout from the user's original
/// triplet, but valid for any consumer that only needs `(rows, cols,
/// vals)` to apply `J^T y` or solve a least-squares system.
fn rebuild_combined_jac(state: &SolverState) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let nnz = state.jac_c_rows.len() + state.jac_d_rows.len();
    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);
    for (k, &kc) in state.jac_c_rows.iter().enumerate() {
        rows.push(state.layout.c_to_combined[kc]);
        cols.push(state.jac_c_cols[k]);
        vals.push(state.jac_c_vals[k]);
    }
    for (k, &kd) in state.jac_d_rows.iter().enumerate() {
        rows.push(state.layout.d_to_combined[kd]);
        cols.push(state.jac_d_cols[k]);
        vals.push(state.jac_d_vals[k]);
    }
    (rows, cols, vals)
}

/// Compute the trial slack `s + α·ds` for the line-search trial (size n_d).
/// Frac-to-bound on `s` is enforced upstream by `compute_alpha_max`
/// (`ipm.rs:3303-3414`), so `s_trial` stays in `(d_L, d_U)` for any
/// `α ≤ alpha_primal_max`. Phase 3d: returns d-form Vec.
fn compute_trial_slack(state: &SolverState, alpha: f64) -> Vec<f64> {
    let n_d = state.layout.n_d;
    let mut s_trial = Vec::with_capacity(n_d);
    for k in 0..n_d {
        s_trial.push(state.s[k] + alpha * state.ds[k]);
    }
    s_trial
}

/// Accumulate `J^T * y` (constraint Jacobian transpose times the
/// equality multipliers) into `target`. Used to assemble several
/// related dual residuals: ∇_x L for the snapshot/barrier-error
/// computations, the active-set z recovery, and the gradient-of-f +
/// J^T y diagnostic used by stall classification.
fn accumulate_jt_y(state: &SolverState, target: &mut [f64]) {
    // Phase 5d: split-form J^T·y, no combined materialisation.
    for (idx, (&kc, &col)) in state.jac_c_rows.iter().zip(state.jac_c_cols.iter()).enumerate() {
        target[col] += state.jac_c_vals[idx] * state.y_c[kc];
    }
    for (idx, (&kd, &col)) in state.jac_d_rows.iter().zip(state.jac_d_cols.iter()).enumerate() {
        target[col] += state.jac_d_vals[idx] * state.y_d[kd];
    }
}

/// Active-set z recovery from `gj = grad_f + J^T y`. Sets
/// `z_l[i] = gj[i]` (or `z_u[i] = -gj[i]`) only when the
/// corresponding bound is finite *and* the resulting product
/// `z * slack` lies within Ipopt's `kappa_d * mu` complementarity
/// envelope (κ_d = 1e10). Used by the post-step optimistic
/// dual-infeasibility probe in track_post_step_acceptable and the
/// stall-classification probe in detect_optimal_at_stall, both of
/// which test "would these multipliers satisfy the KKT
/// complementarity gate?".
fn recover_active_set_z(state: &SolverState, gj: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut zl = vec![0.0; n];
    let mut zu = vec![0.0; n];
    let kc = 1e10;
    for i in 0..n {
        if gj[i] > 0.0 && state.x_l_at(i).is_finite() {
            if gj[i] * slack_xl(state, i) <= kc * state.mu.max(1e-20) {
                zl[i] = gj[i];
            }
        } else if gj[i] < 0.0 && state.x_u_at(i).is_finite() {
            if (-gj[i]) * slack_xu(state, i) <= kc * state.mu.max(1e-20) {
                zu[i] = -gj[i];
            }
        }
    }
    (zl, zu)
}

/// Fraction-to-boundary `tau` factor used by the main step and the
/// Gondzio multiple-centrality corrections. Free mode uses
/// `1 - E_mu` where `E_mu` is Ipopt's unified scaled KKT-error
/// `max(dual_inf/s_d, primal_inf, compl_inf/s_c)`
/// (`IpIpoptCalculatedQuantities::curr_nlp_error`,
/// `IpAdaptiveMuUpdate.cpp:397`); fixed mode uses
/// `1 - mu` (Ipopt's standard `IpAlgorithmRegOp::tau_min`). Both are
/// floored at `options.tau_min` so the primal/dual fraction-to-
/// boundary scan stays strictly positive.
fn compute_tau(
    state: &SolverState,
    options: &SolverOptions,
    mu_state: &MuState,
    primal_inf: f64,
    dual_inf: f64,
    compl_inf: f64,
) -> f64 {
    if mu_state.mode == MuMode::Free {
        let e_mu = compute_e_mu(state, primal_inf, dual_inf, compl_inf);
        (1.0 - e_mu).max(options.tau_min)
    } else {
        (1.0 - state.mu).max(options.tau_min)
    }
}

/// Unified scaled KKT-error `E_mu` matching Ipopt's
/// `IpoptCalculatedQuantities::curr_nlp_error`
/// (`IpIpoptCalculatedQuantities.cpp:3050-3104`):
///
/// ```text
/// E_mu = max( dual_inf / s_d , primal_inf , compl_inf / s_c )
/// ```
///
/// where `s_d = max(s_max, sum|y, z_l, z_u| / N_d) / s_max` and
/// `s_c = max(s_max, sum|z_l, z_u| / N_c) / s_max` with `s_max = 100`.
/// `N_d` / `N_c` are the finite-bound counts (matching Ipopt's
/// `IpIpoptCalculatedQuantities.cpp:3050-3104`, where the denominators
/// are the active multiplier counts, not structural `m+2n` / `2n`).
/// Used by the Free-mode τ formula so it tracks the same scaled error
/// the convergence test uses.
fn compute_e_mu(state: &SolverState, primal_inf: f64, dual_inf: f64, compl_inf: f64) -> f64 {
    let s_d = compute_residual_scaling(compute_multiplier_sum(state), compute_multiplier_count(state));
    let s_c = compute_residual_scaling(compute_bound_multiplier_sum(state), compute_bound_multiplier_count(state));
    (dual_inf / s_d).max(primal_inf).max(compl_inf / s_c)
}

/// Fraction-to-boundary cap on the dual step `α·(dz_l, dz_u)` against
/// the bound multipliers `state.z_l` / `state.z_u`. Returns
/// `min(α_zl, α_zu)` (no `[0, 1]` clamp). Centralises the three sites
/// — main-step `α_dual_max`, Gondzio MCC corrector cap, and Mehrotra
/// affine-predictor cap — that all spell out two
/// `filter::fraction_to_boundary` calls plus a `.min` inline.
fn fraction_to_boundary_dual_z_min(state: &SolverState, dz_l: &[f64], dz_u: &[f64], tau: f64) -> f64 {
    // Phase 6d.4: caller passes compressed dz_l/dz_u (length n_x_l/
    // n_x_u). Direct index k on both z_*_compressed and the input
    // direction; no map indirection needed.
    debug_assert_eq!(dz_l.len(), state.bound_layout.n_x_l);
    debug_assert_eq!(dz_u.len(), state.bound_layout.n_x_u);
    let mut alpha = 1.0_f64;
    for k in 0..state.bound_layout.n_x_l {
        let dsi = dz_l[k];
        if dsi < 0.0 {
            let r = -tau * state.z_l_compressed[k] / dsi;
            if r < alpha {
                alpha = r;
            }
        }
    }
    for k in 0..state.bound_layout.n_x_u {
        let dsi = dz_u[k];
        if dsi < 0.0 {
            let r = -tau * state.z_u_compressed[k] / dsi;
            if r < alpha {
                alpha = r;
            }
        }
    }
    alpha.clamp(0.0, 1.0)
}

/// Fraction-to-boundary cap on `α·dv_L`, `α·dv_U` against the slack-bound
/// multipliers `state.v_l` / `state.v_u`. Returns the minimum across both
/// blocks; together with `fraction_to_boundary_dual_z_min` this gives the
/// full dual-side α_max (Ipopt computes a single α_dual across z_L, z_U,
/// v_L, v_U via `IpFilterLSAcceptor::ComputeAlphaForY` ↔
/// `IpIpoptCalculatedQuantities::CalcFracToBound`).
fn fraction_to_boundary_dual_v_min(state: &SolverState, dv_l: &[f64], dv_u: &[f64], tau: f64) -> f64 {
    // Phase 8c.5: walk compressed slack-multiplier storage. Project
    // the input dv_l/dv_u (length n_d) onto the same compressed
    // index space so v[k]+α·dv[k] is checked only on d-rows with the
    // corresponding finite slack bound. Unbounded sides have v=0
    // (no boundary to hit) and are skipped by construction.
    let dv_l_compressed = state.d_bound_layout.project_l(dv_l);
    let dv_u_compressed = state.d_bound_layout.project_u(dv_u);
    filter::fraction_to_boundary(&state.v_l_compressed, &dv_l_compressed, tau)
        .min(filter::fraction_to_boundary(&state.v_u_compressed, &dv_u_compressed, tau))
}

/// Ipopt's `CalculateSafeSlack` (IpIpoptCalculatedQuantities.cpp:455-537).
/// When a primal/slack-bound slack drops below `s_min = eps * min(1, mu)`
/// it is replaced for downstream computations with
/// `min(max(mu/multiplier, s_min), slack_move*max(1,|bound|) + max(0, slack))`,
/// where `slack_move = eps^0.75 ≈ 1.83e-12` (Ipopt option default
/// `IpIpoptCalculatedQuantities.cpp:163-173`). This safeguard prevents
/// frac-to-boundary from collapsing α to ~machine-precision when an
/// iterate has been pushed against a bound to within ε.
fn safe_slack(slack: f64, mu: f64, multiplier: f64, bound: f64) -> f64 {
    let eps = f64::EPSILON;
    let s_min = {
        let s = eps * mu.min(1.0);
        if s == 0.0 { f64::MIN_POSITIVE } else { s }
    };
    if slack >= s_min {
        return slack;
    }
    let slack_move = eps.powf(0.75); // ≈ 1.83e-12
    let cand = if multiplier > 0.0 {
        (mu / multiplier).max(s_min)
    } else {
        s_min
    };
    let cap = slack_move * bound.abs().max(1.0) + slack.max(0.0);
    cand.min(cap)
}

/// Fraction-to-boundary cap on the primal step `α·dx` against the
/// variable bounds, ignoring the `[0, 1]` clamp. The Mehrotra
/// affine-predictor and the L-BFGS gradient-descent fallback both use
/// this same per-component scan; centralising it keeps the three
/// step-controllers (main step, multiple-centrality corrections,
/// second-order correction) in lockstep.
///
/// Slack values are filtered through `safe_slack` (Ipopt's
/// `CalculateSafeSlack`, `IpIpoptCalculatedQuantities.cpp:455-537`)
/// before the FTB ratio is taken, so degenerate iterates with
/// near-zero slacks behave the same as Ipopt.
fn fraction_to_boundary_primal_x(state: &SolverState, dx: &[f64], tau: f64) -> f64 {
    let mut alpha = 1.0_f64;
    let mu = state.mu;
    for i in 0..state.n {
        if state.x_l_at(i).is_finite() && dx[i] < 0.0 {
            let raw_slack = state.x[i] - state.x_l_at(i);
            let z = state.bound_layout.full_to_x_l[i]
                .map(|k| state.z_l_compressed[k])
                .unwrap_or(0.0);
            let slack = safe_slack(raw_slack, mu, z, state.x_l_at(i));
            alpha = alpha.min(-tau * slack / dx[i]);
        }
        if state.x_u_at(i).is_finite() && dx[i] > 0.0 {
            let raw_slack = state.x_u_at(i) - state.x[i];
            let z = state.bound_layout.full_to_x_u[i]
                .map(|k| state.z_u_compressed[k])
                .unwrap_or(0.0);
            let slack = safe_slack(raw_slack, mu, z, state.x_u_at(i));
            alpha = alpha.min(tau * slack / dx[i]);
        }
    }
    alpha
}

/// Fraction-to-boundary cap on `α·ds` against the slack iterate `s`
/// (Ipopt's `IpIpoptCalculatedQuantities::CalcFracToBound` for the slack
/// block). Equality rows are skipped (their `s` is held at the equality
/// value as a sentinel and `ds` is forced to 0 by `recover_ds`).
///
/// One-sided inequalities use the same one-sided FTB pattern as `x` against
/// its variable bounds: the open side of the bound never produces an `α`
/// limiter even when `ds` points "away" from the finite side.
///
/// Slack values are filtered through `safe_slack` (Ipopt's
/// `CalculateSafeSlack`) before the FTB ratio is taken.
fn fraction_to_boundary_primal_s(state: &SolverState, ds: &[f64], tau: f64) -> f64 {
    debug_assert_eq!(ds.len(), state.layout.n_d);
    let mut alpha = 1.0_f64;
    let mu = state.mu;
    for (k, &i) in state.layout.d_to_combined.iter().enumerate() {
        let l_fin = state.g_l_at(i).is_finite();
        let u_fin = state.g_u_at(i).is_finite();
        if l_fin && ds[k] < 0.0 {
            let raw_slack = state.s[k] - state.g_l_at(i);
            let v = state.d_bound_layout.full_to_d_l[k]
                .map(|kk| state.v_l_compressed[kk])
                .unwrap_or(0.0);
            let slack = safe_slack(raw_slack, mu, v, state.g_l_at(i));
            alpha = alpha.min(-tau * slack / ds[k]);
        }
        if u_fin && ds[k] > 0.0 {
            let raw_slack = state.g_u_at(i) - state.s[k];
            let v = state.d_bound_layout.full_to_d_u[k]
                .map(|kk| state.v_u_compressed[kk])
                .unwrap_or(0.0);
            let slack = safe_slack(raw_slack, mu, v, state.g_u_at(i));
            alpha = alpha.min(tau * slack / ds[k]);
        }
    }
    alpha
}

/// Install the four step components into `state`. Used after the main
/// search-direction solve, after a gradient-descent fallback, and
/// after a Gondzio multiple-centrality correction is accepted.
fn install_step_directions(
    state: &mut SolverState,
    dx: Vec<f64>,
    dy_c: Vec<f64>,
    dy_d: Vec<f64>,
    ds: Vec<f64>,
    dz_l: Vec<f64>,
    dz_u: Vec<f64>,
    dv_l: Vec<f64>,
    dv_u: Vec<f64>,
) {
    debug_assert_eq!(dy_c.len(), state.layout.n_c);
    debug_assert_eq!(dy_d.len(), state.layout.n_d);
    debug_assert_eq!(ds.len(), state.layout.n_d);
    debug_assert_eq!(dv_l.len(), state.layout.n_d);
    debug_assert_eq!(dv_u.len(), state.layout.n_d);
    state.dx = dx;
    state.dy_c = dy_c;
    state.dy_d = dy_d;
    state.ds = ds;
    // Phase 6d.6: project full-`n` dz_l/dz_u from the direction recovery
    // into compressed storage.
    state.dz_l_compressed = state.bound_layout.project_l(&dz_l);
    state.dz_u_compressed = state.bound_layout.project_u(&dz_u);
    // Phase 8d: project the n_d-length dv_l/dv_u directly into the
    // compressed mirrors; the combined storage was dropped.
    state.dv_l_compressed = state.d_bound_layout.project_l(&dv_l);
    state.dv_u_compressed = state.d_bound_layout.project_u(&dv_u);
    // RIPOPT_DX_PROBE=iter,var: dump dx[var], dz_l/u[var], slacks, and
    // norms after the main step is installed. Used to compare KKT
    // direction precision across linear-solver backends (feral vs rmumps)
    // — see docs/A8_FOLLOWUP_arki0003.md "2026-05-04 trace" follow-up.
    if let Ok(spec) = std::env::var("RIPOPT_DX_PROBE") {
        let mut parts = spec.split(',');
        let want_iter: usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
        let want_var: usize  = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        if state.iter == want_iter && want_var < state.dx.len() {
            let xv = state.x[want_var];
            let xl = state.x_l_at(want_var);
            let xu = state.x_u_at(want_var);
            let sl = if xl.is_finite() { xv - xl } else { f64::INFINITY };
            let su = if xu.is_finite() { xu - xv } else { f64::INFINITY };
            let dz_l_v = state.bound_layout.full_to_x_l[want_var]
                .map(|k| state.dz_l_compressed[k]).unwrap_or(0.0);
            let dz_u_v = state.bound_layout.full_to_x_u[want_var]
                .map(|k| state.dz_u_compressed[k]).unwrap_or(0.0);
            let z_l_v = state.bound_layout.full_to_x_l[want_var]
                .map(|k| state.z_l_compressed[k]).unwrap_or(0.0);
            let z_u_v = state.bound_layout.full_to_x_u[want_var]
                .map(|k| state.z_u_compressed[k]).unwrap_or(0.0);
            let dx_inf = state.dx.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let dyc_inf = state.dy_c.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let dyd_inf = state.dy_d.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            eprintln!(
                "[dx-probe] iter={} mu={:.3e} var={} x={:.6e} sl={:.3e} su={:.3e} \
                 dx={:+.6e} z_L={:.3e} z_U={:.3e} dz_L={:+.3e} dz_U={:+.3e} \
                 |dx|_inf={:.3e} |dy_c|_inf={:.3e} |dy_d|_inf={:.3e}",
                state.iter, state.mu, want_var, xv, sl, su,
                state.dx[want_var], z_l_v, z_u_v, dz_l_v, dz_u_v,
                dx_inf, dyc_inf, dyd_inf,
            );
        }
    }
}

/// L-infinity norm of `J^T * c_violation`, where `c_violation` is the
/// signed constraint residual (g - g_l for equalities or below-lower
/// violations, g - g_u for above-upper violations, 0 otherwise). Used
/// by infeasibility-classification heuristics: when ‖∇θ‖_∞ ≈ 0 with
/// θ > 0 the iterate is a stationary point of the feasibility merit
/// function, so the problem is locally infeasible.
fn compute_grad_theta_norm(state: &SolverState) -> f64 {
    // Phase 5e: split-form theta gradient.
    // For c-block: violation[k_c] = c_x[k_c] (= g[i] - c_rhs[k] for
    // equality row i, since c_rhs = g_l = g_u for equalities).
    // For d-block: violation[k_d] = clipped d_x against d_l/d_u.
    let n = state.n;
    let n_c = state.layout.n_c;
    let n_d = state.layout.n_d;
    let mut grad_theta = vec![0.0; n];
    for (idx, (&kc, &col)) in state.jac_c_rows.iter().zip(state.jac_c_cols.iter()).enumerate() {
        let _ = n_c;
        grad_theta[col] += state.jac_c_vals[idx] * state.c_x[kc];
    }
    for (idx, (&kd, &col)) in state.jac_d_rows.iter().zip(state.jac_d_cols.iter()).enumerate() {
        let _ = n_d;
        let dl = state.d_l_at(kd);
        let du = state.d_u_at(kd);
        let dx = state.d_x[kd];
        let v = if dl.is_finite() && dx < dl {
            dx - dl
        } else if du.is_finite() && dx > du {
            dx - du
        } else {
            0.0
        };
        grad_theta[col] += state.jac_d_vals[idx] * v;
    }
    linf_norm(&grad_theta)
}

/// `convergence::dual_infeasibility` at the current iterate using
/// `state.{grad_f, jac_*, y, z_l, z_u}`. This combination of args
/// appears at over a dozen call sites; the helper makes them all
/// uniform. T3.9: kappa_d damping term applied via `state.kappa_d`.
///
/// B7: now also folds in the slack-side residual
/// `||−y_d − v_L + v_U||_∞` for inequality rows so the dual_inf
/// reflects the full IteratesVector dual gradient (Ipopt
/// `IpIpoptCalculatedQuantities::curr_dual_infeasibility` over both
/// x and s blocks).
fn compute_dual_inf_at_state(state: &SolverState) -> f64 {
    // A8.10 / DEV-1: Ipopt's `curr_dual_infeasibility`
    // (`IpIpoptCalculatedQuantities.cpp:2682-2691`) calls the *plain*
    // `curr_grad_lag_x()` / `curr_grad_lag_s()` (lines 1993-2030,
    // 2069-2098) — no κ_d damping. The damped variants
    // (`curr_grad_lag_with_damping_x/s`, lines 2131-2227) are used
    // *only* in the augmented-system RHS (`curr_grad_barrier_obj_x`).
    // Phase 6d.5: materialize compressed z for the convergence helper.
    let z_l_full = state.z_l_combined();
    let z_u_full = state.z_u_combined();
    let x_part = convergence::dual_infeasibility_split(
        &state.grad_f,
        &state.jac_c_rows, &state.jac_c_cols, &state.jac_c_vals, &state.y_c,
        &state.jac_d_rows, &state.jac_d_cols, &state.jac_d_vals, &state.y_d,
        &z_l_full, &z_u_full, state.n,
    );
    x_part.max(slack_dual_inf_max(state))
}

/// Slack-side dual residual `||grad_lag_s||_∞` where
/// `grad_lag_s[i] = −y[i] − v_L[i] + v_U[i]` for inequality rows.
/// Equality rows are skipped (Ipopt has no s for c-rows). The v_L/v_U
/// slots are zero on rows with infinite g_l/g_u so the sum is implicitly
/// projected onto active inequality bounds.
///
/// A8.10: Ipopt's `curr_dual_infeasibility`
/// (`IpIpoptCalculatedQuantities.cpp:2682-2691`) uses the *plain*
/// `curr_grad_lag_s()` — no κ_d damping. The damped variant
/// `curr_grad_lag_with_damping_s` is used only for the augmented-system
/// RHS, not the printed inf_du or convergence test. Pass no damping here.
fn slack_dual_inf_max(state: &SolverState) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..state.m {
        if constraint_is_equality(state, i) {
            continue;
        }
        let r = -state.y_at(i) - state.v_l_at(i) + state.v_u_at(i);
        let a = r.abs();
        if a > m {
            m = a;
        }
    }
    m
}

/// `convergence::dual_infeasibility` at the current iterate using
/// caller-supplied `z_l`/`z_u` (typically the active-set z recovered
/// by [`recover_active_set_z`] for an optimistic optimality probe)
/// instead of `state.z_l`/`state.z_u`.
fn dual_inf_with_z(state: &SolverState, z_l: &[f64], z_u: &[f64]) -> f64 {
    // A8.10 / DEV-1: plain `curr_dual_infeasibility` — no κ_d damping
    // (`IpIpoptCalculatedQuantities.cpp:2682-2691`).
    convergence::dual_infeasibility_split(
        &state.grad_f,
        &state.jac_c_rows, &state.jac_c_cols, &state.jac_c_vals, &state.y_c,
        &state.jac_d_rows, &state.jac_d_cols, &state.jac_d_vals, &state.y_d,
        z_l, z_u, state.n,
    )
}

/// L-infinity primal infeasibility at the current iterate against
/// the current state's `g_l`/`g_u` bounds. Centralises the five
/// state-arg call sites of `convergence::primal_infeasibility_max`.
fn compute_primal_inf_max_at_state(state: &SolverState) -> f64 {
    convergence::primal_infeasibility_max_split(
        &state.g_c(),
        &state.g_d(),
        &state.d_l(),
        &state.d_u(),
    )
}

/// L-infinity slack-coupling residual at the current iterate
/// (`||c||_∞ ∪ ||d − s||_∞`). Used by the scaled (barrier-level)
/// convergence test. Mirrors Ipopt's
/// `IpIpoptCalculatedQuantities::curr_primal_infeasibility(NORM_MAX)`.
fn compute_primal_inf_internal_max_at_state(state: &SolverState) -> f64 {
    convergence::primal_infeasibility_internal_max_split(
        &state.g_c(),
        &state.g_d(),
        &state.s_d(),
    )
}

/// `convergence::dual_infeasibility_scaled` at the current iterate
/// using `state.{grad_f, jac_*, y, z_l, z_u}`. The unscaled (s_d=1)
/// dual residual is what the optimality measures, the post-step
/// diagnostic, the stall classifier, and the MaxIter diagnostic all
/// compare against `options.dual_inf_tol`.
///
/// B7: also folds in `slack_dual_inf_max` so the unscaled view of dual
/// feasibility sees the slack-side residual too.
fn compute_dual_inf_unscaled_at_state(state: &SolverState) -> f64 {
    // A8.10 / DEV-1: plain `curr_dual_infeasibility` — no κ_d damping
    // (`IpIpoptCalculatedQuantities.cpp:2682-2691`).
    // Phase 5c: `dual_infeasibility_scaled` was bit-equivalent to
    // `dual_infeasibility` (same Lagrangian gradient, no per-component
    // divisor — see T3.1). Routes through the split form too.
    // Phase 6d.5: materialize compressed z for the convergence helper.
    //
    // Ipopt's `OrigIpoptNLP::unscaled_curr_dual_infeasibility` divides
    // the NLP-scaled ∇L by `obj_scaling` (df) so the unscaled gate
    // and final summary are reported in the user's NLP space. Without
    // this division `dual_inf_unscaled` was bit-equivalent to
    // `dual_inf` and the convergence test compared scaled-space ∇L
    // against the user-provided `dual_inf_tol`. The cfg(test) helper
    // `compute_convergence_info_from_state` already does this division
    // (verified by `test_convergence_info_dual_inf_unscaled_with_obj_scaling`);
    // the production path was missing it.
    let z_l_full = state.z_l_combined();
    let z_u_full = state.z_u_combined();
    let x_part = convergence::dual_infeasibility_split(
        &state.grad_f,
        &state.jac_c_rows, &state.jac_c_cols, &state.jac_c_vals, &state.y_c,
        &state.jac_d_rows, &state.jac_d_cols, &state.jac_d_vals, &state.y_d,
        &z_l_full, &z_u_full, state.n,
    );
    let scaled = x_part.max(slack_dual_inf_max(state));
    if state.obj_scaling != 1.0 && state.obj_scaling != 0.0 {
        scaled / state.obj_scaling
    } else {
        scaled
    }
}

/// `convergence::complementarity_error` at the current iterate using
/// caller-supplied `z_l`/`z_u` (typically the active-set z recovered
/// by [`recover_active_set_z`] for an optimistic optimality probe)
/// instead of `state.z_l`/`state.z_u`. Always evaluated with `μ = 0`
/// (the optimality complementarity rather than the centered-path one).
fn compl_err_with_z(state: &SolverState, z_l: &[f64], z_u: &[f64]) -> f64 {
    let x_l_full = state.x_l_combined();
    let x_u_full = state.x_u_combined();
    convergence::complementarity_error(
        &state.x, &x_l_full, &x_u_full, z_l, z_u, 0.0,
    )
}

/// Full complementarity error at the current iterate including both
/// variable-bound blocks `(x − x_L)·z_L`, `(x_U − x)·z_U` and the
/// constraint-slack blocks `(g − g_L)·max(y, 0)`, `(g_U − g)·max(−y, 0)`,
/// each compared against `μ = 0`. Mirrors Ipopt's `curr_complementarity`,
/// which sums Asum() over all four projection blocks z_L / z_U / v_L / v_U
/// (`IpIpoptCalculatedQuantities.cpp:2467-2497`). Without the v_L / v_U
/// terms the convergence test cannot detect a stalled inequality
/// constraint where `y` and slack are both nonzero.
fn compute_compl_err_at_state(state: &SolverState) -> f64 {
    // Phase 6d.5: materialize compressed z for the convergence helper.
    let z_l_full = state.z_l_combined();
    let z_u_full = state.z_u_combined();
    let x_l_full = state.x_l_combined();
    let x_u_full = state.x_u_combined();
    convergence::complementarity_error_full_split(
        &state.x, &x_l_full, &x_u_full, &z_l_full, &z_u_full,
        &state.g_d(), &state.d_l(), &state.d_u(),
        &state.v_l_d(), &state.v_u_d(), 0.0,
    )
}

/// Average 1-norm primal-dual barrier system error E_μ used by the
/// soft restoration acceptance test. Mirrors Ipopt's
/// `curr_primal_dual_system_error(mu)` at
/// `IpIpoptCalculatedQuantities.cpp:3198-3256`:
///
///   E_μ = ||grad_lag||_1 / n_dual
///       + ||primal_inf||_1 / n_pri
///       + ||compl - μ·e||_1 / n_compl
///
/// Each component is averaged by the count of contributing entries
/// (variables, constraints, finite bounds). Soft restoration accepts a
/// trial step when E_μ(trial) ≤ 0.9999 · E_μ(current) OR the trial
/// passes the filter (`IpBacktrackingLineSearch.cpp:1187-1200`).
fn compute_pderror_e_mu(state: &SolverState, mu: f64) -> f64 {
    let n = state.n;
    let m = state.m;
    let n_dual = n.max(1) as f64;
    let n_pri = m.max(1) as f64;

    // Dual residual r_i = grad_f_i + (J^T y)_i - z_l_i + z_u_i, 1-norm.
    // Phase 5e: split-form J^T·y, no combined materialisation.
    let mut residual = vec![0.0; n];
    residual[..n].copy_from_slice(&state.grad_f[..n]);
    for (idx, (&kc, &col)) in state.jac_c_rows.iter().zip(state.jac_c_cols.iter()).enumerate() {
        residual[col] += state.jac_c_vals[idx] * state.y_c[kc];
    }
    for (idx, (&kd, &col)) in state.jac_d_rows.iter().zip(state.jac_d_cols.iter()).enumerate() {
        residual[col] += state.jac_d_vals[idx] * state.y_d[kd];
    }
    // Phase 6d.3: walk compressed bound mirrors.
    for k in 0..state.bound_layout.n_x_l {
        let i = state.bound_layout.x_l_to_full[k];
        residual[i] -= state.z_l_compressed[k];
    }
    for k in 0..state.bound_layout.n_x_u {
        let i = state.bound_layout.x_u_to_full[k];
        residual[i] += state.z_u_compressed[k];
    }
    let dual_l1: f64 = residual.iter().map(|r| r.abs()).sum();

    let primal_l1 = convergence::primal_infeasibility_split(
        &state.g_c(),
        &state.g_d(),
        &state.d_l(),
        &state.d_u(),
    );

    // Complementarity 1-norm: |slack·z - μ| over finite bounds, averaged
    // by the count of contributing entries (n_compl). When there are no
    // finite bounds, drop the term.
    let mut compl_l1 = 0.0f64;
    let mut n_compl = 0usize;
    let _ = n;
    // Phase 6d.3: walk compressed bound mirrors.
    for k in 0..state.bound_layout.n_x_l {
        let i = state.bound_layout.x_l_to_full[k];
        let slack = (state.x[i] - state.x_l_at(i)).max(0.0);
        compl_l1 += (slack * state.z_l_compressed[k] - mu).abs();
        n_compl += 1;
    }
    for k in 0..state.bound_layout.n_x_u {
        let i = state.bound_layout.x_u_to_full[k];
        let slack = (state.x_u_at(i) - state.x[i]).max(0.0);
        compl_l1 += (slack * state.z_u_compressed[k] - mu).abs();
        n_compl += 1;
    }
    let compl_term = if n_compl == 0 { 0.0 } else { compl_l1 / n_compl as f64 };

    dual_l1 / n_dual + primal_l1 / n_pri + compl_term
}

/// True iff the current objective and gradient evaluation are
/// numerically valid (finite, no NaN). Used by post-perturbation
/// recovery paths to gate a "did the new iterate evaluate cleanly"
/// check before accepting it. Mirrors Ipopt's `Eval_Error` predicate
/// on the objective + gradient pair (`IpIpoptCalculatedQuantities`).
fn obj_and_grad_finite(state: &SolverState) -> bool {
    state.obj.is_finite() && state.grad_f.iter().all(|v| v.is_finite())
}

/// True iff constraint `i` is an equality (g_l[i] = g_u[i] within
/// 1e-15). Used everywhere ripopt needs to distinguish equality rows
/// (which have a single c-residual `g[i] - g_l[i]` and zero slack
/// multipliers) from inequality rows (which have separate lower/upper
/// slacks). The 1e-15 tolerance matches Ipopt's `equality_tolerance`
/// for `c(x) = 0` vs. `c_L ≤ c(x) ≤ c_U` row classification.
fn constraint_is_equality(state: &SolverState, i: usize) -> bool {
    state.g_l_at(i).is_finite() && state.g_u_at(i).is_finite()
        && (state.g_l_at(i) - state.g_u_at(i)).abs() < 1e-15
}

/// Re-seed the bound multipliers `z_l`, `z_u` from the current
/// slacks so that the perturbed iterate satisfies the
/// complementarity condition `z * s = mu` (with a 1e-20 slack
/// floor for safety). Used by the two perturbation paths that
/// reset x and need a consistent set of bound multipliers before
/// re-evaluation: `try_early_perturbation_recovery` (uses
/// `state.mu`) and `initial_evaluate_with_recovery` (uses
/// `options.mu_init`).
fn reseed_bound_multipliers_from_mu(state: &mut SolverState, mu: f64) {
    // Phase 6d.6: walk compressed bound storage.
    for k in 0..state.bound_layout.n_x_l {
        let i = state.bound_layout.x_l_to_full[k];
        state.z_l_compressed[k] = mu / slack_xl(state, i);
    }
    for k in 0..state.bound_layout.n_x_u {
        let i = state.bound_layout.x_u_to_full[k];
        state.z_u_compressed[k] = mu / slack_xu(state, i);
    }
}

/// Clamp `arr[i]` strictly inside the open variable box at index `i`,
/// using a 1e-14 inset off finite bounds. Callers use this to project
/// trial / SOC / perturbed iterates back to the strict interior so the
/// barrier `log(s)` terms stay finite. Takes `x_l`/`x_u` directly (not
/// a `&SolverState`) so callers can pass `&mut state.x` for `arr`
/// without an aliased state borrow.
fn clamp_to_open_bounds(arr: &mut [f64], x_l: &[f64], x_u: &[f64], i: usize) {
    if x_l[i].is_finite() {
        arr[i] = arr[i].max(x_l[i] + 1e-14);
    }
    if x_u[i].is_finite() {
        arr[i] = arr[i].min(x_u[i] - 1e-14);
    }
}

/// Strictly-positive lower-bound primal slack `max(x - x_l, 1e-20)`,
/// clamped away from zero so callers can divide by it without producing
/// inf/NaN. Caller is responsible for the `x_l[i].is_finite()` guard;
/// without that guard `state.x_l_at(i)` may be -inf and the result is
/// undefined.
fn slack_xl(state: &SolverState, i: usize) -> f64 {
    (state.x[i] - state.x_l_at(i)).max(1e-20)
}

/// Strictly-positive upper-bound primal slack `max(x_u - x, 1e-20)`.
/// See [`slack_xl`] for the finite-guard contract.
fn slack_xu(state: &SolverState, i: usize) -> f64 {
    (state.x_u_at(i) - state.x[i]).max(1e-20)
}

/// Strictly-positive lower-side constraint slack `max(s - g_l, 1e-20)`.
/// Caller is responsible for the `g_l[i].is_finite()` guard (or `v_l[i] > 0`,
/// used by callers that only attached a slack-multiplier when the bound
/// was finite).
///
/// Reads `state.s` (the explicit slack iterate, Ipopt 3.14 alignment).
/// During the B2 transition phase, `state.s` is synced to `state.g` at the
/// top of every iteration (via `sync_state_s_to_g`); once B6 lands and `s`
/// is advanced via Newton step, the sync is removed.
fn slack_gl(state: &SolverState, i: usize) -> f64 {
    (state.s_at(i) - state.g_l_at(i)).max(1e-20)
}

/// Strictly-positive upper-side constraint slack `max(g_u - s, 1e-20)`.
/// See [`slack_gl`] for the finite-guard contract and the `state.s`
/// rationale.
fn slack_gu(state: &SolverState, i: usize) -> f64 {
    (state.g_u_at(i) - state.s_at(i)).max(1e-20)
}

/// Test-only helper: copy `state.g` into `state.s` so the slack iterate
/// tracks the constraint values. For equality rows (`g_l == g_u` exactly),
/// `s[i] = g_l[i]` as a sentinel.
///
/// At runtime the slack iterate is advanced by the Newton step
/// `s ← s + α_p · ds` in `commit_trial_point`; this helper exists only so
/// unit tests that mutate `state.g` directly can keep `state.s` consistent
/// with the constraint values they just installed.
#[cfg(test)]
pub(crate) fn sync_state_s_to_g(state: &mut SolverState) {
    // Phase 3d / 5f: `s` is d-block-native; mirror `d_x` directly.
    for k in 0..state.layout.n_d {
        state.s[k] = state.d_x[k];
    }
}

/// Build a trial point `x + alpha * dx`, clamped strictly inside the
/// finite bounds via `clamp_to_open_bounds`. Used by the regular line
/// search, soft-restoration acceptance, gradient-descent fallback, and
/// the second-order correction.
fn compute_clamped_trial_x(state: &SolverState, dx: &[f64], alpha: f64) -> Vec<f64> {
    let n = state.n;
    let mut x_trial = vec![0.0; n];
    let x_l_full = state.x_l_combined();
    let x_u_full = state.x_u_combined();
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        x_trial[i] = state.x[i] + alpha * dx[i];
        clamp_to_open_bounds(&mut x_trial, &x_l_full, &x_u_full, i);
    }
    x_trial
}

/// L-infinity norm of a slice (max of |v_i|, with a 0.0 floor for
/// empty input). Centralises the eleven sites that spell out
/// `v.iter().map(|x| x.abs()).fold(0.0f64, f64::max)` inline —
/// the affine-step KKT residual norm, the watchdog deflection
/// magnitude, the gradient-descent fallback's L-inf gradient
/// check, the L-BFGS y-norm probes, and the grad-theta norm
/// for infeasibility classification.
fn linf_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0f64, f64::max)
}

/// Minimum of a slice. Centralises the three sites that spell out
/// `v.iter().cloned().fold(f64::INFINITY, f64::min)` inline — the
/// theta-history lookback in classify_exhausted_restoration_attempt,
/// the theta-stall detector in the main loop, and the
/// final-infeasibility check in finalize_after_max_iter. Returns
/// `f64::INFINITY` for an empty slice.
fn slice_min(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}

/// L1 (sum-of-absolute-values) norm of a slice. Centralises the
/// seven sites that spell out
/// `v.iter().map(|x| x.abs()).sum::<f64>()` inline — the three
/// terms of compute_multiplier_sum and its near-feasibility cousin
/// at line 5700–5702, and the dual-error sum-of-absolute-values
/// in the IpoptApprox stop-criterion check.
fn l1_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).sum::<f64>()
}

/// Dot product of two equal-length slices. Centralises the two
/// sites that spell out
/// `a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()`
/// inline — the Gondzio dampening cosine numerator (state.dx · dx_c)
/// and the Mehrotra deflection cosine numerator
/// (dx_orig · dx_dir) in maybe_revert_mehrotra_deflection. Panics
/// on length mismatch only via the iterator zip; callers guard with
/// equal-length invariants.
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()
}

/// Map sentinel "infinity" values (defaults ±1e19) to ±f64::INFINITY
/// in matching lower/upper bound vectors. Centralises the two
/// callers in SolverState::new (variable bounds x_l/x_u and
/// constraint bounds g_l/g_u). Without this remapping, sentinel
/// bounds participate in the slack and complementarity calculations
/// as finite values and block convergence detection.
fn sentinel_bounds_to_infinity(
    lower: &mut [f64],
    upper: &mut [f64],
    options: &SolverOptions,
) {
    for i in 0..lower.len() {
        if lower[i] <= options.nlp_lower_bound_inf {
            lower[i] = f64::NEG_INFINITY;
        }
        if upper[i] >= options.nlp_upper_bound_inf {
            upper[i] = f64::INFINITY;
        }
    }
}

/// `kkt::compute_sigma_compressed` at the current iterate's
/// `state.{x, x_l, x_u, z_l_compressed, z_u_compressed}`. Centralises
/// the two callers — assemble_kkt_systems (main solve) and the
/// perturbation recovery path in try_early_perturbation_recovery.
/// Phase 6c.3: routes through the compressed mirror via BoundLayout.
fn compute_sigma_from_state(state: &SolverState) -> Vec<f64> {
    // Phase 7c: materialize compressed x bounds for kkt::compute_sigma_compressed
    // (still consumes full-`n` x_l/x_u slices).
    let x_l_full = state.x_l_combined();
    let x_u_full = state.x_u_combined();
    kkt::compute_sigma_compressed(
        &state.x,
        &x_l_full,
        &x_u_full,
        &state.z_l_compressed,
        &state.z_u_compressed,
        &state.bound_layout,
    )
}

/// `kkt::recover_dz` (Fiacco bound-multiplier step recovery) at the
/// current iterate's `state.{x, x_l, x_u, z_l, z_u}` for a given
/// primal direction `dx` and centering target `mu`. Centralises the
/// two callers — the main-step recovery in solve_for_search_direction
/// and the Mehrotra affine-predictor probe.
fn recover_dz_from_state(state: &SolverState, dx: &[f64], mu: f64) -> (Vec<f64>, Vec<f64>) {
    // Phase 6d.5: materialize compressed z for kkt::recover_dz (still
    // indexes z by full var idx).
    // Phase 7c: same for compressed x_l/x_u bounds.
    let z_l_full = state.z_l_combined();
    let z_u_full = state.z_u_combined();
    let x_l_full = state.x_l_combined();
    let x_u_full = state.x_u_combined();
    kkt::recover_dz(
        &state.x, &x_l_full, &x_u_full,
        &z_l_full, &z_u_full, dx, mu,
    )
}

/// Recover slack-bound multiplier steps `dv_L`, `dv_U` from the current
/// iterate's `state.{g, g_l, g_u, v_l, v_u}` and Jacobian, for a given
/// primal direction `dx` and centering target `mu`.
fn recover_dv_from_state(state: &SolverState, ds: &[f64], mu: f64) -> (Vec<f64>, Vec<f64>) {
    // Phase 3d: kkt::recover_dv still indexes s/ds in m-form; materialize
    // a combined s view from the d-block storage for the call.
    let s_combined = state.s_combined();
    let v_l_combined = state.v_l_combined();
    let v_u_combined = state.v_u_combined();
    let g_l_combined = state.g_l_combined();
    let g_u_combined = state.g_u_combined();
    kkt::recover_dv(
        state.m,
        &g_l_combined, &g_u_combined, &s_combined,
        &v_l_combined, &v_u_combined, ds, mu,
    )
}

/// Recover the slack-iterate step `ds` for the current iterate.
/// Wraps `kkt::recover_ds`, threading `ic_delta_c` as the per-row
/// `δ_d` perturbation on inequality rows (equality rows get 0).
/// In Ipopt 3.14 the (2,2) block uses a single δ_c that the
/// perturbation handler sets uniformly across c- and d-rows
/// (`IpPDPerturbationHandler.cpp`; recover step uses
/// `delta_x_s · dy` per `IpPDFullSpaceSolver.cpp::SolveOnce`).
fn recover_ds_from_state(
    state: &SolverState,
    dx: &[f64],
    dy: &[f64],
    ic_delta_c: f64,
) -> Vec<f64> {
    let delta_d_vec: Vec<f64> = if ic_delta_c == 0.0 {
        Vec::new()
    } else {
        (0..state.m)
            .map(|i| if constraint_is_equality(state, i) { 0.0 } else { ic_delta_c })
            .collect()
    };
    let s_combined = state.s_combined();
    let g_l_combined = state.g_l_combined();
    let g_u_combined = state.g_u_combined();
    let (jac_rows_m, jac_cols_m, jac_vals_m) = rebuild_combined_jac(state);
    let g_combined = state.g_combined();
    kkt::recover_ds(
        state.n, state.m,
        &jac_rows_m, &jac_cols_m, &jac_vals_m,
        &g_combined, &g_l_combined, &g_u_combined, &s_combined,
        dx, dy, &delta_d_vec,
    )
}

/// `kkt::assemble_kkt` invoked with the standard `state.*` argument
/// pattern (Hessian/Jacobian triplets, primal/dual iterate, bounds, mu,
/// dense vs. sparse choice). Centralises the five callers — main solve
/// (assemble_kkt_systems), the sparse-condensed full-KKT fallback, the
/// full-augmented direction solve, the bandwidth-driven downgrade, and
/// the perturbation recovery path.
fn assemble_kkt_from_state(
    state: &SolverState,
    n: usize,
    m: usize,
    sigma: &[f64],
    use_sparse: bool,
    kappa_d: f64,
) -> kkt::KktSystem {
    debug_assert_eq!(m, state.layout.n_c + state.layout.n_d);
    // Phase 7c: materialize compressed x bounds for kkt::assemble_kkt
    // (still consumes full-`n` x_l/x_u slices).
    let x_l_full = state.x_l_combined();
    let x_u_full = state.x_u_combined();
    // Phase 8c.5: materialize compressed v_l/v_u to length n_d
    // (kkt::assemble_kkt still indexes v by d-block index k).
    let v_l_full = state.d_bound_layout.expand_l(&state.v_l_compressed, 0.0);
    let v_u_full = state.d_bound_layout.expand_u(&state.v_u_compressed, 0.0);
    // Phase 9c.5: same for compressed d_l/d_u → length n_d.
    let d_l_full = state.d_l();
    let d_u_full = state.d_u();
    let mut sys = kkt::assemble_kkt(
        n, state.layout.n_c, state.layout.n_d,
        &state.hess_rows, &state.hess_cols, &state.hess_vals,
        &state.jac_c_rows, &state.jac_c_cols, &state.jac_c_vals,
        &state.jac_d_rows, &state.jac_d_cols, &state.jac_d_vals,
        sigma, &state.grad_f,
        &state.c_x, &state.d_x,
        &d_l_full, &d_u_full,
        &state.s, &state.y_c, &state.y_d,
        &state.x, &x_l_full, &x_u_full, state.mu, kappa_d,
        use_sparse, &v_l_full, &v_u_full, &state.layout,
    );
    // T3.25: snapshot the upstream atags. The assembled matrix is a
    // pure function of (W, J, sigma_x, sigma_s, slacks, multipliers),
    // so the atag tuple uniquely identifies it. The perturbation
    // (δ_x, δ_c) is layered on at factor time and tracked separately
    // by `FactorCache`.
    sys.input_atags = Some(state.kkt_atags);
    sys
}

/// Commit a trial point as the new iterate: writes `x`, `obj`, `g`,
/// and the primal step length. Multipliers and the barrier objective
/// are unaffected — the caller updates those separately when needed
/// (in particular, `phi` is recomputed before the next line search).
fn commit_trial_point(
    state: &mut SolverState,
    x_trial: Vec<f64>,
    obj_trial: f64,
    g_trial: Vec<f64>,
    alpha: f64,
) {
    // A8.4 step trace: env-gated diagnostic for centering-stall debugging.
    // Set RIPOPT_TRACE_STEP=1 to log ‖Δx‖, ‖α·Δx‖, |Δx_eff|, and the
    // achieved relative move ‖Δx_eff‖/‖x‖. A frozen iterate at α=1 with
    // ‖α·Δx‖_inf at machine-epsilon · ‖x‖_inf is the smoking gun for
    // a near-singular augmented system producing a near-zero direction
    // that the linear solver still reports as nonsingular.
    if std::env::var("RIPOPT_TRACE_STEP").is_ok() {
        let dx_inf = linf_norm(&state.dx);
        let mut x_inf = 0.0_f64;
        let mut diff_inf = 0.0_f64;
        for i in 0..state.n {
            let xi = state.x[i].abs();
            if xi > x_inf { x_inf = xi; }
            let d = (x_trial[i] - state.x[i]).abs();
            if d > diff_inf { diff_inf = d; }
        }
        let dy_inf = state
            .dy_c
            .iter()
            .chain(state.dy_d.iter())
            .fold(0.0f64, |a, &b| a.max(b.abs()));
        let rel = if x_inf > 0.0 { diff_inf / x_inf } else { diff_inf };
        // Σ-pin diagnostic: smallest x-slack and largest z give the worst
        // diagonal entry of W + Σ. If min_slack · max_z >> κ_σ·μ the κ_σ
        // clamp has failed to keep z*s in band.
        let mut min_s_x = f64::INFINITY;
        let mut min_s_idx = usize::MAX;
        let mut min_s_side = "";
        let mut max_z_x = 0.0_f64;
        for i in 0..state.n {
            if state.x_l_at(i).is_finite() {
                let s = state.x[i] - state.x_l_at(i);
                if s > 0.0 && s < min_s_x { min_s_x = s; min_s_idx = i; min_s_side = "L"; }
            }
            if state.x_u_at(i).is_finite() {
                let s = state.x_u_at(i) - state.x[i];
                if s > 0.0 && s < min_s_x { min_s_x = s; min_s_idx = i; min_s_side = "U"; }
            }
        }
        // Phase 6d.3: walk compressed bound mirrors for max |z|.
        for &v in &state.z_l_compressed {
            if v.abs() > max_z_x { max_z_x = v.abs(); }
        }
        for &v in &state.z_u_compressed {
            if v.abs() > max_z_x { max_z_x = v.abs(); }
        }
        let (xv, bv) = if min_s_idx < state.n {
            let xv = state.x[min_s_idx];
            let bv = if min_s_side == "L" { state.x_l_at(min_s_idx) } else { state.x_u_at(min_s_idx) };
            (xv, bv)
        } else { (f64::NAN, f64::NAN) };
        eprintln!(
            "[step] α={:.3e} ‖Δx‖={:.3e} rel={:.3e} ‖Δy‖={:.3e} min_s={:.3e}@{}{} x={:.6e} bnd={:.6e} max_z={:.3e} max_Σ≈{:.3e}",
            alpha, dx_inf, rel, dy_inf,
            min_s_x, min_s_idx, min_s_side, xv, bv, max_z_x,
            max_z_x / min_s_x.max(1e-300),
        );
    }
    state.x = x_trial;
    state.obj = obj_trial;
    let (c_trial, d_trial) = split_from_g(state, &g_trial);
    state.c_x = c_trial;
    state.d_x = d_trial;
    // Advance the slack iterate with the same primal step length.
    // Equality rows (`g_l == g_u`) have `ds = 0` (forced by `recover_ds`)
    // and `s` stays at the equality value as a sentinel. For inequality
    // rows the line-searched α_p is the same fraction of the FTB-capped
    // step taken on x, by Ipopt's `alpha_for_y = primal` default
    // (IpFilterLSAcceptor.cpp:617-628, IpIteratesVector.hpp).
    // Phase 3d: state.s and state.ds are d-block-native (size n_d), so
    // advance directly over the d-block — equality rows are not stored.
    for k in 0..state.layout.n_d {
        state.s[k] += alpha * state.ds[k];
    }
    state.alpha_primal = alpha;
    // T3.25: x and g have changed → slacks_x, slacks_s, sigma_x, sigma_s
    // are now stale. The Hessian and Jacobian also typically get
    // re-evaluated against the new x downstream of this commit, so bump
    // them too. This is a coarse-grained, conservative bump: the IPM
    // re-evaluates after every accepted step regardless.
    state.kkt_atags.slacks_x = state.kkt_atags.slacks_x.wrapping_add(1);
    state.kkt_atags.slacks_s = state.kkt_atags.slacks_s.wrapping_add(1);
    state.kkt_atags.sigma_x = state.kkt_atags.sigma_x.wrapping_add(1);
    state.kkt_atags.sigma_s = state.kkt_atags.sigma_s.wrapping_add(1);
    state.kkt_atags.w = state.kkt_atags.w.wrapping_add(1);
    state.kkt_atags.j_c = state.kkt_atags.j_c.wrapping_add(1);
    state.kkt_atags.j_d = state.kkt_atags.j_d.wrapping_add(1);
}

/// Sum of absolute values of all Lagrange multipliers in the iterate:
/// `y` (combined y_c + y_d), bound multipliers `z_l`, `z_u`, and the
/// constraint-slack multipliers `v_l`, `v_u`. Used with
/// `compute_multiplier_count` to form the dual scaling factor `s_d`
/// (`IpIpoptCalculatedQuantities.cpp:3689-3690`,
/// `y_c + y_d + z_L + z_U + v_L + v_U`).
fn compute_multiplier_sum(state: &SolverState) -> f64 {
    // Phase 6d.3 / 8c.5: walk compressed bound mirrors. l1_norm over
    // unbounded zero-padded sides was 0 anyway.
    l1_norm(&state.y_c)
        + l1_norm(&state.y_d)
        + l1_norm(&state.z_l_compressed)
        + l1_norm(&state.z_u_compressed)
        + l1_norm(&state.v_l_compressed)
        + l1_norm(&state.v_u_compressed)
}

/// Sum of absolute values of bound-side multipliers contributing to
/// `s_c`: `z_l + z_u + v_l + v_u`
/// (`IpIpoptCalculatedQuantities.cpp:3677-3687`). Used with
/// `compute_bound_multiplier_count` (finite-bound count over
/// x and inequality g rows) to form the complementarity scaling `s_c`.
fn compute_bound_multiplier_sum(state: &SolverState) -> f64 {
    // Phase 6d.3 / 8c.5: walk compressed bound mirrors. l1_norm over
    // unbounded zero-padded sides was 0 anyway.
    l1_norm(&state.z_l_compressed)
        + l1_norm(&state.z_u_compressed)
        + l1_norm(&state.v_l_compressed)
        + l1_norm(&state.v_u_compressed)
}

/// Count of finite variable bounds (z_L.Dim() + z_U.Dim() in Ipopt) plus
/// finite constraint bounds (v_L.Dim() + v_U.Dim()) used as the
/// denominator for `s_c` per `IpIpoptCalculatedQuantities.cpp:3677-3687`.
/// In ripopt's combined-y representation, each finite g_L[i] contributes
/// one unit (the v_L slot) and each finite g_U[i] contributes one
/// (the v_U slot). Equality constraints have both finite but they
/// represent a single dual, so they each still contribute their two
/// counts separately, matching Ipopt's bookkeeping where equality
/// constraints don't appear in v_L/v_U at all — only inequalities do.
fn compute_bound_multiplier_count(state: &SolverState) -> usize {
    let mut n_bound = state.bound_layout.n_x_l + state.bound_layout.n_x_u;
    for i in 0..state.m {
        if constraint_is_equality(state, i) {
            continue;
        }
        if state.g_l_at(i).is_finite() {
            n_bound += 1;
        }
        if state.g_u_at(i).is_finite() {
            n_bound += 1;
        }
    }
    n_bound
}

/// Count of all dual components contributing to `s_d` per
/// `IpIpoptCalculatedQuantities.cpp:3689-3690`:
/// y_c.Dim() + y_d.Dim() + z_L.Dim() + z_U.Dim() + v_L.Dim() + v_U.Dim().
/// The `y` count is `m` (each constraint has one combined y). The bound
/// counts come from [`compute_bound_multiplier_count`].
fn compute_multiplier_count(state: &SolverState) -> usize {
    state.m + compute_bound_multiplier_count(state)
}

/// Build a `ConvergenceInfo` from the current iterate by computing the
/// max-norm primal infeasibility, dual infeasibility, complementarity
/// error (at mu_target=0), and total multiplier sum from `state`. The
/// caller supplies `mu` because some test paths want to check convergence
/// at mu=0 even though `state.mu` is
/// nonzero.
#[cfg(test)]
fn compute_convergence_info_from_state(
    state: &SolverState,
    mu: f64,
    _n: usize,
    _m: usize,
) -> ConvergenceInfo {
    let primal_inf = compute_primal_inf_max_at_state(state);
    let primal_inf_internal = compute_primal_inf_internal_max_at_state(state);
    let dual_inf = compute_dual_inf_at_state(state);
    let compl_inf = compute_compl_err_at_state(state);
    // Ipopt's unscaled_curr_dual_infeasibility removes the obj_scaling
    // factor that was applied to ∇f and J^T y in the internal scaled
    // problem (`IpOrigIpoptNLP::unscaled_curr_dual_infeasibility`). For
    // ripopt's uniform obj scaling that reduces to a single division.
    let dual_inf_unscaled = if state.obj_scaling != 1.0 && state.obj_scaling != 0.0 {
        dual_inf / state.obj_scaling
    } else {
        dual_inf
    };
    ConvergenceInfo {
        primal_inf,
        primal_inf_internal,
        dual_inf,
        dual_inf_unscaled,
        compl_inf,
        mu,
        objective: state.obj,
        multiplier_sum: compute_multiplier_sum(state),
        multiplier_count: compute_multiplier_count(state),
        bound_multiplier_sum: compute_bound_multiplier_sum(state),
        bound_multiplier_count: compute_bound_multiplier_count(state),
        x_max_abs: linf_norm(&state.x),
    }
}

fn compute_avg_complementarity(state: &SolverState) -> f64 {
    let mut sum_compl = 0.0;
    let mut count = 0;
    // Phase 6d.3: walk compressed bound mirrors. The set
    // {i : x_{l,u}[i].is_finite()} matches `x_{l,u}_to_full[..n_x_{l,u}]`.
    for k in 0..state.bound_layout.n_x_l {
        let i = state.bound_layout.x_l_to_full[k];
        sum_compl += slack_xl(state, i) * state.z_l_compressed[k];
        count += 1;
    }
    for k in 0..state.bound_layout.n_x_u {
        let i = state.bound_layout.x_u_to_full[k];
        sum_compl += slack_xu(state, i) * state.z_u_compressed[k];
        count += 1;
    }
    // B-cross6: include slack-side complementarity v_L·s_L, v_U·s_U
    // unconditionally, matching Ipopt's `IpIpoptCalculatedQuantities::
    // curr_avrg_compl` which sums over all four bound blocks. The
    // previously-conditional fallback (only when no variable bounds
    // existed) was a ripopt-specific tuning that left `avg_compl` blind
    // to slack centrality on mixed-bound problems.
    for i in 0..state.m {
        let l_fin = state.g_l_at(i).is_finite();
        let u_fin = state.g_u_at(i).is_finite();
        if l_fin && u_fin && (state.g_l_at(i) - state.g_u_at(i)).abs() < 1e-14 {
            continue;
        }
        if l_fin {
            sum_compl += state.v_l_at(i) * slack_gl(state, i);
            count += 1;
        }
        if u_fin {
            sum_compl += state.v_u_at(i) * slack_gu(state, i);
            count += 1;
        }
    }
    if count > 0 {
        sum_compl / count as f64
    } else {
        0.0
    }
}

/// Barrier-subproblem optimality error E_μ(x, λ, z), the gate that
/// `update_barrier_parameter_fixed_mode` checks against
/// `barrier_tol_factor * mu` to decide whether the current subproblem
/// is "solved" enough to decrement μ. Mirrors Ipopt's
/// `IpoptCalculatedQuantities::curr_barrier_error()`
/// (`IpIpoptCalculatedQuantities.cpp:3148-3196`):
///
///   E_μ = max( ‖∇L‖_∞ / s_d,
///              ‖c‖_∞,
///              max_i |slack_i · z_i − μ| / s_c )
///
/// where `s_d`, `s_c` are the L1-mean multiplier scalings from
/// `ComputeOptimalityErrorScaling` (`IpIpoptCalculatedQuantities.cpp:3663-3700`,
/// `s_max = 100`). All three components use the L∞ norm; only s_d/s_c
/// internally average over multiplier counts.
///
/// The previous ripopt implementation used L1/n on the dual term and
/// the L1 mean on complementarity (instead of L∞), and added a
/// `du_floor = 0.1 * unscaled_du` heuristic that had no Ipopt analogue
/// and prevented μ from decreasing whenever NLP-level dual_inf was
/// large — exactly the dual-stagnation symptom seen on arki0003 where
/// μ froze at lg(μ)=2.83e-3 for 500+ iterations.
fn compute_barrier_error(state: &SolverState) -> f64 {
    let (e, _, _, _) = compute_barrier_error_components(state);
    e
}

/// Print the largest-magnitude complementarity outliers to stderr.
/// For diagnostics only; gated on `RIPOPT_TRACE_COMPL`.
fn dump_compl_outliers(state: &SolverState) {
    if std::env::var("RIPOPT_TRACE_COMPL").is_err() {
        return;
    }
    let mut entries: Vec<(f64, String)> = Vec::new();
    for k in 0..state.bound_layout.n_x_l {
        let i = state.bound_layout.x_l_to_full[k];
        let s = slack_xl(state, i);
        let z = state.z_l_compressed[k];
        let prod = s * z;
        entries.push((prod.abs(), format!("xL[{}] s={:.3e} z={:.3e} s*z={:.3e}", i, s, z, prod)));
    }
    for k in 0..state.bound_layout.n_x_u {
        let i = state.bound_layout.x_u_to_full[k];
        let s = slack_xu(state, i);
        let z = state.z_u_compressed[k];
        let prod = s * z;
        entries.push((prod.abs(), format!("xU[{}] s={:.3e} z={:.3e} s*z={:.3e}", i, s, z, prod)));
    }
    for i in 0..state.m {
        if state.g_l_at(i).is_finite() {
            let s = slack_gl(state, i);
            let z = state.v_l_at(i);
            let prod = s * z;
            entries.push((prod.abs(), format!("gL[{}] s={:.3e} z={:.3e} s*z={:.3e}", i, s, z, prod)));
        }
        if state.g_u_at(i).is_finite() {
            let s = slack_gu(state, i);
            let z = state.v_u_at(i);
            let prod = s * z;
            entries.push((prod.abs(), format!("gU[{}] s={:.3e} z={:.3e} s*z={:.3e}", i, s, z, prod)));
        }
    }
    entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    eprintln!("ripopt: compl-outliers iter={} mu={:.3e} (top 5 by |s*z|):", state.iter, state.mu);
    for (_, line) in entries.iter().take(5) {
        eprintln!("  {}", line);
    }
}

/// Same as `compute_barrier_error` but also returns the (dual_err,
/// compl_err, primal_err) triple. Used for diagnostics.
fn compute_barrier_error_components(state: &SolverState) -> (f64, f64, f64, f64) {
    // ∇L = ∇f + J^T y − z_l + z_u (un-damped; matches
    // `curr_grad_lag_x` at IpIpoptCalculatedQuantities.cpp:1993-2030)
    let mut grad_lag = state.grad_f.clone();
    accumulate_jt_y(state, &mut grad_lag);
    // Phase 6d.3: walk compressed bound mirrors.
    for k in 0..state.bound_layout.n_x_l {
        let i = state.bound_layout.x_l_to_full[k];
        grad_lag[i] -= state.z_l_compressed[k];
    }
    for k in 0..state.bound_layout.n_x_u {
        let i = state.bound_layout.x_u_to_full[k];
        grad_lag[i] += state.z_u_compressed[k];
    }
    let s_d = compute_s_d_at_state(state);
    let dual_err = linf_norm(&grad_lag) / s_d;

    // L∞ of the perturbed complementarity (`X·z − μ·e` per
    // IpIpoptCalculatedQuantities.cpp:2799-2871, scaled by s_c).
    let s_c = compute_residual_scaling(
        compute_bound_multiplier_sum(state),
        compute_bound_multiplier_count(state),
    );
    let mut compl_max: f64 = 0.0;
    // Phase 6d.3: walk compressed bound mirrors.
    for k in 0..state.bound_layout.n_x_l {
        let i = state.bound_layout.x_l_to_full[k];
        let r = (slack_xl(state, i) * state.z_l_compressed[k] - state.mu).abs();
        if r > compl_max {
            compl_max = r;
        }
    }
    for k in 0..state.bound_layout.n_x_u {
        let i = state.bound_layout.x_u_to_full[k];
        let r = (slack_xu(state, i) * state.z_u_compressed[k] - state.mu).abs();
        if r > compl_max {
            compl_max = r;
        }
    }
    let compl_err = compl_max / s_c;

    // Primal infeasibility ‖c‖_∞ (no s-divisor in Ipopt;
    // `curr_primal_infeasibility(NORM_MAX)` at line 2570-2610).
    // `state.constraint_violation()` returns the L1 sum used by the
    // filter; the barrier-error gate needs the L∞ version.
    let primal_err = convergence::primal_infeasibility_max_split(
        &state.g_c(),
        &state.g_d(),
        &state.d_l(),
        &state.d_u(),
    );

    let e = dual_err.max(compl_err).max(primal_err);
    (e, dual_err, compl_err, primal_err)
}


/// Build the final solve result.
/// Computes z from stationarity for more accurate output multipliers.
/// Unscales all values from the internal scaled space to the original NLP space.
/// Populate the `final_*` measures on a cloned `SolverDiagnostics`:
/// μ, ‖θ‖_∞ primal infeasibility, dual infeasibility (raw), and the
/// complementarity error at μ=0. Also computes the dual scaling
/// factor `s_d` from the multiplier sum. Same formulas as
/// `check_convergence`.
fn populate_final_diagnostics(state: &SolverState) -> SolverDiagnostics {
    let mut diag = state.diagnostics.clone();
    diag.final_mu = state.mu;
    diag.final_primal_inf = compute_primal_inf_max_at_state(state);
    diag.final_dual_inf = compute_dual_inf_at_state(state);
    diag.final_compl = compute_compl_err_at_state(state);
    diag.final_s_d = compute_s_d_at_state(state);
    // T3.25 follow-up: surface cache counters so tests can verify the
    // cached entry point fired and benchmarks can quantify hit rates.
    diag.factor_cache_hits = state.factor_cache.hits;
    diag.factor_cache_misses = state.factor_cache.misses;
    diag.factor_cache_factor_calls = state.factor_cache.factor_calls;
    diag.n_obj_evals = state.n_obj_evals;
    diag.n_grad_evals = state.n_grad_evals;
    diag.n_constr_evals = state.n_constr_evals;
    diag.n_jac_evals = state.n_jac_evals;
    diag.n_hess_evals = state.n_hess_evals;
    diag
}

/// Unscale the iterate's multipliers and constraint values for
/// reporting (Ipopt semantics):
///   - `z_unscaled = z_scaled / obj_scaling`
///   - `y_unscaled[i] = y_scaled[i] * g_scaling[i] / obj_scaling`
///   - `g_unscaled[i] = g_scaled[i] / g_scaling[i]`
/// Returns `(z_l_out, z_u_out, y_out, g_out)`.
fn unscale_solution_vectors(state: &SolverState) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = state.n;
    let m = state.m;
    // Phase 6d.1: materialize z_l/z_u from the compressed mirrors
    // (BoundLayout::expand_l/u pads unbounded sides to 0).
    let mut z_l_out = state.z_l_combined();
    let mut z_u_out = state.z_u_combined();
    for i in 0..n {
        z_l_out[i] /= state.obj_scaling;
        z_u_out[i] /= state.obj_scaling;
    }
    // Reconstruct combined-indexed y_out / g_out from the split storage
    // (Phase 3): per-row scaling lives in c_scaling[k] for equalities,
    // d_scaling[k] for inequalities.
    let mut y_out = state.y_combined();
    let mut g_out = state.g_combined();
    for i in 0..m {
        let scale_i = if let Some(k) = state.layout.eq_pos[i] {
            state.c_scaling.get(k).copied().unwrap_or(1.0)
        } else if let Some(k) = state.layout.ineq_pos[i] {
            state.d_scaling.get(k).copied().unwrap_or(1.0)
        } else {
            1.0
        };
        y_out[i] = y_out[i] * scale_i / state.obj_scaling;
        g_out[i] /= scale_i;
    }
    (z_l_out, z_u_out, y_out, g_out)
}

fn make_result(state: &SolverState, status: SolveStatus) -> SolveResult {
    let diag = populate_final_diagnostics(state);
    let (z_l_out, z_u_out, y_out, g_out) = unscale_solution_vectors(state);

    SolveResult {
        x: state.x.clone(),
        objective: state.obj / state.obj_scaling,
        constraint_multipliers: y_out,
        bound_multipliers_lower: z_l_out,
        bound_multipliers_upper: z_u_out,
        constraint_values: g_out,
        status,
        iterations: state.iter,
        diagnostics: diag,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a minimal SolverState for testing private mu/complementarity helpers.
    // The caller supplies only fields the test exercises; everything else is zeroed.
    fn minimal_state(n: usize, m: usize) -> SolverState {
        SolverState {
            x: vec![0.0; n],
            // Phase 3b: test-only constructor uses an all-inequality layout
            // (n_c = 0, n_d = m), so y_c is empty and y_d is m-sized.
            y_c: Vec::new(),
            y_d: vec![0.0; m],
            dx: vec![0.0; n],
            // Phase 3c: test-only constructor (all-inequality layout) →
            // empty dy_c, m-length dy_d.
            dy_c: Vec::new(),
            dy_d: vec![0.0; m],
            // Phase 3d: test-only constructor (all-inequality layout) → m-length s/ds.
            s: vec![0.0; m],
            ds: vec![0.0; m],
            mu: 0.1,
            alpha_primal: 0.0,
            alpha_dual: 0.0,
            iter: 0,
            // Phase 9d: combined `d_l`/`d_u` storage dropped. Test-only
            // constructor's no-bounds slack layout (n_d_l=n_d_u=0) leaves
            // the compressed mirrors empty, set below.
            c_rhs: Vec::new(),
            n,
            m,
            obj: 0.0,
            grad_f: vec![0.0; n],
            // Phase 5a: native split constraint storage. Test-only
            // constructor uses an all-inequality layout (n_c=0, n_d=m),
            // so c_x is empty and d_x mirrors g.
            c_x: Vec::new(),
            d_x: vec![0.0; m],
            jac_rows: Vec::new(),
            jac_cols: Vec::new(),
            // Phase 4a: empty split-Jacobian storage in minimal_state;
            // tests that exercise Jacobian-consuming code build their
            // own structure and call set_jac_vals_combined() (or write
            // jac_c_vals/jac_d_vals directly) as needed.
            jac_c_rows: Vec::new(),
            jac_c_cols: Vec::new(),
            jac_c_vals: Vec::new(),
            jac_d_rows: Vec::new(),
            jac_d_cols: Vec::new(),
            jac_d_vals: Vec::new(),
            jac_c_combined_idx: Vec::new(),
            jac_d_combined_idx: Vec::new(),
            hess_rows: Vec::new(),
            hess_cols: Vec::new(),
            hess_vals: Vec::new(),
            consecutive_acceptable: 0,
            obj_scaling: 1.0,
            // Test-only constructor uses an all-inequality layout (no
            // equalities) — n_c=0, n_d=m. c_scaling is empty.
            c_scaling: Vec::new(),
            d_scaling: vec![1.0; m],
            layout: crate::constraint_layout::ConstraintLayout::new(
                &vec![f64::NEG_INFINITY; m],
                &vec![f64::INFINITY; m],
            ),
            // Phase 6b: test-only constructor uses a no-bounds layout
            // (n_x_l = n_x_u = 0); compressed mirrors are empty.
            bound_layout: crate::bound_layout::BoundLayout::new(
                &vec![f64::NEG_INFINITY; n],
                &vec![f64::INFINITY; n],
            ),
            z_l_compressed: Vec::new(),
            z_u_compressed: Vec::new(),
            dz_l_compressed: Vec::new(),
            dz_u_compressed: Vec::new(),
            x_l_compressed: Vec::new(),
            x_u_compressed: Vec::new(),
            // Phase 8a/b: test-only constructor uses a no-bounds slack
            // layout (n_d_l = n_d_u = 0); compressed mirrors empty.
            d_bound_layout: crate::d_bound_layout::DBoundLayout::new(
                &vec![f64::NEG_INFINITY; m],
                &vec![f64::INFINITY; m],
            ),
            v_l_compressed: Vec::new(),
            v_u_compressed: Vec::new(),
            dv_l_compressed: Vec::new(),
            dv_u_compressed: Vec::new(),
            d_l_compressed: Vec::new(),
            d_u_compressed: Vec::new(),
            diagnostics: SolverDiagnostics::default(),
            x_last_eval: vec![f64::NAN; n],
            adjusted_slacks_count: 0,
            is_square: false,
            acceptable_iterate: None,
            last_obj_for_acceptable: None,
            kkt_atags: kkt::KktInputAtags::default(),
            factor_cache: kkt::FactorCache::new(),
            n_obj_evals: 0,
            n_grad_evals: 0,
            n_constr_evals: 0,
            n_jac_evals: 0,
            n_hess_evals: 0,
        }
    }

    /// Phase 3f test helper: replace the constraint bounds on a
    /// `minimal_state`-derived fixture and rebuild the dependent storage
    /// (layout, c_rhs/d_l/d_u, and per-layout vector sizes for y_c/y_d
    /// /v_l/v_u/dy_c/dy_d/dv_l/dv_u/s/ds/c_scaling/d_scaling).
    ///
    /// Combined-form `g_l`/`g_u` are also written for parity with the
    /// legacy `state.g_l = vec![…]` patterns the older fixtures used.
    fn set_constraint_bounds(
        state: &mut SolverState,
        g_l: Vec<f64>,
        g_u: Vec<f64>,
    ) {
        assert_eq!(g_l.len(), state.m);
        assert_eq!(g_u.len(), state.m);
        let layout = crate::constraint_layout::ConstraintLayout::new(&g_l, &g_u);
        let n_c = layout.n_c;
        let n_d = layout.n_d;
        state.c_rhs = layout.c_to_combined.iter().map(|&i| g_l[i]).collect();
        let d_l_d = layout.project_d(&g_l);
        let d_u_d = layout.project_d(&g_u);
        state.layout = layout;
        // Phase 8b: rebuild slack-bound layout + compressed v/dv mirrors.
        state.d_bound_layout =
            crate::d_bound_layout::DBoundLayout::new(&d_l_d, &d_u_d);
        state.v_l_compressed = vec![0.0; state.d_bound_layout.n_d_l];
        state.v_u_compressed = vec![0.0; state.d_bound_layout.n_d_u];
        state.dv_l_compressed = vec![0.0; state.d_bound_layout.n_d_l];
        state.dv_u_compressed = vec![0.0; state.d_bound_layout.n_d_u];
        // Phase 9b/d: compressed slack-bound storage (combined dropped).
        state.d_l_compressed = state.d_bound_layout.project_l(&d_l_d);
        state.d_u_compressed = state.d_bound_layout.project_u(&d_u_d);
        state.y_c.resize(n_c, 0.0);
        state.y_d.resize(n_d, 0.0);
        state.dy_c.resize(n_c, 0.0);
        state.dy_d.resize(n_d, 0.0);
        state.s.resize(n_d, 0.0);
        state.ds.resize(n_d, 0.0);
        state.c_scaling.resize(n_c, 1.0);
        state.d_scaling.resize(n_d, 1.0);
        state.c_x.resize(n_c, 0.0);
        state.d_x.resize(n_d, 0.0);
    }

    /// Phase 6b test helper: install variable bounds + matching
    /// `BoundLayout`. Resizes the four compressed mirrors to the new
    /// `n_x_l`/`n_x_u` (zeros).
    fn set_variable_bounds(state: &mut SolverState, x_l: Vec<f64>, x_u: Vec<f64>) {
        assert_eq!(x_l.len(), state.n);
        assert_eq!(x_u.len(), state.n);
        state.bound_layout = crate::bound_layout::BoundLayout::new(&x_l, &x_u);
        state.z_l_compressed = vec![0.0; state.bound_layout.n_x_l];
        state.z_u_compressed = vec![0.0; state.bound_layout.n_x_u];
        state.dz_l_compressed = vec![0.0; state.bound_layout.n_x_l];
        state.dz_u_compressed = vec![0.0; state.bound_layout.n_x_u];
        state.x_l_compressed = state.bound_layout.project_l(&x_l);
        state.x_u_compressed = state.bound_layout.project_u(&x_u);
    }

    #[test]
    fn test_phase6b_compressed_z_layout() {
        // 4 vars: var 0 lower-only, var 1 upper-only, var 2 free, var 3 two-sided.
        let mut state = minimal_state(4, 0);
        set_variable_bounds(
            &mut state,
            vec![0.0, f64::NEG_INFINITY, f64::NEG_INFINITY, -1.0],
            vec![f64::INFINITY, 5.0, f64::INFINITY, 1.0],
        );
        assert_eq!(state.bound_layout.n_x_l, 2);
        assert_eq!(state.bound_layout.n_x_u, 2);

        // Write directly to the compressed mirrors (the canonical form).
        state.z_l_compressed = vec![1.0, 4.0];
        state.z_u_compressed = vec![2.0, 5.0];
        state.dz_l_compressed = vec![0.1, 0.4];
        state.dz_u_compressed = vec![0.2, 0.5];

        // Materialized full-n views match expansion via bound_layout.
        assert_eq!(state.z_l_combined(), vec![1.0, 0.0, 0.0, 4.0]);
        assert_eq!(state.z_u_combined(), vec![0.0, 2.0, 0.0, 5.0]);
        assert_eq!(state.dz_l_combined(), vec![0.1, 0.0, 0.0, 0.4]);
        assert_eq!(state.dz_u_combined(), vec![0.0, 0.2, 0.0, 0.5]);
    }

    #[test]
    fn test_phase6b_advance_z_to_trial_compressed() {
        // Single-bound problem on 1 var.
        let mut state = minimal_state(1, 0);
        set_variable_bounds(&mut state, vec![0.0], vec![10.0]);
        state.z_l_compressed = vec![1.0];
        state.z_u_compressed = vec![2.0];
        state.dz_l_compressed = vec![0.5];
        state.dz_u_compressed = vec![-0.5];

        // After advance_z_to_trial(α=1), z_l = max(1.0+0.5, 1e-20) = 1.5
        // and z_u = max(2.0-0.5, 1e-20) = 1.5.
        advance_z_to_trial(&mut state, 1.0);
        assert!((state.z_l_compressed[0] - 1.5).abs() < 1e-12);
        assert!((state.z_u_compressed[0] - 1.5).abs() < 1e-12);
    }

    /// Phase 4a: rebuild the split Jacobian structure from a combined
    /// triplet (test helper). Mirrors the work `SolverState::new` does
    /// for production states; tests that hand-craft jac_rows/jac_cols
    /// via `minimal_state` use this to populate jac_c_*/jac_d_*.
    fn rebuild_split_jac_structure(state: &mut SolverState) {
        state.jac_c_rows.clear();
        state.jac_c_cols.clear();
        state.jac_c_combined_idx.clear();
        state.jac_d_rows.clear();
        state.jac_d_cols.clear();
        state.jac_d_combined_idx.clear();
        for (idx, (&row, &col)) in state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate() {
            if let Some(k_c) = state.layout.eq_pos[row] {
                state.jac_c_rows.push(k_c);
                state.jac_c_cols.push(col);
                state.jac_c_combined_idx.push(idx);
            } else if let Some(k_d) = state.layout.ineq_pos[row] {
                state.jac_d_rows.push(k_d);
                state.jac_d_cols.push(col);
                state.jac_d_combined_idx.push(idx);
            }
        }
        state.jac_c_vals = vec![0.0; state.jac_c_rows.len()];
        state.jac_d_vals = vec![0.0; state.jac_d_rows.len()];
    }

    #[test]
    fn test_phase5a_split_constraints_mirror_combined() {
        // 3 constraints, 1 var: row 0 ineq lo, row 1 equality (g_l=g_u=2.5),
        // row 2 ineq hi. Native split:
        //   layout.eq_pos  = [None, Some(0), None]
        //   layout.ineq_pos= [Some(0), None,    Some(1)]
        //   c_to_combined  = [1]
        //   d_to_combined  = [0, 2]
        // c_rhs = [g_l[1]] = [2.5]; for g = [10.0, 7.5, 30.0]:
        //   c_x = [g[1] - 2.5] = [5.0]
        //   d_x = [g[0], g[2]] = [10.0, 30.0]
        let mut state = minimal_state(1, 3);
        set_constraint_bounds(
            &mut state,
            vec![1.0, 2.5, 5.0],
            vec![f64::INFINITY, 2.5, f64::INFINITY],
        );
        assert_eq!(state.layout.n_c, 1);
        assert_eq!(state.layout.n_d, 2);
        assert_eq!(state.c_rhs, vec![2.5]);
        assert_eq!(state.c_x.len(), 1);
        assert_eq!(state.d_x.len(), 2);

        state.set_g_combined(&[10.0, 7.5, 30.0]);
        assert_eq!(state.c_x, vec![5.0]);
        assert_eq!(state.d_x, vec![10.0, 30.0]);

        // Reactivity: re-splitting a fresh combined snapshot must propagate.
        state.set_g_combined(&[11.0, 4.5, 30.0]);
        assert_eq!(state.c_x, vec![2.0]);
        assert_eq!(state.d_x, vec![11.0, 30.0]);
    }

    #[test]
    fn test_phase4a_split_jac_mirrors_combined() {
        // 2 constraints, 2 vars: row 0 equality (g_l=g_u=0), row 1 ineq.
        // Combined Jacobian: [[1.0, 2.0], [3.0, 4.0]] in dense layout.
        // Triplet form: rows=[0,0,1,1], cols=[0,1,0,1], vals=[1,2,3,4].
        // Split: c-block has rows=[0,0], cols=[0,1], vals=[1,2];
        //        d-block has rows=[0,0], cols=[0,1], vals=[3,4].
        let mut state = minimal_state(2, 2);
        set_constraint_bounds(&mut state, vec![0.0, 1.0], vec![0.0, f64::INFINITY]);
        state.jac_rows = vec![0, 0, 1, 1];
        state.jac_cols = vec![0, 1, 0, 1];
        rebuild_split_jac_structure(&mut state);
        state.set_jac_vals_combined(&[1.0, 2.0, 3.0, 4.0]);

        assert_eq!(state.jac_c_rows, vec![0, 0]);
        assert_eq!(state.jac_c_cols, vec![0, 1]);
        assert_eq!(state.jac_c_vals, vec![1.0, 2.0]);
        assert_eq!(state.jac_d_rows, vec![0, 0]);
        assert_eq!(state.jac_d_cols, vec![0, 1]);
        assert_eq!(state.jac_d_vals, vec![3.0, 4.0]);

        // Re-splitting after mutation: split mirror tracks combined.
        state.set_jac_vals_combined(&[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(state.jac_c_vals, vec![10.0, 20.0]);
        assert_eq!(state.jac_d_vals, vec![30.0, 40.0]);
    }

    #[test]
    fn test_iterate_snapshot_capture_and_restore() {
        let mut state = minimal_state(2, 1);
        // Phase 6d.2: snapshot now stores compressed; install proper
        // two-sided bounds via the test helper so the bound_layout has
        // n_x_l=n_x_u=2 and the compressed mirror has matching shape.
        set_variable_bounds(&mut state, vec![-10.0, -10.0], vec![10.0, 10.0]);
        state.x = vec![1.5, 2.5];
        state.set_y_combined(&[0.7]);
        state.z_l_compressed = vec![0.1, 0.2];
        state.z_u_compressed = vec![0.3, 0.4];
        state.mu = 1e-3;
        state.obj = 42.0;
        let mut filter = Filter::new(1e4);
        filter.add(0.5, 10.0);
        let snap = IterateSnapshot::capture(&state, &filter, 7);
        // Mutate state and filter to simulate further iterations.
        state.x = vec![9.0, 9.0];
        state.set_y_combined(&[9.0]);
        state.z_l_compressed = vec![0.0, 0.0];
        state.z_u_compressed = vec![0.0, 0.0];
        state.mu = 1.0;
        state.obj = 0.0;
        filter.add(99.0, 99.0);
        // Restore and verify.
        snap.restore(&mut state, &mut filter);
        assert_eq!(state.x, vec![1.5, 2.5]);
        assert_eq!(state.y_combined(), vec![0.7]);
        assert_eq!(state.z_l_compressed, vec![0.1, 0.2]);
        assert_eq!(state.z_u_compressed, vec![0.3, 0.4]);
        assert_eq!(state.mu, 1e-3);
        assert_eq!(state.obj, 42.0);
        assert_eq!(snap.iteration, 7);
        assert_eq!(filter.entries().len(), 1);
        // Stored corner per AugmentFilter: (1-γ_θ)·0.5.
        assert!((filter.entries()[0].theta - (1.0 - 1e-5) * 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_almost_feasible_guard_no_snapshot_aborts() {
        // Ipopt 3.14 IpBacktrackingLineSearch.cpp:597 throws
        // STEP_COMPUTATION_FAILED when the almost-feasible guard fires
        // and no acceptable iterate is cached. Mirror that: return
        // NumericalError.
        let mut state = minimal_state(1, 0);
        let mut filter = Filter::new(1e4);
        let opts = SolverOptions::default();
        let result = check_almost_feasible_guard(
            &mut state, &opts, &mut filter,
            1e-12, 1e-12,
        );
        assert!(result.is_some());
        assert_eq!(result.unwrap().status, SolveStatus::NumericalError);
    }

    #[test]
    fn test_almost_feasible_guard_predicate_blocks() {
        let mut state = minimal_state(1, 0);
        state.x = vec![1.0];
        let mut filter = Filter::new(1e4);
        state.acceptable_iterate = Some(IterateSnapshot::capture(&state, &filter, 0));
        state.x = vec![5.0];
        let opts = SolverOptions::default();
        // Predicate fails: primal_inf large.
        let result = check_almost_feasible_guard(
            &mut state, &opts, &mut filter,
            1.0, 1.0,
        );
        assert!(result.is_none());
        // Snapshot retained because guard did not fire.
        assert!(state.acceptable_iterate.is_some());
        assert_eq!(state.x, vec![5.0]);
    }

    #[test]
    fn test_almost_feasible_guard_fires_with_snapshot() {
        let mut state = minimal_state(1, 0);
        state.x = vec![1.0];
        let mut filter = Filter::new(1e4);
        state.acceptable_iterate = Some(IterateSnapshot::capture(&state, &filter, 3));
        state.x = vec![5.0];
        let opts = SolverOptions::default();
        // Predicate holds: pinf < 1e-2 * tol = 1e-10, pinf_max < 1e-1 * 1e-4 = 1e-5
        let result = check_almost_feasible_guard(
            &mut state, &opts, &mut filter,
            1e-12, 1e-12,
        );
        assert!(result.is_some());
        assert_eq!(result.unwrap().status, SolveStatus::Acceptable);
        assert_eq!(state.x, vec![1.0]);
        assert!(state.acceptable_iterate.is_none());
    }

    #[test]
    fn test_avg_compl_variable_bounds_only() {
        // 2 vars, lower-bound only. x = [1.5, 2.0], x_l = [1.0, 1.0], z_l = [2.0, 3.0].
        // slacks = [0.5, 1.0]; avg_compl = (0.5*2.0 + 1.0*3.0) / 2 = 4.0 / 2 = 2.0
        let mut state = minimal_state(2, 0);
        state.x = vec![1.5, 2.0];
        set_variable_bounds(&mut state, vec![1.0, 1.0], vec![f64::INFINITY, f64::INFINITY]);
        state.z_l_compressed = vec![2.0, 3.0];
        let avg = compute_avg_complementarity(&state);
        assert!((avg - 2.0).abs() < 1e-12, "expected 2.0, got {}", avg);
    }

    #[test]
    fn test_avg_compl_both_bounds() {
        // 1 var, both bounds. x = 1.5, x_l = 1.0, x_u = 2.0, z_l = 2.0, z_u = 3.0.
        // avg = (0.5*2.0 + 0.5*3.0) / 2 = 2.5 / 2 = 1.25
        let mut state = minimal_state(1, 0);
        state.x = vec![1.5];
        set_variable_bounds(&mut state, vec![1.0], vec![2.0]);
        state.z_l_compressed = vec![2.0];
        state.z_u_compressed = vec![3.0];
        let avg = compute_avg_complementarity(&state);
        assert!((avg - 1.25).abs() < 1e-12, "expected 1.25, got {}", avg);
    }

    #[test]
    fn test_avg_compl_inequality_fallback() {
        // No variable bounds, but an inequality constraint with v_l > 0 triggers fallback.
        // g = 2.0, g_l = 1.0, v_l = 0.5 -> slack = 1.0, contrib = 0.5; avg = 0.5 / 1 = 0.5.
        let mut state = minimal_state(1, 1);
        set_constraint_bounds(&mut state, vec![1.0], vec![f64::INFINITY]);
        state.set_g_combined(&[2.0]);
        state.v_l_compressed = vec![0.5];
        sync_state_s_to_g(&mut state);
        let avg = compute_avg_complementarity(&state);
        assert!((avg - 0.5).abs() < 1e-12, "fallback path: expected 0.5, got {}", avg);
    }

    #[test]
    fn test_avg_compl_includes_slack_side_with_var_bounds() {
        // B-cross6: avg_compl now ALWAYS includes the slack-side
        // complementarity v_L·s_L / v_U·s_U, matching Ipopt's
        // `IpIpoptCalculatedQuantities::curr_avrg_compl`. The previous
        // ripopt-only "fallback unless no var bounds" guard biased the
        // adaptive μ oracle on mixed-bound problems and has been retired.
        // Variable bound: slack=1.0, z_L=1.0 → contrib=1.0.
        // Constraint slack (one-sided lower): s_L = s−g_l = 1.0, v_L=99
        // → contrib=99.0. avg = (1.0 + 99.0) / 2 = 50.
        let mut state = minimal_state(1, 1);
        state.x = vec![2.0];
        set_variable_bounds(&mut state, vec![1.0], vec![f64::INFINITY]);
        state.z_l_compressed = vec![1.0];
        set_constraint_bounds(&mut state, vec![1.0], vec![f64::INFINITY]);
        state.set_g_combined(&[2.0]);
        state.v_l_compressed = vec![99.0];
        sync_state_s_to_g(&mut state);
        let avg = compute_avg_complementarity(&state);
        assert!((avg - 50.0).abs() < 1e-12,
            "expected aligned avg = (1 + 99)/2 = 50; got {}", avg);
    }

    #[test]
    fn test_avg_compl_no_bounds_anywhere() {
        // Unconstrained, no bounds: avg_compl = 0.0 (count stays at 0).
        let state = minimal_state(3, 0);
        let avg = compute_avg_complementarity(&state);
        assert_eq!(avg, 0.0);
    }

    #[test]
    fn test_bound_multiplier_count_finite_bounds_only() {
        // T0.2: 3 vars but only 1 finite bound; count must be 1, not 6.
        let mut state = minimal_state(3, 0);
        set_variable_bounds(
            &mut state,
            vec![0.0, f64::NEG_INFINITY, f64::NEG_INFINITY], // finite lower on var 0
            vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        );
        let count = compute_bound_multiplier_count(&state);
        assert_eq!(count, 1, "only 1 finite variable bound; got {}", count);
    }

    #[test]
    fn test_bound_multiplier_count_with_constraint_bounds() {
        // T0.2: variable bounds + inequality constraint bounds.
        //   x_l[0] finite, x_u[0] finite => 2
        //   x_l[1] = -inf, x_u[1] = -inf, x_u[1] = +inf => 0
        //   constraint 0: g_l finite, g_u = inf, ineq => +1
        //   constraint 1: g_l = g_u (equality) => 0 (skipped)
        // Total = 3.
        let mut state = minimal_state(2, 2);
        set_variable_bounds(
            &mut state,
            vec![0.0, f64::NEG_INFINITY],
            vec![10.0, f64::INFINITY],
        );
        set_constraint_bounds(&mut state, vec![0.0, 5.0], vec![f64::INFINITY, 5.0]);
        let count = compute_bound_multiplier_count(&state);
        assert_eq!(count, 3, "expected 3 finite bounds, got {}", count);
    }

    #[test]
    fn test_convergence_info_dual_inf_unscaled_with_obj_scaling() {
        // T0.4: when obj_scaling = 0.5, dual_inf_unscaled = dual_inf / 0.5 = 2 * dual_inf.
        // Build a state with grad_f = [1.0] (scaled), no constraints, no z.
        // dual_inf = max|grad_f - z_l + z_u| = 1.0.
        // dual_inf_unscaled = 1.0 / 0.5 = 2.0.
        let mut state = minimal_state(1, 0);
        state.obj_scaling = 0.5;
        state.grad_f = vec![1.0];
        let info = compute_convergence_info_from_state(&state, 0.0, 1, 0);
        assert!((info.dual_inf - 1.0).abs() < 1e-12, "dual_inf = {}", info.dual_inf);
        assert!((info.dual_inf_unscaled - 2.0).abs() < 1e-12,
            "dual_inf_unscaled with obj_scaling=0.5 should be 2*dual_inf, got {}",
            info.dual_inf_unscaled);
    }

    #[test]
    fn test_convergence_info_dual_inf_unscaled_obj_scaling_one() {
        // T0.4: obj_scaling = 1.0 leaves dual_inf_unscaled == dual_inf.
        let mut state = minimal_state(1, 0);
        state.obj_scaling = 1.0;
        state.grad_f = vec![3.0];
        let info = compute_convergence_info_from_state(&state, 0.0, 1, 0);
        assert!((info.dual_inf - info.dual_inf_unscaled).abs() < 1e-15);
        assert!((info.dual_inf - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_compl_err_includes_constraint_slack() {
        // T0.3: inequality constraint with slack > 0 and v_l > 0 must contribute
        // to compl error. No variable bounds.
        // g = 3.0, g_l = 1.0, g_u = +inf, v_l = 0.5; slack = 2.0.
        // Variable-bound block: 0.0.
        // Constraint-slack block: |2.0 * 0.5 - 0.0| = 1.0.
        // Mirrors Ipopt's curr_complementarity which uses the dedicated v_L
        // multipliers, not max(y,0) (IpIpoptCalculatedQuantities.cpp:2467-2497).
        let mut state = minimal_state(1, 1);
        state.x = vec![0.0];
        set_constraint_bounds(&mut state, vec![1.0], vec![f64::INFINITY]);
        state.set_g_combined(&[3.0]);
        state.v_l_compressed = vec![0.5];
        sync_state_s_to_g(&mut state);
        let err = compute_compl_err_at_state(&state);
        assert!((err - 1.0).abs() < 1e-12,
            "expected slack*v_l = 1.0, got {}", err);
    }

    #[test]
    fn test_compl_err_constraint_upper_block() {
        // T0.3: upper-bound side uses v_u directly.
        // g = 1.0, g_l = -inf, g_u = 4.0, v_u = 0.25; slack = 3.0.
        // Constraint-slack block: |3.0 * 0.25 - 0.0| = 0.75.
        let mut state = minimal_state(1, 1);
        state.x = vec![0.0];
        set_constraint_bounds(&mut state, vec![f64::NEG_INFINITY], vec![4.0]);
        state.set_g_combined(&[1.0]);
        state.v_u_compressed = vec![0.25];
        sync_state_s_to_g(&mut state);
        let err = compute_compl_err_at_state(&state);
        assert!((err - 0.75).abs() < 1e-12,
            "expected slack*v_u = 0.75, got {}", err);
    }

    #[test]
    fn test_multiplier_count_includes_all_y() {
        // T0.2: multiplier_count = m + finite_bound_count.
        // m=2, 1 finite var bound => count = 2 + 1 = 3.
        let mut state = minimal_state(3, 2);
        set_variable_bounds(
            &mut state,
            vec![0.0, f64::NEG_INFINITY, f64::NEG_INFINITY],
            vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        );
        let count = compute_multiplier_count(&state);
        assert_eq!(count, 3);
    }

    /// Minimal `NlpProblem` whose evaluations return user-controlled
    /// non-finite values so we can exercise `FiniteCheckedProblem`.
    struct PoisonProblem {
        bad_obj: bool,
        bad_grad: bool,
        bad_g: bool,
        bad_jac: bool,
        bad_hess: bool,
    }

    impl NlpProblem for PoisonProblem {
        fn num_variables(&self) -> usize { 1 }
        fn num_constraints(&self) -> usize { 1 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 0.0; g_u[0] = 0.0;
        }
        fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; }
        fn objective(&self, _x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = if self.bad_obj { f64::NAN } else { 1.0 };
            true
        }
        fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = if self.bad_grad { f64::INFINITY } else { 1.0 };
            true
        }
        fn constraints(&self, _x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g[0] = if self.bad_g { f64::NAN } else { 0.0 };
            true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0], vec![0])
        }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals[0] = if self.bad_jac { f64::NEG_INFINITY } else { 1.0 };
            true
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0], vec![0])
        }
        fn hessian_values(
            &self, _x: &[f64], _new_x: bool, _of: f64, _l: &[f64], vals: &mut [f64],
        ) -> bool {
            vals[0] = if self.bad_hess { f64::NAN } else { 1.0 };
            true
        }
    }

    #[test]
    fn finite_checked_passes_finite_values_through() {
        let p = PoisonProblem {
            bad_obj: false, bad_grad: false, bad_g: false,
            bad_jac: false, bad_hess: false,
        };
        let w = FiniteCheckedProblem::new(&p);
        let x = [0.0];
        let mut obj = 0.0;
        let mut grad = [0.0];
        let mut g = [0.0];
        let mut jac = [0.0];
        let mut hess = [0.0];
        assert!(w.objective(&x, true, &mut obj));
        assert!(w.gradient(&x, true, &mut grad));
        assert!(w.constraints(&x, true, &mut g));
        assert!(w.jacobian_values(&x, true, &mut jac));
        assert!(w.hessian_values(&x, true, 1.0, &[0.0], &mut hess));
    }

    #[test]
    fn finite_checked_rejects_non_finite_values() {
        let x = [0.0];
        let mut buf1 = [0.0];
        let mut buf_g = [0.0];

        let p_obj = PoisonProblem {
            bad_obj: true, bad_grad: false, bad_g: false, bad_jac: false, bad_hess: false,
        };
        let mut obj = 0.0;
        assert!(!FiniteCheckedProblem::new(&p_obj).objective(&x, true, &mut obj));

        let p_grad = PoisonProblem {
            bad_obj: false, bad_grad: true, bad_g: false, bad_jac: false, bad_hess: false,
        };
        assert!(!FiniteCheckedProblem::new(&p_grad).gradient(&x, true, &mut buf1));

        let p_g = PoisonProblem {
            bad_obj: false, bad_grad: false, bad_g: true, bad_jac: false, bad_hess: false,
        };
        assert!(!FiniteCheckedProblem::new(&p_g).constraints(&x, true, &mut buf_g));

        let p_jac = PoisonProblem {
            bad_obj: false, bad_grad: false, bad_g: false, bad_jac: true, bad_hess: false,
        };
        assert!(!FiniteCheckedProblem::new(&p_jac).jacobian_values(&x, true, &mut buf1));

        let p_hess = PoisonProblem {
            bad_obj: false, bad_grad: false, bad_g: false, bad_jac: false, bad_hess: true,
        };
        assert!(!FiniteCheckedProblem::new(&p_hess)
            .hessian_values(&x, true, 1.0, &[0.0], &mut buf1));
    }

    /// Ipopt `MinC_1NrmRestorationPhase::PerformRestoration`
    /// (`IpRestoMinC_1Nrm.cpp:374-419`): with no x step
    /// (s_cur == s_trial), the synthetic Newton step
    /// δz = (μ − z·s_trial)/s_curr drives `z + α·δz` to μ/s when α=1.
    #[test]
    fn test_resto_handoff_no_x_step_recovers_mu_slack() {
        let mut state = minimal_state(1, 0);
        state.x = vec![1.1];
        set_variable_bounds(&mut state, vec![1.0], vec![f64::INFINITY]);
        state.mu = 0.01;
        // Parent's pre-resto z lives in state.z_l_compressed.
        state.z_l_compressed[0] = 0.5;
        // s_cur = s_trial = 0.1; z=0.5 → δz = (0.01 − 0.05)/0.1 = −0.4.
        // α_dual FTB: −0.99·0.5 / −0.4 = 1.2375 → clamp to 1.
        // z_new = 0.5 + (−0.4) = 0.1 = μ/s.
        let opts = SolverOptions::default();
        let x_cur = vec![1.1];
        let s_cur: Vec<f64> = vec![];
        let nuclear = update_bound_multipliers_after_restoration(
            &mut state, &opts, &x_cur, &s_cur,
        );
        assert!((state.z_l_compressed[0] - 0.1).abs() < 1e-12,
            "z_l should converge to μ/s = 0.1 via Newton step, got {}",
            state.z_l_compressed[0]);
        assert!(!nuclear, "z_max=0.1 should not trigger nuclear reset");
    }

    /// Ipopt `IpRestoMinC_1Nrm.cpp:402-419`: when the post-step max
    /// multiplier exceeds `bound_mult_reset_threshold` (default 1e3),
    /// **all four blocks (z_L, z_U, v_L, v_U) are reset to 1.0**. The
    /// nuclear reset is the safety valve that prevents inflated parent
    /// z's from poisoning the next factorization.
    #[test]
    fn test_resto_handoff_nuclear_reset_above_threshold() {
        let mut state = minimal_state(1, 0);
        state.x = vec![1.1];
        set_variable_bounds(&mut state, vec![1.0], vec![f64::INFINITY]);
        state.mu = 1e-6;
        // Pre-resto z huge (1e10); s_cur=s_trial=0.1.
        // δz = (1e-6 − 1e10·0.1)/0.1 = (1e-6 − 1e9)/0.1 ≈ −1e10.
        // FTB: −0.99·1e10 / −1e10 = 0.99 → step = −0.99·1e10.
        // z_new ≈ 1e10·(1 − 0.99) = 1e8 → above 1e3 → nuclear reset.
        state.z_l_compressed[0] = 1.0e10;
        let opts = SolverOptions::default();
        let x_cur = vec![1.1];
        let s_cur: Vec<f64> = vec![];
        let nuclear = update_bound_multipliers_after_restoration(
            &mut state, &opts, &x_cur, &s_cur,
        );
        assert!(nuclear, "z_new ≈ 1e8 should trigger nuclear reset");
        assert_eq!(state.z_l_compressed[0], 1.0,
            "nuclear reset must drive z_l to 1.0, got {}", state.z_l_compressed[0]);
    }

    /// Ipopt `IpRestoMinC_1Nrm.cpp:438-453`: with an x step (s shrinks
    /// from s_cur=0.5 to s_trial=0.1), the synthetic Newton uses both
    /// s_curr and s_trial, not just one. Verify δz computation:
    ///   δz = (μ − z·s_trial) / s_curr = (0.05 − 0.1·0.1) / 0.5
    ///      = (0.05 − 0.01) / 0.5 = 0.08
    /// δz > 0 → α_dual = 1 (no FTB cap needed). z_new = 0.1 + 0.08 = 0.18.
    #[test]
    fn test_resto_handoff_with_x_step_uses_both_slacks() {
        let mut state = minimal_state(1, 0);
        // Trial: x=1.1 → s_trial=0.1. Pre-resto: x_cur=1.5 → s_cur=0.5.
        state.x = vec![1.1];
        set_variable_bounds(&mut state, vec![1.0], vec![f64::INFINITY]);
        state.mu = 0.05;
        state.z_l_compressed[0] = 0.1;
        let opts = SolverOptions::default();
        let x_cur = vec![1.5];
        let s_cur: Vec<f64> = vec![];
        update_bound_multipliers_after_restoration(
            &mut state, &opts, &x_cur, &s_cur,
        );
        assert!((state.z_l_compressed[0] - 0.18).abs() < 1e-12,
            "z_l should be 0.18 from synthetic Newton step, got {}",
            state.z_l_compressed[0]);
    }

    /// Trivial 1-D NLP for filter-gating tests. Objective and
    /// constraints are user-pluggable closures (boxed-fn'd into
    /// the trait via the wrapper).
    struct MockNlp {
        n: usize,
        m: usize,
        obj_at: Box<dyn Fn(&[f64]) -> f64>,
        cons_at: Box<dyn Fn(&[f64], &mut [f64])>,
    }

    impl NlpProblem for MockNlp {
        fn num_variables(&self) -> usize { self.n }
        fn num_constraints(&self) -> usize { self.m }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            for v in x_l.iter_mut() { *v = f64::NEG_INFINITY; }
            for v in x_u.iter_mut() { *v = f64::INFINITY; }
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            for v in g_l.iter_mut() { *v = 0.0; }
            for v in g_u.iter_mut() { *v = 0.0; }
        }
        fn initial_point(&self, x0: &mut [f64]) { for v in x0.iter_mut() { *v = 0.0; } }
        fn objective(&self, x: &[f64], _: bool, obj: &mut f64) -> bool { *obj = (self.obj_at)(x); true }
        fn gradient(&self, _: &[f64], _: bool, grad: &mut [f64]) -> bool { for v in grad.iter_mut() { *v = 0.0; } true }
        fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool { (self.cons_at)(x, g); true }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0; self.m], (0..self.m).collect()) }
        fn jacobian_values(&self, _: &[f64], _: bool, vals: &mut [f64]) -> bool { for v in vals.iter_mut() { *v = 1.0; } true }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
        fn hessian_values(&self, _: &[f64], _: bool, _: f64, _: &[f64], _: &mut [f64]) -> bool { true }
    }

    /// T0.9: when the resto-returned iterate is rejected by the
    /// existing filter (and is not feasibility-recovery), the
    /// success handler must commit nothing and return false. The
    /// pre-existing filter entries must remain intact (Ipopt
    /// IpRestoFilterConvCheck::TestOrigProgress). This is the
    /// behavior that the old code obscured by clearing the filter
    /// before checking.
    #[test]
    fn test_apply_restoration_success_filter_rejects_dominated_iterate() {
        let mut state = minimal_state(1, 1);
        // Old x; constraint g(x) = x; equality g = 0.
        state.x = vec![0.5];
        set_constraint_bounds(&mut state, vec![0.0], vec![0.0]);
        state.set_g_combined(&[0.5]);
        // x_l = -inf, x_u = +inf via minimal_state default (no bounds).
        state.mu = 0.1;

        // Mock NLP: objective = x[0], constraints = x[0] = 0.
        // Trial restored point x = 2.0 → theta = 2.0, phi = 2.0.
        let problem = MockNlp {
            n: 1, m: 1,
            obj_at: Box::new(|x| x[0]),
            cons_at: Box::new(|x, g| { g[0] = x[0]; }),
        };

        let mut filter = Filter::new(1e4);
        // Seed the filter with an entry that DOMINATES (theta=2, phi=2):
        //   theta_e=1, phi_e=1. With gamma_theta=1e-5, gamma_phi=1e-8,
        //   acceptance demands theta < (1-γθ)·1 ≈ 1 OR phi < 1 - γφ·1.
        //   Trial (2, 2) fails both → not acceptable.
        filter.add(1.0, 1.0);
        let entries_before = filter.entries().to_vec();
        let theta_min_before = filter.theta_max(); // proxy for filter state

        let mut mu_state = MuState::new();
        let opts = SolverOptions::default();
        let mut lbfgs_state: Option<LbfgsIpmState> = None;

        let x_new = vec![2.0]; // theta=2, phi=2 → filter-rejected
        let committed = apply_restoration_success(
            &mut state, &mut filter, &mut mu_state, &opts, 1, 1,
            &problem, &x_new, None,
            None, false, &mut lbfgs_state,
        );

        assert!(!committed,
            "filter-dominated iterate must NOT commit (T0.9)");
        // x must NOT be updated.
        assert!((state.x[0] - 0.5).abs() < 1e-15,
            "state.x must be unchanged on rejection, got {}", state.x[0]);
        // Filter entries must be intact (no reset occurred).
        assert_eq!(filter.entries().len(), entries_before.len(),
            "filter must NOT be cleared on rejected resto (T0.9)");
        // Stored corner per AugmentFilter: ((1-γ_θ)·1, 1 - γ_φ·1).
        let theta_corner = (1.0 - 1e-5) * 1.0;
        let phi_corner = 1.0 - 1e-8 * 1.0;
        assert!((filter.entries()[0].theta - theta_corner).abs() < 1e-15,
            "pre-existing filter entry theta must persist");
        assert!((filter.entries()[0].phi - phi_corner).abs() < 1e-15,
            "pre-existing filter entry phi must persist");
        // theta_max unchanged
        assert!((filter.theta_max() - theta_min_before).abs() < 1e-15);
    }

    /// T0.9: when the restored iterate IS acceptable to the filter,
    /// the success handler commits and leaves filter entries
    /// untouched (no reset).
    #[test]
    fn test_apply_restoration_success_filter_accepts_and_preserves_entries() {
        let mut state = minimal_state(1, 1);
        state.x = vec![0.5];
        set_constraint_bounds(&mut state, vec![0.0], vec![0.0]);
        state.set_g_combined(&[0.5]);
        // x_l = -inf, x_u = +inf via minimal_state default (no bounds).
        state.mu = 0.1;

        // Trial point: x = 1e-3 → theta = 1e-3, phi = 1e-3. Better
        // than entry (1.0, 1.0) on both axes → acceptable.
        let problem = MockNlp {
            n: 1, m: 1,
            obj_at: Box::new(|x| x[0]),
            cons_at: Box::new(|x, g| { g[0] = x[0]; }),
        };

        let mut filter = Filter::new(1e4);
        filter.add(1.0, 1.0);
        let entries_before_len = filter.entries().len();

        let mut mu_state = MuState::new();
        let opts = SolverOptions::default();
        let mut lbfgs_state: Option<LbfgsIpmState> = None;

        let x_new = vec![1.0e-3];
        let committed = apply_restoration_success(
            &mut state, &mut filter, &mut mu_state, &opts, 1, 1,
            &problem, &x_new, None,
            None, false, &mut lbfgs_state,
        );

        assert!(committed, "acceptable iterate must commit");
        assert!((state.x[0] - 1.0e-3).abs() < 1e-15, "state.x must be updated");
        // T0.9: filter must NOT be cleared on success either.
        assert_eq!(filter.entries().len(), entries_before_len,
            "filter must retain entries on resto success (T0.9)");
        // Stored corner per AugmentFilter: ((1-γ_θ)·1, ...).
        assert!((filter.entries()[0].theta - (1.0 - 1e-5)).abs() < 1e-15);
    }

    #[test]
    fn test_push_initial_point_one_sided_lower_large_magnitude() {
        // Variable with x_L = 1e3 and no upper bound, starting at x = 1e3.
        // Ipopt pushes by κ1 · max(|x_L|, 1) = 0.01 · 1e3 = 10, so x ≥ 1010.
        // The pre-fix formula pushed by only `bound_push = 0.01` absolute.
        let opts = SolverOptions::default();
        let mut x = vec![1e3];
        let x_l = vec![1e3];
        let x_u = vec![f64::INFINITY];
        push_initial_point_from_bounds(&mut x, &x_l, &x_u, &opts);
        assert!(
            x[0] >= 1e3 + 0.01 * 1e3 - 1e-12,
            "expected x >= 1010 after magnitude-scaled push, got x = {}",
            x[0]
        );
    }

    #[test]
    fn test_push_initial_point_one_sided_lower_unit_magnitude() {
        // |x_L| < 1 falls back to max(|x_L|, 1) = 1, so push is bound_push.
        let opts = SolverOptions::default();
        let mut x = vec![0.0];
        let x_l = vec![0.0];
        let x_u = vec![f64::INFINITY];
        push_initial_point_from_bounds(&mut x, &x_l, &x_u, &opts);
        assert!(
            (x[0] - opts.bound_push).abs() < 1e-12,
            "expected x = bound_push = {}, got x = {}",
            opts.bound_push,
            x[0]
        );
    }

    #[test]
    fn test_relax_fixed_variable_bounds_default_widens() {
        // Default option (Phase 11: MakeParameter). The IPM-layer fallback
        // `relax_fixed_variable_bounds` widens for both RelaxBounds and
        // MakeParameter (the latter is a safety net for callers that bypass
        // the preprocessor and reach the IPM with literally-equal bounds).
        // x_l = x_u = 5.0 should be widened to [5 - 5e-8, 5 + 5e-8].
        let opts = SolverOptions::default();
        let mut x_l = vec![5.0];
        let mut x_u = vec![5.0];
        relax_fixed_variable_bounds(&mut x_l, &mut x_u, &opts);
        let expected_relax = 1e-8 * 5.0;
        assert!((x_l[0] - (5.0 - expected_relax)).abs() < 1e-15);
        assert!((x_u[0] - (5.0 + expected_relax)).abs() < 1e-15);
    }

    #[test]
    fn test_relax_fixed_variable_bounds_make_parameter_widens_at_solver_layer() {
        // MakeParameter eliminates fixed vars upstream via PreprocessedProblem.
        // If bounds still reach the solver-layer `relax_fixed_variable_bounds`
        // (e.g. when the user disables preprocessing AND MakeParameter
        // elimination didn't run), they are widened identically to RelaxBounds
        // so the IPM has a non-empty interior either way.
        let mut opts = SolverOptions::default();
        opts.fixed_variable_treatment = FixedVariableTreatment::MakeParameter;
        let mut x_l = vec![5.0];
        let mut x_u = vec![5.0];
        relax_fixed_variable_bounds(&mut x_l, &mut x_u, &opts);
        assert!(x_l[0] < 5.0 && x_u[0] > 5.0);
    }

    #[test]
    fn test_relax_fixed_variable_bounds_skips_non_fixed() {
        let opts = SolverOptions::default();
        let mut x_l = vec![1.0, f64::NEG_INFINITY];
        let mut x_u = vec![3.0, f64::INFINITY];
        relax_fixed_variable_bounds(&mut x_l, &mut x_u, &opts);
        assert_eq!(x_l[0], 1.0);
        assert_eq!(x_u[0], 3.0);
        assert_eq!(x_l[1], f64::NEG_INFINITY);
        assert_eq!(x_u[1], f64::INFINITY);
    }

    #[test]
    fn test_init_bound_multipliers_default_is_constant_one() {
        // Default options: bound_mult_init_method = Constant, val = 1.0.
        // mu_init = 0.1 should NOT appear in z_l[0]; the constant 1.0 should.
        let opts = SolverOptions::default();
        let x = vec![1.5];
        let x_l = vec![0.0];
        let x_u = vec![f64::INFINITY];
        let (z_l, z_u) = init_bound_multipliers(&x, &x_l, &x_u, 0.1, &opts);
        assert_eq!(z_l[0], 1.0, "constant default should give z_l = 1.0");
        assert_eq!(z_u[0], 0.0, "no upper bound -> z_u stays at 0");
    }

    #[test]
    fn test_init_bound_multipliers_mu_based_uses_mu_over_slack() {
        let mut opts = SolverOptions::default();
        opts.bound_mult_init_method = BoundMultInitMethod::MuBased;
        let x = vec![1.5];
        let x_l = vec![0.5];
        let x_u = vec![f64::INFINITY];
        let (z_l, _z_u) = init_bound_multipliers(&x, &x_l, &x_u, 0.1, &opts);
        // slack = 1.5 - 0.5 = 1.0 -> z_l = 0.1 / 1.0 = 0.1
        assert!((z_l[0] - 0.1).abs() < 1e-12, "got z_l = {}", z_l[0]);
    }

    #[test]
    fn test_push_initial_point_two_sided_uses_min_of_kappa1_kappa2() {
        // Two-sided narrow range: [0, 0.01]. κ2·range = 0.01·0.01 = 1e-4
        // vs κ1·max(|x_L|, 1) = 0.01·1 = 0.01. The min picks 1e-4.
        let opts = SolverOptions::default();
        let mut x = vec![0.0];
        let x_l = vec![0.0];
        let x_u = vec![0.01];
        push_initial_point_from_bounds(&mut x, &x_l, &x_u, &opts);
        // p_l = min(0.01, 1e-4) = 1e-4
        assert!((x[0] - 1e-4).abs() < 1e-12, "got x = {}", x[0]);
    }

    /// Set up a 1×1 LS system whose augmented solve yields y = 2.
    ///
    /// System:
    ///   [ I  J^T ] [r]   [grad_f - z_L + z_U]
    ///   [ J   0  ] [y] = [0                 ]
    /// With grad_f=2, J=1, z=0 → r=0, y=2.
    fn ls_y_equals_two_state(g: f64, g_eq: bool) -> SolverState {
        let mut s = minimal_state(1, 1);
        s.grad_f = vec![2.0];
        s.jac_rows = vec![0];
        s.jac_cols = vec![0];
        if g_eq {
            // Layout flip first — set_y_combined must write into the
            // post-flip y_c/y_d sizing.
            set_constraint_bounds(&mut s, vec![0.0], vec![0.0]);
        }
        // Phase 5f.4b: split storage is canonical. Build the split
        // Jacobian structure from the m-form pattern, then write split
        // values + split constraint state directly via the combined
        // setters (the values lookup via `combined_idx` mirrors what
        // `evaluate_with_linear` does at runtime).
        rebuild_split_jac_structure(&mut s);
        s.set_jac_vals_combined(&[1.0]);
        s.set_g_combined(&[g]);
        s.set_y_combined(&[999.0]);  // sentinel; recalc must overwrite
        s
    }

    #[test]
    fn test_recalc_y_off_by_default_does_not_overwrite() {
        let mut state = ls_y_equals_two_state(0.0, true);
        let opts = SolverOptions::default();
        maybe_recalc_y_post_step(&mut state, &opts, 1, 1, false);
        assert_eq!(state.y_combined(), vec![999.0], "default off must not touch y");
    }

    #[test]
    fn test_recalc_y_lbfgs_mode_recomputes_when_feasible() {
        let mut state = ls_y_equals_two_state(0.0, true); // viol = 0 < tol
        let opts = SolverOptions::default();
        maybe_recalc_y_post_step(&mut state, &opts, 1, 1, true);
        assert!((state.y_at(0) - 2.0).abs() < 1e-10, "lbfgs gate must recompute, got {}", state.y_at(0));
    }

    #[test]
    fn test_recalc_y_explicit_on_recomputes_when_feasible() {
        let mut state = ls_y_equals_two_state(0.0, true);
        let opts = SolverOptions { recalc_y: true, ..SolverOptions::default() };
        maybe_recalc_y_post_step(&mut state, &opts, 1, 1, false);
        assert!((state.y_at(0) - 2.0).abs() < 1e-10, "recalc_y=true must recompute, got {}", state.y_at(0));
    }

    #[test]
    fn test_recalc_y_skipped_when_constraint_violation_above_tol() {
        // g = 1e-3, equality constraint => constraint_violation = 1e-3 > recalc_y_feas_tol (1e-6)
        let mut state = ls_y_equals_two_state(1e-3, true);
        let opts = SolverOptions { recalc_y: true, ..SolverOptions::default() };
        maybe_recalc_y_post_step(&mut state, &opts, 1, 1, false);
        assert_eq!(state.y_combined(), vec![999.0], "infeasible iterate must skip recalc_y");
    }

    /// T3.30 full augmented system: on an inequality constraint with
    /// nonzero `(v_L − v_U)`, the post-step recalc must produce a
    /// different y than the reduced 2-block system. Setup:
    ///   n=1, m=1, lower-only inequality g_l=0 g_u=+inf, J=1, grad_f=2,
    ///   z=0, v_L=1, v_U=0. After eliminating slack:
    ///     y_d = J·sol_x − (v_L − v_U) = sol_x − 1
    ///     sol_x + J·y_d = grad_f  ⇒  2·sol_x − 1 = 2  ⇒  sol_x = 1.5
    ///     y_d = 0.5 (positive → consistent with lower-bound sign).
    ///   The reduced 2-block system would yield y = 2 (no v coupling).
    #[test]
    fn test_recalc_y_full_augmented_inequality_uses_v_l_v_u() {
        let mut state = ls_y_equals_two_state(0.5, false);
        set_constraint_bounds(&mut state, vec![0.0], vec![f64::INFINITY]);
        state.v_l_compressed = vec![1.0];
        let opts = SolverOptions { recalc_y: true, ..SolverOptions::default() };
        maybe_recalc_y_post_step(&mut state, &opts, 1, 1, false);
        assert!(
            (state.y_at(0) - 0.5).abs() < 1e-10,
            "full-augmented LS y mismatch: got {} expected 0.5", state.y_at(0)
        );
    }

    /// T3.30: on equality rows the full and reduced systems must agree.
    /// Ipopt's slack/v_d coupling only enters for inequality rows.
    #[test]
    fn test_recalc_y_full_augmented_equality_matches_reduced() {
        let mut state = ls_y_equals_two_state(0.0, true);
        // Equality row: v_L/v_U values must be ignored.
        // Phase 8d: equality rows have no compressed v slot (n_d_l=n_d_u=0
        // for an all-equality layout); writes here would be no-ops.
        debug_assert_eq!(state.d_bound_layout.n_d_l, 0);
        debug_assert_eq!(state.d_bound_layout.n_d_u, 0);
        let opts = SolverOptions { recalc_y: true, ..SolverOptions::default() };
        maybe_recalc_y_post_step(&mut state, &opts, 1, 1, false);
        assert!(
            (state.y_at(0) - 2.0).abs() < 1e-10,
            "equality row should match reduced: got {}", state.y_at(0)
        );
    }

    /// T3.32 closed-form 1D minimizer for `MinDualInfeas`. Setup:
    ///   n=1, m=1, equality (g_l = g_u = 0), grad_f_trial = 2,
    ///   J = 1, y = 0.5, z = 0, dy = -3, v = 0.
    ///   r_x = 2 + 1·0.5 = 2.5
    ///   Jt_dy = 1·(-3) = -3
    ///   r_s = 0, dy_d = 0 (equality row)
    ///   a = 9 + 0 = 9
    ///   b = 2.5·(-3) − 0 = -7.5
    ///   α* = -b/a = 7.5/9 ≈ 0.8333
    ///   Clamped to [0, 1] → 0.8333.
    #[test]
    fn test_min_dual_infeas_closed_form_equality_row() {
        struct ConstantProblem;
        impl NlpProblem for ConstantProblem {
            fn num_variables(&self) -> usize { 1 }
            fn num_constraints(&self) -> usize { 1 }
            fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
                x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
            }
            fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
                g_l[0] = 0.0; g_u[0] = 0.0;
            }
            fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; }
            fn objective(&self, _: &[f64], _: bool, obj: &mut f64) -> bool { *obj = 0.0; true }
            fn gradient(&self, _: &[f64], _: bool, grad: &mut [f64]) -> bool { grad[0] = 2.0; true }
            fn constraints(&self, _: &[f64], _: bool, g: &mut [f64]) -> bool { g[0] = 0.0; true }
            fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
            fn jacobian_values(&self, _: &[f64], _: bool, vals: &mut [f64]) -> bool { vals[0] = 1.0; true }
            fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
            fn hessian_values(&self, _: &[f64], _: bool, _: f64, _: &[f64], _: &mut [f64]) -> bool { true }
        }
        let mut state = minimal_state(1, 1);
        set_constraint_bounds(&mut state, vec![0.0], vec![0.0]);
        state.set_y_combined(&[0.5]);
        state.set_dy_combined(&[-3.0]);
        state.jac_rows = vec![0];
        state.jac_cols = vec![0];
        rebuild_split_jac_structure(&mut state);
        state.set_jac_vals_combined(&[1.0]);

        let alpha = compute_min_dual_infeas_alpha(
            &state, &ConstantProblem, AlphaForY::MinDualInfeas, 0.5, 0.5,
        );
        assert!(
            (alpha - 7.5 / 9.0).abs() < 1e-10,
            "MinDualInfeas got {}, expected {}", alpha, 7.5 / 9.0
        );
    }

    /// T3.32 SaferMinDualInfeas: same closed form but clipped to
    /// `[min(α_p, α_d), max(α_p, α_d)]`. With α_p = 0.2, α_d = 0.5,
    /// the interval is [0.2, 0.5] and α* = 0.833 clamps to 0.5.
    #[test]
    fn test_safer_min_dual_infeas_clips_to_alpha_bracket() {
        struct ConstantProblem;
        impl NlpProblem for ConstantProblem {
            fn num_variables(&self) -> usize { 1 }
            fn num_constraints(&self) -> usize { 1 }
            fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
                x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
            }
            fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
                g_l[0] = 0.0; g_u[0] = 0.0;
            }
            fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; }
            fn objective(&self, _: &[f64], _: bool, obj: &mut f64) -> bool { *obj = 0.0; true }
            fn gradient(&self, _: &[f64], _: bool, grad: &mut [f64]) -> bool { grad[0] = 2.0; true }
            fn constraints(&self, _: &[f64], _: bool, g: &mut [f64]) -> bool { g[0] = 0.0; true }
            fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
            fn jacobian_values(&self, _: &[f64], _: bool, vals: &mut [f64]) -> bool { vals[0] = 1.0; true }
            fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
            fn hessian_values(&self, _: &[f64], _: bool, _: f64, _: &[f64], _: &mut [f64]) -> bool { true }
        }
        let mut state = minimal_state(1, 1);
        set_constraint_bounds(&mut state, vec![0.0], vec![0.0]);
        state.set_y_combined(&[0.5]);
        state.set_dy_combined(&[-3.0]);
        state.jac_rows = vec![0];
        state.jac_cols = vec![0];
        rebuild_split_jac_structure(&mut state);
        state.set_jac_vals_combined(&[1.0]);

        let alpha = compute_min_dual_infeas_alpha(
            &state, &ConstantProblem, AlphaForY::SaferMinDualInfeas, 0.2, 0.5,
        );
        assert!(
            (alpha - 0.5).abs() < 1e-10,
            "SaferMinDualInfeas should clamp to max(α_p, α_d) = 0.5, got {}", alpha
        );
    }

    // ---- Magic step (T2.24, spec §5.3) ----

    /// Lower-bound-only entries push s up by `d - s` when `d > s`, and
    /// take no step when `d <= s`. Mirrors the `delta_s_magic_L`
    /// branch of `BacktrackingLineSearch::PerformMagicStep`
    /// (`IpBacktrackingLineSearch.cpp:1028-1032`).
    #[test]
    fn test_magic_step_delta_lower_only() {
        let s = vec![1.0, 1.0, 1.0];
        let d = vec![2.5, 0.5, 1.0]; // d > s, d < s, d == s
        let d_l = vec![0.0, 0.0, 0.0];
        let d_u = vec![f64::INFINITY; 3];
        let mut delta = vec![0.0; 3];
        let nnz = compute_magic_step_delta(&s, &d, &d_l, &d_u, &mut delta);
        assert_eq!(delta, vec![1.5, 0.0, 0.0], "lower-only: push up only when d > s");
        assert_eq!(nnz, 1);
    }

    /// Upper-bound-only entries push s down by `d - s` when `d < s`,
    /// and take no step when `d >= s`. Mirrors the `delta_s_magic_U`
    /// branch (`IpBacktrackingLineSearch.cpp:1036-1040`).
    #[test]
    fn test_magic_step_delta_upper_only() {
        let s = vec![1.0, 1.0, 1.0];
        let d = vec![0.4, 1.5, 1.0];
        let d_l = vec![f64::NEG_INFINITY; 3];
        let d_u = vec![10.0, 10.0, 10.0];
        let mut delta = vec![0.0; 3];
        let nnz = compute_magic_step_delta(&s, &d, &d_l, &d_u, &mut delta);
        assert_eq!(delta, vec![-0.6, 0.0, 0.0], "upper-only: push down only when d < s");
        assert_eq!(nnz, 1);
    }

    /// Doubly-bounded entries: take the candidate when it reduces the
    /// centering measure `|d_L + d_U - 2 s|`, suppress otherwise. With
    /// `d_L = 0`, `d_U = 4`, `s = 1`: centering measure is 2. Candidate
    /// `delta = +1` (since d > s) gives s_new = 2, centering 0 ≤ 2,
    /// accepted. With `s = 3`, candidate `delta = -2` gives s_new = 1,
    /// centering 2 ≤ 2 (equal), accepted. With `s = 1`, d = -3:
    /// candidate `delta = -4` gives s_new = -3, centering 10 > 2,
    /// suppressed. Mirrors the `tmp` indicator logic in
    /// `IpBacktrackingLineSearch.cpp:1054-1082`.
    #[test]
    fn test_magic_step_delta_double_bound_suppresses() {
        let d_l = vec![0.0, 0.0, 0.0];
        let d_u = vec![4.0, 4.0, 4.0];
        // Component 0: candidate improves centering -> kept.
        // Component 1: candidate keeps centering equal -> kept.
        // Component 2: candidate worsens centering -> suppressed.
        let s = vec![1.0, 3.0, 1.0];
        let d = vec![2.5, 1.0, -3.0];
        let mut delta = vec![0.0; 3];
        let _ = compute_magic_step_delta(&s, &d, &d_l, &d_u, &mut delta);
        assert!((delta[0] - 1.5).abs() < 1e-12, "kept: {}", delta[0]);
        assert!((delta[1] - (-2.0)).abs() < 1e-12, "kept on tie: {}", delta[1]);
        assert_eq!(delta[2], 0.0, "suppressed when centering worsens");
    }

    /// Free (no bounds) entries always produce zero delta.
    #[test]
    fn test_magic_step_delta_unbounded_no_step() {
        let s = vec![1.0, -2.0, 100.0];
        let d = vec![1e6, -1e6, 0.0];
        let d_l = vec![f64::NEG_INFINITY; 3];
        let d_u = vec![f64::INFINITY; 3];
        let mut delta = vec![0.0; 3];
        let nnz = compute_magic_step_delta(&s, &d, &d_l, &d_u, &mut delta);
        assert_eq!(delta, vec![0.0, 0.0, 0.0]);
        assert_eq!(nnz, 0);
    }

    /// `apply_magic_step` is a no-op in ripopt's implicit-slack
    /// formulation: x, y, v_l, v_u, z_l, z_u must all be unchanged
    /// regardless of the option flag. This test pins the architectural
    /// invariant noted in the function's documentation; if a future
    /// change adds explicit-slack behavior, the test should be updated
    /// to reflect the new contract.
    #[test]
    fn test_apply_magic_step_is_noop_in_implicit_slack_mode() {
        let mut state = minimal_state(2, 2);
        // Populate with non-trivial values so any accidental mutation is
        // visible. v_l and g_l are configured as Ipopt would expect for
        // an inequality with finite lower bound.
        state.x = vec![3.0, 4.0];
        state.set_y_combined(&[1.5, -0.5]);
        set_variable_bounds(&mut state, vec![0.0, 0.0], vec![10.0, 10.0]);
        state.z_l_compressed = vec![0.7, 0.2];
        state.z_u_compressed = vec![0.0, 0.3];
        set_constraint_bounds(&mut state, vec![0.0, f64::NEG_INFINITY], vec![f64::INFINITY, 5.0]);
        // Phase 8d: write compressed v storage. d_bound_layout has
        // n_d_l=1 (row 0 has finite lower g_l), n_d_u=1 (row 1 has
        // finite upper g_u).
        state.v_l_compressed = vec![0.4];
        state.v_u_compressed = vec![0.6];
        state.set_g_combined(&[1.0, 2.0]);
        state.mu = 0.1;
        let snapshot = (
            state.x.clone(),
            state.y_combined(),
            state.z_l_compressed.clone(),
            state.z_u_compressed.clone(),
            state.v_l_compressed.clone(),
            state.v_u_compressed.clone(),
            state.g_combined(),
        );

        let opts_on = SolverOptions { magic_step: true, ..SolverOptions::default() };
        let n_on = apply_magic_step(&mut state, &opts_on);
        assert_eq!(n_on, 0, "no explicit slack vector exists, so no updates possible");
        assert_eq!(state.x, snapshot.0);
        assert_eq!(state.y_combined(), snapshot.1);
        assert_eq!(state.z_l_compressed, snapshot.2);
        assert_eq!(state.z_u_compressed, snapshot.3);
        assert_eq!(state.v_l_compressed, snapshot.4);
        assert_eq!(state.v_u_compressed, snapshot.5);
        assert_eq!(state.g_combined(), snapshot.6);

        // With option off, identical (no-op) result.
        let opts_off = SolverOptions { magic_step: false, ..SolverOptions::default() };
        let n_off = apply_magic_step(&mut state, &opts_off);
        assert_eq!(n_off, 0);
        assert_eq!(state.x, snapshot.0);
        assert_eq!(state.y_combined(), snapshot.1);
    }

    /// The option flag short-circuits the helper: when disabled the
    /// function returns 0 without inspecting state. We verify by
    /// passing a deliberately mis-shaped state (m == 0) and confirming
    /// no panic.
    #[test]
    fn test_apply_magic_step_disabled_short_circuits() {
        let mut state = minimal_state(0, 0);
        let opts = SolverOptions { magic_step: false, ..SolverOptions::default() };
        let n = apply_magic_step(&mut state, &opts);
        assert_eq!(n, 0);
    }

    // ---- T2.23 Quality Function μ oracle (spec §3.5) ----

    /// Golden-section on a unimodal convex function `(log σ + 0.5)²`
    /// whose minimum sits at σ = exp(−0.5) ≈ 0.6065 should converge
    /// inside 8 steps for a [1e-6, 1e2] bracket.
    #[test]
    fn test_golden_section_converges_in_eight_steps_synthetic() {
        let target = (-0.5_f64).exp();
        let f = |sigma: f64| -> f64 {
            let l = sigma.ln();
            (l + 0.5).powi(2)
        };
        let s_star = golden_section_minimize(&f, 1e-6, 1e2, 8, 0.01);
        // After 8 golden-section steps the (1−φ)^8 ≈ 0.0214 fraction of the
        // log-bracket remains, so the minimiser is within ~0.4 (in log) of the
        // true minimum at exp(−0.5).
        let log_err = (s_star.ln() - target.ln()).abs();
        assert!(log_err < 0.5, "golden section did not narrow enough: log_err={}", log_err);
        // And it should be inside the original bracket.
        assert!(s_star >= 1e-6 && s_star <= 1e2);
    }

    /// Golden-section on a flat function should not panic and should
    /// return some point inside the bracket.
    #[test]
    fn test_golden_section_flat_function_returns_inside_bracket() {
        let f = |_sigma: f64| -> f64 { 1.0 };
        let s = golden_section_minimize(&f, 1e-6, 1e2, 8, 0.01);
        assert!(s >= 1e-6 && s <= 1e2);
    }

    /// Build a tiny but realistic primal-dual state with one variable bound
    /// and no constraints, so `compute_quality_function_mu` can assemble +
    /// factor an actual KKT matrix.
    fn qf_minimal_state() -> SolverState {
        let mut s = minimal_state(2, 0);
        // Two variables x_l = 0 ≤ x with H = I, grad = (1, 1). Optimum at
        // x → 0 along both axes; KKT system has unique solution.
        s.x = vec![1.0, 1.0];
        set_variable_bounds(&mut s, vec![0.0, 0.0], vec![f64::INFINITY, f64::INFINITY]);
        s.z_l_compressed = vec![0.5, 0.5];
        s.grad_f = vec![1.0, 1.0];
        s.hess_rows = vec![0, 1];
        s.hess_cols = vec![0, 1];
        s.hess_vals = vec![1.0, 1.0];
        s.mu = 0.1;
        s
    }

    /// On a small well-posed iterate the QF oracle must return Some, and
    /// the chosen μ must lie in `[μ_min, mu_max_fact * initial_avg_compl]`
    /// (T3.2 — replaced the historical 1e5 hard cap with the lazy mu_max).
    #[test]
    fn test_compute_quality_function_mu_returns_clamped_some() {
        let state = qf_minimal_state();
        let opts = SolverOptions::default();
        let mut mu_state = MuState::new();
        let avg = compute_avg_complementarity(&state);
        assert!(avg > 0.0, "test state must have positive avg_compl");
        let mut cache = kkt::FactorCache::new();
        let mu = compute_quality_function_mu(&state, &opts, &mut mu_state, avg, false, &mut cache);
        let mu = mu.expect("QF oracle should produce μ on a well-formed iterate");
        let cap = opts.mu_max_fact * avg;
        assert!(mu >= opts.mu_min, "μ below mu_min: {}", mu);
        assert!(mu <= cap, "μ above mu_max_fact*avg cap: {}", mu);
        assert!(mu.is_finite(), "μ must be finite");
    }

    /// The QF oracle returns None when avg_compl ≤ 0 (no active bound
    /// products), so the caller falls back to the linear-decrease branch.
    #[test]
    fn test_compute_quality_function_mu_none_on_zero_avg_compl() {
        let state = qf_minimal_state();
        let opts = SolverOptions::default();
        let mut mu_state = MuState::new();
        let mut cache = kkt::FactorCache::new();
        let mu = compute_quality_function_mu(&state, &opts, &mut mu_state, 0.0, false, &mut cache);
        assert!(mu.is_none(), "QF oracle must reject avg_compl ≤ 0");
    }

    /// Centrality-on exercises the `quality_function_centrality` branch
    /// end-to-end. Both must produce finite, clamped μ; the two need not
    /// disagree on every iterate (when the trial step is the same the
    /// reciprocal-centrality term cancels out at the optimum), so the
    /// assertion is on shape, not on a specific μ delta.
    #[test]
    fn test_compute_quality_function_mu_centrality_branch_lives() {
        let mut state = qf_minimal_state();
        state.x = vec![10.0, 0.01];
        state.z_l_compressed = vec![0.001, 5.0];
        let avg = compute_avg_complementarity(&state);

        let opts_off = SolverOptions { quality_function_centrality: false, ..SolverOptions::default() };
        let mut mus_off = MuState::new();
        let mut cache_off = kkt::FactorCache::new();
        let mu_off = compute_quality_function_mu(&state, &opts_off, &mut mus_off, avg, false, &mut cache_off)
            .expect("QF off");
        let opts_on = SolverOptions { quality_function_centrality: true, ..SolverOptions::default() };
        let mut mus_on = MuState::new();
        let mut cache_on = kkt::FactorCache::new();
        let mu_on = compute_quality_function_mu(&state, &opts_on, &mut mus_on, avg, false, &mut cache_on)
            .expect("QF on");
        let cap = opts_off.mu_max_fact * avg;
        for mu in [mu_off, mu_on] {
            assert!(mu.is_finite(), "μ must be finite, got {}", mu);
            assert!(mu >= opts_off.mu_min, "μ below mu_min: {}", mu);
            assert!(mu <= cap, "μ above mu_max_fact*avg cap: {}", mu);
        }
    }

    /// End-to-end: a tiny QP `min 0.5·(x-1)² + 0.5·(y-1)² s.t. x ≥ 0, y ≥ 0`
    /// (separable, optimum at (1, 1)) solves to optimal under both
    /// `mu_oracle_quality_function = true` and `false`, exercising the QF
    /// dispatch path through the live IPM loop.
    struct QfQpProblem;
    impl NlpProblem for QfQpProblem {
        fn num_variables(&self) -> usize { 2 }
        fn num_constraints(&self) -> usize { 0 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = 0.0; x_l[1] = 0.0;
            x_u[0] = f64::INFINITY; x_u[1] = f64::INFINITY;
        }
        fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
        fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.5; x0[1] = 0.5; }
        fn objective(&self, x: &[f64], _: bool, obj: &mut f64) -> bool {
            *obj = 0.5 * ((x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2));
            true
        }
        fn gradient(&self, x: &[f64], _: bool, grad: &mut [f64]) -> bool {
            grad[0] = x[0] - 1.0; grad[1] = x[1] - 1.0; true
        }
        fn constraints(&self, _: &[f64], _: bool, _: &mut [f64]) -> bool { true }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
        fn jacobian_values(&self, _: &[f64], _: bool, _: &mut [f64]) -> bool { true }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1], vec![0, 1])
        }
        fn hessian_values(&self, _: &[f64], _: bool, obj_factor: f64, _: &[f64], vals: &mut [f64]) -> bool {
            vals[0] = obj_factor; vals[1] = obj_factor; true
        }
    }

    #[test]
    fn test_quality_function_mu_end_to_end_small_nlp() {
        let problem = QfQpProblem;

        let opts_off = SolverOptions { mu_oracle_quality_function: false, ..SolverOptions::default() };
        let r_off = solve_ipm(&problem, &opts_off);
        assert_eq!(r_off.status, SolveStatus::Optimal,
            "Loqo path must solve the QP, got {:?}", r_off.status);

        let opts_on = SolverOptions { mu_oracle_quality_function: true, ..SolverOptions::default() };
        let r_on = solve_ipm(&problem, &opts_on);
        assert_eq!(r_on.status, SolveStatus::Optimal,
            "QF path must also solve the QP, got {:?}", r_on.status);

        // Both should reach the optimum (1, 1).
        assert!((r_off.x[0] - 1.0).abs() < 1e-6);
        assert!((r_on.x[0] - 1.0).abs() < 1e-6);

        // Iteration counts within a 2x band (we don't require QF to be
        // faster, only comparable — small problems hit the μ_min floor
        // regardless of the oracle).
        let off = r_off.iterations.max(1) as f64;
        let on = r_on.iterations.max(1) as f64;
        let ratio = on / off;
        assert!(ratio < 2.0 && ratio > 0.5,
            "QF iterations diverge from Loqo: off={} on={}", r_off.iterations, r_on.iterations);
    }

    // T3.25 follow-up factor-cache wiring tests removed in A7.6 — they
    // exercised the condensed-path-specific factor cache (`CachedQpProblem`,
    // `test_factor_cache_*`). A7.7 will reintroduce equivalent tests once the
    // factor cache is ported to `kkt_aug`.

    // ---- T2.22 kappa_resto / restoration acceptance (spec §7.7) ----

    /// Spec §7.7 / T2.22 item B: when restoration ends with theta_new just
    /// inside `kappa_resto * theta_current` (and is filter-acceptable),
    /// the outcome must be `Success`. Tightening kappa_resto must reject.
    #[test]
    fn test_classify_restoration_kappa_resto_gates_success() {
        let filter = crate::filter::Filter::new(1e8);
        let mut opts = SolverOptions::default();
        opts.kappa_resto = 0.9;
        opts.constr_viol_tol = 1e-4;
        opts.tol = 1e-8;
        let outcome = classify_restoration_outcome(&filter, &opts, 1.0, 0.85, 0.0, true);
        assert!(matches!(outcome, RestorationOutcome::Success),
            "expected Success at kappa=0.9, got {:?}",
            std::mem::discriminant(&outcome));

        opts.kappa_resto = 0.5;
        let outcome2 = classify_restoration_outcome(&filter, &opts, 1.0, 0.85, 0.0, true);
        assert!(matches!(outcome2, RestorationOutcome::LocalInfeasibility),
            "expected LocalInfeasibility at kappa=0.5 (inner_converged=true), got {:?}",
            std::mem::discriminant(&outcome2));

        let outcome3 = classify_restoration_outcome(&filter, &opts, 1.0, 0.85, 0.0, false);
        assert!(matches!(outcome3, RestorationOutcome::Failed),
            "expected Failed when inner did not converge, got {:?}",
            std::mem::discriminant(&outcome3));
    }

    /// Spec §7.7: small_threshold = min(tol, constr_viol_tol).
    #[test]
    fn test_classify_restoration_feasibility_threshold() {
        let filter = crate::filter::Filter::new(1e8);
        let mut opts = SolverOptions::default();
        opts.tol = 1e-8;
        opts.constr_viol_tol = 1e-6;
        opts.kappa_resto = 0.9;
        let outcome = classify_restoration_outcome(&filter, &opts, 6e-9, 5e-9, 0.0, true);
        assert!(matches!(outcome, RestorationOutcome::Success));
    }

    // -------------------------------------------------------------------
    // T2.21 (spec §4): tiny-step exit and watchdog filter fidelity.
    // -------------------------------------------------------------------

    /// `detect_tiny_step` must NOT mutate `state.mu` or the filter; it
    /// only sets `mu_state.tiny_step` and the `tiny_step_last_iter` latch.
    /// With prior latch true and a tiny detection, fires `tiny_step` flag.
    #[test]
    fn test_detect_tiny_step_no_mu_mutation() {
        let mut state = minimal_state(1, 0);
        state.x = vec![1.0];
        state.dx = vec![1e-20];
        let initial_mu = state.mu;

        let opts = SolverOptions::default();
        let mut mu_state = MuState::new();
        let mut filter = crate::filter::Filter::new(1e10);
        let initial_filter_len = filter.len();
        let mut last_iter_latch = true; // simulate prior iter latched

        detect_tiny_step(
            &mut state, &opts, &mut mu_state, &mut filter,
            &mut last_iter_latch, 0.0,
        );

        assert!(mu_state.tiny_step,
            "tiny_step flag must fire when current detection AND prior latch");
        assert!(last_iter_latch,
            "latch refreshes true (m=0 ⇒ dy_amax=0 < tiny_step_y_tol)");
        assert_eq!(state.mu, initial_mu,
            "detect_tiny_step must not mutate mu (Ipopt §IpBacktrackingLineSearch)");
        assert_eq!(filter.len(), initial_filter_len,
            "detect_tiny_step must not reset/augment the filter");
    }

    /// First-iter detection (no prior latch) must NOT yet fire the
    /// `tiny_step` flag — Ipopt requires two consecutive iters
    /// (IpBacktrackingLineSearch.cpp:407-411).
    #[test]
    fn test_detect_tiny_step_requires_prior_latch() {
        let mut state = minimal_state(1, 0);
        state.x = vec![1.0];
        state.dx = vec![1e-20];

        let opts = SolverOptions::default();
        let mut mu_state = MuState::new();
        let mut filter = crate::filter::Filter::new(1e10);
        let mut last_iter_latch = false;

        detect_tiny_step(
            &mut state, &opts, &mut mu_state, &mut filter,
            &mut last_iter_latch, 0.0,
        );

        assert!(!mu_state.tiny_step,
            "tiny_step must NOT fire on first detection without prior latch");
        assert!(last_iter_latch, "latch must arm for next iter");
    }

    /// When the relative step exceeds `10·eps`, the flag must clear and
    /// the latch must reset.
    #[test]
    fn test_detect_tiny_step_clears_when_step_grows() {
        let mut state = minimal_state(1, 0);
        state.x = vec![1.0];
        state.dx = vec![0.5];

        let opts = SolverOptions::default();
        let mut mu_state = MuState::new();
        mu_state.tiny_step = true;
        let mut filter = crate::filter::Filter::new(1e10);
        let mut last_iter_latch = true;

        detect_tiny_step(
            &mut state, &opts, &mut mu_state, &mut filter,
            &mut last_iter_latch, 0.0,
        );

        assert!(!mu_state.tiny_step, "tiny_step must clear on a real step");
        assert!(!last_iter_latch, "latch must clear when detection fails");
    }

    /// A tiny x-step combined with a *large* dual step is still detected
    /// (Ipopt's DetectTinyStep doesn't include Δy), but the latch must
    /// NOT arm — so the flag won't fire on the next iter
    /// (IpBacktrackingLineSearch.cpp:421-424).
    #[test]
    fn test_detect_tiny_step_dy_only_gates_latch() {
        let mut state = minimal_state(1, 1);
        state.x = vec![1.0];
        state.dx = vec![1e-20];
        state.set_y_combined(&[0.0]);
        state.set_dy_combined(&[1.0]); // dy_amax = 1.0, well above default 1e-2

        let opts = SolverOptions::default();
        let mut mu_state = MuState::new();
        let mut filter = crate::filter::Filter::new(1e10);
        let mut last_iter_latch = true; // prior iter armed

        detect_tiny_step(
            &mut state, &opts, &mut mu_state, &mut filter,
            &mut last_iter_latch, 0.0,
        );

        // Detection IS tiny, prior latch IS true → flag fires this iter.
        assert!(mu_state.tiny_step,
            "tiny_step fires on detection-tiny + prior latch (Δy not in detection)");
        // But the latch must NOT re-arm because Δy is too large.
        assert!(!last_iter_latch,
            "Δy ≥ tiny_step_y_tol must clear the latch for next iter");
    }

    /// Detection requires `cviol ≤ 1e-4`
    /// (IpBacktrackingLineSearch.cpp:1269-1273). High primal infeasibility
    /// must block the latch from arming and the flag from firing.
    #[test]
    fn test_detect_tiny_step_blocked_by_cviol() {
        let mut state = minimal_state(1, 0);
        state.x = vec![1.0];
        state.dx = vec![1e-20];

        let opts = SolverOptions::default();
        let mut mu_state = MuState::new();
        let mut filter = crate::filter::Filter::new(1e10);
        let mut last_iter_latch = true;

        // primal_inf = 1.0 > 1e-4 → detection should fail.
        detect_tiny_step(
            &mut state, &opts, &mut mu_state, &mut filter,
            &mut last_iter_latch, 1.0,
        );

        assert!(!mu_state.tiny_step,
            "high cviol must block the tiny_step flag");
        assert!(!last_iter_latch,
            "high cviol must clear the latch");
    }

    /// Watchdog fidelity (spec §4): on a watchdog accept the filter is
    /// not augmented. Pin via the raw Filter API.
    #[test]
    fn test_watchdog_accept_does_not_augment_filter() {
        let mut filter = crate::filter::Filter::new(1e10);
        let theta_current = 1.0;
        let phi_current = 5.0;
        let theta_trial = 0.5;
        let phi_trial = 4.0;
        let grad_phi_step = -1.0;

        let (acceptable, _) = filter.check_acceptability(
            theta_current, phi_current, theta_trial, phi_trial,
            grad_phi_step, 1.0, false,
        );
        assert!(acceptable, "trial must pass the filter for the test setup");
        let len_before = filter.len();
        assert_eq!(filter.len(), len_before,
            "watchdog branch must not augment filter on accept");
        let (acceptable_next, _) = filter.check_acceptability(
            theta_current, phi_current,
            theta_current * 0.95, phi_current,
            grad_phi_step, 1.0, false,
        );
        assert!(acceptable_next,
            "filter must still accept near-θ_current iterates after watchdog");
    }

    /// `SolveStatus::StopAtTinyStep` round-trip through `make_result`.
    #[test]
    fn test_stop_at_tiny_step_status_roundtrip() {
        let state = minimal_state(2, 0);
        let result = make_result(&state, SolveStatus::StopAtTinyStep);
        assert_eq!(result.status, SolveStatus::StopAtTinyStep);
        assert_eq!(result.x.len(), 2);
    }

    /// T-MIT-C: scaling mock with explicit gradient and Jacobian rows
    /// for `compute_nlp_scaling` tests. Constraints/obj values are
    /// irrelevant — scaling is computed from `grad` and `jac_rows`/
    /// `jac_vals` only.
    struct ScalingMock {
        n: usize,
        m: usize,
        grad: Vec<f64>,
        jac_rows: Vec<usize>,
        jac_cols: Vec<usize>,
        jac_vals: Vec<f64>,
    }
    impl NlpProblem for ScalingMock {
        fn num_variables(&self) -> usize { self.n }
        fn num_constraints(&self) -> usize { self.m }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            for v in x_l.iter_mut() { *v = f64::NEG_INFINITY; }
            for v in x_u.iter_mut() { *v = f64::INFINITY; }
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            for v in g_l.iter_mut() { *v = 0.0; }
            for v in g_u.iter_mut() { *v = 0.0; }
        }
        fn initial_point(&self, x0: &mut [f64]) { for v in x0.iter_mut() { *v = 0.0; } }
        fn objective(&self, _: &[f64], _: bool, obj: &mut f64) -> bool { *obj = 0.0; true }
        fn gradient(&self, _: &[f64], _: bool, grad: &mut [f64]) -> bool {
            grad.copy_from_slice(&self.grad);
            true
        }
        fn constraints(&self, _: &[f64], _: bool, g: &mut [f64]) -> bool {
            for v in g.iter_mut() { *v = 0.0; } true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (self.jac_rows.clone(), self.jac_cols.clone())
        }
        fn jacobian_values(&self, _: &[f64], _: bool, vals: &mut [f64]) -> bool {
            vals.copy_from_slice(&self.jac_vals);
            true
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
        fn hessian_values(&self, _: &[f64], _: bool, _: f64, _: &[f64], _: &mut [f64]) -> bool { true }
    }

    /// T-MIT-C: `nlp_scaling_method = None` returns identity scales,
    /// then `obj_scaling_factor` multiplies in. Mirrors
    /// `IpNLPScaling.cpp:276` semantics: factor applied even for the
    /// `NoNLPScalingObject` path.
    #[test]
    fn test_nlp_scaling_none_applies_obj_scaling_factor() {
        use crate::options::NlpScalingMethod;
        let p = ScalingMock {
            n: 2, m: 1,
            grad: vec![1e6, 1e6], // would trigger gradient scaling if enabled
            jac_rows: vec![0, 0],
            jac_cols: vec![0, 1],
            jac_vals: vec![1e9, 1e9], // ditto
        };
        let mut opts = SolverOptions::default();
        opts.nlp_scaling_method = NlpScalingMethod::None;
        opts.obj_scaling_factor = -1.0; // canonical "maximize" idiom
        let (os, gs) = compute_nlp_scaling(&p, &opts, &[0.0, 0.0], &p.jac_rows);
        assert_eq!(os, -1.0, "method=None must skip gradient scaling and apply factor");
        assert_eq!(gs, vec![1.0], "method=None must leave constraint scaling identity");
    }

    /// T-MIT-C: gradient-based scaling with default options scales the
    /// objective by `max_gradient / ||grad||_inf` when above threshold.
    #[test]
    fn test_nlp_scaling_gradient_obj_above_threshold() {
        let p = ScalingMock {
            n: 2, m: 0,
            grad: vec![0.0, 1000.0],
            jac_rows: vec![],
            jac_cols: vec![],
            jac_vals: vec![],
        };
        let opts = SolverOptions::default(); // method = Gradient, max=100
        let (os, _) = compute_nlp_scaling(&p, &opts, &[0.0, 0.0], &p.jac_rows);
        assert!((os - 0.1).abs() < 1e-12, "expected 100/1000 = 0.1, got {os}");
    }

    /// T-MIT-C: `nlp_scaling_obj_target_gradient > 0` overrides the
    /// `max_gradient` gate — the objective is rescaled even when the
    /// gradient is below `max_gradient`. Matches
    /// `IpGradientScaling.cpp:108-127`.
    #[test]
    fn test_nlp_scaling_gradient_obj_target_override() {
        let p = ScalingMock {
            n: 1, m: 0,
            grad: vec![10.0], // below max_gradient=100, would normally yield os=1
            jac_rows: vec![], jac_cols: vec![], jac_vals: vec![],
        };
        let mut opts = SolverOptions::default();
        opts.nlp_scaling_obj_target_gradient = 1.0; // force os = 1/10 = 0.1
        let (os, _) = compute_nlp_scaling(&p, &opts, &[0.0], &p.jac_rows);
        assert!((os - 0.1).abs() < 1e-12, "target override should give 0.1, got {os}");
    }

    /// T-MIT-C: `nlp_scaling_constr_target_gradient > 0` makes every
    /// constraint row receive the same scale `target / global_max`.
    #[test]
    fn test_nlp_scaling_gradient_constr_target_override() {
        let p = ScalingMock {
            n: 2, m: 2,
            grad: vec![0.0, 0.0],
            jac_rows: vec![0, 1],
            jac_cols: vec![0, 1],
            jac_vals: vec![1.0, 50.0], // global max = 50, row maxes 1 and 50
        };
        let mut opts = SolverOptions::default();
        opts.nlp_scaling_constr_target_gradient = 5.0; // every row -> 5/50 = 0.1
        let (_, gs) = compute_nlp_scaling(&p, &opts, &[0.0, 0.0], &p.jac_rows);
        assert_eq!(gs.len(), 2);
        assert!((gs[0] - 0.1).abs() < 1e-12, "row 0: expected 0.1, got {}", gs[0]);
        assert!((gs[1] - 0.1).abs() < 1e-12, "row 1: expected 0.1, got {}", gs[1]);
    }

    /// T-MIT-C: `obj_scaling_factor` composes multiplicatively with the
    /// gradient-based result. Mirrors
    /// `StandardScalingBase::DetermineScaling` at `IpNLPScaling.cpp:276`.
    #[test]
    fn test_nlp_scaling_factor_composes_with_gradient() {
        let p = ScalingMock {
            n: 1, m: 0,
            grad: vec![1000.0],
            jac_rows: vec![], jac_cols: vec![], jac_vals: vec![],
        };
        let mut opts = SolverOptions::default();
        opts.obj_scaling_factor = -2.0;
        let (os, _) = compute_nlp_scaling(&p, &opts, &[0.0], &p.jac_rows);
        // Gradient pass would yield 100/1000=0.1, then * -2 = -0.2.
        assert!((os - (-0.2)).abs() < 1e-12, "expected -0.2, got {os}");
    }

    /// T-MIT-C: `nlp_scaling_method = User` reads from
    /// `user_obj_scaling` / `user_g_scaling`, ignoring the gradient.
    #[test]
    fn test_nlp_scaling_user_method_uses_user_values() {
        use crate::options::NlpScalingMethod;
        let p = ScalingMock {
            n: 1, m: 2,
            grad: vec![1e6], // would scale to 1e-4 under Gradient
            jac_rows: vec![0, 1], jac_cols: vec![0, 0], jac_vals: vec![1.0, 1.0],
        };
        let mut opts = SolverOptions::default();
        opts.nlp_scaling_method = NlpScalingMethod::User;
        opts.user_obj_scaling = Some(0.5);
        opts.user_g_scaling = Some(vec![2.0, 3.0]);
        let (os, gs) = compute_nlp_scaling(&p, &opts, &[0.0], &p.jac_rows);
        assert_eq!(os, 0.5);
        assert_eq!(gs, vec![2.0, 3.0]);
    }
}
