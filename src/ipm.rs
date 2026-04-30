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
use crate::linear_solver::{KktMatrix, LinearSolver, SymmetricMatrix};
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
use crate::restoration::{OuterBarrierContext, RestorationPhase};
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
}

/// NLP problem wrapper that applies gradient-based scaling.
///
/// Scales objective by `obj_scaling` and each constraint `i` by `g_scaling[i]`
/// so that the max gradient norm at the initial point is ≤ 100.
/// This matches Ipopt's `nlp_scaling_method = gradient-based`.
struct ScaledProblem<'a, P: NlpProblem> {
    inner: &'a P,
    obj_scaling: f64,
    g_scaling: Vec<f64>,
    jac_rows: Vec<usize>,
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
        self.inner.constraint_bounds(g_l, g_u);
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
}

/// Snapshot of the most recent iterate that satisfied the acceptable-level
/// thresholds (Ipopt's `RestoreAcceptablePoint`). When restoration would
/// otherwise be triggered on a near-feasible iterate (where the resto NLP
/// is ill-defined), the IPM rolls back to this point and exits with
/// `SolveStatus::Acceptable`. Mirrors Ipopt's `STOP_AT_ACCEPTABLE_POINT`
/// path in `IpIpoptAlgorithm.cpp`.
struct IterateSnapshot {
    x: Vec<f64>,
    y: Vec<f64>,
    z_l: Vec<f64>,
    z_u: Vec<f64>,
    v_l: Vec<f64>,
    v_u: Vec<f64>,
    s: Vec<f64>,
    mu: f64,
    obj: f64,
    g: Vec<f64>,
    grad_f: Vec<f64>,
    filter_entries: Vec<FilterEntry>,
    iteration: usize,
}

impl IterateSnapshot {
    fn capture(state: &SolverState, filter: &Filter, iteration: usize) -> Self {
        Self {
            x: state.x.clone(),
            y: state.y.clone(),
            z_l: state.z_l.clone(),
            z_u: state.z_u.clone(),
            v_l: state.v_l.clone(),
            v_u: state.v_u.clone(),
            s: state.s.clone(),
            mu: state.mu,
            obj: state.obj,
            g: state.g.clone(),
            grad_f: state.grad_f.clone(),
            filter_entries: filter.save_entries(),
            iteration,
        }
    }

    fn restore(&self, state: &mut SolverState, filter: &mut Filter) {
        state.x = self.x.clone();
        state.y = self.y.clone();
        state.z_l = self.z_l.clone();
        state.z_u = self.z_u.clone();
        state.v_l = self.v_l.clone();
        state.v_u = self.v_u.clone();
        state.s = self.s.clone();
        state.mu = self.mu;
        state.obj = self.obj;
        state.g = self.g.clone();
        state.grad_f = self.grad_f.clone();
        filter.restore_entries(self.filter_entries.clone());
    }
}

/// Saved state for the watchdog mechanism.
struct WatchdogSavedState {
    x: Vec<f64>,
    y: Vec<f64>,
    z_l: Vec<f64>,
    z_u: Vec<f64>,
    v_l: Vec<f64>,
    v_u: Vec<f64>,
    s: Vec<f64>,
    mu: f64,
    obj: f64,
    g: Vec<f64>,
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
            y: state.y.clone(),
            z_l: state.z_l.clone(),
            z_u: state.z_u.clone(),
            v_l: state.v_l.clone(),
            v_u: state.v_u.clone(),
            s: state.s.clone(),
            mu: state.mu,
            obj: state.obj,
            g: state.g.clone(),
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
        state.y = self.y.clone();
        state.z_l = self.z_l.clone();
        state.z_u = self.z_u.clone();
        state.v_l = self.v_l.clone();
        state.v_u = self.v_u.clone();
        state.s = self.s.clone();
        state.mu = self.mu;
        state.obj = self.obj;
        state.g = self.g.clone();
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
    /// kappa_d damping coefficient (T3.9). Read from
    /// `options.kappa_d` at construction; used by convergence helpers
    /// to add the one-sided-bound damping term to grad_lag_x without
    /// threading `options` through every call site.
    pub kappa_d: f64,
    /// Current primal variables.
    pub x: Vec<f64>,
    /// Current constraint multipliers (lambda/y).
    pub y: Vec<f64>,
    /// Lower bound multipliers.
    pub z_l: Vec<f64>,
    /// Upper bound multipliers.
    pub z_u: Vec<f64>,
    /// Constraint slack lower-bound multipliers (Ipopt's v_L).
    /// v_l[i] > 0 for inequality constraints with finite g_l[i], 0 otherwise.
    pub v_l: Vec<f64>,
    /// Constraint slack upper-bound multipliers (Ipopt's v_U).
    /// v_u[i] > 0 for inequality constraints with finite g_u[i], 0 otherwise.
    pub v_u: Vec<f64>,
    /// Search direction: primal.
    pub dx: Vec<f64>,
    /// Search direction: constraint multipliers.
    pub dy: Vec<f64>,
    /// Search direction: lower bound multipliers.
    pub dz_l: Vec<f64>,
    /// Search direction: upper bound multipliers.
    pub dz_u: Vec<f64>,
    /// Search direction: slack lower-bound multipliers (Ipopt's dv_L).
    pub dv_l: Vec<f64>,
    /// Search direction: slack upper-bound multipliers (Ipopt's dv_U).
    pub dv_u: Vec<f64>,
    /// Explicit slack iterate (Ipopt's `s`). For inequality rows: pushed to interior
    /// of `[g_l, g_u]` at init via slack_bound_push/slack_bound_frac, then advanced
    /// each iteration by `s ← s + α_p · ds`. For equality rows: held at `g_l[i]`
    /// as a sentinel; consumers MUST skip equality rows (FTB, barrier, Σ_s).
    /// Source: Ipopt 3.14 IpIteratesVector.hpp slot 1.
    pub s: Vec<f64>,
    /// Search direction for slack iterate (Ipopt's `delta_s`).
    /// Computed by `recover_ds`: `ds[i] = (J·dx)[i] + (g[i] - s[i]) - δ_d·dy[i]`
    /// for inequality rows, 0 for equalities. Source: IpStdAugSystemSolver.cpp:431-465.
    pub ds: Vec<f64>,
    /// Barrier parameter.
    pub mu: f64,
    /// Primal step size.
    pub alpha_primal: f64,
    /// Dual step size.
    pub alpha_dual: f64,
    /// Iteration counter.
    pub iter: usize,
    /// Variable lower bounds.
    pub x_l: Vec<f64>,
    /// Variable upper bounds.
    pub x_u: Vec<f64>,
    /// Constraint lower bounds.
    pub g_l: Vec<f64>,
    /// Constraint upper bounds.
    pub g_u: Vec<f64>,
    /// Number of variables.
    pub n: usize,
    /// Number of constraints.
    pub m: usize,
    /// Current objective value.
    pub obj: f64,
    /// Current gradient.
    pub grad_f: Vec<f64>,
    /// Current constraint values.
    pub g: Vec<f64>,
    /// Jacobian structure and values.
    pub jac_rows: Vec<usize>,
    pub jac_cols: Vec<usize>,
    pub jac_vals: Vec<f64>,
    /// Hessian structure and values.
    pub hess_rows: Vec<usize>,
    pub hess_cols: Vec<usize>,
    pub hess_vals: Vec<f64>,
    /// Consecutive acceptable iterations.
    pub consecutive_acceptable: usize,
    /// Objective scaling factor (for NLP scaling / result unscaling).
    pub obj_scaling: f64,
    /// Constraint scaling factors (for NLP scaling / result unscaling).
    pub g_scaling: Vec<f64>,
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
    z_l: Vec<f64>,
    z_u: Vec<f64>,
    v_l: Vec<f64>,
    v_u: Vec<f64>,
}

impl MuState {
    fn new() -> Self {
        Self {
            mode: MuMode::Free,
            ref_vals: Vec::with_capacity(8),
            num_refs_max: 4,
            refs_red_fact: 0.999,
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

    /// Compute ∇_x L = ∇f + J^T λ
    fn compute_lagrangian_gradient(
        grad_f: &[f64],
        jac_rows: &[usize],
        jac_cols: &[usize],
        jac_vals: &[f64],
        lambda: &[f64],
        n: usize,
    ) -> Vec<f64> {
        let mut lag_grad = grad_f.to_vec();
        for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
            // J^T λ: column `col` of J^T gets contribution from row `row`
            if col < n {
                lag_grad[col] += jac_vals[idx] * lambda[row];
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
        sentinel_bounds_to_infinity(&mut x_l, &mut x_u, options);
        sentinel_bounds_to_infinity(&mut g_l, &mut g_u, options);

        // Ipopt's bound_relax_factor: widen every finite variable AND
        // constraint bound outward by min(constr_viol_tol, factor·max(|b|,1)).
        // Mirrors IpOrigIpoptNLP.cpp:355-358. Must run AFTER infinity
        // sentinels (so we don't relax 1e30) and BEFORE bound_push /
        // fixed-variable handling.
        apply_bound_relax_factor(
            &mut x_l, &mut x_u,
            options.bound_relax_factor, options.constr_viol_tol,
        );
        apply_bound_relax_factor(
            &mut g_l, &mut g_u,
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
            problem, options, &x, &z_l, &z_u, &jac_rows, &jac_cols, &g_l, &g_u, n, m, jac_nnz,
        );

        if options.warm_start {
            apply_warm_start_multipliers(problem, &mut y, &mut z_l, &mut z_u);
        }

        let m_eq = (0..m).filter(|&i| g_l[i] == g_u[i]).count();
        let is_square = m == n || m_eq == n;

        Self {
            kappa_d: options.kappa_d,
            x,
            y,
            z_l,
            z_u,
            v_l: vec![0.0; m],
            v_u: vec![0.0; m],
            dx: vec![0.0; n],
            dy: vec![0.0; m],
            dz_l: vec![0.0; n],
            dz_u: vec![0.0; n],
            dv_l: vec![0.0; m],
            dv_u: vec![0.0; m],
            // Slack iterate `s` and step `ds`. At construction `s[i] = g_l[i]` for
            // equality rows (sentinel) and `s[i] = 0.0` for inequality rows; the
            // proper push-to-interior init runs in `initialize_slack_iterate` (B1.2).
            s: vec![0.0; m],
            ds: vec![0.0; m],

            mu: initial_mu,
            alpha_primal: 0.0,
            alpha_dual: 0.0,
            iter: 0,
            x_l,
            x_u,
            g_l,
            g_u,
            n,
            m,
            obj: 0.0,
            grad_f: vec![0.0; n],
            g: vec![0.0; m],
            jac_rows,
            jac_cols,
            jac_vals: vec![0.0; jac_nnz],
            hess_rows,
            hess_cols,
            hess_vals: vec![0.0; hess_nnz],
            consecutive_acceptable: 0,
            obj_scaling: 1.0,
            g_scaling: vec![1.0; m],
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
        self.n_obj_evals += 1;
        if !problem.objective(&self.x, new_x, &mut self.obj) { return false; }
        if !self.obj.is_finite() { return false; }
        self.n_grad_evals += 1;
        if !problem.gradient(&self.x, false, &mut self.grad_f) { return false; }
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
            if !problem.constraints(&self.x, false, &mut self.g) { return false; }
            self.n_jac_evals += 1;
            if !problem.jacobian_values(&self.x, false, &mut self.jac_vals) { return false; }
        }
        self.x_last_eval.copy_from_slice(&self.x);
        if skip_hessian {
            return true;
        }
        self.n_hess_evals += 1;
        if let Some(flags) = linear_constraints {
            let mut lambda_for_hess = self.y.clone();
            for (i, &is_lin) in flags.iter().enumerate() {
                if is_lin {
                    lambda_for_hess[i] = 0.0;
                }
            }
            if !problem.hessian_values(&self.x, false, obj_factor, &lambda_for_hess, &mut self.hess_vals) { return false; }
        } else {
            if !problem.hessian_values(&self.x, false, obj_factor, &self.y, &mut self.hess_vals) { return false; }
        }
        true
    }

    /// Compute the barrier objective:
    /// f(x) - mu * sum(ln(x_i - x_l_i) + ln(x_u_i - x_i))
    /// Optionally includes constraint slack log-barriers when enabled.
    fn barrier_objective(&self, options: &SolverOptions) -> f64 {
        compute_barrier_phi(
            self.obj, &self.x, &self.g, self,
            self.n, self.m, options.constraint_slack_barrier,
            options.kappa_d,
        )
    }

    /// Compute constraint violation (theta).
    fn constraint_violation(&self) -> f64 {
        convergence::primal_infeasibility(&self.g, &self.g_l, &self.g_u)
    }

    /// Compute the directional derivative of the barrier objective along the search direction.
    ///
    /// ∇φ·dx = (∇f - μ/(x-x_l) + μ/(x_u-x))·dx
    /// Optionally includes constraint slack derivative terms when enabled.
    fn barrier_directional_derivative(&self, options: &SolverOptions) -> f64 {
        let mut grad_phi_dx = 0.0;
        let kappa_d = options.kappa_d;
        for i in 0..self.n {
            let l_fin = self.x_l[i].is_finite();
            let u_fin = self.x_u[i].is_finite();
            let mut grad_phi_i = self.grad_f[i];
            if l_fin {
                grad_phi_i -= self.mu / slack_xl(self, i);
            }
            if u_fin {
                grad_phi_i += self.mu / slack_xu(self, i);
            }
            // kappa_d damping gradient: +kappa_d*mu if only x_l finite
            // (slack = x - x_l), -kappa_d*mu if only x_u finite
            // (slack = x_u - x).
            if kappa_d > 0.0 && (l_fin ^ u_fin) {
                if l_fin {
                    grad_phi_i += kappa_d * self.mu;
                } else {
                    grad_phi_i -= kappa_d * self.mu;
                }
            }
            grad_phi_dx += grad_phi_i * self.dx[i];
        }
        if options.constraint_slack_barrier && self.m > 0 {
            // Compute J * dx (directional change in constraints)
            let mut jdx = vec![0.0; self.m];
            for (idx, (&row, &col)) in
                self.jac_rows.iter().zip(self.jac_cols.iter()).enumerate()
            {
                jdx[row] += self.jac_vals[idx] * self.dx[col];
            }
            for i in 0..self.m {
                if constraint_is_equality(self, i) {
                    continue;
                }
                if self.g_l[i].is_finite() {
                    let slack = self.g[i] - self.g_l[i];
                    if slack > self.mu * 1e-2 {
                        grad_phi_dx -= self.mu * jdx[i] / slack;
                    }
                }
                if self.g_u[i].is_finite() {
                    let slack = self.g_u[i] - self.g[i];
                    if slack > self.mu * 1e-2 {
                        grad_phi_dx += self.mu * jdx[i] / slack;
                    }
                }
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
        SolveStatus::Unbounded => "Diverging Iterates -- Problem May Be Unbounded.",
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
        let l_fin = state.x_l[i].is_finite();
        let u_fin = state.x_u[i].is_finite();
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
        let l_fin = state.g_l[i].is_finite();
        let u_fin = state.g_u[i].is_finite();
        let is_eq = l_fin && u_fin && (state.g_l[i] - state.g_u[i]).abs() < 1e-14;
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
    let mut jac_nnz_eq = 0usize;
    let mut jac_nnz_ineq = 0usize;
    for &r in &state.jac_rows {
        if r < m && row_is_eq[r] { jac_nnz_eq += 1; } else { jac_nnz_ineq += 1; }
    }
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
            &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
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
    Return(SolveResult),
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
    g: &[f64],
    state: &SolverState,
    n: usize,
    m: usize,
    constraint_slack_barrier: bool,
    kappa_d: f64,
) -> f64 {
    let mut phi = obj;
    for i in 0..n {
        let l_fin = state.x_l[i].is_finite();
        let u_fin = state.x_u[i].is_finite();
        if l_fin {
            let slack = (x[i] - state.x_l[i]).max(1e-20);
            phi -= state.mu * slack.ln();
        }
        if u_fin {
            let slack = (state.x_u[i] - x[i]).max(1e-20);
            phi -= state.mu * slack.ln();
        }
        // kappa_d damping: penalize drift toward the open side for
        // variables with exactly one finite bound. Mirrors Ipopt 3.14
        // CalcBarrierTerm in IpIpoptCalculatedQuantities.cpp.
        if kappa_d > 0.0 && (l_fin ^ u_fin) {
            let s_oneside = if l_fin {
                (x[i] - state.x_l[i]).max(0.0)
            } else {
                (state.x_u[i] - x[i]).max(0.0)
            };
            phi += kappa_d * state.mu * s_oneside;
        }
    }
    if constraint_slack_barrier {
        for i in 0..m {
            if constraint_is_equality(state, i) {
                continue;
            }
            if state.g_l[i].is_finite() {
                let slack = g[i] - state.g_l[i];
                if slack > state.mu * 1e-2 {
                    phi -= state.mu * slack.ln();
                }
            }
            if state.g_u[i].is_finite() {
                let slack = state.g_u[i] - g[i];
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

    let mut obj_trial = f64::INFINITY;
    state.n_obj_evals += 1;
    let obj_ok = problem.objective(&x_trial, true, &mut obj_trial);
    let mut g_trial = vec![0.0; m];
    let constr_ok = if m > 0 {
        state.n_constr_evals += 1;
        problem.constraints(&x_trial, true, &mut g_trial)
    } else {
        true
    };

    if !obj_ok || !constr_ok || obj_trial.is_nan() || obj_trial.is_infinite()
        || g_trial.iter().any(|v| v.is_nan() || v.is_infinite())
    {
        return None;
    }

    let theta_trial = theta_for_g(state, &g_trial);
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
    start_time: Instant,
    early_timeout: f64,
    trace_meta: &mut TraceMetadata,
    ls_steps: &mut usize,
    aug_solver: &mut dyn LinearSolver,
    aug_kkt: &crate::kkt_aug::AugKktSystem,
) -> LineSearchOutcome {
    let mut alpha = alpha_primal_max;
    let mut step_accepted = false;
    *ls_steps = 0;

    for _ls_iter in 0..40 {
        // Intra-iteration early stall check (scaled by problem size).
        // Square problems can have legitimately slow first iterations
        // (mirrors IpBacktrackingLineSearch.cpp:276-280), so we never
        // declare infeasibility on time alone in that branch.
        if !state.is_square
            && iteration < 3
            && options.early_stall_timeout > 0.0
            && start_time.elapsed().as_secs_f64() > early_timeout
        {
            return LineSearchOutcome::Return(make_result(state, SolveStatus::NumericalError));
        }
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

        // Barrier objective at trial
        let phi_trial = compute_barrier_phi(
            obj_trial, &x_trial, &g_trial, state, n, m, options.constraint_slack_barrier,
            options.kappa_d,
        );

        let (acceptable, used_switching) = filter.check_acceptability(
            theta_current,
            phi_current,
            theta_trial,
            phi_trial,
            grad_phi_step,
            alpha,
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
            if !used_switching {
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
    let n = state.n;
    let m = state.m;
    for i in 0..n {
        if state.x_l[i].is_finite() {
            state.z_l[i] = (state.z_l[i] + alpha_d * state.dz_l[i]).max(1e-20);
        }
        if state.x_u[i].is_finite() {
            state.z_u[i] = (state.z_u[i] + alpha_d * state.dz_u[i]).max(1e-20);
        }
    }
    // Slack-bound multipliers v_L, v_U: same Newton update as z_L, z_U
    // (Ipopt's `IpIpoptAlg.cpp:652-770` advances all four blocks with the
    // shared α_dual). Skip equality constraints (v stays at 0 there).
    for i in 0..m {
        if state.g_l[i].is_finite() && state.g_u[i].is_finite()
            && (state.g_l[i] - state.g_u[i]).abs() < 1e-14
        {
            continue;
        }
        if state.g_l[i].is_finite() {
            state.v_l[i] = (state.v_l[i] + alpha_d * state.dv_l[i]).max(1e-20);
        }
        if state.g_u[i].is_finite() {
            state.v_u[i] = (state.v_u[i] + alpha_d * state.dv_u[i]).max(1e-20);
        }
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
    let n = state.n;
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
    for i in 0..n {
        if state.x_l[i].is_finite() {
            let s_l = slack_xl(state, i);
            let z_lo = mu_ks / (kappa_sigma * s_l);
            let z_hi = kappa_sigma * mu_ks / s_l;
            state.z_l[i] = state.z_l[i].clamp(z_lo, z_hi);
        }
        if state.x_u[i].is_finite() {
            let s_u = slack_xu(state, i);
            let z_lo = mu_ks / (kappa_sigma * s_u);
            let z_hi = kappa_sigma * mu_ks / s_u;
            state.z_u[i] = state.z_u[i].clamp(z_lo, z_hi);
        }
    }
    // Apply the same κ_σ band to the slack-bound multipliers v_L, v_U
    // (Ipopt's `correct_bound_multiplier` runs over ALL FOUR blocks,
    // `IpIpoptAlg.cpp:721-758`). Skip equality constraints.
    for i in 0..m {
        if state.g_l[i].is_finite() && state.g_u[i].is_finite()
            && (state.g_l[i] - state.g_u[i]).abs() < 1e-14
        {
            continue;
        }
        if state.g_l[i].is_finite() {
            let s_l = slack_gl(state, i);
            let v_lo = mu_ks / (kappa_sigma * s_l);
            let v_hi = kappa_sigma * mu_ks / s_l;
            state.v_l[i] = state.v_l[i].clamp(v_lo, v_hi);
        }
        if state.g_u[i].is_finite() {
            let s_u = slack_gu(state, i);
            let v_lo = mu_ks / (kappa_sigma * s_u);
            let v_hi = kappa_sigma * mu_ks / s_u;
            state.v_u[i] = state.v_u[i].clamp(v_lo, v_hi);
        }
    }
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
    for i in 0..n {
        if state.x_l[i].is_finite() {
            let s_l = state.x[i] - state.x_l[i];
            if s_l < s_min {
                let z = state.z_l[i];
                let from_mu = if z > 0.0 { mu / z } else { f64::INFINITY };
                let cap = slack_move * state.x_l[i].abs().max(1.0) + s_l;
                let new_s = from_mu.max(s_min).min(cap);
                state.x_l[i] -= new_s - s_l;
                adjusted += 1;
            }
        }
        if state.x_u[i].is_finite() {
            let s_u = state.x_u[i] - state.x[i];
            if s_u < s_min {
                let z = state.z_u[i];
                let from_mu = if z > 0.0 { mu / z } else { f64::INFINITY };
                let cap = slack_move * state.x_u[i].abs().max(1.0) + s_u;
                let new_s = from_mu.max(s_min).min(cap);
                state.x_u[i] += new_s - s_u;
                adjusted += 1;
            }
        }
    }
    // B-cross8: extend slack_move to the constraint slack iterate `s`
    // against `[g_l, g_u]` (Ipopt's CalculateSafeSlack runs over all
    // four slack blocks: x_L, x_U, s_L, s_U; see
    // IpIpoptCalculatedQuantities.cpp:455-537). Skip equality rows
    // (their s is held at the equality value as a sentinel).
    for i in 0..m {
        let l_fin = state.g_l[i].is_finite();
        let u_fin = state.g_u[i].is_finite();
        if l_fin && u_fin && (state.g_l[i] - state.g_u[i]).abs() < 1e-14 {
            continue;
        }
        if l_fin {
            let s_l = state.s[i] - state.g_l[i];
            if s_l < s_min {
                let v = state.v_l[i];
                let from_mu = if v > 0.0 { mu / v } else { f64::INFINITY };
                let cap = slack_move * state.g_l[i].abs().max(1.0) + s_l;
                let new_s = from_mu.max(s_min).min(cap);
                state.g_l[i] -= new_s - s_l;
                adjusted += 1;
            }
        }
        if u_fin {
            let s_u = state.g_u[i] - state.s[i];
            if s_u < s_min {
                let v = state.v_u[i];
                let from_mu = if v > 0.0 { mu / v } else { f64::INFINITY };
                let cap = slack_move * state.g_u[i].abs().max(1.0) + s_u;
                let new_s = from_mu.max(s_min).min(cap);
                state.g_u[i] += new_s - s_u;
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
    for i in 0..state.m {
        state.y[i] += alpha_y * state.dy[i];
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
    let mut grad_f_trial = vec![0.0_f64; n];
    if !problem.gradient(&state.x, true, &mut grad_f_trial) {
        return alpha_p; // graceful fallback
    }
    let mut jac_trial = vec![0.0_f64; state.jac_vals.len()];
    if !problem.jacobian_values(&state.x, false, &mut jac_trial) {
        return alpha_p;
    }

    // r_x = grad_lag_x(trial) = grad_f_trial + J(trial)^T · y_curr − z_L + z_U.
    // (The kappa_d damping is omitted here to match Ipopt's formula on
    // line 977 which uses the raw `grad_lag_x` without the damping
    // term — Ipopt resets y_c/y_d to current via `BackupCurrent` at
    // 975-980 then queries `curr_grad_lag_x_amax_func()`.)
    let mut r_x = grad_f_trial.clone();
    for (idx, (&row, &col)) in state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate() {
        r_x[col] += jac_trial[idx] * state.y[row];
    }
    for i in 0..n {
        r_x[i] -= state.z_l[i];
        r_x[i] += state.z_u[i];
    }

    // r_s and dy_d are zero on equality rows; on inequality rows
    // r_s_i = -y_i - v_l_i + v_u_i and dy_d_i = state.dy[i].
    let mut r_s = vec![0.0_f64; m];
    let mut dy_d = vec![0.0_f64; m];
    for i in 0..m {
        if !constraint_is_equality(state, i) {
            r_s[i] = -state.y[i] - state.v_l[i] + state.v_u[i];
            dy_d[i] = state.dy[i];
        }
    }

    // Jt_dy = J(trial)^T · dy (for all rows; equality contribution
    // automatically participates in the y_c half of the formula).
    let mut jt_dy = vec![0.0_f64; n];
    for (idx, (&row, &col)) in state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate() {
        jt_dy[col] += jac_trial[idx] * state.dy[row];
    }

    // a = ||Jt_dy||² + ||dy_d||²
    let a: f64 = jt_dy.iter().map(|v| v * v).sum::<f64>()
        + dy_d.iter().map(|v| v * v).sum::<f64>();
    // b = r_x · Jt_dy − r_s · dy_d
    let b: f64 = r_x.iter().zip(jt_dy.iter()).map(|(rx, jd)| rx * jd).sum::<f64>()
        - r_s.iter().zip(dy_d.iter()).map(|(rs, dd)| rs * dd).sum::<f64>();

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
    let alpha_y = match options.alpha_for_y {
        AlphaForY::Primal => alpha_p,
        AlphaForY::BoundMult => alpha_d,
        AlphaForY::Min => alpha_p.min(alpha_d),
        AlphaForY::Max => alpha_p.max(alpha_d),
        AlphaForY::Full => 1.0,
        AlphaForY::PrimalAndFull => {
            if alpha_p >= options.alpha_for_y_tol { 1.0 } else { alpha_p }
        }
        AlphaForY::DualAndFull => {
            if alpha_d >= options.alpha_for_y_tol { 1.0 } else { alpha_d }
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
    let made_progress = filter.is_acceptable(theta_now, phi_now)
        && (theta_now < (1.0 - 1e-5) * saved.theta
            || phi_now < saved.phi - 1e-5 * saved.theta);

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
/// disabled, KKT dim > 50000, or the early-stall timeout has nearly
/// elapsed within the first 3 iterations). Returns true when restoration
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
    iteration: usize,
    fail_count: usize,
    n: usize,
    m: usize,
    start_time: Instant,
    early_timeout: f64,
    theta_current: f64,
) -> bool {
    let kkt_dim = n + m;
    let skip_nlp_restoration = !state.is_square
        && iteration < 3
        && options.early_stall_timeout > 0.0
        && start_time.elapsed().as_secs_f64() > early_timeout * 0.5;
    if !((fail_count == 2 || fail_count == 4)
        && !options.disable_nlp_restoration
        && kkt_dim <= 50000
        && !skip_nlp_restoration)
    {
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

/// Run the fast Gauss–Newton restoration. Returns true if GN restoration
/// succeeded and `apply_restoration_success` was invoked (caller should
/// short-circuit the cascade with `Continue`); false otherwise (caller
/// proceeds to the recovery / NLP-restoration fallbacks).
#[allow(clippy::too_many_arguments)]
fn try_gn_restoration<P: NlpProblem>(
    state: &mut SolverState,
    problem: &P,
    options: &SolverOptions,
    filter: &mut Filter,
    mu_state: &mut MuState,
    lbfgs_state: &mut Option<LbfgsIpmState>,
    lbfgs_mode: bool,
    linear_constraints: Option<&[bool]>,
    restoration: &mut RestorationPhase,
    n: usize,
    m: usize,
    deadline: Option<Instant>,
) -> bool {
    // T3.12: outer-NLP barrier context for TestOrigProgress.
    let outer_ctx = OuterBarrierContext {
        mu_outer: state.mu,
        x_l: &state.x_l,
        x_u: &state.x_u,
    };
    let (x_rest, gn_success) = restoration.restore_with_outer(
        &state.x,
        &state.x_l,
        &state.x_u,
        &state.g_l,
        &state.g_u,
        &state.jac_rows,
        &state.jac_cols,
        n,
        m,
        options,
        &|theta, phi| filter.is_acceptable(theta, phi),
        &|x_eval, g_out| problem.constraints(x_eval, true, g_out),
        &|x_eval, jac_out| problem.jacobian_values(x_eval, true, jac_out),
        Some(&|x_eval: &[f64], obj_out: &mut f64| problem.objective(x_eval, true, obj_out)),
        deadline,
        Some(&outer_ctx),
    );

    if gn_success {
        state.diagnostics.restoration_count += 1;
        // Gauss-Newton restoration is primal-only; no z is returned.
        // Pass None so apply_restoration_success falls back to the
        // μ/slack reset (T0.8 only applies when the resto NLP path
        // produced fresh bound multipliers). T0.9: the function now
        // returns false if the filter rejects the restored point;
        // propagate that to fall through to recovery.
        apply_restoration_success(
            state, filter, mu_state, options, n, m, problem, &x_rest, None,
            linear_constraints, lbfgs_mode, lbfgs_state,
        )
    } else {
        false
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
    restoration: &mut RestorationPhase,
    iteration: usize,
    n: usize,
    m: usize,
    start_time: Instant,
    deadline: Option<Instant>,
    early_timeout: f64,
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

    // Phase 1: Fast GN restoration
    log::debug!("Line search failed at iteration {}, entering restoration", iteration);

    if try_gn_restoration(
        state, problem, options, filter, mu_state, lbfgs_state, lbfgs_mode,
        linear_constraints, restoration, n, m, deadline,
    ) {
        return RestorationCascadeDecision::Continue;
    }

    // GN restoration failed — recovery logic with NLP restoration as last resort.
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
        early_timeout, theta_current,
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
/// `x_pre_step`) up to 5 times; if that fails, invoke the restoration
/// phase; if restoration also fails, return `NumericalError`.
///
/// On success (or successful recovery via α-halving or restoration)
/// returns `Proceed` / `Continue` for the main loop's control flow.
fn reevaluate_after_step<P: NlpProblem>(
    state: &mut SolverState,
    problem: &P,
    options: &SolverOptions,
    lbfgs_state: &mut Option<LbfgsIpmState>,
    filter: &mut Filter,
    restoration: &mut RestorationPhase,
    timings: &mut PhaseTimings,
    x_pre_step: &[f64],
    linear_constraints: Option<&[bool]>,
    lbfgs_mode: bool,
    deadline: Option<Instant>,
) -> PostStepEvalDecision {
    let n = state.n;
    let m = state.m;
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

    // α halving exhausted: try restoration.
    // T3.12: outer-NLP barrier context for TestOrigProgress.
    let outer_ctx = OuterBarrierContext {
        mu_outer: state.mu,
        x_l: &state.x_l,
        x_u: &state.x_u,
    };
    let (x_rest, success) = restoration.restore_with_outer(
        &state.x, &state.x_l, &state.x_u, &state.g_l, &state.g_u,
        &state.jac_rows, &state.jac_cols, n, m, options,
        &|theta, phi| filter.is_acceptable(theta, phi),
        &|x_eval, g_out| problem.constraints(x_eval, true, g_out),
        &|x_eval, jac_out| problem.jacobian_values(x_eval, true, jac_out),
        Some(&|x_eval: &[f64], obj_out: &mut f64| problem.objective(x_eval, true, obj_out)),
        deadline,
        Some(&outer_ctx),
    );
    if success {
        state.x = x_rest;
        state.alpha_primal = 0.0;
        // T3.25 follow-up: hand-rolled `state.x = ...` outside the line
        // search choke point. Bump atags and drop the cache.
        state.bump_all_kkt_atags();
        state.factor_cache.invalidate();
        if state.evaluate_with_linear(problem, 1.0, linear_constraints, lbfgs_mode) {
            update_lbfgs_hessian(lbfgs_state, state);
            return PostStepEvalDecision::Continue;
        }
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
    y: Vec<f64>,
    z_l: Vec<f64>,
    z_u: Vec<f64>,
    s: Vec<f64>,
    obj: f64,
    g: Vec<f64>,
    grad_f: Vec<f64>,
    jac_vals: Vec<f64>,
    alpha_primal: f64,
}
impl SoftRestoSnapshot {
    fn take(state: &SolverState) -> Self {
        Self {
            x: state.x.clone(),
            y: state.y.clone(),
            z_l: state.z_l.clone(),
            z_u: state.z_u.clone(),
            s: state.s.clone(),
            obj: state.obj,
            g: state.g.clone(),
            grad_f: state.grad_f.clone(),
            jac_vals: state.jac_vals.clone(),
            alpha_primal: state.alpha_primal,
        }
    }
    fn restore(self, state: &mut SolverState) {
        state.x = self.x;
        state.y = self.y;
        state.z_l = self.z_l;
        state.z_u = self.z_u;
        state.s = self.s;
        state.obj = self.obj;
        state.g = self.g;
        state.grad_f = self.grad_f;
        state.jac_vals = self.jac_vals;
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
        state.y[i] += alpha_d * state.dy[i];
    }
    for i in 0..n {
        state.z_l[i] = (state.z_l[i] + alpha_d * state.dz_l[i]).max(0.0);
        state.z_u[i] = (state.z_u[i] + alpha_d * state.dz_u[i]).max(0.0);
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

    let theta_trial = theta_for_g(state, &state.g);
    let phi_trial = compute_barrier_phi(
        state.obj, &state.x, &state.g, state, n, m, options.constraint_slack_barrier,
        options.kappa_d,
    );

    let filter_ok = filter.is_acceptable(theta_trial, phi_trial);
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

    let alpha_dual_z = fraction_to_boundary_dual_z_min(state, &state.dz_l, &state.dz_u, tau);
    let alpha_dual_v = fraction_to_boundary_dual_v_min(state, &state.dv_l, &state.dv_u, tau);
    let alpha_dual_max = alpha_dual_z.min(alpha_dual_v);

    if std::env::var("RIPOPT_TRACE_STEP").is_ok() {
        let dvl_inf = state.dv_l.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let dvu_inf = state.dv_u.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let vl_min = state.v_l.iter().cloned().fold(f64::INFINITY, f64::min);
        let vu_min = state.v_u.iter().cloned().fold(f64::INFINITY, f64::min);
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
            if state.x_l[i].is_finite() && state.dx[i] < 0.0 {
                let slack = state.x[i] - state.x_l[i];
                let a = -tau * slack / state.dx[i];
                if a < lim_alpha {
                    lim_alpha = a;
                    lim_idx = i;
                    lim_side = "L";
                    lim_block = "x";
                }
            }
            if state.x_u[i].is_finite() && state.dx[i] > 0.0 {
                let slack = state.x_u[i] - state.x[i];
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
            let l_fin = state.g_l[i].is_finite();
            let u_fin = state.g_u[i].is_finite();
            if l_fin && u_fin && (state.g_l[i] - state.g_u[i]).abs() < 1e-14 {
                continue;
            }
            if l_fin && state.ds[i] < 0.0 {
                let slack = (state.s[i] - state.g_l[i]).max(0.0);
                let a = -tau * slack / state.ds[i];
                if a < lim_alpha {
                    lim_alpha = a;
                    lim_idx = i;
                    lim_side = "L";
                    lim_block = "s";
                }
            }
            if u_fin && state.ds[i] > 0.0 {
                let slack = (state.g_u[i] - state.s[i]).max(0.0);
                let a = tau * slack / state.ds[i];
                if a < lim_alpha {
                    lim_alpha = a;
                    lim_idx = i;
                    lim_side = "U";
                    lim_block = "s";
                }
            }
        }
        let dx_inf = state.dx.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let dy_inf = state.dy.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let dzl_inf = state.dz_l.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let dzu_inf = state.dz_u.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        if lim_idx != usize::MAX {
            let (xv, xb, dxv, slack) = if lim_block == "s" {
                let xv = state.s[lim_idx];
                let xb = if lim_side == "L" { state.g_l[lim_idx] } else { state.g_u[lim_idx] };
                (xv, xb, state.ds[lim_idx], (xv - xb).abs())
            } else {
                let xv = state.x[lim_idx];
                let xb = if lim_side == "L" { state.x_l[lim_idx] } else { state.x_u[lim_idx] };
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

/// Ipopt-style tiny-step detection: set the `tiny_step` flag when the
/// raw search direction is at machine-precision noise and the dual step
/// is also small.
///
/// Mirrors `IpBacktrackingLineSearch.cpp::DetectTinyStep` (Ipopt 3.14,
/// lines 1219-1278) and the latch logic at lines 407-434:
/// - Threshold is `max_i |Δx_i| / (1 + |x_i|) < 10·eps ≈ 2.22e-15` on
///   the raw direction (NOT scaled by `α_primal_max` — that is a
///   ripopt-specific bug that mis-flagged normal short steps near a
///   bound as "tiny").
/// - The latch (`tiny_step_flag`) requires the dual step to also be
///   small: `‖Δy‖_∞ < tiny_step_y_tol` (default 1e-2), unscaled
///   (Ipopt's `Max(delta_y_c->Amax(), delta_y_d->Amax())`,
///   `IpBacktrackingLineSearch.cpp:421`). Without this, iterates still
///   making dual progress get latched.
/// - `consecutive_tiny_steps` is the two-iteration counter that mirrors
///   Ipopt's `tiny_step_last_iteration_` latch reset at line 434.
///   Crucially, `mu_state.tiny_step` only goes true when the counter
///   reaches 2 — Ipopt's `Set_tiny_step_flag(true)` at line 410 fires
///   only when the prior iter was also tiny. Latching on iter 1 would
///   force a μ decrease one iteration earlier than Ipopt does.
///
/// The actual `STOP_AT_TINY_STEP` exit is *not* triggered here — it
/// fires from `update_barrier_parameter` when `tiny_step && new_μ == μ`
/// (`IpMonotoneMuUpdate.cpp:158-160`, `IpAdaptiveMuUpdate.cpp:329,377`).
/// The main loop consumes the resulting `pending_tiny_step_exit` flag
/// at the *top* of the next iteration, AFTER `check_convergence` has
/// run, so KKT-clean tiny-step iterates exit Optimal first.
fn detect_tiny_step(
    state: &mut SolverState,
    options: &SolverOptions,
    mu_state: &mut MuState,
    _filter: &mut Filter,
    consecutive_tiny_steps: &mut usize,
    primal_inf: f64,
) {
    let n = state.n;
    let m = state.m;
    let tiny_tol = 10.0 * f64::EPSILON;

    let max_rel_dx: f64 = (0..n)
        .map(|i| state.dx[i].abs() / (state.x[i].abs() + 1.0))
        .fold(0.0f64, f64::max);
    // Ipopt uses raw `Amax(delta_y)` (unscaled L∞ norm) compared
    // directly against `tiny_step_y_tol_`
    // (`IpBacktrackingLineSearch.cpp:421`). Do NOT divide by `1+|y|`.
    let dy_amax: f64 = if m == 0 {
        0.0
    } else {
        state.dy.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
    };

    let x_tiny = max_rel_dx < tiny_tol;
    let y_tiny = dy_amax < options.tiny_step_y_tol;

    if x_tiny && y_tiny && primal_inf < 1e-4 {
        *consecutive_tiny_steps += 1;
        // Ipopt's `Set_tiny_step_flag(true)` (line 410) fires only when
        // the previous iter also latched — i.e. on iter 2+ of a run of
        // consecutive tiny steps. Mirror that with the >= 2 gate.
        mu_state.tiny_step = *consecutive_tiny_steps >= 2;
    } else {
        *consecutive_tiny_steps = 0;
        mu_state.tiny_step = false;
    }
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
    for i in 0..state.n {
        if state.x_l[i].is_finite() {
            min_compl = min_compl.min(slack_xl(state, i) * state.z_l[i]);
        }
        if state.x_u[i].is_finite() {
            min_compl = min_compl.min(slack_xu(state, i) * state.z_u[i]);
        }
    }
    for i in 0..state.m {
        let l_fin = state.g_l[i].is_finite();
        let u_fin = state.g_u[i].is_finite();
        if l_fin && u_fin && (state.g_l[i] - state.g_u[i]).abs() < 1e-14 {
            continue;
        }
        if l_fin {
            min_compl = min_compl.min(slack_gl(state, i) * state.v_l[i]);
        }
        if u_fin {
            min_compl = min_compl.min(slack_gu(state, i) * state.v_u[i]);
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
    let rhs_aff = kkt::affine_predictor_rhs(
        &kkt.rhs, &state.x, &state.x_l, &state.x_u, state.mu, options.kappa_d,
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
        let alpha_p = fraction_to_boundary_primal_x(state, &dx, 1.0)
            .min(fraction_to_boundary_primal_s(state, &ds, 1.0))
            .clamp(0.0, 1.0);
        let alpha_d = fraction_to_boundary_dual_z_min(state, &dz_l, &dz_u, 1.0)
            .min(fraction_to_boundary_dual_v_min(state, &dv_l, &dv_u, 1.0))
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
        for i in 0..n {
            if state.x_l[i].is_finite() {
                let s_plus = (slack_xl(state, i) + alpha_p * dx[i]).max(1e-20);
                let z_plus = (state.z_l[i] + alpha_d * dz_l[i]).max(1e-20);
                compl_max = compl_max.max(s_plus * z_plus);
            }
            if state.x_u[i].is_finite() {
                let s_plus = (slack_xu(state, i) - alpha_p * dx[i]).max(1e-20);
                let z_plus = (state.z_u[i] + alpha_d * dz_u[i]).max(1e-20);
                compl_max = compl_max.max(s_plus * z_plus);
            }
        }
        for i in 0..m {
            let l_fin = state.g_l[i].is_finite();
            let u_fin = state.g_u[i].is_finite();
            if l_fin && u_fin && (state.g_l[i] - state.g_u[i]).abs() < 1e-14 {
                continue;
            }
            if l_fin {
                let s_plus = (slack_gl(state, i) + alpha_p * ds[i]).max(1e-20);
                let v_plus = (state.v_l[i] + alpha_d * dv_l[i]).max(1e-20);
                compl_max = compl_max.max(s_plus * v_plus);
            }
            if u_fin {
                let s_plus = (slack_gu(state, i) - alpha_p * ds[i]).max(1e-20);
                let v_plus = (state.v_u[i] + alpha_d * dv_u[i]).max(1e-20);
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
            for i in 0..n {
                if state.x_l[i].is_finite() {
                    let s_plus = (slack_xl(state, i) + alpha_p * dx[i]).max(1e-20);
                    let z_plus = (state.z_l[i] + alpha_d * dz_l[i]).max(1e-20);
                    let sz = s_plus * z_plus;
                    sum_sz += sz;
                    if sz < min_sz { min_sz = sz; }
                    nb += 1;
                }
                if state.x_u[i].is_finite() {
                    let s_plus = (slack_xu(state, i) - alpha_p * dx[i]).max(1e-20);
                    let z_plus = (state.z_u[i] + alpha_d * dz_u[i]).max(1e-20);
                    let sz = s_plus * z_plus;
                    sum_sz += sz;
                    if sz < min_sz { min_sz = sz; }
                    nb += 1;
                }
            }
            for i in 0..m {
                let l_fin = state.g_l[i].is_finite();
                let u_fin = state.g_u[i].is_finite();
                if l_fin && u_fin && (state.g_l[i] - state.g_u[i]).abs() < 1e-14 {
                    continue;
                }
                if l_fin {
                    let s_plus = (slack_gl(state, i) + alpha_p * ds[i]).max(1e-20);
                    let v_plus = (state.v_l[i] + alpha_d * dv_l[i]).max(1e-20);
                    let sv = s_plus * v_plus;
                    sum_sz += sv;
                    if sv < min_sz { min_sz = sv; }
                    nb += 1;
                }
                if u_fin {
                    let s_plus = (slack_gu(state, i) - alpha_p * ds[i]).max(1e-20);
                    let v_plus = (state.v_u[i] + alpha_d * dv_u[i]).max(1e-20);
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
    // When there are no variable bounds, mu serves no barrier purpose but is
    // still used for KKT regularization and the filter line search. We decrease
    // mu superlinearly (mu^1.5) rather than collapsing it instantly to mu_min,
    // which would destroy filter protection against infeasible steps. This
    // prevents the PENTAGON-type failure where mu=1e-11 at iteration 1 causes
    // the switching condition to accept a step that destroys feasibility.
    let has_bounds = (0..n).any(|i| state.x_l[i].is_finite() || state.x_u[i].is_finite());
    if !has_bounds {
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
            state.y = snap.y;
            state.z_l = snap.z_l;
            state.z_u = snap.z_u;
            state.v_l = snap.v_l;
            state.v_u = snap.v_u;
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
            y: state.y.clone(),
            z_l: state.z_l.clone(),
            z_u: state.z_u.clone(),
            v_l: state.v_l.clone(),
            v_u: state.v_u.clone(),
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
        // T2.3: dropped the ripopt-specific `μ/5` floor that fired when
        // barrier_err > κ_eps·μ; Ipopt's monotone-mode update uses only
        // `mu_min` as the lower clamp.
        let mu_cap = mu_state.mu_max_cap(options, avg_compl);
        state.mu = (avg_compl / options.kappa).clamp(options.mu_min, mu_cap);
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
        // Cap at 4 to bound work.
        const MAX_FAST_DECREASES: usize = 4;
        let mut decreases = 0usize;
        let mut tiny_step = mu_state.tiny_step;
        loop {
            let barrier_err = compute_barrier_error(state);
            let solved = barrier_err <= options.barrier_tol_factor * state.mu;
            if !(solved || tiny_step) { break; }
            let new_mu = (options.mu_linear_decrease_factor * state.mu)
                .min(state.mu.powf(options.mu_superlinear_decrease_power))
                .max(options.mu_min);
            let mu_changed = (new_mu - state.mu).abs() > 1e-20;
            if !mu_changed { break; }
            state.mu = new_mu;
            decreases += 1;
            tiny_step = false;
            log::debug!("Fixed mode: mu decreased to {:.2e}", state.mu);
            if !options.mu_allow_fast_monotone_decrease { break; }
            if decreases >= MAX_FAST_DECREASES { break; }
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
            Some(make_result(state, SolveStatus::Unbounded))
        }
        ConvergenceStatus::NotConverged => None,
    }
}

/// Check wall-clock and early-stall time limits at the top of each iteration.
///
/// Returns `Some(SolveResult)` to terminate the loop if either:
/// - `max_wall_time` has been exceeded → `MaxIterations`
/// - `early_stall_timeout` was hit during the first 5 iterations
///   (scaled by problem size) → `NumericalError`
///
/// Wall-clock is polled every iteration during the first 10, then every 10
/// thereafter to keep overhead negligible on long runs.
///
/// Extracted from `solve_ipm` as part of the v0.8 main-loop decomposition
/// (pre-work step 2). Pure guard function.
fn check_time_limits(
    state: &SolverState,
    iteration: usize,
    start_time: Instant,
    early_timeout: f64,
    options: &SolverOptions,
) -> Option<SolveResult> {
    // Check wall-clock time limit (every iteration in early phase, every 10 after)
    if (iteration < 10 || iteration % 10 == 0) && options.max_wall_time > 0.0 {
        if start_time.elapsed().as_secs_f64() >= options.max_wall_time {
            return Some(make_result(state, SolveStatus::MaxIterations));
        }
    }

    // Early stall detection: bail out if stuck in early iterations.
    // `early_timeout` is pre-scaled by problem size in the caller (see
    // scaling formula in solve_ipm). Square problems are exempt — they
    // can have legitimately slow first iterations (mirrors Ipopt
    // IpBacktrackingLineSearch.cpp:276-280).
    if !state.is_square && iteration < 5 && options.early_stall_timeout > 0.0 {
        if start_time.elapsed().as_secs_f64() > early_timeout {
            if options.print_level >= 3 {
                rip_log!(
                    "ripopt: Early stall at iteration {} ({:.1}s elapsed), terminating",
                    iteration, start_time.elapsed().as_secs_f64()
                );
            }
            return Some(make_result(state, SolveStatus::NumericalError));
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

/// Overall-progress stall tracker: best primal/dual infeasibility seen
/// so far and the consecutive-no-progress counter. Threaded through
/// the stall-detection helpers (`detect_and_handle_progress_stall` and
/// the μ-boost recovery paths) instead of three parallel locals.
struct ProgressStallTracker {
    best_pr: f64,
    best_du: f64,
    no_progress_count: usize,
}

impl ProgressStallTracker {
    fn new() -> Self {
        Self {
            best_pr: f64::INFINITY,
            best_du: f64::INFINITY,
            no_progress_count: 0,
        }
    }

    fn reset(&mut self) {
        self.best_pr = f64::INFINITY;
        self.best_du = f64::INFINITY;
        self.no_progress_count = 0;
    }
}

/// Stall-recovery cleanup: re-seed the filter from the current θ and
/// clear the no-progress window so the next iteration starts fresh.
/// Used by every branch in `handle_near_tolerance_stall` /
/// `try_boost_mu_for_stall` that mutates μ to escape a stall — the
/// metric history before the μ change is meaningless once μ jumps.
fn reset_stall_counters_and_filter(
    state: &SolverState,
    filter: &mut Filter,
    stall: &mut ProgressStallTracker,
) {
    reset_filter_with_current_theta(state, filter);
    stall.reset();
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
/// Ipopt acceptable-tolerance gate and the overall-progress stall
/// detector, then update `state.consecutive_acceptable`.
///
/// Scaling matches `IpIpoptCalculatedQuantities::ComputeOptimalityErrorScaling`
/// with `s_max=100` and the 1e4 cap preserved for compatibility with
/// the rest of ripopt's tolerance pipeline. Acceptable thresholds
/// match `IpOptErrorConvCheck.cpp:70-121` defaults
/// (acceptable_tol=1e-6, acceptable_constr_viol_tol=1e-2,
/// acceptable_dual_inf_tol=1e10, acceptable_compl_inf_tol=1e-2).
///
/// Returns `s_d_for_acc` because it is consumed downstream by
/// `detect_and_handle_progress_stall`.
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

/// Detect unboundedness by requiring 10 consecutive iterations of
/// `obj < -1e20` at a feasible iterate. The counter prevents false
/// positives from transient dips.
///
/// **ripopt-specific (T3.22, no Ipopt analog).** Ipopt 3.14 has no
/// `Unbounded` exit status: it relies on the user to set
/// `nlp_lower_bound_inf` (default −1e19) and to bound objectives that
/// could legitimately diverge; an unbounded NLP typically manifests as
/// `Diverging_Iterates` (`x` magnitude exceeding the
/// `diverging_iterates_tol` threshold, default 1e20) rather than as an
/// objective-value test. This routine is a defensive ripopt convenience
/// for callers that pass an objective with no lower bound — kept until
/// T3.21 ports Ipopt's diverging-iterates check, after which
/// `SolveStatus::Unbounded` itself is also up for review.
fn detect_unbounded(
    state: &SolverState,
    options: &SolverOptions,
    primal_inf: f64,
    consecutive_unbounded: &mut usize,
) -> Option<SolveResult> {
    if state.obj < -1e20 && primal_inf < options.constr_viol_tol {
        *consecutive_unbounded += 1;
        if *consecutive_unbounded >= 10 {
            return Some(make_result(state, SolveStatus::Unbounded));
        }
    } else {
        *consecutive_unbounded = 0;
    }
    None
}

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

/// Outcome of the overall-progress stall detector.
enum StallDecision {
    /// No stall (or not yet activated) — fall through to normal step.
    Proceed,
    /// Stall detected and recovered by bumping μ / resetting the filter —
    /// caller must `continue` the main loop immediately (no Newton step
    /// this iteration).
    Continue,
    /// Stall detected and no recovery possible — caller must return this
    /// result immediately.
    Return(SolveResult),
}

// T3.21 (2026-04-27): retired `check_stall_near_tolerance_via_optimal_duals`.
// It recomputed "optimal" bound multipliers from ∇L, ran a two-gate
// near-tolerance check, and exited with NumericalError when both gates
// passed. The handler was unreachable under defaults (gated behind
// `options.stall_iter_limit > 0`, default 0 since T2.27) and has no
// Ipopt analog — Ipopt has no near-tolerance escape hatch on stall.
// Removed alongside its callsite in `detect_and_handle_progress_stall`.

/// In Fixed (monotone) μ mode, when stall recovery hasn't already kicked
/// in via a μ boost and μ is still above mu_min, force a μ decrease at
/// the min(linear, superlinear) rate. Resets stall counters and the
/// filter. Returns true when the decrease was applied (caller should
/// `continue` the loop), false when the gate didn't fire.
#[allow(clippy::too_many_arguments)]
fn try_force_mu_decrease_in_fixed_mode(
    state: &mut SolverState,
    options: &SolverOptions,
    primal_inf: f64,
    dual_inf: f64,
    compl_inf: f64,
    filter: &mut Filter,
    stall: &mut ProgressStallTracker,
) -> bool {
    if !(!options.mu_strategy_adaptive && state.mu > options.mu_min) {
        return false;
    }
    let new_mu = (options.mu_linear_decrease_factor * state.mu)
        .min(state.mu.powf(options.mu_superlinear_decrease_power))
        .max(options.mu_min);
    if options.print_level >= 3 {
        rip_log!(
            "ripopt: Fixed mode stall near tolerance (pr={:.2e}, du={:.2e}, co={:.2e}), forcing mu {:.2e} -> {:.2e}",
            primal_inf, dual_inf, compl_inf, state.mu, new_mu
        );
    }
    state.mu = new_mu;
    reset_stall_counters_and_filter(state, filter, stall);
    true
}

/// Handle a stall whose current iterate is near-tolerance (1000x tol on
/// each metric). Three sub-decisions in priority order:
/// 1. mu has outrun feasibility (mu < 0.01*pr_max while pr_max above
///    constr_viol_tol): boost mu to 0.1*pr_max, switch to Fixed mode,
///    return Continue.
/// 2. Fixed (monotone) mode + mu still above mu_min: force a mu decrease
///    by min(linear, superlinear) rate (the barrier subproblem is
///    effectively solved at the current mu), return Continue.
/// 3. Otherwise, classify as Acceptable when both unscaled (1e-2) and
///    scaled (1e-6 * s_d) tolerances pass; else return NumericalError.
#[allow(clippy::too_many_arguments)]
fn handle_near_tolerance_stall(
    state: &mut SolverState,
    options: &SolverOptions,
    primal_inf: f64,
    primal_inf_max: f64,
    dual_inf: f64,
    compl_inf: f64,
    s_d_for_acc: f64,
    filter: &mut Filter,
    mu_state: &mut MuState,
    stall: &mut ProgressStallTracker,
) -> StallDecision {
    // A8.13: μ-boost on stall is a ripopt-specific recovery with no
    // analogue in Ipopt's `IpMonotoneMuUpdate.cpp`, where μ decreases
    // monotonically and stall handling is left to the filter line
    // search + AcceptableLevel termination. Restrict to adaptive
    // strategy only — under monotone (the Ipopt default) a μ-boost
    // violates the monotone invariant and produced an iter-685 spike
    // on arki0003 (μ jumped 1e-3 → 2.21e2). See A8 follow-up doc.
    if options.mu_strategy_adaptive
        && state.mu < primal_inf_max * 0.01
        && primal_inf_max > options.constr_viol_tol
    {
        let new_mu = (primal_inf_max * 0.1).max(1e-6);
        if options.print_level >= 3 {
            rip_log!(
                "ripopt: Near-tolerance stall: boosting mu {:.2e} -> {:.2e} (pr_max={:.2e})",
                state.mu, new_mu, primal_inf_max
            );
        }
        boost_mu_and_switch_to_fixed_with_stall_reset(
            state, new_mu, mu_state, filter, stall,
        );
        return StallDecision::Continue;
    }
    if try_force_mu_decrease_in_fixed_mode(
        state, options, primal_inf, dual_inf, compl_inf, filter, stall,
    ) {
        return StallDecision::Continue;
    }
    classify_near_tolerance_stall_outcome(
        state, options, primal_inf, primal_inf_max, dual_inf, compl_inf, s_d_for_acc,
    )
}

/// Final classification used after the μ-boost and forced-Fixed-decrease
/// branches of handle_near_tolerance_stall did not fire. Returns
/// `Acceptable` when both unscaled (1e-2) and scaled (1e-6 · s_d) gates
/// pass on all three metrics; otherwise `NumericalError`.
#[allow(clippy::too_many_arguments)]
fn classify_near_tolerance_stall_outcome(
    state: &SolverState,
    options: &SolverOptions,
    primal_inf: f64,
    primal_inf_max: f64,
    dual_inf: f64,
    compl_inf: f64,
    s_d_for_acc: f64,
) -> StallDecision {
    let acc_pr_ok = primal_inf_max <= 1e-2;
    let acc_du_ok = dual_inf <= 1e10;
    let acc_co_ok = compl_inf <= 1e-2;
    let acc_scaled_ok = primal_inf_max <= 1e-6
        && dual_inf <= 1e-6 * s_d_for_acc
        && compl_inf <= 1e-6 * s_d_for_acc;
    if acc_pr_ok && acc_du_ok && acc_co_ok && acc_scaled_ok {
        if options.print_level >= 3 {
            rip_log!(
                "ripopt: Stalled at acceptable tolerance (pr={:.2e}, du={:.2e}, co={:.2e}), returning Acceptable",
                primal_inf, dual_inf, compl_inf
            );
        }
        return StallDecision::Return(make_result(state, SolveStatus::Acceptable));
    }
    if options.print_level >= 3 {
        rip_log!(
            "ripopt: Stalled but near-tolerance (pr={:.2e}, du={:.2e}, co={:.2e}), returning NumericalError",
            primal_inf, dual_inf, compl_inf
        );
    }
    StallDecision::Return(make_result(state, SolveStatus::NumericalError))
}

/// Last-chance stall recovery: when primal feasibility is reasonable
/// (< 0.1) but μ has outrun it (μ < pr_max·0.01), bump μ back to
/// pr_max·0.1 (floor 1e-6), reset the filter, clear stall counters,
/// and switch to Fixed mode. Returns Some(Continue) on recovery, None
/// if the trigger isn't met.
fn try_boost_mu_for_stall(
    state: &mut SolverState,
    options: &SolverOptions,
    filter: &mut Filter,
    mu_state: &mut MuState,
    primal_inf_max: f64,
    stall: &mut ProgressStallTracker,
) -> Option<StallDecision> {
    // A8.13: same gate as `handle_near_tolerance_stall` — μ-boost has
    // no analogue in Ipopt's monotone path; restrict to adaptive.
    if !(options.mu_strategy_adaptive
        && primal_inf_max < 0.1
        && state.mu < primal_inf_max * 0.01)
    {
        return None;
    }
    let new_mu = (primal_inf_max * 0.1).max(1e-6);
    if options.print_level >= 3 {
        rip_log!(
            "ripopt: Stall recovery: boosting mu {:.2e} -> {:.2e} (pr_max={:.2e})",
            state.mu, new_mu, primal_inf_max
        );
    }
    boost_mu_and_switch_to_fixed_with_stall_reset(
        state, new_mu, mu_state, filter, stall,
    );
    Some(StallDecision::Continue)
}

/// Update best-so-far primal/dual metrics and the no-progress counter,
/// and return `true` when the stall limit has been reached.
///
/// Counts an iteration as "improving" when either `primal_inf_max` or
/// `dual_inf` shrinks by at least 1% of the previous best. Improving
/// resets the no-progress counter; non-improving increments it. The
/// effective stall limit is halved when both step lengths are
/// negligible (`alpha_primal < 1e-8 && alpha_dual < 1e-4`) so truly
/// stuck iterations terminate sooner.
fn update_stall_counters_and_check_limit(
    stall: &mut ProgressStallTracker,
    state: &SolverState,
    options: &SolverOptions,
    primal_inf_max: f64,
    dual_inf: f64,
) -> bool {
    let pr_improved = primal_inf_max < 0.99 * stall.best_pr;
    let du_improved = dual_inf < 0.99 * stall.best_du;
    if pr_improved {
        stall.best_pr = primal_inf_max;
    }
    if du_improved {
        stall.best_du = dual_inf;
    }
    if pr_improved || du_improved {
        stall.no_progress_count = 0;
        return false;
    }
    stall.no_progress_count += 1;
    let tiny_alpha = state.alpha_primal < 1e-8 && state.alpha_dual < 1e-4;
    let stall_limit = if tiny_alpha { options.stall_iter_limit / 2 } else { options.stall_iter_limit };
    stall.no_progress_count >= stall_limit
}

fn detect_and_handle_progress_stall<P: NlpProblem>(
    state: &mut SolverState,
    _problem: &P,
    options: &SolverOptions,
    iteration: usize,
    primal_inf: f64,
    primal_inf_max: f64,
    dual_inf: f64,
    compl_inf: f64,
    s_d_for_acc: f64,
    filter: &mut Filter,
    mu_state: &mut MuState,
    stall: &mut ProgressStallTracker,
    _linear_constraints: Option<&[bool]>,
    _lbfgs_mode: bool,
) -> StallDecision {
    if iteration <= 50 || options.stall_iter_limit == 0 {
        return StallDecision::Proceed;
    }

    if !update_stall_counters_and_check_limit(
        stall, state, options, primal_inf_max, dual_inf,
    ) {
        return StallDecision::Proceed;
    }

    // Before declaring NumericalError, check if the current point is
    // near-tolerance (1000x tol).
    let stall_near_tol = options.tol * 1000.0;
    let stall_pr_ok = primal_inf_max <= stall_near_tol.max(10.0 * options.constr_viol_tol);
    let stall_du_ok = dual_inf <= (stall_near_tol * s_d_for_acc).max(1e-2);
    let stall_co_ok = compl_inf <= (stall_near_tol * s_d_for_acc).max(1e-2);
    if stall_pr_ok && stall_du_ok && stall_co_ok {
        return handle_near_tolerance_stall(
            state, options, primal_inf, primal_inf_max, dual_inf, compl_inf,
            s_d_for_acc, filter, mu_state, stall,
        );
    }
    if let Some(decision) = try_boost_mu_for_stall(
        state, options, filter, mu_state, primal_inf_max, stall,
    ) {
        return decision;
    }
    if options.print_level >= 3 {
        rip_log!(
            "ripopt: Stalled for {} iterations without progress (alpha_p={:.2e}, pr={:.2e}, du={:.2e}), terminating",
            stall.no_progress_count, state.alpha_primal, primal_inf, dual_inf
        );
    }
    StallDecision::Return(make_result(state, SolveStatus::NumericalError))
}

/// Track constraint-violation history and optionally short-circuit
/// with `LocalInfeasibility`.
///
/// Pushes the current `primal_inf` into `theta_history`, updates the
/// "ever feasible" flag, and (when proactive infeasibility detection
/// is enabled) looks for a stagnated θ with a near-stationary ∇θ
/// over a 100-iteration window. Returns `Some(SolveResult)` when
/// infeasibility is declared.
///
/// Ipopt uses its `RestoFilterConvCheck` for a broadly analogous
/// purpose; this path is ripopt-specific and fires before the main
/// IPM would otherwise burn iterations waiting for restoration to
/// reach the same conclusion.
///
/// Extracted from `solve_ipm` as part of the v0.8 main-loop
/// decomposition.
#[allow(clippy::too_many_arguments)]
fn track_feasibility_and_detect_infeasibility(
    state: &SolverState,
    options: &SolverOptions,
    iteration: usize,
    primal_inf: f64,
    feas: &mut FeasibilityTracker,
) -> Option<SolveResult> {
    let m = state.m;

    if feas.history.len() >= feas.history_len {
        feas.history.remove(0);
    }
    feas.history.push(primal_inf);

    if primal_inf < options.constr_viol_tol {
        feas.ever_feasible = true;
    }

    // Proactive infeasibility detection: if θ has stagnated over the history
    // window AND ‖∇θ‖_∞ is near zero, declare infeasibility instead of
    // waiting for restoration to reach the same conclusion.
    if options.proactive_infeasibility_detection
        && !feas.ever_feasible
        && m > 0
        && iteration >= 50
        && primal_inf > options.constr_viol_tol
        && feas.history.len() >= feas.history_len
    {
        let theta_min_h = slice_min(&feas.history);
        let theta_max_h = feas.history.iter().cloned().fold(0.0f64, f64::max);
        if theta_max_h > 0.0 && (theta_max_h - theta_min_h) < 0.01 * primal_inf {
            feas.stall_count += 1;
        } else {
            feas.stall_count = 0;
        }
        if feas.stall_count >= 10 {
            let grad_theta_norm = compute_grad_theta_norm(state);
            let stationarity_tol = 1e-3 * primal_inf.max(1.0);
            if grad_theta_norm < stationarity_tol {
                log::info!(
                    "Proactive infeasibility at iter {}: θ stagnated at {:.2e}, ‖∇θ‖={:.2e}",
                    iteration, primal_inf, grad_theta_norm
                );
                return Some(make_result(state, SolveStatus::LocalInfeasibility));
            }
            // Stationarity not met — reset counter to check again next window.
            feas.stall_count = 0;
        }
    } else if feas.ever_feasible {
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
    for i in 0..state.n {
        let mut s_i = 0.0_f64;
        if state.x_l[i].is_finite() {
            s_i += state.z_l[i] / slack_xl(state, i);
        }
        if state.x_u[i].is_finite() {
            s_i += state.z_u[i] / slack_xu(state, i);
        }
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
    let dzl_inf = linf_norm(&state.dz_l);
    let dzu_inf = linf_norm(&state.dz_u);
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
    for i in 0..n {
        if state.x_l[i].is_finite() {
            x_l_viol[i] = (state.x_l[i] - state.x[i]).max(0.0);
            compl_xl[i] = (state.x[i] - state.x_l[i]) * state.z_l[i];
        }
        if state.x_u[i].is_finite() {
            x_u_viol[i] = (state.x[i] - state.x_u[i]).max(0.0);
            compl_xu[i] = (state.x_u[i] - state.x[i]) * state.z_u[i];
        }
    }
    // grad_lag = grad_f + J^T y - z_l + z_u
    let mut grad_lag = state.grad_f.clone();
    accumulate_jt_y(state, &mut grad_lag);
    for i in 0..n {
        grad_lag[i] -= state.z_l[i];
        grad_lag[i] += state.z_u[i];
    }
    let mut constr_viol = vec![0.0; m];
    let mut compl_g_vec = vec![0.0; m];
    for i in 0..m {
        if state.g_l[i].is_finite() && state.g[i] < state.g_l[i] {
            constr_viol[i] = state.g_l[i] - state.g[i];
        } else if state.g_u[i].is_finite() && state.g[i] > state.g_u[i] {
            constr_viol[i] = state.g[i] - state.g_u[i];
        }
        // Complementarity: lambda_i * c_i where c_i is the active constraint slack
        if state.g_l[i].is_finite() && state.g_u[i].is_finite() {
            // Equality or range: use min slack
            compl_g_vec[i] = state.y[i] * (state.g[i] - state.g_l[i]).min(state.g_u[i] - state.g[i]);
        } else if state.g_l[i].is_finite() {
            compl_g_vec[i] = state.y[i] * (state.g[i] - state.g_l[i]);
        } else if state.g_u[i].is_finite() {
            compl_g_vec[i] = state.y[i] * (state.g_u[i] - state.g[i]);
        }
    }
    IterateSnapshot {
        x: state.x.clone(),
        z_l: state.z_l.clone(),
        z_u: state.z_u.clone(),
        g: state.g.clone(),
        lambda: state.y.clone(),
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
    rip_log!(
        "{:>4}  {:>14.7e}  {:>10.2e}  {:>10.2e}  {:>10.2e}  {:>10.2e}  {:>8.2e}  {:>8.2e}  {:>3}",
        iteration,
        state.obj / state.obj_scaling,
        primal_inf,
        dual_inf,
        compl_inf,
        state.mu,
        state.alpha_primal,
        state.alpha_dual,
        ls_steps,
    );
    *log_line_count += 1;
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
    let kappa1 = options.bound_push;
    let kappa2 = options.bound_frac;
    for i in 0..m {
        if constraint_is_equality(state, i) {
            // Equality row sentinel: s = g_l so the slack-derived diagnostics
            // (which skip equality rows) see a benign value.
            state.s[i] = state.g_l[i];
            continue;
        }
        let l_fin = state.g_l[i].is_finite();
        let u_fin = state.g_u[i].is_finite();
        let mut s_i = state.g[i];
        if l_fin && u_fin {
            let range = state.g_u[i] - state.g_l[i];
            let p_l = (kappa1 * state.g_l[i].abs().max(1.0)).min(kappa2 * range);
            let p_u = (kappa1 * state.g_u[i].abs().max(1.0)).min(kappa2 * range);
            s_i = s_i.max(state.g_l[i] + p_l).min(state.g_u[i] - p_u);
        } else if l_fin {
            let p_l = kappa1 * state.g_l[i].abs().max(1.0);
            s_i = s_i.max(state.g_l[i] + p_l);
        } else if u_fin {
            let p_u = kappa1 * state.g_u[i].abs().max(1.0);
            s_i = s_i.min(state.g_u[i] - p_u);
        }
        state.s[i] = s_i;
    }
}

/// Initialize constraint slack barrier multipliers `v_l`, `v_u` (Ipopt's
/// `v_L`, `v_U`). For each inequality constraint side,
/// `v = mu_init / max(slack, 1e-20)`. Equality rows (`g_l ≈ g_u`) are
/// skipped. Mirrors Ipopt's `IpDefaultIterateInitializer.cpp`: slack-bound
/// multipliers are initialized to `bound_mult_init_val` (default 1.0), the
/// same constant used for x-bound multipliers, NOT to `mu_init / slack`.
///
/// A8.3: When `least_squares_mult_init` is OFF, ripopt forces the slack
/// dual residual `−y_d − v_L + v_U` to zero at iter 0 by setting
/// `y_d := v_U − v_L`. This is the algebraic elimination that ignores
/// the `J_d J_c^T` off-diagonal coupling — fine when y_c is also ≈ 0,
/// but a 1000× iter-0 dual blow-up on problems where the inequality
/// Jacobian has columns with O(1e3) summed coefficients (e.g.
/// Mittelmann arki0003).
///
/// When `least_squares_mult_init` is ON (Ipopt's default), the LS solve
/// already chose `y_c` to minimize `‖∇f − z_L + z_U + J^T y‖²` with
/// `y_d = 0` implicitly; overwriting `y_d` here would re-introduce the
/// J_d^T·(±1) contribution that the LS picked specifically to avoid. So
/// we leave y untouched in that path. This trades exact slack-side
/// stationarity at iter 0 (which Ipopt also does not enforce — its
/// `IpLeastSquareMults.cpp:53-81` 4-block LS chooses (y_c, y_d) jointly,
/// not piecewise) for the much larger reduction in iter-0 ‖J^T y‖.
fn initialize_constraint_slack_multipliers(state: &mut SolverState, m: usize, options: &SolverOptions) {
    let v_init = options.bound_mult_init_val;
    let ls_active = options.least_squares_mult_init && m > 0 && m != state.n;
    for i in 0..m {
        if constraint_is_equality(state, i) {
            continue;
        }
        if state.g_l[i].is_finite() {
            state.v_l[i] = v_init;
        }
        if state.g_u[i].is_finite() {
            state.v_u[i] = v_init;
        }
        if !ls_active {
            state.y[i] = state.v_u[i] - state.v_l[i];
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
        let len = init_y.len().min(state.y.len());
        state.y[..len].copy_from_slice(&init_y[..len]);
    }
    if let Some(ref init_z_l) = options.warm_start_z_l {
        let len = init_z_l.len().min(state.z_l.len());
        state.z_l[..len].copy_from_slice(&init_z_l[..len]);
    }
    if let Some(ref init_z_u) = options.warm_start_z_u {
        let len = init_z_u.len().min(state.z_u.len());
        state.z_u[..len].copy_from_slice(&init_z_u[..len]);
    }
    // B9: warm-start the slack iterate `s` and its bound multipliers
    // `v_l`, `v_u`. The default slack-push initializer ran before us
    // (`initialize_slack_iterate`); we overwrite with user values, then
    // project s back into a strict interior to keep the barrier well-
    // defined (use the same κ1/κ2 push the cold-start path uses).
    if let Some(ref init_s) = options.warm_start_s {
        let len = init_s.len().min(state.s.len());
        state.s[..len].copy_from_slice(&init_s[..len]);
        let kappa1 = options.bound_push;
        let kappa2 = options.bound_frac;
        for i in 0..len {
            if constraint_is_equality(state, i) {
                state.s[i] = state.g_l[i];
                continue;
            }
            let l_fin = state.g_l[i].is_finite();
            let u_fin = state.g_u[i].is_finite();
            if l_fin && u_fin {
                let range = state.g_u[i] - state.g_l[i];
                let p_l = (kappa1 * state.g_l[i].abs().max(1.0)).min(kappa2 * range);
                let p_u = (kappa1 * state.g_u[i].abs().max(1.0)).min(kappa2 * range);
                state.s[i] = state.s[i].max(state.g_l[i] + p_l).min(state.g_u[i] - p_u);
            } else if l_fin {
                let p_l = kappa1 * state.g_l[i].abs().max(1.0);
                state.s[i] = state.s[i].max(state.g_l[i] + p_l);
            } else if u_fin {
                let p_u = kappa1 * state.g_u[i].abs().max(1.0);
                state.s[i] = state.s[i].min(state.g_u[i] - p_u);
            }
        }
    }
    if let Some(ref init_v_l) = options.warm_start_v_l {
        let len = init_v_l.len().min(state.v_l.len());
        state.v_l[..len].copy_from_slice(&init_v_l[..len]);
    }
    if let Some(ref init_v_u) = options.warm_start_v_u {
        let len = init_v_u.len().min(state.v_u.len());
        state.v_u[..len].copy_from_slice(&init_v_u[..len]);
    }
    state.mu = WarmStartInitializer::initialize(
        &mut state.x,
        &mut state.z_l,
        &mut state.z_u,
        &state.x_l,
        &state.x_u,
        options,
    );
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
            if state.x_l[i].is_finite() && state.x_u[i].is_finite() {
                let range = state.x_u[i] - state.x_l[i];
                let push = push_factor * range;
                if range > 2.0 * push {
                    state.x[i] = state.x[i].max(state.x_l[i] + push).min(state.x_u[i] - push);
                } else {
                    state.x[i] = 0.5 * (state.x_l[i] + state.x_u[i]);
                }
            } else if state.x_l[i].is_finite() {
                let push = push_factor * state.x_l[i].abs().max(1.0);
                state.x[i] = state.x[i].max(state.x_l[i] + push);
            } else if state.x_u[i].is_finite() {
                let push = push_factor * state.x_u[i].abs().max(1.0);
                state.x[i] = state.x[i].min(state.x_u[i] - push);
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
    state.g_scaling = g_scaling;
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
    let mut restoration = RestorationPhase::new(500);
    restoration.set_square(state.is_square);

    // Initialize filter
    let mut filter = Filter::new(1e4);
    filter.set_obj_max_inc(options.obj_max_inc);
    filter.set_alpha_min_frac(options.alpha_min_frac);
    filter.set_filter_reset_options(options.filter_reset_trigger, options.max_filter_resets);

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

    // Tiny step counter (Ipopt: accept full step when relative step < 10*eps for 2 consecutive)
    let mut consecutive_tiny_steps: usize = 0;

    // STOP_AT_TINY_STEP exit flag: set by `update_barrier_parameter` when
    // `tiny_step && new_μ == μ` (no-op mu update — Ipopt's
    // `IpMonotoneMuUpdate.cpp:158-160`, `IpAdaptiveMuUpdate.cpp:329,377`),
    // consumed at the top of the *next* iteration after `check_convergence`
    // so KKT-clean tiny-step iterates exit Optimal first.
    let mut pending_tiny_step_exit: bool = false;

    // Overall progress stall detection: if neither primal nor dual infeasibility
    // improves by at least 1% over many consecutive iterations, terminate early.
    let mut stall = ProgressStallTracker::new();

    // Line-search backtrack count for the previous iteration (printed in table).
    let mut ls_steps: usize = 0;
    // Hessian regularization delta from previous iteration (for intermediate callback).
    let prev_ic_delta_w: f64 = 0.0;

    // Consecutive iterations with obj < -1e20 for robust unbounded detection
    let mut consecutive_unbounded: usize = 0;

    // Initial evaluation with NaN/Inf recovery by bound-push perturbation.
    if let Err(result) = initial_evaluate_with_recovery(
        &mut state, problem, &mut lbfgs_state, linear_constraints.as_deref(), lbfgs_mode, n, options,
    ) {
        return result;
    }

    initialize_slack_iterate(&mut state, m, options);
    initialize_constraint_slack_multipliers(&mut state, m, options);

    // Set filter parameters based on initial constraint violation
    let theta_init = state.constraint_violation();
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

        // Early-stall timeout scaled by problem size: medium-scale problems
        // (n+m > 1000) can legitimately spend 30-60s on restoration or line
        // search during early iterations.
        let early_timeout = options.early_stall_timeout * ((n + m) as f64 / 200.0).max(1.0);
        if let Some(result) = check_time_limits(&state, iteration, start_time, early_timeout, options) {
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
            for (idx, (&row, &col)) in state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate() {
                jty[col] += state.jac_vals[idx] * state.y[row];
            }
            let jty_inf = jty.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let zl_inf = state.z_l.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let zu_inf = state.z_u.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let y_inf = state.y.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let y_sum: f64 = state.y.iter().map(|v| v.abs()).sum();
            let zl_sum: f64 = state.z_l.iter().map(|v| v.abs()).sum();
            let zu_sum: f64 = state.z_u.iter().map(|v| v.abs()).sum();
            let mut grad_lag = state.grad_f.clone();
            for i in 0..state.n {
                grad_lag[i] += jty[i] - state.z_l[i] + state.z_u[i];
            }
            let (gl_idx, gl_inf) = grad_lag.iter().enumerate().fold(
                (0usize, 0.0f64),
                |(ai, av), (i, &v)| if v.abs() > av { (i, v.abs()) } else { (ai, av) },
            );
            let x_l_fin = state.x_l[gl_idx].is_finite();
            let x_u_fin = state.x_u[gl_idx].is_finite();
            let x_inf = state.x.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            rip_log!(
                "ripopt: iter0-probe: |grad_f|_inf={:.3e}@var{}, |J^T y|_inf={:.3e}, |y|_inf={:.3e}, |z_L|_inf={:.3e}, |z_U|_inf={:.3e}, sum|y|={:.3e}, sum|z_L|={:.3e}, sum|z_U|={:.3e}, |x|_inf={:.3e}, n={} m={}",
                gf_inf, gf_inf_idx, jty_inf, y_inf, zl_inf, zu_inf, y_sum, zl_sum, zu_sum, x_inf, state.n, state.m
            );
            rip_log!(
                "ripopt: iter0-probe: |grad_lag|_inf={:.3e}@var{} (grad_f={:.3e}, J^T y={:.3e}, z_L={:.3e}, z_U={:.3e}, x_l_fin={}, x_u_fin={}, obj_scaling={:.3e})",
                gl_inf, gl_idx, state.grad_f[gl_idx], jty[gl_idx], state.z_l[gl_idx], state.z_u[gl_idx], x_l_fin, x_u_fin, state.obj_scaling
            );
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

        let s_d_for_acc = track_consecutive_acceptable(
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

        if let Some(result) = detect_unbounded(
            &state,
            options,
            primal_inf,
            &mut consecutive_unbounded,
        ) {
            return result;
        }

        // Overall-progress stall detection — see helper doc comment.
        match detect_and_handle_progress_stall(
            &mut state,
            problem,
            options,
            iteration,
            primal_inf,
            primal_inf_max,
            dual_inf,
            compl_inf,
            s_d_for_acc,
            &mut filter,
            &mut mu_state,
            &mut stall,
            linear_constraints.as_deref(),
            lbfgs_mode,
        ) {
            StallDecision::Return(r) => return r,
            StallDecision::Continue => continue,
            StallDecision::Proceed => {}
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
        let (step, _dc, mu_new_opt, aug_kkt) = if probing {
            let avg_compl = compute_avg_complementarity(&state);
            let mu_max = mu_state.mu_max_cap(options, avg_compl);
            match crate::kkt_aug::aug_step_from_state_mehrotra(
                n, &state.grad_f,
                &state.hess_rows, &state.hess_cols, &state.hess_vals,
                &state.jac_rows, &state.jac_cols, &state.jac_vals,
                &state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u,
                &state.s, &state.g, &state.g_l, &state.g_u, &state.y,
                &state.v_l, &state.v_u,
                state.mu, options.kappa_d,
                crate::kkt_aug::PROBING_SIGMA_MAX_DEFAULT,
                options.mu_min, mu_max,
                use_sparse,
                aug_solver.as_mut(),
                &mut inertia_params,
            ) {
                Ok((step, mu_new, _dw, dc, aug)) => (step, dc, Some(mu_new), aug),
                Err(_e) => {
                    timings.direction_solve += t_dir.elapsed();
                    return make_result(&state, SolveStatus::NumericalError);
                }
            }
        } else {
            match crate::kkt_aug::aug_step_from_state(
                n, &state.grad_f,
                &state.hess_rows, &state.hess_cols, &state.hess_vals,
                &state.jac_rows, &state.jac_cols, &state.jac_vals,
                &state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u,
                &state.s, &state.g, &state.g_l, &state.g_u, &state.y,
                &state.v_l, &state.v_u,
                state.mu, options.kappa_d,
                use_sparse,
                aug_solver.as_mut(),
                &mut inertia_params,
            ) {
                Ok((step, _dw, dc, aug)) => (step, dc, None, aug),
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
            &mut state, step.dx, step.dy_m, step.ds,
            step.dz_l, step.dz_u, step.dv_l, step.dv_u,
        );

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
            &mut consecutive_tiny_steps,
            primal_inf,
        );

        // Line search
        let t_ls = Instant::now();
        let theta_current = primal_inf;
        let phi_current = state.barrier_objective(options);
        let grad_phi_step = state.barrier_directional_derivative(options);

        let mut step_accepted;
        let min_alpha = filter.compute_alpha_min(theta_current, grad_phi_step);
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
            start_time,
            early_timeout,
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
            LineSearchOutcome::Return(result) => return result,
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
            filter.add(theta_current, phi_current);
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
                &mut restoration,
                iteration,
                n,
                m,
                start_time,
                deadline,
                early_timeout,
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
            &mut restoration,
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
        // active, the two-iter counter has tripped, and the mu update
        // could not advance — this is exactly the Ipopt throw condition.
        // Consumed at the top of the next iteration AFTER check_convergence,
        // so a KKT-clean iterate still exits Optimal.
        if mu_state.tiny_step
            && consecutive_tiny_steps >= 2
            && state.mu == mu_before_update
        {
            pending_tiny_step_exit = true;
        }

        track_post_step_acceptable(&mut state, options);

        // A8.6+ dual-divergence trace: env-gated per-iter snapshot of
        // the dual state. Set RIPOPT_TRACE_DUAL=1 to log ‖y‖_∞,
        // worst-y_i index, α_pr/α_du, μ, and μ-mode at the end of
        // every accepted iteration. Used to identify the iter where a
        // diverging trajectory first deviates from the Ipopt log.
        if std::env::var("RIPOPT_TRACE_DUAL").is_ok() {
            let mut y_inf = 0.0_f64;
            let mut y_idx = usize::MAX;
            for (i, &yi) in state.y.iter().enumerate() {
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
            let dy_inf = linf_norm(&state.dy);
            let dzl_inf = linf_norm(&state.dz_l);
            let dzu_inf = linf_norm(&state.dz_u);
            // Worst (z·s)/μ ratio: should be bounded by κ_Σ (default 1e10)
            // per Ipopt's reset_slack_multipliers / IpIpoptCalculatedQuantities.
            // Ratios >> 1 indicate the κ_Σ clamp is not enforcing.
            let mut worst_zs_ratio = 0.0_f64;
            let mut worst_zs_idx = usize::MAX;
            let mut worst_zs_side = "";
            for i in 0..state.n {
                if state.x_l[i].is_finite() {
                    let s = state.x[i] - state.x_l[i];
                    if s > 0.0 {
                        let r = (state.z_l[i] * s).abs() / state.mu.max(1e-300);
                        if r > worst_zs_ratio { worst_zs_ratio = r; worst_zs_idx = i; worst_zs_side = "L"; }
                    }
                }
                if state.x_u[i].is_finite() {
                    let s = state.x_u[i] - state.x[i];
                    if s > 0.0 {
                        let r = (state.z_u[i] * s).abs() / state.mu.max(1e-300);
                        if r > worst_zs_ratio { worst_zs_ratio = r; worst_zs_idx = i; worst_zs_side = "U"; }
                    }
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
    n: usize,
    m: usize,
    theta_current: f64,
    phi_current: f64,
    grad_phi_step: f64,
    alpha: f64,
    kappa_soc: f64,
    theta_prev_soc: &mut f64,
) -> SocTrialOutcome {
    let mut obj_soc = f64::INFINITY;
    if !problem.objective(&x_soc, true, &mut obj_soc) || !obj_soc.is_finite() {
        return SocTrialOutcome::Abort;
    }
    let mut g_soc = vec![0.0; m];
    if !problem.constraints(&x_soc, false, &mut g_soc) {
        return SocTrialOutcome::Abort;
    }
    if g_soc.iter().any(|v| !v.is_finite()) {
        return SocTrialOutcome::Abort;
    }

    let theta_soc = theta_for_g(state, &g_soc);
    if theta_soc >= kappa_soc * *theta_prev_soc {
        return SocTrialOutcome::Abort;
    }
    *theta_prev_soc = theta_soc;

    let phi_soc = compute_barrier_phi(
        obj_soc, &x_soc, &g_soc, state, n, m, options.constraint_slack_barrier,
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
        alpha,
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

    let partition = crate::kkt_aug::ConstraintPartition::new(&state.g_l, &state.g_u);
    let n_c = partition.n_c;
    let n_d = partition.n_d;

    // c_soc[k]    = curr_c[k]            = g[i] - g_l[i]   for equality row i.
    // dms_soc[k]  = curr_d_minus_s[k]    = g[i] - s[i]     for inequality row i.
    let mut c_soc = vec![0.0; n_c];
    let mut dms_soc = vec![0.0; n_d];
    for i in 0..m {
        if let Some(k) = partition.eq_pos[i] {
            c_soc[k] = state.g[i] - state.g_l[i];
        } else if let Some(k) = partition.ineq_pos[i] {
            dms_soc[k] = state.g[i] - state.s[i];
        }
    }

    // First-iteration trial residuals: g_trial / s_trial = s + α·ds.
    let mut latest_trial_c = vec![0.0; n_c];
    let mut latest_trial_dms = vec![0.0; n_d];
    for i in 0..m {
        if let Some(k) = partition.eq_pos[i] {
            latest_trial_c[k] = g_trial[i] - state.g_l[i];
        } else if let Some(k) = partition.ineq_pos[i] {
            let s_trial_i = state.s[i] + alpha * state.ds[i];
            latest_trial_dms[k] = g_trial[i] - s_trial_i;
        }
    }

    let kappa_soc = 0.99;
    let tau = (1.0 - state.mu).max(options.tau_min);

    let mut alpha_primal_soc = alpha;
    let mut theta_prev_soc = theta_for_g(state, g_trial);

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
            &state.jac_rows, &state.jac_cols, &state.jac_vals,
            &state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u,
            &state.s, &state.g, &state.g_l, &state.g_u, &state.y,
            &state.v_l, &state.v_u,
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

        match evaluate_soc_trial_and_check(
            state, problem, options, filter, x_soc, n, m,
            theta_current, phi_current, grad_phi_step, alpha,
            kappa_soc, &mut theta_prev_soc,
        ) {
            SocTrialOutcome::Accepted { x_soc, obj_soc, g_soc } => {
                return Some((x_soc, obj_soc, g_soc, alpha_primal_soc));
            }
            SocTrialOutcome::Abort => return None,
            SocTrialOutcome::NotAccepted { g_soc } => {
                // Refresh latest_trial_* using the rejected SOC trial:
                //   s_soc[i] = state.s[i] + α_p_soc · ds_d_soc[k]   for ineq row i.
                for i in 0..m {
                    if let Some(k) = partition.eq_pos[i] {
                        latest_trial_c[k] = g_soc[i] - state.g_l[i];
                    } else if let Some(k) = partition.ineq_pos[i] {
                        let s_soc_i = state.s[i] + alpha_primal_soc * ds_d_soc[k];
                        latest_trial_dms[k] = g_soc[i] - s_soc_i;
                    }
                }
            }
        }
    }

    None
}

/// Reset bound multipliers after restoration, matching Ipopt
/// IpRestoMinC_1Nrm.cpp:374-419:
///   1. Tentatively set z = mu/slack per bound (no element-wise clamp).
///   2. If max(|z|) exceeds bound_mult_reset_threshold (1e3),
///      *nuclear reset*: set ALL z_L, z_U to 1.0.
///   3. Otherwise keep z = mu/slack as-is.
/// An element-wise clamp at 1e3 leaves inf_du stuck at ~mu/slack - 1000
/// when slack is tight — the least-squares y computed after can't absorb
/// that. Returns whether the nuclear reset was triggered (for downstream
/// v_L/v_U handling).
/// T0.8: install resto-returned bound multipliers, clamped via the
/// Ipopt κ_σ safeguard. For each finite bound, take the resto-NLP z
/// for that variable (the original-x block of the resto solve) and
/// apply the one-shot Newton step described in `IPOPT_ALGORITHM_SPEC.md`
/// §7.8 / `IpRestoMinC_1Nrm.cpp:378-399`:
///
///   δz_i = (μ - z_i·(s_trial - s_cur))/s_cur − z_i
///
/// followed by a fraction-to-boundary step on z (τ ≈ 1 − μ, capped at
/// 0.99) and the standard κ_σ ∈ \[μ/(κ_σ·s), κ_σ·μ/s\] safeguard at
/// the new slack. `x_cur` is the pre-restoration x (used in s_cur) and
/// `state.x` already holds x_trial at call time.
///
/// Bounds with no resto z (infinite x bound) are zeroed. The "nuclear
/// reset" return value tracks whether the post-clamp z_max is above
/// the watchdog threshold so the v-multiplier path mirrors the
/// existing semantics.
fn apply_kappa_sigma_clamp_to_resto_z(
    state: &mut SolverState,
    n: usize,
    zl_resto: &[f64],
    zu_resto: &[f64],
    x_cur: &[f64],
) -> bool {
    let kappa_sigma = 1e10;
    let mu = state.mu.max(1e-20);
    let bound_mult_reset_threshold = 1000.0;
    // Match Ipopt IpFilterLSAcceptor: τ_min = max(0.99, 1 − μ).
    let tau = (1.0 - mu).max(0.99);
    let mut z_max: f64 = 0.0;
    for i in 0..n {
        if state.x_l[i].is_finite() {
            let s_cur = (x_cur[i] - state.x_l[i]).max(1e-12);
            let s_trial = (state.x[i] - state.x_l[i]).max(1e-12);
            let z_in = if i < zl_resto.len() { zl_resto[i].max(1e-20) } else { mu / s_cur };
            // One-shot Newton: δz = (μ − z·(s_trial − s_cur))/s_cur − z
            let dz = (mu - z_in * (s_trial - s_cur)) / s_cur - z_in;
            // α_d FTB on z: keep z + α·δz ≥ (1−τ)·z when δz < 0.
            let alpha_d = if dz < 0.0 {
                ((-tau * z_in) / dz).min(1.0).max(0.0)
            } else {
                1.0
            };
            let z_new = z_in + alpha_d * dz;
            // κ_σ safeguard at the new slack.
            let z_lo = mu / (kappa_sigma * s_trial);
            let z_hi = kappa_sigma * mu / s_trial;
            state.z_l[i] = z_new.clamp(z_lo, z_hi);
            z_max = z_max.max(state.z_l[i]);
        } else {
            state.z_l[i] = 0.0;
        }
        if state.x_u[i].is_finite() {
            let s_cur = (state.x_u[i] - x_cur[i]).max(1e-12);
            let s_trial = (state.x_u[i] - state.x[i]).max(1e-12);
            let z_in = if i < zu_resto.len() { zu_resto[i].max(1e-20) } else { mu / s_cur };
            let dz = (mu - z_in * (s_trial - s_cur)) / s_cur - z_in;
            let alpha_d = if dz < 0.0 {
                ((-tau * z_in) / dz).min(1.0).max(0.0)
            } else {
                1.0
            };
            let z_new = z_in + alpha_d * dz;
            let z_lo = mu / (kappa_sigma * s_trial);
            let z_hi = kappa_sigma * mu / s_trial;
            state.z_u[i] = z_new.clamp(z_lo, z_hi);
            z_max = z_max.max(state.z_u[i]);
        } else {
            state.z_u[i] = 0.0;
        }
    }
    z_max > bound_mult_reset_threshold
}

fn reset_bound_multipliers_after_restoration(state: &mut SolverState, n: usize) -> bool {
    let bound_mult_reset_threshold = 1000.0;
    let mu_for_reset = state.mu;
    let mut z_max: f64 = 0.0;
    for i in 0..n {
        if state.x_l[i].is_finite() {
            let slack = (state.x[i] - state.x_l[i]).max(1e-12);
            state.z_l[i] = mu_for_reset / slack;
            z_max = z_max.max(state.z_l[i]);
        } else {
            state.z_l[i] = 0.0;
        }
        if state.x_u[i].is_finite() {
            let slack = (state.x_u[i] - state.x[i]).max(1e-12);
            state.z_u[i] = mu_for_reset / slack;
            z_max = z_max.max(state.z_u[i]);
        } else {
            state.z_u[i] = 0.0;
        }
    }
    let nuclear_reset = z_max > bound_mult_reset_threshold;
    if nuclear_reset {
        for i in 0..n {
            state.z_l[i] = if state.x_l[i].is_finite() { 1.0 } else { 0.0 };
            state.z_u[i] = if state.x_u[i].is_finite() { 1.0 } else { 0.0 };
        }
    }
    nuclear_reset
}

/// Recompute y at the restored point via the augmented-saddle-point
/// least-squares multiplier estimate, INCLUDING the reset z_L/z_U
/// contribution. Otherwise any deviation between z_true = mu/slack
/// (huge at tight slack) and the reset value (1.0) appears entirely
/// in inf_du, driving a bad first Newton step.
///
/// Uses the Ipopt-exact augmented saddle-point system
///   [ I   J^T ] [ r ] = [ grad_f - z_L + z_U ]
///   [ J    0  ] [ y ]   [ 0                   ]
/// (matches IpLeastSquareMults::CalculateMultipliers with W=0, δ=0).
/// This is far better conditioned than the normal equations
/// J*J^T*y = rhs when J is nearly rank-deficient (as happens on
/// AC-OPF with gauge freedom — case30_ieee hits this at a
/// post-restoration feasible iterate).
///
/// If the augmented solve fails or the result exceeds
/// `constr_mult_init_max` (matching Ipopt
/// DefaultIterateInitializer::least_square_mults), zero out y.
/// LS y refresh after restoration entry. Uses the reduced 2-block
/// system (no slack-multiplier coupling) and applies the
/// `constr_mult_init_max` magnitude clamp as a robustness guard for
/// the post-restoration handoff. This is ripopt-specific defense; Ipopt
/// has no equivalent reset-time clamp because its restoration path
/// hands back fresh y from `RestoIpoptNLP`. The post-step recalc_y
/// uses `recompute_y_post_step_full_augmented` instead.
fn recompute_y_after_restoration(
    state: &mut SolverState,
    options: &SolverOptions,
    n: usize,
    m: usize,
) {
    if m == 0 {
        return;
    }
    let y_ls_result = compute_ls_multiplier_estimate_augmented(
        &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
        &state.g_l, &state.g_u, n, m,
        Some(&state.z_l), Some(&state.z_u),
        None,
    );
    let y_accepted = match y_ls_result {
        Some(y_ls) => {
            let max_abs = linf_norm(&y_ls);
            if max_abs > options.constr_mult_init_max { None } else { Some(y_ls) }
        }
        None => None,
    };
    // Keep current y on LS failure or magnitude rejection. Ipopt's
    // IpDefaultIterateInitializer::least_square_mults zeros y only at the
    // *initial* iterate; during iteration `IpIpoptAlgorithm::ActualizeHessianAndConstraints`
    // leaves y unchanged when the LS estimate is rejected. Zeroing mid-run
    // discards information from a converged dual estimate and biases the
    // next Newton direction.
    if let Some(y_ls) = y_accepted {
        state.y.copy_from_slice(&y_ls);
    }
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
    let y_ls_result = compute_ls_multiplier_estimate_augmented(
        &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
        &state.g_l, &state.g_u, n, m,
        Some(&state.z_l), Some(&state.z_u),
        Some((&state.v_l, &state.v_u)),
    );
    if let Some(y_ls) = y_ls_result {
        state.y.copy_from_slice(&y_ls);
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
    let theta_max = convergence::primal_infeasibility_max(&state.g, &state.g_l, &state.g_u);
    if theta_max >= options.recalc_y_feas_tol {
        return;
    }
    // T3.30: Ipopt-aligned post-step recalc uses the full 4-block LS system
    // (slack/v_L/v_U coupling) and accepts unconditionally on solver success.
    recompute_y_post_step_full_augmented(state, n, m);
}

/// Reset constraint-slack barrier multipliers v_L, v_U after restoration,
/// mirroring the nuclear-reset semantics of bound multipliers. Equality
/// constraints get v=0 (no slack barrier). For inequality constraints, if
/// the bound-multiplier reset triggered the all-to-1.0 path, also reset v
/// to 1.0; else use v = mu/slack uncapped.
fn reset_constraint_slack_multipliers_after_restoration(
    state: &mut SolverState,
    m: usize,
    nuclear_reset: bool,
) {
    let mu_r = state.mu;
    for i in 0..m {
        if constraint_is_equality(state, i) {
            state.v_l[i] = 0.0;
            state.v_u[i] = 0.0;
            continue;
        }
        if state.g_l[i].is_finite() {
            let slack = (state.g[i] - state.g_l[i]).max(1e-12);
            state.v_l[i] = if nuclear_reset { 1.0 } else { mu_r / slack };
        } else {
            state.v_l[i] = 0.0;
        }
        if state.g_u[i].is_finite() {
            let slack = (state.g_u[i] - state.g[i]).max(1e-12);
            state.v_u[i] = if nuclear_reset { 1.0 } else { mu_r / slack };
        } else {
            state.v_u[i] = 0.0;
        }
    }
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
        let mut g_check = vec![0.0; m];
        let mut phi_check = f64::INFINITY;
        let g_ok = m == 0 || problem.constraints(x_new, true, &mut g_check);
        let phi_ok = problem.objective(x_new, true, &mut phi_check) && phi_check.is_finite();
        if !g_ok || !phi_ok {
            return false;
        }
        let theta_check = if m == 0 { 0.0 } else { theta_for_g(state, &g_check) };
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

    // T0.8 / spec §7.8: when the resto NLP returned bound multipliers,
    // apply the one-shot Newton z update + α_d FTB + κ_σ clamp to
    // those values (mirrors Ipopt main-phase init after
    // RestoIpoptNLP::finalize_solution). Otherwise (Gauss-Newton path
    // or missing z) fall back to the legacy μ/slack reset.
    let nuclear_reset = if let Some((zl_resto, zu_resto)) = resto_z {
        apply_kappa_sigma_clamp_to_resto_z(state, n, zl_resto, zu_resto, &x_cur)
    } else {
        reset_bound_multipliers_after_restoration(state, n)
    };
    recompute_y_after_restoration(state, options, n, m);

    reset_constraint_slack_multipliers_after_restoration(state, m, nuclear_reset);

    // T0.9: do NOT clear the filter — Ipopt keeps the existing entries
    // (including the augmentation added at resto entry) so future
    // iterations cannot revisit the pre-resto basin. Only the
    // consecutive_acceptable counter is reset.
    state.consecutive_acceptable = 0;

    // Recompute mu from current complementarity after restoration
    let mu_compl = compute_avg_complementarity(state);
    if mu_compl > 0.0 {
        state.mu = mu_compl.max(options.mu_min).min(1e5);
    }

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
/// prevent recursion, sets mu_init to resto_mu, disables stall detection
/// (restoration makes slow steady progress that would trip the 30-iter
/// stall limit), and relaxes tol to 1e-7 (we want feasibility, not
/// optimality). Propagates the remaining wall-time budget so the inner
/// solve can't outlive the outer fallback cascade; returns None when the
/// remaining budget is < 0.5s. Scales early_stall_timeout by restoration
/// NLP size so large restorations get the full default timeout.
fn configure_restoration_inner_options(
    options: &SolverOptions,
    resto_mu: f64,
    resto_dim: usize,
    start_time: Instant,
) -> Option<SolverOptions> {
    let mut inner_opts = options.clone();
    inner_opts.max_iter = options.restoration_max_iter.max(500);
    inner_opts.disable_nlp_restoration = true;
    inner_opts.print_level = if options.print_level >= 5 { 3 } else { 0 };
    inner_opts.mu_init = resto_mu;
    inner_opts.stall_iter_limit = 0;
    inner_opts.tol = 1e-7;

    if options.max_wall_time > 0.0 {
        let remaining = options.max_wall_time - start_time.elapsed().as_secs_f64();
        if remaining < 0.5 {
            return None;
        }
        inner_opts.max_wall_time = remaining;
    }
    inner_opts.early_stall_timeout = if options.early_stall_timeout > 0.0 {
        if resto_dim > 500 {
            options.early_stall_timeout
        } else {
            options.early_stall_timeout.min(3.0)
        }
    } else {
        3.0
    };
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

    // Ipopt's resto_mu = max(curr_mu, ||c(x_r)||_inf) (IpRestoIterateInitializer.cpp:58).
    // Using a mu_init consistent with the current infeasibility makes the
    // closed-form (p,n) init well-conditioned: when theta ≫ mu, the slacks
    // would otherwise be pinned near 0 with enormous bound multipliers.
    let c_inf = compute_primal_inf_max_at_state(state);
    let resto_mu = state.mu.max(c_inf);

    // Build restoration NLP using the same resto_mu for p/n quadratic init.
    let resto_nlp = RestorationNlp::new(problem, &state.x, resto_mu, rho, 1.0);

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

    // Evaluate original constraints at the restored point
    let mut g_new = vec![0.0; m];
    if !problem.constraints(&x_nlp, true, &mut g_new)
        || g_new.iter().any(|v| !v.is_finite())
    {
        return (x_nlp, resto_z, RestorationOutcome::Failed);
    }
    let theta_new = theta_for_g(state, &g_new);

    // Evaluate original objective at the restored point
    let mut phi_new = f64::INFINITY;
    if !problem.objective(&x_nlp, false, &mut phi_new) || !phi_new.is_finite() {
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
/// 2. theta_new ≤ 0.5*theta_current → Success (50% reduction, ripopt-specific
///    lenient path that helps NLP restoration recover when θ_current is small;
///    subsumed by gate 3 when `kappa_resto >= 0.5`).
/// 3. theta_new ≤ kappa_resto * theta_current AND filter-acceptable → Success
///    (Ipopt's primary `required_infeasibility_reduction` gate, default 0.9).
/// 4. inner_converged but no feasibility improvement → LocalInfeasibility
///    (the restoration NLP itself reached a stationary point of the
///    L1-feasibility objective with positive residual).
/// 5. Otherwise → Failed.
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
    if theta_new <= 0.5 * theta_current {
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

/// Build the RHS `b = -J·(grad_f − z_L + z_U)` for the normal-equations
/// LS multiplier estimate. Treats `None` for `z_l`/`z_u` as zero
/// (cold-start). Shared by the dense and sparse variants of
/// `compute_ls_multiplier_estimate_*`.
fn compute_ls_multiplier_rhs(
    grad_f: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    n: usize,
    m: usize,
    z_l: Option<&[f64]>,
    z_u: Option<&[f64]>,
) -> Vec<f64> {
    let mut rhs_grad = grad_f.to_vec();
    if let Some(zl) = z_l {
        for i in 0..n { rhs_grad[i] -= zl[i]; }
    }
    if let Some(zu) = z_u {
        for i in 0..n { rhs_grad[i] += zu[i]; }
    }
    let mut b = vec![0.0; m];
    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        b[row] -= jac_vals[idx] * rhs_grad[col];
    }
    b
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

/// Compute initial constraint multipliers via least-squares estimate.
/// Solves the normal equations `(J·Jᵀ) y = -J·grad_f` (Ipopt convention
/// `L = f + yᵀ g`); the inner `compute_ls_multiplier_estimate_with_z`
/// dispatches to a sparse `J·Jᵀ` factorization for `m > 500` so this
/// scales without an outer cap (T2.2: prior `m ≤ 500` cutoff dropped).
/// Returns `vec![0.0; m]` when LS init is disabled, m == 0, evaluation
/// fails, the LS solve fails, or estimates exceed `constr_mult_init_max`.
#[allow(clippy::too_many_arguments)]
fn compute_initial_y_with_ls<P: NlpProblem>(
    problem: &P,
    options: &SolverOptions,
    x: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
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
    // The LS estimate is unreliable when J is square (equivalently expects
    // grad_f ∈ range(J^T) which is too restrictive at a poor initial point).
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
    // A8.1+A8.2: thread z_L, z_U through so the LS RHS becomes
    //   ∇f − P_L·z_L + P_U·z_U
    // matching Ipopt 3.14 `IpLeastSquareMults.cpp:53-81` exactly. Without
    // this, the LS estimate over-fits to a sparse ∇f and produces y with
    // ‖y‖_∞ in the hundreds (well below the 1000 discard threshold) on
    // problems where most z*-padded RHS entries are O(1).
    compute_ls_multiplier_estimate_with_z(
        &grad_f_init,
        jac_rows,
        jac_cols,
        &jac_vals_init,
        g_l,
        g_u,
        n,
        m,
        options.constr_mult_init_max,
        Some(z_l),
        Some(z_u),
    )
    .unwrap_or_else(|| vec![0.0; m])
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
    for j in 0..m {
        ssm.triplet_rows.push(n + j);
        ssm.triplet_cols.push(n + j);
        let diag = match inequality_diag {
            Some(flags) if flags[j] => -1.0,
            _ => 0.0,
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

    let matrix = KktMatrix::Sparse(build_ls_augmented_matrix(
        jac_rows, jac_cols, jac_vals, n, m, inequality_flags.as_deref(),
    ));
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

    let mut y_ls: Vec<f64> = sol[n..].to_vec();
    fix_inequality_mult_signs(&mut y_ls, g_l, g_u, m);
    Some(y_ls)
}

/// LS multiplier estimate with optional bound-multiplier contributions.
///
/// Minimizes ||grad_f + J^T y - z_L + z_U||², yielding the normal equation
///   (J J^T) y = -J * (grad_f - z_L + z_U)
/// If `z_l`/`z_u` are `None`, they are treated as zero (cold-start / pre-z
/// initialization). Post-restoration callers must pass them so the reset z
/// values are absorbed into y, otherwise inf_du = ||grad_f + J^T y - z_L + z_U||
/// stays large and the next Newton step is ill-directed.
fn compute_ls_multiplier_estimate_with_z(
    grad_f: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    n: usize,
    m: usize,
    max_abs_threshold: f64,
    z_l: Option<&[f64]>,
    z_u: Option<&[f64]>,
) -> Option<Vec<f64>> {
    // For large problems, use sparse J*J^T factorization
    if m > 500 {
        return compute_ls_multiplier_estimate_sparse(
            grad_f, jac_rows, jac_cols, jac_vals, g_l, g_u, n, m, max_abs_threshold, None,
            z_l, z_u,
        );
    }
    if m == 0 {
        return None;
    }

    let b = compute_ls_multiplier_rhs(grad_f, jac_rows, jac_cols, jac_vals, n, m, z_l, z_u);

    // Compute A = J * J^T  (m x m dense symmetric matrix)
    let mut j_dense = vec![0.0; m * n];
    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        j_dense[row * n + col] = jac_vals[idx];
    }
    let mut a_mat = SymmetricMatrix::zeros(m);
    for i in 0..m {
        let row_i = &j_dense[i * n..(i + 1) * n];
        for j in 0..=i {
            let row_j = &j_dense[j * n..(j + 1) * n];
            a_mat.set(i, j, dot_product(row_i, row_j));
        }
    }

    // Solve (J J^T) y = b using DenseLdl
    let mut ls_solver = DenseLdl::new();
    let mut y_ls = vec![0.0; m];
    let factored = ls_solver.bunch_kaufman_factor(&a_mat);
    let solved = factored.is_ok() && ls_solver.solve(&b, &mut y_ls).is_ok();

    if !solved {
        return None;
    }

    let max_abs = linf_norm(&y_ls);
    if max_abs > max_abs_threshold {
        return None;
    }

    fix_inequality_mult_signs(&mut y_ls, g_l, g_u, m);
    Some(y_ls)
}

/// Sparse variant of LS multiplier estimate for large problems.
/// Builds sparse J*J^T in COO format and factors with the sparse solver.
/// If `solver` is Some, reuses the solver (for recalc_y caching).
/// Build the sparse normal matrix `J·Jᵀ + reg·I` (upper triangle) as
/// a `SparseSymmetricMatrix`. Groups Jacobian entries by column, then
/// for each column accumulates the outer product `v·vᵀ` into a
/// HashMap keyed by `(row, col)` with `row ≤ col`. A small `1e-12`
/// regularization on the diagonal preserves numerical stability when
/// `J` is rank-deficient. Used by the sparse LS multiplier estimate.
fn build_sparse_normal_matrix_jjt(
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    n: usize,
    m: usize,
) -> crate::linear_solver::SparseSymmetricMatrix {
    use crate::linear_solver::SparseSymmetricMatrix;

    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        col_entries[col].push((row, jac_vals[idx]));
    }

    let total_col_nnz: usize = col_entries.iter().map(|c| c.len()).sum();
    let nnz_est = total_col_nnz * 4;

    use std::collections::HashMap;
    let mut triplet_map: HashMap<(usize, usize), f64> = HashMap::with_capacity(nnz_est);

    for k in 0..n {
        let entries = &col_entries[k];
        for &(i, vi) in entries.iter() {
            for &(j, vj) in entries.iter() {
                if i >= j {
                    *triplet_map.entry((j, i)).or_insert(0.0) += vi * vj;
                }
            }
        }
    }

    let reg = 1e-12;
    for i in 0..m {
        *triplet_map.entry((i, i)).or_insert(0.0) += reg;
    }

    let nnz = triplet_map.len();
    let mut ssm = SparseSymmetricMatrix {
        n: m,
        triplet_rows: Vec::with_capacity(nnz),
        triplet_cols: Vec::with_capacity(nnz),
        triplet_vals: Vec::with_capacity(nnz),
    };
    for (&(r, c), &v) in &triplet_map {
        ssm.triplet_rows.push(r);
        ssm.triplet_cols.push(c);
        ssm.triplet_vals.push(v);
    }
    ssm
}

fn compute_ls_multiplier_estimate_sparse(
    grad_f: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    n: usize,
    m: usize,
    max_abs_threshold: f64,
    solver: Option<&mut Box<dyn LinearSolver>>,
    z_l: Option<&[f64]>,
    z_u: Option<&[f64]>,
) -> Option<Vec<f64>> {
    if m == 0 {
        return None;
    }

    let b = compute_ls_multiplier_rhs(grad_f, jac_rows, jac_cols, jac_vals, n, m, z_l, z_u);

    let ssm = build_sparse_normal_matrix_jjt(jac_rows, jac_cols, jac_vals, n, m);
    let matrix = KktMatrix::Sparse(ssm);

    // Factor and solve
    let mut y_ls = vec![0.0; m];
    let solved = if let Some(ls) = solver {
        ls.factor(&matrix).is_ok() && ls.solve(&b, &mut y_ls).is_ok()
    } else {
        let mut ls = new_sparse_solver();
        ls.factor(&matrix).is_ok() && ls.solve(&b, &mut y_ls).is_ok()
    };

    if !solved {
        return None;
    }

    let max_abs = linf_norm(&y_ls);
    if max_abs > max_abs_threshold {
        return None;
    }

    fix_inequality_mult_signs(&mut y_ls, g_l, g_u, m);
    Some(y_ls)
}

/// Fix signs of inequality constraint multipliers from LS estimate.
/// Ipopt convention (L = f + y^T g): g >= g_l → y >= 0, g <= g_u → y <= 0.
fn fix_inequality_mult_signs(y_ls: &mut [f64], g_l: &[f64], g_u: &[f64], m: usize) {
    for i in 0..m {
        if convergence::is_equality_constraint(g_l[i], g_u[i]) {
            continue;
        }
        let has_lower = g_l[i].is_finite();
        let has_upper = g_u[i].is_finite();
        if has_lower && !has_upper && y_ls[i] < 0.0 {
            y_ls[i] = 0.0;
        } else if has_upper && !has_lower && y_ls[i] > 0.0 {
            y_ls[i] = 0.0;
        } else if !has_lower && !has_upper {
            y_ls[i] = 0.0;
        }
    }
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

/// Constraint violation theta evaluated at an arbitrary `g` against
/// the current state's `g_l`/`g_u` bounds. Centralises the four
/// trial-point theta sites (regular line search, soft restoration,
/// second-order correction, and the IIE search-direction probe) that
/// otherwise each repeat `convergence::primal_infeasibility(g,
/// &state.g_l, &state.g_u)`.
fn theta_for_g(state: &SolverState, g: &[f64]) -> f64 {
    convergence::primal_infeasibility(g, &state.g_l, &state.g_u)
}

/// Accumulate `J^T * y` (constraint Jacobian transpose times the
/// equality multipliers) into `target`. Used to assemble several
/// related dual residuals: ∇_x L for the snapshot/barrier-error
/// computations, the active-set z recovery, and the gradient-of-f +
/// J^T y diagnostic used by stall classification.
fn accumulate_jt_y(state: &SolverState, target: &mut [f64]) {
    for (idx, (&row, &col)) in state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate() {
        target[col] += state.jac_vals[idx] * state.y[row];
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
        if gj[i] > 0.0 && state.x_l[i].is_finite() {
            if gj[i] * slack_xl(state, i) <= kc * state.mu.max(1e-20) {
                zl[i] = gj[i];
            }
        } else if gj[i] < 0.0 && state.x_u[i].is_finite() {
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
    filter::fraction_to_boundary(&state.z_l, dz_l, tau)
        .min(filter::fraction_to_boundary(&state.z_u, dz_u, tau))
}

/// Fraction-to-boundary cap on `α·dv_L`, `α·dv_U` against the slack-bound
/// multipliers `state.v_l` / `state.v_u`. Returns the minimum across both
/// blocks; together with `fraction_to_boundary_dual_z_min` this gives the
/// full dual-side α_max (Ipopt computes a single α_dual across z_L, z_U,
/// v_L, v_U via `IpFilterLSAcceptor::ComputeAlphaForY` ↔
/// `IpIpoptCalculatedQuantities::CalcFracToBound`).
fn fraction_to_boundary_dual_v_min(state: &SolverState, dv_l: &[f64], dv_u: &[f64], tau: f64) -> f64 {
    filter::fraction_to_boundary(&state.v_l, dv_l, tau)
        .min(filter::fraction_to_boundary(&state.v_u, dv_u, tau))
}

/// Fraction-to-boundary cap on the primal step `α·dx` against the
/// variable bounds, ignoring the `[0, 1]` clamp. The Mehrotra
/// affine-predictor and the L-BFGS gradient-descent fallback both use
/// this same per-component scan; centralising it keeps the three
/// step-controllers (main step, multiple-centrality corrections,
/// second-order correction) in lockstep.
fn fraction_to_boundary_primal_x(state: &SolverState, dx: &[f64], tau: f64) -> f64 {
    let mut alpha = 1.0_f64;
    for i in 0..state.n {
        if state.x_l[i].is_finite() && dx[i] < 0.0 {
            let slack = state.x[i] - state.x_l[i];
            alpha = alpha.min(-tau * slack / dx[i]);
        }
        if state.x_u[i].is_finite() && dx[i] > 0.0 {
            let slack = state.x_u[i] - state.x[i];
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
fn fraction_to_boundary_primal_s(state: &SolverState, ds: &[f64], tau: f64) -> f64 {
    let mut alpha = 1.0_f64;
    for i in 0..state.m {
        let l_fin = state.g_l[i].is_finite();
        let u_fin = state.g_u[i].is_finite();
        if l_fin && u_fin && (state.g_l[i] - state.g_u[i]).abs() < 1e-14 {
            continue;
        }
        if l_fin && ds[i] < 0.0 {
            let slack = (state.s[i] - state.g_l[i]).max(0.0);
            alpha = alpha.min(-tau * slack / ds[i]);
        }
        if u_fin && ds[i] > 0.0 {
            let slack = (state.g_u[i] - state.s[i]).max(0.0);
            alpha = alpha.min(tau * slack / ds[i]);
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
    dy: Vec<f64>,
    ds: Vec<f64>,
    dz_l: Vec<f64>,
    dz_u: Vec<f64>,
    dv_l: Vec<f64>,
    dv_u: Vec<f64>,
) {
    state.dx = dx;
    state.dy = dy;
    state.ds = ds;
    state.dz_l = dz_l;
    state.dz_u = dz_u;
    state.dv_l = dv_l;
    state.dv_u = dv_u;
}

/// L-infinity norm of `J^T * c_violation`, where `c_violation` is the
/// signed constraint residual (g - g_l for equalities or below-lower
/// violations, g - g_u for above-upper violations, 0 otherwise). Used
/// by infeasibility-classification heuristics: when ‖∇θ‖_∞ ≈ 0 with
/// θ > 0 the iterate is a stationary point of the feasibility merit
/// function, so the problem is locally infeasible.
fn compute_grad_theta_norm(state: &SolverState) -> f64 {
    let n = state.n;
    let m = state.m;
    let mut violation = vec![0.0; m];
    for i in 0..m {
        if constraint_is_equality(state, i) {
            violation[i] = state.g[i] - state.g_l[i];
        } else if state.g_l[i].is_finite() && state.g[i] < state.g_l[i] {
            violation[i] = state.g[i] - state.g_l[i];
        } else if state.g_u[i].is_finite() && state.g[i] > state.g_u[i] {
            violation[i] = state.g[i] - state.g_u[i];
        }
    }
    let mut grad_theta = vec![0.0; n];
    for (idx, (&row, &col)) in state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate() {
        grad_theta[col] += state.jac_vals[idx] * violation[row];
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
    let x_part = convergence::dual_infeasibility(
        &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
        &state.y, &state.z_l, &state.z_u, state.n,
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
        let r = -state.y[i] - state.v_l[i] + state.v_u[i];
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
    convergence::dual_infeasibility(
        &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
        &state.y, z_l, z_u, state.n,
    )
}

/// L-infinity primal infeasibility at the current iterate against
/// the current state's `g_l`/`g_u` bounds. Centralises the five
/// state-arg call sites of `convergence::primal_infeasibility_max`.
fn compute_primal_inf_max_at_state(state: &SolverState) -> f64 {
    convergence::primal_infeasibility_max(&state.g, &state.g_l, &state.g_u)
}

/// L-infinity slack-coupling residual at the current iterate
/// (`||c||_∞ ∪ ||d − s||_∞`). Used by the scaled (barrier-level)
/// convergence test. Mirrors Ipopt's
/// `IpIpoptCalculatedQuantities::curr_primal_infeasibility(NORM_MAX)`.
fn compute_primal_inf_internal_max_at_state(state: &SolverState) -> f64 {
    convergence::primal_infeasibility_internal_max(
        &state.g, &state.s, &state.g_l, &state.g_u,
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
    let x_part = convergence::dual_infeasibility_scaled(
        &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
        &state.y, &state.z_l, &state.z_u, state.n,
    );
    x_part.max(slack_dual_inf_max(state))
}

/// `convergence::complementarity_error` at the current iterate using
/// caller-supplied `z_l`/`z_u` (typically the active-set z recovered
/// by [`recover_active_set_z`] for an optimistic optimality probe)
/// instead of `state.z_l`/`state.z_u`. Always evaluated with `μ = 0`
/// (the optimality complementarity rather than the centered-path one).
fn compl_err_with_z(state: &SolverState, z_l: &[f64], z_u: &[f64]) -> f64 {
    convergence::complementarity_error(
        &state.x, &state.x_l, &state.x_u, z_l, z_u, 0.0,
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
    convergence::complementarity_error_full(
        &state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u,
        &state.g, &state.g_l, &state.g_u, &state.v_l, &state.v_u, 0.0,
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
    let mut residual = vec![0.0; n];
    residual[..n].copy_from_slice(&state.grad_f[..n]);
    for (idx, (&row, &col)) in state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate() {
        residual[col] += state.jac_vals[idx] * state.y[row];
    }
    for i in 0..n {
        residual[i] -= state.z_l[i];
        residual[i] += state.z_u[i];
    }
    let dual_l1: f64 = residual.iter().map(|r| r.abs()).sum();

    let primal_l1 = convergence::primal_infeasibility(&state.g, &state.g_l, &state.g_u);

    // Complementarity 1-norm: |slack·z - μ| over finite bounds, averaged
    // by the count of contributing entries (n_compl). When there are no
    // finite bounds, drop the term.
    let mut compl_l1 = 0.0f64;
    let mut n_compl = 0usize;
    for i in 0..n {
        if state.x_l[i].is_finite() {
            let slack = (state.x[i] - state.x_l[i]).max(0.0);
            compl_l1 += (slack * state.z_l[i] - mu).abs();
            n_compl += 1;
        }
        if state.x_u[i].is_finite() {
            let slack = (state.x_u[i] - state.x[i]).max(0.0);
            compl_l1 += (slack * state.z_u[i] - mu).abs();
            n_compl += 1;
        }
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
    state.g_l[i].is_finite() && state.g_u[i].is_finite()
        && (state.g_l[i] - state.g_u[i]).abs() < 1e-15
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
    let n = state.n;
    for i in 0..n {
        if state.x_l[i].is_finite() {
            state.z_l[i] = mu / slack_xl(state, i);
        }
        if state.x_u[i].is_finite() {
            state.z_u[i] = mu / slack_xu(state, i);
        }
    }
}

/// Boost μ to `new_mu`, reset the filter and overall-progress stall
/// counters, and pin the strategy in Fixed mode. Used by the two
/// stall-recovery branches (`handle_near_tolerance_stall` near-tol
/// boost and `try_boost_mu_for_stall`) — both first decide a target
/// μ from `primal_inf_max`, log it, then run this exact mutation.
/// Does NOT increment `diagnostics.mu_mode_switches` (these
/// stall-driven flips are tracked separately from the Free↔Fixed
/// transitions in `switch_mu_mode`).
fn boost_mu_and_switch_to_fixed_with_stall_reset(
    state: &mut SolverState,
    new_mu: f64,
    mu_state: &mut MuState,
    filter: &mut Filter,
    stall: &mut ProgressStallTracker,
) {
    state.mu = new_mu;
    reset_stall_counters_and_filter(state, filter, stall);
    mu_state.mode = MuMode::Fixed;
    mu_state.first_iter_in_mode = true;
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
/// without that guard `state.x_l[i]` may be -inf and the result is
/// undefined.
fn slack_xl(state: &SolverState, i: usize) -> f64 {
    (state.x[i] - state.x_l[i]).max(1e-20)
}

/// Strictly-positive upper-bound primal slack `max(x_u - x, 1e-20)`.
/// See [`slack_xl`] for the finite-guard contract.
fn slack_xu(state: &SolverState, i: usize) -> f64 {
    (state.x_u[i] - state.x[i]).max(1e-20)
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
    (state.s[i] - state.g_l[i]).max(1e-20)
}

/// Strictly-positive upper-side constraint slack `max(g_u - s, 1e-20)`.
/// See [`slack_gl`] for the finite-guard contract and the `state.s`
/// rationale.
fn slack_gu(state: &SolverState, i: usize) -> f64 {
    (state.g_u[i] - state.s[i]).max(1e-20)
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
    for i in 0..state.m {
        let is_eq = state.g_l[i].is_finite()
            && state.g_u[i].is_finite()
            && (state.g_l[i] - state.g_u[i]).abs() < 1e-14;
        state.s[i] = if is_eq { state.g_l[i] } else { state.g[i] };
    }
}

/// Build a trial point `x + alpha * dx`, clamped strictly inside the
/// finite bounds via `clamp_to_open_bounds`. Used by the regular line
/// search, soft-restoration acceptance, gradient-descent fallback, and
/// the second-order correction.
fn compute_clamped_trial_x(state: &SolverState, dx: &[f64], alpha: f64) -> Vec<f64> {
    let n = state.n;
    let mut x_trial = vec![0.0; n];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        x_trial[i] = state.x[i] + alpha * dx[i];
        clamp_to_open_bounds(&mut x_trial, &state.x_l, &state.x_u, i);
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

/// `kkt::compute_sigma` at the current iterate's
/// `state.{x, x_l, x_u, z_l, z_u}`. Centralises the two callers —
/// assemble_kkt_systems (main solve) and the perturbation recovery
/// path in try_early_perturbation_recovery.
fn compute_sigma_from_state(state: &SolverState) -> Vec<f64> {
    kkt::compute_sigma(&state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u)
}

/// `kkt::recover_dz` (Fiacco bound-multiplier step recovery) at the
/// current iterate's `state.{x, x_l, x_u, z_l, z_u}` for a given
/// primal direction `dx` and centering target `mu`. Centralises the
/// two callers — the main-step recovery in solve_for_search_direction
/// and the Mehrotra affine-predictor probe.
fn recover_dz_from_state(state: &SolverState, dx: &[f64], mu: f64) -> (Vec<f64>, Vec<f64>) {
    kkt::recover_dz(
        &state.x, &state.x_l, &state.x_u,
        &state.z_l, &state.z_u, dx, mu,
    )
}

/// Recover slack-bound multiplier steps `dv_L`, `dv_U` from the current
/// iterate's `state.{g, g_l, g_u, v_l, v_u}` and Jacobian, for a given
/// primal direction `dx` and centering target `mu`.
fn recover_dv_from_state(state: &SolverState, ds: &[f64], mu: f64) -> (Vec<f64>, Vec<f64>) {
    kkt::recover_dv(
        state.m,
        &state.g_l, &state.g_u, &state.s,
        &state.v_l, &state.v_u, ds, mu,
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
    kkt::recover_ds(
        state.n, state.m,
        &state.jac_rows, &state.jac_cols, &state.jac_vals,
        &state.g, &state.g_l, &state.g_u, &state.s,
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
    let mut sys = kkt::assemble_kkt(
        n, m,
        &state.hess_rows, &state.hess_cols, &state.hess_vals,
        &state.jac_rows, &state.jac_cols, &state.jac_vals,
        sigma, &state.grad_f, &state.g, &state.g_l, &state.g_u,
        &state.s, &state.y, &state.z_l, &state.z_u,
        &state.x, &state.x_l, &state.x_u, state.mu, kappa_d,
        use_sparse, &state.v_l, &state.v_u,
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
        let alpha_dx_inf = alpha * dx_inf;
        let mut x_inf = 0.0_f64;
        let mut diff_inf = 0.0_f64;
        for i in 0..state.n {
            let xi = state.x[i].abs();
            if xi > x_inf { x_inf = xi; }
            let d = (x_trial[i] - state.x[i]).abs();
            if d > diff_inf { diff_inf = d; }
        }
        let dy_inf = linf_norm(&state.dy);
        let rel = if x_inf > 0.0 { diff_inf / x_inf } else { diff_inf };
        // Σ-pin diagnostic: smallest x-slack and largest z give the worst
        // diagonal entry of W + Σ. If min_slack · max_z >> κ_σ·μ the κ_σ
        // clamp has failed to keep z*s in band.
        let mut min_s_x = f64::INFINITY;
        let mut min_s_idx = usize::MAX;
        let mut min_s_side = "";
        let mut max_z_x = 0.0_f64;
        for i in 0..state.n {
            if state.x_l[i].is_finite() {
                let s = state.x[i] - state.x_l[i];
                if s > 0.0 && s < min_s_x { min_s_x = s; min_s_idx = i; min_s_side = "L"; }
            }
            if state.x_u[i].is_finite() {
                let s = state.x_u[i] - state.x[i];
                if s > 0.0 && s < min_s_x { min_s_x = s; min_s_idx = i; min_s_side = "U"; }
            }
            if state.z_l[i].abs() > max_z_x { max_z_x = state.z_l[i].abs(); }
            if state.z_u[i].abs() > max_z_x { max_z_x = state.z_u[i].abs(); }
        }
        let (xv, bv) = if min_s_idx < state.n {
            let xv = state.x[min_s_idx];
            let bv = if min_s_side == "L" { state.x_l[min_s_idx] } else { state.x_u[min_s_idx] };
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
    state.g = g_trial;
    // Advance the slack iterate with the same primal step length.
    // Equality rows (`g_l == g_u`) have `ds = 0` (forced by `recover_ds`)
    // and `s` stays at the equality value as a sentinel. For inequality
    // rows the line-searched α_p is the same fraction of the FTB-capped
    // step taken on x, by Ipopt's `alpha_for_y = primal` default
    // (IpFilterLSAcceptor.cpp:617-628, IpIteratesVector.hpp).
    for i in 0..state.m {
        let is_eq = state.g_l[i].is_finite()
            && state.g_u[i].is_finite()
            && (state.g_l[i] - state.g_u[i]).abs() < 1e-14;
        if is_eq {
            state.s[i] = state.g_l[i];
        } else {
            state.s[i] += alpha * state.ds[i];
        }
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
    l1_norm(&state.y)
        + l1_norm(&state.z_l)
        + l1_norm(&state.z_u)
        + l1_norm(&state.v_l)
        + l1_norm(&state.v_u)
}

/// Sum of absolute values of bound-side multipliers contributing to
/// `s_c`: `z_l + z_u + v_l + v_u`
/// (`IpIpoptCalculatedQuantities.cpp:3677-3687`). Used with
/// `compute_bound_multiplier_count` (finite-bound count over
/// x and inequality g rows) to form the complementarity scaling `s_c`.
fn compute_bound_multiplier_sum(state: &SolverState) -> f64 {
    l1_norm(&state.z_l)
        + l1_norm(&state.z_u)
        + l1_norm(&state.v_l)
        + l1_norm(&state.v_u)
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
    let mut n_bound = 0usize;
    for i in 0..state.n {
        if state.x_l[i].is_finite() {
            n_bound += 1;
        }
        if state.x_u[i].is_finite() {
            n_bound += 1;
        }
    }
    for i in 0..state.m {
        if constraint_is_equality(state, i) {
            continue;
        }
        if state.g_l[i].is_finite() {
            n_bound += 1;
        }
        if state.g_u[i].is_finite() {
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
    // Variable bound complementarity: z_l * (x - x_l), z_u * (x_u - x)
    for i in 0..state.n {
        if state.x_l[i].is_finite() {
            sum_compl += slack_xl(state, i) * state.z_l[i];
            count += 1;
        }
        if state.x_u[i].is_finite() {
            sum_compl += slack_xu(state, i) * state.z_u[i];
            count += 1;
        }
    }
    // B-cross6: include slack-side complementarity v_L·s_L, v_U·s_U
    // unconditionally, matching Ipopt's `IpIpoptCalculatedQuantities::
    // curr_avrg_compl` which sums over all four bound blocks. The
    // previously-conditional fallback (only when no variable bounds
    // existed) was a ripopt-specific tuning that left `avg_compl` blind
    // to slack centrality on mixed-bound problems.
    for i in 0..state.m {
        let l_fin = state.g_l[i].is_finite();
        let u_fin = state.g_u[i].is_finite();
        if l_fin && u_fin && (state.g_l[i] - state.g_u[i]).abs() < 1e-14 {
            continue;
        }
        if l_fin {
            sum_compl += state.v_l[i] * slack_gl(state, i);
            count += 1;
        }
        if u_fin {
            sum_compl += state.v_u[i] * slack_gu(state, i);
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
    let n = state.n;

    // ∇L = ∇f + J^T y − z_l + z_u (un-damped; matches
    // `curr_grad_lag_x` at IpIpoptCalculatedQuantities.cpp:1993-2030)
    let mut grad_lag = state.grad_f.clone();
    accumulate_jt_y(state, &mut grad_lag);
    for i in 0..n {
        if state.x_l[i].is_finite() {
            grad_lag[i] -= state.z_l[i];
        }
        if state.x_u[i].is_finite() {
            grad_lag[i] += state.z_u[i];
        }
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
    for i in 0..n {
        if state.x_l[i].is_finite() {
            let r = (slack_xl(state, i) * state.z_l[i] - state.mu).abs();
            if r > compl_max {
                compl_max = r;
            }
        }
        if state.x_u[i].is_finite() {
            let r = (slack_xu(state, i) * state.z_u[i] - state.mu).abs();
            if r > compl_max {
                compl_max = r;
            }
        }
    }
    let compl_err = compl_max / s_c;

    // Primal infeasibility ‖c‖_∞ (no s-divisor in Ipopt;
    // `curr_primal_infeasibility(NORM_MAX)` at line 2570-2610).
    // `state.constraint_violation()` returns the L1 sum used by the
    // filter; the barrier-error gate needs the L∞ version.
    let primal_err = convergence::primal_infeasibility_max(&state.g, &state.g_l, &state.g_u);

    dual_err.max(compl_err).max(primal_err)
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
    let mut z_l_out = vec![0.0; n];
    let mut z_u_out = vec![0.0; n];
    for i in 0..n {
        z_l_out[i] = state.z_l[i] / state.obj_scaling;
        z_u_out[i] = state.z_u[i] / state.obj_scaling;
    }
    let mut y_out = state.y.clone();
    for i in 0..m {
        y_out[i] = state.y[i] * state.g_scaling[i] / state.obj_scaling;
    }
    let mut g_out = state.g.clone();
    for i in 0..m {
        g_out[i] /= state.g_scaling[i];
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
            kappa_d: 0.0,
            x: vec![0.0; n],
            y: vec![0.0; m],
            z_l: vec![0.0; n],
            z_u: vec![0.0; n],
            v_l: vec![0.0; m],
            v_u: vec![0.0; m],
            dx: vec![0.0; n],
            dy: vec![0.0; m],
            dz_l: vec![0.0; n],
            dz_u: vec![0.0; n],
            dv_l: vec![0.0; m],
            dv_u: vec![0.0; m],
            s: vec![0.0; m],
            ds: vec![0.0; m],
            mu: 0.1,
            alpha_primal: 0.0,
            alpha_dual: 0.0,
            iter: 0,
            x_l: vec![f64::NEG_INFINITY; n],
            x_u: vec![f64::INFINITY; n],
            g_l: vec![f64::NEG_INFINITY; m],
            g_u: vec![f64::INFINITY; m],
            n,
            m,
            obj: 0.0,
            grad_f: vec![0.0; n],
            g: vec![0.0; m],
            jac_rows: Vec::new(),
            jac_cols: Vec::new(),
            jac_vals: Vec::new(),
            hess_rows: Vec::new(),
            hess_cols: Vec::new(),
            hess_vals: Vec::new(),
            consecutive_acceptable: 0,
            obj_scaling: 1.0,
            g_scaling: vec![1.0; m],
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

    #[test]
    fn test_iterate_snapshot_capture_and_restore() {
        let mut state = minimal_state(2, 1);
        state.x = vec![1.5, 2.5];
        state.y = vec![0.7];
        state.z_l = vec![0.1, 0.2];
        state.z_u = vec![0.3, 0.4];
        state.mu = 1e-3;
        state.obj = 42.0;
        let mut filter = Filter::new(1e4);
        filter.add(0.5, 10.0);
        let snap = IterateSnapshot::capture(&state, &filter, 7);
        // Mutate state and filter to simulate further iterations.
        state.x = vec![9.0, 9.0];
        state.y = vec![9.0];
        state.mu = 1.0;
        state.obj = 0.0;
        filter.add(99.0, 99.0);
        // Restore and verify.
        snap.restore(&mut state, &mut filter);
        assert_eq!(state.x, vec![1.5, 2.5]);
        assert_eq!(state.y, vec![0.7]);
        assert_eq!(state.z_l, vec![0.1, 0.2]);
        assert_eq!(state.z_u, vec![0.3, 0.4]);
        assert_eq!(state.mu, 1e-3);
        assert_eq!(state.obj, 42.0);
        assert_eq!(snap.iteration, 7);
        assert_eq!(filter.entries().len(), 1);
        assert!((filter.entries()[0].theta - 0.5).abs() < 1e-15);
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
        state.x_l = vec![1.0, 1.0];
        state.z_l = vec![2.0, 3.0];
        let avg = compute_avg_complementarity(&state);
        assert!((avg - 2.0).abs() < 1e-12, "expected 2.0, got {}", avg);
    }

    #[test]
    fn test_avg_compl_both_bounds() {
        // 1 var, both bounds. x = 1.5, x_l = 1.0, x_u = 2.0, z_l = 2.0, z_u = 3.0.
        // avg = (0.5*2.0 + 0.5*3.0) / 2 = 2.5 / 2 = 1.25
        let mut state = minimal_state(1, 0);
        state.x = vec![1.5];
        state.x_l = vec![1.0];
        state.x_u = vec![2.0];
        state.z_l = vec![2.0];
        state.z_u = vec![3.0];
        let avg = compute_avg_complementarity(&state);
        assert!((avg - 1.25).abs() < 1e-12, "expected 1.25, got {}", avg);
    }

    #[test]
    fn test_avg_compl_inequality_fallback() {
        // No variable bounds, but an inequality constraint with v_l > 0 triggers fallback.
        // g = 2.0, g_l = 1.0, v_l = 0.5 -> slack = 1.0, contrib = 0.5; avg = 0.5 / 1 = 0.5.
        let mut state = minimal_state(1, 1);
        state.g = vec![2.0];
        state.g_l = vec![1.0];
        state.g_u = vec![f64::INFINITY];
        state.v_l = vec![0.5];
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
        state.x_l = vec![1.0];
        state.z_l = vec![1.0];
        state.g = vec![2.0];
        state.g_l = vec![1.0];
        state.v_l = vec![99.0];
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
        state.x_l[0] = 0.0; // finite lower on var 0
        // var 1 and var 2: x_l = -inf, x_u = +inf (default)
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
        state.x_l[0] = 0.0;
        state.x_u[0] = 10.0;
        state.g_l[0] = 0.0;
        state.g_u[0] = f64::INFINITY;
        state.g_l[1] = 5.0;
        state.g_u[1] = 5.0;
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
        state.g = vec![3.0];
        state.g_l = vec![1.0];
        state.g_u = vec![f64::INFINITY];
        state.v_l = vec![0.5];
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
        state.g = vec![1.0];
        state.g_l = vec![f64::NEG_INFINITY];
        state.g_u = vec![4.0];
        state.v_u = vec![0.25];
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
        state.x_l[0] = 0.0;
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

    /// Spec §7.8: when restoration leaves x unchanged (s_cur == s_trial),
    /// the one-shot Newton step δz = μ/s − z drives z to μ/s exactly,
    /// then κ_σ safeguards. Mirrors Ipopt IpRestoMinC_1Nrm.cpp:378-399.
    #[test]
    fn test_one_shot_newton_z_no_x_step_recovers_mu_slack() {
        let mut state = minimal_state(1, 0);
        state.x = vec![1.1];
        state.x_l = vec![1.0];
        state.x_u = vec![f64::INFINITY];
        state.mu = 0.01;
        // s_cur = s_trial = 0.1; z_in = 0.5 → δz = μ/s − z = -0.4.
        // α_d FTB: −0.99·0.5 / −0.4 = 1.2375 → clamp to α=1.
        // z_new = 0.5 − 0.4 = 0.1 = μ/s. κ_σ band [1e-12, 1e9] → unchanged.
        let zl_resto = vec![0.5];
        let zu_resto = vec![0.0];
        let x_cur = vec![1.1];
        let nuclear = apply_kappa_sigma_clamp_to_resto_z(
            &mut state, 1, &zl_resto, &zu_resto, &x_cur,
        );
        assert!((state.z_l[0] - 0.1).abs() < 1e-12,
            "z_l should converge to μ/s = 0.1 via one-shot Newton, got {}",
            state.z_l[0]);
        assert_eq!(state.z_u[0], 0.0, "infinite x_u → z_u = 0");
        assert!(!nuclear, "z_max=0.1 should not trigger nuclear-reset flag");
    }

    /// Spec §7.8: a wildly large resto z must be clamped down by the
    /// κ_σ upper band (κ_σ·μ/s_trial). With z huge and s_cur ≈ s_trial,
    /// the Newton δz is large and negative; α_d FTB caps the step at
    /// (1−τ)·z_in (~1% of original), then κ_σ takes over.
    #[test]
    fn test_one_shot_newton_z_caps_huge_resto_z() {
        let mut state = minimal_state(1, 0);
        state.x = vec![1.1];
        state.x_l = vec![1.0];
        state.x_u = vec![f64::INFINITY];
        state.mu = 1e-6;
        // s_cur=s_trial=0.1, μ=1e-6 → κ_σ upper = 1e10·1e-6/0.1 = 1e5.
        // z_in=1e20: δz = μ/s − z = 1e-5 − 1e20 ≈ −1e20.
        // α_d FTB: −0.99·1e20 / −1e20 = 0.99 → step = 0.99·(−1e20).
        // z_new = 1e20·(1 − 0.99) = 1e18. κ_σ clamp → 1e5.
        let zl_resto = vec![1.0e20];
        let zu_resto = vec![0.0];
        let x_cur = vec![1.1];
        apply_kappa_sigma_clamp_to_resto_z(
            &mut state, 1, &zl_resto, &zu_resto, &x_cur,
        );
        let z_hi = 1e10 * 1e-6 / 0.1;
        assert!((state.z_l[0] - z_hi).abs() / z_hi < 1e-12,
            "z_l should be clamped to κ_σ·μ/s = {}, got {}", z_hi, state.z_l[0]);
    }

    /// Spec §7.8: when restoration moves x toward the bound (s shrinks
    /// from s_cur=0.5 to s_trial=0.1), the Newton step should bias z
    /// upward to maintain z·s ≈ μ. Verify δz computation directly.
    #[test]
    fn test_one_shot_newton_z_with_x_step_uses_s_cur_in_formula() {
        let mut state = minimal_state(1, 0);
        // x moved from 1.5 (s=0.5) to 1.1 (s=0.1).
        state.x = vec![1.1];
        state.x_l = vec![1.0];
        state.x_u = vec![f64::INFINITY];
        state.mu = 0.05;
        // z_in = 0.1, s_cur = 0.5, s_trial = 0.1, μ = 0.05.
        // δz = (μ − z·(s_trial−s_cur))/s_cur − z
        //    = (0.05 − 0.1·(−0.4))/0.5 − 0.1
        //    = (0.05 + 0.04)/0.5 − 0.1
        //    = 0.18 − 0.1 = 0.08
        // δz > 0 → α=1, z_new = 0.18. κ_σ band at s=0.1: [5e-12, 5e8] → kept.
        let zl_resto = vec![0.1];
        let zu_resto = vec![0.0];
        let x_cur = vec![1.5];
        apply_kappa_sigma_clamp_to_resto_z(
            &mut state, 1, &zl_resto, &zu_resto, &x_cur,
        );
        assert!((state.z_l[0] - 0.18).abs() < 1e-12,
            "z_l should be 0.18 from Newton step, got {}", state.z_l[0]);
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
        state.g = vec![0.5];
        state.g_l = vec![0.0];
        state.g_u = vec![0.0];
        state.x_l = vec![f64::NEG_INFINITY];
        state.x_u = vec![f64::INFINITY];
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
        assert!((filter.entries()[0].theta - 1.0).abs() < 1e-15,
            "pre-existing filter entry theta must persist");
        assert!((filter.entries()[0].phi - 1.0).abs() < 1e-15,
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
        state.g = vec![0.5];
        state.g_l = vec![0.0];
        state.g_u = vec![0.0];
        state.x_l = vec![f64::NEG_INFINITY];
        state.x_u = vec![f64::INFINITY];
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
        assert!((filter.entries()[0].theta - 1.0).abs() < 1e-15);
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
        // Default option: RelaxBounds. x_l = x_u = 5.0 should be widened
        // to [5 - 5e-8, 5 + 5e-8].
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
        s.jac_vals = vec![1.0];
        s.g = vec![g];
        s.y = vec![999.0];          // sentinel; recalc must overwrite
        if g_eq {
            s.g_l = vec![0.0];
            s.g_u = vec![0.0];
        }
        s
    }

    #[test]
    fn test_recalc_y_off_by_default_does_not_overwrite() {
        let mut state = ls_y_equals_two_state(0.0, true);
        let opts = SolverOptions::default();
        maybe_recalc_y_post_step(&mut state, &opts, 1, 1, false);
        assert_eq!(state.y, vec![999.0], "default off must not touch y");
    }

    #[test]
    fn test_recalc_y_lbfgs_mode_recomputes_when_feasible() {
        let mut state = ls_y_equals_two_state(0.0, true); // viol = 0 < tol
        let opts = SolverOptions::default();
        maybe_recalc_y_post_step(&mut state, &opts, 1, 1, true);
        assert!((state.y[0] - 2.0).abs() < 1e-10, "lbfgs gate must recompute, got {}", state.y[0]);
    }

    #[test]
    fn test_recalc_y_explicit_on_recomputes_when_feasible() {
        let mut state = ls_y_equals_two_state(0.0, true);
        let opts = SolverOptions { recalc_y: true, ..SolverOptions::default() };
        maybe_recalc_y_post_step(&mut state, &opts, 1, 1, false);
        assert!((state.y[0] - 2.0).abs() < 1e-10, "recalc_y=true must recompute, got {}", state.y[0]);
    }

    #[test]
    fn test_recalc_y_skipped_when_constraint_violation_above_tol() {
        // g = 1e-3, equality constraint => constraint_violation = 1e-3 > recalc_y_feas_tol (1e-6)
        let mut state = ls_y_equals_two_state(1e-3, true);
        let opts = SolverOptions { recalc_y: true, ..SolverOptions::default() };
        maybe_recalc_y_post_step(&mut state, &opts, 1, 1, false);
        assert_eq!(state.y, vec![999.0], "infeasible iterate must skip recalc_y");
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
        state.g_l = vec![0.0];
        state.g_u = vec![f64::INFINITY];
        state.v_l = vec![1.0];
        state.v_u = vec![0.0];
        let opts = SolverOptions { recalc_y: true, ..SolverOptions::default() };
        maybe_recalc_y_post_step(&mut state, &opts, 1, 1, false);
        assert!(
            (state.y[0] - 0.5).abs() < 1e-10,
            "full-augmented LS y mismatch: got {} expected 0.5", state.y[0]
        );
    }

    /// T3.30: on equality rows the full and reduced systems must agree.
    /// Ipopt's slack/v_d coupling only enters for inequality rows.
    #[test]
    fn test_recalc_y_full_augmented_equality_matches_reduced() {
        let mut state = ls_y_equals_two_state(0.0, true);
        // Equality row: v_L/v_U values must be ignored.
        state.v_l = vec![100.0];
        state.v_u = vec![50.0];
        let opts = SolverOptions { recalc_y: true, ..SolverOptions::default() };
        maybe_recalc_y_post_step(&mut state, &opts, 1, 1, false);
        assert!(
            (state.y[0] - 2.0).abs() < 1e-10,
            "equality row should match reduced: got {}", state.y[0]
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
        state.g_l = vec![0.0];
        state.g_u = vec![0.0];
        state.y = vec![0.5];
        state.dy = vec![-3.0];
        state.jac_rows = vec![0];
        state.jac_cols = vec![0];
        state.jac_vals = vec![1.0];

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
        state.g_l = vec![0.0];
        state.g_u = vec![0.0];
        state.y = vec![0.5];
        state.dy = vec![-3.0];
        state.jac_rows = vec![0];
        state.jac_cols = vec![0];
        state.jac_vals = vec![1.0];

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
        state.y = vec![1.5, -0.5];
        state.z_l = vec![0.7, 0.2];
        state.z_u = vec![0.0, 0.3];
        state.v_l = vec![0.4, 0.0];
        state.v_u = vec![0.0, 0.6];
        state.g_l = vec![0.0, f64::NEG_INFINITY];
        state.g_u = vec![f64::INFINITY, 5.0];
        state.g = vec![1.0, 2.0];
        state.mu = 0.1;
        let snapshot = (
            state.x.clone(),
            state.y.clone(),
            state.z_l.clone(),
            state.z_u.clone(),
            state.v_l.clone(),
            state.v_u.clone(),
            state.g.clone(),
        );

        let opts_on = SolverOptions { magic_step: true, ..SolverOptions::default() };
        let n_on = apply_magic_step(&mut state, &opts_on);
        assert_eq!(n_on, 0, "no explicit slack vector exists, so no updates possible");
        assert_eq!(state.x, snapshot.0);
        assert_eq!(state.y, snapshot.1);
        assert_eq!(state.z_l, snapshot.2);
        assert_eq!(state.z_u, snapshot.3);
        assert_eq!(state.v_l, snapshot.4);
        assert_eq!(state.v_u, snapshot.5);
        assert_eq!(state.g, snapshot.6);

        // With option off, identical (no-op) result.
        let opts_off = SolverOptions { magic_step: false, ..SolverOptions::default() };
        let n_off = apply_magic_step(&mut state, &opts_off);
        assert_eq!(n_off, 0);
        assert_eq!(state.x, snapshot.0);
        assert_eq!(state.y, snapshot.1);
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
        s.x_l = vec![0.0, 0.0];
        s.x_u = vec![f64::INFINITY, f64::INFINITY];
        s.z_l = vec![0.5, 0.5];
        s.z_u = vec![0.0, 0.0];
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
        state.z_l = vec![0.001, 5.0];
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
    /// only sets `mu_state.tiny_step` and bumps `consecutive_tiny_steps`.
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
        let mut consecutive: usize = 1;

        detect_tiny_step(
            &mut state, &opts, &mut mu_state, &mut filter,
            &mut consecutive, 0.0,
        );

        assert!(mu_state.tiny_step, "tiny_step flag must be set");
        assert_eq!(consecutive, 2, "consecutive counter must increment");
        assert_eq!(state.mu, initial_mu,
            "detect_tiny_step must not mutate mu (Ipopt §IpBacktrackingLineSearch)");
        assert_eq!(filter.len(), initial_filter_len,
            "detect_tiny_step must not reset/augment the filter");
    }

    /// When the relative step exceeds `10·eps`, the flag must clear and
    /// the consecutive counter must reset.
    #[test]
    fn test_detect_tiny_step_clears_when_step_grows() {
        let mut state = minimal_state(1, 0);
        state.x = vec![1.0];
        state.dx = vec![0.5];

        let opts = SolverOptions::default();
        let mut mu_state = MuState::new();
        mu_state.tiny_step = true;
        let mut filter = crate::filter::Filter::new(1e10);
        let mut consecutive: usize = 1;

        detect_tiny_step(
            &mut state, &opts, &mut mu_state, &mut filter,
            &mut consecutive, 0.0,
        );

        assert!(!mu_state.tiny_step, "tiny_step must clear on a real step");
        assert_eq!(consecutive, 0, "consecutive counter must reset");
    }

    /// A tiny x-step combined with a *large* dual step must NOT latch
    /// the tiny-step flag (Ipopt's `tiny_step_y_tol` gate at
    /// `IpBacktrackingLineSearch.cpp:421-424`).
    #[test]
    fn test_detect_tiny_step_blocked_by_large_dy() {
        let mut state = minimal_state(1, 1);
        state.x = vec![1.0];
        state.dx = vec![1e-20];
        state.y = vec![0.0];
        state.dy = vec![1.0]; // |dy|/(1+|y|) = 1.0, well above default 1e-2

        let opts = SolverOptions::default();
        let mut mu_state = MuState::new();
        let mut filter = crate::filter::Filter::new(1e10);
        let mut consecutive: usize = 1;

        detect_tiny_step(
            &mut state, &opts, &mut mu_state, &mut filter,
            &mut consecutive, 0.0,
        );

        assert!(!mu_state.tiny_step,
            "tiny x-step must not latch tiny_step when dual step is still large");
        assert_eq!(consecutive, 0, "counter must reset when dy gate fails");
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
            grad_phi_step, 1.0,
        );
        assert!(acceptable, "trial must pass the filter for the test setup");
        let len_before = filter.len();
        assert_eq!(filter.len(), len_before,
            "watchdog branch must not augment filter on accept");
        let (acceptable_next, _) = filter.check_acceptability(
            theta_current, phi_current,
            theta_current * 0.95, phi_current,
            grad_phi_step, 1.0,
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
