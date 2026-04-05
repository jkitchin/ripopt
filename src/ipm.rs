use std::time::{Duration, Instant};

use crate::convergence::{self, check_convergence, ConvergenceInfo, ConvergenceStatus};
use crate::filter::{self, Filter, FilterEntry};
use crate::kkt::{self, InertiaCorrectionParams};
use crate::linear_solver::banded::BandedLdl;
use crate::linear_solver::dense::DenseLdl;
#[cfg(all(feature = "faer", not(feature = "rmumps")))]
use crate::linear_solver::sparse::SparseLdl;
#[cfg(feature = "rmumps")]
use crate::linear_solver::multifrontal::MultifrontalLdl;
#[cfg(feature = "rmumps")]
use crate::linear_solver::iterative::IterativeMinres;
#[cfg(feature = "rmumps")]
use crate::linear_solver::hybrid::HybridSolver;
use crate::linear_solver::{KktMatrix, LinearSolver, SymmetricMatrix};
use crate::options::LinearSolverChoice;

/// Create a new sparse linear solver using the best available backend.
/// Prefers rmumps (multifrontal) when available, falls back to faer (SparseLdl).
fn new_sparse_solver() -> Box<dyn LinearSolver> {
    new_sparse_solver_with_choice(LinearSolverChoice::Direct)
}

/// Create a sparse linear solver with the specified choice.
fn new_sparse_solver_with_choice(choice: LinearSolverChoice) -> Box<dyn LinearSolver> {
    match choice {
        LinearSolverChoice::Direct => {
            #[cfg(feature = "rmumps")]
            { return Box::new(MultifrontalLdl::new()); }
            #[cfg(all(not(feature = "rmumps"), feature = "faer"))]
            { return Box::new(SparseLdl::new()); }
            #[cfg(not(any(feature = "rmumps", feature = "faer")))]
            { return Box::new(DenseLdl::new()); }
        }
        LinearSolverChoice::Iterative => {
            #[cfg(feature = "rmumps")]
            { return Box::new(IterativeMinres::new()); }
            #[cfg(not(feature = "rmumps"))]
            {
                log::warn!("Iterative solver requires rmumps feature; falling back to direct");
                return new_sparse_solver_with_choice(LinearSolverChoice::Direct);
            }
        }
        LinearSolverChoice::Hybrid => {
            #[cfg(feature = "rmumps")]
            { return Box::new(HybridSolver::new()); }
            #[cfg(not(feature = "rmumps"))]
            {
                log::warn!("Hybrid solver requires rmumps feature; falling back to direct");
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
use crate::options::SolverOptions;
use crate::problem::NlpProblem;
use crate::restoration::RestorationPhase;
use crate::restoration_nlp::RestorationNlp;
use crate::result::{SolveResult, SolverDiagnostics, SolveStatus};
use crate::slack_formulation::SlackFormulation;
use crate::warmstart::WarmStartInitializer;
use crate::logging::rip_log;

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
}

/// Saved state for the watchdog mechanism.
struct WatchdogSavedState {
    x: Vec<f64>,
    y: Vec<f64>,
    z_l: Vec<f64>,
    z_u: Vec<f64>,
    v_l: Vec<f64>,
    v_u: Vec<f64>,
    mu: f64,
    obj: f64,
    g: Vec<f64>,
    grad_f: Vec<f64>,
    filter_entries: Vec<FilterEntry>,
    theta: f64,
    phi: f64,
}

/// Central state struct for the IPM solver.
pub(crate) struct SolverState {
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

        let sy: f64 = s_k.iter().zip(y_k.iter()).map(|(s, y)| s * y).sum();

        // Compute B_k * s_k for Powell damping
        let bs = self.multiply_bk(&s_k);
        let sbs: f64 = s_k.iter().zip(bs.iter()).map(|(s, b)| s * b).sum();

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
        let sy_damped: f64 = s_k.iter().zip(y_k.iter()).map(|(s, y)| s * y).sum();
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

            let sbs: f64 = s.iter().zip(bs.iter()).map(|(si, bsi)| si * bsi).sum();
            let sy: f64 = s.iter().zip(y.iter()).map(|(si, yi)| si * yi).sum();

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
        for i in 0..n {
            if x_l[i] <= options.nlp_lower_bound_inf {
                x_l[i] = f64::NEG_INFINITY;
            }
            if x_u[i] >= options.nlp_upper_bound_inf {
                x_u[i] = f64::INFINITY;
            }
        }
        for i in 0..m {
            if g_l[i] <= options.nlp_lower_bound_inf {
                g_l[i] = f64::NEG_INFINITY;
            }
            if g_u[i] >= options.nlp_upper_bound_inf {
                g_u[i] = f64::INFINITY;
            }
        }

        let mut x = vec![0.0; n];
        problem.initial_point(&mut x);

        // Relax fixed variables: when x_l == x_u, the variable is fixed.
        // Interior-point methods require strictly interior starting points,
        // so we relax the bounds slightly (Ipopt's relax_bounds approach).
        for i in 0..n {
            if x_l[i].is_finite() && x_u[i].is_finite() && (x_u[i] - x_l[i]).abs() < 1e-10 {
                let center = (x_l[i] + x_u[i]) / 2.0;
                let relax = 1e-8 * center.abs().max(1.0);
                x_l[i] = center - relax;
                x_u[i] = center + relax;
            }
        }

        // Push initial point away from bounds
        for i in 0..n {
            if x_l[i].is_finite() && x_u[i].is_finite() {
                let range = x_u[i] - x_l[i];
                let push = options.bound_push.min(options.bound_frac * range);
                x[i] = x[i].max(x_l[i] + push).min(x_u[i] - push);
            } else if x_l[i].is_finite() {
                x[i] = x[i].max(x_l[i] + options.bound_push);
            } else if x_u[i].is_finite() {
                x[i] = x[i].min(x_u[i] - options.bound_push);
            }
        }

        // Initialize bound multipliers
        let mut z_l = vec![0.0; n];
        let mut z_u = vec![0.0; n];
        for i in 0..n {
            if x_l[i].is_finite() {
                let slack = (x[i] - x_l[i]).max(1e-20);
                z_l[i] = options.mu_init / slack;
            }
            if x_u[i].is_finite() {
                let slack = (x_u[i] - x[i]).max(1e-20);
                z_u[i] = options.mu_init / slack;
            }
        }

        let (jac_rows, jac_cols) = problem.jacobian_structure();
        let jac_nnz = jac_rows.len();
        let (hess_rows, hess_cols) = if options.hessian_approximation_lbfgs {
            dense_lower_triangle_pattern(n)
        } else {
            problem.hessian_structure()
        };
        let hess_nnz = hess_rows.len();

        // Initialize constraint multipliers via least-squares estimate if enabled.
        // Solves min ||∇f + J^T y||^2  ⟹  (J J^T) y = -J ∇f
        // Gate LS mult init on problem size: dense J*J^T is O(m^2*n), too slow for large problems
        let ls_init_dim_limit = 500;
        let y = if options.least_squares_mult_init && m > 0 && (m + n) <= ls_init_dim_limit {
            let mut grad_f_init = vec![0.0; n];
            let grad_ok = problem.gradient(&x, true, &mut grad_f_init);

            let mut jac_vals_init = vec![0.0; jac_nnz];
            let jac_ok = problem.jacobian_values(&x, false, &mut jac_vals_init);
            if !grad_ok || !jac_ok {
                // Evaluation failed during LS mult init; skip and use default multipliers
                vec![0.0; m]
            } else {
                compute_ls_multiplier_estimate(
                    &grad_f_init, &jac_rows, &jac_cols, &jac_vals_init,
                    &g_l, &g_u, n, m, options.constr_mult_init_max,
                ).unwrap_or_else(|| vec![0.0; m])
            }
        } else {
            vec![0.0; m]
        };

        Self {
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

            mu: options.mu_init,
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
        if !problem.objective(&self.x, new_x, &mut self.obj) { return false; }
        if !problem.gradient(&self.x, false, &mut self.grad_f) { return false; }
        if self.m > 0 {
            if !problem.constraints(&self.x, false, &mut self.g) { return false; }
            if !problem.jacobian_values(&self.x, false, &mut self.jac_vals) { return false; }
        }
        self.x_last_eval.copy_from_slice(&self.x);
        if skip_hessian {
            return true;
        }
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
        let mut phi = self.obj;
        for i in 0..self.n {
            if self.x_l[i].is_finite() {
                let slack = (self.x[i] - self.x_l[i]).max(1e-20);
                phi -= self.mu * slack.ln();
            }
            if self.x_u[i].is_finite() {
                let slack = (self.x_u[i] - self.x[i]).max(1e-20);
                phi -= self.mu * slack.ln();
            }
        }
        if options.constraint_slack_barrier {
            for i in 0..self.m {
                // Skip equality constraints (g_l == g_u): slack is zero by definition
                let is_eq = self.g_l[i].is_finite() && self.g_u[i].is_finite()
                    && (self.g_l[i] - self.g_u[i]).abs() < 1e-15;
                if is_eq {
                    continue;
                }
                if self.g_l[i].is_finite() {
                    let slack = self.g[i] - self.g_l[i];
                    if slack > self.mu * 1e-2 {
                        phi -= self.mu * slack.ln();
                    }
                }
                if self.g_u[i].is_finite() {
                    let slack = self.g_u[i] - self.g[i];
                    if slack > self.mu * 1e-2 {
                        phi -= self.mu * slack.ln();
                    }
                }
            }
        }
        phi
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
        for i in 0..self.n {
            let mut grad_phi_i = self.grad_f[i];
            if self.x_l[i].is_finite() {
                let slack = (self.x[i] - self.x_l[i]).max(1e-20);
                grad_phi_i -= self.mu / slack;
            }
            if self.x_u[i].is_finite() {
                let slack = (self.x_u[i] - self.x[i]).max(1e-20);
                grad_phi_i += self.mu / slack;
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
                let is_eq = self.g_l[i].is_finite() && self.g_u[i].is_finite()
                    && (self.g_l[i] - self.g_u[i]).abs() < 1e-15;
                if is_eq {
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

/// Wrapper that reformulates an overdetermined nonlinear equations (NE) problem
/// as an unconstrained least-squares problem.
///
/// Original:  min 0  subject to  g_i(x) = target_i,  i = 1,...,m   (m > n)
/// Reformulated:  min 0.5 * Σ (g_i(x) - target_i)^2   (no constraints, keep variable bounds)
///
/// Gradient: ∇f_LS = J^T * r   where r_i = g_i(x) - target_i
/// Hessian:  H_LS ≈ J^T * J   (Gauss-Newton approximation)
struct LeastSquaresProblem<'a, P: NlpProblem> {
    inner: &'a P,
    /// Targets for each constraint (g_l == g_u for equalities).
    targets: Vec<f64>,
    /// Jacobian structure cached from inner problem.
    jac_rows: Vec<usize>,
    jac_cols: Vec<usize>,
    /// Hessian structure: lower triangle of J^T*J + ∑ r_i ∇²g_i.
    hess_rows: Vec<usize>,
    hess_cols: Vec<usize>,
    /// Mapping from inner hessian entries to our dense lower triangle index.
    /// inner_hess_map[k] = index into our vals[] for inner hessian entry k.
    inner_hess_map: Vec<usize>,
}

impl<P: NlpProblem> LeastSquaresProblem<'_, P> {
    fn new(inner: &P) -> LeastSquaresProblem<'_, P> {
        let n = inner.num_variables();
        let m = inner.num_constraints();
        let mut g_l = vec![0.0; m];
        let mut g_u = vec![0.0; m];
        inner.constraint_bounds(&mut g_l, &mut g_u);
        let targets: Vec<f64> = (0..m).map(|i| 0.5 * (g_l[i] + g_u[i])).collect();

        let (jac_rows, jac_cols) = inner.jacobian_structure();

        // Build Hessian structure for J^T*J + ∑r_i∇²g_i (lower triangle, n x n dense).
        // Since J^T*J is generally dense, use full lower triangle.
        let mut hess_rows = Vec::with_capacity(n * (n + 1) / 2);
        let mut hess_cols = Vec::with_capacity(n * (n + 1) / 2);
        for i in 0..n {
            for j in 0..=i {
                hess_rows.push(i);
                hess_cols.push(j);
            }
        }

        // Build mapping from inner hessian entries to our dense lower triangle.
        // Our layout: for (i,j) with i >= j, index = i*(i+1)/2 + j
        let (inner_hess_rows, inner_hess_cols) = inner.hessian_structure();
        let mut inner_hess_map = Vec::with_capacity(inner_hess_rows.len());
        for k in 0..inner_hess_rows.len() {
            let (r, c) = (inner_hess_rows[k], inner_hess_cols[k]);
            // Ensure lower triangle (r >= c)
            let (i, j) = if r >= c { (r, c) } else { (c, r) };
            let idx = i * (i + 1) / 2 + j;
            inner_hess_map.push(idx);
        }

        LeastSquaresProblem {
            inner,
            targets,
            jac_rows,
            jac_cols,
            hess_rows,
            hess_cols,
            inner_hess_map,
        }
    }
}

impl<P: NlpProblem> NlpProblem for LeastSquaresProblem<'_, P> {
    fn num_variables(&self) -> usize {
        self.inner.num_variables()
    }
    fn num_constraints(&self) -> usize {
        0 // unconstrained LS
    }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        self.inner.bounds(x_l, x_u);
    }
    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {
        // no constraints
    }
    fn initial_point(&self, x0: &mut [f64]) {
        self.inner.initial_point(x0);
    }
    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        let m = self.targets.len();
        let mut g = vec![0.0; m];
        if !self.inner.constraints(x, _new_x, &mut g) { return false; }
        let mut sum = 0.0;
        for i in 0..m {
            let r = g[i] - self.targets[i];
            sum += r * r;
        }
        *obj = 0.5 * sum;
        true
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        let n = self.inner.num_variables();
        let m = self.targets.len();
        let mut g = vec![0.0; m];
        if !self.inner.constraints(x, _new_x, &mut g) { return false; }
        let mut r = vec![0.0; m];
        for i in 0..m {
            r[i] = g[i] - self.targets[i];
        }
        let jac_nnz = self.jac_rows.len();
        let mut jac_vals = vec![0.0; jac_nnz];
        if !self.inner.jacobian_values(x, _new_x, &mut jac_vals) { return false; }
        for i in 0..n {
            grad[i] = 0.0;
        }
        for (idx, (&row, &col)) in self.jac_rows.iter().zip(self.jac_cols.iter()).enumerate() {
            grad[col] += jac_vals[idx] * r[row];
        }
        true
    }
    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool { true }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) -> bool { true }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.hess_rows.clone(), self.hess_cols.clone())
    }
    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        let n = self.inner.num_variables();
        let m = self.targets.len();

        let mut g = vec![0.0; m];
        if !self.inner.constraints(x, _new_x, &mut g) { return false; }
        let mut r = vec![0.0; m];
        for i in 0..m {
            r[i] = g[i] - self.targets[i];
        }

        let jac_nnz = self.jac_rows.len();
        let mut jac_vals = vec![0.0; jac_nnz];
        if !self.inner.jacobian_values(x, _new_x, &mut jac_vals) { return false; }

        let mut j_dense = vec![0.0; m * n];
        for (idx, (&row, &col)) in self.jac_rows.iter().zip(self.jac_cols.iter()).enumerate() {
            j_dense[row * n + col] += jac_vals[idx];
        }

        let mut idx = 0;
        for i in 0..n {
            for j in 0..=i {
                let mut dot = 0.0;
                for k in 0..m {
                    dot += j_dense[k * n + i] * j_dense[k * n + j];
                }
                vals[idx] = obj_factor * dot;
                idx += 1;
            }
        }

        let inner_hess_nnz = self.inner_hess_map.len();
        if inner_hess_nnz > 0 {
            let mut inner_hess_vals = vec![0.0; inner_hess_nnz];
            if !self.inner.hessian_values(x, _new_x, 0.0, &r, &mut inner_hess_vals) { return false; }
            for (k, &v) in inner_hess_vals.iter().enumerate() {
                vals[self.inner_hess_map[k]] += obj_factor * v;
            }
        }
        true
    }
}

/// Detect if a problem is a nonlinear equation system (square or overdetermined).
///
/// Returns true if ALL of:
/// - f(x0) ≈ 0 (zero objective)
/// - ∇f(x0) ≈ 0 (zero gradient)
/// - All constraints are equalities (g_l[i] == g_u[i])
/// - m >= n (square or more constraints than variables)
/// Dense Cholesky solve: solve A*x = b where A is n×n symmetric positive definite.
/// Returns None if A is not positive definite (factorization fails).
fn dense_cholesky_solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    // Cholesky: A = L * L^T
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j * n + k] * l[j * n + k];
        }
        let diag = a[j * n + j] - sum;
        if diag <= 0.0 {
            return None;
        }
        l[j * n + j] = diag.sqrt();

        for i in (j + 1)..n {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
        }
    }

    // Forward solve: L * y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..i {
            sum += l[i * n + k] * y[k];
        }
        y[i] = (b[i] - sum) / l[i * n + i];
    }

    // Back solve: L^T * x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for k in (i + 1)..n {
            sum += l[k * n + i] * x[k];
        }
        x[i] = (y[i] - sum) / l[i * n + i];
    }

    Some(x)
}

fn detect_ne_problem<P: NlpProblem>(problem: &P) -> bool {
    let n = problem.num_variables();
    let m = problem.num_constraints();

    if m < n || m == 0 || n == 0 {
        return false;
    }

    // Square systems (m == n) are better solved by direct constrained IPM
    // (Newton on g(x)=0), which typically converges in a few iterations.
    // LS reformulation min 0.5*||g||^2 has a harder landscape for square systems.
    if m == n {
        return false;
    }

    // Check objective and gradient at initial point
    let mut x0 = vec![0.0; n];
    problem.initial_point(&mut x0);

    let mut f0 = 0.0;
    if !problem.objective(&x0, true, &mut f0) {
        return false;
    }
    if f0.abs() > 1e-10 {
        return false;
    }

    let mut grad = vec![0.0; n];
    if !problem.gradient(&x0, false, &mut grad) {
        return false;
    }
    let grad_max = grad.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if grad_max > 1e-10 {
        return false;
    }

    // Check all constraints are equalities
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    problem.constraint_bounds(&mut g_l, &mut g_u);
    for i in 0..m {
        if !convergence::is_equality_constraint(g_l[i], g_u[i]) {
            return false;
        }
    }

    // If constraints are already satisfied at x0, no need to reformulate —
    // the standard IPM can handle it (e.g., FBRAIN3 starts feasible).
    let mut g0 = vec![0.0; m];
    if !problem.constraints(&x0, false, &mut g0) {
        return false;
    }
    let theta0 = convergence::primal_infeasibility(&g0, &g_l, &g_u);
    if theta0 < 1e-8 {
        return false;
    }

    true
}

/// Diagnosis of why the IPM failed, used to select targeted recovery strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureDiagnosis {
    /// Constraint violation is large and not decreasing.
    StallAtInfeasibility,
    /// All metrics are small but strict tolerances not met.
    StallNearOptimal,
    /// Factorization failed, NaN/Inf, or other numerical breakdown.
    NumericalBreakdown,
    /// Making progress but hit max iterations.
    SlowConvergence,
    /// Dual variables growing unboundedly (ill-conditioned Hessian).
    DualDivergence,
}

/// Classify the failure mode from diagnostics to select the best recovery strategy.
fn diagnose_failure(result: &SolveResult) -> FailureDiagnosis {
    let d = &result.diagnostics;
    match result.status {
        SolveStatus::NumericalError => {
            if d.final_dual_inf > 1e4 {
                FailureDiagnosis::DualDivergence
            } else {
                FailureDiagnosis::NumericalBreakdown
            }
        }
        SolveStatus::RestorationFailed => FailureDiagnosis::StallAtInfeasibility,
        SolveStatus::MaxIterations => {
            // Check dual divergence first — large dual infeasibility indicates
            // the Hessian or problem scaling is the root issue, not convergence speed.
            if d.final_dual_inf > 1e4 {
                FailureDiagnosis::DualDivergence
            } else if d.final_primal_inf > 1e-2 {
                FailureDiagnosis::StallAtInfeasibility
            } else if d.final_primal_inf < 1e-6 && d.final_compl < 1e-4 {
                FailureDiagnosis::StallNearOptimal
            } else {
                FailureDiagnosis::SlowConvergence
            }
        }
        _ => FailureDiagnosis::SlowConvergence,
    }
}

/// Check if `candidate` is strictly better than `current`.
fn is_strictly_better(current: &SolveResult, candidate: &SolveResult) -> bool {
    let candidate_solved = matches!(candidate.status, SolveStatus::Optimal);
    let current_solved = matches!(current.status, SolveStatus::Optimal);
    candidate_solved
        && (!current_solved
            || candidate.objective < current.objective)
}

/// Prepare options for a fallback solve: cap iterations and set remaining time budget.
/// Returns `None` if there is no time budget remaining.
fn prepare_fallback_opts(options: &SolverOptions, solve_start: &Instant) -> Option<SolverOptions> {
    let mut opts = options.clone();
    opts.max_iter = options.max_iter.min(500).max(options.max_iter / 3);
    if options.max_wall_time > 0.0 {
        let remaining = options.max_wall_time - solve_start.elapsed().as_secs_f64();
        if remaining <= 0.1 {
            return None;
        }
        opts.max_wall_time = remaining;
    }
    Some(opts)
}

/// Solve the NLP using the interior point method.
pub fn solve<P: NlpProblem>(problem: &P, options: &SolverOptions) -> SolveResult {
    let solve_start = Instant::now();

    // Capture initial objective and feasibility for slow-optimal detection.
    // NOTE: disabled -- extra problem evaluations here change CUTEst FP state and cause regressions.
    let (initial_obj, initial_feasible) = (f64::INFINITY, false);

    // --- Preprocessing: eliminate fixed variables and redundant constraints ---
    if options.enable_preprocessing {
        let prep = crate::preprocessing::PreprocessedProblem::new(problem as &dyn NlpProblem, options.bound_push);
        if prep.did_reduce() {
            if options.print_level >= 5 {
                rip_log!(
                    "ripopt: Preprocessing reduced problem: {} fixed vars, {} redundant constraints ({}x{} -> {}x{})",
                    prep.num_fixed(), prep.num_redundant(),
                    problem.num_variables(), problem.num_constraints(),
                    prep.num_variables(), prep.num_constraints(),
                );
            }
            let mut prep_opts = options.clone();
            prep_opts.enable_preprocessing = false; // prevent re-preprocessing
            // Limit wall time for the preprocessed solve to half the remaining budget.
            // Without this, the preprocessed fallback chain can consume the full budget,
            // leaving no time for the unpreprocessed retry (which often succeeds when
            // preprocessing changes the problem structure, e.g., ganges.gms 273x273->356x273).
            if options.max_wall_time > 0.0 {
                let elapsed = solve_start.elapsed().as_secs_f64();
                let remaining = (options.max_wall_time - elapsed).max(1.0);
                prep_opts.max_wall_time = remaining * 0.5;
            }
            let reduced_result = solve(&prep, &prep_opts);
            let result = prep.unmap_solution(&reduced_result);
            // If preprocessing made things worse, fall back to solving without it
            if matches!(result.status, SolveStatus::Optimal) {
                return result;
            }
            if options.print_level >= 5 {
                rip_log!(
                    "ripopt: Preprocessed solve failed ({:?}), retrying without preprocessing",
                    result.status
                );
            }
            // Fall through to solve without preprocessing
        }
    }

    // --- NE-to-LS Detection and Reformulation ---
    // Detect overdetermined nonlinear equation problems (m > n, f≡0, all equalities)
    // and reformulate as least-squares: min 0.5*||g(x)-target||^2.
    if detect_ne_problem(problem) {
        let n = problem.num_variables();
        let m = problem.num_constraints();
        if options.print_level >= 5 {
            rip_log!(
                "ripopt: Detected overdetermined NE problem (n={}, m={}), reformulating as least-squares",
                n, m
            );
        }
        let ls_problem = LeastSquaresProblem::new(problem);
        // For square systems (m == n), cap LS iterations — they often fail the LS
        // approach because there are no "extra" equations to drive residuals down.
        let mut ls_opts = options.clone();
        if m == n {
            ls_opts.max_iter = (options.max_iter / 10).min(100);
        }
        let ls_result = solve_ipm(&ls_problem, &ls_opts);

        // Evaluate original constraint violation at the LS solution
        let mut g_final = vec![0.0; m];
        if !problem.constraints(&ls_result.x, true, &mut g_final) {
            // Evaluation failed; skip polishing, return LS result as-is
            let mut diag = ls_result.diagnostics.clone();
            diag.fallback_used = Some("ne-to-ls".into());
            return SolveResult {
                x: ls_result.x,
                objective: ls_result.objective,
                constraint_multipliers: vec![0.0; m],
                bound_multipliers_lower: ls_result.bound_multipliers_lower,
                bound_multipliers_upper: ls_result.bound_multipliers_upper,
                constraint_values: g_final,
                status: SolveStatus::EvaluationError,
                iterations: ls_result.iterations,
                diagnostics: diag,
            };
        }
        let mut g_l = vec![0.0; m];
        let mut g_u = vec![0.0; m];
        problem.constraint_bounds(&mut g_l, &mut g_u);
        let mut theta = convergence::primal_infeasibility(&g_final, &g_l, &g_u);

        // Newton polish: if theta is close but not quite at tol, try a few
        // Gauss-Newton steps on the original system g(x) = target to drive
        // constraint violation below tol.
        let mut polished_x = ls_result.x.clone();
        if theta > options.tol && theta < 1e-2 {
            let mut x_l_var = vec![0.0; n];
            let mut x_u_var = vec![0.0; n];
            problem.bounds(&mut x_l_var, &mut x_u_var);
            let (jac_rows, jac_cols) = problem.jacobian_structure();
            let nnz = jac_rows.len();

            // Target values for each constraint (midpoint of [g_l, g_u] for equalities)
            let target: Vec<f64> = (0..m).map(|i| {
                if (g_u[i] - g_l[i]).abs() < 1e-15 { g_l[i] } else { 0.5 * (g_l[i] + g_u[i]) }
            }).collect();

            let max_newton_iters = 20;
            for newton_iter in 0..max_newton_iters {
                // Residual: r = g(x) - target
                let r: Vec<f64> = (0..m).map(|i| g_final[i] - target[i]).collect();

                // Get Jacobian at current point
                let mut jac_vals = vec![0.0; nnz];
                if !problem.jacobian_values(&polished_x, true, &mut jac_vals) {
                    break; // Eval failed, stop polishing
                }

                // Build dense J (m x n)
                let mut j_dense = vec![0.0; m * n];
                for k in 0..nnz {
                    j_dense[jac_rows[k] * n + jac_cols[k]] += jac_vals[k];
                }

                // Solve for dx using normal equations: (J^T J) dx = -J^T r
                // Form J^T J (n x n) and J^T r (n)
                let mut jtj = vec![0.0; n * n];
                let mut jtr = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        let mut s = 0.0;
                        for k in 0..m {
                            s += j_dense[k * n + i] * j_dense[k * n + j];
                        }
                        jtj[i * n + j] = s;
                    }
                    let mut s = 0.0;
                    for k in 0..m {
                        s += j_dense[k * n + i] * r[k];
                    }
                    jtr[i] = s;
                }

                // Add small regularization for numerical stability
                for i in 0..n {
                    jtj[i * n + i] += 1e-14;
                }

                // Solve with dense Cholesky (J^T J is SPD when J has full column rank)
                let dx = match dense_cholesky_solve(&jtj, &jtr, n) {
                    Some(dx) => dx,
                    None => break, // J^T J singular, stop polishing
                };

                // Line search with fraction-to-boundary for variable bounds
                let mut alpha = 1.0;
                let tau = 0.995;
                for i in 0..n {
                    if dx[i] < 0.0 && x_l_var[i].is_finite() {
                        let max_step = -tau * (polished_x[i] - x_l_var[i]) / dx[i];
                        if max_step < alpha { alpha = max_step; }
                    }
                    if dx[i] > 0.0 && x_u_var[i].is_finite() {
                        let max_step = tau * (x_u_var[i] - polished_x[i]) / dx[i];
                        if max_step < alpha { alpha = max_step; }
                    }
                }
                alpha = alpha.max(0.0).min(1.0);

                // Backtracking: ensure theta actually decreases
                let mut trial_x = vec![0.0; n];
                let mut trial_g = vec![0.0; m];
                let mut best_alpha = alpha;
                let mut best_theta = theta;
                for _ in 0..10 {
                    for i in 0..n {
                        trial_x[i] = polished_x[i] - best_alpha * dx[i];
                    }
                    if !problem.constraints(&trial_x, true, &mut trial_g) {
                        best_alpha *= 0.5;
                        continue; // Eval failed, try shorter step
                    }
                    let trial_theta = convergence::primal_infeasibility(&trial_g, &g_l, &g_u);
                    if trial_theta < theta {
                        best_theta = trial_theta;
                        break;
                    }
                    best_alpha *= 0.5;
                }

                if best_theta >= theta * 0.999 {
                    break; // No progress
                }

                // Accept step
                for i in 0..n {
                    polished_x[i] -= best_alpha * dx[i];
                }
                if !problem.constraints(&polished_x, true, &mut g_final) {
                    break; // Eval failed, stop polishing
                }
                theta = best_theta;

                if options.print_level >= 5 {
                    rip_log!(
                        "ripopt: Newton polish iter {}: theta={:.2e}, alpha={:.4}",
                        newton_iter + 1, theta, best_alpha,
                    );
                }

                if theta < options.tol {
                    break; // Converged!
                }
            }
        }

        // g_final is from the original (unscaled) problem, no unscaling needed
        let g_out = g_final;

        let status = if theta < options.tol {
            SolveStatus::Optimal
        } else {
            SolveStatus::LocalInfeasibility
        };

        if options.print_level >= 5 {
            rip_log!(
                "ripopt: NE-to-LS result: obj_LS={:.4e}, constraint_violation={:.4e}, status={:?}",
                ls_result.objective, theta, status
            );
        }

        // Fall back to constrained IPM when LS reports infeasibility:
        // - For square systems (m==n): LS may have found a non-root critical point
        //   of ||g||^2; constrained IPM can find the actual root.
        // - For non-square systems: only fall back if LS didn't converge
        //   (MaxIterations/RestorationFailed); if LS converged with high theta,
        //   the system is genuinely inconsistent.
        let ls_converged = matches!(ls_result.status, SolveStatus::Optimal);
        if status == SolveStatus::LocalInfeasibility && (m == n || !ls_converged) {
            if options.print_level >= 5 {
                rip_log!(
                    "ripopt: LS reformulation reports infeasibility (theta={:.4e}, ls_status={:?}), falling back to constrained IPM",
                    theta, ls_result.status
                );
            }
            let mut fallback_opts = options.clone();
            if options.max_wall_time > 0.0 {
                let remaining = options.max_wall_time - solve_start.elapsed().as_secs_f64();
                if remaining <= 0.1 {
                    return SolveResult {
                        x: ls_result.x,
                        objective: 0.0,
                        constraint_multipliers: vec![0.0; m],
                        bound_multipliers_lower: ls_result.bound_multipliers_lower,
                        bound_multipliers_upper: ls_result.bound_multipliers_upper,
                        constraint_values: g_out,
                        status: SolveStatus::MaxIterations,
                        iterations: ls_result.iterations,
                        diagnostics: SolverDiagnostics::default(),
                    };
                }
                fallback_opts.max_wall_time = remaining;
            }
            let ipm_result = solve_ipm(problem, &fallback_opts);
            if matches!(ipm_result.status, SolveStatus::Optimal) {
                return ipm_result;
            }
            // IPM fallback failed too — try AL fallback for square NE systems
            // (e.g. HEART6 where both LS and constrained IPM fail)
            if options.enable_al_fallback {
                if options.print_level >= 5 {
                    rip_log!(
                        "ripopt: NE constrained IPM fallback failed ({:?}), trying AL",
                        ipm_result.status
                    );
                }
                let mut al_opts = options.clone();
                al_opts.max_iter = options.max_iter.min(500).max(options.max_iter / 3);
                if options.max_wall_time > 0.0 {
                    let remaining = options.max_wall_time - solve_start.elapsed().as_secs_f64();
                    if remaining <= 0.1 {
                        return ipm_result;
                    }
                    al_opts.max_wall_time = remaining;
                }
                let al_result = crate::augmented_lagrangian::solve(problem, &al_opts);
                if matches!(al_result.status, SolveStatus::Optimal) {
                    if options.print_level >= 5 {
                        rip_log!(
                            "ripopt: NE AL fallback succeeded ({:?}, obj={:.6e})",
                            al_result.status, al_result.objective
                        );
                    }
                    return al_result;
                }
            }
            return ipm_result;
        }

        // L-BFGS retry on LS problem when IPM found a local min with nonzero residual
        let (final_x, final_status, final_g, final_iters, final_zl, final_zu) =
            if status == SolveStatus::LocalInfeasibility && options.enable_lbfgs_fallback {
                if options.print_level >= 5 {
                    rip_log!(
                        "ripopt: NE-to-LS LocalInfeasibility (theta={:.4e}), trying L-BFGS on LS",
                        theta
                    );
                }
                let lbfgs_ls = crate::lbfgs::solve(&ls_problem, options);
                let mut g_lb = vec![0.0; m];
                let theta_lb = if problem.constraints(&lbfgs_ls.x, true, &mut g_lb) {
                    convergence::primal_infeasibility(&g_lb, &g_l, &g_u)
                } else {
                    f64::INFINITY // Eval failed, skip this fallback
                };

                if theta_lb < theta {
                    let new_status = if theta_lb < options.tol {
                        SolveStatus::Optimal
                    } else {
                        SolveStatus::LocalInfeasibility
                    };
                    if options.print_level >= 5 {
                        rip_log!(
                            "ripopt: L-BFGS improved NE-to-LS (theta: {:.4e} -> {:.4e}, status={:?})",
                            theta, theta_lb, new_status
                        );
                    }
                    (lbfgs_ls.x, new_status, g_lb, lbfgs_ls.iterations,
                     lbfgs_ls.bound_multipliers_lower, lbfgs_ls.bound_multipliers_upper)
                } else {
                    if options.print_level >= 5 {
                        rip_log!(
                            "ripopt: L-BFGS did not improve NE-to-LS (theta_lb={:.4e} >= theta={:.4e})",
                            theta_lb, theta
                        );
                    }
                    (polished_x, status, g_out, ls_result.iterations,
                     ls_result.bound_multipliers_lower, ls_result.bound_multipliers_upper)
                }
            } else {
                (polished_x, status, g_out, ls_result.iterations,
                 ls_result.bound_multipliers_lower, ls_result.bound_multipliers_upper)
            };

        return SolveResult {
            x: final_x,
            objective: 0.0, // Original objective is f≡0
            constraint_multipliers: vec![0.0; m],
            bound_multipliers_lower: final_zl,
            bound_multipliers_upper: final_zu,
            constraint_values: final_g,
            status: final_status,
            iterations: final_iters,
            diagnostics: SolverDiagnostics::default(),
        };
    }

    // For unconstrained problems, try L-BFGS first (O(n·m) vs O(n³) per iteration)
    // then fall back to IPM if needed.
    let mut result = if options.enable_lbfgs_fallback && problem.num_constraints() == 0 {
        let lbfgs_result = crate::lbfgs::solve(problem, options);
        if matches!(lbfgs_result.status, SolveStatus::Optimal) {
            if options.print_level >= 5 {
                rip_log!(
                    "ripopt: L-BFGS solved unconstrained problem ({:?}, obj={:.6e})",
                    lbfgs_result.status, lbfgs_result.objective
                );
            }
            lbfgs_result
        } else {
            if options.print_level >= 5 {
                rip_log!(
                    "ripopt: L-BFGS failed ({:?}, obj={:.6e}), trying IPM",
                    lbfgs_result.status, lbfgs_result.objective
                );
            }
            let ipm_result = solve_ipm(problem, options);
            if matches!(ipm_result.status, SolveStatus::Optimal) {
                ipm_result
            } else if lbfgs_result.objective < ipm_result.objective {
                lbfgs_result
            } else {
                ipm_result
            }
        }
    } else {
        // For constrained problems with wall time limits, reserve budget for fallbacks.
        // Without this, the first solve_ipm consumes the full max_wall_time, leaving
        // nothing for SQP/slack/AL fallbacks that might succeed.
        if options.max_wall_time > 0.0 && problem.num_constraints() > 0 {
            let mut main_opts = options.clone();
            main_opts.max_wall_time = options.max_wall_time * 0.5;
            solve_ipm(problem, &main_opts)
        } else {
            solve_ipm(problem, options)
        }
    };

    // --- Diagnostic-driven recovery ---
    // Instead of trying every fallback in a fixed order, diagnose the failure
    // and select targeted recovery strategies.
    let diagnosis = diagnose_failure(&result);
    let has_constraints = problem.num_constraints() > 0;
    let has_inequalities = has_inequality_constraints(problem);

    if options.print_level >= 5 && !matches!(result.status, SolveStatus::Optimal) {
        rip_log!("ripopt: Failure diagnosis: {:?}", diagnosis);
    }

    // Helper closure: try L-BFGS Hessian fallback
    let try_lbfgs_hessian = |result: &mut SolveResult| {
        if !options.enable_lbfgs_hessian_fallback || options.hessian_approximation_lbfgs {
            return;
        }
        if let Some(mut opts) = prepare_fallback_opts(options, &solve_start) {
            opts.hessian_approximation_lbfgs = true;
            opts.enable_lbfgs_hessian_fallback = false;
            opts.stall_iter_limit = 0;
            if options.print_level >= 5 {
                rip_log!("ripopt: Trying L-BFGS Hessian fallback ({:?})", diagnosis);
            }
            let candidate = solve_ipm(problem, &opts);
            if is_strictly_better(result, &candidate) {
                if options.print_level >= 5 {
                    rip_log!(
                        "ripopt: L-BFGS Hessian fallback succeeded ({:?}, obj={:.6e})",
                        candidate.status, candidate.objective
                    );
                }
                *result = candidate;
                result.diagnostics.fallback_used = Some("lbfgs_hessian".into());
            } else if options.print_level >= 5 {
                rip_log!("ripopt: L-BFGS Hessian fallback did not improve ({:?})", candidate.status);
            }
        }
    };

    // Helper closure: try AL fallback
    let try_al = |result: &mut SolveResult| {
        if !options.enable_al_fallback || !has_constraints {
            return;
        }
        if let Some(opts) = prepare_fallback_opts(options, &solve_start) {
            if options.print_level >= 5 {
                rip_log!("ripopt: Trying AL fallback ({:?})", diagnosis);
            }
            let candidate = crate::augmented_lagrangian::solve(problem, &opts);
            if is_strictly_better(result, &candidate) {
                if options.print_level >= 5 {
                    rip_log!(
                        "ripopt: AL fallback succeeded ({:?}, obj={:.6e})",
                        candidate.status, candidate.objective
                    );
                }
                *result = candidate;
                result.diagnostics.fallback_used = Some("augmented_lagrangian".into());
            } else if options.print_level >= 5 {
                rip_log!("ripopt: AL fallback did not improve ({:?})", candidate.status);
            }
        }
    };

    // Helper closure: try SQP fallback
    let try_sqp = |result: &mut SolveResult| {
        if !options.enable_sqp_fallback || !has_constraints {
            return;
        }
        if let Some(opts) = prepare_fallback_opts(options, &solve_start) {
            if options.print_level >= 5 {
                rip_log!("ripopt: Trying SQP fallback ({:?})", diagnosis);
            }
            let candidate = crate::sqp::solve(problem, &opts);
            if is_strictly_better(result, &candidate) {
                if options.print_level >= 5 {
                    rip_log!(
                        "ripopt: SQP fallback succeeded ({:?}, obj={:.6e})",
                        candidate.status, candidate.objective
                    );
                }
                *result = candidate;
                result.diagnostics.fallback_used = Some("sqp".into());
            } else if options.print_level >= 5 {
                rip_log!("ripopt: SQP fallback did not improve ({:?})", candidate.status);
            }
        }
    };

    // Helper closure: try slack reformulation fallback
    let try_slack = |result: &mut SolveResult| -> Option<SolveResult> {
        if !options.enable_slack_fallback || !has_inequalities {
            return None;
        }
        if let Some(mut opts) = prepare_fallback_opts(options, &solve_start) {
            opts.enable_slack_fallback = false; // prevent recursion
            if options.print_level >= 5 {
                rip_log!("ripopt: Trying slack fallback ({:?})", diagnosis);
            }
            let slack_prob = SlackFormulation::new(problem, &result.x);
            let candidate = solve_ipm(&slack_prob, &opts);
            if is_strictly_better(result, &candidate) {
                let n = problem.num_variables();
                let m = problem.num_constraints();
                if options.print_level >= 5 {
                    rip_log!(
                        "ripopt: Slack fallback succeeded ({:?}, obj={:.6e})",
                        candidate.status, candidate.objective
                    );
                }
                let x_out = candidate.x[..n].to_vec();
                let mut g_out = vec![0.0; m];
                let _ = problem.constraints(&x_out, true, &mut g_out); // best-effort
                let mut diag = candidate.diagnostics;
                diag.fallback_used = Some("slack".into());
                diag.wall_time_secs = solve_start.elapsed().as_secs_f64();
                return Some(SolveResult {
                    x: x_out,
                    objective: candidate.objective,
                    constraint_multipliers: candidate.constraint_multipliers,
                    bound_multipliers_lower: candidate.bound_multipliers_lower[..n].to_vec(),
                    bound_multipliers_upper: candidate.bound_multipliers_upper[..n].to_vec(),
                    constraint_values: g_out,
                    status: candidate.status,
                    iterations: result.iterations + candidate.iterations,
                    diagnostics: diag,
                });
            } else if options.print_level >= 5 {
                rip_log!("ripopt: Slack fallback did not improve ({:?})", candidate.status);
            }
        }
        None
    };

    // Conservative IPM retry: revert v0.4.0 algorithmic changes (Gondzio MCC,
    // Mehrotra PC, stall detection) to recover the pre-regression trajectory.
    // This is the most reliable recovery for problems sensitive to Newton direction
    // changes (TRO3X3, STRATEC, MGH10LS, ACOPR30).
    let n_problem = problem.num_variables();
    if n_problem <= 200
        && !matches!(result.status, SolveStatus::Optimal)
    {
        if let Some(mut opts) = prepare_fallback_opts(options, &solve_start) {
            opts.gondzio_mcc_max = 0;
            opts.mehrotra_pc = false;
            opts.stall_iter_limit = 0;
            opts.proactive_infeasibility_detection = true;
            // Full iteration budget for small problems (ACOPR30 needed 1047, MGH10LS ~1800)
            opts.max_iter = options.max_iter;
            // Give most of remaining time — this is the best recovery strategy
            if options.max_wall_time > 0.0 {
                let remaining = options.max_wall_time - solve_start.elapsed().as_secs_f64();
                opts.max_wall_time = remaining * 0.7;
            }
            if options.print_level >= 5 {
                rip_log!("ripopt: Trying conservative IPM retry (no Gondzio/Mehrotra, no stall detection)");
            }
            let candidate = solve_ipm(problem, &opts);
            if is_strictly_better(&result, &candidate) {
                if options.print_level >= 5 {
                    rip_log!(
                        "ripopt: Conservative retry succeeded ({:?}, obj={:.6e})",
                        candidate.status, candidate.objective
                    );
                }
                result = candidate;
                result.diagnostics.fallback_used = Some("conservative_ipm".into());
            } else if options.print_level >= 5 {
                rip_log!("ripopt: Conservative retry did not improve ({:?})", candidate.status);
            }
        }
    }

    // Dispatch based on diagnosis — try targeted strategies rather than
    // a fixed sequence of every possible fallback.
    if !matches!(result.status, SolveStatus::Optimal) {
        match diagnosis {
            FailureDiagnosis::StallAtInfeasibility => {
                // Constraint violation stuck high — slack reformulation changes the
                // problem structure, SQP handles infeasible starts better than IPM
                if let Some(slack_result) = try_slack(&mut result) {
                    return slack_result;
                }
                try_sqp(&mut result);
            }
            FailureDiagnosis::NumericalBreakdown => {
                // Factorization issues — L-BFGS avoids the user-provided Hessian
                // which is often the root cause
                try_lbfgs_hessian(&mut result);
            }
            FailureDiagnosis::DualDivergence => {
                // Large dual infeasibility often means advanced Newton corrections
                // (Gondzio MCC, Mehrotra PC) steered the solver into a bad basin.
                // Retry with plain IPM (no corrections) as a targeted fix.
                if let Some(mut opts) = prepare_fallback_opts(options, &solve_start) {
                    opts.gondzio_mcc_max = 0;
                    opts.mehrotra_pc = false;
                    opts.stall_iter_limit = 0;
                    if options.print_level >= 5 {
                        rip_log!("ripopt: Trying plain IPM retry (no corrections) for DualDivergence");
                    }
                    let candidate = solve_ipm(problem, &opts);
                    if is_strictly_better(&result, &candidate) {
                        if options.print_level >= 5 {
                            rip_log!(
                                "ripopt: Plain IPM retry succeeded ({:?}, obj={:.6e})",
                                candidate.status, candidate.objective
                            );
                        }
                        result = candidate;
                        result.diagnostics.fallback_used = Some("plain_ipm".into());
                    } else if options.print_level >= 5 {
                        rip_log!("ripopt: Plain IPM retry did not improve ({:?})", candidate.status);
                    }
                }
                // If plain IPM didn't help, try L-BFGS Hessian
                if !matches!(result.status, SolveStatus::Optimal) {
                    try_lbfgs_hessian(&mut result);
                }
            }
            FailureDiagnosis::SlowConvergence => {
                // IPM is making progress but too slowly — try a completely
                // different algorithm (AL or SQP)
                try_al(&mut result);
                if !matches!(result.status, SolveStatus::Optimal) {
                    try_sqp(&mut result);
                }
            }
            FailureDiagnosis::StallNearOptimal => {
                // Close to optimal but can't meet strict tolerances —
                // SQP may refine the solution
                try_sqp(&mut result);
            }
        }
    }

    // Slow-optimal slack fallback: if the initial IPM was Optimal but started from
    // a feasible point and the objective worsened (or didn't improve) while consuming
    // >5% of the wall-time budget, it likely converged to a bad local minimum.
    // The threshold is 5% so zigzag (237s / 3600s = 6.6% of budget) triggers it.
    if matches!(result.status, SolveStatus::Optimal)
        && has_inequalities
        && options.enable_slack_fallback
        && options.max_wall_time > 0.0
    {
        let time_used = solve_start.elapsed().as_secs_f64();
        // "Worsened from feasible start": started feasible AND final obj is not
        // better than initial obj (allowing 0.1% tolerance for numerical noise).
        let worsened_from_feasible = initial_feasible
            && initial_obj.is_finite()
            && result.objective > initial_obj - 1e-3 * initial_obj.abs().max(1.0);
        if time_used > 0.05 * options.max_wall_time && worsened_from_feasible {
            if options.print_level >= 5 {
                rip_log!(
                    "ripopt: Slow-optimal detected (obj={:.4e}, init_obj={:.4e}, time={:.1}s/{:.1}s), trying slack fallback",
                    result.objective, initial_obj, time_used, options.max_wall_time
                );
            }
            if let Some(slack_result) = try_slack(&mut result) {
                return slack_result;
            }
        }
    }

    // --- Late early-out: NumericalError but state meets acceptable convergence ---
    // Applied AFTER all fallbacks so the conservative retry has a chance to fix
    // wrong local minima before we promote them.  This handles GAMS-style problems
    // where the IPM converges near-optimal and then fails (e.g., factorization
    // breakdown at near-zero barrier parameter), and all fallbacks time out quickly.
    // Use z_opt-based dual infeasibility (final_dual_inf_scaled) rather than iterative z.
    if matches!(result.status, SolveStatus::NumericalError) {
        let d = &result.diagnostics;
        let du_tol = options.tol * 1000.0; // e.g. 1e-5 with default tol=1e-8
        let pr_ok = d.final_primal_inf <= options.constr_viol_tol;
        let du_ok = d.final_dual_inf_scaled <= du_tol;
        let co_ok = d.final_compl <= options.compl_inf_tol;
        if pr_ok && du_ok && co_ok {
            if options.print_level >= 5 {
                rip_log!(
                    "ripopt: NumericalError but state meets acceptable convergence \
                     (pr={:.2e}, du_opt={:.2e}, co={:.2e}), returning Optimal",
                    d.final_primal_inf, d.final_dual_inf_scaled, d.final_compl
                );
            }
            result.status = SolveStatus::Optimal;
        }
    }

    result.diagnostics.wall_time_secs = solve_start.elapsed().as_secs_f64();
    result
}

/// Check if a problem has any inequality constraints (g_l[i] != g_u[i]).
fn has_inequality_constraints<P: NlpProblem>(problem: &P) -> bool {
    let m = problem.num_constraints();
    if m == 0 {
        return false;
    }
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    problem.constraint_bounds(&mut g_l, &mut g_u);
    (0..m).any(|i| (g_l[i] - g_u[i]).abs() > 0.0)
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

/// Dump a KKT system to disk for external solver benchmarking.
///
/// Writes two files to `dir`:
/// - `<name>_<iter:04>.mtx` — Matrix Market symmetric format, lower triangle, 1-indexed
/// - `<name>_<iter:04>.json` — Metadata: problem_name, iteration, n, m, rhs, inertia, status
///
/// All IO errors are logged as warnings and never propagate to the caller.
fn dump_kkt_matrix(
    dir: &std::path::Path,
    name: &str,
    iteration: usize,
    kkt: &kkt::KktSystem,
    inertia: Option<(usize, usize, usize)>,
    delta_w: f64,
    delta_c: f64,
) {
    use std::io::Write;

    if let Err(e) = std::fs::create_dir_all(dir) {
        log::warn!("kkt_dump: cannot create directory {}: {}", dir.display(), e);
        return;
    }

    let stem = format!("{}_{:04}", name, iteration);
    let dim = kkt.dim;

    // --- Matrix Market (.mtx) ---
    let mtx_path = dir.join(format!("{}.mtx", stem));
    let write_mtx = || -> std::io::Result<()> {
        // Collect lower-triangle entries (1-indexed for Matrix Market).
        let entries: Vec<(usize, usize, f64)> = match &kkt.matrix {
            KktMatrix::Dense(d) => {
                let mut v = Vec::with_capacity(dim * (dim + 1) / 2);
                for j in 0..dim {
                    for i in j..dim {
                        let val = d.get(i, j);
                        if val != 0.0 {
                            v.push((i + 1, j + 1, val));
                        }
                    }
                }
                v
            }
            KktMatrix::Sparse(s) => {
                // Triplets are upper triangle (row <= col). Flip each entry to lower
                // triangle by swapping indices, then aggregate duplicates.
                let mut map: std::collections::HashMap<(usize, usize), f64> =
                    std::collections::HashMap::with_capacity(s.triplet_rows.len());
                for k in 0..s.triplet_rows.len() {
                    let r = s.triplet_rows[k]; // r <= c (upper tri)
                    let c = s.triplet_cols[k];
                    // Lower-triangle key: larger index first → (c, r) with c >= r.
                    *map.entry((c, r)).or_insert(0.0) += s.triplet_vals[k];
                }
                let mut v: Vec<(usize, usize, f64)> = map
                    .into_iter()
                    .filter(|(_, val)| *val != 0.0)
                    .map(|((i, j), val)| (i + 1, j + 1, val))
                    .collect();
                // Sort column-major for reader convenience.
                v.sort_unstable_by_key(|&(i, j, _)| (j, i));
                v
            }
        };

        let mut file = std::fs::File::create(&mtx_path)?;
        writeln!(file, "%%MatrixMarket matrix coordinate real symmetric")?;
        writeln!(file, "{} {} {}", dim, dim, entries.len())?;
        for (i, j, v) in &entries {
            writeln!(file, "{} {} {:.17e}", i, j, v)?;
        }
        Ok(())
    };
    if let Err(e) = write_mtx() {
        log::warn!("kkt_dump: failed to write {}.mtx: {}", stem, e);
        return;
    }

    // --- JSON sidecar (.json) ---
    let json_path = dir.join(format!("{}.json", stem));
    let (pos, neg, zer) = inertia.unwrap_or((0, 0, 0));
    let write_json = || -> std::io::Result<()> {
        let meta = serde_json::json!({
            "problem_name": name,
            "iteration": iteration,
            "n": kkt.n,
            "m": kkt.m,
            "rhs": kkt.rhs,
            "inertia": { "positive": pos, "negative": neg, "zero": zer },
            "delta_w": delta_w,
            "delta_c": delta_c,
            "status": "ongoing"
        });
        let mut file = std::fs::File::create(&json_path)?;
        write!(file, "{}", meta)?;
        Ok(())
    };
    if let Err(e) = write_json() {
        log::warn!("kkt_dump: failed to write {}.json: {}", stem, e);
    }
}

/// Core IPM solver implementation.
fn solve_ipm<P: NlpProblem>(problem: &P, options: &SolverOptions) -> SolveResult {
    // --- NLP Scaling (gradient-based, matching Ipopt's nlp_scaling_method) ---
    // Scale objective and constraints so max gradient norm at x0 is ≤ 100.
    let n_sc = problem.num_variables();
    let m_sc = problem.num_constraints();

    let mut x0 = vec![0.0; n_sc];
    problem.initial_point(&mut x0);

    // --- Problem scaling ---
    // User-provided scaling takes priority over automatic gradient-based scaling.
    let (jac_rows_sc, _) = problem.jacobian_structure();
    let (obj_scaling, g_scaling) = if options.user_obj_scaling.is_some() || options.user_g_scaling.is_some() {
        let os = options.user_obj_scaling.unwrap_or(1.0);
        let gs = options.user_g_scaling.clone().unwrap_or_else(|| vec![1.0; m_sc]);
        (os, gs)
    } else {
        // Automatic gradient-based scaling
        let nlp_scaling_max_gradient = 100.0;
        let nlp_scaling_min_value = 1e-2;
        let mut grad_f0 = vec![0.0; n_sc];
        let grad_ok = problem.gradient(&x0, true, &mut grad_f0);
        let grad_max = if grad_ok {
            grad_f0.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
        } else {
            0.0
        };
        let os = if m_sc > 0 && grad_max > nlp_scaling_max_gradient && grad_max.is_finite() {
            (nlp_scaling_max_gradient / grad_max).max(nlp_scaling_min_value)
        } else {
            1.0
        };

        let mut gs = vec![1.0; m_sc];
        if m_sc > 0 {
            let mut g0_sc = vec![0.0; m_sc];
            let constr_ok = problem.constraints(&x0, false, &mut g0_sc);
            let mut g_l_sc = vec![0.0; m_sc];
            let mut g_u_sc = vec![0.0; m_sc];
            problem.constraint_bounds(&mut g_l_sc, &mut g_u_sc);
            let init_cv = if constr_ok {
                convergence::primal_infeasibility(&g0_sc, &g_l_sc, &g_u_sc)
            } else {
                f64::INFINITY
            };

            if init_cv < 1e6 {
                let mut jac_vals0 = vec![0.0; jac_rows_sc.len()];
                if !problem.jacobian_values(&x0, false, &mut jac_vals0) {
                    // Jacobian eval failed, skip constraint scaling
                } else {
                    let mut row_max = vec![0.0f64; m_sc];
                    for (idx, &row) in jac_rows_sc.iter().enumerate() {
                        let v = jac_vals0[idx].abs();
                        if v.is_finite() && v > row_max[row] {
                            row_max[row] = v;
                        }
                    }
                    for i in 0..m_sc {
                        if row_max[i] > nlp_scaling_max_gradient {
                            gs[i] = (nlp_scaling_max_gradient / row_max[i]).max(nlp_scaling_min_value);
                        }
                    }
                }
            }
        }
        (os, gs)
    };

    if options.print_level >= 5
        && (obj_scaling != 1.0 || g_scaling.iter().any(|&s| s != 1.0))
    {
        let n_scaled_g = g_scaling.iter().filter(|&&s| s != 1.0).count();
        rip_log!(
            "ripopt: NLP scaling: obj_scaling={:.4e}, {}/{} constraints scaled",
            obj_scaling, n_scaled_g, m_sc
        );
    }

    // --- Linear constraint detection (on original unscaled problem for accuracy) ---
    let linear_constraints: Option<Vec<bool>> = if options.detect_linear_constraints && m_sc > 0 {
        let flags = crate::linearity::detect_linear_constraints(problem, &x0);
        let n_linear = flags.iter().filter(|&&f| f).count();
        if n_linear > 0 {
            if options.print_level >= 5 {
                rip_log!(
                    "ripopt: Detected {}/{} linear constraints (Hessian contribution skipped)",
                    n_linear, m_sc
                );
            }
            Some(flags)
        } else {
            None
        }
    } else {
        None
    };

    let scaled = ScaledProblem {
        inner: problem,
        obj_scaling,
        g_scaling: g_scaling.clone(),
        jac_rows: jac_rows_sc,
    };
    let problem = &scaled; // shadow: all subsequent code uses the scaled problem

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

    // Handle warm-start
    if options.warm_start {
        // Copy user-provided initial multipliers before WarmStartInitializer adjusts them
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
        state.mu = WarmStartInitializer::initialize(
            &mut state.x,
            &mut state.z_l,
            &mut state.z_u,
            &state.x_l,
            &state.x_u,
            options,
        );
    }

    // Initialize linear solver — use sparse for large KKT systems
    let use_sparse = (n + m) >= options.sparse_threshold;
    let mut lin_solver: Box<dyn LinearSolver> = if use_sparse {
        // For constrained problems, enable KKT-aware CB pivot search
        // for numerically stable primal-dual 2×2 pivots
        #[cfg(feature = "rmumps")]
        {
            if m > 0 {
                Box::new(MultifrontalLdl::new_kkt(n))
            } else {
                new_sparse_solver_with_choice(options.linear_solver)
            }
        }
        #[cfg(not(feature = "rmumps"))]
        {
            new_sparse_solver_with_choice(options.linear_solver)
        }
    } else {
        Box::new(DenseLdl::new())
    };
    let mut inertia_params = InertiaCorrectionParams::default();
    let mut restoration = RestorationPhase::new(500);

    // Estimate Schur complement density from Jacobian structure.
    // If J^T·D·J would be denser than the full augmented KKT system,
    // disable sparse condensed and use the full (n+m)×(n+m) system instead.
    let mut disable_sparse_condensed = if use_sparse && m > 0 {
        let (jac_rows_est, _) = problem.jacobian_structure();
        // Build row counts
        let mut row_nnz = vec![0usize; m];
        for &r in &jac_rows_est {
            row_nnz[r] += 1;
        }
        // Estimate Schur complement nnz: Σ k_i*(k_i+1)/2 (before dedup)
        let schur_nnz_upper: usize = row_nnz.iter().map(|&k| k * (k + 1) / 2).sum();
        // Augmented KKT nnz: hess_nnz + jac_nnz + n (diagonal)
        let (hess_rows_est, _) = problem.hessian_structure();
        let augmented_nnz = hess_rows_est.len() + jac_rows_est.len() + n;
        let disable = schur_nnz_upper > 2 * augmented_nnz;
        if disable && options.print_level >= 3 {
            rip_log!(
                "ripopt: Disabling sparse condensed KKT: Schur complement nnz estimate ({}) > 2× augmented KKT nnz ({})",
                schur_nnz_upper, augmented_nnz
            );
        }
        disable
    } else {
        false
    };

    // Initialize filter
    let mut filter = Filter::new(1e4);

    // Mehrotra centering parameter from the last iteration's predictor step.
    // Used in the Free-mode mu update: when sigma is available, mu = sigma * mu_current
    // gives a more aggressive (and adaptive) decrease than the Loqo oracle.
    let mut last_mehrotra_sigma: Option<f64> = None;

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
    let mut consecutive_shortened: usize = 0;
    let mut watchdog_active: bool = false;
    let mut watchdog_trial_count: usize = 0;
    let mut watchdog_saved: Option<WatchdogSavedState> = None;

    // Constraint violation history for infeasibility detection
    let theta_history_len: usize = 100;
    let mut theta_history: Vec<f64> = Vec::with_capacity(theta_history_len);

    // Track whether the problem was ever feasible (theta < constr_viol_tol)
    // to prevent false infeasibility declarations on feasible problems.
    let mut ever_feasible = false;

    // Tiny step counter (Ipopt: accept full step when relative step < 10*eps for 2 consecutive)
    let mut consecutive_tiny_steps: usize = 0;

    // Overall progress stall detection: if neither primal nor dual infeasibility
    // improves by at least 1% over many consecutive iterations, terminate early.
    let mut stall_best_pr: f64 = f64::INFINITY;
    let mut stall_best_du: f64 = f64::INFINITY;
    let mut stall_no_progress_count: usize = 0;

    // Line-search backtrack count for the previous iteration (printed in table).
    let mut ls_steps: usize = 0;

    // Primal divergence detection: track consecutive iterations where pr is growing.
    // When pr grows steadily post-restoration, re-trigger restoration rather than
    // continuing for many iterations with worsening feasibility.
    let mut pr_prev_for_divergence: f64 = f64::INFINITY;
    let mut pr_at_divergence_start: f64 = f64::INFINITY;
    let mut consecutive_pr_increase: usize = 0;

    // Consecutive iterations with obj < -1e20 for robust unbounded detection
    let mut consecutive_unbounded: usize = 0;

    // Consecutive iterations where theta (primal infeasibility) stagnated.
    // Used by proactive infeasibility detection to exit early.
    let mut theta_stall_count: usize = 0;

    // Best feasible point tracking: save the best (lowest obj) point that is feasible
    let mut best_x: Option<Vec<f64>> = None;
    let mut best_obj: f64 = f64::INFINITY;

    // Best-du point tracking
    let mut best_du_x: Option<Vec<f64>> = None;
    let mut best_du_val: f64 = f64::INFINITY;
    let mut best_du_y: Option<Vec<f64>> = None;
    let mut best_du_zl: Option<Vec<f64>> = None;
    let mut best_du_zu: Option<Vec<f64>> = None;

    // Dual stagnation detection: track best du improvement.
    // If du hasn't improved significantly over many iterations and we have a
    // best feasible point, restore it and restart with fresh parameters.
    let mut dual_stall_last_good_du: f64 = f64::INFINITY;
    let mut dual_stall_last_good_iter: usize = 0;
    let mut dual_stall_triggered: bool = false;

    // Strategy 1: Iterate averaging for oscillation recovery
    const AVG_WINDOW: usize = 6;
    let mut du_history: Vec<f64> = Vec::with_capacity(AVG_WINDOW + 1);
    let mut iterate_history: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = Vec::new(); // (x, y, z_l, z_u)
    let mut tried_iterate_averaging: bool = false;

    // Strategy 2: Damped multiplier updates when oscillation detected
    let mut prev_dy: Option<Vec<f64>> = None;
    let mut dy_sign_change_count: Vec<u8> = vec![0u8; m]; // per-component consecutive sign change count

    // Strategy 3: Active set reduced KKT solve
    let mut tried_active_set: bool = false;

    // Strategy 4: Complementarity polishing — force mu small when compl is bottleneck
    let mut _tried_compl_polish: bool = false;

    // Initial evaluation
    let init_eval_ok = state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode);
        if let Some(ref mut lbfgs) = lbfgs_state {
            let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
            );
            lbfgs.update(&state.x, &lag_grad);
            lbfgs.fill_hessian(&mut state.hess_vals);
        }

    // NaN/Inf guard on initial evaluation — try perturbation before giving up
    if !init_eval_ok || state.obj.is_nan() || state.obj.is_infinite()
        || state.grad_f.iter().any(|v| v.is_nan() || v.is_infinite())
    {
        let mut recovered = false;
        let x_saved = state.x.clone();
        for &push_factor in &[1e-2, 1e-1, 0.5] {
            // Reset to saved point and apply stronger push
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
            // Re-initialize bound multipliers after perturbation
            for i in 0..n {
                if state.x_l[i].is_finite() {
                    let slack = (state.x[i] - state.x_l[i]).max(1e-20);
                    state.z_l[i] = options.mu_init / slack;
                }
                if state.x_u[i].is_finite() {
                    let slack = (state.x_u[i] - state.x[i]).max(1e-20);
                    state.z_u[i] = options.mu_init / slack;
                }
            }
            let perturb_ok = state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode);
        if let Some(ref mut lbfgs) = lbfgs_state {
            let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
            );
            lbfgs.update(&state.x, &lag_grad);
            lbfgs.fill_hessian(&mut state.hess_vals);
        }
            if perturb_ok && !state.obj.is_nan() && !state.obj.is_infinite()
                && !state.grad_f.iter().any(|v| v.is_nan() || v.is_infinite())
            {
                recovered = true;
                break;
            }
        }
        if !recovered {
            return make_result(&state, SolveStatus::EvaluationError);
        }
    }

    // Initialize constraint slack barrier multipliers v_l, v_u (Ipopt's v_L, v_U).
    // For each inequality constraint side: v = mu_init / slack.
    // This matches Ipopt's IpDefaultIterateInitializer.cpp.
    for i in 0..m {
        let is_eq = state.g_l[i].is_finite() && state.g_u[i].is_finite()
            && (state.g_l[i] - state.g_u[i]).abs() < 1e-15;
        if is_eq {
            continue;
        }
        if state.g_l[i].is_finite() {
            let slack = (state.g[i] - state.g_l[i]).max(1e-20);
            state.v_l[i] = options.mu_init / slack;
        }
        if state.g_u[i].is_finite() {
            let slack = (state.g_u[i] - state.g[i]).max(1e-20);
            state.v_u[i] = options.mu_init / slack;
        }
    }

    // Set filter parameters based on initial constraint violation
    let theta_init = state.constraint_violation();
    filter.set_theta_min_from_initial(theta_init);

    // Print iteration table header (shown at print_level >= 3, reprinted every 25 rows)
    let mut log_line_count: usize = 0;
    if options.print_level >= 3 {
        rip_log!(
            "{:>4}  {:>14}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}  {:>3}",
            "iter", "objective", "inf_pr", "inf_du", "mu", "alpha_pr", "alpha_du", "ls"
        );
    }

    if options.print_level >= 5 {
        rip_log!("ripopt: Starting main loop (n={}, m={})", n, m);
    }

    // Main IPM loop
    for iteration in 0..options.max_iter {
        state.iter = iteration;

        // Check wall-clock time limit (every iteration in early phase, every 10 after)
        if (iteration < 10 || iteration % 10 == 0) && options.max_wall_time > 0.0 {
            if start_time.elapsed().as_secs_f64() >= options.max_wall_time {
                return make_result(&state, SolveStatus::MaxIterations);
            }
        }

        // Early stall detection: bail out if stuck in early iterations.
        // Scale timeout by problem size: medium-scale problems (n+m > 1000) can
        // legitimately spend 30-60s on restoration or line search in early iterations.
        let early_timeout = options.early_stall_timeout * ((n + m) as f64 / 200.0).max(1.0);
        if iteration < 5 && options.early_stall_timeout > 0.0 {
            if start_time.elapsed().as_secs_f64() > early_timeout {
                if options.print_level >= 3 {
                    rip_log!(
                        "ripopt: Early stall at iteration {} ({:.1}s elapsed), terminating",
                        iteration, start_time.elapsed().as_secs_f64()
                    );
                }
                return make_result(&state, SolveStatus::NumericalError);
            }
        }

        // --- Dual stagnation detection (runs every iteration, including restoration) ---
        // Track best du seen. If du hasn't improved for 500+ iterations and we have a
        // best feasible point, restore it with fresh filter/mu.
        // This catches cases where restoration cycling pushes the solver off a good
        // region and it gets stuck for thousands of iterations.
        if iteration > 0 {
            let current_du = convergence::dual_infeasibility(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
                &state.y, &state.z_l, &state.z_u, n,
            );
            if current_du < 0.5 * dual_stall_last_good_du {
                dual_stall_last_good_du = current_du;
                dual_stall_last_good_iter = iteration;
            }

            let stall_iters = iteration.saturating_sub(dual_stall_last_good_iter);
            if stall_iters >= 500
                && !dual_stall_triggered
                && current_du > 100.0 * options.tol
                && best_x.is_some()
            {
                // Dual stagnation detected. Restore the best-du point (which had
                // du=best_du_val with stored x, y, z). This point was near-converged
                // but got disrupted by restoration cycling.
                if let Some(ref bdx) = best_du_x {
                    log::debug!(
                        "Dual stagnation at iter {}: du={:.2e}, restoring best-du point (du={:.2e} at iter {})",
                        iteration, current_du, dual_stall_last_good_du, dual_stall_last_good_iter
                    );
                    state.x.copy_from_slice(bdx);
                    if let Some(ref bdy) = best_du_y { state.y.copy_from_slice(bdy); }
                    if let Some(ref bdzl) = best_du_zl { state.z_l.copy_from_slice(bdzl); }
                    if let Some(ref bdzu) = best_du_zu { state.z_u.copy_from_slice(bdzu); }
                    let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode);
        if let Some(ref mut lbfgs) = lbfgs_state {
            let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
            );
            lbfgs.update(&state.x, &lag_grad);
            lbfgs.fill_hessian(&mut state.hess_vals);
        }

                    // Reset filter and bump mu for a fresh start from the good point.
                    filter.reset();
                    let theta_restart = state.constraint_violation();
                    filter.set_theta_min_from_initial(theta_restart);
                    state.mu = (state.mu * 100.0).max(1e-4).min(1e-1);
                    if options.mu_strategy_adaptive {
                        mu_state.mode = MuMode::Free;
                    }
                    mu_state.first_iter_in_mode = true;
                    mu_state.consecutive_restoration_failures = 0;
                    inertia_params.delta_w_last = 0.0;

                    // Check if the restored point meets acceptable convergence.
                    // The point had du=best_du_val which may already be excellent.
                    let rest_pr = state.constraint_violation();
                    let rest_du = convergence::dual_infeasibility(
                        &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
                        &state.y, &state.z_l, &state.z_u, n,
                    );
                    let rest_co = convergence::complementarity_error(
                        &state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u, 0.0,
                    );
                    // Use relaxed tolerances (acceptable level)
                    let s_max = 100.0_f64;
                    let mult_sum: f64 = state.y.iter().map(|v| v.abs()).sum::<f64>()
                        + state.z_l.iter().map(|v| v.abs()).sum::<f64>()
                        + state.z_u.iter().map(|v| v.abs()).sum::<f64>();
                    let s_d = if (m + 2 * n) > 0 {
                        (s_max.max(mult_sum / (m + 2 * n) as f64) / s_max).min(1e4)
                    } else { 1.0 };
                    let near_tol = 100.0 * options.tol;
                    let du_tol = (near_tol * s_d).max(1e-2);
                    let co_tol = (near_tol * s_d).max(1e-2);
                    let pr_tol = near_tol.max(10.0 * options.constr_viol_tol);
                    if rest_pr <= pr_tol && rest_du <= du_tol && rest_co <= co_tol {
                        log::debug!(
                            "Restored best-du point passes near-tolerance (pr={:.2e}, du={:.2e}, co={:.2e})",
                            rest_pr, rest_du, rest_co
                        );
                        // Near-tolerance but promotion strategies unavailable here — return NumericalError
                        return make_result(&state, SolveStatus::NumericalError);
                    }

                    dual_stall_triggered = true;
                    // Fall through to normal iteration from good point
                }
            }
        }

        // Compute optimality measures.
        let primal_inf = state.constraint_violation();

        // Compute z_opt from stationarity for the scaled convergence check.
        // At optimality, grad_f + J^T y - z_l + z_u = 0.
        // z_opt captures the true bound multiplier for active bounds.
        //
        // Complementarity gate: only use z_opt when z_opt * slack is consistent
        // with the barrier problem (z*s ~ mu). If z_opt * slack >> mu, the point
        // is not a barrier-optimal point and z_opt would hide a true infeasibility.
        let (z_l_opt, z_u_opt) = {
            let mut grad_jty = state.grad_f.clone();
            for (idx, (&row, &col)) in
                state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate()
            {
                grad_jty[col] += state.jac_vals[idx] * state.y[row];
            }
            let mut zl = vec![0.0; n];
            let mut zu = vec![0.0; n];
            let kappa_compl = 1e12;
            for i in 0..n {
                if grad_jty[i] > 0.0 && state.x_l[i].is_finite() {
                    let s_l = (state.x[i] - state.x_l[i]).max(1e-20);
                    if grad_jty[i] * s_l <= kappa_compl * state.mu.max(1e-20) {
                        zl[i] = grad_jty[i];
                    }
                } else if grad_jty[i] < 0.0 && state.x_u[i].is_finite() {
                    let s_u = (state.x_u[i] - state.x[i]).max(1e-20);
                    if (-grad_jty[i]) * s_u <= kappa_compl * state.mu.max(1e-20) {
                        zu[i] = -grad_jty[i];
                    }
                }
            }
            (zl, zu)
        };

        // Scaled dual infeasibility uses z_opt (for fast convergence detection)
        let dual_inf = convergence::dual_infeasibility(
            &state.grad_f,
            &state.jac_rows,
            &state.jac_cols,
            &state.jac_vals,
            &state.y,
            &z_l_opt,
            &z_u_opt,
            n,
        );

        // Unscaled dual infeasibility uses iterative z with component-wise scaling
        // (catches false convergence while being insensitive to gradient magnitude)
        let dual_inf_unscaled = convergence::dual_infeasibility_scaled(
            &state.grad_f,
            &state.jac_rows,
            &state.jac_cols,
            &state.jac_vals,
            &state.y,
            &state.z_l,
            &state.z_u,
            n,
        );
        let compl_inf = convergence::complementarity_error(
            &state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u, 0.0,
        );
        // Also compute complementarity using z_opt (NLP multipliers from stationarity).
        // When mu is stuck high, kappa_sigma safeguard inflates iterative z, making
        // compl_inf huge even at the NLP optimum. z_opt correctly reflects the NLP solution.
        let compl_inf_opt = convergence::complementarity_error(
            &state.x, &state.x_l, &state.x_u, &z_l_opt, &z_u_opt, 0.0,
        );
        let compl_inf_best = compl_inf.min(compl_inf_opt);

        if options.print_level >= 3 {
            // Reprint header every 25 data rows for readability
            if log_line_count > 0 && log_line_count % 25 == 0 {
                rip_log!(
                    "{:>4}  {:>14}  {:>10}  {:>10}  {:>7}  {:>8}  {:>8}  {:>3}",
                    "iter", "objective", "inf_pr", "inf_du", "lg(mu)", "alpha_pr", "alpha_du", "ls"
                );
            }
            rip_log!(
                "{:>4}  {:>14.7e}  {:>10.2e}  {:>10.2e}  {:>10.2e}  {:>8.2e}  {:>8.2e}  {:>3}",
                iteration,
                state.obj / state.obj_scaling,
                primal_inf,
                dual_inf,
                state.mu,
                state.alpha_primal,
                state.alpha_dual,
                ls_steps,
            );
            log_line_count += 1;
        }

        // z_opt component-wise scaled dual infeasibility (fallback for unscaled gate)
        let dual_inf_unscaled_opt = convergence::dual_infeasibility_scaled(
            &state.grad_f,
            &state.jac_rows,
            &state.jac_cols,
            &state.jac_vals,
            &state.y,
            &z_l_opt,
            &z_u_opt,
            n,
        );

        // Invoke intermediate callback (if registered)
        if !crate::intermediate::invoke_intermediate(
            iteration,
            state.obj / state.obj_scaling,
            primal_inf,
            dual_inf,
            state.mu,
            state.alpha_primal,
            state.alpha_dual,
            ls_steps,
        ) {
            if options.print_level >= 5 {
                rip_log!("ripopt: User requested stop via intermediate callback");
            }
            return make_result(&state, SolveStatus::UserRequestedStop);
        }

        // Compute multiplier scaling for convergence check
        let multiplier_sum: f64 = state.y.iter().map(|v| v.abs()).sum::<f64>()
            + state.z_l.iter().map(|v| v.abs()).sum::<f64>()
            + state.z_u.iter().map(|v| v.abs()).sum::<f64>();
        let multiplier_count = m + 2 * n;

        // Check convergence (use best complementarity: min of iterative z and z_opt)
        let conv_info = ConvergenceInfo {
            primal_inf,
            dual_inf,
            dual_inf_unscaled,
            dual_inf_unscaled_opt,
            compl_inf: compl_inf_best,
            compl_inf_opt,
            mu: state.mu,
            objective: state.obj,
            multiplier_sum,
            multiplier_count,
        };

        // Track iterate history for oscillation detection (Strategy 1)
        du_history.push(dual_inf);
        iterate_history.push((state.x.clone(), state.y.clone(), state.z_l.clone(), state.z_u.clone()));
        if du_history.len() > AVG_WINDOW {
            du_history.remove(0);
            iterate_history.remove(0);
        }

        match check_convergence(&conv_info, options, state.consecutive_acceptable) {
            ConvergenceStatus::Converged => {
                if options.print_level >= 5 {
                    timings.print_summary(iteration + 1, ipm_start.elapsed());
                }
                return make_result(&state, SolveStatus::Optimal);
            }
            ConvergenceStatus::Acceptable => {
                // Strategy 1: Try iterate averaging before declaring Acceptable
                if !tried_iterate_averaging && du_history.len() == AVG_WINDOW {
                    // Check for oscillation: count sign changes in du differences
                    let mut sign_changes = 0;
                    for w in 1..du_history.len() - 1 {
                        let d1 = du_history[w] - du_history[w - 1];
                        let d2 = du_history[w + 1] - du_history[w];
                        if d1 * d2 < 0.0 {
                            sign_changes += 1;
                        }
                    }
                    if sign_changes >= AVG_WINDOW / 2 {
                        _ = std::mem::replace(&mut tried_iterate_averaging, true);
                        // Average the iterates
                        let len = iterate_history.len() as f64;
                        let mut avg_x = vec![0.0; n];
                        let mut avg_y = vec![0.0; m];
                        let mut avg_zl = vec![0.0; n];
                        let mut avg_zu = vec![0.0; n];
                        for (hx, hy, hzl, hzu) in &iterate_history {
                            for i in 0..n { avg_x[i] += hx[i] / len; }
                            for i in 0..m { avg_y[i] += hy[i] / len; }
                            for i in 0..n { avg_zl[i] += hzl[i] / len; }
                            for i in 0..n { avg_zu[i] += hzu[i] / len; }
                        }
                        // Clamp averaged point to bounds and ensure z >= 0
                        for i in 0..n {
                            avg_x[i] = avg_x[i].clamp(
                                if state.x_l[i].is_finite() { state.x_l[i] + 1e-15 } else { f64::NEG_INFINITY },
                                if state.x_u[i].is_finite() { state.x_u[i] - 1e-15 } else { f64::INFINITY },
                            );
                            avg_zl[i] = avg_zl[i].max(0.0);
                            avg_zu[i] = avg_zu[i].max(0.0);
                        }
                        // Evaluate convergence at averaged point
                        let saved_x = state.x.clone();
                        let saved_y = state.y.clone();
                        let saved_zl = state.z_l.clone();
                        let saved_zu = state.z_u.clone();
                        state.x.copy_from_slice(&avg_x);
                        state.y.copy_from_slice(&avg_y);
                        state.z_l.copy_from_slice(&avg_zl);
                        state.z_u.copy_from_slice(&avg_zu);
                        let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode);
                        let avg_pr = convergence::primal_infeasibility(&state.g, &state.g_l, &state.g_u);
                        let avg_du = convergence::dual_infeasibility(
                            &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
                            &avg_y, &avg_zl, &avg_zu, n,
                        );
                        let avg_compl = convergence::complementarity_error(
                            &avg_x, &state.x_l, &state.x_u, &avg_zl, &avg_zu, 0.0,
                        );
                        let avg_conv = ConvergenceInfo {
                            primal_inf: avg_pr, dual_inf: avg_du,
                            dual_inf_unscaled: avg_du,
                            dual_inf_unscaled_opt: avg_du,
                            compl_inf: avg_compl,
                            compl_inf_opt: avg_compl,
                            mu: state.mu, objective: state.obj,
                            multiplier_sum: avg_y.iter().map(|v| v.abs()).sum::<f64>()
                                + avg_zl.iter().map(|v| v.abs()).sum::<f64>()
                                + avg_zu.iter().map(|v| v.abs()).sum::<f64>(),
                            multiplier_count: m + 2 * n,
                        };
                        if let ConvergenceStatus::Converged = check_convergence(&avg_conv, options, 0) {
                            if options.print_level >= 3 {
                                rip_log!("ripopt: Iterate averaging promoted near-tolerance -> Optimal (du={:.2e})", avg_du);
                            }
                            return make_result(&state, SolveStatus::Optimal);
                        }
                        // Restore original state if averaging didn't help
                        state.x.copy_from_slice(&saved_x);
                        state.y.copy_from_slice(&saved_y);
                        state.z_l.copy_from_slice(&saved_zl);
                        state.z_u.copy_from_slice(&saved_zu);
                        let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode);
                    }
                }

                // Strategy 3: Try active set identification + reduced solve
                if !tried_active_set {
                    _ = std::mem::replace(&mut tried_active_set, true);
                    if let Some(result) = try_active_set_solve(&mut state, problem, options, linear_constraints.as_deref(), lbfgs_mode) {
                        if options.print_level >= 3 {
                            rip_log!("ripopt: Active set solve promoted Acceptable -> Optimal");
                        }
                        return result;
                    }
                }

                // Strategy 4: Complementarity polishing via multiplier snap
                // When complementarity is the bottleneck (primal/dual already good enough),
                // snap bound multipliers to reduce complementarity, then recheck convergence.
                if !_tried_compl_polish {
                    let compl_inf_now = conv_info.compl_inf;
                    let s_d_now = {
                        let s_max: f64 = 100.0;
                        let s_d_max: f64 = 1e4;
                        if conv_info.multiplier_count > 0 {
                            ((s_max.max(conv_info.multiplier_sum / conv_info.multiplier_count as f64)) / s_max).min(s_d_max)
                        } else { 1.0 }
                    };
                    let compl_tol_scaled = options.tol * s_d_now;
                    if compl_inf_now > compl_tol_scaled
                        && conv_info.primal_inf <= 100.0 * options.tol
                        && conv_info.dual_inf <= 100.0 * options.tol * s_d_now
                    {
                        _tried_compl_polish = true;
                        // For variables near bounds, snap multipliers to reduce complementarity:
                        // If x_i ≈ x_l_i (gap < tol), keep z_l_i from stationarity (z_opt)
                        // If x_i is interior (gap > tol), set z_l_i = 0
                        let saved_zl = state.z_l.clone();
                        let saved_zu = state.z_u.clone();
                        let gap_tol = 1e-6;
                        for i in 0..n {
                            let gap_l = if state.x_l[i].is_finite() { state.x[i] - state.x_l[i] } else { f64::INFINITY };
                            let gap_u = if state.x_u[i].is_finite() { state.x_u[i] - state.x[i] } else { f64::INFINITY };
                            // If clearly interior to lower bound, zero out z_l
                            if gap_l > gap_tol {
                                state.z_l[i] = 0.0;
                            }
                            // If clearly interior to upper bound, zero out z_u
                            if gap_u > gap_tol {
                                state.z_u[i] = 0.0;
                            }
                        }
                        // Recompute convergence with snapped multipliers
                        let snap_du = convergence::dual_infeasibility(
                            &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
                            &state.y, &state.z_l, &state.z_u, n,
                        );
                        let snap_compl = convergence::complementarity_error(
                            &state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u, 0.0,
                        );
                        let snap_mult_sum: f64 = state.y.iter().map(|v| v.abs()).sum::<f64>()
                            + state.z_l.iter().map(|v| v.abs()).sum::<f64>()
                            + state.z_u.iter().map(|v| v.abs()).sum::<f64>();
                        let snap_conv = ConvergenceInfo {
                            primal_inf: conv_info.primal_inf,
                            dual_inf: snap_du,
                            dual_inf_unscaled: snap_du,
                            dual_inf_unscaled_opt: snap_du,
                            compl_inf: snap_compl,
                            compl_inf_opt: snap_compl,
                            mu: state.mu,
                            objective: state.obj,
                            multiplier_sum: snap_mult_sum,
                            multiplier_count: m + 2 * n,
                        };
                        if let ConvergenceStatus::Converged = check_convergence(&snap_conv, options, 0) {
                            if options.print_level >= 3 {
                                rip_log!(
                                    "ripopt: Complementarity snap promoted near-tolerance -> Optimal (compl {:.2e} -> {:.2e}, du {:.2e})",
                                    compl_inf_now, snap_compl, snap_du
                                );
                            }
                            return make_result(&state, SolveStatus::Optimal);
                        }
                        // Snap didn't work — restore multipliers
                        state.z_l.copy_from_slice(&saved_zl);
                        state.z_u.copy_from_slice(&saved_zu);
                    }
                }

                if options.print_level >= 5 {
                    timings.print_summary(iteration + 1, ipm_start.elapsed());
                }
                return make_result(&state, SolveStatus::NumericalError);
            }
            ConvergenceStatus::Diverging => {
                return make_result(&state, SolveStatus::Unbounded);
            }
            ConvergenceStatus::NotConverged => {}
        }

        // Track consecutive acceptable iterations (using same criteria as check_convergence)
        let s_d_for_acc = {
            let s_max: f64 = 100.0;
            let s_d_max: f64 = 1e4;
            if (m + 2 * n) > 0 {
                ((s_max.max(multiplier_sum / (m + 2 * n) as f64)) / s_max).min(s_d_max)
            } else {
                1.0
            }
        };
        let meets_acc_scaled = primal_inf <= 100.0 * options.tol
            && dual_inf <= 100.0 * options.tol * s_d_for_acc
            && compl_inf_best <= 100.0 * options.tol * s_d_for_acc;
        let meets_acc_unscaled = primal_inf <= 10.0 * options.constr_viol_tol
            && dual_inf_unscaled <= 10.0 * options.dual_inf_tol
            && compl_inf_best <= 10.0 * options.compl_inf_tol;
        if meets_acc_scaled && meets_acc_unscaled {
            state.consecutive_acceptable += 1;
        } else {
            state.consecutive_acceptable = 0;
        }

        // Track best-du point for cycling/stall detection at max_iter exit.
        if dual_inf < best_du_val {
            best_du_val = dual_inf;
            best_du_x = Some(state.x.clone());
            best_du_y = Some(state.y.clone());
            best_du_zl = Some(state.z_l.clone());
            best_du_zu = Some(state.z_u.clone());
        }

        // Track constraint violation history for infeasibility detection
        if theta_history.len() >= theta_history_len {
            theta_history.remove(0);
        }
        theta_history.push(primal_inf);

        // Track whether we've ever been feasible
        if primal_inf < options.constr_viol_tol {
            ever_feasible = true;
        }

        // Proactive infeasibility detection: if θ has stagnated for many consecutive
        // iterations AND the gradient of the violation is near-zero, declare infeasibility
        // earlier rather than burning iterations until restoration eventually fires.
        if options.proactive_infeasibility_detection
            && !ever_feasible
            && m > 0
            && iteration >= 50
            && primal_inf > options.constr_viol_tol
            && theta_history.len() >= theta_history_len
        {
            let theta_min_h = theta_history.iter().cloned().fold(f64::INFINITY, f64::min);
            let theta_max_h = theta_history.iter().cloned().fold(0.0f64, f64::max);
            // "Stagnated" = less than 1% relative variation over the history window
            if theta_max_h > 0.0 && (theta_max_h - theta_min_h) < 0.01 * primal_inf {
                theta_stall_count += 1;
            } else {
                theta_stall_count = 0;
            }
            // After 10 consecutive stagnation windows, check stationarity of ∇θ
            if theta_stall_count >= 10 {
                let mut violation = vec![0.0; m];
                for i in 0..m {
                    let is_eq = state.g_l[i].is_finite() && state.g_u[i].is_finite()
                        && (state.g_l[i] - state.g_u[i]).abs() < 1e-15;
                    if is_eq {
                        violation[i] = state.g[i] - state.g_l[i];
                    } else if state.g_l[i].is_finite() && state.g[i] < state.g_l[i] {
                        violation[i] = state.g[i] - state.g_l[i];
                    } else if state.g_u[i].is_finite() && state.g[i] > state.g_u[i] {
                        violation[i] = state.g[i] - state.g_u[i];
                    }
                }
                let mut grad_theta = vec![0.0; n];
                for (idx, (&row, &col)) in
                    state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate()
                {
                    grad_theta[col] += state.jac_vals[idx] * violation[row];
                }
                let grad_theta_norm = grad_theta.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                let stationarity_tol = 1e-3 * primal_inf.max(1.0);
                if grad_theta_norm < stationarity_tol {
                    log::info!(
                        "Proactive infeasibility at iter {}: θ stagnated at {:.2e}, ‖∇θ‖={:.2e}",
                        iteration, primal_inf, grad_theta_norm
                    );
                    return make_result(&state, SolveStatus::LocalInfeasibility);
                }
                // Stationarity not met — reset counter to check again in another window
                theta_stall_count = 0;
            }
        } else if ever_feasible {
            theta_stall_count = 0;
        }

        // Unbounded detection: objective diverging negatively with satisfied constraints.
        // Require 10 consecutive iterations to avoid false positives from transient dips.
        if state.obj < -1e20 && primal_inf < options.constr_viol_tol {
            consecutive_unbounded += 1;
            if consecutive_unbounded >= 10 {
                return make_result(&state, SolveStatus::Unbounded);
            }
        } else {
            consecutive_unbounded = 0;
        }

        // Overall progress stall detection: terminate when the solver is stuck
        // with no improvement in either primal or dual infeasibility.
        // Two triggers: (1) tiny steps for 15 iterations, (2) no metric improvement
        // for 30 iterations regardless of step size.
        // Only activate after 50 iterations to avoid tripping during early phases.
        if iteration > 50 && options.stall_iter_limit > 0 {
            let pr_improved = primal_inf < 0.99 * stall_best_pr;
            let du_improved = dual_inf < 0.99 * stall_best_du;
            if pr_improved {
                stall_best_pr = primal_inf;
            }
            if du_improved {
                stall_best_du = dual_inf;
            }
            if pr_improved || du_improved {
                stall_no_progress_count = 0;
            } else {
                stall_no_progress_count += 1;
                let tiny_alpha = state.alpha_primal < 1e-8 && state.alpha_dual < 1e-4;
                // Terminate after half the limit with truly negligible steps,
                // or the full limit with no metric improvement regardless of step size
                let stall_limit = if tiny_alpha { options.stall_iter_limit / 2 } else { options.stall_iter_limit };
                if stall_no_progress_count >= stall_limit {
                    // Before declaring NumericalError, check if the current point
                    // is near-tolerance (1000x tol). Such points are close but not optimal.
                    let stall_near_tol = options.tol * 1000.0;
                    let stall_pr_ok = primal_inf <= stall_near_tol.max(10.0 * options.constr_viol_tol);
                    let stall_du_ok = dual_inf <= (stall_near_tol * s_d_for_acc).max(1e-2);
                    let stall_co_ok = compl_inf_best <= (stall_near_tol * s_d_for_acc).max(1e-2);
                    if stall_pr_ok && stall_du_ok && stall_co_ok {
                        // In Fixed (monotone) mode, stalling near tolerance means the barrier
                        // subproblem is solved at the current mu — force a mu decrease to
                        // continue toward the NLP optimum instead of returning NumericalError.
                        if !options.mu_strategy_adaptive && state.mu > options.mu_min {
                            let new_mu = (options.mu_linear_decrease_factor * state.mu)
                                .min(state.mu.powf(options.mu_superlinear_decrease_power))
                                .max(options.mu_min);
                            if options.print_level >= 3 {
                                rip_log!(
                                    "ripopt: Fixed mode stall near tolerance (pr={:.2e}, du={:.2e}, co={:.2e}), forcing mu {:.2e} -> {:.2e}",
                                    primal_inf, dual_inf, compl_inf_best, state.mu, new_mu
                                );
                            }
                            state.mu = new_mu;
                            filter.reset();
                            let theta_new = state.constraint_violation();
                            filter.set_theta_min_from_initial(theta_new);
                            stall_no_progress_count = 0;
                            stall_best_pr = f64::INFINITY;
                            stall_best_du = f64::INFINITY;
                            continue;
                        } else {
                            if options.print_level >= 3 {
                                rip_log!(
                                    "ripopt: Stalled but near-tolerance (pr={:.2e}, du={:.2e}, co={:.2e}), returning NumericalError",
                                    primal_inf, dual_inf, compl_inf_best
                                );
                            }
                            return make_result(&state, SolveStatus::NumericalError);
                        }
                    }
                    // Full two-gate near-tolerance check with optimal dual multipliers.
                    // When duals have diverged but the primal point is near-optimal
                    // (e.g., HS116: obj close to optimal, small primal_inf), the simple
                    // check above fails because it uses the current (diverged) duals.
                    // Recompute optimal duals from the gradient to get a cleaner picture.
                    {
                        let mut gj = state.grad_f.clone();
                        for (idx, (&row, &col)) in
                            state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate()
                        {
                            gj[col] += state.jac_vals[idx] * state.y[row];
                        }
                        let mut opt_zl = vec![0.0; n];
                        let mut opt_zu = vec![0.0; n];
                        let kc = 1e10;
                        for i in 0..n {
                            if gj[i] > 0.0 && state.x_l[i].is_finite() {
                                let sl = (state.x[i] - state.x_l[i]).max(1e-20);
                                if gj[i] * sl <= kc * state.mu.max(1e-20) {
                                    opt_zl[i] = gj[i];
                                }
                            } else if gj[i] < 0.0 && state.x_u[i].is_finite() {
                                let su = (state.x_u[i] - state.x[i]).max(1e-20);
                                if (-gj[i]) * su <= kc * state.mu.max(1e-20) {
                                    opt_zu[i] = -gj[i];
                                }
                            }
                        }
                        let opt_du = convergence::dual_infeasibility(
                            &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
                            &state.y, &opt_zl, &opt_zu, n,
                        );
                        let opt_co = convergence::complementarity_error(
                            &state.x, &state.x_l, &state.x_u, &opt_zl, &opt_zu, 0.0,
                        );
                        let opt_co_best = compl_inf_best.min(opt_co);
                        let fmult: f64 = state.y.iter().map(|v| v.abs()).sum::<f64>()
                            + opt_zl.iter().map(|v| v.abs()).sum::<f64>()
                            + opt_zu.iter().map(|v| v.abs()).sum::<f64>();
                        let fsd = if (m + 2 * n) > 0 {
                            ((100.0f64.max(fmult / (m + 2 * n) as f64)) / 100.0).min(1e4)
                        } else {
                            1.0
                        };
                        let stall_fdu_tol = (stall_near_tol * fsd).max(1e-2);
                        let stall_fco_tol = (stall_near_tol * fsd).max(1e-2);
                        let stall_fpr_tol = stall_near_tol.max(10.0 * options.constr_viol_tol);
                        // Scaled gate with optimal duals
                        let sc = primal_inf <= stall_fpr_tol
                            && opt_du <= stall_fdu_tol
                            && opt_co_best <= stall_fco_tol;
                        // Unscaled gate with original duals (component-wise scaled)
                        let du_u = convergence::dual_infeasibility_scaled(
                            &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
                            &state.y, &state.z_l, &state.z_u, n,
                        );
                        let usc = primal_inf <= 10.0 * options.constr_viol_tol
                            && du_u <= 10.0 * options.dual_inf_tol
                            && opt_co_best <= 10.0 * options.compl_inf_tol;
                        if sc && usc {
                            if options.print_level >= 3 {
                                rip_log!(
                                    "ripopt: Stalled but near-tolerance via optimal duals (pr={:.2e}, du_opt={:.2e}, co={:.2e}), returning NumericalError",
                                    primal_inf, opt_du, opt_co_best
                                );
                            }
                            return make_result(&state, SolveStatus::NumericalError);
                        }
                    }
                    if options.print_level >= 3 {
                        rip_log!(
                            "ripopt: Stalled for {} iterations without progress (alpha_p={:.2e}, pr={:.2e}, du={:.2e}), terminating",
                            stall_no_progress_count, state.alpha_primal, primal_inf, dual_inf
                        );
                    }
                    return make_result(&state, SolveStatus::NumericalError);
                }
            }
        }

        // Primal divergence detection: when pr is growing for several consecutive
        // iterations, force re-entry into restoration rather than continuing with
        // worsening feasibility. This catches the pattern where after NLP restoration,
        // the filter accepts steps via slight phi decrease while theta grows steadily.
        let mut force_restoration = false;
        if m > 0 && iteration > 5 && primal_inf > options.constr_viol_tol {
            if primal_inf > pr_prev_for_divergence * (1.0 + 1e-6) {
                if consecutive_pr_increase == 0 {
                    pr_at_divergence_start = pr_prev_for_divergence;
                }
                consecutive_pr_increase += 1;
            } else {
                consecutive_pr_increase = 0;
            }
            // After 8 consecutive increases AND pr has grown by at least 20% total,
            // force restoration to find a more feasible point. The growth check
            // prevents triggering on tiny numerical oscillations.
            if consecutive_pr_increase >= 8
                && primal_inf > 1.2 * pr_at_divergence_start
            {
                log::info!(
                    "Primal divergence at iter {}: pr grew for {} consecutive iterations ({:.2e} -> {:.2e}), forcing restoration",
                    iteration, consecutive_pr_increase, pr_at_divergence_start, primal_inf
                );
                if options.print_level >= 3 {
                    rip_log!(
                        "ripopt: Primal divergence detected (pr grew {:.2e} -> {:.2e} over {} iters), re-entering restoration",
                        pr_at_divergence_start, primal_inf, consecutive_pr_increase
                    );
                }
                force_restoration = true;
                consecutive_pr_increase = 0;
            }
        } else {
            consecutive_pr_increase = 0;
        }
        pr_prev_for_divergence = primal_inf;

        // Compute sigma (barrier diagonal)
        let sigma = kkt::compute_sigma(&state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u);

        // Use dense condensed KKT (Schur complement) when m >> n and n is small.
        // Condensed cost is O(n^2*m + n^3) vs O((n+m)^3) — strictly better when m >> n.
        // Allow this even for "sparse" problems when n is tiny — an n×n dense solve
        // is always faster than an (n+m)×(n+m) sparse solve for small n.
        let use_condensed = m >= 2 * n && n > 0 && (!use_sparse || n <= 100);

        // Use sparse condensed KKT when problem is large and has constraints,
        // but only if the Schur complement is actually sparser than the augmented system.
        let use_sparse_condensed = use_sparse && m > 0 && !use_condensed && !disable_sparse_condensed;

        let t_kkt = Instant::now();
        let condensed_system = if use_condensed {
            Some(kkt::assemble_condensed_kkt(
                n, m,
                &state.hess_rows, &state.hess_cols, &state.hess_vals,
                &state.jac_rows, &state.jac_cols, &state.jac_vals,
                &sigma, &state.grad_f, &state.g, &state.g_l, &state.g_u,
                &state.y, &state.z_l, &state.z_u,
                &state.x, &state.x_l, &state.x_u, state.mu,
                &state.v_l, &state.v_u,
            ))
        } else {
            None
        };

        let mut sparse_condensed_system = if use_sparse_condensed {
            Some(kkt::assemble_sparse_condensed_kkt(
                n, m,
                &state.hess_rows, &state.hess_cols, &state.hess_vals,
                &state.jac_rows, &state.jac_cols, &state.jac_vals,
                &sigma, &state.grad_f, &state.g, &state.g_l, &state.g_u,
                &state.y, &state.z_l, &state.z_u,
                &state.x, &state.x_l, &state.x_u, state.mu,
                &state.v_l, &state.v_u,
            ))
        } else {
            None
        };

        // Build full KKT only when not using any condensed path
        let mut kkt_system_opt: Option<kkt::KktSystem> = if !use_condensed && !use_sparse_condensed {
            Some(kkt::assemble_kkt(
                n, m,
                &state.hess_rows, &state.hess_cols, &state.hess_vals,
                &state.jac_rows, &state.jac_cols, &state.jac_vals,
                &sigma, &state.grad_f, &state.g, &state.g_l, &state.g_u,
                &state.y, &state.z_l, &state.z_u,
                &state.x, &state.x_l, &state.x_u, state.mu,
                use_sparse, &state.v_l, &state.v_u,
            ))
        } else {
            None
        };
        timings.kkt_assembly += t_kkt.elapsed();

        // On first iteration with sparse condensed, detect bandwidth for the condensed system.
        // If the condensed Schur complement is essentially dense (bandwidth > n/2),
        // abandon it and switch to the full augmented KKT system which MUMPS can
        // reorder efficiently.
        if iteration == 0 && use_sparse_condensed {
            // Detect bandwidth and decide whether to keep sparse condensed or switch to full augmented.
            let sc_bw = sparse_condensed_system.as_ref().map(|sc| {
                BandedLdl::compute_bandwidth(&sc.matrix.triplet_rows, &sc.matrix.triplet_cols)
            });
            if let Some(bw) = sc_bw {
                if bw > n / 2 {
                    // Condensed Schur complement is essentially dense - switch to full
                    // augmented KKT system which MUMPS can reorder efficiently.
                    if options.print_level >= 3 {
                        rip_log!(
                            "ripopt: Sparse condensed S has bandwidth {} for n={}, switching to full augmented KKT",
                            bw, n
                        );
                    }
                    disable_sparse_condensed = true;
                    sparse_condensed_system = None;
                    kkt_system_opt = Some(kkt::assemble_kkt(
                        n, m,
                        &state.hess_rows, &state.hess_cols, &state.hess_vals,
                        &state.jac_rows, &state.jac_cols, &state.jac_vals,
                        &sigma, &state.grad_f, &state.g, &state.g_l, &state.g_u,
                        &state.y, &state.z_l, &state.z_u,
                        &state.x, &state.x_l, &state.x_u, state.mu,
                        use_sparse, &state.v_l, &state.v_u,
                    ));
                } else if bw * bw <= n {
                    if options.print_level >= 5 {
                        rip_log!("ripopt: Sparse condensed S has bandwidth {} for n={}, using banded solver", bw, n);
                    }
                    lin_solver = Box::new(BandedLdl::new());
                } else if options.print_level >= 5 {
                    rip_log!("ripopt: Sparse condensed S has bandwidth {} for n={}, using sparse solver", bw, n);
                }
            }
        }

        // Factor with inertia correction (only for non-condensed path)
        // Track regularization values for iterative refinement against original system
        let mut ic_delta_w = 0.0f64;
        let mut ic_delta_c = 0.0f64;
        if let Some(ref mut kkt_system) = kkt_system_opt {
        let t_fact = Instant::now();
        let inertia_result =
            kkt::factor_with_inertia_correction(kkt_system, lin_solver.as_mut(), &mut inertia_params);
        timings.factorization += t_fact.elapsed();

        match &inertia_result {
            Ok((dw, dc)) => { ic_delta_w = *dw; ic_delta_c = *dc; }
            _ => {}
        }

        // KKT matrix dump for external solver benchmarking (e.g. FERAL).
        // Only fires when options.kkt_dump_dir is Some and factorization succeeded.
        if let Some(ref dump_dir) = options.kkt_dump_dir {
            if inertia_result.is_ok() {
                dump_kkt_matrix(
                    dump_dir,
                    &options.kkt_dump_name,
                    iteration,
                    kkt_system,
                    Some((n, m, 0)),
                    ic_delta_w,
                    ic_delta_c,
                );
            }
        }

        if let Err(e) = inertia_result {
            log::warn!("KKT factorization failed: {}", e);

            // Early-iteration perturbation: if factorization fails in the first 5
            // iterations, the starting point is likely degenerate (singular Jacobian).
            // Try more aggressive perturbation scales before other recovery methods.
            if iteration < 5 {
                let mut early_recovered = false;
                for &perturb_scale in &[1e-4, 1e-3, 1e-2, 5e-2, 1e-1] {
                    let x_saved = state.x.clone();
                    for i in 0..n {
                        let mag = state.x[i].abs().max(1.0);
                        // Deterministic pseudo-random sign based on index and attempt
                        let sign = if (i * 7 + iteration * 13 + (perturb_scale * 1e4) as usize * 3) % 3 == 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        state.x[i] += sign * perturb_scale * mag;
                        if state.x_l[i].is_finite() {
                            state.x[i] = state.x[i].max(state.x_l[i] + 1e-14);
                        }
                        if state.x_u[i].is_finite() {
                            state.x[i] = state.x[i].min(state.x_u[i] - 1e-14);
                        }
                    }
                    // Re-initialize bound multipliers after perturbation
                    for i in 0..n {
                        if state.x_l[i].is_finite() {
                            let slack = (state.x[i] - state.x_l[i]).max(1e-20);
                            state.z_l[i] = state.mu / slack;
                        }
                        if state.x_u[i].is_finite() {
                            let slack = (state.x_u[i] - state.x[i]).max(1e-20);
                            state.z_u[i] = state.mu / slack;
                        }
                    }
                    let pert_eval_ok = state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode);
        if let Some(ref mut lbfgs) = lbfgs_state {
            let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
            );
            lbfgs.update(&state.x, &lag_grad);
            lbfgs.fill_hessian(&mut state.hess_vals);
        }
                    if pert_eval_ok && !state.obj.is_nan() && !state.obj.is_infinite()
                        && !state.grad_f.iter().any(|v| v.is_nan() || v.is_infinite())
                    {
                        // Re-try factorization at perturbed point
                        let sigma_p = kkt::compute_sigma(
                            &state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u,
                        );
                        let mut kkt_p = kkt::assemble_kkt(
                            n, m, &state.hess_rows, &state.hess_cols, &state.hess_vals,
                            &state.jac_rows, &state.jac_cols, &state.jac_vals, &sigma_p,
                            &state.grad_f, &state.g, &state.g_l, &state.g_u,
                            &state.y, &state.z_l, &state.z_u,
                            &state.x, &state.x_l, &state.x_u, state.mu,
                            use_sparse, &state.v_l, &state.v_u,
                        );
                        if kkt::factor_with_inertia_correction(
                            &mut kkt_p, lin_solver.as_mut(), &mut inertia_params,
                        ).is_ok() {
                            log::debug!(
                                "Early perturbation (scale={:.0e}) recovered factorization at iter {}",
                                perturb_scale, iteration
                            );
                            filter.reset();
                            let theta_p = state.constraint_violation();
                            filter.set_theta_min_from_initial(theta_p);
                            early_recovered = true;
                            break;
                        }
                    }
                    // Restore if this perturbation didn't help
                    state.x.copy_from_slice(&x_saved);
                }
                if early_recovered {
                    continue;
                }
            }

            // Try gradient descent fallback
            if let Some(fallback) = gradient_descent_fallback(&state) {
                state.dx = fallback.0;
                state.dy = fallback.1;
                state.dz_l = vec![0.0; n];
                state.dz_u = vec![0.0; n];

                // Simple Armijo backtracking with the gradient step
                let mut alpha_fb = 1.0;
                let obj_current = state.obj;
                let mut fb_accepted = false;
                for _ in 0..20 {
                    let mut x_trial = vec![0.0; n];
                    for i in 0..n {
                        x_trial[i] = state.x[i] + alpha_fb * state.dx[i];
                        if state.x_l[i].is_finite() {
                            x_trial[i] = x_trial[i].max(state.x_l[i] + 1e-14);
                        }
                        if state.x_u[i].is_finite() {
                            x_trial[i] = x_trial[i].min(state.x_u[i] - 1e-14);
                        }
                    }
                    let mut obj_trial = f64::INFINITY;
                    let obj_ok = problem.objective(&x_trial, true, &mut obj_trial);
                    if obj_ok && !obj_trial.is_nan() && obj_trial < obj_current {
                        state.x = x_trial;
                        state.obj = obj_trial;
                        state.alpha_primal = alpha_fb;
                        fb_accepted = true;
                        break;
                    }
                    alpha_fb *= 0.5;
                }
                if fb_accepted {
                    let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode);
        if let Some(ref mut lbfgs) = lbfgs_state {
            let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
            );
            lbfgs.update(&state.x, &lag_grad);
            lbfgs.fill_hessian(&mut state.hess_vals);
        }
                    continue;
                }
            }
            // Try restoration instead of giving up
            let (x_rest, success) = restoration.restore(
                &state.x, &state.x_l, &state.x_u, &state.g_l, &state.g_u,
                &state.jac_rows, &state.jac_cols, n, m, options,
                &|theta, phi| filter.is_acceptable(theta, phi),
                &|x_eval, g_out| problem.constraints(x_eval, true, g_out),
                &|x_eval, jac_out| problem.jacobian_values(x_eval, true, jac_out),
                Some(&|x_eval: &[f64], obj_out: &mut f64| problem.objective(x_eval, true, obj_out)),
                deadline,
            );
            if success {
                state.x = x_rest;
                state.alpha_primal = 0.0;
                let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode);
        if let Some(ref mut lbfgs) = lbfgs_state {
            let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
            );
            lbfgs.update(&state.x, &lag_grad);
            lbfgs.fill_hessian(&mut state.hess_vals);
        }
                continue;
            }
            // Last resort: perturb x and retry factorization
            let mut recovered_from_perturb = false;
            for &perturb_scale in &[1e-3, 1e-2, 1e-1] {
                for i in 0..n {
                    let mag = state.x[i].abs().max(1.0);
                    let sign = if (i * 7 + iteration * 13) % 3 == 0 { -1.0 } else { 1.0 };
                    state.x[i] += sign * perturb_scale * mag;
                    if state.x_l[i].is_finite() {
                        state.x[i] = state.x[i].max(state.x_l[i] + 1e-14);
                    }
                    if state.x_u[i].is_finite() {
                        state.x[i] = state.x[i].min(state.x_u[i] - 1e-14);
                    }
                }
                let pert2_ok = state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode);
        if let Some(ref mut lbfgs) = lbfgs_state {
            let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
            );
            lbfgs.update(&state.x, &lag_grad);
            lbfgs.fill_hessian(&mut state.hess_vals);
        }
                if pert2_ok && !state.obj.is_nan() && !state.obj.is_infinite() {
                    recovered_from_perturb = true;
                    break;
                }
            }
            if recovered_from_perturb {
                filter.reset();
                let theta_new = state.constraint_violation();
                filter.set_theta_min_from_initial(theta_new);
                continue;
            }
            return make_result(&state, SolveStatus::NumericalError);
        }
        } // end: if let Some(ref mut kkt_system) = kkt_system_opt

        // Solve for search direction
        let t_dir = Instant::now();
        let mut cond_solver_for_soc: Option<DenseLdl> = None;
        let (dx, dy) = if let Some(ref cond) = condensed_system {
            // Try condensed solve first (faster for m >> n)
            let mut cond_solver = DenseLdl::new();
            let cond_ok = cond_solver.bunch_kaufman_factor(&cond.matrix).is_ok();
            let cond_result = if cond_ok {
                kkt::solve_condensed(cond, &mut cond_solver).ok()
            } else {
                None
            };

            if let Some(d) = cond_result {
                cond_solver_for_soc = Some(cond_solver);
                d
            } else {
                // Condensed failed — build full KKT on demand
                let mut kkt = kkt::assemble_kkt(
                    n, m,
                    &state.hess_rows, &state.hess_cols, &state.hess_vals,
                    &state.jac_rows, &state.jac_cols, &state.jac_vals,
                    &sigma, &state.grad_f, &state.g, &state.g_l, &state.g_u,
                    &state.y, &state.z_l, &state.z_u,
                    &state.x, &state.x_l, &state.x_u, state.mu,
                    use_sparse, &state.v_l, &state.v_u,
                );
                let fb_ic = kkt::factor_with_inertia_correction(
                    &mut kkt, lin_solver.as_mut(), &mut inertia_params,
                );
                if fb_ic.is_err() {
                    let (x_rest, success) = restoration.restore(
                        &state.x, &state.x_l, &state.x_u, &state.g_l, &state.g_u,
                        &state.jac_rows, &state.jac_cols, n, m, options,
                        &|theta, phi| filter.is_acceptable(theta, phi),
                        &|x_eval, g_out| problem.constraints(x_eval, true, g_out),
                        &|x_eval, jac_out| problem.jacobian_values(x_eval, true, jac_out),
                        Some(&|x_eval: &[f64], obj_out: &mut f64| problem.objective(x_eval, true, obj_out)),
                        deadline,
                    );
                    if success {
                        state.x = x_rest;
                        state.alpha_primal = 0.0;
                        let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode);
        if let Some(ref mut lbfgs) = lbfgs_state {
            let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
            );
            lbfgs.update(&state.x, &lag_grad);
            lbfgs.fill_hessian(&mut state.hess_vals);
        }
                        continue;
                    }
                    return make_result(&state, SolveStatus::NumericalError);
                }
                let (fb_dw, fb_dc) = fb_ic.unwrap();
                match kkt::solve_for_direction(&kkt, lin_solver.as_mut(), fb_dw, fb_dc) {
                    Ok(d) => {
                        let _ = (fb_dw, fb_dc); // used by main path's solve_for_direction
                        kkt_system_opt = Some(kkt);
                        d
                    },
                    Err(e) => {
                        log::warn!("KKT solve failed: {}", e);
                        let (x_rest, success) = restoration.restore(
                            &state.x, &state.x_l, &state.x_u, &state.g_l, &state.g_u,
                            &state.jac_rows, &state.jac_cols, n, m, options,
                            &|theta, phi| filter.is_acceptable(theta, phi),
                            &|x_eval, g_out| problem.constraints(x_eval, true, g_out),
                            &|x_eval, jac_out| problem.jacobian_values(x_eval, true, jac_out),
                            Some(&|x_eval: &[f64], obj_out: &mut f64| problem.objective(x_eval, true, obj_out)),
                            deadline,
                        );
                        if success {
                            state.x = x_rest;
                            state.alpha_primal = 0.0;
                            let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode);
        if let Some(ref mut lbfgs) = lbfgs_state {
            let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
            );
            lbfgs.update(&state.x, &lag_grad);
            lbfgs.fill_hessian(&mut state.hess_vals);
        }
                            continue;
                        }
                        return make_result(&state, SolveStatus::NumericalError);
                    }
                }
            }
        } else if let Some(ref sc) = sparse_condensed_system {
            // Sparse condensed path: factor S = H + Σ + J^T·D_c^{-1}·J with banded/sparse solver
            let kkt_sc = KktMatrix::Sparse(sc.matrix.clone());
            let factor_ok = lin_solver.factor(&kkt_sc).is_ok();
            if factor_ok {
                match kkt::solve_sparse_condensed(sc, lin_solver.as_mut()) {
                    Ok(d) => d,
                    Err(_) => {
                        // Fall back to full KKT
                        let mut kkt = kkt::assemble_kkt(
                            n, m,
                            &state.hess_rows, &state.hess_cols, &state.hess_vals,
                            &state.jac_rows, &state.jac_cols, &state.jac_vals,
                            &sigma, &state.grad_f, &state.g, &state.g_l, &state.g_u,
                            &state.y, &state.z_l, &state.z_u,
                            &state.x, &state.x_l, &state.x_u, state.mu,
                            use_sparse, &state.v_l, &state.v_u,
                        );
                        let mut fallback_solver = new_fallback_solver(use_sparse);
                        if let Ok((fb_dw, fb_dc)) = kkt::factor_with_inertia_correction(
                            &mut kkt, fallback_solver.as_mut(), &mut inertia_params,
                        ) {
                            kkt::solve_for_direction(&kkt, fallback_solver.as_mut(), fb_dw, fb_dc)
                                .unwrap_or_else(|_| gradient_descent_fallback(&state)
                                    .unwrap_or_else(|| (vec![0.0; n], vec![0.0; m])))
                        } else {
                            gradient_descent_fallback(&state)
                                .unwrap_or_else(|| (vec![0.0; n], vec![0.0; m]))
                        }
                    }
                }
            } else {
                // Factor failed — try full KKT
                let mut kkt = kkt::assemble_kkt(
                    n, m,
                    &state.hess_rows, &state.hess_cols, &state.hess_vals,
                    &state.jac_rows, &state.jac_cols, &state.jac_vals,
                    &sigma, &state.grad_f, &state.g, &state.g_l, &state.g_u,
                    &state.y, &state.z_l, &state.z_u,
                    &state.x, &state.x_l, &state.x_u, state.mu,
                    use_sparse, &state.v_l, &state.v_u,
                );
                let mut fallback_solver = new_fallback_solver(use_sparse);
                if let Ok((fb_dw, fb_dc)) = kkt::factor_with_inertia_correction(
                    &mut kkt, fallback_solver.as_mut(), &mut inertia_params,
                ) {
                    kkt::solve_for_direction(&kkt, fallback_solver.as_mut(), fb_dw, fb_dc)
                        .unwrap_or_else(|_| gradient_descent_fallback(&state)
                            .unwrap_or_else(|| (vec![0.0; n], vec![0.0; m])))
                } else {
                    gradient_descent_fallback(&state)
                        .unwrap_or_else(|| (vec![0.0; n], vec![0.0; m]))
                }
            }
        } else {
            // Save original RHS for Mehrotra PC deflection check
            let saved_rhs = if options.mehrotra_pc {
                kkt_system_opt.as_ref().map(|k| k.rhs.clone())
            } else {
                None
            };
            let mut mehrotra_applied = false;

            // Mehrotra predictor-corrector (PC): probe the affine-scaling direction to
            // estimate a better barrier parameter μ before solving the main step.
            //
            // Algorithm:
            //   1. Solve affine predictor (μ=0 in RHS) — same factored matrix, new RHS.
            //   2. Compute max step α_aff for the predictor using fraction-to-boundary.
            //   3. Compute μ_aff = average complementarity after the affine step.
            //   4. Set σ = (μ_aff / μ)³  (centering parameter).
            //   5. Update KKT RHS to use μ_new = σ·μ (≤ μ → more aggressive decrease).
            //   6. Solve the main corrector with the improved RHS.
            //
            // Cost: one extra triangular solve (no re-factorization).
            // Reference: Mehrotra (1992, SIAM J. Optim.); Nocedal/Wächter (2006).
            if options.mehrotra_pc {
                let has_bounds = (0..n).any(|i| state.x_l[i].is_finite() || state.x_u[i].is_finite());
                if has_bounds {
                    // Scope the immutable borrow so it ends before the mutable update below.
                    let pc_rhs: Option<Vec<f64>> = {
                        let kkt = kkt_system_opt.as_ref().unwrap();
                        let rhs_aff = kkt::affine_predictor_rhs(
                            &kkt.rhs, &state.x, &state.x_l, &state.x_u, state.mu,
                        );
                        if let Ok((dx_aff, _)) = kkt::solve_with_custom_rhs(
                            kkt.n, kkt.dim, lin_solver.as_mut(), &rhs_aff,
                        ) {
                            // Complementarity steps for the affine predictor (μ=0)
                            let (dz_l_aff, dz_u_aff) = kkt::recover_dz(
                                &state.x, &state.x_l, &state.x_u,
                                &state.z_l, &state.z_u, &dx_aff, 0.0,
                            );
                            // Compute α_aff = max step along affine direction
                            let tau_aff = 1.0 - 1e-3;
                            let aff_zl = filter::fraction_to_boundary(&state.z_l, &dz_l_aff, tau_aff);
                            let aff_zu = filter::fraction_to_boundary(&state.z_u, &dz_u_aff, tau_aff);
                            let mut alpha_aff = aff_zl.min(aff_zu).min(1.0);
                            for i in 0..n {
                                if state.x_l[i].is_finite() && dx_aff[i] < 0.0 {
                                    let s = (state.x[i] - state.x_l[i]).max(1e-20);
                                    alpha_aff = alpha_aff.min(tau_aff * s / (-dx_aff[i]));
                                }
                                if state.x_u[i].is_finite() && dx_aff[i] > 0.0 {
                                    let s = (state.x_u[i] - state.x[i]).max(1e-20);
                                    alpha_aff = alpha_aff.min(tau_aff * s / dx_aff[i]);
                                }
                            }
                            alpha_aff = alpha_aff.clamp(0.0, 1.0);
                            // Compute μ_aff = average complementarity after affine step
                            let mut mu_aff_sum = 0.0_f64;
                            let mut nb: usize = 0;
                            for i in 0..n {
                                if state.x_l[i].is_finite() {
                                    let s = (state.x[i] + alpha_aff * dx_aff[i]
                                        - state.x_l[i]).max(1e-20);
                                    let z = (state.z_l[i] + alpha_aff * dz_l_aff[i]).max(1e-20);
                                    mu_aff_sum += s * z;
                                    nb += 1;
                                }
                                if state.x_u[i].is_finite() {
                                    let s = (state.x_u[i] - state.x[i]
                                        - alpha_aff * dx_aff[i]).max(1e-20);
                                    let z = (state.z_u[i] + alpha_aff * dz_u_aff[i]).max(1e-20);
                                    mu_aff_sum += s * z;
                                    nb += 1;
                                }
                            }
                            if nb > 0 {
                                let mu_aff = mu_aff_sum / nb as f64;
                                let sigma = (mu_aff / state.mu).powi(3).clamp(0.0, 1.0);
                                // Store sigma for the cross-iteration mu update
                                last_mehrotra_sigma = Some(sigma);
                                let mu_pc = (sigma * state.mu).max(options.mu_min);
                                // Apply PC only when the centering parameter suggests
                                // a meaningful μ decrease (σ < 0.95) and the probe is
                                // not degenerate. Skip early iterations to avoid
                                // amplifying noise at poorly-scaled starting points.
                                // Also skip when sigma is very small (near convergence,
                                // α_aff → 1) to avoid over-aggressive barrier decrease.
                                let sigma_skip_min = 0.05_f64;
                                if mu_pc < state.mu * 0.95 && sigma >= sigma_skip_min && iteration >= 2 {
                                    log::debug!(
                                        "Mehrotra PC iter {}: σ={:.4} α_aff={:.4} μ: {:.2e}→{:.2e}",
                                        iteration, sigma, alpha_aff, state.mu, mu_pc
                                    );
                                    Some(kkt::rebuild_rhs_with_mu(
                                        &kkt.rhs, &state.x, &state.x_l, &state.x_u,
                                        state.mu, mu_pc,
                                    ))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }; // immutable borrow of kkt_system_opt ends here

                    // Apply the improved RHS to the KKT system
                    if let Some(new_rhs) = pc_rhs {
                        kkt_system_opt.as_mut().unwrap().rhs = new_rhs;
                        mehrotra_applied = true;
                    }
                }
            }

            let dir_result = kkt::solve_for_direction(kkt_system_opt.as_ref().unwrap(), lin_solver.as_mut(), ic_delta_w, ic_delta_c);
            let (mut dx_dir, mut dy_dir) = match dir_result {
                Ok(d) => d,
                Err(e) => {
                    log::warn!("KKT solve failed: {}", e);
                    // Try gradient descent fallback before restoration
                    if let Some(fallback) = gradient_descent_fallback(&state) {
                        fallback
                    } else {
                        let (x_rest, success) = restoration.restore(
                            &state.x, &state.x_l, &state.x_u, &state.g_l, &state.g_u,
                            &state.jac_rows, &state.jac_cols, n, m, options,
                            &|theta, phi| filter.is_acceptable(theta, phi),
                            &|x_eval, g_out| problem.constraints(x_eval, true, g_out),
                            &|x_eval, jac_out| problem.jacobian_values(x_eval, true, jac_out),
                            Some(&|x_eval: &[f64], obj_out: &mut f64| problem.objective(x_eval, true, obj_out)),
                            deadline,
                        );
                        if success {
                            state.x = x_rest;
                            state.alpha_primal = 0.0;
                            let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode);
        if let Some(ref mut lbfgs) = lbfgs_state {
            let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
            );
            lbfgs.update(&state.x, &lag_grad);
            lbfgs.fill_hessian(&mut state.hess_vals);
        }
                            continue;
                        }
                        return make_result(&state, SolveStatus::NumericalError);
                    }
                }
            };

            // Mehrotra PC deflection check: if the PC direction deflects > 30% from
            // the original (non-PC) direction, revert to the original direction.
            if mehrotra_applied {
                if let Some(ref orig_rhs) = saved_rhs {
                    if let Some(ref kkt) = kkt_system_opt {
                        if let Ok((dx_orig, dy_orig)) = kkt::solve_with_custom_rhs(kkt.n, kkt.dim, lin_solver.as_mut(), orig_rhs) {
                            let norm_orig: f64 = dx_orig.iter().map(|v| v * v).sum::<f64>().sqrt();
                            let norm_pc: f64 = dx_dir.iter().map(|v| v * v).sum::<f64>().sqrt();
                            if norm_orig > 1e-30 && norm_pc > 1e-30 {
                                let dot: f64 = dx_orig.iter().zip(dx_dir.iter()).map(|(a, b)| a * b).sum::<f64>();
                                let cos_angle = dot / (norm_orig * norm_pc);
                                if cos_angle < 0.7 {
                                    log::debug!(
                                        "Mehrotra PC deflection too large (cos={:.3}), reverting",
                                        cos_angle
                                    );
                                    dx_dir = dx_orig;
                                    dy_dir = dy_orig;
                                    // Restore original RHS for Gondzio MCC
                                    kkt_system_opt.as_mut().unwrap().rhs = orig_rhs.clone();
                                }
                            }
                        }
                    }
                }
            }

            (dx_dir, dy_dir)
        };

        timings.direction_solve += t_dir.elapsed();

        // Recover bound multiplier steps
        let (dz_l, dz_u) =
            kkt::recover_dz(&state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u, &dx, state.mu);

        state.dx = dx;
        state.dy = dy;
        state.dz_l = dz_l;
        state.dz_u = dz_u;

        // Gondzio multiple centrality corrections (MCC).
        //
        // After computing the main search direction (possibly Mehrotra-corrected),
        // perform up to `gondzio_mcc_max` additional centrality corrections.
        // Each correction uses the SAME factored KKT matrix (one extra backsolve each)
        // to drive complementarity pairs that are far from μ back toward the central path.
        //
        // Acceptance criterion: the correction is accepted only if it does not reduce
        // the maximum step length by more than 10%.
        //
        // Reference: Gondzio (1994, Comput. Optim. Appl.); Gondzio (2007).
        if options.gondzio_mcc_max > 0 {
            if let Some(ref kkt) = kkt_system_opt {
                // Compute a preliminary max step for the current direction
                let tau_mcc = if mu_state.mode == MuMode::Free {
                    let nlp_error = primal_inf + dual_inf + compl_inf_best;
                    (1.0 - nlp_error).max(options.tau_min)
                } else {
                    (1.0 - state.mu).max(options.tau_min)
                };
                let mcc_zl = filter::fraction_to_boundary(&state.z_l, &state.dz_l, tau_mcc);
                let mcc_zu = filter::fraction_to_boundary(&state.z_u, &state.dz_u, tau_mcc);
                let mut alpha_mcc = mcc_zl.min(mcc_zu).min(1.0);
                for i in 0..n {
                    if state.x_l[i].is_finite() && state.dx[i] < 0.0 {
                        let s = state.x[i] - state.x_l[i];
                        alpha_mcc = alpha_mcc.min(tau_mcc * s / (-state.dx[i]));
                    }
                    if state.x_u[i].is_finite() && state.dx[i] > 0.0 {
                        let s = state.x_u[i] - state.x[i];
                        alpha_mcc = alpha_mcc.min(tau_mcc * s / state.dx[i]);
                    }
                }
                alpha_mcc = alpha_mcc.clamp(0.0, 1.0);

                let mu_target = state.mu;
                let beta_min = 0.01_f64;  // centrality lower bound: z·s ≥ β_min·μ
                let beta_max = 100.0_f64; // centrality upper bound: z·s ≤ β_max·μ

                // Save original direction norm for deflection check
                let dx_norm_orig: f64 = state.dx.iter().map(|v| v * v).sum::<f64>().sqrt();

                for _mcc_iter in 0..options.gondzio_mcc_max {
                    // Build centrality correction RHS: target z·s → μ for outliers
                    let mut rhs_mcc = vec![0.0_f64; kkt.dim];
                    let mut needs_correction = false;

                    for i in 0..n {
                        if state.x_l[i].is_finite() {
                            let s_t = (state.x[i] + alpha_mcc * state.dx[i]
                                - state.x_l[i]).max(1e-20);
                            let z_t = (state.z_l[i] + alpha_mcc * state.dz_l[i]).max(1e-20);
                            let c = z_t * s_t;
                            if c < beta_min * mu_target || c > beta_max * mu_target {
                                rhs_mcc[i] += (mu_target - c) / s_t;
                                needs_correction = true;
                            }
                        }
                        if state.x_u[i].is_finite() {
                            let s_t = (state.x_u[i] - state.x[i]
                                - alpha_mcc * state.dx[i]).max(1e-20);
                            let z_t = (state.z_u[i] + alpha_mcc * state.dz_u[i]).max(1e-20);
                            let c = z_t * s_t;
                            if c < beta_min * mu_target || c > beta_max * mu_target {
                                rhs_mcc[i] -= (mu_target - c) / s_t;
                                needs_correction = true;
                            }
                        }
                    }

                    if !needs_correction {
                        break;
                    }

                    // Solve for the centrality correction direction
                    match kkt::solve_with_custom_rhs(kkt.n, kkt.dim, lin_solver.as_mut(), &rhs_mcc) {
                        Ok((ddx, ddy)) => {
                            // Compute bound-multiplier corrections from the Newton step:
                            //   S_l · ddz_l + Z_l · ddx = 0  (no centering in correction)
                            //   ddz_l[i] = -(z_l[i] / s_l[i]) * ddx[i]
                            //   ddz_u[i] =  (z_u[i] / s_u[i]) * ddx[i]
                            // NOTE: do NOT use recover_dz(mu=0) here — that adds the
                            // affine centering term (-z_l[i]) which would drive z_l to zero.
                            let mut ddz_l = vec![0.0_f64; n];
                            let mut ddz_u = vec![0.0_f64; n];
                            for i in 0..n {
                                if state.x_l[i].is_finite() {
                                    let s_l = (state.x[i] - state.x_l[i]).max(1e-20);
                                    ddz_l[i] = -(state.z_l[i] / s_l) * ddx[i];
                                }
                                if state.x_u[i].is_finite() {
                                    let s_u = (state.x_u[i] - state.x[i]).max(1e-20);
                                    ddz_u[i] = (state.z_u[i] / s_u) * ddx[i];
                                }
                            }

                            // Tentatively update direction
                            let mut dx_c: Vec<f64> = state.dx.iter().zip(ddx.iter()).map(|(a, b)| a + b).collect();
                            let mut dy_c: Vec<f64> = state.dy.iter().zip(ddy.iter()).map(|(a, b)| a + b).collect();
                            let mut dz_l_c: Vec<f64> = state.dz_l.iter().zip(ddz_l.iter()).map(|(a, b)| a + b).collect();
                            let mut dz_u_c: Vec<f64> = state.dz_u.iter().zip(ddz_u.iter()).map(|(a, b)| a + b).collect();

                            // Deflection check: if correction deflects direction > 30%,
                            // dampen to prevent basin-switching on nonconvex problems
                            if dx_norm_orig > 1e-30 {
                                let dx_c_norm: f64 = dx_c.iter().map(|v| v * v).sum::<f64>().sqrt();
                                if dx_c_norm > 1e-30 {
                                    let dot: f64 = state.dx.iter().zip(dx_c.iter()).map(|(a, b)| a * b).sum::<f64>();
                                    let cos_angle = dot / (dx_norm_orig * dx_c_norm);
                                    if cos_angle < 0.7 {
                                        let alpha_damp = 0.3;
                                        for i in 0..n { dx_c[i] = (1.0 - alpha_damp) * state.dx[i] + alpha_damp * dx_c[i]; }
                                        for i in 0..m { dy_c[i] = (1.0 - alpha_damp) * state.dy[i] + alpha_damp * dy_c[i]; }
                                        for i in 0..n { dz_l_c[i] = (1.0 - alpha_damp) * state.dz_l[i] + alpha_damp * dz_l_c[i]; }
                                        for i in 0..n { dz_u_c[i] = (1.0 - alpha_damp) * state.dz_u[i] + alpha_damp * dz_u_c[i]; }
                                        log::debug!(
                                            "Gondzio MCC iter {}: dampened correction (cos={:.3})",
                                            iteration, cos_angle
                                        );
                                    }
                                }
                            }

                            // Compute new alpha for the corrected direction
                            let new_zl = filter::fraction_to_boundary(&state.z_l, &dz_l_c, tau_mcc);
                            let new_zu = filter::fraction_to_boundary(&state.z_u, &dz_u_c, tau_mcc);
                            let mut alpha_new = new_zl.min(new_zu).min(1.0);
                            for i in 0..n {
                                if state.x_l[i].is_finite() && dx_c[i] < 0.0 {
                                    let s = state.x[i] - state.x_l[i];
                                    alpha_new = alpha_new.min(tau_mcc * s / (-dx_c[i]));
                                }
                                if state.x_u[i].is_finite() && dx_c[i] > 0.0 {
                                    let s = state.x_u[i] - state.x[i];
                                    alpha_new = alpha_new.min(tau_mcc * s / dx_c[i]);
                                }
                            }
                            alpha_new = alpha_new.clamp(0.0, 1.0);

                            // Accept only if the correction doesn't shrink alpha by >10%
                            if alpha_new >= 0.9 * alpha_mcc {
                                // dx_c needs explicit type annotation for vec addition
                                let _ = &mut dx_c; // silence unused_mut warning
                                state.dx = dx_c;
                                state.dy = dy_c;
                                state.dz_l = dz_l_c;
                                state.dz_u = dz_u_c;
                                alpha_mcc = alpha_new;
                                log::debug!(
                                    "Gondzio MCC iter {}: correction accepted, α_mcc={:.4}",
                                    iteration, alpha_mcc
                                );
                            } else {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
            }
        }

        // Compute maximum step sizes using fraction-to-boundary rule.
        // Free mode: tau based on NLP error. Fixed mode: tau based on mu.
        let tau = if mu_state.mode == MuMode::Free {
            let nlp_error = primal_inf + dual_inf + compl_inf_best;
            (1.0 - nlp_error).max(options.tau_min)
        } else {
            (1.0 - state.mu).max(options.tau_min)
        };

        // Primal step: ensure x + alpha*dx stays within variable bounds
        let mut alpha_primal_max: f64 = 1.0;
        for i in 0..n {
            if state.x_l[i].is_finite() && state.dx[i] < 0.0 {
                let slack = state.x[i] - state.x_l[i];
                let ratio = -tau * slack / state.dx[i];
                alpha_primal_max = alpha_primal_max.min(ratio);
            }
            if state.x_u[i].is_finite() && state.dx[i] > 0.0 {
                let slack = state.x_u[i] - state.x[i];
                let ratio = tau * slack / state.dx[i];
                alpha_primal_max = alpha_primal_max.min(ratio);
            }
        }

        alpha_primal_max = alpha_primal_max.clamp(0.0, 1.0);

        // Dual step: ensure z + alpha*dz > 0
        let alpha_dual_max_l = filter::fraction_to_boundary(&state.z_l, &state.dz_l, tau);
        let alpha_dual_max_u = filter::fraction_to_boundary(&state.z_u, &state.dz_u, tau);
        let alpha_dual_max = alpha_dual_max_l.min(alpha_dual_max_u);

        // Ipopt-like tiny step detection: if relative step size is < 10*eps for
        // 2 consecutive iterations, force mu decrease and accept the full step.
        {
            let max_rel_step: f64 = (0..n)
                .map(|i| (alpha_primal_max * state.dx[i]).abs() / (state.x[i].abs() + 1.0))
                .fold(0.0f64, f64::max);
            if max_rel_step < 1e-14 && primal_inf < 1e-4 {
                consecutive_tiny_steps += 1;
                mu_state.tiny_step = true;
                if consecutive_tiny_steps >= 2 {
                    // Force mu decrease (Ipopt: monotone decrease on tiny step)
                    let new_mu = (options.mu_linear_decrease_factor * state.mu)
                        .min(state.mu.powf(options.mu_superlinear_decrease_power))
                        .max(options.mu_min);
                    if (new_mu - state.mu).abs() < 1e-20 {
                        log::debug!("Tiny step with mu at minimum, checking acceptability");
                    } else {
                        state.mu = new_mu;
                        filter.reset();
                        let theta_new = state.constraint_violation();
                        filter.set_theta_min_from_initial(theta_new);
                        log::debug!("Tiny step detected, forced mu decrease to {:.2e}", state.mu);
                    }
                    consecutive_tiny_steps = 0;
                }
            } else {
                consecutive_tiny_steps = 0;
                mu_state.tiny_step = false;
            }
        }

        // Line search
        let t_ls = Instant::now();
        let theta_current = primal_inf;
        let phi_current = state.barrier_objective(options);
        let grad_phi_step = state.barrier_directional_derivative(options);

        let mut alpha = alpha_primal_max;
        let mut step_accepted = false;
        let min_alpha = filter.compute_alpha_min(theta_current, grad_phi_step);

        // If primal divergence was detected, skip line search to force restoration entry.
        // The !step_accepted branch below handles filter updates and restoration.
        // No-op here; just skip the line search loop.

        ls_steps = 0; // reset backtrack counter for this iteration
        for _ls_iter in 0..40 {
            if force_restoration {
                break;
            }
            // Intra-iteration early stall check (scaled by problem size)
            if iteration < 3 && options.early_stall_timeout > 0.0 {
                if start_time.elapsed().as_secs_f64() > early_timeout {
                    return make_result(&state, SolveStatus::NumericalError);
                }
            }
            if alpha < min_alpha {
                break;
            }

            // Compute trial point
            let mut x_trial = vec![0.0; n];
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                x_trial[i] = state.x[i] + alpha * state.dx[i];
                // Safeguard: ensure strictly within bounds
                if state.x_l[i].is_finite() {
                    x_trial[i] = x_trial[i].max(state.x_l[i] + 1e-14);
                }
                if state.x_u[i].is_finite() {
                    x_trial[i] = x_trial[i].min(state.x_u[i] - 1e-14);
                }
            }

            // Evaluate at trial point
            let mut obj_trial = f64::INFINITY;
            let obj_ok = problem.objective(&x_trial, true, &mut obj_trial);
            let mut g_trial = vec![0.0; m];
            let constr_ok = if m > 0 {
                problem.constraints(&x_trial, true, &mut g_trial)
            } else {
                true
            };

            // NaN guard: reject trial points with NaN/Inf values or eval failures
            if !obj_ok || !constr_ok || obj_trial.is_nan() || obj_trial.is_infinite()
                || g_trial.iter().any(|v| v.is_nan() || v.is_infinite())
            {
                alpha *= 0.5;
                ls_steps += 1;
                continue;
            }

            let theta_trial =
                convergence::primal_infeasibility(&g_trial, &state.g_l, &state.g_u);

            // Watchdog mode: accept full step unconditionally (bypass filter)
            if watchdog_active && alpha == alpha_primal_max {
                state.x = x_trial;
                state.obj = obj_trial;
                state.g = g_trial;
                state.alpha_primal = alpha;
                step_accepted = true;
                break;
            }

            // Compute barrier objective at trial
            let mut phi_trial = obj_trial;
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                if state.x_l[i].is_finite() {
                    let slack = (x_trial[i] - state.x_l[i]).max(1e-20);
                    phi_trial -= state.mu * slack.ln();
                }
                if state.x_u[i].is_finite() {
                    let slack = (state.x_u[i] - x_trial[i]).max(1e-20);
                    phi_trial -= state.mu * slack.ln();
                }
            }
            if options.constraint_slack_barrier {
                for i in 0..m {
                    let is_eq = state.g_l[i].is_finite() && state.g_u[i].is_finite()
                        && (state.g_l[i] - state.g_u[i]).abs() < 1e-15;
                    if is_eq {
                        continue;
                    }
                    if state.g_l[i].is_finite() {
                        let slack = g_trial[i] - state.g_l[i];
                        if slack > state.mu * 1e-2 {
                            phi_trial -= state.mu * slack.ln();
                        }
                    }
                    if state.g_u[i].is_finite() {
                        let slack = state.g_u[i] - g_trial[i];
                        if slack > state.mu * 1e-2 {
                            phi_trial -= state.mu * slack.ln();
                        }
                    }
                }
            }

            // Check acceptability
            let (acceptable, _used_switching) = filter.check_acceptability(
                theta_current,
                phi_current,
                theta_trial,
                phi_trial,
                grad_phi_step,
                alpha,
            );

            if acceptable {
                // Accept step
                state.x = x_trial;
                state.obj = obj_trial;
                state.g = g_trial;
                state.alpha_primal = alpha;
                step_accepted = true;

                // Add to filter if not using switching condition
                if !_used_switching {
                    filter.add(theta_current, phi_current);
                }
                break;
            }

            // Second-order correction (SOC) — try on every backtracking step where theta increases
            if theta_trial > theta_current && options.max_soc > 0 {
                let soc_accepted = if let (Some(ref cond), Some(ref mut cs)) = (&condensed_system, &mut cond_solver_for_soc) {
                    // Use condensed SOC (avoids building full KKT)
                    attempt_soc_condensed(
                        &state, problem, &g_trial, cs, cond, &filter,
                        theta_current, phi_current, grad_phi_step, alpha, options,
                    )
                } else if let Some(ref sc) = sparse_condensed_system {
                    // Use sparse condensed SOC
                    attempt_soc_sparse_condensed(
                        &state, problem, &g_trial, lin_solver.as_mut(), sc, &filter,
                        theta_current, phi_current, grad_phi_step, alpha, options,
                    )
                } else if let Some(ref kkt) = kkt_system_opt {
                    attempt_soc(
                        &state, problem, &x_trial, &g_trial,
                        lin_solver.as_mut(), kkt, &filter,
                        theta_current, phi_current, grad_phi_step, alpha, options,
                    )
                } else {
                    None
                };

                if let Some((x_soc, obj_soc, g_soc, alpha_soc)) = soc_accepted {
                    state.diagnostics.soc_corrections += 1;
                    state.x = x_soc;
                    state.obj = obj_soc;
                    state.g = g_soc;
                    state.alpha_primal = alpha_soc;
                    step_accepted = true;
                    filter.add(theta_current, phi_current);
                    break;
                }
            }

            // Backtrack
            alpha *= 0.5;
            ls_steps += 1;
        }

        if !step_accepted {
            state.diagnostics.filter_rejects += 1;

            // Add current point to filter before entering restoration (Ipopt convention).
            filter.add(theta_current, phi_current);
            filter.augment_for_restoration(theta_current);

            // Phase 1: Fast GN restoration
            log::debug!("Line search failed at iteration {}, entering restoration", iteration);

            let (x_rest, gn_success) = restoration.restore(
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
            );

            if gn_success {
                state.diagnostics.restoration_count += 1;
                // GN restoration succeeded — apply standard restoration success handling
                apply_restoration_success(
                    &mut state, &mut filter, &mut mu_state, options, n, m, problem, &x_rest,
                    linear_constraints.as_deref(), lbfgs_mode, &mut lbfgs_state,
                );
                continue;
            }

            // GN restoration failed — recovery logic with NLP restoration as last resort
            {
                // Bail out of recovery cascade if wall time is nearly exhausted.
                // Without this, the cascade (especially NLP restoration) can consume
                // remaining time and prevent the outer fallback strategies from running.
                if options.max_wall_time > 0.0 {
                    let remaining = options.max_wall_time - start_time.elapsed().as_secs_f64();
                    if remaining < 1.0 {
                        return make_result(&state, SolveStatus::MaxIterations);
                    }
                }

                mu_state.consecutive_restoration_failures += 1;
                let fail_count = mu_state.consecutive_restoration_failures;

                // At fail_count == 2: try full NLP restoration early.
                // The NLP restoration is the most robust approach (Ipopt's primary method).
                // Try it before exhausting simpler recovery strategies.
                // Skip for large problems: NLP restoration doubles the problem size,
                // making it prohibitively expensive for n+m > 10000.
                let kkt_dim = n + m;
                // Early stall: skip expensive NLP restoration if we've already exceeded timeout
                let skip_nlp_restoration = iteration < 3
                    && options.early_stall_timeout > 0.0
                    && start_time.elapsed().as_secs_f64() > early_timeout * 0.5;
                if (fail_count == 2 || fail_count == 4) && !options.disable_nlp_restoration && kkt_dim <= 50000 && !skip_nlp_restoration {
                    state.diagnostics.nlp_restoration_count += 1;
                    let (x_nlp, outcome) = attempt_nlp_restoration(
                        problem, &state, &filter, options, theta_current, start_time,
                    );
                    match outcome {
                        RestorationOutcome::Success => {
                            apply_restoration_success(
                                &mut state, &mut filter, &mut mu_state, options, n, m,
                                problem, &x_nlp,
                                linear_constraints.as_deref(), lbfgs_mode, &mut lbfgs_state,
                            );
                            continue;
                        }
                        RestorationOutcome::LocalInfeasibility
                        | RestorationOutcome::Failed => {
                            // Fall through to continue recovery.
                            // Don't immediately return LocalInfeasibility — the existing
                            // infeasibility detection at fail_count > 6 uses stationarity
                            // checks which are more reliable.
                        }
                    }
                }

                // For large problems (no NLP restoration), give up sooner
                let max_restore_attempts = if kkt_dim > 50000 { 3 } else { 6 };
                if fail_count > max_restore_attempts {
                    // Exhausted recovery attempts: check infeasibility and give up
                    log::warn!("Restoration failed at iteration {} (attempt #{})", iteration, fail_count);
                    let current_theta = state.constraint_violation();

                    // Check stationarity of violation
                    if current_theta > options.constr_viol_tol && !ever_feasible {
                        let mut violation = vec![0.0; m];
                        for i in 0..m {
                            let is_eq = state.g_l[i].is_finite() && state.g_u[i].is_finite()
                                && (state.g_l[i] - state.g_u[i]).abs() < 1e-15;
                            if is_eq {
                                violation[i] = state.g[i] - state.g_l[i];
                            } else if state.g_l[i].is_finite() && state.g[i] < state.g_l[i] {
                                violation[i] = state.g[i] - state.g_l[i];
                            } else if state.g_u[i].is_finite() && state.g[i] > state.g_u[i] {
                                violation[i] = state.g[i] - state.g_u[i];
                            }
                        }
                        let mut grad_theta = vec![0.0; n];
                        for (idx, (&row, &col)) in
                            state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate()
                        {
                            grad_theta[col] += state.jac_vals[idx] * violation[row];
                        }
                        let grad_theta_norm = grad_theta.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                        let stationarity_tol = 1e-4 * current_theta.max(1.0);
                        if grad_theta_norm < stationarity_tol {
                            log::info!(
                                "Local infeasibility detected: theta={:.2e}, ||∇theta||={:.2e}",
                                current_theta, grad_theta_norm
                            );
                            return make_result(&state, SolveStatus::LocalInfeasibility);
                        }
                    }

                    if !ever_feasible && current_theta > 1e4 && iteration > 500 && theta_history.len() >= theta_history_len {
                        let min_theta = theta_history.iter().cloned().fold(f64::INFINITY, f64::min);
                        if current_theta > 0.01 * min_theta {
                            return make_result(&state, SolveStatus::Infeasible);
                        }
                    }
                    return make_result(&state, SolveStatus::RestorationFailed);
                }

                // Recovery strategies: cycle through mode switches and mu perturbations
                log::debug!("Restoration failed (attempt #{}), trying recovery", fail_count);
                let mu_factors: [f64; 6] = [10.0, 0.1, 100.0, 0.01, 1000.0, 0.001];

                match fail_count {
                    1 => {
                        // First failure: switch mode
                        if mu_state.mode == MuMode::Free {
                            state.diagnostics.mu_mode_switches += 1;
                            mu_state.mode = MuMode::Fixed;
                            mu_state.first_iter_in_mode = true;
                            let avg_compl = compute_avg_complementarity(&state);
                            if avg_compl > 0.0 {
                                state.mu = (options.adaptive_mu_monotone_init_factor * avg_compl)
                                    .clamp(options.mu_min, 1e5);
                            } else {
                                state.mu = (options.mu_linear_decrease_factor * state.mu)
                                    .max(options.mu_min);
                            }
                        } else {
                            // Force mu decrease
                            let new_mu = (options.mu_linear_decrease_factor * state.mu)
                                .min(state.mu.powf(options.mu_superlinear_decrease_power))
                                .max(options.mu_min);
                            state.mu = new_mu;
                        }
                    }
                    _ => {
                        // Subsequent failures: try varied mu perturbation
                        let factor = mu_factors[(fail_count - 2) % mu_factors.len()];
                        state.mu = (state.mu * factor).max(options.mu_min).min(1e5);
                    }
                }
                filter.reset();
                let theta_now = state.constraint_violation();
                filter.set_theta_min_from_initial(theta_now);
                inertia_params.delta_w_last = 0.0;

                // On attempts 3+: also perturb x to escape current basin
                if fail_count >= 3 {
                    for i in 0..n {
                        let range = if state.x_l[i].is_finite() && state.x_u[i].is_finite() {
                            state.x_u[i] - state.x_l[i]
                        } else {
                            state.x[i].abs().max(1.0)
                        };
                        let sign = if (i * 7 + fail_count * 13) % 3 == 0 { -1.0 } else { 1.0 };
                        state.x[i] += sign * 1e-4 * range;
                        if state.x_l[i].is_finite() {
                            state.x[i] = state.x[i].max(state.x_l[i] + 1e-14);
                        }
                        if state.x_u[i].is_finite() {
                            state.x[i] = state.x[i].min(state.x_u[i] - 1e-14);
                        }
                    }
                    let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode);
        if let Some(ref mut lbfgs) = lbfgs_state {
            let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
            );
            lbfgs.update(&state.x, &lag_grad);
            lbfgs.fill_hessian(&mut state.hess_vals);
        }
                }
                continue;
            }
        }

        // Step was accepted — reset consecutive restoration failure counter
        mu_state.consecutive_restoration_failures = 0;

        // Watchdog: track consecutive shortened steps
        if state.alpha_primal < alpha_primal_max * 0.99 {
            consecutive_shortened += 1;
        } else {
            consecutive_shortened = 0;
        }

        // Watchdog activation: save state when too many shortened steps
        if !watchdog_active
            && consecutive_shortened >= options.watchdog_shortened_iter_trigger
        {
            state.diagnostics.watchdog_activations += 1;
            watchdog_active = true;
            watchdog_trial_count = 0;
            let wd_theta = state.constraint_violation();
            let wd_phi = state.barrier_objective(options);
            watchdog_saved = Some(WatchdogSavedState {
                x: state.x.clone(),
                y: state.y.clone(),
                z_l: state.z_l.clone(),
                z_u: state.z_u.clone(),
                v_l: state.v_l.clone(),
                v_u: state.v_u.clone(),
                mu: state.mu,
                obj: state.obj,
                g: state.g.clone(),
                grad_f: state.grad_f.clone(),
                filter_entries: filter.save_entries(),
                theta: wd_theta,
                phi: wd_phi,
            });
            consecutive_shortened = 0;
            log::debug!(
                "Watchdog activated at iteration {} (theta={:.2e}, phi={:.2e})",
                iteration, wd_theta, wd_phi
            );
        }

        // Watchdog progress check
        if watchdog_active {
            watchdog_trial_count += 1;
            if let Some(ref saved) = watchdog_saved {
                let theta_now = state.constraint_violation();
                let phi_now = state.barrier_objective(options);
                // Check if current point is filter-acceptable from saved state
                let made_progress = filter.is_acceptable(theta_now, phi_now)
                    && (theta_now < (1.0 - 1e-5) * saved.theta
                        || phi_now < saved.phi - 1e-5 * saved.theta);

                if made_progress {
                    log::debug!(
                        "Watchdog succeeded at trial {} (theta: {:.2e} -> {:.2e})",
                        watchdog_trial_count, saved.theta, theta_now
                    );
                    watchdog_active = false;
                    watchdog_trial_count = 0;
                    watchdog_saved = None;
                } else if watchdog_trial_count >= options.watchdog_trial_iter_max {
                    // Revert to saved state
                    log::debug!(
                        "Watchdog reverting after {} trials",
                        watchdog_trial_count
                    );
                    // Add explored region to filter before reverting
                    let theta_now = state.constraint_violation();
                    let phi_now = state.barrier_objective(options);
                    filter.restore_entries(saved.filter_entries.clone());
                    filter.add(theta_now, phi_now);

                    state.x = saved.x.clone();
                    state.y = saved.y.clone();
                    state.z_l = saved.z_l.clone();
                    state.z_u = saved.z_u.clone();
                    state.v_l = saved.v_l.clone();
                    state.v_u = saved.v_u.clone();
                    state.mu = saved.mu;
                    state.obj = saved.obj;
                    state.g = saved.g.clone();
                    state.grad_f = saved.grad_f.clone();
                    let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode);
        if let Some(ref mut lbfgs) = lbfgs_state {
            let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
            );
            lbfgs.update(&state.x, &lag_grad);
            lbfgs.fill_hessian(&mut state.hess_vals);
        }

                    watchdog_active = false;
                    watchdog_trial_count = 0;
                    watchdog_saved = None;
                    continue;
                }
            }
        }

        timings.line_search += t_ls.elapsed();

        // Update dual variables (with damping for oscillating components)
        let alpha_d = alpha_dual_max;
        let near_convergence = state.consecutive_acceptable >= 1;
        for i in 0..m {
            let sign_change = if let Some(ref pdy) = prev_dy {
                pdy[i] * state.dy[i] < 0.0
            } else {
                false
            };
            if near_convergence && sign_change {
                dy_sign_change_count[i] = dy_sign_change_count[i].saturating_add(1);
            } else if !sign_change {
                dy_sign_change_count[i] = 0;
            }
            let dy_i = if near_convergence && dy_sign_change_count[i] >= 3 {
                // Persistent oscillation (≥3 consecutive sign changes): damp to stabilize
                0.5 * state.dy[i]
            } else {
                state.dy[i]
            };
            state.y[i] += alpha_d * dy_i;
        }
        prev_dy = Some(state.dy.clone());
        // Ipopt kappa_sigma safeguard: keep z*s in [mu_ks/kappa_sigma, kappa_sigma*mu_ks]
        let kappa_sigma = 1e10;
        let mu_ks = if mu_state.mode == MuMode::Free {
            compute_avg_complementarity(&state)
                .max(state.mu)
                .min(1e3)
        } else {
            state.mu
        };
        for i in 0..n {
            if state.x_l[i].is_finite() {
                let z_new = (state.z_l[i] + alpha_d * state.dz_l[i]).max(1e-20);
                let s_l = (state.x[i] - state.x_l[i]).max(1e-20);
                let z_lo = mu_ks / (kappa_sigma * s_l);
                let z_hi = kappa_sigma * mu_ks / s_l;
                state.z_l[i] = z_new.clamp(z_lo, z_hi);
            }
            if state.x_u[i].is_finite() {
                let z_new = (state.z_u[i] + alpha_d * state.dz_u[i]).max(1e-20);
                let s_u = (state.x_u[i] - state.x[i]).max(1e-20);
                let z_lo = mu_ks / (kappa_sigma * s_u);
                let z_hi = kappa_sigma * mu_ks / s_u;
                state.z_u[i] = z_new.clamp(z_lo, z_hi);
            }
        }

        state.alpha_dual = alpha_d;

        // Re-evaluate at new point
        let t_eval = Instant::now();
        if !state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode) {
            return make_result(&state, SolveStatus::EvaluationError);
        }
        if let Some(ref mut lbfgs) = lbfgs_state {
            let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
            );
            lbfgs.update(&state.x, &lag_grad);
            lbfgs.fill_hessian(&mut state.hess_vals);
        }
        timings.problem_eval += t_eval.elapsed();

        // Reset v_l, v_u from barrier equilibrium v = mu_ks / slack.
        // Simple reset rather than Newton update (our dv is approximate since we
        // lack explicit slacks, and FTB on v can restrict alpha_d too much).
        for i in 0..m {
            if state.v_l[i] > 0.0 && state.g_l[i].is_finite() {
                let slack = (state.g[i] - state.g_l[i]).max(1e-20);
                state.v_l[i] = mu_ks / slack;
            }
            if state.v_u[i] > 0.0 && state.g_u[i].is_finite() {
                let slack = (state.g_u[i] - state.g[i]).max(1e-20);
                state.v_u[i] = mu_ks / slack;
            }
        }

        // NaN/Inf guard on evaluation
        if state.obj.is_nan() || state.obj.is_infinite() {
            // Try restoration from current point
            let (x_rest, success) = restoration.restore(
                &state.x, &state.x_l, &state.x_u, &state.g_l, &state.g_u,
                &state.jac_rows, &state.jac_cols, n, m, options,
                &|theta, phi| filter.is_acceptable(theta, phi),
                &|x_eval, g_out| problem.constraints(x_eval, true, g_out),
                &|x_eval, jac_out| problem.jacobian_values(x_eval, true, jac_out),
                Some(&|x_eval: &[f64], obj_out: &mut f64| problem.objective(x_eval, true, obj_out)),
                deadline,
            );
            if success {
                state.x = x_rest;
                state.alpha_primal = 0.0;
                let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints.as_deref(), lbfgs_mode);
        if let Some(ref mut lbfgs) = lbfgs_state {
            let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
            );
            lbfgs.update(&state.x, &lag_grad);
            lbfgs.fill_hessian(&mut state.hess_vals);
        }
                if !state.obj.is_nan() && !state.obj.is_infinite() {
                    continue;
                }
            }
            return make_result(&state, SolveStatus::NumericalError);
        }

        // Track best feasible point for max_iter exit
        {
            let theta_now = state.constraint_violation();
            if theta_now < options.constr_viol_tol && state.obj < best_obj {
                best_obj = state.obj;
                best_x = Some(state.x.clone());
            }
        }

        // --- Barrier parameter update (free/fixed mode) ---
        // When there are no variable bounds, mu serves no barrier purpose but is
        // still used for KKT regularization and the filter line search. We decrease
        // mu superlinearly (mu^1.5) rather than collapsing it instantly to mu_min,
        // which would destroy filter protection against infeasible steps. This
        // prevents the PENTAGON-type failure where mu=1e-11 at iteration 1 causes
        // the switching condition to accept a step that destroys feasibility.
        let has_bounds = (0..n).any(|i| state.x_l[i].is_finite() || state.x_u[i].is_finite());
        if !has_bounds {
            state.mu = state.mu.powf(options.mu_superlinear_decrease_power).max(options.mu_min);
        } else {
            let kkt_error = {
                let pi = state.constraint_violation();
                let di = convergence::dual_infeasibility(
                    &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
                    &state.y, &state.z_l, &state.z_u, n,
                );
                let ci = convergence::complementarity_error(
                    &state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u, 0.0,
                );
                pi * pi + di * di + ci * ci
            };

            let sufficient = mu_state.check_sufficient_progress(kkt_error);

            match mu_state.mode {
                MuMode::Free => {
                    // Consume Mehrotra sigma for use as quality function candidate
                    let sigma_mu = last_mehrotra_sigma.take();
                    if sufficient && !mu_state.tiny_step {
                        mu_state.consecutive_insufficient = 0;
                        mu_state.remember_accepted(kkt_error);
                        let avg_compl = compute_avg_complementarity(&state);
                        if options.mu_oracle_quality_function && avg_compl > 0.0 {
                            // Quality function: evaluate Q(mu) for explicit candidates and
                            // pick the one with lowest barrier KKT error.
                            let mu_loqo = avg_compl / options.kappa;
                            let mu_linear = options.mu_linear_decrease_factor * state.mu;
                            let mu_third = state.mu / 3.0;
                            let mu_tenth = state.mu / 10.0;

                            // Only allow aggressive decrease if barrier subproblem is solved
                            let barrier_err = compute_barrier_error(&state);
                            let mu_floor = if barrier_err <= options.barrier_tol_factor * state.mu {
                                options.mu_min
                            } else {
                                (state.mu / 5.0).max(options.mu_min)
                            };

                            // Build candidate list, optionally including Mehrotra sigma
                            let mut candidates = vec![mu_loqo, mu_linear, mu_third, mu_tenth];
                            if let Some(sigma) = sigma_mu {
                                candidates.push((sigma * state.mu).max(options.mu_min));
                            }

                            let pi = state.constraint_violation();
                            let di = convergence::dual_infeasibility(
                                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
                                &state.y, &state.z_l, &state.z_u, state.n,
                            );
                            let fixed_q = pi * pi + di * di;

                            let mut best_mu = mu_loqo;
                            let mut best_q = f64::INFINITY;
                            for &mu_c in &candidates {
                                let mu_c = mu_c.clamp(mu_floor, state.mu);
                                let ci = convergence::complementarity_error(
                                    &state.x, &state.x_l, &state.x_u,
                                    &state.z_l, &state.z_u, mu_c,
                                );
                                let q = fixed_q + ci * ci;
                                if q < best_q {
                                    best_q = q;
                                    best_mu = mu_c;
                                }
                            }
                            state.mu = best_mu.clamp(mu_floor, 1e5);
                        } else if avg_compl > 0.0 {
                            // Loqo fallback with rate limit (at most 5x decrease if subproblem not solved)
                            let barrier_err = compute_barrier_error(&state);
                            let mu_floor = if barrier_err <= options.barrier_tol_factor * state.mu {
                                options.mu_min
                            } else {
                                (state.mu / 5.0).max(options.mu_min)
                            };
                            state.mu = (avg_compl / options.kappa).clamp(mu_floor, 1e5);
                        } else {
                            // No complementarity products (all bounds inactive) → decrease mu
                            state.mu = (options.mu_linear_decrease_factor * state.mu)
                                .max(options.mu_min);
                        }
                        // In free mode: reset filter each iteration
                        filter.reset();
                        let theta_new = state.constraint_violation();
                        filter.set_theta_min_from_initial(theta_new);
                    } else {
                        mu_state.consecutive_insufficient += 1;
                        if mu_state.consecutive_insufficient >= 2 {
                            // Switch to fixed mode after 2 consecutive insufficient iterations
                            mu_state.consecutive_insufficient = 0;
                            log::debug!("Switching to fixed mu mode (insufficient progress or tiny step)");
                            state.diagnostics.mu_mode_switches += 1;
                            mu_state.mode = MuMode::Fixed;
                            mu_state.first_iter_in_mode = true;
                            let avg_compl = compute_avg_complementarity(&state);
                            if avg_compl > 0.0 {
                                state.mu = (options.adaptive_mu_monotone_init_factor * avg_compl)
                                    .clamp(options.mu_min, 1e5);
                            } else {
                                state.mu = (options.mu_linear_decrease_factor * state.mu)
                                    .max(options.mu_min);
                            }
                            filter.reset();
                            let theta_new = state.constraint_violation();
                            filter.set_theta_min_from_initial(theta_new);
                        } else {
                            // Stay in Free mode with conservative mu decrease
                            let avg_compl = compute_avg_complementarity(&state);
                            if avg_compl > 0.0 {
                                let barrier_err = compute_barrier_error(&state);
                                let mu_floor = if barrier_err <= options.barrier_tol_factor * state.mu {
                                    options.mu_min
                                } else {
                                    (state.mu / 5.0).max(options.mu_min)
                                };
                                state.mu = (avg_compl / options.kappa).clamp(mu_floor, 1e5);
                            } else {
                                state.mu = (options.mu_linear_decrease_factor * state.mu)
                                    .max(options.mu_min);
                            }
                        }
                    }
                }
                MuMode::Fixed => {
                    if options.mu_strategy_adaptive && sufficient && !mu_state.tiny_step && !mu_state.first_iter_in_mode {
                        // Switch back to free mode (only in adaptive strategy)
                        log::debug!("Switching back to free mu mode (sufficient progress)");
                        state.diagnostics.mu_mode_switches += 1;
                        mu_state.mode = MuMode::Free;
                        mu_state.remember_accepted(kkt_error);
                        mu_state.first_iter_in_mode = true;
                    } else {
                        mu_state.first_iter_in_mode = false;
                        // Check if subproblem is solved (barrier error small enough)
                        let barrier_err = compute_barrier_error(&state);
                        if barrier_err <= options.barrier_tol_factor * state.mu || mu_state.tiny_step {
                            let new_mu = (options.mu_linear_decrease_factor * state.mu)
                                .min(state.mu.powf(options.mu_superlinear_decrease_power))
                                .max(options.mu_min);
                            if !(mu_state.tiny_step && (new_mu - state.mu).abs() < 1e-20) {
                                state.mu = new_mu;
                                filter.reset();
                                let theta_new = state.constraint_violation();
                                filter.set_theta_min_from_initial(theta_new);
                                log::debug!("Fixed mode: mu decreased to {:.2e}", state.mu);
                            }
                        }
                    }
                }
            }
        }

        // Post-step acceptable convergence tracking.
        // This catches cases where the step just taken pushes the state into the
        // acceptable region but the pre-step check at the top of the loop missed it.
        {
            let post_primal = state.constraint_violation();
            let (post_zl_opt, post_zu_opt) = {
                let mut gj = state.grad_f.clone();
                for (idx, (&row, &col)) in
                    state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate()
                {
                    gj[col] += state.jac_vals[idx] * state.y[row];
                }
                let mut zl = vec![0.0; n];
                let mut zu = vec![0.0; n];
                let kc = 1e10;
                for i in 0..n {
                    if gj[i] > 0.0 && state.x_l[i].is_finite() {
                        let sl = (state.x[i] - state.x_l[i]).max(1e-20);
                        if gj[i] * sl <= kc * state.mu.max(1e-20) {
                            zl[i] = gj[i];
                        }
                    } else if gj[i] < 0.0 && state.x_u[i].is_finite() {
                        let su = (state.x_u[i] - state.x[i]).max(1e-20);
                        if (-gj[i]) * su <= kc * state.mu.max(1e-20) {
                            zu[i] = -gj[i];
                        }
                    }
                }
                (zl, zu)
            };
            let post_du = convergence::dual_infeasibility(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
                &state.y, &post_zl_opt, &post_zu_opt, n,
            );
            let post_du_unsc = convergence::dual_infeasibility_scaled(
                &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
                &state.y, &state.z_l, &state.z_u, n,
            );
            let post_compl = convergence::complementarity_error(
                &state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u, 0.0,
            );
            let post_compl_opt = convergence::complementarity_error(
                &state.x, &state.x_l, &state.x_u, &post_zl_opt, &post_zu_opt, 0.0,
            );
            let post_compl_best = post_compl.min(post_compl_opt);
            let post_mult_sum: f64 = state.y.iter().map(|v| v.abs()).sum::<f64>()
                + state.z_l.iter().map(|v| v.abs()).sum::<f64>()
                + state.z_u.iter().map(|v| v.abs()).sum::<f64>();
            let post_sd = if (m + 2 * n) > 0 {
                ((100.0f64.max(post_mult_sum / (m + 2 * n) as f64)) / 100.0).min(1e4)
            } else {
                1.0
            };
            let post_near_scaled = post_primal <= 100.0 * options.tol
                && post_du <= 100.0 * options.tol * post_sd
                && post_compl_best <= 100.0 * options.tol * post_sd;
            let post_near_unscaled = post_primal <= 10.0 * options.constr_viol_tol
                && post_du_unsc <= 10.0 * options.dual_inf_tol
                && post_compl_best <= 10.0 * options.compl_inf_tol;
            if post_near_scaled && post_near_unscaled {
                state.consecutive_acceptable += 1;
            }
            // Don't reset here — the pre-step check handles resets
        }
    }

    // At max_iter: log convergence diagnostics using same z_opt as convergence check (with gate)
    {
        let final_primal_inf = state.constraint_violation();
        let (z_l_opt_final, z_u_opt_final) = {
            let mut grad_jty = state.grad_f.clone();
            for (idx, (&row, &col)) in
                state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate()
            {
                grad_jty[col] += state.jac_vals[idx] * state.y[row];
            }
            let mut zl = vec![0.0; n];
            let mut zu = vec![0.0; n];
            let kc = 1e10;
            for i in 0..n {
                if grad_jty[i] > 0.0 && state.x_l[i].is_finite() {
                    let sl = (state.x[i] - state.x_l[i]).max(1e-20);
                    if grad_jty[i] * sl <= kc * state.mu.max(1e-20) {
                        zl[i] = grad_jty[i];
                    }
                } else if grad_jty[i] < 0.0 && state.x_u[i].is_finite() {
                    let su = (state.x_u[i] - state.x[i]).max(1e-20);
                    if (-grad_jty[i]) * su <= kc * state.mu.max(1e-20) {
                        zu[i] = -grad_jty[i];
                    }
                }
            }
            (zl, zu)
        };
        let final_dual_inf = convergence::dual_infeasibility(
            &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
            &state.y, &z_l_opt_final, &z_u_opt_final, n,
        );
        let final_dual_inf_unscaled = convergence::dual_infeasibility(
            &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
            &state.y, &state.z_l, &state.z_u, n,
        );
        let final_compl = convergence::complementarity_error(
            &state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u, 0.0,
        );
        let final_compl_opt = convergence::complementarity_error(
            &state.x, &state.x_l, &state.x_u, &z_l_opt_final, &z_u_opt_final, 0.0,
        );
        let final_compl_best = final_compl.min(final_compl_opt);
        let mult_sum: f64 = state.y.iter().map(|v| v.abs()).sum::<f64>()
            + state.z_l.iter().map(|v| v.abs()).sum::<f64>()
            + state.z_u.iter().map(|v| v.abs()).sum::<f64>();
        let s_max: f64 = 100.0;
        let s_d = if (m + 2 * n) > 0 {
            ((s_max.max(mult_sum / (m + 2 * n) as f64)) / s_max).min(1e4)
        } else {
            1.0
        };
        rip_log!(
            "ripopt: MaxIter diag: pr={:.2e} du={:.2e}(t={:.2e}) du_u={:.2e}(t={:.0e}) co={:.2e} co_opt={:.2e} co_best={:.2e}(t={:.2e}/{:.2e}) mu={:.2e} sd={:.1} ac={}",
            final_primal_inf,
            final_dual_inf, options.tol * s_d,
            final_dual_inf_unscaled, options.dual_inf_tol,
            final_compl, final_compl_opt, final_compl_best,
            100.0 * options.tol * s_d, 10.0 * options.compl_inf_tol,
            state.mu, s_d, state.consecutive_acceptable,
        );
    }

    // At max_iter: check if the problem is actually infeasible.
    // Never declare infeasible if we were ever feasible.
    let final_theta = state.constraint_violation();
    if !ever_feasible && final_theta > options.constr_viol_tol {
        // Check stationarity of violation: if ||∇θ|| ≈ 0, declare local infeasibility
        let mut violation = vec![0.0; m];
        for i in 0..m {
            let is_eq = state.g_l[i].is_finite() && state.g_u[i].is_finite()
                && (state.g_l[i] - state.g_u[i]).abs() < 1e-15;
            if is_eq {
                violation[i] = state.g[i] - state.g_l[i];
            } else if state.g_l[i].is_finite() && state.g[i] < state.g_l[i] {
                violation[i] = state.g[i] - state.g_l[i];
            } else if state.g_u[i].is_finite() && state.g[i] > state.g_u[i] {
                violation[i] = state.g[i] - state.g_u[i];
            }
        }
        let mut grad_theta = vec![0.0; n];
        for (idx, (&row, &col)) in
            state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate()
        {
            grad_theta[col] += state.jac_vals[idx] * violation[row];
        }
        let grad_theta_norm = grad_theta.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let stationarity_tol = 1e-4 * final_theta.max(1.0);
        if grad_theta_norm < stationarity_tol {
            return make_result(&state, SolveStatus::LocalInfeasibility);
        }

        // Fallback: check if theta hasn't improved over recent history
        if final_theta > 1e4 && theta_history.len() >= theta_history_len {
            let min_theta = theta_history.iter().cloned().fold(f64::INFINITY, f64::min);
            if final_theta > 0.01 * min_theta {
                return make_result(&state, SolveStatus::Infeasible);
            }
        }
    }
    if options.print_level >= 5 {
        timings.print_summary(options.max_iter, ipm_start.elapsed());
    }
    make_result(&state, SolveStatus::MaxIterations)
}

/// Attempt a second-order correction step.
///
/// If the trial point has worse constraint violation than the current point,
/// try to correct by solving a modified system targeting the trial constraint values.
#[allow(clippy::too_many_arguments)]
fn attempt_soc<P: NlpProblem>(
    state: &SolverState,
    problem: &P,
    _x_trial: &[f64],
    g_trial: &[f64],
    solver: &mut dyn LinearSolver,
    kkt: &kkt::KktSystem,
    filter: &Filter,
    theta_current: f64,
    phi_current: f64,
    grad_phi_step: f64,
    alpha: f64,
    options: &SolverOptions,
) -> Option<(Vec<f64>, f64, Vec<f64>, f64)> {
    let n = state.n;
    let m = state.m;

    if m == 0 {
        return None;
    }

    // Compute constraint residual at trial point, respecting constraint type
    let mut c_soc = vec![0.0; m];
    for i in 0..m {
        let is_equality = state.g_l[i].is_finite() && state.g_u[i].is_finite()
            && (state.g_l[i] - state.g_u[i]).abs() < 1e-15;
        if is_equality || state.g_l[i].is_finite() {
            c_soc[i] = g_trial[i] - state.g_l[i];
        } else if state.g_u[i].is_finite() {
            c_soc[i] = g_trial[i] - state.g_u[i];
        }
    }

    let kappa_soc = 0.99;
    let mut theta_prev_soc = convergence::primal_infeasibility(g_trial, &state.g_l, &state.g_u);

    for _soc_iter in 0..options.max_soc {
        // Modify RHS for SOC: replace primal residual with trial constraint residual
        let mut rhs_soc = kkt.rhs.clone();
        for i in 0..m {
            rhs_soc[n + i] = -c_soc[i];
        }

        // Solve with same factored matrix
        let mut sol_soc = vec![0.0; n + m];
        if solver.solve(&rhs_soc, &mut sol_soc).is_err() {
            return None;
        }

        let dx_soc = &sol_soc[..n];

        // Compute SOC trial point
        #[allow(clippy::needless_range_loop)]
        let mut x_soc = vec![0.0; n];
        for i in 0..n {
            x_soc[i] = state.x[i] + alpha * dx_soc[i];
            if state.x_l[i].is_finite() {
                x_soc[i] = x_soc[i].max(state.x_l[i] + 1e-14);
            }
            if state.x_u[i].is_finite() {
                x_soc[i] = x_soc[i].min(state.x_u[i] - 1e-14);
            }
        }

        let mut obj_soc = f64::INFINITY;
        if !problem.objective(&x_soc, true, &mut obj_soc) { return None; }
        let mut g_soc = vec![0.0; m];
        if !problem.constraints(&x_soc, false, &mut g_soc) { return None; }

        let theta_soc = convergence::primal_infeasibility(&g_soc, &state.g_l, &state.g_u);

        // Stop SOC iterations if theta isn't decreasing sufficiently
        if theta_soc >= kappa_soc * theta_prev_soc {
            return None;
        }
        theta_prev_soc = theta_soc;

        // Compute barrier objective
        let mut phi_soc = obj_soc;
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            if state.x_l[i].is_finite() {
                let slack = (x_soc[i] - state.x_l[i]).max(1e-20);
                phi_soc -= state.mu * slack.ln();
            }
            if state.x_u[i].is_finite() {
                let slack = (state.x_u[i] - x_soc[i]).max(1e-20);
                phi_soc -= state.mu * slack.ln();
            }
        }
        if options.constraint_slack_barrier {
            for i in 0..m {
                let is_eq = state.g_l[i].is_finite() && state.g_u[i].is_finite()
                    && (state.g_l[i] - state.g_u[i]).abs() < 1e-15;
                if is_eq {
                    continue;
                }
                if state.g_l[i].is_finite() {
                    let slack = g_soc[i] - state.g_l[i];
                    if slack > state.mu * 1e-2 {
                        phi_soc -= state.mu * slack.ln();
                    }
                }
                if state.g_u[i].is_finite() {
                    let slack = state.g_u[i] - g_soc[i];
                    if slack > state.mu * 1e-2 {
                        phi_soc -= state.mu * slack.ln();
                    }
                }
            }
        }

        let (acceptable, _) = filter.check_acceptability(
            theta_current,
            phi_current,
            theta_soc,
            phi_soc,
            grad_phi_step,
            alpha,
        );

        if acceptable {
            return Some((x_soc, obj_soc, g_soc, alpha));
        }

        // Update c_soc for next SOC iteration (respecting constraint type)
        for i in 0..m {
            let is_equality = state.g_l[i].is_finite() && state.g_u[i].is_finite()
                && (state.g_l[i] - state.g_u[i]).abs() < 1e-15;
            if is_equality || state.g_l[i].is_finite() {
                c_soc[i] = g_soc[i] - state.g_l[i];
            } else if state.g_u[i].is_finite() {
                c_soc[i] = g_soc[i] - state.g_u[i];
            }
        }
    }

    None
}

/// Attempt a second-order correction step using the condensed KKT system.
///
/// Same logic as `attempt_soc` but uses the condensed system to avoid building
/// the full (n+m)×(n+m) KKT matrix. Uses `solve_condensed_soc` which rebuilds
/// only the n-dimensional condensed RHS with the modified constraint residual.
#[allow(clippy::too_many_arguments)]
fn attempt_soc_condensed<P: NlpProblem>(
    state: &SolverState,
    problem: &P,
    g_trial: &[f64],
    solver: &mut DenseLdl,
    condensed: &kkt::CondensedKktSystem,
    filter: &Filter,
    theta_current: f64,
    phi_current: f64,
    grad_phi_step: f64,
    alpha: f64,
    options: &SolverOptions,
) -> Option<(Vec<f64>, f64, Vec<f64>, f64)> {
    let n = state.n;
    let m = state.m;

    if m == 0 {
        return None;
    }

    // Compute constraint residual at trial point, respecting constraint type
    let mut c_soc = vec![0.0; m];
    for i in 0..m {
        let is_equality = state.g_l[i].is_finite() && state.g_u[i].is_finite()
            && (state.g_l[i] - state.g_u[i]).abs() < 1e-15;
        if is_equality || state.g_l[i].is_finite() {
            c_soc[i] = g_trial[i] - state.g_l[i];
        } else if state.g_u[i].is_finite() {
            c_soc[i] = g_trial[i] - state.g_u[i];
        }
    }

    let kappa_soc = 0.99;
    let mut theta_prev_soc = convergence::primal_infeasibility(g_trial, &state.g_l, &state.g_u);

    for _soc_iter in 0..options.max_soc {
        // Solve condensed system with modified constraint residual
        let dx_soc = match kkt::solve_condensed_soc(condensed, solver, &c_soc) {
            Ok(d) => d,
            Err(_) => return None,
        };

        // Compute SOC trial point
        #[allow(clippy::needless_range_loop)]
        let mut x_soc = vec![0.0; n];
        for i in 0..n {
            x_soc[i] = state.x[i] + alpha * dx_soc[i];
            if state.x_l[i].is_finite() {
                x_soc[i] = x_soc[i].max(state.x_l[i] + 1e-14);
            }
            if state.x_u[i].is_finite() {
                x_soc[i] = x_soc[i].min(state.x_u[i] - 1e-14);
            }
        }

        let mut obj_soc = f64::INFINITY;
        if !problem.objective(&x_soc, true, &mut obj_soc) { return None; }
        let mut g_soc = vec![0.0; m];
        if !problem.constraints(&x_soc, false, &mut g_soc) { return None; }

        let theta_soc = convergence::primal_infeasibility(&g_soc, &state.g_l, &state.g_u);

        // Stop SOC iterations if theta isn't decreasing sufficiently
        if theta_soc >= kappa_soc * theta_prev_soc {
            return None;
        }
        theta_prev_soc = theta_soc;

        // Compute barrier objective
        let mut phi_soc = obj_soc;
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            if state.x_l[i].is_finite() {
                let slack = (x_soc[i] - state.x_l[i]).max(1e-20);
                phi_soc -= state.mu * slack.ln();
            }
            if state.x_u[i].is_finite() {
                let slack = (state.x_u[i] - x_soc[i]).max(1e-20);
                phi_soc -= state.mu * slack.ln();
            }
        }
        if options.constraint_slack_barrier {
            for i in 0..m {
                let is_eq = state.g_l[i].is_finite() && state.g_u[i].is_finite()
                    && (state.g_l[i] - state.g_u[i]).abs() < 1e-15;
                if is_eq {
                    continue;
                }
                if state.g_l[i].is_finite() {
                    let slack = g_soc[i] - state.g_l[i];
                    if slack > state.mu * 1e-2 {
                        phi_soc -= state.mu * slack.ln();
                    }
                }
                if state.g_u[i].is_finite() {
                    let slack = state.g_u[i] - g_soc[i];
                    if slack > state.mu * 1e-2 {
                        phi_soc -= state.mu * slack.ln();
                    }
                }
            }
        }

        let (acceptable, _) = filter.check_acceptability(
            theta_current,
            phi_current,
            theta_soc,
            phi_soc,
            grad_phi_step,
            alpha,
        );

        if acceptable {
            return Some((x_soc, obj_soc, g_soc, alpha));
        }

        // Update c_soc for next SOC iteration (respecting constraint type)
        for i in 0..m {
            let is_equality = state.g_l[i].is_finite() && state.g_u[i].is_finite()
                && (state.g_l[i] - state.g_u[i]).abs() < 1e-15;
            if is_equality || state.g_l[i].is_finite() {
                c_soc[i] = g_soc[i] - state.g_l[i];
            } else if state.g_u[i].is_finite() {
                c_soc[i] = g_soc[i] - state.g_u[i];
            }
        }
    }

    None
}

/// SOC using sparse condensed KKT system.
fn attempt_soc_sparse_condensed<P: NlpProblem>(
    state: &SolverState,
    problem: &P,
    g_trial: &[f64],
    solver: &mut dyn LinearSolver,
    condensed: &kkt::SparseCondensedKktSystem,
    filter: &Filter,
    theta_current: f64,
    phi_current: f64,
    grad_phi_step: f64,
    alpha: f64,
    options: &SolverOptions,
) -> Option<(Vec<f64>, f64, Vec<f64>, f64)> {
    let n = state.n;
    let m = state.m;
    if m == 0 { return None; }

    let mut c_soc = vec![0.0; m];
    for i in 0..m {
        let is_equality = state.g_l[i].is_finite() && state.g_u[i].is_finite()
            && (state.g_l[i] - state.g_u[i]).abs() < 1e-15;
        if is_equality || state.g_l[i].is_finite() {
            c_soc[i] = g_trial[i] - state.g_l[i];
        } else if state.g_u[i].is_finite() {
            c_soc[i] = g_trial[i] - state.g_u[i];
        }
    }

    let kappa_soc = 0.99;
    let mut theta_prev_soc = convergence::primal_infeasibility(g_trial, &state.g_l, &state.g_u);

    for _soc_iter in 0..options.max_soc {
        let dx_soc = match kkt::solve_sparse_condensed_soc(condensed, solver, &c_soc) {
            Ok(d) => d,
            Err(_) => return None,
        };

        let mut x_soc = vec![0.0; n];
        for i in 0..n {
            x_soc[i] = state.x[i] + alpha * dx_soc[i];
            if state.x_l[i].is_finite() { x_soc[i] = x_soc[i].max(state.x_l[i] + 1e-14); }
            if state.x_u[i].is_finite() { x_soc[i] = x_soc[i].min(state.x_u[i] - 1e-14); }
        }

        let mut obj_soc = f64::INFINITY;
        if !problem.objective(&x_soc, true, &mut obj_soc) { return None; }
        let mut g_soc = vec![0.0; m];
        if !problem.constraints(&x_soc, false, &mut g_soc) { return None; }

        let theta_soc = convergence::primal_infeasibility(&g_soc, &state.g_l, &state.g_u);
        if theta_soc >= kappa_soc * theta_prev_soc { return None; }
        theta_prev_soc = theta_soc;

        let mut phi_soc = obj_soc;
        for i in 0..n {
            if state.x_l[i].is_finite() {
                phi_soc -= state.mu * (x_soc[i] - state.x_l[i]).max(1e-20).ln();
            }
            if state.x_u[i].is_finite() {
                phi_soc -= state.mu * (state.x_u[i] - x_soc[i]).max(1e-20).ln();
            }
        }
        if options.constraint_slack_barrier {
            for i in 0..m {
                let is_eq = state.g_l[i].is_finite() && state.g_u[i].is_finite()
                    && (state.g_l[i] - state.g_u[i]).abs() < 1e-15;
                if is_eq { continue; }
                if state.g_l[i].is_finite() {
                    let slack = g_soc[i] - state.g_l[i];
                    if slack > state.mu * 1e-2 { phi_soc -= state.mu * slack.ln(); }
                }
                if state.g_u[i].is_finite() {
                    let slack = state.g_u[i] - g_soc[i];
                    if slack > state.mu * 1e-2 { phi_soc -= state.mu * slack.ln(); }
                }
            }
        }

        let (acceptable, _) = filter.check_acceptability(
            theta_current, phi_current, theta_soc, phi_soc, grad_phi_step, alpha,
        );
        if acceptable { return Some((x_soc, obj_soc, g_soc, alpha)); }

        for i in 0..m {
            let is_equality = state.g_l[i].is_finite() && state.g_u[i].is_finite()
                && (state.g_l[i] - state.g_u[i]).abs() < 1e-15;
            if is_equality || state.g_l[i].is_finite() {
                c_soc[i] = g_soc[i] - state.g_l[i];
            } else if state.g_u[i].is_finite() {
                c_soc[i] = g_soc[i] - state.g_u[i];
            }
        }
    }

    None
}

/// Apply post-restoration success handling: update state, reset multipliers, filter, and mu.
fn apply_restoration_success<P: NlpProblem>(
    state: &mut SolverState,
    filter: &mut Filter,
    mu_state: &mut MuState,
    options: &SolverOptions,
    n: usize,
    m: usize,
    problem: &P,
    x_new: &[f64],
    linear_constraints: Option<&[bool]>,
    lbfgs_mode: bool,
    lbfgs_state: &mut Option<LbfgsIpmState>,
) {
    state.x.copy_from_slice(x_new);
    state.alpha_primal = 0.0;
    let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints, lbfgs_mode);
    if let Some(ref mut lbfgs) = lbfgs_state {
        let lag_grad = LbfgsIpmState::compute_lagrangian_gradient(
            &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals, &state.y, state.n,
        );
        lbfgs.update(&state.x, &lag_grad);
        lbfgs.fill_hessian(&mut state.hess_vals);
    }

    // Reset multipliers after restoration (Ipopt-style).
    let bound_mult_reset_threshold = 1000.0;
    let mu_for_reset = state.mu;
    let mut any_large = false;
    for i in 0..n {
        if state.x_l[i].is_finite() {
            let slack = (state.x[i] - state.x_l[i]).max(1e-12);
            let z_new = mu_for_reset / slack;
            if z_new > bound_mult_reset_threshold {
                any_large = true;
            }
            state.z_l[i] = z_new.min(bound_mult_reset_threshold);
        }
        if state.x_u[i].is_finite() {
            let slack = (state.x_u[i] - state.x[i]).max(1e-12);
            let z_new = mu_for_reset / slack;
            if z_new > bound_mult_reset_threshold {
                any_large = true;
            }
            state.z_u[i] = z_new.min(bound_mult_reset_threshold);
        }
    }
    if any_large {
        for i in 0..n {
            if state.x_l[i].is_finite() {
                let slack = (state.x[i] - state.x_l[i]).max(1e-12);
                state.z_l[i] = (mu_for_reset / slack).min(bound_mult_reset_threshold);
            } else {
                state.z_l[i] = 0.0;
            }
            if state.x_u[i].is_finite() {
                let slack = (state.x_u[i] - state.x[i]).max(1e-12);
                state.z_u[i] = (mu_for_reset / slack).min(bound_mult_reset_threshold);
            } else {
                state.z_u[i] = 0.0;
            }
        }
    }
    // Compute least-squares multiplier estimate at the restored point.
    // This avoids the "dead start" (y=0) that causes the filter to reject every
    // post-restoration step. Use a higher dimension limit than initialization since
    // restoration runs infrequently.
    let ls_restoration_dim_limit = 1000;
    if m > 0 && (m + n) <= ls_restoration_dim_limit {
        if let Some(y_ls) = compute_ls_multiplier_estimate(
            &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
            &state.g_l, &state.g_u, n, m, options.constr_mult_init_max,
        ) {
            state.y.copy_from_slice(&y_ls);
        } else {
            for i in 0..m {
                state.y[i] = 0.0;
            }
        }
    } else {
        for i in 0..m {
            state.y[i] = 0.0;
        }
    }

    // Reset constraint slack barrier multipliers v_l, v_u from mu/slack.
    let mu_r = state.mu;
    for i in 0..m {
        let is_eq = state.g_l[i].is_finite() && state.g_u[i].is_finite()
            && (state.g_l[i] - state.g_u[i]).abs() < 1e-15;
        if is_eq {
            state.v_l[i] = 0.0;
            state.v_u[i] = 0.0;
            continue;
        }
        if state.g_l[i].is_finite() {
            let slack = (state.g[i] - state.g_l[i]).max(1e-12);
            state.v_l[i] = (mu_r / slack).min(bound_mult_reset_threshold);
        } else {
            state.v_l[i] = 0.0;
        }
        if state.g_u[i].is_finite() {
            let slack = (state.g_u[i] - state.g[i]).max(1e-12);
            state.v_u[i] = (mu_r / slack).min(bound_mult_reset_threshold);
        } else {
            state.v_u[i] = 0.0;
        }
    }

    // Reset filter and re-initialize from restored point
    filter.reset();
    let theta_restored = state.constraint_violation();
    filter.set_theta_min_from_initial(theta_restored);
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
fn attempt_nlp_restoration<P: NlpProblem>(
    problem: &P,
    state: &SolverState,
    filter: &Filter,
    options: &SolverOptions,
    theta_current: f64,
    start_time: Instant,
) -> (Vec<f64>, RestorationOutcome) {
    let n = state.n;
    let m = state.m;

    if options.print_level >= 5 {
        rip_log!(
            "ripopt: Entering NLP restoration (theta={:.2e}, mu={:.2e})",
            theta_current, state.mu
        );
    }

    // Adaptive rho: just large enough to exceed current multiplier magnitude.
    // Ipopt (Wächter & Biegler 2006, §3.3) uses rho = max(rho_prev, 2*||y||_inf + rho_small).
    // Static rho=1000 causes ill-conditioning: dual infeasibility stagnates because
    // the KKT system must offset the huge penalty gradient. Floor of 100 ensures
    // slacks are still penalized heavily enough to drive them toward zero.
    let y_inf = state.y.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let rho = (2.0 * y_inf + 1.0).max(100.0).min(1000.0);

    // Build restoration NLP
    let resto_nlp = RestorationNlp::new(problem, &state.x, state.mu, rho, 1.0);

    // Configure inner solver options
    let mut inner_opts = options.clone();
    inner_opts.max_iter = options.restoration_max_iter.max(500);
    inner_opts.disable_nlp_restoration = true; // prevent recursion
    inner_opts.print_level = if options.print_level >= 5 { 3 } else { 0 };
    inner_opts.mu_init = state.mu.max(1e-2);
    // Disable stall detection for inner solve: the restoration NLP makes slow
    // but steady progress toward feasibility, and the 30-iter stall limit kills
    // it prematurely. max_iter provides a hard cap.
    inner_opts.stall_iter_limit = 0;
    // Relax convergence tolerances — we just need feasibility, not optimality
    inner_opts.tol = 1e-7;

    // Propagate remaining wall time so the inner solve doesn't get a fresh clock.
    // Without this, the inner solve can run for the full max_wall_time, causing
    // the outer solve to timeout before its fallback cascade can run.
    if options.max_wall_time > 0.0 {
        let remaining = options.max_wall_time - start_time.elapsed().as_secs_f64();
        if remaining < 0.5 {
            // Not enough time left for a meaningful inner solve
            return (state.x[..n].to_vec(), RestorationOutcome::Failed);
        }
        inner_opts.max_wall_time = remaining;
    }
    // Scale early stall timeout by restoration NLP size — large restoration NLPs
    // need more time than the default 3s cap.
    let resto_dim = resto_nlp.num_variables() + resto_nlp.num_constraints();
    inner_opts.early_stall_timeout = if options.early_stall_timeout > 0.0 {
        if resto_dim > 500 {
            options.early_stall_timeout // Full timeout for large restoration NLPs
        } else {
            options.early_stall_timeout.min(3.0)
        }
    } else {
        3.0
    };

    // Solve the restoration NLP
    let result = solve_ipm(&resto_nlp, &inner_opts);

    // Extract x_orig from the restoration solution
    let x_nlp: Vec<f64> = result.x[..n].to_vec();

    // Evaluate original constraints at the restored point
    let mut g_new = vec![0.0; m];
    if !problem.constraints(&x_nlp, true, &mut g_new) {
        return (x_nlp, RestorationOutcome::Failed);
    }
    let theta_new = convergence::primal_infeasibility(&g_new, &state.g_l, &state.g_u);

    // Evaluate original objective at the restored point
    let mut phi_new = f64::INFINITY;
    if !problem.objective(&x_nlp, false, &mut phi_new) {
        return (x_nlp, RestorationOutcome::Failed);
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

    // Check success criteria — require meaningful improvement
    if theta_new < options.constr_viol_tol {
        // Achieved feasibility
        return (x_nlp, RestorationOutcome::Success);
    }

    // Require >=50% reduction for non-feasible improvement (stricter than GN's 10%)
    // to avoid marginal "success" that prevents recovery mechanisms from engaging.
    if theta_new <= 0.5 * theta_current {
        return (x_nlp, RestorationOutcome::Success);
    }

    // Check if acceptable to outer filter AND has meaningful reduction
    if theta_new < 0.9 * theta_current {
        let filter_acceptable = {
            let entries = filter.entries();
            let theta_max = filter.theta_max();
            let gamma_theta = filter.gamma_theta();
            let gamma_phi = filter.gamma_phi();

            if theta_new.is_nan() || phi_new.is_nan() || theta_new > theta_max {
                false
            } else {
                let mut ok = true;
                for entry in entries {
                    if theta_new >= (1.0 - gamma_theta) * entry.theta
                        && phi_new >= entry.phi - gamma_phi * entry.theta
                    {
                        ok = false;
                        break;
                    }
                }
                ok
            }
        };

        if filter_acceptable {
            return (x_nlp, RestorationOutcome::Success);
        }
    }

    // Inner solve converged but didn't improve feasibility → locally infeasible
    if inner_converged {
        return (x_nlp, RestorationOutcome::LocalInfeasibility);
    }

    (x_nlp, RestorationOutcome::Failed)
}

/// Compute average complementarity for recomputing mu after restoration.
/// Quality function for barrier parameter selection.
///
/// Evaluates Q(mu) = dual_inf² + primal_inf² + compl_err(mu)² for log-spaced
/// candidate mu values and returns the minimizer. The first two terms are fixed
/// at the current iterate; only the complementarity term varies with mu.
///
/// This replaces the Loqo oracle (mu = avg_compl / kappa) with a global search
/// that can make aggressive mu decreases when the iterate is well-centered.
#[allow(dead_code)]
fn quality_function_mu(state: &SolverState, mu_lower: f64, mu_upper: f64, n_candidates: usize) -> f64 {
    if mu_upper <= mu_lower || n_candidates < 2 {
        return mu_upper;
    }

    let n = state.n;
    let pi = state.constraint_violation();
    let di = convergence::dual_infeasibility(
        &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
        &state.y, &state.z_l, &state.z_u, n,
    );
    let fixed_part = pi * pi + di * di;

    let log_min = mu_lower.max(1e-20).ln();
    let log_max = mu_upper.ln();

    let mut best_mu = mu_upper;
    let mut best_q = f64::INFINITY;

    for k in 0..n_candidates {
        let t = k as f64 / (n_candidates - 1) as f64;
        let mu_candidate = (log_min + t * (log_max - log_min)).exp();

        let ci = convergence::complementarity_error(
            &state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u, mu_candidate,
        );
        let q = fixed_part + ci * ci;

        if q < best_q {
            best_q = q;
            best_mu = mu_candidate;
        }
    }

    best_mu
}

/// Compute least-squares multiplier estimate: min ||grad_f + J^T y||^2.
/// Solves the normal equations (J J^T) y = -J grad_f using dense Bunch-Kaufman LDL.
/// Returns Some(y) if successful and all estimates are within threshold; None otherwise.
fn compute_ls_multiplier_estimate(
    grad_f: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    n: usize,
    m: usize,
    max_abs_threshold: f64,
) -> Option<Vec<f64>> {
    if m == 0 {
        return None;
    }

    // Compute b = -J * grad_f  (m-vector)
    let mut b = vec![0.0; m];
    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        b[row] -= jac_vals[idx] * grad_f[col];
    }

    // Compute A = J * J^T  (m x m dense symmetric matrix)
    let mut j_dense = vec![0.0; m * n];
    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        j_dense[row * n + col] = jac_vals[idx];
    }
    let mut a_mat = SymmetricMatrix::zeros(m);
    for i in 0..m {
        for j in 0..=i {
            let mut dot = 0.0;
            for k in 0..n {
                dot += j_dense[i * n + k] * j_dense[j * n + k];
            }
            a_mat.set(i, j, dot);
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

    let max_abs = y_ls.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if max_abs > max_abs_threshold {
        return None;
    }

    // For inequality constraints, zero out multipliers with wrong sign.
    // Ipopt convention (L = f + y^T g):
    //   g >= g_l (lower bound only): y >= 0
    //   g <= g_u (upper bound only): y <= 0
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

    Some(y_ls)
}

fn compute_avg_complementarity(state: &SolverState) -> f64 {
    let mut sum_compl = 0.0;
    let mut count = 0;
    // Variable bound complementarity: z_l * (x - x_l), z_u * (x_u - x)
    for i in 0..state.n {
        if state.x_l[i].is_finite() {
            let slack = (state.x[i] - state.x_l[i]).max(1e-20);
            sum_compl += slack * state.z_l[i];
            count += 1;
        }
        if state.x_u[i].is_finite() {
            let slack = (state.x_u[i] - state.x[i]).max(1e-20);
            sum_compl += slack * state.z_u[i];
            count += 1;
        }
    }
    // If no variable bounds exist but inequality constraints do, include
    // constraint slack complementarity v_l*(g-g_l), v_u*(g_u-g) as fallback.
    // This prevents avg_compl=0 for problems with only inequality constraints
    // (e.g., OET2/6/7 with m=1002 inequalities and no variable bounds),
    // which otherwise causes mu to collapse to mu_min prematurely.
    //
    // When variable bounds exist, their z*slack products already drive mu;
    // adding v*slack (which ≈ mu since v = mu/slack) would bias avg_compl
    // and slow convergence (causes TP044/TP116 regressions).
    if count == 0 {
        for i in 0..state.m {
            if state.v_l[i] > 0.0 {
                let slack = (state.g[i] - state.g_l[i]).max(1e-20);
                sum_compl += state.v_l[i] * slack;
                count += 1;
            }
            if state.v_u[i] > 0.0 {
                let slack = (state.g_u[i] - state.g[i]).max(1e-20);
                sum_compl += state.v_u[i] * slack;
                count += 1;
            }
        }
    }
    if count > 0 {
        sum_compl / count as f64
    } else {
        0.0
    }
}

/// Compute barrier error for fixed-mode subproblem convergence check.
/// This is the optimality error of the current barrier subproblem (for fixed mu).
fn compute_barrier_error(state: &SolverState) -> f64 {
    let n = state.n;

    // Dual infeasibility of barrier problem:
    // grad_f + J^T y - z_l + z_u
    let mut grad_lag = state.grad_f.clone();
    for (idx, (&row, &col)) in state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate() {
        grad_lag[col] += state.jac_vals[idx] * state.y[row];
    }
    for i in 0..n {
        if state.x_l[i].is_finite() {
            grad_lag[i] -= state.z_l[i];
        }
        if state.x_u[i].is_finite() {
            grad_lag[i] += state.z_u[i];
        }
    }

    let sd = n.max(1) as f64;
    let dual_err = grad_lag.iter().map(|v| v.abs()).sum::<f64>() / sd;

    // Complementarity error (relative to mu)
    let mut compl_err = 0.0;
    let mut count = 0;
    for i in 0..n {
        if state.x_l[i].is_finite() {
            let slack = (state.x[i] - state.x_l[i]).max(1e-20);
            compl_err += (slack * state.z_l[i] - state.mu).abs();
            count += 1;
        }
        if state.x_u[i].is_finite() {
            let slack = (state.x_u[i] - state.x[i]).max(1e-20);
            compl_err += (slack * state.z_u[i] - state.mu).abs();
            count += 1;
        }
    }
    if count > 0 {
        compl_err /= count as f64;
    }

    // Primal infeasibility
    let primal_err = state.constraint_violation();

    dual_err.max(compl_err).max(primal_err)
}

/// Strategy 3: Try active set identification + reduced KKT solve.
///
/// At near-optimal points, identify variables at their bounds (active set),
/// fix them, solve the reduced KKT system for free variables, and check
/// if the result meets strict convergence tolerances.
fn try_active_set_solve<P: NlpProblem>(
    state: &mut SolverState,
    problem: &P,
    options: &SolverOptions,
    linear_constraints: Option<&[bool]>,
    lbfgs_mode: bool,
) -> Option<SolveResult> {
    let n = state.n;
    let m = state.m;

    // Identify active bounds using complementarity gap.
    // A variable is "active at lower bound" if x_i is close to x_l_i and z_l_i is significant.
    let tol_bound = 1e-6;
    let mut is_free = vec![true; n];
    let mut active_lower = vec![false; n];
    let mut active_upper = vec![false; n];
    let mut n_free = 0usize;

    for i in 0..n {
        let at_lower = state.x_l[i].is_finite()
            && (state.x[i] - state.x_l[i]).abs() < tol_bound * (1.0 + state.x_l[i].abs());
        let at_upper = state.x_u[i].is_finite()
            && (state.x_u[i] - state.x[i]).abs() < tol_bound * (1.0 + state.x_u[i].abs());

        if at_lower && state.z_l[i] > 1e-8 {
            is_free[i] = false;
            active_lower[i] = true;
        } else if at_upper && state.z_u[i] > 1e-8 {
            is_free[i] = false;
            active_upper[i] = true;
        } else {
            n_free += 1;
        }
    }

    // Need at least one active bound for this strategy to help
    if n_free == n {
        return None;
    }

    // Don't attempt if reduced system is too large for dense solve
    let dim = n_free + m;
    if dim > 500 {
        return None;
    }
    if dim == 0 {
        return None;
    }

    // Build mapping: free_idx[k] = original index of k-th free variable
    let mut free_idx = Vec::with_capacity(n_free);
    let mut orig_to_free = vec![usize::MAX; n]; // usize::MAX = not free
    for i in 0..n {
        if is_free[i] {
            orig_to_free[i] = free_idx.len();
            free_idx.push(i);
        }
    }

    // Fix active variables at their bounds (save full state for restoration)
    let saved_x = state.x.clone();
    let saved_y = state.y.clone();
    let saved_zl = state.z_l.clone();
    let saved_zu = state.z_u.clone();
    for i in 0..n {
        if active_lower[i] {
            state.x[i] = state.x_l[i];
        } else if active_upper[i] {
            state.x[i] = state.x_u[i];
        }
    }

    // Re-evaluate at the snapped point
    let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints, lbfgs_mode);

    // Build reduced KKT system (dense):
    // [ H_ff   J_f^T ] [ dx_f ]   [ -grad_f_f ]
    // [ J_f    0     ] [ dy   ] = [ g_l/g_u - g ]
    //
    // where H_ff is the Hessian restricted to free-free, J_f is Jacobian cols for free vars.

    let mut kkt = vec![0.0; dim * dim];
    let mut rhs = vec![0.0; dim];

    // Fill H_ff block (top-left n_free x n_free)
    for (idx, (&row, &col)) in state.hess_rows.iter().zip(state.hess_cols.iter()).enumerate() {
        let fr = orig_to_free[row];
        let fc = orig_to_free[col];
        if fr != usize::MAX && fc != usize::MAX {
            kkt[fr * dim + fc] += state.hess_vals[idx];
            if fr != fc {
                kkt[fc * dim + fr] += state.hess_vals[idx]; // symmetric
            }
        }
    }

    // Fill J_f block (bottom-left m x n_free) and J_f^T (top-right n_free x m)
    for (idx, (&row, &col)) in state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate() {
        let fc = orig_to_free[col];
        if fc != usize::MAX {
            let r = n_free + row; // row in KKT
            kkt[r * dim + fc] += state.jac_vals[idx];       // J_f block
            kkt[fc * dim + r] += state.jac_vals[idx];       // J_f^T block
        }
    }

    // RHS: top part = -grad_f for free variables
    for k in 0..n_free {
        rhs[k] = -state.grad_f[free_idx[k]];
        // Subtract contribution of fixed active variables via Hessian
        // (H * x_active terms absorbed into gradient already since we re-evaluated)
    }

    // RHS: bottom part = constraint target - g(x)
    // For equality constraints (g_l == g_u): rhs = g_l - g
    // For inequality constraints: use the active bound side
    for i in 0..m {
        if (state.g_l[i] - state.g_u[i]).abs() < 1e-20 {
            // Equality
            rhs[n_free + i] = state.g_l[i] - state.g[i];
        } else if state.g[i] <= state.g_l[i] + 1e-10 {
            rhs[n_free + i] = state.g_l[i] - state.g[i];
        } else if state.g[i] >= state.g_u[i] - 1e-10 {
            rhs[n_free + i] = state.g_u[i] - state.g[i];
        } else {
            // Inactive constraint: target is current value (no correction needed)
            rhs[n_free + i] = 0.0;
        }
    }

    // Solve the dense symmetric system via LDL^T (Bunch-Kaufman-like pivoting)
    let solution = dense_symmetric_solve(dim, &mut kkt, &mut rhs);
    if solution.is_none() {
        // Singular system, restore and bail
        state.x.copy_from_slice(&saved_x);
        state.y.copy_from_slice(&saved_y);
        state.z_l.copy_from_slice(&saved_zl);
        state.z_u.copy_from_slice(&saved_zu);
        let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints, lbfgs_mode);
        return None;
    }
    let sol = solution.unwrap();

    // Apply the step for free variables (with damping for safety)
    let alpha = 1.0; // full Newton step
    for k in 0..n_free {
        let i = free_idx[k];
        state.x[i] += alpha * sol[k];
        // Clamp to bounds
        if state.x_l[i].is_finite() {
            state.x[i] = state.x[i].max(state.x_l[i]);
        }
        if state.x_u[i].is_finite() {
            state.x[i] = state.x[i].min(state.x_u[i]);
        }
    }

    // Update y from the solve
    for i in 0..m {
        state.y[i] = sol[n_free + i];
    }

    // Re-evaluate at the new point
    let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints, lbfgs_mode);

    // Recover z from stationarity: ∇f + J^T y = z_l - z_u
    let mut grad_jty = state.grad_f.clone();
    for (idx, (&row, &col)) in state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate() {
        grad_jty[col] += state.jac_vals[idx] * state.y[row];
    }
    for i in 0..n {
        state.z_l[i] = 0.0;
        state.z_u[i] = 0.0;
        if state.x_l[i].is_finite() && grad_jty[i] > 0.0 {
            state.z_l[i] = grad_jty[i];
        } else if state.x_u[i].is_finite() && grad_jty[i] < 0.0 {
            state.z_u[i] = -grad_jty[i];
        }
    }

    // Check strict convergence
    let primal_inf = state.constraint_violation();
    let dual_inf = convergence::dual_infeasibility(
        &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
        &state.y, &state.z_l, &state.z_u, n,
    );
    let compl_inf = convergence::complementarity_error(
        &state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u, 0.0,
    );
    let multiplier_sum: f64 = state.y.iter().map(|v| v.abs()).sum::<f64>()
        + state.z_l.iter().map(|v| v.abs()).sum::<f64>()
        + state.z_u.iter().map(|v| v.abs()).sum::<f64>();
    let multiplier_count = m + 2 * n;

    let conv_info = ConvergenceInfo {
        primal_inf,
        dual_inf,
        dual_inf_unscaled: dual_inf, // same z used for both
        dual_inf_unscaled_opt: dual_inf,
        compl_inf,
        compl_inf_opt: compl_inf,
        mu: 0.0, // at the solution, mu should be zero
        objective: state.obj,
        multiplier_sum,
        multiplier_count,
    };

    if let ConvergenceStatus::Converged = check_convergence(&conv_info, options, 0) {
        return Some(make_result(state, SolveStatus::Optimal));
    }

    // Didn't converge; restore original state and re-evaluate
    state.x.copy_from_slice(&saved_x);
    state.y.copy_from_slice(&saved_y);
    state.z_l.copy_from_slice(&saved_zl);
    state.z_u.copy_from_slice(&saved_zu);
    let _ = state.evaluate_with_linear(problem, 1.0, linear_constraints, lbfgs_mode);

    None
}

/// Dense symmetric indefinite solve using diagonal pivoting (LDL^T).
/// Solves A*x = b in-place. Returns Some(x) on success, None if singular.
/// `a` is row-major dim x dim, `b` is length dim.
fn dense_symmetric_solve(dim: usize, a: &mut [f64], b: &mut [f64]) -> Option<Vec<f64>> {
    // Gaussian elimination with partial pivoting for symmetric indefinite systems.
    // Simple but sufficient for small systems (dim < 500).
    let mut piv = vec![0usize; dim];
    for i in 0..dim {
        piv[i] = i;
    }

    for k in 0..dim {
        // Find pivot: largest diagonal element in remaining submatrix
        let mut max_val = a[k * dim + k].abs();
        let mut max_idx = k;
        for i in (k + 1)..dim {
            if a[i * dim + i].abs() > max_val {
                max_val = a[i * dim + i].abs();
                max_idx = i;
            }
        }

        if max_val < 1e-15 {
            return None; // Singular
        }

        // Swap rows/cols k and max_idx
        if max_idx != k {
            piv.swap(k, max_idx);
            // Swap rows
            for j in 0..dim {
                let tmp = a[k * dim + j];
                a[k * dim + j] = a[max_idx * dim + j];
                a[max_idx * dim + j] = tmp;
            }
            // Swap cols
            for i in 0..dim {
                let tmp = a[i * dim + k];
                a[i * dim + k] = a[i * dim + max_idx];
                a[i * dim + max_idx] = tmp;
            }
            b.swap(k, max_idx);
        }

        let pivot = a[k * dim + k];
        // Eliminate below
        for i in (k + 1)..dim {
            let factor = a[i * dim + k] / pivot;
            a[i * dim + k] = factor;
            for j in (k + 1)..dim {
                a[i * dim + j] -= factor * a[k * dim + j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    let mut x = b.to_vec();
    for k in (0..dim).rev() {
        for j in (k + 1)..dim {
            x[k] -= a[k * dim + j] * x[j];
        }
        x[k] /= a[k * dim + k];
    }

    Some(x)
}

/// Compute a steepest-descent fallback direction when KKT solve fails.
///
/// Returns (dx, dy) where dx = -alpha * grad_f (scaled gradient step)
/// and dy = 0. This is crude but prevents immediate failure.
fn gradient_descent_fallback(state: &SolverState) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = state.n;
    let m = state.m;
    let grad_norm: f64 = state.grad_f.iter().map(|g| g * g).sum::<f64>().sqrt();
    if grad_norm < 1e-20 {
        return None;
    }
    let alpha = 1e-4 / grad_norm; // small step
    let mut dx = vec![0.0; n];
    for i in 0..n {
        dx[i] = -alpha * state.grad_f[i];
    }
    let dy = vec![0.0; m];
    Some((dx, dy))
}

/// Build the final solve result.
/// Computes z from stationarity for more accurate output multipliers.
/// Unscales all values from the internal scaled space to the original NLP space.
fn make_result(state: &SolverState, status: SolveStatus) -> SolveResult {
    let n = state.n;
    let m = state.m;

    // Fill in final convergence measures for diagnostics
    let mut diag = state.diagnostics.clone();
    diag.final_mu = state.mu;
    diag.final_primal_inf = state.constraint_violation();
    diag.final_dual_inf = convergence::dual_infeasibility(
        &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
        &state.y, &state.z_l, &state.z_u, n,
    );
    diag.final_compl = convergence::complementarity_error(
        &state.x, &state.x_l, &state.x_u, &state.z_l, &state.z_u, 0.0,
    );

    // Compute dual scaling factor s_d (same formula as check_convergence)
    {
        let s_max: f64 = 100.0;
        let s_d_max: f64 = 1e4;
        let mult_sum: f64 = state.y.iter().map(|v| v.abs()).sum::<f64>()
            + state.z_l.iter().map(|v| v.abs()).sum::<f64>()
            + state.z_u.iter().map(|v| v.abs()).sum::<f64>();
        let mult_count = m + 2 * n;
        diag.final_s_d = if mult_count > 0 {
            ((s_max.max(mult_sum / mult_count as f64)) / s_max).min(s_d_max)
        } else {
            1.0
        };
    }

    // Compute optimal z from stationarity in scaled space: ∇f_s + J_s^T y_s - z_l_s + z_u_s = 0
    let mut grad_jty = state.grad_f.clone();
    for (idx, (&row, &col)) in state.jac_rows.iter().zip(state.jac_cols.iter()).enumerate() {
        grad_jty[col] += state.jac_vals[idx] * state.y[row];
    }

    // z from stationarity (scaled), then unscale: z_unscaled = z_scaled / obj_scaling
    let mut z_l_opt_scaled = vec![0.0; n];
    let mut z_u_opt_scaled = vec![0.0; n];
    let mut z_l_out = vec![0.0; n];
    let mut z_u_out = vec![0.0; n];
    for i in 0..n {
        if grad_jty[i] > 0.0 && state.x_l[i].is_finite() {
            z_l_opt_scaled[i] = grad_jty[i];
            z_l_out[i] = grad_jty[i] / state.obj_scaling;
        } else if grad_jty[i] < 0.0 && state.x_u[i].is_finite() {
            z_u_opt_scaled[i] = -grad_jty[i];
            z_u_out[i] = -grad_jty[i] / state.obj_scaling;
        }
    }

    // z_opt-based dual infeasibility (used in scaled convergence gate)
    diag.final_dual_inf_scaled = convergence::dual_infeasibility(
        &state.grad_f, &state.jac_rows, &state.jac_cols, &state.jac_vals,
        &state.y, &z_l_opt_scaled, &z_u_opt_scaled, n,
    );

    // Unscale constraint multipliers: y_unscaled[i] = y_scaled[i] * g_scaling[i] / obj_scaling
    let mut y_out = state.y.clone();
    for i in 0..m {
        y_out[i] = state.y[i] * state.g_scaling[i] / state.obj_scaling;
    }

    // Unscale constraint values: g_unscaled[i] = g_scaled[i] / g_scaling[i]
    let mut g_out = state.g.clone();
    for i in 0..m {
        g_out[i] /= state.g_scaling[i];
    }

    // No Acceptable status anymore — pass status through directly.
    let validated_status = status;

    SolveResult {
        x: state.x.clone(),
        objective: state.obj / state.obj_scaling,
        constraint_multipliers: y_out,
        bound_multipliers_lower: z_l_out,
        bound_multipliers_upper: z_u_out,
        constraint_values: g_out,
        status: validated_status,
        iterations: state.iter,
        diagnostics: diag,
    }
}
