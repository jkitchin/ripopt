use std::cell::Cell;

use crate::filter::FilterEntry;
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
    /// Parent NLP's max-bound-violation `||c(x_R) − π[g_l,g_u](c(x_R))||_∞`
    /// at restoration entry. The early-exit hook compares the current
    /// inner-iterate's parent violation against
    /// `parent_kappa_resto · parent_theta_entry`.
    /// `0.0` disables early exit (no parent target injected).
    parent_theta_entry: f64,
    /// Required infeasibility-reduction factor (Ipopt's
    /// `required_infeasibility_reduction`, default 0.9). When the
    /// parent's max-bound-violation falls to this fraction of the
    /// entry value, the inner solve exits Optimal.
    parent_kappa_resto: f64,
    /// Cached parent constraint bounds (used by `resto_early_exit` to
    /// recompute the parent max-bound-violation without going back
    /// through the wrapper stack).
    parent_g_l: Vec<f64>,
    parent_g_u: Vec<f64>,
    /// Optional acceptable tolerance for early exit
    /// (`min(tol, constr_viol_tol)`); a parent violation below this
    /// floor exits even if `kappa_resto` would not yet trigger.
    parent_small_threshold: f64,
    /// 1-norm of the parent's bound-violation at restoration entry —
    /// used for the filter / sufficient-progress gates (Ipopt
    /// `IpFilterLSAcceptor::IsAcceptableToCurrentIterate`,
    /// IpFilterLSAcceptor.cpp:497-498). Distinct from
    /// `parent_theta_entry` (max-norm) which feeds the κ_resto gate
    /// (Ipopt `IpRestoConvCheck.cpp:184-190`).
    parent_theta_entry_l1: f64,
    /// Parent's `f(x_R)` at restoration entry. φ for the inner solve
    /// is taken to be the original NLP objective only — matching
    /// ripopt's existing post-restoration `classify_restoration_outcome`
    /// metric (`src/ipm.rs:8562`); this keeps the inner early-exit
    /// gate and the post-resto Success classifier consistent so the
    /// parent never rejects a point the inner just exited on.
    parent_phi_entry: f64,
    /// Parent filter snapshot at restoration entry (Ipopt's parent
    /// `FilterLSAcceptor`'s entry list, frozen for the duration of
    /// the inner solve — `IpFilterLSAcceptor.cpp:497`).
    parent_filter_entries: Vec<FilterEntry>,
    /// Parent filter `theta_max` (envelope cap; rejects trials whose
    /// 1-norm θ exceeds this).
    parent_theta_max: f64,
    /// Parent filter `gamma_theta` (sufficient-θ-reduction margin;
    /// Ipopt default 1e-5).
    parent_gamma_theta: f64,
    /// Parent filter `gamma_phi` (sufficient-φ-reduction margin;
    /// Ipopt default 1e-8).
    parent_gamma_phi: f64,
    /// Parent's curr_mu at restoration entry. Used to compute the
    /// **barrier** φ_trial = f(x) − μ·Σ ln(slack) for the parent-filter
    /// gate (D5 fix: matches `IpRestoConvCheck.cpp:193`'s use of
    /// `orig_ip_cq->trial_barrier_obj()`). Without this the parent
    /// filter would be tested with raw `f(x)` against entries that
    /// store barrier-φ — a category mismatch that loosens the gate.
    parent_mu_entry: f64,
    /// Parent x lower bounds (n_orig). Finite entries contribute
    /// `−μ·ln(x − x_l)` to the parent's barrier φ.
    parent_x_l: Vec<f64>,
    /// Parent x upper bounds (n_orig). Finite entries contribute
    /// `−μ·ln(x_u − x)` to the parent's barrier φ.
    parent_x_u: Vec<f64>,
    /// `IpRestoFilterConvCheck::CheckConvergence` skips the early-exit
    /// test on the very first inner iteration (Ipopt's
    /// `first_resto_iter_` flag, IpRestoConvCheck.cpp:73-78); without
    /// this guard the resto solve can declare success at iter 0
    /// before the slack variables have moved. Cell so the immutable
    /// `resto_early_exit(&self, ...)` hook can flip it.
    first_iter_seen: Cell<bool>,
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
            parent_theta_entry: 0.0,
            parent_kappa_resto: 0.0,
            parent_g_l: g_l,
            parent_g_u: g_u,
            parent_small_threshold: 0.0,
            parent_theta_entry_l1: 0.0,
            parent_phi_entry: 0.0,
            parent_filter_entries: Vec::new(),
            parent_theta_max: f64::INFINITY,
            parent_gamma_theta: 1e-5,
            parent_gamma_phi: 1e-8,
            parent_mu_entry: mu_entry.max(0.0),
            parent_x_l: Vec::new(),
            parent_x_u: Vec::new(),
            first_iter_seen: Cell::new(false),
        }
    }

    /// Inject the parent NLP's restoration-entry violation
    /// `theta_entry` (max-norm bound-violation, used by the κ_resto
    /// gate per `IpRestoConvCheck.cpp:184-190`), `theta_entry_l1`
    /// (1-norm bound-violation, used by the filter and
    /// sufficient-progress gates per
    /// `IpFilterLSAcceptor.cpp:497-498`), `phi_entry` (parent's
    /// `f(x_R)`), the required-reduction factor `kappa_resto`
    /// (Ipopt's `required_infeasibility_reduction`, default 0.9), the
    /// small-threshold floor (`min(tol, constr_viol_tol)`), the
    /// parent filter parameters (`theta_max`, `gamma_theta`,
    /// `gamma_phi`) and a snapshot of the parent's filter entries.
    ///
    /// Together these support the full
    /// `IpRestoFilterConvCheck::TestOrigProgress`
    /// (`IpRestoFilterConvCheck.cpp:53-80`) early-exit gate:
    /// (1) skip on first inner iter; (2) max-norm κ_resto reduction;
    /// (3) parent-filter acceptance; (4) sufficient progress on
    /// (θ_entry, φ_entry).
    pub fn set_parent_target(
        &mut self,
        theta_entry: f64,
        theta_entry_l1: f64,
        phi_entry: f64,
        kappa_resto: f64,
        small_threshold: f64,
        theta_max: f64,
        gamma_theta: f64,
        gamma_phi: f64,
        filter_entries: Vec<FilterEntry>,
        mu_entry: f64,
        x_l: Vec<f64>,
        x_u: Vec<f64>,
    ) {
        self.parent_theta_entry = theta_entry.max(0.0);
        self.parent_theta_entry_l1 = theta_entry_l1.max(0.0);
        self.parent_phi_entry = phi_entry;
        self.parent_kappa_resto = kappa_resto.max(0.0);
        self.parent_small_threshold = small_threshold.max(0.0);
        self.parent_theta_max = theta_max;
        self.parent_gamma_theta = gamma_theta.max(0.0);
        self.parent_gamma_phi = gamma_phi.max(0.0);
        self.parent_filter_entries = filter_entries;
        self.parent_mu_entry = mu_entry.max(0.0);
        self.parent_x_l = x_l;
        self.parent_x_u = x_u;
        self.first_iter_seen.set(false);
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

    /// Ipopt `IpRestoFilterConvCheck::TestOrigProgress`
    /// (`IpRestoFilterConvCheck.cpp:53-80`,
    /// `IpRestoConvCheck.cpp:71-248`): the inner solve exits early
    /// when ALL of the following gates pass at the current inner
    /// iterate `x_n`:
    ///
    /// 1. **Not first inner iter** — Ipopt's `first_resto_iter_`
    ///    flag (IpRestoConvCheck.cpp:73-78) skips the test on iter 0
    ///    so the slack variables have a chance to move.
    /// 2. **κ_resto reduction (max-norm)** —
    ///    `θ_max(x_n) ≤ max(κ_resto · θ_entry_max, small_threshold)`,
    ///    Ipopt's primary `required_infeasibility_reduction` gate
    ///    (`IpRestoConvCheck.cpp:184-190`).
    /// 3. **Parent filter acceptance (1-norm θ, raw f as φ)** — the
    ///    `(θ_l1, φ)` pair must not be dominated by any parent
    ///    filter entry, mirroring `Filter::IsAcceptable`
    ///    (`IpFilterLSAcceptor.cpp:497`).
    /// 4. **Sufficient progress vs parent iterate** —
    ///    `θ_l1(x_n) ≤ (1−γ_θ)·θ_entry_l1` OR
    ///    `φ(x_n) − φ_entry ≤ −γ_φ · θ_entry_l1`, mirroring
    ///    `IsAcceptableToCurrentIterate` (`IpFilterLSAcceptor.cpp:497-498`).
    ///
    /// The φ used here is the **parent barrier objective**
    /// `φ(x) = f(x) − μ_parent · Σ ln(slack)` over finite x bounds
    /// (D5 fix). This matches `IpRestoConvCheck.cpp:193`
    /// (`orig_ip_cq->trial_barrier_obj()`); the parent filter stores
    /// barrier-φ entries, so testing raw `f` against them was a
    /// category mismatch. With μ_parent captured at restoration entry
    /// the gate is consistent with the parent's filter semantics.
    fn resto_early_exit(&self, x: &[f64]) -> bool {
        let n = self.n_orig;
        let m = self.m_orig;
        if self.parent_theta_entry <= 0.0 || x.len() < n {
            return false;
        }
        // Gate 1: skip on first inner iter (Ipopt first_resto_iter_).
        if !self.first_iter_seen.get() {
            self.first_iter_seen.set(true);
            return false;
        }

        // Compute parent c(x_n).
        let mut g = vec![0.0; m];
        if m > 0 && !self.inner.constraints(&x[..n], true, &mut g) {
            return false;
        }
        let mut max_viol = 0.0_f64;
        let mut l1_viol = 0.0_f64;
        for i in 0..m {
            let lo = self.parent_g_l[i];
            let hi = self.parent_g_u[i];
            let v = if g[i] < lo {
                lo - g[i]
            } else if g[i] > hi {
                g[i] - hi
            } else {
                0.0
            };
            if v > max_viol {
                max_viol = v;
            }
            l1_viol += v;
        }

        // Gate 2: κ_resto reduction in max-norm.
        let threshold_max = (self.parent_kappa_resto * self.parent_theta_entry)
            .max(self.parent_small_threshold);
        let gate_kappa = max_viol <= threshold_max;

        // φ_trial = f(x_n) − μ_parent · Σ ln(slack) — parent barrier
        // objective. D5 fix: the parent filter stores barrier-φ
        // entries (`compute_barrier_phi` is the parent's metric); the
        // raw `f(x)` would mismatch when slacks are tight.
        let mut f_val = 0.0;
        if !self.inner.objective(&x[..n], false, &mut f_val) || !f_val.is_finite() {
            return false;
        }
        let mut phi_trial = f_val;
        if self.parent_mu_entry > 0.0 {
            let nb = self.parent_x_l.len().min(self.parent_x_u.len()).min(n);
            for i in 0..nb {
                if self.parent_x_l[i].is_finite() {
                    let s = x[i] - self.parent_x_l[i];
                    if s <= 0.0 {
                        return false;
                    }
                    phi_trial -= self.parent_mu_entry * s.ln();
                }
                if self.parent_x_u[i].is_finite() {
                    let s = self.parent_x_u[i] - x[i];
                    if s <= 0.0 {
                        return false;
                    }
                    phi_trial -= self.parent_mu_entry * s.ln();
                }
            }
            if !phi_trial.is_finite() {
                return false;
            }
        }

        // Gate 3: parent filter acceptance on (θ_l1, φ).
        let theta_entry_l1 = self.parent_theta_entry_l1;
        let phi_entry = self.parent_phi_entry;
        let gamma_theta = self.parent_gamma_theta;
        let gamma_phi = self.parent_gamma_phi;
        let gate_filter = if l1_viol.is_nan() || phi_trial.is_nan() {
            false
        } else if l1_viol > self.parent_theta_max {
            false
        } else {
            !self.parent_filter_entries.iter().any(|e| {
                l1_viol >= (1.0 - gamma_theta) * e.theta
                    && phi_trial >= e.phi - gamma_phi * e.theta
            })
        };

        // Gate 4: sufficient progress vs (θ_entry_l1, φ_entry).
        let theta_progress = l1_viol <= (1.0 - gamma_theta) * theta_entry_l1;
        let phi_progress = (phi_trial - phi_entry) <= -gamma_phi * theta_entry_l1;
        let gate_progress = theta_progress || phi_progress;

        let exit = gate_kappa && gate_filter && gate_progress;
        if std::env::var("RIPOPT_TRACE_RESTO_EXIT").is_ok() {
            eprintln!(
                "  resto-exit-probe: max_viol={:.3e}/thr={:.3e} l1={:.3e}/entry_l1={:.3e} \
                 phi={:.3e}/entry={:.3e} κ={} flt={} prog={} → exit={}",
                max_viol, threshold_max, l1_viol, theta_entry_l1,
                phi_trial, phi_entry,
                gate_kappa, gate_filter, gate_progress, exit,
            );
        }
        exit
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

    /// Spec §7.4 (T2.22): Hessian of the restoration objective uses D_R^2,
    /// not D_R, and the slack rows/cols are zero. The original Hessian must
    /// be evaluated with `obj_factor=0` (P22, `IpRestoIpoptNLP.cpp:691`).
    #[test]
    fn test_hessian_uses_d_r_squared_and_zeroes_slacks() {
        let prob = SimpleConstrained;
        let x_r = vec![2.0_f64, 4.0];
        let mu: f64 = 0.25;
        let eta = 1.0 * mu.sqrt();
        let resto = RestorationNlp::new(&prob, &x_r, mu, 1000.0, 1.0);
        let (hrows, hcols) = resto.hessian_structure();
        let nnz = hrows.len();
        let mut vals = vec![0.0; nnz];
        let x = vec![3.0, 5.0, 0.1, 0.1];
        let lambda = vec![0.0];
        resto.hessian_values(&x, true, 1.0, &lambda, &mut vals);

        let mut diag_x = vec![0.0; 2];
        let mut max_off_x_or_slack = 0.0_f64;
        for k in 0..nnz {
            let r = hrows[k];
            let c = hcols[k];
            if r == c && r < 2 {
                diag_x[r] = vals[k];
            } else if vals[k].abs() > max_off_x_or_slack {
                max_off_x_or_slack = vals[k].abs();
            }
        }
        assert!(
            (diag_x[0] - eta * 0.25).abs() < 1e-12,
            "diag_x[0] should be eta*D_R^2[0] = {} but was {}",
            eta * 0.25, diag_x[0]
        );
        assert!(
            (diag_x[1] - eta * 0.0625).abs() < 1e-12,
            "diag_x[1] should be eta*D_R^2[1] = {} but was {}",
            eta * 0.0625, diag_x[1]
        );
        assert!((diag_x[0] - eta * 0.5).abs() > 1e-3, "regression: D_R not D_R^2");
        assert!(
            max_off_x_or_slack < 1e-12,
            "slack rows/cols and x off-diag must be zero, max abs = {}",
            max_off_x_or_slack
        );
    }

    /// Spec §7.2 / T2.22 item A1: residual is c(x) − p + n.
    #[test]
    fn test_eval_g_signs_match_spec() {
        let prob = SimpleConstrained;
        let x_r = vec![0.0, 0.0];
        let resto = RestorationNlp::new(&prob, &x_r, 0.1, 1000.0, 1.0);
        let mut g = vec![0.0; 1];
        let x = vec![1.0, 3.0, 1.0, 0.0];
        resto.constraints(&x, true, &mut g);
        assert!((g[0] - 3.0).abs() < 1e-12, "expected 3 (= 4-1+0), got {}", g[0]);
        let x2 = vec![1.0, 3.0, 0.0, 1.0];
        resto.constraints(&x2, true, &mut g);
        assert!((g[0] - 5.0).abs() < 1e-12, "expected 5 (= 4-0+1), got {}", g[0]);
    }

    /// Spec §7.2 / T2.22 item A2: Jacobian slack columns [−I_p, +I_n].
    #[test]
    fn test_eval_jac_g_slack_signs_match_spec() {
        let prob = SimpleConstrained;
        let x_r = vec![0.0, 0.0];
        let resto = RestorationNlp::new(&prob, &x_r, 0.1, 1000.0, 1.0);
        let (rows, cols) = resto.jacobian_structure();
        let mut vals = vec![0.0; rows.len()];
        resto.jacobian_values(&[0.5, 0.5, 0.1, 0.1], true, &mut vals);
        for (k, (&r, &c)) in rows.iter().zip(cols.iter()).enumerate() {
            assert_eq!(r, 0);
            match c {
                0 | 1 => assert!((vals[k] - 1.0).abs() < 1e-12, "J_x[{}]=1, got {}", c, vals[k]),
                2 => assert!((vals[k] - (-1.0)).abs() < 1e-12, "p-col should be -1, got {}", vals[k]),
                3 => assert!((vals[k] - 1.0).abs() < 1e-12, "n-col should be +1, got {}", vals[k]),
                _ => panic!("unexpected col {}", c),
            }
        }
    }

    /// Spec §7.2 / T2.22 item A3: gradient `[η·D_R²·(x−x_R), ρ·1, ρ·1]`.
    #[test]
    fn test_eval_grad_f_uses_d_r_squared() {
        let prob = SimpleConstrained;
        let x_r = vec![2.0_f64, 4.0];
        let mu: f64 = 0.25;
        let eta = mu.sqrt();
        let rho = 1000.0;
        let resto = RestorationNlp::new(&prob, &x_r, mu, rho, 1.0);
        let nv = resto.num_variables();
        let mut grad = vec![0.0; nv];
        let x = vec![3.0, 5.0, 0.7, 0.3];
        resto.gradient(&x, true, &mut grad);
        assert!((grad[0] - eta * 0.25).abs() < 1e-12, "grad[0] = {}", grad[0]);
        assert!((grad[1] - eta * 0.0625).abs() < 1e-12, "grad[1] = {}", grad[1]);
        assert!((grad[2] - rho).abs() < 1e-12);
        assert!((grad[3] - rho).abs() < 1e-12);
        assert!((grad[0] - eta * 0.5).abs() > 1e-3, "regression: grad uses D_R, not D_R^2");
    }
}
