//! TRON (Trust-Region Newton method) for bound-constrained problems:
//!
//!     min f(x)  s.t.  l ≤ x ≤ u
//!
//! Reference: Lin, C.-J. and Moré, J.J. "Newton's method for large
//! bound-constrained optimization problems", SIAM J. Optim. 9(4):1100-1127, 1999.
//! https://doi.org/10.1137/S1052623498345075
//!
//! # When to use this instead of `ripopt::solve`
//!
//! [`solve_bc`] requires `problem.num_constraints() == 0` and at least one
//! finite bound. It is **not** auto-dispatched — `ripopt::solve` always uses
//! the IPM. Pick `solve_bc` when:
//!
//! - The problem is pure bound-constrained (`m == 0`), and
//! - The objective is well-conditioned (small/moderate `n`, smooth `f`).
//!
//! On convex/quadratic BC problems TRON typically converges in 1–3
//! iterations versus the IPM's 5–8 (no barrier sequence to walk). On highly
//! nonlinear / ill-conditioned BC problems (e.g. the PALMER nonlinear
//! least-squares family) the IPM is currently faster because TRON's inner
//! CG is plain (no incomplete-Cholesky preconditioning) and there is no
//! inner active-set restart (Lin & Moré §4). TRON and the IPM can converge
//! to different local minima on multimodal landscapes; the explicit entry
//! point makes that choice visible.
//!
//! Algorithm structure (per-iteration):
//!   1. Convergence test on projected-gradient infinity norm.
//!   2. Cauchy step: piecewise-quadratic line search along the projected
//!      steepest-descent path P[x − α·g], constrained to the trust region.
//!      Identifies a working active set at the Cauchy point.
//!   3. Subspace minimization: truncated CG on the QP restricted to the
//!      free variables, with Steihaug-Toint termination (negative curvature
//!      or trust-region boundary) and bound-clipping (a CG step that would
//!      exit [l, u] is shortened to the bound and the loop exits — the
//!      next outer iteration picks up the new active set).
//!   4. Trust-region acceptance test on actual / predicted reduction;
//!      radius update by ratio.
//!
//! Implementation choices for ripopt's first cut:
//!   - Single dense Hessian per outer iteration. The standard CUTEst BC
//!     workload (the PALMER / PFIT data-fitting families that motivated
//!     issue #22) has n ≤ 10, so dense storage and dense mat-vec are
//!     cheap and avoid sparse-triplet bookkeeping in the hot loop.
//!     Generalization to sparse mat-vec is a future refinement.
//!   - No preconditioner (plain CG, not PCG). The Lin-Moré reference uses
//!     incomplete Cholesky; deferred until profiling justifies it.
//!   - Bound clipping inside CG, not the inner active-set restart from
//!     Lin-Moré §4. We rely on the outer TR loop to re-identify newly
//!     active bounds. Costs more outer iterations on problems with many
//!     active-set changes but converges to the same point and is a small
//!     fraction of the implementation complexity.

use crate::{NlpProblem, SolveResult, SolveStatus, SolverDiagnostics};

/// Configuration for the bound-constrained solver.
#[derive(Debug, Clone)]
pub struct BcOptions {
    /// Convergence tolerance on the projected-gradient infinity norm
    /// `‖P[x − g] − x‖_∞`. Default `1e-8` (matches IPM `tol` default).
    pub tol: f64,
    /// Maximum number of outer iterations. Default 200 (TRON typically
    /// finishes in ≪ 100; the cap mostly catches pathological cases).
    pub max_iter: usize,
    /// Per-outer-iteration cap on CG iterations. Default `max(10, n)`.
    /// Used as a hard cap on top of the natural CG residual-reduction
    /// stop.
    pub max_cg_iter: Option<usize>,
    /// CG residual-reduction stop: terminate when `‖r_k‖ ≤ cg_tol·‖r_0‖`.
    /// Default `0.1` (Eisenstat-Walker style; conservative). Lower values
    /// (e.g. `1e-2`) push for tighter inner solves.
    pub cg_residual_reduction: f64,
    /// Acceptance threshold on the actual/predicted-reduction ratio ρ.
    /// Trial step accepted iff `ρ > η_0`. Default `1e-4` (Lin-Moré).
    pub eta0: f64,
    /// TR shrink threshold: `ρ < η_1` ⇒ shrink. Default `0.25` (Lin-Moré).
    pub eta1: f64,
    /// TR expand threshold: `ρ > η_2` and step on TR boundary ⇒ expand.
    /// Default `0.75` (Lin-Moré).
    pub eta2: f64,
    /// TR shrink factor when ρ < η_1. Default `0.25`.
    pub gamma1: f64,
    /// TR expand factor when ρ > η_2 and step is on TR boundary.
    /// Default `4.0`.
    pub gamma2: f64,
    /// Initial trust-region radius. Default `1.0`. Lin-Moré §6 recommends
    /// `‖g_0‖` for least-squares; we use a fixed default and let the
    /// first few iterations adapt via the ρ test.
    pub initial_tr_radius: f64,
    /// Maximum trust-region radius. Default `1e10`.
    pub max_tr_radius: f64,
    /// Print level. 0 = silent, 1 = banner + final, 2 = per-iter table,
    /// 3 = per-iter + CG inner trace.
    pub print_level: u8,
}

impl Default for BcOptions {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            max_iter: 200,
            max_cg_iter: None,
            cg_residual_reduction: 0.1,
            eta0: 1e-4,
            eta1: 0.25,
            eta2: 0.75,
            gamma1: 0.25,
            gamma2: 4.0,
            initial_tr_radius: 1.0,
            max_tr_radius: 1e10,
            print_level: 0,
        }
    }
}

/// Solve a bound-constrained NLP via TRON.
///
/// The problem must satisfy `num_constraints() == 0`. Returns
/// `InternalError` if called on a problem with general constraints
/// (caller should dispatch to the IPM in that case).
pub fn solve_bc<P: NlpProblem>(problem: &P, opts: &BcOptions) -> SolveResult {
    let n = problem.num_variables();
    let m = problem.num_constraints();

    if m != 0 {
        return error_result(
            n,
            SolveStatus::InternalError,
            "solve_bc requires num_constraints == 0",
        );
    }

    let mut x_l = vec![f64::NEG_INFINITY; n];
    let mut x_u = vec![f64::INFINITY; n];
    problem.bounds(&mut x_l, &mut x_u);

    // Validate bounds.
    for i in 0..n {
        if !(x_l[i] <= x_u[i]) {
            return error_result(
                n,
                SolveStatus::InternalError,
                "bound validation failed: x_l > x_u",
            );
        }
    }

    let mut x = vec![0.0; n];
    problem.initial_point(&mut x);
    project_in_place(&mut x, &x_l, &x_u);

    let mut g = vec![0.0; n];
    let mut h_dense = vec![0.0; n * n];
    let mut f_x = 0.0;

    // Hessian sparsity is fixed across the solve.
    let (h_rows, h_cols) = problem.hessian_structure();
    let mut h_vals = vec![0.0; h_rows.len()];
    let lambda: Vec<f64> = Vec::new(); // m=0

    if !problem.objective(&x, true, &mut f_x) {
        return error_result(n, SolveStatus::EvaluationError, "f(x0) failed");
    }
    if !problem.gradient(&x, false, &mut g) {
        return error_result(n, SolveStatus::EvaluationError, "∇f(x0) failed");
    }

    let mut tr_radius = opts.initial_tr_radius;
    let max_cg = opts.max_cg_iter.unwrap_or_else(|| n.max(10));

    if opts.print_level >= 2 {
        eprintln!("{:>4} {:>14} {:>10} {:>10} {:>10} {:>10} {:>5}",
            "iter", "f", "‖Pg‖∞", "Δ", "‖s‖", "ρ", "cg");
    }

    for iter in 0..opts.max_iter {
        // Convergence test: projected-gradient infinity norm.
        let pg_inf = projected_gradient_inf_norm(&x, &g, &x_l, &x_u);

        if opts.print_level >= 2 {
            eprintln!("{:>4} {:>14.6e} {:>10.2e} {:>10.2e}",
                iter, f_x, pg_inf, tr_radius);
        }

        if pg_inf <= opts.tol {
            return success_result(
                problem, &x, f_x, &g, &x_l, &x_u, iter, SolveStatus::Optimal,
            );
        }

        // Compute Hessian at current iterate.
        if !problem.hessian_values(&x, false, 1.0, &lambda, &mut h_vals) {
            return error_result(n, SolveStatus::EvaluationError, "Hessian eval failed");
        }
        densify_lower_triangle(n, &h_rows, &h_cols, &h_vals, &mut h_dense);

        // --- Step 1: Cauchy step (projected gradient + breakpoint search) ---
        let (s_cauchy, _alpha_c, q_cauchy) = cauchy_step(
            &x, &g, &h_dense, &x_l, &x_u, tr_radius,
        );

        // --- Step 2: Subspace minimization via truncated CG on free vars ---
        // Active set at Cauchy point. A variable is "active" if it sits on a
        // bound after the Cauchy step. Free variables are everything else.
        let mut x_after_cauchy = x.clone();
        for i in 0..n {
            x_after_cauchy[i] = (x[i] + s_cauchy[i]).clamp(x_l[i], x_u[i]);
        }
        let active = active_set(&x_after_cauchy, &x_l, &x_u);

        let (s_total, q_total, cg_iters) = subspace_step(
            &x, &g, &h_dense, &x_l, &x_u,
            &s_cauchy, q_cauchy, &active,
            tr_radius, opts.cg_residual_reduction, max_cg,
            opts.print_level,
        );

        // Trial point.
        let mut x_trial = vec![0.0; n];
        for i in 0..n {
            x_trial[i] = (x[i] + s_total[i]).clamp(x_l[i], x_u[i]);
        }

        let mut f_trial = 0.0;
        let f_ok = problem.objective(&x_trial, true, &mut f_trial);

        // Predicted reduction (model decrease) — strictly positive when the
        // step makes progress on the model. q_total is q(s) − q(0) = q at s
        // (with q(0) = 0 by construction), so pred = −q_total.
        let pred_red = -q_total;
        let act_red = if f_ok { f_x - f_trial } else { f64::NEG_INFINITY };

        // Step norm in the Euclidean metric (Lin-Moré uses ‖·‖_M with a
        // diagonal scaling; we use I for the first cut).
        let s_norm = s_total.iter().map(|v| v * v).sum::<f64>().sqrt();
        let on_boundary = (s_norm - tr_radius).abs() <= 1e-12 * tr_radius.max(1.0);

        let rho = if pred_red > 0.0 && f_ok {
            act_red / pred_red
        } else {
            f64::NEG_INFINITY
        };

        if opts.print_level >= 2 {
            eprintln!("    └─ trial f={:.6e} s_norm={:.2e} pred={:.2e} act={:.2e} ρ={:.2e} cg={}",
                f_trial, s_norm, pred_red, act_red, rho, cg_iters);
        }

        // Accept / reject.
        let accept = rho > opts.eta0 && f_ok;
        if accept {
            x = x_trial;
            f_x = f_trial;
            if !problem.gradient(&x, true, &mut g) {
                return error_result(n, SolveStatus::EvaluationError, "∇f failed");
            }
        }
        // (else: x, f_x, g unchanged — re-evaluate H next iter at same x)

        // Trust-region radius update (Lin-Moré Algorithm 5.1).
        tr_radius = if rho < opts.eta1 {
            (opts.gamma1 * tr_radius.min(s_norm)).max(1e-16)
        } else if rho > opts.eta2 && on_boundary {
            (opts.gamma2 * tr_radius).min(opts.max_tr_radius)
        } else {
            tr_radius
        };

        // Tiny-step guard: if Δ collapses below machine precision we'll
        // never make progress. Equivalent to Ipopt's STOP_AT_TINY_STEP.
        if tr_radius < 1e-15 {
            return success_result(
                problem, &x, f_x, &g, &x_l, &x_u, iter + 1, SolveStatus::StopAtTinyStep,
            );
        }
    }

    success_result(
        problem, &x, f_x, &g, &x_l, &x_u, opts.max_iter, SolveStatus::MaxIterations,
    )
}

// ---------------------------------------------------------------------------
// Projection and projected-gradient utilities
// ---------------------------------------------------------------------------

#[inline]
fn project_in_place(x: &mut [f64], l: &[f64], u: &[f64]) {
    for i in 0..x.len() {
        if x[i] < l[i] { x[i] = l[i]; }
        else if x[i] > u[i] { x[i] = u[i]; }
    }
}

/// Compute `‖P[x − g] − x‖_∞`, the infinity norm of the projected
/// gradient. Standard first-order optimality measure for BC problems
/// (Lin-Moré eq. (2.1)).
fn projected_gradient_inf_norm(x: &[f64], g: &[f64], l: &[f64], u: &[f64]) -> f64 {
    let mut max = 0.0_f64;
    for i in 0..x.len() {
        let p = (x[i] - g[i]).clamp(l[i], u[i]) - x[i];
        let a = p.abs();
        if a > max { max = a; }
    }
    max
}

/// Variable is "active" if it sits exactly on a bound. Returns a vec of
/// `i64` flags: −1 = at lower, +1 = at upper, 0 = free.
fn active_set(x: &[f64], l: &[f64], u: &[f64]) -> Vec<i8> {
    let n = x.len();
    let mut a = vec![0_i8; n];
    for i in 0..n {
        if x[i] <= l[i] { a[i] = -1; }
        else if x[i] >= u[i] { a[i] = 1; }
    }
    a
}

// ---------------------------------------------------------------------------
// Hessian densification (lower triangle → full symmetric)
// ---------------------------------------------------------------------------

fn densify_lower_triangle(
    n: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
    out: &mut [f64],
) {
    debug_assert_eq!(out.len(), n * n);
    for v in out.iter_mut() { *v = 0.0; }
    for k in 0..rows.len() {
        let (mut i, mut j) = (rows[k], cols[k]);
        // Some problem implementations may emit upper-triangle entries;
        // canonicalize to (i ≥ j).
        if i < j { std::mem::swap(&mut i, &mut j); }
        out[i * n + j] += vals[k];
        if i != j {
            out[j * n + i] += vals[k];
        }
    }
}

#[inline]
fn matvec(n: usize, h: &[f64], v: &[f64], out: &mut [f64]) {
    for i in 0..n {
        let row_off = i * n;
        let mut s = 0.0;
        for j in 0..n {
            s += h[row_off + j] * v[j];
        }
        out[i] = s;
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() { s += a[i] * b[i]; }
    s
}

// ---------------------------------------------------------------------------
// Cauchy step: piecewise-quadratic search along P[x − α·g]
// ---------------------------------------------------------------------------

/// Compute the Cauchy point in the path direction `−g` projected onto
/// `[l, u]` and clipped to the trust region. Returns `(s, alpha, q(s))`
/// where `s = P[x − α·g] − x` is the Cauchy step.
///
/// The objective along the path is piecewise-quadratic in `α`: each
/// breakpoint is where one component hits a bound. Between breakpoints
/// the quadratic model `q(α) = g^T s + 0.5 s^T H s` is smooth in α; we
/// walk segments in order of increasing α and stop at the first segment
/// whose minimizer is interior or whose endpoint hits the trust region.
fn cauchy_step(
    x: &[f64],
    g: &[f64],
    h: &[f64],
    l: &[f64],
    u: &[f64],
    tr_radius: f64,
) -> (Vec<f64>, f64, f64) {
    let n = x.len();

    // Direction d(α) along the path P[x − α·g] − x. Component i:
    //   d_i(α) = max(l_i − x_i, min(u_i − x_i, −α g_i))
    // The breakpoint of component i is where −α g_i hits one of the
    // bounds. If g_i = 0 the component never moves. Otherwise:
    //   α_brk_i = (x_i − l_i) / g_i  if g_i > 0  (would hit lower)
    //   α_brk_i = (x_i − u_i) / g_i  if g_i < 0  (would hit upper)
    let mut breakpoints: Vec<(f64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        if g[i] > 0.0 && x[i] > l[i] && l[i].is_finite() {
            let a = (x[i] - l[i]) / g[i];
            if a.is_finite() && a > 0.0 { breakpoints.push((a, i)); }
        } else if g[i] < 0.0 && x[i] < u[i] && u[i].is_finite() {
            let a = (x[i] - u[i]) / g[i];
            if a.is_finite() && a > 0.0 { breakpoints.push((a, i)); }
        }
    }
    breakpoints.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Build the path parametrically. On each segment [α_prev, α_next] the
    // active-flag pattern is constant; the path direction p(α) is
    // affine in α on the free components and constant on the active ones.
    //
    // Let A_k = set of indices already pinned at the start of segment k.
    // For i ∈ A_k: d_i is fixed at l_i − x_i or u_i − x_i.
    // For i ∉ A_k: d_i(α) = −α g_i.
    //
    // The model along the segment is q(α) = g^T d(α) + 0.5 d(α)^T H d(α);
    // it's a scalar quadratic in α we minimize analytically and
    // intersect with the trust-region ball ‖d‖ ≤ Δ.
    //
    // d(α) = c + α p where:
    //   c_i = d_i_pinned for i ∈ A_k, else 0
    //   p_i = 0 for i ∈ A_k, else −g_i
    let mut c = vec![0.0; n];
    let mut p = vec![0.0; n];
    for i in 0..n { p[i] = -g[i]; }

    // Mark variables that start the search already pinned (already on a
    // bound and gradient pushes outward).
    for i in 0..n {
        if g[i] > 0.0 && x[i] <= l[i] {
            // already at or below lower bound, gradient pushes left → pin
            c[i] = l[i] - x[i];
            p[i] = 0.0;
        } else if g[i] < 0.0 && x[i] >= u[i] {
            c[i] = u[i] - x[i];
            p[i] = 0.0;
        } else if !g[i].is_finite() || g[i] == 0.0 {
            p[i] = 0.0;
        } else if (g[i] > 0.0 && !l[i].is_finite()) || (g[i] < 0.0 && !u[i].is_finite()) {
            // unbounded in the descent direction: free, will follow until TR.
        }
    }

    let mut alpha_prev = 0.0;
    let mut tmp = vec![0.0; n];
    let mut hp = vec![0.0; n];

    // Helper: optimal α* on the current segment, intersected with [α_prev, α_max].
    // q(α) = q(c) + α · (g^T p + c^T H p) + 0.5 α² · (p^T H p)
    // Set dq/dα = 0 → α* = −(g^T p + c^T H p) / (p^T H p) (when p^T H p > 0).
    let mut best_alpha = 0.0;
    let mut best_d = vec![0.0; n];
    let mut best_q = 0.0;

    let mut bp_iter = breakpoints.iter().peekable();

    loop {
        // Next breakpoint α (or +∞ if none left).
        let alpha_next = match bp_iter.peek() {
            Some(&&(a, _)) => a,
            None => f64::INFINITY,
        };

        // TR-boundary α on this segment. Solve ‖c + α p‖² = Δ²:
        //   (p·p) α² + 2 (c·p) α + (c·c − Δ²) = 0
        let pp = dot(&p, &p);
        let cp = dot(&c, &p);
        let cc = dot(&c, &c);
        let alpha_tr = if pp <= 0.0 {
            f64::INFINITY
        } else {
            let disc = cp * cp - pp * (cc - tr_radius * tr_radius);
            if disc < 0.0 {
                // c already outside TR — segment unreachable.
                f64::INFINITY
            } else {
                let s = disc.sqrt();
                // Two roots; we want the larger non-negative one (the path
                // exits the ball at the +root when traveling from c).
                let r1 = (-cp - s) / pp;
                let r2 = (-cp + s) / pp;
                let cand = r2.max(r1);
                if cand >= alpha_prev { cand } else { f64::INFINITY }
            }
        };

        let alpha_max_seg = alpha_next.min(alpha_tr);

        // Quadratic minimizer on this segment.
        // q'(α) at α = (g·p + c·Hp) + α·(p·Hp). Compute Hp.
        matvec(n, h, &p, &mut hp);
        let php = dot(&p, &hp);
        let chp = dot(&c, &hp);
        let gp = dot(g, &p);
        let alpha_star = if php > 0.0 {
            -(gp + chp) / php
        } else {
            // p^T H p ≤ 0: q is unbounded below or affine-decreasing along p.
            // Go all the way to α_max_seg.
            alpha_max_seg
        };

        // Choose the per-segment optimizer, clamped to [α_prev, α_max_seg].
        let alpha_opt = alpha_star.clamp(alpha_prev, alpha_max_seg);
        if alpha_opt > alpha_prev {
            // Compute candidate step and its model value.
            for i in 0..n { tmp[i] = c[i] + alpha_opt * p[i]; }
            // q(s) = g·s + 0.5 s·H s
            matvec(n, h, &tmp, &mut hp);
            let qval = dot(g, &tmp) + 0.5 * dot(&tmp, &hp);
            if qval < best_q || best_alpha == 0.0 {
                best_q = qval;
                best_alpha = alpha_opt;
                best_d.clone_from(&tmp);
            }
        }

        // If we hit the TR boundary or there are no more breakpoints, stop.
        if alpha_max_seg >= alpha_tr - f64::EPSILON || alpha_next.is_infinite() {
            break;
        }
        // Sufficient-decrease early exit: if the segment minimizer was
        // interior (strictly less than α_max_seg) and produced negative q,
        // we can stop — further breakpoints can only pin more components,
        // which won't help once we've found an unconstrained interior min.
        if alpha_star < alpha_next && alpha_star > alpha_prev && php > 0.0 && best_q < 0.0 {
            break;
        }

        // Advance past the breakpoint: pin component i and update c, p.
        if let Some(&(a_bp, i)) = bp_iter.next() {
            // Lock i at its bound: d_i = l_i − x_i or u_i − x_i.
            // The actual d_i at α = a_bp equals c_i + a_bp · p_i; this is
            // exactly the bound by construction of the breakpoint.
            let d_at_brk = c[i] + a_bp * p[i];
            c[i] = d_at_brk;
            p[i] = 0.0;
            alpha_prev = a_bp;
        } else {
            break;
        }
    }

    // Edge case: if no segment yielded a candidate, the projected-gradient
    // direction was unproductive (all components active outward, or
    // p^T H p ≤ 0 with TR=∞). Fall back to a tiny step along −P[g].
    if best_alpha == 0.0 {
        // Take the projected-gradient direction itself, scaled to TR.
        for i in 0..n { best_d[i] = (x[i] - g[i]).clamp(l[i], u[i]) - x[i]; }
        let nrm = dot(&best_d, &best_d).sqrt();
        if nrm > tr_radius && nrm > 0.0 {
            let s = tr_radius / nrm;
            for i in 0..n { best_d[i] *= s; }
        }
        matvec(n, h, &best_d, &mut hp);
        best_q = dot(g, &best_d) + 0.5 * dot(&best_d, &hp);
        best_alpha = 1.0;
    }

    (best_d, best_alpha, best_q)
}

// ---------------------------------------------------------------------------
// Subspace minimization: truncated CG on free variables
// ---------------------------------------------------------------------------

/// Starting from the Cauchy step, run truncated CG on the QP restricted
/// to the free variables (those not active at the Cauchy point) to
/// further improve the model. Returns `(total_step, q(total_step),
/// n_cg_iters)`.
///
/// Terminates on:
///   - residual reduction `‖r_k‖ ≤ cg_tol·‖r_0‖`
///   - negative curvature (`p^T H_FF p ≤ 0`): step to TR boundary along p
///   - trust-region exit: step to TR boundary along p
///   - bound exit: step to nearest bound on a free variable
///   - `max_cg_iter` cap
#[allow(clippy::too_many_arguments)]
fn subspace_step(
    x: &[f64],
    g: &[f64],
    h: &[f64],
    l: &[f64],
    u: &[f64],
    s_cauchy: &[f64],
    q_cauchy: f64,
    active: &[i8],
    tr_radius: f64,
    cg_tol: f64,
    max_cg: usize,
    print_level: u8,
) -> (Vec<f64>, f64, usize) {
    let n = x.len();
    // Indexing on free variables only.
    let free: Vec<usize> = (0..n).filter(|&i| active[i] == 0).collect();
    let nf = free.len();

    if nf == 0 {
        // Everything active at Cauchy point — nothing to improve.
        return (s_cauchy.to_vec(), q_cauchy, 0);
    }

    // CG operates on a step `d` (in full-dim space) starting from `s_cauchy`,
    // with d_i = 0 for i ∈ active. The QP is:
    //   min_{d_F} g_F^T (s_F^C + d_F) + 0.5 (s^C + d)^T H (s^C + d)
    //         s.t. ‖s^C + d‖ ≤ Δ
    //              l − x ≤ s^C + d ≤ u − x  (componentwise)
    //
    // The reduced-system gradient at d = 0 is:
    //   r = − (g_F + (H (s^C))_F)   (negative gradient of q wrt d_F)
    // Equivalently r = − ∇_{d_F} q(s^C + d) |_{d=0}.

    let mut hs = vec![0.0; n];
    matvec(n, h, s_cauchy, &mut hs);

    // d ∈ ℝ^n with d_active = 0. r, p indexed similarly (zero on active).
    let mut d = vec![0.0; n];
    let mut r = vec![0.0; n];
    for &i in &free {
        r[i] = -(g[i] + hs[i]);
    }
    let r0_norm = dot(&r, &r).sqrt();
    if r0_norm == 0.0 {
        // s_cauchy already at the unconstrained minimum on F.
        return (s_cauchy.to_vec(), q_cauchy, 0);
    }

    let mut p = r.clone();
    let mut hp = vec![0.0; n];
    let mut r_tr = dot(&r, &r);

    let mut cg_k = 0;
    let mut hit_boundary = false;

    while cg_k < max_cg {
        cg_k += 1;
        // Hp restricted to free variables (we zero out active components
        // before/after to keep the iteration on the free subspace).
        for &i in (0..n).collect::<Vec<_>>().iter().filter(|&&i| active[i] != 0) {
            // p_active is already 0 by construction.
            debug_assert_eq!(p[i], 0.0);
        }
        matvec(n, h, &p, &mut hp);
        // Restrict Hp to free components.
        for i in 0..n { if active[i] != 0 { hp[i] = 0.0; } }

        let php = dot(&p, &hp);

        // s_total = s_cauchy + d at start of this iteration.
        // Trust-region boundary in d along p: ‖s_cauchy + d + α·p‖ = Δ.
        // Let v = s_cauchy + d; solve ‖v + α p‖² = Δ²:
        //   pp α² + 2 (v·p) α + (v·v − Δ²) = 0
        let mut v = vec![0.0; n];
        for i in 0..n { v[i] = s_cauchy[i] + d[i]; }
        let pp = dot(&p, &p);
        let vp = dot(&v, &p);
        let vv = dot(&v, &v);
        let alpha_tr = {
            let disc = vp * vp - pp * (vv - tr_radius * tr_radius);
            if pp == 0.0 || disc < 0.0 { f64::INFINITY } else {
                let s = disc.sqrt();
                let pos = (-vp + s) / pp;
                if pos >= 0.0 { pos } else { f64::INFINITY }
            }
        };

        // Negative-curvature check: take step to TR boundary and stop.
        if php <= 0.0 {
            let alpha = if alpha_tr.is_finite() { alpha_tr } else { 0.0 };
            for i in 0..n { d[i] += alpha * p[i]; }
            hit_boundary = true;
            break;
        }

        let alpha_unc = r_tr / php;

        // Bound-clipping: step that keeps s_cauchy + d + α p within [l−x, u−x].
        let mut alpha_bnd = f64::INFINITY;
        let mut hit_idx: Option<usize> = None;
        for i in 0..n {
            if active[i] != 0 || p[i] == 0.0 { continue; }
            let s_total_i = s_cauchy[i] + d[i];
            // Allowed range for s_total + α p:
            //   l[i] − x[i] ≤ s_total + α p[i] ≤ u[i] − x[i]
            if p[i] > 0.0 && u[i].is_finite() {
                let cand = (u[i] - x[i] - s_total_i) / p[i];
                if cand >= 0.0 && cand < alpha_bnd { alpha_bnd = cand; hit_idx = Some(i); }
            } else if p[i] < 0.0 && l[i].is_finite() {
                let cand = (l[i] - x[i] - s_total_i) / p[i];
                if cand >= 0.0 && cand < alpha_bnd { alpha_bnd = cand; hit_idx = Some(i); }
            }
        }

        let alpha_max = alpha_tr.min(alpha_bnd);

        if alpha_unc >= alpha_max {
            // Truncate at TR or bound.
            let alpha = alpha_max;
            for i in 0..n { d[i] += alpha * p[i]; }
            if alpha_bnd < alpha_tr {
                if print_level >= 3 {
                    eprintln!("    cg{}: bound clip i={:?} α={:.2e}",
                        cg_k, hit_idx, alpha);
                }
            } else if print_level >= 3 {
                eprintln!("    cg{}: TR boundary α={:.2e}", cg_k, alpha);
            }
            hit_boundary = true;
            break;
        }

        // Standard CG update.
        let alpha = alpha_unc;
        for i in 0..n { d[i] += alpha * p[i]; }
        for i in 0..n { r[i] -= alpha * hp[i]; }
        // Re-zero r on active vars (defensive — should already be 0).
        for i in 0..n { if active[i] != 0 { r[i] = 0.0; } }

        let r_tr_new = dot(&r, &r);
        if r_tr_new.sqrt() <= cg_tol * r0_norm {
            break;
        }
        let beta = r_tr_new / r_tr;
        for i in 0..n { p[i] = r[i] + beta * p[i]; }
        for i in 0..n { if active[i] != 0 { p[i] = 0.0; } }
        r_tr = r_tr_new;
    }

    // Total step and its model value.
    let mut s_total = vec![0.0; n];
    for i in 0..n { s_total[i] = s_cauchy[i] + d[i]; }

    let mut hs_total = vec![0.0; n];
    matvec(n, h, &s_total, &mut hs_total);
    let q_total = dot(g, &s_total) + 0.5 * dot(&s_total, &hs_total);

    // Sanity: the subspace step should not worsen the model relative to
    // the Cauchy point. If it does (numerical failure or bad cut), fall
    // back to the Cauchy step.
    if q_total > q_cauchy {
        let _ = hit_boundary;
        return (s_cauchy.to_vec(), q_cauchy, cg_k);
    }

    (s_total, q_total, cg_k)
}

// ---------------------------------------------------------------------------
// Result construction
// ---------------------------------------------------------------------------

fn success_result<P: NlpProblem>(
    problem: &P,
    x: &[f64],
    f_x: f64,
    g: &[f64],
    l: &[f64],
    u: &[f64],
    iters: usize,
    status: SolveStatus,
) -> SolveResult {
    let n = x.len();
    let m = problem.num_constraints();

    // KKT bound multipliers via complementary-slackness reading on the
    // projected gradient: at a stationary point of the BC problem,
    //   ∇f(x*)_i + z_U_i − z_L_i = 0, with z_L, z_U ≥ 0 and complementary
    //   slackness. We back them out from sign(g) on the active set.
    let mut z_l = vec![0.0; n];
    let mut z_u = vec![0.0; n];
    for i in 0..n {
        if x[i] <= l[i] && g[i] > 0.0 {
            z_l[i] = g[i];
        } else if x[i] >= u[i] && g[i] < 0.0 {
            z_u[i] = -g[i];
        }
    }

    let mut diagnostics = SolverDiagnostics::default();
    diagnostics.final_dual_inf = projected_gradient_inf_norm(x, g, l, u);
    diagnostics.final_primal_inf = 0.0; // bounds always satisfied (we project)
    diagnostics.final_compl = 0.0;       // no slack-bound complementarity in BC
    diagnostics.final_mu = 0.0;          // no barrier in TRON

    SolveResult {
        x: x.to_vec(),
        objective: f_x,
        constraint_multipliers: vec![0.0; m],
        bound_multipliers_lower: z_l,
        bound_multipliers_upper: z_u,
        constraint_values: vec![],
        status,
        iterations: iters,
        diagnostics,
    }
}

fn error_result(n: usize, status: SolveStatus, msg: &str) -> SolveResult {
    eprintln!("solve_bc: {}", msg);
    SolveResult {
        x: vec![0.0; n],
        objective: f64::NAN,
        constraint_multipliers: vec![],
        bound_multipliers_lower: vec![0.0; n],
        bound_multipliers_upper: vec![0.0; n],
        constraint_values: vec![],
        status,
        iterations: 0,
        diagnostics: SolverDiagnostics::default(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-crafted unconstrained quadratic: min 0.5 (x − x*)^T A (x − x*).
    /// A = diag(1, 4), x* = (1, 2). Optimum is x*, f* = 0.
    struct QuadProb {
        x_star: [f64; 2],
        a_diag: [f64; 2],
    }

    impl NlpProblem for QuadProb {
        fn num_variables(&self) -> usize { 2 }
        fn num_constraints(&self) -> usize { 0 }
        fn bounds(&self, l: &mut [f64], u: &mut [f64]) {
            l[0] = f64::NEG_INFINITY; l[1] = f64::NEG_INFINITY;
            u[0] = f64::INFINITY; u[1] = f64::INFINITY;
        }
        fn constraint_bounds(&self, _l: &mut [f64], _u: &mut [f64]) {}
        fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; x0[1] = 0.0; }
        fn objective(&self, x: &[f64], _: bool, obj: &mut f64) -> bool {
            let dx0 = x[0] - self.x_star[0];
            let dx1 = x[1] - self.x_star[1];
            *obj = 0.5 * (self.a_diag[0] * dx0 * dx0 + self.a_diag[1] * dx1 * dx1);
            true
        }
        fn gradient(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
            g[0] = self.a_diag[0] * (x[0] - self.x_star[0]);
            g[1] = self.a_diag[1] * (x[1] - self.x_star[1]);
            true
        }
        fn constraints(&self, _x: &[f64], _: bool, _g: &mut [f64]) -> bool { true }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
        fn jacobian_values(&self, _x: &[f64], _: bool, _v: &mut [f64]) -> bool { true }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1], vec![0, 1])
        }
        fn hessian_values(&self, _x: &[f64], _: bool, of: f64, _l: &[f64], v: &mut [f64]) -> bool {
            v[0] = of * self.a_diag[0];
            v[1] = of * self.a_diag[1];
            true
        }
    }

    #[test]
    fn unconstrained_diag_quadratic_solves_one_step() {
        // For an unconstrained quadratic with positive-definite H, TR
        // Newton (with Δ large enough) takes one Newton step → exact min.
        let prob = QuadProb { x_star: [1.0, 2.0], a_diag: [1.0, 4.0] };
        let mut opts = BcOptions::default();
        opts.initial_tr_radius = 10.0;
        let r = super::solve_bc(&prob, &opts);
        assert_eq!(r.status, SolveStatus::Optimal);
        assert!((r.x[0] - 1.0).abs() < 1e-8, "x[0]={}", r.x[0]);
        assert!((r.x[1] - 2.0).abs() < 1e-8, "x[1]={}", r.x[1]);
        assert!(r.iterations <= 3, "iters={}", r.iterations);
    }

    /// Same quadratic with bounds clipping the unconstrained minimum.
    /// x* = (1, 2) but bounds [0, 0.5]² → constrained optimum at (0.5, 0.5).
    /// Both bounds active.
    #[test]
    fn quadratic_with_active_upper_bounds() {
        let prob = QuadProb { x_star: [1.0, 2.0], a_diag: [1.0, 4.0] };
        struct Wrapped { inner: QuadProb }
        impl NlpProblem for Wrapped {
            fn num_variables(&self) -> usize { 2 }
            fn num_constraints(&self) -> usize { 0 }
            fn bounds(&self, l: &mut [f64], u: &mut [f64]) {
                l[0] = 0.0; l[1] = 0.0;
                u[0] = 0.5; u[1] = 0.5;
            }
            fn constraint_bounds(&self, _l: &mut [f64], _u: &mut [f64]) {}
            fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.25; x0[1] = 0.25; }
            fn objective(&self, x: &[f64], n: bool, o: &mut f64) -> bool { self.inner.objective(x, n, o) }
            fn gradient(&self, x: &[f64], n: bool, g: &mut [f64]) -> bool { self.inner.gradient(x, n, g) }
            fn constraints(&self, x: &[f64], n: bool, g: &mut [f64]) -> bool { self.inner.constraints(x, n, g) }
            fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { self.inner.jacobian_structure() }
            fn jacobian_values(&self, x: &[f64], n: bool, v: &mut [f64]) -> bool { self.inner.jacobian_values(x, n, v) }
            fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { self.inner.hessian_structure() }
            fn hessian_values(&self, x: &[f64], n: bool, of: f64, l: &[f64], v: &mut [f64]) -> bool {
                self.inner.hessian_values(x, n, of, l, v)
            }
        }
        let wrapped = Wrapped { inner: prob };
        let opts = BcOptions { initial_tr_radius: 10.0, ..Default::default() };
        let r = super::solve_bc(&wrapped, &opts);
        assert_eq!(r.status, SolveStatus::Optimal);
        assert!((r.x[0] - 0.5).abs() < 1e-8, "x[0]={}", r.x[0]);
        assert!((r.x[1] - 0.5).abs() < 1e-8, "x[1]={}", r.x[1]);
        // Both upper bound multipliers should be active and positive:
        // z_U_i = −g_i = a_i (x*_i − x_i) > 0 at the constrained optimum.
        assert!(r.bound_multipliers_upper[0] > 0.0);
        assert!(r.bound_multipliers_upper[1] > 0.0);
    }

    /// Rosenbrock with bounds. Standard non-quadratic test; the optimum
    /// is interior and the Hessian is highly non-uniform.
    #[test]
    fn rosenbrock_bounded() {
        struct Ros;
        impl NlpProblem for Ros {
            fn num_variables(&self) -> usize { 2 }
            fn num_constraints(&self) -> usize { 0 }
            fn bounds(&self, l: &mut [f64], u: &mut [f64]) {
                l[0] = -2.0; l[1] = -2.0;
                u[0] = 2.0;  u[1] = 2.0;
            }
            fn constraint_bounds(&self, _l: &mut [f64], _u: &mut [f64]) {}
            fn initial_point(&self, x0: &mut [f64]) { x0[0] = -1.2; x0[1] = 1.0; }
            fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool {
                let a = 1.0 - x[0];
                let b = x[1] - x[0] * x[0];
                *o = a * a + 100.0 * b * b;
                true
            }
            fn gradient(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
                let b = x[1] - x[0] * x[0];
                g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * b;
                g[1] = 200.0 * b;
                true
            }
            fn constraints(&self, _x: &[f64], _: bool, _g: &mut [f64]) -> bool { true }
            fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
            fn jacobian_values(&self, _x: &[f64], _: bool, _v: &mut [f64]) -> bool { true }
            fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
                (vec![0, 1, 1], vec![0, 0, 1])
            }
            fn hessian_values(&self, x: &[f64], _: bool, of: f64, _l: &[f64], v: &mut [f64]) -> bool {
                let b = x[1] - x[0] * x[0];
                v[0] = of * (2.0 - 400.0 * b + 800.0 * x[0] * x[0]); // d²/dx0²
                v[1] = of * (-400.0 * x[0]);                          // d²/dx0 dx1
                v[2] = of * 200.0;                                    // d²/dx1²
                true
            }
        }
        let r = super::solve_bc(&Ros, &BcOptions::default());
        assert_eq!(r.status, SolveStatus::Optimal, "iter={}", r.iterations);
        assert!((r.x[0] - 1.0).abs() < 1e-6, "x[0]={}", r.x[0]);
        assert!((r.x[1] - 1.0).abs() < 1e-6, "x[1]={}", r.x[1]);
        assert!(r.objective < 1e-12, "f={}", r.objective);
    }

    /// 1-D quadratic where the unconstrained minimum is below the lower
    /// bound: forces a single active bound. Optimum: x = lower bound.
    #[test]
    fn one_d_lower_bound_active() {
        struct Q;
        impl NlpProblem for Q {
            fn num_variables(&self) -> usize { 1 }
            fn num_constraints(&self) -> usize { 0 }
            fn bounds(&self, l: &mut [f64], u: &mut [f64]) {
                l[0] = 1.0; u[0] = 10.0;
            }
            fn constraint_bounds(&self, _l: &mut [f64], _u: &mut [f64]) {}
            fn initial_point(&self, x0: &mut [f64]) { x0[0] = 5.0; }
            fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool {
                *o = 0.5 * x[0] * x[0]; true
            }
            fn gradient(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
                g[0] = x[0]; true
            }
            fn constraints(&self, _x: &[f64], _: bool, _g: &mut [f64]) -> bool { true }
            fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
            fn jacobian_values(&self, _x: &[f64], _: bool, _v: &mut [f64]) -> bool { true }
            fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
            fn hessian_values(&self, _x: &[f64], _: bool, of: f64, _l: &[f64], v: &mut [f64]) -> bool {
                v[0] = of * 1.0; true
            }
        }
        let r = super::solve_bc(&Q, &BcOptions::default());
        assert_eq!(r.status, SolveStatus::Optimal);
        assert!((r.x[0] - 1.0).abs() < 1e-10);
        assert!(r.bound_multipliers_lower[0] > 0.5); // z_L = g(x*) = 1.0
    }

    #[test]
    fn projected_gradient_norm_basic() {
        // x = 0.5 in [0, 1], g = -2 → P[x − g] = P[0.5 + 2] = P[2.5] = 1
        // proj_grad = 1 − 0.5 = 0.5
        let n = projected_gradient_inf_norm(&[0.5], &[-2.0], &[0.0], &[1.0]);
        assert!((n - 0.5).abs() < 1e-15);

        // x = 0.0 (on lower bound), g = +2 → P[0 − 2] = P[−2] = 0
        // proj_grad = 0 − 0 = 0 (KKT-stationary at lower bound)
        let n = projected_gradient_inf_norm(&[0.0], &[2.0], &[0.0], &[1.0]);
        assert!(n < 1e-15);
    }

    #[test]
    fn density_lower_triangle_sums_correctly() {
        // 3x3 with off-diagonal: rows=[0,1,2,2], cols=[0,0,1,2]
        // L = [a 0 0; b c 0; 0 d e] -> H = [a b 0; b c d; 0 d e]
        let n = 3;
        let rows = vec![0, 1, 2, 2];
        let cols = vec![0, 0, 1, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; n * n];
        densify_lower_triangle(n, &rows, &cols, &vals, &mut out);
        // Row 0
        assert_eq!(out[0], 1.0); assert_eq!(out[1], 2.0); assert_eq!(out[2], 0.0);
        // Row 1
        assert_eq!(out[3], 2.0); assert_eq!(out[4], 0.0); assert_eq!(out[5], 3.0);
        // Row 2
        assert_eq!(out[6], 0.0); assert_eq!(out[7], 3.0); assert_eq!(out[8], 4.0);
    }
}
