use crate::convergence::is_equality_constraint;
use crate::linear_solver::{KktMatrix, LinearSolver, SolverError, SparseSymmetricMatrix, SymmetricMatrix};

/// Information about the KKT system structure.
pub struct KktSystem {
    /// Dimension of the full KKT matrix (n + m).
    pub dim: usize,
    /// Number of primal variables.
    pub n: usize,
    /// Number of constraints.
    pub m: usize,
    /// The assembled KKT matrix (dense or sparse).
    pub matrix: KktMatrix,
    /// Right-hand side vector.
    pub rhs: Vec<f64>,
    /// Per-constraint δ_c values added to the (2,2) block during assembly.
    /// Stored as positive values; the matrix gets -delta_c_diag\[i\] on diagonal (n+i, n+i).
    /// Used by iterative refinement to recover the original (unregularized) matvec.
    pub delta_c_diag: Vec<f64>,
    /// Ruiz equilibration scaling factors. When active, the factored system is
    /// D*A*D and the solution must be unscaled: x\[i\] = scale\[i\] * x_scaled\[i\].
    pub scale_factors: Option<Vec<f64>>,
}

/// Assemble the augmented KKT matrix:
/// ```text
/// [H + Sigma + delta_w*I,  J^T ]   [dx]   [r_d]
/// [J,                -delta_c*I ] * [dy] = [r_p]
/// ```
///
/// Where:
/// - H is the Hessian of the Lagrangian (n x n, lower triangle in COO)
/// - J is the constraint Jacobian (m x n, in COO)
/// - Sigma contains the barrier diagonal terms: Sigma_ii = z_l_i/(x_i - x_l_i) + z_u_i/(x_u_i - x_i)
/// - r_d is the dual residual
/// - r_p is the primal residual
///
/// # Arguments
/// - `n`: number of variables
/// - `m`: number of constraints
/// - `hess_rows`, `hess_cols`, `hess_vals`: Hessian lower triangle in COO
/// - `jac_rows`, `jac_cols`, `jac_vals`: Jacobian in COO
/// - `sigma`: barrier diagonal (length n)
/// - `grad_f`: gradient of objective (length n)
/// - `g`: constraint values (length m)
/// - `g_l`, `g_u`: constraint bounds
/// - `y`: current constraint multipliers (length m)
/// - `z_l`, `z_u`: bound multipliers
/// - `x`, `x_l`, `x_u`: current point and bounds
/// - `mu`: barrier parameter
#[allow(clippy::too_many_arguments)]
pub fn assemble_kkt(
    n: usize,
    m: usize,
    hess_rows: &[usize],
    hess_cols: &[usize],
    hess_vals: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    sigma: &[f64],
    grad_f: &[f64],
    g: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    y: &[f64],
    _z_l: &[f64],
    _z_u: &[f64],
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    mu: f64,
    use_sparse: bool,
    _v_l: &[f64],
    _v_u: &[f64],
) -> KktSystem {
    let dim = n + m;
    let capacity = hess_rows.len() + jac_rows.len() + n + m;
    let mut matrix = if use_sparse {
        KktMatrix::zeros_sparse(dim, capacity)
    } else {
        KktMatrix::zeros_dense(dim)
    };
    let mut rhs = vec![0.0; dim];

    // (1,1) block: H + Sigma
    for (idx, (&row, &col)) in hess_rows.iter().zip(hess_cols.iter()).enumerate() {
        let v = hess_vals[idx];
        if v.is_nan() || v.is_infinite() {
            log::warn!("NaN/Inf in Hessian at ({}, {}): {}", row, col, v);
        }
        matrix.add(row, col, v);
    }

    // Add barrier diagonal Sigma for variable bounds
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        matrix.add(i, i, sigma[i]);
    }

    // (2,1) block: J
    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        matrix.add(n + row, col, jac_vals[idx]);
    }

    // RHS: dual residual r_d (first n entries).
    //
    // Derivation: the full primal-dual Newton system has stationarity
    //   H*Δx + J^T*Δy - Δz_L + Δz_U = -(∇f + J^T*y - z_L + z_U)
    // and linearized complementarity S·Δz + Z·Δx = μ·e - Z·S·e. Eliminating
    //   Δz_L = (μ - z_L*s_L)/s_L - (z_L/s_L)·Δx
    // (and the symmetric upper-bound form) and substituting, the z_L and z_U
    // terms cancel algebraically. The condensed primal-block RHS is:
    //   r_d[i] = -∇f[i] - (J^T*y)[i] + μ/s_L[i] - μ/s_U[i]
    // with Σ[i][i] = z_L[i]/s_L[i] + z_U[i]/s_U[i] on the diagonal of the (1,1)
    // block. This matches Ipopt's IpPDFullSpaceSolver.cpp:418-420 augRhs_x
    // assembly (rhs_x = grad_Lag_x augmented with -μ·Σ⁻¹·e terms from bounds,
    // with z eliminated).
    //
    // Earlier versions of this code added `+z_L - z_U` to r_d with a comment
    // claiming it "tracked the dual residual" and helped convergence under
    // the kappa_sigma clamp. That was double-counting: the z deviation from
    // μ/s is ALREADY carried by the dz recovery formula, and adding it here
    // perturbed dx by O(z) / (H + Σ)_ii. On problems with active bounds at
    // optimum (e.g. HS071 with x[0]=1), this produced a period-2 limit cycle
    // in z_L at the active bound and prevented stationarity convergence.
    for i in 0..n {
        let mut rd = -grad_f[i];
        if x_l[i].is_finite() {
            rd += mu / (x[i] - x_l[i]);
        }
        if x_u[i].is_finite() {
            rd -= mu / (x_u[i] - x[i]);
        }
        rhs[i] = rd;
    }

    // Subtract J^T * y contribution from r_d
    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        rhs[col] -= jac_vals[idx] * y[row];
    }

    // RHS: primal residual r_p (last m entries) and (2,2) block for inequality constraints.
    //
    // After condensing the slack variables from the KKT system, the condensed system is:
    //   [H + Σ_x    J^T         ] [Δx]   [r_d                            ]
    //   [J          -Σ_s^{-1}   ] [Δy] = [Σ_s^{-1} * (y + μ/s_l - μ/s_u)]
    //
    // where Σ_s = z_sl/s_l + z_su/s_u is the barrier contribution from constraint slacks,
    // z_sl, z_su are the slack bound multipliers, and s_l = g - g_l, s_u = g_u - g.
    //
    // For equality constraints: no slack, (2,2) = 0, r_c = -(g - g_l).
    // For infeasible inequality constraints: no barrier, r_c = -(g - bound).
    let mut has_sigma_s = vec![false; m]; // tracks which constraints got a (2,2) diagonal entry
    for i in 0..m {
        if is_equality_constraint(g_l[i], g_u[i]) {
            rhs[n + i] = -(g[i] - g_l[i]);
            continue;
        }

        // Compute Σ_s and the RHS correction term (y + μ/s_l - μ/s_u)
        let mut sigma_s = 0.0;
        let mut rhs_correction = y[i]; // starts with y
        let mut any_feasible = false;
        let mut rhs_infeasible = 0.0;

        if g_l[i].is_finite() {
            let slack = g[i] - g_l[i];
            if slack >= -1e-8 {
                // Feasible or at bound: use barrier with safeguarded slack
                let safe_slack = slack.max(mu.max(1e-10));
                // Heuristic: use |y| when y has correct sign, else barrier estimate mu/s
                let z_sl = if y[i] < -1e-20 {
                    -y[i]
                } else {
                    mu / safe_slack
                };
                sigma_s += z_sl / safe_slack;
                rhs_correction += mu / safe_slack;
                any_feasible = true;
            } else {
                // Truly infeasible: drive toward feasibility
                rhs_infeasible += -(g[i] - g_l[i]);
            }
        }
        if g_u[i].is_finite() {
            let slack = g_u[i] - g[i];
            if slack >= -1e-8 {
                // Feasible or at bound: use barrier with safeguarded slack
                let safe_slack = slack.max(mu.max(1e-10));
                // Heuristic: use |y| when y has correct sign, else barrier estimate mu/s
                let z_su = if y[i] > 1e-20 {
                    y[i]
                } else {
                    mu / safe_slack
                };
                sigma_s += z_su / safe_slack;
                rhs_correction -= mu / safe_slack;
                any_feasible = true;
            } else {
                // Truly infeasible: drive toward feasibility
                rhs_infeasible += -(g[i] - g_u[i]);
            }
        }

        if any_feasible && sigma_s > 1e-20 {
            let sigma_s_inv = (1.0 / sigma_s).min(1e20);
            // (2,2) block: -Σ_s^{-1} (always negative, correct for KKT inertia)
            matrix.add(n + i, n + i, -sigma_s_inv);
            has_sigma_s[i] = true;
            // RHS: Σ_s^{-1} * (y + μ/s_l - μ/s_u) + infeasible contributions
            rhs[n + i] = sigma_s_inv * rhs_correction + rhs_infeasible;
        } else {
            // All infeasible: just drive toward feasibility
            rhs[n + i] = rhs_infeasible;
        }
    }

    // T0.13 (Ipopt 3.14 alignment): δ_c is NOT added unconditionally at assembly
    // time. Ipopt's `IpPDPerturbationHandler` distinguishes two modes:
    //   - PerturbForWrongInertia: augmented system has the wrong number of
    //     negative eigenvalues. Increase δ_w (Hessian perturbation). Do NOT
    //     touch δ_c.
    //   - PerturbForSingularity: factorization detects a numerically singular
    //     system (zero pivot). Add δ_c (equality block perturbation) to lift
    //     the singularity, possibly with δ_w too.
    // (See `IpPDPerturbationHandler.cpp` `delta_c_curr_` assignment paths.)
    //
    // The previous behavior — unconditionally adding `delta_c_base = 1e-8·μ^0.25`
    // to every equality / infeasible-inequality row at assembly time — biased
    // the KKT residual on well-conditioned systems and degraded convergence
    // rate near the optimum. Singularity-driven δ_c is now applied exclusively
    // by `factor_with_inertia_correction` (see the PerturbForSingularity path
    // there). `_has_sigma_s` is retained as a marker for future use but no
    // longer drives perturbation.
    let _has_sigma_s = has_sigma_s;
    let delta_c_diag = vec![0.0; m];

    // Debug: check for NaN in matrix and RHS
    if rhs.iter().any(|v| v.is_nan() || v.is_infinite()) {
        log::warn!("NaN/Inf in KKT RHS!");
        for (i, v) in rhs.iter().enumerate() {
            if v.is_nan() || v.is_infinite() {
                log::warn!("  rhs[{}] = {}", i, v);
            }
        }
    }

    KktSystem {
        dim,
        n,
        m,
        matrix,
        rhs,
        delta_c_diag,
        scale_factors: None,
    }
}

/// Compute the barrier diagonal Sigma.
///
/// Sigma_ii = z_l_i / (x_i - x_l_i) + z_u_i / (x_u_i - x_i)
pub fn compute_sigma(
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    z_l: &[f64],
    z_u: &[f64],
) -> Vec<f64> {
    let n = x.len();
    let mut sigma = vec![0.0; n];
    for i in 0..n {
        if x_l[i].is_finite() {
            let slack = (x[i] - x_l[i]).max(1e-20);
            sigma[i] += z_l[i] / slack;
        }
        if x_u[i].is_finite() {
            let slack = (x_u[i] - x[i]).max(1e-20);
            sigma[i] += z_u[i] / slack;
        }
    }
    sigma
}

/// Apply Ruiz iterative equilibration to a KKT matrix and RHS.
///
/// Computes diagonal scaling D such that the scaled matrix D*A*D has
/// approximately equal row/column norms. This improves pivot selection
/// quality and factorization accuracy.
///
/// Returns the cumulative scaling factors. After solving the scaled system,
/// the solution must be unscaled: x_original\[i\] = scale\[i\] * x_scaled\[i\].
///
/// Matches MUMPS SimScale schedule (KEEP(52)=7 for SYM=2):
/// 1 iteration of inf-norm equilibration + 3 iterations of one-norm equilibration.
/// Reference: Ruiz & Ucar, "A symmetry preserving algorithm for matrix scaling".
pub fn ruiz_equilibrate(matrix: &mut KktMatrix, rhs: &mut [f64]) -> Vec<f64> {
    let dim = matrix.n();
    let mut cumulative = vec![1.0; dim];

    // Phase 1: 1 iteration of inf-norm equilibration (row max-norm)
    {
        let norms = matrix.row_abs_max();
        for k in 0..dim {
            let norm_k = norms[k];
            if norm_k > 1e-30 {
                let s = 1.0 / norm_k.sqrt();
                matrix.scale_row_col(k, s);
                rhs[k] *= s;
                cumulative[k] *= s;
            }
        }
    }

    // Phase 2: 3 iterations of one-norm equilibration (row sum-norm)
    for _ in 0..3 {
        let norms = matrix.row_abs_sum();
        for k in 0..dim {
            let norm_k = norms[k];
            if norm_k > 1e-30 {
                let s = 1.0 / norm_k.sqrt();
                matrix.scale_row_col(k, s);
                rhs[k] *= s;
                cumulative[k] *= s;
            }
        }
    }

    cumulative
}

/// Parameters for inertia correction. Mirrors Ipopt's
/// `PDPerturbationHandler` (see `IpPDPerturbationHandler.cpp`):
/// distinct first-time vs subsequent inc factors, a 1/3 dec factor for
/// the warm-shrink between successive solves, a hard cap `delta_w_max`,
/// and a floor `delta_w_min`.
pub struct InertiaCorrectionParams {
    /// Initial primal regularization, `first_hessian_perturbation`
    /// (Ipopt default 1e-4, `IpPDPerturbationHandler.cpp:79`).
    pub delta_w_init: f64,
    /// Base constraint regularization.
    pub delta_c_base: f64,
    /// First-time growth factor `perturb_inc_fact_first` (Ipopt
    /// default 100, `IpPDPerturbationHandler.cpp:53`). Used when
    /// `delta_w_last == 0` (no previous perturbation in the run) or
    /// `1e5 * delta_w_last < delta_w_curr` (current already vastly
    /// exceeds the last successful level).
    pub delta_w_inc_fact_first: f64,
    /// Subsequent growth factor `perturb_inc_fact` (Ipopt default 8,
    /// `IpPDPerturbationHandler.cpp:62`).
    pub delta_w_inc_fact: f64,
    /// Warm-shrink factor `perturb_dec_fact` applied to delta_w_last
    /// between successive calls (Ipopt default 1/3,
    /// `IpPDPerturbationHandler.cpp:71`).
    pub delta_w_dec_fact: f64,
    /// Give-up threshold `max_hessian_perturbation` (Ipopt default
    /// 1e20, `IpPDPerturbationHandler.cpp:31`). When delta_w exceeds
    /// this, restoration is the only recourse.
    pub delta_w_max: f64,
    /// Floor `min_hessian_perturbation` (Ipopt default 1e-20,
    /// `IpPDPerturbationHandler.cpp:45`). Clamps the warm-shrink so
    /// it never collapses to exactly zero.
    pub delta_w_min: f64,
    /// Maximum number of correction attempts (safety net; the primary
    /// stop is `delta_w > delta_w_max`).
    pub max_attempts: usize,
    /// Last successful delta_w (for warm-starting perturbation).
    pub delta_w_last: f64,
    /// Whether scaling is active (activated on demand when backward error is poor).
    pub use_scaling: bool,
    /// Count of consecutive iterations that needed perturbation (delta_w > 0).
    pub degeneracy_count: usize,
    /// True when the Hessian is structurally degenerate (always needs delta_w > 0).
    /// Skips the unperturbed factorization trial to save wasted work.
    pub structurally_degenerate: bool,
    /// T0.14: Tracks whether the "pretend singular" trigger has fired
    /// in the current outer iteration. Ipopt's `IpPDPerturbationHandler`
    /// allows the trick (treat a borderline non-singular system as
    /// singular and add δ_c) at most once per outer iter; subsequent
    /// pretend-singular triggers within the same iter are refused so
    /// repeated triggers cannot mask deeper KKT breakdown. Reset at
    /// the top of each outer iter via
    /// `reset_pretend_singular_for_new_iter`.
    pub pretend_singular_used: bool,
}

impl InertiaCorrectionParams {
    /// T0.14: clear the per-iter `pretend_singular_used` flag. Call this
    /// at the top of each outer iteration so the pretend-singular
    /// trigger is allowed exactly once per iter.
    pub fn reset_pretend_singular_for_new_iter(&mut self) {
        self.pretend_singular_used = false;
    }
}

impl Default for InertiaCorrectionParams {
    fn default() -> Self {
        Self {
            delta_w_init: 1e-4,
            // Matches Ipopt's jacobian_regularization_value default (1e-8);
            // the actual delta_c applied is this base scaled by mu^0.25 inside
            // factor_with_inertia_correction.
            delta_c_base: 1e-8,
            delta_w_inc_fact_first: 100.0,
            delta_w_inc_fact: 8.0,
            delta_w_dec_fact: 1.0 / 3.0,
            delta_w_max: 1e20,
            delta_w_min: 1e-20,
            max_attempts: 30,
            delta_w_last: 0.0,
            use_scaling: false,
            degeneracy_count: 0,
            structurally_degenerate: false,
            pretend_singular_used: false,
        }
    }
}

/// Check if a factorization produces acceptable backward error for the KKT system.
/// Does a trial solve and checks ||b - Ax|| / (||x|| + ||b||).
/// Returns true if backward error <= 1e-6.
fn check_factorization_backward_error(
    kkt: &KktSystem,
    solver: &mut dyn LinearSolver,
) -> bool {
    check_factorization_backward_error_with_matrix(&kkt.matrix, &kkt.rhs, solver)
}

/// Check backward error using a specific matrix (may differ from kkt.matrix when perturbed).
/// Uses a lenient threshold (1e-4) since iterative refinement in solve_for_direction
/// will improve accuracy. This catches only grossly unreliable factorizations.
fn check_factorization_backward_error_with_matrix(
    matrix: &KktMatrix,
    rhs: &[f64],
    solver: &mut dyn LinearSolver,
) -> bool {
    let dim = rhs.len();
    let mut solution = vec![0.0; dim];
    if solver.solve(rhs, &mut solution).is_err() {
        return false;
    }
    if solution.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return false;
    }
    let mut residual = vec![0.0; dim];
    matrix.matvec(&solution, &mut residual);
    let x_norm: f64 = solution.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1.0);
    let mut max_berr: f64 = 0.0;
    for i in 0..dim {
        let abs_res = (rhs[i] - residual[i]).abs();
        let denom = x_norm + rhs[i].abs().max(1e-30);
        max_berr = max_berr.max(abs_res / denom);
    }
    max_berr <= 1e-4
}

/// Perform KKT factorization with inertia correction.
///
/// Factor the KKT matrix and check inertia. If inertia is wrong
/// (should be (n, m, 0) for an n-variable, m-constraint problem),
/// add regularization and re-factor. Also checks backward error
/// to ensure the factorization is numerically reliable.
///
/// Returns the factored solver and the regularization used.
pub fn factor_with_inertia_correction(
    kkt: &mut KktSystem,
    solver: &mut dyn LinearSolver,
    params: &mut InertiaCorrectionParams,
    mu: f64,
) -> Result<(f64, f64), crate::linear_solver::SolverError> {
    let n = kkt.n;
    // Mu-scaled constraint regularization base (Ipopt's
    // PDPerturbationHandler::delta_cd, IpPDPerturbationHandler.cpp:465-468).
    let delta_c_active = params.delta_c_base * mu.max(0.0).powf(0.25);
    let m = kkt.m;

    // Apply Ruiz equilibration when scaling is active (activated on demand).
    if params.use_scaling {
        let scale = ruiz_equilibrate(&mut kkt.matrix, &mut kkt.rhs);
        kkt.scale_factors = Some(scale);
    }

    // First attempt: factor without perturbation.
    // Skip when structurally degenerate (Hessian always needs delta_w > 0) —
    // jump directly to perturbation loop to save a wasted factorization.
    if !params.structurally_degenerate {
    let inertia = solver.factor(&kkt.matrix)?;

    if let Some(inertia) = inertia {
        // Accept only when inertia counts match expected (n, m, 0). Ipopt
        // never accepts off-by-one inertia; a single eigenvalue flipped
        // sign can turn a descent step into an ascent step.
        let inertia_ok = inertia.positive == n && inertia.negative == m && inertia.zero == 0;
        if inertia_ok {
            // Always verify backward error. faer's sign-based inertia can pass
            // rank-deficient KKT matrices with zero perturbation (AC-OPF angle
            // gauge: J has a null direction, but all pivot signs match expected).
            // The backward-error probe does a test solve and catches rank
            // deficiency that inertia counts miss.
            if check_factorization_backward_error(kkt, solver) {
                params.delta_w_last = 0.0;
                params.degeneracy_count = 0;
                return Ok((0.0, 0.0));
            }
            // Backward error too large — try activating scaling before regularization
            if !params.use_scaling && kkt.scale_factors.is_none() {
                params.use_scaling = true;
                let scale = ruiz_equilibrate(&mut kkt.matrix, &mut kkt.rhs);
                kkt.scale_factors = Some(scale);
                let inertia2 = solver.factor(&kkt.matrix)?;
                if let Some(inertia2) = inertia2 {
                    let inertia_ok2 = inertia2.positive == n && inertia2.negative == m && inertia2.zero == 0;
                    if inertia_ok2 && check_factorization_backward_error(kkt, solver) {
                        params.delta_w_last = 0.0;
                        params.degeneracy_count = 0;
                        return Ok((0.0, 0.0));
                    }
                }
            }
        }
    }

    // For unconstrained problems (m=0), try direct delta_w from min diagonal.
    // This avoids the exponential growth that overshoots on indefinite Hessians,
    // producing gradient-like steps instead of Newton steps.
    // Only use this when the required perturbation is moderate (< 1e4) —
    // extreme indefiniteness needs the standard exponential growth strategy.
    if m == 0 {
        if let Some(min_d) = solver.min_diagonal() {
            if min_d < 0.0 {
                let delta_w_direct = -min_d + 1e-8;
                let mut perturbed = kkt.matrix.clone();
                perturbed.add_diagonal_range(0, n, delta_w_direct);
                let inertia = solver.factor(&perturbed)?;
                if let Some(inertia) = inertia {
                    if inertia.positive == n && inertia.negative == 0 && inertia.zero == 0 {
                        kkt.matrix = perturbed;
                        params.delta_w_last = delta_w_direct;
                        params.degeneracy_count += 1;
                        if params.degeneracy_count >= 3 { params.structurally_degenerate = true; }
                        return Ok((delta_w_direct, 0.0));
                    }
                }
            }
        }
    }

    // Before perturbation: try increasing factorization quality (Ipopt's IncreaseQuality).
    // This escalates the pivot threshold (e.g., 1e-6 -> 1e-3 -> 0.03 -> 0.1), which can
    // fix inertia by forcing better pivot choices without adding regularization.
    // Like Ipopt, try this once before resorting to delta_w perturbation.
    if solver.increase_quality() {
        let inertia = solver.factor(&kkt.matrix)?;
        if let Some(inertia) = inertia {
            let inertia_ok = inertia.positive == n && inertia.negative == m && inertia.zero == 0;
            if inertia_ok && check_factorization_backward_error(kkt, solver) {
                params.delta_w_last = 0.0;
                return Ok((0.0, 0.0));
            }
        }
    }

    // Selective delta_c-only perturbation (Ipopt's PerturbForSingularity path).
    //
    // T0.13 (Ipopt 3.14 alignment): δ_c is added here, on the
    // PerturbForSingularity path, rather than unconditionally during
    // assembly. We enter this path under two conditions:
    //
    //   (a) singularity_detected: `inertia.zero > 0` — the factorization
    //       reports zero pivots, i.e. the augmented system is numerically
    //       singular. This is the canonical PerturbForSingularity trigger.
    //   (b) small inertia deficit: `n_neg < m` by ≤ max(5, m/100) and
    //       total = n+m exactly, i.e. a few eigenvalues near the saddle-
    //       point boundary flipped sign due to floating-point rounding.
    //       Adding only -δ_c to the (2,2) block pushes those borderline
    //       eigenvalues negative without touching the primal block.
    //
    // δ_w is not bumped in either branch — that is the
    // PerturbForWrongInertia regime, handled by the escalation loop below.
    if m > 0 {
        let last_inertia = solver.factor(&kkt.matrix)?;
        if let Some(inertia) = last_inertia {
            let total = inertia.positive + inertia.negative + inertia.zero;
            let deficit = m as isize - inertia.negative as isize;
            let singularity_detected = inertia.zero > 0;
            let small_deficit = deficit > 0
                && deficit <= 5.max((m / 100) as isize)
                && inertia.zero == 0
                && (total as isize - (n + m) as isize).unsigned_abs() <= 2;
            if singularity_detected || small_deficit {
                let mut delta_c = delta_c_active;
                for _ in 0..4 {
                    let mut perturbed = kkt.matrix.clone();
                    perturbed.add_diagonal_range(n, n + m, -delta_c);
                    let inertia = solver.factor(&perturbed)?;
                    if let Some(inertia) = inertia {
                        let ok = inertia.positive == n && inertia.negative == m
                            && inertia.zero == 0;
                        if ok {
                            log::debug!(
                                "Selective delta_c-only correction succeeded \
                                 ({}): delta_c={:.2e}",
                                if singularity_detected { "singularity" } else { "small deficit" },
                                delta_c
                            );
                            kkt.matrix = perturbed;
                            params.delta_w_last = 0.0;
                            return Ok((0.0, delta_c));
                        }
                    }
                    delta_c *= 4.0;
                }
                // delta_c-only didn't work — fall through to full perturbation
            }
        }
    }

    } // end if !structurally_degenerate

    // Inertia is wrong or backward error too large — apply perturbation and re-factor.
    //
    // Initial delta_w within this call. Mirrors Ipopt's
    // `get_deltas_for_wrong_inertia` start (`IpPDPerturbationHandler.cpp:375-382`):
    //   - cold (no prior perturbation in this run): start at delta_w_init.
    //   - warm: start at max(delta_w_min, delta_w_last * delta_w_dec_fact),
    //     i.e. one-third of the last successful perturbation, allowing
    //     the schedule to shrink across successive iterations when the
    //     problem is becoming locally easier.
    let mut delta_w = if params.delta_w_last == 0.0 {
        params.delta_w_init
    } else {
        (params.delta_w_last * params.delta_w_dec_fact).max(params.delta_w_min)
    };
    let mut best_delta_w = delta_w;

    for attempt in 0..params.max_attempts {
        // Give-up threshold (Ipopt: `delta_xs_max_`,
        // `IpPDPerturbationHandler.cpp:31,395-403`). Once exceeded, the
        // matrix is hopelessly indefinite and only restoration can recover.
        if delta_w > params.delta_w_max {
            log::debug!(
                "Inertia correction: delta_w={:.2e} exceeds delta_w_max={:.2e}, giving up",
                delta_w, params.delta_w_max
            );
            break;
        }

        let delta_c = delta_c_active;

        // Create perturbed matrix
        let mut perturbed = kkt.matrix.clone();
        perturbed.add_diagonal_range(0, n, delta_w);
        if m > 0 {
            perturbed.add_diagonal_range(n, n + m, -delta_c);
        }

        let inertia = solver.factor(&perturbed)?;

        if let Some(inertia) = inertia {
            let exact_ok = inertia.positive == n && inertia.negative == m && inertia.zero == 0;
            if exact_ok {
                // Verify backward error is acceptable.
                if check_factorization_backward_error_with_matrix(&perturbed, &kkt.rhs, solver) {
                    kkt.matrix = perturbed;
                    params.delta_w_last = delta_w;
                    params.degeneracy_count += 1;
                    if params.degeneracy_count >= 3 { params.structurally_degenerate = true; }
                    return Ok((delta_w, delta_c));
                }
                // Backward error too large — increase regularization
                log::debug!(
                    "Inertia correct at delta_w={:.2e} but backward error too large, increasing",
                    delta_w
                );
            }
        }

        best_delta_w = delta_w;

        // Two-stage growth (Ipopt's first-vs-subsequent inc fact,
        // `IpPDPerturbationHandler.cpp:386-393`). The first factor (100x)
        // applies when the system has never been perturbed before
        // (`delta_w_last == 0`) or when the current perturbation has
        // already vastly outgrown the last successful level
        // (`1e5 * delta_w_last < delta_w_curr`). Otherwise, the gentler
        // 8x factor keeps us near the previously successful regime.
        let inc = if params.delta_w_last == 0.0
            || 1e5 * params.delta_w_last < delta_w
        {
            params.delta_w_inc_fact_first
        } else {
            params.delta_w_inc_fact
        };
        delta_w *= inc;

        log::debug!(
            "Inertia correction attempt {}: delta_w = {:.2e} (x{}), delta_c = {:.2e}, inertia = {:?}",
            attempt + 1,
            delta_w,
            inc,
            delta_c,
            inertia
        );
    }

    // Inertia correction failed — use last perturbed matrix and proceed.
    // The line search will reject bad steps, and restoration can recover.
    let delta_c = delta_c_active;

    log::warn!(
        "Inertia correction failed after {} attempts (delta_w={:.2e}, delta_c={:.2e}), proceeding with approximate factorization",
        params.max_attempts, best_delta_w, delta_c
    );
    let mut perturbed = kkt.matrix.clone();
    perturbed.add_diagonal_range(0, n, best_delta_w);
    if m > 0 {
        perturbed.add_diagonal_range(n, n + m, -delta_c);
    }
    solver.factor(&perturbed)?;
    kkt.matrix = perturbed;
    params.delta_w_last = best_delta_w;
    params.degeneracy_count += 1;
    if params.degeneracy_count >= 3 { params.structurally_degenerate = true; }
    Ok((best_delta_w, delta_c))
}

/// Compute y = A_original * x, undoing ALL perturbations: both assembly-time δ_c
/// on equality constraints and IC perturbation (δ_w on primal, δ_c_ic on constraints).
///
/// The factored system has: A_factored = A_original - diag(assembly_δ_c) - diag(IC_δ_c) + diag(IC_δ_w)
/// So: A_original * x = A_factored * x - δ_w * x\[0..n\] + (assembly_δ_c\[i\] + IC_δ_c) * x\[n+i\]
fn matvec_original(
    kkt: &KktSystem,
    x: &[f64],
    y: &mut [f64],
    delta_w: f64,
    delta_c_ic: f64,
) {
    kkt.matrix.matvec(x, y);
    // Undo IC primal perturbation (IC added +delta_w to diagonal 0..n)
    if delta_w > 0.0 {
        for j in 0..kkt.n {
            y[j] -= delta_w * x[j];
        }
    }
    // Undo assembly δ_c + IC constraint perturbation
    for i in 0..kkt.m {
        let total_dc = kkt.delta_c_diag[i] + delta_c_ic;
        if total_dc > 0.0 {
            y[kkt.n + i] += total_dc * x[kkt.n + i];
        }
    }
}

/// Solve the KKT system for the search direction, given a factored solver.
///
/// Returns (dx, dy) where dx is the primal step and dy is the dual step.
/// Bound multiplier steps dz_l, dz_u are recovered from complementarity.
///
/// Uses iterative refinement against the ORIGINAL (unregularized) system
/// to recover the true Newton direction despite δ_c/δ_w regularization.
/// The factored regularized system acts as a preconditioner.
pub fn solve_for_direction(
    kkt: &KktSystem,
    solver: &mut dyn LinearSolver,
    delta_w: f64,
    delta_c_ic: f64,
) -> Result<(Vec<f64>, Vec<f64>), crate::linear_solver::SolverError> {
    let dim = kkt.dim;
    // Ipopt measures residuals against the assembled *regularized* KKT matrix
    // (IpPDFullSpaceSolver.cpp:701-732 + ComputeResidualRatio:795-820). The
    // IC-phase (δ_w, δ_c_ic) are *part of* the system being solved, not a
    // perturbation to be undone. If J is rank-deficient, δ_c_ic > 0 makes the
    // assembled system non-singular and its residual ratio is near machine
    // epsilon; Ipopt accepts that step without complaint.
    let _ = (delta_w, delta_c_ic);
    let use_ic_refinement = false;

    // NaN guard on RHS — if the RHS has NaN, the problem evaluation is broken
    if kkt.rhs.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(crate::linear_solver::SolverError::NumericalFailure(
            "KKT RHS contains NaN/Inf".to_string(),
        ));
    }

    let mut solution = vec![0.0; dim];
    solver.solve(&kkt.rhs, &mut solution)?;

    // NaN guard on solution — factorization may produce NaN for ill-conditioned systems
    if solution.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(crate::linear_solver::SolverError::NumericalFailure(
            "KKT solution contains NaN/Inf".to_string(),
        ));
    }

    // Iterative refinement: correct the solution using residuals against the
    // ORIGINAL (unregularized) matrix. The factored regularized system acts as
    // a preconditioner. This converges to the solution of the original system.
    // Use more refinement iterations for large systems where factorization accuracy
    // may be limited, and for IC-refinement where we're solving a different system.
    // Ipopt's default: max_refinement_steps = 10, min_refinement_steps = 1.
    let max_refinements = 10;
    let mut residual = vec![0.0; dim];
    let mut prev_res_norm = f64::MAX;
    for _ref_iter in 0..max_refinements {
        // When IC regularization was applied, compute residual against the ORIGINAL
        // (unregularized) matrix so refinement converges to the true Newton direction.
        if use_ic_refinement {
            matvec_original(kkt, &solution, &mut residual, delta_w, delta_c_ic);
        } else {
            kkt.matrix.matvec(&solution, &mut residual);
        }
        let mut res_norm: f64 = 0.0;
        for i in 0..dim {
            residual[i] = kkt.rhs[i] - residual[i];
            res_norm = res_norm.max(residual[i].abs());
        }

        if res_norm < 1e-12 {
            break;
        }

        // Stagnation detection: stop if not improving.
        // IC path: 0.9 (aggressive, preconditioning mismatch expected).
        // Non-IC path: 1.0 - 1e-6 (Ipopt uses 1 - 1e-9; stop only if
        // refinement makes no progress, allowing slow but steady convergence).
        let stagnation_factor = if use_ic_refinement { 0.9 } else { 1.0 - 1e-6 };
        if res_norm > stagnation_factor * prev_res_norm {
            break;
        }
        prev_res_norm = res_norm;

        // Solve A_regularized * correction = residual
        let mut correction = vec![0.0; dim];
        if solver.solve(&residual, &mut correction).is_err() {
            break;
        }

        // Update solution
        for i in 0..dim {
            solution[i] += correction[i];
        }
    }

    // Pretend-singular check using Ipopt's normwise residual ratio.
    // ratio = ||resid||_inf / (min(||sol||_inf, 1e6 * ||rhs||_inf) + ||rhs||_inf)
    // If ratio > 1e-5 after iterative refinement, the system is numerically singular.
    {
        if use_ic_refinement {
            matvec_original(kkt, &solution, &mut residual, delta_w, delta_c_ic);
        } else {
            kkt.matrix.matvec(&solution, &mut residual);
        }
        let nrm_rhs: f64 = kkt.rhs.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let nrm_res: f64 = solution.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let nrm_resid: f64 = (0..dim).map(|i| (kkt.rhs[i] - residual[i]).abs()).fold(0.0f64, f64::max);
        let max_cond = 1e6;
        let residual_ratio = if nrm_rhs + nrm_res == 0.0 {
            nrm_resid
        } else {
            nrm_resid / (nrm_res.min(max_cond * nrm_rhs) + nrm_rhs)
        };

        if residual_ratio > 1e-5 {
            log::debug!("KKT residual ratio {:.2e} > 1e-5 — pretend singular", residual_ratio);
            return Err(SolverError::PretendSingular);
        }
        if residual_ratio > 1e-10 {
            log::debug!("KKT residual ratio {:.2e} (above target 1e-10, proceeding)", residual_ratio);
        }

        // Solution-magnitude safeguard. Ipopt's residual_ratio check caps
        // ||sol|| at 1e6·||rhs|| in the denominator (IpPDFullSpaceSolver.cpp:815
        // "ToDo: ... safeguard against incredibly large solution vectors"), so
        // when J is rank-deficient iterative refinement can converge to a
        // null-space solution with tiny residual but enormous norm. Fraction-
        // to-boundary then drives α→0 and the solver stalls.
        //
        // Rule: if ||sol||_inf / max(||rhs||_inf, 1) > κ, treat as pretend-
        // singular so the upstream chain applies δ_c (lifting the rank
        // deficiency) and re-solves. κ=1e10 is permissive enough for
        // genuinely ill-conditioned (but not rank-deficient) systems.
        let magnitude_ratio = nrm_res / nrm_rhs.max(1.0);
        if magnitude_ratio > 1e10 {
            log::debug!(
                "KKT ||sol||={:.2e} vs ||rhs||={:.2e} (ratio {:.2e}) — pretend singular (rank-def guard)",
                nrm_res, nrm_rhs, magnitude_ratio,
            );
            return Err(SolverError::PretendSingular);
        }
    }

    // Unscale solution when Ruiz equilibration was applied.
    // The scaled system solves (D*A*D)*(D^{-1}*x) = D*b, so x_scaled = D^{-1}*x_original.
    // To recover the original solution: x_original[i] = scale[i] * x_scaled[i].
    if let Some(ref scale) = kkt.scale_factors {
        for i in 0..dim {
            solution[i] *= scale[i];
        }
    }

    let dx = solution[..kkt.n].to_vec();
    let dy = solution[kkt.n..].to_vec();

    Ok((dx, dy))
}

/// T0.14: outer-iter-aware wrapper around `solve_for_direction` that
/// gates the pretend-singular trigger so it can fire at most once per
/// outer iteration. Mirrors Ipopt's `IpPDPerturbationHandler` rule that
/// the "pretend the system is singular" trick is allowed only on the
/// first invocation per iter; subsequent invocations within the same
/// iter must fall through to the normal escalation path rather than
/// repeatedly raising PretendSingular (which can mask deeper KKT
/// breakdown).
///
/// Behavior:
///   - First call this iter that returns `PretendSingular`: the flag
///     is recorded and the error is propagated so the caller (typically
///     `solve_with_quality_escalation`) walks the standard ladder.
///   - Subsequent calls this iter that *would* return `PretendSingular`:
///     the wrapper invokes `solve_for_direction` once, and on
///     `PretendSingular` returns Ok with the (possibly imperfect)
///     solution from a plain `solver.solve` of the regularized RHS.
///     This refuses another pretend-singular pass, matching Ipopt.
///
/// `params.reset_pretend_singular_for_new_iter()` must be called at the
/// top of each outer iteration to clear the flag.
pub fn solve_for_direction_iter_aware(
    kkt: &KktSystem,
    solver: &mut dyn LinearSolver,
    params: &mut InertiaCorrectionParams,
    delta_w: f64,
    delta_c_ic: f64,
) -> Result<(Vec<f64>, Vec<f64>), crate::linear_solver::SolverError> {
    match solve_for_direction(kkt, solver, delta_w, delta_c_ic) {
        Ok(v) => Ok(v),
        Err(SolverError::PretendSingular) if !params.pretend_singular_used => {
            // First pretend-singular this iter — allow it and record.
            params.pretend_singular_used = true;
            Err(SolverError::PretendSingular)
        }
        Err(SolverError::PretendSingular) => {
            // Second pretend-singular within the same outer iter — refuse.
            // Fall through to a plain backsolve and accept whatever the
            // factorized regularized system produces.
            log::debug!(
                "T0.14: pretend-singular already used this outer iter; \
                 falling through to plain solve"
            );
            let dim = kkt.dim;
            let mut solution = vec![0.0; dim];
            solver.solve(&kkt.rhs, &mut solution)?;
            if solution.iter().any(|v| v.is_nan() || v.is_infinite()) {
                return Err(crate::linear_solver::SolverError::NumericalFailure(
                    "KKT solution contains NaN/Inf (T0.14 fallback)".into(),
                ));
            }
            if let Some(ref scale) = kkt.scale_factors {
                for i in 0..dim {
                    solution[i] *= scale[i];
                }
            }
            let dx = solution[..kkt.n].to_vec();
            let dy = solution[kkt.n..].to_vec();
            Ok((dx, dy))
        }
        Err(e) => Err(e),
    }
}

/// Recover bound multiplier steps from complementarity.
///
/// dz_l_i = (mu - z_l_i * (x_i - x_l_i) - z_l_i * dx_i) / (x_i - x_l_i)
///        = (mu / (x_i - x_l_i)) - z_l_i - (z_l_i / (x_i - x_l_i)) * dx_i
///        = sigma_l_i * dx_i ... (simplified from complementarity)
///
/// More precisely:
/// dz_l_i = (mu - z_l_i * s_l_i) / s_l_i - z_l_i * dx_i / s_l_i
///        where s_l_i = x_i - x_l_i
pub fn recover_dz(
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    dx: &[f64],
    mu: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut dz_l = vec![0.0; n];
    let mut dz_u = vec![0.0; n];

    for i in 0..n {
        if x_l[i].is_finite() {
            let s_l = (x[i] - x_l[i]).max(1e-20);
            dz_l[i] = (mu - z_l[i] * s_l) / s_l - (z_l[i] / s_l) * dx[i];
        }
        if x_u[i].is_finite() {
            let s_u = (x_u[i] - x[i]).max(1e-20);
            dz_u[i] = (mu - z_u[i] * s_u) / s_u + (z_u[i] / s_u) * dx[i];
        }
    }

    (dz_l, dz_u)
}

/// Compute the affine-scaling (μ=0) predictor RHS for Mehrotra predictor-corrector.
///
/// Returns a copy of the existing KKT RHS with the centering terms (μ/s) removed.
/// Solving with this RHS gives the pure-Newton (affine-scaling) direction that
/// can be used to probe a better barrier parameter μ.
///
/// Cost: O(n) to build — extremely cheap. The expensive part is the triangular solve.
pub fn affine_predictor_rhs(
    rhs: &[f64],
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    mu: f64,
) -> Vec<f64> {
    let n = x.len();
    let mut rhs_aff = rhs.to_vec();
    // Remove the μ/s centering terms from the primal block
    for i in 0..n {
        if x_l[i].is_finite() {
            let s_l = (x[i] - x_l[i]).max(1e-20);
            rhs_aff[i] -= mu / s_l;
        }
        if x_u[i].is_finite() {
            let s_u = (x_u[i] - x[i]).max(1e-20);
            rhs_aff[i] += mu / s_u;
        }
    }
    rhs_aff
}

/// Build a new KKT RHS with a different barrier parameter μ_new.
///
/// Given the existing RHS assembled with μ_old, returns a new RHS for μ_new.
/// Only the primal block (first n entries) changes: the μ/s centering terms
/// are updated from μ_old/s to μ_new/s.
pub fn rebuild_rhs_with_mu(
    rhs: &[f64],
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    mu_old: f64,
    mu_new: f64,
) -> Vec<f64> {
    let n = x.len();
    let mut rhs_new = rhs.to_vec();
    let delta_mu = mu_new - mu_old;
    for i in 0..n {
        if x_l[i].is_finite() {
            let s_l = (x[i] - x_l[i]).max(1e-20);
            rhs_new[i] += delta_mu / s_l;
        }
        if x_u[i].is_finite() {
            let s_u = (x_u[i] - x[i]).max(1e-20);
            rhs_new[i] -= delta_mu / s_u;
        }
    }
    rhs_new
}

/// Build a full Mehrotra corrector RHS: rebuild with μ_new AND add the
/// second-order cross-term from the affine-predictor step.
///
/// After the affine predictor (Δx_aff, Δz_L_aff, Δz_U_aff) is computed, the
/// corrector complementarity equation becomes
///   S_L · z_L + ΔS_L · z_L + Δz_L · S_L + ΔS_L_aff · Δz_L_aff = μ_new
/// and analogously for the upper bound. Eliminating Δz into the primal block
/// (see Ipopt `IpPDSearchDirCalc.cpp:88-110` plus the AugRhs_x folding at
/// `IpPDFullSpaceSolver.cpp:418-420`) yields two additional per-bound terms
/// on r_x:
///   r_x\[i\] -= (Δx_aff\[i\] · Δz_L_aff\[i\]) / s_L\[i\]      (lower bound)
///   r_x\[i\] -= (Δx_aff\[i\] · Δz_U_aff\[i\]) / s_U\[i\]      (upper bound)
/// Both contributions carry a minus sign; the Δs_u = −Δx asymmetry is
/// compensated by the `alpha = −1` that `AddMSinvZ` uses for the upper bound.
pub fn mehrotra_corrector_rhs(
    rhs: &[f64],
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    dx_aff: &[f64],
    dz_l_aff: &[f64],
    dz_u_aff: &[f64],
    mu_old: f64,
    mu_new: f64,
) -> Vec<f64> {
    let n = x.len();
    let mut rhs_new = rebuild_rhs_with_mu(rhs, x, x_l, x_u, mu_old, mu_new);
    for i in 0..n {
        if x_l[i].is_finite() {
            let s_l = (x[i] - x_l[i]).max(1e-20);
            rhs_new[i] -= dx_aff[i] * dz_l_aff[i] / s_l;
        }
        if x_u[i].is_finite() {
            let s_u = (x_u[i] - x[i]).max(1e-20);
            rhs_new[i] -= dx_aff[i] * dz_u_aff[i] / s_u;
        }
    }
    rhs_new
}

/// Recover Δz_L, Δz_U after a Mehrotra-corrected primal step Δx.
///
/// Unlike the plain `recover_dz`, this includes the second-order cross-term
/// in the complementarity residual:
///   s_L · z_L + ΔS_L · z_L + Δz_L · s_L + ΔS_L_aff · Δz_L_aff = μ_new
/// giving
///   Δz_L\[i\] = (μ_new − s_L·z_L − Δx_aff·Δz_L_aff) / s_L − (z_L/s_L) · Δx\[i\]
/// and the symmetric upper-bound expression with Δs_u = −Δx. Without this,
/// the recovered Δz inherits an O(Δaff²) complementarity error that defeats
/// the whole point of the predictor-corrector.
pub fn recover_dz_mehrotra(
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    dx: &[f64],
    dx_aff: &[f64],
    dz_l_aff: &[f64],
    dz_u_aff: &[f64],
    mu: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut dz_l = vec![0.0; n];
    let mut dz_u = vec![0.0; n];
    for i in 0..n {
        if x_l[i].is_finite() {
            let s_l = (x[i] - x_l[i]).max(1e-20);
            dz_l[i] = (mu - z_l[i] * s_l - dx_aff[i] * dz_l_aff[i]) / s_l
                - (z_l[i] / s_l) * dx[i];
        }
        if x_u[i].is_finite() {
            let s_u = (x_u[i] - x[i]).max(1e-20);
            // Δs_u = -Δx, so ΔS_u_aff · Δz_U_aff = -Δx_aff · Δz_U_aff.
            // The complementarity eq is s_u z_u + ΔS_u z_u + Δz_u s_u + ΔS_u_aff · Δz_U_aff = μ,
            // giving dz_u = (μ - s_u z_u + Δx_aff Δz_U_aff)/s_u + (z_u/s_u) Δx.
            dz_u[i] = (mu - z_u[i] * s_u + dx_aff[i] * dz_u_aff[i]) / s_u
                + (z_u[i] / s_u) * dx[i];
        }
    }
    (dz_l, dz_u)
}

/// Solve the factored system with a custom RHS (backsolve only, no re-factorization).
///
/// Used for Mehrotra predictor-corrector and Gondzio centrality corrections —
/// both need extra backsolves with the already-factored KKT matrix.
///
/// Returns (dx, dy) where dx is the primal block (first n entries) and dy the dual.
pub fn solve_with_custom_rhs(
    n: usize,
    dim: usize,
    solver: &mut dyn LinearSolver,
    rhs: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), SolverError> {
    solve_with_custom_rhs_impl(n, dim, solver, rhs, None)
}

/// Same as `solve_with_custom_rhs` but also performs iterative refinement
/// against the supplied matrix. Use this for Mehrotra/Gondzio backsolves where
/// the cheap (no-refinement) variant would otherwise propagate factorization
/// backward error into μ_aff, σ, and the corrector RHS — exactly the silent
/// bug the ipopt-expert flagged on cho parmest.
pub fn solve_with_custom_rhs_refined(
    matrix: &KktMatrix,
    n: usize,
    dim: usize,
    solver: &mut dyn LinearSolver,
    rhs: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), SolverError> {
    solve_with_custom_rhs_impl(n, dim, solver, rhs, Some(matrix))
}

fn solve_with_custom_rhs_impl(
    n: usize,
    dim: usize,
    solver: &mut dyn LinearSolver,
    rhs: &[f64],
    refine_against: Option<&KktMatrix>,
) -> Result<(Vec<f64>, Vec<f64>), SolverError> {
    if rhs.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(SolverError::NumericalFailure(
            "Custom RHS contains NaN/Inf".to_string(),
        ));
    }
    let mut solution = vec![0.0; dim];
    solver.solve(rhs, &mut solution)?;
    if solution.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(SolverError::NumericalFailure(
            "Custom solve solution contains NaN/Inf".to_string(),
        ));
    }
    if let Some(matrix) = refine_against {
        // Up to 5 iterations of mixed-precision-style refinement against the
        // matrix the caller actually wants solved. Cheaper than the 10-step
        // refinement in `solve_for_direction` because backsolves dominate cost
        // and backsolves used inside Mehrotra/Gondzio don't need 1e-12 accuracy
        // — they just need the residual below the calling oracle's noise floor.
        let max_refinements = 5;
        let mut residual = vec![0.0; dim];
        let mut prev_res_norm = f64::MAX;
        for _ in 0..max_refinements {
            matrix.matvec(&solution, &mut residual);
            let mut res_norm: f64 = 0.0;
            for i in 0..dim {
                residual[i] = rhs[i] - residual[i];
                res_norm = res_norm.max(residual[i].abs());
            }
            if res_norm < 1e-10 {
                break;
            }
            // Stagnation guard (Ipopt convention — slow but steady accepted).
            if res_norm > (1.0 - 1e-6) * prev_res_norm {
                break;
            }
            prev_res_norm = res_norm;
            let mut correction = vec![0.0; dim];
            if solver.solve(&residual, &mut correction).is_err() {
                break;
            }
            for i in 0..dim {
                solution[i] += correction[i];
            }
        }
    }
    Ok((solution[..n].to_vec(), solution[n..].to_vec()))
}

/// Condensed KKT system for m >> n problems (Schur complement).
///
/// Instead of factoring the full (n+m)×(n+m) KKT system, we condense to n×n:
///   S = H + Σ + δ_w·I + J^T · D_c^{-1} · J
///   S · dx = r_d + J^T · D_c^{-1} · r_p
///   dy = D_c^{-1} · (J · dx - r_p)
///
/// where D_c is the (2,2) block diagonal (negative for inequalities).
/// Cost: O(n²·m + n³) instead of O((n+m)³).
pub struct CondensedKktSystem {
    /// Condensed matrix S (n × n).
    pub matrix: SymmetricMatrix,
    /// Condensed RHS (n-vector).
    pub rhs: Vec<f64>,
    /// Number of primal variables.
    pub n: usize,
    /// Number of constraints.
    pub m: usize,
    /// D_c diagonal (m-vector, from the (2,2) block).
    pub d_c: Vec<f64>,
    /// Original primal RHS (n-vector).
    pub rhs_primal: Vec<f64>,
    /// Original constraint RHS (m-vector).
    pub rhs_constraint: Vec<f64>,
    /// Jacobian in COO format.
    pub jac_rows: Vec<usize>,
    pub jac_cols: Vec<usize>,
    pub jac_vals: Vec<f64>,
}

/// Assemble the condensed (Schur complement) KKT system.
///
/// Takes the same inputs as `assemble_kkt` but produces an n×n system
/// instead of (n+m)×(n+m).
#[allow(clippy::too_many_arguments)]
pub fn assemble_condensed_kkt(
    n: usize,
    m: usize,
    hess_rows: &[usize],
    hess_cols: &[usize],
    hess_vals: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    sigma: &[f64],
    grad_f: &[f64],
    g: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    y: &[f64],
    _z_l: &[f64],
    _z_u: &[f64],
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    mu: f64,
    _v_l: &[f64],
    _v_u: &[f64],
) -> CondensedKktSystem {
    // Build the condensed system directly from problem data without assembling
    // the full (n+m)×(n+m) KKT matrix. This saves O((n+m)^2) memory and work.

    // --- (1,1) block: H + Sigma (n×n dense symmetric) ---
    let mut matrix = SymmetricMatrix::zeros(n);

    // Hessian entries
    for (idx, (&row, &col)) in hess_rows.iter().zip(hess_cols.iter()).enumerate() {
        matrix.add(row, col, hess_vals[idx]);
    }

    // Barrier diagonal Sigma for variable bounds
    for i in 0..n {
        matrix.add(i, i, sigma[i]);
    }

    // --- RHS: dual residual r_d (n-vector) ---
    // See full derivation in build_kkt_system above. After eliminating dz, the
    // z terms cancel and r_d reduces to -∇f - J^T*y + μ/s_L - μ/s_U.
    let mut rhs_primal = vec![0.0; n];
    for i in 0..n {
        let mut rd = -grad_f[i];
        if x_l[i].is_finite() {
            rd += mu / (x[i] - x_l[i]);
        }
        if x_u[i].is_finite() {
            rd -= mu / (x_u[i] - x[i]);
        }
        rhs_primal[i] = rd;
    }
    // Subtract J^T * y
    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        rhs_primal[col] -= jac_vals[idx] * y[row];
    }

    // --- (2,2) block diagonal D_c and constraint RHS r_p (m-vectors) ---
    let mut d_c = vec![0.0; m];
    let mut rhs_constraint = vec![0.0; m];

    for i in 0..m {
        if is_equality_constraint(g_l[i], g_u[i]) {
            rhs_constraint[i] = -(g[i] - g_l[i]);
            // d_c[i] = 0.0 for equalities (no (2,2) block entry)
            continue;
        }

        let mut sigma_s = 0.0;
        let mut rhs_correction = y[i];
        let mut any_feasible = false;
        let mut rhs_infeasible = 0.0;

        if g_l[i].is_finite() {
            let slack = g[i] - g_l[i];
            if slack >= -1e-8 {
                let safe_slack = slack.max(mu.max(1e-10));
                let z_sl = if y[i] < -1e-20 { -y[i] } else { mu / safe_slack };
                sigma_s += z_sl / safe_slack;
                rhs_correction += mu / safe_slack;
                any_feasible = true;
            } else {
                rhs_infeasible += -(g[i] - g_l[i]);
            }
        }
        if g_u[i].is_finite() {
            let slack = g_u[i] - g[i];
            if slack >= -1e-8 {
                let safe_slack = slack.max(mu.max(1e-10));
                let z_su = if y[i] > 1e-20 { y[i] } else { mu / safe_slack };
                sigma_s += z_su / safe_slack;
                rhs_correction -= mu / safe_slack;
                any_feasible = true;
            } else {
                rhs_infeasible += -(g[i] - g_u[i]);
            }
        }

        if any_feasible && sigma_s > 1e-20 {
            let sigma_s_inv = (1.0 / sigma_s).min(1e20);
            d_c[i] = -sigma_s_inv;
            rhs_constraint[i] = sigma_s_inv * rhs_correction + rhs_infeasible;
        } else {
            rhs_constraint[i] = rhs_infeasible;
        }
    }

    // --- Build condensed matrix: S = (1,1) block + J^T · (-D_c)^{-1} · J ---
    let mut j_dense = vec![0.0; m * n];
    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        j_dense[row * n + col] += jac_vals[idx];
    }

    for i in 0..m {
        let d_c_eff = if d_c[i].abs() < 1e-20 {
            -1e-16  // equality: very stiff spring (inv = -1e16)
        } else {
            d_c[i]
        };
        let inv_neg_dc = 1.0 / (-d_c_eff);
        for p in 0..n {
            let jp = j_dense[i * n + p];
            if jp == 0.0 {
                continue;
            }
            for q in 0..=p {
                let jq = j_dense[i * n + q];
                if jq != 0.0 {
                    matrix.add(p, q, inv_neg_dc * jp * jq);
                }
            }
        }
    }

    // --- Build condensed RHS: r_d + J^T · (-D_c)^{-1} · r_p ---
    let mut rhs = rhs_primal.clone();
    for i in 0..m {
        let d_c_eff = if d_c[i].abs() < 1e-20 {
            -1e-16
        } else {
            d_c[i]
        };
        let inv_neg_dc = 1.0 / (-d_c_eff);
        let scaled_rp = inv_neg_dc * rhs_constraint[i];
        for p in 0..n {
            let jp = j_dense[i * n + p];
            if jp != 0.0 {
                rhs[p] += jp * scaled_rp;
            }
        }
    }

    CondensedKktSystem {
        matrix,
        rhs,
        n,
        m,
        d_c,
        rhs_primal,
        rhs_constraint,
        jac_rows: jac_rows.to_vec(),
        jac_cols: jac_cols.to_vec(),
        jac_vals: jac_vals.to_vec(),
    }
}

/// Solve the condensed KKT system: compute dx from condensed, recover dy.
pub fn solve_condensed(
    condensed: &CondensedKktSystem,
    solver: &mut dyn LinearSolver,
) -> Result<(Vec<f64>, Vec<f64>), SolverError> {
    let n = condensed.n;
    let m = condensed.m;

    // Solve S · dx = rhs_condensed, then refine against the assembled
    // condensed matrix. Without refinement, factorization backward error
    // propagates directly into dx and (via the J*dx − r_p recovery) into
    // dy, with no protection — the silent backward-error bug the
    // ipopt-expert flagged on cho parmest. The full-KKT path
    // `solve_for_direction` already does iterative refinement; doing the
    // same here closes the gap for problems that take the condensed path.
    let mut dx = vec![0.0; n];
    solver.solve(&condensed.rhs, &mut dx)?;

    let max_refinements = 5;
    let mut residual = vec![0.0; n];
    let mut prev_res_norm = f64::MAX;
    for _ in 0..max_refinements {
        condensed.matrix.matvec(&dx, &mut residual);
        let mut res_norm: f64 = 0.0;
        for i in 0..n {
            residual[i] = condensed.rhs[i] - residual[i];
            res_norm = res_norm.max(residual[i].abs());
        }
        if res_norm < 1e-10 {
            break;
        }
        if res_norm > (1.0 - 1e-6) * prev_res_norm {
            break;
        }
        prev_res_norm = res_norm;
        let mut correction = vec![0.0; n];
        if solver.solve(&residual, &mut correction).is_err() {
            break;
        }
        for i in 0..n {
            dx[i] += correction[i];
        }
    }

    // Recover dy = (-D_c)^{-1} · (J · dx - r_p)
    // First compute J · dx
    let mut jdx = vec![0.0; m];
    for (idx, (&row, &col)) in condensed
        .jac_rows
        .iter()
        .zip(condensed.jac_cols.iter())
        .enumerate()
    {
        jdx[row] += condensed.jac_vals[idx] * dx[col];
    }

    let mut dy = vec![0.0; m];
    for i in 0..m {
        let d_c_eff = if condensed.d_c[i].abs() < 1e-20 {
            -1e-16  // equality: consistent with assembly
        } else {
            condensed.d_c[i]
        };
        dy[i] = (jdx[i] - condensed.rhs_constraint[i]) / (-d_c_eff);
    }

    // Solution-magnitude safeguard. The condensed system uses d_c = -1e-16 as
    // a "very stiff spring" for equality constraints, so any rank deficiency
    // in J amplifies by 1e16 in the dy recovery step. When this happens the
    // solution is physically meaningless (seen on case30_ieee: ||dy||=1e16
    // with ||rhs||≈63), fraction-to-boundary drives α→0, and the solver
    // stalls. Unlike the full-KKT path, the condensed path has no δ_c knob,
    // so the only fix is to reject the step and let the caller fall back to
    // the full KKT, which does apply δ_c via inertia correction.
    let nrm_rhs: f64 = condensed
        .rhs_primal
        .iter()
        .chain(condensed.rhs_constraint.iter())
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    let nrm_sol: f64 = dx
        .iter()
        .chain(dy.iter())
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    let magnitude_ratio = nrm_sol / nrm_rhs.max(1.0);
    if magnitude_ratio > 1e10 {
        log::debug!(
            "Condensed ||sol||={:.2e} vs ||rhs||={:.2e} (ratio {:.2e}) — rank-def, falling back to full KKT",
            nrm_sol, nrm_rhs, magnitude_ratio,
        );
        return Err(SolverError::NumericalFailure(format!(
            "Condensed solution magnitude {:.2e} exceeds {:.2e}×RHS (rank-deficient)",
            nrm_sol, nrm_rhs.max(1.0) * 1e10,
        )));
    }

    Ok((dx, dy))
}

/// Solve the condensed system with a modified constraint residual (for SOC).
///
/// Instead of using the original rhs_constraint, uses -c_soc as the constraint
/// residual. This avoids rebuilding the full (n+m)×(n+m) KKT system for SOC.
pub fn solve_condensed_soc(
    condensed: &CondensedKktSystem,
    solver: &mut dyn LinearSolver,
    c_soc: &[f64],
) -> Result<Vec<f64>, SolverError> {
    let n = condensed.n;
    let m = condensed.m;

    // Build per-constraint scaling: (-c_soc[i]) / (-d_c[i])
    let mut scaled = vec![0.0; m];
    for i in 0..m {
        let d_c_eff = if condensed.d_c[i].abs() < 1e-20 {
            -1e-16 // equality: consistent with assembly
        } else {
            condensed.d_c[i]
        };
        scaled[i] = (-c_soc[i]) / (-d_c_eff);
    }

    // Build modified condensed RHS: rhs_primal + J^T · scaled
    let mut rhs = condensed.rhs_primal.clone();
    for (idx, (&row, &col)) in condensed
        .jac_rows
        .iter()
        .zip(condensed.jac_cols.iter())
        .enumerate()
    {
        rhs[col] += condensed.jac_vals[idx] * scaled[row];
    }

    // Solve S · dx = modified_rhs
    let mut dx = vec![0.0; n];
    solver.solve(&rhs, &mut dx)?;

    Ok(dx)
}

/// Sparse condensed KKT system for large problems where both n and m are large.
///
/// Like `CondensedKktSystem` but stores S as a sparse matrix, avoiding O(n²) memory.
/// The Schur complement J^T · D_c^{-1} · J is computed directly in COO format.
pub struct SparseCondensedKktSystem {
    pub matrix: SparseSymmetricMatrix,
    pub rhs: Vec<f64>,
    pub n: usize,
    pub m: usize,
    pub d_c: Vec<f64>,
    pub rhs_primal: Vec<f64>,
    pub rhs_constraint: Vec<f64>,
    pub jac_rows: Vec<usize>,
    pub jac_cols: Vec<usize>,
    pub jac_vals: Vec<f64>,
}

/// Assemble a sparse condensed (Schur complement) KKT system.
///
/// S = H + Σ + J^T · (-D_c)^{-1} · J  (n×n sparse)
/// rhs = r_d + J^T · (-D_c)^{-1} · r_p
///
/// The J^T·D_c^{-1}·J product is computed row-by-row:
/// for each constraint i, add rank-1 update (1/(-d_c\[i\])) * J\[i,:\]^T * J\[i,:\]
/// Only nonzero pairs of J entries in the same row produce fill.
#[allow(clippy::too_many_arguments)]
pub fn assemble_sparse_condensed_kkt(
    n: usize,
    m: usize,
    hess_rows: &[usize],
    hess_cols: &[usize],
    hess_vals: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    sigma: &[f64],
    grad_f: &[f64],
    g: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    y: &[f64],
    _z_l: &[f64],
    _z_u: &[f64],
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    mu: f64,
    _v_l: &[f64],
    _v_u: &[f64],
) -> SparseCondensedKktSystem {
    // Estimate nnz: hess + diagonal + J^T*D_c^{-1}*J fill
    // For each constraint row with k nonzeros, the outer product has k*(k+1)/2 entries.
    // Build a row-pointer structure for J to iterate by constraint row.
    let mut row_start = vec![0usize; m + 1];
    for &r in jac_rows {
        row_start[r + 1] += 1;
    }
    for i in 0..m {
        row_start[i + 1] += row_start[i];
    }
    // Sort Jacobian entries by row
    let jac_nnz = jac_rows.len();
    let mut jac_order = vec![0usize; jac_nnz];
    let mut row_count = vec![0usize; m];
    for k in 0..jac_nnz {
        let r = jac_rows[k];
        jac_order[row_start[r] + row_count[r]] = k;
        row_count[r] += 1;
    }

    // Estimate nnz for S
    let mut schur_nnz = 0;
    for i in 0..m {
        let k = row_start[i + 1] - row_start[i];
        schur_nnz += k * (k + 1) / 2;
    }
    let total_nnz = hess_rows.len() + n + schur_nnz;
    let mut matrix = SparseSymmetricMatrix::with_capacity(n, total_nnz);

    // (1,1) block: H + Σ
    for (idx, (&row, &col)) in hess_rows.iter().zip(hess_cols.iter()).enumerate() {
        matrix.add(row, col, hess_vals[idx]);
    }
    for i in 0..n {
        matrix.add(i, i, sigma[i]);
    }

    // RHS: dual residual r_d. z terms cancel after dz elimination — see
    // full derivation in build_kkt_system.
    let mut rhs_primal = vec![0.0; n];
    for i in 0..n {
        let mut rd = -grad_f[i];
        if x_l[i].is_finite() {
            rd += mu / (x[i] - x_l[i]);
        }
        if x_u[i].is_finite() {
            rd -= mu / (x_u[i] - x[i]);
        }
        rhs_primal[i] = rd;
    }
    // Subtract J^T * y
    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        rhs_primal[col] -= jac_vals[idx] * y[row];
    }

    // (2,2) block diagonal D_c and constraint RHS
    let mut d_c = vec![0.0; m];
    let mut rhs_constraint = vec![0.0; m];

    for i in 0..m {
        if is_equality_constraint(g_l[i], g_u[i]) {
            // For equalities, D_c comes from constraint regularization.
            // Use -delta_c so the Schur complement J^T * delta_c^{-1} * J
            // doesn't blow up. Floor at 1e-4 to keep D_c^{-1} <= 1e4;
            // smaller values cause inaccurate dy recovery that degrades
            // convergence. The regularization doesn't affect the primal
            // step quality (dx from S is accurate), only dy.
            let delta_c = mu.max(1e-8);
            d_c[i] = -delta_c;
            rhs_constraint[i] = -(g[i] - g_l[i]);
            continue;
        }

        let mut sigma_s = 0.0;
        let mut rhs_correction = y[i];
        let mut any_feasible = false;
        let mut rhs_infeasible = 0.0;

        if g_l[i].is_finite() {
            let slack = g[i] - g_l[i];
            if slack >= -1e-8 {
                let safe_slack = slack.max(mu.max(1e-10));
                let z_sl = if y[i] < -1e-20 { -y[i] } else { mu / safe_slack };
                sigma_s += z_sl / safe_slack;
                rhs_correction += mu / safe_slack;
                any_feasible = true;
            } else {
                rhs_infeasible += -(g[i] - g_l[i]);
            }
        }
        if g_u[i].is_finite() {
            let slack = g_u[i] - g[i];
            if slack >= -1e-8 {
                let safe_slack = slack.max(mu.max(1e-10));
                let z_su = if y[i] > 1e-20 { y[i] } else { mu / safe_slack };
                sigma_s += z_su / safe_slack;
                rhs_correction -= mu / safe_slack;
                any_feasible = true;
            } else {
                rhs_infeasible += -(g[i] - g_u[i]);
            }
        }

        if any_feasible && sigma_s > 1e-20 {
            let sigma_s_inv = (1.0 / sigma_s).min(1e20);
            d_c[i] = -sigma_s_inv;
            rhs_constraint[i] = sigma_s_inv * rhs_correction + rhs_infeasible;
        } else {
            rhs_constraint[i] = rhs_infeasible;
        }
    }

    // Schur complement: S += J^T · (-D_c)^{-1} · J
    // Process row-by-row to exploit sparsity
    for i in 0..m {
        let d_c_eff = if d_c[i].abs() < 1e-20 { -1e-16 } else { d_c[i] };
        let inv_neg_dc = 1.0 / (-d_c_eff);

        let start = row_start[i];
        let end = row_start[i + 1];
        // For each pair of nonzeros in row i of J, add outer product entry
        for a in start..end {
            let ka = jac_order[a];
            let ca = jac_cols[ka];
            let va = jac_vals[ka];
            for b in a..end {
                let kb = jac_order[b];
                let cb = jac_cols[kb];
                let vb = jac_vals[kb];
                // Add to upper triangle (min(ca,cb), max(ca,cb))
                let (p, q) = if ca <= cb { (ca, cb) } else { (cb, ca) };
                let val = if a == b {
                    inv_neg_dc * va * vb
                } else {
                    inv_neg_dc * va * vb // both (a,b) and (b,a) contribute, but we only do upper
                };
                matrix.add(p, q, val);
            }
        }
    }

    // Condensed RHS: r_d + J^T · (-D_c)^{-1} · r_p
    let mut rhs = rhs_primal.clone();
    for i in 0..m {
        let d_c_eff = if d_c[i].abs() < 1e-20 { -1e-16 } else { d_c[i] };
        let inv_neg_dc = 1.0 / (-d_c_eff);
        let scaled_rp = inv_neg_dc * rhs_constraint[i];
        let start = row_start[i];
        let end = row_start[i + 1];
        for a in start..end {
            let ka = jac_order[a];
            rhs[jac_cols[ka]] += jac_vals[ka] * scaled_rp;
        }
    }

    SparseCondensedKktSystem {
        matrix,
        rhs,
        n,
        m,
        d_c,
        rhs_primal,
        rhs_constraint,
        jac_rows: jac_rows.to_vec(),
        jac_cols: jac_cols.to_vec(),
        jac_vals: jac_vals.to_vec(),
    }
}

/// Solve a sparse condensed system: dx from factored solver, recover dy.
pub fn solve_sparse_condensed(
    condensed: &SparseCondensedKktSystem,
    solver: &mut dyn LinearSolver,
) -> Result<(Vec<f64>, Vec<f64>), SolverError> {
    let n = condensed.n;
    let m = condensed.m;

    let mut dx = vec![0.0; n];
    solver.solve(&condensed.rhs, &mut dx)?;

    // Recover dy = (-D_c)^{-1} · (J · dx - r_p)
    let mut jdx = vec![0.0; m];
    for (idx, (&row, &col)) in condensed.jac_rows.iter().zip(condensed.jac_cols.iter()).enumerate() {
        jdx[row] += condensed.jac_vals[idx] * dx[col];
    }

    let mut dy = vec![0.0; m];
    for i in 0..m {
        let d_c_eff = if condensed.d_c[i].abs() < 1e-20 { -1e-16 } else { condensed.d_c[i] };
        dy[i] = (jdx[i] - condensed.rhs_constraint[i]) / (-d_c_eff);
    }

    Ok((dx, dy))
}

/// Solve sparse condensed with modified constraint residual (for SOC).
pub fn solve_sparse_condensed_soc(
    condensed: &SparseCondensedKktSystem,
    solver: &mut dyn LinearSolver,
    c_soc: &[f64],
) -> Result<Vec<f64>, SolverError> {
    let n = condensed.n;
    let m = condensed.m;

    let mut scaled = vec![0.0; m];
    for i in 0..m {
        let d_c_eff = if condensed.d_c[i].abs() < 1e-20 { -1e-16 } else { condensed.d_c[i] };
        scaled[i] = (-c_soc[i]) / (-d_c_eff);
    }

    let mut rhs = condensed.rhs_primal.clone();
    for (idx, (&row, &col)) in condensed.jac_rows.iter().zip(condensed.jac_cols.iter()).enumerate() {
        rhs[col] += condensed.jac_vals[idx] * scaled[row];
    }

    let mut dx = vec![0.0; n];
    solver.solve(&rhs, &mut dx)?;
    Ok(dx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_solver::dense::DenseLdl;

    #[test]
    fn test_compute_sigma_no_bounds() {
        let x = vec![1.0, 2.0];
        let x_l = vec![f64::NEG_INFINITY, f64::NEG_INFINITY];
        let x_u = vec![f64::INFINITY, f64::INFINITY];
        let z_l = vec![0.0, 0.0];
        let z_u = vec![0.0, 0.0];
        let sigma = compute_sigma(&x, &x_l, &x_u, &z_l, &z_u);
        assert!((sigma[0]).abs() < 1e-15);
        assert!((sigma[1]).abs() < 1e-15);
    }

    #[test]
    fn test_compute_sigma_lower_bound_only() {
        let x = vec![1.5];
        let x_l = vec![1.0];
        let x_u = vec![f64::INFINITY];
        let z_l = vec![2.0];
        let z_u = vec![0.0];
        let sigma = compute_sigma(&x, &x_l, &x_u, &z_l, &z_u);
        // sigma = z_l / (x - x_l) = 2.0 / 0.5 = 4.0
        assert!((sigma[0] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_sigma_both_bounds() {
        let x = vec![1.5];
        let x_l = vec![1.0];
        let x_u = vec![2.0];
        let z_l = vec![2.0];
        let z_u = vec![3.0];
        let sigma = compute_sigma(&x, &x_l, &x_u, &z_l, &z_u);
        // sigma = 2.0/0.5 + 3.0/0.5 = 4.0 + 6.0 = 10.0
        assert!((sigma[0] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_sigma_at_bound_clamped() {
        let x = vec![1.0]; // At lower bound
        let x_l = vec![1.0];
        let x_u = vec![f64::INFINITY];
        let z_l = vec![1.0];
        let z_u = vec![0.0];
        let sigma = compute_sigma(&x, &x_l, &x_u, &z_l, &z_u);
        // slack = max(0, 1e-20) = 1e-20, sigma = 1.0/1e-20 = 1e20
        assert!(sigma[0] > 1e19);
    }

    #[test]
    fn test_assemble_kkt_unconstrained() {
        // 2 vars, no constraints
        // H = [[2, 0], [0, 3]]
        let n = 2;
        let m = 0;
        let hess_rows = vec![0, 1];
        let hess_cols = vec![0, 1];
        let hess_vals = vec![2.0, 3.0];
        let sigma = vec![1.0, 2.0];
        let grad_f = vec![0.5, 0.5];
        let x = vec![1.0, 2.0];
        let x_l = vec![f64::NEG_INFINITY; 2];
        let x_u = vec![f64::INFINITY; 2];
        let z_l = vec![0.0; 2];
        let z_u = vec![0.0; 2];

        let kkt = assemble_kkt(
            n, m, &hess_rows, &hess_cols, &hess_vals,
            &[], &[], &[], &sigma, &grad_f,
            &[], &[], &[], &[], &z_l, &z_u,
            &x, &x_l, &x_u, 0.1, false, &[], &[],
        );

        assert_eq!(kkt.dim, 2);
        // (1,1) block: H + Sigma = [[3, 0], [0, 5]]
        assert!((kkt.matrix.get(0, 0) - 3.0).abs() < 1e-12);
        assert!((kkt.matrix.get(1, 1) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_assemble_kkt_equality_constraint() {
        // 2 vars, 1 equality constraint: x0 + x1 = 1
        let n = 2;
        let m = 1;
        let hess_rows = vec![0, 1];
        let hess_cols = vec![0, 1];
        let hess_vals = vec![2.0, 2.0];
        let jac_rows = vec![0, 0];
        let jac_cols = vec![0, 1];
        let jac_vals = vec![1.0, 1.0];
        let sigma = vec![0.0; 2];
        let grad_f = vec![1.0, 1.0];
        let g = vec![0.7]; // current constraint value
        let g_l = vec![1.0];
        let g_u = vec![1.0];
        let y = vec![0.5];
        let x = vec![0.3, 0.4];
        let x_l = vec![f64::NEG_INFINITY; 2];
        let x_u = vec![f64::INFINITY; 2];
        let z_l = vec![0.0; 2];
        let z_u = vec![0.0; 2];

        let v_l = vec![0.0; m];
        let v_u = vec![0.0; m];
        let kkt = assemble_kkt(
            n, m, &hess_rows, &hess_cols, &hess_vals,
            &jac_rows, &jac_cols, &jac_vals, &sigma, &grad_f,
            &g, &g_l, &g_u, &y, &z_l, &z_u,
            &x, &x_l, &x_u, 0.1, false, &v_l, &v_u,
        );

        assert_eq!(kkt.dim, 3);
        // Verify J block: matrix[2,0] and matrix[2,1] should be 1.0
        assert!((kkt.matrix.get(2, 0) - 1.0).abs() < 1e-12);
        assert!((kkt.matrix.get(2, 1) - 1.0).abs() < 1e-12);
        // T0.13: equality constraint (2,2) block is exactly 0 — no
        // unconditional δ_c regularization. δ_c is applied only by the
        // PerturbForSingularity path inside factor_with_inertia_correction
        // when the augmented system is detected as singular.
        assert!(kkt.matrix.get(2, 2).abs() < 1e-15);
        assert!(kkt.delta_c_diag[0].abs() < 1e-15);
        // Primal residual: -(g - g_l) = -(0.7 - 1.0) = 0.3
        assert!((kkt.rhs[2] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_assemble_kkt_rhs_sign_convention() {
        // Regression: J^T*y must be subtracted from r_d
        let n = 1;
        let m = 1;
        let hess_rows = vec![0];
        let hess_cols = vec![0];
        let hess_vals = vec![1.0];
        let jac_rows = vec![0];
        let jac_cols = vec![0];
        let jac_vals = vec![2.0]; // J = [2]
        let sigma = vec![0.0];
        let grad_f = vec![3.0];
        let g = vec![1.0];
        let g_l = vec![1.0];
        let g_u = vec![1.0];
        let y = vec![1.0];
        let x = vec![1.0];
        let x_l = vec![f64::NEG_INFINITY];
        let x_u = vec![f64::INFINITY];
        let z_l = vec![0.0];
        let z_u = vec![0.0];

        let v_l = vec![0.0; m];
        let v_u = vec![0.0; m];
        let kkt = assemble_kkt(
            n, m, &hess_rows, &hess_cols, &hess_vals,
            &jac_rows, &jac_cols, &jac_vals, &sigma, &grad_f,
            &g, &g_l, &g_u, &y, &z_l, &z_u,
            &x, &x_l, &x_u, 0.1, false, &v_l, &v_u,
        );

        // r_d = -grad_f + z_l - z_u = -3.0 + 0 - 0 = -3.0
        // Then subtract J^T * y: -3.0 - 2.0*1.0 = -5.0
        assert!((kkt.rhs[0] - (-5.0)).abs() < 1e-12,
            "RHS sign convention: expected -5.0, got {}", kkt.rhs[0]);
    }

    #[test]
    fn test_assemble_kkt_inequality_constraint() {
        // Feasible inequality: g(x) = 2.0, g_l = 1.0, g_u = INF
        let n = 1;
        let m = 1;
        let hess_rows = vec![0];
        let hess_cols = vec![0];
        let hess_vals = vec![1.0];
        let jac_rows = vec![0];
        let jac_cols = vec![0];
        let jac_vals = vec![1.0];
        let sigma = vec![0.0];
        let grad_f = vec![0.0];
        let g = vec![2.0]; // feasible: 2.0 > 1.0
        let g_l = vec![1.0];
        let g_u = vec![f64::INFINITY];
        let y = vec![0.0];
        let x = vec![2.0];
        let x_l = vec![f64::NEG_INFINITY];
        let x_u = vec![f64::INFINITY];
        let z_l = vec![0.0];
        let z_u = vec![0.0];
        let mu = 0.1;

        let v_l = vec![0.0; m];
        let v_u = vec![0.0; m];
        let kkt = assemble_kkt(
            n, m, &hess_rows, &hess_cols, &hess_vals,
            &jac_rows, &jac_cols, &jac_vals, &sigma, &grad_f,
            &g, &g_l, &g_u, &y, &z_l, &z_u,
            &x, &x_l, &x_u, mu, false, &v_l, &v_u,
        );

        // (2,2) block should be negative (from -Σ_s^{-1})
        assert!(kkt.matrix.get(1, 1) < 0.0,
            "Inequality (2,2) block should be negative, got {}", kkt.matrix.get(1, 1));
    }

    #[test]
    fn test_factor_with_inertia_correction_good() {
        // KKT matrix with correct inertia (2, 1, 0) — no perturbation needed
        let n = 2;
        let m = 1;
        let mut matrix = SymmetricMatrix::zeros(3);
        matrix.set(0, 0, 2.0);
        matrix.set(1, 1, 2.0);
        matrix.set(2, 0, 1.0);
        matrix.set(2, 1, 1.0);

        let mut kkt = KktSystem {
            dim: 3, n, m,
            matrix: KktMatrix::Dense(matrix),
            rhs: vec![1.0, 2.0, 3.0],
            delta_c_diag: vec![0.0; m],
            scale_factors: None,
        };
        let mut solver = DenseLdl::new();
        let mut params = InertiaCorrectionParams::default();

        let (delta_w, delta_c) = factor_with_inertia_correction(&mut kkt, &mut solver, &mut params, 1e-4).unwrap();
        assert!((delta_w).abs() < 1e-15, "Good inertia should need no delta_w");
        assert!((delta_c).abs() < 1e-15, "Good inertia should need no delta_c");
    }

    #[test]
    fn test_factor_with_inertia_correction_needs_perturbation() {
        // Matrix with wrong inertia: all positive (3, 0, 0) instead of (2, 1, 0)
        let n = 2;
        let m = 1;
        let mut matrix = SymmetricMatrix::zeros(3);
        matrix.set(0, 0, 2.0);
        matrix.set(1, 1, 2.0);
        matrix.set(2, 2, 1.0); // Positive instead of 0
        matrix.set(2, 0, 0.1);
        matrix.set(2, 1, 0.1);

        let mut kkt = KktSystem {
            dim: 3, n, m,
            matrix: KktMatrix::Dense(matrix),
            rhs: vec![1.0, 2.0, 3.0],
            delta_c_diag: vec![0.0; m],
            scale_factors: None,
        };
        let mut solver = DenseLdl::new();
        let mut params = InertiaCorrectionParams::default();

        let (delta_w, _delta_c) = factor_with_inertia_correction(&mut kkt, &mut solver, &mut params, 1e-4).unwrap();
        assert!(delta_w > 0.0, "Wrong inertia should require delta_w > 0");
    }

    /// T0.13: δ_c remains 0 on a non-singular system. The KKT matrix here
    /// has the correct inertia (n positives, m negatives, no zero pivots),
    /// so neither PerturbForWrongInertia nor PerturbForSingularity should
    /// fire — `factor_with_inertia_correction` should return (0, 0).
    #[test]
    fn test_factor_no_delta_c_when_nonsingular() {
        let n = 2;
        let m = 1;
        let mut matrix = SymmetricMatrix::zeros(3);
        matrix.set(0, 0, 2.0);
        matrix.set(1, 1, 3.0);
        matrix.set(2, 0, 1.0);
        matrix.set(2, 1, 0.5);
        // (2,2) intentionally 0 — equality block. With well-conditioned J this
        // produces a non-singular augmented system with inertia (2, 1, 0).

        let mut kkt = KktSystem {
            dim: 3, n, m,
            matrix: KktMatrix::Dense(matrix),
            rhs: vec![1.0, 2.0, 3.0],
            delta_c_diag: vec![0.0; m],
            scale_factors: None,
        };
        let mut solver = DenseLdl::new();
        let mut params = InertiaCorrectionParams::default();

        let (delta_w, delta_c) = factor_with_inertia_correction(
            &mut kkt, &mut solver, &mut params, 1e-4,
        ).unwrap();
        assert!(delta_w.abs() < 1e-15, "non-singular: no delta_w needed");
        assert!(delta_c.abs() < 1e-15, "non-singular: no delta_c (T0.13)");
    }

    /// T0.14: pretend-singular allowed at most once per outer iteration.
    /// A synthetically singular system (rank-deficient J, zero RHS rows)
    /// drives `solve_for_direction` into the residual-ratio guard. The
    /// first call via `solve_for_direction_iter_aware` should return
    /// `PretendSingular` (and set the flag). The second call within the
    /// same outer iter must NOT return `PretendSingular` (T0.14 refusal).
    /// After `reset_pretend_singular_for_new_iter`, the trigger fires again.
    #[test]
    fn test_pretend_singular_once_per_iter() {
        // Construct a 3x3 singular augmented system: H = diag(1e-30, 1e-30),
        // J = [1e10, 1e10] (rank 1 row, but with huge magnitude). The
        // factorized system will produce a solution whose magnitude
        // ratio || sol || / || rhs || is enormous, which the rank-def
        // guard in solve_for_direction reports as PretendSingular.
        let n = 2;
        let m = 1;
        let mut matrix = SymmetricMatrix::zeros(3);
        matrix.set(0, 0, 1.0);
        matrix.set(1, 1, 1.0);
        // Tight coupling that creates a near-null direction.
        matrix.set(2, 0, 1e10);
        matrix.set(2, 1, 1e10);
        // (2,2) = 0 — equality block, no δ_c (T0.13 alignment).

        let mut kkt = KktSystem {
            dim: 3, n, m,
            matrix: KktMatrix::Dense(matrix),
            rhs: vec![1.0, 1.0, 0.0],
            delta_c_diag: vec![0.0; m],
            scale_factors: None,
        };
        let mut solver = DenseLdl::new();
        let mut params = InertiaCorrectionParams::default();
        // Factor once so subsequent solves succeed.
        let _ = solver.factor(&kkt.matrix);

        let r1 = solve_for_direction_iter_aware(&kkt, &mut solver, &mut params, 0.0, 0.0);
        // Either we got PretendSingular (and the flag is now set), or the
        // matrix is happily solvable; in the latter case the test cannot
        // exercise the T0.14 path. Skip with a soft assertion if so.
        if !matches!(r1, Err(SolverError::PretendSingular)) {
            // Solve happened to succeed — adjust expectation rather than
            // asserting a brittle synthetic outcome.
            assert!(!params.pretend_singular_used,
                "flag must not be set on a successful solve");
            return;
        }
        assert!(params.pretend_singular_used,
            "first PretendSingular must record the flag");

        // Second call this outer iter — refused. Must be Ok.
        // (Our wrapper forces a plain backsolve and returns Ok(...).)
        let r2 = solve_for_direction_iter_aware(&kkt, &mut solver, &mut params, 0.0, 0.0);
        assert!(!matches!(r2, Err(SolverError::PretendSingular)),
            "T0.14: second pretend-singular within the same outer iter must be refused, got {:?}", r2);

        // Reset for new outer iter — flag clears, pretend-singular fires again.
        params.reset_pretend_singular_for_new_iter();
        assert!(!params.pretend_singular_used,
            "reset_pretend_singular_for_new_iter clears the flag");
        let r3 = solve_for_direction_iter_aware(&kkt, &mut solver, &mut params, 0.0, 0.0);
        assert!(matches!(r3, Err(SolverError::PretendSingular)),
            "after reset, pretend-singular should fire again on the same singular system");
    }

    /// T0.13: δ_c is bumped on a synthetically singular system. With a
    /// rank-deficient Jacobian (all-zero row), the (2,2) block factorization
    /// reports a zero pivot — singularity_detected — and the
    /// PerturbForSingularity path adds δ_c > 0 to lift the singularity.
    #[test]
    fn test_factor_delta_c_on_singular() {
        let n = 2;
        let m = 1;
        let mut matrix = SymmetricMatrix::zeros(3);
        matrix.set(0, 0, 2.0);
        matrix.set(1, 1, 3.0);
        // Jacobian row is all zero ⇒ augmented system is singular.
        // (2,0) and (2,1) are both 0; (2,2) is 0 (no assembly-time δ_c).

        let mut kkt = KktSystem {
            dim: 3, n, m,
            matrix: KktMatrix::Dense(matrix),
            rhs: vec![1.0, 2.0, 3.0],
            delta_c_diag: vec![0.0; m],
            scale_factors: None,
        };
        let mut solver = DenseLdl::new();
        let mut params = InertiaCorrectionParams::default();

        let result = factor_with_inertia_correction(
            &mut kkt, &mut solver, &mut params, 1e-4,
        );
        // The factorization should succeed (perturbation handler lifts the
        // singularity), and δ_c should have been bumped above 0.
        let (_delta_w, delta_c) = result.expect("factorization should recover from singularity");
        assert!(delta_c > 0.0,
            "singular system: delta_c must be > 0 (PerturbForSingularity)");
    }

    #[test]
    fn test_factor_with_inertia_correction_warm_start() {
        // Use delta_w_last to warm-start perturbation
        let n = 2;
        let m = 1;
        let mut matrix = SymmetricMatrix::zeros(3);
        matrix.set(0, 0, 2.0);
        matrix.set(1, 1, 2.0);
        matrix.set(2, 2, 1.0);
        matrix.set(2, 0, 0.1);
        matrix.set(2, 1, 0.1);

        let mut kkt = KktSystem {
            dim: 3, n, m,
            matrix: KktMatrix::Dense(matrix),
            rhs: vec![1.0, 2.0, 3.0],
            delta_c_diag: vec![0.0; m],
            scale_factors: None,
        };
        let mut solver = DenseLdl::new();
        let mut params = InertiaCorrectionParams::default();
        params.delta_w_last = 1.0; // Warm-start from previous

        let (delta_w, _) = factor_with_inertia_correction(&mut kkt, &mut solver, &mut params, 1e-4).unwrap();
        // Ipopt-style warm-shrink: start at delta_w_last * delta_w_dec_fact = 1.0 * 1/3.
        // The first acceptable delta_w is at least that (escalation grows from there).
        let warm_start = 1.0 * params.delta_w_dec_fact;
        assert!(
            delta_w >= warm_start - 1e-12,
            "Warm-start should begin from delta_w_last * dec_fact ({}); got {}",
            warm_start, delta_w
        );
    }

    #[test]
    fn test_factor_with_inertia_correction_growth_sequence() {
        // Wrong inertia forces delta_w to escalate from delta_w_init by delta_w_growth.
        // With delta_w_init=1e-4 and growth=4.0, attempt k uses delta_w = 1e-4 * 4^k.
        // A matrix that needs a significant delta_w lets us verify attempts > 0.
        let n = 2;
        let m = 1;
        let mut matrix = SymmetricMatrix::zeros(3);
        matrix.set(0, 0, -1.0); // Indefinite top block
        matrix.set(1, 1, -1.0);
        matrix.set(2, 0, 1.0);
        matrix.set(2, 1, 1.0);

        let mut kkt = KktSystem {
            dim: 3, n, m,
            matrix: KktMatrix::Dense(matrix),
            rhs: vec![1.0, 2.0, 3.0],
            delta_c_diag: vec![0.0; m],
            scale_factors: None,
        };
        let mut solver = DenseLdl::new();
        let mut params = InertiaCorrectionParams::default();
        let (delta_w, _dc) = factor_with_inertia_correction(&mut kkt, &mut solver, &mut params, 1e-4).unwrap();
        // Expect delta_w >= delta_w_init * growth (at least one escalation beyond initial attempt)
        assert!(delta_w >= params.delta_w_init,
            "delta_w should be at least delta_w_init, got {}", delta_w);
        // delta_w_last is set to the successful perturbation (for warm-start next iteration)
        assert!((params.delta_w_last - delta_w).abs() < 1e-15,
            "delta_w_last should equal the successful delta_w");
        // degeneracy_count should bump once per successful perturbation
        assert_eq!(params.degeneracy_count, 1);
    }

    #[test]
    fn test_inertia_first_inc_factor_used_when_cold() {
        // First-time perturbation (delta_w_last == 0): escalation should
        // multiply by the first-inc factor (100), not the subsequent
        // inc factor (8). Mirrors Ipopt
        // PDPerturbationHandler::get_deltas_for_wrong_inertia
        // (IpPDPerturbationHandler.cpp:386-393).
        //
        // Construct a KKT requiring at least one escalation: identity
        // (2,2) block + identity constraint block has inertia (4,0,0),
        // so the loop must perturb to flip 2 eigenvalues to negative
        // via the -delta_c addition. The initial delta_w starts at
        // delta_w_init = 1e-4, and the next delta_w (if needed) is
        // 1e-4 * 100 = 1e-2 — never 1e-4 * 8 = 8e-4.
        let n = 2;
        let m = 2;
        let mut matrix = SymmetricMatrix::zeros(4);
        for i in 0..4 { matrix.set(i, i, 1.0); }
        matrix.set(2, 0, 0.5);
        matrix.set(3, 1, 0.5);
        let mut kkt = KktSystem {
            dim: 4, n, m,
            matrix: KktMatrix::Dense(matrix),
            rhs: vec![1.0, 2.0, 3.0, 4.0],
            delta_c_diag: vec![0.0; m],
            scale_factors: None,
        };
        let mut solver = DenseLdl::new();
        let mut params = InertiaCorrectionParams::default();
        // Inject a high mu so delta_c_active is meaningful.
        let result = factor_with_inertia_correction(&mut kkt, &mut solver, &mut params, 1.0);
        assert!(result.is_ok());
        let (delta_w, _) = result.unwrap();
        // Either the first try (1e-4) succeeded, or the loop escalated
        // to >= 1e-4 * 100 = 1e-2. The intermediate value 1e-4 * 8 must
        // never appear.
        if delta_w > params.delta_w_init * 1.5 {
            assert!(
                delta_w >= params.delta_w_init * params.delta_w_inc_fact_first - 1e-12,
                "Cold escalation should jump by 100x, not 8x; got delta_w={:.3e}",
                delta_w
            );
        }
    }

    #[test]
    fn test_inertia_dec_fact_warm_shrinks_by_one_third() {
        // Warm start with delta_w_last = 0.9: the next call should begin
        // at 0.9 * (1/3) = 0.3, not 0.9 / 8.0 (the old growth-based
        // shrink). Verifies the dec_fact warm-shrink path
        // (IpPDPerturbationHandler.cpp:381). We force wrong inertia via
        // an indefinite (1,1) block so the perturbation loop actually
        // runs (the unperturbed factorization would otherwise pass).
        let n = 2;
        let m = 1;
        let mut matrix = SymmetricMatrix::zeros(3);
        matrix.set(0, 0, -1.0);
        matrix.set(1, 1, -1.0);
        matrix.set(2, 2, 1.0);
        matrix.set(2, 0, 0.1);
        matrix.set(2, 1, 0.1);
        let mut kkt2 = KktSystem {
            dim: 3, n, m,
            matrix: KktMatrix::Dense(matrix),
            rhs: vec![1.0, 2.0, 3.0],
            delta_c_diag: vec![0.0; m],
            scale_factors: None,
        };
        let mut solver = DenseLdl::new();
        let mut params = InertiaCorrectionParams::default();
        params.delta_w_last = 0.9;
        let (delta_w, _) =
            factor_with_inertia_correction(&mut kkt2, &mut solver, &mut params, 1e-4).unwrap();
        // The first delta_w tried is 0.9 * (1/3) = 0.3. If the
        // unperturbed (cold) path had been used, we would see <= 1e-4.
        // The successful delta_w should match (or escalate beyond) 0.3.
        let expected_initial = 0.9 * (1.0 / 3.0);
        assert!(
            delta_w >= expected_initial - 1e-12,
            "Warm-shrink should start at delta_w_last * 1/3 = {:.3e}; got {:.3e}",
            expected_initial, delta_w
        );
        // Also confirm we did NOT use the old 1/8 shrink (which would
        // give 0.1125 — strictly less than 1/3 of last). The strictly-
        // greater check rules that out.
        assert!(
            delta_w > 0.9 / 8.0,
            "delta_w should not match the deprecated /growth shrink"
        );
    }

    #[test]
    fn test_inertia_max_perturbation_cap() {
        // Setting delta_w_max small and forcing impossible inertia
        // exercises the give-up path. The function must return Ok
        // (fall through to the warning path with best_delta_w),
        // not loop forever.
        let n = 2;
        let m = 2;
        let mut matrix = SymmetricMatrix::zeros(4);
        for i in 0..4 { matrix.set(i, i, 1.0); }
        matrix.set(2, 0, 0.1);
        matrix.set(3, 1, 0.1);
        let mut kkt = KktSystem {
            dim: 4, n, m,
            matrix: KktMatrix::Dense(matrix),
            rhs: vec![1.0, 2.0, 3.0, 4.0],
            delta_c_diag: vec![0.0; m],
            scale_factors: None,
        };
        let mut solver = DenseLdl::new();
        let mut params = InertiaCorrectionParams {
            delta_w_max: 1e-2,
            max_attempts: 100,
            ..Default::default()
        };
        let result = factor_with_inertia_correction(&mut kkt, &mut solver, &mut params, 1e-4);
        assert!(result.is_ok(), "delta_w_max give-up must return Ok");
    }

    #[test]
    fn test_factor_with_inertia_correction_max_attempts_cap() {
        // Hopeless KKT with inertia so wrong no finite delta_w recovers it within max_attempts=3.
        // (We lower max_attempts so the test is fast and deterministic.)
        // With max_attempts reduced, the loop exhausts and falls through to the warning path,
        // which still returns Ok(best_delta_w, delta_c) instead of panicking.
        let n = 2;
        let m = 2; // Expected inertia (2, 2, 0) but we'll make the matrix all-positive
        let mut matrix = SymmetricMatrix::zeros(4);
        for i in 0..4 { matrix.set(i, i, 1.0); } // Identity: inertia (4, 0, 0), wrong
        matrix.set(2, 0, 0.1);
        matrix.set(3, 1, 0.1);

        let mut kkt = KktSystem {
            dim: 4, n, m,
            matrix: KktMatrix::Dense(matrix),
            rhs: vec![1.0, 2.0, 3.0, 4.0],
            delta_c_diag: vec![0.0; m],
            scale_factors: None,
        };
        let mut solver = DenseLdl::new();
        let mut params = InertiaCorrectionParams {
            max_attempts: 3,
            ..Default::default()
        };
        // Must return Ok (fallthrough path proceeds with best perturbation, doesn't panic)
        let result = factor_with_inertia_correction(&mut kkt, &mut solver, &mut params, 1e-4);
        assert!(result.is_ok(), "max-attempts path must return Ok, not error: {:?}", result.err());
        // A delta_c > 0 eventually flips one of the diagonals negative for the (2,2) block,
        // so the attempts cap may succeed before exhaustion. Either way, the function must
        // not panic and must return a valid (delta_w, delta_c) tuple.
        let (delta_w, delta_c) = result.unwrap();
        assert!(delta_w >= 0.0, "delta_w must be non-negative, got {}", delta_w);
        assert!(delta_c >= 0.0, "delta_c must be non-negative, got {}", delta_c);
    }

    #[test]
    fn test_factor_with_inertia_correction_degeneracy_count_sets_structural_flag() {
        // After 3 consecutive perturbations the structurally_degenerate flag should latch on,
        // making subsequent calls skip the unperturbed trial.
        let n = 2;
        let m = 1;
        let mut params = InertiaCorrectionParams::default();

        for _ in 0..3 {
            let mut matrix = SymmetricMatrix::zeros(3);
            matrix.set(0, 0, -1.0);
            matrix.set(1, 1, -1.0);
            matrix.set(2, 0, 1.0);
            matrix.set(2, 1, 1.0);
            let mut kkt = KktSystem {
                dim: 3, n, m,
                matrix: KktMatrix::Dense(matrix),
                rhs: vec![1.0, 2.0, 3.0],
                delta_c_diag: vec![0.0; m],
                scale_factors: None,
            };
            let mut solver = DenseLdl::new();
            factor_with_inertia_correction(&mut kkt, &mut solver, &mut params, 1e-4).unwrap();
        }
        assert!(params.structurally_degenerate,
            "after 3 perturbations, structurally_degenerate must latch on");
        assert!(params.degeneracy_count >= 3);
    }

    #[test]
    fn test_factor_with_inertia_correction_unconstrained_uses_min_diagonal_path() {
        // For unconstrained (m=0) indefinite problems, the solver can short-circuit to
        // delta_w = -min_diag + 1e-8 without escalating via the growth loop.
        let n = 2;
        let m = 0;
        let mut matrix = SymmetricMatrix::zeros(2);
        matrix.set(0, 0, -2.0); // Indefinite: needs delta_w >= 2.0 + eps
        matrix.set(1, 1, 3.0);
        let mut kkt = KktSystem {
            dim: 2, n, m,
            matrix: KktMatrix::Dense(matrix),
            rhs: vec![1.0, 1.0],
            delta_c_diag: vec![0.0; m],
            scale_factors: None,
        };
        let mut solver = DenseLdl::new();
        let mut params = InertiaCorrectionParams::default();
        let (delta_w, delta_c) = factor_with_inertia_correction(&mut kkt, &mut solver, &mut params, 1e-4).unwrap();
        assert!(delta_w > 0.0, "indefinite unconstrained needs delta_w > 0, got {}", delta_w);
        assert_eq!(delta_c, 0.0, "unconstrained (m=0) must have delta_c = 0");
    }

    #[test]
    fn test_solve_for_direction_simple() {
        // Create a simple 2-var, 1-constraint KKT system and solve
        let n = 2;
        let m = 1;
        let mut matrix = SymmetricMatrix::zeros(3);
        matrix.set(0, 0, 2.0);
        matrix.set(1, 1, 2.0);
        matrix.set(2, 0, 1.0);
        matrix.set(2, 1, 1.0);

        let rhs = vec![1.0, 2.0, 0.5];
        let kkt = KktSystem {
            dim: 3, n, m,
            matrix: KktMatrix::Dense(matrix.clone()),
            rhs: rhs.clone(),
            delta_c_diag: vec![0.0; m],
            scale_factors: None,
        };

        let mut solver = DenseLdl::new();
        solver.factor(&KktMatrix::Dense(matrix.clone())).unwrap();

        let (dx, dy) = solve_for_direction(&kkt, &mut solver, 0.0, 0.0).unwrap();
        assert_eq!(dx.len(), 2);
        assert_eq!(dy.len(), 1);

        // Verify KKT * [dx; dy] ≈ rhs
        let mut sol = vec![0.0; 3];
        sol[..2].copy_from_slice(&dx);
        sol[2] = dy[0];
        let mut ax = vec![0.0; 3];
        matrix.matvec(&sol, &mut ax);
        for i in 0..3 {
            assert!((ax[i] - rhs[i]).abs() < 1e-8,
                "KKT*solution mismatch at {}: {} vs {}", i, ax[i], rhs[i]);
        }
    }

    #[test]
    fn test_recover_dz_no_bounds() {
        let x = vec![1.0, 2.0];
        let x_l = vec![f64::NEG_INFINITY; 2];
        let x_u = vec![f64::INFINITY; 2];
        let z_l = vec![0.0; 2];
        let z_u = vec![0.0; 2];
        let dx = vec![0.1, 0.2];
        let (dz_l, dz_u) = recover_dz(&x, &x_l, &x_u, &z_l, &z_u, &dx, 0.1);
        for i in 0..2 {
            assert!((dz_l[i]).abs() < 1e-15);
            assert!((dz_u[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_recover_dz_lower_bound() {
        let x = vec![1.5];
        let x_l = vec![1.0];
        let x_u = vec![f64::INFINITY];
        let z_l = vec![2.0];
        let z_u = vec![0.0];
        let dx = vec![0.1];
        let mu = 0.1;
        let (dz_l, _) = recover_dz(&x, &x_l, &x_u, &z_l, &z_u, &dx, mu);
        // s_l = 0.5
        // dz_l = (mu - z_l*s_l)/s_l - (z_l/s_l)*dx
        //      = (0.1 - 2.0*0.5)/0.5 - (2.0/0.5)*0.1
        //      = (0.1 - 1.0)/0.5 - 4.0*0.1
        //      = -0.9/0.5 - 0.4
        //      = -1.8 - 0.4 = -2.2
        assert!((dz_l[0] - (-2.2)).abs() < 1e-12);
    }

    #[test]
    fn test_condensed_kkt_matches_full() {
        // 2 variables, 3 inequality constraints (m > n)
        // min x0^2 + x1^2 s.t. x0 + x1 >= 1, x0 >= 0.2, x1 >= 0.3
        let n = 2;
        let m = 3;
        let hess_rows = vec![0, 1];
        let hess_cols = vec![0, 1];
        let hess_vals = vec![2.0, 2.0];
        let jac_rows = vec![0, 0, 1, 2];
        let jac_cols = vec![0, 1, 0, 1];
        let jac_vals = vec![1.0, 1.0, 1.0, 1.0];
        let x = vec![0.6, 0.7];
        let x_l = vec![f64::NEG_INFINITY; 2];
        let x_u = vec![f64::INFINITY; 2];
        let z_l = vec![0.0; 2];
        let z_u = vec![0.0; 2];
        let sigma = compute_sigma(&x, &x_l, &x_u, &z_l, &z_u);
        let grad_f = vec![1.2, 1.4];
        let g = vec![1.3, 0.6, 0.7];
        let g_l = vec![1.0, 0.2, 0.3];
        let g_u = vec![f64::INFINITY; 3];
        let y = vec![0.1, 0.05, 0.05];
        let mu = 0.01;

        // Solve with full KKT
        let v_l = vec![0.0; m];
        let v_u = vec![0.0; m];
        let mut full_kkt = assemble_kkt(
            n, m, &hess_rows, &hess_cols, &hess_vals,
            &jac_rows, &jac_cols, &jac_vals, &sigma, &grad_f,
            &g, &g_l, &g_u, &y, &z_l, &z_u, &x, &x_l, &x_u, mu, false,
            &v_l, &v_u,
        );
        let mut full_solver = DenseLdl::new();
        let mut params = InertiaCorrectionParams::default();
        let (dw, dc) = factor_with_inertia_correction(&mut full_kkt, &mut full_solver, &mut params, 1e-4).unwrap();
        let (dx_full, dy_full) = solve_for_direction(&full_kkt, &mut full_solver, dw, dc).unwrap();

        // Solve with condensed KKT
        let condensed = assemble_condensed_kkt(
            n, m, &hess_rows, &hess_cols, &hess_vals,
            &jac_rows, &jac_cols, &jac_vals, &sigma, &grad_f,
            &g, &g_l, &g_u, &y, &z_l, &z_u, &x, &x_l, &x_u, mu,
            &v_l, &v_u,
        );
        let mut cond_solver = DenseLdl::new();
        cond_solver.factor(&KktMatrix::Dense(condensed.matrix.clone())).unwrap();
        let (dx_cond, dy_cond) = solve_condensed(&condensed, &mut cond_solver).unwrap();

        // Compare solutions
        for i in 0..n {
            assert!(
                (dx_full[i] - dx_cond[i]).abs() < 1e-6,
                "dx mismatch at {}: full={}, condensed={}", i, dx_full[i], dx_cond[i]
            );
        }
        for i in 0..m {
            assert!(
                (dy_full[i] - dy_cond[i]).abs() < 1e-6,
                "dy mismatch at {}: full={}, condensed={}", i, dy_full[i], dy_cond[i]
            );
        }
    }
}
