//! Full augmented KKT matrix assembly matching Ipopt 3.14's `IpStdAugSystemSolver`.
//!
//! Variable order: `[x ; s ; y_c ; y_d]` with dimension `n + n_d + n_c + n_d`,
//! where `n_c` is the number of equality constraints and `n_d` is the number
//! of inequality constraints (each carries one slack `s`).
//!
//! Block layout (lower triangle only — the `KktMatrix` container symmetrizes):
//!
//! ```text
//!             x          s         y_c       y_d
//!        +----------+----------+---------+---------+
//!    x   | W+Σx+δxI |          |         |         |   (0,0)
//!    s   |          |  Σs+δsI  |         |         |   (1,1)
//!   y_c  |   J_c    |          |  -δc·I  |         |   (2,0), (2,2)
//!   y_d  |   J_d    |   -I     |         |  -δd·I  |   (3,0), (3,1), (3,3)
//!        +----------+----------+---------+---------+
//! ```
//!
//! Cross-references (paths under `ref/Ipopt/src/Algorithm/`):
//!   * Block layout, signs, perturbation merge: `IpStdAugSystemSolver.cpp:251-465`
//!   * Σ_x / Σ_s materialization (where `Px_L`/`Px_U`/`Pd_L`/`Pd_U` are absorbed
//!     into Σ before being passed in): `IpIpoptCalculatedQuantities.cpp:3501-3549`
//!   * Caller (passes Σ, δ, J_c, J_d to assembler): `IpPDFullSpaceSolver.cpp:475`
//!
//! This module implements ONLY the matrix assembly + Σ helpers + the
//! constraint-partition mapping. RHS construction (A2), Δz/Δv recovery (A3),
//! iterative refinement (A4), inertia target (A5), perturbation ladder (A6),
//! and IPM wiring (A7) live in separate modules / tasks.

use crate::convergence::is_equality_constraint;
use crate::linear_solver::{KktMatrix, LinearSolver, SolverError};

/// Splits a flat `m`-constraint vector into equality and inequality subsets,
/// matching Ipopt's internal split between `J_c` (equalities, dim `n_c`) and
/// `J_d` (inequalities with slacks, dim `n_d`).
///
/// `eq_pos[i]` is `Some(k)` iff constraint `i` is an equality and is the
/// `k`-th entry of the equality subset. `ineq_pos[i]` is `Some(k)` for
/// inequalities. Exactly one of the two is `Some` for each `i`.
#[derive(Debug, Clone)]
pub struct ConstraintPartition {
    pub n_c: usize,
    pub n_d: usize,
    pub eq_pos: Vec<Option<usize>>,
    pub ineq_pos: Vec<Option<usize>>,
    /// Inverse of `ineq_pos`: `ineq_to_constraint[k]` is the global constraint
    /// index `i` of the `k`-th inequality. Useful when the slack vector `s`
    /// is indexed in `[0, n_d)`-space.
    pub ineq_to_constraint: Vec<usize>,
}

impl ConstraintPartition {
    pub fn new(g_l: &[f64], g_u: &[f64]) -> Self {
        let m = g_l.len();
        debug_assert_eq!(g_u.len(), m);
        let mut eq_pos = vec![None; m];
        let mut ineq_pos = vec![None; m];
        let mut ineq_to_constraint = Vec::new();
        let mut n_c = 0usize;
        let mut n_d = 0usize;
        for i in 0..m {
            if is_equality_constraint(g_l[i], g_u[i]) {
                eq_pos[i] = Some(n_c);
                n_c += 1;
            } else {
                ineq_pos[i] = Some(n_d);
                ineq_to_constraint.push(i);
                n_d += 1;
            }
        }
        Self { n_c, n_d, eq_pos, ineq_pos, ineq_to_constraint }
    }
}

/// Σ_s = Pd_L · diag(v_L / s_L) + Pd_U · diag(v_U / s_U), one entry per
/// inequality constraint (length `n_d`). The projection matrices `Pd_L` /
/// `Pd_U` zero out unbounded sides, matching `IpIpoptCalculatedQuantities.cpp:3540-3543`.
///
/// Reads `s` (the slack iterate, length `m`), `g_l` / `g_u` (constraint
/// bounds, length `m`), `v_l` / `v_u` (slack-bound multipliers, length `m`),
/// and the partition (which selects the `n_d` inequality rows).
pub fn compute_sigma_s(
    partition: &ConstraintPartition,
    s: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    v_l: &[f64],
    v_u: &[f64],
) -> Vec<f64> {
    let mut sigma_s = vec![0.0; partition.n_d];
    for (k, &i) in partition.ineq_to_constraint.iter().enumerate() {
        if g_l[i].is_finite() {
            let slack = (s[i] - g_l[i]).max(1e-20);
            sigma_s[k] += v_l[i] / slack;
        }
        if g_u[i].is_finite() {
            let slack = (g_u[i] - s[i]).max(1e-20);
            sigma_s[k] += v_u[i] / slack;
        }
    }
    sigma_s
}

/// Result of an augmented-KKT assembly. Mirrors `kkt::KktSystem` minus
/// fields that don't apply to the new layout (e.g. condensed Σ_s
/// post-recovery; that lives in A3).
pub struct AugKktSystem {
    /// Total dimension `n + n_d + n_c + n_d`.
    pub dim: usize,
    pub n: usize,
    pub n_c: usize,
    pub n_d: usize,
    /// The 4-block symmetric matrix.
    pub matrix: KktMatrix,
    /// δ_c diagonal stored as a positive scalar; the matrix entry at
    /// `(n+n_d+k, n+n_d+k)` for `k in 0..n_c` is `-delta_c`. Per-row
    /// storage is unnecessary because Ipopt's caller passes a single
    /// scalar (`IpPDFullSpaceSolver.cpp:475`); kept here for symmetry
    /// with `kkt::KktSystem.delta_c_diag`.
    pub delta_c: f64,
    /// δ_d diagonal stored as a positive scalar; matrix entry at
    /// `(n+n_d+n_c+k, n+n_d+n_c+k)` for `k in 0..n_d` is `-delta_d`.
    pub delta_d: f64,
    /// δ_x / δ_s already merged into the (0,0) and (1,1) diagonals.
    /// Stored here so iterative refinement (A4) can reconstruct the
    /// original (unperturbed) matvec.
    pub delta_x: f64,
    pub delta_s: f64,
}

/// Assemble the 4-block augmented KKT matrix. RHS is the caller's
/// responsibility — see A2.
///
/// Σ_x and Σ_s arrive already-projected: `sigma_x[i] = 0` for fully
/// unbounded variables, ditto `sigma_s[k]` for fully unbounded inequality
/// rows. This matches Ipopt's `IpoptCalculatedQuantities` output where
/// `AddMSinvZ` zeros out unbounded slots before the assembler is called.
///
/// `partition` carries the eq/ineq split derived from the constraint bounds
/// (`ConstraintPartition::new`).
///
/// Jacobian triplets `(jac_rows, jac_cols, jac_vals)` are in flat
/// `m × n` triplet form; each entry is routed to either the `(2,0)` block
/// (if the row is an equality) or `(3,0)` (if it's an inequality).
///
/// Hessian triplets `(hess_rows, hess_cols, hess_vals)` populate the (0,0)
/// block. The caller must pass them in lower-triangular form, matching the
/// existing `kkt::assemble_kkt` convention.
#[allow(clippy::too_many_arguments)]
pub fn assemble_aug_kkt(
    n: usize,
    partition: &ConstraintPartition,
    hess_rows: &[usize],
    hess_cols: &[usize],
    hess_vals: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    sigma_x: &[f64],
    sigma_s: &[f64],
    delta_x: f64,
    delta_s: f64,
    delta_c: f64,
    delta_d: f64,
    use_sparse: bool,
) -> AugKktSystem {
    let n_c = partition.n_c;
    let n_d = partition.n_d;
    let dim = n + n_d + n_c + n_d;

    debug_assert_eq!(sigma_x.len(), n);
    debug_assert_eq!(sigma_s.len(), n_d);

    // Block start row/col (each block is square except the cross blocks):
    let s_off = n;                  // (1,1) start
    let yc_off = n + n_d;           // (2,0)/(2,2) start
    let yd_off = n + n_d + n_c;     // (3,0)/(3,1)/(3,3) start

    // Capacity hint: H + Σx + Σs + δc·I + (-I) + δd·I + J_c + J_d.
    let capacity = hess_rows.len() + jac_rows.len() + n + n_d + n_c + n_d;
    let mut matrix = if use_sparse {
        KktMatrix::zeros_sparse(dim, capacity)
    } else {
        KktMatrix::zeros_dense(dim)
    };

    // (0,0): W + diag(Σ_x + δ_x). Per IpStdAugSystemSolver.cpp:331-371.
    for (idx, (&row, &col)) in hess_rows.iter().zip(hess_cols.iter()).enumerate() {
        let v = hess_vals[idx];
        if v.is_nan() || v.is_infinite() {
            log::warn!("NaN/Inf in Hessian at ({}, {}): {}", row, col, v);
        }
        matrix.add(row, col, v);
    }
    for i in 0..n {
        matrix.add(i, i, sigma_x[i] + delta_x);
    }

    // (1,1): diag(Σ_s + δ_s). Per IpStdAugSystemSolver.cpp:374-398.
    // Unbounded inequality rows have sigma_s[k] = 0, leaving just δ_s on the
    // diagonal — that's what makes the slack variable well-defined when Σ_s
    // would otherwise vanish.
    for k in 0..n_d {
        matrix.add(s_off + k, s_off + k, sigma_s[k] + delta_s);
    }

    // (2,0): J_c (equality rows of the Jacobian). Per IpStdAugSystemSolver.cpp:401.
    // (3,0): J_d (inequality rows of the Jacobian). Per IpStdAugSystemSolver.cpp:432.
    for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
        let v = jac_vals[idx];
        if let Some(k_c) = partition.eq_pos[row] {
            matrix.add(yc_off + k_c, col, v);
        } else if let Some(k_d) = partition.ineq_pos[row] {
            matrix.add(yd_off + k_d, col, v);
        }
    }

    // (3,1): -I (slack coupling). Per IpStdAugSystemSolver.cpp:436-438.
    // Every inequality row k_d gets a -1 at (yd_off+k_d, s_off+k_d).
    for k in 0..n_d {
        matrix.add(yd_off + k, s_off + k, -1.0);
    }

    // (2,2): -δ_c · I. Per IpStdAugSystemSolver.cpp:415, 423.
    if delta_c != 0.0 {
        for k in 0..n_c {
            matrix.add(yc_off + k, yc_off + k, -delta_c);
        }
    }

    // (3,3): -δ_d · I. Per IpStdAugSystemSolver.cpp:451, 459.
    if delta_d != 0.0 {
        for k in 0..n_d {
            matrix.add(yd_off + k, yd_off + k, -delta_d);
        }
    }

    AugKktSystem {
        dim,
        n,
        n_c,
        n_d,
        matrix,
        delta_c,
        delta_d,
        delta_x,
        delta_s,
    }
}

/// The eight outer RHS components of Ipopt's primal-dual Newton system,
/// **stored in Ipopt's pre-flip convention** (i.e. `+grad_lag`, `+complementarity`).
/// The augmented solve below uses the negated/folded form so its linear-solver
/// output is the actual Newton step — matching ripopt's convention that
/// `solver.solve(rhs, sol)` returns `sol = K^{-1}·rhs` with no further sign flip.
///
/// Slots match `IteratesVector` from `IpIteratesVector.hpp`:
/// `(x, s, y_c, y_d, z_L, z_U, v_L, v_U)`. See
/// `IpPDSearchDirCalc.cpp:75-118` for how Ipopt fills the eight slots.
///
/// The four bound-multiplier slots (`z_L`, `z_U`, `v_L`, `v_U`) are kept
/// here in `m`/`n`-space (one entry per bound, zero for unbounded) so they
/// can be reused both for `AddMSinvZ` folding and `SinvBlrmZMTdBr` recovery.
pub struct OuterRhs {
    /// `rhs_x = grad_lag_x = ∇f + Jc^T·y_c + Jd^T·y_d − Px_L·z_L + Px_U·z_U + κ_d damping`.
    /// Indexed `[0, n)`.
    pub rhs_x: Vec<f64>,
    /// `rhs_s = grad_lag_s = −y_d − Pd_L·v_L + Pd_U·v_U + κ_d damping`.
    /// Indexed in `n_d`-space (one entry per inequality slack).
    pub rhs_s: Vec<f64>,
    /// `rhs_y_c = c(x)` for equality constraints. Indexed in `n_c`-space.
    pub rhs_y_c: Vec<f64>,
    /// `rhs_y_d = d(x) − s` for inequality constraints. Indexed in `n_d`-space.
    pub rhs_y_d: Vec<f64>,
    /// `rhs_z_L = (x − x_L) · z_L − μ` (per-variable, zero for unbounded).
    /// Indexed `[0, n)`.
    pub rhs_z_l: Vec<f64>,
    pub rhs_z_u: Vec<f64>,
    /// `rhs_v_L = (s − g_L) · v_L − μ` (per-inequality, zero for unbounded sides).
    /// Indexed in `m`-space — caller can ignore equality entries which are zero.
    pub rhs_v_l: Vec<f64>,
    pub rhs_v_u: Vec<f64>,
}

/// Build the eight outer RHS components.
///
/// Inputs come straight from the IPM state. `j_t_y` is `J^T·y` (the full
/// `m`-row Jacobian transposed against the full `m`-vector of multipliers).
/// We let the caller compute it once and pass it in — it is also needed
/// for the IR matvec.
///
/// `kappa_d` is the bound damping coefficient; pass 0.0 to disable. Damping
/// adds `±κ_d·μ` to one-sided-bounded entries of `rhs_x` and `rhs_s` per
/// `IpIpoptCalculatedQuantities.cpp:2131-2227`.
#[allow(clippy::too_many_arguments)]
pub fn build_outer_rhs(
    n: usize,
    partition: &ConstraintPartition,
    grad_f: &[f64],
    j_t_y: &[f64],
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    s: &[f64],
    g: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    y: &[f64],
    v_l: &[f64],
    v_u: &[f64],
    mu: f64,
    kappa_d: f64,
) -> OuterRhs {
    let n_c = partition.n_c;
    let n_d = partition.n_d;

    // rhs_x = ∇f + J^T·y − Px_L·z_L + Px_U·z_U + κ_d damping for one-sided bounds.
    let mut rhs_x = vec![0.0; n];
    for i in 0..n {
        let mut r = grad_f[i] + j_t_y[i];
        let l_fin = x_l[i].is_finite();
        let u_fin = x_u[i].is_finite();
        if l_fin {
            r -= z_l[i];
        }
        if u_fin {
            r += z_u[i];
        }
        if kappa_d > 0.0 && (l_fin ^ u_fin) {
            // One-sided bound: damping pushes toward the open side.
            // Sign mirrors `IpIpoptCalculatedQuantities.cpp:2172-2173`:
            //   +κ_d·μ on lower-only, −κ_d·μ on upper-only.
            if l_fin {
                r += kappa_d * mu;
            } else {
                r -= kappa_d * mu;
            }
        }
        rhs_x[i] = r;
    }

    // rhs_s (n_d entries) = −y_d − Pd_L·v_L + Pd_U·v_U + κ_d damping per
    // `IpIpoptCalculatedQuantities.cpp:2182-2227`. Iterate in n_d-space.
    let mut rhs_s = vec![0.0; n_d];
    for (k, &i) in partition.ineq_to_constraint.iter().enumerate() {
        let mut r = -y[i];
        let l_fin = g_l[i].is_finite();
        let u_fin = g_u[i].is_finite();
        if l_fin {
            r -= v_l[i];
        }
        if u_fin {
            r += v_u[i];
        }
        if kappa_d > 0.0 && (l_fin ^ u_fin) {
            if l_fin {
                r += kappa_d * mu;
            } else {
                r -= kappa_d * mu;
            }
        }
        rhs_s[k] = r;
    }

    // rhs_y_c = c(x) for equality rows.
    let mut rhs_y_c = vec![0.0; n_c];
    for i in 0..g.len() {
        if let Some(k) = partition.eq_pos[i] {
            // Equality target value is g_l[i] = g_u[i].
            rhs_y_c[k] = g[i] - g_l[i];
        }
    }

    // rhs_y_d = d(x) − s for inequality rows.
    let mut rhs_y_d = vec![0.0; n_d];
    for (k, &i) in partition.ineq_to_constraint.iter().enumerate() {
        rhs_y_d[k] = g[i] - s[i];
    }

    // Bound complementarity residuals: rhs_z_L = z_L · (x − x_L) − μ for
    // bounded variables; zero for unbounded. Per `IpIpoptCalculatedQuantities.cpp:2422-2428`.
    let mut rhs_z_l = vec![0.0; n];
    let mut rhs_z_u = vec![0.0; n];
    for i in 0..n {
        if x_l[i].is_finite() {
            let s_l = (x[i] - x_l[i]).max(1e-20);
            rhs_z_l[i] = z_l[i] * s_l - mu;
        }
        if x_u[i].is_finite() {
            let s_u = (x_u[i] - x[i]).max(1e-20);
            rhs_z_u[i] = z_u[i] * s_u - mu;
        }
    }

    // Slack-bound complementarity residuals (m-indexed for caller convenience;
    // equality rows stay 0).
    let m = g.len();
    let mut rhs_v_l = vec![0.0; m];
    let mut rhs_v_u = vec![0.0; m];
    for &i in partition.ineq_to_constraint.iter() {
        if g_l[i].is_finite() {
            let s_l = (s[i] - g_l[i]).max(1e-20);
            rhs_v_l[i] = v_l[i] * s_l - mu;
        }
        if g_u[i].is_finite() {
            let s_u = (g_u[i] - s[i]).max(1e-20);
            rhs_v_u[i] = v_u[i] * s_u - mu;
        }
    }

    OuterRhs {
        rhs_x,
        rhs_s,
        rhs_y_c,
        rhs_y_d,
        rhs_z_l,
        rhs_z_u,
        rhs_v_l,
        rhs_v_u,
    }
}

/// Fold the eight-block outer RHS into the four-block augmented RHS
/// **with the sign convention that `K · sol = aug_rhs` directly returns
/// the Newton step `Δ`** (no post-hoc flip needed).
///
/// Ipopt's flow is: `Solve(α=−1)` builds `augRhs_ipopt = +grad_lag + AddMSinvZ(rhs_z…)`
/// (`IpPDFullSpaceSolver.cpp:418-424`), solves `K · sol = +augRhs_ipopt`, and
/// finally negates `sol` (`:355` `res.Scal(α=-1)`). We bake the negation in:
/// `aug_rhs = −augRhs_ipopt`, so the linear solver output IS the step.
///
/// Algebraically, after substituting `rhs_z_L = S_L_x·z_L − μ` etc., the
/// folded x-block reduces to `−∇f − J^T·y + μ/s_L − μ/s_U` which exactly
/// matches `kkt::assemble_kkt`'s existing convention — see the comment
/// block at `kkt.rs:201-243` for the equivalence derivation.
///
/// Returns a single dense vector of length `n + n_d + n_c + n_d` indexed in
/// the same order as `assemble_aug_kkt`: `[x; s; y_c; y_d]`.
pub fn fold_aug_rhs(
    n: usize,
    partition: &ConstraintPartition,
    rhs: &OuterRhs,
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    s: &[f64],
    g_l: &[f64],
    g_u: &[f64],
) -> Vec<f64> {
    let n_c = partition.n_c;
    let n_d = partition.n_d;
    let dim = n + n_d + n_c + n_d;
    let mut aug = vec![0.0; dim];

    // x-block: aug.x = −rhs_x − Px_L·(rhs_z_L / S_L_x) + Px_U·(rhs_z_U / S_U_x).
    // Per `IpPDFullSpaceSolver.cpp:418-419`, before the α=−1 outer flip it would
    // be `+rhs_x + Px_L·… − Px_U·…`. We negate everything to fold the flip in.
    for i in 0..n {
        let mut r = -rhs.rhs_x[i];
        if x_l[i].is_finite() {
            let s_l = (x[i] - x_l[i]).max(1e-20);
            r -= rhs.rhs_z_l[i] / s_l;
        }
        if x_u[i].is_finite() {
            let s_u = (x_u[i] - x[i]).max(1e-20);
            r += rhs.rhs_z_u[i] / s_u;
        }
        aug[i] = r;
    }

    // s-block: same pattern with Pd_L / Pd_U and slack_s_L / slack_s_U.
    for (k, &i) in partition.ineq_to_constraint.iter().enumerate() {
        let mut r = -rhs.rhs_s[k];
        if g_l[i].is_finite() {
            let s_l = (s[i] - g_l[i]).max(1e-20);
            r -= rhs.rhs_v_l[i] / s_l;
        }
        if g_u[i].is_finite() {
            let s_u = (g_u[i] - s[i]).max(1e-20);
            r += rhs.rhs_v_u[i] / s_u;
        }
        aug[n + k] = r;
    }

    // y_c block: aug.y_c = −rhs_y_c. Per `:475` rhs.y_c flows through unchanged
    // before the α=−1 flip; we negate.
    for k in 0..n_c {
        aug[n + n_d + k] = -rhs.rhs_y_c[k];
    }

    // y_d block: aug.y_d = −rhs_y_d.
    for k in 0..n_d {
        aug[n + n_d + n_c + k] = -rhs.rhs_y_d[k];
    }

    aug
}

/// The eight-block Newton step recovered from a successful augmented solve.
/// Δx, Δs, Δy_c, Δy_d come directly from the linear-solver output (already
/// in step convention thanks to the negated RHS); Δz_L, Δz_U, Δv_L, Δv_U
/// are recovered by inverting the `AddMSinvZ` substitution per
/// `IpPDFullSpaceSolver.cpp:653-656`.
pub struct AugStep {
    pub dx: Vec<f64>,
    /// Slack step in m-space (zero for equality rows, the inequality entries
    /// carry the n_d-space step recovered from the augmented solve).
    pub ds: Vec<f64>,
    pub dy_c: Vec<f64>,
    pub dy_d: Vec<f64>,
    /// Combined constraint multiplier step in m-space, formed by writing
    /// `dy_c` into the equality positions and `dy_d` into the inequality
    /// positions per `ConstraintPartition`. Convenient for IPM-side wiring
    /// (the legacy `state.y` is m-indexed and combines both subsets).
    pub dy_m: Vec<f64>,
    /// Per-variable Δz_L (zero for variables without a lower bound).
    pub dz_l: Vec<f64>,
    pub dz_u: Vec<f64>,
    /// Per-constraint Δv_L (m-indexed; zero for equalities and unbounded sides).
    pub dv_l: Vec<f64>,
    pub dv_u: Vec<f64>,
}

/// Split the augmented solve output into (Δx, Δs, Δy_c, Δy_d) and recover
/// the four bound-multiplier blocks.
///
/// `aug_sol` is the linear solver's output for `K · sol = fold_aug_rhs(...)`.
/// Because `fold_aug_rhs` baked Ipopt's α=−1 flip into the RHS, `aug_sol`
/// IS the Newton step (no further negation).
///
/// SinvBlrmZMTdBr semantics from `IpMatrix.hpp:101-112`:
/// `X = S^{-1} · (R + α · Z · M^T · D)` with the `(α, M)` arguments
/// `(−1, Px_L)`, `(+1, Px_U)`, `(−1, Pd_L)`, `(+1, Pd_U)` per
/// `IpPDFullSpaceSolver.cpp:653-656`.
///
/// In the post-flip step convention (Δx is the actual step), the formulas
/// reduce to:
///   Δz_L = (μ − z_L·s_L)/s_L − (z_L/s_L) · Δx_L
///   Δz_U = (μ − z_U·s_U)/s_U + (z_U/s_U) · Δx_U
///   Δv_L = (μ − v_L·s_L)/s_L − (v_L/s_L) · Δs[k]
///   Δv_U = (μ − v_U·s_U)/s_U + (v_U/s_U) · Δs[k]
///
/// matching `kkt::recover_dz` / `kkt::recover_dv` exactly. The algebra is
/// derived at the head of `kkt_aug.rs` near `fold_aug_rhs`.
#[allow(clippy::too_many_arguments)]
pub fn recover_step(
    n: usize,
    partition: &ConstraintPartition,
    aug_sol: &[f64],
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    s: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    v_l: &[f64],
    v_u: &[f64],
    mu: f64,
) -> AugStep {
    let n_c = partition.n_c;
    let n_d = partition.n_d;
    let m = g_l.len();
    let expected = n + n_d + n_c + n_d;
    debug_assert_eq!(aug_sol.len(), expected);

    let dx = aug_sol[0..n].to_vec();
    let ds_nd = aug_sol[n..n + n_d].to_vec();
    let dy_c = aug_sol[n + n_d..n + n_d + n_c].to_vec();
    let dy_d = aug_sol[n + n_d + n_c..expected].to_vec();

    // Bound-multiplier recovery in step convention (post-flip).
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

    // Slack-bound recovery: ds is in n_d-space, indexed by inequality position.
    let mut dv_l = vec![0.0; m];
    let mut dv_u = vec![0.0; m];
    for (k, &i) in partition.ineq_to_constraint.iter().enumerate() {
        let dsk = ds_nd[k];
        if g_l[i].is_finite() {
            let s_l = (s[i] - g_l[i]).max(1e-20);
            dv_l[i] = (mu - v_l[i] * s_l) / s_l - (v_l[i] / s_l) * dsk;
        }
        if g_u[i].is_finite() {
            let s_u = (g_u[i] - s[i]).max(1e-20);
            dv_u[i] = (mu - v_u[i] * s_u) / s_u + (v_u[i] / s_u) * dsk;
        }
    }

    // Promote ds back to m-space (zero for equalities) so downstream code
    // that uses the IPM's m-indexed `state.s` arrays can stay unchanged.
    let mut ds = vec![0.0; m];
    for (k, &i) in partition.ineq_to_constraint.iter().enumerate() {
        ds[i] = ds_nd[k];
    }

    // Build combined m-space dy: dy_c entries land at equality positions,
    // dy_d entries at inequality positions.
    let mut dy_m = vec![0.0; m];
    for (i, eq) in partition.eq_pos.iter().enumerate() {
        if let Some(k) = eq {
            dy_m[i] = dy_c[*k];
        }
    }
    for (k, &i) in partition.ineq_to_constraint.iter().enumerate() {
        dy_m[i] = dy_d[k];
    }

    AugStep { dx, ds, dy_c, dy_d, dy_m, dz_l, dz_u, dv_l, dv_u }
}

/// Result of `solve_aug_with_ir`: the linear-solver output (Newton step Δ in
/// the `[x; s; y_c; y_d]` order) plus IR diagnostics.
pub struct AugSolveResult {
    /// Final solution vector (length `n + n_d + n_c + n_d`).
    pub sol: Vec<f64>,
    /// Number of refinement iterations actually executed (≥ 1; the initial
    /// solve counts as iteration 1).
    pub ir_iters: usize,
    /// Final residual ratio (see `residual_ratio` below). `None` if the
    /// solve produced a NaN/Inf solution that we could not refine.
    pub final_ratio: Option<f64>,
}

/// Compute the residual `A · sol − rhs` of the *augmented* system, where
/// `A` is `aug.matrix` (which already has δ_x, δ_s, δ_c, δ_d folded into
/// its diagonal). Per `IpPDFullSpaceSolver.cpp:280-308`.
fn aug_residual(aug: &AugKktSystem, sol: &[f64], rhs: &[f64]) -> Vec<f64> {
    let mut resid = vec![0.0; aug.dim];
    aug.matrix.matvec(sol, &mut resid);
    for i in 0..aug.dim {
        resid[i] -= rhs[i];
    }
    resid
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()))
}

/// Compute the residual ratio used by Ipopt's IR termination test:
/// ratio = ‖resid‖_∞ / ( min(‖sol‖_∞, 1e6·‖rhs‖_∞) + ‖rhs‖_∞ ).
/// Per `IpPDFullSpaceSolver.cpp:316-329`.
fn residual_ratio(resid: &[f64], sol: &[f64], rhs: &[f64]) -> f64 {
    let nr = inf_norm(resid);
    let ns = inf_norm(sol);
    let nrhs = inf_norm(rhs);
    let denom = ns.min(1e6 * nrhs) + nrhs;
    if denom == 0.0 {
        if nr == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        nr / denom
    }
}

/// Solve `K · sol = rhs_aug` and refine iteratively, matching
/// `IpPDFullSpaceSolver.cpp:253-346`.
///
/// `aug.matrix` must already be factored by the caller (see
/// `factor_with_inertia_correction_cached`); this function only does
/// backsolves and matvec residuals.
///
/// Termination conditions (in priority order):
///   1. NaN/Inf in `sol` → return immediately with `final_ratio = None`.
///   2. `ir_iters ≥ min_iters` AND `ratio ≤ ratio_max` → success.
///   3. `ratio` increased (last_ratio · improvement_factor < ratio) → bail.
///   4. `ir_iters ≥ max_iters` → bail (caller decides whether to escalate δ).
pub fn solve_aug_with_ir(
    solver: &mut dyn LinearSolver,
    aug: &AugKktSystem,
    rhs_aug: &[f64],
    min_iters: usize,
    max_iters: usize,
    ratio_max: f64,
    improvement_factor: f64,
) -> Result<AugSolveResult, SolverError> {
    debug_assert_eq!(rhs_aug.len(), aug.dim);
    let mut sol = vec![0.0; aug.dim];
    solver.solve(rhs_aug, &mut sol)?;

    if sol.iter().any(|x| !x.is_finite()) {
        return Ok(AugSolveResult { sol, ir_iters: 1, final_ratio: None });
    }

    let mut ir_iters = 1usize;
    let mut last_ratio = f64::INFINITY;
    loop {
        let resid = aug_residual(aug, &sol, rhs_aug);
        let ratio = residual_ratio(&resid, &sol, rhs_aug);

        if ir_iters >= min_iters && ratio <= ratio_max {
            return Ok(AugSolveResult { sol, ir_iters, final_ratio: Some(ratio) });
        }
        if ir_iters >= max_iters {
            return Ok(AugSolveResult { sol, ir_iters, final_ratio: Some(ratio) });
        }
        if last_ratio.is_finite() && ratio > last_ratio * improvement_factor {
            return Ok(AugSolveResult { sol, ir_iters, final_ratio: Some(ratio) });
        }

        // Refinement: solve K · Δsol = resid (note: resid = A·sol − rhs, so
        // Δsol = K^{-1} · resid corrects via sol -= Δsol).
        let mut correction = vec![0.0; aug.dim];
        solver.solve(&resid, &mut correction)?;
        if correction.iter().any(|x| !x.is_finite()) {
            return Ok(AugSolveResult { sol, ir_iters, final_ratio: Some(ratio) });
        }
        for i in 0..aug.dim {
            sol[i] -= correction[i];
        }
        last_ratio = ratio;
        ir_iters += 1;
    }
}

/// Apply the four-block perturbation pattern `(δ_x, δ_s, δ_c, δ_d)` to a
/// fresh clone of `base` and return the perturbed matrix.
///
/// Per Ipopt 3.14 `IpPDPerturbationHandler.cpp:405-413`, δ_x and δ_s share
/// a single primal regularization scalar, and δ_c and δ_d share a single
/// constraint regularization scalar. The four arguments here are kept
/// separate so callers can experiment with asymmetric perturbations
/// (e.g. δ_d=0 while δ_c>0), but `factor_aug_with_inertia_correction`
/// always passes δ_x=δ_s and δ_c=δ_d.
pub fn apply_aug_perturbation(
    base: &KktMatrix,
    n: usize,
    n_c: usize,
    n_d: usize,
    delta_x: f64,
    delta_s: f64,
    delta_c: f64,
    delta_d: f64,
) -> KktMatrix {
    let mut p = base.clone();
    if delta_x != 0.0 {
        p.add_diagonal_range(0, n, delta_x);
    }
    if delta_s != 0.0 && n_d > 0 {
        p.add_diagonal_range(n, n + n_d, delta_s);
    }
    if delta_c != 0.0 && n_c > 0 {
        p.add_diagonal_range(n + n_d, n + n_d + n_c, -delta_c);
    }
    if delta_d != 0.0 && n_d > 0 {
        p.add_diagonal_range(n + n_d + n_c, n + n_d + n_c + n_d, -delta_d);
    }
    p
}

/// Drive the PD perturbation handler against the augmented (4-block)
/// system. This is the augmented analog of
/// `kkt::factor_with_inertia_correction`, with two changes:
///
///   * Inertia target: `positive == n + n_d`, `negative == n_c + n_d`,
///     `zero == 0`. Per `IpPDFullSpaceSolver.cpp:539-541`, the linear
///     system has dimension `n + n_d + n_c + n_d` and the negative-eigenvalue
///     count must equal the number of dual rows (`n_c + n_d`).
///   * Perturbation applied to all four diagonal ranges via
///     `apply_aug_perturbation`. δ_x = δ_s and δ_c = δ_d, matching
///     `IpPDPerturbationHandler.cpp:405-413`.
///
/// `aug.matrix` is the unperturbed matrix from `assemble_aug_kkt(...,
/// δ_x=0, δ_s=0, δ_c=0, δ_d=0, ...)` — the perturbations are added on
/// top of it inside this function so the warm-start ladder can probe
/// (δ_x, δ_c) pairs cheaply. On success, the perturbed matrix is
/// installed into `aug.matrix` and `aug.delta_*` are updated.
///
/// References: full reference walk is in `kkt.rs:954-1098`
/// (`factor_with_inertia_correction`); this function inlines the same
/// algorithm with the augmented inertia target.
pub fn factor_aug_with_inertia_correction(
    aug: &mut AugKktSystem,
    solver: &mut dyn LinearSolver,
    params: &mut crate::kkt::InertiaCorrectionParams,
    mu: f64,
    rhs: &[f64],
) -> Result<(f64, f64), SolverError> {
    use crate::kkt::DegenType;
    let n = aug.n;
    let n_c = aug.n_c;
    let n_d = aug.n_d;
    let target_pos = n + n_d;
    let target_neg = n_c + n_d;

    // === consider_new_system ===
    let (mut dx, mut dc) = match consider_new_system_pub(params, mu) {
        Some(pair) => pair,
        None => {
            return Err(SolverError::NumericalFailure(
                "PDPerturbationHandler: delta_x cap exhausted at consider_new_system (aug)"
                    .to_string(),
            ));
        }
    };

    let mut tried_increase_quality = false;
    for _attempt in 0..params.max_attempts {
        // δ_x = δ_s, δ_c = δ_d.
        let perturbed = apply_aug_perturbation(&aug.matrix, n, n_c, n_d, dx, dx, dc, dc);
        let inertia = solver.factor(&perturbed)?;

        let (positive, negative, zero) = match inertia {
            Some(i) => (i.positive, i.negative, i.zero),
            None => {
                // Backend can't report inertia: accept if the probe solve
                // produces a finite vector. Mirrors kkt.rs:1006-1025.
                let mut probe = vec![0.0; aug.dim];
                if solver.solve(rhs, &mut probe).is_ok()
                    && probe.iter().all(|v| v.is_finite())
                {
                    aug.matrix = perturbed;
                    aug.delta_x = dx;
                    aug.delta_s = dx;
                    aug.delta_c = dc;
                    aug.delta_d = dc;
                    if dx > 0.0 {
                        params.delta_w_last = dx;
                    }
                    if dc > 0.0 {
                        params.delta_c_last = dc;
                    }
                    return Ok((dx, dc));
                }
                if !perturb_for_wrong_inertia_pub(params, mu) {
                    return Err(SolverError::NumericalFailure(
                        "PDPerturbationHandler: cap exhausted (no inertia) (aug)".to_string(),
                    ));
                }
                dx = params.delta_x_curr;
                dc = params.delta_c_curr;
                continue;
            }
        };

        if std::env::var("RIPOPT_TRACE_PERTURB").is_ok() {
            eprintln!(
                "  aug perturb-trace: dx={:.2e} dc={:.2e} -> inertia(+{}, -{}, 0:{}) target({}+, {}-, 0)",
                dx, dc, positive, negative, zero, target_pos, target_neg
            );
        }
        let exact_ok = positive == target_pos && negative == target_neg && zero == 0;
        if exact_ok {
            // Probe-solve sanity check (NaN/Inf only — IR handles accuracy).
            let mut probe = vec![0.0; aug.dim];
            let probe_ok = solver.solve(rhs, &mut probe).is_ok()
                && probe.iter().all(|v| v.is_finite());
            if probe_ok {
                aug.matrix = perturbed;
                aug.delta_x = dx;
                aug.delta_s = dx;
                aug.delta_c = dc;
                aug.delta_d = dc;
                if dx > 0.0 {
                    params.delta_w_last = dx;
                }
                if dc > 0.0 {
                    params.delta_c_last = dc;
                }
                return Ok((dx, dc));
            }
            if !tried_increase_quality && solver.increase_quality() {
                tried_increase_quality = true;
                continue;
            }
            if !perturb_for_singularity_pub(params, mu) {
                return Err(SolverError::NumericalFailure(
                    "PDPerturbationHandler: cap exhausted in singularity probe (aug)"
                        .to_string(),
                ));
            }
            dx = params.delta_x_curr;
            dc = params.delta_c_curr;
            continue;
        }

        let singular = zero > 0;
        if singular {
            if !tried_increase_quality && solver.increase_quality() {
                tried_increase_quality = true;
                continue;
            }
            if !perturb_for_singularity_pub(params, mu) {
                return Err(SolverError::NumericalFailure(
                    "PDPerturbationHandler: cap exhausted in singularity probe (aug)"
                        .to_string(),
                ));
            }
        } else {
            if !perturb_for_wrong_inertia_pub(params, mu) {
                return Err(SolverError::NumericalFailure(
                    "PDPerturbationHandler: cap exhausted in wrong-inertia ladder (aug)"
                        .to_string(),
                ));
            }
        }
        dx = params.delta_x_curr;
        dc = params.delta_c_curr;
        // Suppress unused-warning for consts captured from caller.
        let _ = (target_pos, target_neg, DegenType::NotYetDetermined);
    }

    Err(SolverError::NumericalFailure(format!(
        "PDPerturbationHandler: max_attempts={} exhausted (aug, last δ_w={:.2e}, δ_c={:.2e})",
        params.max_attempts, dx, dc
    )))
}

// ---------------------------------------------------------------------
// Crate-private bridges to InertiaCorrectionParams' private methods.
// These exist because `consider_new_system`, `perturb_for_singularity`,
// and `perturb_for_wrong_inertia` are private to kkt.rs. We import
// them via the public surface added in this section to keep kkt.rs
// untouched until A7 wiring.
// ---------------------------------------------------------------------

fn consider_new_system_pub(
    params: &mut crate::kkt::InertiaCorrectionParams,
    mu: f64,
) -> Option<(f64, f64)> {
    params.consider_new_system_aug(mu)
}

fn perturb_for_singularity_pub(
    params: &mut crate::kkt::InertiaCorrectionParams,
    mu: f64,
) -> bool {
    params.perturb_for_singularity_aug(mu)
}

fn perturb_for_wrong_inertia_pub(
    params: &mut crate::kkt::InertiaCorrectionParams,
    mu: f64,
) -> bool {
    params.perturb_for_wrong_inertia_aug(mu)
}

/// Default IR knobs matching Ipopt 3.14 `IpPDFullSpaceSolver`:
///   `residual_ratio_max = 1e-10`, `residual_improvement_factor = 0.999999999`,
///   `min_refinement_steps = 1`, `max_refinement_steps = 10`.
/// (`IpPDFullSpaceSolver.cpp:97-113` — the `Add*Number` registration.)
pub const IR_RATIO_MAX_DEFAULT: f64 = 1e-10;
pub const IR_IMPROVEMENT_FACTOR_DEFAULT: f64 = 0.999_999_999;
pub const IR_MIN_STEPS_DEFAULT: usize = 1;
pub const IR_MAX_STEPS_DEFAULT: usize = 10;

/// End-to-end "compute the next Newton step via the augmented system" driver.
///
/// Bundles A1 (matrix assembly), A2 (RHS construction), A3 (recovery), A4
/// (IR-protected solve), and A5/A6 (perturbation handler) into a single
/// callable surface. The IPM driver (A7 wiring) calls this once per
/// outer-iteration KKT solve in place of the legacy condensed path
/// (`kkt::assemble_kkt` + `kkt::factor_with_inertia_correction` +
/// `kkt::solve_for_direction_with_ir` + `kkt::recover_dz`/`recover_dv`/`recover_ds`).
///
/// Inputs are the IPM state in scattered form to avoid a circular module
/// dependency on `ipm::SolverState`. The caller is the IPM driver, which
/// already has all of these in hand.
///
/// `kkt_atags` is the input-fingerprint tuple that the cached-factor path
/// uses; for the augmented path we don't yet share the cache with the
/// condensed system (T3.25 wiring is condensed-only) so this argument is
/// passed but the cache short-circuit is bypassed. A7 follow-up: extend
/// `FactorCache` to key on the augmented dim/atags pair.
#[allow(clippy::too_many_arguments)]
pub fn aug_step_from_state(
    n: usize,
    grad_f: &[f64],
    hess_rows: &[usize],
    hess_cols: &[usize],
    hess_vals: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    s: &[f64],
    g: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    y: &[f64],
    v_l: &[f64],
    v_u: &[f64],
    mu: f64,
    kappa_d: f64,
    use_sparse: bool,
    solver: &mut dyn LinearSolver,
    perturbation: &mut crate::kkt::InertiaCorrectionParams,
) -> Result<(AugStep, f64, f64, AugKktSystem), SolverError> {
    let m = g.len();
    let partition = ConstraintPartition::new(g_l, g_u);

    // Σ_x: `IpIpoptCalculatedQuantities.cpp:3501-3540`. Pre-projected — zero
    // for fully unbounded variables.
    let mut sigma_x = vec![0.0; n];
    for i in 0..n {
        if x_l[i].is_finite() {
            let s_l = (x[i] - x_l[i]).max(1e-20);
            sigma_x[i] += z_l[i] / s_l;
        }
        if x_u[i].is_finite() {
            let s_u = (x_u[i] - x[i]).max(1e-20);
            sigma_x[i] += z_u[i] / s_u;
        }
    }
    let sigma_s = compute_sigma_s(&partition, s, g_l, g_u, v_l, v_u);

    // Assemble the unperturbed matrix (δ_* zeroed; perturbation layered on
    // by factor_aug_with_inertia_correction).
    let mut aug = assemble_aug_kkt(
        n, &partition,
        hess_rows, hess_cols, hess_vals,
        jac_rows, jac_cols, jac_vals,
        &sigma_x, &sigma_s,
        0.0, 0.0, 0.0, 0.0,
        use_sparse,
    );

    // Compute J^T·y once (used both by RHS construction and the unperturbed-matvec
    // sanity check that the IR loop performs internally).
    let mut j_t_y = vec![0.0; n];
    for (idx, &row) in jac_rows.iter().enumerate() {
        let col = jac_cols[idx];
        let v = jac_vals[idx];
        if row < m && col < n {
            j_t_y[col] += v * y[row];
        }
    }

    // Build the eight outer RHS slots and fold them into the four-block
    // augmented form (with Ipopt's α=−1 flip already baked in so the
    // linear solver output IS the step).
    let outer = build_outer_rhs(
        n, &partition, grad_f, &j_t_y,
        x, x_l, x_u, z_l, z_u,
        s, g, g_l, g_u, y, v_l, v_u,
        mu, kappa_d,
    );
    let aug_rhs = fold_aug_rhs(n, &partition, &outer, x, x_l, x_u, s, g_l, g_u);

    // Drive the perturbation ladder (A5/A6 inertia target = (n+n_d, n_c+n_d, 0)).
    let (dw, dc) =
        factor_aug_with_inertia_correction(&mut aug, solver, perturbation, mu, &aug_rhs)?;

    // IR-protected solve (A4).
    let result = solve_aug_with_ir(
        solver, &aug, &aug_rhs,
        IR_MIN_STEPS_DEFAULT,
        IR_MAX_STEPS_DEFAULT,
        IR_RATIO_MAX_DEFAULT,
        IR_IMPROVEMENT_FACTOR_DEFAULT,
    )?;

    // Recover the eight-block step.
    let step = recover_step(
        n, &partition, &result.sol,
        x, x_l, x_u, z_l, z_u,
        s, g_l, g_u, v_l, v_u, mu,
    );
    // A7.7: return the perturbed `AugKktSystem` so callers (line search → SOC)
    // can reuse the just-installed factorization without reassembling and
    // re-factoring the same matrix.
    Ok((step, dw, dc, aug))
}

/// Probing-oracle defaults from Ipopt 3.14:
///   `sigma_max = 1e2` (registered in `IpQualityFunctionMuOracle.cpp:54-61`,
///   shared default consumed by `IpProbingMuOracle.cpp:117-119`).
pub const PROBING_SIGMA_MAX_DEFAULT: f64 = 1e2;

/// Apply Ipopt's Probing μ oracle to an affine step and return the new μ.
///
/// Ports `IpProbingMuOracle::CalculateMu` (`IpProbingMuOracle.cpp:47-133`) and
/// `IpProbingMuOracle::CalculateAffineMu` (`IpProbingMuOracle.cpp:135-232`).
///
/// Algorithm:
///   1. `α_p_aff = primal_frac_to_the_bound(τ=1, Δx, Δs)` over the four primal
///      slack groups (slack_x_L, slack_x_U, slack_s_L, slack_s_U).
///      `IpProbingMuOracle.cpp:95`.
///   2. `α_d_aff = dual_frac_to_the_bound(τ=1, Δz_L, Δz_U, Δv_L, Δv_U)` over
///      the four bound multipliers. `IpProbingMuOracle.cpp:97`.
///      Note: τ is the literal `1.0`, not `tau_min`. Affine probe is the
///      pure full-step Mehrotra ratio.
///   3. `μ_aff = (Σ_groups Σ_i slack'_i · mult'_i) / ncomp` where
///      `slack' = slack + α_p · Δslack`, `mult' = mult + α_d · Δmult`, and
///      `ncomp = dim(z_L) + dim(z_U) + dim(v_L) + dim(v_U)` (total scalar
///      bound count, NOT count of nonempty groups).
///      `IpProbingMuOracle.cpp:162-231`.
///   4. `σ = min((μ_aff / μ_curr)^q, σ_max)` with `q = 3` hard-coded
///      (`IpProbingMuOracle.cpp:117`) and `σ_max = 1e2` default.
///   5. `μ_new = clamp(σ · μ_curr, μ_min, μ_max)`.
///      `IpProbingMuOracle.cpp:131`.
///
/// Returns `None` if there are no bound multipliers (ncomp = 0) or
/// `mu_curr <= 0`, signalling the IPM should keep its current μ.
#[allow(clippy::too_many_arguments)]
pub fn aug_probing_mu_from_affine(
    n: usize,
    partition: &ConstraintPartition,
    step_aff: &AugStep,
    x: &[f64], x_l: &[f64], x_u: &[f64], z_l: &[f64], z_u: &[f64],
    s: &[f64], g_l: &[f64], g_u: &[f64], v_l: &[f64], v_u: &[f64],
    mu_curr: f64,
    sigma_max: f64,
    mu_min: f64,
    mu_max: f64,
) -> Option<f64> {
    if mu_curr <= 0.0 {
        return None;
    }
    let tau = 1.0_f64;

    // α_p_aff over the four primal slack groups. Per-group test:
    // `slack + α · P^T · Δx > 0` ⇒ `α ≤ τ · slack / (−P^T · Δx)` when the
    // direction is negative. Only Px_L and Pd_L contribute with sign +α
    // (slack growth direction is +Δx); Px_U and Pd_U with sign −α.
    let mut alpha_p = 1.0_f64;
    for i in 0..n {
        if x_l[i].is_finite() && step_aff.dx[i] < 0.0 {
            let s_l = (x[i] - x_l[i]).max(1e-20);
            alpha_p = alpha_p.min(tau * s_l / (-step_aff.dx[i]));
        }
        if x_u[i].is_finite() && step_aff.dx[i] > 0.0 {
            let s_u = (x_u[i] - x[i]).max(1e-20);
            alpha_p = alpha_p.min(tau * s_u / step_aff.dx[i]);
        }
    }
    for &i in partition.ineq_to_constraint.iter() {
        let dsi = step_aff.ds[i];
        if g_l[i].is_finite() && dsi < 0.0 {
            let s_l = (s[i] - g_l[i]).max(1e-20);
            alpha_p = alpha_p.min(tau * s_l / (-dsi));
        }
        if g_u[i].is_finite() && dsi > 0.0 {
            let s_u = (g_u[i] - s[i]).max(1e-20);
            alpha_p = alpha_p.min(tau * s_u / dsi);
        }
    }

    // α_d_aff over the four bound multipliers. Each multiplier z must stay
    // ≥ 0, so when Δz < 0 the bound is `α ≤ τ · z / (−Δz)`.
    let mut alpha_d = 1.0_f64;
    for i in 0..n {
        if x_l[i].is_finite() && step_aff.dz_l[i] < 0.0 {
            alpha_d = alpha_d.min(tau * z_l[i].max(1e-20) / (-step_aff.dz_l[i]));
        }
        if x_u[i].is_finite() && step_aff.dz_u[i] < 0.0 {
            alpha_d = alpha_d.min(tau * z_u[i].max(1e-20) / (-step_aff.dz_u[i]));
        }
    }
    for &i in partition.ineq_to_constraint.iter() {
        if g_l[i].is_finite() && step_aff.dv_l[i] < 0.0 {
            alpha_d = alpha_d.min(tau * v_l[i].max(1e-20) / (-step_aff.dv_l[i]));
        }
        if g_u[i].is_finite() && step_aff.dv_u[i] < 0.0 {
            alpha_d = alpha_d.min(tau * v_u[i].max(1e-20) / (-step_aff.dv_u[i]));
        }
    }
    let alpha_p = alpha_p.clamp(0.0, 1.0);
    let alpha_d = alpha_d.clamp(0.0, 1.0);

    // μ_aff via CalculateAffineMu. Sum slack' · mult' over the four groups.
    // For Px_L: slack' = slack_x_L + α_p · Δx, mult' = z_L + α_d · Δz_L.
    // For Px_U: slack' = slack_x_U − α_p · Δx, mult' = z_U + α_d · Δz_U.
    let mut sum = 0.0_f64;
    let mut ncomp = 0_usize;
    for i in 0..n {
        if x_l[i].is_finite() {
            let s_l = (x[i] - x_l[i]).max(1e-20);
            let s_new = s_l + alpha_p * step_aff.dx[i];
            let z_new = z_l[i] + alpha_d * step_aff.dz_l[i];
            sum += s_new * z_new;
            ncomp += 1;
        }
        if x_u[i].is_finite() {
            let s_u = (x_u[i] - x[i]).max(1e-20);
            let s_new = s_u - alpha_p * step_aff.dx[i];
            let z_new = z_u[i] + alpha_d * step_aff.dz_u[i];
            sum += s_new * z_new;
            ncomp += 1;
        }
    }
    for &i in partition.ineq_to_constraint.iter() {
        let dsi = step_aff.ds[i];
        if g_l[i].is_finite() {
            let s_l = (s[i] - g_l[i]).max(1e-20);
            let s_new = s_l + alpha_p * dsi;
            let v_new = v_l[i] + alpha_d * step_aff.dv_l[i];
            sum += s_new * v_new;
            ncomp += 1;
        }
        if g_u[i].is_finite() {
            let s_u = (g_u[i] - s[i]).max(1e-20);
            let s_new = s_u - alpha_p * dsi;
            let v_new = v_u[i] + alpha_d * step_aff.dv_u[i];
            sum += s_new * v_new;
            ncomp += 1;
        }
    }
    if ncomp == 0 {
        return None;
    }
    let mu_aff = sum / ncomp as f64;
    if !mu_aff.is_finite() || mu_aff < 0.0 {
        return None;
    }
    let sigma = (mu_aff / mu_curr).powi(3).min(sigma_max);
    let mu_new = (sigma * mu_curr).max(mu_min).min(mu_max);
    if !mu_new.is_finite() {
        return None;
    }
    Some(mu_new)
}

/// Augmented-system Probing-oracle driver: solve once for the affine step,
/// derive μ_new, then solve a second time at μ_new reusing the factor.
///
/// Ports the dispatcher pattern of `IpProbingMuOracle::CalculateMu` followed
/// by the regular primal-dual step at the new μ. Factor reuse matches Ipopt's
/// `dummy_cache_` flow at `IpPDFullSpaceSolver.cpp:429-482`: the matrix
/// `W + Σ` does not depend on μ, so the same factorization serves both the
/// affine probe and the Newton step at μ_new.
///
/// Returns `(step, μ_new, δ_w, δ_c)`. The IPM driver should overwrite
/// `state.mu` with `μ_new` so subsequent line-search and convergence
/// computations see the chosen μ.
///
/// Falls back to the plain `aug_step_from_state` semantics (no oracle) when
/// `aug_probing_mu_from_affine` returns `None` — the affine step was unable
/// to suggest a μ (no bound multipliers, or NaN/Inf), so we keep `mu_curr`.
#[allow(clippy::too_many_arguments)]
pub fn aug_step_from_state_mehrotra(
    n: usize,
    grad_f: &[f64],
    hess_rows: &[usize],
    hess_cols: &[usize],
    hess_vals: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    s: &[f64],
    g: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    y: &[f64],
    v_l: &[f64],
    v_u: &[f64],
    mu_curr: f64,
    kappa_d: f64,
    sigma_max: f64,
    mu_min: f64,
    mu_max: f64,
    use_sparse: bool,
    solver: &mut dyn LinearSolver,
    perturbation: &mut crate::kkt::InertiaCorrectionParams,
) -> Result<(AugStep, f64, f64, f64, AugKktSystem), SolverError> {
    let m = g.len();
    let partition = ConstraintPartition::new(g_l, g_u);

    // Σ_x and Σ_s do not depend on μ — they're functions of bound multipliers
    // and slacks at the current iterate. Reused across the affine and Newton
    // solves. (Per `IpIpoptCalculatedQuantities.cpp:3501-3540`.)
    let mut sigma_x = vec![0.0; n];
    for i in 0..n {
        if x_l[i].is_finite() {
            let s_l = (x[i] - x_l[i]).max(1e-20);
            sigma_x[i] += z_l[i] / s_l;
        }
        if x_u[i].is_finite() {
            let s_u = (x_u[i] - x[i]).max(1e-20);
            sigma_x[i] += z_u[i] / s_u;
        }
    }
    let sigma_s = compute_sigma_s(&partition, s, g_l, g_u, v_l, v_u);

    let mut aug = assemble_aug_kkt(
        n, &partition,
        hess_rows, hess_cols, hess_vals,
        jac_rows, jac_cols, jac_vals,
        &sigma_x, &sigma_s,
        0.0, 0.0, 0.0, 0.0,
        use_sparse,
    );

    let mut j_t_y = vec![0.0; n];
    for (idx, &row) in jac_rows.iter().enumerate() {
        let col = jac_cols[idx];
        let v = jac_vals[idx];
        if row < m && col < n {
            j_t_y[col] += v * y[row];
        }
    }

    // Affine RHS per `IpProbingMuOracle.cpp:63-72`:
    //   x      ← grad_lag_x (no κ_d damping)
    //   s      ← grad_lag_s (no κ_d damping)
    //   y_c    ← c(x)
    //   y_d    ← d(x) − s
    //   z_*/v_* ← slack · mult (no μ subtracted)
    // build_outer_rhs(mu=0, kappa_d=0) produces this exactly.
    let outer_aff = build_outer_rhs(
        n, &partition, grad_f, &j_t_y,
        x, x_l, x_u, z_l, z_u,
        s, g, g_l, g_u, y, v_l, v_u,
        0.0, 0.0,
    );
    let aug_rhs_aff = fold_aug_rhs(n, &partition, &outer_aff, x, x_l, x_u, s, g_l, g_u);

    // Drive the perturbation ladder using the affine RHS as the probe.
    let (dw, dc) = factor_aug_with_inertia_correction(
        &mut aug, solver, perturbation, mu_curr, &aug_rhs_aff,
    )?;

    // Affine solve (IR-protected). Reuses the factorization just installed.
    let aff_result = solve_aug_with_ir(
        solver, &aug, &aug_rhs_aff,
        IR_MIN_STEPS_DEFAULT, IR_MAX_STEPS_DEFAULT,
        IR_RATIO_MAX_DEFAULT, IR_IMPROVEMENT_FACTOR_DEFAULT,
    )?;
    let step_aff = recover_step(
        n, &partition, &aff_result.sol,
        x, x_l, x_u, z_l, z_u,
        s, g_l, g_u, v_l, v_u, 0.0,
    );

    // Probing oracle → μ_new. Falls back to mu_curr when no bound multipliers
    // exist or the affine step is degenerate.
    let mu_new = aug_probing_mu_from_affine(
        n, &partition, &step_aff,
        x, x_l, x_u, z_l, z_u,
        s, g_l, g_u, v_l, v_u,
        mu_curr, sigma_max, mu_min, mu_max,
    ).unwrap_or(mu_curr);

    // Newton step at μ_new (with kappa_d damping). Build a fresh OuterRhs
    // and fold; the matrix is unchanged so we reuse the existing factor.
    let outer = build_outer_rhs(
        n, &partition, grad_f, &j_t_y,
        x, x_l, x_u, z_l, z_u,
        s, g, g_l, g_u, y, v_l, v_u,
        mu_new, kappa_d,
    );
    let aug_rhs = fold_aug_rhs(n, &partition, &outer, x, x_l, x_u, s, g_l, g_u);

    let result = solve_aug_with_ir(
        solver, &aug, &aug_rhs,
        IR_MIN_STEPS_DEFAULT, IR_MAX_STEPS_DEFAULT,
        IR_RATIO_MAX_DEFAULT, IR_IMPROVEMENT_FACTOR_DEFAULT,
    )?;
    let step = recover_step(
        n, &partition, &result.sol,
        x, x_l, x_u, z_l, z_u,
        s, g_l, g_u, v_l, v_u, mu_new,
    );

    Ok((step, mu_new, dw, dc, aug))
}

/// Second-order correction (SOC) step for the augmented system.
///
/// Ports `IpFilterLSAcceptor::TrySecondOrderCorrection` (`IpFilterLSAcceptor.cpp:550-640`)
/// `soc_method = 0`: build a fresh Newton RHS at the current iterate, then
/// overwrite the y_c and y_d slots with the SOC-accumulated `c_soc`/`dms_soc`
/// residuals. Bound-multiplier slots stay at the regular `slack·mult − μ`
/// form (Ipopt's `curr_relaxed_compl_*` differs from `curr_compl_*` only by
/// κ_σ damping; we use the standard form for v0.8).
///
/// Re-factors the matrix per call. A7.7 (factor caching) — see
/// `aug_soc_solve_dx_factored` for the cached entry that reuses the upstream
/// Newton step's factorization.
///
/// Returns `Some((dx_soc, ds_d_soc))` on success — `dx_soc` length `n`,
/// `ds_d_soc` length `n_d` (inequality-only slack step indexed by
/// `partition.ineq_to_constraint[k]`). `None` if the linear solve fails.
#[allow(clippy::too_many_arguments)]
pub fn aug_soc_solve_dx(
    n: usize,
    grad_f: &[f64],
    hess_rows: &[usize],
    hess_cols: &[usize],
    hess_vals: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    s: &[f64],
    g: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    y: &[f64],
    v_l: &[f64],
    v_u: &[f64],
    mu: f64,
    kappa_d: f64,
    use_sparse: bool,
    solver: &mut dyn LinearSolver,
    perturbation: &mut crate::kkt::InertiaCorrectionParams,
    // SOC-modified constraint residuals. Lengths must match partition sizes.
    c_soc: &[f64],
    dms_soc: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let m = g.len();
    let partition = ConstraintPartition::new(g_l, g_u);
    if c_soc.len() != partition.n_c || dms_soc.len() != partition.n_d {
        return None;
    }
    let n_d = partition.n_d;

    let mut sigma_x = vec![0.0; n];
    for i in 0..n {
        if x_l[i].is_finite() {
            let s_l = (x[i] - x_l[i]).max(1e-20);
            sigma_x[i] += z_l[i] / s_l;
        }
        if x_u[i].is_finite() {
            let s_u = (x_u[i] - x[i]).max(1e-20);
            sigma_x[i] += z_u[i] / s_u;
        }
    }
    let sigma_s = compute_sigma_s(&partition, s, g_l, g_u, v_l, v_u);

    let mut aug = assemble_aug_kkt(
        n, &partition,
        hess_rows, hess_cols, hess_vals,
        jac_rows, jac_cols, jac_vals,
        &sigma_x, &sigma_s,
        0.0, 0.0, 0.0, 0.0,
        use_sparse,
    );

    let mut j_t_y = vec![0.0; n];
    for (idx, &row) in jac_rows.iter().enumerate() {
        let col = jac_cols[idx];
        let v = jac_vals[idx];
        if row < m && col < n {
            j_t_y[col] += v * y[row];
        }
    }

    // Newton RHS at current state (x/s/z_*/v_* slots), then overwrite y_c/y_d
    // with the SOC-accumulated residuals (`IpFilterLSAcceptor.cpp:581-582`,
    // method 0).
    let mut outer = build_outer_rhs(
        n, &partition, grad_f, &j_t_y,
        x, x_l, x_u, z_l, z_u,
        s, g, g_l, g_u, y, v_l, v_u,
        mu, kappa_d,
    );
    outer.rhs_y_c.copy_from_slice(c_soc);
    outer.rhs_y_d.copy_from_slice(dms_soc);
    let aug_rhs = fold_aug_rhs(n, &partition, &outer, x, x_l, x_u, s, g_l, g_u);

    // Factor (with inertia correction) and IR-solve. The matrix is identical
    // to the upstream Newton step — A7.7 will share this factorization.
    if factor_aug_with_inertia_correction(&mut aug, solver, perturbation, mu, &aug_rhs)
        .is_err()
    {
        return None;
    }
    let result = match solve_aug_with_ir(
        solver, &aug, &aug_rhs,
        IR_MIN_STEPS_DEFAULT, IR_MAX_STEPS_DEFAULT,
        IR_RATIO_MAX_DEFAULT, IR_IMPROVEMENT_FACTOR_DEFAULT,
    ) {
        Ok(r) => r,
        Err(_) => return None,
    };
    if result.sol.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let dx_soc = result.sol[..n].to_vec();
    let ds_d_soc = result.sol[n..n + n_d].to_vec();
    Some((dx_soc, ds_d_soc))
}

/// Factor-cached SOC solve (A7.7). Reuses the `solver`'s existing
/// factorization of `aug` (installed by the upstream Newton step's
/// `factor_aug_with_inertia_correction`). Only rebuilds the RHS with the
/// SOC-overwritten y_c / y_d slots and runs IR.
///
/// Preconditions:
/// - `aug` is the perturbed `AugKktSystem` returned by `aug_step_from_state`
///   or `aug_step_from_state_mehrotra` for the same iterate.
/// - `solver` still holds the LDLᵀ factorization of `aug.matrix` (no other
///   `factor` call between the upstream step and this call).
///
/// All Σ-, Hessian-, and Jacobian-derived inputs are inputs only because
/// the RHS construction needs them (`build_outer_rhs` → J^T·y, gradient,
/// kappa_d damping); they MUST be identical to the values used to assemble
/// `aug` upstream — otherwise the factorization is stale.
#[allow(clippy::too_many_arguments)]
pub fn aug_soc_solve_dx_factored(
    n: usize,
    grad_f: &[f64],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[f64],
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    s: &[f64],
    g: &[f64],
    g_l: &[f64],
    g_u: &[f64],
    y: &[f64],
    v_l: &[f64],
    v_u: &[f64],
    mu: f64,
    kappa_d: f64,
    solver: &mut dyn LinearSolver,
    aug: &AugKktSystem,
    c_soc: &[f64],
    dms_soc: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let m = g.len();
    let partition = ConstraintPartition::new(g_l, g_u);
    if c_soc.len() != partition.n_c || dms_soc.len() != partition.n_d {
        return None;
    }
    let n_d = partition.n_d;

    let mut j_t_y = vec![0.0; n];
    for (idx, &row) in jac_rows.iter().enumerate() {
        let col = jac_cols[idx];
        let v = jac_vals[idx];
        if row < m && col < n {
            j_t_y[col] += v * y[row];
        }
    }

    let mut outer = build_outer_rhs(
        n, &partition, grad_f, &j_t_y,
        x, x_l, x_u, z_l, z_u,
        s, g, g_l, g_u, y, v_l, v_u,
        mu, kappa_d,
    );
    outer.rhs_y_c.copy_from_slice(c_soc);
    outer.rhs_y_d.copy_from_slice(dms_soc);
    let aug_rhs = fold_aug_rhs(n, &partition, &outer, x, x_l, x_u, s, g_l, g_u);

    let result = match solve_aug_with_ir(
        solver, aug, &aug_rhs,
        IR_MIN_STEPS_DEFAULT, IR_MAX_STEPS_DEFAULT,
        IR_RATIO_MAX_DEFAULT, IR_IMPROVEMENT_FACTOR_DEFAULT,
    ) {
        Ok(r) => r,
        Err(_) => return None,
    };
    if result.sol.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let dx_soc = result.sol[..n].to_vec();
    let ds_d_soc = result.sol[n..n + n_d].to_vec();
    Some((dx_soc, ds_d_soc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_pure_equality() {
        let g_l = vec![1.0, 2.0, 3.0];
        let g_u = vec![1.0, 2.0, 3.0];
        let p = ConstraintPartition::new(&g_l, &g_u);
        assert_eq!(p.n_c, 3);
        assert_eq!(p.n_d, 0);
        assert_eq!(p.eq_pos, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(p.ineq_pos, vec![None, None, None]);
        assert!(p.ineq_to_constraint.is_empty());
    }

    #[test]
    fn partition_pure_inequality() {
        let g_l = vec![0.0, f64::NEG_INFINITY];
        let g_u = vec![10.0, 5.0];
        let p = ConstraintPartition::new(&g_l, &g_u);
        assert_eq!(p.n_c, 0);
        assert_eq!(p.n_d, 2);
        assert_eq!(p.eq_pos, vec![None, None]);
        assert_eq!(p.ineq_pos, vec![Some(0), Some(1)]);
        assert_eq!(p.ineq_to_constraint, vec![0, 1]);
    }

    #[test]
    fn partition_mixed_preserves_order() {
        // eq, ineq, eq, ineq, ineq -> n_c=2, n_d=3 with positions in order
        let g_l = vec![1.0, 0.0, 5.0, f64::NEG_INFINITY, -1.0];
        let g_u = vec![1.0, 10.0, 5.0, 7.0, 1.0];
        let p = ConstraintPartition::new(&g_l, &g_u);
        assert_eq!(p.n_c, 2);
        assert_eq!(p.n_d, 3);
        assert_eq!(p.eq_pos, vec![Some(0), None, Some(1), None, None]);
        assert_eq!(p.ineq_pos, vec![None, Some(0), None, Some(1), Some(2)]);
        assert_eq!(p.ineq_to_constraint, vec![1, 3, 4]);
    }

    #[test]
    fn sigma_s_zero_for_unbounded() {
        // Two ineq rows: row 0 has only lower bound, row 1 fully unbounded.
        let g_l = vec![0.0, f64::NEG_INFINITY];
        let g_u = vec![f64::INFINITY, f64::INFINITY];
        let p = ConstraintPartition::new(&g_l, &g_u);
        // Row 1 is fully unbounded, so it's not really a "constraint" in
        // the usual sense, but `is_equality_constraint` returns false and
        // it lands in the inequality bucket. sigma_s for it should be 0.
        let s = vec![1.0, 0.0];
        let v_l = vec![2.0, 0.0];
        let v_u = vec![0.0, 0.0];
        let sigma = compute_sigma_s(&p, &s, &g_l, &g_u, &v_l, &v_u);
        assert_eq!(sigma.len(), 2);
        assert!((sigma[0] - 2.0).abs() < 1e-12); // v_l/(s-g_l) = 2/1 = 2
        assert_eq!(sigma[1], 0.0);
    }

    #[test]
    fn sigma_s_two_sided_sums_both_contributions() {
        let g_l = vec![0.0];
        let g_u = vec![10.0];
        let p = ConstraintPartition::new(&g_l, &g_u);
        let s = vec![2.0];
        let v_l = vec![3.0];
        let v_u = vec![4.0];
        let sigma = compute_sigma_s(&p, &s, &g_l, &g_u, &v_l, &v_u);
        // 3/(2-0) + 4/(10-2) = 1.5 + 0.5 = 2.0
        assert!((sigma[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn assemble_dimensions_and_block_signs() {
        // 2 vars, 1 eq + 2 ineq -> dim = 2 + 2 + 1 + 2 = 7.
        let n = 2usize;
        let g_l = vec![5.0, 0.0, f64::NEG_INFINITY];
        let g_u = vec![5.0, 10.0, 1.0];
        let p = ConstraintPartition::new(&g_l, &g_u);
        assert_eq!(p.n_c, 1);
        assert_eq!(p.n_d, 2);

        // H = diag(7, 11). Lower triangle only.
        let hess_rows = vec![0, 1];
        let hess_cols = vec![0, 1];
        let hess_vals = vec![7.0, 11.0];

        // J: row 0 (eq) -> [1, 2]; row 1 (ineq) -> [3, 0]; row 2 (ineq) -> [0, 4].
        let jac_rows = vec![0, 0, 1, 2];
        let jac_cols = vec![0, 1, 0, 1];
        let jac_vals = vec![1.0, 2.0, 3.0, 4.0];

        let sigma_x = vec![0.5, 0.25];
        let sigma_s = vec![0.1, 0.2];
        let sys = assemble_aug_kkt(
            n, &p,
            &hess_rows, &hess_cols, &hess_vals,
            &jac_rows, &jac_cols, &jac_vals,
            &sigma_x, &sigma_s,
            1e-3, 2e-3, 3e-4, 4e-4,
            false, // dense
        );
        assert_eq!(sys.dim, 7);
        assert_eq!(sys.n_c, 1);
        assert_eq!(sys.n_d, 2);

        // (0,0): H + Σx + δx
        assert!((sys.matrix.get(0, 0) - (7.0 + 0.5 + 1e-3)).abs() < 1e-12);
        assert!((sys.matrix.get(1, 1) - (11.0 + 0.25 + 1e-3)).abs() < 1e-12);

        // (1,1): Σs + δs at (n+0, n+0) and (n+1, n+1)
        assert!((sys.matrix.get(2, 2) - (0.1 + 2e-3)).abs() < 1e-12);
        assert!((sys.matrix.get(3, 3) - (0.2 + 2e-3)).abs() < 1e-12);

        // (2,0): J_c row at (n+n_d+0, ·) = (4, ·): values 1, 2 from constraint 0.
        assert!((sys.matrix.get(4, 0) - 1.0).abs() < 1e-12);
        assert!((sys.matrix.get(4, 1) - 2.0).abs() < 1e-12);

        // (3,0): J_d. Inequality row 0 (global constraint 1) -> (5, ·): [3, 0].
        //        Inequality row 1 (global constraint 2) -> (6, ·): [0, 4].
        assert!((sys.matrix.get(5, 0) - 3.0).abs() < 1e-12);
        assert!((sys.matrix.get(5, 1) - 0.0).abs() < 1e-12);
        assert!((sys.matrix.get(6, 0) - 0.0).abs() < 1e-12);
        assert!((sys.matrix.get(6, 1) - 4.0).abs() < 1e-12);

        // (3,1): -I at (5, 2) and (6, 3).
        assert!((sys.matrix.get(5, 2) - (-1.0)).abs() < 1e-12);
        assert!((sys.matrix.get(6, 3) - (-1.0)).abs() < 1e-12);

        // (2,2): -δc on (4, 4). (3,3): -δd on (5, 5) and (6, 6).
        // Note (5,5) and (6,6) overlap with the (3,3) block, NOT (1,1). Σs lives at (2,2)/(3,3) of the matrix.
        assert!((sys.matrix.get(4, 4) - (-3e-4)).abs() < 1e-12);
        assert!((sys.matrix.get(5, 5) - (-4e-4)).abs() < 1e-12);
        assert!((sys.matrix.get(6, 6) - (-4e-4)).abs() < 1e-12);
    }

    #[test]
    fn outer_rhs_pure_equality_no_slack_entries() {
        // 2 vars unbounded, 1 equality g(x) = 0 with current g(x)=0.5.
        let n = 2usize;
        let g_l = vec![0.0];
        let g_u = vec![0.0];
        let p = ConstraintPartition::new(&g_l, &g_u);
        let grad_f = vec![1.0, 2.0];
        let j_t_y = vec![0.1, 0.2];
        let x = vec![0.0, 0.0];
        let x_l = vec![f64::NEG_INFINITY; 2];
        let x_u = vec![f64::INFINITY; 2];
        let z_l = vec![0.0; 2];
        let z_u = vec![0.0; 2];
        let s = vec![0.0];
        let g = vec![0.5];
        let y = vec![0.3];
        let v_l = vec![0.0];
        let v_u = vec![0.0];
        let r = build_outer_rhs(
            n, &p, &grad_f, &j_t_y,
            &x, &x_l, &x_u, &z_l, &z_u,
            &s, &g, &g_l, &g_u, &y, &v_l, &v_u,
            1e-3, 0.0,
        );
        // No bounds → rhs_x = grad_f + j_t_y.
        assert!((r.rhs_x[0] - 1.1).abs() < 1e-12);
        assert!((r.rhs_x[1] - 2.2).abs() < 1e-12);
        // Pure equality → rhs_s and rhs_y_d empty, rhs_v_l/u all zero.
        assert!(r.rhs_s.is_empty());
        assert!(r.rhs_y_d.is_empty());
        // rhs_y_c = g - g_l = 0.5.
        assert_eq!(r.rhs_y_c.len(), 1);
        assert!((r.rhs_y_c[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn outer_rhs_kappa_d_one_sided_only() {
        // 3 vars: var 0 lower-only, var 1 upper-only, var 2 two-sided.
        // Damping should land on 0 (+κ_d·μ) and 1 (−κ_d·μ); var 2 untouched.
        let n = 3usize;
        let g_l: Vec<f64> = vec![];
        let g_u: Vec<f64> = vec![];
        let p = ConstraintPartition::new(&g_l, &g_u);
        let grad_f = vec![0.0, 0.0, 0.0];
        let j_t_y = vec![0.0, 0.0, 0.0];
        let x = vec![1.0, 1.0, 1.0];
        let x_l = vec![0.0, f64::NEG_INFINITY, 0.0];
        let x_u = vec![f64::INFINITY, 2.0, 2.0];
        let z_l = vec![5.0, 0.0, 7.0];
        let z_u = vec![0.0, 11.0, 13.0];
        let s = vec![];
        let g = vec![];
        let y = vec![];
        let v_l = vec![];
        let v_u = vec![];
        let mu = 0.01;
        let kappa_d = 1e-5;
        let r = build_outer_rhs(
            n, &p, &grad_f, &j_t_y,
            &x, &x_l, &x_u, &z_l, &z_u,
            &s, &g, &g_l, &g_u, &y, &v_l, &v_u,
            mu, kappa_d,
        );
        // var 0 (lower-only): -z_l + κ_d·μ
        assert!((r.rhs_x[0] - (-5.0 + kappa_d * mu)).abs() < 1e-12);
        // var 1 (upper-only): +z_u - κ_d·μ
        assert!((r.rhs_x[1] - (11.0 - kappa_d * mu)).abs() < 1e-12);
        // var 2 (two-sided): -z_l + z_u, no damping
        assert!((r.rhs_x[2] - (-7.0 + 13.0)).abs() < 1e-12);
    }

    #[test]
    fn fold_then_recover_round_trip_diagonal() {
        // Construct a tiny augmented system, fold the RHS, "solve" with
        // ratio-1 mock solver (exact dense LDLT via faer), recover the step,
        // and check that re-applying the matvec reproduces the folded RHS.
        // This is the smoke test the spec called for.
        use crate::linear_solver::dense::DenseLdl;
        let n = 1usize;
        // 1 ineq with both bounds.
        let g_l = vec![0.0];
        let g_u = vec![10.0];
        let p = ConstraintPartition::new(&g_l, &g_u);
        // x bounded in [0, ∞), so only z_l contributes.
        let x_l = vec![0.0];
        let x_u = vec![f64::INFINITY];
        let x = vec![1.0];
        let z_l = vec![0.5];
        let z_u = vec![0.0];
        let s = vec![3.0];
        let g = vec![3.5];     // residual d-s = 0.5
        let y = vec![0.7];
        let v_l = vec![0.4];
        let v_u = vec![0.6];
        let mu = 0.1;

        // Σ_x = z_l/(x-x_l) = 0.5/1 = 0.5.
        let sigma_x = vec![0.5];
        // Σ_s = v_l/(s-g_l) + v_u/(g_u-s) = 0.4/3 + 0.6/7
        let sigma_s = compute_sigma_s(&p, &s, &g_l, &g_u, &v_l, &v_u);

        // Hessian H = [[2.0]], Jacobian J row 0 (ineq): [3.0].
        let hess_rows = vec![0];
        let hess_cols = vec![0];
        let hess_vals = vec![2.0];
        let jac_rows = vec![0];
        let jac_cols = vec![0];
        let jac_vals = vec![3.0];

        let aug = assemble_aug_kkt(
            n, &p,
            &hess_rows, &hess_cols, &hess_vals,
            &jac_rows, &jac_cols, &jac_vals,
            &sigma_x, &sigma_s,
            0.0, 0.0, 0.0, 0.0,
            false,
        );
        assert_eq!(aug.dim, 3); // n=1, n_d=1, n_c=0 → 1+1+0+1

        let grad_f = vec![10.0];
        let j_t_y = vec![3.0 * 0.7]; // J^T·y
        let outer = build_outer_rhs(
            n, &p, &grad_f, &j_t_y,
            &x, &x_l, &x_u, &z_l, &z_u,
            &s, &g, &g_l, &g_u, &y, &v_l, &v_u,
            mu, 0.0,
        );
        let aug_rhs = fold_aug_rhs(n, &p, &outer, &x, &x_l, &x_u, &s, &g_l, &g_u);
        assert_eq!(aug_rhs.len(), 3);

        // Solve K·sol = aug_rhs with a dense solver.
        let mut solver = DenseLdl::new();
        solver.factor(&aug.matrix).unwrap();
        let mut sol = vec![0.0; 3];
        solver.solve(&aug_rhs, &mut sol).unwrap();

        // Verify residual is small.
        let mut check = vec![0.0; 3];
        aug.matrix.matvec(&sol, &mut check);
        for i in 0..3 {
            assert!(
                (check[i] - aug_rhs[i]).abs() < 1e-9,
                "row {}: matvec={}, rhs={}", i, check[i], aug_rhs[i]
            );
        }

        // Recover full step and sanity-check shapes.
        let step = recover_step(
            n, &p, &sol,
            &x, &x_l, &x_u, &z_l, &z_u,
            &s, &g_l, &g_u, &v_l, &v_u, mu,
        );
        assert_eq!(step.dx.len(), 1);
        assert_eq!(step.ds.len(), 1); // m-space
        assert_eq!(step.dy_c.len(), 0);
        assert_eq!(step.dy_d.len(), 1);
        // Lower bound on x is finite, so dz_l[0] should be defined.
        let s_l_x = x[0] - x_l[0];
        let expected_dz_l = (mu - z_l[0] * s_l_x) / s_l_x - (z_l[0] / s_l_x) * step.dx[0];
        assert!((step.dz_l[0] - expected_dz_l).abs() < 1e-12);
        assert_eq!(step.dz_u[0], 0.0);
        // Both v bounds are active.
        assert!(step.dv_l[0] != 0.0);
        assert!(step.dv_u[0] != 0.0);
    }

    #[test]
    fn aug_step_from_state_unconstrained_quadratic_lands_on_minimum() {
        // min 0.5·x^2 over x ∈ R, no constraints, no bounds.
        // KKT system is just W·dx = -∇f = -x. From x=2, expected dx = -2.
        use crate::linear_solver::dense::DenseLdl;
        let n = 1usize;
        let grad_f = vec![2.0];
        let hess_rows = vec![0];
        let hess_cols = vec![0];
        let hess_vals = vec![1.0];
        let jac_rows: Vec<usize> = vec![];
        let jac_cols: Vec<usize> = vec![];
        let jac_vals: Vec<f64> = vec![];
        let x = vec![2.0];
        let x_l = vec![f64::NEG_INFINITY];
        let x_u = vec![f64::INFINITY];
        let z_l = vec![0.0];
        let z_u = vec![0.0];
        let s: Vec<f64> = vec![];
        let g: Vec<f64> = vec![];
        let g_l: Vec<f64> = vec![];
        let g_u: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];
        let v_l: Vec<f64> = vec![];
        let v_u: Vec<f64> = vec![];
        let mu = 1e-3;
        let mut solver: Box<dyn LinearSolver> = Box::new(DenseLdl::new());
        let mut params = crate::kkt::InertiaCorrectionParams::default();
        let (step, _dw, _dc, _aug) = aug_step_from_state(
            n, &grad_f,
            &hess_rows, &hess_cols, &hess_vals,
            &jac_rows, &jac_cols, &jac_vals,
            &x, &x_l, &x_u, &z_l, &z_u,
            &s, &g, &g_l, &g_u, &y, &v_l, &v_u,
            mu, 0.0, false,
            solver.as_mut(), &mut params,
        ).unwrap();
        // Expected Newton step toward the minimizer at x=0: dx = -2.
        assert!((step.dx[0] - (-2.0)).abs() < 1e-9, "dx = {}", step.dx[0]);
        assert!(step.ds.is_empty());
        assert!(step.dy_c.is_empty());
        assert!(step.dy_d.is_empty());
    }

    #[test]
    fn apply_aug_perturbation_lands_on_correct_diagonals() {
        // n=2, n_c=1, n_d=1 → dim = 2+1+1+1 = 5.
        let mut base = KktMatrix::zeros_dense(5);
        // Identity-ish base so the perturbation is the only diagonal contribution.
        for i in 0..5 {
            base.add(i, i, 1.0);
        }
        let p = apply_aug_perturbation(&base, 2, 1, 1, 0.1, 0.2, 0.3, 0.4);
        // x-block: 1 + 0.1
        assert!((p.get(0, 0) - 1.1).abs() < 1e-12);
        assert!((p.get(1, 1) - 1.1).abs() < 1e-12);
        // s-block (index 2): 1 + 0.2
        assert!((p.get(2, 2) - 1.2).abs() < 1e-12);
        // y_c block (index 3): 1 - 0.3
        assert!((p.get(3, 3) - 0.7).abs() < 1e-12);
        // y_d block (index 4): 1 - 0.4
        assert!((p.get(4, 4) - 0.6).abs() < 1e-12);
    }

    #[test]
    fn solve_aug_with_ir_diagonal_reaches_target_ratio() {
        use crate::linear_solver::dense::DenseLdl;
        // Tiny diagonal SPD system: K = diag(2, 3, 5), rhs = [1, 1, 1].
        // Expected sol = [0.5, 1/3, 0.2]. IR should hit the ratio target in 1 step.
        let mut matrix = KktMatrix::zeros_dense(3);
        matrix.add(0, 0, 2.0);
        matrix.add(1, 1, 3.0);
        matrix.add(2, 2, 5.0);
        let aug = AugKktSystem {
            dim: 3, n: 1, n_c: 1, n_d: 1,
            matrix,
            delta_c: 0.0, delta_d: 0.0, delta_x: 0.0, delta_s: 0.0,
        };
        let rhs = vec![1.0, 1.0, 1.0];
        let mut solver = DenseLdl::new();
        solver.factor(&aug.matrix).unwrap();
        let res = solve_aug_with_ir(
            &mut solver, &aug, &rhs,
            IR_MIN_STEPS_DEFAULT,
            IR_MAX_STEPS_DEFAULT,
            IR_RATIO_MAX_DEFAULT,
            IR_IMPROVEMENT_FACTOR_DEFAULT,
        ).unwrap();
        assert!((res.sol[0] - 0.5).abs() < 1e-12);
        assert!((res.sol[1] - 1.0/3.0).abs() < 1e-12);
        assert!((res.sol[2] - 0.2).abs() < 1e-12);
        // Direct LDLT on a tiny diagonal hits machine zero — IR exits after 1
        // residual check.
        assert!(res.ir_iters <= 2);
        assert!(res.final_ratio.unwrap() < IR_RATIO_MAX_DEFAULT);
    }

    #[test]
    fn assemble_pure_equality_no_slack_block() {
        // 2 vars, 2 equality constraints -> dim = 2 + 0 + 2 + 0 = 4.
        let n = 2usize;
        let g_l = vec![0.0, 1.0];
        let g_u = vec![0.0, 1.0];
        let p = ConstraintPartition::new(&g_l, &g_u);
        assert_eq!(p.n_d, 0);
        let hess_rows = vec![0, 1];
        let hess_cols = vec![0, 1];
        let hess_vals = vec![1.0, 1.0];
        let jac_rows = vec![0, 1];
        let jac_cols = vec![0, 1];
        let jac_vals = vec![5.0, 6.0];
        let sys = assemble_aug_kkt(
            n, &p,
            &hess_rows, &hess_cols, &hess_vals,
            &jac_rows, &jac_cols, &jac_vals,
            &[0.0, 0.0], &[],
            0.0, 0.0, 0.0, 0.0,
            false,
        );
        assert_eq!(sys.dim, 4);
        // J_c rows 0,1 land at (2, 0), (3, 1)
        assert!((sys.matrix.get(2, 0) - 5.0).abs() < 1e-12);
        assert!((sys.matrix.get(3, 1) - 6.0).abs() < 1e-12);
    }
}
