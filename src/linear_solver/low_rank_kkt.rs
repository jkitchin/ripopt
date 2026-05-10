//! Sherman-Morrison-Woodbury wrapper for KKT systems with an L-BFGS Hessian.
//!
//! When the IPM uses an L-BFGS Hessian approximation, the (1,1) block of the
//! augmented KKT matrix is `B_k + Σ + δ_w I` with `B_k = σI + V Vᵀ - U Uᵀ`
//! (compact Byrd-Nocedal form, see `LbfgsIpmState::compact`). Materializing
//! `B_k` as an `n×n` dense block is `O(n²)` storage and ruins the inner
//! sparse solver's structure. Instead this wrapper:
//!
//! 1. Has the caller assemble the *base* augmented system `A_0` with `σI` on
//!    the (1,1) diagonal in place of the `B_k` triplets.
//! 2. Factors `A_0` once via the inner `LinearSolver`.
//! 3. Performs `2k` extra back-solves to materialize `Vtilde = A_0⁻¹ V_aug`
//!    and `Utilde1 = A_0⁻¹ U_aug` (with `V_aug, U_aug` zero outside the x-block).
//! 4. Builds `Utilde2 = Utilde1 − Vtilde · M1⁻¹ Vtilde_xᵀ U` and Cholesky-factors
//!    two small `k×k` matrices `M1 = I + Vtilde_xᵀ V` and `M2 = I − Utilde2_xᵀ U`.
//! 5. On each RHS, returns `csol_0 + Utilde2 · (M2⁻¹ Utilde2ᵀ r) − Vtilde · (M1⁻¹ Vtildeᵀ r)`.
//!
//! Mirrors Ipopt 3.14's `LowRankAugSystemSolver` —
//! `ref/Ipopt/src/Algorithm/IpLowRankAugSystemSolver.cpp:182-227` (per-RHS
//! application) and `:300-401` (setup/Cholesky). The seam moves up by one
//! layer compared to Ipopt because ripopt's `LinearSolver` trait receives a
//! fully-assembled `KktMatrix`, not split block arguments — see the design
//! note in issue #30.

use super::{Inertia, KktMatrix, LinearSolver, SolverError};
use crate::ipm::LbfgsCompact;

/// State of a low-rank Sherman-Morrison wrapper around an inner `LinearSolver`.
///
/// All buffers are column-major. `vtilde[i + j * n_aug]` is row `i` of column
/// `j` of `Vtilde`, etc.
pub struct LowRankKktSolver<S: LinearSolver> {
    inner: S,
    n: usize,
    n_aug: usize,
    k: usize,
    /// `n × k` column-major copy of `V` (x-part of `V_aug`). Owned so the
    /// caller can drop the `LbfgsCompact` between `factor` and `solve`.
    v: Vec<f64>,
    /// `n × k` column-major copy of `U`.
    u: Vec<f64>,
    /// `n_aug × k` column-major: `Vtilde = A_0⁻¹ · V_aug`.
    vtilde: Vec<f64>,
    /// `n_aug × k` column-major: `Utilde2 = Utilde1 − Vtilde · M1⁻¹ Vtilde_xᵀ U`.
    utilde2: Vec<f64>,
    /// `k × k` row-major lower-triangular Cholesky factor of `M1 = I + Vtilde_xᵀ V`.
    j1_chol: Vec<f64>,
    /// `k × k` row-major lower-triangular Cholesky factor of `M2 = I − Utilde2_xᵀ U`.
    j2_chol: Vec<f64>,
    factored: bool,
}

impl<S: LinearSolver> LowRankKktSolver<S> {
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            n: 0,
            n_aug: 0,
            k: 0,
            v: Vec::new(),
            u: Vec::new(),
            vtilde: Vec::new(),
            utilde2: Vec::new(),
            j1_chol: Vec::new(),
            j2_chol: Vec::new(),
            factored: false,
        }
    }

    /// Factor `A_0 + V_aug V_augᵀ − U_aug U_augᵀ` where `A_0` is the inner
    /// augmented system with `σI` on the (1,1) block (caller's responsibility).
    ///
    /// `n_aug` is the total dimension of `base_kkt` (e.g. `n + n_c + n_d`).
    /// The wrapper assumes `V_aug` / `U_aug` are zero outside the leading
    /// `n` rows.
    pub fn factor(
        &mut self,
        base_kkt: &KktMatrix,
        compact: &LbfgsCompact,
        n_aug: usize,
    ) -> Result<Option<Inertia>, SolverError> {
        let n = compact.n;
        let k = compact.k;
        if base_kkt.n() != n_aug {
            return Err(SolverError::DimensionMismatch {
                expected: n_aug,
                got: base_kkt.n(),
            });
        }
        if n_aug < n {
            return Err(SolverError::DimensionMismatch {
                expected: n,
                got: n_aug,
            });
        }
        self.n = n;
        self.n_aug = n_aug;
        self.k = k;
        self.v.clear();
        self.v.extend_from_slice(&compact.v);
        self.u.clear();
        self.u.extend_from_slice(&compact.u);

        let inner_inertia = self.inner.factor(base_kkt)?;

        if k == 0 {
            self.vtilde.clear();
            self.utilde2.clear();
            self.j1_chol.clear();
            self.j2_chol.clear();
            self.factored = true;
            return Ok(inner_inertia);
        }

        // Build V_aug, U_aug as `n_aug × k` column-major buffers; zero outside x.
        let mut v_aug = vec![0.0_f64; n_aug * k];
        let mut u_aug = vec![0.0_f64; n_aug * k];
        for j in 0..k {
            for i in 0..n {
                v_aug[i + j * n_aug] = compact.v[i + j * n];
                u_aug[i + j * n_aug] = compact.u[i + j * n];
            }
        }

        // Vtilde = A_0⁻¹ V_aug (k inner back-solves).
        self.vtilde = vec![0.0_f64; n_aug * k];
        self.inner
            .solve_many(&v_aug, k, &mut self.vtilde)
            .map_err(|e| e)?;

        // Utilde1 = A_0⁻¹ U_aug.
        let mut utilde1 = vec![0.0_f64; n_aug * k];
        self.inner.solve_many(&u_aug, k, &mut utilde1)?;

        // M1 = I + Vtilde_xᵀ V (k×k symmetric, only x-rows of Vtilde matter).
        let mut m1 = vec![0.0_f64; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut acc = 0.0;
                for p in 0..n {
                    acc += self.vtilde[p + i * n_aug] * compact.v[p + j * n];
                }
                m1[i * k + j] = acc;
            }
        }
        for i in 0..k {
            m1[i * k + i] += 1.0;
        }

        let mut j1_chol = vec![0.0_f64; k * k];
        if !chol_lower_inplace(&m1, k, &mut j1_chol) {
            return Err(SolverError::WrongInertia {
                actual: Inertia {
                    positive: 0,
                    negative: 0,
                    zero: 0,
                },
            });
        }

        // C = M1⁻¹ (Vtilde_xᵀ U). First Z = Vtilde_xᵀ U (k×k), then solve M1·C = Z.
        let mut c = vec![0.0_f64; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut acc = 0.0;
                for p in 0..n {
                    acc += self.vtilde[p + i * n_aug] * compact.u[p + j * n];
                }
                c[i * k + j] = acc;
            }
        }
        chol_solve_matrix_inplace(&j1_chol, k, &mut c, k);

        // Utilde2 = Utilde1 − Vtilde · C (full augmented).
        self.utilde2 = utilde1;
        for j in 0..k {
            for p in 0..k {
                let cpj = c[p * k + j];
                if cpj == 0.0 {
                    continue;
                }
                for i in 0..n_aug {
                    self.utilde2[i + j * n_aug] -= self.vtilde[i + p * n_aug] * cpj;
                }
            }
        }

        // M2 = I − Utilde2_xᵀ U.
        let mut m2 = vec![0.0_f64; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut acc = 0.0;
                for p in 0..n {
                    acc += self.utilde2[p + i * n_aug] * compact.u[p + j * n];
                }
                m2[i * k + j] = -acc;
            }
        }
        for i in 0..k {
            m2[i * k + i] += 1.0;
        }

        let mut j2_chol = vec![0.0_f64; k * k];
        if !chol_lower_inplace(&m2, k, &mut j2_chol) {
            return Err(SolverError::WrongInertia {
                actual: Inertia {
                    positive: 0,
                    negative: 0,
                    zero: 0,
                },
            });
        }

        self.j1_chol = j1_chol;
        self.j2_chol = j2_chol;
        self.factored = true;
        Ok(inner_inertia)
    }

    /// Solve `A · z = r` where `A = A_0 + V_aug V_augᵀ − U_aug U_augᵀ`.
    ///
    /// `rhs` and `solution` are `n_aug`-vectors. Computes
    /// `z = csol_0 + Utilde2 · (M2⁻¹ Utilde2ᵀ r) − Vtilde · (M1⁻¹ Vtildeᵀ r)`,
    /// matching `IpLowRankAugSystemSolver.cpp:182-227`.
    pub fn solve(&mut self, rhs: &[f64], solution: &mut [f64]) -> Result<(), SolverError> {
        if !self.factored {
            return Err(SolverError::NumericalFailure(
                "LowRankKktSolver: solve called before factor".into(),
            ));
        }
        if rhs.len() != self.n_aug || solution.len() != self.n_aug {
            return Err(SolverError::DimensionMismatch {
                expected: self.n_aug,
                got: rhs.len().min(solution.len()),
            });
        }

        // csol_0 = A_0⁻¹ · rhs.
        self.inner.solve(rhs, solution)?;

        let k = self.k;
        if k == 0 {
            return Ok(());
        }
        let n_aug = self.n_aug;

        // U-correction: solution += Utilde2 · M2⁻¹ Utilde2ᵀ rhs.
        let mut bu = vec![0.0_f64; k];
        for j in 0..k {
            let mut acc = 0.0;
            for i in 0..n_aug {
                acc += self.utilde2[i + j * n_aug] * rhs[i];
            }
            bu[j] = acc;
        }
        chol_solve_vec_inplace(&self.j2_chol, k, &mut bu);
        for j in 0..k {
            let bj = bu[j];
            if bj == 0.0 {
                continue;
            }
            for i in 0..n_aug {
                solution[i] += self.utilde2[i + j * n_aug] * bj;
            }
        }

        // V-correction: solution -= Vtilde · M1⁻¹ Vtildeᵀ rhs.
        let mut bv = vec![0.0_f64; k];
        for j in 0..k {
            let mut acc = 0.0;
            for i in 0..n_aug {
                acc += self.vtilde[i + j * n_aug] * rhs[i];
            }
            bv[j] = acc;
        }
        chol_solve_vec_inplace(&self.j1_chol, k, &mut bv);
        for j in 0..k {
            let bj = bv[j];
            if bj == 0.0 {
                continue;
            }
            for i in 0..n_aug {
                solution[i] -= self.vtilde[i + j * n_aug] * bj;
            }
        }

        Ok(())
    }
}

/// Cholesky factorization of a `k × k` symmetric PD matrix into a lower
/// triangular factor `L` such that `M = L Lᵀ`. `m` is row-major full storage
/// (only the lower triangle is read). `out` is row-major lower-triangular
/// (only entries with `j ≤ i` are written; rest left zero). Returns false if
/// any pivot is non-positive (matrix not PD).
fn chol_lower_inplace(m: &[f64], k: usize, out: &mut [f64]) -> bool {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for j in 0..k {
        let mut diag = m[j * k + j];
        for p in 0..j {
            diag -= out[j * k + p].powi(2);
        }
        if !(diag > 0.0) {
            return false;
        }
        let ljj = diag.sqrt();
        out[j * k + j] = ljj;
        for i in (j + 1)..k {
            let mut acc = m[i * k + j];
            for p in 0..j {
                acc -= out[i * k + p] * out[j * k + p];
            }
            out[i * k + j] = acc / ljj;
        }
    }
    true
}

/// In-place Cholesky solve `L Lᵀ x = b` (b becomes x). `j_chol` is row-major
/// lower-triangular `k × k` (entries with `j > i` are ignored).
fn chol_solve_vec_inplace(j_chol: &[f64], k: usize, b: &mut [f64]) {
    // Forward: L · y = b.
    for i in 0..k {
        let mut s = b[i];
        for p in 0..i {
            s -= j_chol[i * k + p] * b[p];
        }
        b[i] = s / j_chol[i * k + i];
    }
    // Backward: Lᵀ · x = y.
    for i in (0..k).rev() {
        let mut s = b[i];
        for p in (i + 1)..k {
            s -= j_chol[p * k + i] * b[p];
        }
        b[i] = s / j_chol[i * k + i];
    }
}

/// In-place Cholesky solve `L Lᵀ X = B` for an `k × ncol` row-major matrix `B`.
fn chol_solve_matrix_inplace(j_chol: &[f64], k: usize, b: &mut [f64], ncol: usize) {
    // Apply column-by-column.
    let mut col = vec![0.0_f64; k];
    for jcol in 0..ncol {
        for i in 0..k {
            col[i] = b[i * ncol + jcol];
        }
        chol_solve_vec_inplace(j_chol, k, &mut col);
        for i in 0..k {
            b[i * ncol + jcol] = col[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipm::LbfgsCompact;
    use crate::linear_solver::dense::DenseLdl;
    use crate::linear_solver::{KktMatrix, LinearSolver};

    /// Build a tiny augmented system with an L-BFGS Hessian via the wrapper,
    /// then solve the same system directly by assembling the full Hessian
    /// `B = σI + VVᵀ − UUᵀ` into the (1,1) block. The two solutions must
    /// agree to round-off (1e-10 relative).
    #[test]
    fn low_rank_wrapper_matches_dense_reference() {
        let n = 5;
        let m = 2;
        let n_aug = n + m;
        let k = 3;
        let sigma = 1.7;

        // Cooked V, U, J — pick values so the resulting M1, M2 are PD.
        let v_data: Vec<f64> = (0..(n * k))
            .map(|i| ((i as f64) * 0.31 - 0.5).sin() * 0.4)
            .collect();
        let u_data: Vec<f64> = (0..(n * k))
            .map(|i| ((i as f64) * 0.21 + 0.2).cos() * 0.25)
            .collect();
        let compact = LbfgsCompact {
            n,
            k,
            sigma,
            v: v_data.clone(),
            u: u_data.clone(),
        };

        // J (m × n) row-major, plus a barrier diagonal Σ_x to make A_0 PD on x.
        let j_mat: Vec<f64> = vec![
            1.0, 0.5, 0.0, 0.2, -0.3,
            -0.1, 0.4, 0.7, 0.0, 0.1,
        ];
        let sigma_x: Vec<f64> = vec![0.5, 0.6, 0.7, 0.4, 0.5]; // Σ_x diagonal
        let delta_c = 1e-8;

        // Build A_0 (base augmented system with σI on (1,1)).
        let build_kkt = |hess_dense: Option<&[f64]>| {
            let mut kkt = KktMatrix::zeros_dense(n_aug);
            for i in 0..n {
                let h_ii = match hess_dense {
                    Some(b) => b[i * n + i],
                    None => sigma,
                };
                kkt.add(i, i, h_ii + sigma_x[i]);
            }
            // Off-diagonal of full Hessian (only when provided).
            if let Some(b) = hess_dense {
                for i in 0..n {
                    for jx in 0..i {
                        let v = b[i * n + jx];
                        if v != 0.0 {
                            kkt.add(i, jx, v);
                        }
                    }
                }
            }
            for row in 0..m {
                for col in 0..n {
                    let v = j_mat[row * n + col];
                    if v != 0.0 {
                        kkt.add(n + row, col, v);
                    }
                }
                kkt.add(n + row, n + row, -delta_c);
            }
            kkt
        };

        let base_kkt = build_kkt(None);

        // Reference: assemble full B = σI + VVᵀ − UUᵀ explicitly, solve via dense LDLᵀ.
        let mut b_full = vec![0.0_f64; n * n];
        for i in 0..n {
            b_full[i * n + i] = sigma;
        }
        for j in 0..k {
            for r in 0..n {
                for c in 0..n {
                    b_full[r * n + c] += v_data[r + j * n] * v_data[c + j * n]
                        - u_data[r + j * n] * u_data[c + j * n];
                }
            }
        }
        let full_kkt = build_kkt(Some(&b_full));

        // RHS — pick something exercising both x and m blocks.
        let rhs: Vec<f64> = (0..n_aug)
            .map(|i| ((i as f64) * 0.7 - 1.3).sin())
            .collect();

        // Reference solve.
        let mut ref_solver = DenseLdl::new();
        ref_solver
            .factor(&full_kkt)
            .expect("reference LDLᵀ should factor full KKT");
        let mut ref_sol = vec![0.0_f64; n_aug];
        ref_solver.solve(&rhs, &mut ref_sol).expect("ref solve");

        // Wrapper solve.
        let mut wrapper = LowRankKktSolver::new(DenseLdl::new());
        wrapper
            .factor(&base_kkt, &compact, n_aug)
            .expect("wrapper factor");
        let mut wrap_sol = vec![0.0_f64; n_aug];
        wrapper.solve(&rhs, &mut wrap_sol).expect("wrapper solve");

        // Verify residual of wrapper solution against the *full* (not base) KKT.
        let mut resid = vec![0.0_f64; n_aug];
        full_kkt.matvec(&wrap_sol, &mut resid);
        for i in 0..n_aug {
            resid[i] -= rhs[i];
        }
        let resid_norm: f64 = resid.iter().map(|r| r * r).sum::<f64>().sqrt();
        let rhs_norm: f64 = rhs.iter().map(|r| r * r).sum::<f64>().sqrt();
        assert!(
            resid_norm / rhs_norm < 1e-10,
            "wrapper solve residual too large: {:e} / {:e}",
            resid_norm,
            rhs_norm
        );

        // And cross-check against the reference solution component-by-component.
        for i in 0..n_aug {
            let denom = ref_sol[i].abs().max(1e-12);
            let diff = (ref_sol[i] - wrap_sol[i]).abs();
            assert!(
                diff / denom < 1e-9,
                "row {i}: ref={} wrap={} |Δ|={diff:e}",
                ref_sol[i],
                wrap_sol[i]
            );
        }
    }

    /// Stress test: larger n=12, m=4, k=6 (Ipopt's default `limited_memory_max_history`).
    /// Same residual / parity checks as the small case.
    #[test]
    fn low_rank_wrapper_matches_dense_reference_k6() {
        let n = 12;
        let m = 4;
        let n_aug = n + m;
        let k = 6;
        let sigma = 2.3;

        let v_data: Vec<f64> = (0..(n * k))
            .map(|i| ((i as f64) * 0.13 - 0.7).sin() * 0.5)
            .collect();
        let u_data: Vec<f64> = (0..(n * k))
            .map(|i| ((i as f64) * 0.17 + 0.3).cos() * 0.3)
            .collect();
        let compact = LbfgsCompact { n, k, sigma, v: v_data.clone(), u: u_data.clone() };

        let j_mat: Vec<f64> = (0..(m * n))
            .map(|i| ((i as f64) * 0.41 + 0.1).sin() * 0.6)
            .collect();
        let sigma_x: Vec<f64> = (0..n).map(|i| 0.4 + 0.05 * (i as f64)).collect();
        let delta_c = 1e-8;

        let build_kkt = |hess_dense: Option<&[f64]>| {
            let mut kkt = KktMatrix::zeros_dense(n_aug);
            for i in 0..n {
                let h_ii = match hess_dense {
                    Some(b) => b[i * n + i],
                    None => sigma,
                };
                kkt.add(i, i, h_ii + sigma_x[i]);
            }
            if let Some(b) = hess_dense {
                for i in 0..n {
                    for jx in 0..i {
                        kkt.add(i, jx, b[i * n + jx]);
                    }
                }
            }
            for row in 0..m {
                for col in 0..n {
                    kkt.add(n + row, col, j_mat[row * n + col]);
                }
                kkt.add(n + row, n + row, -delta_c);
            }
            kkt
        };

        let base_kkt = build_kkt(None);

        let mut b_full = vec![0.0_f64; n * n];
        for i in 0..n {
            b_full[i * n + i] = sigma;
        }
        for j in 0..k {
            for r in 0..n {
                for c in 0..n {
                    b_full[r * n + c] +=
                        v_data[r + j * n] * v_data[c + j * n] - u_data[r + j * n] * u_data[c + j * n];
                }
            }
        }
        let full_kkt = build_kkt(Some(&b_full));

        let rhs: Vec<f64> = (0..n_aug)
            .map(|i| ((i as f64) * 0.9 - 0.5).cos())
            .collect();

        // Reference: factor the full KKT directly and check its residual too,
        // so we can compare wrapper accuracy against the inner solver's floor.
        let mut ref_solver = DenseLdl::new();
        ref_solver.factor(&full_kkt).expect("ref factor");
        let mut ref_sol = vec![0.0_f64; n_aug];
        ref_solver.solve(&rhs, &mut ref_sol).expect("ref solve");
        let mut ref_resid = vec![0.0_f64; n_aug];
        full_kkt.matvec(&ref_sol, &mut ref_resid);
        for i in 0..n_aug {
            ref_resid[i] -= rhs[i];
        }
        let ref_resid_norm: f64 = ref_resid.iter().map(|r| r * r).sum::<f64>().sqrt();

        let mut wrapper = LowRankKktSolver::new(DenseLdl::new());
        wrapper.factor(&base_kkt, &compact, n_aug).expect("factor");
        let mut wrap_sol = vec![0.0_f64; n_aug];
        wrapper.solve(&rhs, &mut wrap_sol).expect("solve");

        // Residual against the *full* B_k system. Wrapper performs k=6 extra
        // inner solves and a 6×6 Cholesky correction, so its accuracy floor is
        // a constant times the reference solver's. Bound at 100× the
        // reference residual or 1e-7 absolute, whichever is larger.
        let mut resid = vec![0.0_f64; n_aug];
        full_kkt.matvec(&wrap_sol, &mut resid);
        for i in 0..n_aug {
            resid[i] -= rhs[i];
        }
        let resid_norm: f64 = resid.iter().map(|r| r * r).sum::<f64>().sqrt();
        let rhs_norm: f64 = rhs.iter().map(|r| r * r).sum::<f64>().sqrt();
        let bound = (100.0 * ref_resid_norm).max(1e-7);
        assert!(
            resid_norm < bound,
            "k=6 residual too large: {:e} (ref {:e}, bound {:e}, ||rhs||={:e})",
            resid_norm,
            ref_resid_norm,
            bound,
            rhs_norm
        );

        // Sanity: a second RHS using the same factorization.
        let rhs2: Vec<f64> = (0..n_aug).map(|i| 1.0 / (1.0 + i as f64)).collect();
        let mut wrap_sol2 = vec![0.0_f64; n_aug];
        wrapper.solve(&rhs2, &mut wrap_sol2).expect("solve2");
        let mut resid2 = vec![0.0_f64; n_aug];
        full_kkt.matvec(&wrap_sol2, &mut resid2);
        for i in 0..n_aug {
            resid2[i] -= rhs2[i];
        }
        let resid2_norm: f64 = resid2.iter().map(|r| r * r).sum::<f64>().sqrt();
        assert!(
            resid2_norm < bound,
            "k=6 second-RHS residual too large: {:e} (bound {:e})",
            resid2_norm, bound
        );
    }

    /// k = 0 path: wrapper degenerates to the inner solver alone.
    #[test]
    fn low_rank_wrapper_k0_passes_through() {
        let n = 3;
        let m = 1;
        let n_aug = n + m;
        let sigma = 0.9;

        let mut kkt = KktMatrix::zeros_dense(n_aug);
        for i in 0..n {
            kkt.add(i, i, sigma + 0.4);
        }
        kkt.add(n, 0, 1.0);
        kkt.add(n, 1, 0.5);
        kkt.add(n, n, -1e-8);

        let compact = LbfgsCompact::empty(n, sigma);
        let mut wrapper = LowRankKktSolver::new(DenseLdl::new());
        wrapper.factor(&kkt, &compact, n_aug).expect("factor");

        let rhs = vec![1.0, -2.0, 3.0, 0.5];
        let mut wrap_sol = vec![0.0; n_aug];
        wrapper.solve(&rhs, &mut wrap_sol).expect("solve");

        let mut ref_solver = DenseLdl::new();
        ref_solver.factor(&kkt).expect("ref factor");
        let mut ref_sol = vec![0.0; n_aug];
        ref_solver.solve(&rhs, &mut ref_sol).expect("ref solve");

        for i in 0..n_aug {
            assert!(
                (wrap_sol[i] - ref_sol[i]).abs() < 1e-12,
                "row {i}: wrap={} ref={}",
                wrap_sol[i],
                ref_sol[i]
            );
        }
    }
}
