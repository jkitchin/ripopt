//! Sparse symmetric indefinite multifrontal LDL^T solver backed by feral.
//!
//! This is the v0.8 default sparse linear solver, replacing the rmumps-backed
//! `MultifrontalLdl`. Feral is a pure-Rust multifrontal solver with
//! Bunch-Kaufman 1×1/2×2 pivoting, certified inertia counts, MC64 scaling,
//! AMD/METIS ordering, and best-iterate iterative refinement.
//!
//! The wrapper:
//! - converts the upper-triangle COO triplet store from `KktMatrix::Sparse`
//!   to feral's lower-triangle CSC by swapping `(row, col) -> (col, row)`,
//! - caches the symbolic factorization, the CSC structure, and a
//!   COO→CSC index mapping across calls with the same sparsity pattern,
//! - re-uses a `FactorWorkspace` so per-supernode buffers are not
//!   reallocated each IPM iteration,
//! - escalates the Bunch-Kaufman column-relative pivot threshold on
//!   `increase_quality()` using the same MA27-style rule as feral's
//!   `Solver::increase_quality`.

use super::{Inertia, KktMatrix, LinearSolver, SolverError};
use feral::numeric::factorize::{
    factorize_multifrontal_with_workspace, FactorWorkspace, NumericParams, SparseFactors,
};
use feral::numeric::solve::solve_sparse;
use feral::scaling::ScalingStrategy;
use feral::symbolic::supernode::SupernodeParams;
use feral::symbolic::{symbolic_factorize, SymbolicFactorization};
use feral::{CscMatrix, FeralError, ZeroPivotAction};

/// Two-stage `increase_quality` state, mirroring feral's
/// `Solver::QualityLevel`. T3.37 lifts the same state machine into
/// the `FeralLdl` wrapper so we get Ipopt's full
/// `IpTSymLinearSolver::IncreaseQuality` cascade: scaling first,
/// pivot threshold second.
#[derive(Debug, Clone, Copy, PartialEq)]
enum QualityLevel {
    Baseline,
    ScalingEnabled,
    PivotRaised,
    Exhausted,
}

/// Multifrontal sparse symmetric indefinite solver backed by feral.
pub struct FeralLdl {
    n: usize,
    factored: bool,

    // Symbolic + numeric state, cached across calls with the same pattern.
    symbolic: Option<SymbolicFactorization>,
    factors: Option<SparseFactors>,
    workspace: FactorWorkspace,
    snode_params: SupernodeParams,
    numeric_params: NumericParams,

    // CSC structure cached after the first factor; values are scattered
    // in-place from the COO triplets on subsequent calls.
    csc: Option<CscMatrix>,
    /// Mapping from COO triplet index → CSC value index (after the
    /// upper→lower swap and column sort). Same pattern as the rmumps
    /// wrapper — see `multifrontal.rs::build_coo_to_csc_mapping`.
    coo_to_csc: Vec<usize>,

    /// `pivtol_max` for `increase_quality`. Mirrors the value used by
    /// `feral::Solver` (MA27-style 0.5).
    pivtol_max: f64,

    /// Two-stage escalation state (T3.37).
    quality_level: QualityLevel,
}

impl Default for FeralLdl {
    fn default() -> Self {
        Self::new()
    }
}

impl FeralLdl {
    pub fn new() -> Self {
        // ZeroPivotAction::ForceAccept keeps the factor available even when
        // a near-zero pivot is hit, mirroring rmumps' "factor with possibly
        // wrong inertia" semantic. ripopt's inertia-correction loop in
        // `kkt::factor_with_inertia_correction` then perturbs and retries.
        //
        // T3.37: default scaling is Identity, not Auto. ripopt owns the KKT
        // scaling decision via `kkt::ruiz_equilibrate` (activated on demand
        // through `params.use_scaling`); double-scaling at the linear-solver
        // layer changes the factored matrix shape and corrupts the inertia
        // signal that the IPM uses to drive its perturbation handler.
        // `increase_quality()` flips to `InfNorm` as stage-1 escalation,
        // matching feral's Solver::increase_quality and Ipopt's
        // IpTSymLinearSolver::IncreaseQuality cascade.
        let mut numeric_params = NumericParams::default();
        numeric_params.bk.on_zero_pivot = ZeroPivotAction::ForceAccept;
        numeric_params.scaling = ScalingStrategy::Identity;

        Self {
            n: 0,
            factored: false,
            symbolic: None,
            factors: None,
            workspace: FactorWorkspace::new(),
            snode_params: SupernodeParams::default(),
            numeric_params,
            csc: None,
            coo_to_csc: Vec::new(),
            pivtol_max: 0.5,
            quality_level: QualityLevel::Baseline,
        }
    }

    /// Convenience constructor for KKT systems. Currently identical to
    /// `new()`; kept for parity with `MultifrontalLdl::new_kkt` so the
    /// IPM factory can call either uniformly.
    pub fn new_kkt(_n_primal: usize) -> Self {
        Self::new()
    }

    fn convert_inertia(i: feral::Inertia) -> Inertia {
        Inertia {
            positive: i.positive,
            negative: i.negative,
            zero: i.zero,
        }
    }

    fn map_error(e: FeralError) -> SolverError {
        match e {
            FeralError::NumericallyRankDeficient => SolverError::SingularMatrix,
            FeralError::DimensionMismatch { expected, got } => {
                SolverError::DimensionMismatch { expected, got }
            }
            other => SolverError::NumericalFailure(format!("feral: {}", other)),
        }
    }

    /// Build COO→CSC mapping for the given lower-triangle triplets and CSC
    /// structure. Each triplet's CSC slot is found by binary-searching its
    /// row in the relevant column slice. Multiple triplets may map to the
    /// same slot; that is handled by the scatter pass which sums duplicates.
    fn build_coo_to_csc_mapping(
        lower_rows: &[usize],
        lower_cols: &[usize],
        csc: &CscMatrix,
    ) -> Vec<usize> {
        let mut mapping = Vec::with_capacity(lower_rows.len());
        for k in 0..lower_rows.len() {
            let row = lower_rows[k];
            let col = lower_cols[k];
            let col_start = csc.col_ptr[col];
            let col_end = csc.col_ptr[col + 1];
            let slice = &csc.row_idx[col_start..col_end];
            let pos = slice.binary_search(&row).unwrap_or_else(|_| {
                panic!("COO entry ({}, {}) not found in CSC structure", row, col)
            });
            mapping.push(col_start + pos);
        }
        mapping
    }

    /// Zero csc_values, then accumulate triplet values via the cached mapping.
    fn scatter(coo_to_csc: &[usize], triplet_vals: &[f64], csc_values: &mut [f64]) {
        for v in csc_values.iter_mut() {
            *v = 0.0;
        }
        for (k, &val) in triplet_vals.iter().enumerate() {
            csc_values[coo_to_csc[k]] += val;
        }
    }

    /// Convert ripopt's upper-triangle (i ≤ j) triplet to feral's
    /// lower-triangle (row ≥ col) by swap. Returns `(row, col) = (j, i)`
    /// when `i <= j` and `(i, j)` otherwise (idempotent for diagonals).
    #[inline]
    fn to_lower(i: usize, j: usize) -> (usize, usize) {
        if i <= j {
            (j, i)
        } else {
            (i, j)
        }
    }
}

impl LinearSolver for FeralLdl {
    fn factor(&mut self, matrix: &KktMatrix) -> Result<Option<Inertia>, SolverError> {
        let sparse = match matrix {
            KktMatrix::Sparse(s) => s,
            KktMatrix::Dense(_) => {
                return Err(SolverError::NumericalFailure(
                    "FeralLdl requires KktMatrix::Sparse".into(),
                ))
            }
        };

        self.n = sparse.n;
        if self.n == 0 {
            self.factored = true;
            return Ok(Some(Inertia {
                positive: 0,
                negative: 0,
                zero: 0,
            }));
        }

        let n_triplets = sparse.triplet_rows.len();

        // Build lower-triangle row/col arrays once (values are reused as-is
        // because the matrix is symmetric and only indices need swapping).
        let mut lower_rows = Vec::with_capacity(n_triplets);
        let mut lower_cols = Vec::with_capacity(n_triplets);
        for k in 0..n_triplets {
            let (r, c) = Self::to_lower(sparse.triplet_rows[k], sparse.triplet_cols[k]);
            lower_rows.push(r);
            lower_cols.push(c);
        }

        let first_call = self.csc.is_none();

        if first_call {
            let csc = CscMatrix::from_triplets(
                sparse.n,
                &lower_rows,
                &lower_cols,
                &sparse.triplet_vals,
            )
            .map_err(Self::map_error)?;
            self.coo_to_csc =
                Self::build_coo_to_csc_mapping(&lower_rows, &lower_cols, &csc);
            // Fresh symbolic for the new pattern.
            let symbolic =
                symbolic_factorize(&csc, &self.snode_params).map_err(Self::map_error)?;
            self.csc = Some(csc);
            self.symbolic = Some(symbolic);
        } else {
            // Possibly extend the COO→CSC mapping if more triplets were
            // appended (e.g. an `add_diagonal` between factor calls).
            if n_triplets > self.coo_to_csc.len() {
                let csc_ref = match self.csc.as_ref() {
                    Some(c) => c,
                    None => {
                        return Err(SolverError::NumericalFailure(
                            "FeralLdl: csc cache lost".into(),
                        ))
                    }
                };
                let extra = Self::build_coo_to_csc_mapping(
                    &lower_rows[self.coo_to_csc.len()..],
                    &lower_cols[self.coo_to_csc.len()..],
                    csc_ref,
                );
                self.coo_to_csc.extend(extra);
            }
            let csc = match self.csc.as_mut() {
                Some(c) => c,
                None => {
                    return Err(SolverError::NumericalFailure(
                        "FeralLdl: csc cache lost".into(),
                    ))
                }
            };
            Self::scatter(&self.coo_to_csc, &sparse.triplet_vals, &mut csc.values);
        }

        let csc = match self.csc.as_ref() {
            Some(c) => c,
            None => {
                return Err(SolverError::NumericalFailure(
                    "FeralLdl: csc cache lost".into(),
                ))
            }
        };
        let symbolic = match self.symbolic.as_ref() {
            Some(s) => s,
            None => {
                return Err(SolverError::NumericalFailure(
                    "FeralLdl: symbolic cache lost".into(),
                ))
            }
        };

        match factorize_multifrontal_with_workspace(
            csc,
            symbolic,
            &self.numeric_params,
            &mut self.workspace,
        ) {
            Ok((factors, inertia)) => {
                self.factors = Some(factors);
                self.factored = true;
                Ok(Some(Self::convert_inertia(inertia)))
            }
            Err(e) => {
                self.factors = None;
                self.factored = false;
                Err(Self::map_error(e))
            }
        }
    }

    fn solve(&mut self, rhs: &[f64], solution: &mut [f64]) -> Result<(), SolverError> {
        if !self.factored {
            return Err(SolverError::NumericalFailure(
                "matrix not factored".to_string(),
            ));
        }
        if rhs.len() != self.n || solution.len() != self.n {
            return Err(SolverError::DimensionMismatch {
                expected: self.n,
                got: rhs.len(),
            });
        }
        if self.n == 0 {
            return Ok(());
        }

        let factors = match self.factors.as_ref() {
            Some(f) => f,
            None => {
                return Err(SolverError::NumericalFailure(
                    "FeralLdl: factor cache lost".into(),
                ))
            }
        };
        // T3.24: feral's `solve_sparse_refined` ran an internal IR loop
        // (up to 10 steps) on top of the KKT-layer IR in `kkt.rs`. Ipopt's
        // contract is that the linear solver does a single back-solve and
        // the IPM owns refinement, so we use plain `solve_sparse` and let
        // `solve_for_direction_with_ir` drive any refinement.
        let out = solve_sparse(factors, rhs).map_err(Self::map_error)?;
        if out.len() != solution.len() {
            return Err(SolverError::DimensionMismatch {
                expected: solution.len(),
                got: out.len(),
            });
        }
        solution.copy_from_slice(&out);
        Ok(())
    }

    fn provides_inertia(&self) -> bool {
        true
    }

    fn min_diagonal(&self) -> Option<f64> {
        self.factors.as_ref().and_then(|f| f.min_diagonal())
    }

    fn increase_quality(&mut self) -> bool {
        // T3.37: two-stage escalation matching
        // `IpTSymLinearSolver::IncreaseQuality` and feral's
        // `Solver::increase_quality`.
        //   Stage 1: flip Identity → InfNorm scaling (skipped if already non-Identity).
        //   Stage 2: bump the BK column-relative pivot threshold.
        const FIRST_PIVOT_THRESHOLD: f64 = 0.01;
        const PIVOT_EXPONENT: f64 = 0.75;
        const EPS_CAP: f64 = 1e-12;

        match self.quality_level {
            QualityLevel::Exhausted => false,
            QualityLevel::Baseline => {
                if matches!(self.numeric_params.scaling, ScalingStrategy::Identity) {
                    log::debug!("FeralLdl: escalating scaling Identity -> InfNorm");
                    self.numeric_params.scaling = ScalingStrategy::InfNorm;
                    self.quality_level = QualityLevel::ScalingEnabled;
                } else {
                    self.bump_pivot_threshold(FIRST_PIVOT_THRESHOLD, PIVOT_EXPONENT, EPS_CAP);
                }
                true
            }
            QualityLevel::ScalingEnabled | QualityLevel::PivotRaised => {
                self.bump_pivot_threshold(FIRST_PIVOT_THRESHOLD, PIVOT_EXPONENT, EPS_CAP);
                true
            }
        }
    }
}

impl FeralLdl {
    /// Apply stage-2 pivot escalation and update `quality_level`.
    fn bump_pivot_threshold(&mut self, first_jump: f64, exponent: f64, eps_cap: f64) {
        let pivtol = &mut self.numeric_params.bk.pivot_threshold;
        let prev = *pivtol;
        *pivtol = if *pivtol == 0.0 {
            first_jump
        } else {
            pivtol.powf(exponent).min(self.pivtol_max)
        };
        log::debug!(
            "FeralLdl: escalating pivot threshold {:.2e} -> {:.2e}",
            prev,
            *pivtol
        );
        self.quality_level = if *pivtol >= self.pivtol_max - eps_cap {
            QualityLevel::Exhausted
        } else {
            QualityLevel::PivotRaised
        };
    }
}
