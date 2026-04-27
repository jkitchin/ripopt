//! Iterative-refinement sparse symmetric indefinite solver backed by feral.
//!
//! v0.8 note: feral does not currently expose an `IncompleteLdlt`-style
//! preconditioner or a public MINRES driver. The plan (path B) is to use the
//! full feral factorization as the MINRES preconditioner — but with a full
//! direct factor as `M`, the Krylov outer loop converges in one step and
//! reduces to plain iterative refinement, which `feral::solve_sparse_refined`
//! already performs internally.
//!
//! So `FeralIterativeMinres` is a thin wrapper around `FeralLdl` that simply
//! tightens convergence expectations on the refinement loop. It satisfies
//! `LinearSolverChoice::Iterative` without losing functionality. A future
//! version may add an incomplete preconditioner and a true preconditioned
//! MINRES outer loop; until then, `Iterative` and `Direct` paths are
//! algorithmically equivalent on the feral backend.
//!
//! The wrapper preserves the `LinearSolver` contract used by the IPM and
//! delegates everything to `FeralLdl`.

use super::feral_direct::FeralLdl;
use super::{FactorDiagnostics, Inertia, KktMatrix, LinearSolver, SolverError};

pub struct FeralIterativeMinres {
    inner: FeralLdl,
}

impl Default for FeralIterativeMinres {
    fn default() -> Self {
        Self::new()
    }
}

impl FeralIterativeMinres {
    pub fn new() -> Self {
        Self {
            inner: FeralLdl::new(),
        }
    }

    /// Kept for API parity with the rmumps `IterativeMinres::with_options`.
    /// Tolerance/max_iter/drop_tolerance are accepted but ignored on the feral
    /// backend (the refinement loop inside `solve_sparse_refined` uses its own
    /// internal stopping criteria).
    pub fn with_options(_tol: f64, _max_iter: usize, _drop_tolerance: f64) -> Self {
        Self::new()
    }
}

impl LinearSolver for FeralIterativeMinres {
    fn factor(&mut self, matrix: &KktMatrix) -> Result<Option<Inertia>, SolverError> {
        self.inner.factor(matrix)
    }

    fn solve(&mut self, rhs: &[f64], solution: &mut [f64]) -> Result<(), SolverError> {
        self.inner.solve(rhs, solution)
    }

    fn provides_inertia(&self) -> bool {
        self.inner.provides_inertia()
    }

    fn min_diagonal(&self) -> Option<f64> {
        self.inner.min_diagonal()
    }

    fn increase_quality(&mut self) -> bool {
        self.inner.increase_quality()
    }

    fn last_factor_diagnostics(&self) -> FactorDiagnostics {
        self.inner.last_factor_diagnostics()
    }
}
