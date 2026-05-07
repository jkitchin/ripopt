pub mod banded;
pub mod dense;
#[cfg(feature = "rmumps")]
pub mod multifrontal;
#[cfg(feature = "rmumps")]
pub mod iterative;
#[cfg(feature = "rmumps")]
pub mod hybrid;
#[cfg(feature = "feral")]
pub mod feral_direct;
#[cfg(feature = "feral")]
pub mod feral_iterative;
#[cfg(feature = "feral")]
pub mod feral_hybrid;
#[cfg(feature = "faer")]
pub mod sparse;

use std::fmt;

/// Inertia of a symmetric matrix after LDL^T factorization.
/// Counts the signs of diagonal entries in D.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inertia {
    /// Number of positive eigenvalues.
    pub positive: usize,
    /// Number of negative eigenvalues.
    pub negative: usize,
    /// Number of zero eigenvalues.
    pub zero: usize,
}

impl fmt::Display for Inertia {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(+{}, -{}, 0:{})",
            self.positive, self.negative, self.zero
        )
    }
}

/// Error from a linear solver.
///
/// T3.34: `WrongInertia` is distinct from `SingularMatrix` to match Ipopt's
/// `ESymSolverStatus` (SUCCESS / SINGULAR / WRONG_INERTIA). The IPM's
/// perturbation handler escalates `delta_w`/`delta_c` differently for the
/// two cases — singular triggers a fresh factor with bumped delta_c; wrong
/// inertia triggers a delta_w escalation. Collapsing them into one variant
/// loses that signal.
#[derive(Debug, Clone)]
pub enum SolverError {
    /// Matrix is structurally singular (e.g., zero row).
    SingularMatrix,
    /// Factorization succeeded but inertia does not match the expected
    /// (n positive, m negative, 0 zero) signature for an augmented KKT
    /// system. Carries the actual inertia so callers can log/diagnose.
    WrongInertia { actual: Inertia },
    /// Numerical failure during factorization.
    NumericalFailure(String),
    /// Dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Factorization has correct inertia but solve residual ratio is too large.
    /// Signals the caller to increase perturbation and re-factorize (Ipopt's pretend_singular).
    PretendSingular,
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::SingularMatrix => write!(f, "singular matrix"),
            SolverError::WrongInertia { actual } => write!(f, "wrong inertia: {}", actual),
            SolverError::NumericalFailure(msg) => write!(f, "numerical failure: {}", msg),
            SolverError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
            SolverError::PretendSingular => write!(f, "pretend singular (residual ratio too large)"),
        }
    }
}

impl std::error::Error for SolverError {}

/// Symmetric matrix stored as dense lower triangle, column-major.
#[derive(Debug, Clone)]
pub struct SymmetricMatrix {
    /// Dimension of the matrix.
    pub n: usize,
    /// Lower triangle entries stored column-major.
    /// For column j, rows j..n are stored.
    /// Entry (i,j) where i >= j is at index: j*n - j*(j+1)/2 + i
    pub data: Vec<f64>,
}

impl SymmetricMatrix {
    /// Create a new zero symmetric matrix of dimension n.
    pub fn zeros(n: usize) -> Self {
        let nnz = n * (n + 1) / 2;
        Self {
            n,
            data: vec![0.0; nnz],
        }
    }

    /// Get the index into data for entry (i, j) where i >= j.
    #[inline]
    fn packed_index(n: usize, i: usize, j: usize) -> usize {
        debug_assert!(i >= j);
        debug_assert!(i < n);
        j * n - j * (j + 1) / 2 + i
    }

    /// Get element (i, j), automatically handling symmetry.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i >= j {
            self.data[Self::packed_index(self.n, i, j)]
        } else {
            self.data[Self::packed_index(self.n, j, i)]
        }
    }

    /// Set element (i, j), automatically handling symmetry.
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        if i >= j {
            self.data[Self::packed_index(self.n, i, j)] = val;
        } else {
            self.data[Self::packed_index(self.n, j, i)] = val;
        }
    }

    /// Add val to element (i, j), automatically handling symmetry.
    pub fn add(&mut self, i: usize, j: usize, val: f64) {
        if i >= j {
            self.data[Self::packed_index(self.n, i, j)] += val;
        } else {
            self.data[Self::packed_index(self.n, j, i)] += val;
        }
    }

    /// Add delta to all diagonal entries.
    pub fn add_diagonal(&mut self, delta: f64) {
        for i in 0..self.n {
            self.data[Self::packed_index(self.n, i, i)] += delta;
        }
    }

    /// Add delta to diagonal entries in range [start, end).
    pub fn add_diagonal_range(&mut self, start: usize, end: usize, delta: f64) {
        for i in start..end {
            self.data[Self::packed_index(self.n, i, i)] += delta;
        }
    }

    /// Compute y = A * x (symmetric matrix-vector product).
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            y[i] = 0.0;
        }
        for j in 0..n {
            // Diagonal
            let ajj = self.data[Self::packed_index(n, j, j)];
            y[j] += ajj * x[j];
            // Off-diagonal (lower triangle)
            for i in (j + 1)..n {
                let aij = self.data[Self::packed_index(n, i, j)];
                y[i] += aij * x[j];
                y[j] += aij * x[i];
            }
        }
    }

    /// Compute the infinity norm of each row/column (identical for symmetric matrices).
    /// Returns a vector of length n where entry k = max_j |A_{k,j}|.
    pub fn row_abs_max(&self) -> Vec<f64> {
        let n = self.n;
        let mut norms = vec![0.0f64; n];
        for j in 0..n {
            let ajj = self.data[Self::packed_index(n, j, j)].abs();
            norms[j] = norms[j].max(ajj);
            for i in (j + 1)..n {
                let aij = self.data[Self::packed_index(n, i, j)].abs();
                norms[i] = norms[i].max(aij);
                norms[j] = norms[j].max(aij);
            }
        }
        norms
    }

    /// Compute the one-norm (absolute sum) of each row/column (identical for symmetric matrices).
    /// Returns a vector of length n where entry k = sum_j |A_{k,j}|.
    pub fn row_abs_sum(&self) -> Vec<f64> {
        let n = self.n;
        let mut norms = vec![0.0f64; n];
        for j in 0..n {
            let ajj = self.data[Self::packed_index(n, j, j)].abs();
            norms[j] += ajj;
            for i in (j + 1)..n {
                let aij = self.data[Self::packed_index(n, i, j)].abs();
                norms[i] += aij;
                norms[j] += aij;
            }
        }
        norms
    }

    /// Scale row k and column k by alpha (symmetric scaling: A' = D*A*D where D\[k\]=alpha).
    /// Diagonal (k,k) is scaled by alpha^2, off-diagonals by alpha.
    pub fn scale_row_col(&mut self, k: usize, alpha: f64) {
        let n = self.n;
        let alpha2 = alpha * alpha;
        // Diagonal
        self.data[Self::packed_index(n, k, k)] *= alpha2;
        // Column k: entries (i, k) for i > k
        for i in (k + 1)..n {
            self.data[Self::packed_index(n, i, k)] *= alpha;
        }
        // Row k: entries (k, j) for j < k, stored as (k, j) in column j
        for j in 0..k {
            self.data[Self::packed_index(n, k, j)] *= alpha;
        }
    }
}

#[cfg(test)]
impl SymmetricMatrix {
    /// Convert to full dense matrix (row-major) for debugging/testing.
    pub fn to_full(&self) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0; self.n]; self.n];
        for i in 0..self.n {
            for j in 0..=i {
                let v = self.get(i, j);
                m[i][j] = v;
                m[j][i] = v;
            }
        }
        m
    }

    /// Compute eigenvalues using the Jacobi eigenvalue algorithm.
    /// Only suitable for small matrices (n <= ~50).
    /// Returns eigenvalues sorted in ascending order.
    pub fn eigenvalues(&self) -> Vec<f64> {
        let n = self.n;
        if n == 0 {
            return vec![];
        }
        // Work with full dense matrix
        let mut m = self.to_full();

        let max_sweeps = 100;
        for _sweep in 0..max_sweeps {
            // Find largest off-diagonal |m[p][q]|
            let mut max_val = 0.0f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if m[i][j].abs() > max_val {
                        max_val = m[i][j].abs();
                        p = i;
                        q = j;
                    }
                }
            }

            // Convergence check: off-diagonal is small relative to diagonal
            let diag_max = (0..n)
                .map(|i| m[i][i].abs())
                .fold(1e-300, f64::max);
            if max_val < 1e-12 * diag_max {
                break;
            }

            // Apply Jacobi rotation to zero m[p][q]
            let diff = m[q][q] - m[p][p];
            let t = if diff.abs() < 1e-20 * max_val {
                1.0
            } else {
                let phi = diff / (2.0 * m[p][q]);
                phi.signum() / (phi.abs() + (1.0 + phi * phi).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;
            let tau = s / (1.0 + c);

            let apq = m[p][q];
            m[p][q] = 0.0;
            m[q][p] = 0.0;
            m[p][p] -= t * apq;
            m[q][q] += t * apq;

            for r in 0..n {
                if r != p && r != q {
                    let rp = m[r][p];
                    let rq = m[r][q];
                    m[r][p] = rp - s * (rq + tau * rp);
                    m[p][r] = m[r][p];
                    m[r][q] = rq + s * (rp - tau * rq);
                    m[q][r] = m[r][q];
                }
            }
        }

        let mut eigs: Vec<f64> = (0..n).map(|i| m[i][i]).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        eigs
    }
}

/// Sparse symmetric matrix in COO (triplet) format.
///
/// Entries are stored as (row, col, val) triplets in upper triangle (row <= col).
/// Duplicate entries at the same (row, col) are summed during CSC conversion.
#[derive(Debug, Clone)]
pub struct SparseSymmetricMatrix {
    /// Dimension of the matrix.
    pub n: usize,
    /// Row indices (row <= col for upper triangle).
    pub triplet_rows: Vec<usize>,
    /// Column indices (col >= row for upper triangle).
    pub triplet_cols: Vec<usize>,
    /// Values.
    pub triplet_vals: Vec<f64>,
}

impl SparseSymmetricMatrix {
    /// Create a new empty sparse symmetric matrix of dimension n.
    /// Pre-populates all diagonal entries as structural zeros so the
    /// sparsity pattern is stable even when add_diagonal is called later.
    pub fn zeros(n: usize) -> Self {
        let mut m = Self {
            n,
            triplet_rows: Vec::with_capacity(n),
            triplet_cols: Vec::with_capacity(n),
            triplet_vals: Vec::with_capacity(n),
        };
        for i in 0..n {
            m.triplet_rows.push(i);
            m.triplet_cols.push(i);
            m.triplet_vals.push(0.0);
        }
        m
    }

    /// Create with pre-allocated capacity for triplets.
    /// Pre-populates all diagonal entries as structural zeros.
    pub fn with_capacity(n: usize, capacity: usize) -> Self {
        let cap = capacity.max(n);
        let mut m = Self {
            n,
            triplet_rows: Vec::with_capacity(cap),
            triplet_cols: Vec::with_capacity(cap),
            triplet_vals: Vec::with_capacity(cap),
        };
        for i in 0..n {
            m.triplet_rows.push(i);
            m.triplet_cols.push(i);
            m.triplet_vals.push(0.0);
        }
        m
    }

    /// Add val to element (i, j), storing in upper triangle.
    /// Note: zero values are NOT skipped so that the structural sparsity
    /// pattern remains stable across iterations (required for cached
    /// symbolic factorization).
    pub fn add(&mut self, i: usize, j: usize, val: f64) {
        // Store in upper triangle: row <= col
        if i <= j {
            self.triplet_rows.push(i);
            self.triplet_cols.push(j);
        } else {
            self.triplet_rows.push(j);
            self.triplet_cols.push(i);
        }
        self.triplet_vals.push(val);
    }

    /// Add delta to all diagonal entries.
    pub fn add_diagonal(&mut self, delta: f64) {
        for i in 0..self.n {
            self.triplet_rows.push(i);
            self.triplet_cols.push(i);
            self.triplet_vals.push(delta);
        }
    }

    /// Add delta to diagonal entries in range [start, end).
    pub fn add_diagonal_range(&mut self, start: usize, end: usize, delta: f64) {
        for i in start..end {
            self.triplet_rows.push(i);
            self.triplet_cols.push(i);
            self.triplet_vals.push(delta);
        }
    }

    /// Compute y = A * x (symmetric matrix-vector product) using triplets.
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            y[i] = 0.0;
        }
        for k in 0..self.triplet_rows.len() {
            let i = self.triplet_rows[k];
            let j = self.triplet_cols[k];
            let v = self.triplet_vals[k];
            // Upper triangle: i <= j
            y[i] += v * x[j];
            if i != j {
                y[j] += v * x[i];
            }
        }
    }

    /// Compute the infinity norm of each row/column (identical for symmetric matrices).
    pub fn row_abs_max(&self) -> Vec<f64> {
        let mut norms = vec![0.0f64; self.n];
        for k in 0..self.triplet_rows.len() {
            let i = self.triplet_rows[k];
            let j = self.triplet_cols[k];
            let v = self.triplet_vals[k].abs();
            norms[i] = norms[i].max(v);
            if i != j {
                norms[j] = norms[j].max(v);
            }
        }
        norms
    }

    /// Compute the one-norm (absolute sum) of each row/column (identical for symmetric matrices).
    pub fn row_abs_sum(&self) -> Vec<f64> {
        let mut norms = vec![0.0f64; self.n];
        for k in 0..self.triplet_rows.len() {
            let i = self.triplet_rows[k];
            let j = self.triplet_cols[k];
            let v = self.triplet_vals[k].abs();
            norms[i] += v;
            if i != j {
                norms[j] += v;
            }
        }
        norms
    }

    /// Scale row k and column k by alpha (symmetric scaling).
    /// For triplet (i, j, v): if i==k or j==k, multiply v by alpha.
    /// If both i==k and j==k (diagonal), multiply by alpha^2.
    pub fn scale_row_col(&mut self, k: usize, alpha: f64) {
        for idx in 0..self.triplet_rows.len() {
            let i = self.triplet_rows[idx];
            let j = self.triplet_cols[idx];
            if i == k && j == k {
                self.triplet_vals[idx] *= alpha * alpha;
            } else if i == k || j == k {
                self.triplet_vals[idx] *= alpha;
            }
        }
    }

    /// Convert to faer SparseColMat (upper triangle, duplicates summed).
    #[cfg(feature = "faer")]
    pub fn to_upper_csc(&self) -> faer::sparse::SparseColMat<usize, f64> {
        let triplets: Vec<(usize, usize, f64)> = self
            .triplet_rows
            .iter()
            .zip(self.triplet_cols.iter())
            .zip(self.triplet_vals.iter())
            .map(|((&r, &c), &v)| (r, c, v))
            .collect();

        faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(
            self.n,
            self.n,
            &triplets,
        )
        .expect("SparseSymmetricMatrix: invalid triplets")
    }
}

/// KKT matrix that wraps either a dense or sparse symmetric matrix.
///
/// Provides a unified interface for assembly (add, add_diagonal, etc.)
/// and for passing to linear solvers.
#[derive(Debug, Clone)]
pub enum KktMatrix {
    Dense(SymmetricMatrix),
    Sparse(SparseSymmetricMatrix),
}

impl KktMatrix {
    /// Create a dense zero matrix of dimension n.
    pub fn zeros_dense(n: usize) -> Self {
        KktMatrix::Dense(SymmetricMatrix::zeros(n))
    }

    /// Create a sparse zero matrix of dimension n with given triplet capacity.
    pub fn zeros_sparse(n: usize, capacity: usize) -> Self {
        KktMatrix::Sparse(SparseSymmetricMatrix::with_capacity(n, capacity))
    }

    /// Dimension of the matrix.
    pub fn n(&self) -> usize {
        match self {
            KktMatrix::Dense(d) => d.n,
            KktMatrix::Sparse(s) => s.n,
        }
    }

    /// Add val to element (i, j).
    pub fn add(&mut self, i: usize, j: usize, val: f64) {
        match self {
            KktMatrix::Dense(d) => d.add(i, j, val),
            KktMatrix::Sparse(s) => s.add(i, j, val),
        }
    }

    /// Get element (i, j). Note: O(nnz) for sparse matrices.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        match self {
            KktMatrix::Dense(d) => d.get(i, j),
            KktMatrix::Sparse(s) => {
                // Linear scan — only used for tests/debugging
                let (ri, ci) = if i <= j { (i, j) } else { (j, i) };
                let mut val = 0.0;
                for k in 0..s.triplet_rows.len() {
                    if s.triplet_rows[k] == ri && s.triplet_cols[k] == ci {
                        val += s.triplet_vals[k];
                    }
                }
                val
            }
        }
    }

    /// Add delta to all diagonal entries.
    pub fn add_diagonal(&mut self, delta: f64) {
        match self {
            KktMatrix::Dense(d) => d.add_diagonal(delta),
            KktMatrix::Sparse(s) => s.add_diagonal(delta),
        }
    }

    /// Add delta to diagonal entries in range [start, end).
    pub fn add_diagonal_range(&mut self, start: usize, end: usize, delta: f64) {
        match self {
            KktMatrix::Dense(d) => d.add_diagonal_range(start, end, delta),
            KktMatrix::Sparse(s) => s.add_diagonal_range(start, end, delta),
        }
    }

    /// Compute y = A * x.
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        match self {
            KktMatrix::Dense(d) => d.matvec(x, y),
            KktMatrix::Sparse(s) => s.matvec(x, y),
        }
    }

    /// Scale row k and column k by alpha (symmetric scaling).
    /// Compute the infinity norm of each row/column.
    pub fn row_abs_max(&self) -> Vec<f64> {
        match self {
            KktMatrix::Dense(d) => d.row_abs_max(),
            KktMatrix::Sparse(s) => s.row_abs_max(),
        }
    }

    /// Compute the one-norm (absolute sum) of each row/column.
    pub fn row_abs_sum(&self) -> Vec<f64> {
        match self {
            KktMatrix::Dense(d) => d.row_abs_sum(),
            KktMatrix::Sparse(s) => s.row_abs_sum(),
        }
    }

    pub fn scale_row_col(&mut self, k: usize, alpha: f64) {
        match self {
            KktMatrix::Dense(d) => d.scale_row_col(k, alpha),
            KktMatrix::Sparse(s) => s.scale_row_col(k, alpha),
        }
    }
}

/// Diagnostic snapshot from the most recent factorization. T3.38: lets
/// the IPM log/inspect implementation-defined factorization quality
/// signals (delayed pivots, 2x2 blocks, fill, scaling info, resolved
/// algorithm), without baking those into the trait surface.
///
/// All fields are `Option`-typed so backends that cannot report a
/// given quantity simply return `None` for it.
#[derive(Debug, Clone, Default)]
pub struct FactorDiagnostics {
    /// Number of delayed pivots (Bunch-Kaufman pivots that could not
    /// be eliminated in their natural supernode and were pushed to a
    /// parent / column-swap path).
    pub n_delayed: Option<usize>,
    /// Number of 2x2 pivot blocks selected during BK factorization.
    pub n_2x2: Option<usize>,
    /// Non-zeros in the factor L (post-fill).
    pub factor_nnz: Option<usize>,
    /// Smallest diagonal entry (in absolute value) of D.
    pub min_diagonal: Option<f64>,
    /// One-line, human-readable description of the scaling that was
    /// applied (e.g. `"Identity"`, `"InfNorm"`, `"MC64"`).
    pub scaling_info: Option<String>,
    /// Backend-specific name of the algorithm that was actually
    /// resolved (e.g. `"multifrontal"`, `"supernodal-ldlt"`).
    pub resolved_method: Option<String>,
}

/// Trait for linear solvers used within the IPM.
pub trait LinearSolver {
    /// Factor the symmetric matrix. Returns inertia if the solver can compute it.
    fn factor(&mut self, matrix: &KktMatrix) -> Result<Option<Inertia>, SolverError>;

    /// Solve the system using the most recent factorization.
    /// Reads rhs and writes the solution into `solution`.
    fn solve(&mut self, rhs: &[f64], solution: &mut [f64]) -> Result<(), SolverError>;

    /// Batched multi-RHS backsolve with the most recent factorization.
    ///
    /// `rhs` and `solution` are column-major `n × nrhs` buffers of length
    /// `n * nrhs`. The default impl loops single-RHS solves; backends like
    /// feral override to share workspace and supernode traversal across
    /// columns (feral F1.1 `solve_sparse_many`). Use when several RHSes
    /// against the same factor are known up front (QF oracle, Gondzio).
    fn solve_many(
        &mut self,
        rhs: &[f64],
        nrhs: usize,
        solution: &mut [f64],
    ) -> Result<(), SolverError> {
        if nrhs == 0 {
            return Ok(());
        }
        let n = rhs.len() / nrhs;
        if rhs.len() != n * nrhs || solution.len() != n * nrhs {
            return Err(SolverError::DimensionMismatch {
                expected: n * nrhs,
                got: rhs.len().min(solution.len()),
            });
        }
        for c in 0..nrhs {
            let off = c * n;
            self.solve(&rhs[off..off + n], &mut solution[off..off + n])?;
        }
        Ok(())
    }

    /// Whether this solver can report inertia.
    fn provides_inertia(&self) -> bool;

    /// Return the minimum diagonal entry of D after LDL^T factorization.
    /// Used for direct inertia correction on unconstrained problems.
    fn min_diagonal(&self) -> Option<f64> {
        None
    }

    /// Increase factorization quality (e.g., raise pivot threshold).
    /// Called when inertia correction fails. Returns true if quality was improved.
    /// Matches Ipopt's IncreaseQuality() / MUMPS pivtol escalation.
    fn increase_quality(&mut self) -> bool {
        false
    }

    /// Diagnostic snapshot from the most recent factorization. Default
    /// returns an empty struct; backends fill in the fields they can
    /// surface.
    fn last_factor_diagnostics(&self) -> FactorDiagnostics {
        FactorDiagnostics::default()
    }

    /// Estimate kappa_1(A) = ||A||_1 * ||A^{-1}||_1 using the Hager-Higham
    /// power iteration on the cached factorization (feral F2.1). Returns
    /// `None` if the backend does not implement the estimator. Cost is
    /// 3-5 backsolves plus an O(nnz) pass — call on demand from logging
    /// or diagnostics paths, not every iteration.
    fn estimate_condition_1norm(&mut self) -> Option<f64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_creates_correct_size() {
        // n=0
        let m = SymmetricMatrix::zeros(0);
        assert_eq!(m.n, 0);
        assert_eq!(m.data.len(), 0);
        // n=1
        let m = SymmetricMatrix::zeros(1);
        assert_eq!(m.n, 1);
        assert_eq!(m.data.len(), 1);
        // n=4
        let m = SymmetricMatrix::zeros(4);
        assert_eq!(m.n, 4);
        assert_eq!(m.data.len(), 10); // 4*5/2
    }

    #[test]
    fn test_packed_index_j_zero() {
        // Regression: old formula j*(j-1)/2 underflowed for j=0 with usize
        // New formula: j*n - j*(j+1)/2 + i
        // For n=3, (0,0) should be index 0
        assert_eq!(SymmetricMatrix::packed_index(3, 0, 0), 0);
        // (1,0) should be 1
        assert_eq!(SymmetricMatrix::packed_index(3, 1, 0), 1);
        // (2,0) should be 2
        assert_eq!(SymmetricMatrix::packed_index(3, 2, 0), 2);
        // (1,1) should be 3
        assert_eq!(SymmetricMatrix::packed_index(3, 1, 1), 3);
        // (2,1) should be 4
        assert_eq!(SymmetricMatrix::packed_index(3, 2, 1), 4);
        // (2,2) should be 5
        assert_eq!(SymmetricMatrix::packed_index(3, 2, 2), 5);
    }

    #[test]
    fn test_get_set_symmetry() {
        let mut m = SymmetricMatrix::zeros(3);
        m.set(2, 1, 7.5);
        assert!((m.get(2, 1) - 7.5).abs() < 1e-15);
        assert!((m.get(1, 2) - 7.5).abs() < 1e-15);
    }

    #[test]
    fn test_add_accumulates() {
        let mut m = SymmetricMatrix::zeros(2);
        m.set(0, 0, 3.0);
        m.add(0, 0, 2.0);
        assert!((m.get(0, 0) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_add_diagonal() {
        let mut m = SymmetricMatrix::zeros(3);
        m.add_diagonal(5.0);
        for i in 0..3 {
            assert!((m.get(i, i) - 5.0).abs() < 1e-15);
        }
        // Off-diagonal should remain 0
        assert!((m.get(1, 0)).abs() < 1e-15);
    }

    #[test]
    fn test_add_diagonal_range() {
        let mut m = SymmetricMatrix::zeros(4);
        m.add_diagonal_range(1, 3, 2.0);
        assert!((m.get(0, 0)).abs() < 1e-15);
        assert!((m.get(1, 1) - 2.0).abs() < 1e-15);
        assert!((m.get(2, 2) - 2.0).abs() < 1e-15);
        assert!((m.get(3, 3)).abs() < 1e-15);
    }

    #[test]
    fn test_matvec_identity() {
        let mut m = SymmetricMatrix::zeros(3);
        for i in 0..3 {
            m.set(i, i, 1.0);
        }
        let x = [2.0, 3.0, 4.0];
        let mut y = [0.0; 3];
        m.matvec(&x, &mut y);
        for i in 0..3 {
            assert!((y[i] - x[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_matvec_symmetric() {
        // A = [[2, 1, 0], [1, 3, 1], [0, 1, 4]]
        let mut m = SymmetricMatrix::zeros(3);
        m.set(0, 0, 2.0);
        m.set(1, 0, 1.0);
        m.set(1, 1, 3.0);
        m.set(2, 1, 1.0);
        m.set(2, 2, 4.0);
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0; 3];
        m.matvec(&x, &mut y);
        // y = [2*1+1*2+0*3, 1*1+3*2+1*3, 0*1+1*2+4*3] = [4, 10, 14]
        assert!((y[0] - 4.0).abs() < 1e-15);
        assert!((y[1] - 10.0).abs() < 1e-15);
        assert!((y[2] - 14.0).abs() < 1e-15);
    }

    #[test]
    fn test_matvec_zero_vector() {
        let mut m = SymmetricMatrix::zeros(3);
        m.set(0, 0, 5.0);
        m.set(1, 1, 3.0);
        let x = [0.0, 0.0, 0.0];
        let mut y = [0.0; 3];
        m.matvec(&x, &mut y);
        for i in 0..3 {
            assert!((y[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_to_full_round_trip() {
        let mut m = SymmetricMatrix::zeros(3);
        m.set(0, 0, 1.0);
        m.set(1, 0, 2.0);
        m.set(1, 1, 3.0);
        m.set(2, 0, 4.0);
        m.set(2, 1, 5.0);
        m.set(2, 2, 6.0);
        let full = m.to_full();
        assert!((full[0][0] - 1.0).abs() < 1e-15);
        assert!((full[0][1] - 2.0).abs() < 1e-15);
        assert!((full[1][0] - 2.0).abs() < 1e-15);
        assert!((full[1][1] - 3.0).abs() < 1e-15);
        assert!((full[0][2] - 4.0).abs() < 1e-15);
        assert!((full[2][0] - 4.0).abs() < 1e-15);
        assert!((full[2][2] - 6.0).abs() < 1e-15);
    }

    #[test]
    fn test_eigenvalues_identity() {
        let mut m = SymmetricMatrix::zeros(3);
        for i in 0..3 {
            m.set(i, i, 1.0);
        }
        let eigs = m.eigenvalues();
        assert_eq!(eigs.len(), 3);
        for e in &eigs {
            assert!((e - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_eigenvalues_known_spectrum() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3
        let mut m = SymmetricMatrix::zeros(2);
        m.set(0, 0, 2.0);
        m.set(1, 0, 1.0);
        m.set(1, 1, 2.0);
        let eigs = m.eigenvalues();
        assert_eq!(eigs.len(), 2);
        assert!((eigs[0] - 1.0).abs() < 1e-10);
        assert!((eigs[1] - 3.0).abs() < 1e-10);
    }

    // --- KktMatrix tests ---

    #[test]
    fn test_kkt_matrix_dense_add_get() {
        let mut m = KktMatrix::zeros_dense(3);
        m.add(0, 0, 2.0);
        m.add(1, 0, 1.5);
        m.add(2, 2, 3.0);
        assert!((m.get(0, 0) - 2.0).abs() < 1e-15);
        assert!((m.get(1, 0) - 1.5).abs() < 1e-15);
        assert!((m.get(0, 1) - 1.5).abs() < 1e-15);
        assert!((m.get(2, 2) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_kkt_matrix_sparse_add_get() {
        let mut m = KktMatrix::zeros_sparse(3, 10);
        m.add(0, 0, 2.0);
        m.add(1, 0, 1.5);
        m.add(2, 2, 3.0);
        assert!((m.get(0, 0) - 2.0).abs() < 1e-15);
        assert!((m.get(1, 0) - 1.5).abs() < 1e-15);
        assert!((m.get(0, 1) - 1.5).abs() < 1e-15);
        assert!((m.get(2, 2) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_kkt_matrix_sparse_matvec() {
        // A = [[2, 1, 0], [1, 3, 1], [0, 1, 4]]
        let mut m = KktMatrix::zeros_sparse(3, 10);
        m.add(0, 0, 2.0);
        m.add(1, 0, 1.0);
        m.add(1, 1, 3.0);
        m.add(2, 1, 1.0);
        m.add(2, 2, 4.0);
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0; 3];
        m.matvec(&x, &mut y);
        assert!((y[0] - 4.0).abs() < 1e-15);
        assert!((y[1] - 10.0).abs() < 1e-15);
        assert!((y[2] - 14.0).abs() < 1e-15);
    }

    #[cfg(feature = "faer")]
    #[test]
    fn test_sparse_to_upper_csc() {
        let mut s = SparseSymmetricMatrix::zeros(3);
        s.add(0, 0, 2.0);
        s.add(1, 0, 1.0); // stored as (0, 1)
        s.add(1, 1, 3.0);
        s.add(2, 2, 4.0);
        let csc = s.to_upper_csc();
        assert_eq!(csc.nrows(), 3);
        assert_eq!(csc.ncols(), 3);
    }
}
