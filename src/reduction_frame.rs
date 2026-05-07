//! Internal preprocessing reduction frames.
//!
//! Reduction frames are intentionally crate-private. They carry the mapping
//! needed to transfer a reduced solve result back to the problem that produced
//! the reduced view, without advertising a public transform-stack API.

use crate::logging::rip_log;
use crate::options::SolverOptions;
use crate::problem::NlpProblem;
use crate::result::SolveResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RemovedMultiplierRecovery {
    None,
    AuxiliaryStationarity,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ReductionFrame {
    n_orig: usize,
    m_orig: usize,
    /// Reduced variable index -> original variable index.
    var_map: Vec<usize>,
    /// Reduced row index -> original row index.
    row_map: Vec<usize>,
    /// Original-space values for variables not present in the reduced problem.
    fixed_values: Vec<f64>,
    multiplier_recovery: RemovedMultiplierRecovery,
}

impl ReductionFrame {
    pub(crate) fn new(
        n_orig: usize,
        m_orig: usize,
        var_map: Vec<usize>,
        row_map: Vec<usize>,
        fixed_values: Vec<f64>,
        multiplier_recovery: RemovedMultiplierRecovery,
    ) -> Self {
        debug_assert_eq!(fixed_values.len(), n_orig);
        Self {
            n_orig,
            m_orig,
            var_map,
            row_map,
            fixed_values,
            multiplier_recovery,
        }
    }

    pub(crate) fn did_reduce(&self) -> bool {
        self.var_map.len() < self.n_orig || self.row_map.len() < self.m_orig
    }

    pub(crate) fn num_removed_vars(&self) -> usize {
        self.n_orig - self.var_map.len()
    }

    pub(crate) fn num_removed_rows(&self) -> usize {
        self.m_orig - self.row_map.len()
    }

    pub(crate) fn removed_vars(&self) -> Vec<usize> {
        removed_indices(self.n_orig, &self.var_map)
    }

    pub(crate) fn removed_rows(&self) -> Vec<usize> {
        removed_indices(self.m_orig, &self.row_map)
    }

    pub(crate) fn reduced_x_scaling(&self, scaling: &[f64]) -> Option<Vec<f64>> {
        if scaling.len() != self.n_orig {
            return None;
        }
        Some(self.var_map.iter().map(|&orig| scaling[orig]).collect())
    }

    pub(crate) fn reduced_g_scaling(&self, scaling: &[f64]) -> Option<Vec<f64>> {
        if scaling.len() != self.m_orig {
            return None;
        }
        Some(self.row_map.iter().map(|&orig| scaling[orig]).collect())
    }

    pub(crate) fn expand_x(&self, x_reduced: &[f64]) -> Vec<f64> {
        let mut x_full = self.fixed_values.clone();
        for (reduced, &orig) in self.var_map.iter().enumerate() {
            x_full[orig] = x_reduced[reduced];
        }
        x_full
    }

    pub(crate) fn unmap_solution(
        &self,
        inner: &dyn NlpProblem,
        reduced: &SolveResult,
    ) -> SolveResult {
        self.unmap_solution_with_options(inner, reduced, None)
    }

    pub(crate) fn unmap_solution_with_options(
        &self,
        inner: &dyn NlpProblem,
        reduced: &SolveResult,
        options: Option<&SolverOptions>,
    ) -> SolveResult {
        let x_full = self.expand_x(&reduced.x);

        let mut constraint_multipliers = vec![0.0; self.m_orig];
        for (reduced_idx, &orig_idx) in self.row_map.iter().enumerate() {
            if reduced_idx < reduced.constraint_multipliers.len() {
                constraint_multipliers[orig_idx] = reduced.constraint_multipliers[reduced_idx];
            }
        }

        let mut bound_multipliers_lower = vec![0.0; self.n_orig];
        let mut bound_multipliers_upper = vec![0.0; self.n_orig];
        for (reduced_idx, &orig_idx) in self.var_map.iter().enumerate() {
            if reduced_idx < reduced.bound_multipliers_lower.len() {
                bound_multipliers_lower[orig_idx] = reduced.bound_multipliers_lower[reduced_idx];
            }
            if reduced_idx < reduced.bound_multipliers_upper.len() {
                bound_multipliers_upper[orig_idx] = reduced.bound_multipliers_upper[reduced_idx];
            }
        }

        if self.multiplier_recovery == RemovedMultiplierRecovery::AuxiliaryStationarity {
            if let Err(reason) = self.reconstruct_removed_constraint_multipliers(
                inner,
                &x_full,
                &mut constraint_multipliers,
                &bound_multipliers_lower,
                &bound_multipliers_upper,
            ) {
                if let Some(options) = options {
                    if options.print_level >= 5 {
                        rip_log!(
                            "ripopt: Auxiliary multiplier reconstruction skipped: {}",
                            reason
                        );
                    }
                }
            }
        }

        let mut objective = reduced.objective;
        let _ = inner.objective(&x_full, true, &mut objective);

        let mut constraint_values = vec![0.0; self.m_orig];
        if self.m_orig > 0 {
            let _ = inner.constraints(&x_full, true, &mut constraint_values);
        }

        SolveResult {
            x: x_full,
            objective,
            constraint_multipliers,
            bound_multipliers_lower,
            bound_multipliers_upper,
            constraint_values,
            status: reduced.status,
            iterations: reduced.iterations,
            diagnostics: reduced.diagnostics.clone(),
        }
    }

    fn reconstruct_removed_constraint_multipliers(
        &self,
        inner: &dyn NlpProblem,
        x_full: &[f64],
        constraint_multipliers: &mut [f64],
        bound_multipliers_lower: &[f64],
        bound_multipliers_upper: &[f64],
    ) -> Result<(), &'static str> {
        let removed_rows = self.removed_rows();
        let removed_vars = self.removed_vars();
        if removed_rows.is_empty() && removed_vars.is_empty() {
            return Ok(());
        }
        if removed_rows.len() != removed_vars.len() {
            return Err("removed auxiliary row/variable counts are not square");
        }
        if removed_rows.is_empty() {
            return Ok(());
        }
        if self.removed_vars_have_bound_ambiguity(inner, &removed_vars, x_full) {
            return Err("removed auxiliary variable is active at a finite bound");
        }

        let mut grad = vec![0.0; self.n_orig];
        if !inner.gradient(x_full, true, &mut grad) {
            return Err("full gradient evaluation failed");
        }

        let (jac_rows, jac_cols) = inner.jacobian_structure();
        if jac_rows.len() != jac_cols.len() {
            return Err("Jacobian structure has mismatched row/column lengths");
        }
        let mut jac_vals = vec![0.0; jac_rows.len()];
        if !inner.jacobian_values(x_full, true, &mut jac_vals) {
            return Err("full Jacobian evaluation failed");
        }

        let row_pos = index_positions(self.m_orig, &removed_rows);
        let var_pos = index_positions(self.n_orig, &removed_vars);
        let dim = removed_rows.len();
        let mut system = vec![0.0; dim * dim];
        let mut rhs: Vec<f64> = removed_vars
            .iter()
            .map(|&var| -(grad[var] - bound_multipliers_lower[var] + bound_multipliers_upper[var]))
            .collect();

        for (idx, (&row, &col)) in jac_rows.iter().zip(jac_cols.iter()).enumerate() {
            if row >= self.m_orig || col >= self.n_orig {
                continue;
            }
            let val = jac_vals[idx];
            if !val.is_finite() {
                return Err("Jacobian contains a non-finite value");
            }
            let Some(var_local) = var_pos[col] else {
                continue;
            };
            if let Some(row_local) = row_pos[row] {
                system[var_local * dim + row_local] += val;
            } else {
                rhs[var_local] -= val * constraint_multipliers[row];
            }
        }

        let lambda = solve_dense_square_system(system, rhs, dim)?;
        for (local, &row) in removed_rows.iter().enumerate() {
            constraint_multipliers[row] = lambda[local];
        }
        Ok(())
    }

    fn removed_vars_have_bound_ambiguity(
        &self,
        inner: &dyn NlpProblem,
        removed_vars: &[usize],
        x_full: &[f64],
    ) -> bool {
        let mut x_l = vec![0.0; self.n_orig];
        let mut x_u = vec![0.0; self.n_orig];
        inner.bounds(&mut x_l, &mut x_u);
        removed_vars.iter().any(|&var| {
            let x = x_full[var];
            let scale_l = x.abs().max(x_l[var].abs()).max(1.0);
            let scale_u = x.abs().max(x_u[var].abs()).max(1.0);
            (x_l[var].is_finite() && x <= x_l[var] + 1e-8 * scale_l)
                || (x_u[var].is_finite() && x >= x_u[var] - 1e-8 * scale_u)
        })
    }
}

pub(crate) struct ReductionStack<'a> {
    layers: Vec<ReductionLayer<'a>>,
}

struct ReductionLayer<'a> {
    frame: &'a ReductionFrame,
    inner: &'a dyn NlpProblem,
}

impl<'a> ReductionStack<'a> {
    pub(crate) fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub(crate) fn push(mut self, frame: &'a ReductionFrame, inner: &'a dyn NlpProblem) -> Self {
        self.layers.push(ReductionLayer { frame, inner });
        self
    }

    pub(crate) fn unmap_solution_with_options(
        &self,
        reduced: &SolveResult,
        options: Option<&SolverOptions>,
    ) -> SolveResult {
        let mut current = reduced.clone();
        for layer in &self.layers {
            current = layer
                .frame
                .unmap_solution_with_options(layer.inner, &current, options);
        }
        current
    }
}

fn removed_indices(total: usize, kept: &[usize]) -> Vec<usize> {
    let mut is_kept = vec![false; total];
    for &idx in kept {
        if idx < total {
            is_kept[idx] = true;
        }
    }
    (0..total).filter(|&idx| !is_kept[idx]).collect()
}

fn index_positions(total: usize, indices: &[usize]) -> Vec<Option<usize>> {
    let mut positions = vec![None; total];
    for (pos, &idx) in indices.iter().enumerate() {
        if idx < total {
            positions[idx] = Some(pos);
        }
    }
    positions
}

pub(crate) fn solve_dense_square_system(
    mut matrix: Vec<f64>,
    mut rhs: Vec<f64>,
    dim: usize,
) -> Result<Vec<f64>, &'static str> {
    if dim == 0 {
        return Ok(Vec::new());
    }
    if matrix.len() != dim * dim || rhs.len() != dim {
        return Err("dense system dimensions are inconsistent");
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err("dense system matrix contains a non-finite value");
    }
    if rhs.iter().any(|value| !value.is_finite()) {
        return Err("dense system RHS contains a non-finite value");
    }

    let matrix_norm = matrix
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    if matrix_norm == 0.0 {
        return Err("dense system matrix is zero");
    }

    let original_matrix = matrix.clone();
    let original_rhs = rhs.clone();
    let pivot_tol = (dim as f64) * matrix_norm * 1e-10;

    for col in 0..dim {
        let pivot_row = (col..dim)
            .max_by(|&a, &b| {
                matrix[a * dim + col]
                    .abs()
                    .partial_cmp(&matrix[b * dim + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or("dense system has no pivot row")?;
        let pivot_abs = matrix[pivot_row * dim + col].abs();
        if pivot_abs <= pivot_tol {
            return Err("dense system is singular or ill-conditioned");
        }
        if pivot_row != col {
            for j in col..dim {
                matrix.swap(col * dim + j, pivot_row * dim + j);
            }
            rhs.swap(col, pivot_row);
        }

        let pivot = matrix[col * dim + col];
        for row in (col + 1)..dim {
            let factor = matrix[row * dim + col] / pivot;
            matrix[row * dim + col] = 0.0;
            for j in (col + 1)..dim {
                matrix[row * dim + j] -= factor * matrix[col * dim + j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    let mut solution = vec![0.0; dim];
    for i in (0..dim).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..dim {
            sum -= matrix[i * dim + j] * solution[j];
        }
        let pivot = matrix[i * dim + i];
        if pivot.abs() <= pivot_tol {
            return Err("dense system is singular or ill-conditioned");
        }
        solution[i] = sum / pivot;
    }
    if solution.iter().any(|value| !value.is_finite()) {
        return Err("dense system solution is non-finite");
    }

    let residual = dense_residual_inf(&original_matrix, &solution, &original_rhs, dim);
    let rhs_norm = original_rhs
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    if residual > 1e-8 * rhs_norm.max(1.0) {
        return Err("dense system residual is too large");
    }

    Ok(solution)
}

fn dense_residual_inf(matrix: &[f64], x: &[f64], rhs: &[f64], dim: usize) -> f64 {
    let mut residual: f64 = 0.0;
    for row in 0..dim {
        let mut ax = 0.0;
        for col in 0..dim {
            ax += matrix[row * dim + col] * x[col];
        }
        residual = residual.max((ax - rhs[row]).abs());
    }
    residual
}
