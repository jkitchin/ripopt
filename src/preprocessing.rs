//! Preprocessing: eliminate fixed variables and redundant constraints before solving.
//!
//! Wraps an `NlpProblem` to present a reduced problem with:
//! - Fixed variables (x_l == x_u) removed and set to their fixed values
//! - Redundant constraints (duplicate Jacobian rows) removed
//! - Variable bounds tightened from single-variable linear constraints

use crate::problem::NlpProblem;
use crate::result::SolveResult;

/// NLP problem wrapper that eliminates fixed variables and redundant constraints.
///
/// Uses dynamic dispatch (`&dyn NlpProblem`) to avoid monomorphization issues
/// when called from inside the generic `solve<P>`.
pub struct PreprocessedProblem<'a> {
    inner: &'a dyn NlpProblem,
    n_orig: usize,
    m_orig: usize,
    /// Reduced variable index -> original variable index
    var_map: Vec<usize>,
    /// Reduced constraint index -> original constraint index
    constr_map: Vec<usize>,
    /// Full-size vector with fixed values filled in (NaN for free variables)
    fixed_values: Vec<f64>,
    /// Original variable index -> reduced variable index (None if fixed)
    _orig_to_reduced_var: Vec<Option<usize>>,
    /// Remapped Jacobian sparsity
    jac_rows: Vec<usize>,
    jac_cols: Vec<usize>,
    /// For each reduced Jacobian entry, index into inner Jacobian values
    jac_entry_map: Vec<usize>,
    /// Remapped Hessian sparsity
    hess_rows: Vec<usize>,
    hess_cols: Vec<usize>,
    /// For each reduced Hessian entry, index into inner Hessian values
    hess_entry_map: Vec<usize>,
    /// Tightened lower bounds for reduced variables
    tightened_x_l: Vec<f64>,
    /// Tightened upper bounds for reduced variables
    tightened_x_u: Vec<f64>,
}

/// Tolerance for detecting fixed variables (x_l ≈ x_u).
const FIXED_TOL: f64 = 1e-10;

impl<'a> PreprocessedProblem<'a> {
    /// Analyze and build the preprocessed problem wrapper.
    pub fn new(inner: &'a dyn NlpProblem, bound_push: f64) -> Self {
        let n = inner.num_variables();
        let m = inner.num_constraints();

        let mut x_l = vec![0.0; n];
        let mut x_u = vec![0.0; n];
        inner.bounds(&mut x_l, &mut x_u);

        // --- Fixed variable detection ---
        let mut is_fixed = vec![false; n];
        let mut fixed_values = vec![f64::NAN; n];
        for i in 0..n {
            if x_l[i].is_finite() && x_u[i].is_finite() && (x_u[i] - x_l[i]).abs() < FIXED_TOL {
                is_fixed[i] = true;
                fixed_values[i] = (x_l[i] + x_u[i]) / 2.0;
            }
        }

        // Build variable mapping
        let mut var_map = Vec::new();
        let mut orig_to_reduced_var = vec![None; n];
        for i in 0..n {
            if !is_fixed[i] {
                orig_to_reduced_var[i] = Some(var_map.len());
                var_map.push(i);
            }
        }

        // --- Bound tightening from single-variable linear constraints ---
        // Evaluate Jacobian at initial point to find single-nonzero rows
        let mut tightened_x_l: Vec<f64> = var_map.iter().map(|&i| x_l[i]).collect();
        let mut tightened_x_u: Vec<f64> = var_map.iter().map(|&i| x_u[i]).collect();

        let (inner_jac_rows, inner_jac_cols) = inner.jacobian_structure();
        let jac_nnz = inner_jac_rows.len();

        'bound_tightening: {
        if m > 0 && jac_nnz > 0 {
            let mut x0 = vec![0.0; n];
            inner.initial_point(&mut x0);
            // Fill fixed values into x0 for evaluation
            for i in 0..n {
                if is_fixed[i] {
                    x0[i] = fixed_values[i];
                }
            }

            let mut jac_vals = vec![0.0; jac_nnz];
            // If Jacobian eval fails, skip bound tightening (safe default)
            if !inner.jacobian_values(&x0, true, &mut jac_vals) {
                break 'bound_tightening;
            }

            // Verify linearity: evaluate Jacobian at a perturbed point
            let mut x1 = x0.clone();
            for i in 0..n {
                if !is_fixed[i] {
                    let pert = 1.0 + 0.1 * x0[i].abs();
                    x1[i] = x0[i] + pert;
                    if x_u[i].is_finite() {
                        x1[i] = x1[i].min(x_u[i]);
                    }
                    if (x1[i] - x0[i]).abs() < 1e-4 {
                        x1[i] = x0[i] - pert;
                        if x_l[i].is_finite() {
                            x1[i] = x1[i].max(x_l[i]);
                        }
                    }
                }
            }
            let mut jac_vals1 = vec![0.0; jac_nnz];
            if !inner.jacobian_values(&x1, true, &mut jac_vals1) {
                break 'bound_tightening;
            }

            // Identify which constraints are truly linear (Jacobian doesn't change)
            let mut is_linear_constraint = vec![true; m];
            for k in 0..jac_nnz {
                let row = inner_jac_rows[k];
                if !is_linear_constraint[row] {
                    continue;
                }
                let diff = (jac_vals[k] - jac_vals1[k]).abs();
                let scale = jac_vals[k].abs().max(jac_vals1[k].abs()).max(1.0);
                if diff > 1e-10 * scale {
                    is_linear_constraint[row] = false;
                }
            }

            let mut g_l = vec![0.0; m];
            let mut g_u = vec![0.0; m];
            inner.constraint_bounds(&mut g_l, &mut g_u);

            // For each LINEAR constraint with exactly one free-variable nonzero,
            // tighten the variable's bounds.
            for ci in 0..m {
                if !is_linear_constraint[ci] {
                    continue; // Only tighten bounds from verified linear constraints
                }

                let mut free_entries: Vec<(usize, f64)> = Vec::new(); // (orig_col, value)
                let mut fixed_contribution = 0.0;

                for (k, (&r, &c)) in inner_jac_rows.iter().zip(inner_jac_cols.iter()).enumerate() {
                    if r == ci && jac_vals[k].abs() > 1e-20 {
                        if is_fixed[c] {
                            fixed_contribution += jac_vals[k] * fixed_values[c];
                        } else {
                            free_entries.push((c, jac_vals[k]));
                        }
                    }
                }

                if free_entries.len() == 1 {
                    let (orig_j, a_j) = free_entries[0];
                    if let Some(red_j) = orig_to_reduced_var[orig_j] {
                        // Constraint: g_l <= a_j * x_j + fixed_part <= g_u
                        // => (g_l - fixed_part) / a_j <= x_j <= (g_u - fixed_part) / a_j (if a_j > 0)
                        // Flip if a_j < 0
                        let adj_l = if g_l[ci].is_finite() {
                            (g_l[ci] - fixed_contribution) / a_j
                        } else if a_j > 0.0 {
                            f64::NEG_INFINITY
                        } else {
                            f64::INFINITY
                        };
                        let adj_u = if g_u[ci].is_finite() {
                            (g_u[ci] - fixed_contribution) / a_j
                        } else if a_j > 0.0 {
                            f64::INFINITY
                        } else {
                            f64::NEG_INFINITY
                        };
                        let (new_l, new_u) = if a_j > 0.0 {
                            (adj_l, adj_u)
                        } else {
                            (adj_u, adj_l)
                        };
                        // Compute candidate bounds after this tightening
                        let candidate_l = if new_l.is_finite() && new_l > tightened_x_l[red_j] {
                            new_l
                        } else {
                            tightened_x_l[red_j]
                        };
                        let candidate_u = if new_u.is_finite() && new_u < tightened_x_u[red_j] {
                            new_u
                        } else {
                            tightened_x_u[red_j]
                        };
                        // Skip tightening if it would create a range smaller than 2*bound_push
                        // (the initial point needs at least bound_push slack on each side)
                        if candidate_l.is_finite() && candidate_u.is_finite()
                            && (candidate_u - candidate_l) < 2.0 * bound_push
                        {
                            continue;
                        }
                        if new_l.is_finite() && new_l > tightened_x_l[red_j] {
                            tightened_x_l[red_j] = new_l;
                        }
                        if new_u.is_finite() && new_u < tightened_x_u[red_j] {
                            tightened_x_u[red_j] = new_u;
                        }
                    }
                }
            }
        }
        } // 'bound_tightening

        // --- Redundant constraint detection ---
        // Two constraints are redundant if they have the same Jacobian structure
        // (same columns with same values at TWO points) and same bounds.
        // Checking at two points prevents false matches from nonlinear constraints
        // that happen to have the same Jacobian at one particular point.
        let mut constr_map: Vec<usize> = (0..m).collect();

        if m > 1 && jac_nnz > 0 {
            let mut x0 = vec![0.0; n];
            inner.initial_point(&mut x0);
            for i in 0..n {
                if is_fixed[i] {
                    x0[i] = fixed_values[i];
                }
            }
            let mut jac_vals0 = vec![0.0; jac_nnz];
            let jac0_ok = inner.jacobian_values(&x0, true, &mut jac_vals0);

            // Also evaluate at a perturbed point for verification
            let mut x1 = x0.clone();
            for i in 0..n {
                if !is_fixed[i] {
                    let pert = 1.0 + 0.1 * x0[i].abs();
                    x1[i] = x0[i] + pert;
                    if x_u[i].is_finite() {
                        x1[i] = x1[i].min(x_u[i]);
                    }
                    if (x1[i] - x0[i]).abs() < 1e-4 {
                        x1[i] = x0[i] - pert;
                        if x_l[i].is_finite() {
                            x1[i] = x1[i].max(x_l[i]);
                        }
                    }
                }
            }
            let mut jac_vals1 = vec![0.0; jac_nnz];
            let jac1_ok = inner.jacobian_values(&x1, true, &mut jac_vals1);

            // Also compare constraint values at the perturbed point
            let mut g0 = vec![0.0; m];
            let mut g1 = vec![0.0; m];
            let g0_ok = inner.constraints(&x0, true, &mut g0);
            let g1_ok = inner.constraints(&x1, true, &mut g1);

            let mut g_l = vec![0.0; m];
            let mut g_u = vec![0.0; m];
            inner.constraint_bounds(&mut g_l, &mut g_u);

            // Build per-constraint entry lists at both points
            let mut constr_entries0: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
            let mut constr_entries1: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
            for (k, (&r, &c)) in inner_jac_rows.iter().zip(inner_jac_cols.iter()).enumerate() {
                if !is_fixed[c] {
                    constr_entries0[r].push((c, jac_vals0[k]));
                    constr_entries1[r].push((c, jac_vals1[k]));
                }
            }
            for entries in &mut constr_entries0 {
                entries.sort_by_key(|&(c, _)| c);
            }
            for entries in &mut constr_entries1 {
                entries.sort_by_key(|&(c, _)| c);
            }

            let mut is_redundant = vec![false; m];
            for i in 0..m {
                if is_redundant[i] {
                    continue;
                }
                for j in (i + 1)..m {
                    if is_redundant[j] {
                        continue;
                    }
                    // Check same bounds
                    if (g_l[i] - g_l[j]).abs() > FIXED_TOL || (g_u[i] - g_u[j]).abs() > FIXED_TOL {
                        continue;
                    }
                    // Check same Jacobian entries at BOTH points
                    if constr_entries0[i].len() != constr_entries0[j].len() {
                        continue;
                    }
                    let same_at_x0 = constr_entries0[i]
                        .iter()
                        .zip(constr_entries0[j].iter())
                        .all(|(&(ci, vi), &(cj, vj))| ci == cj && (vi - vj).abs() < FIXED_TOL);
                    if !same_at_x0 {
                        continue;
                    }
                    let same_at_x1 = constr_entries1[i]
                        .iter()
                        .zip(constr_entries1[j].iter())
                        .all(|(&(ci, vi), &(cj, vj))| ci == cj && (vi - vj).abs() < FIXED_TOL);
                    if !same_at_x1 {
                        continue;
                    }
                    // Also check constraint values match at x1
                    let g_diff = (g0[i] - g0[j]).abs().max((g1[i] - g1[j]).abs());
                    let g_scale = g0[i].abs().max(g0[j].abs()).max(1.0);
                    if g_diff > FIXED_TOL * g_scale {
                        continue;
                    }
                    is_redundant[j] = true;
                }
            }

            constr_map = (0..m).filter(|&i| !is_redundant[i]).collect();
        }

        // --- Remap Jacobian sparsity ---
        let mut jac_rows_new = Vec::new();
        let mut jac_cols_new = Vec::new();
        let mut jac_entry_map = Vec::new();

        // Build reverse constraint map: orig -> reduced
        let mut orig_to_reduced_constr = vec![None; m];
        for (red_i, &orig_i) in constr_map.iter().enumerate() {
            orig_to_reduced_constr[orig_i] = Some(red_i);
        }

        for (k, (&r, &c)) in inner_jac_rows.iter().zip(inner_jac_cols.iter()).enumerate() {
            if let (Some(red_r), Some(red_c)) = (orig_to_reduced_constr[r], orig_to_reduced_var[c])
            {
                jac_rows_new.push(red_r);
                jac_cols_new.push(red_c);
                jac_entry_map.push(k);
            }
        }

        // --- Remap Hessian sparsity ---
        let (inner_hess_rows, inner_hess_cols) = inner.hessian_structure();
        let mut hess_rows_new = Vec::new();
        let mut hess_cols_new = Vec::new();
        let mut hess_entry_map = Vec::new();

        for (k, (&r, &c)) in inner_hess_rows.iter().zip(inner_hess_cols.iter()).enumerate() {
            if let (Some(red_r), Some(red_c)) = (orig_to_reduced_var[r], orig_to_reduced_var[c]) {
                // Maintain lower triangle: row >= col
                if red_r >= red_c {
                    hess_rows_new.push(red_r);
                    hess_cols_new.push(red_c);
                } else {
                    hess_rows_new.push(red_c);
                    hess_cols_new.push(red_r);
                }
                hess_entry_map.push(k);
            }
        }

        PreprocessedProblem {
            inner,
            n_orig: n,
            m_orig: m,
            var_map,
            constr_map,
            fixed_values,
            _orig_to_reduced_var: orig_to_reduced_var,
            jac_rows: jac_rows_new,
            jac_cols: jac_cols_new,
            jac_entry_map,
            hess_rows: hess_rows_new,
            hess_cols: hess_cols_new,
            hess_entry_map,
            tightened_x_l,
            tightened_x_u,
        }
    }

    /// Whether preprocessing actually reduced the problem.
    pub fn did_reduce(&self) -> bool {
        self.var_map.len() < self.n_orig || self.constr_map.len() < self.m_orig
    }

    /// Number of fixed variables eliminated.
    pub fn num_fixed(&self) -> usize {
        self.n_orig - self.var_map.len()
    }

    /// Number of redundant constraints eliminated.
    pub fn num_redundant(&self) -> usize {
        self.m_orig - self.constr_map.len()
    }

    /// Build a full-size x vector from a reduced x vector.
    fn expand_x(&self, x_reduced: &[f64]) -> Vec<f64> {
        let mut x_full = self.fixed_values.clone();
        for (red_i, &orig_i) in self.var_map.iter().enumerate() {
            x_full[orig_i] = x_reduced[red_i];
        }
        x_full
    }

    /// Map the solution from the reduced problem back to the original problem dimensions.
    pub fn unmap_solution(&self, reduced: &SolveResult) -> SolveResult {
        let n = self.n_orig;
        let m = self.m_orig;

        // Expand x
        let x_full = self.expand_x(&reduced.x);

        // Expand bound multipliers
        let mut z_l_full = vec![0.0; n];
        let mut z_u_full = vec![0.0; n];
        for (red_i, &orig_i) in self.var_map.iter().enumerate() {
            z_l_full[orig_i] = reduced.bound_multipliers_lower[red_i];
            z_u_full[orig_i] = reduced.bound_multipliers_upper[red_i];
        }

        // Expand constraint multipliers and values
        let mut y_full = vec![0.0; m];
        let mut g_full = vec![0.0; m];

        // Evaluate original constraints at the full solution
        if m > 0 {
            let _ = self.inner.constraints(&x_full, true, &mut g_full);
        }

        for (red_i, &orig_i) in self.constr_map.iter().enumerate() {
            if red_i < reduced.constraint_multipliers.len() {
                y_full[orig_i] = reduced.constraint_multipliers[red_i];
            }
        }

        // Recompute objective at full solution
        let mut obj = 0.0;
        let _ = self.inner.objective(&x_full, true, &mut obj);

        SolveResult {
            x: x_full,
            objective: obj,
            constraint_multipliers: y_full,
            bound_multipliers_lower: z_l_full,
            bound_multipliers_upper: z_u_full,
            constraint_values: g_full,
            status: reduced.status,
            iterations: reduced.iterations,
            diagnostics: reduced.diagnostics.clone(),
        }
    }
}

impl NlpProblem for PreprocessedProblem<'_> {
    fn num_variables(&self) -> usize {
        self.var_map.len()
    }

    fn num_constraints(&self) -> usize {
        self.constr_map.len()
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for (red_i, _) in self.var_map.iter().enumerate() {
            x_l[red_i] = self.tightened_x_l[red_i];
            x_u[red_i] = self.tightened_x_u[red_i];
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        let m = self.m_orig;
        let mut g_l_full = vec![0.0; m];
        let mut g_u_full = vec![0.0; m];
        self.inner.constraint_bounds(&mut g_l_full, &mut g_u_full);
        for (red_i, &orig_i) in self.constr_map.iter().enumerate() {
            g_l[red_i] = g_l_full[orig_i];
            g_u[red_i] = g_u_full[orig_i];
        }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        let mut x0_full = vec![0.0; self.n_orig];
        self.inner.initial_point(&mut x0_full);
        for (red_i, &orig_i) in self.var_map.iter().enumerate() {
            x0[red_i] = x0_full[orig_i];
        }
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        let x_full = self.expand_x(x);
        self.inner.objective(&x_full, _new_x, obj)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        let x_full = self.expand_x(x);
        let mut grad_full = vec![0.0; self.n_orig];
        if !self.inner.gradient(&x_full, _new_x, &mut grad_full) {
            return false;
        }
        for (red_i, &orig_i) in self.var_map.iter().enumerate() {
            grad[red_i] = grad_full[orig_i];
        }
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        let x_full = self.expand_x(x);
        let mut g_full = vec![0.0; self.m_orig];
        if !self.inner.constraints(&x_full, _new_x, &mut g_full) {
            return false;
        }
        for (red_i, &orig_i) in self.constr_map.iter().enumerate() {
            g[red_i] = g_full[orig_i];
        }
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.jac_rows.clone(), self.jac_cols.clone())
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        let x_full = self.expand_x(x);
        let inner_nnz = self.jac_entry_map.iter().copied().max().map_or(0, |m| m + 1);
        if inner_nnz == 0 {
            return true;
        }
        // Get all inner Jacobian values
        let (inner_jac_rows, _) = self.inner.jacobian_structure();
        let mut inner_vals = vec![0.0; inner_jac_rows.len()];
        if !self.inner.jacobian_values(&x_full, _new_x, &mut inner_vals) {
            return false;
        }
        for (red_k, &orig_k) in self.jac_entry_map.iter().enumerate() {
            vals[red_k] = inner_vals[orig_k];
        }
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.hess_rows.clone(), self.hess_cols.clone())
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        let x_full = self.expand_x(x);

        // Expand lambda to original constraint indices
        let mut lambda_full = vec![0.0; self.m_orig];
        for (red_i, &orig_i) in self.constr_map.iter().enumerate() {
            lambda_full[orig_i] = lambda[red_i];
        }

        // Get all inner Hessian values
        let (inner_hess_rows, _) = self.inner.hessian_structure();
        let inner_nnz = inner_hess_rows.len();
        let mut inner_vals = vec![0.0; inner_nnz];
        if !self.inner
            .hessian_values(&x_full, _new_x, obj_factor, &lambda_full, &mut inner_vals) {
            return false;
        }

        for (red_k, &orig_k) in self.hess_entry_map.iter().enumerate() {
            vals[red_k] = inner_vals[orig_k];
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::SolveStatus;

    /// Problem with fixed variables: min (x0-1)^2 + (x1-2)^2 + (x2-3)^2
    /// with x1 fixed at 2.0 (x_l[1] == x_u[1] == 2.0)
    /// and constraint x0 + x2 = 4
    struct FixedVarProblem;

    impl NlpProblem for FixedVarProblem {
        fn num_variables(&self) -> usize {
            3
        }
        fn num_constraints(&self) -> usize {
            1
        }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = f64::NEG_INFINITY;
            x_u[0] = f64::INFINITY;
            x_l[1] = 2.0;
            x_u[1] = 2.0; // fixed
            x_l[2] = f64::NEG_INFINITY;
            x_u[2] = f64::INFINITY;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 4.0;
            g_u[0] = 4.0;
        }
        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 1.0;
            x0[1] = 2.0;
            x0[2] = 3.0;
        }
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2) + (x[2] - 3.0).powi(2);
            true
        }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = 2.0 * (x[0] - 1.0);
            grad[1] = 2.0 * (x[1] - 2.0);
            grad[2] = 2.0 * (x[2] - 3.0);
            true
        }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g[0] = x[0] + x[2];
            true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 0], vec![0, 2])
        }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals[0] = 1.0;
            vals[1] = 1.0;
            true
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1, 2], vec![0, 1, 2])
        }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
            vals[0] = 2.0 * obj_factor;
            vals[1] = 2.0 * obj_factor;
            vals[2] = 2.0 * obj_factor;
            true
        }
    }

    #[test]
    fn test_fixed_var_elimination() {
        let prob = FixedVarProblem;
        let prep = PreprocessedProblem::new(&prob as &dyn NlpProblem, 1e-2);

        assert!(prep.did_reduce());
        assert_eq!(prep.num_fixed(), 1);
        assert_eq!(prep.num_variables(), 2); // x0, x2
        assert_eq!(prep.num_constraints(), 1);

        // Check variable mapping: var_map should be [0, 2]
        assert_eq!(prep.var_map, vec![0, 2]);
    }

    #[test]
    fn test_fixed_var_objective() {
        let prob = FixedVarProblem;
        let prep = PreprocessedProblem::new(&prob as &dyn NlpProblem, 1e-2);

        // With reduced x = [1.0, 3.0] (x0=1, x2=3), x1 is fixed at 2.0
        // obj = (1-1)^2 + (2-2)^2 + (3-3)^2 = 0
        let x_red = vec![1.0, 3.0];
        let mut obj_val = 0.0;
        prep.objective(&x_red, true, &mut obj_val);
        assert!((obj_val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_var_gradient() {
        let prob = FixedVarProblem;
        let prep = PreprocessedProblem::new(&prob as &dyn NlpProblem, 1e-2);

        let x_red = vec![1.0, 3.0];
        let mut grad = vec![0.0; 2];
        prep.gradient(&x_red, true, &mut grad);
        assert!((grad[0] - 0.0).abs() < 1e-10); // d/dx0 = 0
        assert!((grad[1] - 0.0).abs() < 1e-10); // d/dx2 = 0
    }

    #[test]
    fn test_fixed_var_constraints() {
        let prob = FixedVarProblem;
        let prep = PreprocessedProblem::new(&prob as &dyn NlpProblem, 1e-2);

        let x_red = vec![1.5, 2.5]; // x0=1.5, x2=2.5
        let mut g = vec![0.0; 1];
        prep.constraints(&x_red, true, &mut g);
        assert!((g[0] - 4.0).abs() < 1e-10); // 1.5 + 2.5 = 4.0
    }

    #[test]
    fn test_unmap_solution() {
        let prob = FixedVarProblem;
        let prep = PreprocessedProblem::new(&prob as &dyn NlpProblem, 1e-2);

        let reduced_result = SolveResult {
            x: vec![2.0, 2.0],
            objective: 2.0,
            constraint_multipliers: vec![1.0],
            bound_multipliers_lower: vec![0.0, 0.0],
            bound_multipliers_upper: vec![0.0, 0.0],
            constraint_values: vec![4.0],
            status: SolveStatus::Optimal,
            iterations: 10,
            diagnostics: Default::default(),
        };

        let full = prep.unmap_solution(&reduced_result);
        assert_eq!(full.x.len(), 3);
        assert!((full.x[0] - 2.0).abs() < 1e-10);
        assert!((full.x[1] - 2.0).abs() < 1e-10); // fixed value
        assert!((full.x[2] - 2.0).abs() < 1e-10);
        assert_eq!(full.constraint_multipliers.len(), 1);
        assert_eq!(full.bound_multipliers_lower.len(), 3);
    }

    /// Problem with duplicate constraints for redundancy detection
    struct DuplicateConstraintProblem;

    impl NlpProblem for DuplicateConstraintProblem {
        fn num_variables(&self) -> usize {
            2
        }
        fn num_constraints(&self) -> usize {
            3
        }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = 0.0;
            x_u[0] = 10.0;
            x_l[1] = 0.0;
            x_u[1] = 10.0;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            // Constraints 0 and 2 are identical
            g_l[0] = 1.0;
            g_u[0] = 5.0;
            g_l[1] = 0.0;
            g_u[1] = 3.0;
            g_l[2] = 1.0; // duplicate of constraint 0
            g_u[2] = 5.0;
        }
        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 1.0;
            x0[1] = 1.0;
        }
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = x[0] * x[0] + x[1] * x[1];
            true
        }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = 2.0 * x[0];
            grad[1] = 2.0 * x[1];
            true
        }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g[0] = x[0] + x[1]; // same as g[2]
            g[1] = x[0] - x[1];
            g[2] = x[0] + x[1]; // duplicate of g[0]
            true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            // Dense: rows [0,0,1,1,2,2], cols [0,1,0,1,0,1]
            (vec![0, 0, 1, 1, 2, 2], vec![0, 1, 0, 1, 0, 1])
        }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals[0] = 1.0;
            vals[1] = 1.0; // dg0/dx1
            vals[2] = 1.0;
            vals[3] = -1.0; // dg1/dx1
            vals[4] = 1.0;
            vals[5] = 1.0; // dg2/dx1 (same as dg0)
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
    fn test_redundant_constraint_detection() {
        let prob = DuplicateConstraintProblem;
        let prep = PreprocessedProblem::new(&prob as &dyn NlpProblem, 1e-2);

        assert!(prep.did_reduce());
        assert_eq!(prep.num_redundant(), 1);
        assert_eq!(prep.num_constraints(), 2); // 3 - 1 duplicate
        assert_eq!(prep.constr_map, vec![0, 1]); // constraint 2 removed
    }

    #[test]
    fn test_no_reduction() {
        // A problem with nothing to reduce
        struct NothingToReduce;
        impl NlpProblem for NothingToReduce {
            fn num_variables(&self) -> usize {
                2
            }
            fn num_constraints(&self) -> usize {
                1
            }
            fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
                x_l[0] = 0.0;
                x_u[0] = 10.0;
                x_l[1] = 0.0;
                x_u[1] = 10.0;
            }
            fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
                g_l[0] = 1.0;
                g_u[0] = 5.0;
            }
            fn initial_point(&self, x0: &mut [f64]) {
                x0[0] = 1.0;
                x0[1] = 1.0;
            }
            fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = x[0] + x[1];
            true
        }
            fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
                grad[0] = 1.0;
                grad[1] = 1.0;
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
                (vec![], vec![])
            }
            fn hessian_values(
                &self,
                _x: &[f64],
                _new_x: bool,
                _obj_factor: f64,
                _lambda: &[f64],
                _vals: &mut [f64],
            ) -> bool { true }
        }

        let prob = NothingToReduce;
        let prep = PreprocessedProblem::new(&prob as &dyn NlpProblem, 1e-2);
        assert!(!prep.did_reduce());
    }

    /// Problem: 1 free variable, 1 linear constraint 2*x >= 4
    struct BoundTightenPositive;

    impl NlpProblem for BoundTightenPositive {
        fn num_variables(&self) -> usize { 1 }
        fn num_constraints(&self) -> usize { 1 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = f64::NEG_INFINITY;
            x_u[0] = f64::INFINITY;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 4.0;
            g_u[0] = f64::INFINITY;
        }
        fn initial_point(&self, x0: &mut [f64]) { x0[0] = 5.0; }
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = x[0] * x[0];
            true
        }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = 2.0 * x[0];
            true
        }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool { g[0] = 2.0 * x[0];
            true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0], vec![0])
        }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool { vals[0] = 2.0;
            true
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0], vec![0])
        }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
            vals[0] = 2.0 * obj_factor;
            true
        }
    }

    #[test]
    fn test_bound_tightening_positive_coeff() {
        let prob = BoundTightenPositive;
        let prep = PreprocessedProblem::new(&prob as &dyn NlpProblem, 1e-2);

        // Constraint 2*x >= 4 => x >= 2.0
        let mut xl = vec![0.0; 1];
        let mut xu = vec![0.0; 1];
        prep.bounds(&mut xl, &mut xu);
        assert!(xl[0] >= 2.0 - 1e-10, "expected x_l >= 2.0, got {}", xl[0]);
    }

    /// Problem: 1 free variable, 1 linear constraint -3*x <= -6
    struct BoundTightenNegative;

    impl NlpProblem for BoundTightenNegative {
        fn num_variables(&self) -> usize { 1 }
        fn num_constraints(&self) -> usize { 1 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = f64::NEG_INFINITY;
            x_u[0] = f64::INFINITY;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = f64::NEG_INFINITY;
            g_u[0] = -6.0;
        }
        fn initial_point(&self, x0: &mut [f64]) { x0[0] = 5.0; }
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = x[0] * x[0];
            true
        }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = 2.0 * x[0];
            true
        }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool { g[0] = -3.0 * x[0];
            true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0], vec![0])
        }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool { vals[0] = -3.0;
            true
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0], vec![0])
        }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
            vals[0] = 2.0 * obj_factor;
            true
        }
    }

    #[test]
    fn test_bound_tightening_negative_coeff() {
        let prob = BoundTightenNegative;
        let prep = PreprocessedProblem::new(&prob as &dyn NlpProblem, 1e-2);

        // Constraint -3*x <= -6 => x >= 2.0 (divide by -3, flip)
        let mut xl = vec![0.0; 1];
        let mut xu = vec![0.0; 1];
        prep.bounds(&mut xl, &mut xu);
        assert!(xl[0] >= 2.0 - 1e-10, "expected x_l >= 2.0, got {}", xl[0]);
    }

    /// Problem: 1 free variable, 1 linear constraint 1 <= 2*x <= 6 (both-sided)
    struct BoundTightenBothSided;

    impl NlpProblem for BoundTightenBothSided {
        fn num_variables(&self) -> usize { 1 }
        fn num_constraints(&self) -> usize { 1 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = f64::NEG_INFINITY;
            x_u[0] = f64::INFINITY;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 1.0;
            g_u[0] = 6.0;
        }
        fn initial_point(&self, x0: &mut [f64]) { x0[0] = 1.0; }
        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = x[0] * x[0];
            true
        }
        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool { grad[0] = 2.0 * x[0];
            true
        }
        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool { g[0] = 2.0 * x[0];
            true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0], vec![0])
        }
        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool { vals[0] = 2.0;
            true
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0], vec![0])
        }
        fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
            vals[0] = 2.0 * obj_factor;
            true
        }
    }

    #[test]
    fn test_bound_tightening_both_sided() {
        let prob = BoundTightenBothSided;
        let prep = PreprocessedProblem::new(&prob as &dyn NlpProblem, 1e-2);

        // Constraint 1 <= 2*x <= 6 => 0.5 <= x <= 3.0
        let mut xl = vec![0.0; 1];
        let mut xu = vec![0.0; 1];
        prep.bounds(&mut xl, &mut xu);
        assert!(xl[0] >= 0.5 - 1e-10, "expected x_l >= 0.5, got {}", xl[0]);
        assert!(xu[0] <= 3.0 + 1e-10, "expected x_u <= 3.0, got {}", xu[0]);
    }
}
