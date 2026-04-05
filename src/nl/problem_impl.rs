use std::collections::HashMap;

use super::autodiff::Tape;
use super::parser::NlFileData;
use crate::NlpProblem;

/// An NLP problem parsed from an NL file, implementing the NlpProblem trait.
pub struct NlProblem {
    n: usize,
    m: usize,
    x_l: Vec<f64>,
    x_u: Vec<f64>,
    g_l: Vec<f64>,
    g_u: Vec<f64>,
    x0: Vec<f64>,
    maximize: bool,

    /// Tape for the nonlinear part of the objective.
    obj_tape: Option<Tape>,
    /// Linear coefficients for the objective: (var_idx, coeff).
    obj_linear: Vec<(usize, f64)>,

    /// Tape for the nonlinear part of each constraint.
    con_tapes: Vec<Option<Tape>>,
    /// Linear coefficients for each constraint.
    con_linear: Vec<Vec<(usize, f64)>>,

    /// Jacobian sparsity: (row_indices, col_indices).
    jac_rows: Vec<usize>,
    jac_cols: Vec<usize>,
    /// Map from (row, col) to position in Jacobian values array.
    jac_map: HashMap<(usize, usize), usize>,

    /// Hessian sparsity (sparse lower triangle).
    hess_rows: Vec<usize>,
    hess_cols: Vec<usize>,
    /// Map from (row, col) lower-triangle pair to position in hessian values array.
    hess_map: HashMap<(usize, usize), usize>,
}

impl NlProblem {
    /// Build an NlProblem from parsed NL file data.
    pub fn from_nl_data(data: NlFileData) -> Self {
        let n = data.header.n_vars;
        let m = data.header.n_constrs;

        // Build tapes for objective
        let (obj_idx, maximize, obj_expr) = data
            .obj_exprs
            .into_iter()
            .next()
            .unwrap_or((0, false, None));

        // Pre-build common expression tapes to avoid exponential inlining blowup
        use super::autodiff::CommonExprCache;
        let ce_cache = CommonExprCache::build(&data.common_exprs, n);

        let obj_tape = obj_expr.map(|expr| Tape::build_cached(&expr, &data.common_exprs, n, &ce_cache));

        let obj_linear = if obj_idx < data.obj_linear.len() {
            data.obj_linear[obj_idx].clone()
        } else {
            Vec::new()
        };

        // Build tapes for constraints
        let con_tapes: Vec<Option<Tape>> = data
            .con_exprs
            .iter()
            .map(|expr| expr.as_ref().map(|e| Tape::build_cached(e, &data.common_exprs, n, &ce_cache)))
            .collect();

        // Build Jacobian sparsity from con_linear entries + nonlinear variables
        let mut jac_entries: Vec<(usize, usize)> = Vec::new();
        let mut jac_set: HashMap<(usize, usize), usize> = HashMap::new();

        for (i, linear) in data.con_linear.iter().enumerate() {
            for &(var_idx, _) in linear {
                let key = (i, var_idx);
                if !jac_set.contains_key(&key) {
                    let pos = jac_entries.len();
                    jac_set.insert(key, pos);
                    jac_entries.push(key);
                }
            }
        }

        // Also add entries for nonlinear variables in each constraint
        for (i, tape) in con_tapes.iter().enumerate() {
            if let Some(tape) = tape {
                for op in &tape.ops {
                    if let super::autodiff::TapeOp::Var(j) = op {
                        let key = (i, *j);
                        if !jac_set.contains_key(&key) {
                            let pos = jac_entries.len();
                            jac_set.insert(key, pos);
                            jac_entries.push(key);
                        }
                    }
                }
            }
        }

        let jac_rows: Vec<usize> = jac_entries.iter().map(|&(r, _)| r).collect();
        let jac_cols: Vec<usize> = jac_entries.iter().map(|&(_, c)| c).collect();

        // Compute exact sparse Hessian structure via sparsity propagation through tapes.
        // This tracks which variables influence each tape node and emits structural
        // nonzero pairs at each nonlinear op (like ASL does internally).
        use std::collections::BTreeSet;
        let mut hess_set: BTreeSet<(usize, usize)> = BTreeSet::new();

        if let Some(ref tape) = obj_tape {
            hess_set.extend(tape.hessian_sparsity());
        }
        for tape in &con_tapes {
            if let Some(tape) = tape {
                hess_set.extend(tape.hessian_sparsity());
            }
        }

        // Add diagonal entries for all nonlinear variables (for regularization)
        let mut all_nonlinear_vars = BTreeSet::new();
        if let Some(ref tape) = obj_tape {
            for &v in &tape.variables() {
                all_nonlinear_vars.insert(v);
            }
        }
        for tape in &con_tapes {
            if let Some(tape) = tape {
                for &v in &tape.variables() {
                    all_nonlinear_vars.insert(v);
                }
            }
        }
        for &v in &all_nonlinear_vars {
            hess_set.insert((v, v));
        }

        log::info!("Hessian sparsity: {} structural nonzeros (from {} nonlinear vars)", hess_set.len(), all_nonlinear_vars.len());

        let mut hess_rows = Vec::with_capacity(hess_set.len());
        let mut hess_cols = Vec::with_capacity(hess_set.len());
        let mut hess_map = HashMap::with_capacity(hess_set.len());
        for (idx, &(r, c)) in hess_set.iter().enumerate() {
            hess_rows.push(r);
            hess_cols.push(c);
            hess_map.insert((r, c), idx);
        }

        NlProblem {
            n,
            m,
            x_l: data.x_l,
            x_u: data.x_u,
            g_l: data.g_l,
            g_u: data.g_u,
            x0: data.x0,
            maximize,
            obj_tape,
            obj_linear,
            con_tapes,
            con_linear: data.con_linear,
            jac_rows,
            jac_cols,
            jac_map: jac_set,
            hess_rows,
            hess_cols,
            hess_map,
        }
    }

    /// Compute the gradient of the objective (nonlinear + linear).
    fn obj_gradient(&self, x: &[f64], grad: &mut [f64]) {
        grad.iter_mut().for_each(|v| *v = 0.0);

        // Nonlinear part via reverse AD
        if let Some(tape) = &self.obj_tape {
            tape.gradient(x, grad);
        }

        // Linear part
        for &(idx, coeff) in &self.obj_linear {
            if idx < grad.len() {
                grad[idx] += coeff;
            }
        }

        // Negate if maximizing (solver minimizes)
        if self.maximize {
            for g in grad.iter_mut() {
                *g = -*g;
            }
        }
    }

    /// Compute the gradient of constraint i (nonlinear + linear).
    fn con_gradient(&self, i: usize, x: &[f64], grad: &mut [f64]) {
        grad.iter_mut().for_each(|v| *v = 0.0);

        // Nonlinear part via reverse AD
        if let Some(tape) = &self.con_tapes[i] {
            tape.gradient(x, grad);
        }

        // Linear part
        for &(idx, coeff) in &self.con_linear[i] {
            if idx < grad.len() {
                grad[idx] += coeff;
            }
        }
    }

}

impl NlpProblem for NlProblem {
    fn num_variables(&self) -> usize {
        self.n
    }

    fn num_constraints(&self) -> usize {
        self.m
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l.copy_from_slice(&self.x_l);
        x_u.copy_from_slice(&self.x_u);
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l.copy_from_slice(&self.g_l);
        g_u.copy_from_slice(&self.g_u);
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0.copy_from_slice(&self.x0);
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        let mut val = 0.0;

        // Nonlinear part
        if let Some(tape) = &self.obj_tape {
            val += tape.eval(x);
        }

        // Linear part
        for &(idx, coeff) in &self.obj_linear {
            val += coeff * x[idx];
        }

        *obj = if self.maximize {
            -val
        } else {
            val
        };
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        self.obj_gradient(x, grad);
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        for i in 0..self.m {
            let mut val = 0.0;

            // Nonlinear part
            if let Some(tape) = &self.con_tapes[i] {
                val += tape.eval(x);
            }

            // Linear part
            for &(idx, coeff) in &self.con_linear[i] {
                val += coeff * x[idx];
            }

            g[i] = val;
        }
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.jac_rows.clone(), self.jac_cols.clone())
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals.iter_mut().for_each(|v| *v = 0.0);

        let mut grad = vec![0.0; self.n];

        for i in 0..self.m {
            self.con_gradient(i, x, &mut grad);

            // Scatter into vals using jac_map
            for j in 0..self.n {
                if grad[j] != 0.0 {
                    if let Some(&pos) = self.jac_map.get(&(i, j)) {
                        vals[pos] = grad[j];
                    }
                }
            }
        }
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.hess_rows.clone(), self.hess_cols.clone())
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        // Analytical Hessian via forward-over-reverse AD on each tape.
        vals.iter_mut().for_each(|v| *v = 0.0);

        // Objective Hessian contribution
        if let Some(ref tape) = self.obj_tape {
            let weight = if self.maximize { -obj_factor } else { obj_factor };
            tape.hessian_accumulate(x, weight, &self.hess_map, vals);
        }

        // Constraint Hessian contributions
        for (i, tape) in self.con_tapes.iter().enumerate() {
            if let Some(tape) = tape {
                tape.hessian_accumulate(x, lambda[i], &self.hess_map, vals);
            }
        }
        true
    }
}
