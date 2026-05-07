use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::autodiff::{ExternalResolver, Tape};
use super::expr::ExprNode;
use super::external::ExternalLibrary;
use super::parser::{ImportedFunc, NlFileData};
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
    ///
    /// If the problem uses AMPL imported (external) functions, the libraries
    /// listed in the `AMPLFUNC` environment variable are loaded via
    /// [`ExternalLibrary`] and each declared `F` segment is resolved against
    /// the union of their registered functions. An error is returned if any
    /// function cannot be resolved.
    pub fn from_nl_data(data: NlFileData) -> Result<Self, String> {
        // Phase-2: build a resolver mapping each Funcall id -> (library, name).
        let resolver = resolve_externals(&data)?;

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
        let ce_cache = CommonExprCache::build_with_externals(&data.common_exprs, n, &resolver);

        let obj_tape = obj_expr.map(|expr| {
            Tape::build_cached_with_externals(&expr, &data.common_exprs, n, &ce_cache, &resolver)
        });

        let obj_linear = if obj_idx < data.obj_linear.len() {
            data.obj_linear[obj_idx].clone()
        } else {
            Vec::new()
        };

        // Build tapes for constraints
        let con_tapes: Vec<Option<Tape>> = data
            .con_exprs
            .iter()
            .map(|expr| {
                expr.as_ref().map(|e| {
                    Tape::build_cached_with_externals(
                        e,
                        &data.common_exprs,
                        n,
                        &ce_cache,
                        &resolver,
                    )
                })
            })
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

        // No diagonal padding: σ_x regularization is added directly to the
        // KKT matrix in kkt.rs (the (1,1) block diagonal), so the NLP-level
        // Hessian only needs structural nonzeros. hessian_sparsity() is
        // exact — padding (v,v) for vars that appear only bilinearly would
        // emit guaranteed-zero entries (e.g. arki0003: 534 such phantoms).
        log::info!("Hessian sparsity: {} structural nonzeros", hess_set.len());

        let mut hess_rows = Vec::with_capacity(hess_set.len());
        let mut hess_cols = Vec::with_capacity(hess_set.len());
        let mut hess_map = HashMap::with_capacity(hess_set.len());
        for (idx, &(r, c)) in hess_set.iter().enumerate() {
            hess_rows.push(r);
            hess_cols.push(c);
            hess_map.insert((r, c), idx);
        }

        Ok(NlProblem {
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
        })
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

/// Collect the set of `Funcall` ids referenced anywhere in the NL data
/// (objective, constraints, common subexpressions).
fn collect_funcall_ids(data: &NlFileData) -> HashSet<usize> {
    let mut ids: HashSet<usize> = HashSet::new();
    if let Some((_, _, Some(expr))) = data.obj_exprs.first() {
        walk_collect_ids(expr, &mut ids);
    }
    for expr in data.con_exprs.iter().flatten() {
        walk_collect_ids(expr, &mut ids);
    }
    for expr in &data.common_exprs {
        walk_collect_ids(expr, &mut ids);
    }
    ids
}

fn walk_collect_ids(expr: &ExprNode, ids: &mut HashSet<usize>) {
    match expr {
        ExprNode::Funcall { id, args } => {
            ids.insert(*id);
            for a in args {
                walk_collect_ids(a, ids);
            }
        }
        ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::StringLiteral(_) => {}
        ExprNode::Binary(_, l, r) => {
            walk_collect_ids(l, ids);
            walk_collect_ids(r, ids);
        }
        ExprNode::Unary(_, a) => walk_collect_ids(a, ids),
        ExprNode::Nary(_, args) => {
            for a in args {
                walk_collect_ids(a, ids);
            }
        }
        ExprNode::If(c, t, e) => {
            walk_collect_ids(c, ids);
            walk_collect_ids(t, ids);
            walk_collect_ids(e, ids);
        }
    }
}

/// Split the `AMPLFUNC` environment variable into candidate library paths.
/// AMPL accepts either newline- or (on UNIX) colon-separated lists.
fn amplfunc_paths() -> Vec<std::path::PathBuf> {
    match std::env::var("AMPLFUNC") {
        Err(_) => Vec::new(),
        Ok(s) => s
            .split(|c: char| c == '\n' || c == ':')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(std::path::PathBuf::from)
            .collect(),
    }
}

/// Build the tape builder's [`ExternalResolver`] by:
///   1. collecting every `Funcall` id referenced in the NL data,
///   2. loading every shared library listed in `AMPLFUNC`,
///   3. for each imported F-segment, looking up a library that registered
///      a function by the declared name.
/// Returns the resolver, or a `String` error describing what's missing.
fn resolve_externals(data: &NlFileData) -> Result<ExternalResolver, String> {
    let used_ids = collect_funcall_ids(data);
    if used_ids.is_empty() {
        return Ok(ExternalResolver::default());
    }

    // Load every library listed in AMPLFUNC. Keep the Arc handles so later
    // lookups succeed and so the libraries stay alive with each tape.
    let lib_paths = amplfunc_paths();
    if lib_paths.is_empty() {
        let first_name = data
            .imported_funcs
            .iter()
            .find(|f: &&ImportedFunc| used_ids.contains(&f.id))
            .map(|f| f.name.clone())
            .unwrap_or_else(|| "<unknown>".to_string());
        return Err(format!(
            "problem uses external function '{first_name}' but AMPLFUNC is not set. \
             Set AMPLFUNC to a newline- or colon-separated list of shared-library paths \
             (e.g. the IDAES Helmholtz extension)."
        ));
    }

    let mut libs: Vec<Arc<ExternalLibrary>> = Vec::with_capacity(lib_paths.len());
    for path in &lib_paths {
        let lib = ExternalLibrary::load(path).map_err(|e| {
            format!("failed to load AMPLFUNC library '{}': {}", path.display(), e)
        })?;
        libs.push(Arc::new(lib));
    }

    // Resolve each referenced F-segment against the loaded libraries.
    let mut funcs_by_id: HashMap<usize, (Arc<ExternalLibrary>, String)> = HashMap::new();
    for id in &used_ids {
        let func = data
            .imported_funcs
            .iter()
            .find(|f: &&ImportedFunc| f.id == *id)
            .ok_or_else(|| {
                format!("problem references Funcall id={id} but no F-segment declares it")
            })?;
        let name = func.name.clone();
        let matching = libs.iter().find(|lib| lib.get(&name).is_some());
        match matching {
            Some(lib) => {
                funcs_by_id.insert(*id, (lib.clone(), name));
            }
            None => {
                let loaded: Vec<String> = lib_paths
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect();
                return Err(format!(
                    "external function '{name}' (F{id}) was not registered by any \
                     AMPLFUNC library; loaded: [{}]",
                    loaded.join(", ")
                ));
            }
        }
    }

    Ok(ExternalResolver { funcs_by_id })
}
