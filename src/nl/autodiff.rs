use std::collections::HashMap;
use std::sync::Arc;

use super::expr::{BinaryOp, ExprNode, NaryOp, UnaryOp};
use super::external::{ExternalArg, ExternalLibrary};

/// One argument to an AMPL external (imported) function call on the tape.
///
/// The AMPL `arglist` ABI preserves the positional order of arguments while
/// routing reals through `ra[]` and strings through `sa[]`. We mirror that:
/// real args are represented as tape indices so their values are fetched
/// from the running `vals[]`, while string args are owned literals.
#[derive(Debug, Clone)]
pub enum FuncallArg {
    /// Index into the tape — real-valued child result.
    Tape(usize),
    /// String literal passed verbatim to the external library.
    Str(String),
}

/// Resolution of NL-declared `ImportedFunc` ids to a live shared library
/// plus the name the library registered under. Produced once in
/// `NlProblem::from_nl_data` and consumed by tape builders.
#[derive(Default, Clone)]
pub struct ExternalResolver {
    /// `Funcall { id }` -> (library, registered function name).
    pub funcs_by_id: HashMap<usize, (Arc<ExternalLibrary>, String)>,
}

impl ExternalResolver {
    pub fn is_empty(&self) -> bool {
        self.funcs_by_id.is_empty()
    }
}

/// A single operation in the flattened tape.
#[derive(Debug, Clone)]
pub enum TapeOp {
    Const(f64),
    Var(usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Pow(usize, usize),
    Mod(usize, usize),
    Atan2(usize, usize),
    Less(usize, usize),
    IntDiv(usize, usize),
    Neg(usize),
    Abs(usize),
    Floor(usize),
    Ceil(usize),
    Sqrt(usize),
    Exp(usize),
    Log(usize),
    Log10(usize),
    Sin(usize),
    Cos(usize),
    Tan(usize),
    Asin(usize),
    Acos(usize),
    Atan(usize),
    Sinh(usize),
    Cosh(usize),
    Tanh(usize),
    Asinh(usize),
    Acosh(usize),
    Atanh(usize),
    /// AMPL imported (external) function call. The library is kept alive
    /// by the Arc; `name` is used to dispatch to the registered `rfunc`.
    Funcall {
        lib: Arc<ExternalLibrary>,
        name: String,
        args: Vec<FuncallArg>,
    },
}

/// Flattened expression tape for efficient forward evaluation and reverse-mode AD.
#[derive(Debug, Clone)]
pub struct Tape {
    pub ops: Vec<TapeOp>,
    pub n_vars: usize,
}

/// Pre-built common expression tapes, keyed by index.
/// Each entry stores the tape ops and the index of the result within those ops.
pub struct CommonExprCache {
    /// For each common expression: (ops, result_index)
    entries: Vec<Option<(Vec<TapeOp>, usize)>>,
}

impl CommonExprCache {
    /// Build a cache of all common expression tapes (no external functions).
    pub fn build(common_exprs: &[ExprNode], n_vars: usize) -> Self {
        Self::build_with_externals(common_exprs, n_vars, &ExternalResolver::default())
    }

    /// Build a cache of all common expression tapes, resolving AMPL external
    /// function calls via `resolver`.
    pub fn build_with_externals(
        common_exprs: &[ExprNode],
        n_vars: usize,
        resolver: &ExternalResolver,
    ) -> Self {
        let mut entries: Vec<Option<(Vec<TapeOp>, usize)>> = Vec::with_capacity(common_exprs.len());
        for i in 0..common_exprs.len() {
            let mut ops = Vec::new();
            let result_idx = build_recursive_cached(
                &common_exprs[i],
                common_exprs,
                n_vars,
                &mut ops,
                &entries,
                resolver,
            );
            entries.push(Some((ops, result_idx)));
        }
        CommonExprCache { entries }
    }
}

impl Tape {
    /// Build a tape from an expression tree (no external functions).
    pub fn build(expr: &ExprNode, common_exprs: &[ExprNode], n_vars: usize) -> Self {
        Self::build_with_externals(expr, common_exprs, n_vars, &ExternalResolver::default())
    }

    /// Build a tape from an expression tree, resolving any AMPL external
    /// function calls via `resolver`.
    pub fn build_with_externals(
        expr: &ExprNode,
        common_exprs: &[ExprNode],
        n_vars: usize,
        resolver: &ExternalResolver,
    ) -> Self {
        let mut ops = Vec::new();
        build_recursive(expr, common_exprs, n_vars, &mut ops, resolver);
        Tape { ops, n_vars }
    }

    /// Build a tape using pre-cached common expressions (no externals).
    pub fn build_cached(
        expr: &ExprNode,
        common_exprs: &[ExprNode],
        n_vars: usize,
        cache: &CommonExprCache,
    ) -> Self {
        Self::build_cached_with_externals(
            expr,
            common_exprs,
            n_vars,
            cache,
            &ExternalResolver::default(),
        )
    }

    /// Build a tape using pre-cached common expressions, resolving AMPL
    /// external function calls via `resolver`.
    pub fn build_cached_with_externals(
        expr: &ExprNode,
        common_exprs: &[ExprNode],
        n_vars: usize,
        cache: &CommonExprCache,
        resolver: &ExternalResolver,
    ) -> Self {
        let mut ops = Vec::new();
        build_recursive_cached(
            expr,
            common_exprs,
            n_vars,
            &mut ops,
            &cache.entries,
            resolver,
        );
        Tape { ops, n_vars }
    }

    /// Forward-evaluate the tape at the given variable values.
    /// Returns the vector of all intermediate values; the last element is the result.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut vals: Vec<f64> = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let v = match op {
                TapeOp::Const(c) => *c,
                TapeOp::Var(i) => x[*i],
                TapeOp::Add(a, b) => vals[*a] + vals[*b],
                TapeOp::Sub(a, b) => vals[*a] - vals[*b],
                TapeOp::Mul(a, b) => vals[*a] * vals[*b],
                TapeOp::Div(a, b) => vals[*a] / vals[*b],
                TapeOp::Pow(a, b) => vals[*a].powf(vals[*b]),
                TapeOp::Mod(a, b) => vals[*a] % vals[*b],
                TapeOp::Atan2(a, b) => vals[*a].atan2(vals[*b]),
                TapeOp::Less(a, b) => {
                    if vals[*a] < vals[*b] {
                        vals[*a]
                    } else {
                        vals[*b]
                    }
                }
                TapeOp::IntDiv(a, b) => (vals[*a] / vals[*b]).floor(),
                TapeOp::Neg(a) => -vals[*a],
                TapeOp::Abs(a) => vals[*a].abs(),
                TapeOp::Floor(a) => vals[*a].floor(),
                TapeOp::Ceil(a) => vals[*a].ceil(),
                TapeOp::Sqrt(a) => vals[*a].sqrt(),
                TapeOp::Exp(a) => vals[*a].exp(),
                TapeOp::Log(a) => vals[*a].ln(),
                TapeOp::Log10(a) => vals[*a].log10(),
                TapeOp::Sin(a) => vals[*a].sin(),
                TapeOp::Cos(a) => vals[*a].cos(),
                TapeOp::Tan(a) => vals[*a].tan(),
                TapeOp::Asin(a) => vals[*a].asin(),
                TapeOp::Acos(a) => vals[*a].acos(),
                TapeOp::Atan(a) => vals[*a].atan(),
                TapeOp::Sinh(a) => vals[*a].sinh(),
                TapeOp::Cosh(a) => vals[*a].cosh(),
                TapeOp::Tanh(a) => vals[*a].tanh(),
                TapeOp::Asinh(a) => vals[*a].asinh(),
                TapeOp::Acosh(a) => vals[*a].acosh(),
                TapeOp::Atanh(a) => vals[*a].atanh(),
                TapeOp::Funcall { lib, name, args } => {
                    let call_args = funcall_ext_args(args, &vals);
                    let res = lib
                        .eval(name, &call_args, false, false)
                        .unwrap_or_else(|e| {
                            panic!("external function '{name}' forward eval failed: {e}")
                        });
                    res.value
                }
            };
            vals.push(v);
        }
        vals
    }

    /// Evaluate the tape and return just the scalar result.
    pub fn eval(&self, x: &[f64]) -> f64 {
        let vals = self.forward(x);
        *vals.last().unwrap_or(&0.0)
    }

    /// Compute the gradient via reverse-mode AD.
    /// `grad` is zeroed and filled with df/dx_i for each problem variable.
    pub fn gradient(&self, x: &[f64], grad: &mut [f64]) {
        let vals = self.forward(x);
        self.reverse(&vals, grad);
    }

    /// Return the sorted set of variable indices that this tape depends on.
    pub fn variables(&self) -> Vec<usize> {
        let mut vars: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for op in &self.ops {
            if let TapeOp::Var(j) = op {
                vars.insert(*j);
            }
        }
        vars.into_iter().collect()
    }

    /// Reverse pass: given forward values, accumulate gradients into `grad`.
    /// `grad` must be pre-zeroed by the caller if a clean gradient is needed.
    pub fn reverse(&self, vals: &[f64], grad: &mut [f64]) {
        let n = self.ops.len();
        if n == 0 {
            return;
        }
        let mut adj = vec![0.0f64; n];
        adj[n - 1] = 1.0;

        for i in (0..n).rev() {
            let a = adj[i];
            if a == 0.0 {
                continue;
            }
            match &self.ops[i] {
                TapeOp::Const(_) => {}
                TapeOp::Var(j) => {
                    if *j < grad.len() {
                        grad[*j] += a;
                    }
                }
                TapeOp::Add(l, r) => {
                    adj[*l] += a;
                    adj[*r] += a;
                }
                TapeOp::Sub(l, r) => {
                    adj[*l] += a;
                    adj[*r] -= a;
                }
                TapeOp::Mul(l, r) => {
                    adj[*l] += a * vals[*r];
                    adj[*r] += a * vals[*l];
                }
                TapeOp::Div(l, r) => {
                    let rv = vals[*r];
                    adj[*l] += a / rv;
                    adj[*r] -= a * vals[*l] / (rv * rv);
                }
                TapeOp::Pow(l, r) => {
                    let lv = vals[*l];
                    let rv = vals[*r];
                    // d/dl: r * l^(r-1)
                    if rv != 0.0 {
                        adj[*l] += a * rv * lv.powf(rv - 1.0);
                    }
                    // d/dr: l^r * ln(l)
                    if lv > 0.0 {
                        adj[*r] += a * vals[i] * lv.ln();
                    }
                }
                TapeOp::Mod(l, _r) => {
                    // d(a%b)/da = 1, d(a%b)/db is discontinuous; approximate as 0
                    adj[*l] += a;
                }
                TapeOp::Atan2(l, r) => {
                    let lv = vals[*l];
                    let rv = vals[*r];
                    let denom = lv * lv + rv * rv;
                    if denom > 0.0 {
                        adj[*l] += a * rv / denom;
                        adj[*r] -= a * lv / denom;
                    }
                }
                TapeOp::Less(l, r) => {
                    // Subgradient: pass through to the min argument
                    if vals[*l] < vals[*r] {
                        adj[*l] += a;
                    } else {
                        adj[*r] += a;
                    }
                }
                TapeOp::IntDiv(l, _r) => {
                    // Floor division: gradient is 0 almost everywhere; pass through
                    adj[*l] += a;
                }
                TapeOp::Neg(j) => {
                    adj[*j] -= a;
                }
                TapeOp::Abs(j) => {
                    if vals[*j] >= 0.0 {
                        adj[*j] += a;
                    } else {
                        adj[*j] -= a;
                    }
                }
                TapeOp::Floor(_) | TapeOp::Ceil(_) => {
                    // Zero gradient (piecewise constant)
                }
                TapeOp::Sqrt(j) => {
                    let sv = vals[i];
                    if sv > 0.0 {
                        adj[*j] += a * 0.5 / sv;
                    }
                }
                TapeOp::Exp(j) => {
                    adj[*j] += a * vals[i]; // d(e^x)/dx = e^x
                }
                TapeOp::Log(j) => {
                    adj[*j] += a / vals[*j];
                }
                TapeOp::Log10(j) => {
                    adj[*j] += a / (vals[*j] * std::f64::consts::LN_10);
                }
                TapeOp::Sin(j) => {
                    adj[*j] += a * vals[*j].cos();
                }
                TapeOp::Cos(j) => {
                    adj[*j] -= a * vals[*j].sin();
                }
                TapeOp::Tan(j) => {
                    let c = vals[*j].cos();
                    adj[*j] += a / (c * c);
                }
                TapeOp::Asin(j) => {
                    adj[*j] += a / (1.0 - vals[*j] * vals[*j]).sqrt();
                }
                TapeOp::Acos(j) => {
                    adj[*j] -= a / (1.0 - vals[*j] * vals[*j]).sqrt();
                }
                TapeOp::Atan(j) => {
                    adj[*j] += a / (1.0 + vals[*j] * vals[*j]);
                }
                TapeOp::Sinh(j) => {
                    adj[*j] += a * vals[*j].cosh();
                }
                TapeOp::Cosh(j) => {
                    adj[*j] += a * vals[*j].sinh();
                }
                TapeOp::Tanh(j) => {
                    let tv = vals[i];
                    adj[*j] += a * (1.0 - tv * tv);
                }
                TapeOp::Asinh(j) => {
                    adj[*j] += a / (vals[*j] * vals[*j] + 1.0).sqrt();
                }
                TapeOp::Acosh(j) => {
                    adj[*j] += a / (vals[*j] * vals[*j] - 1.0).sqrt();
                }
                TapeOp::Atanh(j) => {
                    adj[*j] += a / (1.0 - vals[*j] * vals[*j]);
                }
                TapeOp::Funcall { lib, name, args } => {
                    // Re-enter the library with want_derivs=true to get df/dx_k
                    // for each real arg k (in `ra[]` order, which matches the
                    // order of FuncallArg::Tape entries).
                    let call_args = funcall_ext_args(args, vals);
                    let res = lib
                        .eval(name, &call_args, true, false)
                        .unwrap_or_else(|e| {
                            panic!("external function '{name}' reverse eval failed: {e}")
                        });
                    let derivs = res.derivs.expect("want_derivs=true returns derivs");
                    let mut k = 0usize;
                    for arg in args {
                        if let FuncallArg::Tape(idx) = arg {
                            adj[*idx] += a * derivs[k];
                            k += 1;
                        }
                    }
                }
            }
        }
    }

    /// Forward tangent sweep: compute dot[i] = d(v_i)/d(x_seed) for all tape nodes.
    fn forward_tangent(&self, vals: &[f64], seed_var: usize) -> Vec<f64> {
        let n = self.ops.len();
        let mut dot = vec![0.0f64; n];
        for i in 0..n {
            dot[i] = match &self.ops[i] {
                TapeOp::Const(_) => 0.0,
                TapeOp::Var(k) => if *k == seed_var { 1.0 } else { 0.0 },
                TapeOp::Add(a, b) => dot[*a] + dot[*b],
                TapeOp::Sub(a, b) => dot[*a] - dot[*b],
                TapeOp::Mul(a, b) => dot[*a] * vals[*b] + vals[*a] * dot[*b],
                TapeOp::Div(a, b) => {
                    let vb = vals[*b];
                    (dot[*a] * vb - vals[*a] * dot[*b]) / (vb * vb)
                }
                TapeOp::Pow(a, b) => {
                    let u = vals[*a];
                    let r = vals[*b];
                    let du = dot[*a];
                    let dr = dot[*b];
                    let mut result = 0.0;
                    if r != 0.0 && u != 0.0 {
                        result += r * u.powf(r - 1.0) * du;
                    }
                    if u > 0.0 {
                        result += vals[i] * u.ln() * dr;
                    }
                    result
                }
                TapeOp::Mod(a, _) => dot[*a],
                TapeOp::Atan2(a, b) => {
                    let y = vals[*a]; let xv = vals[*b];
                    let d = y * y + xv * xv;
                    if d > 0.0 { (dot[*a] * xv - y * dot[*b]) / d } else { 0.0 }
                }
                TapeOp::Less(a, b) => if vals[*a] < vals[*b] { dot[*a] } else { dot[*b] },
                TapeOp::IntDiv(a, _) => dot[*a],
                TapeOp::Neg(a) => -dot[*a],
                TapeOp::Abs(a) => if vals[*a] >= 0.0 { dot[*a] } else { -dot[*a] },
                TapeOp::Floor(_) | TapeOp::Ceil(_) => 0.0,
                TapeOp::Sqrt(a) => {
                    let sv = vals[i];
                    if sv > 0.0 { dot[*a] * 0.5 / sv } else { 0.0 }
                }
                TapeOp::Exp(a) => dot[*a] * vals[i],
                TapeOp::Log(a) => dot[*a] / vals[*a],
                TapeOp::Log10(a) => dot[*a] / (vals[*a] * std::f64::consts::LN_10),
                TapeOp::Sin(a) => dot[*a] * vals[*a].cos(),
                TapeOp::Cos(a) => -dot[*a] * vals[*a].sin(),
                TapeOp::Tan(a) => { let c = vals[*a].cos(); dot[*a] / (c * c) }
                TapeOp::Asin(a) => dot[*a] / (1.0 - vals[*a] * vals[*a]).sqrt(),
                TapeOp::Acos(a) => -dot[*a] / (1.0 - vals[*a] * vals[*a]).sqrt(),
                TapeOp::Atan(a) => dot[*a] / (1.0 + vals[*a] * vals[*a]),
                TapeOp::Sinh(a) => dot[*a] * vals[*a].cosh(),
                TapeOp::Cosh(a) => dot[*a] * vals[*a].sinh(),
                TapeOp::Tanh(a) => { let tv = vals[i]; dot[*a] * (1.0 - tv * tv) }
                TapeOp::Asinh(a) => dot[*a] / (vals[*a] * vals[*a] + 1.0).sqrt(),
                TapeOp::Acosh(a) => dot[*a] / (vals[*a] * vals[*a] - 1.0).sqrt(),
                TapeOp::Atanh(a) => dot[*a] / (1.0 - vals[*a] * vals[*a]),
                TapeOp::Funcall { lib, name, args } => {
                    // dot[i] = sum_k (df/dx_k) * dot[arg_k_tape_idx]
                    let call_args = funcall_ext_args(args, vals);
                    let res = lib
                        .eval(name, &call_args, true, false)
                        .unwrap_or_else(|e| {
                            panic!("external function '{name}' tangent eval failed: {e}")
                        });
                    let derivs = res.derivs.expect("want_derivs=true returns derivs");
                    let mut acc = 0.0;
                    let mut k = 0usize;
                    for arg in args {
                        if let FuncallArg::Tape(idx) = arg {
                            acc += derivs[k] * dot[*idx];
                            k += 1;
                        }
                    }
                    acc
                }
            };
        }
        dot
    }

    /// Compute Hessian via forward-over-reverse AD and accumulate weighted entries.
    ///
    /// For each variable j in this tape, does a forward tangent sweep (seed e_j),
    /// then a reverse second-order sweep to get column j of the Hessian.
    /// Results are accumulated as `vals[pos] += weight * H[i][j]` using `hess_map`.
    pub fn hessian_accumulate(
        &self,
        x: &[f64],
        weight: f64,
        hess_map: &std::collections::HashMap<(usize, usize), usize>,
        vals: &mut [f64],
    ) {
        let n = self.ops.len();
        if n == 0 || weight == 0.0 {
            return;
        }

        let v = self.forward(x);
        let var_indices = self.variables();

        for &j in &var_indices {
            let dot = self.forward_tangent(&v, j);

            // Second-order reverse sweep:
            // adj[i] = standard adjoint, adj_dot[i] = second-order adjoint
            let mut adj = vec![0.0f64; n];
            let mut adj_dot = vec![0.0f64; n];
            adj[n - 1] = 1.0;
            // adj_dot[n-1] = 0 (no seed for second order)

            for i in (0..n).rev() {
                let w = adj[i];
                let wd = adj_dot[i];
                if w == 0.0 && wd == 0.0 {
                    continue;
                }
                match &self.ops[i] {
                    TapeOp::Const(_) => {}
                    TapeOp::Var(k) => {
                        // Accumulate H[k][j] (second-order adjoint).
                        // Only accumulate lower triangle entries where k >= j
                        // to avoid double-counting off-diagonal entries.
                        if wd != 0.0 && *k >= j {
                            if let Some(&pos) = hess_map.get(&(*k, j)) {
                                vals[pos] += weight * wd;
                            }
                        }
                    }
                    // --- Linear ops: partials are constant, no second-order contribution ---
                    TapeOp::Add(a, b) => {
                        adj[*a] += w; adj[*b] += w;
                        adj_dot[*a] += wd; adj_dot[*b] += wd;
                    }
                    TapeOp::Sub(a, b) => {
                        adj[*a] += w; adj[*b] -= w;
                        adj_dot[*a] += wd; adj_dot[*b] -= wd;
                    }
                    // --- Mul: p_a = v[b], dp_a = dot[b]; p_b = v[a], dp_b = dot[a] ---
                    TapeOp::Mul(a, b) => {
                        adj[*a] += w * v[*b];
                        adj[*b] += w * v[*a];
                        adj_dot[*a] += wd * v[*b] + w * dot[*b];
                        adj_dot[*b] += wd * v[*a] + w * dot[*a];
                    }
                    // --- Div: v[i] = v[a]/v[b] ---
                    TapeOp::Div(a, b) => {
                        let vb = v[*b];
                        let vb2 = vb * vb;
                        let vb3 = vb2 * vb;
                        // p_a = 1/vb, dp_a = -dot[b]/vb^2
                        adj[*a] += w / vb;
                        adj_dot[*a] += wd / vb + w * (-dot[*b] / vb2);
                        // p_b = -v[a]/vb^2, dp_b = -dot[a]/vb^2 + 2*v[a]*dot[b]/vb^3
                        adj[*b] += w * (-v[*a] / vb2);
                        adj_dot[*b] += wd * (-v[*a] / vb2)
                            + w * (-dot[*a] / vb2 + 2.0 * v[*a] * dot[*b] / vb3);
                    }
                    // --- Pow: v[i] = v[a]^v[b] ---
                    TapeOp::Pow(a, b) => {
                        let u = v[*a];
                        let r = v[*b];
                        let du = dot[*a];
                        let dr = dot[*b];
                        // p_a = r * u^(r-1), dp_a = d(p_a)/dx_j
                        if r != 0.0 {
                            if u != 0.0 {
                                let p_a = r * u.powf(r - 1.0);
                                adj[*a] += w * p_a;
                                let mut dp_a = dr * u.powf(r - 1.0);
                                if u > 0.0 {
                                    dp_a += r * u.powf(r - 1.0) * ((r - 1.0) * du / u + dr * u.ln());
                                } else {
                                    dp_a += r * (r - 1.0) * u.powf(r - 2.0) * du;
                                }
                                adj_dot[*a] += wd * p_a + w * dp_a;
                            } else if r >= 2.0 {
                                // u == 0: p_a = r * 0^(r-1) = 0 for r>1, but
                                // dp_a = r*(r-1)*0^(r-2)*du which is nonzero when r=2
                                let p_a = 0.0; // r * 0^(r-1) = 0 for r > 1
                                adj[*a] += w * p_a;
                                let dp_a = if r == 2.0 {
                                    2.0 * du // 2 * 1 * 0^0 * du = 2*du
                                } else {
                                    r * (r - 1.0) * (0.0_f64).powf(r - 2.0) * du
                                };
                                adj_dot[*a] += wd * p_a + w * dp_a;
                            }
                        }
                        // p_b = v[i] * ln(u)
                        if u > 0.0 {
                            let ln_u = u.ln();
                            let p_b = v[i] * ln_u;
                            adj[*b] += w * p_b;
                            let dur = v[i] * (r * du / u + dr * ln_u);
                            let dp_b = dur * ln_u + v[i] * du / u;
                            adj_dot[*b] += wd * p_b + w * dp_b;
                        }
                    }
                    TapeOp::Atan2(a, b) => {
                        let y = v[*a]; let xv = v[*b];
                        let d = y * y + xv * xv;
                        if d > 0.0 {
                            let d2 = d * d;
                            let dy = dot[*a]; let dx = dot[*b];
                            let dd = 2.0 * y * dy + 2.0 * xv * dx;
                            // p_a = xv/d
                            adj[*a] += w * xv / d;
                            let dp_a = (dx * d - xv * dd) / d2;
                            adj_dot[*a] += wd * xv / d + w * dp_a;
                            // p_b = -y/d
                            adj[*b] += w * (-y / d);
                            let dp_b = (-dy * d + y * dd) / d2;
                            adj_dot[*b] += wd * (-y / d) + w * dp_b;
                        }
                    }
                    TapeOp::Mod(a, _) | TapeOp::IntDiv(a, _) => {
                        adj[*a] += w;
                        adj_dot[*a] += wd;
                    }
                    TapeOp::Less(a, b) => {
                        if v[*a] < v[*b] {
                            adj[*a] += w; adj_dot[*a] += wd;
                        } else {
                            adj[*b] += w; adj_dot[*b] += wd;
                        }
                    }
                    // --- Unary ops: adj_dot[a] += wd * f'(u) + w * f''(u) * dot[a] ---
                    TapeOp::Neg(a) => {
                        adj[*a] -= w;
                        adj_dot[*a] -= wd;
                    }
                    TapeOp::Abs(a) => {
                        let s = if v[*a] >= 0.0 { 1.0 } else { -1.0 };
                        adj[*a] += w * s;
                        adj_dot[*a] += wd * s; // f'' = 0
                    }
                    TapeOp::Floor(_) | TapeOp::Ceil(_) => {} // f' = 0
                    TapeOp::Sqrt(a) => {
                        let sv = v[i];
                        if sv > 0.0 {
                            let fp = 0.5 / sv; // f' = 1/(2*sqrt(u))
                            let fpp = -0.25 / (v[*a] * sv); // f'' = -1/(4*u^(3/2))
                            adj[*a] += w * fp;
                            adj_dot[*a] += wd * fp + w * fpp * dot[*a];
                        }
                    }
                    TapeOp::Exp(a) => {
                        let ev = v[i]; // exp(u)
                        adj[*a] += w * ev;
                        adj_dot[*a] += wd * ev + w * ev * dot[*a]; // f'' = exp(u)
                    }
                    TapeOp::Log(a) => {
                        let u = v[*a];
                        adj[*a] += w / u;
                        adj_dot[*a] += wd / u + w * (-1.0 / (u * u)) * dot[*a];
                    }
                    TapeOp::Log10(a) => {
                        let u = v[*a];
                        let ln10 = std::f64::consts::LN_10;
                        adj[*a] += w / (u * ln10);
                        adj_dot[*a] += wd / (u * ln10) + w * (-1.0 / (u * u * ln10)) * dot[*a];
                    }
                    TapeOp::Sin(a) => {
                        let u = v[*a];
                        let cu = u.cos();
                        adj[*a] += w * cu;
                        adj_dot[*a] += wd * cu + w * (-u.sin()) * dot[*a];
                    }
                    TapeOp::Cos(a) => {
                        let u = v[*a];
                        let su = u.sin();
                        adj[*a] -= w * su;
                        adj_dot[*a] += wd * (-su) + w * (-u.cos()) * dot[*a];
                    }
                    TapeOp::Tan(a) => {
                        let u = v[*a];
                        let c = u.cos();
                        let sec2 = 1.0 / (c * c);
                        let t = u.tan();
                        adj[*a] += w * sec2;
                        adj_dot[*a] += wd * sec2 + w * 2.0 * t * sec2 * dot[*a];
                    }
                    TapeOp::Asin(a) => {
                        let u = v[*a];
                        let s = (1.0 - u * u).sqrt();
                        adj[*a] += w / s;
                        adj_dot[*a] += wd / s + w * (u / (s * s * s)) * dot[*a];
                    }
                    TapeOp::Acos(a) => {
                        let u = v[*a];
                        let s = (1.0 - u * u).sqrt();
                        adj[*a] -= w / s;
                        adj_dot[*a] += wd * (-1.0 / s) + w * (-u / (s * s * s)) * dot[*a];
                    }
                    TapeOp::Atan(a) => {
                        let u = v[*a];
                        let d = 1.0 + u * u;
                        adj[*a] += w / d;
                        adj_dot[*a] += wd / d + w * (-2.0 * u / (d * d)) * dot[*a];
                    }
                    TapeOp::Sinh(a) => {
                        let u = v[*a];
                        let ch = u.cosh();
                        adj[*a] += w * ch;
                        adj_dot[*a] += wd * ch + w * u.sinh() * dot[*a];
                    }
                    TapeOp::Cosh(a) => {
                        let u = v[*a];
                        let sh = u.sinh();
                        adj[*a] += w * sh;
                        adj_dot[*a] += wd * sh + w * u.cosh() * dot[*a];
                    }
                    TapeOp::Tanh(a) => {
                        let tv = v[i]; // tanh(u)
                        let sech2 = 1.0 - tv * tv;
                        adj[*a] += w * sech2;
                        adj_dot[*a] += wd * sech2 + w * (-2.0 * tv * sech2) * dot[*a];
                    }
                    TapeOp::Asinh(a) => {
                        let u = v[*a];
                        let s = (u * u + 1.0).sqrt();
                        adj[*a] += w / s;
                        adj_dot[*a] += wd / s + w * (-u / (s * s * s)) * dot[*a];
                    }
                    TapeOp::Acosh(a) => {
                        let u = v[*a];
                        let s = (u * u - 1.0).sqrt();
                        adj[*a] += w / s;
                        adj_dot[*a] += wd / s + w * (-u / (s * s * s)) * dot[*a];
                    }
                    TapeOp::Atanh(a) => {
                        let u = v[*a];
                        let d = 1.0 - u * u;
                        adj[*a] += w / d;
                        adj_dot[*a] += wd / d + w * (2.0 * u / (d * d)) * dot[*a];
                    }
                    TapeOp::Funcall { lib, name, args } => {
                        // Full second-order propagation through an external
                        // function. Treat F: R^nr -> R; let p_k = dF/dra[k]
                        // and H_kl = d^2F/dra[k]/dra[l] (packed upper-tri).
                        //
                        //   adj[ti[k]]     += w * p_k
                        //   adj_dot[ti[k]] += wd * p_k
                        //                    + w * sum_l ( H_kl * dot[ti[l]] )
                        let call_args = funcall_ext_args(args, &v);
                        let res = lib
                            .eval(name, &call_args, true, true)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "external function '{name}' 2nd-order eval failed: {e}"
                                )
                            });
                        let derivs =
                            res.derivs.expect("want_derivs=true returns derivs");
                        let hes = res.hessian.expect("want_hes=true returns hessian");

                        // Collect the tape indices of real args (in ra[] order).
                        let real_tape: Vec<usize> = args
                            .iter()
                            .filter_map(|a| match a {
                                FuncallArg::Tape(t) => Some(*t),
                                FuncallArg::Str(_) => None,
                            })
                            .collect();

                        for (k, &tk) in real_tape.iter().enumerate() {
                            adj[tk] += w * derivs[k];
                            let mut second_term = 0.0;
                            for (l, &tl) in real_tape.iter().enumerate() {
                                let (lo, hi) = if k <= l { (k, l) } else { (l, k) };
                                let h_kl = hes[lo + hi * (hi + 1) / 2];
                                second_term += h_kl * dot[tl];
                            }
                            adj_dot[tk] += wd * derivs[k] + w * second_term;
                        }
                    }
                }
            }
        }
    }

    /// Compute the exact structural Hessian sparsity for this tape.
    ///
    /// Propagates variable dependency sets forward through the tape, then at each
    /// nonlinear op, emits the cross products of variable sets from its children.
    /// Returns the set of (row, col) pairs (lower triangle, row >= col) that have
    /// structurally nonzero second derivatives.
    ///
    /// **Needed-set optimization.** Naively propagating var_sets through every
    /// tape op gives O(n²) time/memory on chained-sum objectives like
    /// `sum_i f(x_i)`: each successive Add in the outer reduction tree clones a
    /// linearly-growing dependency BTreeSet, even though no nonlinear ancestor
    /// ever consumes the union. The needed mask below identifies the tape
    /// nodes whose var_set is actually read — by some nonlinear emission point
    /// directly, or transitively through linear ancestors — and skips var_set
    /// materialization elsewhere. For bearing_400 (n=160k, separable
    /// `sum_i f(x_i)`) this collapses the >180 s, OOM-killed setup down to
    /// proportional to the union of inner cones.
    pub fn hessian_sparsity(&self) -> std::collections::BTreeSet<(usize, usize)> {
        use std::collections::BTreeSet;

        let n = self.ops.len();
        let mut hess_pairs: BTreeSet<(usize, usize)> = BTreeSet::new();

        // ---- Phase 1: mark which nodes' var_sets we actually need ----
        // A node N is `needed` iff:
        //   (a) N is a direct input to a nonlinear emission op (Mul/Div/Pow/
        //       Atan2/sqrt/exp/.../Funcall) — the emission reads var_set[N];
        //   (b) some parent op P propagates var_set[N] into a needed
        //       var_set[P] (linear ops Add/Sub/Neg/Abs/Mod/IntDiv/Less, or
        //       nonlinear ops whose own var_set has a needed consumer).
        // Floor/Ceil/Const have empty var_sets and read nothing.
        //
        // Tape is topologically sorted (children before parents); iterating
        // in reverse visits each parent before its children, so we can
        // propagate "needed" downward in a single pass.
        let mut needed = vec![false; n];
        for i in (0..n).rev() {
            let propagate = needed[i];
            match &self.ops[i] {
                // Nonlinear binary: children's var_sets always needed for emission.
                TapeOp::Mul(a, b) | TapeOp::Div(a, b)
                | TapeOp::Pow(a, b) | TapeOp::Atan2(a, b) => {
                    needed[*a] = true;
                    needed[*b] = true;
                }
                // Nonlinear unary.
                TapeOp::Sqrt(a) | TapeOp::Exp(a) | TapeOp::Log(a) | TapeOp::Log10(a)
                | TapeOp::Sin(a) | TapeOp::Cos(a) | TapeOp::Tan(a)
                | TapeOp::Asin(a) | TapeOp::Acos(a) | TapeOp::Atan(a)
                | TapeOp::Sinh(a) | TapeOp::Cosh(a) | TapeOp::Tanh(a)
                | TapeOp::Asinh(a) | TapeOp::Acosh(a) | TapeOp::Atanh(a) => {
                    needed[*a] = true;
                }
                TapeOp::Funcall { args, .. } => {
                    for arg in args {
                        if let FuncallArg::Tape(t) = arg {
                            needed[*t] = true;
                        }
                    }
                }
                // Linear ops: only propagate if this node's var_set is itself needed.
                TapeOp::Add(a, b) | TapeOp::Sub(a, b)
                | TapeOp::Mod(a, b) | TapeOp::IntDiv(a, b) | TapeOp::Less(a, b) => {
                    if propagate {
                        needed[*a] = true;
                        needed[*b] = true;
                    }
                }
                TapeOp::Neg(a) | TapeOp::Abs(a) => {
                    if propagate {
                        needed[*a] = true;
                    }
                }
                TapeOp::Floor(_) | TapeOp::Ceil(_) | TapeOp::Const(_) | TapeOp::Var(_) => {}
            }
        }

        // ---- Phase 2: forward pass, materializing var_sets only when needed,
        //              emitting Hessian pairs at every nonlinear op. ----
        let mut var_sets: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        // Sentinel reused for ops whose var_set is not needed.
        let empty: BTreeSet<usize> = BTreeSet::new();

        let emit_cross = |s1: &BTreeSet<usize>, s2: &BTreeSet<usize>, pairs: &mut BTreeSet<(usize, usize)>| {
            for &v1 in s1 {
                for &v2 in s2 {
                    let (r, c) = if v1 >= v2 { (v1, v2) } else { (v2, v1) };
                    pairs.insert((r, c));
                }
            }
        };
        let emit_self = |s: &BTreeSet<usize>, pairs: &mut BTreeSet<(usize, usize)>| {
            let vars: Vec<usize> = s.iter().copied().collect();
            for (ai, &vi) in vars.iter().enumerate() {
                for &vj in &vars[..=ai] {
                    let (r, c) = if vi >= vj { (vi, vj) } else { (vj, vi) };
                    pairs.insert((r, c));
                }
            }
        };

        for i in 0..n {
            match &self.ops[i] {
                TapeOp::Const(_) => { /* empty */ }
                TapeOp::Var(j) => {
                    if needed[i] {
                        var_sets[i].insert(*j);
                    }
                }
                TapeOp::Add(a, b) | TapeOp::Sub(a, b)
                | TapeOp::Mod(a, b) | TapeOp::IntDiv(a, b) | TapeOp::Less(a, b) => {
                    if needed[i] {
                        var_sets[i] = var_sets[*a].union(&var_sets[*b]).copied().collect();
                    }
                }
                TapeOp::Neg(a) | TapeOp::Abs(a) => {
                    if needed[i] {
                        var_sets[i] = var_sets[*a].clone();
                    }
                }
                TapeOp::Floor(_) | TapeOp::Ceil(_) => { /* empty */ }
                TapeOp::Mul(a, b) => {
                    emit_cross(&var_sets[*a], &var_sets[*b], &mut hess_pairs);
                    if needed[i] {
                        var_sets[i] = var_sets[*a].union(&var_sets[*b]).copied().collect();
                    }
                }
                TapeOp::Div(a, b) => {
                    emit_cross(&var_sets[*a], &var_sets[*b], &mut hess_pairs);
                    emit_self(&var_sets[*b], &mut hess_pairs);
                    if needed[i] {
                        var_sets[i] = var_sets[*a].union(&var_sets[*b]).copied().collect();
                    }
                }
                TapeOp::Pow(a, b) | TapeOp::Atan2(a, b) => {
                    let combined: BTreeSet<usize> =
                        var_sets[*a].union(&var_sets[*b]).copied().collect();
                    emit_self(&combined, &mut hess_pairs);
                    if needed[i] {
                        var_sets[i] = combined;
                    }
                }
                TapeOp::Sqrt(a) | TapeOp::Exp(a) | TapeOp::Log(a) | TapeOp::Log10(a)
                | TapeOp::Sin(a) | TapeOp::Cos(a) | TapeOp::Tan(a)
                | TapeOp::Asin(a) | TapeOp::Acos(a) | TapeOp::Atan(a)
                | TapeOp::Sinh(a) | TapeOp::Cosh(a) | TapeOp::Tanh(a)
                | TapeOp::Asinh(a) | TapeOp::Acosh(a) | TapeOp::Atanh(a) => {
                    emit_self(&var_sets[*a], &mut hess_pairs);
                    if needed[i] {
                        var_sets[i] = var_sets[*a].clone();
                    }
                }
                TapeOp::Funcall { args, .. } => {
                    let mut combined: BTreeSet<usize> = BTreeSet::new();
                    for arg in args {
                        if let FuncallArg::Tape(t) = arg {
                            for &v in &var_sets[*t] {
                                combined.insert(v);
                            }
                        }
                    }
                    emit_self(&combined, &mut hess_pairs);
                    if needed[i] {
                        var_sets[i] = combined;
                    }
                }
            }
        }
        // Suppress unused warning when no branch references `empty` (kept for
        // future passes that may want an explicit zero-set sentinel).
        let _ = &empty;

        hess_pairs
    }
}

/// Build the `ExternalArg` slice passed to `ExternalLibrary::eval`, reading
/// current real-arg values from the running tape `vals[]`. The positional
/// ordering (mixed real/string) is preserved exactly.
fn funcall_ext_args<'a>(args: &'a [FuncallArg], vals: &[f64]) -> Vec<ExternalArg<'a>> {
    args.iter()
        .map(|a| match a {
            FuncallArg::Tape(t) => ExternalArg::Real(vals[*t]),
            FuncallArg::Str(s) => ExternalArg::Str(s.as_str()),
        })
        .collect()
}

/// Recursively flatten an expression tree into tape operations.
/// Returns the index of the result in the tape.
fn build_recursive(
    expr: &ExprNode,
    common_exprs: &[ExprNode],
    n_vars: usize,
    ops: &mut Vec<TapeOp>,
    resolver: &ExternalResolver,
) -> usize {
    match expr {
        ExprNode::Const(c) => {
            let idx = ops.len();
            ops.push(TapeOp::Const(*c));
            idx
        }
        ExprNode::Var(i) => {
            if *i < n_vars {
                let idx = ops.len();
                ops.push(TapeOp::Var(*i));
                idx
            } else {
                // Common sub-expression: inline it
                let ce_idx = *i - n_vars;
                if ce_idx < common_exprs.len() {
                    build_recursive(&common_exprs[ce_idx], common_exprs, n_vars, ops, resolver)
                } else {
                    // Missing common expr, treat as 0
                    let idx = ops.len();
                    ops.push(TapeOp::Const(0.0));
                    idx
                }
            }
        }
        ExprNode::Binary(op, left, right) => {
            let l = build_recursive(left, common_exprs, n_vars, ops, resolver);
            let r = build_recursive(right, common_exprs, n_vars, ops, resolver);
            let idx = ops.len();
            ops.push(match op {
                BinaryOp::Add => TapeOp::Add(l, r),
                BinaryOp::Sub => TapeOp::Sub(l, r),
                BinaryOp::Mul => TapeOp::Mul(l, r),
                BinaryOp::Div => TapeOp::Div(l, r),
                BinaryOp::Mod => TapeOp::Mod(l, r),
                BinaryOp::Pow => TapeOp::Pow(l, r),
                BinaryOp::Atan2 => TapeOp::Atan2(l, r),
                BinaryOp::Less => TapeOp::Less(l, r),
                BinaryOp::IntDiv => TapeOp::IntDiv(l, r),
            });
            idx
        }
        ExprNode::Unary(op, arg) => {
            let a = build_recursive(arg, common_exprs, n_vars, ops, resolver);
            let idx = ops.len();
            ops.push(match op {
                UnaryOp::Abs => TapeOp::Abs(a),
                UnaryOp::Neg => TapeOp::Neg(a),
                UnaryOp::Floor => TapeOp::Floor(a),
                UnaryOp::Ceil => TapeOp::Ceil(a),
                UnaryOp::Tanh => TapeOp::Tanh(a),
                UnaryOp::Tan => TapeOp::Tan(a),
                UnaryOp::Sqrt => TapeOp::Sqrt(a),
                UnaryOp::Sinh => TapeOp::Sinh(a),
                UnaryOp::Sin => TapeOp::Sin(a),
                UnaryOp::Log10 => TapeOp::Log10(a),
                UnaryOp::Log => TapeOp::Log(a),
                UnaryOp::Exp => TapeOp::Exp(a),
                UnaryOp::Cosh => TapeOp::Cosh(a),
                UnaryOp::Cos => TapeOp::Cos(a),
                UnaryOp::Atanh => TapeOp::Atanh(a),
                UnaryOp::Atan => TapeOp::Atan(a),
                UnaryOp::Asinh => TapeOp::Asinh(a),
                UnaryOp::Asin => TapeOp::Asin(a),
                UnaryOp::Acosh => TapeOp::Acosh(a),
                UnaryOp::Acos => TapeOp::Acos(a),
            });
            idx
        }
        ExprNode::Nary(op, args) => {
            if args.is_empty() {
                let idx = ops.len();
                ops.push(TapeOp::Const(match op {
                    NaryOp::Sum => 0.0,
                    NaryOp::Min => f64::INFINITY,
                    NaryOp::Max => f64::NEG_INFINITY,
                }));
                return idx;
            }
            // For Sum, chain binary adds. For Min/Max, use Less or binary comparisons.
            let mut acc = build_recursive(&args[0], common_exprs, n_vars, ops, resolver);
            for arg in &args[1..] {
                let next = build_recursive(arg, common_exprs, n_vars, ops, resolver);
                match op {
                    NaryOp::Sum => {
                        let idx = ops.len();
                        ops.push(TapeOp::Add(acc, next));
                        acc = idx;
                    }
                    NaryOp::Min => {
                        let idx = ops.len();
                        ops.push(TapeOp::Less(acc, next));
                        acc = idx;
                    }
                    NaryOp::Max => {
                        // max(a, b) = -(min(-a, -b))
                        let neg_acc_idx = ops.len();
                        ops.push(TapeOp::Neg(acc));
                        let neg_next_idx = ops.len();
                        ops.push(TapeOp::Neg(next));
                        let min_idx = ops.len();
                        ops.push(TapeOp::Less(neg_acc_idx, neg_next_idx));
                        let result_idx = ops.len();
                        ops.push(TapeOp::Neg(min_idx));
                        acc = result_idx;
                    }
                }
            }
            acc
        }
        ExprNode::If(cond, then_expr, else_expr) => {
            // If-then-else: approximate as "then" branch for AD; most NLP
            // problems don't use it meaningfully.
            let _c = build_recursive(cond, common_exprs, n_vars, ops, resolver);
            let t = build_recursive(then_expr, common_exprs, n_vars, ops, resolver);
            let _e = build_recursive(else_expr, common_exprs, n_vars, ops, resolver);
            t
        }
        ExprNode::StringLiteral(_) => {
            let _idx = ops.len();
            ops.push(TapeOp::Const(0.0));
            _idx
        }
        ExprNode::Funcall { id, args } => {
            let (lib, name) = resolver.funcs_by_id.get(id).cloned().unwrap_or_else(|| {
                panic!(
                    "build_recursive: no ExternalResolver entry for Funcall id={id}; \
                     NlProblem::from_nl_data should have loaded this library"
                )
            });
            // Positional args: strings go through verbatim; every other
            // ExprNode is built recursively and stored as a tape index.
            let built_args: Vec<FuncallArg> = args
                .iter()
                .map(|a| match a {
                    ExprNode::StringLiteral(s) => FuncallArg::Str(s.clone()),
                    other => FuncallArg::Tape(build_recursive(
                        other,
                        common_exprs,
                        n_vars,
                        ops,
                        resolver,
                    )),
                })
                .collect();
            let idx = ops.len();
            ops.push(TapeOp::Funcall {
                lib,
                name,
                args: built_args,
            });
            idx
        }
    }
}

/// Remap a TapeOp's internal indices by adding `offset` to all tape-internal references.
fn remap_op(op: &TapeOp, offset: usize) -> TapeOp {
    match op {
        TapeOp::Const(c) => TapeOp::Const(*c),
        TapeOp::Var(i) => TapeOp::Var(*i), // Var indices are problem variables, not tape indices
        TapeOp::Add(a, b) => TapeOp::Add(a + offset, b + offset),
        TapeOp::Sub(a, b) => TapeOp::Sub(a + offset, b + offset),
        TapeOp::Mul(a, b) => TapeOp::Mul(a + offset, b + offset),
        TapeOp::Div(a, b) => TapeOp::Div(a + offset, b + offset),
        TapeOp::Pow(a, b) => TapeOp::Pow(a + offset, b + offset),
        TapeOp::Mod(a, b) => TapeOp::Mod(a + offset, b + offset),
        TapeOp::Atan2(a, b) => TapeOp::Atan2(a + offset, b + offset),
        TapeOp::Less(a, b) => TapeOp::Less(a + offset, b + offset),
        TapeOp::IntDiv(a, b) => TapeOp::IntDiv(a + offset, b + offset),
        TapeOp::Neg(a) => TapeOp::Neg(a + offset),
        TapeOp::Abs(a) => TapeOp::Abs(a + offset),
        TapeOp::Floor(a) => TapeOp::Floor(a + offset),
        TapeOp::Ceil(a) => TapeOp::Ceil(a + offset),
        TapeOp::Sqrt(a) => TapeOp::Sqrt(a + offset),
        TapeOp::Exp(a) => TapeOp::Exp(a + offset),
        TapeOp::Log(a) => TapeOp::Log(a + offset),
        TapeOp::Log10(a) => TapeOp::Log10(a + offset),
        TapeOp::Sin(a) => TapeOp::Sin(a + offset),
        TapeOp::Cos(a) => TapeOp::Cos(a + offset),
        TapeOp::Tan(a) => TapeOp::Tan(a + offset),
        TapeOp::Asin(a) => TapeOp::Asin(a + offset),
        TapeOp::Acos(a) => TapeOp::Acos(a + offset),
        TapeOp::Atan(a) => TapeOp::Atan(a + offset),
        TapeOp::Sinh(a) => TapeOp::Sinh(a + offset),
        TapeOp::Cosh(a) => TapeOp::Cosh(a + offset),
        TapeOp::Tanh(a) => TapeOp::Tanh(a + offset),
        TapeOp::Asinh(a) => TapeOp::Asinh(a + offset),
        TapeOp::Acosh(a) => TapeOp::Acosh(a + offset),
        TapeOp::Atanh(a) => TapeOp::Atanh(a + offset),
        TapeOp::Funcall { lib, name, args } => TapeOp::Funcall {
            lib: lib.clone(),
            name: name.clone(),
            args: args
                .iter()
                .map(|a| match a {
                    FuncallArg::Tape(t) => FuncallArg::Tape(t + offset),
                    FuncallArg::Str(s) => FuncallArg::Str(s.clone()),
                })
                .collect(),
        },
    }
}

/// Like build_recursive, but uses pre-built common expression tapes
/// to avoid exponential blowup from repeated inlining.
fn build_recursive_cached(
    expr: &ExprNode,
    common_exprs: &[ExprNode],
    n_vars: usize,
    ops: &mut Vec<TapeOp>,
    cache: &[Option<(Vec<TapeOp>, usize)>],
    resolver: &ExternalResolver,
) -> usize {
    match expr {
        ExprNode::Const(c) => {
            let idx = ops.len();
            ops.push(TapeOp::Const(*c));
            idx
        }
        ExprNode::Var(i) => {
            if *i < n_vars {
                let idx = ops.len();
                ops.push(TapeOp::Var(*i));
                idx
            } else {
                let ce_idx = *i - n_vars;
                if ce_idx < cache.len() {
                    if let Some((ce_ops, ce_result)) = &cache[ce_idx] {
                        // Embed the cached tape ops with remapped indices
                        let offset = ops.len();
                        for op in ce_ops {
                            ops.push(remap_op(op, offset));
                        }
                        offset + ce_result
                    } else {
                        let idx = ops.len();
                        ops.push(TapeOp::Const(0.0));
                        idx
                    }
                } else {
                    let idx = ops.len();
                    ops.push(TapeOp::Const(0.0));
                    idx
                }
            }
        }
        ExprNode::Binary(op, left, right) => {
            let l = build_recursive_cached(left, common_exprs, n_vars, ops, cache, resolver);
            let r = build_recursive_cached(right, common_exprs, n_vars, ops, cache, resolver);
            let idx = ops.len();
            ops.push(match op {
                BinaryOp::Add => TapeOp::Add(l, r),
                BinaryOp::Sub => TapeOp::Sub(l, r),
                BinaryOp::Mul => TapeOp::Mul(l, r),
                BinaryOp::Div => TapeOp::Div(l, r),
                BinaryOp::Mod => TapeOp::Mod(l, r),
                BinaryOp::Pow => TapeOp::Pow(l, r),
                BinaryOp::Atan2 => TapeOp::Atan2(l, r),
                BinaryOp::Less => TapeOp::Less(l, r),
                BinaryOp::IntDiv => TapeOp::IntDiv(l, r),
            });
            idx
        }
        ExprNode::Unary(op, arg) => {
            let a = build_recursive_cached(arg, common_exprs, n_vars, ops, cache, resolver);
            let idx = ops.len();
            ops.push(match op {
                UnaryOp::Abs => TapeOp::Abs(a),
                UnaryOp::Neg => TapeOp::Neg(a),
                UnaryOp::Floor => TapeOp::Floor(a),
                UnaryOp::Ceil => TapeOp::Ceil(a),
                UnaryOp::Tanh => TapeOp::Tanh(a),
                UnaryOp::Tan => TapeOp::Tan(a),
                UnaryOp::Sqrt => TapeOp::Sqrt(a),
                UnaryOp::Sinh => TapeOp::Sinh(a),
                UnaryOp::Sin => TapeOp::Sin(a),
                UnaryOp::Log10 => TapeOp::Log10(a),
                UnaryOp::Log => TapeOp::Log(a),
                UnaryOp::Exp => TapeOp::Exp(a),
                UnaryOp::Cosh => TapeOp::Cosh(a),
                UnaryOp::Cos => TapeOp::Cos(a),
                UnaryOp::Atanh => TapeOp::Atanh(a),
                UnaryOp::Atan => TapeOp::Atan(a),
                UnaryOp::Asinh => TapeOp::Asinh(a),
                UnaryOp::Asin => TapeOp::Asin(a),
                UnaryOp::Acosh => TapeOp::Acosh(a),
                UnaryOp::Acos => TapeOp::Acos(a),
            });
            idx
        }
        ExprNode::Nary(op, args) => {
            if args.is_empty() {
                let idx = ops.len();
                ops.push(TapeOp::Const(match op {
                    NaryOp::Sum => 0.0,
                    NaryOp::Min => f64::INFINITY,
                    NaryOp::Max => f64::NEG_INFINITY,
                }));
                return idx;
            }
            let mut acc = build_recursive_cached(&args[0], common_exprs, n_vars, ops, cache, resolver);
            for arg in &args[1..] {
                let next = build_recursive_cached(arg, common_exprs, n_vars, ops, cache, resolver);
                match op {
                    NaryOp::Sum => {
                        let idx = ops.len();
                        ops.push(TapeOp::Add(acc, next));
                        acc = idx;
                    }
                    NaryOp::Min => {
                        let idx = ops.len();
                        ops.push(TapeOp::Less(acc, next));
                        acc = idx;
                    }
                    NaryOp::Max => {
                        let neg_acc_idx = ops.len();
                        ops.push(TapeOp::Neg(acc));
                        let neg_next_idx = ops.len();
                        ops.push(TapeOp::Neg(next));
                        let min_idx = ops.len();
                        ops.push(TapeOp::Less(neg_acc_idx, neg_next_idx));
                        let result_idx = ops.len();
                        ops.push(TapeOp::Neg(min_idx));
                        acc = result_idx;
                    }
                }
            }
            acc
        }
        ExprNode::If(cond, then_expr, else_expr) => {
            let _c = build_recursive_cached(cond, common_exprs, n_vars, ops, cache, resolver);
            let t = build_recursive_cached(then_expr, common_exprs, n_vars, ops, cache, resolver);
            let _e = build_recursive_cached(else_expr, common_exprs, n_vars, ops, cache, resolver);
            t
        }
        ExprNode::StringLiteral(_) => {
            let idx = ops.len();
            ops.push(TapeOp::Const(0.0));
            idx
        }
        ExprNode::Funcall { id, args } => {
            let (lib, name) = resolver.funcs_by_id.get(id).cloned().unwrap_or_else(|| {
                panic!(
                    "build_recursive_cached: no ExternalResolver entry for Funcall \
                     id={id}; NlProblem::from_nl_data should have loaded this library"
                )
            });
            let built_args: Vec<FuncallArg> = args
                .iter()
                .map(|a| match a {
                    ExprNode::StringLiteral(s) => FuncallArg::Str(s.clone()),
                    other => FuncallArg::Tape(build_recursive_cached(
                        other,
                        common_exprs,
                        n_vars,
                        ops,
                        cache,
                        resolver,
                    )),
                })
                .collect();
            let idx = ops.len();
            ops.push(TapeOp::Funcall {
                lib,
                name,
                args: built_args,
            });
            idx
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::expr::*;

    #[test]
    fn tape_build_and_eval_polynomial() {
        // 3*x0^2 + 2*x1
        let expr = ExprNode::Binary(
            BinaryOp::Add,
            Box::new(ExprNode::Binary(
                BinaryOp::Mul,
                Box::new(ExprNode::Const(3.0)),
                Box::new(ExprNode::Binary(
                    BinaryOp::Pow,
                    Box::new(ExprNode::Var(0)),
                    Box::new(ExprNode::Const(2.0)),
                )),
            )),
            Box::new(ExprNode::Binary(
                BinaryOp::Mul,
                Box::new(ExprNode::Const(2.0)),
                Box::new(ExprNode::Var(1)),
            )),
        );
        let tape = Tape::build(&expr, &[], 2);
        let val = tape.eval(&[2.0, 3.0]);
        assert!((val - 18.0).abs() < 1e-10);
    }

    #[test]
    fn tape_gradient_polynomial() {
        // 3*x0^2 + 2*x1
        let expr = ExprNode::Binary(
            BinaryOp::Add,
            Box::new(ExprNode::Binary(
                BinaryOp::Mul,
                Box::new(ExprNode::Const(3.0)),
                Box::new(ExprNode::Binary(
                    BinaryOp::Pow,
                    Box::new(ExprNode::Var(0)),
                    Box::new(ExprNode::Const(2.0)),
                )),
            )),
            Box::new(ExprNode::Binary(
                BinaryOp::Mul,
                Box::new(ExprNode::Const(2.0)),
                Box::new(ExprNode::Var(1)),
            )),
        );
        let tape = Tape::build(&expr, &[], 2);
        let mut grad = vec![0.0; 2];
        tape.gradient(&[2.0, 3.0], &mut grad);
        // d/dx0 = 6*x0 = 12.0, d/dx1 = 2.0
        assert!((grad[0] - 12.0).abs() < 1e-10);
        assert!((grad[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn tape_gradient_transcendental() {
        // exp(x0) + sin(x1) + log(x0) + sqrt(x1)
        let expr = ExprNode::Binary(
            BinaryOp::Add,
            Box::new(ExprNode::Binary(
                BinaryOp::Add,
                Box::new(ExprNode::Unary(UnaryOp::Exp, Box::new(ExprNode::Var(0)))),
                Box::new(ExprNode::Unary(UnaryOp::Sin, Box::new(ExprNode::Var(1)))),
            )),
            Box::new(ExprNode::Binary(
                BinaryOp::Add,
                Box::new(ExprNode::Unary(UnaryOp::Log, Box::new(ExprNode::Var(0)))),
                Box::new(ExprNode::Unary(UnaryOp::Sqrt, Box::new(ExprNode::Var(1)))),
            )),
        );
        let tape = Tape::build(&expr, &[], 2);
        let x = [1.0, 1.0];
        let val = tape.eval(&x);
        let expected_val = 1.0_f64.exp() + 1.0_f64.sin() + 0.0 + 1.0;
        assert!((val - expected_val).abs() < 1e-10);

        let mut grad = vec![0.0; 2];
        tape.gradient(&x, &mut grad);
        // grad[0] = exp(1) + 1/1
        let expected_g0 = 1.0_f64.exp() + 1.0;
        // grad[1] = cos(1) + 0.5/sqrt(1)
        let expected_g1 = 1.0_f64.cos() + 0.5;
        assert!((grad[0] - expected_g0).abs() < 1e-10);
        assert!((grad[1] - expected_g1).abs() < 1e-10);
    }

    #[test]
    fn tape_common_expr_inlining() {
        // common_exprs[0] = x0 + x1 (ce0)
        // expr = Var(2)^2, where Var(2) refers to ce0 since n_vars=2
        let common_exprs = vec![ExprNode::Binary(
            BinaryOp::Add,
            Box::new(ExprNode::Var(0)),
            Box::new(ExprNode::Var(1)),
        )];
        let expr = ExprNode::Binary(
            BinaryOp::Pow,
            Box::new(ExprNode::Var(2)),
            Box::new(ExprNode::Const(2.0)),
        );
        let tape = Tape::build(&expr, &common_exprs, 2);
        let val = tape.eval(&[3.0, 4.0]);
        assert!((val - 49.0).abs() < 1e-10);

        let mut grad = vec![0.0; 2];
        tape.gradient(&[3.0, 4.0], &mut grad);
        // d/dx0 = 2*(x0+x1) = 14, d/dx1 = 2*(x0+x1) = 14
        assert!((grad[0] - 14.0).abs() < 1e-10);
        assert!((grad[1] - 14.0).abs() < 1e-10);
    }

    #[test]
    fn tape_nary_max_min() {
        // max(1, x0, 2) at x0=3 => 3
        let expr_max = ExprNode::Nary(
            NaryOp::Max,
            vec![ExprNode::Const(1.0), ExprNode::Var(0), ExprNode::Const(2.0)],
        );
        let tape_max = Tape::build(&expr_max, &[], 1);
        let val_max = tape_max.eval(&[3.0]);
        assert!((val_max - 3.0).abs() < 1e-10);

        // min(5, x0, 2) at x0=3 => 2
        let expr_min = ExprNode::Nary(
            NaryOp::Min,
            vec![ExprNode::Const(5.0), ExprNode::Var(0), ExprNode::Const(2.0)],
        );
        let tape_min = Tape::build(&expr_min, &[], 1);
        let val_min = tape_min.eval(&[3.0]);
        assert!((val_min - 2.0).abs() < 1e-10);
    }

    /// Helper: compute analytical Hessian and compare against FD of gradient.
    fn check_hessian_vs_fd(tape: &Tape, x: &[f64], tol: f64) {
        use std::collections::HashMap;
        let vars = tape.variables();
        let n = x.len();

        // Build hess_map for all variable pairs (lower triangle)
        let mut hess_map = HashMap::new();
        let mut idx = 0;
        for (ai, &vi) in vars.iter().enumerate() {
            for &vj in &vars[..=ai] {
                let (r, c) = if vi >= vj { (vi, vj) } else { (vj, vi) };
                hess_map.entry((r, c)).or_insert_with(|| { let i = idx; idx += 1; i });
            }
        }
        let nnz = idx;

        // Analytical Hessian
        let mut vals_ad = vec![0.0; nnz];
        tape.hessian_accumulate(x, 1.0, &hess_map, &mut vals_ad);

        // FD Hessian
        let mut vals_fd = vec![0.0; nnz];
        let mut x_pert = x.to_vec();
        let mut gp = vec![0.0; n];
        let mut gm = vec![0.0; n];
        for &j in &vars {
            let h = (1e-7_f64).max(x[j].abs() * 1e-7);
            x_pert[j] = x[j] + h;
            gp.iter_mut().for_each(|v| *v = 0.0);
            tape.gradient(&x_pert, &mut gp);
            x_pert[j] = x[j] - h;
            gm.iter_mut().for_each(|v| *v = 0.0);
            tape.gradient(&x_pert, &mut gm);
            x_pert[j] = x[j];
            for &i in &vars {
                if i >= j {
                    if let Some(&pos) = hess_map.get(&(i, j)) {
                        vals_fd[pos] = (gp[i] - gm[i]) / (2.0 * h);
                    }
                }
            }
        }

        for (&(r, c), &pos) in &hess_map {
            let ad = vals_ad[pos];
            let fd = vals_fd[pos];
            let err = (ad - fd).abs();
            let scale = fd.abs().max(1.0);
            assert!(
                err / scale < tol,
                "Hessian mismatch at ({},{}): AD={:.10e}, FD={:.10e}, err={:.2e}",
                r, c, ad, fd, err
            );
        }
    }

    #[test]
    fn hessian_quadratic() {
        // f(x,y) = 3*x^2 + 2*x*y + y^2
        // H = [[6, 2], [2, 2]]
        let expr = ExprNode::Nary(NaryOp::Sum, vec![
            ExprNode::Binary(BinaryOp::Mul, Box::new(ExprNode::Const(3.0)),
                Box::new(ExprNode::Binary(BinaryOp::Pow, Box::new(ExprNode::Var(0)), Box::new(ExprNode::Const(2.0))))),
            ExprNode::Binary(BinaryOp::Mul, Box::new(ExprNode::Const(2.0)),
                Box::new(ExprNode::Binary(BinaryOp::Mul, Box::new(ExprNode::Var(0)), Box::new(ExprNode::Var(1))))),
            ExprNode::Binary(BinaryOp::Pow, Box::new(ExprNode::Var(1)), Box::new(ExprNode::Const(2.0))),
        ]);
        let tape = Tape::build(&expr, &[], 2);
        check_hessian_vs_fd(&tape, &[2.0, 3.0], 1e-5);
    }

    #[test]
    fn hessian_transcendental() {
        // f(x,y) = exp(x) + sin(y) + log(x) + sqrt(y) + x*y
        let expr = ExprNode::Nary(NaryOp::Sum, vec![
            ExprNode::Unary(UnaryOp::Exp, Box::new(ExprNode::Var(0))),
            ExprNode::Unary(UnaryOp::Sin, Box::new(ExprNode::Var(1))),
            ExprNode::Unary(UnaryOp::Log, Box::new(ExprNode::Var(0))),
            ExprNode::Unary(UnaryOp::Sqrt, Box::new(ExprNode::Var(1))),
            ExprNode::Binary(BinaryOp::Mul, Box::new(ExprNode::Var(0)), Box::new(ExprNode::Var(1))),
        ]);
        let tape = Tape::build(&expr, &[], 2);
        check_hessian_vs_fd(&tape, &[1.5, 2.0], 1e-5);
    }

    #[test]
    fn hessian_division_and_trig() {
        // f(x,y) = x/y + cos(x) + tan(y) + atan(x)
        let expr = ExprNode::Nary(NaryOp::Sum, vec![
            ExprNode::Binary(BinaryOp::Div, Box::new(ExprNode::Var(0)), Box::new(ExprNode::Var(1))),
            ExprNode::Unary(UnaryOp::Cos, Box::new(ExprNode::Var(0))),
            ExprNode::Unary(UnaryOp::Tan, Box::new(ExprNode::Var(1))),
            ExprNode::Unary(UnaryOp::Atan, Box::new(ExprNode::Var(0))),
        ]);
        let tape = Tape::build(&expr, &[], 2);
        check_hessian_vs_fd(&tape, &[0.5, 1.2], 1e-5);
    }

    #[test]
    fn hessian_hyperbolic() {
        // f(x) = sinh(x) + cosh(x) + tanh(x) + asinh(x) + acosh(x+2) + atanh(x/2)
        let expr = ExprNode::Nary(NaryOp::Sum, vec![
            ExprNode::Unary(UnaryOp::Sinh, Box::new(ExprNode::Var(0))),
            ExprNode::Unary(UnaryOp::Cosh, Box::new(ExprNode::Var(0))),
            ExprNode::Unary(UnaryOp::Tanh, Box::new(ExprNode::Var(0))),
            ExprNode::Unary(UnaryOp::Asinh, Box::new(ExprNode::Var(0))),
            ExprNode::Unary(UnaryOp::Acosh, Box::new(ExprNode::Binary(
                BinaryOp::Add, Box::new(ExprNode::Var(0)), Box::new(ExprNode::Const(2.0))))),
            ExprNode::Unary(UnaryOp::Atanh, Box::new(ExprNode::Binary(
                BinaryOp::Div, Box::new(ExprNode::Var(0)), Box::new(ExprNode::Const(2.0))))),
        ]);
        let tape = Tape::build(&expr, &[], 1);
        check_hessian_vs_fd(&tape, &[0.5], 1e-5);
    }

    #[test]
    fn hessian_rosenbrock() {
        // f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        let expr = ExprNode::Nary(NaryOp::Sum, vec![
            ExprNode::Binary(BinaryOp::Pow,
                Box::new(ExprNode::Binary(BinaryOp::Sub,
                    Box::new(ExprNode::Const(1.0)), Box::new(ExprNode::Var(0)))),
                Box::new(ExprNode::Const(2.0))),
            ExprNode::Binary(BinaryOp::Mul,
                Box::new(ExprNode::Const(100.0)),
                Box::new(ExprNode::Binary(BinaryOp::Pow,
                    Box::new(ExprNode::Binary(BinaryOp::Sub,
                        Box::new(ExprNode::Var(1)),
                        Box::new(ExprNode::Binary(BinaryOp::Pow,
                            Box::new(ExprNode::Var(0)),
                            Box::new(ExprNode::Const(2.0)))))),
                    Box::new(ExprNode::Const(2.0))))),
        ]);
        let tape = Tape::build(&expr, &[], 2);
        check_hessian_vs_fd(&tape, &[1.0, 1.0], 1e-5);
        check_hessian_vs_fd(&tape, &[-1.5, 2.3], 1e-5);
    }

    #[test]
    fn hessian_sparsity_separable() {
        // f(x0, x1, x2) = sin(x0) + x1*x2
        // Hessian: d²f/dx0² from sin, d²f/dx1dx2 from x1*x2
        // Structural nonzeros: (0,0), (2,1)
        let expr = ExprNode::Nary(NaryOp::Sum, vec![
            ExprNode::Unary(UnaryOp::Sin, Box::new(ExprNode::Var(0))),
            ExprNode::Binary(BinaryOp::Mul,
                Box::new(ExprNode::Var(1)),
                Box::new(ExprNode::Var(2))),
        ]);
        let tape = Tape::build(&expr, &[], 3);
        let sparsity = tape.hessian_sparsity();
        assert!(sparsity.contains(&(0, 0)), "should have (0,0) from sin(x0)");
        assert!(sparsity.contains(&(2, 1)), "should have (2,1) from x1*x2");
        assert!(!sparsity.contains(&(1, 0)), "should NOT have (1,0) - separable");
        assert!(!sparsity.contains(&(2, 0)), "should NOT have (2,0) - separable");
    }

    #[test]
    fn hessian_sparsity_matches_numerical() {
        // f(x0,x1,x2) = exp(x0*x1) + x2^2
        // Hessian has entries: (0,0), (1,0), (1,1) from exp(x0*x1), and (2,2) from x2^2
        // But NOT (2,0) or (2,1)
        let expr = ExprNode::Nary(NaryOp::Sum, vec![
            ExprNode::Unary(UnaryOp::Exp,
                Box::new(ExprNode::Binary(BinaryOp::Mul,
                    Box::new(ExprNode::Var(0)),
                    Box::new(ExprNode::Var(1))))),
            ExprNode::Binary(BinaryOp::Pow,
                Box::new(ExprNode::Var(2)),
                Box::new(ExprNode::Const(2.0))),
        ]);
        let tape = Tape::build(&expr, &[], 3);
        let sparsity = tape.hessian_sparsity();
        // exp(x0*x1) couples x0 and x1
        assert!(sparsity.contains(&(0, 0)));
        assert!(sparsity.contains(&(1, 0)));
        assert!(sparsity.contains(&(1, 1)));
        // x2^2 only has diagonal
        assert!(sparsity.contains(&(2, 2)));
        // No cross-coupling between {x0,x1} and x2
        assert!(!sparsity.contains(&(2, 0)));
        assert!(!sparsity.contains(&(2, 1)));
        assert_eq!(sparsity.len(), 4);
    }
}
