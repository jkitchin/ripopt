use super::expr::{self, ExprNode};
use super::header::{self, NlHeader};

/// AMPL suffix kind. Encoded in the low two bits of the `S` segment
/// flags byte (matches AMPL's `ASL_Sufkind_*` macros).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuffixKind {
    Variable,
    Constraint,
    Objective,
    Problem,
}

/// One AMPL suffix table read from an `S` segment.
///
/// Format: header line `S<flags> <count> <name>` followed by `count`
/// `<index> <value>` lines. Bits in `flags`:
///   - `flags & 3`  → kind (variable/constraint/objective/problem)
///   - `flags & 4`  → 0 = integer-valued, 4 = float-valued
///
/// Mirrors AMPL's `AmplSuffixHandler` convention (`AmplTNLP.cpp:1110-1165`).
#[derive(Debug, Clone)]
pub struct NlSuffix {
    pub name: String,
    pub kind: SuffixKind,
    pub float_valued: bool,
    /// Sparse `(index, value)` pairs as written in the file.
    pub entries: Vec<(usize, f64)>,
}

/// AMPL imported (external) function declaration from an `F` segment.
#[derive(Debug, Clone)]
pub struct ImportedFunc {
    pub id: usize,
    /// 0 = real-valued, 1 = string-valued (per AMPL's funcadd ABI).
    pub kind: usize,
    /// Declared number of arguments (may be -1 for variable-arity in the spec;
    /// we store the raw decoded value as read).
    pub nargs: i64,
    pub name: String,
}

/// Raw parsed data from an NL file.
#[derive(Debug)]
pub struct NlFileData {
    pub header: NlHeader,
    /// Nonlinear expression for each constraint (None if purely linear).
    pub con_exprs: Vec<Option<ExprNode>>,
    /// (objective_index, maximize, nonlinear_expr).
    pub obj_exprs: Vec<(usize, bool, Option<ExprNode>)>,
    /// Common sub-expressions (defined variables), in order.
    /// Each is the full expression (linear + nonlinear combined).
    pub common_exprs: Vec<ExprNode>,
    /// Linear coefficients per constraint: con_linear\[i\] = [(var_idx, coeff), ...].
    pub con_linear: Vec<Vec<(usize, f64)>>,
    /// Linear gradient coefficients per objective: obj_linear\[i\] = [(var_idx, coeff), ...].
    pub obj_linear: Vec<Vec<(usize, f64)>>,
    /// Variable bounds.
    pub x_l: Vec<f64>,
    pub x_u: Vec<f64>,
    /// Constraint bounds.
    pub g_l: Vec<f64>,
    pub g_u: Vec<f64>,
    /// Initial primal values.
    pub x0: Vec<f64>,
    /// Initial dual values.
    pub y0: Vec<f64>,
    /// Jacobian column pointers (cumulative), from k segment.
    pub jac_col_ptrs: Vec<usize>,
    /// AMPL imported functions declared via `F` segments.
    pub imported_funcs: Vec<ImportedFunc>,
    /// AMPL suffix tables declared via `S` segments. Empty for files
    /// without suffix data.
    pub suffixes: Vec<NlSuffix>,
}

impl NlFileData {
    /// Extract `scaling_factor` suffix entries into dense per-source
    /// vectors aligned with AmplTNLP's convention
    /// (`AmplTNLP.cpp:1110-1165`): the same suffix name `scaling_factor`
    /// is read against three AMPL kinds (objective, variable,
    /// constraint). Missing entries default to 1.0; a single-entry
    /// objective suffix collapses to a scalar.
    pub fn scaling_factors(&self) -> NlScalingFactors {
        let n = self.header.n_vars;
        let m = self.header.n_constrs;
        let mut out = NlScalingFactors::default();
        for s in &self.suffixes {
            if s.name != "scaling_factor" {
                continue;
            }
            match s.kind {
                SuffixKind::Objective => {
                    // Conventionally only objective 0; take the first.
                    if let Some(&(_, v)) = s.entries.first() {
                        out.obj = Some(v);
                    }
                }
                SuffixKind::Variable => {
                    let mut x = vec![1.0f64; n];
                    for &(i, v) in &s.entries {
                        if i < n {
                            x[i] = v;
                        }
                    }
                    out.x = Some(x);
                }
                SuffixKind::Constraint => {
                    let mut g = vec![1.0f64; m];
                    for &(i, v) in &s.entries {
                        if i < m {
                            g[i] = v;
                        }
                    }
                    out.g = Some(g);
                }
                SuffixKind::Problem => {}
            }
        }
        out
    }
}

/// Aggregated `scaling_factor` suffix values from an NL file.
#[derive(Debug, Default, Clone)]
pub struct NlScalingFactors {
    pub obj: Option<f64>,
    pub x: Option<Vec<f64>>,
    pub g: Option<Vec<f64>>,
}

impl NlScalingFactors {
    pub fn is_empty(&self) -> bool {
        self.obj.is_none() && self.x.is_none() && self.g.is_none()
    }
}

/// Parse an NL file from its text content.
pub fn parse_nl_file(content: &str) -> Result<NlFileData, String> {
    let mut lines = content.lines();

    let header = header::parse_header(&mut lines)?;

    let n = header.n_vars;
    let m = header.n_constrs;

    let mut data = NlFileData {
        con_exprs: vec![None; m],
        obj_exprs: Vec::new(),
        common_exprs: Vec::new(),
        con_linear: vec![Vec::new(); m],
        obj_linear: Vec::new(),
        x_l: vec![f64::NEG_INFINITY; n],
        x_u: vec![f64::INFINITY; n],
        g_l: vec![f64::NEG_INFINITY; m],
        g_u: vec![f64::INFINITY; m],
        x0: vec![0.0; n],
        y0: vec![0.0; m],
        jac_col_ptrs: Vec::new(),
        imported_funcs: Vec::new(),
        suffixes: Vec::new(),
        header,
    };

    // Collect remaining lines so we can peek
    let remaining: Vec<&str> = lines.collect();
    let mut pos = 0;

    while pos < remaining.len() {
        let line = remaining[pos].trim();
        if line.is_empty() {
            pos += 1;
            continue;
        }

        let first = line.as_bytes()[0];
        match first {
            b'C' => {
                let idx = parse_segment_index(line, 'C')?;
                pos += 1;
                let mut sub = remaining[pos..].iter().copied();
                let expr = expr::parse_expr(&mut sub)?;
                // Count consumed lines
                let consumed = count_consumed(&remaining[pos..], &sub);
                pos += consumed;
                if idx < m {
                    data.con_exprs[idx] = Some(expr);
                }
            }
            b'O' => {
                // O<idx> <maximize_flag>
                let parts: Vec<&str> = line.split_whitespace().collect();
                let idx = parse_segment_index(parts[0], 'O')?;
                let maximize = parts.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0) != 0;
                pos += 1;
                let mut sub = remaining[pos..].iter().copied();
                let expr = expr::parse_expr(&mut sub)?;
                let consumed = count_consumed(&remaining[pos..], &sub);
                pos += consumed;
                data.obj_exprs.push((idx, maximize, Some(expr)));
            }
            b'V' => {
                // V<idx> <n_linear> <type>
                let parts: Vec<&str> = line.split_whitespace().collect();
                let n_linear: usize = parts
                    .get(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                pos += 1;

                // Read linear terms
                let mut linear_sum: Option<ExprNode> = None;
                for _ in 0..n_linear {
                    if pos >= remaining.len() {
                        break;
                    }
                    let lparts: Vec<&str> = remaining[pos].trim().split_whitespace().collect();
                    if lparts.len() >= 2 {
                        let var_idx: usize = lparts[0].parse().unwrap_or(0);
                        let coeff: f64 = lparts[1].parse().unwrap_or(0.0);
                        let term = if coeff == 1.0 {
                            ExprNode::Var(var_idx)
                        } else {
                            ExprNode::Binary(
                                expr::BinaryOp::Mul,
                                Box::new(ExprNode::Const(coeff)),
                                Box::new(ExprNode::Var(var_idx)),
                            )
                        };
                        linear_sum = Some(match linear_sum {
                            None => term,
                            Some(acc) => ExprNode::Binary(
                                expr::BinaryOp::Add,
                                Box::new(acc),
                                Box::new(term),
                            ),
                        });
                    }
                    pos += 1;
                }

                // Read nonlinear expression
                let mut sub = remaining[pos..].iter().copied();
                let nl_expr = expr::parse_expr(&mut sub)?;
                let consumed = count_consumed(&remaining[pos..], &sub);
                pos += consumed;

                // Combine linear + nonlinear
                let full_expr = match linear_sum {
                    None => nl_expr,
                    Some(lin) => {
                        // Check if nonlinear part is just constant 0
                        if matches!(&nl_expr, ExprNode::Const(c) if *c == 0.0) {
                            lin
                        } else {
                            ExprNode::Binary(
                                expr::BinaryOp::Add,
                                Box::new(lin),
                                Box::new(nl_expr),
                            )
                        }
                    }
                };
                data.common_exprs.push(full_expr);
            }
            b'r' => {
                pos += 1;
                for i in 0..m {
                    if pos >= remaining.len() {
                        break;
                    }
                    let rline = remaining[pos].trim();
                    let parts: Vec<&str> = rline.split_whitespace().collect();
                    if let Some(type_code) = parts.first().and_then(|s| s.parse::<usize>().ok()) {
                        match type_code {
                            0 => {
                                // Range: lower upper
                                data.g_l[i] = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(f64::NEG_INFINITY);
                                data.g_u[i] = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(f64::INFINITY);
                            }
                            1 => {
                                // Upper bound only
                                data.g_u[i] = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(f64::INFINITY);
                                data.g_l[i] = f64::NEG_INFINITY;
                            }
                            2 => {
                                // Lower bound only
                                data.g_l[i] = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(f64::NEG_INFINITY);
                                data.g_u[i] = f64::INFINITY;
                            }
                            3 => {
                                // No bounds
                                data.g_l[i] = f64::NEG_INFINITY;
                                data.g_u[i] = f64::INFINITY;
                            }
                            4 => {
                                // Equality
                                let val = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                data.g_l[i] = val;
                                data.g_u[i] = val;
                            }
                            5 => {
                                // Complementarity (treat as no bounds for now)
                                data.g_l[i] = f64::NEG_INFINITY;
                                data.g_u[i] = f64::INFINITY;
                            }
                            _ => {}
                        }
                    }
                    pos += 1;
                }
            }
            b'b' => {
                pos += 1;
                for i in 0..n {
                    if pos >= remaining.len() {
                        break;
                    }
                    let bline = remaining[pos].trim();
                    let parts: Vec<&str> = bline.split_whitespace().collect();
                    if let Some(type_code) = parts.first().and_then(|s| s.parse::<usize>().ok()) {
                        match type_code {
                            0 => {
                                data.x_l[i] = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(f64::NEG_INFINITY);
                                data.x_u[i] = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(f64::INFINITY);
                            }
                            1 => {
                                data.x_u[i] = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(f64::INFINITY);
                                data.x_l[i] = f64::NEG_INFINITY;
                            }
                            2 => {
                                data.x_l[i] = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(f64::NEG_INFINITY);
                                data.x_u[i] = f64::INFINITY;
                            }
                            3 => {
                                data.x_l[i] = f64::NEG_INFINITY;
                                data.x_u[i] = f64::INFINITY;
                            }
                            4 => {
                                let val = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                data.x_l[i] = val;
                                data.x_u[i] = val;
                            }
                            _ => {}
                        }
                    }
                    pos += 1;
                }
            }
            b'x' => {
                let count = parse_segment_count(line)?;
                pos += 1;
                for _ in 0..count {
                    if pos >= remaining.len() {
                        break;
                    }
                    let parts: Vec<&str> = remaining[pos].trim().split_whitespace().collect();
                    if parts.len() >= 2 {
                        let idx: usize = parts[0].parse().unwrap_or(0);
                        let val: f64 = parts[1].parse().unwrap_or(0.0);
                        if idx < n {
                            data.x0[idx] = val;
                        }
                    }
                    pos += 1;
                }
            }
            b'd' => {
                let count = parse_segment_count(line)?;
                pos += 1;
                for _ in 0..count {
                    if pos >= remaining.len() {
                        break;
                    }
                    let parts: Vec<&str> = remaining[pos].trim().split_whitespace().collect();
                    if parts.len() >= 2 {
                        let idx: usize = parts[0].parse().unwrap_or(0);
                        let val: f64 = parts[1].parse().unwrap_or(0.0);
                        if idx < m {
                            data.y0[idx] = val;
                        }
                    }
                    pos += 1;
                }
            }
            b'k' => {
                // k<n_vars - 1>: cumulative Jacobian column counts
                let count = parse_segment_count(line)?;
                pos += 1;
                data.jac_col_ptrs = Vec::with_capacity(count);
                for _ in 0..count {
                    if pos >= remaining.len() {
                        break;
                    }
                    let val: usize = remaining[pos]
                        .trim()
                        .parse()
                        .unwrap_or(0);
                    data.jac_col_ptrs.push(val);
                    pos += 1;
                }
            }
            b'J' => {
                // J<constraint_idx> <count>: linear Jacobian entries
                let idx = parse_segment_index(line, 'J')?;
                let count = parse_segment_count_after_space(line)?;
                pos += 1;
                if idx < m {
                    for _ in 0..count {
                        if pos >= remaining.len() {
                            break;
                        }
                        let parts: Vec<&str> = remaining[pos].trim().split_whitespace().collect();
                        if parts.len() >= 2 {
                            let var_idx: usize = parts[0].parse().unwrap_or(0);
                            let coeff: f64 = parts[1].parse().unwrap_or(0.0);
                            data.con_linear[idx].push((var_idx, coeff));
                        }
                        pos += 1;
                    }
                } else {
                    // Skip entries for out-of-range constraint
                    pos += count;
                }
            }
            b'G' => {
                // G<objective_idx> <count>: linear gradient entries
                let idx = parse_segment_index(line, 'G')?;
                let count = parse_segment_count_after_space(line)?;
                pos += 1;
                // Ensure we have space
                while data.obj_linear.len() <= idx {
                    data.obj_linear.push(Vec::new());
                }
                for _ in 0..count {
                    if pos >= remaining.len() {
                        break;
                    }
                    let parts: Vec<&str> = remaining[pos].trim().split_whitespace().collect();
                    if parts.len() >= 2 {
                        let var_idx: usize = parts[0].parse().unwrap_or(0);
                        let coeff: f64 = parts[1].parse().unwrap_or(0.0);
                        data.obj_linear[idx].push((var_idx, coeff));
                    }
                    pos += 1;
                }
            }
            b'S' => {
                // Suffix segment header: `S<flags> <count> <name>`.
                // `flags & 3` selects kind (variable=0, constraint=1,
                // objective=2, problem=3); `flags & 4` set means the
                // values are floats (else integers, but we always store
                // as f64). Mirrors AMPL's `AmplSuffixHandler` ABI.
                let parts: Vec<&str> = line.split_whitespace().collect();
                let flags: u32 = parts
                    .first()
                    .and_then(|s| s.get(1..))
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let count: usize = parts
                    .get(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let name = parts.get(2).copied().unwrap_or("").to_string();
                let kind = match flags & 3 {
                    0 => SuffixKind::Variable,
                    1 => SuffixKind::Constraint,
                    2 => SuffixKind::Objective,
                    _ => SuffixKind::Problem,
                };
                let float_valued = (flags & 4) != 0;
                let mut entries = Vec::with_capacity(count);
                for off in 1..=count {
                    if pos + off >= remaining.len() {
                        break;
                    }
                    let row = remaining[pos + off].trim();
                    let cols: Vec<&str> = row.split_whitespace().collect();
                    if cols.len() >= 2 {
                        let i: usize = cols[0].parse().unwrap_or(usize::MAX);
                        let v: f64 = cols[1].parse().unwrap_or(0.0);
                        if i != usize::MAX {
                            entries.push((i, v));
                        }
                    }
                }
                data.suffixes.push(NlSuffix {
                    name,
                    kind,
                    float_valued,
                    entries,
                });
                pos += 1 + count;
            }
            b'F' => {
                // AMPL imported (external) function declaration:
                // `F<k> <type> <nargs> <name>` — all on one line.
                let parts: Vec<&str> = line.split_whitespace().collect();
                let id = parse_segment_index(parts[0], 'F')?;
                let kind: usize = parts
                    .get(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let nargs: i64 = parts
                    .get(2)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let name = parts.get(3).copied().unwrap_or("").to_string();
                data.imported_funcs.push(ImportedFunc {
                    id,
                    kind,
                    nargs,
                    name,
                });
                pos += 1;
            }
            _ => {
                pos += 1;
            }
        }
    }

    // Ensure obj_linear has at least one entry
    if data.obj_linear.is_empty() {
        data.obj_linear.push(Vec::new());
    }

    // If no objective expression was parsed, add a zero objective
    if data.obj_exprs.is_empty() {
        data.obj_exprs.push((0, false, None));
    }

    Ok(data)
}

/// Parse segment index from "X<idx>" or "X<idx> ...".
fn parse_segment_index(line: &str, prefix: char) -> Result<usize, String> {
    let s = line.trim();
    // Find first char after prefix that's a space or end of string
    let num_part = &s[1..].split_whitespace().next().unwrap_or("0");
    num_part
        .parse()
        .map_err(|e| format!("Bad {} segment index '{}': {}", prefix, num_part, e))
}

/// Parse count from "x<count>" segment header.
fn parse_segment_count(line: &str) -> Result<usize, String> {
    let s = line.trim();
    let num_part = &s[1..].split_whitespace().next().unwrap_or("0");
    num_part
        .parse()
        .map_err(|e| format!("Bad segment count '{}': {}", num_part, e))
}

/// Parse count from "X<idx> <count>" format.
fn parse_segment_count_after_space(line: &str) -> Result<usize, String> {
    let parts: Vec<&str> = line.trim().split_whitespace().collect();
    if parts.len() >= 2 {
        parts[1]
            .parse()
            .map_err(|e| format!("Bad segment count '{}': {}", parts[1], e))
    } else {
        Ok(0)
    }
}

/// Count how many lines were consumed from `slice` by the expression parser.
/// `remaining_iter` is the iterator state after parsing.
fn count_consumed<'a>(
    slice: &[&'a str],
    remaining: &impl Iterator<Item = &'a str>,
) -> usize {
    // Since we can't directly inspect iterator position,
    // we count by comparing: total - remaining.
    // But we consumed the iterator by reference so we can't count remaining.
    // Instead, use a different approach: count during parsing.
    // Actually, let's use a counter wrapper.
    // For now, use the expression depth to count lines.
    let _ = remaining;
    count_expr_lines(slice)
}

/// Count the number of lines consumed by one expression at the start of `lines`.
fn count_expr_lines(lines: &[&str]) -> usize {
    let mut pos = 0;
    count_expr_lines_recursive(lines, &mut pos);
    pos
}

fn count_expr_lines_recursive(lines: &[&str], pos: &mut usize) {
    if *pos >= lines.len() {
        return;
    }
    let line = lines[*pos].trim();
    let token = line.split('#').next().unwrap_or("").trim();
    *pos += 1;

    if token.starts_with('n') || token.starts_with('v') || token.starts_with('h') {
        // Leaf node: 1 line
    } else if token.starts_with('f') {
        // Funcall: `f<id> <nargs>` on the current line, then nargs sub-expressions.
        let rest = &token[1..];
        let mut parts = rest.split_whitespace();
        let _id = parts.next().unwrap_or("0");
        let nargs: usize = parts.next().unwrap_or("0").parse().unwrap_or(0);
        for _ in 0..nargs {
            count_expr_lines_recursive(lines, pos);
        }
    } else if token.starts_with('o') {
        let opcode: usize = token[1..].parse().unwrap_or(0);
        match opcode {
            // Binary ops: consume 2 sub-expressions
            0..=5 | 22 | 48 | 55 => {
                count_expr_lines_recursive(lines, pos);
                count_expr_lines_recursive(lines, pos);
            }
            // Unary ops: consume 1 sub-expression
            13..=16 | 37..=47 | 49..=53 => {
                count_expr_lines_recursive(lines, pos);
            }
            // N-ary ops: count line + n sub-expressions
            11 | 12 | 54 => {
                if *pos < lines.len() {
                    let count_str = lines[*pos].trim().split('#').next().unwrap_or("").trim();
                    let count: usize = count_str.parse().unwrap_or(0);
                    *pos += 1;
                    for _ in 0..count {
                        count_expr_lines_recursive(lines, pos);
                    }
                }
            }
            // If-then-else: 3 sub-expressions
            35 => {
                count_expr_lines_recursive(lines, pos);
                count_expr_lines_recursive(lines, pos);
                count_expr_lines_recursive(lines, pos);
            }
            // If-then: 2 sub-expressions
            65 => {
                count_expr_lines_recursive(lines, pos);
                count_expr_lines_recursive(lines, pos);
            }
            _ => {}
        }
    }
}
