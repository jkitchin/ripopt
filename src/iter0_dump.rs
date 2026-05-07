//! Iter-0 KKT-system dump for arki0003-style direction-divergence
//! debugging. Both ripopt and the Ipopt-FFI dump binary write the same
//! schema, and `examples/arki_diff.rs` compares the two.
//!
//! Activation (ripopt side): set `RIPOPT_IR_DUMP=/path/to/file.json` in
//! the environment. The IPM emits the dump after the iter-0 step is
//! computed (post-KKT-solve, post-step recovery, pre-line-search) and
//! exits the iter-0 logging block normally.
//!
//! Schema is intentionally flat (no nested matrices): all arrays are
//! `Vec<f64>` or `Vec<usize>` so JSON parses cleanly and the differ can
//! diff arrays element-wise without traversal.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Iter0Dump {
    /// "ripopt" or "ipopt" — identifies which solver produced this dump.
    pub solver: String,
    /// Free-form notes (problem name, options, build).
    pub note: String,
    /// Number of primal variables `x` (matches `.col` line count).
    pub n: usize,
    /// Number of constraints (matches `.row` line count minus one for objective).
    pub m: usize,
    /// Number of inequality constraints (slack dim, "n_d" in Ipopt notation).
    pub n_d: usize,

    // ===== Initial iterate =====
    /// Primal `x` at iter 0 (length n).
    pub x: Vec<f64>,
    /// Lower variable bounds. `None` encodes "no finite lower bound"
    /// (consumers treat as -∞). JSON round-trips as null. Length n.
    pub x_l: Vec<Option<f64>>,
    pub x_u: Vec<Option<f64>>,

    /// Slack `s` for inequality constraints at iter 0 (length n_d). Populated only on the ripopt side.
    #[serde(default)]
    pub s: Vec<f64>,
    /// Slack lower/upper bounds (length n_d each). `None` encodes
    /// unbounded sides. ripopt-only.
    #[serde(default)]
    pub d_l: Vec<Option<f64>>,
    #[serde(default)]
    pub d_u: Vec<Option<f64>>,

    /// Equality multipliers (length = #eq constraints, ripopt) or all multipliers (length m, ipopt).
    /// `y_layout` distinguishes the two encodings.
    pub y_c: Vec<f64>,
    pub y_d: Vec<f64>,
    /// "split" (ripopt: y_c = eq, y_d = ineq) or "combined" (ipopt: y_c = full λ, y_d = empty)
    pub y_layout: String,

    /// Bound multipliers in **full-n** indexing (zeros at unbounded sides).
    pub z_l: Vec<f64>,
    pub z_u: Vec<f64>,
    /// Slack-bound multipliers (ripopt-only, length n_d each).
    #[serde(default)]
    pub v_l: Vec<f64>,
    #[serde(default)]
    pub v_u: Vec<f64>,

    // ===== Evaluator output at x =====
    pub grad_f: Vec<f64>,
    /// Constraint values g(x) at x_0 (length m).
    pub g: Vec<f64>,
    /// Sparse Jacobian (lower triangle of full J, in row/col/val triplets).
    pub jac_rows: Vec<usize>,
    pub jac_cols: Vec<usize>,
    pub jac_vals: Vec<f64>,
    /// Sparse Hessian of the Lagrangian at (x, sigma=1, y=y_init), lower triangle.
    pub hess_rows: Vec<usize>,
    pub hess_cols: Vec<usize>,
    pub hess_vals: Vec<f64>,

    // ===== Scaling (NLP-level, applied before IPM) =====
    pub obj_scaling: f64,
    /// Per-variable scaling (length n). All-1.0 if no scaling.
    pub x_scaling: Vec<f64>,
    /// Per-constraint scaling (length m).
    pub c_scaling: Vec<f64>,

    // ===== Derived KKT-system inputs (ripopt-only) =====
    /// Σ_x diagonal at iter 0: z_L/(x-x_l) + z_U/(x_u-x), full-n.
    #[serde(default)]
    pub sigma_x: Vec<f64>,
    /// Σ_s diagonal at iter 0 (length n_d), ripopt-only.
    #[serde(default)]
    pub sigma_s: Vec<f64>,
    /// Augmented-system RHS (length n + n_d + m), ripopt-only.
    #[serde(default)]
    pub aug_rhs: Vec<f64>,
    /// Perturbations applied at iter 0 (ripopt-only).
    #[serde(default)]
    pub delta_w_used: f64,
    #[serde(default)]
    pub delta_c_used: f64,

    // ===== Iter-0 step direction =====
    /// Δx (length n).
    pub dx: Vec<f64>,
    /// Δs (length n_d), ripopt-only.
    #[serde(default)]
    pub ds: Vec<f64>,
    /// Δy combined (length m).
    pub dy: Vec<f64>,
    /// Δz_L, Δz_U in full-n indexing.
    pub dz_l: Vec<f64>,
    pub dz_u: Vec<f64>,
    /// Δv_L, Δv_U (length n_d), ripopt-only.
    #[serde(default)]
    pub dv_l: Vec<f64>,
    #[serde(default)]
    pub dv_u: Vec<f64>,

    /// α_pr / α_du after the iter-0 line search (Ipopt fills these by reading
    /// the IntermediateCallback values; ripopt fills them post-step).
    #[serde(default)]
    pub alpha_pr: f64,
    #[serde(default)]
    pub alpha_du: f64,

    /// μ at iter 0.
    pub mu: f64,
}

impl Iter0Dump {
    /// Write the dump as pretty JSON to `path`. Errors are logged via
    /// `eprintln!` and swallowed (this is a debug hook, not a code path
    /// the IPM should be derailed by).
    pub fn write(&self, path: &str) {
        match serde_json::to_string_pretty(self) {
            Ok(s) => {
                if let Err(e) = std::fs::write(path, s) {
                    eprintln!("iter0_dump: failed to write {}: {}", path, e);
                }
            }
            Err(e) => {
                eprintln!("iter0_dump: failed to serialize: {}", e);
            }
        }
    }

    /// Read a dump JSON from `path`. Returns the parsed struct.
    pub fn read(path: &str) -> Result<Self, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
        serde_json::from_str(&s).map_err(|e| format!("parse {}: {}", path, e))
    }
}
