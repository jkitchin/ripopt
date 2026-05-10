//! Issue #22 audit: classify CUTEst m=0 problems as either
//! free (no bounds) or bound-constrained (BC). Print a compact CSV
//! that downstream `jq` / shell commands can join against
//! benchmarks/cutest/results.json.
//!
//! Usage: cargo run --example bc_audit --features cutest --release -- [list_path]
//! list_path defaults to benchmarks/cutest/problem_list.txt.

use ripopt::NlpProblem;
#[path = "../benchmarks/cutest/cutest_problem.rs"]
mod cutest_problem;
#[path = "../benchmarks/cutest/cutest_ffi.rs"]
mod cutest_ffi;
use cutest_problem::CutestProblem;

fn main() {
    let suite_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks").join("cutest");
    let problems_dir = suite_dir.join("problems");
    let list_path = std::env::args().nth(1)
        .unwrap_or_else(|| suite_dir.join("problem_list.txt").to_string_lossy().into());
    let names: Vec<String> = std::fs::read_to_string(&list_path)
        .unwrap_or_else(|_| panic!("cannot read {}", list_path))
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
        .map(|l| l.trim().to_string())
        .collect();

    println!("name,n,m,n_x_l_finite,n_x_u_finite,class");
    for name in &names {
        let lib_path = problems_dir.join(format!("lib{}.{}", name, std::env::consts::DLL_EXTENSION));
        let outsdif_path = problems_dir.join(format!("{}_OUTSDIF.d", name));
        if !lib_path.exists() || !outsdif_path.exists() {
            continue;
        }
        // CUTEst is global state: load → query → drop. Run each in its own
        // subprocess invocation? No — sequential is fine if we drop between.
        let prob = match CutestProblem::load(
            name,
            lib_path.to_str().unwrap(),
            outsdif_path.to_str().unwrap(),
        ) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let n = prob.n;
        let m = prob.m;
        if m != 0 {
            // Free / drop and skip non-BC candidates.
            drop(prob);
            continue;
        }
        let mut x_l = vec![0.0; n];
        let mut x_u = vec![0.0; n];
        prob.bounds(&mut x_l, &mut x_u);
        let n_l = x_l.iter().filter(|v| v.is_finite()).count();
        let n_u = x_u.iter().filter(|v| v.is_finite()).count();
        let class = if n_l == 0 && n_u == 0 { "U" } else { "BC" };
        println!("{},{},{},{},{},{}", name, n, m, n_l, n_u, class);
        drop(prob);
    }
}
