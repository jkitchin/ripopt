use ripopt::NlpProblem;
#[path = "../benchmarks/cutest/cutest_problem.rs"]
mod cutest_problem;
#[path = "../benchmarks/cutest/cutest_ffi.rs"]
mod cutest_ffi;
use cutest_problem::CutestProblem;

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "ALLINITA".to_string());
    let suite_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks").join("cutest");
    let problems_dir = suite_dir.join("problems");
    let lib_path = problems_dir.join(format!("lib{}.{}", name, std::env::consts::DLL_EXTENSION));
    let outsdif_path = problems_dir.join(format!("{}_OUTSDIF.d", name));
    let problem = CutestProblem::load(&name, lib_path.to_str().unwrap(), outsdif_path.to_str().unwrap()).expect("load");

    use ripopt::{solve, SolverOptions};
    println!("== {} (n={} m={}) ==", name, problem.num_variables(), problem.num_constraints());
    for (label, adaptive) in [("monotone", false), ("adaptive", true)] {
        let opts = SolverOptions {
            tol: 1e-8,
            max_iter: 3000,
            print_level: 0,
            mu_strategy_adaptive: adaptive,
            max_wall_time: 30.0,
            ..Default::default()
        };
        let r = solve(&problem, &opts);
        println!("{:>10}: {:?} obj={:.10e} iter={} pr={:.2e} du={:.2e} compl={:.2e} mu={:.2e}",
            label, r.status, r.objective, r.iterations,
            r.diagnostics.final_primal_inf, r.diagnostics.final_dual_inf,
            r.diagnostics.final_compl, r.diagnostics.final_mu);
    }
}
