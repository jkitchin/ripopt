//! A/B probe: solve a CUTEst BC (m=0 with finite bounds) problem with
//! both ripopt's IPM and the new TRON solver (`solve_bc`), and print a
//! one-line comparison.
//!
//! Usage:
//!   cargo run --example tron_vs_ipm --features cutest --release -- PROB1 [PROB2 ...]
//!
//! Each problem is loaded in its own subprocess invocation? No — CUTEst is
//! global state but `cleanup()` followed by reload of a different .dylib
//! is supported within one process. We chain problems sequentially.

use ripopt::bc_solver::{solve_bc, BcOptions};
use ripopt::{solve, NlpProblem, SolveStatus, SolverOptions};

#[path = "../benchmarks/cutest/cutest_problem.rs"]
mod cutest_problem;
#[path = "../benchmarks/cutest/cutest_ffi.rs"]
mod cutest_ffi;
use cutest_problem::CutestProblem;

fn fmt_status(s: SolveStatus) -> &'static str {
    match s {
        SolveStatus::Optimal => "Optimal",
        SolveStatus::Acceptable => "Acceptable",
        SolveStatus::Infeasible => "Infeasible",
        SolveStatus::LocalInfeasibility => "LocInfeas",
        SolveStatus::MaxIterations => "MaxIter",
        SolveStatus::MaxTimeExceeded => "Timeout",
        SolveStatus::NumericalError => "NumErr",
        SolveStatus::DivergingIterates => "Diverge",
        SolveStatus::RestorationFailed => "RestoFail",
        SolveStatus::EvaluationError => "EvalErr",
        SolveStatus::UserRequestedStop => "UserStop",
        SolveStatus::StopAtTinyStep => "TinyStep",
        SolveStatus::InternalError => "IntErr",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: tron_vs_ipm PROB1 [PROB2 ...]");
        std::process::exit(2);
    }

    let suite_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks").join("cutest");
    let problems_dir = suite_dir.join("problems");

    // Header.
    println!(
        "{:<12} {:>4} {:>4} | {:>10} {:>5} {:>14} | {:>10} {:>5} {:>14} | {:>10}",
        "name", "n", "BC", "IPM_status", "iters", "IPM_obj",
        "TRON_stat", "iters", "TRON_obj", "Δobj",
    );
    println!("{}", "-".repeat(110));

    for name in &args {
        let lib_path = problems_dir
            .join(format!("lib{}.{}", name, std::env::consts::DLL_EXTENSION));
        let outsdif_path = problems_dir.join(format!("{}_OUTSDIF.d", name));
        if !lib_path.exists() || !outsdif_path.exists() {
            eprintln!("{}: missing lib or OUTSDIF — skipping", name);
            continue;
        }

        let prob = match CutestProblem::load(
            name,
            lib_path.to_str().unwrap(),
            outsdif_path.to_str().unwrap(),
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{}: load failed: {}", name, e);
                continue;
            }
        };
        let n = prob.n;
        let m = prob.m;
        if m != 0 {
            eprintln!("{}: m={} not BC — skipping", name, m);
            prob.cleanup();
            continue;
        }
        let mut x_l = vec![0.0; n];
        let mut x_u = vec![0.0; n];
        prob.bounds(&mut x_l, &mut x_u);
        let any_bound = x_l.iter().any(|v| v.is_finite())
            || x_u.iter().any(|v| v.is_finite());

        // --- IPM ---
        let mut ipm_opts = SolverOptions::default();
        ipm_opts.print_level = 0;
        let ipm_res = solve(&prob, &ipm_opts);

        // --- TRON ---
        let mut bc_opts = BcOptions::default();
        // Lin-Moré §6: initial TR radius for least-squares problems.
        // Compute ‖g(x0)‖ here (after projection onto bounds).
        {
            let mut x0 = vec![0.0; n];
            prob.initial_point(&mut x0);
            for i in 0..n {
                if x0[i] < x_l[i] { x0[i] = x_l[i]; }
                else if x0[i] > x_u[i] { x0[i] = x_u[i]; }
            }
            let mut g0 = vec![0.0; n];
            prob.gradient(&x0, true, &mut g0);
            let g0n = g0.iter().map(|v| v * v).sum::<f64>().sqrt();
            if g0n.is_finite() && g0n > 0.0 {
                bc_opts.initial_tr_radius = g0n;
            }
        }
        bc_opts.max_iter = 500;
        let tron_res = solve_bc(&prob, &bc_opts);

        let dobj = (ipm_res.objective - tron_res.objective).abs();

        println!(
            "{:<12} {:>4} {:>4} | {:>10} {:>5} {:>14.6e} | {:>10} {:>5} {:>14.6e} | {:>10.2e}",
            name,
            n,
            if any_bound { "BC" } else { "U" },
            fmt_status(ipm_res.status),
            ipm_res.iterations,
            ipm_res.objective,
            fmt_status(tron_res.status),
            tron_res.iterations,
            tron_res.objective,
            dobj,
        );

        prob.cleanup();
    }
}
