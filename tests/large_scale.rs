//! Large-scale benchmark problems for testing the sparse LDL^T solver.
//!
//! Run with: cargo test --release -- --ignored large_scale --nocapture

#[path = "common/large_scale_problems.rs"]
mod problems;
use problems::*;

use ripopt::{NlpProblem, SolveStatus, SolverOptions};
use std::time::Instant;

fn default_options() -> SolverOptions {
    SolverOptions {
        max_wall_time: 300.0,
        tol: 1e-6,
        print_level: 5,
        ..SolverOptions::default()
    }
}

/// Compute max constraint violation from constraint values and bounds.
fn max_cv(problem: &dyn NlpProblem, g: &[f64]) -> f64 {
    let m = problem.num_constraints();
    if m == 0 {
        return 0.0;
    }
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    problem.constraint_bounds(&mut g_l, &mut g_u);
    let mut cv = 0.0_f64;
    for i in 0..m {
        let viol_l = (g_l[i] - g[i]).max(0.0);
        let viol_u = (g[i] - g_u[i]).max(0.0);
        cv = cv.max(viol_l).max(viol_u);
    }
    cv
}

#[test]
#[ignore]
fn large_scale_chained_rosenbrock_5k() {
    let problem = ChainedRosenbrock { n: 5000 };
    let options = default_options();
    eprintln!("\n=== Chained Rosenbrock 5K (n=5000, m=0, KKT=5000) ===");
    eprintln!("Solving...");
    let start = Instant::now();
    let result = ripopt::solve(&problem, &options);
    let elapsed = start.elapsed();

    eprintln!(
        "RESULT: status={:?}, obj={:.6e}, iters={}, time={:.3}s",
        result.status, result.objective, result.iterations, elapsed.as_secs_f64()
    );

    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal/Acceptable, got {:?}",
        result.status
    );
    assert!(
        result.objective < 1e-4,
        "f* should be ~0, got {:.6e}",
        result.objective
    );
}

#[test]
#[ignore]
fn large_scale_bratu_10k() {
    let problem = BratuProblem::new(10000);
    let options = default_options();
    eprintln!("\n=== Bratu BVP 10K (n=10000, m=9998, KKT=19998) ===");
    eprintln!("Solving...");
    let start = Instant::now();
    let result = ripopt::solve(&problem, &options);
    let elapsed = start.elapsed();

    let cv = max_cv(&problem, &result.constraint_values);
    eprintln!(
        "RESULT: status={:?}, obj={:.6e}, cv={:.6e}, iters={}, time={:.3}s",
        result.status, result.objective, cv, result.iterations,
        elapsed.as_secs_f64()
    );

    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal/Acceptable, got {:?}",
        result.status
    );
    assert!(cv < 1e-4, "Constraint violation should be small, got {:.6e}", cv);
}

#[test]
#[ignore]
fn large_scale_optimal_control_20k() {
    let problem = OptimalControl::new(9999);
    let options = default_options();
    eprintln!("\n=== Optimal Control 20K (n=19999, m=10000, KKT=29999) ===");
    eprintln!("Solving...");
    let start = Instant::now();
    let result = ripopt::solve(&problem, &options);
    let elapsed = start.elapsed();

    let cv = max_cv(&problem, &result.constraint_values);
    eprintln!(
        "RESULT: status={:?}, obj={:.6e}, cv={:.6e}, iters={}, time={:.3}s",
        result.status, result.objective, cv, result.iterations,
        elapsed.as_secs_f64()
    );

    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal/Acceptable, got {:?}",
        result.status
    );
    assert!(cv < 1e-4, "Constraint violation should be small, got {:.6e}", cv);
}

#[test]
#[ignore]
fn large_scale_poisson_control_50k() {
    let problem = PoissonControl::new(158);
    let options = default_options();
    eprintln!("\n=== Poisson Control 50K (n=49928, m=24964, KKT=74892) ===");
    eprintln!("Solving...");
    let start = Instant::now();
    let result = ripopt::solve(&problem, &options);
    let elapsed = start.elapsed();

    let cv = max_cv(&problem, &result.constraint_values);
    eprintln!(
        "RESULT: status={:?}, obj={:.6e}, cv={:.6e}, iters={}, time={:.3}s",
        result.status, result.objective, cv, result.iterations,
        elapsed.as_secs_f64()
    );

    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal/Acceptable, got {:?}",
        result.status
    );
    assert!(cv < 1e-3, "Constraint violation should be small, got {:.6e}", cv);
}

#[test]
#[ignore]
fn large_scale_sparse_qp_100k() {
    let problem = SparseQP { n: 50000 };
    let options = default_options();
    eprintln!("\n=== Sparse QP 100K (n=50000, m=50000, KKT=100000) ===");
    eprintln!("Solving...");
    let start = Instant::now();
    let result = ripopt::solve(&problem, &options);
    let elapsed = start.elapsed();

    let mut f0 = 0.0; problem.objective(&vec![0.5; 50000], true, &mut f0);
    let cv = max_cv(&problem, &result.constraint_values);
    eprintln!(
        "RESULT: status={:?}, obj={:.6e}, cv={:.6e}, iters={}, time={:.3}s, f0={:.6e}",
        result.status, result.objective, cv, result.iterations,
        elapsed.as_secs_f64(), f0
    );

    assert!(
        result.status == SolveStatus::Optimal,
        "Expected Optimal/Acceptable, got {:?}",
        result.status
    );
    assert!(
        result.objective < f0,
        "f* should be less than f(x0)={:.6e}, got {:.6e}",
        f0,
        result.objective
    );
}

// ===========================================================================
// Smaller-scale tests (500, 1000, 2500 variables) for quick validation
// ===========================================================================

#[test]
#[ignore]
fn large_scale_rosenbrock_500() {
    let problem = ChainedRosenbrock { n: 500 };
    let options = default_options();
    eprintln!("\n=== Chained Rosenbrock 500 (n=500, m=0, KKT=500) ===");
    eprintln!("Solving...");
    let start = Instant::now();
    let result = ripopt::solve(&problem, &options);
    let elapsed = start.elapsed();
    eprintln!(
        "RESULT: status={:?}, obj={:.6e}, iters={}, time={:.3}s",
        result.status, result.objective, result.iterations, elapsed.as_secs_f64()
    );
    assert!(
        result.status == SolveStatus::Optimal,
        "got {:?}", result.status
    );
    assert!(result.objective < 1e-4, "f*={:.6e}", result.objective);
}

#[test]
#[ignore]
fn large_scale_bratu_1000() {
    let problem = BratuProblem::new(1000);
    let options = default_options();
    eprintln!("\n=== Bratu BVP 1K (n=1000, m=998, KKT=1998) ===");
    eprintln!("Solving...");
    let start = Instant::now();
    let result = ripopt::solve(&problem, &options);
    let elapsed = start.elapsed();
    let cv = max_cv(&problem, &result.constraint_values);
    eprintln!(
        "RESULT: status={:?}, obj={:.6e}, cv={:.6e}, iters={}, time={:.3}s",
        result.status, result.objective, cv, result.iterations, elapsed.as_secs_f64()
    );
    assert!(
        result.status == SolveStatus::Optimal,
        "got {:?}", result.status
    );
    assert!(cv < 1e-4, "cv={:.6e}", cv);
}

#[test]
#[ignore]
fn large_scale_optimal_control_2500() {
    let problem = OptimalControl::new(1249); // n=2499, m=1250
    let options = default_options();
    eprintln!("\n=== Optimal Control 2.5K (n=2499, m=1250, KKT=3749) ===");
    eprintln!("Solving...");
    let start = Instant::now();
    let result = ripopt::solve(&problem, &options);
    let elapsed = start.elapsed();
    let cv = max_cv(&problem, &result.constraint_values);
    eprintln!(
        "RESULT: status={:?}, obj={:.6e}, cv={:.6e}, iters={}, time={:.3}s",
        result.status, result.objective, cv, result.iterations, elapsed.as_secs_f64()
    );
    assert!(
        result.status == SolveStatus::Optimal,
        "got {:?}", result.status
    );
    assert!(cv < 1e-4, "cv={:.6e}", cv);
}

#[test]
#[ignore]
fn large_scale_poisson_control_2500() {
    let problem = PoissonControl::new(35); // n=2*35²=2450, m=35²=1225, KKT=3675
    let options = default_options();
    eprintln!("\n=== Poisson Control 2.5K (n=2450, m=1225, KKT=3675) ===");
    eprintln!("Solving...");
    let start = Instant::now();
    let result = ripopt::solve(&problem, &options);
    let elapsed = start.elapsed();
    let cv = max_cv(&problem, &result.constraint_values);
    eprintln!(
        "RESULT: status={:?}, obj={:.6e}, cv={:.6e}, iters={}, time={:.3}s",
        result.status, result.objective, cv, result.iterations, elapsed.as_secs_f64()
    );
    assert!(
        result.status == SolveStatus::Optimal,
        "got {:?}", result.status
    );
    assert!(cv < 1e-3, "cv={:.6e}", cv);
}

#[test]
#[ignore]
fn large_scale_sparse_qp_1000() {
    let n = 500;
    let problem = SparseQP { n };
    let options = default_options();
    eprintln!("\n=== Sparse QP 1K (n=500, m=500, KKT=1000) ===");
    eprintln!("Solving...");
    let start = Instant::now();
    let result = ripopt::solve(&problem, &options);
    let elapsed = start.elapsed();
    let mut f0 = 0.0; problem.objective(&vec![0.5; n], true, &mut f0);
    let cv = max_cv(&problem, &result.constraint_values);
    eprintln!(
        "RESULT: status={:?}, obj={:.6e}, cv={:.6e}, iters={}, time={:.3}s, f0={:.6e}",
        result.status, result.objective, cv, result.iterations, elapsed.as_secs_f64(), f0
    );
    assert!(
        result.status == SolveStatus::Optimal,
        "got {:?}", result.status
    );
    assert!(result.objective < f0, "f*={:.6e} >= f0={:.6e}", result.objective, f0);
}
