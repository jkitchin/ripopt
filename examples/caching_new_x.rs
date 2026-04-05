//! Demonstrates how `new_x` enables efficient caching of shared computations.
//!
//! Many real-world NLP problems (phase equilibria, process simulation, etc.) share
//! expensive intermediate results between objective and constraint evaluations.
//! The `new_x` flag tells the model when `x` has changed, so these intermediates
//! can be computed once per point and reused across objective, gradient, constraint,
//! Jacobian, and Hessian evaluations at the same point.
//!
//! This example solves HS071 with a simulated "expensive shared computation"
//! and tracks how many full recomputations are saved by checking `new_x`.
//!
//! Run with: cargo run --example caching_new_x

use ripopt::{NlpProblem, SolveStatus, SolverOptions};
use std::cell::RefCell;

/// Cached intermediate results shared between objective and constraints.
/// In a real application this might be phase equilibria, equation of state
/// evaluations, or other expensive model computations.
struct SharedCache {
    /// The x at which the cache was computed.
    x: Vec<f64>,
    /// Products used by both objective and constraints.
    prod_all: f64,    // x0 * x1 * x2 * x3
    sum_sq: f64,      // x0^2 + x1^2 + x2^2 + x3^2
    sum_first3: f64,  // x0 + x1 + x2
}

/// Counters for cache hits and misses.
#[derive(Default)]
struct CacheStats {
    full_evals: usize,
    cache_hits: usize,
}

struct CachedHs071 {
    cache: RefCell<Option<SharedCache>>,
    stats: RefCell<CacheStats>,
}

impl CachedHs071 {
    fn new() -> Self {
        Self {
            cache: RefCell::new(None),
            stats: RefCell::new(CacheStats::default()),
        }
    }

    /// Compute (or reuse) the shared intermediates for the given point.
    /// When `new_x` is false, the cached values are returned directly.
    fn ensure_cache(&self, x: &[f64], new_x: bool) {
        let needs_recompute = if new_x {
            true
        } else {
            // Double-check: cache should exist and match x
            let cache = self.cache.borrow();
            cache.is_none() || cache.as_ref().unwrap().x != x
        };

        if needs_recompute {
            self.stats.borrow_mut().full_evals += 1;
            // Simulate an expensive shared computation
            let shared = SharedCache {
                x: x.to_vec(),
                prod_all: x[0] * x[1] * x[2] * x[3],
                sum_sq: x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3],
                sum_first3: x[0] + x[1] + x[2],
            };
            *self.cache.borrow_mut() = Some(shared);
        } else {
            self.stats.borrow_mut().cache_hits += 1;
        }
    }

    fn print_stats(&self) {
        let stats = self.stats.borrow();
        let total = stats.full_evals + stats.cache_hits;
        println!("Cache statistics:");
        println!("  Total evaluation calls:     {}", total);
        println!("  Full recomputations:        {}", stats.full_evals);
        println!("  Cache hits (avoided work):  {}", stats.cache_hits);
        if total > 0 {
            println!(
                "  Hit rate:                   {:.1}%",
                100.0 * stats.cache_hits as f64 / total as f64
            );
        }
    }
}

impl NlpProblem for CachedHs071 {
    fn num_variables(&self) -> usize {
        4
    }

    fn num_constraints(&self) -> usize {
        2
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..4 {
            x_l[i] = 1.0;
            x_u[i] = 5.0;
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 25.0;
        g_u[0] = f64::INFINITY;
        g_l[1] = 40.0;
        g_u[1] = 40.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 1.0;
        x0[1] = 5.0;
        x0[2] = 5.0;
        x0[3] = 1.0;
    }

    fn objective(&self, x: &[f64], new_x: bool, obj: &mut f64) -> bool {
        self.ensure_cache(x, new_x);
        let cache = self.cache.borrow();
        let c = cache.as_ref().unwrap();
        // f(x) = x0 * x3 * (x0 + x1 + x2) + x2
        *obj = x[0] * x[3] * c.sum_first3 + x[2];
        true
    }

    fn gradient(&self, x: &[f64], new_x: bool, grad: &mut [f64]) -> bool {
        self.ensure_cache(x, new_x);
        grad[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        grad[1] = x[0] * x[3];
        grad[2] = x[0] * x[3] + 1.0;
        grad[3] = x[0] * (x[0] + x[1] + x[2]);
        true
    }

    fn constraints(&self, x: &[f64], new_x: bool, g: &mut [f64]) -> bool {
        self.ensure_cache(x, new_x);
        let cache = self.cache.borrow();
        let c = cache.as_ref().unwrap();
        g[0] = c.prod_all;
        g[1] = c.sum_sq;
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (
            vec![0, 0, 0, 0, 1, 1, 1, 1],
            vec![0, 1, 2, 3, 0, 1, 2, 3],
        )
    }

    fn jacobian_values(&self, x: &[f64], new_x: bool, vals: &mut [f64]) -> bool {
        self.ensure_cache(x, new_x);
        vals[0] = x[1] * x[2] * x[3];
        vals[1] = x[0] * x[2] * x[3];
        vals[2] = x[0] * x[1] * x[3];
        vals[3] = x[0] * x[1] * x[2];
        vals[4] = 2.0 * x[0];
        vals[5] = 2.0 * x[1];
        vals[6] = 2.0 * x[2];
        vals[7] = 2.0 * x[3];
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (
            vec![0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
        )
    }

    fn hessian_values(
        &self,
        x: &[f64],
        new_x: bool,
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) -> bool {
        self.ensure_cache(x, new_x);
        vals[0] = obj_factor * 2.0 * x[3] + lambda[1] * 2.0;
        vals[1] = obj_factor * x[3] + lambda[0] * x[2] * x[3];
        vals[2] = lambda[1] * 2.0;
        vals[3] = obj_factor * x[3] + lambda[0] * x[1] * x[3];
        vals[4] = lambda[0] * x[0] * x[3];
        vals[5] = lambda[1] * 2.0;
        vals[6] = obj_factor * (2.0 * x[0] + x[1] + x[2]) + lambda[0] * x[1] * x[2];
        vals[7] = obj_factor * x[0] + lambda[0] * x[0] * x[2];
        vals[8] = obj_factor * x[0] + lambda[0] * x[0] * x[1];
        vals[9] = lambda[1] * 2.0;
        true
    }
}

fn main() {
    let problem = CachedHs071::new();
    let options = SolverOptions {
        max_iter: 500,
        tol: 1e-8,
        print_level: 0,
        ..SolverOptions::default()
    };

    let result = ripopt::solve(&problem, &options);

    println!("HS071 with new_x caching");
    println!("========================");
    println!("Status:     {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Objective:  {:.10}", result.objective);
    println!(
        "Solution:   ({:.6}, {:.6}, {:.6}, {:.6})",
        result.x[0], result.x[1], result.x[2], result.x[3]
    );
    println!();
    problem.print_stats();
    println!();
    println!(
        "Without caching, every call to objective/gradient/constraints/jacobian/hessian"
    );
    println!(
        "would recompute the shared intermediates. With new_x, only the first call at"
    );
    println!(
        "each new point pays the cost; subsequent calls at the same point reuse the cache."
    );

    assert_eq!(result.status, SolveStatus::Optimal);
    assert!(
        (result.objective - 17.014).abs() < 0.01,
        "Unexpected objective: {}",
        result.objective
    );
    assert!(
        problem.stats.borrow().cache_hits > 0,
        "Expected cache hits from new_x=false calls"
    );
}
