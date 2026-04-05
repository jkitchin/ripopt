//! Issue #7 reproducer: dual infeasibility stalls at ~1e-3 on exp/log objectives
//!
//! Reproduces the exact problem from the issue, matching discopt's formulation:
//! - DENSE Jacobian and Hessian (PyO3 bindings use m*n and n*(n+1)/2)
//! - '>=' constraints normalized to '<=' with negated body (Expression.__ge__)
//! - Constraint bodies compiled as `body - rhs` with rhs=0
//!
//! discopt formulation of constraints:
//!   x1+x2 >= 3  →  body = 3-(x1+x2), sense <=, bounds [-inf, 0]
//!   x1 <= 4*y1  →  body = x1-4*y1, sense <=, bounds [-inf, 0]
//!   x2 <= 4*y2  →  body = x2-4*y2, sense <=, bounds [-inf, 0]
//!   y1+y2 <= 1+y3 → body = y1+y2-1-y3, sense <=, bounds [-inf, 0]

use ripopt::{NlpProblem, SolveStatus, SolverOptions};

struct ExpActivation {
    x_l: [f64; 5],
    x_u: [f64; 5],
    x0: [f64; 5],
}

impl ExpActivation {
    fn relaxation() -> Self {
        Self {
            x_l: [0.0, 0.0, 0.0, 0.0, 0.0],
            x_u: [4.0, 4.0, 1.0, 1.0, 1.0],
            x0: [2.0, 2.0, 0.5, 0.5, 0.5],
        }
    }

    fn branch(fixes: &[(usize, f64)], parent_x: &[f64; 5]) -> Self {
        let mut x_l = [0.0, 0.0, 0.0, 0.0, 0.0];
        let mut x_u = [4.0, 4.0, 1.0, 1.0, 1.0];
        for &(idx, val) in fixes {
            x_l[idx] = val;
            x_u[idx] = val;
        }
        let mut x0 = *parent_x;
        for i in 0..5 {
            x0[i] = x0[i].clamp(x_l[i], x_u[i]);
        }
        Self { x_l, x_u, x0 }
    }
}

impl NlpProblem for ExpActivation {
    fn num_variables(&self) -> usize { 5 }
    fn num_constraints(&self) -> usize { 4 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l.copy_from_slice(&self.x_l);
        x_u.copy_from_slice(&self.x_u);
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        // All constraints normalized to <= 0 (discopt style)
        for i in 0..4 {
            g_l[i] = f64::NEG_INFINITY;
            g_u[i] = 0.0;
        }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0.copy_from_slice(&self.x0);
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = x[0].exp() + x[1].exp() + 5.0 * x[2] + 6.0 * x[3] + 8.0 * x[4];
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad[0] = x[0].exp();
        grad[1] = x[1].exp();
        grad[2] = 5.0;
        grad[3] = 6.0;
        grad[4] = 8.0;
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        // discopt normalization: >= becomes negated <=
        g[0] = 3.0 - x[0] - x[1];            // 3-(x1+x2) <= 0
        g[1] = x[0] - 4.0 * x[2];             // x1-4*y1 <= 0
        g[2] = x[1] - 4.0 * x[3];             // x2-4*y2 <= 0
        g[3] = x[2] + x[3] - 1.0 - x[4];      // y1+y2-1-y3 <= 0
        true
    }

    // Dense Jacobian: 4*5 = 20 entries (matching PyO3 bindings)
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let mut rows = Vec::with_capacity(20);
        let mut cols = Vec::with_capacity(20);
        for i in 0..4 {
            for j in 0..5 {
                rows.push(i);
                cols.push(j);
            }
        }
        (rows, cols)
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        vals.fill(0.0);
        // g0 = 3-x1-x2: dg/dx1=-1, dg/dx2=-1
        vals[0 * 5 + 0] = -1.0;
        vals[0 * 5 + 1] = -1.0;
        // g1 = x1-4*y1: dg/dx1=1, dg/dy1=-4
        vals[1 * 5 + 0] = 1.0;
        vals[1 * 5 + 2] = -4.0;
        // g2 = x2-4*y2: dg/dx2=1, dg/dy2=-4
        vals[2 * 5 + 1] = 1.0;
        vals[2 * 5 + 3] = -4.0;
        // g3 = y1+y2-1-y3: dg/dy1=1, dg/dy2=1, dg/dy3=-1
        vals[3 * 5 + 2] = 1.0;
        vals[3 * 5 + 3] = 1.0;
        vals[3 * 5 + 4] = -1.0;
        true
    }

    // Dense lower-triangle Hessian: 5*(5+1)/2 = 15 entries
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let mut rows = Vec::with_capacity(15);
        let mut cols = Vec::with_capacity(15);
        for i in 0..5 {
            for j in 0..=i {
                rows.push(i);
                cols.push(j);
            }
        }
        (rows, cols)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) -> bool {
        vals.fill(0.0);
        vals[0] = obj_factor * x[0].exp();  // H[0,0]
        vals[2] = obj_factor * x[1].exp();  // H[1,1]
        true
    }
}

fn run_problem(name: &str, problem: &ExpActivation, opts: &SolverOptions) -> (SolveStatus, usize, f64, [f64; 5]) {
    println!("\n=== {} ===", name);
    let result = ripopt::solve(problem, opts);
    println!("  Status:     {:?}", result.status);
    println!("  Objective:  {:.10e}", result.objective);
    println!("  Solution:   x=[{:.4}, {:.4}] y=[{:.4}, {:.4}, {:.4}]",
        result.x[0], result.x[1], result.x[2], result.x[3], result.x[4]);
    println!("  Iterations: {}", result.iterations);
    if result.status != SolveStatus::Optimal {
        println!("  >>> NON-OPTIMAL STATUS");
    }
    let mut sol = [0.0; 5];
    sol.copy_from_slice(&result.x[..5]);
    (result.status, result.iterations, result.objective, sol)
}

fn main() {
    let opts = SolverOptions {
        print_level: 5,
        tol: 1e-7,
        max_iter: 3000,
        // Disable fallback solvers to expose the raw IPM stall
        enable_lbfgs_fallback: false,
        enable_al_fallback: false,
        enable_sqp_fallback: false,
        enable_slack_fallback: false,
        enable_lbfgs_hessian_fallback: false,
        disable_nlp_restoration: true,
        ..SolverOptions::default()
    };

    // Node 1: Root relaxation
    let (s1, i1, _, root_sol) = run_problem("Node 1: Root relaxation",
        &ExpActivation::relaxation(), &opts);

    // Node 2: y1=1
    let (s2, i2, _, sol2) = run_problem("Node 2: y1=1",
        &ExpActivation::branch(&[(2, 1.0)], &root_sol), &opts);

    // Node 3: y1=0
    let (s3, i3, _, sol3) = run_problem("Node 3: y1=0",
        &ExpActivation::branch(&[(2, 0.0)], &root_sol), &opts);

    // Node 4: y1=1, y2=1
    let (s4, i4, _, _) = run_problem("Node 4: y1=1, y2=1",
        &ExpActivation::branch(&[(2, 1.0), (3, 1.0)], &sol2), &opts);

    // Node 5: y1=1, y2=0
    let (s5, i5, _, _) = run_problem("Node 5: y1=1, y2=0",
        &ExpActivation::branch(&[(2, 1.0), (3, 0.0)], &sol2), &opts);

    // Node 6: y1=0, y2=1
    let (s6, i6, _, sol6) = run_problem("Node 6: y1=0, y2=1",
        &ExpActivation::branch(&[(2, 0.0), (3, 1.0)], &sol3), &opts);

    // Node 7: y1=0, y2=1, y3=0
    let (s7, i7, _, _) = run_problem("Node 7: y1=0, y2=1, y3=0",
        &ExpActivation::branch(&[(2, 0.0), (3, 1.0), (4, 0.0)], &sol6), &opts);

    // Summary
    println!("\n=== SUMMARY ===");
    let results = vec![
        ("Root", s1, i1),
        ("y1=1", s2, i2),
        ("y1=0", s3, i3),
        ("y1=1,y2=1", s4, i4),
        ("y1=1,y2=0", s5, i5),
        ("y1=0,y2=1", s6, i6),
        ("y1=0,y2=1,y3=0", s7, i7),
    ];
    let mut any_stall = false;
    for (name, status, iters) in &results {
        let flag = if *status != SolveStatus::Optimal { " *** STALL" } else { "" };
        if *status != SolveStatus::Optimal { any_stall = true; }
        println!("  {:25} {:?} ({} iters){}", name, status, iters, flag);
    }
    if any_stall {
        println!("\nREPRODUCED: dual stall on exp() B&B subproblems");
    } else {
        println!("\nNOT REPRODUCED");
    }
}
