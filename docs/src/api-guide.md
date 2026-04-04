# API Guide

## Defining a problem

Implement the `NlpProblem` trait. Every method has a default that returns zeros or empty vectors, so you only need to override what your problem uses.

### Full example: HS071

HS071 is the classic Hock-Schittkowski test problem #71:

```
min   x₁·x₄·(x₁+x₂+x₃) + x₃
s.t.  x₁·x₂·x₃·x₄ ≥ 25
      x₁²+x₂²+x₃²+x₄² = 40
      1 ≤ xᵢ ≤ 5
x* ≈ (1.000, 4.743, 3.821, 1.379),  f* ≈ 17.014
```

```rust
use ripopt::{NlpProblem, SolveStatus, SolverOptions};

struct Hs071;

impl NlpProblem for Hs071 {
    fn num_variables(&self) -> usize { 4 }
    fn num_constraints(&self) -> usize { 2 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..4 { x_l[i] = 1.0; x_u[i] = 5.0; }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 25.0; g_u[0] = f64::INFINITY;  // inequality: product >= 25
        g_l[1] = 40.0; g_u[1] = 40.0;            // equality: sum of squares = 40
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0.copy_from_slice(&[1.0, 5.0, 5.0, 1.0]);
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        grad[1] = x[0] * x[3];
        grad[2] = x[0] * x[3] + 1.0;
        grad[3] = x[0] * (x[0] + x[1] + x[2]);
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];
    }

    // Jacobian: list of (row, col) pairs for non-zero entries
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0,0,0,0, 1,1,1,1], vec![0,1,2,3, 0,1,2,3])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = x[1]*x[2]*x[3]; vals[1] = x[0]*x[2]*x[3];
        vals[2] = x[0]*x[1]*x[3]; vals[3] = x[0]*x[1]*x[2];
        vals[4] = 2.0*x[0]; vals[5] = 2.0*x[1];
        vals[6] = 2.0*x[2]; vals[7] = 2.0*x[3];
    }

    // Hessian of Lagrangian: lower triangle only. L = obj_factor*f + Σ λᵢ*gᵢ
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0,1,1,2,2,2,3,3,3,3], vec![0,0,1,0,1,2,0,1,2,3])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * 2.0*x[3]        + lambda[1]*2.0;
        vals[1] = obj_factor * x[3]             + lambda[0]*x[2]*x[3];
        vals[2] =                                 lambda[1]*2.0;
        vals[3] = obj_factor * x[3]             + lambda[0]*x[1]*x[3];
        vals[4] =                                 lambda[0]*x[0]*x[3];
        vals[5] =                                 lambda[1]*2.0;
        vals[6] = obj_factor*(2.0*x[0]+x[1]+x[2]) + lambda[0]*x[1]*x[2];
        vals[7] = obj_factor * x[0]             + lambda[0]*x[0]*x[2];
        vals[8] = obj_factor * x[0]             + lambda[0]*x[0]*x[1];
        vals[9] =                                 lambda[1]*2.0;
    }
}

fn main() {
    let result = ripopt::solve(&Hs071, &SolverOptions::default());
    assert_eq!(result.status, SolveStatus::Optimal);
    println!("f* = {:.6}", result.objective);          // 17.014017
    println!("x* = {:?}", result.x);                   // [1.0, 4.743, 3.821, 1.379]
    println!("iters = {}", result.iterations);         // ~8
}
```

## Trait reference

| Method | Required | Description |
|---|---|---|
| `num_variables()` | **Yes** | Number of decision variables n |
| `num_constraints()` | **Yes** | Number of constraints m |
| `bounds(x_l, x_u)` | **Yes** | Variable bounds (use ±∞ for unbounded) |
| `constraint_bounds(g_l, g_u)` | **Yes** | Constraint bounds (g_l = g_u for equality) |
| `initial_point(x0)` | **Yes** | Starting point — must be strictly interior to bounds |
| `objective(x)` | **Yes** | Objective value f(x) |
| `gradient(x, grad)` | **Yes** | Gradient ∇f(x) |
| `constraints(x, g)` | if m > 0 | Constraint values g(x) |
| `jacobian_structure()` | if m > 0 | Sparsity pattern as (row, col) triplets |
| `jacobian_values(x, vals)` | if m > 0 | Jacobian values in same order as structure |
| `hessian_structure()` | Recommended | Lower triangle sparsity of ∇²L |
| `hessian_values(x, obj_factor, lambda, vals)` | Recommended | ∇²L = obj_factor·∇²f + Σ λᵢ·∇²gᵢ |

**Note:** If `hessian_structure()` / `hessian_values()` are not implemented, ripopt automatically falls back to L-BFGS Hessian approximation.

## The Lagrangian sign convention

ripopt uses `L = f + yᵀg` (same as Ipopt). The Hessian requested is:

```
obj_factor · ∇²f(x) + Σᵢ λᵢ · ∇²gᵢ(x)
```

Only the **lower triangle** is needed: entries (i, j) with i ≥ j.

## Reading the result

```rust
let result = ripopt::solve(&problem, &opts);

match result.status {
    SolveStatus::Optimal => {
        println!("x = {:?}", result.x);       // optimal point
        println!("f = {}", result.objective); // objective value
        println!("y = {:?}", result.y);       // constraint multipliers
        println!("z_l = {:?}", result.z_l);   // lower bound multipliers
        println!("z_u = {:?}", result.z_u);   // upper bound multipliers
    }
    SolveStatus::MaxIterations => { /* increase max_iter or adjust options */ }
    SolveStatus::NumericalError => { /* try fallbacks, different starting point */ }
    SolveStatus::LocalInfeasibility => { /* problem is locally infeasible */ }
    SolveStatus::RestorationFailed => { /* feasibility recovery failed */ }
}
```

## Sensitivity analysis

After a successful solve, compute how the optimal solution changes under parameter perturbations:

```rust
// Parametric sensitivity: dx/dp at optimum
let (result, sens) = ripopt::solve_with_sensitivity(&problem, &opts);
if let Some(s) = sens {
    println!("dx/dp = {:?}", s.dx_dp);  // primal sensitivity
    println!("dy/dp = {:?}", s.dy_dp);  // multiplier sensitivity
}
```

See `src/sensitivity.rs` for the sIPOPT-style implementation.

## Customizing options

```rust
let opts = SolverOptions {
    tol: 1e-10,
    max_iter: 500,
    print_level: 3,
    hessian_approximation_lbfgs: true,  // skip exact Hessian
    ..SolverOptions::default()
};
let result = ripopt::solve(&problem, &opts);
```

See [Solver Options](options.md) for the full reference.
