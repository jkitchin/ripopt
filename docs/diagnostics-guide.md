# ripopt Solver Diagnostics Guide

## Overview

`SolverDiagnostics` captures structured data about solver behavior during a
solve. It is available both programmatically via `result.diagnostics` and as a
stderr summary block (printed when `print_level >= 5`).

The diagnostic block looks like:

```
--- ripopt diagnostics ---
status: Optimal
iterations: 8
wall_time: 0.001s
final_mu: 4.14e-9
final_primal_inf: 5.55e-17
final_dual_inf: 1.04e-12
final_compl: 7.75e-9
restoration_count: 0
nlp_restoration_count: 0
mu_mode_switches: 2
filter_rejects: 0
watchdog_activations: 0
soc_corrections: 0
--- end diagnostics ---
```

## Fields

| Field | Meaning |
|---|---|
| `restoration_count` | GN (Gauss-Newton) restoration entries |
| `nlp_restoration_count` | Full NLP restoration entries (heavier, Ipopt-style) |
| `mu_mode_switches` | Barrier mode transitions (Free <-> Fixed) |
| `filter_rejects` | Line search failures (backtracking exhausted) |
| `watchdog_activations` | Watchdog triggered by consecutive short steps |
| `soc_corrections` | Second-order corrections accepted |
| `final_mu` | Barrier parameter at termination |
| `final_primal_inf` | Constraint violation at termination |
| `final_dual_inf` | Dual infeasibility (stationarity error) |
| `final_compl` | Complementarity error at termination |
| `wall_time_secs` | Total wall-clock time |
| `fallback_used` | Which fallback succeeded, if any (`lbfgs_hessian`, `augmented_lagrangian`, `sqp`, `slack`) |

## Reading the diagnostics

**Healthy solve** (HS071-like): 0 restorations, 0 filter rejects, 2-4 mu mode
switches, `final_mu` near `1e-9`, `final_primal_inf` and `final_dual_inf` both
below `tol`.

**Struggling solve**: Many filter rejects, multiple restorations, `final_mu`
stuck above `1e-4`, or a fallback was used.

**Key patterns and what to try:**

| Pattern | Likely cause | Options to adjust |
|---|---|---|
| `filter_rejects` > 5 | Line search fighting constraints | Increase `mu_init`, reduce `kappa` |
| `restoration_count` > 3 | Repeated feasibility recovery | Try `enable_slack_fallback`, or increase `mu_init` |
| `mu_mode_switches` > 10 | Free/Fixed cycling | Set `mu_strategy_adaptive: false` for monotone mode |
| `final_mu` stuck > 1e-4 | Barrier parameter not decreasing | Increase `max_iter`, reduce `mu_linear_decrease_factor` |
| `fallback_used: Some(...)` | Primary IPM failed | Examine which fallback; consider changing Hessian strategy |
| `soc_corrections` > 0 | Nonlinear constraints causing step rejection | Normal; increase `max_soc` if filter rejects are also high |
| `watchdog_activations` > 0 | Tiny steps detected | Try `hessian_approximation_lbfgs: true` |

---

## Example 1: Easy problem (HS071)

HS071 is a 4-variable, 2-constraint nonlinear program. ripopt solves it in ~8
iterations with default options.

```rust
use ripopt::{NlpProblem, SolveStatus, SolverOptions};

// min  x1*x4*(x1+x2+x3) + x3
// s.t. x1*x2*x3*x4 >= 25
//      x1^2+x2^2+x3^2+x4^2 = 40
//      1 <= xi <= 5
// x0 = (1, 5, 5, 1),  f* ~ 17.014

struct Hs071;

impl NlpProblem for Hs071 {
    fn num_variables(&self) -> usize { 4 }
    fn num_constraints(&self) -> usize { 2 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..4 { x_l[i] = 1.0; x_u[i] = 5.0; }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 25.0; g_u[0] = f64::INFINITY;  // product >= 25
        g_l[1] = 40.0; g_u[1] = 40.0;            // sum of squares = 40
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 1.0; x0[1] = 5.0; x0[2] = 5.0; x0[3] = 1.0;
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
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0,0,0,0, 1,1,1,1], vec![0,1,2,3, 0,1,2,3])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = x[1]*x[2]*x[3]; vals[1] = x[0]*x[2]*x[3];
        vals[2] = x[0]*x[1]*x[3]; vals[3] = x[0]*x[1]*x[2];
        vals[4] = 2.0*x[0]; vals[5] = 2.0*x[1];
        vals[6] = 2.0*x[2]; vals[7] = 2.0*x[3];
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0,1,1,2,2,2,3,3,3,3], vec![0,0,1,0,1,2,0,1,2,3])
    }

    fn hessian_values(&self, x: &[f64], s: f64, l: &[f64], v: &mut [f64]) {
        v[0] = s*2.0*x[3] + l[1]*2.0;
        v[1] = s*x[3] + l[0]*x[2]*x[3];
        v[2] = l[1]*2.0;
        v[3] = s*x[3] + l[0]*x[1]*x[3];
        v[4] = l[0]*x[0]*x[3];
        v[5] = l[1]*2.0;
        v[6] = s*(2.0*x[0]+x[1]+x[2]) + l[0]*x[1]*x[2];
        v[7] = s*x[0] + l[0]*x[0]*x[2];
        v[8] = s*x[0] + l[0]*x[0]*x[1];
        v[9] = l[1]*2.0;
    }
}

fn main() {
    let result = ripopt::solve(&Hs071, &SolverOptions::default());

    assert_eq!(result.status, SolveStatus::Optimal);
    assert!((result.objective - 17.014).abs() < 0.01);

    // Diagnostics: expect clean solve
    let d = &result.diagnostics;
    println!("iterations: {}", result.iterations);   // ~8
    println!("filter_rejects: {}", d.filter_rejects); // 0
    println!("restorations: {}", d.restoration_count); // 0
}
```

**Expected diagnostics:**
```
status: Optimal
iterations: 8
filter_rejects: 0
restoration_count: 0
mu_mode_switches: 2
final_mu: ~4e-9
```

No drama. The solver converges in a straight line.

---

## Example 2: Hard problem (TP374 — unsolved)

TP374 has 10 variables, 35 nonlinear inequality constraints involving
trigonometric sums. ripopt hits `MaxIterations` at 2999 iterations.
Known optimal: f* = 0.233264.

```rust
use ripopt::{NlpProblem, SolverOptions};
use std::f64::consts::PI;

struct TP374;

fn tp374_a(z: f64, x: &[f64]) -> f64 {
    (1..=9).map(|k| x[k-1] * (k as f64 * z).cos()).sum()
}
fn tp374_b(z: f64, x: &[f64]) -> f64 {
    (1..=9).map(|k| x[k-1] * (k as f64 * z).sin()).sum()
}
fn tp374_g(z: f64, x: &[f64]) -> f64 {
    let (a, b) = (tp374_a(z, x), tp374_b(z, x));
    a*a + b*b
}

impl NlpProblem for TP374 {
    fn num_variables(&self) -> usize { 10 }
    fn num_constraints(&self) -> usize { 35 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..10 { x_l[i] = f64::NEG_INFINITY; x_u[i] = f64::INFINITY; }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for i in 0..35 { g_l[i] = 0.0; g_u[i] = f64::INFINITY; }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        for i in 0..10 { x0[i] = 0.1; }
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 { x[9] }

    fn gradient(&self, _x: &[f64], grad: &mut [f64]) {
        for i in 0..9 { grad[i] = 0.0; }
        grad[9] = 1.0;
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        for i in 0..10 {
            let z = PI/4.0 * (i as f64 * 0.1);
            g[i] = tp374_g(z, x) - (1.0 - x[9]).powi(2);
        }
        for i in 10..20 {
            let z = PI/4.0 * ((i-10) as f64 * 0.1);
            g[i] = (1.0 + x[9]).powi(2) - tp374_g(z, x);
        }
        for i in 20..35 {
            let z = PI/4.0 * (1.2 + (i-20) as f64 * 0.2);
            g[i] = x[9].powi(2) - tp374_g(z, x);
        }
    }

    // ... jacobian and hessian implementations omitted for brevity
    // (see examples/debug_tp374.rs for the full implementation)
#   fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
#       let mut rows = Vec::new(); let mut cols = Vec::new();
#       for i in 0..35 { for j in 0..10 { rows.push(i); cols.push(j); } }
#       (rows, cols)
#   }
#   fn jacobian_values(&self, _x: &[f64], _v: &mut [f64]) { /* ... */ }
#   fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
#       let mut rows = Vec::new(); let mut cols = Vec::new();
#       for i in 0..10 { for j in 0..=i { rows.push(i); cols.push(j); } }
#       (rows, cols)
#   }
#   fn hessian_values(&self, _x: &[f64], _s: f64, _l: &[f64], _v: &mut [f64]) { /* ... */ }
}

fn main() {
    // Attempt 1: default options
    let opts1 = SolverOptions {
        print_level: 5,
        max_iter: 3000,
        ..SolverOptions::default()
    };
    let r1 = ripopt::solve(&TP374, &opts1);
    println!("Attempt 1: {:?}, obj={:.6}, iters={}", r1.status, r1.objective, r1.iterations);
    println!("  filter_rejects={}, restorations={}, mu_switches={}",
        r1.diagnostics.filter_rejects,
        r1.diagnostics.restoration_count,
        r1.diagnostics.mu_mode_switches);

    // Attempt 2: adjust based on diagnostics
    // If filter_rejects is high -> raise mu_init, lower kappa
    // If mu stuck -> try monotone mode
    let opts2 = SolverOptions {
        print_level: 5,
        max_iter: 3000,
        mu_init: 1.0,
        kappa: 3.0,
        hessian_approximation_lbfgs: true,
        ..SolverOptions::default()
    };
    let r2 = ripopt::solve(&TP374, &opts2);
    println!("Attempt 2: {:?}, obj={:.6}, iters={}", r2.status, r2.objective, r2.iterations);
}
```

**Why it's hard:** The 35 trigonometric inequality constraints create a
narrow feasible region with many near-degenerate active sets. The solver
accumulates filter rejects and restorations, and the barrier parameter
fails to decrease toward zero. Constraint multipliers grow to ~1e11,
signaling numerical instability.

**What diagnostics tell you:**
```
status: MaxIterations
iterations: 2999
filter_rejects: ~50+
restoration_count: ~10+
mu_mode_switches: ~20+
final_mu: ~1e-2 (stuck, should be ~1e-9)
final_primal_inf: ~1e-1 (not feasible)
```

**Strategies to try (informed by diagnostics):**

1. **High filter_rejects** -> Increase `mu_init` to 1.0 or 10.0, giving the
   solver more room for infeasible steps early on
2. **Restorations dominating** -> Enable `enable_slack_fallback: true` to
   reformulate inequalities with explicit slacks
3. **mu stuck** -> Try `mu_strategy_adaptive: false` for monotone decrease,
   or reduce `kappa` to slow mu reduction
4. **Large multipliers** -> Try `hessian_approximation_lbfgs: true` to avoid
   ill-conditioned exact Hessians
5. **Still stuck** -> Try a different starting point; the initial `x0 = 0.1`
   may be in a bad basin

---

## Starting Point Strategies

Interior point methods are local solvers. The starting point determines which
basin of attraction the solver lands in. Changing `x0` is often more effective
than tuning `SolverOptions`.

### When to try a new starting point

Look for these diagnostic patterns:

- `final_primal_inf` is large → solver never reached feasibility from `x0`
- `restoration_count` is high → solver kept losing feasibility, suggesting
  `x0` is far from the feasible region
- Status is `MaxIterations` with `final_mu` stuck → solver is cycling in a
  bad region

### How to change starting points

The mechanism depends on how the problem is defined:

**`.nl` file problems (AMPL/CUTEst):**

The starting point is embedded in the `.nl` file. Claude Code can:

1. **Edit the `.nl` file directly** — the initial `x` values appear in the
   `x` segment of the file. Claude can parse and rewrite them:

   ```bash
   # Solve with default x0
   ripopt_ampl problem.nl print_level=5

   # Claude reads diagnostics, decides to try new x0
   # Edits the x segment in problem.nl with new values
   # Re-solves
   ripopt_ampl problem.nl print_level=5
   ```

2. **Use the `.sol` file as warm-start** — after a failed solve, the `.sol`
   file contains the best point found. Copy it back as the new `.nl` starting
   point and re-solve with `warm_start_init_point=yes`:

   ```bash
   ripopt_ampl problem.nl print_level=5 warm_start_init_point=yes
   ```

**Rust-defined problems (examples, tests):**

The starting point is hardcoded in `initial_point()`. Claude Code edits the
source and recompiles (~2 seconds for ripopt). This is the simplest approach:

```
Claude Code:
  1. Runs `cargo run --example debug_tp374 2>&1`
  2. Reads diagnostics: final_primal_inf=0.12, restoration_count=8
  3. Reasons: "x0=[0.1,...,0.1] is far from feasible. The trig constraints
     need G(z,x) ≈ 1 at several z values. Let me try x0 that makes
     A(z,x) ≈ 1, B(z,x) ≈ 0."
  4. Edits initial_point() in the source:
       x0[0] = 1.0;  // cos(z) ≈ 1 at z=0
       for i in 1..9 { x0[i] = 0.0; }
       x0[9] = 0.3;   // near known optimal
  5. Runs `cargo run --example debug_tp374 2>&1`
  6. Reads new diagnostics, compares
```

**Environment variable override (optional pattern):**

For problems where you want to avoid recompilation, you can add a file-based
override to `initial_point()`:

```rust
fn initial_point(&self, x0: &mut [f64]) {
    // Default
    for i in 0..10 { x0[i] = 0.1; }

    // Override from file if RIPOPT_X0 is set
    if let Ok(path) = std::env::var("RIPOPT_X0") {
        if let Ok(contents) = std::fs::read_to_string(&path) {
            for (i, val) in contents.split_whitespace().enumerate() {
                if i < x0.len() {
                    if let Ok(v) = val.parse::<f64>() { x0[i] = v; }
                }
            }
        }
    }
}
```

Then Claude Code writes `x0.txt` and runs without recompiling:

```bash
echo "1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.3" > x0.txt
RIPOPT_X0=x0.txt cargo run --example debug_tp374
```

### Warm-start from previous result

The most natural multistart pattern: use `result.x` from a failed run as the
starting point for the next attempt. This avoids re-exploring territory the
solver already covered.

```rust
// Attempt 1: default x0
let r1 = ripopt::solve(&problem, &opts);

if r1.status != SolveStatus::Optimal {
    // Attempt 2: warm-start from r1's best point
    problem.set_x0(r1.x.clone());
    let mut opts2 = opts.clone();
    opts2.warm_start = true;
    // Resume near where r1 left off
    opts2.mu_init = r1.diagnostics.final_mu.max(1e-4);
    let r2 = ripopt::solve(&problem, &opts2);
}
```

For `.nl` problems via the AMPL interface, warm-start works automatically:
the `.sol` file from the first run becomes the starting point when you pass
`warm_start_init_point=yes`.

### Multistart strategies Claude Code can use

1. **Perturbed multistart** — take the default `x0`, add random or structured
   perturbations, try several. Keep the best result.

2. **Constraint-informed initialization** — analyze the constraint structure
   to pick `x0` that approximately satisfies some constraints. For TP374,
   this means choosing `x[0..9]` so that the trigonometric sums `G(z,x)`
   are in the right range.

3. **Two-phase approach** — first minimize constraint violation (set
   `objective ≡ 0` or use the restoration NLP), then use the feasible point
   as `x0` for the real problem.

4. **Scale-aware initialization** — if gradients at `x0` are huge
   (`obj_scaling` is very small in the diagnostics), the starting point may
   be in a steep region. Move `x0` toward where the gradient is moderate.

5. **Literature / domain knowledge** — for known problem classes (e.g.,
   optimal power flow, chemical equilibrium), there are standard
   initialization heuristics. Claude Code can look these up and apply them.

---

## Using Diagnostics with Claude Code

There are two ways to use the diagnostics-driven steering loop: interactively
inside a Claude Code session, or non-interactively from the command line with
`claude -p`.

### Mode 1: Interactive (inside Claude Code)

You're in a Claude Code session working on your project. You ask Claude to
solve a problem and it does the steering loop for you:

```
You:  "Run the TP374 example. If it doesn't converge, read the diagnostics
       and try adjusting options. Give it 3 attempts."

Claude Code:
  1. Runs `cargo run --example debug_tp374 2>&1`
  2. Reads the `--- ripopt diagnostics ---` block from stderr
  3. Sees: filter_rejects=47, restoration_count=8, final_mu=1.2e-2
  4. Edits examples/debug_tp374.rs: changes mu_init to 1.0, kappa to 3.0
  5. Re-runs, reads new diagnostics
  6. Reports back: "Attempt 2 reduced filter_rejects to 12 but still
     MaxIterations. Trying L-BFGS Hessian..."
  7. Edits again, re-runs
  8. Reports final result with comparison table
```

This is the natural workflow when you're developing interactively. Claude Code
has full context of the codebase, can edit source files, and can reason about
the diagnostic patterns across multiple attempts.

**When to use:** Exploratory work, debugging a specific problem, developing
new solver strategies.

### Mode 2: Non-interactive (`claude -p` from the shell)

You pipe a prompt to Claude Code from a script or the command line. Claude
runs autonomously and returns the result:

```bash
# Single problem, automated steering
claude -p "
  Run: cargo run --example debug_tp374 2>&1
  Parse the '--- ripopt diagnostics ---' block from stderr.
  If status is not Optimal:
    - Read the diagnostics pattern
    - Edit the SolverOptions in examples/debug_tp374.rs based on:
      * High filter_rejects -> increase mu_init, decrease kappa
      * High restoration_count -> try enable_slack_fallback
      * mu stuck high -> try mu_strategy_adaptive: false
      * Large multipliers -> try hessian_approximation_lbfgs: true
    - Re-run and compare
    - Try up to 3 adjustments
  Report the best result found.
"
```

You can also write a shell script that loops over multiple problems:

```bash
#!/bin/bash
# solve_batch.sh — run Claude Code steering on a list of problems
PROBLEMS="debug_tp374 hs071 rosenbrock"

for prob in $PROBLEMS; do
  echo "=== Solving $prob ==="
  claude -p "
    Run cargo run --example $prob 2>&1.
    If it converges (Optimal or Acceptable), report the result.
    If not, read the diagnostics block, adjust SolverOptions, and
    retry up to 3 times. Report what you tried and the best result.
  "
  echo ""
done
```

**When to use:** Batch testing, CI pipelines, automated benchmarking,
running the same steering strategy across many problems.

### Architecture

In both modes, the architecture is the same:

```
ripopt (Rust)                    Claude Code (LLM)
  |                                  |
  |  solve(&problem, &options)       |
  |  -> prints iteration table       |
  |  -> prints diagnostics block     |
  |  -> returns SolveResult          |
  |                                  |
  |  stderr -----> reads output ---->|
  |                                  |  reasons about patterns
  |                                  |  decides new options
  |                                  |  edits SolverOptions
  |  <----- re-runs with new opts <--|
  |                                  |
  (repeat until converged or budget exhausted)
```

The Rust code is purely a reporter. It has no knowledge of Claude. All
intelligence — pattern matching, strategy selection, option adjustment —
lives in Claude Code's reasoning. The `SolverDiagnostics` struct just
makes the solver's internal state visible in a structured, parseable format.

### Programmatic access (Rust code)

If you're writing Rust code that calls ripopt, you can also read diagnostics
directly without parsing stderr:

```rust
let result = ripopt::solve(&problem, &options);

if result.status != SolveStatus::Optimal {
    let d = &result.diagnostics;

    // Decide next options based on diagnostics
    let mut opts2 = options.clone();
    if d.filter_rejects > 5 {
        opts2.mu_init = 1.0;
        opts2.kappa = 3.0;
    }
    if d.restoration_count > 3 {
        opts2.enable_slack_fallback = true;
    }
    if d.final_mu > 1e-4 {
        opts2.mu_strategy_adaptive = false;
    }

    let result2 = ripopt::solve(&problem, &opts2);
}
```

This is useful for building automated tuning loops or integration tests
that adapt options based on solver behavior.

---

## SolverOptions Reference

The key knobs for steering, grouped by what they control:

### Barrier parameter

| Option | Default | Effect |
|---|---|---|
| `mu_init` | 0.1 | Higher = more room for infeasible exploration |
| `mu_min` | 1e-11 | Floor for barrier parameter |
| `kappa` | 10.0 | Lower = slower mu decrease (more conservative) |
| `mu_strategy_adaptive` | true | false = monotone decrease (simpler, sometimes better) |
| `mu_linear_decrease_factor` | 0.2 | Controls monotone mu reduction speed |
| `mu_superlinear_decrease_power` | 1.5 | Exponent for superlinear mu decrease |
| `barrier_tol_factor` | 10.0 | Subproblem tolerance = this * mu |

### Convergence

| Option | Default | Effect |
|---|---|---|
| `tol` | 1e-8 | Optimality tolerance |
| `acceptable_tol` | 1e-4 | Relaxed tolerance for acceptable convergence |
| `acceptable_iter` | 10 | Consecutive acceptable iterations needed |
| `max_iter` | 3000 | Iteration budget |
| `max_wall_time` | 0.0 | Wall-clock limit in seconds (0 = unlimited) |

### Line search and corrections

| Option | Default | Effect |
|---|---|---|
| `max_soc` | 4 | Max second-order corrections per step |
| `watchdog_shortened_iter_trigger` | 10 | Consecutive short steps before watchdog |
| `watchdog_trial_iter_max` | 3 | Watchdog trial iterations |

### Hessian strategy

| Option | Default | Effect |
|---|---|---|
| `hessian_approximation_lbfgs` | false | true = L-BFGS instead of exact Hessian |
| `enable_lbfgs_hessian_fallback` | true | Auto-retry with L-BFGS if exact Hessian fails |

### Fallback strategies

| Option | Default | Effect |
|---|---|---|
| `enable_slack_fallback` | true | Reformulate inequalities with explicit slacks |
| `enable_al_fallback` | true | Augmented Lagrangian for constrained problems |
| `enable_sqp_fallback` | true | SQP for small constrained problems |
| `enable_lbfgs_fallback` | true | L-BFGS for unconstrained problems |

### Advanced

| Option | Default | Effect |
|---|---|---|
| `mehrotra_pc` | false | Mehrotra predictor-corrector (fewer iterations) |
| `gondzio_mcc_max` | 0 | Gondzio centrality corrections (better centering) |
| `warm_start` | false | Reuse previous solution as starting point |
| `enable_preprocessing` | true | Eliminate fixed vars and redundant constraints |
| `detect_linear_constraints` | true | Skip Hessian for linear constraints |
