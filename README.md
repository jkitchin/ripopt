# ripopt

[![Crates.io](https://img.shields.io/crates/v/ripopt.svg)](https://crates.io/crates/ripopt)
[![PyPI - ripopt](https://img.shields.io/pypi/v/ripopt.svg?label=pypi%20ripopt)](https://pypi.org/project/ripopt/)
[![PyPI - pyomo-ripopt](https://img.shields.io/pypi/v/pyomo-ripopt.svg?label=pypi%20pyomo-ripopt)](https://pypi.org/project/pyomo-ripopt/)
[![Tests](https://github.com/jkitchin/ripopt/actions/workflows/test.yml/badge.svg)](https://github.com/jkitchin/ripopt/actions/workflows/test.yml)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19542664-blue.svg)](https://doi.org/10.5281/zenodo.19542664)

![img](./ipopt-rust.png)

A memory-safe interior point optimizer written in Rust, inspired by [Ipopt](https://github.com/coin-or/Ipopt).

## What is this?

ripopt solves nonlinear programming (NLP) problems of the form:

```
min  f(x)
s.t. g_l <= g(x) <= g_u
     x_l <= x    <= x_u
```

It implements a primal-dual interior point method with a barrier formulation, similar to the algorithm described in the Ipopt papers. The solver is written entirely in Rust (~21,700 lines) with no external C/Fortran dependencies.

## Features

- Primal-dual interior point method with logarithmic barrier
- Dense LDL^T factorization via Bunch-Kaufman pivoting with inertia detection
- Sparse multifrontal LDL^T factorization (via [`feral`](https://crates.io/crates/feral) with Bunch-Kaufman 1×1/2×2 pivoting, MC64 scaling, AMD/METIS ordering, and certified inertia) for larger problems (n+m >= 110). The legacy `rmumps` backend is available behind the opt-in `rmumps` feature: `cargo build --no-default-features --features "rmumps faer"`
- Banded LDL^T solver for problems with detected-banded structure (e.g., PDE discretizations)
- Dense condensed KKT (Schur complement) for tall-narrow problems (m >> n, n <= 100)
- Sparse condensed KKT for reducing system size when m > 0
- Filter line search with switching condition and Armijo criterion
- Second-order corrections (SOC) for improved step acceptance
- **Mehrotra predictor-corrector** with Gondzio centrality corrections (enabled by default)
- Adaptive and monotone barrier parameter strategies with Mehrotra sigma-guided mu updates
- Fraction-to-boundary rule for primal and dual step sizes
- Support for equality constraints, inequality constraints, and variable bounds
- Warm-start initialization
- Two-phase restoration: fast Gauss-Newton + full NLP restoration subproblem
- Multi-attempt recovery with systematic barrier landscape perturbation
- Watchdog strategy for escaping narrow feasible corridors
- Automatic NE-to-LS reformulation for overdetermined nonlinear equation systems
- **Convergence polishing**: Newton polish for NE-to-LS, complementarity snap for IPM
- NLP scaling (gradient-based objective and constraint scaling)
- Local infeasibility detection for inconsistent constraint systems
- **Early stall detection**: bail out fast when stuck in early iterations to trigger fallbacks
- **Preprocessing**: Automatic auxiliary equality-block reduction/recovery, fixed-variable elimination, redundant-constraint removal, and bound tightening from single-variable linear constraints
- **Near-linear constraint detection**: Automatically identifies linear constraints and skips their Hessian contribution
- **Limited-memory Hessian approximation**: L-BFGS-in-IPM mode (`hessian_approximation_lbfgs`) replaces exact Hessian with L-BFGS curvature pairs, eliminating the need for second-derivative callbacks
- **Multi-solver fallback architecture**: L-BFGS, Augmented Lagrangian, SQP, and explicit slack reformulation
- **TRON fast path for bound-constrained problems** (`solve_bc`, [Lin & Moré 1999](https://doi.org/10.1137/S1052623498345075)): trust-region projected-Newton with truncated CG. For pure bound-constrained NLPs (`m == 0`, finite bounds) it often converges in 1–3 iterations on convex/quadratic objectives where the IPM needs 5–8. Explicit entry point — not auto-dispatched. See [Bound-constrained fast path](#bound-constrained-fast-path-tron).
- **Parametric sensitivity analysis**: sIPOPT-style post-optimal sensitivity (`ds/dp = -M⁻¹ · Nₚ`) for computing how the optimal solution changes under parameter perturbations, plus reduced Hessian extraction for covariance estimation
- **C API** mirroring the Ipopt C interface for direct linking from C/C++/Python/Julia
- **AMPL NL interface** with Pyomo integration via `SolverFactory('ripopt')`, with `--help` listing all options. Supports AMPL **external functions** via the `funcadd_ASL` ABI (e.g. IDAES `cbrt`, custom property-package libraries) at solve time.
- **GAMS solver link** enabling `option nlp = ripopt;` in GAMS models via the GMO API
- **Julia/JuMP interface** (`Ripopt.jl`) via MathOptInterface, enabling `Model(Ripopt.Optimizer)` with full JuMP support

## Benchmarks

### Hock-Schittkowski Test Suite (120 problems) — retired

> The standalone HS suite has been retired from the ripopt benchmark harness;
> the numbers below are historical (pre-v0.8) and are no longer regenerated.
> HS-family problems remain exercised individually through CUTEst.

| Metric          | ripopt             | Ipopt (native, MUMPS) |
|-----------------|--------------------|-----------------------|
| Problems solved | **118/120 (98.3%)**| 116/120 (96.7%)       |
| Optimal         | 118                | 116                   |
| ripopt only     | 2                  | --                    |
| Ipopt only      | --                 | 0                     |

On 116 commonly-solved problems: **20.8x geometric mean speedup**, median 20.9x, ripopt faster on 114/116 (98%).

### CUTEst Benchmark Suite (727 problems)

| Metric        | ripopt              | Ipopt (C++ with MUMPS) |
|---------------|---------------------|------------------------|
| Total solved  | 564/727 (77.6%)     | 561/727 (77.2%)        |
| Both solve    | 544                 | 544                    |
| ripopt only   | 20                  | --                     |
| Ipopt only    | --                  | 17                     |

ripopt edges out native Ipopt on CUTEst strict-Optimal at v0.8.0 by three problems (564 vs 561). The v0.8 cycle replaced rmumps with the pure-Rust [`feral`](https://crates.io/crates/feral) LDLᵀ solver and aligned the IPM kernel with Ipopt 3.14 (post-restoration handoff, AugmentFilter, μ-update oracles); the dominant remaining failure mode is `RestorationFailed` (74 cases). See CHANGELOG and the manuscript for details.

On 544 commonly-solved problems:

| Metric                          | Value            |
|---------------------------------|------------------|
| Geometric mean speedup          | **8.9x**         |
| Median speedup                  | (see manuscript) |
| Problems where ripopt is faster | majority of 544  |

**Interpreting the speed numbers.** Most CUTEst problems are small (n < 10) and solve in microseconds for ripopt, while Ipopt has a ~1-3ms floor from internal initialization. The per-iteration speedup on small problems comes from stack allocation, the absence of C/Fortran interop, and cache-efficient dense linear algebra. On larger problems, ripopt switches to sparse multifrontal LDL^T with SuiteSparse AMD ordering, and Ipopt's Fortran MUMPS has a per-factorization advantage. Ipopt uses fewer iterations on average on CUTEst (ripopt mean 62.3 vs Ipopt 39.5), reflecting its more mature barrier parameter tuning.

The speed advantage comes from:

1. **Lower per-iteration overhead.** ripopt's dense Bunch-Kaufman factorization avoids sparse symbolic analysis and has minimal allocation. For small-to-medium problems (n < 50), this gives 2-5x per-iteration speedup.
2. **Dense condensed KKT for tall-narrow problems.** When m >> n with n <= 100, ripopt reduces an (n+m)x(n+m) sparse factorization to an nxn dense solve, giving 100-800x speedup on problems like EXPFITC (n=5, m=502) and OET3 (n=4, m=1002).
3. **Mehrotra predictor-corrector with Gondzio corrections.** Enabled by default, reducing iteration counts on many problems.
4. **Fewer iterations on some problems.** NE-to-LS reformulation, two-phase restoration, and multi-solver fallback recover problems that Ipopt cannot solve.

Where Ipopt is faster:

1. **Large sparse problems.** Ipopt's Fortran MUMPS is ~10-15x faster per factorization than ripopt's pure-Rust feral on 50K-100K systems.
2. **Some medium constrained problems.** A handful of problems (CORE1, HAIFAM, NET1) have high per-iteration cost in ripopt's line search or fallback cascade.
3. **Some difficult nonlinear problems.** Ipopt's mature barrier parameter tuning gives it an edge on specific hard problems.

### Large-Scale Benchmarks

Both solvers receive the exact same NlpProblem struct via the Rust trait interface, ensuring a fair comparison. ripopt uses feral (pure-Rust multifrontal LDL^T with Bunch-Kaufman pivoting and AMD/METIS ordering); Ipopt uses MUMPS (Fortran).

| Problem         | n      | m      | ripopt  | time    | Ipopt    | time    | speedup   |
|-----------------|--------|--------|---------|---------|----------|---------|-----------|
| Rosenbrock 500  | 500    | 0      | Optimal | 0.003s  | Optimal  | 0.199s  | **76.2x** |
| Bratu 1K        | 1,000  | 998    | Optimal | 0.002s  | Optimal  | 0.002s  | 1.1x      |
| SparseQP 1K     | 500    | 500    | Optimal | 0.176s  | Optimal  | 0.004s  | 0.02x     |
| OptControl 2.5K | 2,499  | 1,250  | Optimal | 0.006s  | Optimal  | 0.002s  | 0.4x      |

Numbers above are from the v0.7 sweep. The v0.8 cycle replaced the
sparse linear solver (rmumps → feral) and aligned the IPM kernel
with Ipopt 3.14; aggregate CUTEst pass rate went up (564 vs 561) and
the geomean speedup on commonly-solved problems came down to 8.9x as
feral has not yet matched MUMPS's GEMM throughput on the largest
sparse fronts. See the manuscript for the full v0.8 analysis.
Historical numbers from v0.6.2 are preserved in
`benchmarks/large_scale/large_scale_results.txt` snapshots. ripopt's
advantage is strongest on unconstrained problems (L-BFGS fallback);
on large constrained problems Ipopt's Fortran MUMPS is ~10-15x faster
per factorization.

Run the benchmarks yourself: `make benchmark`

### Domain-Specific Benchmarks

| Suite                        | Problems | ripopt           | Ipopt         | Notes                                                              |
|------------------------------|----------|------------------|---------------|--------------------------------------------------------------------|
| Electrolyte thermodynamics   | 13       | **13/13 (100%)** | 12/13 (92.3%) | 17.5x geo mean speedup; ripopt uniquely solves seawater speciation |
| Grid (AC Optimal Power Flow) | 4        | 3/4 (75%)        | **4/4 (100%)**| 2.8x geo mean on 3 commonly-solved; case30_ieee regression         |
| CHO parameter estimation     | 1        | 0/1              | 0/1           | Large-scale (n=21,672, m=21,660); both hit iteration limit         |
| Gas pipeline NLPs            | 4        | see suite README | see suite README | PDE-discretized Euler equations on pipe networks (gaslib11/40, steady/dynamic). Standalone — does not feed `BENCHMARK_REPORT.md` |
| Water distribution NLPs      | 6        | see suite README | see suite README | MINLPLib water network design instances (Hazen-Williams head-loss). Standalone — does not feed `BENCHMARK_REPORT.md` |

Run all benchmarks: `make benchmark`

## Installation

### Prerequisites: Rust and Cargo

ripopt is written in Rust. You need the Rust toolchain (compiler + Cargo build tool) installed.

**Install Rust via rustup** (the official installer, works on macOS, Linux, and WSL):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Follow the prompts (the defaults are fine). Then restart your shell or run:

```bash
source "$HOME/.cargo/env"
```

Verify the installation:

```bash
rustc --version
cargo --version
```

For other installation methods (Homebrew, distro packages, Windows), see the [official Rust installation guide](https://www.rust-lang.org/tools/install).

### Install ripopt

**From crates.io:**

```bash
cargo install ripopt
```

This installs the `ripopt` AMPL solver binary to `~/.cargo/bin/`.

To use ripopt as a library dependency in your Rust project:

```bash
cargo add ripopt
```

**From source** (clone the repository and run `make install`):

```bash
git clone https://github.com/jkitchin/ripopt.git
cd ripopt
make install
```

This does three things:

1. Builds the optimized release binary and shared library
2. Installs the `ripopt` AMPL solver binary to `~/.cargo/bin/` (which rustup already added to your `$PATH`)
3. Copies the shared library (`libripopt.dylib` on macOS, `libripopt.so` on Linux) to `~/.local/lib/`

> **PATH check:** The `ripopt` binary is installed to `~/.cargo/bin/`. If `ripopt --version` doesn't work after installation, make sure `~/.cargo/bin` is on your `$PATH` by adding this to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):
> ```bash
> export PATH="$HOME/.cargo/bin:$PATH"
> ```
> Then restart your shell or run `source ~/.bashrc` (or `~/.zshrc`).

> **Shared library:** If you use the C API, ensure `~/.local/lib` is in your library path:
> ```bash
> export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
> ```

After installation, verify it works:

```bash
ripopt --version
```

### Using ripopt with Pyomo

Once `ripopt` is on your `$PATH` (the `make install` step above handles this), install the Pyomo solver plugin:

```bash
pip install ./pyomo-ripopt
```

This registers `ripopt` as a named solver with Pyomo's `SolverFactory`:

```python
from pyomo.environ import *

model = ConcreteModel()
# ... define your model ...

solver = SolverFactory('ripopt')
result = solver.solve(model, tee=True)
```

> **Note:** If you skip the `pip install` step, you can still use ripopt via the generic AMPL interface with `SolverFactory('asl:ripopt')`, as long as the `ripopt` binary is on your `$PATH`.

### Using ripopt as a Rust library

Add to your project's `Cargo.toml`:

```toml
[dependencies]
ripopt = { git = "https://github.com/jkitchin/ripopt" }
```

Or for a local checkout:

```toml
[dependencies]
ripopt = { path = "/path/to/ripopt" }
```

### Using the shared library (C/Python ctypes/Julia FFI)

After `make install`, the shared library is at `~/.local/lib/libripopt.dylib` (macOS) or `~/.local/lib/libripopt.so` (Linux). The C header `ripopt.h` is in the repository root.

```bash
# Compile a C program against the installed library
cc my_program.c -I/path/to/ripopt -L~/.local/lib -lripopt -lm
```

Or link directly from the build directory without installing:

```bash
cargo build --release
cc my_program.c -I. -Ltarget/release -lripopt \
   -Wl,-rpath,$(pwd)/target/release -o my_program -lm
```

### Using ripopt with GAMS

ripopt includes a GAMS solver link (`gams/`) that bridges between GAMS's GMO API and ripopt's C API. This allows GAMS models to use ripopt as an NLP solver via `option nlp = ripopt;`.

**Build and install** (requires a GAMS installation):

```bash
cargo build --release
make -C gams
sudo make -C gams install   # copies libs to GAMS sysdir, registers in gmscmpun.txt
```

**Use in a GAMS model:**

```gams
option nlp = ripopt;
Solve mymodel using nlp minimizing obj;
```

**Solver options** are set via a `ripopt.opt` file (same key-value format as Ipopt):

```
tol 1e-8
max_iter 1000
print_level 5
```

GAMS iteration and resource limits (`option iterlim`, `option reslim`) are automatically forwarded. The solver link supports NLP, DNLP, and RMINLP model types. When the analytical Hessian is not available (e.g., DNLP models), it automatically falls back to L-BFGS approximation.

**Test:**

```bash
sudo make -C gams test   # solves HS071 and checks the result
```

### Using ripopt with Julia/JuMP

`Ripopt.jl` provides a [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) (MOI) wrapper so ripopt can be used as a drop-in JuMP optimizer.

**Prerequisites:** Julia ≥ 1.9, JuMP, and the ripopt shared library.

**Install once** (adds Ripopt.jl to your global Julia environment):

```bash
cargo build --release      # build libripopt.dylib / libripopt.so

julia -e '
import Pkg
Pkg.develop(path="Ripopt.jl")   # or Pkg.add(url="...") for a remote install
'
```

**Use in a script or notebook:**

```julia
ENV["RIPOPT_LIBRARY_PATH"] = "/path/to/ripopt/target/release"
using JuMP, Ripopt

model = Model(Ripopt.Optimizer)
set_silent(model)

@variable(model, 1 <= x[1:4] <= 5)
set_start_value.(x, [1.0, 5.0, 5.0, 1.0])

@NLobjective(model, Min, x[1]*x[4]*(x[1]+x[2]+x[3]) + x[3])
@NLconstraint(model, x[1]*x[2]*x[3]*x[4] >= 25)
@NLconstraint(model, x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 == 40)

optimize!(model)
println(termination_status(model))   # LOCALLY_SOLVED
println(objective_value(model))      # ≈ 17.014
println(value.(x))                   # ≈ [1.0, 4.743, 3.821, 1.379]
```

Or run the provided examples from the repo root:

```bash
RIPOPT_LIBRARY_PATH=target/release julia --project=@v1.12 Ripopt.jl/examples/jump_hs071.jl
RIPOPT_LIBRARY_PATH=target/release julia --project=@v1.12 Ripopt.jl/examples/jump_rosenbrock.jl
RIPOPT_LIBRARY_PATH=target/release julia --project=@v1.12 Ripopt.jl/examples/c_wrapper_hs071.jl
```

Solver options use the same names as Ipopt:

```julia
set_optimizer_attribute(model, "tol", 1e-10)
set_optimizer_attribute(model, "max_iter", 500)
set_optimizer_attribute(model, "mu_strategy", "adaptive")
set_time_limit_sec(model, 60.0)
```

**Compatibility notes** for users coming from Ipopt:

- ripopt accepts the same option names as Ipopt for the options it implements. Options ripopt does not implement are silently ignored by some host frontends; the C API and the AMPL CLI emit a warning to stderr listing the unknown option name.
- Time limits: ripopt enforces a wall-clock budget via `max_wall_time`. Ipopt's default `max_cpu_time` is accepted as an alias for `max_wall_time` (with a one-time stderr warning) so existing Ipopt scripts keep working — note that the limit is enforced as wall-clock, not CPU time.

Switching between ripopt and Ipopt requires only changing the optimizer constructor; the rest of the model is identical:

```julia
# With ripopt
model = Model(Ripopt.Optimizer)

# With Ipopt (if installed)
using Ipopt
model = Model(Ipopt.Optimizer)
```

### Uninstall

```bash
make uninstall
```

## Usage

### Defining a Problem

Implement the `NlpProblem` trait:

```rust
use ripopt::NlpProblem;

struct Rosenbrock;

impl NlpProblem for Rosenbrock {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        // Unconstrained: use infinity bounds
        for i in 0..2 {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }

    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = -1.0; x0[1] = 1.0; }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
        grad[1] = 200.0 * (x[1] - x[0] * x[0]);
    }

    fn constraints(&self, _x: &[f64], _g: &mut [f64]) {}
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _vals: &mut [f64]) {}

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1], vec![0, 0, 1])  // lower triangle
    }

    fn hessian_values(&self, x: &[f64], obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (-400.0 * (x[1] - 3.0 * x[0] * x[0]) + 2.0);
        vals[1] = obj_factor * (-400.0 * x[0]);
        vals[2] = obj_factor * 200.0;
    }
}
```

### Solving

```rust
use ripopt::{SolverOptions, solve};

let problem = Rosenbrock;
let options = SolverOptions::default();
let result = solve(&problem, &options);

println!("Status: {:?}", result.status);
println!("Objective: {:.6e}", result.objective);
println!("Solution: {:?}", result.x);
println!("Iterations: {}", result.iterations);
```

### Bound-constrained fast path (TRON)

For problems with **only variable bounds** (`num_constraints() == 0` and at least
one finite bound), an alternative entry point implements Lin & Moré's TRON:

```rust
use ripopt::bc_solver::{solve_bc, BcOptions};

let result = solve_bc(&problem, &BcOptions::default());
```

**When to use it**

- You know up front that `m == 0`. TRON cannot handle equality or general
  inequality constraints; it returns `InternalError` if `m != 0`.
- The objective is well-conditioned (small to moderate `n`, smooth `f`). On
  convex/quadratic BC problems TRON typically takes 1–3 iterations vs the
  IPM's 5–8 (no barrier sequence to walk).
- You want a status of `Optimal` rather than the IPM's `Acceptable` on
  problems where the IPM stalls in the centering phase.

**When to prefer the IPM (default `solve()`)**

- Highly nonlinear, ill-conditioned BC problems (e.g. PALMER nonlinear
  least-squares family). TRON converges to the same local minima but takes
  more iterations than the IPM there because the inner CG is currently plain
  (no incomplete-Cholesky preconditioning).
- Multimodal landscapes — TRON and the IPM can converge to different local
  optima from the same `x0`. There is no auto-dispatch precisely so this
  choice stays explicit.
- Anything with general constraints.

**Algorithm summary**

Per outer iteration:
1. Cauchy step along the projected steepest-descent path `P[x − α·g]` with
   piecewise-quadratic search and trust-region clipping.
2. Steihaug-Toint truncated CG on the free variables after the Cauchy point;
   terminates on residual reduction, negative curvature, trust-region
   boundary, or a bound being hit.
3. Trust-region acceptance via the actual/predicted-reduction ratio ρ, with
   shrink/expand factors γ₁=0.25 / γ₂=4.0 at thresholds η₁=0.25 / η₂=0.75.

See `src/bc_solver.rs` for `BcOptions` (tolerance, TR parameters, CG cap,
print level).

### Solver Options

Key options (all have Ipopt-matching defaults):

| Option                          | Default | Description                                                |
|---------------------------------|---------|------------------------------------------------------------|
| `tol`                           | 1e-8    | Convergence tolerance                                      |
| `max_iter`                      | 3000    | Maximum iterations                                         |
| `mu_init`                       | 0.1     | Initial barrier parameter                                  |
| `print_level`                   | 5       | Output verbosity (0=silent, 5=verbose)                     |
| `mu_strategy_adaptive`          | true    | Adaptive vs monotone barrier update                        |
| `max_soc`                       | 4       | Maximum second-order correction steps                      |
| `max_wall_time`                 | 0.0     | Wall-clock time limit in seconds (0=no limit). Ipopt's `max_cpu_time` is accepted as an alias. |
| `warm_start`                    | false   | Enable warm-start initialization                           |
| `constr_viol_tol`               | 1e-4    | Constraint violation tolerance                             |
| `dual_inf_tol`                  | 1.0     | Dual infeasibility tolerance                               |
| `enable_preprocessing`          | true    | Internal auxiliary equality solves/recovery, fixed variables, redundancies |
| `auxiliary_tol`                 | 1e-8    | Accepted residual for internal auxiliary equality solves/recovery |
| `detect_linear_constraints`     | true    | Skip Hessian for linear constraints                        |
| `enable_sqp_fallback`           | true    | SQP fallback for constrained problems                      |
| `hessian_approximation_lbfgs`   | false   | Use L-BFGS Hessian approximation (no exact Hessian needed) |
| `enable_lbfgs_hessian_fallback` | true    | Auto-retry with L-BFGS Hessian when exact Hessian fails    |
| `mehrotra_pc`                   | true    | Mehrotra predictor-corrector for better centering          |
| `gondzio_mcc_max`               | 3       | Maximum Gondzio centrality corrections per iteration       |
| `early_stall_timeout`           | 120.0   | Max seconds for first 3 iterations (0=off)                 |
| `linear_solver`                 | direct  | KKT solver: direct, iterative (MINRES), or hybrid          |

### Result

`SolveResult` contains:

- `x` -- optimal primal variables
- `objective` -- optimal objective value f(x*)
- `constraint_multipliers` -- Lagrange multipliers for constraints (y)
- `bound_multipliers_lower` / `bound_multipliers_upper` -- bound multipliers (z_L, z_U)
- `constraint_values` -- constraint values g(x*)
- `status` -- one of: `Optimal`, `Infeasible`, `LocalInfeasibility`, `MaxIterations`, `NumericalError`, `Unbounded`, `RestorationFailed`, `InternalError`
- `iterations` -- number of IPM iterations
- `diagnostics` -- `SolverDiagnostics`, including residuals, timing/evaluation counters, fallback information, and nested `diagnostics.preprocessing` phase data. The same diagnostics object is included in CLI JSON reports.

## C API

ripopt exposes a C API that mirrors the [Ipopt C interface](https://coin-or.github.io/Ipopt/INTERFACES.html#INTERFACE_C), enabling direct linking from C, C++, Python (`ctypes`/`cffi`), Julia, and any language with C FFI support — without the subprocess/file overhead of the NL interface. If you have existing Ipopt C code, migrating to ripopt requires only header/function renaming; the callback signatures are identical.

### Build the shared library

```bash
cargo build --release
# produces target/release/libripopt.dylib (macOS) or libripopt.so (Linux)
```

### C header

Include `ripopt.h` (repo root) in your C project. It defines version macros, callback typedefs, return status codes, and all public functions:

```c
#include "ripopt.h"

// Check version at compile time
printf("ripopt %s\n", RIPOPT_VERSION);  // "0.8.0"
```

### Callback signatures

The five callback types are identical to the Ipopt C interface. All callbacks return `1` on success, `0` on error (the solver will abort if a callback returns `0`):

```c
typedef int (*Eval_F_CB)   (int n, const double *x, int new_x,
                             double *obj_value, void *user_data);
typedef int (*Eval_Grad_F_CB)(int n, const double *x, int new_x,
                              double *grad_f, void *user_data);
typedef int (*Eval_G_CB)   (int n, const double *x, int new_x,
                             int m, double *g, void *user_data);
typedef int (*Eval_Jac_G_CB)(int n, const double *x, int new_x,
                              int m, int nele_jac,
                              int *iRow, int *jCol, double *values,
                              void *user_data);
typedef int (*Eval_H_CB)   (int n, const double *x, int new_x,
                             double obj_factor,
                             int m, const double *lambda, int new_lambda,
                             int nele_hess,
                             int *iRow, int *jCol, double *values,
                             void *user_data);
```

**Two-call protocol for Jacobian and Hessian:** When `values == NULL`, fill `iRow`/`jCol` with the sparsity pattern (0-based indexing); when `values != NULL`, fill numerical values in the same element order as the pattern. The Hessian uses the **lower triangle** only.

**Sign convention:** ripopt uses the Ipopt convention L = f(x) + y^T g(x). The Hessian callback receives `obj_factor` and `lambda` and should compute `obj_factor * ∇²f + Σ lambda[i] * ∇²g_i`.

### Lifecycle

```c
#include "ripopt.h"

// 1. Create handle
RipoptProblem nlp = ripopt_create(
    n, x_l, x_u,          // variable bounds (use ±1e30 for ±∞)
    m, g_l, g_u,           // constraint bounds (g_l == g_u for equality)
    nele_jac, nele_hess,   // number of nonzeros
    eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h);

// 2. Set options (Ipopt-compatible key names)
ripopt_add_int_option(nlp, "print_level", 5);
ripopt_add_num_option(nlp, "tol",         1e-8);
ripopt_add_str_option(nlp, "mu_strategy", "adaptive");

// 3. Solve  (x: in = initial point, out = solution)
double obj_val;
int status = ripopt_solve(nlp, x, NULL, &obj_val,
                          NULL, NULL, NULL, NULL);
// status == 0  →  RIPOPT_SOLVE_SUCCEEDED

// 4. Free
ripopt_free(nlp);
```

For unconstrained problems, pass `m=0` and `NULL` for `g_l`/`g_u`.

**Infinity bounds:** Use `HUGE_VAL` (from `<math.h>`) for "no bound". Internally, any value beyond `±1e19` is treated as unbounded. Avoid using finite large values like `1e30` — they may cause numerical issues.

### Extracting multipliers

All output pointers except `x` are optional (pass `NULL` to skip). Here is how to extract the full solution including Lagrange multipliers and bound multipliers:

```c
double x[4]      = {1.0, 5.0, 5.0, 1.0};  // initial point
double obj_val   = 0.0;
double g[2]      = {0.0, 0.0};              // constraint values at solution
double mult_g[2] = {0.0, 0.0};              // constraint multipliers (lambda)
double mult_xl[4]= {0.0, 0.0, 0.0, 0.0};   // lower bound multipliers (z_L)
double mult_xu[4]= {0.0, 0.0, 0.0, 0.0};   // upper bound multipliers (z_U)

int status = ripopt_solve(nlp, x, g, &obj_val,
                          mult_g, mult_xl, mult_xu,
                          NULL);  // user_data

// At the solution:
// - x[]       contains the optimal primal variables
// - obj_val   is f(x*)
// - g[]       contains g(x*) — verify constraints are satisfied
// - mult_g[]  contains the Lagrange multipliers for constraints
//             (nonzero for active constraints)
// - mult_xl[] contains z_L (positive when x is at its lower bound)
// - mult_xu[] contains z_U (positive when x is at its upper bound)
```

The `user_data` pointer is forwarded to every callback unchanged — use it to pass problem-specific data (e.g., model parameters) without globals.

### Return status

| Code | Enum constant                          | Meaning                                         |
|------|----------------------------------------|-------------------------------------------------|
| 0    | `RIPOPT_SOLVE_SUCCEEDED`               | Converged to optimal solution                   |
| 2    | `RIPOPT_INFEASIBLE_PROBLEM`            | Problem is locally infeasible                   |
| 5    | `RIPOPT_MAXITER_EXCEEDED`              | Reached iteration limit                         |
| 6    | `RIPOPT_RESTORATION_FAILED`            | Feasibility restoration failed                  |
| 7    | `RIPOPT_ERROR_IN_STEP_COMPUTATION`     | Numerical difficulties                          |
| 10   | `RIPOPT_NOT_ENOUGH_DEGREES_OF_FREEDOM` | Problem has too few free variables              |
| 11   | `RIPOPT_INVALID_PROBLEM_DEFINITION`    | Problem appears unbounded                       |
| -1   | `RIPOPT_INTERNAL_ERROR`                | Internal error                                  |

Status 0 indicates a successful solve. All others indicate failure — check your problem formulation, initial point, or try adjusting options.

### Options reference

Option-setting functions return `1` on success, `0` if the keyword is unknown. All option keywords match Ipopt naming conventions.

**Numeric options** (`ripopt_add_num_option`):

| Option                       | Default | Description                                     |
|------------------------------|---------|-------------------------------------------------|
| `tol`                        | 1e-8    | Convergence tolerance                           |
| `mu_init`                    | 0.1     | Initial barrier parameter                       |
| `mu_min`                     | 1e-11   | Minimum barrier parameter                       |
| `bound_push`                 | 1e-2    | Initial bound push                              |
| `bound_frac`                 | 1e-2    | Initial bound fraction                          |
| `constr_viol_tol`            | 1e-4    | Constraint violation tolerance                  |
| `dual_inf_tol`               | 1.0     | Dual infeasibility tolerance                    |
| `compl_inf_tol`              | 1e-4    | Complementarity tolerance                       |
| `auxiliary_tol`              | 1e-8    | Residual tolerance for internal auxiliary equality solves/recovery |
| `max_wall_time`              | 0.0     | Wall-clock time limit in seconds (0 = no limit). Ipopt's `max_cpu_time` is accepted as an alias. |
| `warm_start_bound_push`      | 1e-3    | Warm-start bound push                           |
| `warm_start_bound_frac`      | 1e-3    | Warm-start bound fraction                       |
| `warm_start_mult_bound_push` | 1e-3    | Warm-start multiplier push                      |
| `nlp_lower_bound_inf`        | -1e20   | Threshold for -infinity bounds                  |
| `nlp_upper_bound_inf`        | 1e20    | Threshold for +infinity bounds                  |
| `kappa`                      | 10.0    | Adaptive mu divisor                             |
| `constr_mult_init_max`       | 1000.0  | Max initial constraint multiplier               |
| `barrier_tol_factor`         | 10.0    | Barrier tolerance factor                        |

**Integer options** (`ripopt_add_int_option`):

| Option                 | Default | Description                                            |
|------------------------|---------|--------------------------------------------------------|
| `max_iter`             | 3000    | Maximum iterations                                     |
| `print_level`          | 5       | Output verbosity (0 = silent, 5 = verbose, 12 = debug) |
| `max_soc`              | 4       | Maximum second-order correction steps                  |
| `sparse_threshold`     | 110     | KKT dimension threshold for sparse solver              |
| `restoration_max_iter` | 200     | Max iterations in NLP restoration subproblem           |

**String options** (`ripopt_add_str_option`):

| Option                          | Default      | Values                        | Description                                       |
|---------------------------------|--------------|-------------------------------|---------------------------------------------------|
| `mu_strategy`                   | `"adaptive"` | `"adaptive"`, `"monotone"`    | Barrier parameter update strategy                 |
| `warm_start_init_point`         | `"no"`       | `"yes"`, `"no"`               | Enable warm-start initialization                  |
| `mu_allow_increase`             | `"yes"`      | `"yes"`, `"no"`               | Allow barrier parameter increase                  |
| `least_squares_mult_init`       | `"yes"`      | `"yes"`, `"no"`               | LS estimate for initial multipliers               |
| `enable_slack_fallback`         | `"yes"`      | `"yes"`, `"no"`               | Slack reformulation fallback                      |
| `enable_lbfgs_fallback`         | `"yes"`      | `"yes"`, `"no"`               | L-BFGS fallback for unconstrained                 |
| `enable_al_fallback`            | `"yes"`      | `"yes"`, `"no"`               | Augmented Lagrangian fallback                     |
| `enable_preprocessing`          | `"yes"`      | `"yes"`, `"no"`               | Internal auxiliary equality solves/recovery, fixed vars, redundancies |
| `detect_linear_constraints`     | `"yes"`      | `"yes"`, `"no"`               | Skip Hessian for linear constraints               |
| `enable_sqp_fallback`           | `"yes"`      | `"yes"`, `"no"`               | SQP fallback for constrained problems             |
| `hessian_approximation`         | `"exact"`    | `"exact"`, `"limited-memory"` | Use L-BFGS Hessian approximation                  |
| `enable_lbfgs_hessian_fallback` | `"yes"`      | `"yes"`, `"no"`               | Auto-retry with L-BFGS Hessian on failure         |

### Error handling

- **Callback errors:** If any callback returns `0`, the solver aborts and returns `RIPOPT_ERROR_IN_STEP_COMPUTATION` (7). Always return `1` from callbacks unless you detect a problem (e.g., NaN in inputs).
- **Unknown options:** `ripopt_add_*_option` returns `0` for unrecognized keywords. Check the return value if you want to detect typos.
- **NULL safety:** `ripopt_free(NULL)` is a no-op (safe to call). All output pointers in `ripopt_solve` except `x` may be `NULL`.
- **Memory:** The problem handle owns all internal memory. Call `ripopt_free()` once when done. Do not use the handle after freeing.

### Migrating from Ipopt

If you have existing Ipopt C code, the migration is straightforward:

1. **Header:** `#include "IpStdCInterface.h"` → `#include "ripopt.h"`
2. **Handle type:** `IpoptProblem` → `RipoptProblem` (both are `void*`)
3. **Functions:** Rename `CreateIpoptProblem` → `ripopt_create`, `FreeIpoptProblem` → `ripopt_free`, `AddIpoptNumOption` → `ripopt_add_num_option`, etc.
4. **Callbacks:** No changes required — signatures are identical
5. **Status codes:** Similar semantics but different enum names (e.g., `Solve_Succeeded` → `RIPOPT_SOLVE_SUCCEEDED`)
6. **Infinity:** Ipopt uses ±2e19 by default; ripopt uses ±1e30 in bounds and ±1e20 for `nlp_*_bound_inf`
7. **Linking:** `-lipopt` → `-lripopt`

### Compile and run the examples

```bash
cargo build --release

# HS071 — constrained NLP with inequality + equality constraints
cc examples/c_api_test.c -I. -Ltarget/release -lripopt \
   -Wl,-rpath,$(pwd)/target/release -o c_api_test -lm
./c_api_test

# Rosenbrock — unconstrained optimization
cc examples/c_rosenbrock.c -I. -Ltarget/release -lripopt \
   -Wl,-rpath,$(pwd)/target/release -o c_rosenbrock -lm
./c_rosenbrock

# HS035 — bound-constrained QP with inequality
cc examples/c_hs035.c -I. -Ltarget/release -lripopt \
   -Wl,-rpath,$(pwd)/target/release -o c_hs035 -lm
./c_hs035

# Full multiplier extraction and options demonstration
cc examples/c_example_with_options.c -I. -Ltarget/release -lripopt \
   -Wl,-rpath,$(pwd)/target/release -o c_example_with_options -lm
./c_example_with_options
```

## Automatic Differentiation

Implementing `NlpProblem` requires hand-derived gradients, Jacobians, and Hessians, which is tedious and error-prone for real-world problems. The [ipopt-ad](https://github.com/prehner/ipopt-ad) crate (by [@prehner](https://github.com/prehner)) eliminates this by computing exact derivatives automatically via the [num-dual](https://crates.io/crates/num-dual) crate. Users write their objective and constraints once as generic Rust functions, and ipopt-ad handles all derivative computation and sparsity detection.

With automatic differentiation, the HS071 objective and constraints reduce to:

```rust
fn objective<D: DualNum<f64> + Copy>(x: SVector<D, 4>) -> D {
    x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
}

fn constraints<D: DualNum<f64> + Copy>(x: SVector<D, 4>) -> SVector<D, 2> {
    SVector::from([
        x[0] * x[1] * x[2] * x[3],
        x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3],
    ])
}
```

No gradient, Jacobian, or Hessian code needed. See `examples/autodiff_num_dual.rs` for a complete working example:

```bash
cargo run --example autodiff_num_dual --features num-dual
```

## Examples

### Rust

```bash
# Rosenbrock function (unconstrained with bounds)
cargo run --example rosenbrock

# HS071 (constrained NLP with inequalities)
cargo run --example hs071

# HS071 with automatic differentiation (no hand-derived derivatives)
cargo run --example autodiff_num_dual --features num-dual

# Benchmark timing across 5 problems
cargo run --release --example benchmark

# Parametric sensitivity analysis
cargo run --release --example sensitivity
```

### C

See [Compile and run the examples](#compile-and-run-the-examples) above for build instructions. The C examples are:

| Example                    | Problem type                | Demonstrates                                                 |
|----------------------------|-----------------------------|--------------------------------------------------------------|
| `c_api_test.c`             | HS071 (constrained)         | Basic usage, all 5 callbacks                                 |
| `c_rosenbrock.c`           | Rosenbrock (unconstrained)  | No constraints, no bounds                                    |
| `c_hs035.c`                | HS035 (bounds + inequality) | Bound multipliers, constraint multiplier                     |
| `c_example_with_options.c` | HS071 (multiple solves)     | Options tuning, multiplier extraction, status interpretation |

## Tests

```bash
cargo test
```

The test suite covers:
- **Unit tests**: Dense LDL factorization, convergence checking, filter line search, fraction-to-boundary, KKT assembly, restoration, preprocessing, linearity detection, SQP, linear solver, autodiff, L-BFGS, sensitivity analysis
- **C API tests**: FFI integration tests
- **Integration tests**: Rosenbrock, SimpleQP, HS071, HS035, PureBoundConstrained, MultipleEqualityConstraints, NE-to-LS reformulation, augmented Lagrangian, NL file parsing, IPM code paths, parametric sensitivity, and more
- **Coverage tests**: Augmented Lagrangian convergence paths, NL parser/solver pipeline, autodiff tape operations, IPM preprocessing/condensed KKT/unbounded detection

## Code Coverage

ripopt uses `cargo-llvm-cov` for code coverage measurement:

```bash
# Run tests with coverage and print summary
cargo llvm-cov test

# Detailed line-by-line report
cargo llvm-cov test --text

# HTML report (opens in browser)
cargo llvm-cov test --html && open target/llvm-cov/html/index.html
```

Current coverage by module:

| Module                   | Line Coverage |
|--------------------------|---------------|
| options.rs               | 100%          |
| slack_formulation.rs     | 97%           |
| warmstart.rs             | 97%           |
| filter.rs                | 95%           |
| restoration_nlp.rs       | 93%           |
| convergence.rs           | 92%           |
| sensitivity.rs           | 91%           |
| dense.rs (linear solver) | 91%           |
| banded.rs                | 91%           |
| nl/header.rs             | 91%           |
| sqp.rs                   | 90%           |
| preprocessing.rs         | 85%           |
| linear_solver/mod.rs     | 84%           |
| sparse.rs                | 82%           |
| kkt.rs                   | 81%           |
| restoration.rs           | 79%           |
| multifrontal.rs          | 78%           |
| nl/autodiff.rs           | 74%           |
| lbfgs.rs                 | 70%           |
| nl/parser.rs             | 70%           |
| augmented_lagrangian.rs  | 66%           |
| c_api.rs                 | 61%           |
| nl/problem_impl.rs       | 58%           |
| linearity.rs             | 53%           |
| ipm.rs                   | 52%           |
| nl/expr.rs               | 37%           |
| hybrid.rs                | 0%            |
| iterative.rs             | 0%            |

**Overall: 62% line coverage** in the last recorded coverage run.

The `hybrid.rs` and `iterative.rs` modules implement opt-in linear
solvers (`LinearSolverChoice::Hybrid`, `::Iterative`) that no test
currently exercises — they are never selected by the default solver
path. Their coverage is reported as 0% until dedicated tests land.

## Architecture

```
src/
  lib.rs              Public API (solve function, re-exports)
  c_api.rs            C FFI layer (extern "C" functions, ripopt.h)
  problem.rs          NlpProblem trait definition
  options.rs          SolverOptions with Ipopt-matching defaults
  result.rs           SolveResult and SolveStatus
  ipm.rs              Main IPM loop, barrier updates, line search, NE-to-LS detection, NLP scaling
  kkt.rs              KKT system assembly, solution, and inertia correction
  convergence.rs      Convergence checking (primal/dual/complementarity)
  filter.rs           Filter line search mechanism
  restoration.rs      Gauss-Newton restoration phase with adaptive LM regularization
  restoration_nlp.rs  Full NLP restoration subproblem (Phase 2)
  lbfgs.rs            L-BFGS solver for unconstrained/bound-constrained problems
  augmented_lagrangian.rs  Augmented Lagrangian fallback for constrained problems
  sqp.rs              SQP fallback for constrained problems
  sensitivity.rs      Parametric sensitivity analysis (sIPOPT-style)
  slack_formulation.rs     Explicit slack reformulation fallback
  preprocessing.rs    Fixed variable elimination, redundant constraint removal, bound tightening
  linearity.rs        Near-linear constraint detection
  warmstart.rs        Warm-start initialization
  linear_solver/
    mod.rs            LinearSolver trait, SymmetricMatrix, KktMatrix
    dense.rs          Dense LDL^T (Bunch-Kaufman) factorization
    banded.rs         Banded LDL^T for problems with small bandwidth
    feral_direct.rs   Multifrontal sparse LDL^T via feral (default, Bunch-Kaufman, MC64, AMD/METIS)
    feral_iterative.rs MINRES wrapper backed by feral (iterative-refinement variant)
    feral_hybrid.rs   Hybrid direct/iterative wrapper for the feral backend
    multifrontal.rs   Multifrontal sparse LDL^T via rmumps (opt-in `rmumps` feature)
    sparse.rs         Sparse LDL^T via faer (optional)
    iterative.rs      MINRES with incomplete LDL^T preconditioner (rmumps feature)
    hybrid.rs         Hybrid direct/iterative solver with automatic switching (rmumps feature)

tests/
  correctness.rs      Integration tests (22 NLP problems)
  ipm_paths.rs        IPM code path tests (condensed KKT, unbounded, NE-to-LS, preprocessing)
  sensitivity.rs      Parametric sensitivity integration tests
  c_api.rs            C API integration tests (12 tests via FFI)
  lbfgs_ipm.rs        L-BFGS Hessian approximation tests
  iterative_solvers.rs  Iterative/hybrid solver tests
  large_scale.rs      Large-scale correctness tests (up to 100K variables)
  large_scale_benchmark.rs  Large-scale ripopt vs Ipopt comparison
  nl_integration.rs   NL file parsing and solving tests

gams/
  gams_ripopt.c       GAMS solver link (GMO API → ripopt C API bridge)
  Makefile            Build, install, and test targets
  install.sh          Registration script for gmscmpun.txt
  test_hs071.gms      Smoke test (HS071 via `option nlp = ripopt`)

examples/
  rosenbrock.rs       Unconstrained optimization
  hs071.rs            Constrained NLP
  sensitivity.rs      Parametric sensitivity analysis demo
  benchmark.rs        Timing benchmark
  c_api_test.c        HS071 via the C API
  c_rosenbrock.c      Unconstrained Rosenbrock via C API
  c_hs035.c           Bound-constrained QP via C API
  c_example_with_options.c  Options and multiplier extraction demo
```

## Algorithm Details

### Preprocessing

Before solving, ripopt automatically analyzes the problem to reduce its size:

1. **Auxiliary equality-system preprocessing/recovery**: Internal auxiliary equalities and variables may be solved before the main NLP or recovered after a reduced main solve when the removed variables are absent from the objective and inequalities
2. **Fixed variable elimination**: Variables with `x_l == x_u` are removed and set to their fixed values in all evaluations
3. **Redundant constraint removal**: Duplicate constraints (same Jacobian structure, values, and bounds) are eliminated
4. **Bound tightening**: Single-variable linear constraints are used to tighten variable bounds

The reduced problem is solved and the solution is mapped back to the original dimensions. Disable with `enable_preprocessing: false`.

### Near-Linear Constraint Detection

The Jacobian is evaluated at two points to identify linear constraints (where all Jacobian entries remain constant). For linear constraints, the Hessian contribution `lambda[i] * nabla^2 g_i` is exactly zero and is skipped, reducing computation in the Hessian evaluation.

### Core Interior Point Method

The solver follows the primal-dual barrier method from the Ipopt papers (Wachter & Biegler, 2006). At each iteration it:

1. Assembles and factors the KKT system using dense LDL^T (Bunch-Kaufman), sparse multifrontal LDL^T (feral by default; rmumps under the opt-in `rmumps` feature), or dense condensed Schur complement for tall-narrow problems
2. Computes inertia of the factorization and applies regularization if needed
3. Applies Mehrotra predictor-corrector with Gondzio centrality corrections (default on)
4. Computes search directions with iterative refinement (up to 3 rounds)
5. Applies second-order corrections (SOC) when the initial step is rejected
6. Uses a filter line search with backtracking to ensure sufficient progress
7. Updates the barrier parameter adaptively using Mehrotra sigma-guided updates

### Multi-Solver Fallback Architecture

When the primary IPM fails, ripopt automatically tries alternative solvers:

1. **L-BFGS**: Tried first for unconstrained problems (m=0, no bounds); used as fallback for bound-constrained problems
2. **L-BFGS Hessian approximation**: Retries the IPM with L-BFGS curvature pairs replacing the exact Hessian (helps when the Hessian is ill-conditioned or buggy)
3. **Augmented Lagrangian**: PHR penalty method for constrained problems, with the IPM solving each AL subproblem
4. **SQP**: Equality-constrained Sequential Quadratic Programming with l1 merit function line search
5. **Explicit slack reformulation**: Converts g(x) to g(x)-s=0 with bounds on s, stabilizing multiplier oscillation at degenerate points
6. **Best-du tracking**: Throughout the solve, tracks the iterate with lowest dual infeasibility and recovers it at max iterations

### Limited-Memory Hessian Approximation (L-BFGS-in-IPM)

When `hessian_approximation_lbfgs = true`, the IPM replaces exact Hessian evaluations with an L-BFGS curvature approximation. This eliminates the need for `hessian_values()` callbacks entirely.

**How it works:**

1. Each IPM iteration, after accepting a step, the solver computes `s_k = x_{k+1} - x_k` and `y_k = ∇L(x_{k+1}) - ∇L(x_k)` from Lagrangian gradient differences
2. Powell damping ensures positive curvature (`s^T y > 0`)
3. An explicit dense B_k matrix is formed from the L-BFGS pairs and used in KKT assembly
4. Up to 10 curvature pairs are stored (O(n·m) memory where m=10)

**When to use it:**

- Neural networks in NLPs (dense Hessian, O(n²) memory prohibitive)
- Problems where second derivatives are unavailable or expensive
- Rapid prototyping (skip Hessian derivation)
- When the exact Hessian is ill-conditioned or buggy

**Automatic fallback:** By default (`enable_lbfgs_hessian_fallback = true`), if the exact-Hessian IPM fails with MaxIterations, NumericalError, or RestorationFailed, the solver automatically retries with L-BFGS Hessian approximation. This helps when the user-provided Hessian is inaccurate.

**Example:**

```rust
let options = SolverOptions {
    hessian_approximation_lbfgs: true,
    ..SolverOptions::default()
};
let result = ripopt::solve(&problem, &options);
```

See `examples/lbfgs_hessian.rs` for complete working examples.

### Parametric Sensitivity Analysis

ripopt provides sIPOPT-style post-optimal sensitivity analysis: after solving an NLP, compute how the optimal solution changes when problem parameters are perturbed, without re-solving. This is useful for:

- **What-if analysis**: How does the optimal design change if a constraint bound shifts?
- **Uncertainty propagation**: Map parameter uncertainty to solution uncertainty via the reduced Hessian
- **Real-time optimization**: Update the solution for small disturbances at near-zero cost

The core equation is `ds/dp = -M⁻¹ · Nₚ` — one backsolve using the already-factored KKT matrix.

**Usage**: Implement the `ParametricNlpProblem` trait (extends `NlpProblem` with parameter derivative methods), then call `solve_with_sensitivity()`:

```rust
use ripopt::{ParametricNlpProblem, SolverOptions};

// Implement ParametricNlpProblem for your problem type...
// (adds num_parameters, jacobian_p_*, hessian_xp_* methods)

let mut ctx = ripopt::solve_with_sensitivity(&problem, &options);

// Compute sensitivity for a parameter perturbation Δp
let dp = [0.1];  // perturbation vector
let sens = ctx.compute_sensitivity(&problem, &[&dp]).unwrap();

// Predict perturbed solution: x(p + Δp) ≈ x* + dx
let x_new: Vec<f64> = ctx.result.x.iter()
    .zip(sens.dx_dp[0].iter())
    .map(|(x, dx)| x + dx)
    .collect();

// Extract reduced Hessian for covariance estimation
let cov = ctx.reduced_hessian().unwrap();
```

On the HS071 problem with a parametric constraint bound, the sensitivity prediction matches re-solve to within 1e-5:

```
Predicted x(p=40.1): (1.000000, 4.751642, 3.824904, 1.375540)
Actual    x(p=40.1): (1.000000, 4.751634, 3.824896, 1.375553)
Prediction errors:   8.4e-6,   8.0e-6,   1.4e-5
```

See `examples/sensitivity.rs` for a complete working example.

### Condensed KKT System

For problems where the number of constraints m exceeds 2n, the solver automatically uses a condensed (Schur complement) formulation. This reduces the factorization cost from O((n+m)^3) to O(n^2 m + n^3), enabling efficient handling of problems with many constraints and few variables.

### NE-to-LS Reformulation

When the solver detects an overdetermined nonlinear equation system (m >= n, f(x) = 0, all equality constraints, starting point not already feasible), it automatically reformulates the problem as unconstrained least-squares minimization:

```
min  0.5 * ||g(x) - target||^2
```

using a full Hessian (J^T J + sum of r_i * nabla^2 g_i). If the residual is small at the solution, the original system is consistent and `Optimal` is reported. Otherwise, `LocalInfeasibility` is reported with the best least-squares solution.

### Two-Phase Restoration

When the filter line search fails:

1. **Phase 1 (Gauss-Newton)**: Fast feasibility solver minimizing ||violation||^2 with gradient descent fallback
2. **Phase 2 (NLP restoration)**: Full barrier subproblem with p/n slack variables (Ipopt formulation)
3. **Multi-attempt recovery**: Up to 6 attempts cycling barrier parameter perturbations [10x, 0.1x, 100x, 0.01x, 1000x, 0.001x] with x perturbation

### Watchdog Strategy

The solver implements a watchdog mechanism that temporarily relaxes the filter acceptance criteria when progress stalls due to shortened steps. This helps escape narrow feasible corridors where strict Armijo conditions are too conservative.

## Profiling

### Built-in Phase Timing

ripopt includes per-iteration phase timing instrumentation. When `print_level >= 5` (the default), a summary table is printed at the end of each solve showing where CPU time is spent:

```
Phase breakdown (47 iterations):
  Problem eval           0.234s (45.2%)
  KKT assembly           0.089s (17.2%)
  Factorization          0.156s (30.1%)
  Direction solve        0.012s  (2.3%)
  Line search            0.021s  (4.1%)
  Other                  0.006s  (1.1%)
  Total                  0.518s
```

To suppress timing output, set `print_level: 0` in `SolverOptions`.

### External Profiling with samply

Release builds include debug symbols (`debug = true` in `[profile.release]`), so external profilers can show function names. [samply](https://github.com/mstange/samply) provides flamegraph visualization on macOS and Linux:

```bash
cargo install samply
cargo build --release --bin hs_suite
samply record target/release/hs_suite
```

This opens a Firefox Profiler UI in the browser with a full call tree and flamegraph. Look for wide bars under `solve_ipm` to identify dominant functions.

On macOS, Instruments (Xcode) also works without any additional setup:

```bash
cargo build --release --bin hs_suite
xcrun xctrace record --template "Time Profiler" --launch target/release/hs_suite
```

## Sign Convention

ripopt uses the Ipopt convention where the Lagrangian is:

```
L = f(x) + y^T g(x)
```

For inequality constraints `g(x) >= g_l`, the multiplier `y` is negative at optimality. For equality constraints, `y` can be positive or negative.

## License

[EPL-2.0](LICENSE) (Eclipse Public License 2.0), consistent with Ipopt.

The `rmumps` workspace member (pure Rust multifrontal solver) is licensed under [CeCILL-C](rmumps/LICENSE), a LGPL-compatible free software license.
