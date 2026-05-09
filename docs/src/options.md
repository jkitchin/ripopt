# Solver Options

All options are set via `SolverOptions`. Defaults match Ipopt where applicable.

```rust
let opts = SolverOptions {
    tol: 1e-8,
    max_iter: 500,
    ..SolverOptions::default()
};
```

## Convergence

| Option | Default | Description |
|---|---|---|
| `tol` | `1e-8` | KKT optimality tolerance (dual infeasibility + complementarity) |
| `constr_viol_tol` | `1e-4` | Constraint violation tolerance |
| `dual_inf_tol` | `1.0` | Dual infeasibility tolerance (scaled) |
| `compl_inf_tol` | `1e-4` | Complementarity tolerance |
| `max_iter` | `3000` | Maximum iterations |
| `max_wall_time` | `0.0` | Wall-clock limit in seconds (0 = unlimited) |
| `stall_iter_limit` | `30` | Max iters without 1% improvement before stall detection (0 = off) |
| `early_stall_timeout` | `120.0` | Max seconds for first 3 iterations (0 = off) |

## Barrier parameter

| Option | Default | Description |
|---|---|---|
| `mu_init` | `0.1` | Initial barrier parameter |
| `mu_min` | `1e-11` | Minimum barrier parameter floor |
| `mu_strategy_adaptive` | `true` | `true` = oracle-based (Free mode); `false` = monotone |
| `kappa` | `10.0` | Barrier decrease divisor: `μ_new = avg_compl / kappa` |
| `mu_linear_decrease_factor` | `0.2` | Monotone mode: `μ_new = factor * μ` |
| `mu_superlinear_decrease_power` | `1.5` | Monotone mode superlinear exponent |
| `barrier_tol_factor` | `10.0` | Subproblem tol = `barrier_tol_factor * μ` |
| `mu_allow_increase` | `true` | Allow μ to increase after restoration/stall recovery |
| `adaptive_mu_monotone_init_factor` | `0.8` | Initial μ factor when entering Fixed (monotone) mode |
| `mu_oracle_quality_function` | `false` | Use quality function for mu selection |

## Linear solver

| Option | Default | Description |
|---|---|---|
| `sparse_threshold` | `110` | Switch to sparse multifrontal solver when `n+m ≥ threshold` |
| `linear_solver` | `Direct` | `Direct` (MUMPS/BK), `Iterative` (MINRES), or `Hybrid` |

## Inertia-free curvature test (IFRd)

Opt-in alternative to inertia-based regularization (IBR). When the linear solver reports
wrong inertia, run the curvature test of Chiang & Zavala (2016, COAP 64:327-354, eq. 28)
on the computed direction; accept if the curvature condition holds, otherwise fall back
to the standard δ-escalation ladder. Mirrors Ipopt 3.14's `IpPDFullSpaceSolver` dispatch.

| Option | Default | Description |
|---|---|---|
| `neg_curv_test_tol` | `0.0` | Curvature acceptance tolerance α_d. `0.0` disables IFRd (pure IBR). Set to `1e-12` to enable. |
| `neg_curv_test_reg` | `true` | Include δ_w·‖(dx,ds)‖² regularization in the curvature sum (matches Ipopt) |

Empirically (CUTEst sweep, 727 problems): default vs. `tol=1e-12` both solve 541 to
Optimal, but the *mix* differs — 19 problems are rescued by IFRd and 19 different
problems regress. Default off; enable on a per-problem basis when a problem doesn't
solve with IBR.

## Mehrotra predictor-corrector

| Option | Default | Description |
|---|---|---|
| `mehrotra_pc` | `true` | Enable Mehrotra predictor-corrector (20–40% fewer iterations) |
| `gondzio_mcc_max` | `3` | Max Gondzio centrality corrections per iteration (0 = off) |

## Line search and corrections

| Option | Default | Description |
|---|---|---|
| `max_soc` | `4` | Max second-order corrections per step |
| `tau_min` | `0.99` | Fraction-to-boundary parameter (τ in step size rule) |
| `constraint_slack_barrier` | `true` | Include constraint slack log-barriers in filter merit function |
| `watchdog_shortened_iter_trigger` | `10` | Consecutive short steps before watchdog |
| `watchdog_trial_iter_max` | `3` | Watchdog trial iterations |

## Hessian strategy

| Option | Default | Description |
|---|---|---|
| `hessian_approximation_lbfgs` | `false` | Use L-BFGS instead of exact Hessian (no `hessian_values` needed) |
| `enable_lbfgs_hessian_fallback` | `true` | Auto-retry with L-BFGS if exact Hessian IPM fails |

## Fallback cascade

| Option | Default | Description |
|---|---|---|
| `enable_slack_fallback` | `true` | Reformulate inequalities with explicit slacks on failure |
| `enable_al_fallback` | `true` | Try Augmented Lagrangian if IPM fails (equality problems) |
| `enable_sqp_fallback` | `true` | Try SQP if AL/slack also fail |
| `enable_lbfgs_fallback` | `true` | Try L-BFGS for unconstrained problems on IPM failure |

## Restoration

| Option | Default | Description |
|---|---|---|
| `restoration_max_iter` | `200` | Max iterations for NLP restoration subproblem |
| `disable_nlp_restoration` | `false` | Disable NLP restoration (prevents recursion in inner solves) |

## Warm start

| Option | Default | Description |
|---|---|---|
| `warm_start` | `false` | Initialize from a previous solution |
| `warm_start_bound_push` | `1e-3` | Bound push for warm-started variables |
| `warm_start_bound_frac` | `1e-3` | Bound fraction for warm-started variables |
| `warm_start_mult_bound_push` | `1e-3` | Multiplier floor for warm start |

## Initial point

| Option | Default | Description |
|---|---|---|
| `bound_push` | `1e-2` | κ₁: push initial x away from bounds by max(κ₁, κ₂·(u−l)) |
| `bound_frac` | `1e-2` | κ₂: fraction of bound gap used for push |
| `slack_bound_push` | `1e-2` | Slack variable bound push |
| `slack_bound_frac` | `1e-2` | Slack variable bound fraction |
| `least_squares_mult_init` | `true` | Initialize y by least-squares on stationarity |
| `constr_mult_init_max` | `1000.0` | Cap on LS multiplier init magnitude |

## Bound thresholds

| Option | Default | Description |
|---|---|---|
| `nlp_lower_bound_inf` | `-1e20` | Treat variable/constraint bounds below this as -∞ |
| `nlp_upper_bound_inf` | `1e20` | Treat variable/constraint bounds above this as +∞ |

## Preprocessing and detection

| Option | Default | Description |
|---|---|---|
| `enable_preprocessing` | `true` | Auxiliary equality-block preprocessing/recovery, fixed-variable elimination, and redundant-constraint removal |
| `auxiliary_tol` | `1e-8` | Accepted residual for auxiliary preprocessing/recovery block solves |
| `detect_linear_constraints` | `true` | Skip Hessian for constraints with zero second derivatives |
| `proactive_infeasibility_detection` | `false` | Early infeasibility detection during iterations |

## Diagnostics

| Option | Default | Description |
|---|---|---|
| `print_level` | `5` | Verbosity: 0 = silent, 5 = full iteration table + diagnostics |

## KKT matrix dump (instrumentation)

| Option | Default | Description |
|---|---|---|
| `kkt_dump_dir` | `None` | If set, write each KKT matrix to this directory after factorization |
| `kkt_dump_name` | `"problem"` | Problem name prefix for dump filenames |

When `kkt_dump_dir` is set, ripopt writes two files per iteration:
- `<name>_<iter:04>.mtx` — Matrix Market symmetric format, lower triangle
- `<name>_<iter:04>.json` — Metadata: n, m, rhs, inertia, status

This is useful for collecting benchmark matrices for external sparse solvers.
