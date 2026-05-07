# Diagnostics

`SolverDiagnostics` captures structured data about solver behavior. It is
available at `result.diagnostics`; the compact summary below is printed to
stderr when `print_level >= 5`. Structured JSON reports include the full
diagnostics object, including preprocessing details.

## Diagnostic block format

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
| `status` | Final solve status |
| `iterations` | Total IPM iterations |
| `wall_time_secs` | Total wall-clock time |
| `final_mu` | Barrier parameter at termination |
| `final_primal_inf` | Constraint violation at termination |
| `final_dual_inf` | Dual infeasibility (stationarity error) at termination |
| `final_compl` | Complementarity error at termination |
| `restoration_count` | Gauss-Newton restoration entries |
| `nlp_restoration_count` | Full NLP restoration entries (heavier) |
| `mu_mode_switches` | Barrier mode transitions (Free ↔ Fixed) |
| `filter_rejects` | Line search failures (backtracking exhausted) |
| `watchdog_activations` | Watchdog triggered by consecutive short steps |
| `soc_corrections` | Second-order corrections accepted |
| `fallback_used` | Which fallback succeeded, if any (`lbfgs_hessian`, `augmented_lagrangian`, `sqp`, `slack`) |
| `preprocessing` | Nested diagnostics for auxiliary presolve, auxiliary postsolve, and standard preprocessing |

## Preprocessing Diagnostics

`diagnostics.preprocessing` is always present in `SolveResult` and in CLI JSON
reports written with `ripopt problem.nl -o report.json`. It has three nested
objects:

| Object | Meaning |
|---|---|
| `preprocessing.presolve` | Auxiliary equality-block reduction attempted before the main solve |
| `preprocessing.postsolve` | Auxiliary equality-block reduction followed by recovery after a reduced solve |
| `preprocessing.standard` | Fixed-variable and redundant-constraint preprocessing |

The auxiliary objects, `presolve` and `postsolve`, report both timing and
structural data:

| Field | Meaning |
|---|---|
| `attempted`, `solved`, `failed` | Whether the phase ran, solved the problem, or rejected/fell back |
| `skipped`, `skip_reason` | Whether a no-op/cost gate skipped the phase before auxiliary solves, and why |
| `total_time_secs` | Total wall-clock time spent in that preprocessing phase |
| `candidate_detection_time_secs` | End-to-end candidate search time |
| `incidence_time_secs` | Time spent building equality incidence data |
| `structural_analysis_time_secs` | Time spent on components, matching, Dulmage-Mendelsohn, and block-triangular analysis |
| `candidate_filter_time_secs` | Time spent rejecting objective- or inequality-coupled variables |
| `auxiliary_solve_time_secs` | Presolve time spent solving accepted auxiliary blocks |
| `recovery_solve_time_secs` | Postsolve time spent recovering eliminated variables |
| `reduction_build_time_secs` | Time spent wrapping the reduced problem |
| `nested_preprocessing_time_secs` | Time spent running standard preprocessing on the auxiliary-reduced problem |
| `reduced_solve_time_secs` | Time spent solving the reduced problem |
| `unmap_time_secs` | Time spent mapping reduced solutions back to the original space |
| `full_space_validation_time_secs` | Time spent recomputing objective, constraints, and residuals on the original problem |
| `equality_rows`, `incident_variables`, `connected_components` | Size of the equality-incidence structure examined |
| `candidates`, `btd_blocks`, `accepted_block_sizes` | Accepted auxiliary candidate and block structure |
| `rejected_blocks`, `rejection_counts` | Rejection totals grouped by reason |
| `auxiliary_blocks_solved`, `auxiliary_iterations`, `auxiliary_*_evals` | Work performed by internal auxiliary solves |
| `original_*`, `reduced_*`, `removed_*`, `nested_*` | Original, reduced, removed, and nested standard-preprocessing dimensions |

The standard object reports `attempted`, `did_reduce`, `total_time_secs`,
`construction_time_secs`, `reduced_solve_time_secs`, `unmap_time_secs`,
original/reduced dimensions, `fixed_variables`, and `redundant_constraints`.
Use these fields to separate "preprocessing found no useful structure" from
"preprocessing reduced the model but the overhead exceeded the iteration
savings."

## Interpreting the diagnostics

**Healthy solve** (HS071-like): 0 restorations, 0 filter rejects, 2–4 mu mode switches, `final_mu` near `1e-9`, `final_primal_inf` and `final_dual_inf` both below `tol`.

**Struggling solve**: Many filter rejects, multiple restorations, `final_mu` stuck above `1e-4`, or a fallback was used.

### Pattern → cause → fix

| Pattern | Likely cause | Options to try |
|---|---|---|
| `filter_rejects` > 5 | Line search fighting constraints | Increase `mu_init`, reduce `kappa` |
| `restoration_count` > 3 | Repeated feasibility recovery | Set `enable_slack_fallback: true`, increase `mu_init` |
| `mu_mode_switches` > 10 | Free/Fixed cycling | Set `mu_strategy_adaptive: false` |
| `final_mu` stuck > 1e-4 | Barrier parameter not decreasing | Increase `max_iter`, reduce `mu_linear_decrease_factor` |
| `fallback_used: Some(...)` | Primary IPM failed | Check which fallback; consider changing Hessian strategy |
| `soc_corrections` > 0 | Nonlinear constraints causing step rejection | Normal; increase `max_soc` if filter rejects are also high |
| `watchdog_activations` > 0 | Tiny steps detected | Try `hessian_approximation_lbfgs: true` |

## Example: healthy solve (HS071)

```rust
let result = ripopt::solve(&Hs071, &SolverOptions::default());
let d = &result.diagnostics;
assert_eq!(d.filter_rejects, 0);
assert_eq!(d.restoration_count, 0);
assert!(d.final_mu < 1e-8);
// iterations ≈ 8
```

## Example: struggling solve — reading and reacting

```rust
let r1 = ripopt::solve(&problem, &opts);

if r1.status != SolveStatus::Optimal {
    let d = &r1.diagnostics;
    let mut opts2 = opts.clone();

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
    if d.mu_mode_switches > 10 {
        opts2.hessian_approximation_lbfgs = true;
    }

    let r2 = ripopt::solve(&problem, &opts2);
}
```

## Using diagnostics with Claude Code

Claude Code can read the `--- ripopt diagnostics ---` block from stderr and automatically adjust solver options:

```bash
claude -p "
  Run: cargo run --example debug_tp374 2>&1
  Parse the diagnostics block.
  If not Optimal:
    - High filter_rejects → increase mu_init, decrease kappa
    - High restoration_count → try enable_slack_fallback
    - mu stuck high → try mu_strategy_adaptive: false
    - Large multipliers → try hessian_approximation_lbfgs: true
  Adjust options, re-run, compare. Up to 3 attempts.
"
```

The Rust code is a pure reporter. All intelligence — pattern matching, strategy selection — lives in Claude's reasoning.
