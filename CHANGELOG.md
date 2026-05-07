# Changelog

## [Unreleased]

### Added
- **Auxiliary equality preprocessing** (PR #32): when `enable_preprocessing` is
  enabled (default), ripopt now detects square auxiliary equality subsystems via
  Dulmage-Mendelsohn partitioning and block-triangular decomposition, solves
  them outside the main IPM, and removes the solved auxiliary variables and
  rows before the main solve. Inequality- and objective-coupled candidates are
  kept on the full-space path. Reduced solutions are rejected and the standard
  preprocessing path is used as fallback unless they validate in the original
  full space at the user's `constr_viol_tol` and `dual_inf_tol`. New option
  `auxiliary_tol` (default 1e-8) controls the accepted residual for internal
  auxiliary solves.
  - Behavior change: with the default `enable_preprocessing = true`, the new
    auxiliary path runs on every solve. Set `enable_preprocessing = false` to
    bypass both the auxiliary and the standard fixed-variable / redundant-
    constraint preprocessing paths.
  - CUTEst impact (vs v0.8.0): +1 Optimal (DENSCHNENE moves
    `LocalInfeasibility` → `Optimal`), 37 problems faster (e.g. HATFLDF
    105 → 0 iters, HEART6 45 → 0), 2 problems slower (RES +16, ACOPP30 +4),
    no lost solves, no objective-match regressions.

## [0.8.0] - 2026-05-06

**BREAKING:** the default sparse linear solver has changed from `rmumps` to
[`feral`](https://crates.io/crates/feral), a pure-Rust multifrontal LDLᵀ solver with Bunch-Kaufman
1×1/2×2 pivoting, certified inertia, MC64 scaling, and AMD/METIS ordering.
`rmumps` remains available behind the opt-in `rmumps` feature for A/B
comparison and regression hunting.

### Changed
- **Default sparse linear solver: feral** (replaces rmumps).
  - `default-features = ["feral", "faer"]` (was `["rmumps", "faer"]`).
  - Build with the legacy backend via
    `cargo build --no-default-features --features "rmumps faer"`.
  - HS suite parity: 118/120 Optimal under both backends with zero status
    flips (HS116/HS374 fail under both). Geomean solve-time within 5%.
  - `LinearSolverChoice::{Direct, Iterative, Hybrid}` now route to
    `FeralLdl` / `FeralIterativeMinres` / `FeralHybrid` by default. Under
    the feral backend, Iterative reduces to Direct + iterative refinement
    (the full LDLᵀ as MINRES preconditioner converges in one step); a true
    incomplete-LDLᵀ + MINRES path is deferred to v0.9.

### Added
- `src/linear_solver/feral_direct.rs` — `FeralLdl` wrapper with cached
  symbolic factorization, COO→CSC value scatter, MA27-style pivot-threshold
  escalation in `increase_quality()`, and `ZeroPivotAction::ForceAccept` so
  ripopt's existing inertia-correction loop drives perturbation.
- `src/linear_solver/feral_iterative.rs`, `feral_hybrid.rs` — companions
  for `LinearSolverChoice::Iterative` and `Hybrid`.
- `tests/feral_solvers.rs` — direct trait-surface tests for `FeralLdl`.

## [0.7.1] - 2026-04-25

Patch release focused on the **Reference-Gap Roadmap**: porting concrete
behavioral details from Ipopt 3.14 that the v0.7.0 ripopt-vs-Ipopt audit
flagged as missing. The biggest functional gains come from a faithful
`IpScaledNLP`-equivalent x-scaling wrapper, the Ipopt
`PDPerturbationHandler` δ-escalation schedule, the soft restoration
phase (`TrySoftRestoStep` with the E_μ test), the restoration
convergence three-gate, and a μ-dependent δ_c regularization. The cycle
also picked up first-class AMPL external-function support and a much
larger unit-test footprint for the filter, μ-oracle, and inertia
correction components. The body of the work (~200 commits) is internal
refactoring of `src/ipm.rs` into named helper functions to keep the
control flow legible as the IPM main loop grows.

### Added
- **AMPL external functions**: `.nl` parser now recognizes external
  function declarations and evaluates them through the
  `funcadd_ASL` ABI at solve time, so models that depend on
  user-supplied function libraries (e.g. IDAES `cbrt`, Sundials, custom
  property packages) load and solve in ripopt without modification.
  Previously these problems errored out at parse.
- **`warm_start_target_mu`** option (roadmap #18): allows callers
  warm-starting from a previous solve to override `mu_init` to the μ
  the prior run finished at, removing the cold-start ramp-up.
- **Centrality term in the Quality-Function μ oracle** (roadmap #14):
  optional centrality penalty in the QF objective, matching Ipopt's
  `centrality_term` knob.

### Changed
- **Full `IpScaledNLP`-equivalent x-scaling wrapper** (roadmap #6).
  ripopt's x-scaling now wraps the NLP exactly like Ipopt's
  `IpScaledNLP`, so all evaluations (objective, gradient, constraints,
  Jacobian, Hessian) see the scaled iterate consistently and the
  convergence check unscales correctly. Closes the largest remaining
  scaling gap from the reference-gap audit.
- **Ipopt `PDPerturbationHandler` δ-escalation** (roadmap #19): the
  KKT inertia-correction routine now follows the exact δ_w / δ_c
  escalation schedule from `IpPDPerturbationHandler.cpp` instead of an
  ad-hoc geometric ramp.
- **Soft restoration phase** (roadmap #12): port of Ipopt's
  `TrySoftRestoStep` with the E_μ acceptance test. When the line
  search fails but the iterate is still close to feasibility, ripopt
  now attempts the cheap soft-restoration step before falling through
  to the full restoration NLP.
- **Restoration convergence three-gate** (roadmap #13): port of
  `RestoFilterConvCheck`. Restoration now terminates on the same
  three-gate criterion Ipopt uses (filter acceptance, KKT residual,
  feasibility), matching the reference behavior at the boundary
  between restoration and the main IPM.
- **Convergence: separate `s_d` / `s_c` scales, drop the 1e4 cap**
  (roadmap #1). The dual-infeasibility and complementarity scaling
  factors are now computed independently as in Ipopt, and the legacy
  `1e4` clamp on the scaled error has been removed. Some problems
  that previously declared `Optimal` early on a loose scaled metric now
  run to the correct stopping point; some that hung at a scaled
  plateau now converge.

### Fixed
- **μ-dependent δ_c regularization** (roadmap #6): the constraint
  regularization now scales with μ as in Ipopt, instead of being held
  fixed.
- **Drop the ±1 inertia acceptance heuristic** (roadmap #5): the KKT
  inertia check now demands the exact `(n, m, 0)` signature Ipopt
  requires; the previous `±1` slack was hiding genuine inertia
  failures.
- **`user_x_scaling` is now rejected with a clear error** (roadmap
  #9) instead of being silently ignored. (No-op acceptance was the
  root cause of several "ripopt doesn't honor my scaling" reports.)
- **Preprocessing redundancy heuristic** (roadmap #8): guarded
  against the degenerate probe case where the heuristic could divide
  by zero on certain rank-deficient Jacobians.

### Refactored
- **~200 `refactor(ipm)` commits** extracting named helpers from
  `src/ipm.rs` (e.g. `compute_barrier_phi`, `apply_gondzio_mcc`,
  `run_line_search_loop`, `run_post_ls_restoration_cascade`,
  `try_soft_restoration`, `try_nlp_restoration_phase`,
  `update_barrier_parameter_free_mode` /
  `_fixed_mode`, all the per-phase trial / commit / snapshot helpers).
  No behavioral change — the IPM main loop now reads as a sequence of
  named phases instead of a 3000-line control-flow blob. Several
  pieces of state that were previously loose locals (`Watchdog`,
  `FeasibilityTracker`, `ProgressStallTracker`, `DualStallTracker`,
  `BestDuIterate`, iterate-averaging, dy-oscillation tracking) are now
  consolidated into named structs.
- **IPM iteration log gains a `compl` column** so users can see why an
  iterate that "looks converged" on `inf_pr`/`inf_du` is still
  rejected (complementarity not yet at tolerance).

### Tests
- **Component-level unit tests** for the filter, μ-oracle, and
  inertia-correction modules (previously only exercised via
  end-to-end solver runs).
- **NL external-function coverage** via the IDAES `cbrt` exact-arity
  type-0 path.
- **HS / CUTEst / electrolyte / grid benchmark fixtures refreshed**
  against the v0.8 work-in-progress.

### Notes
- 216 commits since v0.7.0; ~200 are pure internal refactoring with no
  behavioral change.
- **HS pass rate is unchanged**: ripopt 118/120, native Ipopt 116/120,
  same as v0.7.0.
- **CUTEst pass rate regressed by 4 problems**: ripopt Optimal 560 (v0.7.0)
  → 556 (v0.7.1); native Ipopt is unchanged at 556. Net result: ripopt's
  +4 advantage on CUTEst at v0.7.0 has been wiped out and the two solvers
  are now tied at 556/727 strict-Optimal. Per-problem churn is much larger
  than the net (-4) suggests: 20 problems regressed from `Optimal` (mostly
  to `NumericalError` or `LocalInfeasibility`) and 16 problems newly reach
  `Optimal`. The regressions are concentrated in least-squares /
  rank-deficient Jacobian families (CERI651, MGH, OET, PALMER, NET1,
  HYDC20LS, DECONVBNE, DUALC8, QPCBLEND, SNAKE, FLETCHER, TFI1, TAX13322).
  The most plausible suspects are the stricter `s_d` / `s_c` convergence
  metric (the dropped `1e4` cap means a few iterates that previously
  declared `Optimal` early now run past their numerically-clean stopping
  point) and the new μ-dependent δ_c regularization in factorization.
  Tagged comparison artifacts: `benchmarks/cutest/results_v0.7.0.json`
  vs `benchmarks/cutest/results_v0.7.1.json`.
- Speed on commonly-solved problems improved: HS geo-mean speedup vs
  native Ipopt rose from ~15× to ~21× (median 21×, 114/116 faster);
  CUTEst geo-mean rose from ~10× to ~12× on 521 commonly-solved
  problems.
- Mittelmann ampl-nlp benchmark harness moved from `mittelmann/` to
  `benchmarks/mittelmann/`. ripopt v0.7.1 still cannot run Mittelmann
  problems within the 7200 s timeout — the dense linear solver is the
  bottleneck. The sparse linear-solver swap planned for v0.9 targets
  this benchmark.

## [0.7.0] - 2026-04-23

First release where ripopt solves strictly more HS **and** more CUTEst
problems than native Ipopt (+2 HS, +1 CUTEst). The headline theme of
this cycle is closing the remaining behavioral gaps between ripopt and
Ipopt 3.14.x — most importantly the convergence semantics around the
bound multipliers `z_L`/`z_U`, the post-restoration multiplier reset,
the Mehrotra predictor-corrector, the KKT backward-error probe, and
the barrier-subproblem stop-test gate in Free-mode μ updates.

### Breaking changes
- **`refactor(ipm)!: drop `z_opt`, align convergence with Ipopt's
  iterative-`z` semantics** (e35407c). Previously ripopt kept a
  separate `z_opt` reconstruction used only by the convergence test;
  now the iterative `z_L`/`z_U` themselves are the convergence vector,
  matching Ipopt's `PDFullSpaceSolver` semantics. The
  `SolveDiagnostics::final_dual_inf_scaled` field has been removed —
  callers that consumed it should switch to `final_dual_inf`. Bumped
  to a minor release for this reason.
- `SolveStatus::Acceptable` was removed. Non-`Optimal` statuses now
  always surface honestly (`MaxIterations`, `NumericalError`,
  `RestorationFailed`, `Infeasible`). Consumers that treated
  `Acceptable` as success should audit their integration.

### Added
- **`ripopt-py`: direct Python interface with JAX autodiff** (b89f169,
  a9ba046). Exposes a persistent `Problem` class with dual warm start
  (`Problem.solve(lam0=, z_l0=, z_u0=)`, b95e5b8), accepts Ipopt-style
  option aliases (`mu_strategy`, `sb`, `bound_push`, cyipopt-style
  names, 4779b37), and is now published on PyPI (e32b57b) alongside
  `pyomo-ripopt` (8cce31a).
- **NaN/Inf evaluation hardening and C API parity** (66d77a1,
  42f4015). ripopt now matches Ipopt's behavior when user callbacks
  return NaN/Inf or signal failure: α-halving retry on post-step
  evaluation failure (2b5cb99), soft `EvalError → NumericalError`
  transition instead of hard abort (07cf37f), and structural parity
  with Ipopt's `TNLP::eval_*` error-handling contract.
- **`NlpProblem::new_x` flag for evaluation caching** (d138a25). User
  code can now short-circuit repeated evaluations at the same
  iterate, matching Ipopt's `new_x` contract. Existing trait
  implementations compile unchanged (the parameter defaults to `true`
  on old code paths).
- **`num-dual` automatic-differentiation example** (4aad98a) with a
  dedicated README section showing forward-mode AD through
  `NlpProblem`.
- **TSV direction-diff harness** for step-by-step comparison against
  a reference solver trace (e11832f), with an extended trace schema
  (α_max, τ, Σ condition number, SOC-accepted, c6178d6).
- **GAMS nlpbench benchmark harness** (eac674b). New `gams/Makefile`
  targets `bench-smoke` / `bench-small` / `bench-medium` / `bench-large`
  / `bench-all` drive the (vendored, gitignored) `gams/nlpbench/`
  test-set runner against both ripopt and ipopt and emit
  `BENCHMARK_REPORT_<size>_<version>.md`. Status returned by ripopt is
  mapped to nlpbench's signed-status convention so reports classify
  "locally optimal", "infeasible", and "iteration limit" correctly.
- **Adversary agent sweep** (new `adversary/runs/`). First full run of
  the automated NLP correctness-testing harness: Rosen-Suzuki, HS13,
  Discrete Boundary Value (n ∈ {20, 200, 500, 1000, 2000, 5000}),
  parametric projection, Powell badly-scaled. Four PASS; HS13 filed as
  issue #19 (solver limitation on a known LICQ/MFCQ-degenerate problem).
- **Reference-gap roadmap** (`docs/REFERENCE_GAP_ROADMAP.md`). ~700-line
  ripopt-vs-Ipopt and rmumps-vs-MUMPS gap analysis drafted with the
  ipopt-expert and mumps-expert agents, cataloging known deficiencies
  (D1-D10 for ripopt, (a)-(i) for rmumps), genuine advantages, and a
  ranked roadmap of 20 cross-cutting items (correctness-first).

### Changed
- **Post-restoration multiplier reset matches Ipopt exactly** (07dcdcc,
  20b51ce, af0bf09). The `z_L`/`z_U` reset after restoration now
  absorbs the correct contribution into the least-squares `y`
  re-solve, computed via Ipopt's exact augmented system. This was the
  source of several silent failures at the boundary between
  restoration and the main IPM.
- **Mehrotra predictor-corrector**: removed the skip gate that
  previously disabled corrections on highly infeasible iterates
  (72bae01); fixed the cross-term in the primal RHS and the `dz`
  recovery step (9deaff4); split Mehrotra vs filter-LS RHS
  assembly so the corrector and the line-search share no hidden
  state (eeae3d5).
- **Ipopt barrier-subproblem stop-test gate in Free-mode μ updates**
  (7f333de). μ no longer decreases until the current barrier
  subproblem passes Ipopt's stop test, preventing premature
  centering collapse on marginally feasible iterates.
- **Always verify KKT backward error; no IC-undoing iterative
  refinement** (66bce53). The `n+m ≥ 100` shortcut in
  `factor_with_inertia_correction` is gone — every accepted
  factorization now runs the backward-error probe. Iterative
  refinement no longer tries to undo the inertia-correction
  perturbation; Ipopt treats IC as part of the linear system, and so
  does ripopt now.
- **Reject rank-deficient solutions with huge magnitude** (3211838).
  When the solve produces a correction whose norm is inconsistent
  with the residual, the factorization is rejected and perturbation
  escalates, matching Ipopt's `PDPerturbationHandler` behavior.
- **Iterative refinement in custom-RHS and condensed solves**
  (96329a2). Custom-RHS solves (used by sensitivity, SOC, and the
  condensed-KKT path) now run the same iterative-refinement loop as
  the primary solve.
- **Dropped element-wise NaN checks on `grad_f` and `g`** (0ed77e1).
  Replaced with vector-norm finiteness checks, matching Ipopt.
- **Loqo μ-oracle monotone floor** (02471aa). Re-applies the floor
  that prevents μ from increasing inside a barrier subproblem.

### Fixed
- Documentation: rustdoc intra-doc link warnings from `[i]` index
  brackets — escape as `\[i\]` to prevent rustdoc from parsing them
  as link references.
- Benchmark runners: drop stale `final_dual_inf_scaled` references in
  the CUTEst runner (bc1fa21) and the HS native-ipopt runner
  (2780600).
- CI: gate `cat_a_probe` example behind the `cutest` feature (2b293e2).
- Two cyipopt-style spelling aliases were accepted on both the
  Python and Rust option paths (4779b37).
- **Preprocessing: skip redundancy detection on callback eval failure**
  (ff9144e). `detect_redundant_constraints` previously panicked when
  the user's `constraints()` callback returned `false` at the synthetic
  probe point; it now bails out cleanly, leaves the original
  constraint set intact, and lets the IPM handle the eval failure
  through the normal α-halving path.
- **Julia binding: status-code constants align with `RipoptReturnStatus`**
  (437bba0, 7733576). `Ripopt.jl` and the embedded C wrapper status
  enums had drifted from the Rust-side `RipoptReturnStatus`
  definitions; both are now regenerated from the canonical list so
  `MOI.TerminationStatus` reports the correct Ipopt-compatible code.
- **GAMS bridge: pass `index_style` to `ripopt_create`** (61aa808).
  The GAMS bridge was calling `ripopt_create` without the new
  `index_style` argument, causing 1-based / 0-based indexing confusion
  on GAMS-formulated NLPs.
- **Dead code removed** (6e91f55). LS-y helpers and unused imports
  pruned from `src/ipm.rs`, `src/kkt.rs`, and `src/c_api.rs`. No
  behavioral change.

### Performance
Fresh benchmark results (2026-04-21 on Apple Mac Mini, aarch64-apple-darwin):
- **HS suite**: ripopt **118/120 (98.3%)**, Ipopt 116/120 (96.7%).
  15.0× geometric-mean speedup on 116 commonly-solved, median 14.2×,
  ripopt faster on 113/116 (97%). ripopt-only solves: HS214, HS223.
  Ipopt-only solves: 0.
- **CUTEst suite**: ripopt **562/727 (77.3%)**, Ipopt 561/727 (77.2%).
  9.9× geometric-mean speedup on 525 commonly-solved, median 18.9×,
  ripopt faster on 440/525 (84%). ripopt-only solves: 37;
  Ipopt-only solves: 36.
- **Electrolyte thermodynamics**: ripopt 13/13, Ipopt 12/13. 17.5×
  geometric-mean speedup on 12 commonly-solved. Seawater speciation
  now takes 1,415 iterations (was 22 at v0.6.2) under the stricter
  v0.7.0 convergence semantics, but still solves where Ipopt declares
  Infeasible.
- **Grid (AC OPF)**: ripopt 3/4, Ipopt 4/4. Geometric-mean 2.8× on
  the 3 commonly-solved. See Notes below for the case30_ieee
  regression.

### Notes
- **case30_ieee regression**. At v0.7.0 ripopt reaches `MaxIterations`
  on PGLib-OPF `case30_ieee`, regressing from v0.6.2 which converged
  to a different local minimum (obj=8,609.66, 4.6% above the known
  optimum of 8,081.52, compared to Ipopt's 8,208.52 at 1.6% above).
  The regression is a side-effect of dropping the `n+m ≥ 100`
  shortcut in `factor_with_inertia_correction` (66bce53): the
  stricter backward-error probe now perturbs more aggressively on
  this rank-deficient AC-OPF Jacobian. The v0.6.2 "solve" was in
  fact converging to a different local optimum than Ipopt, so this
  is less of a correctness regression than a robustness regression;
  still, ripopt now solves one fewer grid problem than it did at
  v0.6.2. Tracked for a future patch.
- **Poisson 2.5K large-scale benchmark**. The Poisson 2.5K problem
  exhausts `max_iter` under the v0.7.0 convergence semantics and
  causes `make benchmark` to hang if run to completion; the large-
  scale sweep is therefore reported for the 4 problems that finish
  quickly (Rosenbrock 500, Bratu 1K, SparseQP 1K, OptControl 2.5K).
  Historical v0.6.2 timings for the other large-scale problems
  remain in `benchmarks/large_scale/large_scale_results.txt`. Full
  sweep is gated on a separate investigation.
- **ripopt-py** on PyPI as `ripopt-py` (direct JAX-backed interface)
  and `pyomo-ripopt` (Pyomo plugin). Workspace crates stay on
  semver: `ripopt` → 0.7.0, `rmumps` unchanged at 0.1.1.

## [0.6.2] - 2026-04-12

### Added
- **Reorganized benchmarks under `benchmarks/`** with one self-contained
  subdirectory per suite: `hs/`, `cutest/`, `electrolyte/`, `grid/` (renamed
  from `opf/`), `cho/`, `large_scale/`, `gas/`, and the new `water/`. Each
  suite has its own README and per-suite report. The composite report at
  `benchmarks/BENCHMARK_REPORT.md` aggregates HS + CUTEst + electrolyte +
  grid + CHO + large-scale; the gas and water suites are standalone (AMPL
  `.nl` interface, per-problem `.sol` output).
- **`benchmarks/water/`**: 6 water distribution network design NLPs from
  MINLPLib (Hazen-Williams head-loss formulation). Solved via the AMPL
  interface; ripopt matches the best-known primal bound for `water.nl`
  (963.13) where Ipopt converges to a different local minimum (1001.16).
- **`benchmarks/gas/`**: 4 gas pipeline NLPs from PDE-discretized Euler
  equations on pipe networks (gaslib11/40, steady/dynamic). Solved via the
  AMPL interface.
- **Loqo mu oracle** (`mu_oracle = "loqo"`) enabled by default, matching Ipopt's
  `mu_oracle=quality-function` strategy. Uses centrality measure
  `xi = min(z*s)/avg_compl` to set centering parameter `sigma`, preventing
  premature mu decrease from highly infeasible starting points. On gaslib11_steady:
  164→134 iters, NLP restorations 1→0, mode switches 24→8.
- **Sparse Gauss-Newton restoration**: sparse `J*J^T` factorization for GN restoration
  when m > 500, removing the dense Bunch-Kaufman bottleneck (6s/step → 0.02s for
  gas pipeline NLPs). Sparse LS multiplier estimates for post-restoration `y`
  initialization.
- **Ruiz equilibration KKT scaling** matching MUMPS `SimScale` schedule
  (KEEP(52)=7 for SYM=2): 1 inf-norm iteration + 3 one-norm iterations. Activated
  on demand when backward error is poor. Added `row_abs_sum()` to
  `DenseSymmetric`, `SparseSymmetric`, and `KktMatrix`.
- **Pretend-singular fallback** using Ipopt's normwise residual ratio (threshold
  1e-5). When iterative refinement cannot reach target accuracy, tries `delta_c`
  first (`PerturbForSingularity`), then `delta_w`.
- **Adaptive mu dual infeasibility safeguard**: prevents mu from collapsing to
  1e-11 while dual infeasibility remains large. Adds `du_floor` to barrier error
  and dual-infeasibility stagnation detection.
- **Structural degeneracy detection**: after 3 consecutive iterations requiring
  `delta_w > 0`, skips the unperturbed factorization trial on subsequent iterations.
- **Dense BK `increase_quality()`**: configurable pivot threshold with escalation
  0.64 → 0.8 → 0.95 → 1.0.
- **KKT factorization diagnostics** (dim, nnz, wall time) at `print_level ≥ 5`.
  Line-search rejection details at `print_level ≥ 7`.

### Changed
- **PretendSingular chain now matches Ipopt's `PDFullSpaceSolver`**:
  `solve → refine(fail) → IncreaseQuality → re-solve → perturbation`.
  Previously perturbation was applied before `IncreaseQuality`.
- **Iterative refinement**: max steps raised from 3–5 to 5–10 (matching Ipopt
  default). Stagnation detection extended to the non-IC path with a relaxed
  factor (1−1e−6 vs Ipopt's 1−1e−9).
- **Default sparse threshold** unchanged (n+m ≥ 110) but sparse GN restoration
  now activates at m > 500 even when the outer IPM is dense.
- Workspace version bumped — `ripopt` → 0.6.2, `rmumps` → 0.1.1.
- `ref/` directory and `pyomo-ripopt/build/` now ignored in git.

### Fixed
- **Fallback-result regression (`is_strictly_better`)**: when the main IPM
  reports `NumericalError` at a feasible iterate with a meaningful objective,
  a fallback solver that converges to a **worse** local minimum no longer
  silently replaces the main-IPM result. The comparator now requires either
  strict objective improvement (with primal feasibility ≤ 1e-4) or that the
  current result has no usable objective. This was the root cause of the
  `c_api_hs071_basic`, `c_api_hs071_multiplier_extraction`,
  `c_api_null_output_params`, `test_hs071_sensitivity_vs_finite_differences`,
  and `test_sensitivity_linear_prediction_accuracy` regressions introduced by
  the recent Loqo oracle / KKT quality-chain work.
- **Phosphoric-acid electrolyte test** (`electrolyte_05_phosphoric_acid`)
  pinned to `mu_oracle_quality_function=false`. The Loqo oracle steers the
  solver into a chemically-wrong local minimum (pH ≈ 11.84) of the Gibbs
  free-energy surface; the pre-Loqo default converges to the correct basin
  (pH ≈ 2.25). Documented in the test comment.
- **All compiler warnings** cleaned up in `src/ipm.rs`, `rmumps/src/frontal.rs`,
  and the adversary example suite.
- **13 adversary example files** updated to the current `NlpProblem` trait
  signature (removed `_new_x: bool` parameters and `-> bool` return types).
- `tests/large_scale_benchmark.rs` unused import cleaned up.

### Performance
- Fresh benchmark results (2026-04-11 on Apple Mac Mini, aarch64-apple-darwin):
  - **HS suite**: ripopt 115/120 (95.8%), Ipopt 116/120 (96.7%) — nearly tied.
    14.0× geometric mean speedup on 113 commonly solved, median 15.1×,
    ripopt faster on 111/113 (98%).
  - **CUTEst suite**: ripopt 553/727 (76.1%), Ipopt 561/727 (77.2%). 8.0× geometric
    mean speedup on 513 commonly solved, median 18.8×, ripopt faster on 415/513 (81%).
    ripopt-only solves: 40; Ipopt-only solves: 48.
  - **Electrolyte thermodynamics**: ripopt 13/13 (100%), Ipopt 12/13, 20.8× geo mean.
  - **Grid (AC OPF)**: 4/4 for both, Ipopt faster (0.2× geo mean).
- gas pipeline NLPs: sparse GN restoration reduces per-step cost 300× on m > 500.

### Notes
- The fresh benchmark shows a small shift in both solvers' solve counts vs. the
  0.6.1 numbers reported in the prior CHANGELOG. This reflects run-to-run
  floating-point sensitivity on borderline problems combined with the quality-chain
  changes; the dominant failure modes (NumericalError, LocalInfeasibility) are
  unchanged.

## [0.6.1] - 2026-04-05

### Fixed
- **Dual infeasibility stall on exp/log objectives (#7)**: Added z_opt fallback in the convergence check. When the kappa_sigma safeguard corrupts iterative bound multipliers (z_l, z_u), the unscaled convergence gate now accepts z_opt with component-wise scaling as an alternative, provided z_opt complementarity also passes. This prevents the solver from wasting thousands of iterations on problems where z_opt confirms the point is optimal but iterative z is stuck.

### Changed
- Exclude `adversary/`, `.crucible/`, and `research/` directories from crates.io package

## [0.6.0] - 2026-03-28

### Added
- **Julia/JuMP interface (`Ripopt.jl`)**: MathOptInterface wrapper enabling `Model(Ripopt.Optimizer)` with full JuMP support. Implements the incremental MOI interface (`supports_incremental_interface`, `copy_to`, variable bounds, `NLPBlock`, `Silent`, `TimeLimitSec`, `RawOptimizerAttribute`). Works on ARM/AArch64 (Apple Silicon) — all C callbacks are module-level functions, not closures, avoiding the trampoline requirement.
- **Julia example scripts**: `jump_hs071.jl`, `jump_rosenbrock.jl`, `c_wrapper_hs071.jl` in `Ripopt.jl/examples/`
- **Julia tutorial notebook**: `Ripopt.jl/examples/ripopt_jump_tutorial.ipynb` covering 6 examples (Rosenbrock, HS071, options, maximization, Ipopt comparison, economic dispatch)
- **rmumps genuine delayed pivoting**: CB (Bunch-Bunch) pivot search with look-ahead and delayed pivot acceptance
- **rmumps NEMIN amalgamation**: supernode amalgamation with minimum node size threshold (skipped for n ≥ 10000 where overhead exceeds benefit)
- System information output in benchmark scripts

### Fixed
- **`Ripopt.jl` precompilation**: `const libripopt` replaced with mutable global + `__init__()` so `RIPOPT_LIBRARY_PATH` is read at load time; the precompiled image is no longer tied to a specific build path
- **`Ripopt.jl` `copy_to`**: added `MOI.supports_incremental_interface` and `MOI.copy_to` — required for JuMP to transfer the model to the optimizer
- **`Ripopt.jl` option dispatch**: `Integer` check now precedes `Real` check so integer-valued options (e.g. `max_iter=500`) are routed to `AddRipoptIntOption` instead of `AddRipoptNumOption`
- **`Ripopt.jl` ARM `@cfunction`**: inner callback functions lifted to module level and `$` sigil removed; closure-backed cfunctions are not supported on AArch64
- **`Ripopt.jl` `_dummy_eval_h` ordering**: moved before `CreateRipoptProblem` — compile-time `@cfunction` requires the name to exist at lowering time
- **`Ripopt.jl` callback type widening**: `CreateRipoptProblem` now accepts `Union{Base.CFunction, Ptr{Cvoid}}` for all callback arguments
- **rmumps `factor_csc` threshold pivoting**: fixed correctness bug; reduced threshold to 1e-6
- **rmumps Ruiz scaling**: reverted KKT-aware variant back to standard Ruiz scaling (KKT variant caused CUTEst regressions)
- **CUTEst regression recovery**: recovered 552 → 569/727 Optimal after `nlp_lower_bound_inf` threshold changes caused regressions
- Fixed all compiler warnings in ripopt and rmumps

### Changed
- rmumps matching-based scaling and KKT matching ordering (added then reverted — standard defaults are more robust across problem families)
- rmumps NEMIN amalgamation skipped for n ≥ 10000 (amalgamation overhead exceeds benefit for large systems)
- KKT system improvements for medium-scale equality-constrained problems

### Performance
- CUTEst suite: **569/727** Optimal (up from 516/727 at v0.5.0 release, recovered from 552 after regression)
- HS suite: **118/120** Optimal, ripopt surpasses Ipopt (116/120)
- rmumps: genuine delayed pivoting reduces fill-in on indefinite systems

## [0.5.0] - 2026-03-16

### Breaking Changes
- **Removed `SolveStatus::Acceptable`**: problems that previously returned Acceptable now return either `Optimal` or `NumericalError`. This gives honest reporting — a solve either meets full tolerances or it doesn't. HS suite: 113/120 (was 119/120 with Acceptable); CUTEst: 516/727 (was 596/727).

### Added
- **Domain-specific benchmarks** integrated into `make benchmark`:
  - Electrolyte thermodynamics (13 problems): ripopt 13/13 (100%), Ipopt 12/13 (92.3%), 23.7x geo mean speedup
  - AC Optimal Power Flow (4 problems): ripopt 4/4, Ipopt 4/4
  - CHO parameter estimation (1 large-scale NLP, n=21,672, m=21,660): benchmark infrastructure for .nl file problems
- New Makefile targets: `electrolyte-run`, `opf-run`, `cho-run`
- JSON output from all domain benchmarks for unified reporting
- Domain benchmark sections in `benchmark_report.py` and BENCHMARK_REPORT.md
- `cho_benchmark.rs` example: benchmarks ripopt vs Ipopt on the CHO .nl problem

### Changed
- Convergence polishing: sigma in quality function, delayed mode switch, looser z_opt gate
- Conservative IPM retry added to diagnostic-driven fallback cascade
- NLP restoration: alternative sparse solver retry and relaxed timeout
- Removed overfitting heuristics: adaptive damping, backward-error refinement, scaled dual inf, cost-based fallbacks
- Tests require `Optimal` status; known solver limitations marked as `#[ignore]`
- Manuscript, supporting information, and README updated with current benchmark numbers

### Performance
- HS suite: 113/120 Optimal, geometric mean speedup 16.8x (on 111 commonly-solved)
- CUTEst suite: 516/727 Optimal, geometric mean speedup 11.2x (on 487 commonly-solved)
- Electrolyte suite: 13/13 solved, 23.7x geometric mean speedup vs Ipopt
- Recovered CRESC50 and DISCS via alternative sparse solver fallback
- Recovered MGH10LS via full iteration budget for unconstrained conservative retry

## [0.4.0] - 2026-03-15

### Added
- **Mehrotra predictor-corrector** enabled by default (`mehrotra_pc=true`), with Gondzio centrality corrections (`gondzio_mcc_max=3`) for better centering and fewer iterations
- **Dense condensed KKT for tall-narrow problems**: when m >> n and n <= 100, uses an n x n dense solve instead of (n+m) x (n+m) sparse factorization (up to 845x faster on problems like EXPFITC)
- **SuiteSparse AMD ordering** for rmumps: replaced custom O(n^2) AMD with the `amd` crate, fixing 10s+ ordering times on 40K+ dimensional systems
- **Early stall detection** (`early_stall_timeout=10.0`): bails out fast when stuck in early iterations to trigger fallback strategies
- **CLI `--help` flag**: lists all 50+ solver options organized by category with types, defaults, and descriptions
- **Quality function mu oracle** (disabled by default, `mu_oracle_quality_function`): evaluates barrier KKT error for candidate mu values
- **Large-scale benchmark with Ipopt comparison**: both solvers receive the same NlpProblem via Rust trait, up to 100K variables
- **faer** sparse solver as optional feature (default enabled alongside rmumps)

### Changed
- Default sparse direct solver switched back to rmumps (from faer) after fixing AMD ordering — rmumps multifrontal factorization is 5x faster than faer SparseLdl
- Mehrotra centering parameter (sigma) now feeds into cross-iteration mu update via geometric blend
- Sufficient progress check tightened (`refs_red_fact`: 0.9999 -> 0.999) for earlier Free-to-Fixed mode switching
- Wall-time check runs every iteration in early phase (was every 10)
- Skip expensive NLP restoration when approaching early stall timeout

### Fixed
- Compiler warnings: unused assignment (`tried_compl_polish`), dead code (`HsProblemEntry`)
- Missing fields in `HsSolveResult` for ipopt_native benchmark macro
- Sparse condensed Schur complement with near-full bandwidth (> n/2) now falls back to augmented KKT instead of attempting dense-equivalent sparse factorization

### Performance
- SparseQP 100K: 25.4s -> 4.9s (SuiteSparse AMD + rmumps vs faer)
- EXPFITC (n=5, m=502): 10.1s -> 0.012s (dense condensed path)
- OET3 (n=4, m=1002): 1.4s -> 0.009s (dense condensed path)
- cho_parmest (n=21672, m=21660): first factorization 10.9s -> 0.066s (faer AMD)
- ACOPR14: 60s timeout -> 0.4s (early stall detection + fallback)
- CUTEst geometric mean speedup vs Ipopt: 7.8x -> 9.1x
- HS suite geometric mean speedup vs Ipopt: 15.0x -> 16.5x

## [0.3.0] - 2026-02-14

Initial public release with full IPM implementation, CUTEst/HS benchmarks, C API, AMPL interface, and Pyomo integration.
