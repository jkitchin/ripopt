# A8 follow-up — arki0003 dual stagnation

Pick-up document for continuing the v0.8 alignment work after A7.9.
Written 2026-04-29 immediately after the A7.9 final validation.

## Where things stand (A7.9, 2026-04-29)

- Augmented (4-block) KKT port complete and validated. See §10 of
  `docs/V0.8_IPOPT_ALIGNMENT_PLAN.md` for the full A1–A7.9 evidence.
- HS 113/120, CUTEst 555/727 (vs ipopt 561/727 — within 1.0%),
  electrolyte 13/13, large-scale 10/10. Tests 295 lib green.
- Track C is done except T3.33 (shared restoration solver instance,
  large, deferred — needs a `LinearSolver` trait revision).
- The only `in_progress` task surviving is **#24 "Verify with
  arki0003 and adversary suite"** — and arki0003 still does not
  solve. That's what this doc is about.

## Reproducing the failure

```
timeout 600 ./target/release/ripopt \
    benchmarks/mittelmann/nl/arki0003.nl \
    -AMPL max_iter=3000 \
    > /tmp/arki0003.txt 2>&1
```

Problem size: n=1872, m=2138 (447 eq + 1691 ineq), 1041 linear
constraints (Hessian skipped on those rows), 8515 ineq-Jacobian nnz,
2037 Hess nnz. Dense for the augmented path.

## What the run does

**Iters 0–10**: bound-push clips `|x|_∞ = 1e7` (initial point) into
the box, but the iter-0 probe shows `|grad_f|_∞ = 1.000e0` at var
1871, `|J^T y|_∞ = 1.458e3`, `|grad_lag|_∞ = 1.458e3`. Step sizes
collapse to `α ~ 1e-4` and the filter rejects everything except
infinitesimal moves. inf_pr stuck at 1.16e8.

**Iters 11–50**: large multi-decade swings on the objective:
`-5.4e4 → -3.5e3 → +1.7e4 → -3.1e6 → -1.0e7 → +8.6e5 → +4.0e6`
(this is the "jumping around" the user observed). Eventually
the filter+SOC tame it and obj settles around 4–6 × 10³.

**Iters 76–217**: real descent; obj drops to ~3.83e3, inf_pr down
to 1e-7 levels, mu drops from 1e-1 to 3e-3.

**Iters ~600 onward**: hard dual stagnation. Same iterate every
step, no progress for 60+ iterations until killed:

```
600 — 661   3.7956303e3   7.54e-5   6.54e6   1.50e-1   6.54e-4   1.00e0   9.99e-1   0
```

`inf_pr ≈ 7.5e-5` (~near-feasible), `inf_du = 6.54e6` (frozen),
`compl = 0.15` (frozen), `α_pr = α_du ≈ 1.0` (full step every
iter), `mu = 6.5e-4`. The step is being computed, scored as
acceptable, applied — and the iterate doesn't change. Almost
certainly the perturbation handler is committing a large δ_x and
the resulting Δx, Δy are numerically below the iterate's working
precision once added back to `x`.

## Three candidate root causes (ranked)

### 1. Perturbation ladder absorbing the step (most likely)

`α = 1.0` with frozen iterate is the giveaway. The augmented
factor is succeeding (no `WrongInertia` retry visible) but the
recovered `(Δx, Δs, Δy_c, Δy_d)` is essentially zero relative to
the iterate magnitude. Prime suspect: `PDPerturbationHandler` (T3.27,
`kkt.rs::factor_with_inertia_correction`) committed to a large δ_x
in a previous iteration (`Degenerate` lock after
`DEGEN_ITERS_MAX=3`), and ripopt's `_last` field carries it
forward instead of resetting on a `SUCCESS` no-correction factor —
verify Ipopt's reset semantics (`IpPDPerturbationHandler.cpp::
finalize_test_status_`).

**To check:**
- Add `eprintln!` in `factor_with_inertia_correction` printing
  `(δ_x, δ_c, hess_degen, jac_degen)` per iter, run arki0003 to
  iter 700.
- If `δ_x` is large (>1e-2) and `hess_degen = Degenerate` for
  consecutive iters from ~590 onward, the lock is the bug.
- Cross-check: same trace from Ipopt with `print_level=10` on the
  same .nl. Ipopt should reset δ_x once an unperturbed factor
  succeeds.

### 2. Convergence test refusing to stop on a true KKT point

If inf_du = 6.54e6 is *spurious* (e.g. unscaled while the rest of
the test is scaled), ripopt is stuck checking a wrong residual.

**To check:**
- `convergence.rs::scaled_dual_inf` — confirm the Lagrangian gradient
  norm uses `s_d` correctly (T0.4 fix). Print
  `(grad_f, J^T y, z_L, z_U)` separately at iter 600 and check
  whether one term dominates.
- If `z_L − z_U` is the dominant 6.5e6 term and primal x is good,
  the dual multipliers got stranded. Probably (1) above propagated
  here.

### 3. Initial scaling

iter-0 probe shows `|grad_f|_∞ = 1.000e0` and `|J^T y|_∞ = 1.458e3`
with `obj_scaling = 1.000e0`. Ipopt 3.14's default
`nlp_scaling_method = gradient-based` would set
`obj_scaling = 1/max(1, |grad_f|_∞) = 1.0` (same), but
`con_scaling[i] = 1/max(1, |∇c_i|_∞)`. ripopt scaled 408/2138
constraints — confirm the **threshold**. Ipopt scales every row
where `|∇c_i|_∞ > 100` by default (`nlp_scaling_max_gradient`).
The 1730 unscaled rows might include rows with norms in [1, 100]
that don't need scaling, but check rows above 100 are *all* scaled.

**To check:**
- Print the unscaled rows' `|∇c_i|_∞` distribution. Anything
  ≥100 means ripopt's gradient-based scaler missed it.
- Compare against Ipopt's `nlp_scaling_max_gradient = 100` default
  applied to the same Jacobian.

## Adversary suite

The B-track verification task (#24) was originally going to run an
"adversary suite" — small problems hand-picked to stress slack
handling. None of those have been run since A7 landed. Candidates
already in the tree:
- `tests/lbfgs_ipm.rs` (L-BFGS smoke tests)
- `examples/issue_7_repro.rs` (exp/log dual stagnation reproducer)
- The 38 CUTEst regressions listed in §10's diff: BROWNBSNE,
  CERI651C, CONGIGMZ, CRESC4, DECONVB, FBRAIN3, GROUPING, GULFNE,
  HEART6, HIMMELBJ, HIMMELP5, HIMMELP6, HS16, HS59, LEVYMONE6,
  LEWISPOL, LOGHAIRY, MESH, MGH17, MSS1 + 18 more.

Most of those 38 went from `Optimal` to `MaxIterations` — same
class of issue as arki0003. If the perturbation-lock theory above
is right, fixing it would likely recover several of them.

## Suggested first session

1. **Instrument** `kkt.rs::factor_with_inertia_correction` with a
   per-call trace of `(δ_x_committed, δ_c_committed, trial_status,
   hess_degen, jac_degen, factor_status)`. Gate behind
   `RIPOPT_TRACE_PERT=1`.
2. Run `RIPOPT_TRACE_PERT=1 ./target/release/ripopt
   benchmarks/mittelmann/nl/arki0003.nl -AMPL max_iter=700` and
   diff iter ~580 (last real progress) vs iter ~620 (deep in
   stagnation). Hypothesis: δ_x flips from 0 to ~1e-4 around the
   transition and stays ≥1e-4 through 600+.
3. If confirmed, audit `IpPDPerturbationHandler.cpp::
   finalize_test_status_` (lines ~470–538) and compare to
   `kkt.rs::finalize_test` (~lines 719–774) for the
   `SUCCESS_AFTER_PERT → reset_to_zero` transition.
4. Land the fix as **A8.1**, re-run HS + the 38 CUTEst regressions
   + arki0003. Expected outcome: HS unchanged or +1, CUTEst recovers
   several MaxIterations cases, arki0003 drops below 1000 iters
   (may not solve to Optimal, but should at least make progress
   past the 661-iter freeze).
5. Whatever's left after A8.1: investigate the next dominant
   regression class (probably the `RestorationFailed` cohort:
   BROWNBSNE, CERI651C).

## Key files / line references

- `src/kkt.rs` — augmented + perturbation handler. Suspect: the
  4-cell test_status machine and δ_last reset semantics.
  `factor_with_inertia_correction` ~line 800; `finalize_test`
  ~line 719; `InertiaCorrectionParams` ~line 400.
- `src/kkt_aug.rs` — augmented assembly + step recovery.
  `aug_step_from_state{,_mehrotra}` are the call points;
  `aug_soc_solve_dx_factored` for SOC.
- `src/ipm.rs` — main IPM loop, `solve_full_augmented_direction`
  (~line 3729), `solve_with_quality_escalation` (~line 3274),
  `attempt_soc_aug` (~line 7700).
- `src/convergence.rs` — `scaled_dual_inf` and the residual
  composition (T0.4 territory).

## Don't do these

- **Don't add a "step too small, declare numerical breakdown"
  early-exit** — that hides the bug. The right fix makes the step
  non-zero. Document the failure honestly per CLAUDE.md.
- **Don't tune away the regressions by adjusting tolerances**.
  The CUTEst trade (−25 Optimal / +21 Acceptable / +50 fewer
  NumericalError / +68 MaxIterations) is the real signal of
  a perturbation-lock or convergence-test issue. Fix the cause.
- **Don't re-introduce the retired auto-retry fallbacks** to
  paper over arki0003 — that's the exact pattern A7 retired.

## Pointers

- Pre-A7 baseline: `benchmarks/cutest/results_v0.8.0-dev_baseline.json`
- Post-A7 baseline: `benchmarks/cutest/results_v0.8.0-dev_post-A7.jsonl`
- Plan: `docs/V0.8_IPOPT_ALIGNMENT_PLAN.md` §10
- Algorithm spec: `docs/IPOPT_ALGORITHM_SPEC.md`
- CLAUDE.md alignment-work exception: §3 of project root CLAUDE.md

## A8.1 — A8.4 progress (2026-04-29 follow-up session)

### A8.1+A8.2+A8.3: iter-0 dual init alignment (committed `e9c045d`)

The original A8 doc's hypothesis #1 (perturbation lock) was refuted —
ripopt's PDPerturbationHandler matches Ipopt's `reset_last=false` on the
3-block path and the 4-cell test machine is correct. The actual issue at
iter 0 was that the LS-y multiplier estimate was disabled by default
under a **misleading docstring**: the original comment claimed
`least_squares_mult_init` defaults to `false` "to match Ipopt", but Ipopt
3.14's `IpDefaultIterateInitializer.cpp:340` runs the basic LS-y solve
(via `least_square_mults()` in `IpLeastSquareMults.cpp:669-743`)
unconditionally as long as `constr_mult_init_max > 0` — that's the
default. `least_square_init_duals` (default `no`) is a separate stronger
LS solve and is NOT what controls the basic init.

Three coupled fixes:

- **A8.1**: default `least_squares_mult_init = true` (`src/options.rs`).
- **A8.2**: thread `z_L`, `z_U` through `compute_initial_y_with_ls` so
  the LS RHS becomes `∇f − P_L·z_L + P_U·z_U`, matching
  `IpLeastSquareMults.cpp:53-81` exactly. Without z, the LS over-fits
  a sparse ∇f and `‖y‖_∞` lands in the hundreds (still under the 1000
  discard threshold) on problems with O(1) z init.
- **A8.3**: when LS-y is active, skip the
  `y_d := v_U − v_L` overwrite in
  `initialize_constraint_slack_multipliers`. The piecewise overwrite
  ignores the `J_d J_c^T` off-diagonal coupling and re-introduces
  exactly the `J_d^T·(±1)` contribution the LS picked specifically
  to avoid. Ipopt's 4-block LS chooses `(y_c, y_d)` jointly.

Effect on Mittelmann arki0003 iter 0:
```
                |grad_f|_∞   |J^T y|_∞   |grad_lag|_∞
before A8.1-3:  1.00e0       1.46e3      1.46e3
after  A8.1-3:  1.00e0       1.00e0      1.00e0
ipopt 3.14:     1.00e0       1.49e0      1.49e0
```

arki0003 now reaches `obj = 3.7955e3` at iter ~232 (within 0.01% of
ipopt's optimal `3.7952e3`), a substantial improvement over the
previous max-iter freeze at `obj ≈ 3.83e3`. HS suite: 113/120
unchanged, no regression. 295 lib tests pass.

### A8.4: centering-stall diagnostic (committed `7e50976`)

After A8.1-A8.3, arki0003 makes real progress through iter 232, then
freezes for the remaining ~70 iters (run with max_iter=300). Per
A8 doc's protocol, instrumented `commit_trial_point` with an
`RIPOPT_TRACE_STEP=1`-gated probe that prints `α`, `‖Δx‖_∞`,
`‖α·Δx‖_∞`, `|Δx_eff|_∞`, `‖x‖_∞`, the achieved relative move,
`‖Δy‖_∞`, the smallest x-bound slack and which variable owns it,
the largest `|z|`, and the worst-case `Σ = z/s` diagonal entry.

Freeze-region trace (iter 232+):
```
[step] α=1.000e0 ‖Δx‖=1.665e-7 rel=1.164e-11 ‖Δy‖=1.115e-6
       min_s=1.506e-10 @ var 1650 (lower side)
       x = -9.849414e-9   bnd = -1.000000e-8
       max_z = 3.440e6    max_Σ ≈ 2.285e16
```

Decoded:
- Variable 1650's original lower bound is `0`. With
  `bound_relax_factor = 1e-8` applied (`apply_bound_relax_factor`
  in `src/ipm.rs:7389`), the bound was widened to `−1e−8`.
- Fraction-to-boundary has driven `x[1650]` to `−9.849414e-9`,
  i.e. **just inside the relaxed bound**. The slack against the
  relaxed bound is `1.506e-10` — this is the natural minimum
  slack with `bound_relax_factor = 1e-8` after the iterate
  parks at the bound.
- `z_L[1650] ≈ 3.44e6` to keep `z·s ≈ μ = 5.18e-4`. The κ_σ
  clamp is **silent** here because `z·s` is dead-center in the
  band `[μ/κ_σ, κ_σ·μ] = [5.18e-14, 5.18e6]`.
- The `Σ_x[1650] = z/s ≈ 2.29e16` diagonal entry **pins** that
  variable's component of the Newton solve. The recovered
  `‖Δx‖_∞ = 1.66e-7` on `‖x‖_∞ = 1.43e4` is `rel = 1.16e-11`
  (machine-epsilon noise).

Perturbation handler trace (`RIPOPT_TRACE_PERTURB=1`) in the
freeze region:
```
aug perturb-trace: dx=0.00e0 dc=0.00e0
  -> inertia(+3563, -2138, 0:0) target(3563+, 2138-, 0)
```
**δ_x = 0 throughout the freeze, exact target inertia, no
perturbation triggered.** This rules out the original A8 doc
hypothesis #1 (perturbation lock).

### Comparison with Ipopt 3.14 on arki0003

Ipopt solves arki0003 to optimal in **318 iters** with
`obj = 3.7952009506e3`, `inf_du = 9.59e-12`,
`Constraint violation = 5.42e-9`. During the analogous "tight"
phase (Ipopt iter 290-315), the per-iter `lg(rg)` column shows
`δ_x ∈ [10^-12, 10^-10]` — Ipopt **is committing perturbations**
in this regime even though inertia would be exact. Ripopt is
not.

Per ipopt-expert research session (recorded answers below):

1. **No Σ-pin detector exists in Ipopt.** I grepped the
   `src/Algorithm/` tree. `IpIpoptCalculatedQuantities::curr_sigma_x`
   (lines 3501-3540) computes `Σ` with no magnitude check, no clamp,
   no warning. There is no code that detects "single Σ entry pins
   the direction" and reacts.

2. **Ipopt's defense is upstream**, primarily through
   `bound_relax_factor` (`IpOrigIpoptNLP.cpp:343-358, 459-481`).
   Ripopt has this mechanism, applied identically. The 1.5e-10
   minimum slack is the **expected** result of the relaxation —
   not a bug in ripopt's bound handling.

3. **κ_σ is product-based, not ratio-based** (`IpIpoptAlg.cpp:1090`):
   it clamps `z*s ∈ [μ/κ_σ, κ_σ·μ]`, not `z/s = Σ`. With `z·s ≈ μ`
   in-band (the natural equilibrium), κ_σ is silent. Ripopt
   matches this exactly.

4. **MonotoneMuUpdate has no stall detector**
   (`IpMonotoneMuUpdate.cpp:130-200`). It only decreases μ when
   `sub_problem_error ≤ barrier_tol_factor·μ`. With `inf_du = 5e6`
   and `μ = 5.18e-4`, the condition is unmet → μ frozen, no
   restoration trigger from the μ side.

5. **The actual mechanism Ipopt uses to escape this regime is
   the IR-residual feedback loop in `IpPDFullSpaceSolver`**
   (`IpPDFullSpaceSolver.cpp:240-346, 795-820`). Specifically:
   - The IR loop computes
     `residual_ratio = ‖resid‖_∞ / (min(‖sol‖_∞, 1e6·‖rhs‖_∞) + ‖rhs‖_∞)`.
   - If IR cannot reach `residual_ratio ≤ residual_ratio_max = 1e-10`,
     it first calls `augSysSolver_->IncreaseQuality()`.
   - If that already happened or fails, it sets
     `pretend_singular = true`.
   - The outer perturbation loop then treats `pretend_singular`
     as a singularity and calls
     `perturbHandler_->PerturbForSingularity` (line 532), which
     escalates δ_x.
   - The **only** silent-accept path is when
     `residual_ratio < residual_ratio_singular = 1e-5` (info "S",
     line 323-329). Above that, perturbation is forced.

   At the freeze, the augmented matrix has a Σ ≈ 1e16 diagonal
   entry. With `‖rhs‖_∞ ≈ 5e6` and the recovered `‖sol‖ ≈ 1e-7`,
   the matvec residual is `M·sol − rhs ≈ −rhs` (since `M·sol`
   gets dominated by the Σ-pinned variable's near-zero component
   times its 1e16 row), so
   `residual_ratio ≈ ‖rhs‖_∞ / ‖rhs‖_∞ ≈ 1` — far above
   `residual_ratio_singular = 1e-5`. Ipopt would set
   `pretend_singular` and escalate δ_x.

### A8.5: proposed general fix — IR-residual-driven perturbation

**Problem.** Ripopt has all the components but the wiring is
incomplete:

- `solve_aug_with_ir` (`src/kkt_aug.rs:686-732`) computes
  `final_ratio` correctly via `residual_ratio` (matches the Ipopt
  formula) and `IR_RATIO_MAX_DEFAULT = 1e-10`.
- The caller at `src/kkt_aug.rs:1067-1073` receives
  `result.final_ratio` and **discards it** — only takes
  `result.sol`.

There is no feedback from a bad IR ratio to the perturbation
handler. The perturbation ladder
(`factor_aug_with_inertia_correction`,
`src/kkt_aug.rs:790-925`) is driven only by inertia mismatches,
never by IR residual quality.

**Fix shape (general, problem-agnostic).** Add the IR-residual
"fail-up" path that Ipopt uses:

1. Define `IR_RATIO_SINGULAR_DEFAULT = 1e-5` (from
   `IpPDFullSpaceSolver.cpp:809`) in `kkt_aug.rs`.
2. Wrap the
   `factor_aug_with_inertia_correction` + `solve_aug_with_ir`
   pair in a small loop:
   - On bad IR ratio (`final_ratio > IR_RATIO_MAX_DEFAULT`),
     first try `solver.increase_quality()` and re-IR.
   - If that already happened or still bad, call
     `perturb_for_singularity_pub` and re-factor (matrix is
     same shape, just δ_x escalates), then re-IR.
   - Cap escalations at e.g. 5 to avoid runaway.
   - If `final_ratio < IR_RATIO_SINGULAR_DEFAULT = 1e-5`, accept
     silently (this is Ipopt's info-string-"S" path; the
     direction is "good enough" relative to RHS).
   - If escalation cap hit and ratio still bad, return error
     so the caller can fall back to restoration (this matches
     Ipopt throwing `Restoration_Failed_Convergence_To_Stationary_Point`).
3. Apply the same wiring in `factorize_and_solve_aug_mehrotra`
   for the affine and corrector solves (`src/kkt_aug.rs`
   ~line 1340-1374).

**Why this is general.** This is a problem-agnostic mechanism:
the augmented system's residual quality is the universal signal
that a Newton direction is unreliable. It triggers identically
on Σ-pin (arki0003), on near-singular Hessians (degenerate
LICQ, MFCQ violation), on extreme conditioning from large `mu`
during early iterations, and on slack collapse from any source.
Per the §3 alignment exception in `CLAUDE.md`, the fix is
correctness against the Ipopt reference, not benchmark-tuning.

**Expected impact.**
- arki0003: should escape the freeze. Ipopt's `lg(rg) ≈ −10`
  at iter 290-315 indicates δ_x ≈ 10^−10 is enough. The
  iterate then either advances or enters restoration.
- 38 CUTEst regressions (BROWNBSNE, CERI651C, CONGIGMZ, etc.):
  several likely recover. Same Σ-pin / degenerate-direction
  pattern.
- HS: at most ±1, since HS problems are well-conditioned
  enough that the IR-bad path rarely trips. Worth verifying.

**Validation plan.**
1. Land A8.5 as one commit.
2. Run with `RIPOPT_TRACE_STEP=1` and
   `RIPOPT_TRACE_PERTURB=1` on arki0003 to confirm
   `δ_x` escalates through the previously-frozen iters and
   `‖Δx‖` becomes O(1).
3. Re-run HS suite (regression-free target: 113/120).
4. Re-run the CUTEst regression cohort listed in §10 of the
   v0.8 plan; record per-problem status changes.
5. Confirm arki0003 reaches Optimal or RestorationFailed,
   not MaxIterations with frozen iterate.

**Don't do these (still applies).**
- Don't add a Σ_max threshold — Ipopt has none, and a
  ripopt-specific clamp would diverge from the reference.
- Don't re-introduce the retired auto-retry fallbacks.
- Don't tune away regressions; A8.5's goal is the structural
  fix, not the score.

### Diagnostic infrastructure available for A8.5

- `RIPOPT_TRACE_STEP=1`: per-step `‖Δx‖`, `‖x‖`, slack/z/Σ
  diagnostic in `commit_trial_point`. Committed `7e50976`.
- `RIPOPT_TRACE_PERTURB=1`: per-attempt `(δ_x, δ_c, inertia)`
  trace in `factor_aug_with_inertia_correction`. Already
  present.
- (Optional A8.5-implementation-time addition) Trace
  `final_ratio` from `solve_aug_with_ir` so you can confirm
  the fail-up loop fires only when expected.

### A8.5: implemented and reverted — IR-residual feedback DOES NOT help (2026-04-29)

**Status: reverted. Negative result, kept here so future sessions
do not re-implement the same fix.**

A8.5 was implemented as designed above:

- Added `IR_RATIO_SINGULAR_DEFAULT = 1e-5` and
  `A85_MAX_ESCALATIONS = 5` to `src/kkt_aug.rs`.
- Added `factor_solve_aug_with_feedback` helper that wraps
  `factor_aug_with_inertia_correction` + `solve_aug_with_ir`
  and, on `final_ratio ≥ 1e-5`, calls
  `perturb_for_singularity_pub` and re-factors+re-IRs, capped
  at 5 escalations.
- Wired into `aug_step_from_state` (line ~1063) and
  `aug_step_from_state_mehrotra` (affine probe at ~1336).

**Head-to-head measurement** (`max_iter=1500`,
`max_wall_time=600`, both runs hit the wall-time cap):

| metric          | baseline (no A8.5) | with A8.5 |
|-----------------|--------------------|-----------|
| iterations      | 850                | 450       |
| final obj       | **3.7953e3** ✓     | 3.96e3 ✗  |
| final mu        | 1.57e-4            | **322** (mu went UP) |
| primal_inf      | 1.06e-2            | 1.65e-4   |
| dual_inf        | **1.57e6**         | 5.92e7    |
| compl           | 0.256              | 8.68e4    |
| restoration_count | 1                | 4         |
| mu_mode_switches  | 234              | 122       |

**Why A8.5 is harmful in arki0003 (and why the doc-A8.5
hypothesis was wrong about Σ-pin escape):**

1. **δ-perturbation cannot break a Σ-pin.** At the freeze, the
   pinned variable has Σ_x = z/s ≈ 2.3e16 on the (1,1) diagonal.
   `apply_aug_perturbation` adds δ_x to that diagonal. After 5
   escalations of `get_deltas_for_wrong_inertia`
   (factor 8 each step starting from `delta_w_init = 1e-4`),
   δ_x reaches ~ 0.4 — **30 orders of magnitude smaller than
   the Σ entry**. The matrix is unchanged in the directions
   that matter; the Newton direction is essentially the same as
   before.

2. **The "escaped" steps are noise.** The slightly different
   δ-perturbation produces a Newton step whose residual ratio
   is technically below `1e-5` but whose direction is still
   driven by the Σ-pinned variable. Each such step disturbs
   the dual update at full α=1, and the line search no longer
   has a clean Armijo signal.

3. **Disturbed dual updates → mu blowup.** With the iterate
   slightly off the central path, the adaptive μ strategy
   (mu_mode_switches=122 even with A8.5 active) ramps μ UP
   to drive feasibility, eventually reaching μ=322 — far
   worse than the original "frozen at near-optimal" state.

4. **Doc A8.5 §"Expected impact" was speculative.** The Ipopt
   `lg(rg) ≈ −10` evidence cited δ_x ≈ 1e−10 in Ipopt's
   arki0003 trace. But Ipopt does not have the Σ-pin in the
   same place ripopt does (different bound_relax interaction
   at iter 0; different fraction-to-boundary trajectory).
   Ipopt's δ-escalation works because it does not enter the
   Σ-pin regime at all; it is not "what saves Ipopt from
   Σ-pin."

**Real bottleneck (revealed by the longer baseline run):**
The freeze is not the disease — it is the IPM noise floor on
top of **diverging duals**. After 850 iters:

- x and s are correct (obj = 3.7953e3 matches Ipopt to 0.01%)
- primal_inf = 1.06e-2 (decent, not great)
- dual_inf = **1.57e6** (target: 1.0)
- compl = 0.256 (target: 1e-4)
- 234 mu_mode_switches (i.e., adaptive μ flips every ~3 iters)

Decomposing dual_inf = ‖∇f − J^T·y − z_L + z_U‖_∞:
‖∇f‖_∞ = O(1), ‖z‖_∞ = O(1e6), and on optimal x the J^T·y
contribution must cancel ∇f to within machine precision.
Instead it is contributing 1.57e6, meaning the y values are
≈ 6 orders of magnitude too large.

So the **A8.1-A8.3** iter-0 LS-y fix gave us a clean start
(|J^T·y|_∞ = 1.0 at iter 0, matching Ipopt's 1.49) but **the
dual updates that follow integrate y away** from the correct
values over the centering phase. Ipopt's analogous trajectory
keeps y bounded and reaches dual_inf < 1.0 in 318 iters.

**Hypotheses for the dual divergence (A8.6+ work):**

- The mu-strategy oscillation (234 switches over 850 iters)
  injects high-frequency noise into the y-update, and there
  is no damping. Ipopt's adaptive switch is more conservative
  about flipping (free-mode → fixed-mode is one-way except in
  specific recovery branches). Worth comparing the ripopt
  `MuStrategy::reset` triggers against Ipopt's
  `IpAdaptiveMuUpdate.cpp` mode-switch logic.

- A potential J-row scaling issue: if y is 1e6 times too
  large but J^T·y "looks right" mod-cancellation, the
  individual y_i may be fine but a few are huge. Need a
  per-constraint dual_inf decomposition (which i has the
  largest |y_i| at iter 850?).

- The dual-step fraction-to-boundary recurrence: if α_du is
  consistently capped at a small value (e.g., 0.01) in some
  iters, and α_pr = 1.0, then x advances on Newton's
  schedule but y trails. Over hundreds of iters this is a
  divergence. Worth tracing α_pr vs α_du at every iter.

- The `kappa_d` damping term in the gradient-of-Lagrangian RHS
  may be wrong: `compl_x_inf = max(s·z) − μ·κ_d` per Ipopt.
  Mis-aligned κ_d would steadily bias the dual-update RHS.

**A8.5 code state.** Reverted at HEAD. The instrumentation
constants and helper were removed via `git checkout
src/kkt_aug.rs`. 295 lib tests pass on the revert.

**A8.6+ next steps (do NOT re-implement A8.5).**

1. Add a per-iter trace of `‖y‖_∞`, the worst-|y_i|
   constraint index, α_pr vs α_du, and mu-mode in
   `commit_trial_point`. Run on arki0003 for 100, 300, 500,
   850 iters and look at how y drifts.
2. Compare against an Ipopt log on the same problem at the
   same iters (use `print_user_options=yes
   print_level=4`). Identify the iter where the trajectories
   first diverge in y.
3. From the divergence iter, work backward to the responsible
   subroutine — μ-switch, fraction-to-boundary, or RHS
   construction.

**Lessons.**
- "Implement what Ipopt does" without verifying the
  underlying assumption (Σ-pin escape via δ-perturbation) is
  a load-bearing trap. The Ipopt reference is correct **for
  Ipopt's iterates**; ripopt's iterates may be in a regime
  Ipopt never visits.
- Run the candidate fix to convergence (or wall-time) and
  compare ALL diagnostics, not just the freeze symptom. A8.5
  superficially "escaped the freeze" but in fact made every
  KKT measure worse.
- Always run the head-to-head with a disable env var
  (`RIPOPT_DISABLE_A85=1` here) before committing — same
  binary, two runs, one switch.

## A8.6+ findings — μ-mode mis-switch at iter 1 (2026-04-29)

Added `RIPOPT_TRACE_DUAL=1` per-iter dump (||y||_∞, worst-y_i,
α_pr, α_du, μ, mode, resto) at end of IPM loop. Ran arki0003 to
max_iter=200 and compared against `/tmp/arki0003_ipopt5.txt`
(`print_level=5` Ipopt 3.14 reference).

**Smoking gun at iter 1**:

|         | obj      | inf_pr  | inf_du | μ       | mode  | α_pr     |
|---------|----------|---------|--------|---------|-------|----------|
| Ipopt 1 | 1.13e4   | 1.16e8  | 1.49e0 | 1.0e-1  | Free  | 2.26e-4  |
| ripopt 1| 1.14e4   | 1.16e8  | 1.08e0 | 7.92e4  | Fixed | 2.28e-4  |

Identical primal trajectory (same obj, inf_pr, α_pr to 3 sig figs),
but μ explodes 6 orders of magnitude. From there ripopt's dual
chases the inflated μ: ||y||_∞ goes 0.99 → 2.2 → 28 → 5e4 → 1.5e7
in iters 0..54, all concentrated on row 1904 (an equality
constraint). Ipopt stays in Free mode for all 318 iters and
solves cleanly to obj=3.795e3.

Mode oscillation: 51 Free↔Fixed switches across 200 ripopt iters
(~25% of iters). Each switch back to Fixed re-runs
`switch_to_fixed_mode_with_adaptive_init`, re-seeding μ from
avg_compl × `adaptive_mu_monotone_init_factor`.

**Triggering call site** (src/ipm.rs:4253-4257):
```rust
let du_stagnant = compute_du_stagnant_in_free_mode(mu_state, options);
mu_state.consecutive_insufficient += 1;
if mu_state.consecutive_insufficient >= 2 || du_stagnant {
    switch_to_fixed_mode_with_adaptive_init(state, mu_state, filter, options);
}
```

`du_stagnant` requires window length ≥ 3 and so cannot fire by
iter 1. The trigger is `consecutive_insufficient >= 2`. The
counter is incremented every Free-mode iter that takes the `else`
branch (i.e., is not "sufficient + barrier_subproblem_solved").
On arki0003 this fires at iter 0 (counter=1) and iter 1
(counter=2 → switch). Ipopt does not switch this aggressively;
verifying the exact criterion via ipopt-expert.

**Verified**: A8.7 hoist is numerically equivalent (iters 0-19
bit-identical with/without). Re-applied; commit 8f6a129 stands.

## A8.8 result and diagnosis of dual-stagnation root cause (2026-04-30)

A8.8 commit 45dcf45 fixed the iter-1 mu-mode misswitch. ripopt
now reaches the right primal basin on arki0003: obj=3.7956e3 vs
Ipopt 3.7952e3, inf_pr=3.4e-5 vs Ipopt 5.4e-9.

Remaining symptom: from iter ~800 the iterate is bit-identical
each iter. inf_du=6.3e6 frozen, compl=0.286, μ=6.3e-4.
Per-iter ‖dy‖_∞ in the 1e6–1e8 range, all concentrated on
equality row 1904. ‖dx‖ in 700–30000 range, but α_pr in
1e-5–1e-1 keeps effective dx small.

Ipopt-expert review (`af596942e65477b75`) identified five
ripopt-vs-Ipopt discrepancies. Plan to bring ripopt to parity:

### Discrepancies vs Ipopt (cited from `ref/Ipopt/src/Algorithm/`)

1. **`apply_damped_y_update` heuristic (src/ipm.rs:2249)** — ripopt
   halves `dy` when the same component flips sign 3+ iters in a row
   (`near_convergence && sign_change_count >= 3`). Not in Ipopt.
   Ipopt's `BacktrackingLineSearch::PerformDualStep`
   (`IpBacktrackingLineSearch.cpp:919-1006`) updates y with the
   raw `α_y · dy` from the KKT solve. → **A8.9**

2. **kappa_d damping in `dual_infeasibility` (src/convergence.rs:319)**
   — ripopt's printed inf_du adds `+kappa_d·μ` for one-sided
   bound vars. Per ipopt-expert: `curr_dual_infeasibility`
   (`IpIpoptCalculatedQuantities.cpp:2682-2691`) calls the **plain**
   `curr_grad_lag_x()` without damping. The damping lives only in
   the augmented-RHS `curr_grad_lag_with_damping_x` (lines 2131-2227,
   used in `curr_grad_barrier_obj_x`). ripopt's T3.9 cites lines
   888-899 which are the wrong CQ. The error is small numerically
   (1e-9) but is a convergence-test misalignment. → **A8.10**

3. **`barrier_subproblem_solved` gate in Free-mode μ update
   (src/ipm.rs:4044, called at 4249)** — ripopt's Free-mode μ
   update only fires when `barrier_err <= barrier_tol_factor·μ`.
   This is Fixed-mode logic copied into Free
   (`IpMonotoneMuUpdate.cpp:135-194`). Ipopt's Free-mode `NewMu`
   (`IpAdaptiveMuUpdate.cpp:343-389`) updates μ from the oracle
   whenever `CheckSufficientProgress()` returns true; there's no
   barrier-solved gate. → **A8.11**

4. **DetectTinyStep terminator missing/misaligned**
   — Ipopt's `BacktrackingLineSearch::DetectTinyStep`
   (`IpBacktrackingLineSearch.cpp:1219-1279`): if `‖Δx‖∞/(1+|x|) ≤
   10ε` AND `cviol ≤ 1e-4` for two consecutive iters AND barrier
   subproblem solved, throws `TINY_STEP_DETECTED` →
   `STOP_AT_TINY_STEP` exit (`IpIpoptAlg.cpp:461-466`). Defaults:
   `tiny_step_tol=10ε≈2.22e-15`, `tiny_step_y_tol=1e-2`. ripopt's
   tiny-step path uses different thresholds and doesn't terminate
   at 2 consecutive. → **A8.12** ✅ (2026-04-30)

5. **κ_Σ multiplier reset (`correct_bound_multiplier`,
   `IpIpoptAlg.cpp:1055-1133`)** — Ipopt clamps `z_i ←
   max(min(z_i, κ_Σ·μ/s_i), μ/(κ_Σ·s_i))` after every dual step
   in `AcceptTrialPoint`. Default `κ_Σ=1e10` is essentially inert
   so this is a tertiary concern; verify ripopt's analogue
   (`reset_slack_multipliers`) runs every iter. → **A8.13** (low
   priority unless 1-4 don't suffice)

### Root-cause hypothesis for arki0003 freeze

The dy explosion (1e6-1e8) at row 1904 with correct inertia
suggests the augmented system has a small but non-zero
singular value at the equality row. Ipopt's
`PDPerturbationHandler` only escalates δ_c on detected
singularity (`zero > 0`). If inertia counts are exact but the
factorization is just ill-conditioned, neither Ipopt nor
ripopt would escalate δ_c — but Ipopt's iterates would never
reach this regime because the cumulative effect of 1-4 above
keeps Ipopt on a different trajectory. So the most
profitable fix order is the alignment fixes (1-4), then
re-test. Track whether dy magnitudes shrink.


## A8.11: drop barrier_subproblem_solved gate from Free mode (kept)

**Status: kept.** Verified by ipopt-expert against
`IpAdaptiveMuUpdate.cpp:343-389` (strict 2-way split) and lines 391-436
(unconditional oracle call). Free mode has *no*
`barrier_err <= kappa_eps * mu` test in Ipopt; that gate exists only in
Fixed mode (`IpMonotoneMuUpdate.cpp:135-194`). The previous ripopt
implementation added a third "stay in Free with conservative mu
decrease" branch and a "mu unchanged fall-through" — neither has an
analogue in Ipopt. Both were removed in commit `3f4c82d`.

**First attempt symptom (arki0003):** removing the gate exposed a
*downstream* mismatch: ripopt was running adaptive (Free) by default,
which invoked the QF mu-oracle at iter 1 with `compl ≈ 1e7` from the
infeasible starting iterate. The oracle returned mu=8.55e4 (capped only
by `mu_max_fact * initial_avg_compl = 1e12`), and mu oscillated wildly
through 415 iters.

**Root cause (A8.11.1):** ripopt's default `mu_strategy_adaptive: true`
mismatched Ipopt 3.14's default `mu_strategy = "monotone"`
(`IpAlgBuilder.cpp:355-362`). With the correct monotone default, the
Free-mode path doesn't run on arki0003 at all — Fixed mode holds
`mu = mu_init = 0.1` and decreases monotonically, matching Ipopt's
printed log iter-by-iter. Fixed in commit `fe111d4`
(`mu_strategy_adaptive: false` default in `src/options.rs`).

**Lesson per CLAUDE.md alignment-work principle:** when a
correct-against-Ipopt change regresses, do not revert; find the
upstream mismatch that the heuristic was masking. Here the masking
heuristic (`barrier_subproblem_solved` gate) was hiding a deeper
default-mismatch (`mu_strategy_adaptive`); fixing the deeper issue
makes the surface change benign.

**Follow-ups (open):**
- Long-run on arki0003 with monotone default still shows a mu spike at
  iter ~685 (mu jumped from ~1e-3 to 2.21e2). Suspect the
  ripopt-specific stall-recovery paths (`try_boost_mu_for_stall`,
  `handle_near_tolerance_stall` in `src/ipm.rs:4731,4809`) which
  unconditionally boost mu and switch to Fixed. These have no Ipopt
  analogue and should be either gated on adaptive strategy or removed.
- Audit ripopt's mu-oracle (`compute_quality_function_mu`,
  `compute_loqo_mu`) for safety at infeasible iterates so that adaptive
  strategy itself can be re-enabled per-problem without diverging.

## A8.15 — re-measurement after DEV-1..DEV-24 audit fixes (2026-04-30)

After landing 10 systematic Ipopt-alignment fixes from the v0.8 DEV
audit (commits `0f426d7` DEV-1 → `ace902a` DEV-23: kappa_d removal,
Loqo Free-mode fallback, monotone mu floor + uncapped cascade,
`Diverging_Iterates` rename, restoration success gate alignment,
restoration bound-push proximity, watchdog filter-gamma margins,
filter `theta_max` non-bump, switching-condition non-strict), the
arki0003 dual-stagnation pattern is **still present**. Trace at
`benchmarks/mittelmann/logs/ripopt/arki0003_post_dev_audit.log`.

### Current trajectory (max_iter=1500, hits 5min wall-time)

Cumulative DEV fixes change *when* the cascade fires but not the
underlying behavior:

```
iter  obj           inf_pr    inf_du    compl     lg(mu)
312   1.109e7       1.17e-4   8.35e3    2.32e-1   1.00e-1
313   1.102e7       4.45e-4   6.97e-1   5.15e-1   1.50e-4   ← 3-step cascade
314   1.087e7       2.18e-4   2.84e1    1.37e-1   1.50e-4   ← inf_du blows back
315-450  obj decreasing 1.1e7 → 2.8e6 with inf_du oscillating 100-700, mu pinned at 1.50e-4
```

The cascade math: from `mu=1e-1` Ipopt's `min(linear*mu, mu^super)`
gives 0.1 → 0.02 → 2.83e-3 → 1.50e-4 (three promotions). With
`mu_allow_fast_monotone_decrease=true` (Ipopt's default), Ipopt
loops `while sub_problem_error <= barrier_tol_factor*mu` and so
ripopt's three-step cascade in one iteration is **Ipopt-aligned** —
verified against `IpMonotoneMuUpdate.cpp:130-200`. Not the bug.

The post-cascade blow-up is also expected post-cascade behavior: at
the new `mu=1.5e-4`, the dual residual at the previous iterate is
naturally large because multipliers were tuned for `mu=1e-1`.

The actual fault: **mu stays pinned at 1.5e-4 forever** because
`barrier_subproblem_solved` (`barrier_err <= 10*mu = 1.5e-3`) is
never met again — `inf_du` stays in the hundreds.

### Comparison vs Ipopt (same problem, default options)

```
                    iter   obj         lg(mu)   inf_du
ripopt iter 312     312    1.11e7      -1.0     8.35e3
ipopt iter 302      302    3.7952e3    -8.6     5.50e1
```

Ipopt is at the **optimum's basin** (obj=3795, mu=2.5e-9) at iter
302. ripopt is still in the **far field** (obj=1.1e7, mu=1e-1) at
the same iter count. The cascade is not the problem — the *primal
trajectory* is wrong from much earlier than the cascade fires.

### What the DEV audit did and did not fix

The DEV audit corrected 10 misalignments in convergence test, mu
update, restoration gates, and filter math. None of these are on
the per-iteration step / line-search hot path. arki0003's symptom
is in the iter 1-300 trajectory: ripopt drives `‖y‖_∞ → 1e7` at
row 1904 while Ipopt holds `‖y‖_∞ = O(10)`. The DEV-fixes do not
touch the mechanism that determines this.

### Where to look next

1. **Step computation at the iter 100-300 regime.** The step at
   iter 312 has α_pr=1.37e-3 — typical of late-phase fixed-mu
   stagnation. Compare ripopt's iter-100 KKT solve with Ipopt's
   on the same iterate (need to dump x, y, z values at a specific
   iter and run both solvers from there).

2. **Dual update size limits.** At iter 1, ripopt's ‖y‖_∞ went
   0.99 → 2.2 → 28 → 5e4 → 1.5e7 over 54 iters with growth
   concentrated on row 1904 (an equality). Ipopt's bound on dual
   update size during the centering phase isn't currently
   enforced/matched.

3. **Filter sufficient-progress for h-type vs f-type at large
   theta**. arki0003 spends 70+ iters at `theta ≈ 1e8` (i.e.
   theta >> theta_min). Filter sufficient-progress in this regime
   permits weak θ-decrease only — verify alpha_min computation
   doesn't discard reasonable steps prematurely.

This is the open work for A8.15+ sessions; do not re-implement
the failed A8.5 IR-residual feedback (see §A8.5 above).

## A8.16 — re-measurement after cumulative DEV-30..DEV-36 (2026-04-30)

After landing six more Ipopt-alignment fixes
(`cdd3aab` DEV-30/31 split IsFtype + augmentation logic,
`ae61169` DEV-33 IR loop tries `increase_quality` before pretend-singular,
`3e9da05` DEV-32 alpha_for_y dxnorm,
`c1eead8` DEV-35 drop 40-iter LS cap,
`21c9cb4` DEV-36 wire theta_min_fact / theta_max_fact options),
arki0003 still walls out at the 5 min mittelmann timeout. Trace at
`benchmarks/mittelmann/logs/ripopt/arki0003_post_dev_30_36.log`.

Snapshot at the wall (iter 430, max_iter not reached):

```
iter  obj           inf_pr    inf_du    compl     lg(mu)
312   1.109e7       1.17e-4   8.35e3    2.32e-1   1.00e-1     ← pre-cascade (same as A8.15)
313   1.102e7       4.45e-4   6.97e-1   5.15e-1   1.50e-4     ← 3-step cascade fires
...
430   1.45e6        1.91e-3   3.69e2    6.02e0    1.50e-4     ← still pinned at mu=1.5e-4
```

Conclusion: DEV-30..DEV-36 do not touch the per-iteration step
trajectory in the iter 100-300 regime that drives the pre-cascade
`‖y‖_∞ → 1e7` blowup. As predicted in §A8.15 "What the DEV audit did
and did not fix". Open work items 1-3 in §A8.15 remain the right
direction (step-computation comparison vs Ipopt at iter ~100, dual
update size limits, filter sufficient-progress at large theta).

## A8.17 — DEV-34 and DEV-37 verified as audit false-positives (2026-04-30)

Independent ipopt-expert read of `IpBacktrackingLineSearch.cpp` and
`IpFilterLSAcceptor.cpp` against ripopt's `src/ipm.rs` and
`src/filter.rs` confirms that DEV-34 and DEV-37 are NOT real
misalignments and would be incorrect changes against Ipopt 3.14.

### DEV-34 — "watchdog progress drop missing is_acceptable clause"

**Verdict: ripopt already matches Ipopt.**

Ipopt watchdog acceptance: the forward iterate goes through the
*same* `acceptor_->CheckAcceptabilityOfTrialPoint(...)` call as a
normal step (`IpBacktrackingLineSearch.cpp:773`), with only the
reference iterate swapped from current to saved
(`IpFilterLSAcceptor.cpp:251-268, :510-512`). That single check
includes (a) `theta_max` guard, (b) F-type Armijo on phi or
sufficient reduction vs reference (`:361-374`), and (c)
`IsAcceptableToCurrentFilter` against the *live* filter (`:392`).

ripopt's `process_watchdog_trial` at `src/ipm.rs:2505-2507`:
```rust
let made_progress = filter.is_acceptable(theta_now, phi_now)
    && (theta_now <= (1.0 - gamma_theta) * saved.theta
        || phi_now <= saved.phi - gamma_phi * saved.theta);
```
already requires *both* current-filter acceptability AND sufficient
reduction vs the saved reference — matching Ipopt. Adding another
`is_acceptable` clause would just duplicate the existing filter
check.

### DEV-37 — "theta_max early-rejection should skip SOC/watchdog"

**Verdict: ripopt already matches Ipopt.**

Ipopt's `theta_max` early-reject at `IpFilterLSAcceptor.cpp:341-348`
sits inside `CheckAcceptabilityOfTrialPoint`, and *all* trial-point
contexts go through that single function:
- normal backtracking (`IpBacktrackingLineSearch.cpp:773`),
- SOC trial (`IpFilterLSAcceptor.cpp:629`),
- MPC/PD corrector (`IpFilterLSAcceptor.cpp:848`),
- watchdog forward step (same `:773`, only reference values
  change),
- soft-resto sanity probe (`IpBacktrackingLineSearch.cpp:1172`).

There is no flag to skip `theta_max` for SOC or watchdog. The audit
claim that Ipopt skips this guard in those contexts is incorrect.

ripopt's `Filter::check_acceptability` at `src/filter.rs:235` fires
the `theta_trial > self.theta_max` reject regardless of caller —
matching Ipopt. Skipping it would be a regression against the
reference.

### Status

DEV-34 and DEV-37 marked complete (no-op, verified against Ipopt
3.14). The DEV audit batch from §A8.15-§A8.16 lands at: 12 real
fixes (DEV-1, DEV-2, DEV-3, DEV-4, DEV-7, DEV-9, DEV-11, DEV-13,
DEV-23, DEV-24, DEV-30+31, DEV-32, DEV-33, DEV-35, DEV-36) plus 2
verified-not-misalignments (DEV-34, DEV-37).



## A8.12 result — DetectTinyStep alignment (2026-04-30)

**Status: landed.** Three corrections in `detect_tiny_step`
(`src/ipm.rs:3417-3500`) to match `IpBacktrackingLineSearch.cpp:1219-1278`
and the latch flow at lines 363-435:

1. **Slack-step check added.** Ipopt's `DetectTinyStep` requires
   `max_i |Δs_i|/(1+|s_i|) ≤ tiny_step_tol` in addition to the x-step
   gate. Without it, an iterate making real progress only on
   inequality slacks would be wrongly classified as tiny.

2. **Δy moved out of detection.** Ipopt does NOT include the dual
   step in `DetectTinyStep`. The `tiny_step_y_tol` threshold is the
   gate for the *latch* `tiny_step_last_iteration_` set at line
   421-424, used only to determine whether the *next* iter's
   detection should fire `tiny_step_flag`. ripopt previously gated
   the entire detection on Δy, which made detection more conservative
   than Ipopt and prevented `tiny_step_flag` from firing on iterates
   whose dy was 0.5 (still small but above the 1e-2 threshold).

3. **Two-iter latch via boolean, not counter.** Replaced
   `consecutive_tiny_steps: usize` with `tiny_step_last_iter: bool`,
   matching Ipopt's `tiny_step_last_iteration_`.
   `mu_state.tiny_step` (= Ipopt's `tiny_step_flag`) now fires iff
   *current iter detection* AND *previous iter latched*. The latch is
   refreshed each iter as `detection && (dy_amax < tiny_step_y_tol)`.

**Termination unchanged.** `pending_tiny_step_exit` is still set when
`tiny_step && state.mu == mu_before_update`, mirroring
`IpMonotoneMuUpdate.cpp:158-160` and `IpAdaptiveMuUpdate.cpp:330-332,377-379`.
The redundant `consecutive_tiny_steps >= 2` gate was dropped because
`mu_state.tiny_step` already encodes the two-iter requirement.

**Not done in A8.12 (deferred).** Ipopt also takes a frac-to-bound
primal step *bypassing the line search* when `DetectTinyStep` returns
true (`IpBacktrackingLineSearch.cpp:383-431`). ripopt currently routes
tiny-step detection iterates through the normal line search. This is
a larger surface-area change and is left as a follow-up; it may
matter on problems where the line search keeps shrinking α toward 0
at machine-precision noise.

**Tests.** Five unit tests cover: detection no-op on mu/filter,
prior-latch requirement, step-grows reset, dy-only-gates-latch
(the key correction — detection fires even with large dy if prior
latched), and cviol-blocks-detection. All 294 lib tests pass.

**Effect on arki0003.** No expected delta — the dual-stagnation
trajectory described in §A8.15 doesn't trip the tiny-step gate
(Δx is not at machine-precision noise; the iterate is making "real"
moves in dy). A8.12 is a *correctness* alignment for problems that
naturally enter the machine-precision-step regime, not a fix for
arki0003 specifically. The next focus is the iter-100 KKT-solve
comparison (Task #42) per §A8.15.

## A8.18 — divergence point identified (2026-04-30, Task #42)

**Iter-110 trajectory comparison** between ripopt (post-DEV-audit) and
Ipopt 3.14 on arki0003 reveals an essentially identical iterate state
at iter 110, with bit-different next-iteration behavior:

| iter | obj            | inf_pr  | inf_du  | mu     | α_pr     | α_du     | flag |
|------|----------------|---------|---------|--------|----------|----------|------|
| ripopt 110 | 1.166e7 | 3.12e5 | 5.40e6 | 1e-1 | 1.85e-3 | 1.24e-1 | (regular) |
| ipopt  110 | 1.195e7 | 3.12e5 | 3.57e6 | 1e-1 | 2.21e-4 | 5.30e-3 | (regular) |
| ripopt 111 | 1.166e7 | 3.12e5 | 5.40e6 | 1e-1 | 1.94e-5 | 9.93e-6 | (regular, accepted!) |
| ipopt  111r| 1.195e7 | 3.12e5 | 1.00e3 | 5.5  | 0       | 3.14e-7 | **R** restoration |
| ipopt  112r| 6.075e6 | 3.02e5 | 1.01e3 | 5.5  | 2.00e+11 | 2.94e-5 | R |
| ipopt  113r| 5.016e6 | 2.87e3 | 1.02e3 | 3.4  | 1.02e+09 | 1.04e-3 | R |
| ipopt  114 | 5.016e6 | 2.87e3 | 1.02   | -1.0 | 4.16e+5 | 7.27e-5 | (regular) |

**Diagnosis**: at iter 110→111 Ipopt's regular filter line search
exhausts down to `alpha < alpha_min` without finding an acceptable
trial point, triggering the restoration phase
(`IpBacktrackingLineSearch.cpp:516-602`). Restoration delivers a
massive feasibility improvement (inf_pr 3.12e5 → 2.87e3, two orders
of magnitude) and the regular IPM resumes at a vastly better point.

ripopt at the *same* iterate **accepts** a microscopic primal step
(α_pr=1.94e-5, ls_steps=0 — accepted on first try without any
backtracking). The next iter has α_pr=1.94e-5, then 2.06e-7, then
6.17e-8 — the iterate is bit-identical for at least 5 iters with
no restoration trigger. This means ripopt's filter is accepting
a "no-op" step where Ipopt's filter rejects.

**Hypothesis (to verify next)**: ripopt's `filter.check_acceptability`
or its Armijo/switching decision tree differs from Ipopt's
`IpFilterLSAcceptor::IsAcceptableToCurrentIterate` such that a step
with `theta_trial ≈ theta_current` AND `phi_trial ≈ phi_current`
passes ripopt's test but fails Ipopt's. With theta_current=3.12e5
and γ_theta·theta=3.12 and γ_phi·theta=3.12e-3, both
"sufficient decrease" tests should fail at α=1.94e-5 → trial should
be unacceptable. The fact that ripopt accepts means the test logic
itself is misaligned.

**Next investigation (Task #43)**: instrument ripopt at iter 110-115
on arki0003 with `RIPOPT_TRACE_FILTER=1` (or print_level=7) to log
(theta_trial, phi_trial, theta_current, phi_current, gamma_theta,
gamma_phi, switching_holds, armijo_holds) for the first ls_steps=0
trial. Compare against Ipopt with `print_level=12` filter trace at
the same iter. Identify the exact rule whose result differs.

**Why this was missed by prior DEV audit**: DEV-23/24/30/31/32
all touched filter mechanics but stayed in the abstract test logic
(switching condition, augmentation gate, IsFtype split). None
exercised a regression test where `theta_current ≈ theta_trial` at
machine-relative precision — exactly the scenario at iter 111.

**Status**: §A8.15's "iter-100 KKT-solve comparison" deliverable
collapsed to a much simpler finding — the KKT *solution* is fine
at iter 110 (both solvers compute essentially the same Δx that
respects the bound buffers giving α_max = O(1e-3 to 1e-5)). The
divergence is in the *line search filter test*, not the linear
solve. This shifts focus from KKT/inertia to filter-test alignment.

## A8.19 — filter θ uses box-violation, not slack-coupling (2026-04-30, Task #43)

**Finding**: ripopt's filter line search computes `theta` as the
1-norm of `g(x)`'s box violation against `[g_l, g_u]`, but Ipopt's
`IpCq().curr_constraint_violation()` returns
`||c(x)||_1 + ||d(x) − s||_1` where `s` is the explicit inequality
slack iterate (`IpIpoptCalculatedQuantities.cpp:1468-1473,
2570-2610`). These are different quantities once the IPM iterate
has `s ≠ projection(g(x))`.

Concretely:

- ripopt at `src/ipm.rs:7508` defines
  `theta_for_g(state, g) = primal_infeasibility(g, g_l, g_u)`,
  which is `Σ max(g_l[i]−g[i], 0) + max(g[i]−g_u[i], 0)` —
  the box-violation of `g(x)` alone, **slack-free**.
  This is the function used at every line-search trial site:
  `evaluate_trial_point` (line 1831), `attempt_soft_restoration`
  (line 3269), and the SOC `theta_prev_soc` initialiser
  (line 6240).
- ripopt's `state.constraint_violation()` (`ipm.rs:1263`) calls
  the same `convergence::primal_infeasibility` — slack-free —
  and is used as `theta_current` at the top of each iteration
  (`ipm.rs:5720`) and as the iter-log `inf_pr` column.
- Ipopt's filter test instead reads `IpCq().curr_constraint_violation()`,
  which sums `|g[i] − g_l[i]|` for equality rows and `|g[i] − s[i]|`
  for inequality rows (`IpIpoptCalculatedQuantities.cpp:2596-2602`,
  `Norm1` overload). Slack-coupled.
- ripopt's own helper `convergence::primal_infeasibility_internal`
  (and its `_max` variant) computes the slack-coupled form. This
  helper *is* used in the barrier-level convergence test
  (`compute_primal_inf_internal_max_at_state`, `ipm.rs:7789`) and
  in the SOC RHS (`ipm.rs:6219-6220`), but not in the filter trial
  path nor in `state.constraint_violation()`.

**Evidence**: at iter 111 of arki0003 ripopt's first trial accepts
α=1.94e-5 with `ls_steps=0` because the filter h-type test passes
on `theta_trial ≈ theta_current`. Under box-violation flavour the
trial slack `s_trial = s + α·ds` does not appear in `theta`; the
quantity that *does* move under the slack-Newton step (`d(x) − s`)
is invisible. Ipopt's slack-coupled `theta` is generally larger
than ripopt's box-violation `theta` because for an inequality row
with `g[i] outside [g_l, g_u]`, `|g[i] − s[i]| ≥ |box_violation|`
(s ∈ (g_l, g_u) is strictly inside the box). The acceptance
threshold `gamma_phi · theta_current` is therefore artificially
small in ripopt, making the h-type phi-only test laxer than
Ipopt's.

**Plan to align (Task #43)**:

1. Add `theta_for_g_s(state, g, s)` helper using
   `primal_infeasibility_internal(g, s, g_l, g_u)`.
2. Update `evaluate_trial_point` to compute
   `s_trial = s + α·ds` and pass it to the helper. Frac-to-bound
   on `s` is enforced upstream by `compute_alpha_max`, so
   `s_trial` is feasible for all `α ≤ alpha_primal_max`.
3. Update SOC `theta_prev_soc` (`ipm.rs:6240`) and soft-restoration
   `theta_trial` (`ipm.rs:3269`) to use the slack-coupled form
   with the appropriate trial slack.
4. Switch `state.constraint_violation()` to slack-coupled form so
   the iter-level `theta_current` (and the `inf_pr` column) match
   Ipopt's `curr_constraint_violation`.

Step (4) is the higher blast-radius change — it affects the
restoration cascade entry, the post-cascade convergence check, and
the diagnostic output. Steps (1)-(3) are surgical and limited to
the filter line-search trial path. The fix is in scope for Task #43.

### A8.19 implementation result (2026-04-30, Task #43 closure)

Implemented steps (1)-(3) of the alignment plan: surgical scope
limited to the filter line-search trial path. Step (4)
(`state.constraint_violation()` flip) deferred — higher
blast-radius and the iter-level `theta_current` was localized via
`theta_for_g_s(&state, &state.g, &state.s)` so the filter pipeline
sees slack-coupled θ end-to-end without disturbing the diagnostic
column or restoration entry.

Code changes (src/ipm.rs):

- Replaced `theta_for_g` with `theta_for_g_s(state, g, s)` and
  added `compute_trial_slack(state, alpha) → Vec<f64>` helper
  (line ~7508, ~7563).
- `evaluate_trial_point` (line ~1831): computes `s_trial = s+α·ds`
  and uses slack-coupled θ.
- SOC pipeline (line ~6219-6307): `s_soc` built per inequality row
  from the SOC d-step `ds_d_soc`; `theta_prev_soc` and
  `theta_soc` use slack-coupled θ.
- Soft restoration trial (line ~3269): uses
  `theta_for_g_s(state, &state.g, &state.s)` (no step taken;
  current slack is the trial slack).
- `theta_init` (line ~5416) and per-iter `theta_current`
  (line ~5735): slack-coupled.
- Restoration recovery sites: `theta_for_g_s(state, &g_new,
  &state.s)` since IPM s is preserved across the recovery.

Verification:

- All 294 lib tests pass; `hs_tp044` integration test now passes
  (was MaxIter on baseline) — incidental improvement.
- arki0003 with `max_iter=200`: still terminates at iter 199 with
  `obj=1.22e7`, `inf_pr=7.46e3`. Iter 110-115 still shows the
  microscopic-α acceptance pattern (α=1.94e-5, ls_steps=0, θ
  unchanged at 3.12e5). `filter_rejects=0`, no restoration.

Why arki0003 wasn't fixed: the IPM Newton step is a local descent
direction for θ, so even at α=1.94e-5 the trial yields
`theta_trial ≈ (1−α)·theta_current` ≈ 3.12e5·(1−1.94e-5),
which still satisfies `theta_trial ≤ (1−γ_θ)·theta_ref` because
γ_θ=1e-5 is the *same* small constant. Slack-coupling shifts the
absolute magnitude of θ but does not change the algebraic
relationship `θ_trial/θ_ref = 1−Θ(α)`. So the h-type test passes
identically before and after the fix at this iterate.

The real arki0003 root cause must therefore be elsewhere — likely
in either:
- the α_max (frac-to-bound) computation pinning the first trial
  to such a small value at iter 110→111, or
- the α_min computation: γ_α·γ_θ (≈ 5e-7 with γ_α=0.05) leaves
  α=1.94e-5 well above α_min, so even an exhaustive backtracking
  line search would accept the same trial. Compare to Ipopt's
  α_min — if Ipopt computes a larger α_min at iter 110, the
  microscopic α would be rejected outright and restoration
  triggered.

A8.20 follow-up: trace α_max and α_min at iter 110 in both
solvers. Hypothesis: ripopt's α_max is correct (it's just
frac-to-bound on x and s) but α_min may be missing the
`gamma_phi*theta/(-grad_phi*d)` clause or the `delta*theta^s_theta`
term, which would make it too generous in this regime.

The slack-coupling fix is committed regardless because it is
correct against the Ipopt reference (`IpCq::curr_constraint_violation`)
and resolves the apples-to-oranges issue between the filter test
and the IPM convergence test, and unrelatedly enables `hs_tp044`.

## A8.20 — iter-110 "convergence" was illusory (2026-04-30, Task #44)

**Method**: ran Ipopt 3.14.19 with `print_level=12 max_iter=115` on the
same `arki0003.nl` and compared filter-trace internals against the
post-A8.19 ripopt log.

**Finding**: ripopt and Ipopt had *not* converged to the same iterate by
iter 110. The A8.18 finding that "both reach an essentially identical
iterate at iter 110 (obj~1.17e7, inf_pr=3.12e5)" was misleading because
only the `inf_pr` column matched at machine-printed precision; objectives
and inf_du differed substantially.

Side-by-side iter 100-110 (objective):

| iter | Ipopt obj    | ripopt obj   | Ipopt inf_du | ripopt inf_du |
|------|--------------|--------------|--------------|---------------|
| 100  | 1.1799421e7  | 1.1389183e7  | 1.23e8       | 2.74e6        |
| 105  | 1.1874710e7  | 1.1508713e7  | 2.55e7       | 4.04e6        |
| 109  | 1.1945968e7  | 1.1655002e7  | 3.51e6       | 4.90e6        |
| 110  | 1.1945981e7  | 1.1655116e7  | 3.57e6       | 5.40e6        |

Δobj ≈ 2.5% by iter 100, persisting through iter 110. ripopt's inf_du
profile is qualitatively different from Ipopt's: Ipopt suffers a one-iter
spike to 1.23e8 at iter 100 (and a 7.99e6 spike at iter 106) while
ripopt's inf_du grows monotonically from 2.74e6 → 5.40e6 over the same
range.

**At Ipopt iter 110**: `reference_theta=3.921e5` (iter-109→110 line search
seed) and `reference_gradBarrTDelta=1.252e7 > 0` — no barrier descent, so
Ipopt's only acceptance test is h-type. `ALPHA_MIN = 5.000e-7` matches
ripopt's `α_min = α_min_frac · γ_θ = 0.05 · 1e-5 = 5e-7` exactly.

**Verified `compute_alpha_min` is correctly aligned** (`src/filter.rs:370-393`
vs `IpFilterLSAcceptor.cpp:450-469`). All four clauses present and the
α_min_frac=0.05 multiplier matches.

**The iter-111 microscopic-α acceptance is algebraically inevitable in
both solvers** at the iterates each one is at:

- Newton step satisfies `J·d = -c(x)` so `c(x+αd) ≈ (1−α)c(x)` at first
  order. With α=1.94e-5 > γ_θ=1e-5, the h-type test
  `θ_trial ≤ (1−γ_θ)·θ_ref` passes by an algebraically-required margin.
- Ipopt's iter 110 `Step Calculated` trace shows three "Checking
  sufficient reduction" entries with `reference_theta=3.921e5` and `gBD>0`
  before accepting at `α=2.21e-4` (h-step). Ipopt's iter 111 line search
  presumably also accepts on first try given the same algebra.

**Real divergence point**: ripopt and Ipopt take materially different
α-paths starting from iter 3-9. By iter 9 ripopt is at obj=2.74e5 vs
Ipopt 2.85e5 (4% gap). The gap widens through iter 90-100 where ripopt's
inf_du diverges from Ipopt's spike pattern. This is consistent with a
*step-direction* difference (different KKT solve, scaling, or
perturbation), not a *line-search-acceptance* difference.

**Closes Task #44**: `compute_alpha_min` and `compute_alpha_max`
(frac-to-bound) are correctly aligned. Iter-111 acceptance is correct
behaviour — both solvers do this. The reason ripopt converges differently
on arki0003 is upstream of the line search: the *direction* differs,
likely in early iterations (iter 3-9 already show 4% obj gap) and again
at iter 95-100 where the inf_du profiles diverge qualitatively.

**Implication for the v0.8 alignment effort**: arki0003 is no longer
useful as a single-bug diagnostic. The 2.5% obj gap by iter 100 means
this problem exposes *cumulative drift* across many iterations rather
than one alignable heuristic. Better candidates for further
filter/line-search alignment audit:
- Problems where ripopt and Ipopt agree to <0.1% on objective until a
  specific iter, then diverge sharply (one-shot heuristic mismatch).
- Problems where ripopt's iter-by-iter `α_pr` differs systematically
  from Ipopt's by a constant factor (e.g. 0.5x or 2x), pointing to a
  specific α_init or τ_min difference.

A8.21 (if pursued): pick a smaller problem from the failure set and run
the same iter-by-iter comparison. Alternatively, accept arki0003 as a
"hard problem" that requires Ipopt-level numerical robustness across
the full IPM stack and refocus the alignment effort on cleaner cases.

## A8.21 — element-level iter-0 dx diff vs Ipopt (2026-04-30, Task #45)

**Trigger**: user pushback on A8.20 — "This doesn't make sense. It implies
inaccuracy somewhere in the ripopt/feral stack." Element-level diff to
localize.

**Method**: instrument `src/ipm.rs` post-`install_step_directions` with an
iter-0 probe (`||·||_inf`, top-5 signed/index, per-bound dump at three
canonical slots). Mirror the data in Ipopt 3.14.19 print_level=12 trace
on `arki0003.nl`.

**Findings (per-variable)** — slot index uses ripopt 0-indexed (n=1872);
AMPL annotation is the user-facing variable name from Ipopt's trace:

| slot | AMPL    | ripopt dx                         | Ipopt dx                          | gap     |
|------|---------|-----------------------------------|-----------------------------------|---------|
| 1801 | x1850   | +9.8235739400465982e-2            | +9.8235739400465982e-2            | bit-exact (17 digits) |
| 1871 | x2283   | -4.9989773495065801e+7            | -4.9988622038987763e+7            | 0.0023% |
| 1753 | x1802   | +1.2807967345267247e+1            | +1.2065061724475607e+1            | 6.16%   |

`||dx||_inf = 4.999e7` in both, dominated by the unbounded slot 1871 — the
infinity-norm hides the structural gap. The 6.16% gap at slot 1753
manifests downstream as a 6.2% `dz_L` gap at the same variable.

**Where the gap is *not***:

1. **Perturbation handler** — `delta_w_used = 0`, `delta_c_used = 0`,
   `delta_w_last = 0`, `delta_c_last = 0` at iter 0. Matches Ipopt's
   trace verbatim ("Solving system with delta_x=0.000000e+00
   delta_s=0.000000e+00").

2. **Linear-solver accuracy (feral)** — IR `final_ratio = 1.03e-15` after
   `ir_iters = 1`. The augmented matrix `A` satisfies `A·sol ≈ rhs` to
   ~15 decimals. **feral is not the source.** This rules out the original
   "feral inaccuracy" framing.

3. **Variable scaling** — `nlp_scaling_method = gradient-based` (default
   in both) scales objective and constraints, not variables. Both solvers
   use identity x-scaling unless `user_x_scaling` is supplied.

4. **Initial point** — ripopt: `x[1753] = 9.99999e-3` (i.e. `x_l + bound_push`
   with `bound_push=1e-2`, `x_l = -1e-8`). Both solvers default
   `bound_push = 1e-2`; a 6% x_0 difference would require divergent
   bound-push semantics, which the bit-exact match at slot 1801 rules
   out (same bound layout: `x_l = -1e-8`, `x = 9.99999e-3`).

**Conclusion**: at this seed, the assembled augmented KKT system itself
(matrix `A` or rhs `b`) differs from Ipopt's at the rows touching variable
`x1802`, even though the bound-coupled inputs (`x`, `x_l`, `z_l`) appear
identical. Remaining candidates:

- **Hessian entry** `H[1753, *]` — different sparsity or values from
  Ipopt's `IpEvalHessian`. The objective Hessian on arki0003 is dense in
  certain rows; numerical evaluation differences in AMPL .nl interpretation
  (or one-sided vs structural Hessian sparsity) could leak in.
- **Jacobian column 1753** — at iter 0 with `y_0 = 0`, `J^T·y = 0`, so
  this only matters via the Σ structure folded into the augmented system.
- **Σ_s contributions through slack rows coupled to constraint rows that
  reference variable 1753** — ripopt's slot 1801 (also `x_l = -1e-8`,
  `(x-x_l) = 0.01`) matching bit-exactly while 1753 differs 6% means the
  delta is not in the bound geometry; it's in the constraint coupling.

**Status**: instrumentation shipped; root-cause to (Hessian eval | Jacobian
eval | constraint-row coupling) deferred. The probe is gated on
`print_level >= 6` and `RIPOPT_IR_PROBE=1` — silent in normal runs.

**Refs**: `src/ipm.rs:5703-5775` (iter-0 probe), `src/kkt_aug.rs:1086-1106`
(IR-probe, env-var gated). Closes Task #45.

## A8.22 — scaling-propagation hypothesis ruled out (2026-04-30)

**Trigger**: A8.21 conclusion narrowed the iter-0 dx[1753] gap to "data
assembly". A working hypothesis was that ripopt computes `g_scaling`
correctly but the IPM core operates on raw values (i.e. the
`ScaledProblem` wrapper bypassed somewhere), which would systematically
inflate dx/dy by `1/g_scaling[row]` for affected rows.

**Method**: extended the iter-0 probe to dump per-constraint-row data
(`g`, `g_l`, `g_u`, slack `s`, `dy`, plus full `J[r,*]`) for every row
that touches `x1802`. Re-read the wrapper chain and `SolverState::new`
in full.

**Findings (row 1962 = AMPL e2372, `g_scaling[1962] = 0.0909`)**:

```
g=4.5453545455545456e4  g_l=-inf  g_u=1.0000000000000000e-8
s=-9.9999900000000003e-3  dy=2.1689770521236824e1
J[1962,*]: (1723, 9.090909e-2) (1753, -1.000000e2)
```

Re-interpreted with `ScaledProblem` (`src/ipm.rs:180-248`) applied:

- `g_u = 1e-8`: raw .nl r-segment value is `0` (line "1 0"). Scaled:
  `0 × 0.0909 = 0`. `apply_bound_relax_factor` then pads to `1e-8`.
  **Consistent with scaling applied.**
- `g[1962] = 4.5453e4`: scaled value, raw inner = `5e5` (back-solved
  via `g_scaling[1962] = 0.0909`). Matches Ipopt's `print_level=12`
  internal print (Ipopt prints scaled values).
- `J[1962, 1723] = 9.09e-2`: scaled, raw = `1.0` (clean coefficient).
  `J[1962, 1753] = -100`: scaled, raw = `-1100` (constraint coefficient
  of the relevant magnitude). **Consistent with scaling applied.**

**Code path verified**:

- `solve_ipm` (`src/ipm.rs:5273`) wraps `problem` →
  `ScaledProblem` (5295) → `FiniteCheckedProblem` (5307) → shadow
  `problem` (5308).
- `SolverState::new(problem, options)` (5310) calls
  `problem.constraint_bounds(...)` (1078) and `problem.constraints(...)`
  (1228) through this wrapper chain. State is populated with **scaled**
  values throughout.
- `ScaledProblem::constraint_bounds` (190-200) multiplies `g_l`, `g_u`
  by `g_scaling[i]` when finite. `ScaledProblem::constraints` (216-222)
  multiplies `g[i]` by `g_scaling[i]`. `ScaledProblem::jacobian_values`
  (226-232) multiplies `vals[idx]` by `g_scaling[row]`.

**Conclusion**: scaling propagation is correct. The 6.16% iter-0 dx gap
at slot 1753 is **not** caused by ripopt operating on the unscaled
problem.

Combined with A8.21's elimination of feral, perturbation handler, and
initial point, every locally-verifiable hypothesis at iter 0 is now
ruled out. The remaining candidates require global-system comparison
against Ipopt:

- **Hessian assembly globally** — H[1753, *] is empty at iter 0 (x1802
  is linear in objective, and y_0=0 zeros out constraint Hessians), so
  this can't explain dx[1753] *locally*, but Hessian differences at
  *other* rows propagate through the augmented solve to dx[1753].
- **Jacobian column 1753 raw values** — verified clean above for J row
  1962; J row 1904 entries are `1, -1, 1`. Cross-check against Ipopt's
  print_level=12 Jacobian dump for these rows would be definitive.
- **Σ_s coupling for slack rows touching variable 1753** — ripopt's
  d-block formulation may differ structurally from Ipopt at the slack
  initialization (`s = -9.99e-3` for row 1962, on the upper-side of an
  `x ≤ 0` constraint relaxed to `x ≤ 1e-8`).
- **Floating-point accumulation across the global augmented solve** —
  the system has Σ_x range ~650:1 within row 1753 alone (100 vs 0.154
  from x near vs far from bound), and the IR final_ratio is 1e-15. A
  6% gap from FP differences alone would require systematic ordering
  differences in the elimination tree — feral vs MUMPS is a candidate
  but A8.16 already showed feral and rmumps produce identical results.

**Status**: arki0003 deprioritized per A8.20. The remaining hypothesis
list cannot be discriminated without a global Jacobian/Hessian dump from
Ipopt's print_level=12 trace at iter 0, which is a substantial parsing
exercise. Recommend pivoting to a smaller-scale problem with cleaner
divergence signature before further deep-dive on arki0003.

**Refs**: `src/ipm.rs:5743-5832` (extended conrow probe);
`src/ipm.rs:180-248` (ScaledProblem wrapper).

## A8.23 — current-state re-measurement after P1–P10 alignment pass (2026-05-02)

**Trigger**: pick up arki0003 after the v0.8 alignment pass (P1–P10
defaults audit). User's working hypothesis: ripopt is now equivalent to
Ipopt on arki0003.

**Method**: re-ran arki0003 with `RIPOPT_TRACE_PERTURB=1`,
`max_iter=700`, `print_level=5` against the post-P1–P10 binary
(P3 `watchdog_trial_iter_max` 5→3, P4 `resto_proximity_weight=1.0`
option, P6 `y_d := v_U − v_L` shortcut removed). Pre/post traces saved
at `/tmp/arki_pert.log` and `/tmp/arki_pert_v2.log`.

**Finding 1 — P1–P10 changes are bit-identical on arki0003 through
iter 400**:

| iter | obj         | inf_pr  | inf_du  | compl  | mu      |
|------|-------------|---------|---------|--------|---------|
| 0    | -0.0000e0   | 1.16e8  | 1.00e0  | 1.00e7 | 1.00e-1 |
| 100  | 1.1389e7    | 1.72e6  | 2.74e6  | 5.94e5 | 1.00e-1 |
| 200  | 1.2175e7    | 8.04e3  | 2.01e6  | 3.84e5 | 2.83e-3 |
| 300  | 1.0630e7    | 2.00e0  | 7.32e4  | 5.29e1 | 1.50e-4 |
| 400  | 9.4114e4    | 5.38e-3 | 5.95e3  | 7.94e1 | 1.84e-6 |

The pre-pass and post-pass logs match the displayed columns exactly at
iters 0/10/25/50/100/110/150/200/300/400 — P3/P4/P6 do not move the
arki0003 trajectory because (a) watchdog is not triggered on this
problem (no consecutive shortened-step run), (b) restoration is not
entered (kappa_resto progress holds), and (c) `least_squares_mult_init`
is on by default so the deleted `y_d := v_U − v_L` branch was never
taken.

**Finding 2 — perturbation handler is conclusively NOT the bottleneck**:
across 928 augmented-system factor calls in the trace,
- 543 used `dx=0, dc=0` (no perturbation),
- 385 used a nonzero `dx` perturbation (`3e-5 → 0.18`, the warm-shrink
  ladder operating correctly),
- `dc` is *never* nonzero on this problem.

Inertia is recovered on every observed call (target inertia
`(3563+, 2138-, 0)` reached). This rules out the A8 `_last`-reset bug
hypothesis stated in the followup doc's introduction: ripopt's
`PDPerturbationHandler` is in fact correctly aligned with Ipopt
(verified by direct read of `IpPDPerturbationHandler.cpp` — both
implementations carry `_last` forward when `_curr=0` under
`reset_last_=false`, and `reset_last_` is never flipped to `true` in
the file).

**Finding 3 — actual stagnation signature**: the run progresses well
through iter 400, then asymptotically approaches but does not satisfy
the dual-feasibility tolerance:

- inf_pr: 1.16e8 → 5.38e-3 (10 orders of magnitude reduction) ✅
- compl: 1.00e7 → 7.94e1 (5 orders of magnitude reduction)
- inf_du: 1.00e0 → 5.95e3 (**increased** then frozen at ~6e3) ❌

mu hit its lower bound (`mu_min = 1.84e-6`, set by `mu_target` and
`compl_inf_tol`) at iter ≈280 and stayed there for the rest of the run.
α_pr in the 5e-8 → 1e-3 range across iter 300+ — line search consumes
nearly the entire step but the dual residual never reduces.

**Diagnosis**: the dual-feasibility residual is the bottleneck, not the
perturbation handler, not feasibility, not complementarity. This is
the cumulative-drift signature predicted by A8.20: the per-iteration
direction differs from Ipopt's by a small amount, and 400 iterations
of small differences leave inf_du stuck where Ipopt's would have
converged.

**Concrete next-step recommendation**: per A8.20 / A8.22, arki0003 is
no longer a single-bug diagnostic — it requires an Ipopt
`print_level=12` Jacobian/Hessian/dz dump at iter 0 to bisect against
ripopt's, which is a substantial parsing exercise. The four open
candidates from A8.22 (Hessian assembly, Jacobian column 1753, Σ_s
coupling, FP-accumulation in elimination tree) cannot be ranked
without that dump.

**Two routes for the next session**:

1. **Pivot to a smaller divergence target** (A8.20 / A8.22
   recommendation): pick a CUTEst problem where ripopt and Ipopt
   agree to ≪0.1% on objective until a specific iter, then diverge
   sharply. The step-direction-difference root cause should manifest
   identically but on a problem where the divergence is localizable
   to one alignable heuristic. Suggested filtering pass: across the
   v0.8 baseline `results_v0.8.0-dev_baseline.json` regressions, find
   problems where ripopt iter-by-iter `α_pr` differs from Ipopt's by a
   constant factor or where Ipopt converges in `<50` iterations and
   ripopt hits `MaxIterations`.

2. **Parse the Ipopt print_level=12 dump** for arki0003 iter 0
   (Jacobian, Hessian, dz post-solve) and diff column-by-column
   against ripopt's iter-0 probe (`src/ipm.rs:5703-5775`,
   `src/kkt_aug.rs:1086-1106`, both already gated on
   `RIPOPT_IR_PROBE=1`). The 6.16% gap at slot `dx[1753]`
   (A8.21 finding) is the localizing signal — whichever block
   (H | J | Σ_s | dz) shows a comparable gap at the matching index is
   the assembly-side culprit.

Route 1 has higher expected information-per-hour (clean signature,
small problem, fast iteration). Route 2 is the only way to *close
out* arki0003 specifically.

**Status**: arki0003 remains deprioritized. P1–P10 alignment pass had
zero effect on its trajectory, confirming the gap is not in any of
the audited defaults. Issuing the recommendation to pivot to a smaller
divergence target before resuming arki0003 deep-dive.

**Refs**: `/tmp/arki_pert_v2.log` (post-P1-P10 trace, 1209 lines, iters
0–404, timed out at 3 min); `src/options.rs:692-694` (P3 default);
`src/options.rs:331-334, 705` + `src/ipm.rs:8171-8177` (P4 plumbing);
`src/ipm.rs:6074-6091` (P6 shortcut removal).

## A8.24 — Iter-0 dump differ localizes divergence to .nl evaluator (2026-05-03)

**Trigger**: Route 2 from A8.23 — build cross-binary iter-0 dump
support and diff ripopt vs Ipopt block-by-block at the same `x_init`,
`mu`, and `bound_mult_init`.

**Method**: three-stage pipeline.

1. `src/iter0_dump.rs` — flat JSON schema (lengths declared in the
   header struct) with `Vec<Option<f64>>` for finite/unbounded sides
   so JSON null round-trips (avoids serde_json's NaN→null lossy
   default).
2. `RIPOPT_IR_DUMP=<path>` env-gated dump in `src/ipm.rs`
   (post-KKT-solve, post-step-recovery, pre-line-search at iter 0).
   Materializes x, bounds, multipliers (full-n indexing), Σ_x/Σ_s,
   ∇f, g, sparse J (rebuilt from `rebuild_combined_jac`), sparse H,
   c_scaling (eq+ineq merged), all step deltas, `δ_w/δ_c` used at
   iter 0, and `α_pr/α_du`.
3. `examples/arki_ipopt_log_to_json.rs` — parses Ipopt's
   `print_level=12` text log into the same schema, using `.col`/`.row`
   sidecar files for `{xN}/{eN}` label → ripopt-slot mapping. Handles
   Ipopt's "homogeneous vector, all elements have value V" short form
   (used at iter 0 for z_L = z_U = v_L = v_U = 1.0). Picks
   `jac_d_unscaled_matrix` since Ipopt wraps `jac_d` in a
   `ScaledMatrix`. Picks the first `CompoundVector "delta"` for the
   iter-0 step (post-perturbation solve).
4. `examples/arki_diff.rs` — block ‖·‖_∞ table, top-K element-wise
   mismatches per primal/constraint block, sparse-matrix block diff
   (structural overlap + max|diff| at common entries), and a
   probe-var focus printout.

**Pipeline command**:
```
RIPOPT_IR_DUMP=/tmp/ripopt_iter0_arki0003.json target/release/ripopt \
  benchmarks/mittelmann/nl/arki0003.nl max_iter=1 print_level=0
target/release/examples/arki_ipopt_log_to_json \
  /tmp/ipopt_arki0003_full.log \
  benchmarks/mittelmann/nl/arki0003 \
  /tmp/ipopt_iter0_arki0003.json
target/release/examples/arki_diff \
  /tmp/ripopt_iter0_arki0003.json \
  /tmp/ipopt_iter0_arki0003.json
```

**Measured findings (rel = |a-b|/max(|a|,|b|))**:

Block ‖·‖_∞ table — what matches and what doesn't:

| Block      | ripopt   | ipopt    | rel       | Verdict |
|------------|----------|----------|-----------|---------|
| x          | 1.000e7  | 1.000e7  | 0         | ✅ exact |
| x_l        | 1.430e4  | 1.430e4  | 7.0e-9    | ✅ matches modulo `bound_relax_factor=1e-8` |
| x_u        | 1.000e5  | 1.000e5  | 1.0e-9    | ✅ same |
| z_l, z_u   | 1.0      | 1.0      | 0         | ✅ exact (homogeneous init) |
| v_l, v_u   | 1.0      | 1.0      | 0         | ✅ exact |
| ∇f         | 1.0      | 1.0      | 0         | ✅ exact |
| g          | 1.160e8  | 1.160e8  | 0         | (top-1 OK; **per-element diffs below**) |
| **jac_vals** | **1.000e2** | **2.000e3** | **9.5e-1** | ❌ **20× off in many entries** |
| hess_vals  | 8.71e-2  | 8.92e-2  | 2.4e-2    | ⚠ small max but **534 entries only_in_ripopt** |
| Σ_x, Σ_s   | 1.0e2    | 1.0e2    | 1.0e-6    | ✅ same (1e-6 = bound_relax FP noise) |
| c_scaling  | 1.0      | 1.0      | 0         | ✅ no scaling |
| dx         | 4.999e7  | 4.999e7  | 2.2e-5    | (close at top, **8% gap at slot 30**) |
| y_c        | 9.51e1   | 1.80e2   | 4.7e-1    | ❌ **initial y_c estimate diverges** |
| y_d        | 1.22     | 1.80     | 3.2e-1    | ❌ **initial y_d estimate diverges** |
| dy         | 4.59e3   | 4.39e3   | 4.3e-2    | (downstream of J + y_c) |

**Smoking-gun #1 — Jacobian VALUES disagree at iter 0 with same x**:
`jac_vals` rel=0.95. Sparsity matches exactly (10206 common, 0 only in
either). Top-10 mismatches all show `ripopt=−1.000e2, ipopt=−2.000e3`
(20× factor) at column 483 (`x1720`), 502 (`x1740`), 479, 478, 497,
493, 486, 487, 476, 500. At the probe slot, **J[1962,1753]
(`e2372/x1802`): ripopt=−100, ipopt=−1100** (11× factor).

**Smoking-gun #2 — Constraint VALUES g(x_init) disagree at same x**:

| index | AMPL row | ripopt   | ipopt    | rel  |
|-------|----------|----------|----------|------|
| 1875  | e2223    | -260.0   | -1601.0  | 0.84 |
| 1933  | e2283    | +13.0    | 0.0      | 1.00 |
| 1904  | e2253    | +13.0    | +1.3     | 0.90 |

These three rows are constants in the linearized form (all `dx`
displacements zero into their J-row), so the discrepancy is in the
constraint-value evaluation itself, not in any derivative.

**Smoking-gun #3 — Hessian sparsity divergence**: `common=1503,
only_in_ripopt=534, only_in_ipopt=0`. Max |diff| at common entries is
4.6e-2. Ripopt is reporting 534 zero or near-zero Hessian entries that
Ipopt's ASL evaluator omits. Some are real value disagreements (e.g.,
H[893,417] = +2.08e-2 vs −2.56e-2), not just sparsity-only zeros.

**Verified vs inferred**:

- **Measured**: at iter 0, with identical `x = x_init` and identical
  bound multipliers (z_L = z_U = v_L = v_U = 1.0), ripopt and Ipopt
  produce different J values, different g values, and different H
  sparsity. ‖·‖_∞ tables and per-element diffs above. Files at
  `/tmp/ripopt_iter0_arki0003.json`, `/tmp/ipopt_iter0_arki0003.json`,
  `/tmp/arki_diff_report.txt`.
- **Inferred (not yet verified)**: the J/g/H disagreement originates
  in `src/nl/` (the pure-Rust .nl evaluator) — likely a specific
  opcode (negation? `n2` vs `2`?), a constant-segment handling bug
  (linear part of constraint?), or a `b`/`r`/`d` block being
  consumed differently than ASL does. The factor-20× pattern at many
  jac entries in column range 478–502 (`x1720..x1740`, all in the
  same column band) suggests one shared multiplicative term or one
  variable-segment is mis-evaluated.
- **Inferred**: `y_c[389]` initial estimate divergence
  (`95.13` vs `179.93`) and `y_d` divergence are downstream of the
  J/g disagreement, since `least_squares_mult_init` solves
  `J Jᵀ y = -∇f - z_L + z_U` — wrong J ⇒ wrong y.

**Localization conclusion**: this is **not** a KKT, IPM,
perturbation-handler, line-search, scaling, restoration, or
filter bug. The four open candidates from A8.22 (H assembly,
Jacobian col 1753, Σ_s coupling, FP-accumulation in elimination
tree) are all **eliminated** — the divergence is upstream of the
KKT system. Σ_x and Σ_s match, the linear solver gets the same
inputs except for J/g/H, and the dx top-1 ‖·‖_∞ ratio (2.2e-5)
is set entirely by the ‖dx‖_∞ entry on the bound-violating
slot 1871 (which Ipopt has set with a tighter bound).

The next-stage investigation has a clear target list:

1. **Constraint value mismatch** (largest absolute signal,
   simplest to localize):
   - row `e2223` (slot 1875): -260 vs -1601 — find this row in
     `arki0003.nl`, run ripopt's evaluator on it, compare against
     `nl_grep` or a manual unwind.
   - rows `e2283` (1933) and `e2253` (1904) — both show
     small-integer outputs (+13) that suggest constant terms are
     mishandled.
2. **Jacobian factor-20× pattern**: cluster at `x1720..x1740`
   range (cols 478–502) — find which constraint rows reference
   these vars with a coefficient that ripopt halves or ipopt
   doubles.
3. **Hessian only-in-ripopt entries**: 534 phantom non-zeros —
   likely a structural-zero detection issue in the autodiff
   pass (e.g., emitting a Hessian entry for `0 * x[i] * x[j]`).

**Refs**: `src/iter0_dump.rs` (schema, 150 lines);
`src/ipm.rs:7014-7170` (ripopt-side dump emitter, env-gated);
`examples/arki_ipopt_log_to_json.rs` (Ipopt log parser, ~1100 lines);
`examples/arki_diff.rs` (block-wise differ, ~700 lines);
`/tmp/arki_diff_report.txt` (full diff output).

**Status**: arki0003 root cause is now localized to the **.nl
evaluator** (`src/nl/`). The KKT-system / IPM hypothesis from A8.20–
A8.23 is **falsified** at iter 0. Recommendation: investigate the .nl
evaluator on the three constant-mismatch rows (e2223, e2283, e2253)
first — their small-integer ripopt outputs (-260, +13, +13) vs
Ipopt's varied outputs (-1601, 0, +1.3) point at a specific
op-handling bug that should be reproducible on a 5-line unit test.

### A8.24 — addendum: smoking-gun #2 retracted (2026-05-03, same day)

**Trigger**: before writing a unit test, inspected the .nl parser
directly for the three constant-mismatch rows.

**Method**: pulled the `r` segment (constraint bounds) for rows
1875/1904/1933 and the `x` segment (initial primal) for the
referenced variables out of `benchmarks/mittelmann/nl/arki0003.nl`,
then hand-evaluated.

```
$ grep -n '^r$' arki0003.nl    → line 3636
$ awk '/^r$/{f=NR;next} f && NR<=f+2138 {print NR-f":"$0}' arki0003.nl | awk -F: 'NR==1876||NR==1905||NR==1934'
1876:4 1341
1905:4 11.700000000000001
1934:4 13
```

Type-4 equalities with **non-zero RHS** (1341, 11.7, 13).

```
$ awk '/^x1751$/{n=1;next} n && ($1==894||$1==895||$1==905||$1==1813||$1==389||$1==1694) {print}' arki0003.nl
389 13
894 130
895 130
905 130
1694 13
1813 1300
```

Variables not listed in the `x` segment default to 0.

**Hand-evaluation**:

| row | linear formula | x_init substitution | ripopt g | ipopt g | ipopt = ripopt − r |
|-----|---------------|--------------------|----------|---------|------|
| 1875 / e2223 | −Σ_{894..905} x + x[1813]   | −12·130 + 1300        | −260 ✓ | −1601 | −260 − 1341 = **−1601** ✓ |
| 1904 / e2253 | x[389] − x[1254] + x[1753]  | 13 − 0 + 0.01         | 13 ✓   | 1.3   | 13 − 11.7 = **1.3** ✓     |
| 1933 / e2283 | −x[1284] + x[1694] + x[1783] | 0 + 13 + 0            | 13 ✓   | 0     | 13 − 13 = **0** ✓         |

**Diagnosis**: ripopt's `.nl` evaluator is **correct** on these rows.
The "mismatch" is a **representation convention difference** — Ipopt
internally normalizes type-4 equalities to `g_internal(x) = f(x) − b`
so the equation reads `g_internal = 0`. Ripopt keeps `g(x) = f(x)`
and stores `g_l = g_u = b` as bounds. The Ipopt-log parser
(`examples/arki_ipopt_log_to_json.rs`) dumps Ipopt's already-shifted
`curr_c`/`curr_d`, ripopt's `iter0_dump` emits the unshifted form,
and the differ compares apples to oranges on equality-with-RHS rows.

**Verified vs inferred (revised)**:

- **Measured**: g(x_init) for rows 1875/1904/1933 in ripopt matches
  the .nl-file definition (J coefficients × x_init). Ipopt's reported
  values match `(ripopt's g) − (r-segment RHS)`. Both are correct in
  their own conventions.
- **Falsified**: smoking-gun #2 ("constraint values disagree at same
  x") from A8.24's main entry. No `.nl` evaluator bug on these rows.

**Status of the other A8.24 findings (revised)**:

1. **Jacobian factor-20× cluster (rows 1068–1094, cols 478–502)** —
   still open, but suspect a similar runtime-scaling artifact rather
   than parsing. Both `state.jac_c_vals/jac_d_vals` (ripopt) and
   `jac_d_unscaled_matrix` (Ipopt) may carry NLP gradient-scaling
   that the dumps don't strip. Next-step probe: read
   `src/preprocessing/scaling*.rs` or wherever ripopt computes
   `c_scaling/d_scaling` and check whether `state.jac_*_vals` is
   pre- or post-scaling. Cross-check by setting
   `nlp_scaling_method=none` on both sides and re-diffing.
2. **Hessian 534 only_in_ripopt entries** — still genuine. There's
   no equality-shift counterpart that produces phantom Hessian
   entries (the linearized form is `0` for both sides on linear
   constraints, and the Hessian-of-Lagrangian is just a sum of
   constraint Hessians weighted by y). Likely a real autodiff
   structural-zero detection issue. Investigate
   `src/nl/autodiff.rs` — specifically the `hess_*` builder for
   structural symmetry.
3. **y_c initial 47% divergence** — should be downstream of the
   shift convention only if the least-squares-mult-init formula
   uses constraint values (it doesn't — it uses gradients). So this
   is *not* explained by the convention difference and remains
   genuinely open. Reread `src/ipm.rs` `least_squares_mult_init`
   path against IpDefaultIterateInitializer.cpp.

**Lessons (write to memory)**:

- The differ at `examples/arki_diff.rs` was naive about
  representation conventions. Before claiming a value mismatch,
  check whether the two sides use the same normalization for
  equality constraints (Ipopt: shift to g=0; ripopt: keep g=f(x)
  and use bounds), bound multipliers (sign conventions), and slack
  variables (Ipopt: c(x) − s = 0 with s ≥ 0; ripopt: same
  internally but the dump may differ).
- "Read the function before changing its callsite" extends to
  "read the file format before claiming a value mismatch" — a
  single grep of the `r` segment would have ruled out the bug
  before writing a unit test.

**Status**: smoking-gun #2 retracted. Open candidates reduced to
(a) Jacobian-scaling convention check, (b) Hessian structural-zero
detection, (c) `least_squares_mult_init` formulation. Recommendation
unchanged — investigate the .nl evaluator's autodiff Hessian first
since that's the only non-convention finding left.

**Refs**: `benchmarks/mittelmann/nl/arki0003.nl:3636-...` (r segment);
`benchmarks/mittelmann/nl/arki0003.nl:1884-...` (x segment);
`src/nl/parser.rs:266-313` (r-segment parser, type 4 handler at
:299-303).

## A8.25 — Differ convention bridge + factor-20× retraction (2026-05-03)

**Trigger**: user pushback on A8.24 — "even though it is technically
correct it does not mean the right values are used in ripopt right? I
would expect bit-wise equivalence, not within some operations
equivalent". Investigated whether the Jacobian factor-20× cluster
from A8.24 (rows 1068–1094, cols 478–502) is a real value mismatch or
another dump-layer convention difference.

**Method**: probe ripopt's `state.c_scaling[row]` for the rows in
question and the Ipopt-side `c_scaling[row]` (which the log parser
fills from `d scaling vector`):

```
$ python3 -c "..." (read both JSON dumps)
[1068,418]  ripopt c_scal=5.0000e-02  ipopt c_scal=5.0000e-02
            jac r=-9.9000e-02  i=-1.9800e+00  ratio i/r=20.0000
[1075,425]  ripopt c_scal=5.0000e-02  ipopt c_scal=5.0000e-02
            jac r=-9.9000e-02  i=-1.9800e+00  ratio i/r=20.0000
... (10 of 10 rows show exact 20× = 1/c_scaling)
```

**Diagnosis (root cause)**: ripopt's iter-0 dump emits
`state.jac_c_vals/jac_d_vals`, which `ScaledProblem::jacobian_values`
(`src/ipm.rs:271-277`) populates as **post-NLP-scaling**:

```rust
fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
    if !self.inner.jacobian_values(x, _new_x, vals) { return false; }
    for (idx, &row) in self.jac_rows.iter().enumerate() {
        vals[idx] *= self.g_scaling[row];   // <-- scaling baked in
    }
    true
}
```

Ipopt's log dumps `jac_d_unscaled_matrix`, which is the underlying
GenTMatrix **before** NLP scaling is applied (the NLP scaling is
applied dynamically by `IpScaledMatrix` on each operation). So the
two dumps are at **different abstraction layers** of the IPM state.

The factor 20× = 1 / c_scaling[row] is exactly what you'd predict.

**Bridge fix in `examples/arki_diff.rs`**: optional `--nl=<path>`
flag. The differ now:

1. Parses the .nl `r` segment for type-4 equality rows.
2. Subtracts each rhs from ripopt-side `g[row]` so both dumps are in
   shifted form (matching Ipopt's `c_x = f(x) − b` convention,
   internally consistent with `state.c_x` per docstring at
   `src/ipm.rs:1862`).
3. Multiplies Ipopt-side `jac_vals[k]` by `i.c_scaling[row]` so both
   dumps are in post-NLP-scaling form (matching ripopt's
   `state.jac_*_vals` storage convention).

**Re-run results (post-bridge)**:

| Block       | Pre-bridge rel | Post-bridge rel | Verdict |
|-------------|---------------:|-----------------:|---------|
| g           | 0 (top match coincidental) | **1.9e-16** | ✅ bit-wise equal modulo FP |
| jac_vals    | 9.5e-1 (95%)   | **1.4e-16**  | ✅ bit-wise equal modulo FP |
| J[1962,1753] (probe) | -100 vs -1100 | -100 vs -100 (diff 1.4e-14) | ✅ |
| x, grad_f, x_l/u, z_l/u, v_l/u, c_scaling, s | match | match | ✅ |
| Σ_x, Σ_s    | 1e-6           | 1e-6         | ✅ bound_relax FP only |

**Genuine bit-wise misalignments remaining at iter 0**:

| Block      | rel    | Top entry detail                              |
|------------|-------:|-----------------------------------------------|
| `y_c[389]` | 0.4713 | ripopt=+95.131 vs ipopt=+179.929 (ratio 1.89) |
| `y_c[11]`  | 0.8889 | ripopt=+0.442  vs ipopt=+3.980  (ratio 9.0)   |
| `y_d[1514]` | 1.81  | ripopt=−0.570  vs ipopt=+0.460                |
| `y_d[1662]` | 1.00  | ripopt=0       vs ipopt=+0.896                |
| hess_vals (max diff) | 0.024 | + **534 entries only_in_ripopt** |
| `dx[30..37]` | 6–8% | downstream of y_c (J^T y enters dual residual)|
| `ds[359]`  | 2.1%   | downstream of dy                               |
| `dz_l[1254]` | 0.029 | downstream of dx                              |

**Status of A8.24's three open candidates (final)**:

1. **Jacobian factor-20× cluster** — **RESOLVED**: dump-layer
   convention difference (NLP-scaling), not a bit-wise drift. After
   `i.c_scaling[row]` correction the Jacobian is bit-wise equal at
   iter 0 (max |diff| 1.4e-14, rel 1.4e-16).
2. **g[1875] et al. equality mismatches** — **RESOLVED** (A8.24
   addendum): equality-RHS shift convention. After `r`-segment
   subtraction the constraint vector is bit-wise equal (max |diff|
   2.3e-13, rel 1.9e-16).
3. **Hessian 534 only_in_ripopt + 2.4% rel** — **STILL OPEN**. No
   convention-bridge has been applied for the Hessian; probably needs
   the `obj_factor*obj_scaling + lambda*g_scaling` decomposition
   handled. But 534 phantom non-zeros is more than a scaling
   artifact — it's likely a real autodiff structural-zero issue.
4. **y_c / y_d initial multiplier estimates** — **STILL OPEN**. The
   `least_squares_mult_init` LSQ (`J Jᵀ y = -∇f - z_L + z_U`) uses
   only quantities that we've now confirmed are bit-wise equal
   between ripopt and Ipopt. So either (a) the LSQ formulation
   differs (maybe ripopt regularizes the LSQ system differently
   from Ipopt), or (b) ripopt and Ipopt use different LSQ-norm
   conventions. y_c[389] ratio 1.89 doesn't match any single
   constraint or variable scaling, suggesting a formulation-level
   discrepancy rather than a scaling one.

**Lessons (updated)**:

- "Differ should be format-aware" — added explicit convention
  bridges. Future iter-N dumps must apply the same bridges before
  drawing any "ripopt and Ipopt disagree" conclusions.
- The user's bit-wise standard is the right one; each "explained
  away" mismatch must be backed by a verifiable transformation that
  collapses the diff to FP noise. After applying the transformation,
  the diff *did* collapse — confirming the explanation. If applying
  the transformation had *not* collapsed the diff, the explanation
  would have been wrong.

**Concrete next-step recommendations**:

1. Investigate `least_squares_mult_init` in `src/ipm.rs` against
   `IpDefaultIterateInitializer.cpp:259-298` and `IpEqMultCalculator`
   (Ipopt's LSQ helper). The y_c[389] = 1.89× factor is the
   localizing signal.
2. Build a Hessian convention bridge (multiply Ipopt's W by
   `obj_factor` and per-row `lambda` correction) in
   `examples/arki_diff.rs` and re-run. If 534 phantom entries
   persist after the bridge, that's the real bug — likely in
   `src/nl/autodiff.rs` Hessian sparsity emission.

**Refs**: `examples/arki_diff.rs:117-180` (bridge fix);
`src/ipm.rs:271-277` (ScaledProblem::jacobian_values);
`src/ipm.rs:1860-1869` (set_g_combined docstring confirming
ripopt's internal c_x is shifted, matching Ipopt's curr_c);
`/tmp/arki_diff_v2.txt` (post-bridge diff output).

## 2026-05-04 forward-walk: iter 110 LS no-movement accept

**Setup**: forward-walk row-by-row comparison of ripopt vs Ipopt
(`/tmp/ripopt_iters.txt`, `/tmp/ipopt_iters.txt`). Iters 0-110 are
bit-identical to printed precision (obj, inf_pr, inf_du, lg(mu)).

**Divergence at iter 110→111**:

| iter | ripopt obj/pr/du | ipopt obj/pr/du | tag |
|------|------------------|------------------|-----|
| 110  | 1.195e7 / 3.12e5 / 3.57e6 | same | h 1 |
| 111  | (ripopt) 1.195e7 / 3.12e5 / 3.57e6 | (ipopt 111r) 1.195e7 / 3.12e5 / 1.00e3 | R 4 |
| 114  | (ripopt) 1.195e7 / 3.12e5 / 3.57e6 | (ipopt) 5.0e6 / 2.87e3 / 1.02 | h 1 |

Ipopt enters restoration at iter 110→111 (`R` tag, `ls=4`), runs 3
restoration iters (111r-113r), and emerges at iter 114 with feasibility
improved 100× and objective dropped 60%. **Ripopt continues normal
LS at iter 110→111** and never recovers — it accumulates ~390
no-progress iterates before MaxIter at 500.

### Root cause: variable 533 barrier trap

At iter 110, variable index 533 (`x = x_U = 1341.0`, slack
`x_U - x = 8.31e-5`) dominates the gradient/step calculation:

```
[gBD] iter=110 ∇f·dx=+6.65e4 (xb)·dx=-2.69e6 (sb)·ds=+1.08e6  TOTAL=-1.54e6
[gBD-top] iter=110 top_i=533 side=U term=-1.15e6 slack=8.31e-5 dx=-953.1
[gBD-zmu] iter=110 top_i=533 z_U=776 dz_U=-8.90e9
```

Single-variable contribution: `+μ/slack · dx = +1203 × (-953) ≈ -1.15e6`,
which alone accounts for 75% of `grad_phi_step = -1.54e6`. The
huge `|gBD|` then collapses Ipopt's `alpha_min` formula:

```
alpha_min = α_min_frac × γ_φ × theta / |gBD|
          = 0.05 × 1e-8 × 3.92e5 / 1.54e6
          = 1.27e-10                          (ripopt observed value)
```

With `alpha_min = 1.27e-10` and `alpha_p_max = 2.519e-6` (FTB-clipped
by `dz_U[533] = -8.9e9`), the LS would tolerate ~14 backtracks before
exiting. But:

```
[probe] iter=110 ls=0 alpha=2.519e-6 gBD=-1.544e6
        theta_curr=3.921e5 theta_tr=3.921e5  (Δθ=-0.99)
        phi_curr=1.195e7   phi_tr=1.195e7    (Δφ=-18.3)
        is_ft=true armijo=true suf_phi=true in_filter=true → ACCEPT
```

The first trial passes f-type Armijo trivially: `phi_trial - phi_current
= -18.3 <= η_φ·α·gBD = -3.89e-4`. Relative movement is `|Δθ|/θ = 2.5e-6`
— effectively zero. **The iterate barely moves but the LS reports
"step accepted".**

Ipopt at the same juncture rejects 4 backtrack trials (alphas
~5e-6 → 3.14e-7) and enters restoration. The mechanism is not
visible in the printed log but is one of:
  (a) Ipopt's `alpha_p_max` at iter 110 is ~5e-6 (not 2.5e-6) AND
      `alpha_min` is much larger (~6e-7), so `alpha_p_max/alpha_min`
      is small and 4 backtracks suffice to exit.
  (b) Ipopt's filter contains additional entries that ripopt is
      missing, dominating the iter 110→111 trial.
  (c) Ipopt's `gBD` magnitude differs because dx[533] is smaller
      (small precision differences in KKT solve at iter 110 that
      forward-walk to printed precision didn't catch).

### Why kappa_sigma doesn't help

Default `kappa_sigma=1e10`. With μ=0.1, the per-bound clamp is
`z·s ∈ [1e-11, 1e9]`. At iter 110, `z_U[533]·s_U[533] = 776 × 8.3e-5
= 0.0644` — well inside the band. Kappa_sigma is not designed to
catch this case.

### Why mu-update gate never fires

Companion observation (commit `e8dff73`): `RIPOPT_TRACE_MU` shows
`E_mu` never approaches `barrier_tol_factor × mu = 1.0`. Best
observed `E_mu = 3.92e3` at iter 357 (gap 3920×), dominated by the
complementarity component (`co=1.11e5` terminal).

The `co` outliers are exactly the bound slacks for variables
`xL[1663-1697]` and `xL[1688-1700]` — same family as variable 533,
all sitting at scale 1e6-1e7 with `z`-multipliers near 1.

### Action items (forward-walk)

1. **Filter divergence diagnostic**: instrument ripopt to dump filter
   entry list at iters 100-110, run Ipopt with high print_level to
   extract its filter entries, and diff. If hypothesis (b) is correct,
   ripopt is failing to augment the filter at one or more h-type
   accepts in the iter 100-110 range.

2. **Step-size precision diagnostic**: Add a finer probe that compares
   `dx`, `dz_U[533]`, and `alpha_p_max` at iter 110 between ripopt and
   Ipopt with full FP precision. Requires a custom Ipopt trace patch.

3. **No-movement restoration trigger** (ripopt-specific guardrail):
   if a step is accepted with relative iterate movement
   `||x_trial - x|| / max(||x||, 1) < 1e-6` AND constraint violation
   `theta > 1e-2`, force restoration entry on the next iter. This
   mirrors Ipopt's `DetectTinyStep` (BacktrackingLineSearch.cpp:1219)
   but with the `cviol < 1e-4` gate inverted — Ipopt's gate is for
   "stuck at infeasible local min of theta", ours would be for
   "stuck in barrier trap at high theta".

**Refs**: `/tmp/arki_lsdec.out`, `/tmp/arki_gbd.out`,
`/tmp/arki_lsprobe.out`, `/tmp/arki_probe.out`,
`/tmp/ipopt_arki_full.out`; `src/filter.rs:422` (compute_alpha_min);
`src/ipm.rs:2076` (barrier_directional_derivative);
`src/ipm.rs:2814` (LS alpha_min check);
`ref/Ipopt/src/Algorithm/IpBacktrackingLineSearch.cpp:1219`
(DetectTinyStep).

## 2026-05-04 follow-up: filter contents are NOT the divergence driver

**Setup**: ran Ipopt with `print_level=12` on arki0003 (max_iter=115),
extracted filter contents at iter 110→111 LS time. Compared against
`RIPOPT_FILTER_DUMP=109,111` output from ripopt.

**Filter sizes essentially identical**:
- Ripopt iter 110: 76 entries, last entry `(theta=3.9221e5, phi=1.1945e7)`
- Ipopt  iter 110: 76 entries, last entry `(theta=3.9220e5, phi=1.19454e7, iter=109)`

The off-by-1 at iter 109 vs 110 in entry counts is consistent with
augmentation-cycle timing. Entry contents agree to within Ipopt's
gamma-correction (`(phi - γ_φ·θ, (1-γ_θ)·θ)`); ripopt currently
augments raw `(theta_current, phi_current)` per `src/ipm.rs:2892`.

**Filter is NOT the divergence driver.** Hypothesis (b) from the
previous section is **disproved**.

## True root cause: search direction sign at iter 110

Comparing the iter-110 LS internals between Ipopt and ripopt at the
same iterate:

```
ripopt: phi_curr=1.19450e7  gBD=-1.544e6  (descent direction)
        first trial alpha=2.519e-6 → phi_tr=1.19449e7 (Δφ=-18.3)
        Armijo passes trivially, ACCEPT

ipopt:  phi_ref =1.19454e7  gBD=+1.252e7  (NOT a descent direction)
        first trial alpha=2.515e-6 → phi_tr=1.19456e7 (Δφ=+14.1)
        Sufficient reduction FAILS
        backtracks: 1.26e-6, 6.29e-7 → both fail
        Soft restoration step at 8.63e-8 → fails
        Restoration phase entry (R 4)
```

**Same iterate to printed precision; opposite-sign `grad_phi_step`.**

This means **ripopt's `dx` at iter 110 differs from Ipopt's `dx`** in
a way that flips the directional derivative sign. The breakdown
attribution:

```
ripopt gBD = ∇f·dx + (xb)·dx + (sb)·ds
           = +6.65e4 + (-2.69e6) + (+1.08e6) = -1.54e6
```

The `(xb)·dx` term is dominated by **variable 533** alone:
- `x[533] = x_U[533] = 1341.0`, `slack=8.31e-5`, `μ/slack = +1203`
- `dx[533] = -953` → contribution `= -1.15e6` (75% of `(xb)·dx`)

If `dx[533]` were O(10) (Ipopt's plausible value), the variable-533
contribution shrinks to O(-12000), and `(xb)·dx ≈ -1.54e6 + 1.15e6
= -0.39e6`. Total `gBD ≈ +6.65e4 - 0.39e6 + 1.08e6 = +7.6e5` — same
sign as Ipopt's `+1.25e7`, much closer in magnitude.

**The KKT solve at iter 110 is producing a wildly different `dx[533]`**
between ripopt (using feral) and Ipopt (using MA27/MUMPS). The
inverted-sign gBD then hands the LS a fake "descent direction" that
admits no-movement trials.

### Action items (KKT precision)

1. **Compare KKT residuals at iter 110** between ripopt and Ipopt.
   Dump `r_x` and `r_c` before linear solve in both. If residuals
   match but `dx` differs, the linear solver / regularization is the
   culprit. If residuals differ, the bug is upstream in residual
   assembly.

2. **Try ripopt with rmumps backend** at iter 110 to isolate the
   linear-solver effect. If MUMPS produces dx[533] closer to Ipopt's,
   feral is the source of precision loss; if MUMPS reproduces the
   same dx[533]=-953, the issue is in regularization or the residual.

3. **Check delta_w / delta_c regularization at iter 110**. A near-bound
   variable with σ = z/s = 776/8.3e-5 ≈ 9.4e6 contributes a huge
   diagonal entry to the W matrix; without sufficient regularization,
   the linear solve is ill-conditioned and `dx[533]` becomes noise.
   Compare ripopt's δ_w at iter 110 against Ipopt's `lg(rg) = -1.6`
   (i.e., δ_w ≈ 2.5e-2).

**Refs**: this section's evidence at line ~75 of Ipopt's iter 110→111
filter dump; `src/ipm.rs:2892` (raw augmentation); ripopt
`grad_phi_step` breakdown at `src/ipm.rs:2076`.

## 2026-05-04 trace: full-run mu and |y_d| trajectory

Source: /tmp/arki_probe.out (ripopt MaxIter=500, RIPOPT_*_PROBE
on, baseline feral backend).

**mu pinned for entire 500-iter run**: outer lg(mu) stays at -1.0
(μ=0.1) for every non-restoration iter. The only non-(-1.0) μ
values are the inner restoration iters (lg(mu) ∈ {5.4, 5.2, 5.1,
4.8, 4.7, 4.4, 4.3, 4.0, 3.6, 3.5, 3.1}) — these are the
restoration NLP's own μ, not the parent's. The mu-update gate
fires zero times across the entire run (consistent with prior
RIPOPT_TRACE_MU finding: 488 evaluations, 0 fires, closest E_μ
ever gets to threshold is 3.92e3 / 1.0 = 3920×).

**|y_d| does not "shrink"** — it explodes then gets reset by the
restoration handoff:

| iter |  obj   | inf_pr |   |y_c|   |  |y_d|   |   |z_L|   |  |z_U|   |
|-----:|-------:|-------:|--------:|---------:|---------:|---------:|
|    0 | 0      |1.16e8  | 1.80e2  |  1.80e0  |  1.00e0  |  1.00e0  |
|   24 |        |        | 2.16e2  |  3.46e1  |          |          |
|   99 |        |        | 3.08e6  |  1.85e6  |          |          |
|  124 | 1.20e7 |2.81e5  | 4.58e7  |  2.72e7  |  3.52e7  |  3.53e7  |
|  126 | 1.20e7 |2.80e5  | 4.63e7  |  2.74e7  |  3.66e7  |  3.68e7  |
|*resto*|       |        |         |          |          |          |
|  127 | 1.19e7 |2.39e5  | 0.00e0  |  0.00e0  |  1.00e0  |  1.00e0  |
|  128 | 1.19e7 |2.39e5  | 1.30e1  |  1.90e2  |  1.01e0  |  1.00e0  |
|  299 | 5.6e1  |        | 5.57e1  |  5.55e1  |          |          |
|  499 | 1.0e6  |7.43e1  | 8.13e2  |  3.25e2  |  1.02e2  |  2.68e3  |

The iter-127 row is the post-restoration handoff:
- y reset to 0 by `recompute_y_after_restoration` (Ipopt-default
  `constr_mult_reset_threshold=0.0`, `src/ipm.rs:8123-8128`)
- z reset to 1 by `update_bound_multipliers_after_restoration`'s
  nuclear-reset branch (any multiplier > 1e3 → 1.0,
  `src/ipm.rs:8328`) — fires because pre-resto |z|=3.5e7 > 1e3
- mu carried over from pre-resto (state.mu.max(mu_min) =
  max(0.1, 1e-11) = 0.1, `src/ipm.rs:8354`)

After the reset, iters 128→499 stay stuck because the bound
slacks at xL[1663-1700] sit at scale 1e6-1e7, and the reset z=1
gives complementarity products s·z ≈ 1e7 — which the μ=0.1
target cannot drive down. The barrier-error gate evaluates 488×,
fires 0×.

### What this confirms

**The post-restoration handoff is correct against Ipopt 3.14.**
The doc-comment at `src/ipm.rs:8095-8150` documents the exact
match: `constr_mult_reset_threshold=0.0` is Ipopt's published
default, and the nuclear z-reset for |z|>1e3 is the documented
Ipopt behavior at `IpRestoMinC_1Nrm.cpp:374-419`. The handoff is
doing the right thing **given catastrophically inflated
multipliers**.

### What this does NOT explain

The handoff is downstream of the real bug. From the prior
forward-walk: ripopt and Ipopt are **bit-identical for iters 0-110**.
Ipopt enters restoration at iter 110 (matching the iter-110 LS
divergence section above); ripopt continues normal stepping
until iter 126 because the iter-110 LS accepted on an
inverted-sign gBD (variable 533 barrier trap).

Over iters 111-126, ripopt's solver takes 16 tiny steps that
inflate the multipliers from O(1e3) to O(1e7). The nuclear
reset at iter 127 is then unavoidable, and from there the run is
dead.

**The handoff is not the bug. Iter 110 is the bug.** Fixing
iter 110 (so ripopt enters restoration at the same iter as
Ipopt, with multipliers still O(1e3)) would let the handoff
return to z=1 — but with bound slacks of O(1e3) instead of
O(1e7), so s·z ≈ 1e3 vs μ=1e-1 ratio ~ 1e4 instead of 1e8,
which is recoverable in O(20) outer iters as Ipopt demonstrates
(iter 114 → 318 in 204 iters).

### Follow-up

The KKT-precision follow-up at the end of "2026-05-04 follow-up:
filter contents are NOT the divergence driver" remains the
correct next probe. Specifically: dump KKT residuals (r_x, r_c)
and dx[533], dz_U[533] at iter 110 with the feral backend, then
repeat with rmumps to isolate the linear-solver effect.

**Refs**: /tmp/arki_probe.out (full trace);
src/ipm.rs:8095-8372 (post-resto handoff); prior section
"2026-05-04 follow-up: filter contents are NOT the divergence
driver" (gBD sign inversion at iter 110).

## 2026-05-04 KKT direction probe at iters 105-111 (var 533)

Source: `RIPOPT_DX_PROBE=ITER,VAR` env-gated probe in
`install_step_directions` at `src/ipm.rs:9678` (added this session).

Trajectory of var 533 (the variable identified earlier as the
barrier-trap source) across iters 105-111:

| iter | x[533] | s_U     | dx[533]   | z_U[533] | dz_U[533] | \|dy_d\|∞ |
|-----:|-------:|--------:|----------:|---------:|----------:|----------:|
|  105 |   1341 | 1.11e-4 | +2.94e-5  |  2.68e5  | -1.96e5   |  8.31e6   |
|  106 |   1341 | 1.03e-4 | +1.07e-4  |  1.37e5  | +5.98e3   |  3.82e6   |
|  107 |   1341 | 8.69e-5 | +2.38e-5  |  1.43e5  | -1.02e5   |  1.12e7   |
|  108 |   1341 | 8.09e-5 | +5.06e-5  |  1.22e5  | -4.46e4   |  2.72e6   |
|  109 |   1341 | 7.98e-5 | -1.48e-2  |  7.76e4  | -1.45e7   |  1.44e7   |
|  110 |   1341 | 8.31e-5 | **-9.53e2** | 7.76e2 | **-8.90e9** | **3.13e10** |
|  111 |   1341 | 2.48e-3 | -1.34e3   |  7.76e0  | -4.19e6   |  3.06e12  |

### Three quantitative findings

**(A) z_U[533] decays at exactly the FTB floor.** At iters
109→110 and 110→111, z_U drops by a factor of 100 each iter:
7.76e4 → 7.76e2 → 7.76e0. With Ipopt's default `tau_min=0.99`,
the dual fraction-to-boundary cap is `(1−τ)=0.01` — i.e., z can
drop to at most 1% of its previous value per step. The exact 100×
ratio means FTB is binding twice consecutively on this component.

**(B) dz_U formula is consistent.** The augmented-system
recovery `dz_U = -(z_U/s_U)·dx_U + μ/s_U − z_U` evaluates at
iter 110 to `-(776/8.31e-5)·(953) + 0.1/8.31e-5 − 776 = −8.90e9`
— an exact match to the probe's dz_U=-8.897e9. So the
catastrophic dz_U is a faithful consequence of the catastrophic
`dx[533]=-953`, not a bug in the dz recovery code.

**(C) The catastrophic step is dx[533] itself.** At iters 105-108
the KKT solve produced dx[533] in the range +1e-5 to +1e-4
(microscopic, sane for a variable pinned at its upper bound).
At iter 109 the magnitude jumps to -1.48e-2 (still small but
sign flipped — moving away from U). At iter 110 it explodes to
**-9.53e2** — a 71% relative move on a bound-pinned variable.

### Cross-backend comparison was inconclusive

`cargo build --release --no-default-features --features "rmumps faer"` and rerun:
**rmumps backend exits at iter 0** with "Numerical Difficulties
Encountered" — different early-iter behavior, so apples-to-apples
iter-110 comparison wasn't possible. The iter-0 KKT failure is a
separate backend issue worth investigating but does not bear on
the iter-110 question.

### What this means

The bug is **upstream of the linear solve**. dx[533]=-953 is the
solver's faithful response to whatever the iter-110 KKT RHS row
533 demands — most likely a large dual residual (∇_x L)_533 that
the linear solve translates into a large step. With z_U[533]=776
and the barrier-term µ/s_U=1203 already fighting each other, any
imbalance in the gradient or J^T y assembly at row 533 will
produce a large dx and cascade into the dz_U=-O(z·dx/s)
catastrophe.

This narrows the next probe sharply: dump the KKT system's
**RHS row 533** (the dual residual for var 533) at iters 109 and
110, plus the diagonal Schur entry at row 533 (which controls
how an RHS becomes a dx). If the RHS jumps by a factor of 1e4
between iters 109 and 110, the bug is in residual assembly. If
the RHS is smooth but the diagonal collapses, the bug is in the
Hessian/regularization at that row.

### Follow-up (this replaces the prior KKT-residual action item)

1. **Dump RHS row 533 and diagonal at iters 109-110.** Add an
   env-gated probe inside `assemble_kkt_from_state` (or wherever
   the RHS is finalized, `src/ipm.rs:4606`) that prints
   `rhs[533]` and the assembled diagonal Schur at row 533.
2. **Also dump `state.grad_f[533]`, `(J^T y)_533`, `z_L[533]`,
   `z_U[533]`, and `mu/s_L`, `mu/s_U`** at the same iters. The
   four gradient components compose `r_x[533]`; whichever
   component spikes is the source.
3. **Compare iter-109 vs iter-110 component breakdowns.** A
   single component that jumps by O(1e3) between consecutive
   iters is the bug.

**Refs**: `src/ipm.rs:9678` (probe site);
`/tmp/arki_probe.out` (existing iter-N-probe output for context).

## 2026-05-04 RESOLUTION: missing linear-system equilibration

### What was wrong

The augmented KKT system at iter 110 has rows that span 9+ orders of
magnitude in scale. Variable 533 sits at `x = x_U = 1341` with
`s_U = 8.31e-5`, giving `Σ_U = z_U/s_U ≈ 9.4e6` — a single diagonal
entry that is 7+ orders larger than typical Hessian rows. Without
row/column equilibration, the LDLᵀ factorization loses precision on
that row. The IR loop cannot recover because the residual it iterates
on is itself measured in the unscaled coordinate system.

The faithful symptom: `dx[533] = -953` (a 71% relative move on a
bound-pinned variable), which propagates via the dz recovery formula
into `dz_U = -8.9e9` and `|dy_d|∞ = 3.13e10`. Probe code path:
`src/ipm.rs:9675-9710` (RIPOPT_DX_PROBE).

### What Ipopt does that ripopt was not

Ipopt 3.14 has a `linear_system_scaling` option. Defaults by linear
solver:

  - MA27: `mc19` (Curtis-Reid iterative scaling)
  - MA57: `slack-based`
  - MUMPS: `none` (the only backend ripopt previously parroted)

Confirmed by the ipopt-expert agent reading the Ipopt source.

ripopt's main aug-path (`aug_step_from_state` in `src/kkt_aug.rs`)
had **no equilibration at all** — the existing
`kkt::factor_with_inertia_correction` had a `use_scaling` gate but it
defaulted off and was only triggered as a failure-recovery fallback.

### The fix (env-gated, working)

`src/kkt_aug.rs:1050-1080`: added Ruiz equilibration around the
factor/IR-solve. Calls `kkt::ruiz_equilibrate(&mut aug.matrix, &mut
aug_rhs)` to compute symmetric `D · A · D` row/column scaling on the
augmented matrix, scales the RHS in place, and unscales the solution
after IR. Gated on `RIPOPT_RUIZ_AUG=1` for now.

### Before/after on arki0003 (full run, max_iter=500)

| metric                | baseline (no scaling) | RIPOPT_RUIZ_AUG=1   | Ipopt MA27 (ref) |
|-----------------------|-----------------------|---------------------|------------------|
| terminal status       | MaxIter               | MaxIter             | Optimal (318)    |
| terminal obj          | 9.09e5                | **3.7952233e3**     | 3.7952233e3      |
| terminal lg(mu)       | -1.0 (pinned)         | **-8.6**            | -                |
| iter-110 dx[533]      | -953                  | **-24.7**           | (sane)           |
| iter-110 x[533]       | 1341 (= x_U)          | **22.6 (interior)** | -                |
| iter-110 dz_U[533]    | -8.9e9                | **-1.5e-4**         | -                |
| iter-110 \|dy_d\|∞    | 3.13e10               | **663**             | -                |

The objective matches Ipopt's converged value to printed precision.
The remaining gap to "Optimal" status is dual feasibility:
`inf_du = 1.75e7` (scaled) at iter 499, which prevents formal
convergence at MaxIter — possibly resolvable with more iters or with
`s_d` residual-scaling alignment.

### Open work for the resume point

1. **Promote `RIPOPT_RUIZ_AUG` env-gate to a real `SolverOptions`
   field** (e.g., `linear_system_scaling: enum { None, Ruiz }` in
   `src/options.rs`). Mirror Ipopt's `linear_system_scaling` option
   semantics. Default value is the open question — see (2).

2. **Default value decision needs a benchmark sweep.** Two reasonable
   defaults:
     - `None` (matches Ipopt's MUMPS-default; opt-in fix; zero
       regression risk on the existing CUTEst suite)
     - `Ruiz` (closer in spirit to Ipopt's MA27 default since `feral`
       is a direct factorization, not iterative; fixes arki0003 out
       of the box; needs benchmark sweep to confirm no per-iter
       overhead regressions on currently-converging problems)
   Recommendation: ship the option with `None` default first, run
   `make benchmark` with `Ruiz`, flip the default in a follow-up
   commit if the suite is clean.

3. **Investigate residual dual infeasibility** with Ruiz on. With
   the fix arki0003 reaches the right objective but `inf_du` stays
   at 1.75e7 (scaled). Likely candidates: (a) `s_d` residual
   normalization is being computed in unscaled coordinates and the
   convergence test sees a bigger number than it should; (b) the IR
   tolerance needs to be tightened in scaled coordinates; (c) just
   needs more iterations.

4. **Diagnose rmumps backend iter-0 NumericalDifficulties** on
   arki0003. Cross-backend comparison was inconclusive because the
   rmumps build exits before iter 1 on this problem. Separate issue,
   not a blocker for the equilibration fix.

5. **Decide whether to keep the diagnostic probes long-term.** The
   `RIPOPT_FILTER_DUMP` (`src/ipm.rs:7505-7525`) and
   `RIPOPT_DX_PROBE` (`src/ipm.rs:9675-9710`) probes added during
   this investigation are env-gated (zero cost when off) and were
   useful — recommend keeping until the Ruiz fix is fully promoted
   and the residual dual-inf issue is resolved.

### Files touched in this session

  - `src/kkt_aug.rs:1050-1080` — Ruiz equilibration hook (env-gated).
  - `src/ipm.rs:7505-7525` — `RIPOPT_FILTER_DUMP` env-gated diag.
  - `src/ipm.rs:9675-9710` — `RIPOPT_DX_PROBE` env-gated diag.
  - `examples/arki_probe2.rs` — env-controlled max_iter for probes.
  - `docs/A8_FOLLOWUP_arki0003.md` — this document (5 appended
    sections).

### How to reproduce the fix

```
cargo build --release --example arki_probe2
RIPOPT_RUIZ_AUG=1 RIPOPT_MAX_ITER=500 \
  cargo run --release --example arki_probe2 > /tmp/arki_ruiz.out 2>&1
grep -E '^[ ]*(0|100|200|300|400|499)\b' /tmp/arki_ruiz.out | tail -20
```

Expected terminal obj ≈ 3.795e3 (matches Ipopt). Without
`RIPOPT_RUIZ_AUG=1`: terminal obj ≈ 9.09e5.

## 2026-05-04 RESOLUTION REVISION: Ruiz reverted, real root cause is iter-110 LS acceptance

The "Ruiz fix" above was reverted in a later iteration of this session.
While it does produce the right terminal objective, it masks the actual
bug rather than fixing it. With Ruiz on, ripopt's trajectory diverges
from Ipopt's much earlier than iter 110 (e.g. obj at iter 105 = 3.97e5
with Ruiz vs 1.19e7 bit-identical-with-Ipopt without Ruiz). The
Ruiz-perturbed trajectory happens to converge but is not aligned with
Ipopt's iterate path.

### The iter-109 → 110 → 111 chain (root cause)

Verified via env-gated probes (`RIPOPT_RHS_PROBE`, `RIPOPT_DX_PROBE`)
in baseline ripopt (no Ruiz):

**Iter 109 entry** for variable 533 (both bounds finite):
- `x = x_U = 1341`, `s_U = 8.0e-5`, `s_L = 1341` (lower bound at -1e-8 floor)
- `z_U = 7.76e4`, `z_L = 7.5e-5`
- `z_U·s_U = 6.2` vs target `μ = 0.1` → 60× above target
- `Sigma_x[533] = z_U/s_U = 9.72e8`
- `J^T y = -9.6e4`, `grad_f = 0` (var not in objective)
- `aug_rhs[533] = +9.77e4`
- KKT solve produces `dx[533] = -0.015` (sane), `dz_U[533] = -1.45e7` (huge negative)

**FTB binding at iter 109→110**: dual fraction-to-boundary `(1−τ_min)
= 0.01` caps z_U descent. `α_du = (z_U − floor)/|dz_U| = (7.76e4 −
776)/1.45e7 = 5.30e-3`, taking `z_U` from 7.76e4 to **exactly 776**
(100× drop, FTB-saturated).

**Iter 110 entry** for variable 533:
- `s_U = 8.31e-5` (barely changed), `z_U = 776`
- `Sigma_x[533] = z_U/s_U = 9.33e6` (collapsed 100× from 9.72e8)
- `aug_rhs[533] = +9.46e4` (basically unchanged)
- KKT solve produces `dx[533] = -953` (catastrophic, dominated by
  off-diagonal H/J^T couplings since the diagonal is now 100× weaker)
- `dz_U[533] = -8.9e9`

**Iter 110 LS acceptance** (the actual bug):
The KKT-derived FTB caps `α_p_max = 2.519e-6` (clipped by the huge
dz_U). Ripopt's first LS trial:
```
α=2.519e-6 gBD=-1.544e6 Δφ=-18.3 Δθ=-0.99
is_ft=true armijo=true suf_phi=true in_filter=true → ACCEPT
```
The trial moves the iterate by essentially nothing (`Δθ/θ = 2.5e-6`,
~0.0001%) but Armijo passes because `Δφ = -18.3` is "very negative"
relative to `η_φ·α·gBD = 1e-4 · 2.5e-6 · (-1.5e6) = -3.86e-4`.

**Ipopt at the same iter rejects 4 backtracks then enters
restoration.** Verified via earlier forward-walk diff in the section
"2026-05-04 follow-up: filter contents are NOT the divergence
driver". Both solvers are bit-identical to printed precision through
iter 110, so the entry state for iter 110's LS is identical — the
divergence is purely in LS acceptance.

### What Ipopt does NOT do (verified via ipopt-expert agent)

1. **Ipopt has no `kappa_sigma`-based clamp at iter 109's z_U.**
   Default `kappa_sigma = 1e10` (`IpIpoptAlg.cpp:71-79`) gives clamp
   range `[μ/(κσ·s), κσ·μ/s] = [1.25e-7, 1.25e13]`. `z_U = 7.76e4`
   sits well inside. Ipopt is also not clamping here.

2. **Ipopt has no "α_p_max small → restoration" trigger.**
   `DetectTinyStep` (`IpBacktrackingLineSearch.cpp:1219-1279`) has
   the OPPOSITE semantics: when `max|dx_i|/(1+|x_i|) ≤ 2.2e-15` and
   constraint violation ≤ 1e-4, it ACCEPTS the FTB step unchecked
   and bypasses the LS. It does not enter restoration. Restoration
   is only entered when `α_primal ≤ α_min` AND no trial passed
   (`IpBacktrackingLineSearch.cpp:740`).

3. **Ipopt has no Sigma-collapse monitoring.** `curr_sigma_x`
   (`IpIpoptCalculatedQuantities.cpp:3501-3525`) computes Sigma
   lazily without comparison to prior iter.

So Ipopt's restoration entry at iter 110 must be coming from **trial
rejection by filter/Armijo on the very first trial**, not from any
α-magnitude or Sigma-collapse guard.

### The real divergence (open question)

**Why does ripopt's first LS trial at iter 110 pass Armijo+filter
when Ipopt's same trial gets rejected?**

The candidate diff targets per ipopt-expert:
- (a) `phi` (barrier objective) trial computation
  — `IpIpoptCalculatedQuantities::trial_barrier_obj`
- (b) `gBD = grad_phi^T · d` directional derivative
  — `IpFilterLSAcceptor::CalcBarrierObjDerivative` (eqn 19)
- (c) `alpha_min` formula (γθ, γφ, δ exponents)
  — `IpFilterLSAcceptor.cpp:454-468`
- (d) Filter contents — does ripopt's filter at iter 110 contain the
  entries Ipopt's does that would reject this trial?

The next concrete probe: run Ipopt on arki0003 with `print_level=12`
to capture iter 110 first-trial values for `Δφ`, `gBD`, `α_min`, and
filter membership; then diff against ripopt's existing probe outputs.
If the values disagree, the disagreement IS the bug.

### What was reverted in this session

- `src/kkt_aug.rs`: the `RIPOPT_RUIZ_AUG` env-gated Ruiz hook (lines
  1052-1080 of the prior diff). Removed because it masks the real
  iter-110 LS acceptance bug rather than fixing it.

### What was kept

- `src/ipm.rs`: env-gated diagnostic probes that helped localize this
  finding and remain useful for the next iteration:
    - `RIPOPT_FILTER_DUMP=lo,hi` (filter contents per iter)
    - `RIPOPT_DX_PROBE=ITER,VAR` (post-solve dx/dz dump)
    - `RIPOPT_RHS_PROBE=ITER,VAR` (pre-solve aug-RHS component
      breakdown)
- `examples/arki_probe2.rs`: env-controlled `RIPOPT_MAX_ITER`.

### Resume point: Task #18

The next session should focus on task #18: diff ripopt vs Ipopt LS
acceptance at iter 110 first trial. The ipopt-expert's recommendation
(file:line citations in the agent transcript) is the starting point.

**Key insight**: with Ruiz reverted, ripopt is once again bit-identical
to Ipopt through iter 110. So the iter-110 LS acceptance test runs on
identical input state in both solvers. Any difference in Δφ, gBD, or
α_min is a bug in ripopt's implementation of those formulas.
