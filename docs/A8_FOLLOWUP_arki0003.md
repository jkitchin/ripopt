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

