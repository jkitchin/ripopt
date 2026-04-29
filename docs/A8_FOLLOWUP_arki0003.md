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
