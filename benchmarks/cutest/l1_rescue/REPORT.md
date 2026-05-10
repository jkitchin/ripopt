# ℓ₁-Exact Penalty-Barrier Rescue Probe (Issue #23 Phase 3)

Curated CUTEst suite of 42 NE-suffix (nonlinear equation) problems where
ripopt v0.8.0 baseline fails (RestorationFailed, LocalInfeasibility,
MaxIterations). Each is solved twice:

- **off**: default options
- **on**: `l1_exact_penalty_barrier = true` (BNW outer loop, ρ steered by
  `τ·‖y_eq‖∞ + 1`, honest infeasibility via slack-collapse check)

How to reproduce:
```
RIPOPT_L1_PENALTY=1 ./target/release/cutest_suite --single PROBLEM --solver ripopt
```

## Headline numbers

| Bucket | Count | Notes |
|--------|------:|-------|
| Hard rescues (off≠Optimal → on=Optimal)              |  2 | DENSCHNENE, HATFLDFLNE |
| Honest infeasibility upgrades (off≠LocInf → on=LocInf) | 26 | mostly RestorationFailed → LocalInfeasibility |
| Regressions (off=Optimal → on≠Optimal)               |  0 | by construction the suite has no off=Optimal cases |
| No-change                                             | 14 | wrapper agreed with off-baseline status |

## Hard rescues

| Problem    | off status         | off iters | on iters |
|------------|--------------------|----------:|---------:|
| DENSCHNENE | LocalInfeasibility |        12 |        9 |
| HATFLDFLNE | RestorationFailed  |       413 |        6 |

These are the unambiguous wins — the wrapper finds a feasible Optimal where
the standard solver could not.

## Honest infeasibility upgrades (selected)

Vanilla returns `RestorationFailed` or `MaxIterations` (uninformative); the
wrapper returns `LocalInfeasibility` (KKT of the ℓ₁-penalty problem with
non-collapsing slacks ⇒ provable local infeasibility certificate per BNW).

| Problem    | off status         | off iters | on iters |
|------------|--------------------|----------:|---------:|
| BARDNE     | RestorationFailed  |        13 |       50 |
| HATFLDBNE  | MaxIterations      |      2999 |       55 |
| HS2NE      | RestorationFailed  |       128 |       30 |
| JUDGENE    | RestorationFailed  |        14 |       65 |
| YFITNE     | MaxIterations      |      2999 |       70 |
| WEEDSNE    | RestorationFailed  |       355 |      135 |
| (+20 more PALMER family) |     |           |          |

In particular, HATFLDBNE / YFITNE / PALMER1BNE / PALMER1ENE convert long
MaxIterations runs (3000 it) into fast (~50–200 it) infeasibility
certificates — a substantial wall-time improvement even when the answer is
"no feasible solution".

## Methodology

The probe shells out to `cutest_suite --single` with and without
`RIPOPT_L1_PENALTY=1`. Default tolerances: `tol = 1e-8`, `max_iter = 3000`,
`max_wall_time = 30s`, `mu_strategy_adaptive = true`. The wrapper outer loop
uses `l1_steering_factor = 10.0` and the slack-collapse threshold
`l1_slack_tol = 1e-6` (both `SolverOptions` defaults).

Raw artifacts:
- `problems.txt`         — input list
- `off.jsonl` / `on.jsonl` — one CUTEst result per line per configuration

## Flag-off byte-identical regression evidence

To confirm that the L1 work does not change the default-options trajectory,
the first 100 problems from `problem_list.txt` are run twice:
- **pre**: pre-L1 commit (`dc52017`, in `/tmp/ripopt-pre-l1` worktree)
- **post**: current head, default flags (`l1_exact_penalty_barrier=false`)

| Field                | Bytes |
|----------------------|------:|
| status, iterations, objective, final_primal_inf, final_dual_inf, final_compl, final_mu | identical on 99/100 |
| sole mismatch        | `CRESC132` — `MaxTimeExceeded` outcome (wall-time non-determinism, 340 vs 324 iters) |

Artifacts: `regression_pre_l1.jsonl` / `regression_post_l1_flag_off.jsonl`.

Combined with the structural argument (every `l1_*` field read and the entire
`L1PenaltyBarrierNlp` outer loop are inside `if options.l1_exact_penalty_barrier
{ ... return ...; }` at `src/ipm.rs:3348`, and the public `solve` wrapper at
`src/lib.rs:118` short-circuits on `!options.l1_fallback_on_restoration_failure`
before any extra ipm call), this confirms the L1 work introduces no
regression on the default-options path.
