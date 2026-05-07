# Ipopt Alignment Action List

This document is a narrowed, actionable version of the ripopt-vs-Ipopt audit.
It only lists **live deviations** that should be removed or changed so ripopt
behaves like Ipopt 3.14.

Rules for this list:

- If ripopt has behavior with no Ipopt analogue, the action is to delete it.
- If ripopt implements the same mechanism differently, the action is to replace
  it with the Ipopt version.
- Every item includes a success condition that is testable.
- Historical items already fixed on the current branch are omitted.

Primary references:

- `ref/Ipopt/src/Algorithm/*`
- `docs/IPOPT_ALGORITHM_SPEC.md`
- current ripopt sources under `src/`

## Priority Order

Implement in this order:

1. default/options mismatches,
2. delete ripopt-only preprocessing,
3. replace restoration cascade with Ipopt restoration flow,
4. replace approximate mu oracle behavior,
5. either implement real magic-step support or remove the dead option,
6. replace simplified warm-start logic,
7. update docs to match the code.

## A1. Flip the fixed-variable default to Ipopt's `make_parameter`

Problem:

- `FixedVariableTreatment` defaults to `MakeParameter`, but
  `SolverOptions::default()` still sets `fixed_variable_treatment` to
  `RelaxBounds`.

Ripopt code:

- enum default: `src/options.rs:3-36`
- runtime default: `src/options.rs:660-780`, especially `:776`
- relax-bounds path: `src/ipm.rs:8882-8899`

Ipopt reference:

- `TNLPAdapter` default behavior is `fixed_variable_treatment = make_parameter`.

Action:

1. Change `SolverOptions::default().fixed_variable_treatment` from
   `RelaxBounds` to `MakeParameter`.
2. Keep `RelaxBounds` only as an explicit non-default option.

Delete / replace:

- Replace the current default-path use of `RelaxBounds`.

Success criteria:

- Unit test: default options report `FixedVariableTreatment::MakeParameter`.
- Regression test: a problem with one fixed variable is reduced before the IPM
  solve when using default options.
- No code path in a default solve widens fixed bounds by `1e-8*max(|c|,1)`.

## A2. Delete ripopt-only preprocessing that Ipopt does not do

Problem:

- ripopt preprocessing does more than Ipopt's `TNLPAdapter`:
  bound tightening from linear constraints and redundant-constraint removal.
- This changes the effective NLP before the interior-point algorithm even
  starts.

Ripopt code:

- preprocessing module: `src/preprocessing.rs:1-375`
- reduced solve dispatch: `src/ipm.rs:2235-2280`

Ipopt reference:

- `TNLPAdapter` removes fixed variables and normalizes the NLP, but does not run
  this general-purpose bound-tightening / redundancy pass in the main solver
  path.

Action:

1. Delete the bound-tightening pass in `PreprocessedProblem::with_options`
   (`src/preprocessing.rs:90-231`).
2. Delete the redundant-constraint detection pass
   (`src/preprocessing.rs:233-375`).
3. Retain only the fixed-variable elimination path needed for
   `make_parameter`.
4. Simplify `try_preprocessed_solve` so its only purpose is fixed-variable
   elimination.

Delete / replace:

- Delete ripopt-only preprocessing heuristics.
- Replace `PreprocessedProblem::new(...)` with a fixed-variable-only adapter, or
  route all preprocessing through `new_fixed_only`.

Success criteria:

- Code search: no live code remains that tightens bounds from single-variable
  constraints or drops constraints as redundant before solving.
- Unit tests formerly asserting bound tightening / redundancy elimination are
  removed or rewritten to reflect Ipopt behavior.
- A problem with duplicate or nearly duplicate constraints reaches the solver
  unchanged except for fixed-variable elimination.

## A3. Replace the restoration cascade with Ipopt's restoration flow

Problem:

- ripopt currently runs a broader post-line-search recovery cascade than Ipopt:
  soft restoration, Gauss-Newton restoration, NLP restoration only on selected
  failure counts, mu/mode perturbations, and x-perturbation recovery.
- Ipopt has a specific line-search/restoration transition, not a general
  recovery framework.

Ripopt code:

- cascade entry: `src/ipm.rs:3756-3833`
- soft restoration: `src/ipm.rs:4032-4205`
- restoration NLP wrapper: `src/restoration_nlp.rs:6-253`

Ipopt reference:

- `IpBacktrackingLineSearch.cpp`
- `IpRestoFilterConvCheck.cpp`
- `IpRestoIpoptNLP.cpp`
- `IpRestoMinC_1Nrm.cpp`

Action:

1. Remove fail-count-based gating such as "NLP restoration only on failures
   `{2,4}`".
2. Remove x-perturbation recovery and any non-Ipopt post-LS perturbation logic.
3. Remove the custom "cascade" concept and reframe the control flow as:
   line-search failure -> restoration entry -> restoration success/failure
   decision.
4. Keep only behaviors that correspond to Ipopt's backtracking line search,
   soft restoration, and restoration NLP transitions.
5. Rework restoration termination so it is driven by the Ipopt acceptance and
   convergence checks, not fail counters and ad hoc escalation stages.

Delete / replace:

- Delete `run_post_ls_restoration_cascade` as a cascade abstraction.
- Delete fail-count rotation logic and x-perturbation recovery from this path.
- Replace with an Ipopt-shaped restoration state transition.

Success criteria:

- Code search: no restoration dispatch remains keyed on selected failure counts
  like `{2,4}`.
- Code search: no x-perturbation recovery remains in the post-line-search
  restoration path.
- A failed line search reaches restoration by the same high-level decision path
  as Ipopt: no extra ripopt-only escalation layer.
- Existing restoration tests are updated to assert filter/restoration acceptance
  rather than cascade stage counts.

## A4. Replace the approximate quality-function mu oracle with Ipopt's real one

Problem:

- ripopt's quality-function oracle is explicitly approximate:
  linearized residual model, local refactorization, and `sigma <= 1.0`.
- Ipopt evaluates the true nonlinear residual at trial points.

Ripopt code:

- quality-function oracle and its own tech-debt note:
  `src/ipm.rs:4533-4580`

Ipopt reference:

- `ref/Ipopt/src/Algorithm/IpQualityFunctionMuOracle.cpp`

Action:

1. Remove the linearized `(1-alpha)*current` residual approximation.
2. Remove the ripopt-specific `sigma <= 1.0` cap if it is only there to stabilize
   the approximation.
3. Plumb the problem evaluations needed so the oracle can evaluate the actual
   nonlinear residual terms at the candidate trial point, as Ipopt does.
4. Keep the fallback to Loqo only for genuine factorization/evaluation failure,
   not because the oracle is structurally approximate.

Delete / replace:

- Delete approximation-specific logic in the current QF implementation.
- Replace it with the Ipopt quality-function calculation.

Success criteria:

- The comments in `compute_quality_function_mu` no longer describe linearized
  Q as intentional tech debt.
- The oracle evaluates objective/constraint/duality quantities at the trial
  point rather than only transforming current residuals.
- Tests cover at least:
  - successful QF mu selection on a small nonlinear problem,
  - fallback to Loqo on factorization failure,
  - behavior with and without the centrality term.

## A5. Remove or fully implement `magic_step`

Problem:

- `magic_step` is exposed as an option, but the main implicit-slack path is a
  no-op.
- Ipopt has a real magic-step mechanism for explicit slack variables.

Ripopt code:

- no-op implementation: `src/ipm.rs:4020-4029`
- test pinning no-op behavior: `src/ipm.rs:12108-12115`

Ipopt reference:

- `IpBacktrackingLineSearch.cpp`

Action:

Choose one path and commit to it:

1. If ripopt is staying on the implicit-slack architecture, delete the
   `magic_step` option and remove the dead code path entirely.
2. If strict Ipopt parity is the goal, explicit slack handling must be carried
   through the line search so the real magic step can be implemented.

Delete / replace:

- Preferred for alignment honesty: delete the exposed no-op option unless a full
  explicit-slack implementation is introduced in the same change.

Success criteria:

- Either:
  - no `magic_step` option exists anymore and no code advertises support, or
  - a line-search test shows that enabling magic-step changes the iterate in an
    Ipopt-consistent explicit-slack scenario.
- The current no-op invariant test is deleted if the feature is removed, or
  replaced with a behavior test if fully implemented.

## A6. Replace the simplified warm-start initializer with an Ipopt-style warm-start initializer

Problem:

- ripopt's warm-start initializer is much simpler than Ipopt's and computes the
  starting `mu` from complementarity before pushing `x` off the bounds.

Ripopt code:

- initializer: `src/warmstart.rs:9-78`
- integration: `src/ipm.rs:6061-6145`

Ipopt reference:

- `ref/Ipopt/src/Algorithm/IpWarmStartIterateInitializer.cpp`

Action:

1. Replace `WarmStartInitializer::initialize` with logic matching Ipopt's
   warm-start initializer semantics.
2. Use Ipopt's warm-start push policy for primal variables and multipliers.
3. Align handling of user-supplied `y`, `z_L`, `z_U`, `v_L`, `v_U`, and target
   `mu` with the reference implementation rather than the current simplified
   average-complementarity shortcut.

Delete / replace:

- Replace the current `mu = average complementarity before push` shortcut if it
  does not match Ipopt.

Success criteria:

- Unit tests cover:
  - warm start from a bound-active point,
  - multiplier floor behavior,
  - warm-start with supplied `target_mu`,
  - one-sided and two-sided bounds.
- The warm-start code can be traced directly to the same policy as
  `IpWarmStartIterateInitializer.cpp`, not to a ripopt-specific shortcut.

## A7. Make the public algorithm documentation match the actual Ipopt-aligned implementation

Problem:

- `docs/src/algorithm.md` still describes old ripopt-specific behavior, such as
  free mode as the default and a fallback stack including AL and SQP.

Ripopt docs:

- stale summary: `docs/src/algorithm.md:13-123`

Current code:

- monotone default: `src/options.rs:681-686`
- actual restoration and mu flow: `src/ipm.rs:3756-3833`, `:5046-5155`

Action:

1. Remove descriptions of deleted/non-Ipopt fallback mechanisms.
2. Update defaults to the actual current defaults.
3. Describe the algorithm in the same structural terms as Ipopt:
   explicit slack handling, filter line search, restoration, monotone/adaptive
   mu, and KKT solve.

Delete / replace:

- Delete stale references to AL/SQP-style recovery if those paths are gone.
- Replace the old summary with an Ipopt-structured description.

Success criteria:

- A reader comparing `docs/src/algorithm.md` against `src/options.rs` and
  `src/ipm.rs` finds no default mismatch.
- No documentation claims support for ripopt-only fallback paths that have been
  removed.

## A8. Reassess the condensed KKT path as the last major structural deviation

Problem:

- ripopt's main KKT solve is still a condensed split `(x,y)` system.
- Ipopt's core implementation is the full-space 4-block augmented solve.
- This is the single largest remaining architectural deviation.

Ripopt code:

- KKT assembly: `src/kkt.rs:118-340`
- slack handling around the state: `src/ipm.rs:6000-6058`

Ipopt reference:

- `IpPDFullSpaceSolver.cpp`
- `IpStdAugSystemSolver.cpp`

Action:

1. Decide explicitly whether strict Ipopt alignment requires moving ripopt from
   the condensed KKT path to the full-space augmented formulation.
2. If yes, create a separate implementation track for:
   - explicit `(x, s, y_c, y_d)` Newton system assembly,
   - eliminated `z/v` recovery matching Ipopt,
   - iterative refinement and perturbation behavior on that full system.
3. Treat this as a major architectural replacement, not a small cleanup.

Delete / replace:

- Ultimately replace the condensed system if exact algorithmic parity is the
  objective.

Success criteria:

- The assembled Newton system matches the block structure in
  `docs/IPOPT_ALGORITHM_SPEC.md`.
- Residual formation and step recovery can be line-mapped to
  `IpPDFullSpaceSolver.cpp`.
- Existing condensed-system-only assumptions are removed from tests and docs.

## Acceptance Checklist

This document should be considered complete only when:

1. Every action item above is either:
   - implemented and checked off with tests, or
   - explicitly rejected as incompatible with a chosen non-Ipopt architecture.
2. No option remains exposed in ripopt that advertises an Ipopt mechanism while
   doing nothing.
3. No ripopt-only recovery layer remains in the default solve path unless it is
   documented as a conscious divergence.
4. The default solve path and docs describe the same algorithm.

## Recommended Next Changes

If you want the fastest path to tighter alignment, do these next:

1. `A1` fixed-variable default.
2. `A2` delete preprocessing heuristics.
3. `A3` delete the restoration cascade abstraction and remove fail-count/x-perturbation recovery.
4. `A5` remove the dead `magic_step` option unless full implementation is imminent.
5. `A7` rewrite `docs/src/algorithm.md` after the code changes land.

If you want the path to deepest parity, the long pole is `A8`: replacing the
condensed KKT formulation with Ipopt's full-space augmented solve.
