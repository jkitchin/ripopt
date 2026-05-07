# Ripopt v0.8 Gate Audit vs Ipopt 3.14 Reference

Date: 2026-05-02 — branch `v0.8/feral-default`

Cross-checks ripopt's conditional logic ("gates"), numeric thresholds,
parameter defaults, and state transitions against Ipopt 3.14. Catalog
of ~80 gates produced by the ipopt-expert agent; all critical gates
audited. Outcome:

- **2 bugs fixed** during the audit (one latent, one numeric default).
- **3 known issues** documented — all in the iter-0 LS-init pivoting path
  and tracked against feral upstream.
- **All other audited gates** verified consistent with Ipopt 3.14.
- A small set of advanced/optional Ipopt features are intentionally not
  implemented; absence is consistent with shipping defaults.

## Bugs fixed during audit

### A3 — no-bounds μ-update gate (FIXED)
Location: `src/ipm.rs:4905-4923` (`update_barrier_parameter`)

Ipopt: when both var bounds and inequality slacks are absent
(`IpIpoptCalculatedQuantities::CurrAvrgCompl` → 0), `mu` collapses with
`mu^superlinear_power` only. ripopt previously gated only on the var-bound
predicate (`x_l/x_u empty`), ignoring inequality slacks.

Effect: `qcqp1500-1c` (0 var bounds, 10008 inequalities → real slack
barrier present) collapsed `mu` from 1e-1 to 1e-11 in 6 iterations.

Fix: gate now requires `!has_var_bounds && !has_slack_barrier`. Verified
trajectory on qcqp1500-1c now matches Ipopt monotone descent.

### A7 — `refs_red_fact` default (FIXED)
Location: `src/ipm.rs:987` (`MuState::default`)

Ipopt option `adaptive_mu_kkterror_red_fact` default is `0.9999`. Ripopt
shipped `0.999`. One-character change (323 tests pass unchanged). Effect:
slightly tighter "sufficient progress" classification in Free mode mu
update — fewer iters classified as "sufficient", more KKT-error checks
trigger the alternative path.

## Known issues — documented, not yet resolved

All three concern the iter-0 least-squares multiplier init solve when
J is rank-deficient with many simple linear inequality rows
(e.g. `arki0003`, `qcqp1500-1c`):

1. **feral BK pivoting returns zero `y_d` on rank-deficient + simple
   inequality rows**: feral with default `pivot_threshold = 0` returns
   58 structurally-zero `y_d` entries on `arki0003` rows where MUMPS
   (homebrew Ipopt's actual backend, `cntl(1) = 0.01`) returns y ≈ 0.8.
   86× iter-0 dual infeasibility on qcqp1500-1c. Filed against feral.

2. **`pivot_threshold = 1e-8` no-op on this matrix class**: feral's BK
   threshold formula `max(u·col_max, zero_tol)` never rejects pivots
   when rejected pivot is column-saturated at unit magnitude. Mitigation
   in place (`new_sparse_solver_for_ls()` factory at `src/ipm.rs:32`)
   but doesn't change the offending matrix's output.

3. **SYMSOLVER_SINGULAR fallback inert**: implemented at `ipm.rs:8645`
   (retries with `δ_c = δ_d = 1e-8` on factorization failure). Doesn't
   trigger because feral never signals singular for these matrices.

## Verified consistent with Ipopt 3.14

### Convergence / termination (C1–C16)

| Gate | Where | Ipopt ref | Status |
|------|-------|-----------|--------|
| C1 — `tol = 1e-8` | `options.rs:652` | `IpIpoptData.cpp:27-41` | ✓ |
| C2 — `dual_inf_tol = 1.0` | `options.rs:674` | `IpOptErrorConvCheck.cpp:47-53` | ✓ |
| C3 — `constr_viol_tol = 1e-4` | `options.rs:673` | `IpOptErrorConvCheck.cpp:54-62` | ✓ |
| C4 — `compl_inf_tol = 1e-4` | `options.rs:675` | `IpOptErrorConvCheck.cpp:63-69` | ✓ |
| C5 — `acceptable_tol = 1e-6` | `convergence.rs:106` | `IpOptErrorConvCheck.cpp:70-81` | ✓ |
| C6 — `acceptable_iter = 15` | `options.rs:773`, `convergence.rs:112` | `IpOptErrorConvCheck.cpp:82-89` | ✓ |
| C7 — `acceptable_dual_inf_tol = 1e10` | `convergence.rs:107` | `IpOptErrorConvCheck.cpp:90-97` | ✓ |
| C8 — `acceptable_constr_viol_tol = 1e-2` | `convergence.rs:108` | `IpOptErrorConvCheck.cpp:98-105` | ✓ |
| C9 — `acceptable_compl_inf_tol = 1e-2` | `convergence.rs:109` | `IpOptErrorConvCheck.cpp:106-113` | ✓ |
| C10 — `acceptable_obj_change_tol = 1e20` | `options.rs:772` | `IpOptErrorConvCheck.cpp:114-121` | ✓ |
| C11 — `diverging_iterates_tol = 1e20` | `options.rs:774` | `IpOptErrorConvCheck.cpp:122-128,255` | ✓ |
| C12 — `s_max = 100`; `s_d`/`s_c` formula | `convergence.rs:140-148` | `IpIpoptCalculatedQuantities.cpp:3677,3689,3078-3098` | ✓ |
| C13 — square-problem detection | `ipm.rs:1349`, `restoration.rs:571` | `IpRestoFilterConvCheck.cpp:155` | ✓ |
| C14 — `max_iter = 3000` | `options.rs:653` | `IpOptErrorConvCheck.cpp:27-32` | ✓ |
| C15 — `kappa_sigma = 1e10` multiplier reset | `ipm.rs:2884-2934` | `IpIpoptAlg.cpp:71-79` | ✓ |
| C16 — `max_cpu_time` deadline | `ipm.rs:8045-8061` | `IpOptErrorConvCheck.cpp:33-46` | ✓ |
| Convergence test (scaled max ≤ tol AND 3 unscaled component tests) | `convergence.rs:160-167` | `IpOptErrorConvCheck.cpp:226,240-260` | ✓ |

### Mu update / adaptive (M1–M7, A1–A16)

| Gate | Where | Ipopt ref | Status |
|------|-------|-----------|--------|
| M1 — `mu_linear_decrease_factor = 0.2` (κ_μ) | `options.rs:664` | `IpMonotoneMuUpdate.cpp` | ✓ |
| M2 — `mu_superlinear_decrease_power = 1.5` (θ_μ) | `options.rs:665` | same | ✓ |
| M3 — `barrier_tol_factor = 10` | `options.rs:701` | `IpMonotoneMuUpdate.cpp:139,144` | ✓ |
| M4 — monotone update with `mu_min` floor | `ipm.rs` `update_barrier_parameter` | `IpAdaptiveMuUpdate.cpp:329` | ✓ |
| M5 — `mu_init = 0.1` | `options.rs:654` | `IpMonotoneMuUpdate.cpp:42-48` | ✓ |
| M7 — `tau_min = 0.99` | `options.rs:663` | `IpMonotoneMuUpdate.cpp:86-94` | ✓ |
| A1 — Free→Fixed switch on insufficient progress | `ipm.rs:5121-5180` | `IpAdaptiveMuUpdate.cpp:347` | ✓ |
| A4 — `mu_max_fact = 1000` | `options.rs:656` | `IpAdaptiveMuUpdate.cpp:41-48` | ✓ |
| A6 — `mu_min = 1e-11` | `options.rs:655` | `IpAdaptiveMuUpdate.cpp:57-65` | ✓ |
| A8 — `barrier_tol_factor = 10` (E_μ test) | `options.rs:701` | `IpOptErrorConvCheck.cpp` | ✓ |
| A9 — `num_refs_max = 4` (KKT-error window) | `ipm.rs:986` | `IpAdaptiveMuUpdate.cpp:89-96` | ✓ |
| A12 — `adaptive_mu_restore_previous_iterate = false` | `options.rs:704` | `IpAdaptiveMuUpdate.cpp:126-132` | ✓ |
| A13 — `adaptive_mu_monotone_init_factor = 0.8` | `options.rs:703` | `IpAdaptiveMuUpdate.cpp:133-140` | ✓ |
| A16 — `mu_oracle_quality_function = true` | `options.rs:721` | `IpAlgBuilder.cpp:363-381` | ✓ |
| `mehrotra_pc = false` (default off) | `options.rs:713` | same | ✓ |

### Filter / line search (LS1–LS37 selected)

| Gate | Where | Ipopt ref | Status |
|------|-------|-----------|--------|
| LS1 — `theta_max_fact = 1e4` | `filter.rs:100` | `IpFilterLSAcceptor.cpp:42-51` | ✓ |
| LS2 — `theta_min_fact = 1e-4` | `filter.rs:99` | `IpFilterLSAcceptor.cpp:52-62` | ✓ |
| LS3 — `eta_phi = 1e-8` | `filter.rs:89` | `IpFilterLSAcceptor.cpp:63-70` | ✓ |
| LS4 — `delta = 1.0` | `filter.rs:90` | `IpFilterLSAcceptor.cpp:71-77` | ✓ |
| LS5 — `s_phi = 2.3` | `filter.rs:88` | `IpFilterLSAcceptor.cpp:78-84` | ✓ |
| LS6 — `s_theta = 1.1` | `filter.rs:87` | `IpFilterLSAcceptor.cpp:85-91` | ✓ |
| LS7 — `gamma_phi = 1e-8` | `filter.rs:85` | `IpFilterLSAcceptor.cpp:92-99` | ✓ |
| LS8 — `gamma_theta = 1e-5` | `filter.rs:84` | `IpFilterLSAcceptor.cpp:100-107` | ✓ |
| LS9 — `alpha_min_frac = 0.05` | `filter.rs:92` | `IpFilterLSAcceptor.cpp:108-115` | ✓ |
| LS10 — `max_soc = 4` | `options.rs:682` | `IpFilterLSAcceptor.cpp:116-122` | ✓ |
| LS11 — `compute_alpha_min` formula | `filter.rs:370-393` | `IpFilterLSAcceptor.cpp:450-469` | ✓ |
| LS12 — SOC `kappa_soc = 0.99` | `ipm.rs:7542` | `IpFilterLSAcceptor.cpp:123-131` | ✓ |
| LS13 — `obj_max_inc = 5.0` | `filter.rs:91` | `IpFilterLSAcceptor.cpp:132-139` | ✓ |
| LS14 — `max_filter_resets = 5` | `filter.rs:98` | `IpFilterLSAcceptor.cpp:141-150` | ✓ |
| LS15 — `filter_reset_trigger = 5` | `filter.rs:97` | `IpFilterLSAcceptor.cpp:151-159` | ✓ |
| LS21 — tiny-step 3-guard detection | `ipm.rs:4348-4402` | `IpBacktrackingLineSearch.cpp:410` | ✓ |
| LS22 — `alpha_red_factor = 0.5` (LS backtrack) | `ipm.rs:2721,2809` | `IpBacktrackingLineSearch.cpp:51-58` | ✓ |
| LS25 — `alpha_for_y` 7-mode dispatch | `ipm.rs:3212-3225` | `IpBacktrackingLineSearch.cpp:84-104,937-998` | ✓ |
| LS26 — `alpha_for_y_tol = 10.0` | `options.rs:719` | `IpBacktrackingLineSearch.cpp:98-104` | ✓ |
| LS27 — `tiny_step_tol ≈ 10·eps` | `ipm.rs:4325-4326` | `IpBacktrackingLineSearch.cpp:106-115` | ✓ |
| LS28 — `tiny_step_y_tol = 1e-2` | `options.rs:720` | `IpBacktrackingLineSearch.cpp:116-124` | ✓ |
| LS29 — `watchdog_shortened_iter_trigger = 10` | `options.rs:696` | `IpBacktrackingLineSearch.cpp:125-132` | ✓ |
| LS30 — `watchdog_trial_iter_max = 3` | `options.rs:699` | `IpBacktrackingLineSearch.cpp:133-139` | ✓ |
| LS33 — soft-restoration `pderror` factor 0.9999 | `ipm.rs:4171` | `soft_resto_pderror_reduction_factor` | ✓ |
| LS36 — `kappa_sigma = 1e10` | `ipm.rs:2884-2934` | `IpIpoptAlg.cpp:71-79` | ✓ |
| LS37 — `recalc_y = false`, `recalc_y_feas_tol = 1e-6` | `options.rs:716-717` | `IpIpoptAlg.cpp:80-95` | ✓ |
| `is_ftype` switching condition | `filter.rs:164-167` | `IpFilterLSAcceptor.cpp:362` | ✓ |
| Armijo `phi_trial ≤ phi + eta_phi·alpha·grad` | `filter.rs:195` | same | ✓ |
| Filter accept `θ·(1-γ_θ)`, `φ-γ_φ·θ` | `filter.rs:145-146` | same | ✓ |
| `kappa_d = 1e-5` (linear damping) | `options.rs:668` | `IpIpoptCalculatedQuantities.cpp:154-160` | ✓ |

### KKT regularization (P1–P10)

| Gate | Where | Ipopt ref | Status |
|------|-------|-----------|--------|
| P1-P4 — 4-state machine (DcEq0DxEq0 / DcGt0DxEq0 / DcEq0DxGt0 / DcGt0DxGt0) | `kkt.rs:710-880` | `IpPDPerturbationHandler.cpp:158-450` | ✓ |
| P5 — δ_x first-vs-subsequent inc factor | `kkt.rs:769-775` | `IpPDPerturbationHandler.cpp:386-393` | ✓ |
| P6 — δ_x cap wipes `delta_w_last` | `kkt.rs:780` | `IpPDPerturbationHandler.cpp:399-400` | ✓ |
| P7 — `consider_new_system` test_status reset | `kkt.rs:730-740` | `IpPDPerturbationHandler.cpp:188-202` | ✓ |
| P8 — `perturb_for_wrong_inertia` second-layer recovery | `kkt.rs:863-880` | `IpPDPerturbationHandler.cpp:435-448` | ✓ |
| P9 — `delta_c_base = 1e-8` (`jacobian_regularization_value`) | `kkt.rs:658` | `IpPDPerturbationHandler.cpp:82-87` | ✓ |
| P10 — `delta_c_exp = 0.25` (`jacobian_regularization_exponent`) | `kkt.rs:659` | `IpPDPerturbationHandler.cpp:88-94` | ✓ |
| `delta_w_init = 1e-4` (`first_hessian_perturbation`) | `kkt.rs:657` | `IpPDPerturbationHandler.cpp:75-81` | ✓ |
| `delta_w_inc_fact_first = 100.0` (`perturb_inc_fact_first`) | `kkt.rs:661` | `IpPDPerturbationHandler.cpp:49-57` | ✓ |
| `delta_w_inc_fact = 8.0` (`perturb_inc_fact`) | `kkt.rs:662` | `IpPDPerturbationHandler.cpp:58-65` | ✓ |
| `delta_w_dec_fact = 1/3` (`perturb_dec_fact`) | `kkt.rs:663` | `IpPDPerturbationHandler.cpp:66-74` | ✓ |
| `delta_w_max = 1e20` (`max_hessian_perturbation`) | `kkt.rs:664` | `IpPDPerturbationHandler.cpp:27-40` | ✓ |
| `delta_w_min = 1e-20` (`min_hessian_perturbation`) | `kkt.rs:665` | `IpPDPerturbationHandler.cpp:41-48` | ✓ |
| `perturb_always_cd = false` (default off) | `kkt.rs:666` | `IpPDPerturbationHandler.cpp:95-101` | ✓ |
| `residual_improvement_factor = 0.999999999` | `options.rs:752` | `IpPDFullSpaceSolver.cpp:72-79` | ✓ |
| `min_refinement_steps = 1` (IR1) | `options.rs:748` | `IpPDFullSpaceSolver.cpp:40-47` | ✓ |
| `max_refinement_steps = 10` (IR2) | `options.rs:749` | `IpPDFullSpaceSolver.cpp:48-54` | ✓ |
| `residual_ratio_max = 1e-10` (IR3) | `options.rs:750` | `IpPDFullSpaceSolver.cpp:55-62` | ✓ |
| `residual_ratio_singular = 1e-5` (IR4) | `options.rs:751` | `IpPDFullSpaceSolver.cpp:63-70` | ✓ |

### Restoration (R1–R16)

| Gate | Where | Ipopt ref | Status |
|------|-------|-----------|--------|
| R1 — `kappa_resto = 0.9` | `options.rs:707` | `IpRestoFilterConvCheck.cpp` | ✓ |
| R2 — `rho = 1000` slack penalty | `ipm.rs:8091` | `IpRestoIpoptNLP.cpp:56-62` | ✓ |
| R3 — `max_soft_resto_iters = 10` | `ipm.rs:4112` | `IpBacktrackingLineSearch.cpp:442-444` | ✓ |
| R4 — `resto_proximity_weight = 1.0` | `options.rs:708` | `IpRestoIpoptNLP.cpp:63-70` | ✓ |
| R6 — `bound_mult_reset_threshold = 1000` | `ipm.rs:7670,7715` | `IpRestoMinC_1Nrm.cpp:36-44` | ✓ |
| `restoration_max_iter = 200` | `options.rs:705` | same | ✓ |
| RestorationNlp objective `ρ·(Σp+Σn) + (η/2)·‖D_R(x-x_r)‖²` | `restoration_nlp.rs:9,220-248` | `IpRestoIpoptNLP.cpp` | ✓ |
| `D_R[i] = 1/max(1, |x_r[i]|)` | `restoration_nlp.rs:71` | same | ✓ |
| Closed-form `(p, n)` initial values | `restoration_nlp.rs:143-150` | same | ✓ |
| `η = resto_proximity_weight · √μ` | `restoration_nlp.rs:174` | `IpRestoIpoptNLP.cpp:759` | ✓ |

### Initialization / scaling (I1–I10, HA1–HA9)

| Gate | Where | Ipopt ref | Status |
|------|-------|-----------|--------|
| I1 — `slack_bound_push = 1e-2` | `options.rs:671` | `IpDefaultIterateInitializer.cpp:50-57` | ✓ |
| I2 — `slack_bound_frac = 1e-2` | `options.rs:672` | `IpDefaultIterateInitializer.cpp:58-66` | ✓ |
| I3 — `least_square_init_primal = no` (not exposed; not default) | n/a | `IpDefaultIterateInitializer.cpp:93-101` | ✓ |
| I4 — `least_square_init_duals = no` (not exposed; not default) | n/a | `IpDefaultIterateInitializer.cpp:102-111` | ✓ |
| I5 — fixed-variable treatment via preprocessor (MakeParameter semantics) | `preprocessing.rs:58`, `commit 645724c` | `IpTNLPAdapter.cpp:100-112` | ✓ † |
| I6 — `nlp_lower_bound_inf = -1e19` / `nlp_upper_bound_inf = 1e19` | `options.rs:688-689` | `IpTNLPAdapter.cpp:92-99` | ✓ |
| I7 — `constr_mult_init_max = 1000` clamp | `ipm.rs:8507`, `7793` | `IpDefaultIterateInitializer.cpp` | ✓ |
| I8 — `bound_mult_init_method = Constant`, val=1.0 | `options.rs:769-770` | same | ✓ |
| I9 — `bound_push = bound_frac = 1e-2` | `options.rs:669-670` | same | ✓ |
| I10 — `warm_start_init_point` (`warm_start = false` default) | `options.rs` | `IpDefaultIterateInitializer.cpp:113-120` | ✓ |
| `warm_start_bound_push = 1e-3` | `options.rs:684` | same | ✓ |
| Bound projection: `min(κ1·max(|b|,1), κ2·range)` two-sided, `κ1·max(|b|,1)` one-sided | `ipm.rs:8398-8412,5913-5934` | `IpDefaultIterateInitializer.cpp` | ✓ |
| `bound_relax_factor = 1e-8` | `options.rs:667` | `IpOrigIpoptNLP.cpp:53-64` | ✓ |
| HA1 — `hessian_approximation_lbfgs = false` (= exact) | `options.rs:712` | `IpOrigIpoptNLP.cpp:117-123` | ✓ |
| HA4 — `mu_strategy = monotone` (commit `fe111d4`) | `options.rs` | `IpAlgBuilder.cpp:356-362` | ✓ |
| HA5 — `nlp_scaling_method = gradient-based` | `options.rs:724` | `IpAlgBuilder.cpp:343-353` | ✓ |
| `nlp_scaling_max_gradient = 100.0` | `options.rs:726` | `IpGradientScaling.cpp:18-27` | ✓ |
| `obj_scaling_factor = 1.0` | `options.rs:725` | same | ✓ |

† I5 note: ripopt's `options.fixed_variable_treatment` enum still defaults
to `RelaxBounds` (`options.rs:771`), but the actual user-facing path is
`PreprocessedProblem::new_fixed_only` (`preprocessing.rs:58`) which
implements MakeParameter semantics — matching Ipopt's default. Worth
making the option default mirror the actual behavior for clarity.

## Not implemented / not exposed

These exist in Ipopt 3.14 but are not present in ripopt. None are
default-on; absence is consistent with shipping defaults:

| Gate | Ipopt default | Notes |
|------|---------------|-------|
| `expect_infeasible_problem` (LS31) | `no` | Heuristic to start restoration earlier on suspected infeasibility. |
| `expect_infeasible_problem_ctol` (LS32) | `1e-3` | Threshold below which EIP disabled. |
| `expect_infeasible_problem_ytol` (LS34) | `1e8` | EIP-on `‖y‖_inf` trigger for restoration. |
| `start_with_resto` (LS35) | `no` | Diagnostic mode: begin in restoration. |
| `accept_after_max_steps` (LS24) | `-1` | Accept any trial after N backtracks; -1 disables. |
| `accept_every_trial_step` (LS23) | `false` | Used by Mehrotra preset. |
| `corrector_type` (LS16-20) | `none` | Predictor-corrector flavors; ripopt's `mehrotra_pc` covers the standard variant. |
| `mehrotra_algorithm` preset | `no` | Bundle option that rewrites many defaults; not implemented. |
| `mu_target` | `0.0` | Shifts complementarity termination; ripopt assumes 0. |
| `honor_original_bounds` (R12) | `false` (in 3.14.x) | Project final x into pre-relaxation bounds; ripopt does not. |
| `constr_mult_reset_threshold` (R7) | `0.0` | Post-resto LS-y discard threshold; ripopt always uses constr_mult_init_max=1000. |
| `resto_failure_feasibility_threshold` (R8) | auto = `100·tol` | Resto exit flagged as infeasibility when constr_viol below this; ripopt uses different exit logic. |
| `resto.theta_max_fact` per-phase override (R9) | `1e8` (vs `1e4` regular) | Restoration's looser filter; ripopt reuses outer theta_max. |
| `resto.start_with_resto = no` forced (R10) | hard-coded | Recursion guard; ripopt achieves the same with `consecutive_restoration_failures`. |
| `adaptive_mu_kkterror_red_iters` (A9 separate from `num_refs_max`) | `4` | Combined into ripopt's window-of-4 logic. |
| `adaptive_mu_kkt_norm_type` (A14) | `2-norm-squared` | ripopt uses fixed norm choice. |
| `adaptive_mu_safeguard_factor` (A15) | `0.0` | Undocumented Ipopt safeguard; ripopt has no analog (matches default disabled). |
| `filter_margin_fact` (A10) / `filter_max_margin` (A11) | `1e-5` / `1.0` | Only used in `obj-constr-filter` adaptive globalization (not the default `kkt-error`). |
| `least_square_init_primal/duals` (I3, I4) | `no` | Not exposed; defaults match Ipopt off. |
| `slack_move` | `eps^0.75` | Implementation-internal; ripopt uses equivalent slack handling. |
| `point_perturbation_radius` | `10.0` | Used for FD center; not relevant for AD-based ripopt. |
| `neg_curv_test_tol` (IR6) | `0.0` | Inertia-free test toggle (Zavala/Chiang); ripopt uses inertia. |
| `check_derivatives_for_naninf` (HA8) | `false` | NaN-detection toggle; ripopt always checks. |
| `warm_start_same_structure` (HA9) | `false` | Cache-skip across solves; not a normal-mode option. |
| `limited_memory_aug_solver` (HA3) | `sherman-morrison` | Ripopt uses the extended augmented system unconditionally for L-BFGS. |
| `line_search_method` (HA6) | `filter` | Ripopt only implements filter line search. |
| `hessian_approximation_space` (HA2) | `nonlinear-variables` | Subspace for L-BFGS approx; ripopt uses full space. |

## Summary

Of ~90 gates audited (catalog produced by ipopt-expert):

- **2 latent bugs fixed** (A3 μ-collapse, A7 `refs_red_fact`).
- **~75 gates verified consistent** with Ipopt 3.14 defaults and behavior
  (numeric defaults, state machines, formula matches).
- **3 LS-init pivoting issues** documented as feral-upstream limitations.
- **~15 advanced/optional features not implemented**, all consistent
  with shipping Ipopt defaults (off / disabled / preset-specific).

The mu-collapse fix is verified to restore Ipopt-monotone behavior on
qcqp1500-1c. No other observable misalignments found. Ripopt v0.8 is
structurally aligned with Ipopt 3.14 to the level of detail this
checklist captures, modulo the documented LS-init/feral pivoting
limitations and the I5 enum-vs-actual cleanup.
