//! Integration tests for the Thierry-Biegler ℓ₁-exact penalty-barrier
//! flag wired through `solve()`.
//!
//! Phase 1 (this file): verify (1) flag-off behavior is identical to the
//! unwrapped path on a well-posed problem, (2) flag-on solves the same
//! problem to the same optimum (slacks collapse to ≈ 0 at convergence),
//! and (3) the user-facing result reports user-space dimensions
//! (no augmented variables leaked).

use ripopt::{solve, NlpProblem, SolveStatus, SolverOptions};

/// min x0² + x1², s.t. x0 + x1 = 1.  Optimum: x = (0.5, 0.5), f = 0.5.
struct EqQp;
impl NlpProblem for EqQp {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 1 }
    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
    }
    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 1.0; g_u[0] = 1.0;
    }
    fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; x0[1] = 0.0; }
    fn objective(&self, x: &[f64], _: bool, obj: &mut f64) -> bool {
        *obj = x[0]*x[0] + x[1]*x[1]; true
    }
    fn gradient(&self, x: &[f64], _: bool, grad: &mut [f64]) -> bool {
        grad[0] = 2.0*x[0]; grad[1] = 2.0*x[1]; true
    }
    fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] + x[1]; true
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0], vec![0, 1])
    }
    fn jacobian_values(&self, _: &[f64], _: bool, vals: &mut [f64]) -> bool {
        vals[0] = 1.0; vals[1] = 1.0; true
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }
    fn hessian_values(&self, _: &[f64], _: bool, of: f64, _: &[f64], vals: &mut [f64]) -> bool {
        vals[0] = 2.0*of; vals[1] = 2.0*of; true
    }
}

fn quiet() -> SolverOptions {
    let mut o = SolverOptions::default();
    o.print_level = 0;
    o
}

#[test]
fn flag_off_is_default_path() {
    // With the flag off, calling solve() must follow the same path as
    // calling it on an NLP that has no notion of the flag — a sanity
    // check that the new branch is a no-op when disabled.
    let opts_default = quiet();
    let mut opts_explicit_off = quiet();
    opts_explicit_off.l1_exact_penalty_barrier = false;
    opts_explicit_off.l1_penalty_init = 12345.0; // ignored when flag is off

    let r1 = solve(&EqQp, &opts_default);
    let r2 = solve(&EqQp, &opts_explicit_off);

    assert_eq!(r1.status, SolveStatus::Optimal);
    assert_eq!(r2.status, SolveStatus::Optimal);
    // Iterations and objective must agree to demonstrate the path is identical.
    assert_eq!(r1.iterations, r2.iterations);
    assert!((r1.objective - r2.objective).abs() < 1e-12);
    assert_eq!(r1.x.len(), 2);
    assert_eq!(r2.x.len(), 2);
}

#[test]
fn flag_on_recovers_user_space_solution() {
    let mut opts = quiet();
    opts.l1_exact_penalty_barrier = true;
    opts.l1_penalty_init = 1000.0;

    let result = solve(&EqQp, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);

    // Result must report user-space dimensions: 2 variables, 1
    // constraint — no slacks leaked into x or constraint values.
    assert_eq!(result.x.len(), 2);
    assert_eq!(result.constraint_values.len(), 1);
    assert_eq!(result.bound_multipliers_lower.len(), 2);
    assert_eq!(result.bound_multipliers_upper.len(), 2);
    assert_eq!(result.constraint_multipliers.len(), 1);

    // Optimum: x ≈ (0.5, 0.5), f ≈ 0.5, c ≈ 1.
    assert!((result.x[0] - 0.5).abs() < 1e-6, "x0 = {}", result.x[0]);
    assert!((result.x[1] - 0.5).abs() < 1e-6, "x1 = {}", result.x[1]);
    assert!((result.objective - 0.5).abs() < 1e-6, "f = {}", result.objective);
    assert!((result.constraint_values[0] - 1.0).abs() < 1e-6,
            "c = {} (should equal user-space c, not augmented c−p+n)",
            result.constraint_values[0]);
}

/// Inconsistent equalities — `x = 1 AND x = 2` has no feasible point.
/// Vanilla ripopt declares `LocalInfeasibility` and stops at the
/// midpoint (x = 1.5). The ℓ₁ reformulation drives x to the
/// ℓ₁-optimal least-infeasible point: at x=1 the second constraint's
/// violation is 1 (penalty cost ρ·1) and f=1; at x=2 the first
/// constraint's violation is 1 (penalty cost ρ·1) and f=4. So the
/// ℓ₁ optimum is x=1.
///
/// **The phase-3 win**: a problem the unwrapped path gives up on at
/// x=1.5 is now driven to the ℓ₁-optimal x=1.0 — and the status is
/// honestly reported as `LocalInfeasibility` because the original
/// constraints are still violated there. The user gets a *better*
/// least-infeasible point with a *truthful* status.
#[test]
fn flag_on_finds_l1_optimum_on_inconsistent_equalities() {
    struct InconsistentEq;
    impl NlpProblem for InconsistentEq {
        fn num_variables(&self) -> usize { 1 }
        fn num_constraints(&self) -> usize { 2 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 1.0; g_u[0] = 1.0;
            g_l[1] = 2.0; g_u[1] = 2.0;
        }
        fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; }
        fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool { *o = x[0]*x[0]; true }
        fn gradient(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool { g[0] = 2.0*x[0]; true }
        fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool { g[0] = x[0]; g[1] = x[0]; true }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 1], vec![0, 0]) }
        fn jacobian_values(&self, _: &[f64], _: bool, v: &mut [f64]) -> bool { v[0]=1.0; v[1]=1.0; true }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
        fn hessian_values(&self, _: &[f64], _: bool, of: f64, _: &[f64], v: &mut [f64]) -> bool { v[0]=2.0*of; true }
    }

    // Baseline: unwrapped solve — must fail and stop near the
    // midpoint (x ≈ 1.5).
    let r_off = solve(&InconsistentEq, &quiet());
    assert!(
        matches!(r_off.status, SolveStatus::LocalInfeasibility | SolveStatus::Infeasible | SolveStatus::RestorationFailed),
        "regression: unwrapped path now solves an inconsistent system, status = {:?}", r_off.status
    );

    // ℓ₁-wrapped: status is honestly `LocalInfeasibility` (the
    // original constraints have no feasible point), but x is driven
    // to the ℓ₁-optimal least-infeasible point x ≈ 1.0 (closer to
    // the first equality, which the smaller f(x) selects).
    let mut opts = quiet();
    opts.l1_exact_penalty_barrier = true;
    opts.l1_penalty_init = 1000.0;
    let r_on = solve(&InconsistentEq, &opts);
    assert_eq!(
        r_on.status, SolveStatus::LocalInfeasibility,
        "ℓ₁ flag must report honest infeasibility status when slacks don't collapse"
    );
    assert!((r_on.x[0] - 1.0).abs() < 1e-3,
            "ℓ₁ optimum should be x ≈ 1.0 (smaller |x| beats x=2.0), got {}", r_on.x[0]);
    assert!((r_on.objective - 1.0).abs() < 1e-3,
            "f(x*) should be ≈ 1.0, got {}", r_on.objective);
    // The whole point of the wrapper: x_on must be a *better*
    // least-infeasible point than vanilla's x ≈ 1.5.
    assert!(
        (r_on.x[0] - 1.0).abs() < (r_off.x[0] - 1.0).abs(),
        "ℓ₁ x = {} must be closer to the ℓ₁ optimum than vanilla x = {}",
        r_on.x[0], r_off.x[0]
    );
}

/// Phase 2: feasible problems must not pay an outer-loop tax. The
/// dynamic-ρ loop should observe `Σ(p+n) ≤ l1_slack_tol` on the first
/// inner solve and break — no escalation, no extra IPM solves. We
/// verify this by checking that iteration count with `max_outer_iter
/// = 5` matches iteration count with `max_outer_iter = 1` on `EqQp`.
#[test]
fn phase2_no_outer_loop_tax_on_feasible_problems() {
    let mut opts1 = quiet();
    opts1.l1_exact_penalty_barrier = true;
    opts1.l1_penalty_max_outer_iter = 1;
    let r1 = solve(&EqQp, &opts1);

    let mut opts5 = quiet();
    opts5.l1_exact_penalty_barrier = true;
    opts5.l1_penalty_max_outer_iter = 5;
    let r5 = solve(&EqQp, &opts5);

    assert_eq!(r1.status, SolveStatus::Optimal);
    assert_eq!(r5.status, SolveStatus::Optimal);
    assert_eq!(
        r1.iterations, r5.iterations,
        "feasible problem must terminate after first inner solve regardless of max_outer_iter"
    );
}

/// Phase 3: an infeasible problem causes the outer loop to escalate ρ
/// up to `l1_penalty_max`. We assert that (a) the status is honestly
/// `LocalInfeasibility` (the original constraints cannot all be
/// satisfied) and the returned x is the ℓ₁-best least-infeasible
/// point, and (b) total iterations are bounded by `max_outer_iter *
/// inner_iter_budget` — i.e. the outer loop terminates rather than
/// spiralling.
#[test]
fn phase3_infeasible_reports_local_infeasibility_with_iter_cap() {
    struct InconsistentEq;
    impl NlpProblem for InconsistentEq {
        fn num_variables(&self) -> usize { 1 }
        fn num_constraints(&self) -> usize { 2 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 1.0; g_u[0] = 1.0;
            g_l[1] = 2.0; g_u[1] = 2.0;
        }
        fn initial_point(&self, x0: &mut [f64]) { x0[0] = 0.0; }
        fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool { *o = x[0]*x[0]; true }
        fn gradient(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool { g[0] = 2.0*x[0]; true }
        fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool { g[0] = x[0]; g[1] = x[0]; true }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0, 1], vec![0, 0]) }
        fn jacobian_values(&self, _: &[f64], _: bool, v: &mut [f64]) -> bool { v[0]=1.0; v[1]=1.0; true }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![0], vec![0]) }
        fn hessian_values(&self, _: &[f64], _: bool, of: f64, _: &[f64], v: &mut [f64]) -> bool { v[0]=2.0*of; true }
    }

    let max_outer = 4_usize;
    let inner_budget = 50_usize;
    let mut opts = quiet();
    opts.l1_exact_penalty_barrier = true;
    opts.l1_penalty_init = 1000.0;
    opts.l1_penalty_max = 1e8;
    opts.l1_penalty_increase_factor = 10.0;
    opts.l1_penalty_max_outer_iter = max_outer;
    opts.l1_slack_tol = 1e-6;
    opts.max_iter = inner_budget; // per inner solve

    let r = solve(&InconsistentEq, &opts);
    assert_eq!(
        r.status, SolveStatus::LocalInfeasibility,
        "phase 3 must report honest infeasibility on a structurally inconsistent system"
    );
    assert!((r.x[0] - 1.0).abs() < 1e-3, "x should converge near 1.0, got {}", r.x[0]);
    assert!(
        r.iterations <= max_outer * inner_budget,
        "total iters {} exceeds max_outer * inner_budget = {}",
        r.iterations, max_outer * inner_budget
    );
}

/// Phase 2: max_outer_iter = 1 must reproduce phase-1 (single static
/// ρ) behavior exactly. Regression check that the outer-loop refactor
/// didn't change the inner solve at the first ρ.
#[test]
fn phase2_max_outer_iter_1_matches_phase1() {
    let mut opts = quiet();
    opts.l1_exact_penalty_barrier = true;
    opts.l1_penalty_init = 1000.0;
    opts.l1_penalty_max_outer_iter = 1;

    let r = solve(&EqQp, &opts);
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.x[0] - 0.5).abs() < 1e-6);
    assert!((r.x[1] - 0.5).abs() < 1e-6);
}

/// **Phase 3 headline win on a *feasible* degenerate problem.** The
/// Burke-Han pattern: equality constraint `x₁² + x₂² = 0` with the
/// unique feasible point at the origin, where the constraint gradient
/// is identically zero — extreme LICQ failure. Vanilla ripopt limps to
/// `Acceptable` (the soft-stop status fired by repeated near-feasible
/// iterates), unable to reach `Optimal`. The ℓ₁ wrapper lifts this to
/// `Optimal` with substantially fewer iterations.
///
/// This is the issue-#23 acceptance criterion in test form: a
/// feasible problem where the unwrapped path *thrashes* (Acceptable,
/// many iterations) and the flag turns it into a clean `Optimal` solve.
#[test]
fn flag_on_lifts_burke_han_from_acceptable_to_optimal() {
    struct BurkeHan;
    impl NlpProblem for BurkeHan {
        fn num_variables(&self) -> usize { 2 }
        fn num_constraints(&self) -> usize { 1 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
            x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 0.0; g_u[0] = 0.0;
        }
        fn initial_point(&self, x0: &mut [f64]) { x0[0] = 1.0; x0[1] = 1.0; }
        fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool { *o = x[0]; true }
        fn gradient(&self, _x: &[f64], _: bool, g: &mut [f64]) -> bool {
            g[0] = 1.0; g[1] = 0.0; true
        }
        fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
            g[0] = x[0]*x[0] + x[1]*x[1]; true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 0], vec![0, 1])
        }
        fn jacobian_values(&self, x: &[f64], _: bool, v: &mut [f64]) -> bool {
            v[0] = 2.0*x[0]; v[1] = 2.0*x[1]; true
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1], vec![0, 1])
        }
        fn hessian_values(&self, _: &[f64], _: bool, _of: f64, lam: &[f64], v: &mut [f64]) -> bool {
            v[0] = 2.0 * lam[0]; v[1] = 2.0 * lam[0]; true
        }
    }

    // Vanilla path: stops at Acceptable with many iterations.
    let r_off = solve(&BurkeHan, &quiet());
    assert!(
        matches!(r_off.status, SolveStatus::Acceptable | SolveStatus::MaxIterations | SolveStatus::RestorationFailed | SolveStatus::NumericalError),
        "regression: vanilla now solves Burke-Han to {:?} — the test isn't degenerate enough anymore",
        r_off.status
    );

    // ℓ₁-wrapped: clean Optimal status with fewer iters.
    let mut opts = quiet();
    opts.l1_exact_penalty_barrier = true;
    opts.l1_penalty_init = 1000.0;
    let r_on = solve(&BurkeHan, &opts);

    assert_eq!(
        r_on.status, SolveStatus::Optimal,
        "ℓ₁ flag must turn the degenerate Burke-Han into a clean Optimal solve"
    );
    assert!(
        r_on.iterations < r_off.iterations,
        "ℓ₁ should be cheaper: {} iters vs vanilla's {} iters",
        r_on.iterations, r_off.iterations
    );
    // Both paths should land near the unique feasible point (0, 0).
    assert!(r_on.x[0].abs() < 1e-2, "x0 = {} (expected near 0)", r_on.x[0]);
    assert!(r_on.x[1].abs() < 1e-2, "x1 = {} (expected near 0)", r_on.x[1]);
    // Constraint x₁² + x₂² should be small (we're near the feasible point).
    let c = r_on.constraint_values[0];
    assert!(c.abs() < 1e-3, "c = {} should be small at the feasible point", c);
}

#[test]
fn flag_on_objective_excludes_penalty_term() {
    // The reported objective must be f(x*) only, without ρ·Σ(p+n).
    // For a feasible problem the slacks collapse to ~0 at the optimum,
    // so this test fails loud only when the wiring forgets to
    // re-evaluate the user objective on return.
    let mut opts = quiet();
    opts.l1_exact_penalty_barrier = true;
    // Use a very large rho — if the IPM's penalty contribution leaked
    // into the reported objective, the value would balloon.
    opts.l1_penalty_init = 1e8;

    let r = solve(&EqQp, &opts);
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!(r.objective < 1.0, "reported objective = {} — penalty term leaked", r.objective);
    assert!((r.objective - 0.5).abs() < 1e-4, "f = {}", r.objective);
}

/// Auto-fallback: when the standard solve hits `RestorationFailed` or
/// `LocalInfeasibility`, `l1_fallback_on_restoration_failure = true`
/// should silently retry with the wrapper. On a feasible-degenerate
/// problem (Burke-Han pattern) the retry must elevate the result to
/// Optimal; the original status is replaced.
#[test]
fn l1_fallback_promotes_restoration_failure_to_optimal() {
    // Burke-Han: x₁²+x₂² = 0 (zero gradient at the unique feasible
    // point), objective x₁+x₂. Vanilla either hits RestorationFailed
    // or stalls at Acceptable; the wrapper finds the clean Optimal.
    struct BurkeHan;
    impl NlpProblem for BurkeHan {
        fn num_variables(&self) -> usize { 2 }
        fn num_constraints(&self) -> usize { 1 }
        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = f64::NEG_INFINITY; x_u[0] = f64::INFINITY;
            x_l[1] = f64::NEG_INFINITY; x_u[1] = f64::INFINITY;
        }
        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 0.0; g_u[0] = 0.0;
        }
        fn initial_point(&self, x0: &mut [f64]) { x0[0] = 1.0; x0[1] = 1.0; }
        fn objective(&self, x: &[f64], _: bool, o: &mut f64) -> bool {
            *o = x[0] + x[1]; true
        }
        fn gradient(&self, _: &[f64], _: bool, g: &mut [f64]) -> bool {
            g[0] = 1.0; g[1] = 1.0; true
        }
        fn constraints(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
            g[0] = x[0]*x[0] + x[1]*x[1]; true
        }
        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 0], vec![0, 1])
        }
        fn jacobian_values(&self, x: &[f64], _: bool, v: &mut [f64]) -> bool {
            v[0] = 2.0*x[0]; v[1] = 2.0*x[1]; true
        }
        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1], vec![0, 1])
        }
        fn hessian_values(&self, _: &[f64], _: bool, _: f64, lam: &[f64], v: &mut [f64]) -> bool {
            v[0] = 2.0*lam[0]; v[1] = 2.0*lam[0]; true
        }
    }

    // Without fallback: status is whatever the vanilla solver returns
    // (Acceptable / RestorationFailed / LocalInfeasibility). With
    // fallback on, the auto-retry promotes it to Optimal.
    let mut opts = quiet();
    opts.l1_fallback_on_restoration_failure = true;
    let r = solve(&BurkeHan, &opts);
    assert_eq!(
        r.status, SolveStatus::Optimal,
        "fallback should promote degenerate-equality solve to Optimal, got {:?}",
        r.status
    );
    // x* = (0, 0), f* = 0.
    assert!(r.x[0].abs() < 1e-3, "x[0] = {}", r.x[0]);
    assert!(r.x[1].abs() < 1e-3, "x[1] = {}", r.x[1]);
}

/// Auto-fallback no-op: when the standard solve already returns
/// Optimal, the fallback must not be triggered (no extra iterations,
/// no different result).
#[test]
fn l1_fallback_does_not_disturb_clean_optimal_solves() {
    let mut opts_no_fb = quiet();
    let r_no_fb = solve(&EqQp, &opts_no_fb);
    opts_no_fb.l1_fallback_on_restoration_failure = true;
    let r_fb = solve(&EqQp, &opts_no_fb);
    assert_eq!(r_no_fb.status, SolveStatus::Optimal);
    assert_eq!(r_fb.status, SolveStatus::Optimal);
    assert_eq!(
        r_no_fb.iterations, r_fb.iterations,
        "fallback flag must be a no-op on clean Optimal solves"
    );
}
