use ripopt::nl::{parse_nl_file, NlProblem, write_sol};
use ripopt::{NlpProblem, SolveResult, SolveStatus, SolverOptions};

/// Shared lock for tests that mutate process-global env vars (AMPLFUNC).
/// Tests that would otherwise race for that state take this lock first.
fn env_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

/// Helper: solve an NlProblem with default options (quiet).
fn solve_nl(problem: &NlProblem) -> SolveResult {
    let mut opts = SolverOptions::default();
    opts.print_level = 0;
    ripopt::solve(problem, &opts)
}

// ---------------------------------------------------------------------------
// 1. Parse unconstrained quadratic: min x0^2 + x1^2
// ---------------------------------------------------------------------------
#[test]
fn nl_parse_unconstrained_quadratic() {
    let nl = "\
g3 1 1 0
 2 0 1 0 0
 0 1
 0 0
 2 2 0
 0 0 0 0 0
 0 0 0 0 0
 0 2
 0 0
 0 0 0 0 0
O0 0
o0
o5
v0
n2
o5
v1
n2
b
3
3
G0 2
0 0
1 0
x2
0 0
1 0
";
    let data = parse_nl_file(nl).expect("parse failed");
    assert_eq!(data.header.n_vars, 2);
    assert_eq!(data.header.n_constrs, 0);
    // Free variables: bounds should be -inf / +inf
    assert!(data.x_l[0].is_infinite() && data.x_l[0] < 0.0);
    assert!(data.x_u[0].is_infinite() && data.x_u[0] > 0.0);
    assert!(data.x_l[1].is_infinite() && data.x_l[1] < 0.0);
    assert!(data.x_u[1].is_infinite() && data.x_u[1] > 0.0);
}

// ---------------------------------------------------------------------------
// 2. Parse constrained equality: min x0  s.t.  x0 + x1 = 1
// ---------------------------------------------------------------------------
#[test]
fn nl_parse_constrained_equality() {
    // 2 vars, 1 constraint (equality), 1 objective (linear)
    // Constraint is purely linear: x0 + x1 = 1
    // Objective is purely linear: x0
    let nl = "\
g3 1 1 0
 2 1 1 0 1
 0 1
 0 0
 0 0 0
 0 0 0 0 0
 0 0 0 0 0
 2 1
 0 0
 0 0 0 0 0
O0 0
n0
C0
n0
b
3
3
G0 1
0 1
J0 2
0 1
1 1
k1
1
r
4 1
x2
0 0.5
1 0.5
";
    let data = parse_nl_file(nl).expect("parse failed");
    assert_eq!(data.header.n_vars, 2);
    assert_eq!(data.header.n_constrs, 1);
    assert_eq!(data.header.n_eqns, 1);
    // Equality constraint: g_l == g_u == 1.0
    assert_eq!(data.g_l[0], 1.0);
    assert_eq!(data.g_u[0], 1.0);
    // Linear Jacobian entries for constraint 0
    assert!(!data.con_linear[0].is_empty());
}

// ---------------------------------------------------------------------------
// 3. All variable bound types (codes 0-5 in b segment)
// ---------------------------------------------------------------------------
#[test]
fn nl_parse_all_bound_types() {
    // 6 variables, no constraints, trivial objective (n0)
    let nl = "\
g3 1 1 0
 6 0 1 0 0
 0 1
 0 0
 0 0 0
 0 0 0 0 0
 0 0 0 0 0
 0 0
 0 0
 0 0 0 0 0
O0 0
n0
b
0 -1 1
1 5
2 -3
3
4 7
5
";
    let data = parse_nl_file(nl).expect("parse failed");
    assert_eq!(data.header.n_vars, 6);
    // type 0: range [-1, 1]
    assert_eq!(data.x_l[0], -1.0);
    assert_eq!(data.x_u[0], 1.0);
    // type 1: upper bound only (-inf, 5]
    assert!(data.x_l[1].is_infinite() && data.x_l[1] < 0.0);
    assert_eq!(data.x_u[1], 5.0);
    // type 2: lower bound only [-3, +inf)
    assert_eq!(data.x_l[2], -3.0);
    assert!(data.x_u[2].is_infinite() && data.x_u[2] > 0.0);
    // type 3: free (-inf, +inf)
    assert!(data.x_l[3].is_infinite() && data.x_l[3] < 0.0);
    assert!(data.x_u[3].is_infinite() && data.x_u[3] > 0.0);
    // type 4: fixed at 7
    assert_eq!(data.x_l[4], 7.0);
    assert_eq!(data.x_u[4], 7.0);
    // type 5: complementarity (treated as free or special)
    // Just check it parsed without error
}

// ---------------------------------------------------------------------------
// 4. All constraint bound types (r segment)
// ---------------------------------------------------------------------------
#[test]
fn nl_parse_all_constraint_bound_types() {
    // 1 variable, 6 constraints, each with a different r-type
    let nl = "\
g3 1 1 0
 1 6 1 0 1
 0 1
 0 0
 0 0 0
 0 0 0 0 0
 0 0 0 0 0
 6 1
 0 0
 0 0 0 0 0
O0 0
n0
C0
n0
C1
n0
C2
n0
C3
n0
C4
n0
C5
n0
b
3
J0 1
0 1
J1 1
0 1
J2 1
0 1
J3 1
0 1
J4 1
0 1
J5 1
0 1
k0
r
0 -2 2
1 10
2 -5
3
4 3
5
G0 1
0 0
x1
0 0
";
    let data = parse_nl_file(nl).expect("parse failed");
    assert_eq!(data.header.n_constrs, 6);
    // type 0: range [-2, 2]
    assert_eq!(data.g_l[0], -2.0);
    assert_eq!(data.g_u[0], 2.0);
    // type 1: upper bound only (-inf, 10]
    assert!(data.g_l[1].is_infinite() && data.g_l[1] < 0.0);
    assert_eq!(data.g_u[1], 10.0);
    // type 2: lower bound only [-5, +inf)
    assert_eq!(data.g_l[2], -5.0);
    assert!(data.g_u[2].is_infinite() && data.g_u[2] > 0.0);
    // type 3: free
    assert!(data.g_l[3].is_infinite() && data.g_l[3] < 0.0);
    assert!(data.g_u[3].is_infinite() && data.g_u[3] > 0.0);
    // type 4: equality at 3
    assert_eq!(data.g_l[4], 3.0);
    assert_eq!(data.g_u[4], 3.0);
    // type 5: complementarity
}

// ---------------------------------------------------------------------------
// 5. Nonlinear expression evaluation (add, mul, pow, exp, sqrt)
// ---------------------------------------------------------------------------
#[test]
fn nl_parse_nonlinear_expression() {
    // Objective: sqrt(exp(x0)) + x1^2 * x0
    // = o0( o39(o44(v0)), o2(o5(v1,n2), v0) )
    // At x=(1,2): sqrt(e^1) + 4*1 = sqrt(e) + 4 ≈ 1.6487 + 4 = 5.6487
    let nl = "\
g3 1 1 0
 2 0 1 0 0
 0 1
 0 0
 2 2 0
 0 0 0 0 0
 0 0 0 0 0
 0 2
 0 0
 0 0 0 0 0
O0 0
o0
o39
o44
v0
o2
o5
v1
n2
v0
b
3
3
G0 2
0 0
1 0
x2
0 1
1 2
";
    let data = parse_nl_file(nl).expect("parse failed");
    let problem = NlProblem::from_nl_data(data).expect("build failed");

    // Evaluate objective at initial point x=(1,2)
    let x = vec![1.0, 2.0];
    let mut f = 0.0; problem.objective(&x, true, &mut f);
    let expected = (1.0_f64).exp().sqrt() + 4.0 * 1.0;
    assert!(
        (f - expected).abs() < 1e-10,
        "f={f}, expected={expected}"
    );
}

// ---------------------------------------------------------------------------
// 6. N-ary SUMLIST (opcode 54) with 3 arguments
// ---------------------------------------------------------------------------
#[test]
fn nl_parse_nary_sum() {
    // Objective: SUMLIST(v0, v1, v2) = x0 + x1 + x2
    // At x=(1,2,3): 6
    let nl = "\
g3 1 1 0
 3 0 1 0 0
 0 1
 0 0
 3 3 0
 0 0 0 0 0
 0 0 0 0 0
 0 3
 0 0
 0 0 0 0 0
O0 0
o54
3
v0
v1
v2
b
3
3
3
G0 3
0 0
1 0
2 0
x3
0 1
1 2
2 3
";
    let data = parse_nl_file(nl).expect("parse failed");
    let problem = NlProblem::from_nl_data(data).expect("build failed");

    let x = vec![1.0, 2.0, 3.0];
    let mut f = 0.0; problem.objective(&x, true, &mut f);
    assert!(
        (f - 6.0).abs() < 1e-10,
        "SUMLIST at (1,2,3) = {f}, expected 6.0"
    );
}

// ---------------------------------------------------------------------------
// 7. Full pipeline: parse Rosenbrock NL → NlProblem → solve
// ---------------------------------------------------------------------------
#[test]
fn nl_problem_solve_rosenbrock() {
    // Rosenbrock: min (1-x0)^2 + 100*(x1 - x0^2)^2
    // = o0( o5(o1(n1,v0),n2), o2(n100, o5(o1(v1, o5(v0,n2)), n2)) )
    let nl = "\
g3 1 1 0
 2 0 1 0 0
 0 1
 0 0
 2 2 0
 0 0 0 0 0
 0 0 0 0 0
 0 2
 0 0
 0 0 0 0 0
O0 0
o0
o5
o1
n1
v0
n2
o2
n100
o5
o1
v1
o5
v0
n2
n2
b
3
3
G0 2
0 0
1 0
x2
0 -1.2
1 1
";
    let data = parse_nl_file(nl).expect("parse failed");
    let problem = NlProblem::from_nl_data(data).expect("build failed");
    let result = solve_nl(&problem);

    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "Rosenbrock should converge to Optimal, got {:?}",
        result.status
    );
    assert!(
        (result.x[0] - 1.0).abs() < 1e-4,
        "x0={}, expected ~1.0",
        result.x[0]
    );
    assert!(
        (result.x[1] - 1.0).abs() < 1e-4,
        "x1={}, expected ~1.0",
        result.x[1]
    );
    assert!(
        result.objective.abs() < 1e-6,
        "obj={}, expected ~0",
        result.objective
    );
}

#[test]
fn nl_auxiliary_preprocessing_gate_fixture_matches_fallback() {
    let solve_fixture = |enable_preprocessing: bool| {
        let nl = include_str!("fixtures/issue_10/auxiliary_gate.nl");
        let data = parse_nl_file(nl).expect("parse issue-10 fixture");
        let problem = NlProblem::from_nl_data(data).expect("build issue-10 fixture");
        let options = SolverOptions {
            print_level: 0,
            enable_preprocessing,
            enable_al_fallback: false,
            enable_sqp_fallback: false,
            early_stall_timeout: 0.0,
            tol: 1e-8,
            ..SolverOptions::default()
        };
        let result = ripopt::solve(&problem, &options);
        (problem, result)
    };

    let (pre_problem, preprocessed) = solve_fixture(true);
    let (fallback_problem, fallback) = solve_fixture(false);

    assert_eq!(
        preprocessed.status,
        SolveStatus::Optimal,
        "preprocessed fixture solve should be Optimal, got {:?}",
        preprocessed.status
    );
    assert_eq!(
        fallback.status,
        SolveStatus::Optimal,
        "fallback fixture solve should be Optimal, got {:?}",
        fallback.status
    );

    let pre_violation = max_constraint_violation(&pre_problem, &preprocessed.x);
    let fallback_violation = max_constraint_violation(&fallback_problem, &fallback.x);
    assert!(
        pre_violation <= fallback_violation.max(1e-8) * 10.0 + 1e-8,
        "preprocessed violation {pre_violation} should not be worse than fallback {fallback_violation}"
    );

    let scale = preprocessed
        .objective
        .abs()
        .max(fallback.objective.abs())
        .max(1.0);
    assert!(
        preprocessed.objective <= fallback.objective + 1e-6 * scale,
        "preprocessed objective {} should not be worse than fallback {}",
        preprocessed.objective,
        fallback.objective
    );
    assert!(
        preprocessed.iterations <= fallback.iterations + 5,
        "preprocessed iterations {} should stay near fallback {}",
        preprocessed.iterations,
        fallback.iterations
    );
}

struct Issue23Fixture {
    name: &'static str,
    nl: &'static str,
}

#[derive(Debug)]
struct Issue23Metrics {
    status: SolveStatus,
    objective: f64,
    constraint_violation: f64,
    fallback: Option<String>,
    iterations: usize,
}

fn solve_issue23_fixture(
    fixture: &Issue23Fixture,
    enable_preprocessing: bool,
) -> (NlProblem, SolveResult) {
    let data = parse_nl_file(fixture.nl).expect("parse issue-23 fixture");
    let problem = NlProblem::from_nl_data(data).expect("build issue-23 fixture");
    let options = SolverOptions {
        print_level: 0,
        enable_preprocessing,
        early_stall_timeout: 0.0,
        max_iter: 500,
        tol: 1e-8,
        ..SolverOptions::default()
    };
    let result = ripopt::solve(&problem, &options);
    (problem, result)
}

fn issue23_metrics(problem: &NlProblem, result: &SolveResult) -> Issue23Metrics {
    Issue23Metrics {
        status: result.status,
        objective: result.objective,
        constraint_violation: max_constraint_violation(problem, &result.x),
        fallback: result.diagnostics.fallback_used.clone(),
        iterations: result.iterations,
    }
}

#[test]
fn nl_issue_23_executable_incidence_fixtures_compare_preprocessing() {
    let fixtures = [
        Issue23Fixture {
            name: "tutorial_flow_density",
            nl: include_str!("fixtures/issue_23/tutorial_flow_density.nl"),
        },
        Issue23Fixture {
            name: "tutorial_flow_density_perturbed",
            nl: include_str!("fixtures/issue_23/tutorial_flow_density_perturbed.nl"),
        },
    ];

    for fixture in fixtures {
        let (pre_problem, pre_result) = solve_issue23_fixture(&fixture, true);
        let (fallback_problem, fallback_result) = solve_issue23_fixture(&fixture, false);
        let preprocessed = issue23_metrics(&pre_problem, &pre_result);
        let fallback = issue23_metrics(&fallback_problem, &fallback_result);

        eprintln!(
            "issue 23 {name}: preprocessing={preprocessed:?}, no_preprocessing={fallback:?}",
            name = fixture.name
        );

        assert_eq!(
            preprocessed.status,
            SolveStatus::Optimal,
            "{} preprocessing should solve",
            fixture.name
        );
        assert_eq!(
            fallback.status,
            SolveStatus::Optimal,
            "{} no-preprocessing path should solve",
            fixture.name
        );
        assert!(
            preprocessed.constraint_violation <= 1e-8,
            "{} preprocessing full-space violation too large: {}",
            fixture.name,
            preprocessed.constraint_violation
        );
        assert!(
            fallback.constraint_violation <= 1e-8,
            "{} no-preprocessing full-space violation too large: {}",
            fixture.name,
            fallback.constraint_violation
        );
        assert_eq!(
            preprocessed.fallback.as_deref(),
            None,
            "{} auxiliary preprocessing should not fall back",
            fixture.name
        );
        assert!(
            preprocessed.iterations <= fallback.iterations,
            "{} preprocessing iterations {} should not exceed no-preprocessing {}",
            fixture.name,
            preprocessed.iterations,
            fallback.iterations
        );
        assert!(
            (preprocessed.objective - fallback.objective).abs() <= 1e-8,
            "{} objective mismatch: preprocessing {} vs no-preprocessing {}",
            fixture.name,
            preprocessed.objective,
            fallback.objective
        );
    }
}

fn max_constraint_violation<P: NlpProblem>(problem: &P, x: &[f64]) -> f64 {
    let m = problem.num_constraints();
    if m == 0 {
        return 0.0;
    }

    let mut g = vec![0.0; m];
    if !problem.constraints(x, true, &mut g) {
        return f64::INFINITY;
    }
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    problem.constraint_bounds(&mut g_l, &mut g_u);

    let mut violation: f64 = 0.0;
    for i in 0..m {
        let row = if !g[i].is_finite() {
            f64::INFINITY
        } else if g_l[i].is_finite()
            && g_u[i].is_finite()
            && (g_u[i] - g_l[i]).abs() <= 1e-12
        {
            (g[i] - 0.5 * (g_l[i] + g_u[i])).abs()
        } else {
            let lower = if g_l[i].is_finite() {
                (g_l[i] - g[i]).max(0.0)
            } else {
                0.0
            };
            let upper = if g_u[i].is_finite() {
                (g[i] - g_u[i]).max(0.0)
            } else {
                0.0
            };
            lower.max(upper)
        };
        violation = violation.max(row);
    }
    violation
}

// ---------------------------------------------------------------------------
// 8. SOL writer format check
// ---------------------------------------------------------------------------
#[test]
fn nl_sol_writer() {
    let result = SolveResult {
        x: vec![1.0, 2.0],
        objective: 5.0,
        constraint_multipliers: vec![0.5],
        bound_multipliers_lower: vec![0.0, 0.0],
        bound_multipliers_upper: vec![0.0, 0.0],
        constraint_values: vec![3.0],
        status: SolveStatus::Optimal,
        iterations: 10,
        diagnostics: Default::default(),
    };

    let mut buf: Vec<u8> = Vec::new();
    write_sol(&mut buf, &result, 2, 1).expect("write_sol failed");
    let output = String::from_utf8(buf).expect("not utf8");

    // Check message line contains status
    assert!(
        output.contains("Optimal Solution Found"),
        "SOL should contain status message"
    );
    // Check solve code for Optimal is 0
    assert!(
        output.contains("objno 0 0"),
        "SOL should have objno line with code 0"
    );
    // Check it contains the Options header
    assert!(output.contains("Options"), "SOL should contain Options section");
    // Check variable count lines
    assert!(
        output.contains("\n2\n"),
        "SOL should contain variable count"
    );
    assert!(
        output.contains("\n1\n"),
        "SOL should contain constraint count"
    );
}

// ---------------------------------------------------------------------------
// External (AMPL imported) functions (issue #15)
// ---------------------------------------------------------------------------

/// Parser must accept NL files with F segments and `f<id> <nargs>` expressions
/// without raising "Unknown expression token". Solve-time construction must
/// reject the problem with a clear, named error referring to the function.
#[test]
fn nl_parse_external_function_reports_clean_error() {
    // Problem:
    //   minimize myfunc(x0)
    //   s.t.     x0 >= 0
    // Header carries nfunc=1 on dim line 4 (field 1). The F0 segment declares
    // `myfunc` as a real-valued (type 0) one-argument function. The objective
    // uses the `f0 1` call with a single argument v0.
    let nl = "\
g3 1 1 0
 1 0 1 0 0
 0 1
 0 0
 1 0 1
 0 1 0 0
 0 0 0 0 0
 0 1
 0 0
 0 0 0 0 0
F0 0 1 myfunc
O0 0
f0 1
v0
b
2 0.0
k0
G0 1
0 0
x1
0 1
";
    let data = parse_nl_file(nl).expect("parse should succeed with f/F tokens");
    assert_eq!(data.imported_funcs.len(), 1);
    assert_eq!(data.imported_funcs[0].id, 0);
    assert_eq!(data.imported_funcs[0].name, "myfunc");
    assert_eq!(data.header.n_funcs, 1);

    let err = NlProblem::from_nl_data(data)
        .err()
        .expect("from_nl_data should reject external functions");
    assert!(
        err.contains("myfunc") && err.contains("external function"),
        "error should name the function and mention external functions, got: {err}"
    );
}

/// Regression fixture: the real `.nl` file produced by the IDAES Helmholtz
/// example in issue #15 (CMarcher). Before this patch the parser failed with
/// `Unknown expression token: 'f0 4'`; now the parser must accept all three
/// `F`-segment declarations and the `f<id> <nargs>` calls, and construction
/// must reject the problem with a clear, named error.
#[test]
fn nl_parse_idaes_helmholtz_fixture() {
    let nl = include_str!("fixtures/issue_15/idaes_helmholtz.nl");
    let data = parse_nl_file(nl).expect("IDAES fixture should parse without Unknown token error");

    // Header carries three imported functions (see dim line 4 of the fixture).
    assert_eq!(data.header.n_funcs, 3, "expected nfunc=3");
    assert_eq!(data.imported_funcs.len(), 3);
    let names: Vec<String> = data
        .imported_funcs
        .iter()
        .map(|f| f.name.clone())
        .collect();
    assert!(names.iter().any(|n| n == "vf_hp"), "expected vf_hp in {names:?}");
    assert!(names.iter().any(|n| n == "h_liq_hp"), "expected h_liq_hp in {names:?}");
    assert!(names.iter().any(|n| n == "h_vap_hp"), "expected h_vap_hp in {names:?}");

    // Exercise the "no AMPLFUNC" path. Take the env lock so this doesn't
    // race with nl_build_idaes_helmholtz_with_amplfunc, and clear AMPLFUNC
    // for the duration of the call.
    let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
    let prev = std::env::var_os("AMPLFUNC");
    std::env::remove_var("AMPLFUNC");
    let err_result = NlProblem::from_nl_data(data);
    if let Some(v) = prev {
        std::env::set_var("AMPLFUNC", v);
    }

    let err = err_result
        .err()
        .expect("IDAES Helmholtz problem must be rejected when AMPLFUNC is unset");
    assert!(
        err.contains("external function"),
        "error should mention external functions, got: {err}"
    );
    assert!(
        names.iter().any(|n| err.contains(n)),
        "error should name one of the imported functions, got: {err}"
    );
}

/// With `AMPLFUNC` pointing at the IDAES Helmholtz dylib, the IDAES fixture
/// should actually build into an `NlProblem` — that means tape-level
/// `Funcall` nodes were resolved against the loaded library. Skip when the
/// dylib isn't installed locally; this isn't a ripopt bug.
#[test]
fn nl_build_idaes_helmholtz_with_amplfunc() {
    let home = match std::env::var_os("HOME") {
        Some(h) => std::path::PathBuf::from(h),
        None => return,
    };
    let dylib = home.join(".idaes/bin/general_helmholtz_external.dylib");
    if !dylib.exists() {
        eprintln!("skipping: {} not installed", dylib.display());
        return;
    }

    let nl = include_str!("fixtures/issue_15/idaes_helmholtz.nl");
    let data = parse_nl_file(nl).expect("parse");

    // Safety: env is process-global. This test stomps AMPLFUNC for its run;
    // we restore it at the end so neighbouring tests aren't affected. Take the
    // env lock so this doesn't race with nl_parse_idaes_helmholtz_fixture.
    let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
    let prev = std::env::var_os("AMPLFUNC");
    // SAFETY: `std::env::set_var` is only unsafe in the edition-2024 sense;
    // in this crate's 2021 edition it's a stable safe function.
    std::env::set_var("AMPLFUNC", dylib.as_os_str());
    let result = NlProblem::from_nl_data(data);
    match prev {
        Some(v) => std::env::set_var("AMPLFUNC", v),
        None => std::env::remove_var("AMPLFUNC"),
    }

    let _problem = result.expect("from_nl_data should succeed with AMPLFUNC set");
}

/// When the same `f<id>` token appears in a constraint, the error should
/// still surface — not a parse failure on the token.
#[test]
fn nl_parse_external_function_in_constraint() {
    // Problem:
    //   minimize x0
    //   s.t.     g(x0) == 0   with g(.) = myfunc(.)
    let nl = "\
g3 1 1 0
 1 1 1 0 1
 1 1
 0 0
 1 1 1
 0 1 0 0
 0 0 0 0 0
 1 1
 0 0
 0 0 0 0 0
F0 0 1 myfunc
C0
f0 1
v0
O0 0
n0
r
4 0
b
3
k0
0
J0 1
0 1
G0 1
0 1
x1
0 1
";
    let data = parse_nl_file(nl).expect("parse should succeed");
    let err = NlProblem::from_nl_data(data).err().expect("should reject");
    assert!(
        err.contains("myfunc"),
        "error should name the function, got: {err}"
    );
}
