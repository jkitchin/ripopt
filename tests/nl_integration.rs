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

// ---------------------------------------------------------------------------
// T-MIT-D: S segment ingestion — `scaling_factor` suffix on
// objectives / variables / constraints.
// ---------------------------------------------------------------------------
//
// The S-segment header is `S<flags> <count> <name>` where
//   flags & 3 = kind (0=variable, 1=constraint, 2=objective, 3=problem)
//   flags & 4 = float bit (0=int, 4=float)
// followed by `<count>` lines of `<index> <value>`.
#[test]
fn nl_parse_scaling_factor_suffix() {
    // 2 vars, 1 equality constraint x0 + x1 = 1, linear objective x0,
    // with three `scaling_factor` suffix tables: variable / constraint /
    // objective. flags=4 marks the values as float.
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
S4 2 scaling_factor
0 2.5
1 4.0
S5 1 scaling_factor
0 7.5
S6 1 scaling_factor
0 0.5
";
    let data = parse_nl_file(nl).expect("parse failed");
    assert_eq!(data.suffixes.len(), 3, "expected 3 suffix tables");
    let sf = data.scaling_factors();
    assert_eq!(sf.x.as_deref(), Some(&[2.5, 4.0][..]));
    assert_eq!(sf.g.as_deref(), Some(&[7.5][..]));
    assert_eq!(sf.obj, Some(0.5));
}

/// Suffixes whose name is not `scaling_factor` are still captured but
/// must NOT contribute to `scaling_factors()` extraction.
#[test]
fn nl_parse_unrelated_suffix_ignored_by_scaling_factors() {
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
S0 2 ipopt_zL_in
0 1.0
1 1.0
";
    let data = parse_nl_file(nl).expect("parse failed");
    assert_eq!(data.suffixes.len(), 1);
    assert_eq!(data.suffixes[0].name, "ipopt_zL_in");
    let sf = data.scaling_factors();
    assert!(sf.is_empty(),
        "non-scaling_factor suffix must not populate scaling_factors()");
}

/// `NlScalingFactors::is_empty` correctly distinguishes "no suffixes"
/// from "suffix present".
#[test]
fn nl_scaling_factors_is_empty_default() {
    let sf = ripopt::nl::NlScalingFactors::default();
    assert!(sf.is_empty());
}

// ---------------------------------------------------------------------------
// JSON output (--output FILE.json) — issue #27
// ---------------------------------------------------------------------------

const ROSENBROCK_NL: &str = "\
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

#[test]
fn cli_writes_validated_json_report() {
    // End-to-end check that `ripopt -o out.json prob.nl` produces a JSON
    // report whose validation block reflects the optimum we know analytically.
    let _guard = env_lock().lock().unwrap();

    let tmp = std::env::temp_dir().join(format!(
        "ripopt_issue27_{}_{}.nl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&tmp, ROSENBROCK_NL).expect("write nl");
    let json_path = tmp.with_extension("json");

    let exe = env!("CARGO_BIN_EXE_ripopt");
    let out = std::process::Command::new(exe)
        .arg(tmp.to_str().unwrap())
        .arg("-o")
        .arg(json_path.to_str().unwrap())
        .arg("print_level=0")
        .output()
        .expect("ripopt failed to spawn");
    assert!(
        out.status.success(),
        "ripopt exited with {:?}\nstderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );

    let bytes = std::fs::read(&json_path).expect("read json");
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("parse json");

    assert_eq!(v["solver"]["name"], "ripopt");
    assert_eq!(v["status"], "Optimal");
    assert_eq!(v["problem"]["n_variables"], 2);
    assert_eq!(v["problem"]["n_constraints"], 0);

    let max_bound_v = v["validation"]["max_bound_violation"].as_f64().unwrap();
    let max_constr_v = v["validation"]["max_constraint_violation"].as_f64().unwrap();
    let stat = v["validation"]["stationarity_inf_norm"].as_f64().unwrap();
    let kkt_ok = v["validation"]["kkt_satisfied"].as_bool().unwrap();
    assert_eq!(max_bound_v, 0.0, "no bounds in this problem");
    assert_eq!(max_constr_v, 0.0, "no constraints in this problem");
    assert!(
        stat < 1e-4,
        "stationarity should be small at Rosenbrock minimum, got {}",
        stat
    );
    assert!(kkt_ok, "kkt_satisfied should be true at the optimum");

    // Command line is captured.
    let cmd = v["command"].as_array().expect("command array");
    assert!(cmd.iter().any(|s| s.as_str() == Some("-o")));

    // Options are echoed back so the run can be reproduced.
    assert_eq!(v["options"]["print_level"], 0);

    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&json_path);
}

#[test]
fn cli_default_still_writes_sol() {
    // No -o/--output: must preserve the legacy behavior of writing
    // <stem>.sol next to the input.
    let _guard = env_lock().lock().unwrap();

    let tmp = std::env::temp_dir().join(format!(
        "ripopt_issue27_default_{}_{}.nl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&tmp, ROSENBROCK_NL).expect("write nl");
    let sol_path = tmp.with_extension("sol");
    let _ = std::fs::remove_file(&sol_path);

    let exe = env!("CARGO_BIN_EXE_ripopt");
    let out = std::process::Command::new(exe)
        .arg(tmp.to_str().unwrap())
        .arg("print_level=0")
        .output()
        .expect("ripopt failed to spawn");
    assert!(out.status.success());
    assert!(sol_path.exists(), "default run should produce {sol_path:?}");

    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&sol_path);
}
