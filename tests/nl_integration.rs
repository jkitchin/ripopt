use ripopt::nl::{parse_nl_file, NlProblem, write_sol};
use ripopt::{NlpProblem, SolveResult, SolveStatus, SolverOptions};

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
    let problem = NlProblem::from_nl_data(data);

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
    let problem = NlProblem::from_nl_data(data);

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
    let problem = NlProblem::from_nl_data(data);
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
