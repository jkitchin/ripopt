//! Integration tests for the feral-backed sparse linear solvers.
//!
//! Default-features build (feature = "feral") routes Direct/Iterative/Hybrid
//! to FeralLdl/FeralIterativeMinres/FeralHybrid; these tests exercise the
//! full LinearSolver trait surface (factor, solve, inertia, repeated factor
//! with cached symbolic) on small symmetric indefinite KKTs.

#![cfg(feature = "feral")]

use ripopt::linear_solver::feral_direct::FeralLdl;
use ripopt::linear_solver::{KktMatrix, LinearSolver, SparseSymmetricMatrix};

fn build_kkt_2x2_indef() -> KktMatrix {
    // [ 2  1 ]
    // [ 1 -3 ]
    // Eigenvalues: positive=1, negative=1.
    let mut m = SparseSymmetricMatrix::zeros(2);
    m.add(0, 0, 2.0);
    m.add(0, 1, 1.0);
    m.add(1, 1, -3.0);
    KktMatrix::Sparse(m)
}

#[test]
fn feral_direct_factor_and_solve_2x2_indef() {
    let mut solver = FeralLdl::new();
    let kkt = build_kkt_2x2_indef();

    let inertia = solver
        .factor(&kkt)
        .expect("factor must succeed")
        .expect("feral always reports inertia");
    assert_eq!(inertia.positive, 1);
    assert_eq!(inertia.negative, 1);
    assert_eq!(inertia.zero, 0);

    // RHS [3, -2] -> exact solution should satisfy the system.
    let rhs = vec![3.0, -2.0];
    let mut sol = vec![0.0; 2];
    solver.solve(&rhs, &mut sol).expect("solve must succeed");

    // Verify A x ≈ rhs by computing the residual through the matrix.
    let mut y = vec![0.0; 2];
    kkt.matvec(&sol, &mut y);
    for i in 0..2 {
        assert!((y[i] - rhs[i]).abs() < 1e-10, "residual at {} too large: {} vs {}", i, y[i], rhs[i]);
    }
}

#[test]
fn feral_direct_refactor_reuses_symbolic() {
    // Same sparsity pattern, different values. Second factor must succeed
    // using the cached symbolic factorization (exercising the COO->CSC
    // scatter path inside FeralLdl).
    let mut solver = FeralLdl::new();
    let kkt1 = build_kkt_2x2_indef();
    solver.factor(&kkt1).expect("first factor");

    let mut m = SparseSymmetricMatrix::zeros(2);
    m.add(0, 0, 5.0);
    m.add(0, 1, 0.5);
    m.add(1, 1, -1.0);
    let kkt2 = KktMatrix::Sparse(m);
    let inertia = solver.factor(&kkt2).expect("refactor").expect("inertia");
    assert_eq!(inertia.positive + inertia.negative, 2);

    let rhs = vec![1.0, 1.0];
    let mut sol = vec![0.0; 2];
    solver.solve(&rhs, &mut sol).expect("second solve");

    let mut y = vec![0.0; 2];
    kkt2.matvec(&sol, &mut y);
    for i in 0..2 {
        assert!((y[i] - rhs[i]).abs() < 1e-10);
    }
}

#[test]
fn feral_direct_provides_inertia_and_increase_quality() {
    let mut solver = FeralLdl::new();
    assert!(solver.provides_inertia());

    // increase_quality should escalate at least once (0.0 -> 0.01).
    assert!(solver.increase_quality(), "first escalation should succeed");
    // Subsequent calls should keep escalating until the cap (0.5).
    let mut count = 0;
    while solver.increase_quality() {
        count += 1;
        assert!(count < 50, "increase_quality should saturate");
    }
}
