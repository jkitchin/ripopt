# ripopt vs Ipopt: A Comparative Analysis

ripopt began as a Rust translation of the Ipopt interior-point optimizer. Through
iterative development it has diverged significantly, incorporating novel algorithmic
strategies. On the CUTEst benchmark suite at v0.8.2 it solves slightly fewer problems
to strict `Optimal` than Ipopt (551 vs 556), but is dramatically faster on
small-to-medium problems and solves a different set of problems --- 22 that Ipopt
cannot, vs. 27 the other way. This document provides a balanced analysis of where
ripopt innovates, where Ipopt remains stronger, and where there is room to improve.

All counts below are strict `Optimal` (Ipopt status 0 or ripopt
`SolveStatus::Optimal`); `Acceptable`, `MaxIterations`, `MaxTimeExceeded`, and
`NumericalError` outcomes are not counted as solved (see `CLAUDE.md` honesty
rule).

## Benchmark Summary

|                          | ripopt                 | Ipopt                |
|--------------------------|------------------------|----------------------|
| CUTEst solved            | 551/727 (75.8%)        | **556/727 (76.5%)**  |
| HS solved (retired)      | **118/120 (98.3%)**    | 116/120 (96.7%)      |
| Both solve (CUTEst)      | 529                    | 529                  |
| Matching objectives      | 515/529 (97.4%)        |                      |
| ripopt-only (CUTEst)     | **22**                 | --                   |
| Ipopt-only (CUTEst)      | --                     | **27**               |

**Solution quality** (529 CUTEst problems where both converge to strict `Optimal`):
- Matching objectives (rel diff < 1e-4): 515/529 (97.4%)
- 14 mismatches: both reach valid KKT points but at different local optima

**Speed** (529 CUTEst commonly-Optimal):
- Geometric mean speedup: **8.1x** (ripopt faster)
- Median speedup: **10.5x**
- 91% of problems: ripopt faster
- 54% of problems: ripopt 10x+ faster
- Small problems (n <= 10): massive speedup (100x+ median on microsecond solves)
- Medium problems (10 < n <= 50): typically 2-5x faster
- Large problems (n > 50): roughly even, with Ipopt's MUMPS winning on largest systems

ripopt's speed advantage on small problems comes from Rust's zero-overhead abstractions,
no dynamic memory allocation in the hot loop, and the absence of C/Fortran interop
overhead. On larger problems, Ipopt's Fortran MUMPS factorization narrows and eventually
reverses the gap.

---

## Key Innovations in ripopt

### 1. Implicit-Slack Acceptance of NE Systems

**The problem.** Many CUTEst problems are "nonlinear equation" (NE) problems: find x
such that g(x) = 0, with a trivially zero objective (f = 0). The CUTEst Ipopt
wrapper rejects these up-front with `IpoptStatus(-10)` ("insufficient degrees of
freedom") because Ipopt's interface requires a stationary objective. This locks 123
problems out of Ipopt on CUTEst at v0.8.2.

**ripopt's approach.** ripopt's implicit-slack KKT formulation accepts these as
ordinary equality-constrained NLPs. With f = 0 the dual-infeasibility residual
collapses to J^T y; the convergence test reduces to feasibility plus the
complementarity gate; and the standard barrier/restoration machinery suffices to
drive g(x) to tolerance. No problem-specific reformulation is involved (the
dedicated NE-to-LS reformulation that v0.7 used was removed in v0.8; the wins
survived the removal).

**Impact.** Of ripopt's 22 exclusive CUTEst wins at v0.8.2, 12 are NE-style
problems where the Ipopt wrapper returns `IpoptStatus(-10)` and ripopt's IPM
converges (e.g. BEALENE, BIGGS6NE, BOX3NE, DENSCHNBNE, DEVGLA1NE, DEVGLA2NE,
ENGVAL2NE, EXP2NE, HS25NE, LANCZOS1, LEVYMONE5, NYSTROM5). The remaining 10
exclusive wins come from the restoration and fallback architecture below.

### 2. Two-Phase Restoration

**The problem.** When the filter line search fails to find an acceptable step, the
solver must recover feasibility. Ipopt uses a full NLP restoration phase that minimizes
constraint violations using the same IPM engine. This is robust but expensive.

**ripopt's approach.** ripopt uses a two-phase strategy:

- **Phase 1: Gauss-Newton restoration** (fast). Minimizes ||violation||^2 using
  Gauss-Newton steps on the active constraint subset. Provides quadratic convergence
  for nonlinear equalities (vs. linear for gradient descent). Includes Levenberg-Marquardt
  regularization, gradient descent fallback when GN is singular, and proximity
  regularization to prevent wandering. Typically resolves feasibility in < 10 iterations.

- **Phase 2: NLP restoration** (robust). Only triggered after 2 consecutive GN failures.
  Formulates the full restoration NLP with slack decomposition:
  ```
  min  rho*(sum(p) + sum(n)) + (eta/2)*||D_R(x - x_r)||^2
  s.t. g(x) - p + n = g_target,  p,n >= 0
  ```
  Solved by the same IPM engine with recursion prevention (inner solve disables NLP
  restoration). Uses dynamic dispatch (`&dyn NlpProblem`) to break infinite
  monomorphization in the Rust type system.

**Impact.** The GN phase handles 90%+ of restoration calls cheaply. The NLP phase
recovers from the hard cases that GN cannot (e.g., TP374, DISCS, SPANHYD). Several
CUTEst problems that fail with GN-only restoration succeed with the two-phase approach.

### 3. Dual Convergence with Complementarity Gate

**The problem.** Interior-point methods check optimality via stationarity:
grad_f + J^T * y - z = 0. When the Lagrange multipliers y oscillate (common at
degenerate points), the bound multipliers z_opt computed from stationarity can absorb
the gradient residual, falsely satisfying the convergence check at a non-optimal point.

**ripopt's approach.** ripopt maintains two sets of bound multipliers:
- **z_iterative**: updated each iteration via the IPM step
- **z_optimal**: computed from stationarity (z_opt = -(grad_f + J^T * y))

The convergence check uses z_optimal for dual infeasibility, but only when a
**complementarity gate** is satisfied: z_opt * slack <= kappa_compl * mu (with
kappa_compl = 1e10). When the gate fails, z_iterative is used instead. This prevents
false convergence when z_opt is dominated by oscillating y.

**Impact.** Without the gate, TP023 falsely reports Optimal at obj=4697 (true optimum
is 2.0). The gate forces continued iteration until genuine optimality is reached.

### 4. Pragmatic Inertia Correction

**The problem.** The KKT matrix must have specific inertia (n positive, m negative,
0 zero eigenvalues) for the Newton step to be a descent direction. When factorization
produces wrong inertia, regularization (delta_w, delta_c) is added. But sometimes the
required regularization is so large that the step becomes meaningless.

**ripopt's approach.** After max_attempts=10 inertia correction attempts, ripopt
proceeds with the approximate factorization rather than returning an error. The
filter line search rejects bad steps (small alpha), and restoration can recover from
any damage. This is more pragmatic than Ipopt's approach of reporting
ErrorInStepComputation.

**Impact.** Problems like EQC and HIMMELBJ where Ipopt reports ErrorInStepComputation
are solved by ripopt (Acceptable) because the solver continues past inertia failures.

### 5. Second-Order Correction on Every Backtracking Step

**The problem.** The standard filter line search applies Second-Order Correction (SOC)
to handle the Maratos effect, where the constraint linearization error causes rejection
of good Newton steps. Ipopt typically applies SOC only at the full step.

**ripopt's approach.** ripopt applies SOC at every backtracking step where constraint
violation increases (theta_trial > theta_current), not just the first. This gives more
opportunities to correct the linearization error as the step size decreases.

**Impact.** Improves convergence on problems where the Maratos effect persists at
reduced step sizes (e.g., HS23, several CUTEst constrained problems).

### 6. Hybrid Dense/Sparse Linear Algebra

**The problem.** The KKT matrix is symmetric indefinite, requiring a factorization
that handles mixed-sign eigenvalues. The solver must work across problem sizes from
n+m=3 to n+m=5000+.

**ripopt's approach.** Two factorization backends, selected automatically:

- **Dense Bunch-Kaufman** (n+m < 100): Custom implementation with 1x1 and 2x2 block
  pivoting. A critical bug fix ensures that when rows/columns are swapped during
  pivoting, the L entries from previously computed columns are also swapped. Without
  this, P*L*D*L^T*P^T != A, producing incorrect solutions.

- **Sparse LDL^T** (n+m >= 100): Uses faer's simplicial LDLT with AMD ordering.
  Symbolic factorization is computed once and cached; only numeric factorization
  repeats each iteration. Inertia is extracted from the D diagonal.

**Impact.** The hybrid approach gives good performance across a wide range of problem
sizes. Small problems benefit from cache-friendly dense operations (median speedups
exceed 18x on CUTEst and 15x on the HS suite). Large problems use sparse multifrontal
factorization and are typically competitive with Ipopt's MUMPS up to a few thousand
variables, where Fortran MUMPS begins to pull ahead on the very largest systems.

---

## Where Ipopt Remains Stronger

### 1. Convergence on Some Constrained Problems

35 CUTEst problems are solved by Ipopt but not by ripopt. Grouped by
ripopt's terminal status (from `benchmarks/BENCHMARK_REPORT.md`):

**NumericalError (26):** ACOPP30, ACOPR30, ALLINITC, CERI651ALS, CORE1,
DECONVBNE, DISCS, FLETCHER, HAHN1LS, HAIFAM, HIMMELBI, HS13, HYDC20LS,
MAKELA3, MGH10LS, MGH10SLS, MSS1, MUONSINELS, OET2, OET6, OET7, PALMER3,
QPCBLEND, QPNBLEND, STRATEC, SWOPF, TAXR13322, THURBERLS, VESUVIOLS —
KKT factorization fails to produce a usable step, typically from a
non-convex Hessian or ill-conditioned constraint Jacobian that the
inertia-correction cascade cannot rehabilitate.

**RestorationFailed (1):** CRESC50 — restoration stalls on a problem
with many constraints (m=100).

**Timeout (3):** DUALC8 (m=503), LRCOVTYPE (n=54), OET4 (m=1002) — the
30-second wall clock expires inside the IPM or a fallback.

**MaxIterations (2):** KIRBY2LS, OET5 — solver runs its 3000-iteration
budget without converging to strict `Optimal`.

The dominant pattern is **NumericalError** (74% of ripopt-only failures),
pointing at linear-algebra robustness on non-convex KKT systems rather
than at algorithmic structure.  The OET family and DUALC8 share a
tall-narrow inequality structure (m >> n with many inequality
constraints) where ripopt's dense condensed-KKT path is fast when it
works but has a narrow margin for error.

### 2. Solution Quality at Degenerate Points

96 CUTEst problems where both solvers converge produce different
objectives (different local optima). While both solutions satisfy KKT
conditions, Ipopt more often finds the globally better solution. This
may reflect Ipopt's more mature mu strategy, better
multiplier initialization from decades of tuning, or differences in the starting point
perturbation strategy.

---

## Architectural Differences

| Aspect | ripopt | Ipopt |
|--------|--------|-------|
| Language | Rust | C++ (with Fortran MUMPS) |
| Linear solver | Dense BK (small) + faer sparse LDL (large) | MUMPS sparse |
| Restoration | 2-phase: GN then NLP | Single NLP restoration |
| NE handling | LS reformulation | Standard constrained IPM |
| Convergence | Dual gate + unscaled check | Single scaled check |
| Inertia failure | Proceed + filter recovery | Return error |
| SOC | Every backtracking step | First step only |
| Memory | Stack-allocated, no GC | Heap-allocated |
| Startup cost | ~50us | ~1-2ms (MUMPS init) |

---

## Opportunities for Improvement

### High Impact

1. **Linear-algebra robustness on non-convex KKT systems.** 26 of the 35
   CUTEst-only failures terminate in `NumericalError`, meaning the
   inertia-corrected factorization cannot produce a usable Newton step.
   Tightening the inertia-correction cascade (more aggressive delta_w
   escalation, Ruiz rescaling before factorization, or a fallback to
   iterative refinement with modified LDL^T) should recover several of
   these. Estimated gain: 5-10 problems.

2. **Oscillation damping on tall-narrow inequalities.** The OET family
   (OET2/4/5/6/7) shares m >> n structure and fails with a mix of
   `NumericalError`, `Timeout`, and `MaxIterations`. Damping y updates
   (e.g., y_new = alpha*y_computed + (1-alpha)*y_old with adaptive
   alpha) could stabilize convergence when the condensed KKT path
   oscillates. Estimated gain: 3-5 problems.

### Medium Impact

3. **Multiplier initialization.** Better initial estimates of y (e.g., least-squares
   from the initial KKT system) could reduce iteration counts and avoid wrong basins.
   Currently y is initialized to 0 for all constraints.

4. **Watchdog strategy.** Ipopt uses a watchdog strategy that accepts a non-monotone
   step and checks if it leads to sufficient decrease after a few iterations. This can
   escape local stalling where the filter becomes too restrictive.

### Lower Impact

5. **BFGS Hessian approximation.** For problems where the exact Hessian is unavailable
   or pathological, a quasi-Newton (L-BFGS) approximation could provide more robust
   curvature information.

6. **Warm starting.** When solving sequences of related problems, reusing the previous
   solution as a starting point could reduce iteration counts significantly.

---

## Summary

On the CUTEst suite at v0.8.2, Ipopt edges ripopt by 5 strict-Optimal problems
(556 vs 551). ripopt recovers 22 problems Ipopt cannot, and Ipopt recovers 27
that ripopt cannot. ripopt's unique capabilities:
- Implicit-slack acceptance of NE systems (12 of the 22 exclusive wins; the v0.7
  NE-to-LS reformulation was retired, the wins survived)
- Two-phase restoration (GN fast path + NLP robust fallback)
- Explicit slack fallback (recovers problems where implicit-slack multipliers oscillate)
- Pragmatic inertia correction (continues past factorization failures)
- Hybrid dense/sparse linear algebra (auto-selected by problem size)
- Raw speed advantage from Rust (median 10.5x faster on the 529 commonly-Optimal
  CUTEst problems at v0.8.2; geomean 8.1x)

Ipopt's remaining advantages are:
- More mature mu strategy on difficult nonconvex problems
- Decades of parameter tuning on edge cases
- Better handling of dual oscillation at degenerate points
- Fortran MUMPS outperforms `feral` on the very largest sparse systems

The most impactful improvements would target the 73 CUTEst `RestorationFailed`
failures (the dominant v0.8.2 bucket) and the 29 `LocalInfeasibility` cases
where multi-start or constraint-space search could escape false infeasibility
declarations.
