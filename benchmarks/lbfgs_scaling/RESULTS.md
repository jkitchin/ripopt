# L-BFGS scaling: empirical evidence for issue #30 task 4

Issue #30 predicted that ripopt's L-BFGS path scaled like O(n²) per
iteration because `update_lbfgs_hessian` materialised the full
n*(n+1)/2 dense lower triangle of B_k on every iteration. Task 4
replaced that with the diagonal-only σI fill, delegating the
rank-2k V/U correction to `LowRankKktSolver`.

This harness measures the actual gap. Both runs use identical code
*except* for task 4 itself — same harness binary, same problems, same
options. Times are wall-clock on macOS / Apple Silicon, release build.

## Methodology

Two unconstrained, scalable, sparse-Hessian problems:

- **ARWHEAD** (Conn-Gould-Toint): `f(x) = Σ_{i<n} ((x_i² + x_n²)² - 4x_i + 3)`, min at `(1,…,1,0)`, `f* = 0`.
- **GENROSE** (Toint, generalized Rosenbrock): `f(x) = 1 + Σ_{i≥2} (100(x_i - x_{i-1}²)² + (1 - x_i)²)`, min at `(1,…,1)`, `f* = 1`.

Options: `hessian_approximation_lbfgs = true`, `max_iter = 300`,
defaults otherwise. Status codes and iteration counts match between
pre- and post-task-4 (same algorithm; only the Hessian
representation changed), so a per-iteration time comparison is fair.

- Pre-task-4 commit: `43b2832` (phase 4 bordered solver, dense fill still in place)
- Post-task-4 commit: `6bb4a02` (this commit — diagonal-only fill + mandatory wrapper)

## Results

### ARWHEAD — time per iteration

| N    | pre (s/iter) | post (s/iter) | speedup |
|-----:|-------------:|--------------:|--------:|
|  500 |     0.01011  |    0.000267   |    38×  |
| 1000 |     0.04544  |    0.000500   |    91×  |
| 2000 |     0.26668  |    0.003205   |    83×  |
| 5000 |     2.93387  |    0.003322   |   883×  |

### GENROSE — time per iteration

| N    | pre (s/iter) | post (s/iter) | speedup |
|-----:|-------------:|--------------:|--------:|
|  500 |     0.00912  |    0.000251   |    36×  |
| 1000 |     0.04675  |    0.000523   |    89×  |
| 2000 |     0.26739  |    0.001073   |   249×  |
| 5000 |     3.20789  |    0.002745   |  1168×  |

### GENROSE — total wall time at N=5000

| commit       | wall    | iters | status        | obj      |
|--------------|---------|-------|---------------|----------|
| pre-task-4   | 959.2 s |   299 | MaxIterations | 2439     |
| post-task-4  |   0.82 s |   299 | MaxIterations | 2441     |

## Scaling

Pre-task-4 per-iter cost roughly **quadratic in n**: doubling n from
1000→2000 raises time/iter by ~5.7×; doubling 2000→5000 (×2.5) raises
it by ~11×. Consistent with the n*(n+1)/2 entries that `form_dense_bk`
allocated and `assemble_kkt_from_state` walked each iteration.

Post-task-4 per-iter cost is **sub-linear-to-linear in n**: 500→5000
(×10) raises ARWHEAD time/iter by ~12× and GENROSE by ~11×. Consistent
with the n diagonal entries plus the O(n·k) V/U back-solves.

## Convergence parity (sanity)

- Iteration counts match between pre and post for every (problem, N) cell.
- Status codes match (Optimal/Acceptable/MaxIterations).
- Final objectives match to expected numerical agreement (GENROSE objectives
  differ in the 4th significant figure, consistent with floating-point
  reordering of the SMW-corrected KKT solve vs. the dense factor).

Conclusion: task 4 preserves convergence behaviour and delivers the
asymptotic improvement issue #30 anticipated. At N=5000 the GENROSE
case goes from ~16 minutes to under a second per solve — a regime where
the dense path was already unusable.

## Reproducing

```
cargo build --release --bin lbfgs_scaling
./target/release/lbfgs_scaling ARWHEAD 5000 --json --max-iter 300
./target/release/lbfgs_scaling GENROSE 5000 --json --max-iter 300
```

Sweep, post-task-4:

```
for N in 500 1000 2000 5000; do
  for P in ARWHEAD GENROSE; do
    ./target/release/lbfgs_scaling $P $N --json --max-iter 300
  done
done | tee benchmarks/lbfgs_scaling/results/post_task4.jsonl
```

Then `git checkout 43b2832~0` (parent of task-4) and rerun against
`pre_task4.jsonl` to reproduce the pre-task-4 column.
