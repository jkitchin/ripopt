# ripopt Development Guide

## Linear solver backends
- **Default (v0.8+)**: pure-Rust [`feral`](../feral) multifrontal LDLᵀ. The
  `default-features = ["feral", "faer"]`. Modules: `src/linear_solver/feral_direct.rs`,
  `feral_iterative.rs`, `feral_hybrid.rs`.
- **Legacy**: `rmumps` is preserved behind the opt-in `rmumps` feature.
  Build with `cargo build --no-default-features --features "rmumps faer"`.
  Modules: `src/linear_solver/multifrontal.rs`, `iterative.rs`, `hybrid.rs`.
- The `LinearSolver` trait surface is unchanged; the IPM, KKT, and
  restoration paths are oblivious to the backend.

## Running Tests
- `cargo test` — run all tests (~3 seconds)

## Benchmarks
- `make benchmark` — full benchmark: CUTEst + domain + large-scale + report
- `make benchmark-report` — regenerate report from existing results
- Individual CUTEst problems: `cargo run --bin cutest_suite --features cutest,ipopt-native --release -- PROBLEM1 PROBLEM2`
- Full CUTEst suite: `RESULTS_FILE=benchmarks/cutest/results.json cargo run --bin cutest_suite --features cutest,ipopt-native --release`

## Code Coverage
- Install: `cargo install cargo-llvm-cov` (requires llvm-tools-preview: `rustup component add llvm-tools-preview`)
- Summary: `cargo llvm-cov test` (runs tests with instrumentation and prints coverage)
- Line-by-line: `cargo llvm-cov test --text`
- HTML: `cargo llvm-cov test --html`
- After adding tests, update the coverage table in README.md with current numbers from `cargo llvm-cov test`

## Test Guidelines
- Every test must exercise a specific code branch (not just return true)
- Tests must verify correctness: check status codes, objective values, solution quality
- Prefer small hand-crafted problems over large benchmark problems
- Keep test execution under 1 second per test
- Unit tests (`#[cfg(test)] mod tests`) for module-internal functions
- Integration tests (`tests/`) for cross-module behavior and solver paths

## Honesty in Benchmarks and Tests
**No misleading benchmarks or problem-specific hacks.** The following are explicitly prohibited:
- Counting `NumericalError`, `MaxIterations`, or any non-`Optimal` status as a "solve" in benchmark summaries
- Writing tests that accept failure statuses (e.g., `|| NumericalError`) just to make the pass rate look better
- Tuning solver parameters specifically for individual benchmark problems to inflate scores
- Adding special-case code paths triggered only by specific problem structures seen in benchmarks
- Hiding known failures behind lenient statuses (`Acceptable` was removed for this reason)

**Tests must be honest:** If the solver cannot solve a problem to `Optimal`, the test should either fail (exposing the real limitation), be marked `#[ignore]` with a clear explanation, or be removed. A failing test that documents a real limitation is more valuable than a passing test that hides one.

### Alignment-work exception: regressions are expected and acceptable
During the v0.8 Ipopt-alignment effort (`docs/V0.8_IPOPT_ALIGNMENT_PLAN.md`), the
goal is **correctness of the implementation against the Ipopt 3.14 reference**,
not preservation of benchmark pass-rates. When a planned heuristic deletion
regresses a benchmark suite, that regression is the expected, honest signal
that the deleted heuristic was a benchmark-tuning crutch — exactly the thing
this effort is meant to retire.

**Do not** revert correct alignment changes just because a benchmark count
drops. **Do not** silently re-document a deleted heuristic as "load-bearing"
to keep it alive. If a deletion is correct against the Ipopt reference, ship
it and record the regression in the plan's evidence column. Re-anchoring
(§3.2) is reserved for cases where the heuristic is actually present in Ipopt
(or a documented ripopt-specific kernel difference) and ripopt was just
misplacing it — not for "this heuristic happens to fix N benchmark problems".

Pass rates are downstream metrics we recover by fixing root causes
(convergence test, restoration, scaling), not by keeping post-hoc promotions.

## Working on the solver: efficiency rules

These are rules distilled from sessions that burned time on avoidable mistakes. Follow them for any non-trivial change to `src/ipm.rs`, `src/restoration*.rs`, `src/filter.rs`, `src/convergence.rs`, or `src/kkt.rs`.

### Principled changes
Every set changes should have a testable hypothesis that is backed by an expert opinion. Each set gets its own commit-or-revert decision, measured in isolation. Document the results of failures for future reference to avoid repeating this.


### Read the function before changing its callsite
Numerical code often has a constant whose meaning is set by the callee, not the caller. Before editing a call like `RestorationNlp::new(..., rho, 1.0)`, read `RestorationNlp::new` to understand what the `1.0` becomes. Don't rename or rescale based on the caller-side appearance alone.

### Distinguish verified from inferred
When answering a question about solver behavior, lead with one of: "measured:", "inferred from X:", or "unknown — would need Y". Do not present `Δf ≈ 0.5·H·Δx²` as evidence that x values differ — it's an analytical estimate, not a measurement. If the user asks "did you confirm?", you should be able to point at the measurement.

### Narrow the test target
If the fix is for CONCON, run CONCON. If it's for 9 problems in Category A, run those 9. Only run the full regression suite (48 problems, ~15 min) after the narrow target is green. Full-suite runs per iteration burn ~15 min each and make iteration loops unworkable.

### Bash interrupt does not undo Edit
If you have applied an Edit and the user interrupts a subsequent Bash call, the file on disk is still mutated. Explicitly revert the Edit before changing direction — don't assume "the patch is aborted". Verify with `git diff` if uncertain.

### Don't chain unrelated commands with `&&`
`cargo build && cargo test && cargo run -- benchmark | tail -3` hides intermediate failures: if tests fail silently or the output format changes, you see only the last step. Run build / test / benchmark as separate calls, capture output to known files, and inspect them explicitly. Piping to `tail -3` is especially dangerous because success summaries are often >3 lines.

### One parameterized probe, not N probe files
If you need to diagnose the same thing on 5 problems, write one binary that takes the problem name as a CLI arg. Five probe files (`vesuvio_probe.rs`, `concon_probe.rs`, ...) each trigger a fresh ~30s cargo build; one parameterized probe builds once and runs fast.

### Don't retry a broken external build
If `ipopt-sys` / `ipopt-native` / any external crate fails with a specific cmake or linker error, diagnose the failure before rerunning. Running the same broken configuration twice wastes ~10 min each. Check `brew list`, pkg-config paths, and the build.rs source before trying flags.

### Use memory for durable findings
Facts like "CONCON at iter 48 reaches KKT with pr=du=0 but compl stays at mu=2e-5" or "the LS-y residual only helps when m≤n or J has full rank" are exactly what the memory system exists for. Save them the moment you learn them; future sessions save days.

## Benchmark Versioning
After each release, save tagged benchmark results so we can track improvement and regression across versions. Run `make benchmark` and copy the results:
```
cp benchmarks/BENCHMARK_REPORT.json benchmarks/BENCHMARK_REPORT_vX.Y.Z.json
cp benchmarks/cutest/results.json    benchmarks/cutest/results_vX.Y.Z.json
```
This enables per-problem timing comparisons between versions (e.g. "did problem 12 get faster?") and catches regressions that aggregate pass rates miss.

### GAMS nlpbench reports
The GAMS nlpbench harness (vendored clone at `gams/nlpbench/`, `.gitignore`d) is driven via targets in `gams/Makefile`:
```
make -C gams bench-smoke     # ~10 problems, TIMELIMIT=30
make -C gams bench-small     # 50 problems, TIMELIMIT=60
make -C gams bench-medium    # 50 problems, TIMELIMIT=300
make -C gams bench-large     # 50 problems, TIMELIMIT=900
make -C gams bench-all       # all four
```
Each target runs both `ripopt` and `ipopt`, then writes `gams/nlpbench/BENCHMARK_REPORT_<size>_<version>.md`. Version is read from `Cargo.toml`; override with `make -C gams bench-small VERSION=v0.7.1`. The reports, the report generator, the testset generator, and the curated testsets all live inside the gitignored `gams/nlpbench/` tree because nlpbench is GAMS-licensed and we don't ship any of its derivatives. Testset lists: `gams/nlpbench/testsets/*.gms`, regenerable via `make -C gams testsets`.

<!-- crucible-project -->
## Crucible Knowledge Base

This project has a [Crucible](https://github.com/jkitchin/crucible) knowledge base in `.crucible/`.
Use the `crucible` CLI to ingest sources, search, and maintain the wiki.

Layout: `.crucible/sources/` (primary sources), `.crucible/wiki/` (distilled articles),
`.crucible/crucible.db` (graph database).

Conventions: org-mode with scimax, org-ref citations, narrative prose.
The LLM maintains the wiki; manual edits are the exception.
Run `crucible help all` for the full CLI reference.
<!-- crucible-project -->
