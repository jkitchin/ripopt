# ripopt Release Checklist

A reusable checklist for cutting a new ripopt release. Copy this file when
starting a release and check items off as you go. The order matters: bench
first → docs → versions → verify interfaces → manuscript → tag → publish.

Replace `vX.Y.Z` with the new version number throughout (e.g. `v0.6.2`).

---

## 1. Pre-release verification (clean tree)

- [ ] `git status` — working tree understood (no surprise files)
- [ ] On the correct branch (usually `main`)
- [ ] Pulled latest from origin
- [ ] `cargo check --workspace --all-targets` — no errors
- [ ] `cargo check --examples` — every example compiles (catches stale trait sigs)
- [ ] `cargo build --release` — release build clean, no **new** warnings
  introduced since the previous release (a handful of pre-existing
  `unused_variables`/`unused_imports` warnings in `src/preprocessing.rs` and
  `src/c_api.rs` are tracked separately; the release gate is that the
  warning count has not grown)
- [ ] `cargo test --release --no-fail-fast` — **all** test binaries green
  (use `--no-fail-fast` so a single failure doesn't hide later ones)
- [ ] `cargo test --release -p rmumps` — workspace member tests green
- [ ] `cargo doc --no-deps` — rustdoc builds without warnings
- [ ] **Code coverage**: run `cargo llvm-cov test` and update the coverage
  table in `README.md` with current numbers (this is a release gate)
- [ ] (Optional) `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] (Optional) `cargo fmt --check`

---

## 2. Run the full benchmark suite

- [ ] `make benchmark` — runs HS + CUTEst + large-scale + domain + report
  (~2 hours; do this on a quiet machine)
- [ ] Inspect `benchmarks/BENCHMARK_REPORT.md` — sanity-check the headline numbers
- [ ] Note any regressions vs. the previous release; investigate before tagging
- [ ] Verify all sub-suites actually ran (HS, CUTEst, Electrolyte, Grid, CHO,
  large-scale, gas, water) — if compilation errors silently skipped a suite, the
  numbers will be stale. Check `benchmarks/large_scale/large_scale_results.txt`
  is real output, not error spew.
- [ ] `make gas-run` — gas pipeline NLPs (4 AMPL `.nl` problems). Does NOT
  feed into `BENCHMARK_REPORT.md`; inspect the per-problem console output and
  `benchmarks/gas/*.sol` files manually.
- [ ] `make water-run` — water distribution network NLPs (6 AMPL `.nl`
  problems from MINLPLib). Does NOT feed into `BENCHMARK_REPORT.md`; inspect
  the per-problem console output and `benchmarks/water/*.sol` files manually.

### Domain benchmarks to re-run individually (for debugging a single suite)

`make benchmark` already runs CUTEst, electrolyte, grid, CHO, and
large-scale. Use the targets below to re-run **one** suite in isolation
when you're debugging a regression without paying the full ~2 hours:

- [ ] `make electrolyte-run` — electrolyte thermodynamics
- [ ] `make grid-run` — Grid (AC OPF)
- [ ] `make cho-run` — CHO parameter estimation (if applicable)
- [ ] `make gas-run` — gas pipeline NLPs (also listed above — not in `make benchmark`)
- [ ] `make water-run` — water distribution network NLPs (also listed above — not in `make benchmark`)
- [ ] CUTEst full sweep: `RESULTS_FILE=benchmarks/cutest/results.json cargo run --bin cutest_suite --features cutest,ipopt-native --release`

---

## 3. Update documentation with fresh numbers

Headline metrics that propagate everywhere: HS solved/total, CUTEst
solved/total, both-solve count, geo-mean speedup, median speedup, ripopt-only
and Ipopt-only counts, electrolyte/grid results.

- [ ] `README.md`
  - HS suite table
  - CUTEst suite table
  - Domain benchmarks table
  - Large-scale benchmark table
  - "Interpreting the speed numbers" prose
  - Inline `RIPOPT_VERSION` comment near the C API example
- [ ] `docs/src/benchmarks.md` — mirror of README benchmark sections
- [ ] `docs/src/introduction.md` and other `docs/src/*.md` if API/CLI changed
- [ ] `docs/src/SUMMARY.md` — add entries for any new mdbook pages introduced
  this release (the mdbook TOC does not auto-discover)
- [ ] `mdbook build docs` — rebuild the book and confirm no broken links or
  missing-page warnings
- [ ] `RIPOPT_VS_IPOPT.md` — strategic comparison narrative
- [ ] `benchmarks/electrolyte/electrolyte_benchmark_report.md` — if domain-specific numbers changed
- [ ] `benchmarks/grid/grid_benchmark_report.md` — same
- [ ] `MEMORY.md` (in `~/.claude/projects/...ripopt/memory/`) —
  CUTEst Benchmark Status section
- [ ] `narrative-history.org` — append a new chapter in the same literary
  style covering the work since the last narrative update. Find the prior
  cutoff with `git log -1 --format=%H -- narrative-history.org`, then walk
  `git log --first-parent <prev>..HEAD --format="%ai %h %s"` to gather
  commit messages and dates. Group by theme/issue (not strict chronology),
  cite specific commits and memory-file lessons where they sharpen the
  story, and keep the honesty rule visible. Append after the existing final
  section so prior chapters stay intact.

---

## 4. CHANGELOG

- [ ] Add a new `## [X.Y.Z] - YYYY-MM-DD` section at the top of `CHANGELOG.md`
- [ ] Group changes under `### Added`, `### Changed`, `### Fixed`,
  `### Performance`, `### Notes`
- [ ] Walk `git log <prev-tag>..HEAD --no-merges` to make sure you didn't miss
  anything
- [ ] Mention every test/regression that was fixed during the release prep
- [ ] Include the workspace version bump line (ripopt + rmumps)

---

## 5. Bump version numbers (every place that hard-codes a version)

These must all match. A grep for the **old** version after bumping is the
safest way to confirm nothing was missed. Use two passes: one for quoted
versions (Cargo/pyproject/Markdown prose) and one for raw numeric tokens
(the C header's `#define` macros are unquoted):

```
# Pass 1: quoted "X.Y.Z" occurrences
grep -rn '"X\.Y\.Z"' --include='*.toml' --include='*.h' --include='*.rs' \
     --include='*.md' --include='*.py' . \
  | grep -v target/ | grep -v ref/ | grep -v benchmarks/

# Pass 2: unquoted version tokens in the C header + inline constants
grep -rn -E '\b[0-9]+\.[0-9]+\.[0-9]+\b' ripopt.h
```

- [ ] `Cargo.toml` — `[package].version`
- [ ] `Cargo.toml` — `rmumps = { version = "...", path = "rmumps", ... }`
  dependency line (must match `rmumps/Cargo.toml`)
- [ ] `rmumps/Cargo.toml` — `[package].version` (bump even for ripopt-only
  changes if rmumps is also being released; otherwise leave it)
- [ ] `ripopt.h` — `RIPOPT_VERSION_MAJOR`, `_MINOR`, `_PATCH`,
  and the `RIPOPT_VERSION "X.Y.Z"` string literal
- [ ] `README.md` — the inline `// "X.Y.Z"` comment in the C API example
- [ ] `pyomo-ripopt/pyproject.toml` — `[project].version`
- [ ] `pyomo-ripopt/pyomo_ripopt.egg-info/PKG-INFO` — auto-regenerates on
  `pip install`, but verify it after the install step below
- [ ] `ripopt-py/Cargo.toml` — `[package].version`
- [ ] `ripopt-py/pyproject.toml` — `[project].version` (must match the
  `ripopt-py/Cargo.toml` version; both are bumped together with the main
  `ripopt` crate so a single `vX.Y.Z` tag publishes consistent binaries to
  PyPI via `.github/workflows/publish-ripopt-py.yml`)
- [ ] `Ripopt.jl/Project.toml` — **only if** there are changes that affect
  the Julia binding (FFI signature changes, new C API functions, behavior
  changes Ripopt.jl exposes). Otherwise leave Ripopt.jl on its current
  version. The Julia binding has an independent release cadence.
- [ ] Run `cargo check` after bumping to refresh `Cargo.lock`
- [ ] **Reminder for Section 7b:** any version pin in
  `manuscript/supporting-information.org` (ripopt version, rmumps version,
  Ipopt version, rustc version) must also be bumped. The SI version-pin
  check in Section 7b will catch this, but it's worth noting here so the
  bump is on your radar in one place.

---

## 6. Verify each language interface end-to-end

Each interface should be exercised against the freshly built binary, not a
stale install. Best done in order: native Rust → C → AMPL → Pyomo plugin →
ripopt-py bindings → Julia → GAMS → tutorials, since each later layer
depends (directly or indirectly) on the ones before it.

### 6a. Native Rust library

- [ ] `cargo run --release --example hs071` — known-good Rust example
- [ ] `cargo run --release --example rosenbrock` (or equivalent unconstrained)
- [ ] One large-scale example to exercise sparse path

### 6b. C API + shared library

- [ ] `cargo build --release` produces `target/release/libripopt.{dylib,so}`
- [ ] `ripopt.h` `RIPOPT_VERSION` matches `Cargo.toml`
- [ ] `make test-c` — compiles and runs the bundled C clients
  (`examples/c_api_test.c`, `examples/c_rosenbrock.c`, `examples/c_hs035.c`,
  `examples/c_example_with_options.c`) against the freshly built
  `libripopt.{dylib,so}`
- [ ] `make install` then `ripopt --version` reports the new version

### 6c. AMPL/NL solver binary

- [ ] `cargo build --release --bin ripopt` (the AMPL solver binary)
- [ ] `ripopt --version` and `ripopt --help` work
- [ ] Solve at least one `.nl` file (e.g. an HS problem) and verify it
  reports correct status + objective
- [ ] Confirm `~/.cargo/bin/ripopt` (after `make install`) is on `$PATH` so
  `pyomo-ripopt`'s `SolverFactory('ripopt')` can resolve it in §6d

### 6d. Pyomo solver plugin

- [ ] `pip install -e ./pyomo-ripopt` from a clean Python env
- [ ] `python -c "from pyomo.environ import SolverFactory; s = SolverFactory('ripopt'); print(s.available())"`
- [ ] Run an end-to-end Pyomo model and verify `result.solver.status == 'ok'`
- [ ] Confirm `pyomo_ripopt.egg-info/PKG-INFO` shows the new version
- [ ] If publishing to PyPI: `python -m build` then check the wheel metadata

### 6e. Direct Python bindings (`ripopt-py`, PyPI name `ripopt`)

- [ ] `pip install -e ./ripopt-py` from a clean Python env (builds via maturin)
- [ ] `python -c "from ripopt import minimize; print(minimize)"`
- [ ] Run a small HS/Rosenbrock end-to-end and confirm `success=True`
- [ ] Confirm `ripopt-py/Cargo.toml` and `ripopt-py/pyproject.toml` versions
  match the ripopt `vX.Y.Z` tag — PyPI release is driven by
  `.github/workflows/publish-ripopt-py.yml` on tag push
- [ ] (Optional local smoke) `cd ripopt-py && maturin build --release` and
  inspect the produced wheel in `target/wheels/`

### 6f. Julia/JuMP interface (`Ripopt.jl`)

Skip this section entirely if the release contains no changes that affect
Ripopt.jl (most patch releases). Otherwise:

- [ ] `RIPOPT_LIBRARY_PATH=target/release julia --project=Ripopt.jl Ripopt.jl/examples/jump_hs071.jl`
- [ ] Run `Ripopt.jl/examples/jump_rosenbrock.jl` and `c_wrapper_hs071.jl`
- [ ] Bump `Ripopt.jl/Project.toml` only if Ripopt.jl itself changed
- [ ] Verify `Ripopt.jl/examples/ripopt_jump_tutorial.ipynb` still runs
- [ ] On Apple Silicon: confirm no closure-cfunction warnings (the
  module-level `@cfunction` rule from 0.6.0 must hold)

### 6g. GAMS solver link

- [ ] `cargo build --release` (gams link links against `libripopt`)
- [ ] `make -C gams` (build the GAMS bridge)
- [ ] `sudo make -C gams install` (only if a GAMS install is present)
- [ ] `sudo make -C gams test` — solves HS071 and checks the result
- [ ] If you don't have GAMS locally, document this as a manual step the
  release manager must run on a GAMS-equipped machine

### 6h. Tutorial notebooks

- [ ] Re-run all 15 notebooks in `tutorials/` **in place** against the newly
  built release, top-to-bottom:

  ```bash
  cd tutorials
  for nb in 0*.ipynb 1*.ipynb; do
      jupyter nbconvert --to notebook --execute --inplace "$nb"
  done
  ```

  Any `pyomo-ripopt` API drift or output-format change surfaces as a cell
  failure. Commit the re-executed notebooks so GitHub renders fresh outputs.
- [ ] Spot-check `15_ripopt_in_practice.ipynb` — this notebook exercises
  the most ripopt-specific surface area (solver options, diagnostics,
  architecture), so any behavioral change is most likely to show up here.
- [ ] Verify `tutorials/README.md` still accurately lists every notebook
  and its topic (update if notebooks were added, removed, or retitled)

---

## 7. Manuscript and supporting information

**Important:** `manuscript/ripopt.tex` is a generated artifact — never edit
it by hand. All content lives in `manuscript/ripopt.org`. The `.tex` (and
`.pdf`) are produced by the scimax export workflow described below. The
same rule applies to `manuscript/supporting-information.org` and its
generated `.tex`/`.pdf`.

### 7a. Update `manuscript/ripopt.org`

- [ ] Abstract — re-state the headline benchmark numbers
- [ ] HS suite section
- [ ] CUTEst suite section
- [ ] Domain-specific benchmarks section — add/update entries for any new
  benchmark suites added since the previous release (gas, water, grid,
  electrolyte, CHO, etc.)
- [ ] Failure analysis section — recompute the failure-mode breakdown
  from the fresh CUTEst run
- [ ] Opportunities for improvement section
- [ ] Conclusions section

### 7b. Update `manuscript/supporting-information.org`

The SI is the implementation/reproducibility companion document. It is
the place where every release-blocking detail about *how* the code works
should live. **Walk the SI top-to-bottom against the actual code state
before tagging.**

- [ ] **File path references** — every `src/...`, `rmumps/...`,
  `benchmarks/...` path cited in the SI must still exist. Reorganizations
  break these silently. Grep the SI for any path-shaped strings and
  re-verify each one. (Common reorganizations to watch for: any rename
  inside `benchmarks/`, anything moved between `src/` and `rmumps/`,
  anything moved between `src/` and a new module split.)
- [ ] **Code line-number references** — line numbers shift any time code
  is added or removed above. For each `file.rs:NNN` citation in the SI,
  open the file at that line and confirm it still points to the function,
  block, or comment the SI claims. Update the line number if it has
  drifted; rewrite the surrounding sentence if the code has been
  refactored beyond just shifting.
- [ ] **Excerpted code snippets** — if the SI quotes Rust source verbatim
  (e.g., a function body, a struct definition, an option list), re-pull
  the current source and diff it against the snippet in the SI. Stale
  snippets are worse than no snippets.
- [ ] **New code coverage** — for every algorithmic change in this
  release (new fallback, new option, new oracle, new convergence rule,
  new linear-algebra path, new problem class, etc.) verify the SI either
  documents it or has a stub pointing to where it lives. Walk the
  CHANGELOG `### Added` and `### Changed` sections and tick each item:
  is it in the SI? If no, decide — should it be? Most algorithmic
  changes should at least get a one-paragraph mention.
- [ ] **Command listings** — every shell command in the SI (`cargo run …`,
  `make …`, `ripopt …`, `python …`) must still work as written. If
  paths, target names, CLI flags, or env vars changed in this release,
  the SI command listings need updating. Run a representative subset
  end-to-end if you can.
- [ ] **Version pins** — bump every literal version string. Common ones:
  ripopt version (`0.6.x`), rmumps version (`0.1.x`), Ipopt version
  (`3.14.19`), Rust toolchain. Cross-reference Section 5 (version bump).
- [ ] **Reproducibility metadata** — re-confirm the listed hardware
  (Apple M3? M4? Mac Mini?), the OS (`Darwin 25.x`?), the Ipopt version,
  the rustc version, and the date stamp. The SI numbers were generated
  on a specific machine; be honest about which one.
- [ ] **CLI option listings** — if the SI documents `ripopt --help`
  output, `SolverOptions` defaults, or AMPL keywords, regenerate from
  the actual current binary (`ripopt --help`) and replace the listing.
- [ ] **Cross-references to the manuscript** — any `Section X` or
  `Equation Y` reference that points back to `ripopt.org` must still
  resolve after manuscript edits in 7a.

### 7c. Build the PDFs

- [ ] Build the manuscript PDF. **Must run from inside `manuscript/`**:
  ```bash
  cd manuscript
  scimax export ripopt.org --format pdf
  ```
  This regenerates both `ripopt.tex` and `ripopt.pdf` in one step.
- [ ] Build the supporting-information PDF, also from `manuscript/`:
  ```bash
  cd manuscript
  scimax export supporting-information.org --format pdf
  ```

### 7d. Verify both PDFs render correctly

- [ ] `manuscript/ripopt.pdf` — page count sane, no `??` undefined
  references, bibliography rendered, figures present, all benchmark
  numbers match the prose
- [ ] `manuscript/supporting-information.pdf` — page count sane, no `??`
  undefined references, every code listing renders without minted
  errors, every cross-reference into the main manuscript resolves

### 7e. Clean up intermediate build artifacts

The scimax/LaTeX pipeline leaves these behind, and **none of them should
ever be committed**. All are gitignored already, but cleaning the working
directory keeps `git status` readable and prevents stale artifacts from
confusing the next build:

```bash
cd manuscript
rm -f ripopt.tex ripopt.html ripopt.pyg ripopt.bbl ripopt.blg
rm -f supporting-information.tex supporting-information.html \
      supporting-information.pyg supporting-information.bbl \
      supporting-information.blg
rm -rf _minted
```

After cleanup the only files in `manuscript/` should be `ripopt.org`,
`ripopt.bib`, `ripopt.pdf`, `supporting-information.org`, and
`supporting-information.pdf`.

### 7f. Commit

- [ ] Stage `manuscript/ripopt.org`, `manuscript/supporting-information.org`,
  `manuscript/ripopt.pdf`, and `manuscript/supporting-information.pdf` in
  the same commit so source and PDF artifact stay in sync. The `.tex`
  files are **not** committed (gitignored, regenerated on every build).

---

## 8. Save tagged benchmark artifacts (per CLAUDE.md)

These let us compare per-problem timing across versions later:

- [ ] `cp benchmarks/BENCHMARK_REPORT.json benchmarks/BENCHMARK_REPORT_vX.Y.Z.json`
- [ ] (Optional) `cp benchmarks/cutest/results.json benchmarks/cutest/results_vX.Y.Z.json`
- [ ] (Optional) `cp benchmarks/electrolyte/electrolyte_results.json benchmarks/electrolyte/electrolyte_results_vX.Y.Z.json`
- [ ] (Optional) `cp benchmarks/grid/grid_results.json benchmarks/grid/grid_results_vX.Y.Z.json`
- [ ] (Optional) `cp benchmarks/cho/cho_results.json benchmarks/cho/cho_results_vX.Y.Z.json`

---

## 9. Final pre-tag verification

- [ ] `cargo test --release --no-fail-fast` one more time after all edits
- [ ] `cargo check --examples` one more time (catches any drift introduced
  by docs/manuscript edits)
- [ ] `cargo package --allow-dirty --list -p ripopt` — verify the file list
  matches your `exclude = [...]` rules in `Cargo.toml`. Look for files that
  shouldn't ship: `adversary/`, `.crucible/`, `benchmarks/` (HS, CUTEst,
  electrolyte, grid, cho, gas, water, large_scale), `docs/`, `manuscript/`,
  `research/`, `tutorials/`, large PDFs, `.ipynb` notebooks, `pyomo-ripopt/build/`
- [ ] `cargo package --allow-dirty -p rmumps` — same for rmumps
- [ ] `cargo publish --dry-run -p rmumps` (rmumps must publish first since
  ripopt depends on it)
- [ ] `cargo publish --dry-run -p ripopt`
- [ ] Read the diff of `Cargo.lock` and make sure no surprise dep updates
  snuck in
- [ ] **PyPI publish readiness** — before pushing the tag, confirm the
  Trusted Publishers are configured (one-time setup, but re-verify every
  release):
  - [ ] PyPI project `pyomo-ripopt` has a Trusted Publisher bound to
    this repo + `publish-pyomo.yml` + environment `pypi`
  - [ ] PyPI project `ripopt` has a Trusted Publisher bound to this repo
    + `publish-ripopt-py.yml` + environment `pypi`
  - [ ] GitHub repo has a `pypi` environment configured (Settings →
    Environments)
  - [ ] Neither workflow file has been renamed/moved since the last
    successful release; if renamed, the Trusted Publisher binding on
    PyPI must be updated to match **before** tagging
  - [ ] See `PYPI_PUBLISHING.md` for full setup details
- [ ] **Python wheel manifests** — quick sanity check that the Python
  packages won't ship stray files:
  - [ ] `cd pyomo-ripopt && python -m build --wheel && unzip -l dist/*.whl | head -40`
    — verify no `tests/`, `build/`, or `__pycache__/` in the wheel
  - [ ] `cd ripopt-py && maturin build --release && unzip -l target/wheels/*.whl | head -40`
    — verify the `_ripopt.abi3.so` is present and no probe/test scripts
    leaked in

---

## 10. Tag and publish

Publishing fans out over three independent channels once the tag is
pushed. **Only the crates.io channel has an internal ordering**:
`cargo publish -p rmumps` must land and be indexed before
`cargo publish -p ripopt`, because the ripopt crate depends on rmumps. The
two PyPI workflows (`publish-pyomo.yml` and `publish-ripopt-py.yml`) build
their own copy of ripopt from the tagged repo source — they do **not**
fetch from crates.io and are independent of the cargo publishes below.

```
                              git push origin vX.Y.Z
                                          |
                ,-------------------------+-------------------------,
                |                         |                         |
         crates.io (manual)        publish-pyomo.yml        publish-ripopt-py.yml
         rmumps → ripopt           (auto, tag trigger)      (auto, tag trigger)
```

### 10a. Push the tag

- [ ] `git add -A` (review carefully — there will be benchmark JSON, the
  manuscript pdf, and many docs)
- [ ] `git commit -m "release: vX.Y.Z"` with full release notes in body
- [ ] `git tag -a vX.Y.Z -m "ripopt vX.Y.Z"`
- [ ] `git push origin main`
- [ ] `git push origin vX.Y.Z` — **this starts both PyPI workflows
  immediately**; keep the Actions tab open to monitor them

### 10b. Publish crates.io (manual, ordered)

- [ ] `cargo publish -p rmumps`
- [ ] Wait for crates.io to index rmumps (~30s — `cargo search rmumps`
  should return the new version)
- [ ] `cargo publish -p ripopt`

### 10c. Monitor PyPI workflows (automatic)

- [ ] `.github/workflows/publish-pyomo.yml` — confirm all 5 wheel jobs +
  sdist job + publish job succeeded in the Actions tab
- [ ] `.github/workflows/publish-ripopt-py.yml` — same check (5 wheel
  jobs, publish job; no sdist by design)
- [ ] If either workflow failed during wheel builds (flaky runner, etc.),
  re-run the failed jobs from the Actions UI — the `publish` job will
  wait for them and then upload. Partial uploads are OK: PyPI rejects
  duplicate wheels, so re-running is safe.

### 10d. Other bindings

- [ ] (Optional) Manual PyPI fallback if a workflow is broken beyond a
  re-run: `cd pyomo-ripopt && python -m build && twine upload dist/*`
  (same for `ripopt-py` via `maturin publish`). Requires an API token,
  since Trusted Publishing only fires from the workflow.
- [ ] (Optional) Register/tag `Ripopt.jl` release if bumping it

### 10e. Rollback if something goes wrong

If the release blows up after the tag push but before any artifact hits
the public (crates.io, PyPI), rolling back cleanly:

1. **Cancel in-flight workflows** — GitHub Actions UI → any running
   `publish-*` run → "Cancel workflow". This stops wheel builds and
   prevents the `publish` job from firing.
2. **Delete the tag locally and remotely**:
   ```bash
   git tag -d vX.Y.Z
   git push --delete origin vX.Y.Z
   ```
3. **If a GitHub release was auto-created** (e.g. a downstream workflow
   `gh release create`d one), delete it via `gh release delete vX.Y.Z`.
4. **Zenodo** auto-archives on GitHub release creation, not tag push.
   If a Zenodo version was minted, it cannot be deleted — but you can
   publish a `vX.Y.Z+1` immediately and Zenodo will supersede it.
5. **crates.io**: `cargo yank --version X.Y.Z -p rmumps` (and ripopt) if
   a broken crate was published. Yank is reversible but prevents new
   `cargo install` of that version.
6. **PyPI**: broken wheels can be deleted from the project page within
   a few days, but the version number is burned — you cannot re-upload
   `X.Y.Z`. Bump to `X.Y.Z+1` instead.

Prefer catching problems in §9 to avoid any of this.

---

## 11. GitHub release

- [ ] `gh release create vX.Y.Z --notes-file <(awk '/^## \[X.Y.Z\]/,/^## \[/{if(/^## \[/ && !/X.Y.Z/) exit; print}' CHANGELOG.md)`
  (or via the GitHub UI with the CHANGELOG entry pasted in)
- [ ] Attach `manuscript/ripopt.pdf` and
  `manuscript/supporting-information.pdf` if you publish them with the release
- [ ] Verify the release page renders correctly on GitHub

---

## 12. Post-release

- [ ] On crates.io, confirm both `ripopt` and `rmumps` show the new version
- [ ] `cargo install ripopt --version X.Y.Z` from a clean directory and run
  `ripopt --version` to confirm the published binary works
- [ ] **PyPI end-to-end** — from a **fresh** venv (not the development one):
  ```bash
  python -m venv /tmp/ripopt-release-check
  source /tmp/ripopt-release-check/bin/activate
  pip install pyomo-ripopt==X.Y.Z
  pip install ripopt==X.Y.Z
  python -c "from pyomo.environ import SolverFactory; print(SolverFactory('ripopt').available())"
  python -c "from ripopt import minimize; import numpy as np; r = minimize(lambda x: (x[0]-1)**2, np.zeros(1)); print(r['success'], r['x'])"
  ```
  Both packages should resolve a wheel (not fall back to sdist on a
  supported platform). The sanity checks should return `True` and
  `True [1.]` respectively.
- [ ] **Zenodo DOI** — Zenodo is linked to the GitHub repo and auto-archives
  every published release (metadata in `.zenodo.json`). Wait a few minutes
  after the GitHub release in Section 11, then check
  https://doi.org/10.5281/zenodo.19542664 (the concept DOI — the parent
  record for all versions) and confirm `vX.Y.Z` appears in the version
  list. The README badge is a static `img.shields.io/badge/...` SVG that
  hard-codes the concept DOI, so it doesn't need to be regenerated per
  release. (Avoid both the `zenodo.org/badge/<repo_id>.svg` redirect form
  and the `zenodo.org/badge/DOI/...svg` direct form: Zenodo rate-limits
  GitHub Camo, which then caches a 403 and the badge stops rendering.)
- [ ] Update `MEMORY.md` HS/CUTEst status sections with the new release
  numbers if you didn't already in step 3
- [ ] Bump `Cargo.toml` to the next `+dev` working version if you use that
  convention (ripopt currently does not — versions remain on the released
  number until the next release)
- [ ] Close any GitHub issues fixed in this release with a comment pointing
  at the release notes
- [ ] Tweet/announce/etc. as appropriate
- [ ] **Draft a brief LinkedIn post** summarizing the release: one-line hook,
  2–4 headline numbers or changes (HS/CUTEst deltas, notable new features),
  a link to the GitHub release page, and relevant tags (`#Rust`,
  `#Optimization`, `#OpenSource`). Keep it under ~150 words so it renders
  without a "see more" fold on desktop.

---

## Things to double-check that are easy to forget

- [ ] **`cargo check --examples`** — example files often use trait
  signatures that drift; the test suite will not catch this
- [ ] **Compiler warnings** — they don't fail the build but accumulate;
  clean them before tagging
- [ ] **No `--no-verify` git operations** (CLAUDE.md global rule)
- [ ] **Crucible/adversary/research directories** stay excluded from the
  crates.io package (verified via `cargo package --list`)
- [ ] **Large PDFs and notebooks** stay excluded from the package
- [ ] **Benchmark JSON regenerated by `make benchmark`** — these should be
  committed (they're the source of truth for the report numbers) but be
  aware they generate a large diff
- [ ] **License files** — confirm `LICENSE` (EPL-2.0 for ripopt) and
  `rmumps/LICENSE` (CECILL-C) are present and current
- [ ] **rustdoc links** — broken intra-doc links will silently fail; check
  `cargo doc` output for warnings
- [ ] **The C header** (`ripopt.h`) is the public ABI. If any C API
  function signature changed, that's a SemVer concern — major-bump or
  carefully document

