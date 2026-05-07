# Preprocessing on/off benchmark

This benchmark compares the same `ripopt` binary with:

- `enable_preprocessing=yes`
- `enable_preprocessing=no`

It is intended to produce a compact Markdown table and JSON artifact for the
preprocessing pull request. The generated output is written under
`benchmarks/preprocessing/results/`, which is ignored by git.

## Instances

The script can use these dependency-free `.nl` instances already tracked in the
repo:

| Profile | Instances | Purpose |
| --- | --- | --- |
| `smoke` | issue #10 fixture, both issue #23 incidence fixtures, `gaslib11_steady.nl` | Fast validation that the comparison machinery works |
| `default` | `smoke`, `gaslib11_dynamic.nl`, `gaslib40_steady.nl`, all six water instances | Main PR evidence set without optional external libraries or the largest dynamic gas case |
| `full` | `default`, `gaslib40_dynamic.nl`, CHO parmest | Long/stress run, useful before posting final numbers |

The tracked `tests/fixtures/issue_15/idaes_helmholtz.nl` is not in any profile
because it depends on an external function library in some environments. Run it
explicitly with `--instance tests/fixtures/issue_15/idaes_helmholtz.nl` if the
needed library is available.

## Run

Build a release binary first:

```bash
cargo build --release --bin ripopt
```

For a quick check:

```bash
python3 benchmarks/preprocessing/run_preprocessing_benchmark.py \
  --ripopt target/release/ripopt \
  --profile smoke \
  --repeat 1 \
  --timeout 120
```

For PR-ready numbers, run at least the default profile with repeated solves:

```bash
python3 benchmarks/preprocessing/run_preprocessing_benchmark.py \
  --ripopt target/release/ripopt \
  --profile default \
  --repeat 3 \
  --timeout 300 \
  --output benchmarks/preprocessing/results/default.json \
  --markdown benchmarks/preprocessing/results/default.md
```

For the broadest in-repo run:

```bash
python3 benchmarks/preprocessing/run_preprocessing_benchmark.py \
  --ripopt target/release/ripopt \
  --profile full \
  --repeat 3 \
  --timeout 900 \
  --output benchmarks/preprocessing/results/full.json \
  --markdown benchmarks/preprocessing/results/full.md
```

Use `--list-instances` to see the selected paths, `--instance PATH` to run a
custom subset, or `--manifest PATH` for a text file of instances.

## Reporting

Paste the table from the generated Markdown file into the upstream PR. Include:

- ripopt commit and binary type, preferably `target/release/ripopt`
- profile, repeat count, timeout, and `max_iter`
- the generated summary table
- the JSON artifact if detailed rerun data is needed

The table reports the original shape, auxiliary reduction shape, statuses,
iterations, iteration delta, and median end-to-end elapsed-time ratio. Positive
iteration delta means preprocessing used fewer IPM iterations. Elapsed-time
ratios above 1 mean preprocessing was faster end-to-end. The JSON artifact also
keeps ripopt's reported solver wall time for each run, but the Markdown table
uses process elapsed time so auxiliary preprocessing overhead is included.
