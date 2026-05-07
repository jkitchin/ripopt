# Issue #23 executable incidence fixtures

These fixtures are exported from the local `~/repos/incidence_examples`
tutorial thermodynamics model. They are dependency-free `.nl` snapshots so CI
can exercise ripopt without installing Pyomo, IDAES, or the local example repo.

## Selected cases

| Case | Source | Shape | Local preprocessing result |
| --- | --- | --- | --- |
| `tutorial_flow_density.nl` | Final tutorial repair with the added density/flow relation and solved particle porosity | 19 variables, 19 equality constraints | Helped: `Optimal`, 0 iterations with preprocessing vs. 6 without |
| `tutorial_flow_density_perturbed.nl` | Same repaired model at a shifted operating point | 19 variables, 19 equality constraints | Helped: `Optimal`, 0 iterations with preprocessing vs. 7 without |

Both cases compare `enable_preprocessing=yes` and
`enable_preprocessing=no`, reporting status, objective, full-space constraint
violation, auxiliary fallback status, and iteration counts in
`tests/nl_integration.rs`.

## Survey notes

- `incidence_examples/tutorial/run_tutorial.py` is executable in a fresh
  `.venv` after applying the IDAES `get_solver` compatibility shim used by the
  generator. The earlier `sum_flow` repair also exports and runs, but both
  preprocessing modes currently end in `NumericalError`, so it is not committed
  as a passing regression fixture.
- `incidence_examples/example1/run_clc_dm_example.py` and
  `incidence_examples/example2/run_scc_example.py` were not committed as
  executable fixtures because the current `idaes-pse` package installed in the
  fresh environment does not provide `idaes.gas_solid_contactors`.

## Local validation

The fixtures were regenerated and compared locally with:

```bash
env PATH=/tmp/ripopt-issue23-pixi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin \
  MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/ripopt-issue23-mpl \
  .venv/bin/python tests/fixtures/issue_23/generate_incidence_nl.py \
  --ripopt target/release/ripopt
```

The `.venv` was created with `uv`, and `ipopt` was installed through `pixi`
under `/tmp/ripopt-issue23-pixi` for scripts that expect a Pyomo solver on
`PATH`.
