# ripopt Benchmark Report

Generated: 2026-05-06 21:26:53

## Executive Summary

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Optimal | **562/745** (75.4%) | **572/745** (76.8%) |
| Acceptable | 19 | 5 |
| Total solved (Optimal + Acceptable) | 581 (78.0%) | 577 (77.4%) |
| Solved exclusively | 21 | 17 |
| Both solved | 560 | |
| Matching objectives (< 0.01%) | 532/560 | |
| Acceptable at worse local min | 4 | |

> **Note:** ripopt uses fallback strategies (L-BFGS Hessian, AL, SQP, slack
> reformulation) that Ipopt does not have, which accounts for much of the
> Acceptable count difference. The "Different Local Minima" section below
> lists Acceptable solutions where ripopt converged to a worse local minimum.

## Per-Suite Summary

| Suite | Problems | ripopt solved | Ipopt solved | ripopt only | Ipopt only | Both solved | Match |
|-------|----------|--------------|-------------|-------------|------------|------------|-------|
| CUTEst | 727 | 564 (77.6%) | 561 (77.2%) | 20 | 17 | 544 | 517/544 |
| Electrolyte | 13 | 13 (100.0%) | 12 (92.3%) | 1 | 0 | 12 | 11/12 |
| Grid | 4 | 4 (100.0%) | 4 (100.0%) | 0 | 0 | 4 | 4/4 |
| CHO | 1 | 0 (0.0%) | 0 (0.0%) | 0 | 0 | 0 | 0/1 |

## CUTEst Suite — Performance

On 544 commonly-solved problems:

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Median time | 192us | 2.2ms |
| Total time | 15.48s | 18.46s |
| Mean iterations | 37.5 | 36.6 |
| Median iterations | 14 | 12 |

- **Geometric mean speedup**: 8.9x
- **Median speedup**: 12.8x
- ripopt faster: 497/544 (91%)
- ripopt 10x+ faster: 315/544
- Ipopt faster: 47/544

## Electrolyte Suite — Performance

On 12 commonly-solved problems:

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Median time | 172us | 2.1ms |
| Total time | 33.0ms | 52.3ms |
| Mean iterations | 212.8 | 19.2 |
| Median iterations | 7 | 7 |

- **Geometric mean speedup**: 6.9x
- **Median speedup**: 9.3x
- ripopt faster: 11/12 (92%)
- ripopt 10x+ faster: 5/12
- Ipopt faster: 1/12

## Grid Suite — Performance

On 4 commonly-solved problems:

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Median time | 11.4ms | 10.8ms |
| Total time | 50.2ms | 74.9ms |
| Mean iterations | 15.8 | 12.5 |
| Median iterations | 19 | 14 |

- **Geometric mean speedup**: 1.8x
- **Median speedup**: 2.1x
- ripopt faster: 3/4 (75%)
- ripopt 10x+ faster: 0/4
- Ipopt faster: 1/4

## Failure Analysis

### CUTEst Suite

| Failure Mode | ripopt | Ipopt |
|-------------|--------|-------|
| ErrorInStepComputation | 0 | 2 |
| EvaluationError | 3 | 0 |
| Infeasible | 0 | 11 |
| InvalidNumberDetected | 0 | 1 |
| IpoptStatus(-10) | 0 | 123 |
| IpoptStatus(3) | 0 | 1 |
| IpoptStatus(4) | 0 | 2 |
| LocalInfeasibility | 27 | 0 |
| MaxIterations | 39 | 12 |
| NumericalError | 14 | 0 |
| RestorationFailed | 74 | 4 |
| StopAtTinyStep | 3 | 0 |
| Timeout | 3 | 10 |

### Electrolyte Suite

| Failure Mode | ripopt | Ipopt |
|-------------|--------|-------|
| Infeasible | 0 | 1 |

### CHO Suite

| Failure Mode | ripopt | Ipopt |
|-------------|--------|-------|
| MaxIter | 0 | 1 |
| MaxIterations | 1 | 0 |

## Regressions (Ipopt solves, ripopt fails)

| Problem | Suite | n | m | ripopt status | Ipopt obj |
|---------|-------|---|---|--------------|-----------|
| ACOPR30 | CUTEst | 72 | 172 | MaxIterations | 5.768924e+02 |
| CRESC50 | CUTEst | 6 | 100 | NumericalError | 7.862467e-01 |
| DEMBO7 | CUTEst | 16 | 20 | NumericalError | 1.747870e+02 |
| DISCS | CUTEst | 36 | 66 | MaxIterations | 1.200007e+01 |
| DJTL | CUTEst | 2 | 0 | StopAtTinyStep | -8.951545e+03 |
| ELATTAR | CUTEst | 7 | 102 | NumericalError | 7.420618e+01 |
| FEEDLOC | CUTEst | 90 | 259 | NumericalError | -9.539854e-10 |
| HATFLDFLNE | CUTEst | 3 | 3 | RestorationFailed | 0.000000e+00 |
| HIMMELP4 | CUTEst | 2 | 3 | NumericalError | -5.901318e+01 |
| LOGHAIRY | CUTEst | 2 | 0 | MaxIterations | 1.823216e-01 |
| MSS1 | CUTEst | 90 | 73 | StopAtTinyStep | -1.400000e+01 |
| NET1 | CUTEst | 48 | 57 | RestorationFailed | 9.411943e+05 |
| PFIT4 | CUTEst | 3 | 3 | RestorationFailed | 0.000000e+00 |
| QCNEW | CUTEst | 9 | 3 | NumericalError | -8.065219e+02 |
| SIPOW2 | CUTEst | 2 | 2000 | NumericalError | -1.000000e+00 |
| TAXR13322 | CUTEst | 72 | 1261 | NumericalError | -3.429089e+02 |
| WOMFLET | CUTEst | 3 | 3 | MaxIterations | 6.050000e+00 |

## Wins (ripopt solves, Ipopt fails) — 21 problems

| Problem | Suite | n | m | Ipopt status | ripopt obj |
|---------|-------|---|---|-------------|------------|
| BEALENE | CUTEst | 2 | 3 | IpoptStatus(-10) | 0.000000e+00 |
| BOX3NE | CUTEst | 3 | 10 | IpoptStatus(-10) | 0.000000e+00 |
| BROWNBSNE | CUTEst | 2 | 3 | IpoptStatus(-10) | 0.000000e+00 |
| DECONVB | CUTEst | 63 | 0 | MaxIterations | 3.622754e-03 |
| DENSCHNBNE | CUTEst | 2 | 3 | IpoptStatus(-10) | 0.000000e+00 |
| DEVGLA1NE | CUTEst | 4 | 24 | IpoptStatus(-10) | 0.000000e+00 |
| DEVGLA2NE | CUTEst | 5 | 16 | IpoptStatus(-10) | 0.000000e+00 |
| ENGVAL2NE | CUTEst | 3 | 5 | IpoptStatus(-10) | 0.000000e+00 |
| EQC | CUTEst | 9 | 3 | ErrorInStepComputation | -8.263086e+02 |
| EXP2NE | CUTEst | 2 | 10 | IpoptStatus(-10) | 0.000000e+00 |
| HS25NE | CUTEst | 3 | 99 | IpoptStatus(-10) | 0.000000e+00 |
| LANCZOS1 | CUTEst | 6 | 24 | IpoptStatus(-10) | 0.000000e+00 |
| NYSTROM5 | CUTEst | 18 | 20 | IpoptStatus(-10) | 0.000000e+00 |
| NYSTROM5C | CUTEst | 18 | 20 | IpoptStatus(-10) | 0.000000e+00 |
| PFIT2 | CUTEst | 3 | 3 | RestorationFailed | 0.000000e+00 |
| POLAK3 | CUTEst | 12 | 10 | MaxIterations | 5.933003e+00 |
| POLAK6 | CUTEst | 5 | 4 | MaxIterations | -4.400000e+01 |
| ROBOT | CUTEst | 14 | 2 | IpoptStatus(3) | 6.593299e+00 |
| Seawater speciation | Electrolyte | 15 | 8 | Infeasible | -1.348272e+00 |
| TRO4X4 | CUTEst | 63 | 25 | IpoptStatus(4) | 8.999996e+00 |
| WACHBIEG | CUTEst | 3 | 2 | Infeasible | 1.000000e+00 |

## Different Local Minima — 4 problems

ripopt converged (Acceptable) but to a different — usually worse — local
minimum than Ipopt found. Both solvers satisfied first-order KKT conditions
at their respective solutions. For nonconvex problems this is expected;
for convex problems it indicates the solver trajectory went astray.

| Problem | Suite | n | m | ripopt obj | Ipopt obj | Rel. error |
|---------|-------|---|---|------------|-----------|------------|
| MGH17LS | CUTEst | 5 | 0 | 1.022414e+00 | 7.898394e-05 | 100.0% |
| STREG | CUTEst | 4 | 0 | 4.526251e-01 | 8.901950e-02 | 36.4% |
| HS108 | CUTEst | 9 | 13 | -6.749814e-01 | -5.000000e-01 | 17.5% |
| HS54 | CUTEst | 6 | 1 | -8.498037e-01 | -9.080748e-01 | 5.8% |

## Acceptable (not Optimal) — 19 problems

These problems converged within relaxed tolerances but not strict tolerances.

| Problem | Suite | n | m | Ipopt status | ripopt obj | Ipopt obj |
|---------|-------|---|---|-------------|------------|-----------|
| ALLINITA | CUTEst | 4 | 4 | Optimal | 3.329878e+01 | 3.329611e+01 |
| ALLINITC | CUTEst | 4 | 1 | Optimal | 3.049193e+01 | 3.049261e+01 |
| BT8 | CUTEst | 5 | 2 | Optimal | 1.000000e+00 | 1.000000e+00 |
| DECONVB | CUTEst | 63 | 0 | MaxIterations | 3.622754e-03 | 2.569475e-03 |
| DECONVU | CUTEst | 63 | 0 | Optimal | 4.034019e-13 | 4.146188e-13 |
| EQC | CUTEst | 9 | 3 | ErrorInStepComputation | -8.263086e+02 | -8.651227e+02 |
| HAIFAM | CUTEst | 99 | 150 | Optimal | -4.500036e+01 | -4.500036e+01 |
| HS108 | CUTEst | 9 | 13 | Optimal | -6.749814e-01 | -5.000000e-01 |
| HS13 | CUTEst | 2 | 1 | Optimal | 9.941173e-01 | 9.945785e-01 |
| HS54 | CUTEst | 6 | 1 | Optimal | -8.498037e-01 | -9.080748e-01 |
| HS99EXP | CUTEst | 31 | 21 | Optimal | -1.260006e+12 | -1.260006e+12 |
| HYDC20LS | CUTEst | 99 | 0 | Optimal | 6.586238e-15 | 2.967522e-15 |
| LSC2LS | CUTEst | 3 | 0 | Optimal | 1.334125e+01 | 1.333439e+01 |
| MGH10LS | CUTEst | 3 | 0 | Optimal | 8.794586e+01 | 8.794586e+01 |
| MGH17LS | CUTEst | 5 | 0 | Optimal | 1.022414e+00 | 7.898394e-05 |
| POLAK5 | CUTEst | 3 | 2 | Optimal | 5.000000e+01 | 5.000000e+01 |
| STREG | CUTEst | 4 | 0 | Optimal | 4.526251e-01 | 8.901950e-02 |
| TRO3X3 | CUTEst | 30 | 13 | Optimal | 8.999226e+00 | 8.967478e+00 |
| TRO4X4 | CUTEst | 63 | 25 | IpoptStatus(4) | 8.999996e+00 | -1.957476e+20 |

## Large-Scale Synthetic Problems — ripopt vs Ipopt

Synthetic problems with known structure, up to 100K variables.
Both solvers receive the exact same NlpProblem struct via the Rust trait interface.

| Problem | n | m | ripopt | iters | time | Ipopt | iters | time | speedup |
|---------|---|---|--------|-------|------|-------|-------|------|---------|
| Rosenbrock 500 | 500 | 0 | Optimal | 751 | 0.071s | Optimal | 749 | 0.191s | 2.7x |
| SparseQP 1K | 500 | 500 | Optimal | 6 | 0.005s | Optimal | 6 | 0.004s | 0.8x |
| Bratu 1K | 1,000 | 998 | Optimal | 2 | 0.003s | Optimal | 2 | 0.003s | 0.8x |
| OptControl 2.5K | 2,499 | 1,250 | Optimal | 1 | 0.014s | Optimal | 1 | 0.003s | 0.2x |
| Rosenbrock 5K | 5,000 | 0 | MaxIterations | 2999 | 2.897s | Failed | 3000 | 3.693s | 1.3x |
| Poisson 2.5K | 5,000 | 2,500 | Optimal | 1 | 0.055s | Optimal | 1 | 0.010s | 0.2x |
| Bratu 10K | 10,000 | 9,998 | Optimal | 1 | 0.142s | Optimal | 2 | 0.012s | 0.1x |
| OptControl 20K | 19,999 | 10,000 | Optimal | 1 | 0.296s | Optimal | 1 | 0.020s | 0.1x |
| Poisson 50K | 49,928 | 24,964 | Optimal | 1 | 2.321s | Optimal | 1 | 0.149s | 0.1x |
| SparseQP 100K | 50,000 | 50,000 | Optimal | 6 | 5.569s | Optimal | 6 | 0.294s | 0.1x |

ripopt: **9/10 solved** in 11.4s total
Ipopt: **9/10 solved** in 4.4s total

---
*Generated by benchmark_report.py*