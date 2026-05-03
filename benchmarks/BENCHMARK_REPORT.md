# ripopt Benchmark Report

Generated: 2026-05-02 23:19:50

## Executive Summary

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Optimal | **553/745** (74.2%) | **572/745** (76.8%) |
| Acceptable | 20 | 5 |
| Total solved (Optimal + Acceptable) | 573 (76.9%) | 577 (77.4%) |
| Solved exclusively | 19 | 23 |
| Both solved | 554 | |
| Matching objectives (< 0.01%) | 526/554 | |
| Acceptable at worse local min | 4 | |

> **Note:** ripopt uses fallback strategies (L-BFGS Hessian, AL, SQP, slack
> reformulation) that Ipopt does not have, which accounts for much of the
> Acceptable count difference. The "Different Local Minima" section below
> lists Acceptable solutions where ripopt converged to a worse local minimum.

## Per-Suite Summary

| Suite | Problems | ripopt solved | Ipopt solved | ripopt only | Ipopt only | Both solved | Match |
|-------|----------|--------------|-------------|-------------|------------|------------|-------|
| CUTEst | 727 | 556 (76.5%) | 561 (77.2%) | 18 | 23 | 538 | 511/538 |
| Electrolyte | 13 | 13 (100.0%) | 12 (92.3%) | 1 | 0 | 12 | 11/12 |
| Grid | 4 | 4 (100.0%) | 4 (100.0%) | 0 | 0 | 4 | 4/4 |
| CHO | 1 | 0 (0.0%) | 0 (0.0%) | 0 | 0 | 0 | 0/1 |

## CUTEst Suite — Performance

On 538 commonly-solved problems:

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Median time | 183us | 2.3ms |
| Total time | 17.16s | 19.14s |
| Mean iterations | 39.7 | 39.4 |
| Median iterations | 14 | 12 |

- **Geometric mean speedup**: 9.4x
- **Median speedup**: 13.5x
- ripopt faster: 484/538 (90%)
- ripopt 10x+ faster: 321/538
- Ipopt faster: 54/538

## Electrolyte Suite — Performance

On 12 commonly-solved problems:

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Median time | 113us | 1.2ms |
| Total time | 15.7ms | 41.8ms |
| Mean iterations | 172.2 | 19.2 |
| Median iterations | 7 | 7 |

- **Geometric mean speedup**: 7.9x
- **Median speedup**: 10.7x
- ripopt faster: 11/12 (92%)
- ripopt 10x+ faster: 7/12
- Ipopt faster: 1/12

## Grid Suite — Performance

On 4 commonly-solved problems:

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Median time | 10.2ms | 10.5ms |
| Total time | 52.0ms | 72.1ms |
| Mean iterations | 16.0 | 12.5 |
| Median iterations | 19 | 14 |

- **Geometric mean speedup**: 2.1x
- **Median speedup**: 2.0x
- ripopt faster: 4/4 (100%)
- ripopt 10x+ faster: 0/4
- Ipopt faster: 0/4

## Failure Analysis

### CUTEst Suite

| Failure Mode | ripopt | Ipopt |
|-------------|--------|-------|
| ErrorInStepComputation | 0 | 2 |
| EvaluationError | 3 | 0 |
| Infeasible | 1 | 11 |
| InvalidNumberDetected | 0 | 1 |
| IpoptStatus(-10) | 0 | 123 |
| IpoptStatus(3) | 0 | 1 |
| IpoptStatus(4) | 0 | 2 |
| LocalInfeasibility | 56 | 0 |
| MaxIterations | 83 | 12 |
| NumericalError | 13 | 0 |
| RestorationFailed | 6 | 4 |
| StopAtTinyStep | 2 | 0 |
| Timeout | 7 | 10 |

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
| BT8 | CUTEst | 5 | 2 | MaxIterations | 1.000000e+00 |
| CRESC4 | CUTEst | 6 | 8 | RestorationFailed | 8.718975e-01 |
| CRESC50 | CUTEst | 6 | 100 | NumericalError | 7.862467e-01 |
| DEMBO7 | CUTEst | 16 | 20 | NumericalError | 1.747870e+02 |
| DISCS | CUTEst | 36 | 66 | MaxIterations | 1.200007e+01 |
| ELATTAR | CUTEst | 7 | 102 | MaxIterations | 7.420618e+01 |
| FEEDLOC | CUTEst | 90 | 259 | NumericalError | -9.539854e-10 |
| FLETCHER | CUTEst | 4 | 4 | MaxIterations | 1.165685e+01 |
| HAIFAM | CUTEst | 99 | 150 | NumericalError | -4.500036e+01 |
| HATFLDH | CUTEst | 4 | 7 | MaxIterations | -2.450000e+01 |
| HIMMELP5 | CUTEst | 2 | 3 | NumericalError | -5.901318e+01 |
| HIMMELP6 | CUTEst | 2 | 5 | MaxIterations | -5.901318e+01 |
| HS101 | CUTEst | 7 | 5 | MaxIterations | 1.809765e+03 |
| LOGHAIRY | CUTEst | 2 | 0 | MaxIterations | 1.823216e-01 |
| MSS1 | CUTEst | 90 | 73 | LocalInfeasibility | -1.400000e+01 |
| NET1 | CUTEst | 48 | 57 | Infeasible | 9.411943e+05 |
| PFIT4 | CUTEst | 3 | 3 | MaxIterations | 0.000000e+00 |
| POLAK4 | CUTEst | 3 | 3 | MaxIterations | -9.965951e-09 |
| QCNEW | CUTEst | 9 | 3 | NumericalError | -8.065219e+02 |
| SPANHYD | CUTEst | 97 | 33 | MaxIterations | 2.397380e+02 |
| TAXR13322 | CUTEst | 72 | 1261 | NumericalError | -3.429089e+02 |
| TRO3X3 | CUTEst | 30 | 13 | MaxIterations | 8.967478e+00 |

## Wins (ripopt solves, Ipopt fails) — 19 problems

| Problem | Suite | n | m | Ipopt status | ripopt obj |
|---------|-------|---|---|-------------|------------|
| BEALENE | CUTEst | 2 | 3 | IpoptStatus(-10) | 0.000000e+00 |
| BOX3NE | CUTEst | 3 | 10 | IpoptStatus(-10) | 0.000000e+00 |
| BROWNBSNE | CUTEst | 2 | 3 | IpoptStatus(-10) | 0.000000e+00 |
| DECONVB | CUTEst | 63 | 0 | MaxIterations | 3.622754e-03 |
| DENSCHNBNE | CUTEst | 2 | 3 | IpoptStatus(-10) | 0.000000e+00 |
| DEVGLA1NE | CUTEst | 4 | 24 | IpoptStatus(-10) | 0.000000e+00 |
| DEVGLA2NE | CUTEst | 5 | 16 | IpoptStatus(-10) | 0.000000e+00 |
| EGGCRATENE | CUTEst | 2 | 4 | IpoptStatus(-10) | 0.000000e+00 |
| ENGVAL2NE | CUTEst | 3 | 5 | IpoptStatus(-10) | 0.000000e+00 |
| EXP2NE | CUTEst | 2 | 10 | IpoptStatus(-10) | 0.000000e+00 |
| HS25NE | CUTEst | 3 | 99 | IpoptStatus(-10) | 0.000000e+00 |
| LANCZOS1 | CUTEst | 6 | 24 | IpoptStatus(-10) | 0.000000e+00 |
| LEVYMONE5 | CUTEst | 2 | 4 | IpoptStatus(-10) | 0.000000e+00 |
| LEWISPOL | CUTEst | 6 | 9 | IpoptStatus(-10) | 2.999705e+00 |
| PFIT2 | CUTEst | 3 | 3 | RestorationFailed | 0.000000e+00 |
| POLAK6 | CUTEst | 5 | 4 | MaxIterations | -4.400000e+01 |
| POWELLSQ | CUTEst | 2 | 2 | Infeasible | 0.000000e+00 |
| Seawater speciation | Electrolyte | 15 | 8 | Infeasible | -1.347485e+00 |
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

## Acceptable (not Optimal) — 20 problems

These problems converged within relaxed tolerances but not strict tolerances.

| Problem | Suite | n | m | Ipopt status | ripopt obj | Ipopt obj |
|---------|-------|---|---|-------------|------------|-----------|
| ALLINITA | CUTEst | 4 | 4 | Optimal | 3.329878e+01 | 3.329611e+01 |
| ALLINITC | CUTEst | 4 | 1 | Optimal | 3.049199e+01 | 3.049261e+01 |
| DECONVB | CUTEst | 63 | 0 | MaxIterations | 3.622754e-03 | 2.569475e-03 |
| DECONVU | CUTEst | 63 | 0 | Optimal | 4.034019e-13 | 4.146188e-13 |
| DJTL | CUTEst | 2 | 0 | Acceptable | -8.951545e+03 | -8.951545e+03 |
| HAIFAS | CUTEst | 13 | 9 | Optimal | -4.500000e-01 | -4.500000e-01 |
| HIELOW | CUTEst | 3 | 0 | Optimal | 8.741654e+02 | 8.741654e+02 |
| HS108 | CUTEst | 9 | 13 | Optimal | -6.749814e-01 | -5.000000e-01 |
| HS13 | CUTEst | 2 | 1 | Optimal | 9.936039e-01 | 9.945785e-01 |
| HS54 | CUTEst | 6 | 1 | Optimal | -8.498037e-01 | -9.080748e-01 |
| HS99EXP | CUTEst | 31 | 21 | Optimal | -1.260006e+12 | -1.260006e+12 |
| HYDC20LS | CUTEst | 99 | 0 | Optimal | 6.586238e-15 | 2.967522e-15 |
| LEVYMONT10 | CUTEst | 10 | 0 | Optimal | 1.637009e+02 | 1.637009e+02 |
| LEWISPOL | CUTEst | 6 | 9 | IpoptStatus(-10) | 2.999705e+00 | 0.000000e+00 |
| LSC2LS | CUTEst | 3 | 0 | Optimal | 1.334125e+01 | 1.333439e+01 |
| MGH10LS | CUTEst | 3 | 0 | Optimal | 8.794586e+01 | 8.794586e+01 |
| MGH17LS | CUTEst | 5 | 0 | Optimal | 1.022414e+00 | 7.898394e-05 |
| POLAK5 | CUTEst | 3 | 2 | Optimal | 5.000000e+01 | 5.000000e+01 |
| STREG | CUTEst | 4 | 0 | Optimal | 4.526251e-01 | 8.901950e-02 |
| VESUVIOLS | CUTEst | 8 | 0 | Optimal | 9.914100e+02 | 9.914100e+02 |

## Large-Scale Synthetic Problems — ripopt vs Ipopt

Synthetic problems with known structure, up to 100K variables.
Both solvers receive the exact same NlpProblem struct via the Rust trait interface.

| Problem | n | m | ripopt | iters | time | Ipopt | iters | time | speedup |
|---------|---|---|--------|-------|------|-------|-------|------|---------|
| Rosenbrock 500 | 500 | 0 | Optimal | 751 | 0.092s | Optimal | 749 | 0.183s | 2.0x |
| SparseQP 1K | 500 | 500 | Optimal | 6 | 0.013s | Optimal | 6 | 0.004s | 0.3x |
| Bratu 1K | 1,000 | 998 | Optimal | 2 | 0.008s | Optimal | 2 | 0.002s | 0.3x |
| OptControl 2.5K | 2,499 | 1,250 | Optimal | 1 | 0.027s | Optimal | 1 | 0.003s | 0.1x |
| Rosenbrock 5K | 5,000 | 0 | MaxIterations | 2999 | 3.696s | Failed | 3000 | 3.661s | 1.0x |
| Poisson 2.5K | 5,000 | 2,500 | Optimal | 1 | 5.777s | Optimal | 1 | 0.010s | N/A |
| Bratu 10K | 10,000 | 9,998 | Optimal | 1 | 0.154s | Optimal | 2 | 0.012s | 0.1x |
| OptControl 20K | 19,999 | 10,000 | Optimal | 1 | 1.204s | Optimal | 1 | 0.020s | 0.0x |
| Poisson 50K | 49,928 | 24,964 | Optimal | 1 | 4463.801s | Optimal | 1 | 0.121s | N/A |
| SparseQP 100K | 50,000 | 50,000 | Optimal | 6 | 17.348s | Optimal | 6 | 0.297s | 0.0x |

ripopt: **9/10 solved** in 4492.1s total
Ipopt: **9/10 solved** in 4.3s total

---
*Generated by benchmark_report.py*