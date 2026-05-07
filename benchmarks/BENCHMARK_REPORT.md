# ripopt Benchmark Report

Generated: 2026-05-07 13:45:42

## Executive Summary

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Optimal | **566/745** (76.0%) | **572/745** (76.8%) |
| Acceptable | 18 | 5 |
| Total solved (Optimal + Acceptable) | 584 (78.4%) | 577 (77.4%) |
| Solved exclusively | 25 | 18 |
| Both solved | 559 | |
| Matching objectives (< 0.01%) | 538/559 | |
| Acceptable at worse local min | 4 | |

> **Note:** ripopt uses fallback strategies (L-BFGS Hessian, AL, SQP, slack
> reformulation) that Ipopt does not have, which accounts for much of the
> Acceptable count difference. The "Different Local Minima" section below
> lists Acceptable solutions where ripopt converged to a worse local minimum.

## Per-Suite Summary

| Suite | Problems | ripopt solved | Ipopt solved | ripopt only | Ipopt only | Both solved | Match |
|-------|----------|--------------|-------------|-------------|------------|------------|-------|
| CUTEst | 727 | 567 (78.0%) | 561 (77.2%) | 24 | 18 | 543 | 523/543 |
| Electrolyte | 13 | 13 (100.0%) | 12 (92.3%) | 1 | 0 | 12 | 11/12 |
| Grid | 4 | 4 (100.0%) | 4 (100.0%) | 0 | 0 | 4 | 4/4 |
| CHO | 1 | 0 (0.0%) | 0 (0.0%) | 0 | 0 | 0 | 0/1 |

## CUTEst Suite — Performance

On 543 commonly-solved problems:

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Median time | 143us | 2.3ms |
| Total time | 15.93s | 18.45s |
| Mean iterations | 42.9 | 36.2 |
| Median iterations | 14 | 12 |

- **Geometric mean speedup**: 11.0x
- **Median speedup**: 17.2x
- ripopt faster: 509/543 (94%)
- ripopt 10x+ faster: 366/543
- Ipopt faster: 34/543

## Electrolyte Suite — Performance

On 12 commonly-solved problems:

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Median time | 139us | 1.1ms |
| Total time | 22.2ms | 43.6ms |
| Mean iterations | 212.8 | 19.2 |
| Median iterations | 7 | 7 |

- **Geometric mean speedup**: 7.0x
- **Median speedup**: 10.2x
- ripopt faster: 11/12 (92%)
- ripopt 10x+ faster: 7/12
- Ipopt faster: 1/12

## Grid Suite — Performance

On 4 commonly-solved problems:

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Median time | 6.9ms | 8.3ms |
| Total time | 41.3ms | 68.6ms |
| Mean iterations | 15.8 | 12.5 |
| Median iterations | 19 | 14 |

- **Geometric mean speedup**: 2.0x
- **Median speedup**: 2.4x
- ripopt faster: 4/4 (100%)
- ripopt 10x+ faster: 0/4
- Ipopt faster: 0/4

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
| LocalInfeasibility | 33 | 0 |
| MaxIterations | 38 | 12 |
| NumericalError | 11 | 0 |
| RestorationFailed | 70 | 4 |
| StopAtTinyStep | 2 | 0 |
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
| DISCS | CUTEst | 36 | 66 | MaxIterations | 1.200007e+01 |
| DJTL | CUTEst | 2 | 0 | StopAtTinyStep | -8.951545e+03 |
| ELATTAR | CUTEst | 7 | 102 | MaxIterations | 7.420618e+01 |
| FEEDLOC | CUTEst | 90 | 259 | NumericalError | -9.539854e-10 |
| HATFLDFLNE | CUTEst | 3 | 3 | RestorationFailed | 0.000000e+00 |
| HIMMELP5 | CUTEst | 2 | 3 | NumericalError | -5.901318e+01 |
| HS70 | CUTEst | 4 | 1 | NumericalError | 7.498464e-03 |
| LAUNCH | CUTEst | 25 | 28 | MaxIterations | 9.004902e+00 |
| LOGHAIRY | CUTEst | 2 | 0 | MaxIterations | 1.823216e-01 |
| MSS1 | CUTEst | 90 | 73 | StopAtTinyStep | -1.400000e+01 |
| NET1 | CUTEst | 48 | 57 | RestorationFailed | 9.411943e+05 |
| OET6 | CUTEst | 5 | 1002 | NumericalError | 2.069727e-03 |
| PFIT3 | CUTEst | 3 | 3 | RestorationFailed | 0.000000e+00 |
| PFIT4 | CUTEst | 3 | 3 | RestorationFailed | 0.000000e+00 |
| QCNEW | CUTEst | 9 | 3 | NumericalError | -8.065219e+02 |
| TAXR13322 | CUTEst | 72 | 1261 | NumericalError | -3.429089e+02 |

## Wins (ripopt solves, Ipopt fails) — 25 problems

| Problem | Suite | n | m | Ipopt status | ripopt obj |
|---------|-------|---|---|-------------|------------|
| AVION2 | CUTEst | 49 | 15 | MaxIterations | 9.468013e+07 |
| BEALENE | CUTEst | 2 | 3 | IpoptStatus(-10) | 0.000000e+00 |
| BIGGS6NE | CUTEst | 6 | 13 | IpoptStatus(-10) | 0.000000e+00 |
| BOX3NE | CUTEst | 3 | 10 | IpoptStatus(-10) | 0.000000e+00 |
| BROWNBSNE | CUTEst | 2 | 3 | IpoptStatus(-10) | 0.000000e+00 |
| DENSCHNBNE | CUTEst | 2 | 3 | IpoptStatus(-10) | 0.000000e+00 |
| DEVGLA1NE | CUTEst | 4 | 24 | IpoptStatus(-10) | 0.000000e+00 |
| DIAMON2DLS | CUTEst | 66 | 0 | Timeout | 6.749655e+02 |
| ENGVAL2NE | CUTEst | 3 | 5 | IpoptStatus(-10) | 0.000000e+00 |
| EQC | CUTEst | 9 | 3 | ErrorInStepComputation | -8.299064e+02 |
| EXP2NE | CUTEst | 2 | 10 | IpoptStatus(-10) | 0.000000e+00 |
| GROUPING | CUTEst | 100 | 125 | IpoptStatus(-10) | 1.385040e+01 |
| HS25NE | CUTEst | 3 | 99 | IpoptStatus(-10) | 0.000000e+00 |
| LANCZOS1 | CUTEst | 6 | 24 | IpoptStatus(-10) | 0.000000e+00 |
| LEVYMONE5 | CUTEst | 2 | 4 | IpoptStatus(-10) | 0.000000e+00 |
| NYSTROM5 | CUTEst | 18 | 20 | IpoptStatus(-10) | 0.000000e+00 |
| NYSTROM5C | CUTEst | 18 | 20 | IpoptStatus(-10) | 0.000000e+00 |
| PFIT1 | CUTEst | 3 | 3 | Infeasible | 0.000000e+00 |
| PFIT2 | CUTEst | 3 | 3 | RestorationFailed | 0.000000e+00 |
| POLAK3 | CUTEst | 12 | 10 | MaxIterations | 5.933003e+00 |
| POLAK6 | CUTEst | 5 | 4 | MaxIterations | -4.400000e+01 |
| ROBOT | CUTEst | 14 | 2 | IpoptStatus(3) | 6.593299e+00 |
| SPIRAL | CUTEst | 3 | 2 | Infeasible | -4.988610e-09 |
| Seawater speciation | Electrolyte | 15 | 8 | Infeasible | -1.348272e+00 |
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
| HS54 | CUTEst | 6 | 1 | -8.689894e-01 | -9.080748e-01 | 3.9% |

## Acceptable (not Optimal) — 18 problems

These problems converged within relaxed tolerances but not strict tolerances.

| Problem | Suite | n | m | Ipopt status | ripopt obj | Ipopt obj |
|---------|-------|---|---|-------------|------------|-----------|
| ALLINITC | CUTEst | 4 | 1 | Optimal | 3.049191e+01 | 3.049261e+01 |
| BT8 | CUTEst | 5 | 2 | Optimal | 1.000000e+00 | 1.000000e+00 |
| DECONVU | CUTEst | 63 | 0 | Optimal | 4.034019e-13 | 4.146188e-13 |
| EQC | CUTEst | 9 | 3 | ErrorInStepComputation | -8.299064e+02 | -8.651227e+02 |
| FLETCHER | CUTEst | 4 | 4 | Optimal | 1.165685e+01 | 1.165685e+01 |
| HAIFAM | CUTEst | 99 | 150 | Optimal | -4.500036e+01 | -4.500036e+01 |
| HAIFAS | CUTEst | 13 | 9 | Optimal | -4.500000e-01 | -4.500000e-01 |
| HS108 | CUTEst | 9 | 13 | Optimal | -6.749814e-01 | -5.000000e-01 |
| HS54 | CUTEst | 6 | 1 | Optimal | -8.689894e-01 | -9.080748e-01 |
| HYDC20LS | CUTEst | 99 | 0 | Optimal | 6.586238e-15 | 2.967522e-15 |
| LOGROS | CUTEst | 2 | 0 | Optimal | 0.000000e+00 | 0.000000e+00 |
| LSC2LS | CUTEst | 3 | 0 | Optimal | 1.334125e+01 | 1.333439e+01 |
| MGH10LS | CUTEst | 3 | 0 | Optimal | 8.794586e+01 | 8.794586e+01 |
| MGH17LS | CUTEst | 5 | 0 | Optimal | 1.022414e+00 | 7.898394e-05 |
| MISTAKE | CUTEst | 9 | 13 | Optimal | -1.000000e+00 | -1.000000e+00 |
| POLAK5 | CUTEst | 3 | 2 | Optimal | 5.000000e+01 | 5.000000e+01 |
| STREG | CUTEst | 4 | 0 | Optimal | 4.526251e-01 | 8.901950e-02 |
| TRO3X3 | CUTEst | 30 | 13 | Optimal | 8.999398e+00 | 8.967478e+00 |

---
*Generated by benchmark_report.py*