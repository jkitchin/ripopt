# ripopt Benchmark Report

Generated: 2026-05-22 16:34:04

## Executive Summary

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Optimal (strict) | **568/745** (76.2%) | **572/745** (76.8%) |
| Acceptable (informational, *not* counted as solved) | 20 | 5 |
| Solved exclusively (strict Optimal) | 23 | 27 |
| Both Optimal | 545 | |
| Matching objectives (< 0.01%) | 530/545 | |

> **Note:** All headline counts use strict Optimal status only. `Acceptable`
> means the iterate met relaxed tolerances but not the requested tolerance —
> per CLAUDE.md's "Honesty in Benchmarks" rule it is reported separately and
> never folded into the pass rate. See the "Acceptable (not Optimal)" and
> "Different Local Minima" sections below.

## Per-Suite Summary

| Suite | Problems | ripopt Optimal | Ipopt Optimal | ripopt only | Ipopt only | Both Optimal | Match |
|-------|----------|---------------|--------------|-------------|------------|--------------|-------|
| CUTEst | 727 | 551 (75.8%) | 556 (76.5%) | 22 | 27 | 529 | 515/529 |
| Electrolyte | 13 | 13 (100.0%) | 12 (92.3%) | 1 | 0 | 12 | 11/12 |
| Grid | 4 | 4 (100.0%) | 4 (100.0%) | 0 | 0 | 4 | 4/4 |
| CHO | 1 | 0 (0.0%) | 0 (0.0%) | 0 | 0 | 0 | 0/1 |

## CUTEst Suite — Performance

On 529 commonly-solved problems:

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Median time | 211us | 2.3ms |
| Total time | 20.74s | 19.26s |
| Mean iterations | 34.7 | 30.7 |
| Median iterations | 12 | 12 |

- **Geometric mean speedup**: 8.1x
- **Median speedup**: 10.5x
- ripopt faster: 480/529 (91%)
- ripopt 10x+ faster: 287/529
- Ipopt faster: 49/529

## Electrolyte Suite — Performance

On 12 commonly-solved problems:

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Median time | 162us | 1.2ms |
| Total time | 24.2ms | 42.4ms |
| Mean iterations | 212.8 | 19.2 |
| Median iterations | 7 | 7 |

- **Geometric mean speedup**: 4.6x
- **Median speedup**: 6.9x
- ripopt faster: 11/12 (92%)
- ripopt 10x+ faster: 2/12
- Ipopt faster: 1/12

## Grid Suite — Performance

On 4 commonly-solved problems:

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Median time | 9.3ms | 9.1ms |
| Total time | 46.6ms | 75.3ms |
| Mean iterations | 15.8 | 12.5 |
| Median iterations | 19 | 14 |

- **Geometric mean speedup**: 1.7x
- **Median speedup**: 2.3x
- ripopt faster: 3/4 (75%)
- ripopt 10x+ faster: 0/4
- Ipopt faster: 1/4

## Failure Analysis

### CUTEst Suite

| Failure Mode | ripopt | Ipopt |
|-------------|--------|-------|
| Acceptable | 20 | 5 |
| DivergingIterates | 1 | 0 |
| ErrorInStepComputation | 0 | 2 |
| EvaluationError | 3 | 0 |
| Infeasible | 0 | 11 |
| InvalidNumberDetected | 0 | 1 |
| IpoptStatus(-10) | 0 | 123 |
| IpoptStatus(3) | 0 | 1 |
| IpoptStatus(4) | 0 | 2 |
| LocalInfeasibility | 29 | 0 |
| MaxIterations | 17 | 12 |
| MaxTimeExceeded | 18 | 0 |
| NumericalError | 9 | 0 |
| RestorationFailed | 73 | 4 |
| StopAtTinyStep | 2 | 0 |
| Timeout | 4 | 10 |

### Electrolyte Suite

| Failure Mode | ripopt | Ipopt |
|-------------|--------|-------|
| Infeasible | 0 | 1 |

### CHO Suite

| Failure Mode | ripopt | Ipopt |
|-------------|--------|-------|
| MaxIter | 0 | 1 |
| MaxIterations | 1 | 0 |

## Regressions (Ipopt Optimal, ripopt not Optimal)

| Problem | Suite | n | m | ripopt status | Ipopt obj |
|---------|-------|---|---|--------------|-----------|
| ALLINITA | CUTEst | 4 | 4 | Acceptable | 3.329611e+01 |
| ALLINITC | CUTEst | 4 | 1 | Acceptable | 3.049261e+01 |
| BT8 | CUTEst | 5 | 2 | Acceptable | 1.000000e+00 |
| CRESC4 | CUTEst | 6 | 8 | NumericalError | 8.718975e-01 |
| CRESC50 | CUTEst | 6 | 100 | MaxIterations | 7.862467e-01 |
| DECONVBNE | CUTEst | 63 | 40 | RestorationFailed | 0.000000e+00 |
| DECONVU | CUTEst | 63 | 0 | Acceptable | 4.146188e-13 |
| DISCS | CUTEst | 36 | 66 | RestorationFailed | 1.200007e+01 |
| HAIFAM | CUTEst | 99 | 150 | Acceptable | -4.500036e+01 |
| HATFLDFLNE | CUTEst | 3 | 3 | RestorationFailed | 0.000000e+00 |
| HIMMELP5 | CUTEst | 2 | 3 | NumericalError | -5.901318e+01 |
| HS108 | CUTEst | 9 | 13 | Acceptable | -5.000000e-01 |
| HS54 | CUTEst | 6 | 1 | Acceptable | -9.080748e-01 |
| HS99EXP | CUTEst | 31 | 21 | Acceptable | -1.260006e+12 |
| HYDC20LS | CUTEst | 99 | 0 | Acceptable | 2.967522e-15 |
| LOGHAIRY | CUTEst | 2 | 0 | MaxIterations | 1.823216e-01 |
| LSC2LS | CUTEst | 3 | 0 | Acceptable | 1.333439e+01 |
| MGH10LS | CUTEst | 3 | 0 | Acceptable | 8.794586e+01 |
| MGH17LS | CUTEst | 5 | 0 | Acceptable | 7.898394e-05 |
| MISTAKE | CUTEst | 9 | 13 | Acceptable | -1.000000e+00 |
| MSS1 | CUTEst | 90 | 73 | StopAtTinyStep | -1.400000e+01 |
| PFIT3 | CUTEst | 3 | 3 | RestorationFailed | 0.000000e+00 |
| PFIT4 | CUTEst | 3 | 3 | RestorationFailed | 0.000000e+00 |
| POLAK5 | CUTEst | 3 | 2 | Acceptable | 5.000000e+01 |
| SSINE | CUTEst | 3 | 2 | MaxIterations | 0.000000e+00 |
| STREG | CUTEst | 4 | 0 | Acceptable | 8.901950e-02 |
| TRO3X3 | CUTEst | 30 | 13 | Acceptable | 8.967478e+00 |

## Wins (ripopt Optimal, Ipopt not Optimal) — 23 problems

| Problem | Suite | n | m | Ipopt status | ripopt obj |
|---------|-------|---|---|-------------|------------|
| BEALENE | CUTEst | 2 | 3 | IpoptStatus(-10) | 0.000000e+00 |
| BIGGS6NE | CUTEst | 6 | 13 | IpoptStatus(-10) | 0.000000e+00 |
| BOX3NE | CUTEst | 3 | 10 | IpoptStatus(-10) | 0.000000e+00 |
| DECONVB | CUTEst | 63 | 0 | MaxIterations | 2.569475e-03 |
| DECONVNE | CUTEst | 63 | 40 | Acceptable | 0.000000e+00 |
| DENSCHNDNE | CUTEst | 3 | 3 | Acceptable | 0.000000e+00 |
| DEVGLA1NE | CUTEst | 4 | 24 | IpoptStatus(-10) | 0.000000e+00 |
| DEVGLA2NE | CUTEst | 5 | 16 | IpoptStatus(-10) | 0.000000e+00 |
| DIAMON2DLS | CUTEst | 66 | 0 | Timeout | 6.749655e+02 |
| ENGVAL2NE | CUTEst | 3 | 5 | IpoptStatus(-10) | 0.000000e+00 |
| EXP2NE | CUTEst | 2 | 10 | IpoptStatus(-10) | 0.000000e+00 |
| GROUPING | CUTEst | 100 | 125 | IpoptStatus(-10) | 1.385040e+01 |
| LANCZOS1 | CUTEst | 6 | 24 | IpoptStatus(-10) | 0.000000e+00 |
| LEVYMONE5 | CUTEst | 2 | 4 | IpoptStatus(-10) | 0.000000e+00 |
| NYSTROM5 | CUTEst | 18 | 20 | IpoptStatus(-10) | 0.000000e+00 |
| NYSTROM5C | CUTEst | 18 | 20 | IpoptStatus(-10) | 0.000000e+00 |
| PFIT1 | CUTEst | 3 | 3 | Infeasible | 0.000000e+00 |
| PFIT2 | CUTEst | 3 | 3 | RestorationFailed | 0.000000e+00 |
| POLAK6 | CUTEst | 5 | 4 | MaxIterations | -4.400000e+01 |
| PRICE4NE | CUTEst | 2 | 2 | Acceptable | 0.000000e+00 |
| ROBOT | CUTEst | 14 | 2 | IpoptStatus(3) | 6.593299e+00 |
| Seawater speciation | Electrolyte | 15 | 8 | Infeasible | -1.348272e+00 |
| WACHBIEG | CUTEst | 3 | 2 | Infeasible | 1.000000e+00 |

## Acceptable (not Optimal) — 20 problems

These problems converged within relaxed tolerances but not strict tolerances.

| Problem | Suite | n | m | Ipopt status | ripopt obj | Ipopt obj |
|---------|-------|---|---|-------------|------------|-----------|
| ALLINITA | CUTEst | 4 | 4 | Optimal | 3.329799e+01 | 3.329611e+01 |
| ALLINITC | CUTEst | 4 | 1 | Optimal | 3.049193e+01 | 3.049261e+01 |
| BT8 | CUTEst | 5 | 2 | Optimal | 1.000000e+00 | 1.000000e+00 |
| DECONVU | CUTEst | 63 | 0 | Optimal | 4.034019e-13 | 4.146188e-13 |
| DENSCHNBNE | CUTEst | 2 | 3 | IpoptStatus(-10) | 0.000000e+00 | 0.000000e+00 |
| EQC | CUTEst | 9 | 3 | ErrorInStepComputation | -8.287853e+02 | -8.651227e+02 |
| HAIFAM | CUTEst | 99 | 150 | Optimal | -4.500036e+01 | -4.500036e+01 |
| HS108 | CUTEst | 9 | 13 | Optimal | -6.749814e-01 | -5.000000e-01 |
| HS25NE | CUTEst | 3 | 99 | IpoptStatus(-10) | 0.000000e+00 | 0.000000e+00 |
| HS54 | CUTEst | 6 | 1 | Optimal | -8.683914e-01 | -9.080748e-01 |
| HS99EXP | CUTEst | 31 | 21 | Optimal | -1.260006e+12 | -1.260006e+12 |
| HYDC20LS | CUTEst | 99 | 0 | Optimal | 6.586238e-15 | 2.967522e-15 |
| LSC2LS | CUTEst | 3 | 0 | Optimal | 1.334125e+01 | 1.333439e+01 |
| MGH10LS | CUTEst | 3 | 0 | Optimal | 8.794586e+01 | 8.794586e+01 |
| MGH17LS | CUTEst | 5 | 0 | Optimal | 1.022414e+00 | 7.898394e-05 |
| MISTAKE | CUTEst | 9 | 13 | Optimal | -1.000000e+00 | -1.000000e+00 |
| POLAK5 | CUTEst | 3 | 2 | Optimal | 5.000000e+01 | 5.000000e+01 |
| STREG | CUTEst | 4 | 0 | Optimal | 4.526251e-01 | 8.901950e-02 |
| TRO3X3 | CUTEst | 30 | 13 | Optimal | 7.878175e+00 | 8.967478e+00 |
| TRO4X4 | CUTEst | 63 | 25 | IpoptStatus(4) | 8.997831e+00 | -1.957476e+20 |

## Large-Scale Synthetic Problems — ripopt vs Ipopt

Synthetic problems with known structure, up to 100K variables.
Both solvers receive the exact same NlpProblem struct via the Rust trait interface.

| Problem | n | m | ripopt | iters | time | Ipopt | iters | time | speedup |
|---------|---|---|--------|-------|------|-------|-------|------|---------|
| Rosenbrock 500 | 500 | 0 | Optimal | 751 | 0.072s | Optimal | 749 | 0.188s | 2.6x |
| SparseQP 1K | 500 | 500 | Optimal | 6 | 0.005s | Optimal | 6 | 0.004s | 0.7x |
| Bratu 1K | 1,000 | 998 | Optimal | 2 | 0.002s | Optimal | 2 | 0.003s | 1.2x |
| OptControl 2.5K | 2,499 | 1,250 | Optimal | 1 | 0.010s | Optimal | 1 | 0.003s | 0.3x |
| Rosenbrock 5K | 5,000 | 0 | MaxIterations | 2999 | 3.046s | Failed | 3000 | 3.827s | 1.3x |
| Poisson 2.5K | 5,000 | 2,500 | Optimal | 1 | 0.036s | Optimal | 1 | 0.010s | 0.3x |
| Bratu 10K | 10,000 | 9,998 | Optimal | 1 | 0.020s | Optimal | 2 | 0.012s | 0.6x |
| OptControl 20K | 19,999 | 10,000 | Optimal | 1 | 0.096s | Optimal | 1 | 0.019s | 0.2x |
| Poisson 50K | 49,928 | 24,964 | Optimal | 1 | 0.507s | Optimal | 1 | 0.166s | 0.3x |
| SparseQP 100K | 50,000 | 50,000 | Optimal | 6 | 1.105s | Optimal | 6 | 0.326s | 0.3x |

ripopt: **9/10 solved** in 4.9s total
Ipopt: **9/10 solved** in 4.6s total

---
*Generated by benchmark_report.py*