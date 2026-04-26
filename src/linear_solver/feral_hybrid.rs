//! Hybrid direct/iterative wrapper for the feral backend.
//!
//! Mirrors `hybrid::HybridSolver` but with `FeralLdl` as the direct path and
//! `FeralIterativeMinres` as the iterative path. v0.8 note: because
//! `FeralIterativeMinres` currently delegates to `FeralLdl` (see
//! `feral_iterative.rs`), the hybrid behavior reduces to "direct with extra
//! refinement on retry". The crossover/timeout logic is preserved so the
//! interface can be promoted to a true hybrid in a future release without
//! changing call sites.

use super::feral_direct::FeralLdl;
use super::feral_iterative::FeralIterativeMinres;
use super::{Inertia, KktMatrix, LinearSolver, SolverError};

pub struct FeralHybrid {
    direct: FeralLdl,
    iterative: FeralIterativeMinres,
    mode: HybridMode,
    last_factor_time: f64,
    time_threshold: f64,
    iterative_failures: usize,
    max_iterative_failures: usize,
    direct_failures: usize,
    needs_refactor: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HybridMode {
    Direct,
    Iterative,
}

impl Default for FeralHybrid {
    fn default() -> Self {
        Self::new()
    }
}

impl FeralHybrid {
    pub fn new() -> Self {
        Self {
            direct: FeralLdl::new(),
            iterative: FeralIterativeMinres::new(),
            mode: HybridMode::Direct,
            last_factor_time: 0.0,
            time_threshold: 1.0,
            iterative_failures: 0,
            max_iterative_failures: 3,
            direct_failures: 0,
            needs_refactor: false,
        }
    }

    pub fn with_time_threshold(mut self, seconds: f64) -> Self {
        self.time_threshold = seconds;
        self
    }
}

impl LinearSolver for FeralHybrid {
    fn factor(&mut self, matrix: &KktMatrix) -> Result<Option<Inertia>, SolverError> {
        match self.mode {
            HybridMode::Direct => {
                let start = std::time::Instant::now();
                let result = self.direct.factor(matrix);
                self.last_factor_time = start.elapsed().as_secs_f64();

                match result {
                    Ok(inertia) => {
                        self.direct_failures = 0;
                        if self.last_factor_time > self.time_threshold {
                            log::info!(
                                "FeralHybrid: direct factor took {:.2}s (> {:.2}s), switching to iterative",
                                self.last_factor_time,
                                self.time_threshold
                            );
                            self.mode = HybridMode::Iterative;
                            self.iterative_failures = 0;
                            let iter_result = self.iterative.factor(matrix);
                            if let Ok(iter_inertia) = iter_result {
                                return Ok(iter_inertia);
                            }
                            self.mode = HybridMode::Direct;
                        }
                        Ok(inertia)
                    }
                    Err(e) => {
                        self.direct_failures += 1;
                        log::info!(
                            "FeralHybrid: direct factor failed ({}), switching to iterative",
                            e
                        );
                        self.mode = HybridMode::Iterative;
                        self.iterative_failures = 0;
                        self.iterative.factor(matrix)
                    }
                }
            }
            HybridMode::Iterative => {
                let result = self.iterative.factor(matrix);
                match result {
                    Ok(inertia) => Ok(inertia),
                    Err(e) => {
                        log::info!(
                            "FeralHybrid: iterative factor failed ({}), switching to direct",
                            e
                        );
                        self.mode = HybridMode::Direct;
                        self.direct.factor(matrix)
                    }
                }
            }
        }
    }

    fn solve(&mut self, rhs: &[f64], solution: &mut [f64]) -> Result<(), SolverError> {
        match self.mode {
            HybridMode::Direct => self.direct.solve(rhs, solution),
            HybridMode::Iterative => {
                let result = self.iterative.solve(rhs, solution);
                match result {
                    Ok(()) => {
                        self.iterative_failures = 0;
                        Ok(())
                    }
                    Err(e) => {
                        self.iterative_failures += 1;
                        if self.iterative_failures >= self.max_iterative_failures {
                            log::info!(
                                "FeralHybrid: {} consecutive iterative failures, switching back to direct",
                                self.iterative_failures
                            );
                            self.mode = HybridMode::Direct;
                            self.iterative_failures = 0;
                            self.needs_refactor = true;
                        }
                        Err(e)
                    }
                }
            }
        }
    }

    fn provides_inertia(&self) -> bool {
        true
    }

    fn min_diagonal(&self) -> Option<f64> {
        match self.mode {
            HybridMode::Direct => self.direct.min_diagonal(),
            HybridMode::Iterative => self.iterative.min_diagonal(),
        }
    }

    fn increase_quality(&mut self) -> bool {
        match self.mode {
            HybridMode::Direct => self.direct.increase_quality(),
            HybridMode::Iterative => self.iterative.increase_quality(),
        }
    }
}
