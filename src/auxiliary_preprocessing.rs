//! Internal auxiliary-system preprocessing utilities.
//!
//! This module is intentionally crate-private. The auxiliary preprocessor is an
//! implementation detail of `enable_preprocessing`; it must not expose a public
//! decomposition or transform API.

use crate::logging::rip_log;
use crate::options::SolverOptions;
use crate::problem::NlpProblem;
use crate::reduction_frame::{ReductionFrame, RemovedMultiplierRecovery};
use crate::result::{SolveResult, SolveStatus};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::time::Instant;

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EqualityBlock {
    pub(crate) rows: Vec<usize>,
    pub(crate) vars: Vec<usize>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PresolveCandidate {
    pub(crate) blocks: Vec<EqualityBlock>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AuxiliaryCouplingClass {
    PureEquality,
    ObjectiveCoupled,
    InequalityCoupled,
    ObjectiveAndInequalityCoupled,
}

impl AuxiliaryCouplingClass {
    fn label(self) -> &'static str {
        match self {
            Self::PureEquality => "pure equality",
            Self::ObjectiveCoupled => "objective-coupled",
            Self::InequalityCoupled => "inequality-coupled",
            Self::ObjectiveAndInequalityCoupled => "objective-and-inequality-coupled",
        }
    }

    fn is_coupled(self) -> bool {
        !matches!(self, Self::PureEquality)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AuxiliarySolveOutcome {
    pub(crate) x: Vec<f64>,
    pub(crate) blocks_solved: usize,
    pub(crate) max_residual: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AuxiliarySolveError {
    InvalidBlock {
        block: EqualityBlock,
        reason: &'static str,
    },
    EvaluationFailed {
        block: EqualityBlock,
    },
    TimeBudgetExceeded {
        blocks_solved: usize,
    },
    BlockSolveFailed {
        block: EqualityBlock,
        status: SolveStatus,
        residual: f64,
    },
    RankDeficientBlock {
        block: EqualityBlock,
        rank: usize,
        expected: usize,
    },
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BlockTriangularizationError {
    NonSquare {
        rows: usize,
        vars: usize,
    },
    ImperfectMatching {
        unmatched_rows: Vec<usize>,
        unmatched_vars: Vec<usize>,
    },
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AuxiliaryRejectionReason {
    EmptyComponent,
    NonSquareComponent { rows: usize, vars: usize },
    NonClosedComponent,
    CandidateRowsCoupledToRemainingVariables,
    RankDeficiency { rank: usize, expected: usize },
    BoundActiveAuxiliaryVariable,
    AuxiliarySolveFailure { status: SolveStatus },
    ResidualFailure,
    EvaluationFailure,
    TimeBudgetExceeded,
    InvalidBlock { reason: &'static str },
    CoupledAuxiliaryBlock { coupling: AuxiliaryCouplingClass },
    FullSpaceValidationFailure,
    ReductionDidNotRemoveAnything,
    NoBlocksSolved,
}

impl AuxiliaryRejectionReason {
    fn label(&self) -> &'static str {
        match self {
            Self::EmptyComponent => "empty component",
            Self::NonSquareComponent { .. } => "non-square component",
            Self::NonClosedComponent => "non-closed component",
            Self::CandidateRowsCoupledToRemainingVariables => {
                "candidate rows coupled to remaining variables"
            }
            Self::RankDeficiency { .. } => "rank deficiency",
            Self::BoundActiveAuxiliaryVariable => "bound-active auxiliary variable",
            Self::AuxiliarySolveFailure { .. } => "auxiliary solve failure",
            Self::ResidualFailure => "residual failure",
            Self::EvaluationFailure => "evaluation failure",
            Self::TimeBudgetExceeded => "time budget exceeded",
            Self::InvalidBlock { .. } => "invalid block",
            Self::CoupledAuxiliaryBlock { .. } => "coupled auxiliary block",
            Self::FullSpaceValidationFailure => "full-space validation failure",
            Self::ReductionDidNotRemoveAnything => "reduction did not remove anything",
            Self::NoBlocksSolved => "no blocks solved",
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AuxiliaryRejection {
    pub(crate) block: Option<EqualityBlock>,
    pub(crate) reason: AuxiliaryRejectionReason,
    pub(crate) detail: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct AuxiliaryCandidateDiagnostics {
    pub(crate) equality_rows: usize,
    pub(crate) incident_variables: usize,
    pub(crate) connected_components: usize,
    pub(crate) square_components: usize,
    pub(crate) closed_components: usize,
    pub(crate) btd_blocks: usize,
    pub(crate) rank_accepted_blocks: usize,
    pub(crate) pure_equality_candidates: usize,
    pub(crate) objective_coupled_candidates: usize,
    pub(crate) inequality_coupled_candidates: usize,
    pub(crate) objective_and_inequality_coupled_candidates: usize,
    pub(crate) accepted_blocks: Vec<EqualityBlock>,
    pub(crate) rejections: Vec<AuxiliaryRejection>,
}

impl AuxiliaryCandidateDiagnostics {
    fn from_incidence(incidence: &EqualityIncidence) -> Self {
        Self {
            equality_rows: incidence.row_global.len(),
            incident_variables: incidence
                .var_adj_rows
                .iter()
                .filter(|rows| !rows.is_empty())
                .count(),
            ..Self::default()
        }
    }

    pub(crate) fn rejected_blocks(&self) -> usize {
        self.rejections.len()
    }

    #[allow(dead_code)]
    pub(crate) fn accepted_block_sizes(&self) -> Vec<(usize, usize)> {
        self.accepted_blocks
            .iter()
            .map(|block| (block.rows.len(), block.vars.len()))
            .collect()
    }

    fn record_accepted_blocks(&mut self, blocks: &[EqualityBlock]) {
        self.btd_blocks += blocks.len();
        self.accepted_blocks.extend(blocks.iter().cloned());
    }

    fn record_coupling_class(&mut self, coupling: AuxiliaryCouplingClass) {
        match coupling {
            AuxiliaryCouplingClass::PureEquality => self.pure_equality_candidates += 1,
            AuxiliaryCouplingClass::ObjectiveCoupled => self.objective_coupled_candidates += 1,
            AuxiliaryCouplingClass::InequalityCoupled => self.inequality_coupled_candidates += 1,
            AuxiliaryCouplingClass::ObjectiveAndInequalityCoupled => {
                self.objective_and_inequality_coupled_candidates += 1;
            }
        }
    }

    pub(crate) fn record_rank_accepted_candidates(&mut self, candidates: &[PresolveCandidate]) {
        self.rank_accepted_blocks = candidates
            .iter()
            .map(|candidate| candidate.blocks.len())
            .sum();
    }

    fn reject_block(
        &mut self,
        block: EqualityBlock,
        reason: AuxiliaryRejectionReason,
        detail: Option<String>,
    ) {
        self.rejections.push(AuxiliaryRejection {
            block: Some(block),
            reason,
            detail,
        });
    }

    pub(crate) fn reject_global(
        &mut self,
        reason: AuxiliaryRejectionReason,
        detail: Option<String>,
    ) {
        self.rejections.push(AuxiliaryRejection {
            block: None,
            reason,
            detail,
        });
    }

    pub(crate) fn record_auxiliary_error(&mut self, err: &AuxiliarySolveError, tol: f64) {
        match err {
            AuxiliarySolveError::InvalidBlock { block, reason } => self.reject_block(
                block.clone(),
                AuxiliaryRejectionReason::InvalidBlock { reason: *reason },
                Some((*reason).to_string()),
            ),
            AuxiliarySolveError::EvaluationFailed { block } => self.reject_block(
                block.clone(),
                AuxiliaryRejectionReason::EvaluationFailure,
                None,
            ),
            AuxiliarySolveError::TimeBudgetExceeded { blocks_solved } => self.reject_global(
                AuxiliaryRejectionReason::TimeBudgetExceeded,
                Some(format!("blocks_solved={blocks_solved}")),
            ),
            AuxiliarySolveError::BlockSolveFailed {
                block,
                status,
                residual,
            } => {
                let solved = matches!(status, SolveStatus::Optimal | SolveStatus::Acceptable);
                if solved {
                    self.reject_block(
                        block.clone(),
                        AuxiliaryRejectionReason::ResidualFailure,
                        Some(format!(
                            "status={status:?}, residual={residual:.2e}, auxiliary_tol={tol:.2e}"
                        )),
                    );
                } else {
                    self.reject_block(
                        block.clone(),
                        AuxiliaryRejectionReason::AuxiliarySolveFailure { status: *status },
                        Some(format!("status={status:?}, residual={residual:.2e}")),
                    );
                }
            }
            AuxiliarySolveError::RankDeficientBlock {
                block,
                rank,
                expected,
            } => self.reject_block(
                block.clone(),
                AuxiliaryRejectionReason::RankDeficiency {
                    rank: *rank,
                    expected: *expected,
                },
                Some(format!("rank={rank}, expected={expected}")),
            ),
        }
    }

    fn record_btd_error(&mut self, block: EqualityBlock, err: BlockTriangularizationError) {
        match err {
            BlockTriangularizationError::NonSquare { rows, vars } => self.reject_block(
                block,
                AuxiliaryRejectionReason::NonSquareComponent { rows, vars },
                Some(format!("rows={rows}, vars={vars}")),
            ),
            BlockTriangularizationError::ImperfectMatching {
                unmatched_rows,
                unmatched_vars,
            } => {
                let expected = block.rows.len().min(block.vars.len());
                let missing = unmatched_rows.len().max(unmatched_vars.len());
                let rank = expected.saturating_sub(missing);
                self.reject_block(
                    block,
                    AuxiliaryRejectionReason::RankDeficiency { rank, expected },
                    Some(format!(
                        "structural matching failed, unmatched_rows={unmatched_rows:?}, unmatched_vars={unmatched_vars:?}"
                    )),
                );
            }
        }
    }

    pub(crate) fn log_verbose(&self, label: &str) {
        rip_log!(
            "ripopt: {label} diagnostics: equality_rows={}, incident_variables={}, connected_components={}, square_components={}, closed_components={}, pure_equality_candidates={}, objective_coupled_candidates={}, inequality_coupled_candidates={}, objective_and_inequality_coupled_candidates={}, btd_blocks={}, rank_accepted_blocks={}, rejected_blocks={}",
            self.equality_rows,
            self.incident_variables,
            self.connected_components,
            self.square_components,
            self.closed_components,
            self.pure_equality_candidates,
            self.objective_coupled_candidates,
            self.inequality_coupled_candidates,
            self.objective_and_inequality_coupled_candidates,
            self.btd_blocks,
            self.rank_accepted_blocks,
            self.rejected_blocks(),
        );
        if !self.accepted_blocks.is_empty() {
            rip_log!(
                "ripopt: {label} accepted block sizes: {:?}",
                self.accepted_block_sizes()
            );
            for block in &self.accepted_blocks {
                rip_log!(
                    "ripopt: {label} accepted block rows={:?} vars={:?}",
                    block.rows,
                    block.vars,
                );
            }
        }
        for rejection in &self.rejections {
            let location = match &rejection.block {
                Some(block) => format!("rows={:?} vars={:?}", block.rows, block.vars),
                None => "global".to_string(),
            };
            match &rejection.detail {
                Some(detail) => rip_log!(
                    "ripopt: {label} rejected {location}: {} ({detail})",
                    rejection.reason.label()
                ),
                None => rip_log!(
                    "ripopt: {label} rejected {location}: {}",
                    rejection.reason.label()
                ),
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct EqualityIncidence {
    pub(crate) n_vars: usize,
    pub(crate) m_orig: usize,
    pub(crate) row_global: Vec<usize>,
    pub(crate) row_local_for_global: Vec<Option<usize>>,
    pub(crate) row_adj_vars: Vec<Vec<usize>>,
    pub(crate) var_adj_rows: Vec<Vec<usize>>,
}

#[allow(dead_code)]
pub(crate) struct AuxiliaryBlockProblem<'a> {
    inner: &'a dyn NlpProblem,
    n_orig: usize,
    m_orig: usize,
    rows: Vec<usize>,
    vars: Vec<usize>,
    fixed_x: Vec<f64>,
    jac_rows: Vec<usize>,
    jac_cols: Vec<usize>,
    jac_entry_map: Vec<usize>,
    inner_jac_nnz: usize,
    hess_rows: Vec<usize>,
    hess_cols: Vec<usize>,
    hess_entry_map: Vec<usize>,
    inner_hess_nnz: usize,
}

impl<'a> AuxiliaryBlockProblem<'a> {
    pub(crate) fn new(
        inner: &'a dyn NlpProblem,
        block: &EqualityBlock,
        fixed_x: &[f64],
    ) -> Result<Self, AuxiliarySolveError> {
        let n_orig = inner.num_variables();
        let m_orig = inner.num_constraints();
        validate_auxiliary_block(block, fixed_x, n_orig, m_orig)?;

        let mut row_local_for_global = vec![None; m_orig];
        for (local, &row) in block.rows.iter().enumerate() {
            row_local_for_global[row] = Some(local);
        }
        let mut var_local_for_global = vec![None; n_orig];
        for (local, &var) in block.vars.iter().enumerate() {
            var_local_for_global[var] = Some(local);
        }

        let (inner_jac_rows, inner_jac_cols) = inner.jacobian_structure();
        let mut jac_rows = Vec::new();
        let mut jac_cols = Vec::new();
        let mut jac_entry_map = Vec::new();
        for (idx, (&row, &col)) in inner_jac_rows.iter().zip(inner_jac_cols.iter()).enumerate() {
            if row >= m_orig || col >= n_orig {
                continue;
            }
            if let (Some(local_row), Some(local_col)) =
                (row_local_for_global[row], var_local_for_global[col])
            {
                jac_rows.push(local_row);
                jac_cols.push(local_col);
                jac_entry_map.push(idx);
            }
        }

        let (inner_hess_rows, inner_hess_cols) = inner.hessian_structure();
        let mut hess_rows = Vec::new();
        let mut hess_cols = Vec::new();
        let mut hess_entry_map = Vec::new();
        for (idx, (&row, &col)) in inner_hess_rows
            .iter()
            .zip(inner_hess_cols.iter())
            .enumerate()
        {
            if row >= n_orig || col >= n_orig {
                continue;
            }
            if let (Some(local_row), Some(local_col)) =
                (var_local_for_global[row], var_local_for_global[col])
            {
                if local_row >= local_col {
                    hess_rows.push(local_row);
                    hess_cols.push(local_col);
                } else {
                    hess_rows.push(local_col);
                    hess_cols.push(local_row);
                }
                hess_entry_map.push(idx);
            }
        }

        Ok(Self {
            inner,
            n_orig,
            m_orig,
            rows: block.rows.clone(),
            vars: block.vars.clone(),
            fixed_x: fixed_x.to_vec(),
            jac_rows,
            jac_cols,
            jac_entry_map,
            inner_jac_nnz: inner_jac_rows.len(),
            hess_rows,
            hess_cols,
            hess_entry_map,
            inner_hess_nnz: inner_hess_rows.len(),
        })
    }

    fn block(&self) -> EqualityBlock {
        EqualityBlock {
            rows: self.rows.clone(),
            vars: self.vars.clone(),
        }
    }

    fn expand_x(&self, x_block: &[f64]) -> Vec<f64> {
        let mut x_full = self.fixed_x.clone();
        for (local, &var) in self.vars.iter().enumerate() {
            x_full[var] = x_block[local];
        }
        x_full
    }
}

impl NlpProblem for AuxiliaryBlockProblem<'_> {
    fn num_variables(&self) -> usize {
        self.vars.len()
    }

    fn num_constraints(&self) -> usize {
        self.rows.len()
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        let mut x_l_full = vec![0.0; self.n_orig];
        let mut x_u_full = vec![0.0; self.n_orig];
        self.inner.bounds(&mut x_l_full, &mut x_u_full);
        for (local, &var) in self.vars.iter().enumerate() {
            x_l[local] = x_l_full[var];
            x_u[local] = x_u_full[var];
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        let mut g_l_full = vec![0.0; self.m_orig];
        let mut g_u_full = vec![0.0; self.m_orig];
        self.inner.constraint_bounds(&mut g_l_full, &mut g_u_full);
        for (local, &row) in self.rows.iter().enumerate() {
            g_l[local] = g_l_full[row];
            g_u[local] = g_u_full[row];
        }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        for (local, &var) in self.vars.iter().enumerate() {
            x0[local] = self.fixed_x[var];
        }
    }

    fn objective(&self, _x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        *obj = 0.0;
        true
    }

    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        grad.fill(0.0);
        true
    }

    fn constraints(&self, x: &[f64], new_x: bool, g: &mut [f64]) -> bool {
        let x_full = self.expand_x(x);
        let mut g_full = vec![0.0; self.m_orig];
        if !self.inner.constraints(&x_full, new_x, &mut g_full) {
            return false;
        }
        for (local, &row) in self.rows.iter().enumerate() {
            g[local] = g_full[row];
        }
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.jac_rows.clone(), self.jac_cols.clone())
    }

    fn jacobian_values(&self, x: &[f64], new_x: bool, vals: &mut [f64]) -> bool {
        if self.jac_entry_map.is_empty() {
            return true;
        }
        let x_full = self.expand_x(x);
        let mut inner_vals = vec![0.0; self.inner_jac_nnz];
        if !self.inner.jacobian_values(&x_full, new_x, &mut inner_vals) {
            return false;
        }
        for (local, &inner_idx) in self.jac_entry_map.iter().enumerate() {
            vals[local] = inner_vals[inner_idx];
        }
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.hess_rows.clone(), self.hess_cols.clone())
    }

    fn hessian_values(
        &self,
        x: &[f64],
        new_x: bool,
        _obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) -> bool {
        if self.hess_entry_map.is_empty() {
            return true;
        }
        let x_full = self.expand_x(x);
        let mut lambda_full = vec![0.0; self.m_orig];
        for (local, &row) in self.rows.iter().enumerate() {
            lambda_full[row] = lambda[local];
        }

        let mut inner_vals = vec![0.0; self.inner_hess_nnz];
        if !self
            .inner
            .hessian_values(&x_full, new_x, 0.0, &lambda_full, &mut inner_vals)
        {
            return false;
        }
        for (local, &inner_idx) in self.hess_entry_map.iter().enumerate() {
            vals[local] = inner_vals[inner_idx];
        }
        true
    }
}

#[allow(dead_code)]
pub(crate) struct AuxiliaryReducedProblem<'a> {
    inner: &'a dyn NlpProblem,
    frame: ReductionFrame,
    n_orig: usize,
    m_orig: usize,
    fixed_x: Vec<f64>,
    var_map: Vec<usize>,
    constr_map: Vec<usize>,
    jac_rows: Vec<usize>,
    jac_cols: Vec<usize>,
    jac_entry_map: Vec<usize>,
    inner_jac_nnz: usize,
    hess_rows: Vec<usize>,
    hess_cols: Vec<usize>,
    hess_entry_map: Vec<usize>,
    inner_hess_nnz: usize,
}

impl<'a> AuxiliaryReducedProblem<'a> {
    pub(crate) fn new(
        inner: &'a dyn NlpProblem,
        candidates: &[PresolveCandidate],
        fixed_x: Vec<f64>,
    ) -> Result<Self, AuxiliarySolveError> {
        let n_orig = inner.num_variables();
        let m_orig = inner.num_constraints();
        if fixed_x.len() != n_orig {
            return Err(AuxiliarySolveError::InvalidBlock {
                block: EqualityBlock {
                    rows: Vec::new(),
                    vars: Vec::new(),
                },
                reason: "fixed_x length does not match problem variables",
            });
        }

        let mut fixed_vars = vec![false; n_orig];
        let mut removed_constraints = vec![false; m_orig];
        for candidate in candidates {
            for block in &candidate.blocks {
                validate_auxiliary_block(block, &fixed_x, n_orig, m_orig)?;
                let rank = auxiliary_block_jacobian_rank(inner, block, &fixed_x)?;
                let expected = block.vars.len();
                if rank < expected {
                    return Err(AuxiliarySolveError::RankDeficientBlock {
                        block: block.clone(),
                        rank,
                        expected,
                    });
                }
                for &var in &block.vars {
                    fixed_vars[var] = true;
                }
                for &row in &block.rows {
                    removed_constraints[row] = true;
                }
            }
        }

        let var_map: Vec<_> = (0..n_orig).filter(|&var| !fixed_vars[var]).collect();
        let constr_map: Vec<_> = (0..m_orig)
            .filter(|&row| !removed_constraints[row])
            .collect();

        let mut orig_to_reduced_var = vec![None; n_orig];
        for (reduced, &orig) in var_map.iter().enumerate() {
            orig_to_reduced_var[orig] = Some(reduced);
        }
        let mut orig_to_reduced_constr = vec![None; m_orig];
        for (reduced, &orig) in constr_map.iter().enumerate() {
            orig_to_reduced_constr[orig] = Some(reduced);
        }

        let (inner_jac_rows, inner_jac_cols) = inner.jacobian_structure();
        let mut jac_rows = Vec::new();
        let mut jac_cols = Vec::new();
        let mut jac_entry_map = Vec::new();
        for (idx, (&row, &col)) in inner_jac_rows.iter().zip(inner_jac_cols.iter()).enumerate() {
            if row >= m_orig || col >= n_orig {
                continue;
            }
            if let (Some(reduced_row), Some(reduced_col)) =
                (orig_to_reduced_constr[row], orig_to_reduced_var[col])
            {
                jac_rows.push(reduced_row);
                jac_cols.push(reduced_col);
                jac_entry_map.push(idx);
            }
        }

        let (inner_hess_rows, inner_hess_cols) = inner.hessian_structure();
        let mut hess_rows = Vec::new();
        let mut hess_cols = Vec::new();
        let mut hess_entry_map = Vec::new();
        for (idx, (&row, &col)) in inner_hess_rows
            .iter()
            .zip(inner_hess_cols.iter())
            .enumerate()
        {
            if row >= n_orig || col >= n_orig {
                continue;
            }
            if let (Some(reduced_row), Some(reduced_col)) =
                (orig_to_reduced_var[row], orig_to_reduced_var[col])
            {
                if reduced_row >= reduced_col {
                    hess_rows.push(reduced_row);
                    hess_cols.push(reduced_col);
                } else {
                    hess_rows.push(reduced_col);
                    hess_cols.push(reduced_row);
                }
                hess_entry_map.push(idx);
            }
        }

        let frame = ReductionFrame::new(
            n_orig,
            m_orig,
            var_map.clone(),
            constr_map.clone(),
            fixed_x.clone(),
            RemovedMultiplierRecovery::AuxiliaryStationarity,
        );

        Ok(Self {
            inner,
            frame,
            n_orig,
            m_orig,
            fixed_x,
            var_map,
            constr_map,
            jac_rows,
            jac_cols,
            jac_entry_map,
            inner_jac_nnz: inner_jac_rows.len(),
            hess_rows,
            hess_cols,
            hess_entry_map,
            inner_hess_nnz: inner_hess_rows.len(),
        })
    }

    pub(crate) fn did_reduce(&self) -> bool {
        self.frame.did_reduce()
    }

    pub(crate) fn num_fixed(&self) -> usize {
        self.frame.num_removed_vars()
    }

    pub(crate) fn num_removed_constraints(&self) -> usize {
        self.frame.num_removed_rows()
    }

    pub(crate) fn reduced_x_scaling(&self, scaling: &[f64]) -> Option<Vec<f64>> {
        self.frame.reduced_x_scaling(scaling)
    }

    pub(crate) fn reduced_g_scaling(&self, scaling: &[f64]) -> Option<Vec<f64>> {
        self.frame.reduced_g_scaling(scaling)
    }

    pub(crate) fn reduction_frame(&self) -> &ReductionFrame {
        &self.frame
    }

    fn expand_x(&self, x_reduced: &[f64]) -> Vec<f64> {
        self.frame.expand_x(x_reduced)
    }

    #[cfg(test)]
    pub(crate) fn unmap_solution(&self, reduced: &SolveResult) -> SolveResult {
        self.frame.unmap_solution(self.inner, reduced)
    }
}

impl NlpProblem for AuxiliaryReducedProblem<'_> {
    fn num_variables(&self) -> usize {
        self.var_map.len()
    }

    fn num_constraints(&self) -> usize {
        self.constr_map.len()
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        let mut x_l_full = vec![0.0; self.n_orig];
        let mut x_u_full = vec![0.0; self.n_orig];
        self.inner.bounds(&mut x_l_full, &mut x_u_full);
        for (reduced, &orig) in self.var_map.iter().enumerate() {
            x_l[reduced] = x_l_full[orig];
            x_u[reduced] = x_u_full[orig];
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        let mut g_l_full = vec![0.0; self.m_orig];
        let mut g_u_full = vec![0.0; self.m_orig];
        self.inner.constraint_bounds(&mut g_l_full, &mut g_u_full);
        for (reduced, &orig) in self.constr_map.iter().enumerate() {
            g_l[reduced] = g_l_full[orig];
            g_u[reduced] = g_u_full[orig];
        }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        for (reduced, &orig) in self.var_map.iter().enumerate() {
            x0[reduced] = self.fixed_x[orig];
        }
    }

    fn initial_multipliers(&self, lam_g: &mut [f64], z_l: &mut [f64], z_u: &mut [f64]) -> bool {
        let mut lam_g_full = vec![0.0; self.m_orig];
        let mut z_l_full = vec![0.0; self.n_orig];
        let mut z_u_full = vec![0.0; self.n_orig];
        if !self
            .inner
            .initial_multipliers(&mut lam_g_full, &mut z_l_full, &mut z_u_full)
        {
            return false;
        }
        for (reduced, &orig) in self.constr_map.iter().enumerate() {
            lam_g[reduced] = lam_g_full[orig];
        }
        for (reduced, &orig) in self.var_map.iter().enumerate() {
            z_l[reduced] = z_l_full[orig];
            z_u[reduced] = z_u_full[orig];
        }
        true
    }

    fn objective(&self, x: &[f64], new_x: bool, obj: &mut f64) -> bool {
        let x_full = self.expand_x(x);
        self.inner.objective(&x_full, new_x, obj)
    }

    fn gradient(&self, x: &[f64], new_x: bool, grad: &mut [f64]) -> bool {
        let x_full = self.expand_x(x);
        let mut grad_full = vec![0.0; self.n_orig];
        if !self.inner.gradient(&x_full, new_x, &mut grad_full) {
            return false;
        }
        for (reduced, &orig) in self.var_map.iter().enumerate() {
            grad[reduced] = grad_full[orig];
        }
        true
    }

    fn constraints(&self, x: &[f64], new_x: bool, g: &mut [f64]) -> bool {
        let x_full = self.expand_x(x);
        let mut g_full = vec![0.0; self.m_orig];
        if !self.inner.constraints(&x_full, new_x, &mut g_full) {
            return false;
        }
        for (reduced, &orig) in self.constr_map.iter().enumerate() {
            g[reduced] = g_full[orig];
        }
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.jac_rows.clone(), self.jac_cols.clone())
    }

    fn jacobian_values(&self, x: &[f64], new_x: bool, vals: &mut [f64]) -> bool {
        if self.jac_entry_map.is_empty() {
            return true;
        }
        let x_full = self.expand_x(x);
        let mut inner_vals = vec![0.0; self.inner_jac_nnz];
        if !self.inner.jacobian_values(&x_full, new_x, &mut inner_vals) {
            return false;
        }
        for (reduced, &inner_idx) in self.jac_entry_map.iter().enumerate() {
            vals[reduced] = inner_vals[inner_idx];
        }
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.hess_rows.clone(), self.hess_cols.clone())
    }

    fn hessian_values(
        &self,
        x: &[f64],
        new_x: bool,
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) -> bool {
        if self.hess_entry_map.is_empty() {
            return true;
        }

        let x_full = self.expand_x(x);
        let mut lambda_full = vec![0.0; self.m_orig];
        for (reduced, &orig) in self.constr_map.iter().enumerate() {
            lambda_full[orig] = lambda[reduced];
        }

        let mut inner_vals = vec![0.0; self.inner_hess_nnz];
        if !self
            .inner
            .hessian_values(&x_full, new_x, obj_factor, &lambda_full, &mut inner_vals)
        {
            return false;
        }
        for (reduced, &inner_idx) in self.hess_entry_map.iter().enumerate() {
            vals[reduced] = inner_vals[inner_idx];
        }
        true
    }
}

fn auxiliary_block_jacobian_rank(
    inner: &dyn NlpProblem,
    block: &EqualityBlock,
    fixed_x: &[f64],
) -> Result<usize, AuxiliarySolveError> {
    let block_problem = AuxiliaryBlockProblem::new(inner, block, fixed_x)?;
    let rows = block_problem.rows.len();
    let cols = block_problem.vars.len();
    if rows == 0 || cols == 0 {
        return Ok(0);
    }

    let x_block: Vec<_> = block_problem.vars.iter().map(|&var| fixed_x[var]).collect();
    let mut jac_vals = vec![0.0; block_problem.jac_rows.len()];
    if !block_problem.jacobian_values(&x_block, true, &mut jac_vals) {
        return Err(AuxiliarySolveError::EvaluationFailed {
            block: block.clone(),
        });
    }

    let mut dense = vec![0.0; rows * cols];
    for (idx, (&row, &col)) in block_problem
        .jac_rows
        .iter()
        .zip(block_problem.jac_cols.iter())
        .enumerate()
    {
        dense[row * cols + col] += jac_vals[idx];
    }
    Ok(dense_numeric_rank(&mut dense, rows, cols))
}

fn dense_numeric_rank(matrix: &mut [f64], rows: usize, cols: usize) -> usize {
    let max_abs = matrix
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    if max_abs == 0.0 || !max_abs.is_finite() {
        return 0;
    }

    let tol = (rows.max(cols) as f64) * max_abs * 1e-10;
    let mut rank = 0usize;
    for col in 0..cols {
        let pivot_row = (rank..rows).max_by(|&a, &b| {
            matrix[a * cols + col]
                .abs()
                .partial_cmp(&matrix[b * cols + col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let Some(pivot_row) = pivot_row else {
            break;
        };
        if matrix[pivot_row * cols + col].abs() <= tol {
            continue;
        }

        if pivot_row != rank {
            for j in 0..cols {
                matrix.swap(rank * cols + j, pivot_row * cols + j);
            }
        }

        let pivot = matrix[rank * cols + col];
        for row in (rank + 1)..rows {
            let factor = matrix[row * cols + col] / pivot;
            matrix[row * cols + col] = 0.0;
            for j in (col + 1)..cols {
                matrix[row * cols + j] -= factor * matrix[rank * cols + j];
            }
        }
        rank += 1;
        if rank == rows {
            break;
        }
    }
    rank
}

pub(crate) fn solve_auxiliary_blocks(
    problem: &dyn NlpProblem,
    candidates: &[PresolveCandidate],
    options: &SolverOptions,
    solve_start: Instant,
) -> Result<AuxiliarySolveOutcome, AuxiliarySolveError> {
    let mut x_full = vec![0.0; problem.num_variables()];
    problem.initial_point(&mut x_full);
    solve_auxiliary_blocks_from(problem, candidates, options, solve_start, &mut x_full)
}

pub(crate) fn solve_auxiliary_blocks_from(
    problem: &dyn NlpProblem,
    candidates: &[PresolveCandidate],
    options: &SolverOptions,
    solve_start: Instant,
    x_full: &mut [f64],
) -> Result<AuxiliarySolveOutcome, AuxiliarySolveError> {
    let mut blocks_solved = 0;
    let mut max_residual: f64 = 0.0;

    for candidate in candidates {
        for block in &candidate.blocks {
            let block_problem = AuxiliaryBlockProblem::new(problem, block, x_full)?;
            let Some(aux_options) = auxiliary_solver_options(options, solve_start) else {
                return Err(AuxiliarySolveError::TimeBudgetExceeded { blocks_solved });
            };
            let result = crate::solve(&block_problem, &aux_options);
            let residual = auxiliary_result_residual(&block_problem, &result)?;

            if !(matches!(
                result.status,
                SolveStatus::Optimal | SolveStatus::Acceptable
            ) && residual <= options.auxiliary_tol)
            {
                return Err(AuxiliarySolveError::BlockSolveFailed {
                    block: block_problem.block(),
                    status: result.status,
                    residual,
                });
            }

            for (local, &var) in block.vars.iter().enumerate() {
                x_full[var] = result.x[local];
            }
            blocks_solved += 1;
            max_residual = max_residual.max(residual);
        }
    }

    Ok(AuxiliarySolveOutcome {
        x: x_full.to_vec(),
        blocks_solved,
        max_residual,
    })
}

fn validate_auxiliary_block(
    block: &EqualityBlock,
    fixed_x: &[f64],
    n_orig: usize,
    m_orig: usize,
) -> Result<(), AuxiliarySolveError> {
    if fixed_x.len() != n_orig {
        return Err(AuxiliarySolveError::InvalidBlock {
            block: block.clone(),
            reason: "fixed_x length does not match problem variables",
        });
    }
    if block.rows.is_empty() || block.vars.is_empty() {
        return Err(AuxiliarySolveError::InvalidBlock {
            block: block.clone(),
            reason: "block must contain at least one row and one variable",
        });
    }
    if block.rows.iter().any(|&row| row >= m_orig) {
        return Err(AuxiliarySolveError::InvalidBlock {
            block: block.clone(),
            reason: "block row is out of range",
        });
    }
    if block.vars.iter().any(|&var| var >= n_orig) {
        return Err(AuxiliarySolveError::InvalidBlock {
            block: block.clone(),
            reason: "block variable is out of range",
        });
    }
    if has_duplicates(&block.rows) {
        return Err(AuxiliarySolveError::InvalidBlock {
            block: block.clone(),
            reason: "block rows must be unique",
        });
    }
    if has_duplicates(&block.vars) {
        return Err(AuxiliarySolveError::InvalidBlock {
            block: block.clone(),
            reason: "block variables must be unique",
        });
    }
    Ok(())
}

fn has_duplicates(values: &[usize]) -> bool {
    let mut values = values.to_vec();
    values.sort_unstable();
    values.windows(2).any(|pair| pair[0] == pair[1])
}

fn auxiliary_solver_options(
    options: &SolverOptions,
    solve_start: Instant,
) -> Option<SolverOptions> {
    let mut aux_options = options.clone();
    aux_options.enable_preprocessing = false;
    aux_options.warm_start = false;
    aux_options.warm_start_y = None;
    aux_options.warm_start_z_l = None;
    aux_options.warm_start_z_u = None;
    aux_options.user_obj_scaling = None;
    aux_options.user_g_scaling = None;
    aux_options.user_x_scaling = None;
    if options.max_wall_time > 0.0 {
        let remaining = options.max_wall_time - solve_start.elapsed().as_secs_f64();
        if remaining <= 0.0 {
            return None;
        }
        aux_options.max_wall_time = remaining;
    }
    Some(aux_options)
}

fn auxiliary_result_residual(
    problem: &AuxiliaryBlockProblem<'_>,
    result: &SolveResult,
) -> Result<f64, AuxiliarySolveError> {
    let m = problem.num_constraints();
    let g = if result.constraint_values.len() == m {
        result.constraint_values.clone()
    } else {
        let mut values = vec![0.0; m];
        if !problem.constraints(&result.x, true, &mut values) {
            return Err(AuxiliarySolveError::EvaluationFailed {
                block: problem.block(),
            });
        }
        values
    };

    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    problem.constraint_bounds(&mut g_l, &mut g_u);

    let mut residual: f64 = 0.0;
    for i in 0..m {
        let violation = if !g[i].is_finite() {
            f64::INFINITY
        } else if g_l[i].is_finite() && g_u[i].is_finite() && (g_u[i] - g_l[i]).abs() <= 1e-12 {
            (g[i] - 0.5 * (g_l[i] + g_u[i])).abs()
        } else {
            let lower = if g_l[i].is_finite() {
                (g_l[i] - g[i]).max(0.0)
            } else {
                0.0
            };
            let upper = if g_u[i].is_finite() {
                (g[i] - g_u[i]).max(0.0)
            } else {
                0.0
            };
            lower.max(upper)
        };
        residual = residual.max(violation);
    }
    Ok(residual)
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BipartiteMatching {
    /// Local equality-row index -> matched original variable index.
    pub(crate) row_to_var: Vec<Option<usize>>,
    /// Original variable index -> matched local equality-row index.
    pub(crate) var_to_row: Vec<Option<usize>>,
    /// Unmatched local equality-row indices.
    pub(crate) unmatched_rows: Vec<usize>,
    /// Unmatched original variable indices.
    pub(crate) unmatched_vars: Vec<usize>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DulmageMendelsohnPartition {
    pub(crate) matching: BipartiteMatching,
    /// Overconstrained local equality-row indices.
    pub(crate) overconstrained_rows: Vec<usize>,
    /// Overconstrained original variable indices.
    pub(crate) overconstrained_vars: Vec<usize>,
    /// Square local equality-row indices.
    pub(crate) square_rows: Vec<usize>,
    /// Square original variable indices.
    pub(crate) square_vars: Vec<usize>,
    /// Underconstrained local equality-row indices.
    pub(crate) underconstrained_rows: Vec<usize>,
    /// Underconstrained original variable indices.
    pub(crate) underconstrained_vars: Vec<usize>,
    /// Unmatched local equality-row indices.
    pub(crate) unmatched_rows: Vec<usize>,
    /// Unmatched original variable indices.
    pub(crate) unmatched_vars: Vec<usize>,
}

#[allow(dead_code)]
pub(crate) fn find_presolve_candidates(
    problem: &dyn NlpProblem,
    tol: f64,
) -> Vec<PresolveCandidate> {
    find_presolve_candidates_with_diagnostics(problem, tol).0
}

#[allow(dead_code)]
pub(crate) fn find_presolve_candidates_with_diagnostics(
    problem: &dyn NlpProblem,
    tol: f64,
) -> (Vec<PresolveCandidate>, AuxiliaryCandidateDiagnostics) {
    let incidence = EqualityIncidence::from_problem(problem, tol);
    let mut diagnostics = AuxiliaryCandidateDiagnostics::from_incidence(&incidence);
    if incidence.row_global.is_empty() {
        return (Vec::new(), diagnostics);
    }

    let selected_rows: Vec<_> = (0..incidence.row_adj_vars.len()).collect();
    let selected_vars: Vec<_> = incidence
        .var_adj_rows
        .iter()
        .enumerate()
        .filter_map(|(var, rows)| (!rows.is_empty()).then_some(var))
        .collect();
    let objective_independent = objective_independent_variables(problem, tol);
    let inequality_coupled = variables_in_inequality_rows(problem, tol);

    let components = incidence
        .connected_components(&selected_rows, &selected_vars)
        .into_iter()
        .collect::<Vec<_>>();
    diagnostics.connected_components = components.len();
    let dm = incidence.dulmage_mendelsohn_partition();

    let mut candidates = Vec::new();
    for component in components {
        let whole_component_square = component.rows.len() == component.vars.len();
        if let Some(candidate) = presolve_candidate_from_component(
            &incidence,
            component.clone(),
            &objective_independent,
            &inequality_coupled,
            &mut diagnostics,
        ) {
            candidates.push(candidate);
            continue;
        }

        if whole_component_square {
            continue;
        }

        for dm_component in dm_square_components_for_component(&incidence, &dm, &component) {
            if let Some(candidate) = presolve_candidate_from_dm_square_component(
                &incidence,
                dm_component,
                &objective_independent,
                &inequality_coupled,
                &mut diagnostics,
            ) {
                candidates.push(candidate);
            }
        }
    }
    (candidates, diagnostics)
}

#[allow(dead_code)]
pub(crate) fn find_postsolve_candidates(
    problem: &dyn NlpProblem,
    tol: f64,
) -> Vec<PresolveCandidate> {
    find_postsolve_candidates_with_diagnostics(problem, tol).0
}

#[allow(dead_code)]
pub(crate) fn find_postsolve_candidates_with_diagnostics(
    problem: &dyn NlpProblem,
    tol: f64,
) -> (Vec<PresolveCandidate>, AuxiliaryCandidateDiagnostics) {
    let incidence = EqualityIncidence::from_problem(problem, tol);
    let mut diagnostics = AuxiliaryCandidateDiagnostics::from_incidence(&incidence);
    if incidence.row_global.is_empty() {
        return (Vec::new(), diagnostics);
    }

    let objective_independent = objective_independent_variables(problem, tol);
    let inequality_coupled = variables_in_inequality_rows(problem, tol);
    let selected_vars: Vec<_> = incidence
        .var_adj_rows
        .iter()
        .enumerate()
        .filter_map(|(var, rows)| {
            (!rows.is_empty()
                && objective_independent.get(var).copied().unwrap_or(false)
                && !inequality_coupled.get(var).copied().unwrap_or(true))
            .then_some(var)
        })
        .collect();
    if selected_vars.is_empty() {
        return (Vec::new(), diagnostics);
    }

    let mut selected_row = vec![false; incidence.row_adj_vars.len()];
    for &var in &selected_vars {
        for &row in &incidence.var_adj_rows[var] {
            selected_row[row] = true;
        }
    }
    let selected_rows: Vec<_> = selected_row
        .iter()
        .enumerate()
        .filter_map(|(row, &selected)| selected.then_some(row))
        .collect();

    let components = incidence
        .connected_components(&selected_rows, &selected_vars)
        .into_iter()
        .collect::<Vec<_>>();
    diagnostics.connected_components = components.len();

    let mut candidates = Vec::new();
    for component in components {
        if let Some(candidate) = postsolve_candidate_from_component(
            &incidence,
            component,
            &objective_independent,
            &inequality_coupled,
            &mut diagnostics,
        ) {
            candidates.push(candidate);
        }
    }
    (candidates, diagnostics)
}

fn presolve_candidate_from_component(
    incidence: &EqualityIncidence,
    component: EqualityBlock,
    objective_independent: &[bool],
    inequality_coupled: &[bool],
    diagnostics: &mut AuxiliaryCandidateDiagnostics,
) -> Option<PresolveCandidate> {
    if component.rows.is_empty() || component.vars.is_empty() {
        diagnostics.reject_block(component, AuxiliaryRejectionReason::EmptyComponent, None);
        return None;
    }
    if component.rows.len() != component.vars.len() {
        diagnostics.reject_block(
            component.clone(),
            AuxiliaryRejectionReason::NonSquareComponent {
                rows: component.rows.len(),
                vars: component.vars.len(),
            },
            Some(format!(
                "rows={}, vars={}",
                component.rows.len(),
                component.vars.len()
            )),
        );
        return None;
    }
    diagnostics.square_components += 1;

    let local_rows: Vec<_> = component
        .rows
        .iter()
        .map(|&row| incidence.row_local_for_global[row])
        .collect::<Option<_>>()?;

    if !is_closed_equality_component(incidence, &local_rows, &component.vars) {
        diagnostics.reject_block(
            component,
            AuxiliaryRejectionReason::NonClosedComponent,
            None,
        );
        return None;
    }
    diagnostics.closed_components += 1;

    let coupling =
        classify_auxiliary_coupling(&component, objective_independent, inequality_coupled);
    diagnostics.record_coupling_class(coupling);
    if coupling.is_coupled() {
        diagnostics.reject_block(
            component,
            AuxiliaryRejectionReason::CoupledAuxiliaryBlock { coupling },
            Some(format!(
                "coupling={}, default presolve policy accepts only pure equality auxiliary blocks",
                coupling.label()
            )),
        );
        return None;
    }

    match incidence.block_triangular_decomposition(&local_rows, &component.vars) {
        Ok(blocks) => {
            diagnostics.record_accepted_blocks(&blocks);
            Some(PresolveCandidate { blocks })
        }
        Err(err) => {
            diagnostics.record_btd_error(component, err);
            None
        }
    }
}

fn dm_square_components_for_component(
    incidence: &EqualityIncidence,
    dm: &DulmageMendelsohnPartition,
    component: &EqualityBlock,
) -> Vec<EqualityBlock> {
    let mut component_rows = vec![false; incidence.row_adj_vars.len()];
    for &row in &component.rows {
        if let Some(local_row) = incidence
            .row_local_for_global
            .get(row)
            .and_then(|&local| local)
        {
            component_rows[local_row] = true;
        }
    }

    let mut component_vars = vec![false; incidence.n_vars];
    for &var in &component.vars {
        if var < component_vars.len() {
            component_vars[var] = true;
        }
    }

    let square_rows: Vec<_> = dm
        .square_rows
        .iter()
        .copied()
        .filter(|&row| component_rows.get(row).copied().unwrap_or(false))
        .collect();
    let square_vars: Vec<_> = dm
        .square_vars
        .iter()
        .copied()
        .filter(|&var| component_vars.get(var).copied().unwrap_or(false))
        .collect();

    if square_rows.is_empty() || square_vars.is_empty() {
        return Vec::new();
    }

    incidence.connected_components(&square_rows, &square_vars)
}

fn presolve_candidate_from_dm_square_component(
    incidence: &EqualityIncidence,
    component: EqualityBlock,
    objective_independent: &[bool],
    inequality_coupled: &[bool],
    diagnostics: &mut AuxiliaryCandidateDiagnostics,
) -> Option<PresolveCandidate> {
    if component.rows.is_empty() || component.vars.is_empty() {
        diagnostics.reject_block(component, AuxiliaryRejectionReason::EmptyComponent, None);
        return None;
    }
    if component.rows.len() != component.vars.len() {
        diagnostics.reject_block(
            component.clone(),
            AuxiliaryRejectionReason::NonSquareComponent {
                rows: component.rows.len(),
                vars: component.vars.len(),
            },
            Some(format!(
                "rows={}, vars={}",
                component.rows.len(),
                component.vars.len()
            )),
        );
        return None;
    }
    diagnostics.square_components += 1;

    let local_rows: Vec<_> = component
        .rows
        .iter()
        .map(|&row| incidence.row_local_for_global[row])
        .collect::<Option<_>>()?;

    let external_vars = noncandidate_row_variables(incidence, &local_rows, &component.vars);
    if !external_vars.is_empty() {
        diagnostics.reject_block(
            component,
            AuxiliaryRejectionReason::CandidateRowsCoupledToRemainingVariables,
            Some(format!(
                "candidate equality rows depend on non-candidate variables {external_vars:?}"
            )),
        );
        return None;
    }

    if is_closed_equality_component(incidence, &local_rows, &component.vars) {
        diagnostics.closed_components += 1;
    }

    let coupling =
        classify_auxiliary_coupling(&component, objective_independent, inequality_coupled);
    diagnostics.record_coupling_class(coupling);
    if coupling.is_coupled() {
        diagnostics.reject_block(
            component,
            AuxiliaryRejectionReason::CoupledAuxiliaryBlock { coupling },
            Some(format!(
                "coupling={}, default presolve policy accepts only pure equality auxiliary blocks",
                coupling.label()
            )),
        );
        return None;
    }

    match incidence.block_triangular_decomposition(&local_rows, &component.vars) {
        Ok(blocks) => {
            diagnostics.record_accepted_blocks(&blocks);
            Some(PresolveCandidate { blocks })
        }
        Err(err) => {
            diagnostics.record_btd_error(component, err);
            None
        }
    }
}

fn classify_auxiliary_coupling(
    block: &EqualityBlock,
    objective_independent: &[bool],
    inequality_coupled: &[bool],
) -> AuxiliaryCouplingClass {
    let objective = block
        .vars
        .iter()
        .any(|&var| !objective_independent.get(var).copied().unwrap_or(false));
    let inequality = block
        .vars
        .iter()
        .any(|&var| inequality_coupled.get(var).copied().unwrap_or(true));

    match (objective, inequality) {
        (false, false) => AuxiliaryCouplingClass::PureEquality,
        (true, false) => AuxiliaryCouplingClass::ObjectiveCoupled,
        (false, true) => AuxiliaryCouplingClass::InequalityCoupled,
        (true, true) => AuxiliaryCouplingClass::ObjectiveAndInequalityCoupled,
    }
}

fn postsolve_candidate_from_component(
    incidence: &EqualityIncidence,
    component: EqualityBlock,
    objective_independent: &[bool],
    inequality_coupled: &[bool],
    diagnostics: &mut AuxiliaryCandidateDiagnostics,
) -> Option<PresolveCandidate> {
    if component.rows.is_empty() || component.vars.is_empty() {
        diagnostics.reject_block(component, AuxiliaryRejectionReason::EmptyComponent, None);
        return None;
    }
    if component.rows.len() != component.vars.len() {
        diagnostics.reject_block(
            component.clone(),
            AuxiliaryRejectionReason::NonSquareComponent {
                rows: component.rows.len(),
                vars: component.vars.len(),
            },
            Some(format!(
                "rows={}, vars={}",
                component.rows.len(),
                component.vars.len()
            )),
        );
        return None;
    }
    diagnostics.square_components += 1;

    let local_rows: Vec<_> = component
        .rows
        .iter()
        .map(|&row| incidence.row_local_for_global[row])
        .collect::<Option<_>>()?;

    if is_closed_equality_component(incidence, &local_rows, &component.vars) {
        diagnostics.closed_components += 1;
    }

    let coupling =
        classify_auxiliary_coupling(&component, objective_independent, inequality_coupled);
    diagnostics.record_coupling_class(coupling);

    match incidence.block_triangular_decomposition(&local_rows, &component.vars) {
        Ok(blocks) => {
            diagnostics.record_accepted_blocks(&blocks);
            Some(PresolveCandidate { blocks })
        }
        Err(err) => {
            diagnostics.record_btd_error(component, err);
            None
        }
    }
}

fn is_closed_equality_component(
    incidence: &EqualityIncidence,
    local_rows: &[usize],
    vars: &[usize],
) -> bool {
    let mut selected_rows = vec![false; incidence.row_adj_vars.len()];
    let mut selected_vars = vec![false; incidence.n_vars];

    for &row in local_rows {
        if row >= selected_rows.len() {
            return false;
        }
        selected_rows[row] = true;
    }
    for &var in vars {
        if var >= selected_vars.len() {
            return false;
        }
        selected_vars[var] = true;
    }

    local_rows.iter().all(|&row| {
        incidence.row_adj_vars[row]
            .iter()
            .all(|&var| selected_vars[var])
    }) && vars.iter().all(|&var| {
        incidence.var_adj_rows[var]
            .iter()
            .all(|&row| selected_rows[row])
    })
}

fn noncandidate_row_variables(
    incidence: &EqualityIncidence,
    local_rows: &[usize],
    vars: &[usize],
) -> Vec<usize> {
    let mut selected_vars = vec![false; incidence.n_vars];
    for &var in vars {
        if var < selected_vars.len() {
            selected_vars[var] = true;
        }
    }

    let mut external_vars = Vec::new();
    for &row in local_rows {
        if row >= incidence.row_adj_vars.len() {
            continue;
        }
        external_vars.extend(
            incidence.row_adj_vars[row]
                .iter()
                .copied()
                .filter(|&var| !selected_vars.get(var).copied().unwrap_or(false)),
        );
    }
    external_vars.sort_unstable();
    external_vars.dedup();
    external_vars
}

fn variables_in_inequality_rows(problem: &dyn NlpProblem, tol: f64) -> Vec<bool> {
    let n = problem.num_variables();
    let m = problem.num_constraints();
    let mut coupled = vec![false; n];
    if n == 0 || m == 0 {
        return coupled;
    }

    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    problem.constraint_bounds(&mut g_l, &mut g_u);

    let (jac_rows, jac_cols) = problem.jacobian_structure();
    for (&row, &col) in jac_rows.iter().zip(jac_cols.iter()) {
        if row >= m || col >= n {
            continue;
        }
        if !is_equality_bound(g_l[row], g_u[row], tol) {
            coupled[col] = true;
        }
    }
    coupled
}

fn objective_independent_variables(problem: &dyn NlpProblem, tol: f64) -> Vec<bool> {
    let n = problem.num_variables();
    let mut independent = vec![false; n];
    if n == 0 {
        return independent;
    }

    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    problem.bounds(&mut x_l, &mut x_u);

    let mut x0 = vec![0.0; n];
    problem.initial_point(&mut x0);

    let mut obj0 = 0.0;
    if !problem.objective(&x0, true, &mut obj0) || !obj0.is_finite() {
        return independent;
    }

    let mut grad0 = vec![0.0; n];
    if !problem.gradient(&x0, true, &mut grad0) {
        return independent;
    }

    let grad_all = perturb_all_variables(&x0, &x_l, &x_u).and_then(|x_probe| {
        let mut grad = vec![0.0; n];
        problem
            .gradient(&x_probe, true, &mut grad)
            .then_some(grad)
    });

    let tol = tol.max(1e-10);
    for var in 0..n {
        if !near_zero(grad0[var], tol) {
            continue;
        }
        if grad_all
            .as_ref()
            .is_some_and(|grad| !near_zero(grad[var], tol))
        {
            continue;
        }

        let Some(x_var) = perturb_single_variable(&x0, &x_l, &x_u, var) else {
            continue;
        };
        let mut obj_var = 0.0;
        if !problem.objective(&x_var, true, &mut obj_var) || !obj_var.is_finite() {
            continue;
        }
        let obj_scale = obj0.abs().max(obj_var.abs()).max(1.0);
        if (obj_var - obj0).abs() > tol * obj_scale {
            continue;
        }

        let mut grad_var = vec![0.0; n];
        if !problem.gradient(&x_var, true, &mut grad_var) || !near_zero(grad_var[var], tol) {
            continue;
        }

        independent[var] = true;
    }
    independent
}

fn near_zero(value: f64, tol: f64) -> bool {
    value.is_finite() && value.abs() <= tol
}

fn perturb_all_variables(x: &[f64], x_l: &[f64], x_u: &[f64]) -> Option<Vec<f64>> {
    let mut out = x.to_vec();
    let mut changed = false;
    for i in 0..x.len() {
        if let Some(value) = perturbed_value(x[i], x_l[i], x_u[i]) {
            out[i] = value;
            changed = true;
        }
    }
    changed.then_some(out)
}

fn perturb_single_variable(
    x: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    var: usize,
) -> Option<Vec<f64>> {
    let value = perturbed_value(x[var], x_l[var], x_u[var])?;
    let mut out = x.to_vec();
    out[var] = value;
    Some(out)
}

fn perturbed_value(x: f64, lower: f64, upper: f64) -> Option<f64> {
    if !x.is_finite() {
        return None;
    }
    let scale = x.abs().max(1.0);
    let min_disp = 1e-4 * scale;
    let delta = 1.0 + 0.1 * x.abs();

    for candidate in [x + delta, x - delta] {
        if candidate_is_usable(candidate, x, lower, upper, min_disp) {
            return Some(candidate);
        }
    }

    if lower.is_finite() && upper.is_finite() && upper > lower {
        for candidate in [
            0.5 * (lower + upper),
            lower + 0.25 * (upper - lower),
            upper - 0.25 * (upper - lower),
        ] {
            if candidate_is_usable(candidate, x, lower, upper, min_disp) {
                return Some(candidate);
            }
        }
    }

    None
}

fn candidate_is_usable(candidate: f64, x: f64, lower: f64, upper: f64, min_disp: f64) -> bool {
    candidate.is_finite()
        && (!lower.is_finite() || candidate >= lower)
        && (!upper.is_finite() || candidate <= upper)
        && (candidate - x).abs() >= min_disp
}

fn is_equality_bound(lower: f64, upper: f64, tol: f64) -> bool {
    lower.is_finite() && upper.is_finite() && (lower - upper).abs() <= tol
}

#[derive(Debug, Clone, Copy)]
enum BipartiteNode {
    Row(usize),
    Var(usize),
}

impl EqualityIncidence {
    #[allow(dead_code)]
    pub(crate) fn from_problem(problem: &dyn NlpProblem, tol: f64) -> Self {
        let n_vars = problem.num_variables();
        let m_orig = problem.num_constraints();

        let mut g_l = vec![0.0; m_orig];
        let mut g_u = vec![0.0; m_orig];
        if m_orig > 0 {
            problem.constraint_bounds(&mut g_l, &mut g_u);
        }

        let mut row_global = Vec::new();
        let mut row_local_for_global = vec![None; m_orig];
        for i in 0..m_orig {
            if is_equality_bound(g_l[i], g_u[i], tol) {
                row_local_for_global[i] = Some(row_global.len());
                row_global.push(i);
            }
        }

        let mut row_adj_vars = vec![Vec::new(); row_global.len()];
        let mut var_adj_rows = vec![Vec::new(); n_vars];
        let (jac_rows, jac_cols) = problem.jacobian_structure();
        for (&row, &col) in jac_rows.iter().zip(jac_cols.iter()) {
            if row >= m_orig || col >= n_vars {
                continue;
            }
            if let Some(local_row) = row_local_for_global[row] {
                row_adj_vars[local_row].push(col);
                var_adj_rows[col].push(local_row);
            }
        }

        for adj in &mut row_adj_vars {
            adj.sort_unstable();
            adj.dedup();
        }
        for adj in &mut var_adj_rows {
            adj.sort_unstable();
            adj.dedup();
        }

        Self {
            n_vars,
            m_orig,
            row_global,
            row_local_for_global,
            row_adj_vars,
            var_adj_rows,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn maximum_matching(&self) -> BipartiteMatching {
        let (row_adj_vars, _) = self.deterministic_adjacency();
        hopcroft_karp(self.n_vars, &row_adj_vars)
    }

    #[allow(dead_code)]
    pub(crate) fn dulmage_mendelsohn_partition(&self) -> DulmageMendelsohnPartition {
        let (row_adj_vars, var_adj_rows) = self.deterministic_adjacency();
        let matching = hopcroft_karp(self.n_vars, &row_adj_vars);

        let n_rows = row_adj_vars.len();
        let mut over_rows = vec![false; n_rows];
        let mut over_vars = vec![false; self.n_vars];
        let mut queue = VecDeque::new();

        for &row in &matching.unmatched_rows {
            over_rows[row] = true;
            queue.push_back(BipartiteNode::Row(row));
        }

        while let Some(node) = queue.pop_front() {
            match node {
                BipartiteNode::Row(row) => {
                    for &var in &row_adj_vars[row] {
                        if matching.row_to_var[row] == Some(var) || over_vars[var] {
                            continue;
                        }
                        over_vars[var] = true;
                        queue.push_back(BipartiteNode::Var(var));
                    }
                }
                BipartiteNode::Var(var) => {
                    if let Some(row) = matching.var_to_row[var] {
                        if !over_rows[row] {
                            over_rows[row] = true;
                            queue.push_back(BipartiteNode::Row(row));
                        }
                    }
                }
            }
        }

        let mut under_rows = vec![false; n_rows];
        let mut under_vars = vec![false; self.n_vars];
        let mut queue = VecDeque::new();

        for &var in &matching.unmatched_vars {
            under_vars[var] = true;
            queue.push_back(BipartiteNode::Var(var));
        }

        while let Some(node) = queue.pop_front() {
            match node {
                BipartiteNode::Var(var) => {
                    for &row in &var_adj_rows[var] {
                        if matching.var_to_row[var] == Some(row) || under_rows[row] {
                            continue;
                        }
                        under_rows[row] = true;
                        queue.push_back(BipartiteNode::Row(row));
                    }
                }
                BipartiteNode::Row(row) => {
                    if let Some(var) = matching.row_to_var[row] {
                        if !under_vars[var] {
                            under_vars[var] = true;
                            queue.push_back(BipartiteNode::Var(var));
                        }
                    }
                }
            }
        }

        let overconstrained_rows = indices_where(&over_rows);
        let overconstrained_vars = indices_where(&over_vars);
        let underconstrained_rows = indices_where(&under_rows);
        let underconstrained_vars = indices_where(&under_vars);
        let square_rows = (0..n_rows)
            .filter(|&row| {
                !over_rows[row] && !under_rows[row] && matching.row_to_var[row].is_some()
            })
            .collect();
        let square_vars = (0..self.n_vars)
            .filter(|&var| {
                !over_vars[var] && !under_vars[var] && matching.var_to_row[var].is_some()
            })
            .collect();

        DulmageMendelsohnPartition {
            unmatched_rows: matching.unmatched_rows.clone(),
            unmatched_vars: matching.unmatched_vars.clone(),
            matching,
            overconstrained_rows,
            overconstrained_vars,
            square_rows,
            square_vars,
            underconstrained_rows,
            underconstrained_vars,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn connected_components(
        &self,
        selected_local_rows: &[usize],
        selected_vars: &[usize],
    ) -> Vec<EqualityBlock> {
        let selected_local_rows =
            sorted_unique_bounded(selected_local_rows, self.row_adj_vars.len());
        let selected_vars = sorted_unique_bounded(selected_vars, self.n_vars);
        let mut row_selected = vec![false; self.row_adj_vars.len()];
        let mut var_selected = vec![false; self.n_vars];
        for &row in &selected_local_rows {
            row_selected[row] = true;
        }
        for &var in &selected_vars {
            var_selected[var] = true;
        }

        let (row_adj_vars, var_adj_rows) = self.deterministic_adjacency();
        let mut row_seen = vec![false; self.row_adj_vars.len()];
        let mut var_seen = vec![false; self.n_vars];
        let mut components = Vec::new();

        for &start_row in &selected_local_rows {
            if row_seen[start_row] {
                continue;
            }
            components.push(self.connected_component_from(
                BipartiteNode::Row(start_row),
                &row_selected,
                &var_selected,
                &row_adj_vars,
                &var_adj_rows,
                &mut row_seen,
                &mut var_seen,
            ));
        }

        for &start_var in &selected_vars {
            if var_seen[start_var] {
                continue;
            }
            components.push(self.connected_component_from(
                BipartiteNode::Var(start_var),
                &row_selected,
                &var_selected,
                &row_adj_vars,
                &var_adj_rows,
                &mut row_seen,
                &mut var_seen,
            ));
        }

        components.sort_by_key(equality_block_sort_key);
        components
    }

    #[allow(dead_code)]
    pub(crate) fn block_triangular_decomposition(
        &self,
        selected_local_rows: &[usize],
        selected_vars: &[usize],
    ) -> Result<Vec<EqualityBlock>, BlockTriangularizationError> {
        let selected_local_rows =
            sorted_unique_bounded(selected_local_rows, self.row_adj_vars.len());
        let selected_vars = sorted_unique_bounded(selected_vars, self.n_vars);
        if selected_local_rows.len() != selected_vars.len() {
            return Err(BlockTriangularizationError::NonSquare {
                rows: selected_local_rows.len(),
                vars: selected_vars.len(),
            });
        }

        let (row_adj_vars, _) = self.deterministic_adjacency();
        let mut compact_var_for_global = vec![None; self.n_vars];
        for (compact_var, &global_var) in selected_vars.iter().enumerate() {
            compact_var_for_global[global_var] = Some(compact_var);
        }

        let restricted_row_adj_vars: Vec<Vec<usize>> = selected_local_rows
            .iter()
            .map(|&row| {
                row_adj_vars[row]
                    .iter()
                    .filter_map(|&var| compact_var_for_global[var])
                    .collect()
            })
            .collect();

        let matching = hopcroft_karp(selected_vars.len(), &restricted_row_adj_vars);
        if !matching.unmatched_rows.is_empty() || !matching.unmatched_vars.is_empty() {
            return Err(BlockTriangularizationError::ImperfectMatching {
                unmatched_rows: matching
                    .unmatched_rows
                    .iter()
                    .map(|&row| self.row_global[selected_local_rows[row]])
                    .collect(),
                unmatched_vars: matching
                    .unmatched_vars
                    .iter()
                    .map(|&var| selected_vars[var])
                    .collect(),
            });
        }

        let mut matched_row_for_compact_var = vec![None; selected_vars.len()];
        for (row, &matched_var) in matching.row_to_var.iter().enumerate() {
            if let Some(var) = matched_var {
                matched_row_for_compact_var[var] = Some(row);
            }
        }

        let mut dependencies = vec![Vec::new(); selected_local_rows.len()];
        for (row, adj_vars) in restricted_row_adj_vars.iter().enumerate() {
            for &var in adj_vars {
                if matching.row_to_var[row] == Some(var) {
                    continue;
                }
                if let Some(predecessor_row) = matched_row_for_compact_var[var] {
                    dependencies[predecessor_row].push(row);
                }
            }
        }
        for adj in &mut dependencies {
            adj.sort_unstable();
            adj.dedup();
        }

        let components = tarjan_strongly_connected_components(&dependencies);
        Ok(topologically_order_blocks(
            &components,
            &dependencies,
            &selected_local_rows,
            &selected_vars,
            &matching.row_to_var,
            &self.row_global,
        ))
    }

    fn connected_component_from(
        &self,
        start: BipartiteNode,
        row_selected: &[bool],
        var_selected: &[bool],
        row_adj_vars: &[Vec<usize>],
        var_adj_rows: &[Vec<usize>],
        row_seen: &mut [bool],
        var_seen: &mut [bool],
    ) -> EqualityBlock {
        let mut queue = VecDeque::new();
        let mut rows = Vec::new();
        let mut vars = Vec::new();

        match start {
            BipartiteNode::Row(row) => {
                row_seen[row] = true;
                queue.push_back(BipartiteNode::Row(row));
            }
            BipartiteNode::Var(var) => {
                var_seen[var] = true;
                queue.push_back(BipartiteNode::Var(var));
            }
        }

        while let Some(node) = queue.pop_front() {
            match node {
                BipartiteNode::Row(row) => {
                    rows.push(self.row_global[row]);
                    for &var in &row_adj_vars[row] {
                        if var_selected[var] && !var_seen[var] {
                            var_seen[var] = true;
                            queue.push_back(BipartiteNode::Var(var));
                        }
                    }
                }
                BipartiteNode::Var(var) => {
                    vars.push(var);
                    for &row in &var_adj_rows[var] {
                        if row_selected[row] && !row_seen[row] {
                            row_seen[row] = true;
                            queue.push_back(BipartiteNode::Row(row));
                        }
                    }
                }
            }
        }

        rows.sort_unstable();
        vars.sort_unstable();
        EqualityBlock { rows, vars }
    }

    fn deterministic_adjacency(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut row_adj_vars = Vec::with_capacity(self.row_adj_vars.len());
        let mut var_adj_rows = vec![Vec::new(); self.n_vars];

        for (row, adj) in self.row_adj_vars.iter().enumerate() {
            let mut vars: Vec<usize> = adj
                .iter()
                .copied()
                .filter(|&var| var < self.n_vars)
                .collect();
            vars.sort_unstable();
            vars.dedup();

            for &var in &vars {
                var_adj_rows[var].push(row);
            }
            row_adj_vars.push(vars);
        }

        for adj in &mut var_adj_rows {
            adj.sort_unstable();
            adj.dedup();
        }

        (row_adj_vars, var_adj_rows)
    }
}

fn sorted_unique_bounded(indices: &[usize], upper_bound: usize) -> Vec<usize> {
    let mut indices: Vec<_> = indices
        .iter()
        .copied()
        .filter(|&idx| idx < upper_bound)
        .collect();
    indices.sort_unstable();
    indices.dedup();
    indices
}

fn equality_block_sort_key(block: &EqualityBlock) -> (usize, usize) {
    (
        block.rows.first().copied().unwrap_or(usize::MAX),
        block.vars.first().copied().unwrap_or(usize::MAX),
    )
}

fn hopcroft_karp(n_vars: usize, row_adj_vars: &[Vec<usize>]) -> BipartiteMatching {
    let n_rows = row_adj_vars.len();
    let mut row_to_var = vec![None; n_rows];
    let mut var_to_row = vec![None; n_vars];
    let mut dist = vec![usize::MAX; n_rows];

    while matching_bfs(row_adj_vars, &row_to_var, &var_to_row, &mut dist) {
        for row in 0..n_rows {
            if row_to_var[row].is_none() {
                matching_dfs(
                    row,
                    row_adj_vars,
                    &mut row_to_var,
                    &mut var_to_row,
                    &mut dist,
                );
            }
        }
    }

    let unmatched_rows = row_to_var
        .iter()
        .enumerate()
        .filter_map(|(row, var)| var.is_none().then_some(row))
        .collect();
    let unmatched_vars = var_to_row
        .iter()
        .enumerate()
        .filter_map(|(var, row)| row.is_none().then_some(var))
        .collect();

    BipartiteMatching {
        row_to_var,
        var_to_row,
        unmatched_rows,
        unmatched_vars,
    }
}

fn matching_bfs(
    row_adj_vars: &[Vec<usize>],
    row_to_var: &[Option<usize>],
    var_to_row: &[Option<usize>],
    dist: &mut [usize],
) -> bool {
    let mut queue = VecDeque::new();
    let mut found_unmatched_var = false;

    for row in 0..row_adj_vars.len() {
        if row_to_var[row].is_none() {
            dist[row] = 0;
            queue.push_back(row);
        } else {
            dist[row] = usize::MAX;
        }
    }

    while let Some(row) = queue.pop_front() {
        for &var in &row_adj_vars[row] {
            if let Some(next_row) = var_to_row[var] {
                if dist[next_row] == usize::MAX {
                    dist[next_row] = dist[row] + 1;
                    queue.push_back(next_row);
                }
            } else {
                found_unmatched_var = true;
            }
        }
    }

    found_unmatched_var
}

fn matching_dfs(
    row: usize,
    row_adj_vars: &[Vec<usize>],
    row_to_var: &mut [Option<usize>],
    var_to_row: &mut [Option<usize>],
    dist: &mut [usize],
) -> bool {
    struct Frame {
        row: usize,
        next_edge: usize,
    }

    let mut stack = vec![Frame { row, next_edge: 0 }];
    let mut path_vars = Vec::new();

    while let Some(frame) = stack.last_mut() {
        if frame.next_edge == row_adj_vars[frame.row].len() {
            dist[frame.row] = usize::MAX;
            stack.pop();
            while path_vars.len() >= stack.len() && !path_vars.is_empty() {
                path_vars.pop();
            }
            continue;
        }

        let current_row = frame.row;
        let var = row_adj_vars[current_row][frame.next_edge];
        frame.next_edge += 1;

        if let Some(next_row) = var_to_row[var] {
            if dist[next_row] == dist[current_row] + 1 {
                path_vars.push(var);
                stack.push(Frame {
                    row: next_row,
                    next_edge: 0,
                });
            }
            continue;
        }

        path_vars.push(var);
        for (frame, &path_var) in stack.iter().zip(&path_vars) {
            row_to_var[frame.row] = Some(path_var);
            var_to_row[path_var] = Some(frame.row);
        }
        return true;
    }

    false
}

fn indices_where(flags: &[bool]) -> Vec<usize> {
    flags
        .iter()
        .enumerate()
        .filter_map(|(idx, &flag)| flag.then_some(idx))
        .collect()
}

fn tarjan_strongly_connected_components(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    struct Frame {
        node: usize,
        next_edge: usize,
    }

    let n = adj.len();
    let mut next_index = 0usize;
    let mut index = vec![None; n];
    let mut lowlink = vec![0usize; n];
    let mut scc_stack = Vec::new();
    let mut on_stack = vec![false; n];
    let mut components = Vec::new();

    for start in 0..n {
        if index[start].is_some() {
            continue;
        }

        index[start] = Some(next_index);
        lowlink[start] = next_index;
        next_index += 1;
        scc_stack.push(start);
        on_stack[start] = true;

        let mut dfs_stack = vec![Frame {
            node: start,
            next_edge: 0,
        }];

        while !dfs_stack.is_empty() {
            let frame_idx = dfs_stack.len() - 1;
            let node = dfs_stack[frame_idx].node;

            if dfs_stack[frame_idx].next_edge < adj[node].len() {
                let next = adj[node][dfs_stack[frame_idx].next_edge];
                dfs_stack[frame_idx].next_edge += 1;

                if index[next].is_none() {
                    index[next] = Some(next_index);
                    lowlink[next] = next_index;
                    next_index += 1;
                    scc_stack.push(next);
                    on_stack[next] = true;
                    dfs_stack.push(Frame {
                        node: next,
                        next_edge: 0,
                    });
                } else if on_stack[next] {
                    lowlink[node] = lowlink[node].min(index[next].expect("visited node has index"));
                }
                continue;
            }

            dfs_stack.pop();

            if lowlink[node] == index[node].expect("visited node has index") {
                let mut component = Vec::new();
                while let Some(member) = scc_stack.pop() {
                    on_stack[member] = false;
                    component.push(member);
                    if member == node {
                        break;
                    }
                }
                component.sort_unstable();
                components.push(component);
            }

            if let Some(parent) = dfs_stack.last() {
                lowlink[parent.node] = lowlink[parent.node].min(lowlink[node]);
            }
        }
    }

    components
}

fn topologically_order_blocks(
    components: &[Vec<usize>],
    dependencies: &[Vec<usize>],
    selected_local_rows: &[usize],
    selected_vars: &[usize],
    row_to_compact_var: &[Option<usize>],
    row_global: &[usize],
) -> Vec<EqualityBlock> {
    let mut component_for_row = vec![0; selected_local_rows.len()];
    for (component, rows) in components.iter().enumerate() {
        for &row in rows {
            component_for_row[row] = component;
        }
    }

    let mut component_edges = vec![Vec::new(); components.len()];
    let mut indegree = vec![0usize; components.len()];
    for (from_row, edges) in dependencies.iter().enumerate() {
        let from_component = component_for_row[from_row];
        for &to_row in edges {
            let to_component = component_for_row[to_row];
            if from_component != to_component {
                component_edges[from_component].push(to_component);
            }
        }
    }
    for edges in &mut component_edges {
        edges.sort_unstable();
        edges.dedup();
        for &to_component in edges.iter() {
            indegree[to_component] += 1;
        }
    }

    let blocks: Vec<_> = components
        .iter()
        .map(|rows| {
            let mut block_rows: Vec<_> = rows
                .iter()
                .map(|&row| row_global[selected_local_rows[row]])
                .collect();
            let mut block_vars: Vec<_> = rows
                .iter()
                .filter_map(|&row| row_to_compact_var[row].map(|var| selected_vars[var]))
                .collect();
            block_rows.sort_unstable();
            block_vars.sort_unstable();
            EqualityBlock {
                rows: block_rows,
                vars: block_vars,
            }
        })
        .collect();

    let mut ordered_blocks = Vec::with_capacity(blocks.len());
    let mut ready = BinaryHeap::new();
    for component in 0..components.len() {
        if indegree[component] == 0 {
            ready.push(Reverse((
                equality_block_sort_key(&blocks[component]),
                component,
            )));
        }
    }

    while let Some(Reverse((_key, component))) = ready.pop() {
        ordered_blocks.push(blocks[component].clone());
        for &next in &component_edges[component] {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                ready.push(Reverse((equality_block_sort_key(&blocks[next]), next)));
            }
        }
    }

    ordered_blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::PreprocessedProblem;
    use crate::reduction_frame::{solve_dense_square_system, ReductionStack};

    const TOL: f64 = 1e-10;

    #[derive(Clone)]
    struct GraphProblem {
        n: usize,
        gl: Vec<f64>,
        gu: Vec<f64>,
        edges: Vec<(usize, usize)>,
        objective_vars: Vec<usize>,
    }

    impl NlpProblem for GraphProblem {
        fn num_variables(&self) -> usize {
            self.n
        }

        fn num_constraints(&self) -> usize {
            self.gl.len()
        }

        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            for i in 0..self.n {
                x_l[i] = f64::NEG_INFINITY;
                x_u[i] = f64::INFINITY;
            }
        }

        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l.copy_from_slice(&self.gl);
            g_u.copy_from_slice(&self.gu);
        }

        fn initial_point(&self, x0: &mut [f64]) {
            x0.fill(0.0);
        }

        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = self.objective_vars.iter().map(|&var| x[var]).sum();
            true
        }

        fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad.fill(0.0);
            for &var in &self.objective_vars {
                grad[var] = 1.0;
            }
            true
        }

        fn constraints(&self, _x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g.fill(0.0);
            true
        }

        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            self.edges.iter().copied().unzip()
        }

        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals.fill(1.0);
            true
        }

        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![], vec![])
        }

        fn hessian_values(
            &self,
            _x: &[f64],
            _new_x: bool,
            _obj_factor: f64,
            _lambda: &[f64],
            _vals: &mut [f64],
        ) -> bool {
            true
        }
    }

    fn graph_problem(n: usize, bounds: &[(f64, f64)], edges: &[(usize, usize)]) -> GraphProblem {
        GraphProblem {
            n,
            gl: bounds.iter().map(|b| b.0).collect(),
            gu: bounds.iter().map(|b| b.1).collect(),
            edges: edges.to_vec(),
            objective_vars: Vec::new(),
        }
    }

    fn graph_problem_with_objective(
        n: usize,
        bounds: &[(f64, f64)],
        edges: &[(usize, usize)],
        objective_vars: &[usize],
    ) -> GraphProblem {
        GraphProblem {
            n,
            gl: bounds.iter().map(|b| b.0).collect(),
            gu: bounds.iter().map(|b| b.1).collect(),
            edges: edges.to_vec(),
            objective_vars: objective_vars.to_vec(),
        }
    }

    fn equality_bounds(n_rows: usize) -> Vec<(f64, f64)> {
        vec![(0.0, 0.0); n_rows]
    }

    fn equality_incidence(
        n_vars: usize,
        n_rows: usize,
        edges: &[(usize, usize)],
    ) -> EqualityIncidence {
        let bounds = equality_bounds(n_rows);
        let problem = graph_problem(n_vars, &bounds, edges);
        EqualityIncidence::from_problem(&problem, TOL)
    }

    fn all_indices(n: usize) -> Vec<usize> {
        (0..n).collect()
    }

    fn sorted_values(mut values: Vec<usize>) -> Vec<usize> {
        values.sort_unstable();
        values.dedup();
        values
    }

    fn matching_cardinality(matching: &BipartiteMatching) -> usize {
        matching
            .row_to_var
            .iter()
            .filter(|var| var.is_some())
            .count()
    }

    fn assert_perfect_matching(incidence: &EqualityIncidence) {
        let matching = incidence.maximum_matching();
        assert!(matching.unmatched_rows.is_empty());
        assert!(matching.unmatched_vars.is_empty());
        assert_eq!(
            matching_cardinality(&matching),
            incidence.row_adj_vars.len()
        );
    }

    fn sorted_block_pairs(blocks: &[EqualityBlock]) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut pairs: Vec<_> = blocks
            .iter()
            .map(|block| {
                let mut rows = block.rows.clone();
                let mut vars = block.vars.clone();
                rows.sort_unstable();
                vars.sort_unstable();
                (rows, vars)
            })
            .collect();
        pairs.sort();
        pairs
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum PyomoGasVar {
        Rho,
        Pressure,
        Flow,
        Temperature,
    }

    impl PyomoGasVar {
        fn component_index(self) -> usize {
            match self {
                PyomoGasVar::Rho => 0,
                PyomoGasVar::Pressure => 1,
                PyomoGasVar::Flow => 2,
                PyomoGasVar::Temperature => 3,
            }
        }
    }

    fn pyomo_gas_expansion_incidence(
        n_model: usize,
        fixed_vars: &[(PyomoGasVar, usize)],
    ) -> EqualityIncidence {
        // Mirrors pyomo/contrib/incidence_analysis/tests/models_for_testing.py
        // make_gas_expansion_model. Variable order follows Pyomo component
        // declaration order: rho, P, F, T, each indexed by stream.
        let streams = n_model + 1;
        let mut var_index = vec![vec![None; streams]; 4];
        let mut n_vars = 0;
        for kind in [
            PyomoGasVar::Rho,
            PyomoGasVar::Pressure,
            PyomoGasVar::Flow,
            PyomoGasVar::Temperature,
        ] {
            for stream in 0..streams {
                if fixed_vars.contains(&(kind, stream)) {
                    continue;
                }
                var_index[kind.component_index()][stream] = Some(n_vars);
                n_vars += 1;
            }
        }

        let mut edges = Vec::new();
        let mut push_var = |row: usize, kind: PyomoGasVar, stream: usize| {
            if let Some(var) = var_index[kind.component_index()][stream] {
                edges.push((row, var));
            }
        };

        let mut row = 0;
        for stream in 1..=n_model {
            // mbal[i]: rho[i-1] * F[i-1] - rho[i] * F[i] == 0
            push_var(row, PyomoGasVar::Rho, stream - 1);
            push_var(row, PyomoGasVar::Flow, stream - 1);
            push_var(row, PyomoGasVar::Rho, stream);
            push_var(row, PyomoGasVar::Flow, stream);
            row += 1;
        }
        for stream in 1..=n_model {
            // ebal[i]: rho[i-1] * F[i-1] * T[i-1] - rho[i] * F[i] * T[i] + Q == 0
            push_var(row, PyomoGasVar::Rho, stream - 1);
            push_var(row, PyomoGasVar::Flow, stream - 1);
            push_var(row, PyomoGasVar::Temperature, stream - 1);
            push_var(row, PyomoGasVar::Rho, stream);
            push_var(row, PyomoGasVar::Flow, stream);
            push_var(row, PyomoGasVar::Temperature, stream);
            row += 1;
        }
        for stream in 1..=n_model {
            // expansion[i]: P[i] / P[i-1] - (rho[i] / rho[i-1])**gamma == 0
            push_var(row, PyomoGasVar::Pressure, stream);
            push_var(row, PyomoGasVar::Pressure, stream - 1);
            push_var(row, PyomoGasVar::Rho, stream);
            push_var(row, PyomoGasVar::Rho, stream - 1);
            row += 1;
        }
        for stream in 0..=n_model {
            // ideal_gas[i]: P[i] - rho[i] * R * T[i] == 0
            push_var(row, PyomoGasVar::Pressure, stream);
            push_var(row, PyomoGasVar::Rho, stream);
            push_var(row, PyomoGasVar::Temperature, stream);
            row += 1;
        }

        equality_incidence(n_vars, row, &edges)
    }

    fn quiet_aux_options() -> SolverOptions {
        let mut options = SolverOptions::default();
        options.print_level = 0;
        options.max_iter = 200;
        options.auxiliary_tol = 1e-7;
        options
    }

    struct TriangularAuxProblem;

    impl NlpProblem for TriangularAuxProblem {
        fn num_variables(&self) -> usize {
            2
        }

        fn num_constraints(&self) -> usize {
            2
        }

        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = 0.1;
            x_u[0] = 10.0;
            x_l[1] = 0.0;
            x_u[1] = 10.0;
        }

        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 4.0;
            g_u[0] = 4.0;
            g_l[1] = 0.0;
            g_u[1] = 0.0;
        }

        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 1.0;
            x0[1] = 0.0;
        }

        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = x[0] + x[1];
            true
        }

        fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = 1.0;
            grad[1] = 1.0;
            true
        }

        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g[0] = x[0] * x[0];
            g[1] = x[1] - x[0] - 1.0;
            true
        }

        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1, 1], vec![0, 0, 1])
        }

        fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals[0] = 2.0 * x[0];
            vals[1] = -1.0;
            vals[2] = 1.0;
            true
        }

        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0], vec![0])
        }

        fn hessian_values(
            &self,
            _x: &[f64],
            _new_x: bool,
            _obj_factor: f64,
            lambda: &[f64],
            vals: &mut [f64],
        ) -> bool {
            vals[0] = 2.0 * lambda[0];
            true
        }
    }

    struct InfeasibleBoundAuxProblem;

    impl NlpProblem for InfeasibleBoundAuxProblem {
        fn num_variables(&self) -> usize {
            1
        }

        fn num_constraints(&self) -> usize {
            1
        }

        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = 0.0;
            x_u[0] = 1.0;
        }

        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 2.0;
            g_u[0] = 2.0;
        }

        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 0.5;
        }

        fn objective(&self, _x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = 0.0;
            true
        }

        fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = 0.0;
            true
        }

        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g[0] = x[0];
            true
        }

        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0], vec![0])
        }

        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals[0] = 1.0;
            true
        }

        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![], vec![])
        }

        fn hessian_values(
            &self,
            _x: &[f64],
            _new_x: bool,
            _obj_factor: f64,
            _lambda: &[f64],
            _vals: &mut [f64],
        ) -> bool {
            true
        }
    }

    struct ReducedMappingProblem;

    impl NlpProblem for ReducedMappingProblem {
        fn num_variables(&self) -> usize {
            4
        }

        fn num_constraints(&self) -> usize {
            4
        }

        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = -10.0;
            x_u[0] = 10.0;
            x_l[1] = 2.0;
            x_u[1] = 2.0;
            x_l[2] = -20.0;
            x_u[2] = 20.0;
            x_l[3] = 5.0;
            x_u[3] = 5.0;
        }

        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 0.0;
            g_u[0] = 0.0;
            g_l[1] = f64::NEG_INFINITY;
            g_u[1] = 100.0;
            g_l[2] = 0.0;
            g_u[2] = 0.0;
            g_l[3] = 1.0;
            g_u[3] = 1.0;
        }

        fn initial_point(&self, x0: &mut [f64]) {
            x0.copy_from_slice(&[9.0, 2.0, 8.0, 5.0]);
        }

        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = x[0] * x[0] + 4.0 * x[0] * x[2] + 3.0 * x[2] * x[2] + 5.0 * x[1] + 7.0 * x[3];
            true
        }

        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = 2.0 * x[0] + 4.0 * x[2];
            grad[1] = 5.0;
            grad[2] = 4.0 * x[0] + 6.0 * x[2];
            grad[3] = 7.0;
            true
        }

        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g[0] = x[1] - 2.0;
            g[1] = x[0] + 10.0 * x[1] + 2.0 * x[2];
            g[2] = x[3] - 5.0;
            g[3] = x[0] * x[0] + x[2] - 3.0 * x[3];
            true
        }

        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1, 1, 1, 2, 3, 3, 3], vec![1, 0, 1, 2, 3, 0, 2, 3])
        }

        fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals[0] = 1.0;
            vals[1] = 1.0;
            vals[2] = 10.0;
            vals[3] = 2.0;
            vals[4] = 1.0;
            vals[5] = 2.0 * x[0];
            vals[6] = 1.0;
            vals[7] = -3.0;
            true
        }

        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1, 2, 2, 3, 3], vec![0, 0, 0, 2, 2, 3])
        }

        fn hessian_values(
            &self,
            _x: &[f64],
            _new_x: bool,
            obj_factor: f64,
            lambda: &[f64],
            vals: &mut [f64],
        ) -> bool {
            vals[0] = 2.0 * obj_factor + 2.0 * lambda[3];
            vals[1] = 0.0;
            vals[2] = 4.0 * obj_factor;
            vals[3] = 6.0 * obj_factor;
            vals[4] = 0.0;
            vals[5] = 0.0;
            true
        }
    }

    struct KnownAuxMultiplierProblem {
        bound_active_aux: bool,
        objective_slope: f64,
    }

    impl NlpProblem for KnownAuxMultiplierProblem {
        fn num_variables(&self) -> usize {
            2
        }

        fn num_constraints(&self) -> usize {
            1
        }

        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = if self.bound_active_aux {
                2.0
            } else {
                f64::NEG_INFINITY
            };
            x_u[0] = if self.bound_active_aux {
                2.0
            } else {
                f64::INFINITY
            };
            x_l[1] = f64::NEG_INFINITY;
            x_u[1] = f64::INFINITY;
        }

        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 0.0;
            g_u[0] = 0.0;
        }

        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 2.0;
            x0[1] = 5.0;
        }

        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = self.objective_slope * x[0] + (x[1] - 5.0) * (x[1] - 5.0);
            true
        }

        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = self.objective_slope;
            grad[1] = 2.0 * (x[1] - 5.0);
            true
        }

        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g[0] = x[0] - 2.0;
            true
        }

        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0], vec![0])
        }

        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals[0] = 1.0;
            true
        }

        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![1], vec![1])
        }

        fn hessian_values(
            &self,
            _x: &[f64],
            _new_x: bool,
            obj_factor: f64,
            _lambda: &[f64],
            vals: &mut [f64],
        ) -> bool {
            vals[0] = 2.0 * obj_factor;
            true
        }
    }

    struct ComposedReductionProblem;

    impl NlpProblem for ComposedReductionProblem {
        fn num_variables(&self) -> usize {
            3
        }

        fn num_constraints(&self) -> usize {
            3
        }

        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l[0] = -10.0;
            x_u[0] = 10.0;
            x_l[1] = f64::NEG_INFINITY;
            x_u[1] = f64::INFINITY;
            x_l[2] = 5.0;
            x_u[2] = 5.0;
        }

        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 0.0;
            g_u[0] = 0.0;
            g_l[1] = 6.0;
            g_u[1] = 6.0;
            g_l[2] = 6.0;
            g_u[2] = 6.0;
        }

        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 0.0;
            x0[1] = 2.0;
            x0[2] = 5.0;
        }

        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = 0.5 * x[0] * x[0] + 3.0 * x[1];
            true
        }

        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = x[0];
            grad[1] = 3.0;
            grad[2] = 0.0;
            true
        }

        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g[0] = x[1] - 2.0;
            g[1] = x[0] + x[2];
            g[2] = x[0] + x[2];
            true
        }

        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1, 1, 2, 2], vec![1, 0, 2, 0, 2])
        }

        fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals.fill(1.0);
            true
        }

        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0], vec![0])
        }

        fn hessian_values(
            &self,
            _x: &[f64],
            _new_x: bool,
            obj_factor: f64,
            _lambda: &[f64],
            vals: &mut [f64],
        ) -> bool {
            vals[0] = obj_factor;
            true
        }
    }

    struct RankDeficientAuxProblem;

    impl NlpProblem for RankDeficientAuxProblem {
        fn num_variables(&self) -> usize {
            2
        }

        fn num_constraints(&self) -> usize {
            2
        }

        fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
            x_l.fill(f64::NEG_INFINITY);
            x_u.fill(f64::INFINITY);
        }

        fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
            g_l[0] = 0.0;
            g_u[0] = 0.0;
            g_l[1] = f64::NEG_INFINITY;
            g_u[1] = 10.0;
        }

        fn initial_point(&self, x0: &mut [f64]) {
            x0[0] = 0.0;
            x0[1] = 1.0;
        }

        fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
            *obj = x[1] * x[1];
            true
        }

        fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
            grad[0] = 0.0;
            grad[1] = 2.0 * x[1];
            true
        }

        fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
            g[0] = x[0] * x[0];
            g[1] = x[1];
            true
        }

        fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1], vec![0, 1])
        }

        fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
            vals[0] = 2.0 * x[0];
            vals[1] = 1.0;
            true
        }

        fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
            (vec![0, 1], vec![0, 1])
        }

        fn hessian_values(
            &self,
            _x: &[f64],
            _new_x: bool,
            obj_factor: f64,
            lambda: &[f64],
            vals: &mut [f64],
        ) -> bool {
            vals[0] = 2.0 * lambda[0];
            vals[1] = 2.0 * obj_factor;
            true
        }
    }

    fn reduced_mapping_problem<'a>(problem: &'a dyn NlpProblem) -> AuxiliaryReducedProblem<'a> {
        let candidates = vec![PresolveCandidate {
            blocks: vec![
                EqualityBlock {
                    rows: vec![0],
                    vars: vec![1],
                },
                EqualityBlock {
                    rows: vec![2],
                    vars: vec![3],
                },
            ],
        }];
        AuxiliaryReducedProblem::new(problem, &candidates, vec![9.0, 2.0, 8.0, 5.0]).unwrap()
    }

    #[test]
    fn auxiliary_reduced_problem_expands_evaluations_through_fixed_auxiliaries() {
        let problem = ReducedMappingProblem;
        let reduced = reduced_mapping_problem(&problem);

        assert!(reduced.did_reduce());
        assert_eq!(reduced.num_fixed(), 2);
        assert_eq!(reduced.num_removed_constraints(), 2);
        assert_eq!(reduced.num_variables(), 2);
        assert_eq!(reduced.num_constraints(), 2);
        assert_eq!(reduced.var_map, vec![0, 2]);
        assert_eq!(reduced.constr_map, vec![1, 3]);
        assert_eq!(
            reduced.reduced_x_scaling(&[1.0, 2.0, 3.0, 4.0]),
            Some(vec![1.0, 3.0])
        );
        assert_eq!(
            reduced.reduced_g_scaling(&[10.0, 20.0, 30.0, 40.0]),
            Some(vec![20.0, 40.0])
        );

        let mut x0 = vec![0.0; 2];
        reduced.initial_point(&mut x0);
        assert_eq!(x0, vec![9.0, 8.0]);

        let mut x_l = vec![0.0; 2];
        let mut x_u = vec![0.0; 2];
        reduced.bounds(&mut x_l, &mut x_u);
        assert_eq!(x_l, vec![-10.0, -20.0]);
        assert_eq!(x_u, vec![10.0, 20.0]);

        let mut g_l = vec![0.0; 2];
        let mut g_u = vec![0.0; 2];
        reduced.constraint_bounds(&mut g_l, &mut g_u);
        assert_eq!(g_l, vec![f64::NEG_INFINITY, 1.0]);
        assert_eq!(g_u, vec![100.0, 1.0]);

        let x_reduced = vec![11.0, 13.0];
        let mut obj = 0.0;
        assert!(reduced.objective(&x_reduced, true, &mut obj));
        assert_eq!(obj, 1245.0);

        let mut grad = vec![0.0; 2];
        assert!(reduced.gradient(&x_reduced, true, &mut grad));
        assert_eq!(grad, vec![74.0, 122.0]);

        let mut g = vec![0.0; 2];
        assert!(reduced.constraints(&x_reduced, true, &mut g));
        assert_eq!(g, vec![57.0, 119.0]);
    }

    #[test]
    fn auxiliary_reduced_problem_remaps_jacobian_entries() {
        let problem = ReducedMappingProblem;
        let reduced = reduced_mapping_problem(&problem);

        let (rows, cols) = reduced.jacobian_structure();
        assert_eq!(rows, vec![0, 0, 1, 1]);
        assert_eq!(cols, vec![0, 1, 0, 1]);

        let mut vals = vec![0.0; rows.len()];
        assert!(reduced.jacobian_values(&[11.0, 13.0], true, &mut vals));
        assert_eq!(vals, vec![1.0, 2.0, 22.0, 1.0]);
    }

    #[test]
    fn auxiliary_reduced_problem_remaps_sparse_lower_hessian_entries() {
        let problem = ReducedMappingProblem;
        let reduced = reduced_mapping_problem(&problem);

        let (rows, cols) = reduced.hessian_structure();
        assert_eq!(rows, vec![0, 1, 1]);
        assert_eq!(cols, vec![0, 0, 1]);

        let mut vals = vec![0.0; rows.len()];
        assert!(reduced.hessian_values(&[11.0, 13.0], true, 0.5, &[7.0, 11.0], &mut vals));
        assert_eq!(vals, vec![23.0, 2.0, 3.0]);
    }

    #[test]
    fn auxiliary_reduced_problem_unmaps_full_solution() {
        let problem = ReducedMappingProblem;
        let reduced = reduced_mapping_problem(&problem);
        let reduced_result = SolveResult {
            x: vec![11.0, 13.0],
            objective: -1.0,
            constraint_multipliers: vec![7.0, 11.0],
            bound_multipliers_lower: vec![0.1, 0.2],
            bound_multipliers_upper: vec![0.3, 0.4],
            constraint_values: vec![-1.0, -1.0],
            status: SolveStatus::Optimal,
            iterations: 12,
            diagnostics: Default::default(),
        };

        let full = reduced.unmap_solution(&reduced_result);

        assert_eq!(full.x, vec![11.0, 2.0, 13.0, 5.0]);
        assert_eq!(full.objective, 1245.0);
        assert_eq!(full.constraint_values, vec![0.0, 57.0, 0.0, 119.0]);
        assert_eq!(full.constraint_multipliers, vec![0.0, 7.0, 0.0, 11.0]);
        assert_eq!(full.bound_multipliers_lower, vec![0.1, 0.0, 0.2, 0.0]);
        assert_eq!(full.bound_multipliers_upper, vec![0.3, 0.0, 0.4, 0.0]);
        assert_eq!(full.status, SolveStatus::Optimal);
        assert_eq!(full.iterations, 12);
    }

    #[test]
    fn reduction_stack_composes_auxiliary_then_standard_unmapping() {
        let problem = ComposedReductionProblem;
        let candidates = vec![PresolveCandidate {
            blocks: vec![EqualityBlock {
                rows: vec![0],
                vars: vec![1],
            }],
        }];
        let auxiliary = AuxiliaryReducedProblem::new(&problem, &candidates, vec![0.0, 2.0, 5.0])
            .expect("auxiliary reduction");
        let standard = PreprocessedProblem::new(&auxiliary as &dyn NlpProblem, 1e-2);

        assert_eq!(auxiliary.var_map, vec![0, 2]);
        assert_eq!(auxiliary.constr_map, vec![1, 2]);
        assert!(standard.did_reduce());
        assert_eq!(standard.num_fixed(), 1);
        assert_eq!(standard.num_redundant(), 1);

        let nested_result = SolveResult {
            x: vec![1.0],
            objective: -1.0,
            constraint_multipliers: vec![7.0],
            bound_multipliers_lower: vec![0.1],
            bound_multipliers_upper: vec![0.3],
            constraint_values: vec![6.0],
            status: SolveStatus::Optimal,
            iterations: 4,
            diagnostics: Default::default(),
        };

        let full = ReductionStack::new()
            .push(standard.reduction_frame(), &auxiliary as &dyn NlpProblem)
            .push(auxiliary.reduction_frame(), &problem as &dyn NlpProblem)
            .unmap_solution_with_options(&nested_result, None);

        assert_eq!(full.x, vec![1.0, 2.0, 5.0]);
        assert!((full.objective - 6.5).abs() < 1e-12);
        assert_eq!(full.constraint_values, vec![0.0, 6.0, 6.0]);
        assert_eq!(full.constraint_multipliers, vec![-3.0, 7.0, 0.0]);
        assert_eq!(full.bound_multipliers_lower, vec![0.1, 0.0, 0.0]);
        assert_eq!(full.bound_multipliers_upper, vec![0.3, 0.0, 0.0]);
        assert_eq!(full.status, SolveStatus::Optimal);
        assert_eq!(full.iterations, 4);
    }

    #[test]
    fn auxiliary_reduced_problem_reconstructs_removed_multiplier() {
        let problem = KnownAuxMultiplierProblem {
            bound_active_aux: false,
            objective_slope: 3.0,
        };
        let candidates = vec![PresolveCandidate {
            blocks: vec![EqualityBlock {
                rows: vec![0],
                vars: vec![0],
            }],
        }];
        let reduced = AuxiliaryReducedProblem::new(&problem, &candidates, vec![2.0, 5.0])
            .expect("auxiliary reduction");
        let reduced_result = SolveResult {
            x: vec![5.0],
            objective: 0.0,
            constraint_multipliers: vec![],
            bound_multipliers_lower: vec![0.0],
            bound_multipliers_upper: vec![0.0],
            constraint_values: vec![],
            status: SolveStatus::Optimal,
            iterations: 1,
            diagnostics: Default::default(),
        };

        let full = reduced.unmap_solution(&reduced_result);

        assert_eq!(full.x, vec![2.0, 5.0]);
        assert_eq!(full.constraint_values, vec![0.0]);
        assert!(
            (full.constraint_multipliers[0] + 3.0).abs() < 1e-10,
            "lambda_aux={}, expected -3",
            full.constraint_multipliers[0]
        );
    }

    #[test]
    fn auxiliary_reduced_problem_reconstructs_large_removed_multiplier() {
        let problem = KnownAuxMultiplierProblem {
            bound_active_aux: false,
            objective_slope: 1.0e12,
        };
        let candidates = vec![PresolveCandidate {
            blocks: vec![EqualityBlock {
                rows: vec![0],
                vars: vec![0],
            }],
        }];
        let reduced = AuxiliaryReducedProblem::new(&problem, &candidates, vec![2.0, 5.0])
            .expect("auxiliary reduction");
        let reduced_result = SolveResult {
            x: vec![5.0],
            objective: 0.0,
            constraint_multipliers: vec![],
            bound_multipliers_lower: vec![0.0],
            bound_multipliers_upper: vec![0.0],
            constraint_values: vec![],
            status: SolveStatus::Optimal,
            iterations: 1,
            diagnostics: Default::default(),
        };

        let full = reduced.unmap_solution(&reduced_result);

        let expected = -1.0e12;
        assert!(
            (full.constraint_multipliers[0] - expected).abs() <= expected.abs() * 1e-12,
            "lambda_aux={}, expected {expected}",
            full.constraint_multipliers[0]
        );
    }

    #[test]
    fn auxiliary_reduced_problem_skips_multiplier_reconstruction_for_bound_active_auxiliary() {
        let problem = KnownAuxMultiplierProblem {
            bound_active_aux: true,
            objective_slope: 3.0,
        };
        let candidates = vec![PresolveCandidate {
            blocks: vec![EqualityBlock {
                rows: vec![0],
                vars: vec![0],
            }],
        }];
        let reduced = AuxiliaryReducedProblem::new(&problem, &candidates, vec![2.0, 5.0])
            .expect("auxiliary reduction");
        let reduced_result = SolveResult {
            x: vec![5.0],
            objective: 0.0,
            constraint_multipliers: vec![],
            bound_multipliers_lower: vec![0.0],
            bound_multipliers_upper: vec![0.0],
            constraint_values: vec![],
            status: SolveStatus::Optimal,
            iterations: 1,
            diagnostics: Default::default(),
        };

        let full = reduced.unmap_solution(&reduced_result);

        assert_eq!(full.x, vec![2.0, 5.0]);
        assert_eq!(full.constraint_values, vec![0.0]);
        assert_eq!(
            full.constraint_multipliers[0], 0.0,
            "bound-active auxiliary variables should leave removed multipliers conservative"
        );
    }

    #[test]
    fn auxiliary_multiplier_reconstruction_rejects_singular_system() {
        let matrix = vec![1.0, 2.0, 2.0, 4.0];
        let rhs = vec![1.0, 2.0];

        let result = solve_dense_square_system(matrix, rhs, 2);

        assert!(
            result.is_err(),
            "singular multiplier systems should not be reconstructed"
        );
    }

    #[test]
    fn auxiliary_reduced_problem_rejects_rank_deficient_auxiliary_block() {
        let problem = RankDeficientAuxProblem;
        let candidates = vec![PresolveCandidate {
            blocks: vec![EqualityBlock {
                rows: vec![0],
                vars: vec![0],
            }],
        }];

        let err = match AuxiliaryReducedProblem::new(&problem, &candidates, vec![0.0, 1.0]) {
            Ok(_) => panic!("rank-deficient auxiliary block should not reduce"),
            Err(err) => err,
        };

        match err {
            AuxiliarySolveError::RankDeficientBlock {
                block,
                rank,
                expected,
            } => {
                assert_eq!(block.rows, vec![0]);
                assert_eq!(block.vars, vec![0]);
                assert_eq!(rank, 0);
                assert_eq!(expected, 1);
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn auxiliary_solve_handles_one_variable_nonlinear_equality() {
        let problem = TriangularAuxProblem;
        let candidates = vec![PresolveCandidate {
            blocks: vec![EqualityBlock {
                rows: vec![0],
                vars: vec![0],
            }],
        }];
        let options = quiet_aux_options();

        let outcome =
            solve_auxiliary_blocks(&problem, &candidates, &options, std::time::Instant::now())
                .expect("auxiliary solve");

        assert_eq!(outcome.blocks_solved, 1);
        assert!(outcome.max_residual <= options.auxiliary_tol);
        assert!((outcome.x[0] - 2.0).abs() < 1e-5, "x = {:?}", outcome.x);
        assert_eq!(outcome.x[1], 0.0);
    }

    #[test]
    fn auxiliary_solve_updates_full_vector_between_triangular_blocks() {
        let problem = TriangularAuxProblem;
        let candidates = vec![PresolveCandidate {
            blocks: vec![
                EqualityBlock {
                    rows: vec![0],
                    vars: vec![0],
                },
                EqualityBlock {
                    rows: vec![1],
                    vars: vec![1],
                },
            ],
        }];
        let options = quiet_aux_options();

        let outcome =
            solve_auxiliary_blocks(&problem, &candidates, &options, std::time::Instant::now())
                .expect("auxiliary solve");

        assert_eq!(outcome.blocks_solved, 2);
        assert!(outcome.max_residual <= options.auxiliary_tol);
        assert!((outcome.x[0] - 2.0).abs() < 1e-5, "x = {:?}", outcome.x);
        assert!((outcome.x[1] - 3.0).abs() < 1e-5, "x = {:?}", outcome.x);
    }

    #[test]
    fn auxiliary_solve_failure_returns_structured_failure() {
        let problem = InfeasibleBoundAuxProblem;
        let candidates = find_presolve_candidates(&problem, TOL);
        let options = quiet_aux_options();

        let err =
            solve_auxiliary_blocks(&problem, &candidates, &options, std::time::Instant::now())
                .unwrap_err();

        match err {
            AuxiliarySolveError::BlockSolveFailed {
                block,
                status: _,
                residual,
            } => {
                assert_eq!(block.rows, vec![0]);
                assert_eq!(block.vars, vec![0]);
                assert!(residual > options.auxiliary_tol || !residual.is_finite());
            }
            other => panic!("unexpected auxiliary error: {:?}", other),
        }
    }

    #[test]
    fn auxiliary_solve_stops_when_outer_wall_time_is_exhausted() {
        let problem = TriangularAuxProblem;
        let candidates = vec![PresolveCandidate {
            blocks: vec![EqualityBlock {
                rows: vec![0],
                vars: vec![0],
            }],
        }];
        let mut options = quiet_aux_options();
        options.max_wall_time = 0.01;
        let expired_start = std::time::Instant::now() - std::time::Duration::from_secs(1);

        let err =
            solve_auxiliary_blocks(&problem, &candidates, &options, expired_start).unwrap_err();

        assert_eq!(
            err,
            AuxiliarySolveError::TimeBudgetExceeded { blocks_solved: 0 }
        );
    }

    #[test]
    fn auxiliary_block_problem_respects_original_variable_bounds() {
        let problem = InfeasibleBoundAuxProblem;
        let block = EqualityBlock {
            rows: vec![0],
            vars: vec![0],
        };
        let fixed_x = vec![0.5];
        let aux = AuxiliaryBlockProblem::new(&problem, &block, &fixed_x).unwrap();

        let mut x_l = vec![0.0; 1];
        let mut x_u = vec![0.0; 1];
        aux.bounds(&mut x_l, &mut x_u);
        assert_eq!(x_l, vec![0.0]);
        assert_eq!(x_u, vec![1.0]);

        let result = crate::solve(&aux, &quiet_aux_options());
        assert!(
            result.x[0] >= -1e-10 && result.x[0] <= 1.0 + 1e-10,
            "auxiliary solution violated bounds: {:?}",
            result.x
        );
    }

    #[test]
    fn incidence_handles_no_constraints() {
        let problem = graph_problem(2, &[], &[]);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        assert_eq!(inc.n_vars, 2);
        assert_eq!(inc.m_orig, 0);
        assert!(inc.row_global.is_empty());
        assert!(inc.row_local_for_global.is_empty());
        assert!(inc.row_adj_vars.is_empty());
        assert_eq!(inc.var_adj_rows, vec![Vec::<usize>::new(), Vec::new()]);
    }

    #[test]
    fn incidence_ignores_inequality_rows() {
        let problem = graph_problem(
            3,
            &[(0.0, 1.0), (2.0, f64::INFINITY), (f64::NEG_INFINITY, 0.0)],
            &[(0, 0), (1, 1), (2, 2)],
        );
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        assert!(inc.row_global.is_empty());
        assert_eq!(inc.row_local_for_global, vec![None, None, None]);
        assert!(inc.row_adj_vars.is_empty());
        assert_eq!(
            inc.var_adj_rows,
            vec![Vec::<usize>::new(), Vec::new(), Vec::new()]
        );
    }

    #[test]
    fn incidence_detects_equality_rows_within_tolerance() {
        let problem = graph_problem(
            3,
            &[(1.0, 1.0 + TOL * 0.5), (0.0, 1.0), (2.0, 2.0)],
            &[(0, 0), (1, 1), (2, 2)],
        );
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        assert_eq!(inc.row_global, vec![0, 2]);
        assert_eq!(inc.row_local_for_global, vec![Some(0), None, Some(1)]);
        assert_eq!(inc.row_adj_vars, vec![vec![0], vec![2]]);
        assert_eq!(inc.var_adj_rows, vec![vec![0], Vec::new(), vec![1]]);
    }

    #[test]
    fn incidence_deduplicates_and_sorts_structural_edges() {
        let problem = graph_problem(
            4,
            &[(0.0, 0.0), (1.0, 1.0)],
            &[(0, 3), (0, 1), (0, 3), (1, 2), (1, 0), (1, 2)],
        );
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        assert_eq!(inc.row_global, vec![0, 1]);
        assert_eq!(inc.row_adj_vars, vec![vec![1, 3], vec![0, 2]]);
        assert_eq!(inc.var_adj_rows, vec![vec![1], vec![0], vec![1], vec![0]]);
    }

    #[test]
    fn incidence_matches_mathprogincidence_matrix_and_graph_order_examples() {
        // Mirrors MathProgIncidence.jl/test/incidence_matrix.jl and
        // test/incidence_graph.jl construction-from-constraints cases.
        let inc = equality_incidence(3, 2, &[(0, 0), (0, 1), (1, 1), (1, 2)]);
        assert_eq!(inc.row_adj_vars, vec![vec![0, 1], vec![1, 2]]);
        assert_eq!(inc.var_adj_rows, vec![vec![0], vec![0, 1], vec![1]]);

        // Same two constraints with the variable order reversed. MathProg's
        // expected sparse matrix columns are [3, 2] for eq1 and [2, 1] for eq2
        // in 1-based indexing; here that is [2, 1] and [1, 0].
        let inc = equality_incidence(3, 2, &[(0, 2), (0, 1), (1, 1), (1, 0)]);
        assert_eq!(inc.row_adj_vars, vec![vec![1, 2], vec![0, 1]]);
        assert_eq!(inc.var_adj_rows, vec![vec![1], vec![0, 1], vec![0]]);
    }

    #[test]
    fn incidence_keeps_empty_equality_rows() {
        let problem = graph_problem(2, &[(3.0, 3.0), (0.0, 1.0)], &[(1, 0)]);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        assert_eq!(inc.row_global, vec![0]);
        assert_eq!(inc.row_adj_vars, vec![Vec::<usize>::new()]);
        assert_eq!(inc.var_adj_rows, vec![Vec::<usize>::new(), Vec::new()]);
    }

    #[test]
    fn incidence_ignores_out_of_range_structure_entries() {
        let problem = graph_problem(
            3,
            &[(0.0, 0.0), (1.0, 1.0)],
            &[(0, 0), (99, 1), (1, 99), (1, 2)],
        );
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        assert_eq!(inc.row_adj_vars, vec![vec![0], vec![2]]);
        assert_eq!(inc.var_adj_rows, vec![vec![0], Vec::new(), vec![1]]);
    }

    #[test]
    fn matching_and_dm_square_1x1() {
        let bounds = equality_bounds(1);
        let problem = graph_problem(1, &bounds, &[(0, 0)]);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let matching = inc.maximum_matching();
        assert_eq!(matching.row_to_var, vec![Some(0)]);
        assert_eq!(matching.var_to_row, vec![Some(0)]);
        assert!(matching.unmatched_rows.is_empty());
        assert!(matching.unmatched_vars.is_empty());

        let dm = inc.dulmage_mendelsohn_partition();
        assert_eq!(dm.square_rows, vec![0]);
        assert_eq!(dm.square_vars, vec![0]);
        assert!(dm.overconstrained_rows.is_empty());
        assert!(dm.overconstrained_vars.is_empty());
        assert!(dm.underconstrained_rows.is_empty());
        assert!(dm.underconstrained_vars.is_empty());
        assert!(dm.unmatched_rows.is_empty());
        assert!(dm.unmatched_vars.is_empty());
    }

    #[test]
    fn matching_square_2x2_is_deterministic() {
        let bounds = equality_bounds(2);
        let problem = graph_problem(2, &bounds, &[(0, 1), (1, 1), (0, 0), (1, 0), (0, 1)]);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let matching = inc.maximum_matching();
        assert_eq!(matching.row_to_var, vec![Some(0), Some(1)]);
        assert_eq!(matching.var_to_row, vec![Some(0), Some(1)]);
        assert!(matching.unmatched_rows.is_empty());
        assert!(matching.unmatched_vars.is_empty());

        let dm = inc.dulmage_mendelsohn_partition();
        assert_eq!(dm.square_rows, vec![0, 1]);
        assert_eq!(dm.square_vars, vec![0, 1]);
        assert!(dm.overconstrained_rows.is_empty());
        assert!(dm.underconstrained_rows.is_empty());
    }

    #[test]
    fn matching_reroutes_along_augmenting_path() {
        let bounds = equality_bounds(2);
        let problem = graph_problem(2, &bounds, &[(0, 0), (0, 1), (1, 0)]);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let matching = inc.maximum_matching();
        assert_eq!(matching.row_to_var, vec![Some(1), Some(0)]);
        assert_eq!(matching.var_to_row, vec![Some(1), Some(0)]);
        assert!(matching.unmatched_rows.is_empty());
        assert!(matching.unmatched_vars.is_empty());
    }

    #[test]
    fn matching_matches_mathprogincidence_matrix_example() {
        // Mirrors MathProgIncidence.jl/test/interface.jl::test_maximum_matching_matrix.
        // Julia uses 1-based matrix indices; these assertions use 0-based row/var indices.
        let bounds = equality_bounds(3);
        let problem = graph_problem(3, &bounds, &[(0, 1), (1, 2), (2, 0)]);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let matching = inc.maximum_matching();

        assert_eq!(matching.row_to_var, vec![Some(1), Some(2), Some(0)]);
        assert_eq!(matching.var_to_row, vec![Some(2), Some(0), Some(1)]);
        assert!(matching.unmatched_rows.is_empty());
        assert!(matching.unmatched_vars.is_empty());
    }

    #[test]
    fn matching_matches_pyomo_matrix_gallery_cases() {
        // Mirrors pyomo/contrib/incidence_analysis/tests/test_matching.py.
        let n = 5;

        let identity_edges: Vec<_> = (0..n).map(|i| (i, i)).collect();
        let inc = equality_incidence(n, n, &identity_edges);
        let matching = inc.maximum_matching();
        assert_eq!(matching.row_to_var, (0..n).map(Some).collect::<Vec<_>>());
        assert_eq!(matching.var_to_row, (0..n).map(Some).collect::<Vec<_>>());

        let omit = n / 2;
        let low_rank_diagonal: Vec<_> = (0..n).filter(|&i| i != omit).map(|i| (i, i)).collect();
        let inc = equality_incidence(n, n, &low_rank_diagonal);
        let matching = inc.maximum_matching();
        assert_eq!(matching_cardinality(&matching), n - 1);
        assert_eq!(matching.unmatched_rows, vec![omit]);
        assert_eq!(matching.unmatched_vars, vec![omit]);

        let mut bordered = Vec::new();
        for i in 0..n - 1 {
            bordered.push((n - 1, i));
            bordered.push((i, n - 1));
            bordered.push((i, i));
        }
        assert_perfect_matching(&equality_incidence(n, n, &bordered));

        let mut hessenberg = Vec::new();
        for i in 0..n {
            hessenberg.push((n - 1, i));
            if i == 0 {
                hessenberg.push((0, i));
            } else {
                hessenberg.push((i - 1, i));
            }
        }
        assert_perfect_matching(&equality_incidence(n, n, &hessenberg));

        let mut low_rank_hessenberg = Vec::new();
        for i in 0..n {
            low_rank_hessenberg.push((n - 1, i));
            if i == 0 {
                low_rank_hessenberg.push((0, i));
            } else if i != omit {
                low_rank_hessenberg.push((i - 1, i));
            }
        }
        let matching = equality_incidence(n, n, &low_rank_hessenberg).maximum_matching();
        assert_eq!(matching_cardinality(&matching), n - 1);
        assert!(matching.row_to_var[0].is_some());
        assert!(matching.row_to_var[n - 1].is_some());
        assert!(matching.var_to_row[0].is_some());
        assert!(matching.var_to_row[n - 1].is_some());

        let mut nondecomposable_hessenberg = Vec::new();
        for i in 0..n {
            nondecomposable_hessenberg.push((n - 1, i));
            nondecomposable_hessenberg.push((i, i));
            if i != 0 {
                nondecomposable_hessenberg.push((i - 1, i));
            }
        }
        assert_perfect_matching(&equality_incidence(n, n, &nondecomposable_hessenberg));

        let mut low_rank_nondecomposable_hessenberg = Vec::new();
        for i in 0..n - 1 {
            low_rank_nondecomposable_hessenberg.push((i + 1, i));
            low_rank_nondecomposable_hessenberg.push((i, i + 1));
        }
        let matching =
            equality_incidence(n, n, &low_rank_nondecomposable_hessenberg).maximum_matching();
        assert_eq!(matching_cardinality(&matching), n - 1);
        assert_eq!(matching.unmatched_rows.len(), 1);
        assert_eq!(matching.unmatched_vars.len(), 1);
    }

    #[test]
    fn matching_handles_long_augmenting_path_iteratively() {
        let chain_len = 10_000;
        let bounds = equality_bounds(chain_len + 1);
        let mut edges = Vec::with_capacity(2 * chain_len + 1);
        edges.push((0, 0));
        edges.push((0, chain_len));
        for row in 1..chain_len {
            edges.push((row, row - 1));
            edges.push((row, row));
        }
        edges.push((chain_len, chain_len - 1));

        let problem = graph_problem(chain_len + 1, &bounds, &edges);
        let inc = EqualityIncidence::from_problem(&problem, TOL);
        let matching = inc.maximum_matching();

        assert!(matching.unmatched_rows.is_empty());
        assert!(matching.unmatched_vars.is_empty());
        assert_eq!(matching.row_to_var[0], Some(chain_len));
        for row in 1..chain_len {
            assert_eq!(matching.row_to_var[row], Some(row - 1));
        }
        assert_eq!(matching.row_to_var[chain_len], Some(chain_len - 1));
    }

    #[test]
    fn dm_underconstrained_1_row_2_variables() {
        let bounds = equality_bounds(1);
        let problem = graph_problem(2, &bounds, &[(0, 1), (0, 0)]);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let dm = inc.dulmage_mendelsohn_partition();
        assert_eq!(dm.matching.row_to_var, vec![Some(0)]);
        assert_eq!(dm.matching.var_to_row, vec![Some(0), None]);
        assert!(dm.unmatched_rows.is_empty());
        assert_eq!(dm.unmatched_vars, vec![1]);
        assert!(dm.overconstrained_rows.is_empty());
        assert!(dm.overconstrained_vars.is_empty());
        assert!(dm.square_rows.is_empty());
        assert!(dm.square_vars.is_empty());
        assert_eq!(dm.underconstrained_rows, vec![0]);
        assert_eq!(dm.underconstrained_vars, vec![0, 1]);
    }

    #[test]
    fn dm_overconstrained_2_rows_1_variable() {
        let bounds = equality_bounds(2);
        let problem = graph_problem(1, &bounds, &[(1, 0), (0, 0)]);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let dm = inc.dulmage_mendelsohn_partition();
        assert_eq!(dm.matching.row_to_var, vec![Some(0), None]);
        assert_eq!(dm.matching.var_to_row, vec![Some(0)]);
        assert_eq!(dm.unmatched_rows, vec![1]);
        assert!(dm.unmatched_vars.is_empty());
        assert_eq!(dm.overconstrained_rows, vec![0, 1]);
        assert_eq!(dm.overconstrained_vars, vec![0]);
        assert!(dm.square_rows.is_empty());
        assert!(dm.square_vars.is_empty());
        assert!(dm.underconstrained_rows.is_empty());
        assert!(dm.underconstrained_vars.is_empty());
    }

    #[test]
    fn dm_mixed_square_over_and_under_blocks() {
        let bounds = equality_bounds(5);
        let problem = graph_problem(
            5,
            &bounds,
            &[(0, 0), (1, 1), (2, 2), (3, 2), (4, 4), (4, 3)],
        );
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let dm = inc.dulmage_mendelsohn_partition();
        assert_eq!(
            dm.matching.row_to_var,
            vec![Some(0), Some(1), Some(2), None, Some(3)]
        );
        assert_eq!(
            dm.matching.var_to_row,
            vec![Some(0), Some(1), Some(2), Some(4), None]
        );
        assert_eq!(dm.unmatched_rows, vec![3]);
        assert_eq!(dm.unmatched_vars, vec![4]);
        assert_eq!(dm.overconstrained_rows, vec![2, 3]);
        assert_eq!(dm.overconstrained_vars, vec![2]);
        assert_eq!(dm.square_rows, vec![0, 1]);
        assert_eq!(dm.square_vars, vec![0, 1]);
        assert_eq!(dm.underconstrained_rows, vec![4]);
        assert_eq!(dm.underconstrained_vars, vec![3, 4]);
    }

    #[test]
    fn dm_matches_mathprogincidence_matrix_example() {
        // Mirrors MathProgIncidence.jl/test/interface.jl::test_dulmage_mendelsohn_matrix.
        // Julia uses 1-based matrix indices; these assertions use 0-based row/var indices.
        let bounds = equality_bounds(4);
        let problem = graph_problem(
            4,
            &bounds,
            &[(0, 0), (0, 1), (0, 3), (1, 2), (2, 3), (3, 3)],
        );
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let dm = inc.dulmage_mendelsohn_partition();

        assert_eq!(dm.underconstrained_rows, vec![0]);
        assert_eq!(dm.square_rows, vec![1]);
        assert_eq!(dm.square_vars, vec![2]);
        assert_eq!(dm.overconstrained_vars, vec![3]);

        let mut over_or_unmatched_rows = dm.overconstrained_rows.clone();
        over_or_unmatched_rows.extend_from_slice(&dm.unmatched_rows);
        over_or_unmatched_rows.sort_unstable();
        over_or_unmatched_rows.dedup();
        assert_eq!(over_or_unmatched_rows, vec![2, 3]);

        let mut under_or_unmatched_vars = dm.underconstrained_vars.clone();
        under_or_unmatched_vars.extend_from_slice(&dm.unmatched_vars);
        under_or_unmatched_vars.sort_unstable();
        under_or_unmatched_vars.dedup();
        assert_eq!(under_or_unmatched_vars, vec![0, 1]);
    }

    #[test]
    fn dm_matches_pyomo_gas_expansion_matrix_cases() {
        // Mirrors pyomo/contrib/incidence_analysis/tests/test_dulmage_mendelsohn.py
        // gas-expansion structural matrix cases.
        use PyomoGasVar::*;

        let inc = pyomo_gas_expansion_incidence(4, &[(Flow, 0), (Rho, 0), (Temperature, 0)]);
        assert_eq!(inc.row_adj_vars.len(), 17);
        assert_eq!(inc.n_vars, 17);
        assert_perfect_matching(&inc);

        let dm = inc.dulmage_mendelsohn_partition();
        assert!(dm.unmatched_rows.is_empty());
        assert!(dm.unmatched_vars.is_empty());
        assert!(dm.overconstrained_rows.is_empty());
        assert!(dm.overconstrained_vars.is_empty());
        assert!(dm.underconstrained_rows.is_empty());
        assert!(dm.underconstrained_vars.is_empty());
        assert_eq!(dm.square_rows, all_indices(17));
        assert_eq!(dm.square_vars, all_indices(17));

        let inc = pyomo_gas_expansion_incidence(1, &[(Pressure, 0), (Rho, 0), (Temperature, 0)]);
        assert_eq!(inc.row_adj_vars.len(), 5);
        assert_eq!(inc.n_vars, 5);

        let dm = inc.dulmage_mendelsohn_partition();
        // Row 3 is ideal_gas[0], whose incident variables are all fixed.
        assert_eq!(dm.unmatched_rows, vec![3]);
        assert_eq!(dm.overconstrained_rows, vec![3]);
        assert_eq!(dm.underconstrained_rows, vec![0, 1, 2, 4]);
        assert_eq!(
            sorted_values(dm.underconstrained_vars.clone()),
            all_indices(5)
        );
        assert!(dm.square_rows.is_empty());
        assert!(dm.square_vars.is_empty());

        let inc = pyomo_gas_expansion_incidence(2, &[]);
        assert_eq!(inc.row_adj_vars.len(), 9);
        assert_eq!(inc.n_vars, 12);

        let dm = inc.dulmage_mendelsohn_partition();
        assert!(dm.unmatched_rows.is_empty());
        assert_eq!(dm.unmatched_vars.len(), 3);
        assert!(dm.overconstrained_rows.is_empty());
        assert!(dm.overconstrained_vars.is_empty());
        assert_eq!(dm.underconstrained_rows, all_indices(9));
        assert_eq!(
            sorted_values(dm.underconstrained_vars.clone()),
            all_indices(12)
        );
        assert!(dm.square_rows.is_empty());
        assert!(dm.square_vars.is_empty());
    }

    #[test]
    fn mathprogincidence_degenerate_flow_model_partition_and_components() {
        // Mirrors MathProgIncidence.jl/docs/src/example.md and
        // test/interface.jl::test_dulmage_mendelsohn/test_one_connected_component_cons_vars.
        // Rows: sum_comp_eqn, comp_dens_eqn[1..3], bulk_dens_eqn, comp_flow_eqn[1..3].
        // Vars: x[1..3], flow_comp[1..3], flow, rho.
        let inc = equality_incidence(
            8,
            8,
            &[
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 7),
                (2, 1),
                (2, 7),
                (3, 2),
                (3, 7),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 7),
                (5, 0),
                (5, 3),
                (5, 6),
                (6, 1),
                (6, 4),
                (6, 6),
                (7, 2),
                (7, 5),
                (7, 6),
            ],
        );

        let matching = inc.maximum_matching();
        assert_eq!(matching_cardinality(&matching), 7);

        let dm = inc.dulmage_mendelsohn_partition();
        assert_eq!(dm.underconstrained_rows, vec![5, 6, 7]);
        assert_eq!(
            sorted_values(dm.underconstrained_vars.clone()),
            vec![3, 4, 5, 6]
        );
        assert_eq!(dm.overconstrained_rows, vec![0, 1, 2, 3, 4]);
        assert_eq!(
            sorted_values(dm.overconstrained_vars.clone()),
            vec![0, 1, 2, 7]
        );
        assert!(dm.square_rows.is_empty());
        assert!(dm.square_vars.is_empty());

        let uc_blocks =
            inc.connected_components(&dm.underconstrained_rows, &dm.underconstrained_vars);
        assert_eq!(
            sorted_block_pairs(&uc_blocks),
            vec![(vec![5, 6, 7], vec![3, 4, 5, 6])]
        );
        let oc_blocks =
            inc.connected_components(&dm.overconstrained_rows, &dm.overconstrained_vars);
        assert_eq!(
            sorted_block_pairs(&oc_blocks),
            vec![(vec![0, 1, 2, 3, 4], vec![0, 1, 2, 7])]
        );
    }

    #[test]
    fn dm_matches_pyomo_tutorial_singular_chemical_looping_example() {
        // Mirrors Pyomo's incidence tutorial.dm chemical-looping singular model.
        // Rows: sum_eqn, holdup_eqn[1..3], density_eqn, flow_eqn[1..3].
        // Vars: x[1..3], flow_comp[1..3], flow, density.
        let bounds = equality_bounds(8);
        let problem = graph_problem(
            8,
            &bounds,
            &[
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 7),
                (2, 1),
                (2, 7),
                (3, 2),
                (3, 7),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 7),
                (5, 0),
                (5, 3),
                (5, 6),
                (6, 1),
                (6, 4),
                (6, 6),
                (7, 2),
                (7, 5),
                (7, 6),
            ],
        );
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let dm = inc.dulmage_mendelsohn_partition();

        assert_eq!(dm.underconstrained_rows, vec![5, 6, 7]);
        assert_eq!(sorted_values(dm.underconstrained_vars), vec![3, 4, 5, 6]);
        assert_eq!(dm.overconstrained_rows, vec![0, 1, 2, 3, 4]);
        assert_eq!(sorted_values(dm.overconstrained_vars), vec![0, 1, 2, 7]);
        assert!(dm.square_rows.is_empty());
        assert!(dm.square_vars.is_empty());
    }

    #[test]
    fn pyomo_bt_tutorial_sum_flow_variant_is_structurally_square() {
        // Mirrors Pyomo's incidence tutorial.bt numeric-singularity model.
        // The sum mass-fraction equation is replaced by sum_flow_eqn, giving
        // a perfect structural matching before numeric conditioning is checked.
        // Rows: sum_flow_eqn, holdup_eqn[1..3], density_eqn, flow_eqn[1..3].
        // Vars: x[1..3], flow_comp[1..3], flow, density.
        let inc = equality_incidence(
            8,
            8,
            &[
                (0, 3),
                (0, 4),
                (0, 5),
                (0, 6),
                (1, 0),
                (1, 7),
                (2, 1),
                (2, 7),
                (3, 2),
                (3, 7),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 7),
                (5, 0),
                (5, 3),
                (5, 6),
                (6, 1),
                (6, 4),
                (6, 6),
                (7, 2),
                (7, 5),
                (7, 6),
            ],
        );

        assert_perfect_matching(&inc);
        let dm = inc.dulmage_mendelsohn_partition();
        assert!(dm.unmatched_rows.is_empty());
        assert!(dm.unmatched_vars.is_empty());
        assert!(dm.overconstrained_rows.is_empty());
        assert!(dm.overconstrained_vars.is_empty());
        assert!(dm.underconstrained_rows.is_empty());
        assert!(dm.underconstrained_vars.is_empty());
        assert_eq!(dm.square_rows, all_indices(8));
        assert_eq!(dm.square_vars, all_indices(8));

        let blocks = inc
            .block_triangular_decomposition(&all_indices(8), &all_indices(8))
            .unwrap();
        assert_eq!(
            sorted_values(blocks.iter().flat_map(|block| block.rows.clone()).collect()),
            all_indices(8)
        );
        assert_eq!(
            sorted_values(blocks.iter().flat_map(|block| block.vars.clone()).collect()),
            all_indices(8)
        );
        assert!(blocks
            .iter()
            .all(|block| block.rows.len() == block.vars.len()));
    }

    #[test]
    fn pyomo_tutorial_fixed_chemical_looping_example_is_square() {
        // Mirrors the structurally nonsingular fixed model in Pyomo's
        // incidence tutorial.dm and tutorial.btsolve.
        // Rows: sum_eqn, holdup_eqn[1..3], dens_skel_eqn, dens_bulk_eqn,
        // flow_eqn[1..3], flow_dens_eqn.
        // Vars: x[1..3], flow_comp[1..3], flow, dens_bulk, dens_skel, porosity.
        let bounds = equality_bounds(10);
        let problem = graph_problem(
            10,
            &bounds,
            &[
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 7),
                (2, 1),
                (2, 7),
                (3, 2),
                (3, 7),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 8),
                (5, 7),
                (5, 8),
                (5, 9),
                (6, 0),
                (6, 3),
                (6, 6),
                (7, 1),
                (7, 4),
                (7, 6),
                (8, 2),
                (8, 5),
                (8, 6),
                (9, 6),
                (9, 7),
            ],
        );
        let inc = EqualityIncidence::from_problem(&problem, TOL);
        let selected: Vec<_> = (0..10).collect();

        let matching = inc.maximum_matching();
        assert!(matching.unmatched_rows.is_empty());
        assert!(matching.unmatched_vars.is_empty());

        let dm = inc.dulmage_mendelsohn_partition();
        assert!(dm.unmatched_rows.is_empty());
        assert!(dm.unmatched_vars.is_empty());
        assert!(dm.overconstrained_rows.is_empty());
        assert!(dm.overconstrained_vars.is_empty());
        assert!(dm.underconstrained_rows.is_empty());
        assert!(dm.underconstrained_vars.is_empty());
        assert_eq!(dm.square_rows, selected);
        assert_eq!(dm.square_vars, selected);

        let blocks = inc
            .block_triangular_decomposition(&selected, &selected)
            .unwrap();
        assert_eq!(
            sorted_values(blocks.iter().flat_map(|block| block.rows.clone()).collect()),
            selected
        );
        assert_eq!(
            sorted_values(blocks.iter().flat_map(|block| block.vars.clone()).collect()),
            selected
        );
        assert!(blocks
            .iter()
            .all(|block| block.rows.len() == block.vars.len()));
    }

    #[test]
    fn incidence_examples_tutorial_singular_solid_workflow() {
        // Mirrors ~/repos/incidence_examples/incidence_examples/tutorial/run_tutorial.py.
        // Variable order:
        // material_accumulation[A..C], energy_accumulation, flow_mass_comp[A..C],
        // enth_mass, enth_mol_comp[A..C], flow_mass, mass_frac_comp[A..C],
        // temperature, dens_mass_particle, dens_mass_skeletal, particle_porosity.
        let base_edges = vec![
            (0, 16),
            (0, 12),
            (1, 16),
            (1, 13),
            (2, 16),
            (2, 14),
            (3, 0),
            (3, 11),
            (3, 12),
            (4, 1),
            (4, 11),
            (4, 13),
            (5, 2),
            (5, 11),
            (5, 14),
            (6, 16),
            (6, 7),
            (7, 3),
            (7, 11),
            (7, 7),
            (8, 4),
            (8, 11),
            (8, 12),
            (9, 5),
            (9, 11),
            (9, 13),
            (10, 6),
            (10, 11),
            (10, 14),
            (11, 8),
            (11, 15),
            (12, 9),
            (12, 15),
            (13, 10),
            (13, 15),
            (14, 7),
            (14, 8),
            (14, 9),
            (14, 10),
            (14, 12),
            (14, 13),
            (14, 14),
            (15, 12),
            (15, 13),
            (15, 14),
            (16, 16),
            (16, 17),
            (17, 17),
            (17, 12),
            (17, 13),
            (17, 14),
        ];

        let inc = equality_incidence(18, 18, &base_edges);
        let matching = inc.maximum_matching();
        assert_eq!(matching_cardinality(&matching), 17);
        assert_eq!(matching.unmatched_vars, vec![11]);

        let dm = inc.dulmage_mendelsohn_partition();
        assert_eq!(dm.unmatched_rows.len(), 1);
        assert!(matches!(dm.unmatched_rows[0], 15 | 17));
        assert_eq!(dm.unmatched_vars, vec![11]);
        assert_eq!(dm.underconstrained_rows, vec![3, 4, 5, 7, 8, 9, 10]);
        assert_eq!(
            sorted_values(dm.underconstrained_vars.clone()),
            vec![0, 1, 2, 3, 4, 5, 6, 11]
        );
        assert_eq!(dm.overconstrained_rows, vec![0, 1, 2, 15, 16, 17]);
        assert_eq!(
            sorted_values(dm.overconstrained_vars.clone()),
            vec![12, 13, 14, 16, 17]
        );
        assert_eq!(dm.square_rows, vec![6, 11, 12, 13, 14]);
        assert_eq!(dm.square_vars, vec![7, 8, 9, 10, 15]);

        // The tutorial's first structural fix replaces sum_component_eqn with
        // sum_flow_eqn. This is square and structurally nonsingular, but its
        // BTD exposes a numerically singular 4x4 flow block.
        let sum_flow_edges: Vec<_> = base_edges
            .iter()
            .copied()
            .filter(|&(row, _)| row != 15)
            .map(|(row, var)| {
                if row == 16 {
                    (15, var)
                } else if row == 17 {
                    (16, var)
                } else {
                    (row, var)
                }
            })
            .chain([(17, 4), (17, 5), (17, 6), (17, 11)])
            .collect();
        let inc = equality_incidence(18, 18, &sum_flow_edges);
        assert_perfect_matching(&inc);
        let blocks = inc
            .block_triangular_decomposition(&all_indices(18), &all_indices(18))
            .unwrap();
        assert_eq!(
            sorted_block_pairs(&blocks),
            vec![
                (vec![0, 1, 2, 15, 16], vec![12, 13, 14, 16, 17]),
                (vec![3], vec![0]),
                (vec![4], vec![1]),
                (vec![5], vec![2]),
                (vec![6], vec![7]),
                (vec![7], vec![3]),
                (vec![8, 9, 10, 17], vec![4, 5, 6, 11]),
                (vec![11, 12, 13, 14], vec![8, 9, 10, 15]),
            ]
        );

        // The final tutorial fix restores sum_component_eqn, unfixes
        // particle_porosity, and adds flow_density_eqn. This is both
        // structurally nonsingular and block triangular.
        let final_edges = base_edges
            .iter()
            .copied()
            .chain([(16, 18), (18, 11), (18, 16)])
            .collect::<Vec<_>>();
        let inc = equality_incidence(19, 19, &final_edges);
        assert_perfect_matching(&inc);
        let blocks = inc
            .block_triangular_decomposition(&all_indices(19), &all_indices(19))
            .unwrap();
        assert_eq!(
            sorted_block_pairs(&blocks),
            vec![
                (vec![0, 1, 2, 15], vec![12, 13, 14, 16]),
                (vec![3], vec![0]),
                (vec![4], vec![1]),
                (vec![5], vec![2]),
                (vec![6], vec![7]),
                (vec![7], vec![3]),
                (vec![8], vec![4]),
                (vec![9], vec![5]),
                (vec![10], vec![6]),
                (vec![11, 12, 13, 14], vec![8, 9, 10, 15]),
                (vec![16], vec![18]),
                (vec![17], vec![17]),
                (vec![18], vec![11]),
            ]
        );
    }

    #[test]
    fn incidence_examples_clc_dm_boundary_conditions_detect_wrong_endpoint_unfix() {
        // Mirrors the structural lesson in
        // incidence_examples/example1/run_clc_dm_example.py: zero degrees of
        // freedom is not enough if initial conditions are unfixed at the wrong
        // end of a counter-current dynamic reactor.
        let correct = equality_incidence(
            8,
            8,
            &[
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 0),
                (4, 4),
                (5, 1),
                (5, 5),
                (6, 2),
                (6, 6),
                (7, 3),
                (7, 7),
            ],
        );
        assert_perfect_matching(&correct);

        let wrong_endpoint = equality_incidence(
            8,
            8,
            &[
                (0, 0),
                (1, 1),
                // Rows 2 and 3 are solid inlet boundary equations at xf. In
                // the wrong specification, the xf initial-condition variables
                // remain fixed, so these rows have no unfixed incident vars.
                (4, 0),
                (4, 4),
                (5, 1),
                (5, 5),
                (6, 2),
                (6, 6),
                (7, 3),
                (7, 7),
            ],
        );
        let matching = wrong_endpoint.maximum_matching();
        assert_eq!(matching_cardinality(&matching), 6);
        assert_eq!(matching.unmatched_rows, vec![2, 3]);
        assert_eq!(matching.unmatched_vars.len(), 2);

        let dm = wrong_endpoint.dulmage_mendelsohn_partition();
        assert_eq!(dm.overconstrained_rows, vec![2, 3]);
        assert!(dm.overconstrained_vars.is_empty());
        assert_eq!(dm.underconstrained_rows, vec![6, 7]);
        assert_eq!(dm.underconstrained_vars, vec![2, 3, 6, 7]);
        assert_eq!(dm.square_rows, vec![0, 1, 4, 5]);
        assert_eq!(dm.square_vars, vec![0, 1, 4, 5]);
    }

    #[test]
    fn incidence_examples_clc_scc_initialization_chain_blocks_are_ordered() {
        // Mirrors incidence_examples/example2/run_scc_example.py at the graph
        // level. The IDAES example solves a square moving-bed subsystem by
        // strongly connected components; this fixture checks that repeated
        // local SCCs with upstream dependencies are recovered in topological
        // order.
        let n_cells = 6;
        let mut edges = Vec::new();
        for cell in 0..n_cells {
            let row = 2 * cell;
            let var = 2 * cell;
            edges.push((row, var));
            edges.push((row, var + 1));
            edges.push((row + 1, var));
            edges.push((row + 1, var + 1));
            if cell > 0 {
                edges.push((row, var - 1));
            }
        }

        let inc = equality_incidence(2 * n_cells, 2 * n_cells, &edges);
        assert_perfect_matching(&inc);
        let blocks = inc
            .block_triangular_decomposition(&all_indices(2 * n_cells), &all_indices(2 * n_cells))
            .unwrap();
        assert_eq!(
            blocks,
            (0..n_cells)
                .map(|cell| EqualityBlock {
                    rows: vec![2 * cell, 2 * cell + 1],
                    vars: vec![2 * cell, 2 * cell + 1],
                })
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn matching_matches_incidence_examples_rectangular_gallery_cases() {
        // Mirrors the unmatched-variable and unmatched-constraint gallery cases
        // in Robbybp/incidence_examples/incidence_examples/images/generate_matching_images.py.
        let bounds = equality_bounds(2);
        let under_problem = graph_problem(3, &bounds, &[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]);
        let under_inc = EqualityIncidence::from_problem(&under_problem, TOL);
        let under_matching = under_inc.maximum_matching();
        assert!(under_matching.unmatched_rows.is_empty());
        assert_eq!(under_matching.unmatched_vars.len(), 1);

        let under_dm = under_inc.dulmage_mendelsohn_partition();
        assert_eq!(under_dm.underconstrained_rows, vec![0, 1]);
        assert_eq!(sorted_values(under_dm.underconstrained_vars), vec![0, 1, 2]);
        assert!(under_dm.overconstrained_rows.is_empty());
        assert!(under_dm.overconstrained_vars.is_empty());
        assert!(under_dm.square_rows.is_empty());
        assert!(under_dm.square_vars.is_empty());

        let bounds = equality_bounds(3);
        let over_problem = graph_problem(
            2,
            &bounds,
            &[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
        );
        let over_inc = EqualityIncidence::from_problem(&over_problem, TOL);
        let over_matching = over_inc.maximum_matching();
        assert_eq!(over_matching.unmatched_rows.len(), 1);
        assert!(over_matching.unmatched_vars.is_empty());

        let over_dm = over_inc.dulmage_mendelsohn_partition();
        assert_eq!(over_dm.overconstrained_rows, vec![0, 1, 2]);
        assert_eq!(over_dm.overconstrained_vars, vec![0, 1]);
        assert!(over_dm.underconstrained_rows.is_empty());
        assert!(over_dm.underconstrained_vars.is_empty());
        assert!(over_dm.square_rows.is_empty());
        assert!(over_dm.square_vars.is_empty());
    }

    #[test]
    fn connected_components_match_pyomo_and_mathprogincidence_matrix_cases() {
        // Mirrors Pyomo test_connected.py::test_decomposable_matrix.
        let inc = equality_incidence(
            5,
            5,
            &[
                (0, 0),
                (1, 0),
                (1, 1),
                (2, 2),
                (2, 3),
                (3, 3),
                (3, 4),
                (4, 4),
            ],
        );
        let components = inc.connected_components(&all_indices(5), &all_indices(5));
        assert_eq!(
            sorted_block_pairs(&components),
            vec![(vec![0, 1], vec![0, 1]), (vec![2, 3, 4], vec![2, 3, 4])]
        );

        // Same Pyomo matrix with deterministic row/column permutations.
        let row_perm = [3, 1, 4, 0, 2];
        let col_perm = [2, 4, 0, 3, 1];
        let permuted_edges: Vec<_> = [
            (0, 0),
            (1, 0),
            (1, 1),
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 4),
            (4, 4),
        ]
        .iter()
        .map(|&(row, col)| (row_perm[row], col_perm[col]))
        .collect();
        let components = equality_incidence(5, 5, &permuted_edges)
            .connected_components(&all_indices(5), &all_indices(5));
        let mut expected = vec![
            (
                sorted_values(vec![row_perm[0], row_perm[1]]),
                sorted_values(vec![col_perm[0], col_perm[1]]),
            ),
            (
                sorted_values(vec![row_perm[2], row_perm[3], row_perm[4]]),
                sorted_values(vec![col_perm[2], col_perm[3], col_perm[4]]),
            ),
        ];
        expected.sort();
        assert_eq!(sorted_block_pairs(&components), expected);

        // Mirrors MathProgIncidence.jl/test/interface.jl::test_connected_components_matrix.
        let components = equality_incidence(3, 3, &[(0, 0), (0, 2), (1, 0), (1, 2), (2, 1)])
            .connected_components(&all_indices(3), &all_indices(3));
        assert_eq!(
            sorted_block_pairs(&components),
            vec![(vec![0, 1], vec![0, 2]), (vec![2], vec![1])]
        );

        // Mirrors MathProgIncidence.jl/test/interface.jl::test_multiple_connected_components_igraph.
        let components = equality_incidence(5, 3, &[(0, 0), (0, 2), (1, 1), (1, 3), (2, 4)])
            .connected_components(&all_indices(3), &all_indices(5));
        assert_eq!(
            sorted_block_pairs(&components),
            vec![
                (vec![0], vec![0, 2]),
                (vec![1], vec![1, 3]),
                (vec![2], vec![4])
            ]
        );
    }

    #[test]
    fn connected_components_split_selected_independent_systems() {
        let bounds = equality_bounds(4);
        let problem = graph_problem(5, &bounds, &[(0, 0), (1, 2), (1, 1), (2, 3), (3, 4)]);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let components = inc.connected_components(&[3, 1, 0, 2], &[4, 0, 3, 2, 1]);

        assert_eq!(
            components,
            vec![
                EqualityBlock {
                    rows: vec![0],
                    vars: vec![0],
                },
                EqualityBlock {
                    rows: vec![1],
                    vars: vec![1, 2],
                },
                EqualityBlock {
                    rows: vec![2],
                    vars: vec![3],
                },
                EqualityBlock {
                    rows: vec![3],
                    vars: vec![4],
                },
            ]
        );
    }

    #[test]
    fn connected_components_return_global_equality_rows() {
        let problem = graph_problem(2, &[(0.0, 1.0), (2.0, 2.0), (3.0, 3.0)], &[(1, 0), (2, 1)]);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let components = inc.connected_components(&[0, 1], &[0, 1]);

        assert_eq!(
            components,
            vec![
                EqualityBlock {
                    rows: vec![1],
                    vars: vec![0],
                },
                EqualityBlock {
                    rows: vec![2],
                    vars: vec![1],
                },
            ]
        );
    }

    #[test]
    fn btd_splits_independent_square_systems() {
        let bounds = equality_bounds(3);
        let problem = graph_problem(3, &bounds, &[(2, 2), (0, 0), (1, 1)]);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let blocks = inc
            .block_triangular_decomposition(&[2, 0, 1], &[2, 0, 1])
            .unwrap();

        assert_eq!(
            blocks,
            vec![
                EqualityBlock {
                    rows: vec![0],
                    vars: vec![0],
                },
                EqualityBlock {
                    rows: vec![1],
                    vars: vec![1],
                },
                EqualityBlock {
                    rows: vec![2],
                    vars: vec![2],
                },
            ]
        );
    }

    #[test]
    fn btd_orders_triangular_system_upstream_to_downstream() {
        let bounds = equality_bounds(3);
        let problem = graph_problem(3, &bounds, &[(2, 2), (1, 1), (1, 0), (0, 0), (2, 1)]);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let blocks = inc
            .block_triangular_decomposition(&[2, 1, 0], &[2, 1, 0])
            .unwrap();

        assert_eq!(
            blocks,
            vec![
                EqualityBlock {
                    rows: vec![0],
                    vars: vec![0],
                },
                EqualityBlock {
                    rows: vec![1],
                    vars: vec![1],
                },
                EqualityBlock {
                    rows: vec![2],
                    vars: vec![2],
                },
            ]
        );
    }

    #[test]
    fn btd_returns_cyclic_square_system_as_one_block() {
        let bounds = equality_bounds(2);
        let problem = graph_problem(2, &bounds, &[(0, 1), (1, 0), (0, 0), (1, 1)]);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let blocks = inc
            .block_triangular_decomposition(&[0, 1], &[0, 1])
            .unwrap();

        assert_eq!(
            blocks,
            vec![EqualityBlock {
                rows: vec![0, 1],
                vars: vec![0, 1],
            }]
        );
    }

    #[test]
    fn btd_matches_mathprogincidence_matrix_example() {
        // Mirrors MathProgIncidence.jl/test/interface.jl::test_block_triangularize_matrix.
        // Julia uses 1-based matrix indices; these assertions use 0-based row/var indices.
        let bounds = equality_bounds(3);
        let problem = graph_problem(
            3,
            &bounds,
            &[(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)],
        );
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let blocks = inc
            .block_triangular_decomposition(&[0, 1, 2], &[0, 1, 2])
            .unwrap();

        assert_eq!(
            blocks,
            vec![
                EqualityBlock {
                    rows: vec![2],
                    vars: vec![1],
                },
                EqualityBlock {
                    rows: vec![0, 1],
                    vars: vec![0, 2],
                },
            ]
        );

        // Mirrors MathProgIncidence.jl/test/interface.jl::test_block_triangularize.
        // Rows: eq1, eq2, eq3. Vars: x[1], x[2], x[3].
        let blocks = equality_incidence(3, 3, &[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)])
            .block_triangular_decomposition(&all_indices(3), &all_indices(3))
            .unwrap();
        assert_eq!(
            blocks,
            vec![
                EqualityBlock {
                    rows: vec![1, 2],
                    vars: vec![0, 2],
                },
                EqualityBlock {
                    rows: vec![0],
                    vars: vec![1],
                },
            ]
        );
    }

    #[test]
    fn btd_matches_pyomo_triangularize_matrix_gallery_cases() {
        // Mirrors pyomo/contrib/incidence_analysis/tests/test_triangularize.py.
        let n = 5;

        let identity_edges: Vec<_> = (0..n).map(|i| (i, i)).collect();
        let blocks = equality_incidence(n, n, &identity_edges)
            .block_triangular_decomposition(&all_indices(n), &all_indices(n))
            .unwrap();
        assert_eq!(blocks.len(), n);
        assert!(blocks
            .iter()
            .all(|block| block.rows.len() == 1 && block.rows == block.vars));

        let mut lower_tri = Vec::new();
        lower_tri.extend((0..n).map(|i| (i, i)));
        lower_tri.extend((1..n).map(|i| (i, i - 1)));
        let blocks = equality_incidence(n, n, &lower_tri)
            .block_triangular_decomposition(&all_indices(n), &all_indices(n))
            .unwrap();
        assert_eq!(
            blocks,
            (0..n)
                .map(|i| EqualityBlock {
                    rows: vec![i],
                    vars: vec![i],
                })
                .collect::<Vec<_>>()
        );

        let mut upper_tri = Vec::new();
        upper_tri.extend((0..n).map(|i| (i, i)));
        upper_tri.extend((0..n - 1).map(|i| (i, i + 1)));
        let blocks = equality_incidence(n, n, &upper_tri)
            .block_triangular_decomposition(&all_indices(n), &all_indices(n))
            .unwrap();
        assert_eq!(
            blocks,
            (0..n)
                .rev()
                .map(|i| EqualityBlock {
                    rows: vec![i],
                    vars: vec![i],
                })
                .collect::<Vec<_>>()
        );

        let mut bordered = Vec::new();
        bordered.extend((0..n - 1).map(|i| (i, i)));
        bordered.extend((0..n - 1).map(|i| (n - 1, i)));
        bordered.extend((0..n - 1).map(|i| (i, n - 1)));
        let blocks = equality_incidence(n, n, &bordered)
            .block_triangular_decomposition(&all_indices(n), &all_indices(n))
            .unwrap();
        assert_eq!(
            blocks,
            vec![EqualityBlock {
                rows: all_indices(n),
                vars: all_indices(n),
            }]
        );

        let half = n / 2;
        let mut decomposable_bordered = Vec::new();
        decomposable_bordered.extend((0..n - 1).map(|i| (i, i)));
        decomposable_bordered.extend((0..n - 1).map(|i| (n - 1, i)));
        decomposable_bordered.extend((half..n - 1).map(|i| (i, n - 1)));
        let blocks = equality_incidence(n, n, &decomposable_bordered)
            .block_triangular_decomposition(&all_indices(n), &all_indices(n))
            .unwrap();
        assert_eq!(
            sorted_block_pairs(&blocks),
            vec![
                (vec![0], vec![0]),
                (vec![1], vec![1]),
                (vec![2, 3, 4], vec![2, 3, 4])
            ]
        );

        let mut decomposable_tridiagonal = Vec::new();
        decomposable_tridiagonal.extend((0..n).map(|i| (i, i)));
        decomposable_tridiagonal.extend((1..n).map(|i| (i, i - 1)));
        decomposable_tridiagonal.extend((0..n - 1).filter(|i| i % 2 == 0).map(|i| (i, i + 1)));
        let blocks = equality_incidence(n, n, &decomposable_tridiagonal)
            .block_triangular_decomposition(&all_indices(n), &all_indices(n))
            .unwrap();
        assert_eq!(
            blocks,
            vec![
                EqualityBlock {
                    rows: vec![0, 1],
                    vars: vec![0, 1],
                },
                EqualityBlock {
                    rows: vec![2, 3],
                    vars: vec![2, 3],
                },
                EqualityBlock {
                    rows: vec![4],
                    vars: vec![4],
                },
            ]
        );
    }

    #[test]
    fn btd_rejects_non_square_input() {
        let bounds = equality_bounds(1);
        let problem = graph_problem(2, &bounds, &[(0, 0), (0, 1)]);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let err = inc
            .block_triangular_decomposition(&[0], &[0, 1])
            .unwrap_err();

        assert_eq!(
            err,
            BlockTriangularizationError::NonSquare { rows: 1, vars: 2 }
        );
    }

    #[test]
    fn btd_rejects_square_but_imperfectly_matched_input() {
        let bounds = equality_bounds(2);
        let problem = graph_problem(2, &bounds, &[(0, 0), (1, 0)]);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let err = inc
            .block_triangular_decomposition(&[0, 1], &[0, 1])
            .unwrap_err();

        assert_eq!(
            err,
            BlockTriangularizationError::ImperfectMatching {
                unmatched_rows: vec![1],
                unmatched_vars: vec![1],
            }
        );
    }

    #[test]
    fn btd_handles_long_triangular_dependency_chain_iteratively() {
        let chain_len = 10_000;
        let bounds = equality_bounds(chain_len);
        let mut edges = Vec::with_capacity(2 * chain_len - 1);
        edges.push((0, 0));
        for row in 1..chain_len {
            edges.push((row, row - 1));
            edges.push((row, row));
        }
        let problem = graph_problem(chain_len, &bounds, &edges);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let selected: Vec<_> = (0..chain_len).collect();
        let blocks = inc
            .block_triangular_decomposition(&selected, &selected)
            .unwrap();

        assert_eq!(blocks.len(), chain_len);
        for (idx, block) in blocks.iter().enumerate() {
            assert_eq!(block.rows, vec![idx]);
            assert_eq!(block.vars, vec![idx]);
        }
    }

    #[test]
    fn btd_handles_many_independent_blocks_with_heap_ready_set() {
        let block_count = 10_000;
        let bounds = equality_bounds(block_count);
        let edges: Vec<_> = (0..block_count).map(|idx| (idx, idx)).collect();
        let problem = graph_problem(block_count, &bounds, &edges);
        let inc = EqualityIncidence::from_problem(&problem, TOL);

        let selected: Vec<_> = (0..block_count).rev().collect();
        let blocks = inc
            .block_triangular_decomposition(&selected, &selected)
            .unwrap();

        assert_eq!(blocks.len(), block_count);
        for (idx, block) in blocks.iter().enumerate() {
            assert_eq!(block.rows, vec![idx]);
            assert_eq!(block.vars, vec![idx]);
        }
    }

    #[test]
    fn find_candidates_detects_independent_auxiliary_system() {
        let bounds = equality_bounds(2);
        let problem = graph_problem(3, &bounds, &[(0, 1), (1, 2), (1, 1)]);

        let candidates = find_presolve_candidates(&problem, TOL);

        assert_eq!(
            candidates,
            vec![PresolveCandidate {
                blocks: vec![
                    EqualityBlock {
                        rows: vec![0],
                        vars: vec![1],
                    },
                    EqualityBlock {
                        rows: vec![1],
                        vars: vec![2],
                    },
                ],
            }]
        );
    }

    #[test]
    fn find_candidates_rejects_underconstrained_equality_component() {
        let bounds = equality_bounds(1);
        let problem = graph_problem(2, &bounds, &[(0, 0), (0, 1)]);

        let candidates = find_presolve_candidates(&problem, TOL);

        assert!(candidates.is_empty());
    }

    #[test]
    fn find_candidates_rejects_overconstrained_equality_component() {
        let bounds = equality_bounds(2);
        let problem = graph_problem(1, &bounds, &[(0, 0), (1, 0)]);

        let candidates = find_presolve_candidates(&problem, TOL);

        assert!(candidates.is_empty());
    }

    #[test]
    fn find_candidates_accepts_dm_square_block_embedded_in_underconstrained_component() {
        let bounds = equality_bounds(2);
        let problem = graph_problem(3, &bounds, &[(0, 0), (1, 0), (1, 1), (1, 2)]);

        let (candidates, diagnostics) = find_presolve_candidates_with_diagnostics(&problem, TOL);

        assert_eq!(
            candidates,
            vec![PresolveCandidate {
                blocks: vec![EqualityBlock {
                    rows: vec![0],
                    vars: vec![0],
                }],
            }]
        );
        assert_eq!(diagnostics.connected_components, 1);
        assert_eq!(diagnostics.square_components, 1);
        assert_eq!(diagnostics.closed_components, 0);
        assert_eq!(diagnostics.pure_equality_candidates, 1);
        assert_eq!(diagnostics.accepted_blocks, candidates[0].blocks);
    }

    #[test]
    fn find_candidates_rejects_dm_square_block_with_row_external_variable() {
        let bounds = equality_bounds(3);
        let problem = graph_problem(2, &bounds, &[(0, 0), (0, 1), (1, 1), (2, 1)]);

        let (candidates, diagnostics) = find_presolve_candidates_with_diagnostics(&problem, TOL);

        assert!(candidates.is_empty());
        assert!(diagnostics.rejections.iter().any(|rejection| {
            rejection.reason == AuxiliaryRejectionReason::CandidateRowsCoupledToRemainingVariables
                && rejection.block
                    == Some(EqualityBlock {
                        rows: vec![0],
                        vars: vec![0],
                    })
                && rejection
                    .detail
                    .as_deref()
                    .is_some_and(|detail| detail.contains("non-candidate variables [1]"))
        }));
    }

    #[test]
    fn find_candidates_rejects_objective_coupled_variables_by_default() {
        let bounds = equality_bounds(1);
        let problem = graph_problem_with_objective(2, &bounds, &[(0, 1)], &[1]);

        let (candidates, diagnostics) = find_presolve_candidates_with_diagnostics(&problem, TOL);

        assert!(candidates.is_empty());
        assert_eq!(diagnostics.pure_equality_candidates, 0);
        assert_eq!(diagnostics.objective_coupled_candidates, 1);
        assert_eq!(diagnostics.inequality_coupled_candidates, 0);
        assert_eq!(diagnostics.objective_and_inequality_coupled_candidates, 0);
        assert_eq!(
            diagnostics.rejections[0].reason,
            AuxiliaryRejectionReason::CoupledAuxiliaryBlock {
                coupling: AuxiliaryCouplingClass::ObjectiveCoupled,
            }
        );
    }

    #[test]
    fn find_candidates_rejects_inequality_coupled_variables_by_default() {
        let problem = graph_problem(2, &[(0.0, 0.0), (0.0, 1.0)], &[(0, 1), (1, 1)]);

        let (candidates, diagnostics) = find_presolve_candidates_with_diagnostics(&problem, TOL);

        assert!(candidates.is_empty());
        assert_eq!(diagnostics.pure_equality_candidates, 0);
        assert_eq!(diagnostics.objective_coupled_candidates, 0);
        assert_eq!(diagnostics.inequality_coupled_candidates, 1);
        assert_eq!(diagnostics.objective_and_inequality_coupled_candidates, 0);
        assert_eq!(
            diagnostics.rejections[0].reason,
            AuxiliaryRejectionReason::CoupledAuxiliaryBlock {
                coupling: AuxiliaryCouplingClass::InequalityCoupled,
            }
        );
    }

    #[test]
    fn find_candidates_rejects_objective_and_inequality_coupled_variables_by_default() {
        let problem =
            graph_problem_with_objective(2, &[(0.0, 0.0), (0.0, 1.0)], &[(0, 1), (1, 1)], &[1]);

        let (candidates, diagnostics) = find_presolve_candidates_with_diagnostics(&problem, TOL);

        assert!(candidates.is_empty());
        assert_eq!(diagnostics.pure_equality_candidates, 0);
        assert_eq!(diagnostics.objective_coupled_candidates, 0);
        assert_eq!(diagnostics.inequality_coupled_candidates, 0);
        assert_eq!(diagnostics.objective_and_inequality_coupled_candidates, 1);
        assert_eq!(
            diagnostics.rejections[0].reason,
            AuxiliaryRejectionReason::CoupledAuxiliaryBlock {
                coupling: AuxiliaryCouplingClass::ObjectiveAndInequalityCoupled,
            }
        );
    }

    #[test]
    fn find_postsolve_candidates_allows_recovery_row_with_main_variable() {
        let bounds = equality_bounds(1);
        let problem = graph_problem_with_objective(2, &bounds, &[(0, 0), (0, 1)], &[0]);

        let (candidates, diagnostics) =
            find_postsolve_candidates_with_diagnostics(&problem, TOL);

        assert_eq!(
            candidates,
            vec![PresolveCandidate {
                blocks: vec![EqualityBlock {
                    rows: vec![0],
                    vars: vec![1],
                }],
            }]
        );
        assert_eq!(diagnostics.pure_equality_candidates, 1);
        assert_eq!(diagnostics.objective_coupled_candidates, 0);
        assert_eq!(diagnostics.inequality_coupled_candidates, 0);
        assert_eq!(diagnostics.objective_and_inequality_coupled_candidates, 0);
    }

    #[test]
    fn find_postsolve_candidates_rejects_objective_dependent_variable() {
        let bounds = equality_bounds(1);
        let problem = graph_problem_with_objective(2, &bounds, &[(0, 1)], &[1]);

        let candidates = find_postsolve_candidates(&problem, TOL);

        assert!(candidates.is_empty());
    }

    #[test]
    fn find_postsolve_candidates_rejects_inequality_coupled_variable() {
        let problem = graph_problem_with_objective(
            2,
            &[(0.0, 0.0), (0.0, 1.0)],
            &[(0, 0), (0, 1), (1, 1)],
            &[0],
        );

        let candidates = find_postsolve_candidates(&problem, TOL);

        assert!(candidates.is_empty());
    }

    #[test]
    fn find_postsolve_candidates_rejects_ambiguous_feasibility_system() {
        let bounds = equality_bounds(2);
        let problem =
            graph_problem_with_objective(2, &bounds, &[(0, 0), (0, 1), (1, 0), (1, 1)], &[0]);

        let candidates = find_postsolve_candidates(&problem, TOL);

        assert!(candidates.is_empty());
    }

    #[test]
    fn find_candidates_returns_btd_ordered_blocks() {
        let problem = graph_problem(
            3,
            &[(0.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            &[(1, 0), (2, 0), (2, 1), (3, 1), (3, 2)],
        );

        let candidates = find_presolve_candidates(&problem, TOL);

        assert_eq!(
            candidates,
            vec![PresolveCandidate {
                blocks: vec![
                    EqualityBlock {
                        rows: vec![1],
                        vars: vec![0],
                    },
                    EqualityBlock {
                        rows: vec![2],
                        vars: vec![1],
                    },
                    EqualityBlock {
                        rows: vec![3],
                        vars: vec![2],
                    },
                ],
            }]
        );
    }

    #[test]
    fn presolve_diagnostics_reports_rejected_non_square_component() {
        let bounds = equality_bounds(1);
        let problem = graph_problem(2, &bounds, &[(0, 0), (0, 1)]);

        let (candidates, diagnostics) = find_presolve_candidates_with_diagnostics(&problem, TOL);

        assert!(candidates.is_empty());
        assert_eq!(diagnostics.equality_rows, 1);
        assert_eq!(diagnostics.incident_variables, 2);
        assert_eq!(diagnostics.connected_components, 1);
        assert_eq!(diagnostics.square_components, 0);
        assert_eq!(diagnostics.closed_components, 0);
        assert_eq!(diagnostics.btd_blocks, 0);
        assert_eq!(diagnostics.rejected_blocks(), 1);
        assert_eq!(
            diagnostics.rejections[0],
            AuxiliaryRejection {
                block: Some(EqualityBlock {
                    rows: vec![0],
                    vars: vec![0, 1],
                }),
                reason: AuxiliaryRejectionReason::NonSquareComponent { rows: 1, vars: 2 },
                detail: Some("rows=1, vars=2".to_string()),
            }
        );
    }

    #[test]
    fn presolve_diagnostics_records_btd_block_counts_and_sizes() {
        let problem = graph_problem(
            3,
            &[(0.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            &[(1, 0), (2, 0), (2, 1), (3, 1), (3, 2)],
        );

        let (candidates, mut diagnostics) =
            find_presolve_candidates_with_diagnostics(&problem, TOL);
        diagnostics.record_rank_accepted_candidates(&candidates);

        assert_eq!(candidates.len(), 1);
        assert_eq!(diagnostics.equality_rows, 3);
        assert_eq!(diagnostics.incident_variables, 3);
        assert_eq!(diagnostics.connected_components, 1);
        assert_eq!(diagnostics.square_components, 1);
        assert_eq!(diagnostics.closed_components, 1);
        assert_eq!(diagnostics.pure_equality_candidates, 1);
        assert_eq!(diagnostics.objective_coupled_candidates, 0);
        assert_eq!(diagnostics.inequality_coupled_candidates, 0);
        assert_eq!(diagnostics.objective_and_inequality_coupled_candidates, 0);
        assert_eq!(diagnostics.btd_blocks, 3);
        assert_eq!(diagnostics.rank_accepted_blocks, 3);
        assert_eq!(
            diagnostics.accepted_block_sizes(),
            vec![(1, 1), (1, 1), (1, 1)]
        );
        assert_eq!(
            diagnostics.accepted_blocks,
            vec![
                EqualityBlock {
                    rows: vec![1],
                    vars: vec![0],
                },
                EqualityBlock {
                    rows: vec![2],
                    vars: vec![1],
                },
                EqualityBlock {
                    rows: vec![3],
                    vars: vec![2],
                },
            ]
        );
        assert!(diagnostics.rejections.is_empty());
    }
}
