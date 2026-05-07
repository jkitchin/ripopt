//! Equality / inequality constraint layout — ripopt's mirror of Ipopt's
//! TNLPAdapter c/d split (`c_map`, `d_map`, `P_c_g`, `P_d_g`).
//!
//! Ipopt 3.14 stores `c(x) = 0` (n_c rows) and `d_L ≤ d(x) ≤ d_U` (n_d rows)
//! as two completely independent vectors and never recombines them inside
//! the IPM core. ripopt today still ingests the user's combined `g(x)`
//! through the [`crate::problem::NlpProblem`] trait (matching Ipopt's
//! `TNLP::eval_g`), but the IPM-internal data structures are migrating to
//! Ipopt's split representation. This module owns the index translation
//! between the user/trait-facing combined indexing and the split
//! (c-block / d-block) indexing used inside the IPM.
//!
//! Cross-references (paths under `ref/Ipopt/src/`):
//!   * Split origin in TNLPAdapter::GetSpaces: `Interfaces/IpTNLPAdapter.cpp:576-621`
//!   * Permutation matrices `P_c_g`, `P_d_g`: `Interfaces/IpTNLPAdapter.cpp:926-935`
//!   * Storage on IpoptNLP: `Algorithm/IpIpoptNLP.hpp:109-128, 151-160`
//!
//! See `docs/V0.8_DATA_LAYOUT_REFACTOR_PLAN.md` for the phased migration.

use crate::convergence::is_equality_constraint;

/// Maps each combined-index constraint row to its position in either the
/// equality (c) block or the inequality (d) block, plus the inverse maps
/// from block-index back to combined-index. Built once per `SolverState`
/// from `g_l` / `g_u` (an equality is detected by `g_l[i] == g_u[i]`,
/// matching Ipopt's `TNLPAdapter::GetSpaces` test).
///
/// Phase 1 of the data-layout refactor: this type replaces
/// `kkt_aug::ConstraintPartition` (same n_c/n_d/eq_pos/ineq_pos fields,
/// plus the new `c_to_combined` map). Phase 3 will remove the underlying
/// combined storage entirely; until then this layout is the single source
/// of truth for "which combined row belongs to which block".
#[derive(Debug, Clone)]
pub struct ConstraintLayout {
    /// Number of equality constraints (= dim of `y_c`, `g_c`, `Jac_c` rows).
    pub n_c: usize,
    /// Number of inequality constraints (= dim of `y_d`, `g_d`, `Jac_d` rows,
    /// `s`, `v_l`, `v_u`).
    pub n_d: usize,
    /// `eq_pos[i] = Some(k)` iff combined row `i` is the `k`-th equality;
    /// `None` for inequalities. Mirrors Ipopt's `P_c_g^T` action on a
    /// combined vector.
    pub eq_pos: Vec<Option<usize>>,
    /// `ineq_pos[i] = Some(k)` iff combined row `i` is the `k`-th
    /// inequality; `None` for equalities. Mirrors Ipopt's `P_d_g^T` action.
    pub ineq_pos: Vec<Option<usize>>,
    /// `c_to_combined[k]` is the combined index of the `k`-th equality.
    /// Inverse of `eq_pos`. Mirrors Ipopt's `c_map` from
    /// `TNLPAdapter::GetSpaces`.
    pub c_to_combined: Vec<usize>,
    /// `d_to_combined[k]` is the combined index of the `k`-th inequality.
    /// Inverse of `ineq_pos`. Mirrors Ipopt's `d_map`.
    pub d_to_combined: Vec<usize>,
}

impl ConstraintLayout {
    /// Build the layout by classifying each combined row as equality
    /// (`g_l[i] == g_u[i]` exactly — same convention as
    /// `convergence::is_equality_constraint`) or inequality.
    pub fn new(g_l: &[f64], g_u: &[f64]) -> Self {
        let m = g_l.len();
        debug_assert_eq!(g_u.len(), m);
        let mut eq_pos = vec![None; m];
        let mut ineq_pos = vec![None; m];
        let mut c_to_combined = Vec::new();
        let mut d_to_combined = Vec::new();
        for i in 0..m {
            if is_equality_constraint(g_l[i], g_u[i]) {
                eq_pos[i] = Some(c_to_combined.len());
                c_to_combined.push(i);
            } else {
                ineq_pos[i] = Some(d_to_combined.len());
                d_to_combined.push(i);
            }
        }
        Self {
            n_c: c_to_combined.len(),
            n_d: d_to_combined.len(),
            eq_pos,
            ineq_pos,
            c_to_combined,
            d_to_combined,
        }
    }

    /// Total combined-row count (`n_c + n_d` = `m`).
    pub fn m(&self) -> usize {
        self.n_c + self.n_d
    }

    /// Project a combined-indexed slice (length `m`) onto the c-block,
    /// returning a fresh `Vec<f64>` of length `n_c`.
    pub fn project_c(&self, combined: &[f64]) -> Vec<f64> {
        debug_assert_eq!(combined.len(), self.m());
        self.c_to_combined.iter().map(|&i| combined[i]).collect()
    }

    /// Project a combined-indexed slice onto the d-block (length `n_d`).
    pub fn project_d(&self, combined: &[f64]) -> Vec<f64> {
        debug_assert_eq!(combined.len(), self.m());
        self.d_to_combined.iter().map(|&i| combined[i]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_pure_equality() {
        let g_l = vec![1.0, 2.0, 3.0];
        let g_u = vec![1.0, 2.0, 3.0];
        let layout = ConstraintLayout::new(&g_l, &g_u);
        assert_eq!(layout.n_c, 3);
        assert_eq!(layout.n_d, 0);
        assert_eq!(layout.c_to_combined, vec![0, 1, 2]);
        assert!(layout.d_to_combined.is_empty());
        assert_eq!(layout.eq_pos, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(layout.ineq_pos, vec![None, None, None]);
    }

    #[test]
    fn split_pure_inequality() {
        let g_l = vec![0.0, f64::NEG_INFINITY];
        let g_u = vec![1.0, 5.0];
        let layout = ConstraintLayout::new(&g_l, &g_u);
        assert_eq!(layout.n_c, 0);
        assert_eq!(layout.n_d, 2);
        assert_eq!(layout.d_to_combined, vec![0, 1]);
        assert_eq!(layout.ineq_pos, vec![Some(0), Some(1)]);
    }

    #[test]
    fn split_mixed_preserves_user_order_within_block() {
        // user ordering: [eq, ineq, eq, ineq, eq] → c_block [0,2,4], d_block [1,3]
        let g_l = vec![1.0, 0.0, 2.0, 0.0, 3.0];
        let g_u = vec![1.0, 5.0, 2.0, 5.0, 3.0];
        let layout = ConstraintLayout::new(&g_l, &g_u);
        assert_eq!(layout.n_c, 3);
        assert_eq!(layout.n_d, 2);
        assert_eq!(layout.c_to_combined, vec![0, 2, 4]);
        assert_eq!(layout.d_to_combined, vec![1, 3]);
        assert_eq!(layout.eq_pos, vec![Some(0), None, Some(1), None, Some(2)]);
        assert_eq!(layout.ineq_pos, vec![None, Some(0), None, Some(1), None]);
    }

    #[test]
    fn project_c_and_d_round_trip() {
        let g_l = vec![1.0, 0.0, 2.0, 0.0];
        let g_u = vec![1.0, 5.0, 2.0, 5.0];
        let layout = ConstraintLayout::new(&g_l, &g_u);
        let combined = vec![10.0, 20.0, 30.0, 40.0];
        assert_eq!(layout.project_c(&combined), vec![10.0, 30.0]);
        assert_eq!(layout.project_d(&combined), vec![20.0, 40.0]);
    }

    #[test]
    fn empty_problem() {
        let layout = ConstraintLayout::new(&[], &[]);
        assert_eq!(layout.n_c, 0);
        assert_eq!(layout.n_d, 0);
        assert_eq!(layout.m(), 0);
    }
}
