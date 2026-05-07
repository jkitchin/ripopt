//! Slack-bound layout — ripopt's mirror of Ipopt's `Pd_L_` / `Pd_U_`
//! ExpansionMatrix from `IpOrigIpoptNLP.hpp`.
//!
//! Ipopt 3.14 stores `d_L` (size `n_d_l` = count of finite *lower* slack
//! bounds) and `d_U` (size `n_d_u`) compressed: only finite entries are
//! present, and the expansion matrices `Pd_L_` / `Pd_U_` map a
//! compressed-index `k` back to the d-block index in `0..n_d`. The slack
//! multipliers `v_L` (size `n_d_l`) and `v_U` (size `n_d_u`) live on the
//! same compressed index space.
//!
//! ripopt today stores `v_l`, `v_u`, `dv_l`, `dv_u` at full length `n_d`
//! with `0` for unbounded sides, and `d_l`, `d_u` at full length `n_d`
//! with `±∞`. Phase 8 of the data-layout refactor introduces native
//! compressed slack-multiplier storage; Phase 9 follows for the slack
//! bounds themselves.
//!
//! Cross-references (paths under `ref/Ipopt/src/`):
//!   * `Pd_L_` / `Pd_U_` ExpansionMatrix declarations:
//!     `Algorithm/IpOrigIpoptNLP.hpp:241-263`
//!   * Construction in TNLPAdapter: `Interfaces/IpTNLPAdapter.cpp:944-953`
//!
//! See `docs/V0.8_DATA_LAYOUT_REFACTOR_PLAN.md` Phase 8 for the migration.

#[derive(Debug, Clone)]
pub struct DBoundLayout {
    /// Number of finite lower slack bounds (= dim of `v_L`, `d_L`,
    /// `Pd_L_` columns).
    pub n_d_l: usize,
    /// Number of finite upper slack bounds (= dim of `v_U`, `d_U`,
    /// `Pd_U_` columns).
    pub n_d_u: usize,
    /// `full_to_d_l[k_d] = Some(k)` iff slack-row `k_d` has a finite
    /// lower bound and is the `k`-th compressed lower-bound entry;
    /// `None` otherwise. Mirrors `Pd_L_^T` action on a length-`n_d`
    /// vector.
    pub full_to_d_l: Vec<Option<usize>>,
    /// `full_to_d_u[k_d] = Some(k)` iff slack-row `k_d` has a finite
    /// upper bound. Mirrors `Pd_U_^T`.
    pub full_to_d_u: Vec<Option<usize>>,
    /// `d_l_to_full[k]` is the d-block index of the `k`-th finite
    /// lower bound. Inverse of `full_to_d_l`. Mirrors `Pd_L_`.
    pub d_l_to_full: Vec<usize>,
    /// `d_u_to_full[k]` is the d-block index of the `k`-th finite
    /// upper bound. Inverse of `full_to_d_u`. Mirrors `Pd_U_`.
    pub d_u_to_full: Vec<usize>,
}

impl DBoundLayout {
    /// Build the layout by classifying each d-row's lower and upper
    /// slack bound as finite / infinite. A bound is finite iff
    /// `b.is_finite()`. This matches Ipopt's TNLPAdapter finite-bound
    /// test (`IpTNLPAdapter.cpp:944-953`).
    ///
    /// Inputs are size `n_d` (the number of inequality rows after the
    /// constraint partition); equality rows are not part of this layout.
    pub fn new(d_l: &[f64], d_u: &[f64]) -> Self {
        let n_d = d_l.len();
        debug_assert_eq!(d_u.len(), n_d);
        let mut full_to_d_l = vec![None; n_d];
        let mut full_to_d_u = vec![None; n_d];
        let mut d_l_to_full = Vec::new();
        let mut d_u_to_full = Vec::new();
        for k_d in 0..n_d {
            if d_l[k_d].is_finite() {
                full_to_d_l[k_d] = Some(d_l_to_full.len());
                d_l_to_full.push(k_d);
            }
            if d_u[k_d].is_finite() {
                full_to_d_u[k_d] = Some(d_u_to_full.len());
                d_u_to_full.push(k_d);
            }
        }
        Self {
            n_d_l: d_l_to_full.len(),
            n_d_u: d_u_to_full.len(),
            full_to_d_l,
            full_to_d_u,
            d_l_to_full,
            d_u_to_full,
        }
    }

    /// d-block size (= length of `full_to_d_l` / `full_to_d_u`).
    pub fn n_d(&self) -> usize {
        self.full_to_d_l.len()
    }

    /// Project a length-`n_d` slice onto the compressed lower-bound
    /// block. Result has length `n_d_l`. Mirrors `Pd_L_^T · v`.
    pub fn project_l(&self, full: &[f64]) -> Vec<f64> {
        debug_assert_eq!(full.len(), self.n_d());
        self.d_l_to_full.iter().map(|&k_d| full[k_d]).collect()
    }

    /// Project a length-`n_d` slice onto the compressed upper-bound
    /// block.
    pub fn project_u(&self, full: &[f64]) -> Vec<f64> {
        debug_assert_eq!(full.len(), self.n_d());
        self.d_u_to_full.iter().map(|&k_d| full[k_d]).collect()
    }

    /// Expand a compressed lower-bound vector back to length `n_d`,
    /// padding with `pad` for d-rows without a lower bound. Mirrors
    /// `Pd_L_ · v_compressed`.
    pub fn expand_l(&self, compressed: &[f64], pad: f64) -> Vec<f64> {
        debug_assert_eq!(compressed.len(), self.n_d_l);
        let mut out = vec![pad; self.n_d()];
        for (k, &k_d) in self.d_l_to_full.iter().enumerate() {
            out[k_d] = compressed[k];
        }
        out
    }

    /// Expand a compressed upper-bound vector back to length `n_d`.
    pub fn expand_u(&self, compressed: &[f64], pad: f64) -> Vec<f64> {
        debug_assert_eq!(compressed.len(), self.n_d_u);
        let mut out = vec![pad; self.n_d()];
        for (k, &k_d) in self.d_u_to_full.iter().enumerate() {
            out[k_d] = compressed[k];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_problem() {
        let layout = DBoundLayout::new(&[], &[]);
        assert_eq!(layout.n_d_l, 0);
        assert_eq!(layout.n_d_u, 0);
        assert_eq!(layout.n_d(), 0);
    }

    #[test]
    fn no_slack_bounds() {
        let neg = f64::NEG_INFINITY;
        let pos = f64::INFINITY;
        let layout = DBoundLayout::new(&[neg, neg], &[pos, pos]);
        assert_eq!(layout.n_d_l, 0);
        assert_eq!(layout.n_d_u, 0);
    }

    #[test]
    fn one_sided_mixed_slack_bounds() {
        let neg = f64::NEG_INFINITY;
        let pos = f64::INFINITY;
        // d-row 0: lower only; d-row 1: upper only; d-row 2: free; d-row 3: two-sided.
        let layout = DBoundLayout::new(&[0.0, neg, neg, -1.0], &[pos, 5.0, pos, 1.0]);
        assert_eq!(layout.n_d_l, 2);
        assert_eq!(layout.n_d_u, 2);
        assert_eq!(layout.d_l_to_full, vec![0, 3]);
        assert_eq!(layout.d_u_to_full, vec![1, 3]);
        assert_eq!(layout.full_to_d_l, vec![Some(0), None, None, Some(1)]);
        assert_eq!(layout.full_to_d_u, vec![None, Some(0), None, Some(1)]);
    }

    #[test]
    fn project_round_trip() {
        let neg = f64::NEG_INFINITY;
        let pos = f64::INFINITY;
        let layout = DBoundLayout::new(&[0.0, neg, -1.0], &[pos, 5.0, 1.0]);
        let full = vec![10.0, 20.0, 30.0];
        assert_eq!(layout.project_l(&full), vec![10.0, 30.0]);
        assert_eq!(layout.project_u(&full), vec![20.0, 30.0]);
    }

    #[test]
    fn expand_pads_zero_for_missing_bound() {
        let neg = f64::NEG_INFINITY;
        let pos = f64::INFINITY;
        let layout = DBoundLayout::new(&[0.0, neg, -1.0], &[pos, 5.0, 1.0]);
        let vl_compressed = vec![1.0, 2.0];
        let expanded = layout.expand_l(&vl_compressed, 0.0);
        assert_eq!(expanded, vec![1.0, 0.0, 2.0]);
    }
}
