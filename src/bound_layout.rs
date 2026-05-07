//! Variable-bound layout — ripopt's mirror of Ipopt's `Px_L_` / `Px_U_`
//! ExpansionMatrix from `IpOrigIpoptNLP.hpp`.
//!
//! Ipopt 3.14 stores `x_L` (size `n_x_l` = count of finite *lower* bounds)
//! and `x_U` (size `n_x_u`) compressed: only finite entries are present, and
//! the expansion matrices `Px_L_` / `Px_U_` map a compressed-index `k` back
//! to the full-`n` variable index. The bound multipliers `z_L` (size n_x_l)
//! and `z_U` (size n_x_u) live on the same compressed index space.
//!
//! ripopt today stores `x_l`, `x_u`, `z_l`, `z_u`, `dz_l`, `dz_u` at full
//! length `n` with `±∞` (for bounds) or `0` (for multipliers) for the
//! unbounded sides. Phase 6 of the data-layout refactor introduces native
//! compressed multiplier storage; Phase 7 follows for the bounds.
//!
//! Cross-references (paths under `ref/Ipopt/src/`):
//!   * `Px_L_` / `Px_U_` ExpansionMatrix declarations:
//!     `Algorithm/IpOrigIpoptNLP.hpp:197-219`
//!   * `x_L_` / `x_U_` storage: same file lines 220-231
//!   * Construction in TNLPAdapter: `Interfaces/IpTNLPAdapter.cpp:632-664`
//!
//! See `docs/V0.8_DATA_LAYOUT_REFACTOR_PLAN.md` Phase 6 for the migration.

/// Maps each full-index variable to its position in the compressed
/// lower-bound block (if any) and the compressed upper-bound block (if any),
/// plus the inverse maps from compressed-index back to full-index. Built
/// once per `SolverState` from `x_l` / `x_u` (a finite bound is one with a
/// non-infinite value).
///
/// Mirrors Ipopt's `Px_L_` / `Px_U_` ExpansionMatrix pair.
#[derive(Debug, Clone)]
pub struct BoundLayout {
    /// Number of finite lower bounds (= dim of `z_L`, `x_L`, `Px_L_` columns).
    pub n_x_l: usize,
    /// Number of finite upper bounds (= dim of `z_U`, `x_U`, `Px_U_` columns).
    pub n_x_u: usize,
    /// `full_to_x_l[i] = Some(k)` iff variable `i` has a finite lower bound
    /// and is the `k`-th compressed lower-bound entry; `None` otherwise.
    /// Mirrors Ipopt's `Px_L_^T` action on a full-`n` vector.
    pub full_to_x_l: Vec<Option<usize>>,
    /// `full_to_x_u[i] = Some(k)` iff variable `i` has a finite upper bound.
    /// Mirrors Ipopt's `Px_U_^T` action.
    pub full_to_x_u: Vec<Option<usize>>,
    /// `x_l_to_full[k]` is the full-index of the `k`-th finite lower bound.
    /// Inverse of `full_to_x_l`. Mirrors the ExpansionMatrix column->row map
    /// of `Px_L_` (`IpOrigIpoptNLP.hpp:197`).
    pub x_l_to_full: Vec<usize>,
    /// `x_u_to_full[k]` is the full-index of the `k`-th finite upper bound.
    /// Inverse of `full_to_x_u`. Mirrors `Px_U_`.
    pub x_u_to_full: Vec<usize>,
}

impl BoundLayout {
    /// Build the layout by classifying each variable's lower and upper
    /// bound as finite / infinite. A bound is finite iff `b.is_finite()`
    /// (i.e. neither `±∞` nor `NaN`). This matches Ipopt's TNLPAdapter
    /// finite-bound test (`IpTNLPAdapter.cpp:632-664`).
    pub fn new(x_l: &[f64], x_u: &[f64]) -> Self {
        let n = x_l.len();
        debug_assert_eq!(x_u.len(), n);
        let mut full_to_x_l = vec![None; n];
        let mut full_to_x_u = vec![None; n];
        let mut x_l_to_full = Vec::new();
        let mut x_u_to_full = Vec::new();
        for i in 0..n {
            if x_l[i].is_finite() {
                full_to_x_l[i] = Some(x_l_to_full.len());
                x_l_to_full.push(i);
            }
            if x_u[i].is_finite() {
                full_to_x_u[i] = Some(x_u_to_full.len());
                x_u_to_full.push(i);
            }
        }
        Self {
            n_x_l: x_l_to_full.len(),
            n_x_u: x_u_to_full.len(),
            full_to_x_l,
            full_to_x_u,
            x_l_to_full,
            x_u_to_full,
        }
    }

    /// Variable count (= length of `full_to_x_l` / `full_to_x_u`).
    pub fn n(&self) -> usize {
        self.full_to_x_l.len()
    }

    /// Project a full-`n` slice onto the compressed lower-bound block.
    /// Result has length `n_x_l`; entry `k` reads `full[x_l_to_full[k]]`.
    /// Mirrors Ipopt's `Px_L_^T · v` action.
    pub fn project_l(&self, full: &[f64]) -> Vec<f64> {
        debug_assert_eq!(full.len(), self.n());
        self.x_l_to_full.iter().map(|&i| full[i]).collect()
    }

    /// Project a full-`n` slice onto the compressed upper-bound block.
    pub fn project_u(&self, full: &[f64]) -> Vec<f64> {
        debug_assert_eq!(full.len(), self.n());
        self.x_u_to_full.iter().map(|&i| full[i]).collect()
    }

    /// Expand a compressed lower-bound vector (length `n_x_l`) back to
    /// full `n` length, padding with `pad` for variables without a lower
    /// bound. Mirrors Ipopt's `Px_L_ · v_compressed` (with zero pad for
    /// `MultVector`, but ripopt's combined `z_l` uses `0.0` and `x_l` uses
    /// `f64::NEG_INFINITY`).
    pub fn expand_l(&self, compressed: &[f64], pad: f64) -> Vec<f64> {
        debug_assert_eq!(compressed.len(), self.n_x_l);
        let mut out = vec![pad; self.n()];
        for (k, &i) in self.x_l_to_full.iter().enumerate() {
            out[i] = compressed[k];
        }
        out
    }

    /// Expand a compressed upper-bound vector back to full `n` length.
    pub fn expand_u(&self, compressed: &[f64], pad: f64) -> Vec<f64> {
        debug_assert_eq!(compressed.len(), self.n_x_u);
        let mut out = vec![pad; self.n()];
        for (k, &i) in self.x_u_to_full.iter().enumerate() {
            out[i] = compressed[k];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_problem() {
        let layout = BoundLayout::new(&[], &[]);
        assert_eq!(layout.n_x_l, 0);
        assert_eq!(layout.n_x_u, 0);
        assert_eq!(layout.n(), 0);
    }

    #[test]
    fn no_bounds_at_all() {
        let neg = f64::NEG_INFINITY;
        let pos = f64::INFINITY;
        let layout = BoundLayout::new(&[neg, neg, neg], &[pos, pos, pos]);
        assert_eq!(layout.n_x_l, 0);
        assert_eq!(layout.n_x_u, 0);
        assert_eq!(layout.full_to_x_l, vec![None, None, None]);
        assert_eq!(layout.full_to_x_u, vec![None, None, None]);
    }

    #[test]
    fn two_sided_bounds() {
        let layout = BoundLayout::new(&[0.0, 1.0, -2.0], &[5.0, 4.0, 3.0]);
        assert_eq!(layout.n_x_l, 3);
        assert_eq!(layout.n_x_u, 3);
        assert_eq!(layout.x_l_to_full, vec![0, 1, 2]);
        assert_eq!(layout.x_u_to_full, vec![0, 1, 2]);
        assert_eq!(layout.full_to_x_l, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(layout.full_to_x_u, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn one_sided_mixed() {
        let neg = f64::NEG_INFINITY;
        let pos = f64::INFINITY;
        // var 0: lower only; var 1: upper only; var 2: free; var 3: two-sided
        let layout = BoundLayout::new(&[0.0, neg, neg, -1.0], &[pos, 5.0, pos, 1.0]);
        assert_eq!(layout.n_x_l, 2);
        assert_eq!(layout.n_x_u, 2);
        assert_eq!(layout.x_l_to_full, vec![0, 3]);
        assert_eq!(layout.x_u_to_full, vec![1, 3]);
        assert_eq!(
            layout.full_to_x_l,
            vec![Some(0), None, None, Some(1)]
        );
        assert_eq!(
            layout.full_to_x_u,
            vec![None, Some(0), None, Some(1)]
        );
    }

    #[test]
    fn project_round_trip_lower() {
        let neg = f64::NEG_INFINITY;
        let pos = f64::INFINITY;
        let layout = BoundLayout::new(&[0.0, neg, -1.0], &[pos, 5.0, 1.0]);
        let full = vec![10.0, 20.0, 30.0];
        assert_eq!(layout.project_l(&full), vec![10.0, 30.0]);
        assert_eq!(layout.project_u(&full), vec![20.0, 30.0]);
    }

    #[test]
    fn expand_pads_with_zero_for_missing_bound() {
        let neg = f64::NEG_INFINITY;
        let pos = f64::INFINITY;
        let layout = BoundLayout::new(&[0.0, neg, -1.0], &[pos, 5.0, 1.0]);
        // compressed lower-bound multiplier: var 0 → 1.0, var 2 → 2.0
        let zl_compressed = vec![1.0, 2.0];
        let expanded = layout.expand_l(&zl_compressed, 0.0);
        assert_eq!(expanded, vec![1.0, 0.0, 2.0]);
    }

    #[test]
    fn nan_treated_as_non_finite() {
        // Defensive: NaN bounds should not be classified as finite.
        let layout = BoundLayout::new(&[f64::NAN], &[f64::NAN]);
        assert_eq!(layout.n_x_l, 0);
        assert_eq!(layout.n_x_u, 0);
    }
}
