/// An entry in the filter, representing a constraint violation / objective pair.
#[derive(Debug, Clone, Copy)]
pub struct FilterEntry {
    /// Constraint violation (infeasibility measure).
    pub theta: f64,
    /// Barrier objective value.
    pub phi: f64,
}

/// The filter mechanism for the line search.
///
/// Maintains a set of (theta, phi) pairs. A trial point is acceptable if it
/// "dominates" (improves upon) all entries in the filter by sufficient margin.
pub struct Filter {
    entries: Vec<FilterEntry>,
    /// Maximum constraint violation allowed (theta_max).
    theta_max: f64,
    /// Margin for filter acceptance (gamma_theta).
    gamma_theta: f64,
    /// Margin for filter acceptance (gamma_phi).
    gamma_phi: f64,
    /// Threshold for switching condition (theta_min).
    theta_min: f64,
    /// Exponent for switching condition.
    s_theta: f64,
    /// Exponent for switching condition.
    s_phi: f64,
    /// Armijo parameter (eta_phi).
    eta_phi: f64,
    /// Small constant delta for filter margin.
    delta: f64,
    /// Maximum permitted log10 increase in the barrier objective per step,
    /// used by the `obj_max_inc` divergence guard (Ipopt default `5.0`,
    /// `IpFilterLSAcceptor.cpp:132-139`). The trial is rejected when
    /// `trial_phi > reference_phi` *and* `log10(trial_phi - reference_phi)
    /// > obj_max_inc + basval`, where
    /// `basval = max(1.0, log10(|reference_phi|))`. Exact mirror of
    /// `IpFilterLSAcceptor.cpp:478-493`.
    obj_max_inc: f64,
    /// Whether `set_theta_min_from_initial` has already seeded
    /// `theta_max`/`theta_min` from the initial constraint violation.
    /// Mirrors Ipopt's lazy-init sentinel (theta_max_ < 0.0) at
    /// IpFilterLSAcceptor.cpp:325-339: once seeded, the bounds are
    /// fixed for the rest of the solve.
    theta_init_set: bool,
}

impl Filter {
    /// Create a new filter with the given maximum constraint violation.
    pub fn new(theta_max: f64) -> Self {
        Self {
            entries: Vec::new(),
            theta_max,
            gamma_theta: 1e-5,
            gamma_phi: 1e-8,
            theta_min: 1e-4 * theta_max.max(1e-4),
            s_theta: 1.1,
            s_phi: 2.3,
            eta_phi: 1e-8,
            delta: 1.0,
            obj_max_inc: 5.0,
            theta_init_set: false,
        }
    }

    /// Initialize `theta_min` and `theta_max` from the initial constraint
    /// violation. This is a one-shot operation — subsequent calls are
    /// no-ops, mirroring Ipopt's IpFilterLSAcceptor.cpp:325-339 which
    /// seeds these bounds once at the first acceptability check (when the
    /// `theta_max_ < 0.0` sentinel is still set) and never resets them.
    /// Resetting on every μ change lets the filter envelope grow over
    /// time and admits iterates earlier filter entries had rejected.
    ///
    /// T3.3: floor is `1.0`, matching Ipopt
    /// `IpFilterLSAcceptor.cpp:325-335`'s `Max(Number(1.0), reference_theta_)`.
    /// The previous `1e-4` floor produced 10⁴× tighter theta_max on
    /// near-feasible starts (`theta_init = 1e-6` → `theta_max = 1.0`
    /// instead of Ipopt's `1e4`), rejecting trial points Ipopt would
    /// accept.
    pub fn set_theta_min_from_initial(&mut self, theta_init: f64) {
        if self.theta_init_set {
            return;
        }
        let floor = theta_init.max(1.0);
        self.theta_min = 1e-4 * floor;
        self.theta_max = 1e4 * floor;
        self.theta_init_set = true;
    }

    /// Check if a trial point (theta, phi) is acceptable to the filter.
    /// Returns true if the point is acceptable (not dominated by any filter entry).
    pub fn is_acceptable(&self, theta: f64, phi: f64) -> bool {
        if theta.is_nan() || phi.is_nan() {
            return false;
        }
        if theta > self.theta_max {
            return false;
        }
        for entry in &self.entries {
            if theta >= (1.0 - self.gamma_theta) * entry.theta
                && phi >= entry.phi - self.gamma_phi * entry.theta
            {
                return false;
            }
        }
        true
    }

    /// Check the switching condition: whether we should use the objective (phi)
    /// criterion instead of the filter.
    ///
    /// Returns true if the current constraint violation is small enough and the
    /// directional derivative indicates sufficient objective decrease.
    pub fn switching_condition(
        &self,
        theta_current: f64,
        grad_phi_step: f64,
        alpha: f64,
    ) -> bool {
        // Ipopt switching condition: alpha makes this depend on step length,
        // so as alpha shrinks during backtracking, we properly fall back to
        // h-type (constraint reduction) acceptance.
        grad_phi_step < 0.0
            && theta_current < self.theta_min
            && alpha * (-grad_phi_step).powf(self.s_phi)
                > self.delta * theta_current.powf(self.s_theta)
    }

    /// Check the Armijo sufficient decrease condition.
    pub fn armijo_condition(
        &self,
        phi_current: f64,
        phi_trial: f64,
        grad_phi_step: f64,
        alpha: f64,
    ) -> bool {
        phi_trial <= phi_current + self.eta_phi * alpha * grad_phi_step
    }

    /// Check if a trial point provides sufficient constraint reduction
    /// compared to the current point.
    pub fn sufficient_infeasibility_reduction(
        &self,
        theta_current: f64,
        theta_trial: f64,
    ) -> bool {
        theta_trial <= (1.0 - self.gamma_theta) * theta_current
    }

    /// Check if a trial point is acceptable via either the filter or the
    /// sufficient decrease conditions (Armijo or constraint reduction).
    ///
    /// Returns (acceptable, use_switching) where:
    /// - acceptable: whether the step should be accepted
    /// - use_switching: whether the switching condition was used (affects filter update)
    ///
    /// Follows Ipopt's structure (IpFilterLSAcceptor.cpp:311-437):
    /// 1. Determine step type (f-type via switching/Armijo, or h-type via reduction)
    /// 2. Check filter acceptability
    /// When the switching condition holds, the step is purely f-type — no h-type fallback.
    pub fn check_acceptability(
        &self,
        theta_current: f64,
        phi_current: f64,
        theta_trial: f64,
        phi_trial: f64,
        grad_phi_step: f64,
        alpha: f64,
    ) -> (bool, bool) {
        // Reject if theta exceeds maximum or NaN
        if theta_trial > self.theta_max || theta_trial.is_nan() || phi_trial.is_nan() {
            return (false, false);
        }

        // T3.4: obj_max_inc divergence guard
        // (IpFilterLSAcceptor.cpp:478-493). Reject trials where the
        // barrier objective is climbing faster than `10^obj_max_inc`
        // relative to the current iterate's φ. Gated by
        // `trial_phi > reference_phi`; otherwise this is a no-op.
        if !self.passes_obj_max_inc(phi_current, phi_trial) {
            return (false, false);
        }

        // Determine step type and check type-specific condition
        let (type_ok, is_switching) = if self.switching_condition(theta_current, grad_phi_step, alpha) {
            // f-type: Armijo on barrier objective. No h-type fallback.
            (self.armijo_condition(phi_current, phi_trial, grad_phi_step, alpha), true)
        } else {
            // h-type: sufficient decrease in theta or phi
            let ok = self.sufficient_infeasibility_reduction(theta_current, theta_trial)
                || phi_trial <= phi_current - self.gamma_phi * theta_current;
            (ok, false)
        };

        if !type_ok {
            return (false, false);
        }

        // Check filter acceptability
        if !self.is_acceptable(theta_trial, phi_trial) {
            return (false, false);
        }

        (true, is_switching)
    }

    /// T3.4: Ipopt's `obj_max_inc` divergence guard
    /// (`IpFilterLSAcceptor.cpp:478-493`). Returns `true` when the trial
    /// is acceptable on the divergence criterion (which includes the
    /// case `trial_phi <= reference_phi`); returns `false` only when
    /// `trial_phi > reference_phi` *and*
    /// `log10(trial_phi - reference_phi) > obj_max_inc + basval`, where
    /// `basval = max(1.0, log10(|reference_phi|))`.
    pub fn passes_obj_max_inc(&self, reference_phi: f64, trial_phi: f64) -> bool {
        if !(trial_phi > reference_phi) {
            return true;
        }
        let basval = if reference_phi.abs() > 10.0 {
            reference_phi.abs().log10()
        } else {
            1.0
        };
        (trial_phi - reference_phi).log10() <= self.obj_max_inc + basval
    }

    /// Override the `obj_max_inc` default (5.0). Allows tests and the
    /// IPM driver to plumb `SolverOptions::obj_max_inc` through.
    pub fn set_obj_max_inc(&mut self, value: f64) {
        self.obj_max_inc = value;
    }

    /// Add a (theta, phi) pair to the filter.
    pub fn add(&mut self, theta: f64, phi: f64) {
        // Remove dominated entries
        self.entries.retain(|e| {
            !(theta <= (1.0 - self.gamma_theta) * e.theta
                && phi <= e.phi - self.gamma_phi * e.theta)
        });
        self.entries.push(FilterEntry { theta, phi });
    }

    /// Compute problem-dependent minimum step size for the line search (Ipopt formula).
    /// Returns alpha_min based on filter parameters and current iterate.
    ///
    /// Matches Ipopt's CalculateAlphaMin (IpFilterLSAcceptor.cpp:450-469):
    /// - When gradBarrTDelta >= 0 (no descent): alpha_min = alpha_min_frac * gamma_theta
    /// - When gradBarrTDelta < 0: alpha_min from filter parameters
    pub fn compute_alpha_min(&self, theta_current: f64, grad_phi_step: f64) -> f64 {
        let alpha_min_frac = 0.05; // Ipopt default
        if grad_phi_step >= 0.0 {
            // No barrier descent direction (common for feasibility problems with obj=0).
            // Ipopt uses gamma_theta here, not epsilon — this triggers restoration after
            // ~12 backtracking steps instead of ~50.
            return alpha_min_frac * self.gamma_theta;
        }
        let neg_gphi = -grad_phi_step;
        let term1 = self.gamma_theta;
        let term2 = self.gamma_phi * theta_current / neg_gphi;
        let mut alpha_min = alpha_min_frac * term1.min(term2);
        if theta_current <= self.theta_min {
            let term3 =
                self.delta * theta_current.powf(self.s_theta) / neg_gphi.powf(self.s_phi);
            alpha_min = alpha_min.min(alpha_min_frac * term3);
        }
        alpha_min.max(1e-15)
    }

    /// Augment the filter at restoration entry (Ipopt's PrepareRestoPhaseStart,
    /// IpFilterLSAcceptor.cpp:898-901). Adds an entry at
    /// (phi - gamma_phi*theta, (1 - gamma_theta)*theta) so the restored iterate
    /// cannot hand back a point "as bad or worse" than the pre-restoration one.
    /// Also bumps theta_max.
    pub fn augment_for_restoration(&mut self, theta_current: f64, phi_current: f64) {
        self.theta_max = self.theta_max.max(1e4 * theta_current.max(1e-4));
        let guard_theta = (1.0 - self.gamma_theta) * theta_current;
        let guard_phi = phi_current - self.gamma_phi * theta_current;
        self.add(guard_theta, guard_phi);
    }

    /// Reset the filter (used when barrier parameter decreases).
    pub fn reset(&mut self) {
        self.entries.clear();
    }

    /// Save filter entries for watchdog mechanism.
    pub fn save_entries(&self) -> Vec<FilterEntry> {
        self.entries.clone()
    }

    /// Restore filter entries (watchdog rollback).
    pub fn restore_entries(&mut self, entries: Vec<FilterEntry>) {
        self.entries = entries;
    }

    /// Number of entries in the filter.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the filter is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get theta_max.
    pub fn theta_max(&self) -> f64 {
        self.theta_max
    }

    /// Get gamma_theta.
    pub fn gamma_theta(&self) -> f64 {
        self.gamma_theta
    }

    /// Get gamma_phi.
    pub fn gamma_phi(&self) -> f64 {
        self.gamma_phi
    }

    /// Get the current filter entries (read-only).
    pub fn entries(&self) -> &[FilterEntry] {
        &self.entries
    }
}

/// Compute the maximum step size satisfying the fraction-to-boundary rule.
///
/// Returns the largest alpha in (0, 1] such that:
///   s + alpha * ds >= (1 - tau) * s   for all components
///
/// This ensures slacks/multipliers stay strictly positive.
pub fn fraction_to_boundary(s: &[f64], ds: &[f64], tau: f64) -> f64 {
    let mut alpha_max = 1.0;
    for (si, dsi) in s.iter().zip(ds.iter()) {
        if *dsi < 0.0 {
            let ratio = -tau * si / dsi;
            if ratio < alpha_max {
                alpha_max = ratio;
            }
        }
    }
    alpha_max.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fraction_to_boundary() {
        let s = vec![1.0, 2.0, 0.5];
        let ds = vec![-0.5, 0.1, -0.3];
        let tau = 0.99;
        let alpha = fraction_to_boundary(&s, &ds, tau);
        // For s[0]: -0.99 * 1.0 / (-0.5) = 1.98
        // For s[2]: -0.99 * 0.5 / (-0.3) = 1.65
        // Both > 1.0, so alpha_max = 1.0
        assert!((alpha - 1.0).abs() < 1e-12);

        // Case where step is limited
        let ds2 = vec![-2.0, 0.1, -0.3];
        let alpha2 = fraction_to_boundary(&s, &ds2, tau);
        // For s[0]: -0.99 * 1.0 / (-2.0) = 0.495
        assert!((alpha2 - 0.495).abs() < 1e-12);
    }

    #[test]
    fn test_filter_empty_accepts_everything() {
        let filter = Filter::new(100.0);
        assert!(filter.is_acceptable(1.0, 1.0));
        assert!(filter.is_acceptable(50.0, 50.0));
    }

    #[test]
    fn test_filter_rejects_over_theta_max() {
        let filter = Filter::new(100.0);
        assert!(!filter.is_acceptable(200.0, 0.0));
    }

    #[test]
    fn test_filter_rejects_dominated_point() {
        let mut filter = Filter::new(100.0);
        filter.add(1.0, 1.0);
        // A point that is worse in both theta and phi
        assert!(!filter.is_acceptable(1.0, 1.0));
        // A point that improves sufficiently in theta
        assert!(filter.is_acceptable(0.5, 1.0));
    }

    #[test]
    fn test_filter_reset() {
        let mut filter = Filter::new(100.0);
        filter.add(1.0, 1.0);
        filter.add(0.5, 2.0);
        assert_eq!(filter.len(), 2);
        filter.reset();
        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
    }

    #[test]
    fn test_switching_condition() {
        let mut filter = Filter::new(100.0);
        filter.set_theta_min_from_initial(10.0);
        // theta_min = 1e-4 * 10.0 = 1e-3
        // Small theta + negative directional derivative + alpha=1.0 -> switching
        // alpha * 1.0^2.3 = 1.0 > delta * (1e-4)^1.1 ≈ 6.3e-5 -> true
        assert!(filter.switching_condition(1e-4, -1.0, 1.0));
        // Large theta -> no switching (theta >= theta_min)
        assert!(!filter.switching_condition(10.0, -1.0, 1.0));
        // Positive directional derivative -> no switching
        assert!(!filter.switching_condition(1e-4, 1.0, 1.0));
        // Very small alpha -> switching should turn off
        // alpha=1e-20 * 1.0^2.3 = 1e-20, vs delta * (1e-4)^1.1 ≈ 6.3e-5 -> false
        assert!(!filter.switching_condition(1e-4, -1.0, 1e-20));
    }

    #[test]
    fn test_armijo_condition() {
        let filter = Filter::new(100.0);
        // phi_trial <= phi_current + eta_phi * alpha * grad_phi_step
        // eta_phi = 1e-4, threshold: 10.0 + 1e-4 * 1.0 * (-1.0) = 9.9999
        let phi_current = 10.0;
        let grad_phi_step = -1.0;
        let alpha = 1.0;
        assert!(filter.armijo_condition(phi_current, 9.0, grad_phi_step, alpha));
        assert!(!filter.armijo_condition(phi_current, 10.0, grad_phi_step, alpha));
    }

    #[test]
    fn test_sufficient_infeasibility_reduction() {
        let filter = Filter::new(100.0);
        // gamma_theta = 1e-5
        // theta_trial <= (1 - 1e-5) * theta_current
        let theta_current = 1.0;
        assert!(filter.sufficient_infeasibility_reduction(theta_current, 0.5));
        assert!(!filter.sufficient_infeasibility_reduction(theta_current, 1.0));
    }

    #[test]
    fn test_check_acceptability_switching_mode() {
        let mut filter = Filter::new(100.0);
        filter.set_theta_min_from_initial(10.0);
        // Small theta + negative grad_phi_step → switching condition
        // Then Armijo must pass
        let theta_current = 1e-5;
        let phi_current = 10.0;
        let theta_trial = 1e-6;
        let phi_trial = 9.0; // Satisfies Armijo
        let grad_phi_step = -100.0; // Strong descent
        let alpha = 1.0;

        let (accept, switching) = filter.check_acceptability(
            theta_current, phi_current, theta_trial, phi_trial, grad_phi_step, alpha,
        );
        assert!(accept, "Should be accepted via Armijo");
        assert!(switching, "Should use switching condition");
    }

    #[test]
    fn test_check_acceptability_filter_mode() {
        let mut filter = Filter::new(100.0);
        filter.set_theta_min_from_initial(10.0);
        // Large theta → no switching → use filter
        let theta_current = 5.0;
        let phi_current = 10.0;
        let theta_trial = 2.0; // Sufficient reduction
        let phi_trial = 10.0;
        let grad_phi_step = -0.01; // Weak descent
        let alpha = 1.0;

        let (accept, switching) = filter.check_acceptability(
            theta_current, phi_current, theta_trial, phi_trial, grad_phi_step, alpha,
        );
        assert!(accept, "Should be accepted via filter mode");
        assert!(!switching, "Should NOT use switching");
    }

    #[test]
    fn test_check_acceptability_rejected() {
        let mut filter = Filter::new(100.0);
        filter.set_theta_min_from_initial(10.0);
        // Add a filter entry that dominates the trial
        filter.add(0.5, 5.0);
        // Trial point is worse than filter
        let (accept, _) = filter.check_acceptability(
            1.0, 10.0, 0.6, 6.0, -0.01, 1.0,
        );
        assert!(!accept, "Should be rejected by filter");
    }

    #[test]
    fn test_filter_dominated_entry_removal() {
        let mut filter = Filter::new(100.0);
        // Use entries that don't dominate each other: one has lower theta, other has lower phi
        filter.add(0.5, 10.0);
        filter.add(2.0, 5.0);
        assert_eq!(filter.len(), 2);
        // Add an entry that dominates both
        filter.add(0.01, 0.01);
        // The dominating entry should have removed both
        assert!(filter.len() <= 2);
        // The new entry should be in the filter
        assert!(filter.is_acceptable(0.005, -1.0));
    }

    #[test]
    fn test_filter_rejects_nan() {
        let filter = Filter::new(100.0);
        assert!(!filter.is_acceptable(f64::NAN, 1.0));
        assert!(!filter.is_acceptable(1.0, f64::NAN));
        let (accept, _) = filter.check_acceptability(
            1.0, 10.0, f64::NAN, 5.0, -1.0, 1.0,
        );
        assert!(!accept);
        let (accept2, _) = filter.check_acceptability(
            1.0, 10.0, 0.5, f64::NAN, -1.0, 1.0,
        );
        assert!(!accept2);
    }

    #[test]
    fn test_filter_margin_boundary() {
        // A point exactly at the boundary of the filter margin is rejected
        // (strict-inequality semantics in is_acceptable).
        let mut filter = Filter::new(100.0);
        filter.add(1.0, 1.0);
        // gamma_theta = 1e-5, gamma_phi = 1e-8
        // Boundary: theta >= (1 - 1e-5) * 1.0 = 0.99999 AND phi >= 1.0 - 1e-8 * 1.0
        let theta_boundary = 0.99999;
        let phi_boundary = 1.0 - 1e-8;
        assert!(!filter.is_acceptable(theta_boundary, phi_boundary),
            "At-boundary point should be rejected (dominated)");
        // Just inside the margin (sufficient improvement in theta): accepted
        assert!(filter.is_acceptable(theta_boundary - 1e-6, phi_boundary));
    }

    #[test]
    fn test_compute_alpha_min_no_descent() {
        // grad_phi_step >= 0: no descent direction.
        // alpha_min = alpha_min_frac * gamma_theta = 0.05 * 1e-5 = 5e-7
        let filter = Filter::new(100.0);
        let alpha_min = filter.compute_alpha_min(1.0, 0.0);
        assert!((alpha_min - 5e-7).abs() < 1e-15,
            "no-descent alpha_min should be 0.05*gamma_theta, got {}", alpha_min);
        // Strictly positive grad: same branch
        let alpha_min_pos = filter.compute_alpha_min(1.0, 0.1);
        assert!((alpha_min_pos - 5e-7).abs() < 1e-15);
    }

    #[test]
    fn test_compute_alpha_min_large_theta() {
        // grad_phi_step < 0, theta > theta_min: only gamma_theta and gamma_phi terms.
        let mut filter = Filter::new(100.0);
        filter.set_theta_min_from_initial(10.0); // theta_min = 1e-3
        let theta_current = 1.0; // > theta_min
        let grad = -2.0;
        // term1 = gamma_theta = 1e-5
        // term2 = gamma_phi * theta / |grad| = 1e-8 * 1.0 / 2.0 = 5e-9
        // alpha_min = 0.05 * min(1e-5, 5e-9) = 0.05 * 5e-9 = 2.5e-10
        let alpha_min = filter.compute_alpha_min(theta_current, grad);
        assert!((alpha_min - 2.5e-10).abs() < 1e-20,
            "large-theta alpha_min formula mismatch, got {}", alpha_min);
    }

    #[test]
    fn test_compute_alpha_min_small_theta_adds_switching_term() {
        // grad_phi_step < 0, theta <= theta_min: term3 (switching-condition) refinement kicks in.
        let mut filter = Filter::new(100.0);
        filter.set_theta_min_from_initial(10.0); // theta_min = 1e-3
        let theta_current = 1e-4; // <= theta_min
        let grad = -1.0;
        // alpha_min without term3 = 0.05 * min(1e-5, 1e-8 * 1e-4 / 1) = 0.05 * 1e-12 = 5e-14
        // term3 = delta * theta^s_theta / |grad|^s_phi = 1.0 * (1e-4)^1.1 / 1.0^2.3 ≈ 5.01e-5
        // alpha_min with term3 = min(5e-14, 0.05*5.01e-5=2.51e-6) -> 5e-14
        // The small theta/grad case is dominated by term2; confirm we don't violate the floor.
        let alpha_min = filter.compute_alpha_min(theta_current, grad);
        assert!(alpha_min >= 1e-15, "alpha_min floor not enforced, got {}", alpha_min);
        // Now use a much steeper descent so term2 grows and term3 becomes the binding term.
        let alpha_min_steep = filter.compute_alpha_min(theta_current, -1e-10);
        // With very small |grad|, term2 = 1e-8 * 1e-4 / 1e-10 = 1e-2; term3 bounded by term1 first
        assert!(alpha_min_steep >= 1e-15);
    }

    #[test]
    fn test_watchdog_save_restore_round_trip() {
        let mut filter = Filter::new(100.0);
        filter.add(1.0, 5.0);
        filter.add(0.5, 10.0);
        let saved = filter.save_entries();
        assert_eq!(saved.len(), 2);
        filter.reset();
        assert!(filter.is_empty());
        filter.restore_entries(saved);
        assert_eq!(filter.len(), 2);
        // Restored filter should still reject dominated points
        assert!(!filter.is_acceptable(1.0, 5.0));
        assert!(!filter.is_acceptable(0.5, 10.0));
    }

    #[test]
    fn test_augment_for_restoration_adds_entry() {
        let mut filter = Filter::new(100.0);
        let initial_theta_max = filter.theta_max();
        let theta = 50.0; // Large enough to bump theta_max
        let phi = 10.0;
        filter.augment_for_restoration(theta, phi);
        // Should have added an entry at ((1-gamma_theta)*theta, phi - gamma_phi*theta)
        assert_eq!(filter.len(), 1);
        // theta_max should be bumped to at least 1e4 * theta
        assert!(filter.theta_max() >= 1e4 * theta,
            "theta_max not bumped: {} < {}", filter.theta_max(), 1e4 * theta);
        assert!(filter.theta_max() >= initial_theta_max,
            "augmentation should not shrink theta_max");
        // A point worse than the guard should be rejected
        assert!(!filter.is_acceptable(theta, phi));
    }

    #[test]
    fn test_check_acceptability_h_type_phi_only_reduction() {
        // h-type (non-switching) path accepts on EITHER theta reduction OR phi reduction.
        // Here theta increases slightly but phi drops by more than gamma_phi*theta_current.
        let mut filter = Filter::new(100.0);
        filter.set_theta_min_from_initial(10.0);
        let theta_current = 5.0;
        let phi_current = 100.0;
        // theta_trial slightly worse: (1-gamma_theta)*theta_current = 4.99995; trial 5.0 > that
        let theta_trial = 5.0;
        // phi_trial satisfies phi_trial <= phi_current - gamma_phi * theta_current
        // = 100.0 - 1e-8 * 5.0 = 99.99999995. Pick phi_trial = 50.0 (way below).
        let phi_trial = 50.0;
        let grad = -0.01; // weak descent, theta_current > theta_min -> h-type
        let alpha = 1.0;
        let (accept, switching) = filter.check_acceptability(
            theta_current, phi_current, theta_trial, phi_trial, grad, alpha,
        );
        assert!(accept, "h-type phi-only reduction must be accepted");
        assert!(!switching);
    }

    #[test]
    fn test_check_acceptability_h_type_rejected_no_reduction() {
        // h-type path: trial fails BOTH theta reduction AND phi reduction -> rejected
        let mut filter = Filter::new(100.0);
        filter.set_theta_min_from_initial(10.0);
        let theta_current = 5.0;
        let phi_current = 100.0;
        // Trial: theta equal (no sufficient reduction) and phi equal (no reduction)
        let (accept, _) = filter.check_acceptability(
            theta_current, phi_current, 5.0, 100.0, -0.01, 1.0,
        );
        assert!(!accept, "h-type with no reduction in either must reject");
    }

    #[test]
    fn test_obj_max_inc_passes_when_trial_le_reference() {
        // T3.4: when phi_trial <= phi_reference, the divergence guard
        // is a no-op (returns true).
        let filter = Filter::new(100.0);
        assert!(filter.passes_obj_max_inc(10.0, 5.0), "decrease must pass");
        assert!(filter.passes_obj_max_inc(10.0, 10.0), "equality must pass");
        assert!(filter.passes_obj_max_inc(0.0, -1e6), "negative trial must pass");
    }

    #[test]
    fn test_obj_max_inc_rejects_huge_increase_small_reference() {
        // T3.4: with reference_phi = 1.0 (so |ref| <= 10 → basval = 1),
        // and obj_max_inc = 5 (default), the inequality is
        //   log10(trial - 1) > 5 + 1 = 6  ⇔  trial - 1 > 1e6
        // So trial = 2 is fine (log10(1) = 0 <= 6), but trial = 1e7 + 1 fails.
        let filter = Filter::new(100.0);
        assert!(filter.passes_obj_max_inc(1.0, 1e6 + 0.5), "below 1e6 increase must pass");
        assert!(!filter.passes_obj_max_inc(1.0, 1e7 + 2.0), "above 10^6 increase must fail");
    }

    #[test]
    fn test_obj_max_inc_basval_scales_with_large_reference() {
        // T3.4: when |reference_phi| > 10, basval = log10(|ref|), so the
        // permitted increase scales with |ref|. With ref=1e3, basval=3,
        // permitted log10 increase = 5 + 3 = 8, ie up to 1e8 absolute.
        let filter = Filter::new(100.0);
        assert!(filter.passes_obj_max_inc(1e3, 1e3 + 1e7), "1e7 increase must pass at ref=1e3");
        assert!(!filter.passes_obj_max_inc(1e3, 1e3 + 1e9), "1e9 increase must fail at ref=1e3");
    }

    #[test]
    fn test_obj_max_inc_set_obj_max_inc_setter() {
        // T3.4: tightening the cap to 0.0 makes any strictly positive
        // increase fail (basval >= 1 ⇒ log10(Δ) > 0+1=1 means Δ>10).
        let mut filter = Filter::new(100.0);
        filter.set_obj_max_inc(0.0);
        assert!(filter.passes_obj_max_inc(1.0, 5.0), "Δ=4 < 10, still passes at obj_max_inc=0");
        assert!(!filter.passes_obj_max_inc(1.0, 100.0), "Δ=99 > 10 must fail at obj_max_inc=0");
    }

    #[test]
    fn test_check_acceptability_obj_max_inc_blocks_blowup() {
        // T3.4 end-to-end: a trial that would otherwise pass via h-type
        // (theta sufficient reduction) is rejected when phi explodes
        // beyond the obj_max_inc cap.
        let mut filter = Filter::new(1e10);
        filter.set_theta_min_from_initial(10.0);
        let theta_current = 5.0;
        let phi_current = 1.0;
        let theta_trial = 0.1; // satisfies sufficient infeasibility reduction
        let phi_trial = 1e10; // log10(1e10 - 1) ≈ 10 >> 5 + 1 = 6 ⇒ reject
        let (accept, _) = filter.check_acceptability(
            theta_current, phi_current, theta_trial, phi_trial, -0.01, 1.0,
        );
        assert!(!accept, "obj_max_inc must reject blowup trial");
    }

    #[test]
    fn test_check_acceptability_switching_fails_armijo() {
        // Switching condition holds (small theta + descent) but Armijo fails.
        // Per Ipopt, in this case the step is purely f-type — no h-type fallback.
        let mut filter = Filter::new(100.0);
        filter.set_theta_min_from_initial(10.0);
        let theta_current = 1e-5;
        let phi_current = 10.0;
        let theta_trial = 1e-6; // Would satisfy theta reduction, but that's not allowed here
        let phi_trial = 10.0; // Fails Armijo: phi_trial > phi_current + eta_phi*alpha*grad
        let grad = -100.0;
        let alpha = 1.0;
        let (accept, _) = filter.check_acceptability(
            theta_current, phi_current, theta_trial, phi_trial, grad, alpha,
        );
        assert!(!accept, "Switching+failed-Armijo must reject (no h-type fallback)");
    }

    #[test]
    fn test_set_theta_min_from_initial_is_one_shot() {
        // T0.7: theta_max/theta_min must be seeded ONCE from the initial
        // constraint violation and never reset, mirroring Ipopt
        // IpFilterLSAcceptor.cpp:325-339. Repeated calls (as happens on
        // every μ change in ripopt) must be ignored — otherwise the
        // filter envelope grows over time and admits iterates earlier
        // filter entries had rejected.
        let mut filter = Filter::new(100.0);
        // First call: theta_init = 0.5. T3.3 floor is `Max(1.0, theta_init)`
        // (IpFilterLSAcceptor.cpp:325-335), so theta_max = 1e4 * 1.0 = 1e4.
        filter.set_theta_min_from_initial(0.5);
        let theta_max_init = filter.theta_max();
        assert!((theta_max_init - 1e4).abs() < 1e-9,
            "first init: theta_max should be 1e4 * max(1.0, 0.5) = 1e4, got {}", theta_max_init);

        // Simulate the μ-update path that previously called
        // set_theta_min_from_initial again with a (possibly different) theta.
        // The reset_filter_with_current_theta helper used to do this on every
        // μ change — now it is a no-op for the bounds.
        filter.reset();
        filter.set_theta_min_from_initial(1e-6); // would have shrunk theta_max to 1.0 before T0.7
        assert!((filter.theta_max() - theta_max_init).abs() < 1e-12,
            "T0.7: theta_max must remain seeded at the initial value across resets, got {}",
            filter.theta_max());

        // Even an enormous theta should not bump theta_max via this path
        // (augmentation during restoration is a separate, intentional path).
        filter.set_theta_min_from_initial(1e10);
        assert!((filter.theta_max() - theta_max_init).abs() < 1e-12,
            "T0.7: subsequent calls must be ignored, got {}", filter.theta_max());
    }

    #[test]
    fn test_set_theta_min_from_initial_floor_is_one() {
        // T3.3: floor must be Max(1.0, theta_init), matching Ipopt
        // IpFilterLSAcceptor.cpp:325-335. On a near-feasible start with
        // theta_init = 1e-6 the previous 1e-4 floor produced
        // theta_max = 1e4 * 1e-4 = 1.0, which was 10⁴× tighter than
        // Ipopt's 1e4. The new floor of 1.0 yields theta_max = 1e4.
        let mut filter = Filter::new(100.0);
        filter.set_theta_min_from_initial(1e-6);
        assert!((filter.theta_max() - 1e4).abs() < 1e-9,
            "near-feasible start: theta_max should be 1e4, got {}", filter.theta_max());
        assert!((filter.theta_min - 1e-4).abs() < 1e-12,
            "near-feasible start: theta_min should be 1e-4, got {}", filter.theta_min);

        // For theta_init >= 1.0 the floor passes through unchanged.
        let mut filter2 = Filter::new(100.0);
        filter2.set_theta_min_from_initial(7.5);
        assert!((filter2.theta_max() - 7.5e4).abs() < 1e-6,
            "theta_init=7.5: theta_max should be 7.5e4, got {}", filter2.theta_max());
    }

    #[test]
    fn test_fraction_to_boundary_edge_cases() {
        // All positive steps → alpha = 1.0
        let s = vec![1.0, 2.0, 0.5];
        let ds = vec![0.1, 0.5, 0.3];
        let tau = 0.99;
        let alpha = fraction_to_boundary(&s, &ds, tau);
        assert!((alpha - 1.0).abs() < 1e-12);

        // Tight constraint: s = [0.01], ds = [-1.0]
        let s2 = vec![0.01];
        let ds2 = vec![-1.0];
        let alpha2 = fraction_to_boundary(&s2, &ds2, tau);
        // alpha = -tau * s / ds = 0.99 * 0.01 / 1.0 = 0.0099
        assert!((alpha2 - 0.0099).abs() < 1e-12);
    }
}
