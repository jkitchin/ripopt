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
    /// T3.5: line-search minimum step multiplier. Ipopt's
    /// `alpha_min_frac` (default `0.05`, `IpFilterLSAcceptor.cpp:113,
    /// 222, 468`). Multiplies the final `min` of the gamma_theta /
    /// gamma_phi / switching-condition terms in `compute_alpha_min`.
    alpha_min_frac: f64,
    /// Whether `set_theta_min_from_initial` has already seeded
    /// `theta_max`/`theta_min` from the initial constraint violation.
    /// Mirrors Ipopt's lazy-init sentinel (theta_max_ < 0.0) at
    /// IpFilterLSAcceptor.cpp:325-339: once seeded, the bounds are
    /// fixed for the rest of the solve.
    theta_init_set: bool,
    /// T3.10: tracks whether the most recent rejected line-search trial
    /// (since the last accepted step) was rejected by the filter
    /// dominance test. Mirrors Ipopt's `last_rejection_due_to_filter_`
    /// (`IpFilterLSAcceptor.cpp:380, 397`).
    last_rejection_due_to_filter: bool,
    /// T3.10: counts consecutive accepted steps whose preceding line
    /// search ended with a filter-based rejection.
    count_successive_filter_rejections: u32,
    /// T3.10: how many filter resets have already fired this solve.
    /// Capped at `max_filter_resets` to prevent runaway clearing.
    n_filter_resets: u32,
    /// T3.10: number of consecutive filter rejections that triggers a
    /// reset (Ipopt option `filter_reset_trigger`, default 5).
    filter_reset_trigger: u32,
    /// T3.10: maximum number of times the filter may be reset across
    /// the solve (Ipopt option `max_filter_resets`, default 5; 0
    /// disables the heuristic). Mirrors `IpFilterLSAcceptor.cpp:142,
    /// 230, 407-414`.
    max_filter_resets: u32,
    /// DEV-36: `theta_min_fact` (Ipopt default 1e-4). Used by
    /// `set_theta_min_from_initial` to seed `theta_min` once.
    theta_min_fact: f64,
    /// DEV-36: `theta_max_fact` (Ipopt default 1e4). Used by
    /// `set_theta_min_from_initial` to seed `theta_max` once.
    theta_max_fact: f64,
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
            alpha_min_frac: 0.05,
            theta_init_set: false,
            last_rejection_due_to_filter: false,
            count_successive_filter_rejections: 0,
            n_filter_resets: 0,
            filter_reset_trigger: 5,
            max_filter_resets: 5,
            theta_min_fact: 1e-4,
            theta_max_fact: 1e4,
        }
    }

    /// DEV-36: plumb the Ipopt `theta_min_fact` / `theta_max_fact`
    /// options. Called before `set_theta_min_from_initial`.
    pub fn set_theta_factors(&mut self, theta_min_fact: f64, theta_max_fact: f64) {
        self.theta_min_fact = theta_min_fact;
        self.theta_max_fact = theta_max_fact;
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
        self.theta_min = self.theta_min_fact * floor;
        self.theta_max = self.theta_max_fact * floor;
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

    /// Pure Ipopt `IsFtype` (`IpFilterLSAcceptor.cpp:273-295`):
    /// `gradBarrTDelta < 0  &&  alpha * (-gradBarrTDelta)^s_phi > delta * theta^s_theta`.
    ///
    /// DEV-30: this **does not** include the `theta <= theta_min` gate.
    /// In Ipopt that gate is applied at the call sites in
    /// `CheckAcceptabilityOfTrialPoint` (line 361-362), but the
    /// augmentation logic in `UpdateForNextIteration` (line 881-895)
    /// uses pure `IsFtype` *without* the theta_min gate. Bundling the
    /// two over-augments the filter on h-type accepts that came from
    /// theta>theta_min trials with a still-good Armijo decrease.
    pub fn is_ftype(&self, theta_current: f64, grad_phi_step: f64, alpha: f64) -> bool {
        grad_phi_step < 0.0
            && alpha * (-grad_phi_step).powf(self.s_phi)
                > self.delta * theta_current.powf(self.s_theta)
    }

    /// Combined Ipopt switching gate: `IsFtype(alpha) && theta <= theta_min`.
    /// This is the gate at `IpFilterLSAcceptor.cpp:361-362` that selects
    /// the Armijo branch in `CheckAcceptabilityOfTrialPoint`.
    ///
    /// DEV-30: factored as `is_ftype(...) && theta_current <= theta_min`.
    pub fn switching_condition(
        &self,
        theta_current: f64,
        grad_phi_step: f64,
        alpha: f64,
    ) -> bool {
        // DEV-24: `<= theta_min` (non-strict) matches Ipopt
        // IpFilterLSAcceptor.cpp:362.
        self.is_ftype(theta_current, grad_phi_step, alpha)
            && theta_current <= self.theta_min
    }

    /// Tolerance-aware `<=` mirroring Ipopt's `Compare_le`
    /// (`IpUtils.cpp:294-302`):
    ///   `lhs - rhs <= 10 * eps * |bas_val|`.
    /// Slightly more permissive than bare `<=` near the boundary; used
    /// throughout `IpFilterLSAcceptor.cpp` (Armijo, sufficient-reduction,
    /// h-type theta/phi tests).
    #[inline]
    fn compare_le(lhs: f64, rhs: f64, bas_val: f64) -> bool {
        let mach_eps = f64::EPSILON;
        lhs - rhs <= 10.0 * mach_eps * bas_val.abs()
    }

    /// Check the Armijo sufficient decrease condition. Mirrors Ipopt
    /// `IpFilterLSAcceptor.cpp:445-447` (`ArmijoHolds`):
    ///   `Compare_le(trial_barr - reference_barr, eta_phi * alpha * grad_phi, reference_barr)`.
    pub fn armijo_condition(
        &self,
        phi_current: f64,
        phi_trial: f64,
        grad_phi_step: f64,
        alpha: f64,
    ) -> bool {
        Self::compare_le(
            phi_trial - phi_current,
            self.eta_phi * alpha * grad_phi_step,
            phi_current,
        )
    }

    /// Check if a trial point provides sufficient constraint reduction
    /// compared to the current point. Mirrors the theta clause of
    /// Ipopt `IpFilterLSAcceptor.cpp:497`:
    ///   `Compare_le(trial_theta, (1 - gamma_theta) * reference_theta, reference_theta)`.
    pub fn sufficient_infeasibility_reduction(
        &self,
        theta_current: f64,
        theta_trial: f64,
    ) -> bool {
        Self::compare_le(
            theta_trial,
            (1.0 - self.gamma_theta) * theta_current,
            theta_current,
        )
    }

    /// Check if a trial point is acceptable via either the filter or the
    /// sufficient decrease conditions (Armijo or constraint reduction).
    ///
    /// Returns (acceptable, augment_required) where:
    /// - acceptable: whether the step should be accepted
    /// - augment_required: per Ipopt `UpdateForNextIteration`
    ///   (`IpFilterLSAcceptor.cpp:881-895`), the filter must be
    ///   augmented iff `!IsFtype(alpha) || !ArmijoHolds(alpha)`.
    ///   DEV-31: the augmentation gate uses pure `IsFtype` *without*
    ///   the `theta <= theta_min` clause; bundling them over-augments
    ///   when a step was accepted via h-type but happened to satisfy
    ///   IsFtype + Armijo at the accepted alpha.
    ///
    /// Follows Ipopt's structure (IpFilterLSAcceptor.cpp:311-437):
    /// 1. Determine step type (Armijo branch when `IsFtype && theta<=theta_min`,
    ///    sufficient-reduction otherwise)
    /// 2. Check filter acceptability
    ///
    /// Ipopt-alignment: the `obj_max_inc` divergence guard is **h-type-only**
    /// per `IpFilterLSAcceptor.cpp:480-493`. Ipopt invokes that check from
    /// inside `IsAcceptableToCurrentIterate`, which is itself only called
    /// from the h-type (sufficient-reduction) branch at line 373. The
    /// f-type/Armijo branch at line 361-371 never consults `obj_max_inc`.
    pub fn check_acceptability(
        &mut self,
        theta_current: f64,
        phi_current: f64,
        theta_trial: f64,
        phi_trial: f64,
        grad_phi_step: f64,
        alpha: f64,
        called_from_restoration: bool,
    ) -> (bool, bool) {
        // Reject if theta exceeds maximum or NaN
        if theta_trial > self.theta_max || theta_trial.is_nan() || phi_trial.is_nan() {
            self.last_rejection_due_to_filter = false;
            return (false, false);
        }

        // DEV-30: split IsFtype (pure) from the call-site theta_min gate
        // that selects the acceptance branch.
        let is_ft = self.is_ftype(theta_current, grad_phi_step, alpha);
        let armijo_holds = self.armijo_condition(phi_current, phi_trial, grad_phi_step, alpha);

        // Acceptance branch (IpFilterLSAcceptor.cpp:361-374):
        //   if (IsFtype && theta <= theta_min)  Armijo  (no obj_max_inc check)
        //   else                                 IsAcceptableToCurrentIterate
        //                                        (obj_max_inc FIRST, then
        //                                         sufficient theta-or-phi
        //                                         reduction)
        let type_ok = if is_ft && theta_current <= self.theta_min {
            // f-type/Armijo branch — Ipopt does NOT consult obj_max_inc here.
            armijo_holds
        } else {
            // h-type branch: mirror IsAcceptableToCurrentIterate
            // (IpFilterLSAcceptor.cpp:471-499). The obj_max_inc divergence
            // guard runs FIRST (line 480-493) and is bypassed when
            // `called_from_restoration` (post-restoration handoff); only
            // if it passes do we test the sufficient-reduction OR
            // phi-reduction (line 497-498), both via Compare_le.
            if !self.passes_obj_max_inc(phi_current, phi_trial, called_from_restoration) {
                false
            } else {
                self.sufficient_infeasibility_reduction(theta_current, theta_trial)
                    || Self::compare_le(
                        phi_trial - phi_current,
                        -self.gamma_phi * theta_current,
                        phi_current,
                    )
            }
        };

        if !type_ok {
            self.last_rejection_due_to_filter = false;
            return (false, false);
        }

        // Check filter acceptability
        if !self.is_acceptable(theta_trial, phi_trial) {
            self.last_rejection_due_to_filter = true;
            return (false, false);
        }

        // DEV-31: augmentation gate is pure `!IsFtype || !ArmijoHolds`
        // (IpFilterLSAcceptor.cpp:885-886). No theta_min clause.
        let augment_required = !is_ft || !armijo_holds;
        (true, augment_required)
    }

    /// T3.10: handle the post-acceptance bookkeeping of the
    /// filter-reset heuristic. Mirrors Ipopt's
    /// `IpFilterLSAcceptor.cpp:407-434` block run when a trial point
    /// passes both the sufficient-reduction test and the filter
    /// dominance test. Increments the consecutive-filter-rejection
    /// counter when the most-recent rejected trial in this iteration's
    /// backtracking sequence was filter-caused; clears the counter
    /// otherwise. When the counter hits `filter_reset_trigger` and the
    /// per-solve cap `max_filter_resets` has not been reached, wipes
    /// the filter entries and returns `true`. (`theta_max`/`theta_min`/
    /// `gamma_*`/`eta_phi` are NOT touched — only the entry list is
    /// cleared, matching `IpFilter::Clear`.)
    pub fn note_acceptance(&mut self) -> bool {
        if self.max_filter_resets == 0 || self.n_filter_resets >= self.max_filter_resets {
            self.last_rejection_due_to_filter = false;
            return false;
        }
        let triggered = if self.last_rejection_due_to_filter {
            self.count_successive_filter_rejections += 1;
            if self.count_successive_filter_rejections >= self.filter_reset_trigger {
                self.entries.clear();
                self.count_successive_filter_rejections = 0;
                self.n_filter_resets += 1;
                true
            } else {
                false
            }
        } else {
            self.count_successive_filter_rejections = 0;
            false
        };
        self.last_rejection_due_to_filter = false;
        triggered
    }

    /// Plumb the T3.10 reset-trigger parameters from `SolverOptions`.
    pub fn set_filter_reset_options(&mut self, trigger: u32, max_resets: u32) {
        self.filter_reset_trigger = trigger.max(1);
        self.max_filter_resets = max_resets;
    }

    /// Diagnostic accessor: how many filter resets have already fired.
    pub fn n_filter_resets(&self) -> u32 {
        self.n_filter_resets
    }

    /// T3.4: Ipopt's `obj_max_inc` divergence guard
    /// (`IpFilterLSAcceptor.cpp:478-493`). Returns `true` when the trial
    /// is acceptable on the divergence criterion (which includes the
    /// case `trial_phi <= reference_phi`); returns `false` only when
    /// `trial_phi > reference_phi` *and*
    /// `log10(trial_phi - reference_phi) > obj_max_inc + basval`, where
    /// `basval = max(1.0, log10(|reference_phi|))`.
    ///
    /// Bypassed when `called_from_restoration`: Ipopt skips the guard
    /// on the post-restoration handoff because the restoration phase is
    /// allowed to leave the barrier objective much larger than the
    /// pre-restoration reference (`IpFilterLSAcceptor.cpp:480` —
    /// `if (!called_from_restoration && trial_barr > reference_barr_)`).
    pub fn passes_obj_max_inc(
        &self,
        reference_phi: f64,
        trial_phi: f64,
        called_from_restoration: bool,
    ) -> bool {
        if called_from_restoration {
            return true;
        }
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
        // Ipopt formula (`IpFilterLSAcceptor.cpp:450-469`):
        //   alpha_min = gamma_theta
        //   if gBD < 0:
        //     alpha_min = min(alpha_min, gamma_phi*theta/(-gBD))
        //     if theta <= theta_min:
        //       alpha_min = min(alpha_min, delta*theta^s_theta/(-gBD)^s_phi)
        //   return alpha_min_frac * alpha_min
        // No further floor — Ipopt does not apply a `Max(epsilon, ...)` clamp.
        if grad_phi_step >= 0.0 {
            // No barrier descent direction (common for feasibility problems with obj=0).
            // Falls back to gamma_theta; restoration triggers after ~13 halvings at the
            // default alpha_min_frac=0.05.
            return self.alpha_min_frac * self.gamma_theta;
        }
        let neg_gphi = -grad_phi_step;
        let mut inner = self.gamma_theta.min(self.gamma_phi * theta_current / neg_gphi);
        if theta_current <= self.theta_min {
            inner = inner.min(
                self.delta * theta_current.powf(self.s_theta) / neg_gphi.powf(self.s_phi),
            );
        }
        self.alpha_min_frac * inner
    }

    /// Set the `alpha_min_frac` line-search step multiplier. Mirrors Ipopt
    /// 3.14 `alpha_min_frac` (default `0.05`, `IpFilterLSAcceptor.cpp:113`).
    pub fn set_alpha_min_frac(&mut self, value: f64) {
        self.alpha_min_frac = value;
    }

    /// Augment the filter at restoration entry (Ipopt's PrepareRestoPhaseStart,
    /// IpFilterLSAcceptor.cpp:898-901 → AugmentFilter() at :297-308). Adds an
    /// entry at (phi - gamma_phi*theta, (1 - gamma_theta)*theta) so the
    /// restored iterate cannot hand back a point "as bad or worse" than the
    /// pre-restoration one.
    ///
    /// Note: does NOT bump `theta_max` — Ipopt's `theta_max_` is initialized
    /// lazily once from the first reference theta and is never re-inflated
    /// at restoration entry.
    ///
    /// Ipopt-alignment: this mirrors `FilterLSAcceptor::PrepareRestoPhaseStart`
    /// (`IpFilterLSAcceptor.cpp:898-901`), invoked when the line search hands
    /// off to the restoration phase. ripopt invokes this from `ipm.rs` in the
    /// `!step_accepted` branch at the entry to the restoration cascade
    /// (search for `augment_for_restoration` in `src/ipm.rs`), so the
    /// pre-restoration `(theta, phi)` margin entry is in the filter before
    /// either the almost-feasible guard or the restoration solver runs.
    pub fn augment_for_restoration(&mut self, theta_current: f64, phi_current: f64) {
        // ripopt stores raw (theta, phi); is_acceptable applies the
        // (gamma_theta, gamma_phi) offsets at compare time. Pre-baking the
        // offsets here would double-apply them and hand the restoration
        // kernel a strictly weaker entry than Ipopt's reference. Pass raw.
        self.add(theta_current, phi_current);
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

        let (accept, augment_required) = filter.check_acceptability(
            theta_current, phi_current, theta_trial, phi_trial, grad_phi_step, alpha, false,
        );
        assert!(accept, "Should be accepted via Armijo");
        // DEV-31: f-type accept (IsFtype && ArmijoHolds) → no filter augmentation.
        assert!(!augment_required, "f-type accept must not augment filter");
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

        let (accept, augment_required) = filter.check_acceptability(
            theta_current, phi_current, theta_trial, phi_trial, grad_phi_step, alpha, false,
        );
        assert!(accept, "Should be accepted via filter mode");
        // DEV-31: h-type accept where IsFtype is false → must augment filter.
        assert!(augment_required, "h-type accept must augment filter");
    }

    #[test]
    fn test_check_acceptability_rejected() {
        let mut filter = Filter::new(100.0);
        filter.set_theta_min_from_initial(10.0);
        // Add a filter entry that dominates the trial
        filter.add(0.5, 5.0);
        // Trial point is worse than filter
        let (accept, _) = filter.check_acceptability(
            1.0, 10.0, 0.6, 6.0, -0.01, 1.0, false,
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
        let mut filter = Filter::new(100.0);
        assert!(!filter.is_acceptable(f64::NAN, 1.0));
        assert!(!filter.is_acceptable(1.0, f64::NAN));
        let (accept, _) = filter.check_acceptability(
            1.0, 10.0, f64::NAN, 5.0, -1.0, 1.0, false,
        );
        assert!(!accept);
        let (accept2, _) = filter.check_acceptability(
            1.0, 10.0, 0.5, f64::NAN, -1.0, 1.0, false,
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
        // alpha_min = 0.05 * min(gamma_theta=1e-5, gamma_phi*theta/|grad|=1e-12, term3≈5.01e-5)
        //           = 0.05 * 1e-12 = 5e-14
        // T3.5: no `.max(1e-15)` floor — Ipopt does not clamp.
        let alpha_min = filter.compute_alpha_min(theta_current, grad);
        let expected = 0.05 * 1e-12;
        assert!((alpha_min - expected).abs() < 1e-25,
            "alpha_min should be 0.05*term2 = 5e-14, got {}", alpha_min);
        // Now use a much steeper descent so term2 grows and term1 becomes the binding term.
        let alpha_min_steep = filter.compute_alpha_min(theta_current, -1e-10);
        // term1=1e-5, term2=1e-8*1e-4/1e-10=1e-2, term3 huge → inner = term1 = 1e-5
        let expected_steep = 0.05 * 1e-5;
        assert!((alpha_min_steep - expected_steep).abs() < 1e-15,
            "steep-grad alpha_min should be 0.05*gamma_theta = 5e-7, got {}", alpha_min_steep);
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
        let theta = 50.0;
        let phi = 10.0;
        filter.augment_for_restoration(theta, phi);
        // Should have added an entry at ((1-gamma_theta)*theta, phi - gamma_phi*theta)
        assert_eq!(filter.len(), 1);
        // Ipopt's PrepareRestoPhaseStart only adds the margin entry; theta_max
        // is initialized lazily once and is not re-inflated at restoration entry.
        assert_eq!(filter.theta_max(), initial_theta_max,
            "augment_for_restoration must not mutate theta_max");
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
        let (accept, augment_required) = filter.check_acceptability(
            theta_current, phi_current, theta_trial, phi_trial, grad, alpha, false,
        );
        assert!(accept, "h-type phi-only reduction must be accepted");
        // DEV-31: h-type accept (IsFtype false) → augmentation required.
        assert!(augment_required);
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
            theta_current, phi_current, 5.0, 100.0, -0.01, 1.0, false,
        );
        assert!(!accept, "h-type with no reduction in either must reject");
    }

    #[test]
    fn test_obj_max_inc_passes_when_trial_le_reference() {
        // T3.4: when phi_trial <= phi_reference, the divergence guard
        // is a no-op (returns true).
        let filter = Filter::new(100.0);
        assert!(filter.passes_obj_max_inc(10.0, 5.0, false), "decrease must pass");
        assert!(filter.passes_obj_max_inc(10.0, 10.0, false), "equality must pass");
        assert!(filter.passes_obj_max_inc(0.0, -1e6, false), "negative trial must pass");
    }

    #[test]
    fn test_obj_max_inc_rejects_huge_increase_small_reference() {
        // T3.4: with reference_phi = 1.0 (so |ref| <= 10 → basval = 1),
        // and obj_max_inc = 5 (default), the inequality is
        //   log10(trial - 1) > 5 + 1 = 6  ⇔  trial - 1 > 1e6
        // So trial = 2 is fine (log10(1) = 0 <= 6), but trial = 1e7 + 1 fails.
        let filter = Filter::new(100.0);
        assert!(filter.passes_obj_max_inc(1.0, 1e6 + 0.5, false), "below 1e6 increase must pass");
        assert!(!filter.passes_obj_max_inc(1.0, 1e7 + 2.0, false), "above 10^6 increase must fail");
    }

    #[test]
    fn test_obj_max_inc_basval_scales_with_large_reference() {
        // T3.4: when |reference_phi| > 10, basval = log10(|ref|), so the
        // permitted increase scales with |ref|. With ref=1e3, basval=3,
        // permitted log10 increase = 5 + 3 = 8, ie up to 1e8 absolute.
        let filter = Filter::new(100.0);
        assert!(filter.passes_obj_max_inc(1e3, 1e3 + 1e7, false), "1e7 increase must pass at ref=1e3");
        assert!(!filter.passes_obj_max_inc(1e3, 1e3 + 1e9, false), "1e9 increase must fail at ref=1e3");
    }

    #[test]
    fn test_obj_max_inc_set_obj_max_inc_setter() {
        // T3.4: tightening the cap to 0.0 makes any strictly positive
        // increase fail (basval >= 1 ⇒ log10(Δ) > 0+1=1 means Δ>10).
        let mut filter = Filter::new(100.0);
        filter.set_obj_max_inc(0.0);
        assert!(filter.passes_obj_max_inc(1.0, 5.0, false), "Δ=4 < 10, still passes at obj_max_inc=0");
        assert!(!filter.passes_obj_max_inc(1.0, 100.0, false), "Δ=99 > 10 must fail at obj_max_inc=0");
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
            theta_current, phi_current, theta_trial, phi_trial, -0.01, 1.0, false,
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
            theta_current, phi_current, theta_trial, phi_trial, grad, alpha, false,
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
    fn test_alpha_min_frac_is_configurable() {
        // T3.5: alpha_min_frac is plumbed from SolverOptions, not hard-coded.
        let mut filter = Filter::new(100.0);
        // Default
        let baseline = filter.compute_alpha_min(1.0, -2.0);
        // term1=1e-5, term2=1e-8*1.0/2=5e-9 → inner=5e-9 → 0.05 * 5e-9 = 2.5e-10
        assert!((baseline - 2.5e-10).abs() < 1e-20);
        // Half the multiplier → half the result
        filter.set_alpha_min_frac(0.025);
        let halved = filter.compute_alpha_min(1.0, -2.0);
        assert!((halved - 1.25e-10).abs() < 1e-20,
            "alpha_min should scale linearly with alpha_min_frac, got {}", halved);
    }

    #[test]
    fn test_alpha_min_no_floor_clamp() {
        // T3.5: Ipopt has no `Max(epsilon, ...)` clamp on the returned
        // alpha_min. Confirm tiny but positive values pass through.
        let mut filter = Filter::new(100.0);
        // theta=1e-12, grad=-1.0 → term2 = 1e-8 * 1e-12 / 1 = 1e-20
        // → inner = min(1e-5, 1e-20) = 1e-20 (theta below theta_min default 1e-4*1.0=1e-4
        //   but term3 = 1*(1e-12)^1.1/1^2.3 = 7.94e-14 > 1e-20)
        // Result: 0.05 * 1e-20 = 5e-22, well below the old 1e-15 clamp.
        filter.set_theta_min_from_initial(1.0);
        let alpha_min = filter.compute_alpha_min(1e-12, -1.0);
        assert!(alpha_min < 1e-15,
            "expected sub-1e-15 alpha_min, old code would have clamped to 1e-15, got {}",
            alpha_min);
        assert!(alpha_min > 0.0, "alpha_min must remain positive, got {}", alpha_min);
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

    /// T3.10 helper: simulate one full iteration where a single trial step
    /// was rejected by the filter dominance test (and not by Armijo /
    /// obj_max_inc / theta_max). The post-acceptance hook in `ipm.rs`
    /// runs on the *next* iteration's accepted step, so we mark the bit
    /// directly here.
    fn simulate_filter_rejected_iter(filter: &mut Filter) -> bool {
        // Force a check_acceptability call that fails the filter test.
        // Add a dominating entry, then offer a worse trial that still
        // satisfies sufficient-reduction (h-type theta drop) so we hit
        // the filter branch (not the type_ok branch). Bound checks pass.
        filter.add(0.5, 5.0);
        let theta_current = 1.0;
        let phi_current = 10.0;
        // Trial: theta_trial < theta_current * (1-gamma_theta) so h-type
        // sufficient_infeasibility_reduction passes. But (theta_trial,
        // phi_trial) is dominated by the (0.5, 5.0) filter entry.
        let theta_trial = 0.6;
        let phi_trial = 6.0;
        let (accept, _) = filter.check_acceptability(
            theta_current, phi_current, theta_trial, phi_trial, -0.01, 1.0, false,
        );
        assert!(!accept, "trial should be filter-rejected for the test");
        // Now invoke the post-acceptance hook; it consumes the
        // last_rejection_due_to_filter flag.
        filter.note_acceptance()
    }

    #[test]
    fn test_filter_reset_fires_after_trigger_consecutive_rejections() {
        // T3.10: with default trigger=5, the 5th consecutive accepted
        // step preceded by a filter-rejected trial fires the reset.
        let mut filter = Filter::new(100.0);
        for i in 0..4 {
            let fired = simulate_filter_rejected_iter(&mut filter);
            assert!(!fired, "reset must not fire on iter {}", i);
            assert_eq!(filter.n_filter_resets(), 0);
        }
        let fired = simulate_filter_rejected_iter(&mut filter);
        assert!(fired, "5th consecutive filter-rejected accept must trigger reset");
        assert_eq!(filter.n_filter_resets(), 1);
        assert!(filter.is_empty(), "reset must clear filter entries");
    }

    #[test]
    fn test_filter_reset_counter_resets_on_clean_acceptance() {
        // T3.10: an accepted step *not* preceded by a filter rejection
        // resets the consecutive counter to zero.
        let mut filter = Filter::new(100.0);
        for _ in 0..4 {
            let fired = simulate_filter_rejected_iter(&mut filter);
            assert!(!fired);
        }
        // Clean acceptance (no prior filter rejection): counter should reset.
        let fired_clean = filter.note_acceptance();
        assert!(!fired_clean);
        // Now another filter-rejected iteration: counter starts back at 1,
        // so reset must NOT fire.
        let fired = simulate_filter_rejected_iter(&mut filter);
        assert!(!fired, "counter should have reset after clean acceptance");
        assert_eq!(filter.n_filter_resets(), 0);
    }

    #[test]
    fn test_filter_reset_capped_by_max_filter_resets() {
        // T3.10: after `max_filter_resets` resets, further filter
        // rejections must not trigger another reset.
        let mut filter = Filter::new(100.0);
        // Tight cap to keep the test fast: trigger=2, max_resets=2.
        filter.set_filter_reset_options(2, 2);
        // Reset #1.
        assert!(!simulate_filter_rejected_iter(&mut filter));
        assert!(simulate_filter_rejected_iter(&mut filter));
        assert_eq!(filter.n_filter_resets(), 1);
        // Reset #2.
        assert!(!simulate_filter_rejected_iter(&mut filter));
        assert!(simulate_filter_rejected_iter(&mut filter));
        assert_eq!(filter.n_filter_resets(), 2);
        // Cap reached: further filter-rejected iterations must not reset.
        for _ in 0..10 {
            let fired = simulate_filter_rejected_iter(&mut filter);
            assert!(!fired, "no reset must fire past max_filter_resets");
        }
        assert_eq!(filter.n_filter_resets(), 2);
    }

    #[test]
    fn test_filter_reset_disabled_when_max_resets_zero() {
        // T3.10: `max_filter_resets = 0` disables the heuristic entirely.
        let mut filter = Filter::new(100.0);
        filter.set_filter_reset_options(1, 0);
        for _ in 0..20 {
            let fired = simulate_filter_rejected_iter(&mut filter);
            assert!(!fired, "max_filter_resets=0 must disable resets");
        }
        assert_eq!(filter.n_filter_resets(), 0);
        assert!(!filter.is_empty(), "filter entries must persist when disabled");
    }

    #[test]
    fn test_filter_reset_trigger_set_floored_at_one() {
        // T3.10: trigger=0 would cause a runaway reset on every accepted
        // step; the setter floors it to 1. With trigger=1 the very first
        // filter-rejected accept fires a reset.
        let mut filter = Filter::new(100.0);
        filter.set_filter_reset_options(0, 5);
        let fired = simulate_filter_rejected_iter(&mut filter);
        assert!(fired, "trigger=0 floored to 1 should fire on first rejection");
        assert_eq!(filter.n_filter_resets(), 1);
    }
}
