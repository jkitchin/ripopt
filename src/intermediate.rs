//! Intermediate callback infrastructure for ripopt.
//!
//! Provides a per-iteration callback mechanism matching IPOPT's
//! `SetIntermediateCallback`. The callback is stored in a thread-local
//! (same pattern as logging) and invoked after each iteration in the IPM loop.
//!
//! Also provides thread-local storage for the current iterate, so that
//! `ripopt_get_current_iterate` and `ripopt_get_current_violations` can
//! retrieve solver state during the intermediate callback (matching Ipopt's
//! `GetIpoptCurrentIterate` and `GetIpoptCurrentViolations`).

use std::cell::Cell;
use std::cell::RefCell;
use std::os::raw::c_void;

use crate::c_api::IntermediateCb;

thread_local! {
    static INTERMEDIATE_CALLBACK: Cell<Option<(IntermediateCb, *mut c_void)>> = Cell::new(None);
    static CURRENT_ITERATE: RefCell<Option<IterateSnapshot>> = RefCell::new(None);
}

/// Snapshot of solver state, stored during the intermediate callback so that
/// GetCurrentIterate/GetCurrentViolations can read it.
pub struct IterateSnapshot {
    pub x: Vec<f64>,
    pub z_l: Vec<f64>,
    pub z_u: Vec<f64>,
    pub g: Vec<f64>,
    pub lambda: Vec<f64>,
    // Violations
    pub x_l_violation: Vec<f64>,
    pub x_u_violation: Vec<f64>,
    pub compl_x_l: Vec<f64>,
    pub compl_x_u: Vec<f64>,
    pub grad_lag_x: Vec<f64>,
    pub constraint_violation: Vec<f64>,
    pub compl_g: Vec<f64>,
}

/// Install an intermediate callback for the current thread.
/// Pass `None` to clear.
pub fn set_intermediate_callback(cb: Option<(IntermediateCb, *mut c_void)>) {
    INTERMEDIATE_CALLBACK.with(|cell| cell.set(cb));
}

/// Store a snapshot of the current iterate for GetCurrentIterate/Violations access.
pub fn set_current_iterate(snapshot: Option<IterateSnapshot>) {
    CURRENT_ITERATE.with(|cell| {
        *cell.borrow_mut() = snapshot;
    });
}

/// Access the current iterate snapshot (only valid during intermediate callback).
pub fn with_current_iterate<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&IterateSnapshot) -> R,
{
    CURRENT_ITERATE.with(|cell| {
        cell.borrow().as_ref().map(f)
    })
}

/// Invoke the intermediate callback with current iteration data.
/// Signature matches Ipopt's Intermediate_CB parameters.
/// Returns `true` to continue, `false` to stop (user requested termination).
pub fn invoke_intermediate(
    alg_mod: i32,
    iter: usize,
    obj_value: f64,
    inf_pr: f64,
    inf_du: f64,
    mu: f64,
    d_norm: f64,
    regularization_size: f64,
    alpha_du: f64,
    alpha_pr: f64,
    ls_trials: usize,
) -> bool {
    INTERMEDIATE_CALLBACK.with(|cell| {
        if let Some((cb, user_data)) = cell.get() {
            let result = unsafe {
                cb(
                    alg_mod,
                    iter as i32,
                    obj_value,
                    inf_pr,
                    inf_du,
                    mu,
                    d_norm,
                    regularization_size,
                    alpha_du,
                    alpha_pr,
                    ls_trials as i32,
                    user_data,
                )
            };
            result != 0
        } else {
            true // no callback, continue
        }
    })
}
