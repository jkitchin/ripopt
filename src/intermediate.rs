//! Intermediate callback infrastructure for ripopt.
//!
//! Provides a per-iteration callback mechanism matching IPOPT's
//! `SetIntermediateCallback`. The callback is stored in a thread-local
//! (same pattern as logging) and invoked after each iteration in the IPM loop.

use std::cell::Cell;
use std::os::raw::c_void;

use crate::c_api::IntermediateCb;

thread_local! {
    static INTERMEDIATE_CALLBACK: Cell<Option<(IntermediateCb, *mut c_void)>> = Cell::new(None);
}

/// Install an intermediate callback for the current thread.
/// Pass `None` to clear.
pub fn set_intermediate_callback(cb: Option<(IntermediateCb, *mut c_void)>) {
    INTERMEDIATE_CALLBACK.with(|cell| cell.set(cb));
}

/// Invoke the intermediate callback with current iteration data.
/// Returns `true` to continue, `false` to stop (user requested termination).
pub fn invoke_intermediate(
    iter: usize,
    obj_value: f64,
    inf_pr: f64,
    inf_du: f64,
    mu: f64,
    alpha_pr: f64,
    alpha_du: f64,
    ls_trials: usize,
) -> bool {
    INTERMEDIATE_CALLBACK.with(|cell| {
        if let Some((cb, user_data)) = cell.get() {
            let result = unsafe {
                cb(
                    iter as i32,
                    obj_value,
                    inf_pr,
                    inf_du,
                    mu,
                    alpha_pr,
                    alpha_du,
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
