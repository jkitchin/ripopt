/**
 * ripopt.h — C API for the ripopt nonlinear optimizer.
 *
 * Mirrors the Ipopt C interface so that existing Ipopt C code can be
 * adapted with minimal changes.
 *
 * Usage:
 *   1. Create a problem handle with ripopt_create().
 *   2. Optionally tune options with ripopt_add_num/int/str_option().
 *   3. Call ripopt_solve() with an initial point.
 *   4. Free the handle with ripopt_free().
 *
 * Compile & link example (macOS):
 *   cargo build --release
 *   cc examples/c_api_test.c -I. -Ltarget/release -lripopt \
 *       -Wl,-rpath,target/release -o c_api_test
 *   ./c_api_test
 */

#ifndef RIPOPT_H
#define RIPOPT_H

/* Version information — keep in sync with Cargo.toml */
#define RIPOPT_VERSION_MAJOR 0
#define RIPOPT_VERSION_MINOR 6
#define RIPOPT_VERSION_PATCH 1
#define RIPOPT_VERSION "0.6.1"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a ripopt problem. */
typedef void* RipoptProblem;

/* -------------------------------------------------------------------------
 * Callback signatures (identical to Ipopt C API)
 * -------------------------------------------------------------------------
 * All callbacks return 1 (true) on success, 0 on error.
 * new_x / new_lambda: 1 if x/lambda changed since the last call, 0 otherwise.
 * Ripopt currently passes 1 for every call (conservative / correct).
 *
 * Jacobian / Hessian callbacks are called in two modes:
 *   - values == NULL  → fill iRow/jCol with the sparsity pattern (0-based)
 *   - values != NULL  → fill values in the same element order as the pattern
 * ------------------------------------------------------------------------- */

typedef int (*Eval_F_CB)(
    int n, const double *x, int new_x,
    double *obj_value,
    void *user_data);

typedef int (*Eval_Grad_F_CB)(
    int n, const double *x, int new_x,
    double *grad_f,
    void *user_data);

typedef int (*Eval_G_CB)(
    int n, const double *x, int new_x,
    int m, double *g,
    void *user_data);

typedef int (*Eval_Jac_G_CB)(
    int n, const double *x, int new_x,
    int m, int nele_jac,
    int *iRow, int *jCol,
    double *values,
    void *user_data);

typedef int (*Eval_H_CB)(
    int n, const double *x, int new_x,
    double obj_factor,
    int m, const double *lambda, int new_lambda,
    int nele_hess,
    int *iRow, int *jCol,
    double *values,
    void *user_data);

/* -------------------------------------------------------------------------
 * Return status codes
 * ------------------------------------------------------------------------- */
/* Values match Ipopt's ApplicationReturnStatus for drop-in compatibility. */
typedef enum {
    RIPOPT_SOLVE_SUCCEEDED              =   0,
    /* RIPOPT_ACCEPTABLE_LEVEL          =   1, */  /* not currently returned */
    RIPOPT_INFEASIBLE_PROBLEM           =   2,
    RIPOPT_SEARCH_DIRECTION_TOO_SMALL   =   3,
    RIPOPT_DIVERGING_ITERATES           =   4,
    RIPOPT_USER_REQUESTED_STOP          =   5,
    /* RIPOPT_FEASIBLE_POINT_FOUND      =   6, */  /* not currently returned */
    RIPOPT_MAXITER_EXCEEDED             =  -1,
    RIPOPT_RESTORATION_FAILED           =  -2,
    RIPOPT_ERROR_IN_STEP_COMPUTATION    =  -3,
    /* RIPOPT_MAX_CPUTIME_EXCEEDED      =  -4, */  /* not currently returned */
    RIPOPT_MAX_WALLTIME_EXCEEDED        =  -5,
    RIPOPT_NOT_ENOUGH_DEGREES_OF_FREEDOM = -10,
    RIPOPT_INVALID_PROBLEM_DEFINITION   = -11,
    RIPOPT_INVALID_NUMBER_DETECTED      = -13,
    RIPOPT_INTERNAL_ERROR               = -199
} RipoptReturnStatus;

/* -------------------------------------------------------------------------
 * Lifecycle
 * -------------------------------------------------------------------------
 * ripopt_create — allocate and return a new problem handle.
 *
 *   n           number of primal variables
 *   x_l/x_u    variable lower/upper bounds (length n; use ±1e30 for ±∞)
 *   m           number of constraints
 *   g_l/g_u    constraint lower/upper bounds (length m)
 *   nele_jac   number of nonzeros in the Jacobian
 *   nele_hess  number of nonzeros in the lower-triangular Hessian
 *   index_style 0 = C (0-based indices), 1 = Fortran (1-based indices)
 *   eval_*     callback function pointers (must remain valid until ripopt_free)
 *
 * Returns NULL on allocation failure.
 * ------------------------------------------------------------------------- */
RipoptProblem ripopt_create(
    int n, const double *x_l, const double *x_u,
    int m, const double *g_l, const double *g_u,
    int nele_jac, int nele_hess,
    int index_style,
    Eval_F_CB      eval_f,
    Eval_Grad_F_CB eval_grad_f,
    Eval_G_CB      eval_g,
    Eval_Jac_G_CB  eval_jac_g,
    Eval_H_CB      eval_h);

/** Free a problem handle obtained from ripopt_create(). */
void ripopt_free(RipoptProblem problem);

/* -------------------------------------------------------------------------
 * Log callback
 *
 * When installed, all solver output (iteration table, warnings, diagnostics)
 * is forwarded to the callback instead of being written to stderr.
 * The callback receives a NUL-terminated message string and the user_data
 * pointer provided at registration.
 *
 * The callback is thread-local and is cleared automatically after each
 * ripopt_solve() call.  Pass callback = NULL to revert to stderr output.
 * ------------------------------------------------------------------------- */

typedef void (*RipoptLogCB)(const char *msg, void *user_data);

/** Register a log callback for solver output.
 *
 * Must be called before ripopt_solve().  The callback and user_data pointers
 * must remain valid for the duration of the solve.
 */
void ripopt_set_log_callback(RipoptProblem problem,
                              RipoptLogCB callback,
                              void *user_data);

/* -------------------------------------------------------------------------
 * File logging
 * ------------------------------------------------------------------------- */

/** Open a log file for solver output.
 *
 * All solver output is written to the specified file.  Overrides any
 * previously set log callback.  Returns 1 on success, 0 if the file
 * cannot be opened.
 */
int ripopt_open_output_file(RipoptProblem problem,
                            const char *filename,
                            int print_level);

/* -------------------------------------------------------------------------
 * Intermediate callback
 *
 * Called once per IPM iteration with current solver state.
 * Return 1 to continue, 0 to request early termination
 * (solver returns RIPOPT_USER_REQUESTED_STOP).
 * ------------------------------------------------------------------------- */

/* Signature matches Ipopt's Intermediate_CB. */
typedef int (*RipoptIntermediateCB)(
    int alg_mod,             /* 0 = regular, 1 = restoration */
    int iter,
    double obj_value,
    double inf_pr,
    double inf_du,
    double mu,
    double d_norm,           /* infinity-norm of primal step */
    double regularization_size, /* Hessian regularization delta */
    double alpha_du,
    double alpha_pr,
    int ls_trials,
    void *user_data);

/** Register an intermediate callback.
 *
 * Must be called before ripopt_solve().  The callback and user_data pointers
 * must remain valid for the duration of the solve.
 */
void ripopt_set_intermediate_callback(RipoptProblem problem,
                                      RipoptIntermediateCB callback,
                                      void *user_data);

/* -------------------------------------------------------------------------
 * Problem scaling
 * ------------------------------------------------------------------------- */

/** Set user-provided problem scaling (matches Ipopt's SetIpoptProblemScaling).
 *
 * obj_scaling scales the objective.  x_scaling (length n) scales each
 * variable; pass NULL for no variable scaling.  g_scaling (length m) scales
 * each constraint; pass NULL for no constraint scaling.
 */
void ripopt_set_scaling(RipoptProblem problem,
                        double obj_scaling,
                        const double *x_scaling,
                        const double *g_scaling);

/* -------------------------------------------------------------------------
 * Current iterate / violations (valid ONLY during intermediate callback)
 *
 * These match Ipopt's GetIpoptCurrentIterate and GetIpoptCurrentViolations.
 * Returns 1 on success, 0 if called outside of an intermediate callback.
 * Pass NULL for any output array you don't need.
 * ------------------------------------------------------------------------- */

/** Retrieve the current iterate (primal and dual variables). */
int ripopt_get_current_iterate(RipoptProblem problem,
                               int n,
                               double *x,       /* length n, or NULL */
                               double *z_L,     /* length n, or NULL */
                               double *z_U,     /* length n, or NULL */
                               int m,
                               double *g,       /* length m, or NULL */
                               double *lambda);  /* length m, or NULL */

/** Retrieve current constraint and optimality violations. */
int ripopt_get_current_violations(RipoptProblem problem,
                                  int n,
                                  double *x_L_violation,    /* length n, or NULL */
                                  double *x_U_violation,    /* length n, or NULL */
                                  double *compl_x_L,        /* length n, or NULL */
                                  double *compl_x_U,        /* length n, or NULL */
                                  double *grad_lag_x,       /* length n, or NULL */
                                  int m,
                                  double *constraint_violation, /* length m, or NULL */
                                  double *compl_g);             /* length m, or NULL */

/* -------------------------------------------------------------------------
 * Post-solve statistics (valid after ripopt_solve() returns)
 * ------------------------------------------------------------------------- */

/** Number of IPM iterations in the most recent solve. */
int    ripopt_get_iter_count(RipoptProblem problem);

/** Wall-clock solve time in seconds from the most recent solve. */
double ripopt_get_solve_time(RipoptProblem problem);

/** Final primal infeasibility from the most recent solve. */
double ripopt_get_primal_inf(RipoptProblem problem);

/** Final dual infeasibility from the most recent solve. */
double ripopt_get_dual_inf(RipoptProblem problem);

/** Final complementarity error from the most recent solve. */
double ripopt_get_compl_inf(RipoptProblem problem);

/* -------------------------------------------------------------------------
 * Options (key/value, mirrors Ipopt option names)
 * All functions return 1 on success, 0 if the keyword is unknown.
 * ------------------------------------------------------------------------- */

/** Set a numeric (double) option, e.g. "tol" = 1e-8. */
int ripopt_add_num_option(RipoptProblem problem,
                          const char *keyword, double val);

/** Set an integer option, e.g. "max_iter" = 500. */
int ripopt_add_int_option(RipoptProblem problem,
                          const char *keyword, int val);

/** Set a string option, e.g. "mu_strategy" = "adaptive". */
int ripopt_add_str_option(RipoptProblem problem,
                          const char *keyword, const char *val);

/* -------------------------------------------------------------------------
 * Solve
 *
 *   problem   handle from ripopt_create()
 *   x         [in/out] initial point (length n) → primal solution
 *   g         [out]    constraint values g(x*) at solution, or NULL
 *   obj_val   [out]    objective f(x*) at solution, or NULL
 *   mult_g    [out]    constraint multipliers λ (length m), or NULL
 *   mult_x_L  [out]    lower bound multipliers z_L (length n), or NULL
 *   mult_x_U  [out]    upper bound multipliers z_U (length n), or NULL
 *   user_data arbitrary pointer forwarded to every callback
 *
 * Returns a RipoptReturnStatus code (cast to int).
 * ------------------------------------------------------------------------- */
int ripopt_solve(
    RipoptProblem problem,
    double *x,
    double *g,
    double *obj_val,
    double *mult_g,
    double *mult_x_L,
    double *mult_x_U,
    void   *user_data);

/* -------------------------------------------------------------------------
 * Version
 * ------------------------------------------------------------------------- */

/** Get the ripopt version as major.minor.patch integers. */
void ripopt_get_version(int *major, int *minor, int *patch);

#ifdef __cplusplus
}
#endif

#endif /* RIPOPT_H */
