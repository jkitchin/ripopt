/**
 * c_example_with_options.c — Demonstrates option tuning and multiplier
 * extraction via the ripopt C API.
 *
 * Solves the same HS071 problem twice:
 *   1. Default options
 *   2. With custom tolerances and adaptive mu strategy
 *
 * Shows how to:
 *   - Set numeric, integer, and string options
 *   - Extract all output quantities (x, obj, g, multipliers)
 *   - Interpret return status codes
 *   - Verify constraint satisfaction
 *
 * Build (from repo root):
 *   cargo build --release
 *   cc examples/c_example_with_options.c -I. -Ltarget/release -lripopt \
 *       -Wl,-rpath,$(pwd)/target/release -o c_example_with_options -lm
 *   ./c_example_with_options
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "ripopt.h"

/* --- HS071 callbacks (same as c_api_test.c) --- */

static int eval_f(int n, const double *x, int new_x,
                  double *obj, void *user_data)
{
    (void)n; (void)new_x; (void)user_data;
    *obj = x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2];
    return 1;
}

static int eval_grad_f(int n, const double *x, int new_x,
                       double *grad, void *user_data)
{
    (void)n; (void)new_x; (void)user_data;
    grad[0] = x[3]*(x[0]+x[1]+x[2]) + x[0]*x[3];
    grad[1] = x[0]*x[3];
    grad[2] = x[0]*x[3] + 1.0;
    grad[3] = x[0]*(x[0]+x[1]+x[2]);
    return 1;
}

static int eval_g(int n, const double *x, int new_x,
                  int m, double *g, void *user_data)
{
    (void)n; (void)m; (void)new_x; (void)user_data;
    g[0] = x[0]*x[1]*x[2]*x[3];
    g[1] = x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3];
    return 1;
}

static int eval_jac_g(int n, const double *x, int new_x,
                      int m, int nele_jac,
                      int *iRow, int *jCol, double *values,
                      void *user_data)
{
    (void)n; (void)m; (void)nele_jac; (void)new_x; (void)user_data;
    if (values == NULL) {
        iRow[0]=0; jCol[0]=0;  iRow[1]=0; jCol[1]=1;
        iRow[2]=0; jCol[2]=2;  iRow[3]=0; jCol[3]=3;
        iRow[4]=1; jCol[4]=0;  iRow[5]=1; jCol[5]=1;
        iRow[6]=1; jCol[6]=2;  iRow[7]=1; jCol[7]=3;
    } else {
        values[0]=x[1]*x[2]*x[3]; values[1]=x[0]*x[2]*x[3];
        values[2]=x[0]*x[1]*x[3]; values[3]=x[0]*x[1]*x[2];
        values[4]=2.0*x[0]; values[5]=2.0*x[1];
        values[6]=2.0*x[2]; values[7]=2.0*x[3];
    }
    return 1;
}

static int eval_h(int n, const double *x, int new_x,
                  double obj_factor,
                  int m, const double *lambda, int new_lambda,
                  int nele_hess,
                  int *iRow, int *jCol, double *values,
                  void *user_data)
{
    (void)n; (void)m; (void)nele_hess; (void)new_x; (void)new_lambda;
    (void)user_data;
    if (values == NULL) {
        iRow[0]=0; jCol[0]=0;
        iRow[1]=1; jCol[1]=0;  iRow[2]=1; jCol[2]=1;
        iRow[3]=2; jCol[3]=0;  iRow[4]=2; jCol[4]=1;  iRow[5]=2; jCol[5]=2;
        iRow[6]=3; jCol[6]=0;  iRow[7]=3; jCol[7]=1;  iRow[8]=3; jCol[8]=2;  iRow[9]=3; jCol[9]=3;
    } else {
        values[0] = obj_factor*2.0*x[3];
        values[1] = obj_factor*x[3];       values[2] = 0.0;
        values[3] = obj_factor*x[3];       values[4] = 0.0;        values[5] = 0.0;
        values[6] = obj_factor*(2.0*x[0]+x[1]+x[2]);
        values[7] = obj_factor*x[0];       values[8] = obj_factor*x[0]; values[9] = 0.0;
        /* Constraint 0 */
        values[1] += lambda[0]*x[2]*x[3];
        values[3] += lambda[0]*x[1]*x[3];
        values[4] += lambda[0]*x[0]*x[3];
        values[6] += lambda[0]*x[1]*x[2];
        values[7] += lambda[0]*x[0]*x[2];
        values[8] += lambda[0]*x[0]*x[1];
        /* Constraint 1 */
        values[0] += lambda[1]*2.0;
        values[2] += lambda[1]*2.0;
        values[5] += lambda[1]*2.0;
        values[9] += lambda[1]*2.0;
    }
    return 1;
}

/* --- Helper to print status name --- */

static const char* status_name(int status)
{
    switch (status) {
        case  0: return "SOLVE_SUCCEEDED";
        case  1: return "ACCEPTABLE_LEVEL";
        case  2: return "INFEASIBLE_PROBLEM";
        case  5: return "MAXITER_EXCEEDED";
        case  6: return "RESTORATION_FAILED";
        case  7: return "ERROR_IN_STEP_COMPUTATION";
        case 10: return "NOT_ENOUGH_DEGREES_OF_FREEDOM";
        case 11: return "INVALID_PROBLEM_DEFINITION";
        case -1: return "INTERNAL_ERROR";
        default: return "UNKNOWN";
    }
}

/* --- Solve once with given options, printing full results --- */

static int solve_and_report(const char *label,
                            double tol, int max_iter, int print_level,
                            const char *mu_strategy)
{
    int n = 4, m = 2;
    double x_l[4] = {1.0, 1.0, 1.0, 1.0};
    double x_u[4] = {5.0, 5.0, 5.0, 5.0};
    double g_l[2] = {25.0, 40.0};
    double g_u[2] = {HUGE_VAL, 40.0};

    RipoptProblem nlp = ripopt_create(
        n, x_l, x_u, m, g_l, g_u, 8, 10, 0, /* C-style 0-based indexing */
        eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h);

    if (!nlp) return 1;

    /* Set options */
    ripopt_add_num_option(nlp, "tol", tol);
    ripopt_add_int_option(nlp, "max_iter", max_iter);
    ripopt_add_int_option(nlp, "print_level", print_level);
    ripopt_add_str_option(nlp, "mu_strategy", mu_strategy);

    /* Allocate all outputs */
    double x[4]       = {1.0, 5.0, 5.0, 1.0};
    double obj_val     = 0.0;
    double g[2]        = {0.0, 0.0};
    double mult_g[2]   = {0.0, 0.0};
    double mult_xl[4]  = {0.0, 0.0, 0.0, 0.0};
    double mult_xu[4]  = {0.0, 0.0, 0.0, 0.0};

    int status = ripopt_solve(nlp, x, g, &obj_val,
                              mult_g, mult_xl, mult_xu, NULL);

    printf("\n=== %s ===\n", label);
    printf("Options: tol=%g, max_iter=%d, mu_strategy=%s\n",
           tol, max_iter, mu_strategy);
    printf("Status:  %d (%s)\n", status, status_name(status));
    printf("Obj:     %.10f\n", obj_val);
    printf("x:       [%.6f, %.6f, %.6f, %.6f]\n", x[0], x[1], x[2], x[3]);
    printf("g(x):    [%.6f, %.6f]\n", g[0], g[1]);
    printf("  g[0] = x1*x2*x3*x4 = %.6f  (>= 25)\n", g[0]);
    printf("  g[1] = sum(xi^2)   = %.6f  (== 40)\n", g[1]);
    printf("Constraint multipliers (lambda):\n");
    printf("  mult_g = [%.6e, %.6e]\n", mult_g[0], mult_g[1]);
    printf("Bound multipliers:\n");
    printf("  z_L = [%.6e, %.6e, %.6e, %.6e]\n",
           mult_xl[0], mult_xl[1], mult_xl[2], mult_xl[3]);
    printf("  z_U = [%.6e, %.6e, %.6e, %.6e]\n",
           mult_xu[0], mult_xu[1], mult_xu[2], mult_xu[3]);

    /* Check active bounds */
    printf("Active bounds:\n");
    for (int i = 0; i < n; i++) {
        if (fabs(x[i] - x_l[i]) < 1e-4)
            printf("  x[%d] = %.4f at LOWER bound (z_L=%.4e)\n", i, x[i], mult_xl[i]);
        else if (fabs(x[i] - x_u[i]) < 1e-4)
            printf("  x[%d] = %.4f at UPPER bound (z_U=%.4e)\n", i, x[i], mult_xu[i]);
        else
            printf("  x[%d] = %.4f (free, z_L=%.1e, z_U=%.1e)\n",
                   i, x[i], mult_xl[i], mult_xu[i]);
    }

    int pass = (status == 0 || status == 1) &&
               fabs(obj_val - 17.0140173) < 1e-3;

    ripopt_free(nlp);
    return pass ? 0 : 1;
}

int main(void)
{
    printf("ripopt C API version %s\n", RIPOPT_VERSION);

    int fail = 0;

    /* Solve 1: default-ish options */
    fail |= solve_and_report(
        "HS071 with default options",
        1e-8, 3000, 0, "adaptive");

    /* Solve 2: tighter tolerance */
    fail |= solve_and_report(
        "HS071 with tight tolerance",
        1e-12, 3000, 0, "adaptive");

    /* Solve 3: monotone mu */
    fail |= solve_and_report(
        "HS071 with monotone mu",
        1e-8, 3000, 0, "monotone");

    printf("\n=== Overall: %s ===\n", fail ? "SOME FAILED" : "ALL PASSED");
    return fail;
}
