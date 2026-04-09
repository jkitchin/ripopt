/**
 * c_hs035.c — HS035 bound-constrained QP with an inequality constraint.
 *
 * Problem (Hock & Schittkowski problem 35):
 *   min  9 - 8x1 - 6x2 - 4x3 + 2x1^2 + 2x2^2 + x3^2 + 2x1x2 + 2x1x3
 *   s.t. x1 + x2 + 2x3 <= 3
 *        x1, x2, x3 >= 0
 *
 * Expected solution:
 *   x* = [4/3, 7/9, 4/9] = [1.3333, 0.7778, 0.4444]
 *   f* = 1/9 ≈ 0.1111
 *
 * This example demonstrates:
 *   - Variable bounds (lower bounds only)
 *   - An inequality constraint (upper bound only)
 *   - Extracting constraint multipliers
 *   - Constant Hessian (H = [[4,2,2],[2,4,0],[2,0,2]])
 *
 * Build (from repo root):
 *   cargo build --release
 *   cc examples/c_hs035.c -I. -Ltarget/release -lripopt \
 *       -Wl,-rpath,$(pwd)/target/release -o c_hs035 -lm
 *   ./c_hs035
 */

#include <stdio.h>
#include <math.h>
#include "ripopt.h"

static int eval_f(int n, const double *x, int new_x,
                  double *obj, void *user_data)
{
    (void)n; (void)new_x; (void)user_data;
    *obj = 9.0 - 8.0*x[0] - 6.0*x[1] - 4.0*x[2]
         + 2.0*x[0]*x[0] + 2.0*x[1]*x[1] + x[2]*x[2]
         + 2.0*x[0]*x[1] + 2.0*x[0]*x[2];
    return 1;
}

static int eval_grad_f(int n, const double *x, int new_x,
                       double *grad, void *user_data)
{
    (void)n; (void)new_x; (void)user_data;
    grad[0] = -8.0 + 4.0*x[0] + 2.0*x[1] + 2.0*x[2];
    grad[1] = -6.0 + 2.0*x[0] + 4.0*x[1];
    grad[2] = -4.0 + 2.0*x[0] + 2.0*x[2];
    return 1;
}

static int eval_g(int n, const double *x, int new_x,
                  int m, double *g, void *user_data)
{
    (void)n; (void)m; (void)new_x; (void)user_data;
    g[0] = x[0] + x[1] + 2.0*x[2];
    return 1;
}

/* Jacobian: 3 nonzeros (dense 1x3 row) */
static int eval_jac_g(int n, const double *x, int new_x,
                      int m, int nele_jac,
                      int *iRow, int *jCol, double *values,
                      void *user_data)
{
    (void)n; (void)x; (void)m; (void)nele_jac; (void)new_x; (void)user_data;
    if (values == NULL) {
        iRow[0]=0; jCol[0]=0;
        iRow[1]=0; jCol[1]=1;
        iRow[2]=0; jCol[2]=2;
    } else {
        values[0] = 1.0;
        values[1] = 1.0;
        values[2] = 2.0;
    }
    return 1;
}

/* Hessian: 5 nonzeros (lower triangle of constant H = [[4,2,2],[2,4,0],[2,0,2]]) */
static int eval_h(int n, const double *x, int new_x,
                  double obj_factor,
                  int m, const double *lambda, int new_lambda,
                  int nele_hess,
                  int *iRow, int *jCol, double *values,
                  void *user_data)
{
    (void)n; (void)x; (void)m; (void)nele_hess; (void)new_x; (void)new_lambda;
    (void)lambda; (void)user_data;
    if (values == NULL) {
        /* (0,0), (1,0), (1,1), (2,0), (2,2) — skip (2,1) since H[2][1]=0 */
        iRow[0]=0; jCol[0]=0;
        iRow[1]=1; jCol[1]=0;
        iRow[2]=1; jCol[2]=1;
        iRow[3]=2; jCol[3]=0;
        iRow[4]=2; jCol[4]=2;
    } else {
        values[0] = obj_factor * 4.0;   /* d²f/dx1²       */
        values[1] = obj_factor * 2.0;   /* d²f/dx1dx2     */
        values[2] = obj_factor * 4.0;   /* d²f/dx2²       */
        values[3] = obj_factor * 2.0;   /* d²f/dx1dx3     */
        values[4] = obj_factor * 2.0;   /* d²f/dx3²       */
        /* No constraint Hessian — constraint is linear */
    }
    return 1;
}

int main(void)
{
    int n = 3, m = 1;

    double x_l[3] = {0.0, 0.0, 0.0};              /* x_i >= 0 */
    double x_u[3] = {HUGE_VAL, HUGE_VAL, HUGE_VAL}; /* no upper bounds */
    double g_l[1] = {-HUGE_VAL};                    /* no lower bound on constraint */
    double g_u[1] = {3.0};                         /* x1+x2+2x3 <= 3 */

    int nele_jac  = 3;
    int nele_hess = 5;

    RipoptProblem nlp = ripopt_create(
        n, x_l, x_u,
        m, g_l, g_u,
        nele_jac, nele_hess, 0, /* C-style 0-based indexing */
        eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h);

    if (!nlp) {
        fprintf(stderr, "ripopt_create failed\n");
        return 1;
    }

    ripopt_add_int_option(nlp, "print_level", 0);
    ripopt_add_num_option(nlp, "tol", 1e-8);

    /* Initial point */
    double x[3]       = {0.5, 0.5, 0.5};
    double obj_val     = 0.0;
    double g[1]        = {0.0};
    double mult_g[1]   = {0.0};
    double mult_xl[3]  = {0.0, 0.0, 0.0};
    double mult_xu[3]  = {0.0, 0.0, 0.0};

    int status = ripopt_solve(nlp, x, g, &obj_val,
                              mult_g, mult_xl, mult_xu, NULL);

    double expected_obj = 1.0/9.0;

    printf("=== HS035 (inequality + bounds) ===\n");
    printf("Status      : %d  (0 = Optimal)\n", status);
    printf("Obj         : %.10f  (expected %.10f)\n", obj_val, expected_obj);
    printf("x           : [%.6f, %.6f, %.6f]\n", x[0], x[1], x[2]);
    printf("              (expected [1.3333, 0.7778, 0.4444])\n");
    printf("g(x)        : [%.6f]  (should be <= 3)\n", g[0]);
    printf("mult_g      : [%.6f]  (constraint multiplier)\n", mult_g[0]);
    printf("mult_x_L    : [%.6f, %.6f, %.6f]  (bound multipliers)\n",
           mult_xl[0], mult_xl[1], mult_xl[2]);

    int pass = ((status == 0 || status == 1) &&
                fabs(obj_val - expected_obj) < 1e-3);
    printf("Test        : %s\n", pass ? "PASSED" : "FAILED");

    ripopt_free(nlp);
    return pass ? 0 : 1;
}
