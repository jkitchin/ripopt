/**
 * c_api_test.c — HS071 test problem using the ripopt C API.
 *
 * Problem (Hock & Schittkowski problem 71):
 *
 *   min  x1*x4*(x1+x2+x3) + x3
 *   s.t. x1*x2*x3*x4 >= 25
 *        x1^2+x2^2+x3^2+x4^2 = 40
 *        1 <= x_i <= 5,  i=1..4
 *
 * Expected solution:
 *   x* ≈ [1.000, 4.743, 3.821, 1.379]
 *   f*  ≈ 17.0140173
 *
 * Build (from repo root):
 *   cargo build --release
 *   cc examples/c_api_test.c -I. -Ltarget/release -lripopt \
 *       -Wl,-rpath,$(pwd)/target/release -o c_api_test
 *   ./c_api_test
 */

#include <stdio.h>
#include <math.h>
#include "ripopt.h"

/* -------------------------------------------------------------------------
 * Callbacks
 * ------------------------------------------------------------------------- */

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
    g[0] = x[0]*x[1]*x[2]*x[3];          /* >= 25 */
    g[1] = x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3];  /* == 40 */
    return 1;
}

/* Jacobian has 8 nonzeros (dense 2×4). */
static int eval_jac_g(int n, const double *x, int new_x,
                      int m, int nele_jac,
                      int *iRow, int *jCol, double *values,
                      void *user_data)
{
    (void)n; (void)m; (void)nele_jac; (void)new_x; (void)user_data;
    if (values == NULL) {
        /* Sparsity pattern (0-based) */
        iRow[0]=0; jCol[0]=0;
        iRow[1]=0; jCol[1]=1;
        iRow[2]=0; jCol[2]=2;
        iRow[3]=0; jCol[3]=3;
        iRow[4]=1; jCol[4]=0;
        iRow[5]=1; jCol[5]=1;
        iRow[6]=1; jCol[6]=2;
        iRow[7]=1; jCol[7]=3;
    } else {
        values[0] = x[1]*x[2]*x[3];
        values[1] = x[0]*x[2]*x[3];
        values[2] = x[0]*x[1]*x[3];
        values[3] = x[0]*x[1]*x[2];
        values[4] = 2.0*x[0];
        values[5] = 2.0*x[1];
        values[6] = 2.0*x[2];
        values[7] = 2.0*x[3];
    }
    return 1;
}

/* Hessian of Lagrangian (lower triangle): 10 nonzeros. */
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
        /* Lower-triangular sparsity pattern (0-based) */
        /* (0,0) */  iRow[0]=0; jCol[0]=0;
        /* (1,0) */  iRow[1]=1; jCol[1]=0;
        /* (1,1) */  iRow[2]=1; jCol[2]=1;
        /* (2,0) */  iRow[3]=2; jCol[3]=0;
        /* (2,1) */  iRow[4]=2; jCol[4]=1;
        /* (2,2) */  iRow[5]=2; jCol[5]=2;
        /* (3,0) */  iRow[6]=3; jCol[6]=0;
        /* (3,1) */  iRow[7]=3; jCol[7]=1;
        /* (3,2) */  iRow[8]=3; jCol[8]=2;
        /* (3,3) */  iRow[9]=3; jCol[9]=3;
    } else {
        /* Objective contribution */
        values[0] = obj_factor * 2.0*x[3];        /* (0,0) */
        values[1] = obj_factor * x[3];            /* (1,0) */
        values[2] = 0.0;                          /* (1,1) */
        values[3] = obj_factor * x[3];            /* (2,0) */
        values[4] = 0.0;                          /* (2,1) */
        values[5] = 0.0;                          /* (2,2) */
        values[6] = obj_factor * (2.0*x[0]+x[1]+x[2]); /* (3,0) */
        values[7] = obj_factor * x[0];            /* (3,1) */
        values[8] = obj_factor * x[0];            /* (3,2) */
        values[9] = 0.0;                          /* (3,3) */

        /* Constraint 0: g0 = x0*x1*x2*x3 */
        values[1] += lambda[0]*x[2]*x[3];  /* (1,0) */
        values[3] += lambda[0]*x[1]*x[3];  /* (2,0) */
        values[4] += lambda[0]*x[0]*x[3];  /* (2,1) */
        values[6] += lambda[0]*x[1]*x[2];  /* (3,0) */
        values[7] += lambda[0]*x[0]*x[2];  /* (3,1) */
        values[8] += lambda[0]*x[0]*x[1];  /* (3,2) */

        /* Constraint 1: g1 = x0^2+x1^2+x2^2+x3^2  (Hessian is 2*I) */
        values[0] += lambda[1]*2.0;   /* (0,0) */
        values[2] += lambda[1]*2.0;   /* (1,1) */
        values[5] += lambda[1]*2.0;   /* (2,2) */
        values[9] += lambda[1]*2.0;   /* (3,3) */
    }
    return 1;
}

/* -------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------- */

int main(void)
{
    int n = 4, m = 2;

    double x_l[4] = {1.0, 1.0, 1.0, 1.0};
    double x_u[4] = {5.0, 5.0, 5.0, 5.0};
    double g_l[2] = {25.0, 40.0};
    double g_u[2] = {HUGE_VAL, 40.0};  /* g0 >= 25  (upper = +inf), g1 == 40 */

    int nele_jac  = 8;
    int nele_hess = 10;

    RipoptProblem nlp = ripopt_create(
        n, x_l, x_u,
        m, g_l, g_u,
        nele_jac, nele_hess, 0, /* C-style 0-based indexing */
        eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h);

    if (!nlp) {
        fprintf(stderr, "ripopt_create failed\n");
        return 1;
    }

    /* Options */
    ripopt_add_int_option(nlp, "print_level", 5);
    ripopt_add_num_option(nlp, "tol",         1e-8);

    /* Initial point */
    double x[4]      = {1.0, 5.0, 5.0, 1.0};
    double g[2]      = {0.0, 0.0};
    double obj_val   = 0.0;
    double mult_g[2] = {0.0, 0.0};
    double mult_xl[4]= {0.0, 0.0, 0.0, 0.0};
    double mult_xu[4]= {0.0, 0.0, 0.0, 0.0};

    int status = ripopt_solve(nlp, x, g, &obj_val,
                              mult_g, mult_xl, mult_xu,
                              NULL);

    printf("\n=== HS071 Result ===\n");
    printf("Status : %d  (0 = Optimal)\n", status);
    printf("Obj    : %.10f  (expected ~17.0140173)\n", obj_val);
    printf("x      : [%.6f, %.6f, %.6f, %.6f]\n",
           x[0], x[1], x[2], x[3]);
    printf("         (expected ~[1.0, 4.743, 3.821, 1.379])\n");

    int pass = (status == RIPOPT_SOLVE_SUCCEEDED &&
                fabs(obj_val - 17.0140173) < 1e-4);
    printf("Test   : %s\n", pass ? "PASSED" : "FAILED");

    ripopt_free(nlp);
    return pass ? 0 : 1;
}
