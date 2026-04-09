/**
 * c_rosenbrock.c — Unconstrained Rosenbrock optimization via the ripopt C API.
 *
 * Problem:
 *   min  100*(x2 - x1^2)^2 + (1 - x1)^2
 *   (no constraints, no bounds)
 *
 * Expected solution:
 *   x* = [1.0, 1.0]
 *   f* = 0.0
 *
 * Build (from repo root):
 *   cargo build --release
 *   cc examples/c_rosenbrock.c -I. -Ltarget/release -lripopt \
 *       -Wl,-rpath,$(pwd)/target/release -o c_rosenbrock -lm
 *   ./c_rosenbrock
 */

#include <stdio.h>
#include <math.h>
#include "ripopt.h"

/* Objective: Rosenbrock function */
static int eval_f(int n, const double *x, int new_x,
                  double *obj, void *user_data)
{
    (void)n; (void)new_x; (void)user_data;
    double t1 = x[1] - x[0]*x[0];
    double t2 = 1.0 - x[0];
    *obj = 100.0*t1*t1 + t2*t2;
    return 1;
}

/* Gradient */
static int eval_grad_f(int n, const double *x, int new_x,
                       double *grad, void *user_data)
{
    (void)n; (void)new_x; (void)user_data;
    grad[0] = -400.0*x[0]*(x[1] - x[0]*x[0]) - 2.0*(1.0 - x[0]);
    grad[1] = 200.0*(x[1] - x[0]*x[0]);
    return 1;
}

/* No constraints */
static int eval_g(int n, const double *x, int new_x,
                  int m, double *g, void *user_data)
{
    (void)n; (void)x; (void)new_x; (void)m; (void)g; (void)user_data;
    return 1;
}

static int eval_jac_g(int n, const double *x, int new_x,
                      int m, int nele_jac,
                      int *iRow, int *jCol, double *values,
                      void *user_data)
{
    (void)n; (void)x; (void)new_x; (void)m; (void)nele_jac;
    (void)iRow; (void)jCol; (void)values; (void)user_data;
    return 1;
}

/* Hessian: 3 nonzeros in lower triangle of 2x2 matrix */
static int eval_h(int n, const double *x, int new_x,
                  double obj_factor,
                  int m, const double *lambda, int new_lambda,
                  int nele_hess,
                  int *iRow, int *jCol, double *values,
                  void *user_data)
{
    (void)n; (void)m; (void)nele_hess; (void)new_x; (void)new_lambda;
    (void)lambda; (void)user_data;
    if (values == NULL) {
        /* Lower-triangular sparsity pattern */
        iRow[0]=0; jCol[0]=0;   /* (0,0) */
        iRow[1]=1; jCol[1]=0;   /* (1,0) */
        iRow[2]=1; jCol[2]=1;   /* (1,1) */
    } else {
        values[0] = obj_factor * (-400.0*(x[1] - 3.0*x[0]*x[0]) + 2.0);
        values[1] = obj_factor * (-400.0*x[0]);
        values[2] = obj_factor * 200.0;
    }
    return 1;
}

int main(void)
{
    int n = 2, m = 0;

    /* No bounds (use +/- infinity) */
    double x_l[2] = {-1e30, -1e30};
    double x_u[2] = { 1e30,  1e30};

    RipoptProblem nlp = ripopt_create(
        n, x_l, x_u,
        m, NULL, NULL,    /* no constraints */
        0, 3, 0,          /* 0 Jacobian entries, 3 Hessian entries, C indexing */
        eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h);

    if (!nlp) {
        fprintf(stderr, "ripopt_create failed\n");
        return 1;
    }

    /* Quiet output */
    ripopt_add_int_option(nlp, "print_level", 0);
    ripopt_add_num_option(nlp, "tol", 1e-10);

    /* Initial point */
    double x[2] = {-1.0, 1.0};
    double obj_val = 0.0;

    int status = ripopt_solve(nlp, x, NULL, &obj_val,
                              NULL, NULL, NULL, NULL);

    printf("=== Rosenbrock (unconstrained) ===\n");
    printf("Status : %d  (0 = Optimal)\n", status);
    printf("Obj    : %.10e  (expected 0)\n", obj_val);
    printf("x      : [%.8f, %.8f]  (expected [1, 1])\n", x[0], x[1]);

    int pass = ((status == 0 || status == 1) &&
                fabs(obj_val) < 1e-6 &&
                fabs(x[0] - 1.0) < 1e-3 &&
                fabs(x[1] - 1.0) < 1e-3);
    printf("Test   : %s\n", pass ? "PASSED" : "FAILED");

    ripopt_free(nlp);
    return pass ? 0 : 1;
}
