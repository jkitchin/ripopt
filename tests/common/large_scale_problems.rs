
use ripopt::NlpProblem;
use std::f64::consts::PI;

// ===========================================================================
// Problem 1: Chained Rosenbrock (Extended Rosenbrock)
//   min f(x) = Σ_{i=0}^{n-2} [100*(x_{i+1} - x_i²)² + (1 - x_i)²]
//   Unconstrained, tridiagonal Hessian
//   x* = (1,...,1), f* = 0
// ===========================================================================

pub struct ChainedRosenbrock {
    pub n: usize,
}

impl NlpProblem for ChainedRosenbrock {
    fn num_variables(&self) -> usize {
        self.n
    }

    fn num_constraints(&self) -> usize {
        0
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..self.n {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }

    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        for i in 0..self.n {
            x0[i] = -1.2;
        }
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let mut f = 0.0;
        for i in 0..self.n - 1 {
            let a = 1.0 - x[i];
            let b = x[i + 1] - x[i] * x[i];
            f += a * a + 100.0 * b * b;
        }
        f
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let n = self.n;
        for g in grad.iter_mut() {
            *g = 0.0;
        }
        for i in 0..n - 1 {
            let xi = x[i];
            let xi1 = x[i + 1];
            let r = xi1 - xi * xi;
            grad[i] += -2.0 * (1.0 - xi) + 200.0 * r * (-2.0 * xi);
            grad[i + 1] += 200.0 * r;
        }
    }

    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) {}

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) {}

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Lower triangle: diagonal + sub-diagonal
        // Entries: (0,0), (1,0), (1,1), (2,1), (2,2), ...
        let n = self.n;
        let nnz = 2 * n - 1;
        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        // (0,0)
        rows.push(0);
        cols.push(0);
        for i in 1..n {
            // sub-diagonal (i, i-1)
            rows.push(i);
            cols.push(i - 1);
            // diagonal (i, i)
            rows.push(i);
            cols.push(i);
        }
        (rows, cols)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        let n = self.n;
        // Initialize all to zero
        for v in vals.iter_mut() {
            *v = 0.0;
        }
        // Entry mapping: vals[0] = (0,0), then for i>=1: vals[2*i-1] = (i,i-1), vals[2*i] = (i,i)
        // Index for diagonal(i): if i==0 then 0, else 2*i
        // Index for sub-diagonal(i,i-1) where i>=1: 2*i - 1

        for i in 0..n - 1 {
            let xi = x[i];
            let xi1 = x[i + 1];
            // d²f/dx_i² += 2 + 1200*x_i² - 400*x_{i+1}
            let diag_i_idx = if i == 0 { 0 } else { 2 * i };
            vals[diag_i_idx] += obj_factor * (2.0 + 1200.0 * xi * xi - 400.0 * xi1);
            // d²f/dx_{i+1}dx_i = -400*x_i  (sub-diagonal entry at (i+1, i))
            let sub_idx = 2 * (i + 1) - 1;
            vals[sub_idx] += obj_factor * (-400.0 * xi);
            // d²f/dx_{i+1}² += 200
            let diag_i1_idx = 2 * (i + 1);
            vals[diag_i1_idx] += obj_factor * 200.0;
        }
    }
}

// ===========================================================================
// Problem 2: Bratu BVP
//   Discretize -u'' = λ*exp(u) on [0,1], u(0)=u(1)=0
//   min f = 0
//   s.t. (-x_{i-1} + 2*x_i - x_{i+1})/h² - λ*exp(x_i) = 0, i=1..n-2
//        x_0 = 0, x_{n-1} = 0  (via bounds)
// ===========================================================================

pub struct BratuProblem {
    pub n: usize,
    lambda_bratu: f64,
    h: f64,
}

impl BratuProblem {
    pub fn new(n: usize) -> Self {
        let h = 1.0 / (n as f64 + 1.0);
        Self {
            n,
            lambda_bratu: 1.0,
            h,
        }
    }
}

impl NlpProblem for BratuProblem {
    fn num_variables(&self) -> usize {
        self.n
    }

    fn num_constraints(&self) -> usize {
        self.n - 2
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        // Interior points are free
        for i in 0..self.n {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
        // Boundary conditions: x_0 = 0, x_{n-1} = 0
        x_l[0] = 0.0;
        x_u[0] = 0.0;
        x_l[self.n - 1] = 0.0;
        x_u[self.n - 1] = 0.0;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for j in 0..self.n - 2 {
            g_l[j] = 0.0;
            g_u[j] = 0.0;
        }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        for i in 0..self.n {
            x0[i] = 0.0;
        }
    }

    fn objective(&self, _x: &[f64], _new_x: bool) -> f64 {
        0.0
    }

    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) {
        for g in grad.iter_mut() {
            *g = 0.0;
        }
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let h2 = self.h * self.h;
        // Constraint j corresponds to interior point i = j+1
        for j in 0..self.n - 2 {
            let i = j + 1;
            g[j] = (-x[i - 1] + 2.0 * x[i] - x[i + 1]) / h2
                - self.lambda_bratu * x[i].exp();
        }
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let m = self.n - 2;
        let mut rows = Vec::with_capacity(3 * m);
        let mut cols = Vec::with_capacity(3 * m);
        for j in 0..m {
            let i = j + 1;
            // dg_j/dx_{i-1}
            rows.push(j);
            cols.push(i - 1);
            // dg_j/dx_i
            rows.push(j);
            cols.push(i);
            // dg_j/dx_{i+1}
            rows.push(j);
            cols.push(i + 1);
        }
        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        let h2 = self.h * self.h;
        let m = self.n - 2;
        for j in 0..m {
            let i = j + 1;
            let base = 3 * j;
            vals[base] = -1.0 / h2; // dg_j/dx_{i-1}
            vals[base + 1] = 2.0 / h2 - self.lambda_bratu * x[i].exp(); // dg_j/dx_i
            vals[base + 2] = -1.0 / h2; // dg_j/dx_{i+1}
        }
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Only diagonal entries from constraint Hessians (exp term)
        // Each constraint j has d²g_j/dx_{j+1}² = -λ*exp(x_{j+1})
        // So Hessian has entries at (k,k) for k=1..n-2
        // But we need all n diagonal entries for a complete structure
        let mut rows = Vec::with_capacity(self.n);
        let mut cols = Vec::with_capacity(self.n);
        for k in 0..self.n {
            rows.push(k);
            cols.push(k);
        }
        (rows, cols)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, _obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        // f = 0, so no objective Hessian
        for v in vals.iter_mut() {
            *v = 0.0;
        }
        // Constraint Hessian: d²g_j/dx_{j+1}² = -λ*exp(x_{j+1})
        // lambda[j] * (-λ*exp(x_{j+1})) contributes to vals[j+1]
        for j in 0..self.n - 2 {
            let k = j + 1;
            vals[k] += lambda[j] * (-self.lambda_bratu * x[k].exp());
        }
    }
}

// ===========================================================================
// Problem 3: Discretized Optimal Control (LQR)
//   min h*Σ(y_i - 1)² + α*h*Σu_i²
//   s.t. y_{i+1} = y_i + h*(-y_i + u_i),  i=0..T-1
//        y_0 = 0
//   Variables: [y_0, ..., y_T, u_0, ..., u_{T-1}]
//   n = 2T+1, m = T+1
// ===========================================================================

pub struct OptimalControl {
    t: usize, // number of time steps
    h: f64,
    alpha: f64,
}

impl OptimalControl {
    pub fn new(t: usize) -> Self {
        Self {
            t,
            h: 1.0 / t as f64,
            alpha: 0.01,
        }
    }

    fn n(&self) -> usize {
        2 * self.t + 1
    }
}

impl NlpProblem for OptimalControl {
    fn num_variables(&self) -> usize {
        self.n()
    }

    fn num_constraints(&self) -> usize {
        self.t + 1
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        let n = self.n();
        for i in 0..n {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        let m = self.t + 1;
        for j in 0..m {
            g_l[j] = 0.0;
            g_u[j] = 0.0;
        }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        for v in x0.iter_mut() {
            *v = 0.0;
        }
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let h = self.h;
        let t = self.t;
        let mut f = 0.0;
        // State tracking: h * Σ(y_i - 1)²
        for i in 0..=t {
            let dy = x[i] - 1.0;
            f += h * dy * dy;
        }
        // Control cost: α*h * Σu_i²
        for i in 0..t {
            let u = x[t + 1 + i];
            f += self.alpha * h * u * u;
        }
        f
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let h = self.h;
        let t = self.t;
        // State: grad[i] = 2*h*(y_i - 1)
        for i in 0..=t {
            grad[i] = 2.0 * h * (x[i] - 1.0);
        }
        // Control: grad[T+1+i] = 2*α*h*u_i
        for i in 0..t {
            grad[t + 1 + i] = 2.0 * self.alpha * h * x[t + 1 + i];
        }
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let h = self.h;
        let t = self.t;
        // g[0] = y_0 - 0 = y_0  (initial condition y_0 = 0)
        g[0] = x[0];
        // g[i+1] = y_{i+1} - y_i - h*(-y_i + u_i) = y_{i+1} - (1-h)*y_i - h*u_i
        for i in 0..t {
            g[i + 1] = x[i + 1] - (1.0 - h) * x[i] - h * x[t + 1 + i];
        }
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let t = self.t;
        // Constraint 0: dg_0/dy_0 → (0, 0)
        // Constraint i+1: dg/dy_i → (i+1, i), dg/dy_{i+1} → (i+1, i+1), dg/du_i → (i+1, T+1+i)
        let nnz = 1 + 3 * t;
        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        rows.push(0);
        cols.push(0);
        for i in 0..t {
            rows.push(i + 1);
            cols.push(i);
            rows.push(i + 1);
            cols.push(i + 1);
            rows.push(i + 1);
            cols.push(t + 1 + i);
        }
        (rows, cols)
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) {
        let h = self.h;
        let t = self.t;
        vals[0] = 1.0; // dg_0/dy_0
        for i in 0..t {
            let base = 1 + 3 * i;
            vals[base] = -(1.0 - h); // dg_{i+1}/dy_i
            vals[base + 1] = 1.0; // dg_{i+1}/dy_{i+1}
            vals[base + 2] = -h; // dg_{i+1}/du_i
        }
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Diagonal only (objective is separable quadratic, constraints are linear)
        let n = self.n();
        let mut rows = Vec::with_capacity(n);
        let mut cols = Vec::with_capacity(n);
        for k in 0..n {
            rows.push(k);
            cols.push(k);
        }
        (rows, cols)
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        let h = self.h;
        let t = self.t;
        // State: H[i,i] = 2*h*obj_factor
        for i in 0..=t {
            vals[i] = obj_factor * 2.0 * h;
        }
        // Control: H[T+1+i, T+1+i] = 2*α*h*obj_factor
        for i in 0..t {
            vals[t + 1 + i] = obj_factor * 2.0 * self.alpha * h;
        }
    }
}

// ===========================================================================
// Problem 4: 2D Poisson Control
//   min 0.5*h²*Σ(u_{ij} - u_d(xi,yj))² + (α/2)*h²*Σf_{ij}²
//   s.t. -Δ_h u_{ij} = f_{ij}  (5-point stencil)
//   Variables: [u_{0,0},...,u_{K-1,K-1}, f_{0,0},...,f_{K-1,K-1}]
//   n = 2K², m = K²
// ===========================================================================

pub struct PoissonControl {
    k: usize, // grid points per dimension
    h: f64,
    alpha: f64,
}

impl PoissonControl {
    pub fn new(k: usize) -> Self {
        let h = 1.0 / (k as f64 + 1.0);
        Self {
            k,
            h,
            alpha: 0.01,
        }
    }

    #[inline]
    fn idx_u(&self, i: usize, j: usize) -> usize {
        i + j * self.k
    }

    #[inline]
    fn idx_f(&self, i: usize, j: usize) -> usize {
        self.k * self.k + i + j * self.k
    }

    fn u_desired(&self, i: usize, j: usize) -> f64 {
        let x = (i as f64 + 1.0) * self.h;
        let y = (j as f64 + 1.0) * self.h;
        (PI * x).sin() * (PI * y).sin()
    }
}

impl NlpProblem for PoissonControl {
    fn num_variables(&self) -> usize {
        2 * self.k * self.k
    }

    fn num_constraints(&self) -> usize {
        self.k * self.k
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        let n = self.num_variables();
        for i in 0..n {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        let m = self.num_constraints();
        for j in 0..m {
            g_l[j] = 0.0;
            g_u[j] = 0.0;
        }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        for v in x0.iter_mut() {
            *v = 0.0;
        }
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let k = self.k;
        let h2 = self.h * self.h;
        let mut f = 0.0;
        for j in 0..k {
            for i in 0..k {
                let u = x[self.idx_u(i, j)];
                let ud = self.u_desired(i, j);
                f += 0.5 * h2 * (u - ud) * (u - ud);

                let fi = x[self.idx_f(i, j)];
                f += 0.5 * self.alpha * h2 * fi * fi;
            }
        }
        f
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let k = self.k;
        let h2 = self.h * self.h;
        for v in grad.iter_mut() {
            *v = 0.0;
        }
        for j in 0..k {
            for i in 0..k {
                let u = x[self.idx_u(i, j)];
                let ud = self.u_desired(i, j);
                grad[self.idx_u(i, j)] = h2 * (u - ud);
                grad[self.idx_f(i, j)] = self.alpha * h2 * x[self.idx_f(i, j)];
            }
        }
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let k = self.k;
        let h2 = self.h * self.h;
        for j in 0..k {
            for i in 0..k {
                let c = j * k + i; // constraint index
                let center = x[self.idx_u(i, j)];
                let mut laplacian = 4.0 * center;
                if i > 0 {
                    laplacian -= x[self.idx_u(i - 1, j)];
                }
                if i < k - 1 {
                    laplacian -= x[self.idx_u(i + 1, j)];
                }
                if j > 0 {
                    laplacian -= x[self.idx_u(i, j - 1)];
                }
                if j < k - 1 {
                    laplacian -= x[self.idx_u(i, j + 1)];
                }
                g[c] = laplacian / h2 - x[self.idx_f(i, j)];
            }
        }
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let k = self.k;
        // Each constraint has up to 5 state entries + 1 control entry
        let mut rows = Vec::with_capacity(6 * k * k);
        let mut cols = Vec::with_capacity(6 * k * k);
        for j in 0..k {
            for i in 0..k {
                let c = j * k + i;
                // Center
                rows.push(c);
                cols.push(self.idx_u(i, j));
                // Left
                if i > 0 {
                    rows.push(c);
                    cols.push(self.idx_u(i - 1, j));
                }
                // Right
                if i < k - 1 {
                    rows.push(c);
                    cols.push(self.idx_u(i + 1, j));
                }
                // Down
                if j > 0 {
                    rows.push(c);
                    cols.push(self.idx_u(i, j - 1));
                }
                // Up
                if j < k - 1 {
                    rows.push(c);
                    cols.push(self.idx_u(i, j + 1));
                }
                // Control
                rows.push(c);
                cols.push(self.idx_f(i, j));
            }
        }
        (rows, cols)
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) {
        let k = self.k;
        let h2 = self.h * self.h;
        let mut idx = 0;
        for j in 0..k {
            for i in 0..k {
                // Center: 4/h²
                vals[idx] = 4.0 / h2;
                idx += 1;
                // Left: -1/h²
                if i > 0 {
                    vals[idx] = -1.0 / h2;
                    idx += 1;
                }
                // Right: -1/h²
                if i < k - 1 {
                    vals[idx] = -1.0 / h2;
                    idx += 1;
                }
                // Down: -1/h²
                if j > 0 {
                    vals[idx] = -1.0 / h2;
                    idx += 1;
                }
                // Up: -1/h²
                if j < k - 1 {
                    vals[idx] = -1.0 / h2;
                    idx += 1;
                }
                // Control: -1
                vals[idx] = -1.0;
                idx += 1;
            }
        }
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Diagonal only (objective is separable quadratic, constraints are linear)
        let n = self.num_variables();
        let mut rows = Vec::with_capacity(n);
        let mut cols = Vec::with_capacity(n);
        for k in 0..n {
            rows.push(k);
            cols.push(k);
        }
        (rows, cols)
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        let k = self.k;
        let h2 = self.h * self.h;
        for j in 0..k {
            for i in 0..k {
                vals[self.idx_u(i, j)] = obj_factor * h2;
                vals[self.idx_f(i, j)] = obj_factor * self.alpha * h2;
            }
        }
    }
}

// ===========================================================================
// Problem 5: Sparse QP with Inequality Constraints
//   min 0.5*x^T*Q*x - Σx_i
//   s.t. x_j + x_{(j+1)%n} + x_{(j+2)%n} <= 2.5,  j=0..m-1
//        0 <= x_i <= 10
//   Q = tridiagonal (4 on diagonal, -1 off-diagonal), SPD
//   n = m = 50000
// ===========================================================================

pub struct SparseQP {
    pub n: usize,
}

impl NlpProblem for SparseQP {
    fn num_variables(&self) -> usize {
        self.n
    }

    fn num_constraints(&self) -> usize {
        self.n
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..self.n {
            x_l[i] = 0.0;
            x_u[i] = 10.0;
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for j in 0..self.n {
            g_l[j] = f64::NEG_INFINITY;
            g_u[j] = 2.5;
        }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        for i in 0..self.n {
            x0[i] = 0.5;
        }
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let n = self.n;
        let mut f = 0.0;
        // 0.5 * x^T * Q * x where Q is tridiagonal: 4 on diag, -1 off-diag
        for i in 0..n {
            f += 0.5 * 4.0 * x[i] * x[i];
            if i < n - 1 {
                f += 0.5 * (-1.0) * x[i] * x[i + 1] * 2.0; // symmetric: both (i,i+1) and (i+1,i)
            }
        }
        // - Σx_i
        for i in 0..n {
            f -= x[i];
        }
        f
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let n = self.n;
        // grad = Q*x - 1
        for i in 0..n {
            grad[i] = 4.0 * x[i] - 1.0;
            if i > 0 {
                grad[i] -= x[i - 1];
            }
            if i < n - 1 {
                grad[i] -= x[i + 1];
            }
        }
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let n = self.n;
        for j in 0..n {
            g[j] = x[j] + x[(j + 1) % n] + x[(j + 2) % n];
        }
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.n;
        let mut rows = Vec::with_capacity(3 * n);
        let mut cols = Vec::with_capacity(3 * n);
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            rows.push(j);
            cols.push((j + 1) % n);
            rows.push(j);
            cols.push((j + 2) % n);
        }
        (rows, cols)
    }

    fn jacobian_values(&self, _x: &[f64], _new_x: bool, vals: &mut [f64]) {
        let n = self.n;
        for j in 0..n {
            let base = 3 * j;
            vals[base] = 1.0;
            vals[base + 1] = 1.0;
            vals[base + 2] = 1.0;
        }
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Q is tridiagonal: diagonal + sub-diagonal (lower triangle)
        let n = self.n;
        let mut rows = Vec::with_capacity(2 * n - 1);
        let mut cols = Vec::with_capacity(2 * n - 1);
        rows.push(0);
        cols.push(0);
        for i in 1..n {
            rows.push(i);
            cols.push(i - 1);
            rows.push(i);
            cols.push(i);
        }
        (rows, cols)
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        let n = self.n;
        // Constraints are linear → no Hessian contribution from lambda
        // Q: 4 on diagonal, -1 on sub-diagonal
        // Same layout as ChainedRosenbrock: vals[0]=(0,0), vals[2i-1]=(i,i-1), vals[2i]=(i,i)
        vals[0] = obj_factor * 4.0;
        for i in 1..n {
            vals[2 * i - 1] = obj_factor * (-1.0);
            vals[2 * i] = obj_factor * 4.0;
        }
    }
}
