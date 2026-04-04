#![allow(unused_variables)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::needless_return)]

use ripopt::NlpProblem;


pub struct HsTp001;

impl NlpProblem for HsTp001 {
    fn num_variables(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        0
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
        x_l[1] = -1.5;
        x_u[1] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {

    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = -2.0;
        x0[1] = 1.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        -2.0*x[0] + 100.0*x[0].powi(4) + 100.0*x[1].powi(2) - 200.0*x[1]*x[0].powi(2) + 1.0 + x[0].powi(2)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0*x[0] + 400.0*x[0].powi(3) - 400.0*x[0]*x[1] - 2.0;
        grad[1] = -200.0*x[0].powi(2) + 200.0*x[1];
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let _ = (x, g);
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        let _ = (x, vals);
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1], vec![0, 0, 1])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (1200.0*x[0].powi(2) - 400.0*x[1] + 2.0);
        vals[1] = obj_factor * (-400.0*x[0]);
        vals[2] = obj_factor * (200.000000000000);
    }
}

pub struct HsTp006;

impl NlpProblem for HsTp006 {
    fn num_variables(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        1
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY;
        x_u[1] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = 0.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = -1.2;
        x0[1] = 1.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        -2.0*x[0] + 1.0 + x[0].powi(2)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0*x[0] - 2.0;
        grad[1] = 0.0;
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = -10.0*x[0].powi(2) + 10.0*x[1];
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0], vec![0, 1])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = -20.0*x[0];
        vals[1] = 10.0000000000000;
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (2.0) + lambda[0] * (-20.0000000000000);
    }
}

pub struct HsTp012;

impl NlpProblem for HsTp012 {
    fn num_variables(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        1
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY;
        x_u[1] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = f64::INFINITY;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
        x0[1] = 0.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        0.5*x[0].powi(2) - x[0]*x[1] - 7.0*x[0] + x[1].powi(2) - 7.0*x[1]
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 1.0*x[0] - x[1] - 7.0;
        grad[1] = -x[0] + 2.0*x[1] - 7.0;
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = -4.0*x[0].powi(2) - x[1].powi(2) + 25.0;
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0], vec![0, 1])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = -8.0*x[0];
        vals[1] = -2.0*x[1];
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1], vec![0, 0, 1])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (1.00000000000000) + lambda[0] * (-8.00000000000000);
        vals[1] = obj_factor * (-1.0);
        vals[2] = obj_factor * (2.0) + lambda[0] * (-2.0);
    }
}

pub struct HsTp035;

impl NlpProblem for HsTp035 {
    fn num_variables(&self) -> usize {
        3
    }

    fn num_constraints(&self) -> usize {
        1
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 0.0;
        x_u[0] = f64::INFINITY;
        x_l[1] = 0.0;
        x_u[1] = f64::INFINITY;
        x_l[2] = 0.0;
        x_u[2] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = f64::INFINITY;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.5;
        x0[1] = 0.5;
        x0[2] = 0.5;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        -8.0*x[0] + 2.0*x[0].powi(2) - 6.0*x[1] + 2.0*x[1].powi(2) - 4.0*x[2] + 2.0*x[0]*x[1] + 2.0*x[0]*x[2] + 9.0 + x[2].powi(2)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 4.0*x[0] + 2.0*x[1] + 2.0*x[2] - 8.0;
        grad[1] = 2.0*x[0] + 4.0*x[1] - 6.0;
        grad[2] = 2.0*x[0] + 2.0*x[2] - 4.0;
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = -x[0] - x[1] - 2.0*x[2] + 3.0;
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 0], vec![0, 1, 2])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = -1.0;
        vals[1] = -1.0;
        vals[2] = -2.00000000000000;
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1, 2, 2], vec![0, 0, 1, 0, 2])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (4.00000000000000);
        vals[1] = obj_factor * (2.00000000000000);
        vals[2] = obj_factor * (4.00000000000000);
        vals[3] = obj_factor * (2.00000000000000);
        vals[4] = obj_factor * (2.0);
    }
}

pub struct HsTp044;

impl NlpProblem for HsTp044 {
    fn num_variables(&self) -> usize {
        4
    }

    fn num_constraints(&self) -> usize {
        6
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 0.0;
        x_u[0] = f64::INFINITY;
        x_l[1] = 0.0;
        x_u[1] = f64::INFINITY;
        x_l[2] = 0.0;
        x_u[2] = f64::INFINITY;
        x_l[3] = 0.0;
        x_u[3] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = f64::INFINITY;
        g_l[1] = 0.0;
        g_u[1] = f64::INFINITY;
        g_l[2] = 0.0;
        g_u[2] = f64::INFINITY;
        g_l[3] = 0.0;
        g_u[3] = f64::INFINITY;
        g_l[4] = 0.0;
        g_u[4] = f64::INFINITY;
        g_l[5] = 0.0;
        g_u[5] = f64::INFINITY;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
        x0[1] = 0.0;
        x0[2] = 0.0;
        x0[3] = 0.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        -x[0]*x[2] + x[0]*x[3] + x[0] + x[1]*x[2] - x[1]*x[3] - x[1] - x[2]
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = -x[2] + x[3] + 1.0;
        grad[1] = x[2] - x[3] - 1.0;
        grad[2] = -x[0] + x[1] - 1.0;
        grad[3] = x[0] - x[1];
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = -x[0] - 2.0*x[1] + 8.0;
        g[1] = -4.0*x[0] - x[1] + 12.0;
        g[2] = -3.0*x[0] - 4.0*x[1] + 12.0;
        g[3] = -2.0*x[2] - x[3] + 8.0;
        g[4] = -x[2] - 2.0*x[3] + 8.0;
        g[5] = -x[2] - x[3] + 5.0;
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], vec![0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = -1.0;
        vals[1] = -2.00000000000000;
        vals[2] = -4.00000000000000;
        vals[3] = -1.0;
        vals[4] = -3.00000000000000;
        vals[5] = -4.00000000000000;
        vals[6] = -2.00000000000000;
        vals[7] = -1.0;
        vals[8] = -1.0;
        vals[9] = -2.00000000000000;
        vals[10] = -1.0;
        vals[11] = -1.0;
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![2, 2, 3, 3], vec![0, 1, 0, 1])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (-1.0);
        vals[1] = obj_factor * (1.0);
        vals[2] = obj_factor * (1.0);
        vals[3] = obj_factor * (-1.0);
    }
}

pub struct HsTp045;

impl NlpProblem for HsTp045 {
    fn num_variables(&self) -> usize {
        5
    }

    fn num_constraints(&self) -> usize {
        0
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 0.0;
        x_u[0] = 1.0;
        x_l[1] = 0.0;
        x_u[1] = 2.0;
        x_l[2] = 0.0;
        x_u[2] = 3.0;
        x_l[3] = 0.0;
        x_u[3] = 4.0;
        x_l[4] = 0.0;
        x_u[4] = 5.0;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {

    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 2.0;
        x0[1] = 2.0;
        x0[2] = 2.0;
        x0[3] = 2.0;
        x0[4] = 2.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        -0.00833333333333333*x[0]*x[1]*x[2]*x[3]*x[4] + 2.0
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = -0.00833333333333333*x[1]*x[2]*x[3]*x[4];
        grad[1] = -0.00833333333333333*x[0]*x[2]*x[3]*x[4];
        grad[2] = -0.00833333333333333*x[0]*x[1]*x[3]*x[4];
        grad[3] = -0.00833333333333333*x[0]*x[1]*x[2]*x[4];
        grad[4] = -0.00833333333333333*x[0]*x[1]*x[2]*x[3];
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let _ = (x, g);
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        let _ = (x, vals);
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![1, 2, 2, 3, 3, 3, 4, 4, 4, 4], vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (-0.00833333333333333*x[2]*x[3]*x[4]);
        vals[1] = obj_factor * (-0.00833333333333333*x[1]*x[3]*x[4]);
        vals[2] = obj_factor * (-0.00833333333333333*x[0]*x[3]*x[4]);
        vals[3] = obj_factor * (-0.00833333333333333*x[1]*x[2]*x[4]);
        vals[4] = obj_factor * (-0.00833333333333333*x[0]*x[2]*x[4]);
        vals[5] = obj_factor * (-0.00833333333333333*x[0]*x[1]*x[4]);
        vals[6] = obj_factor * (-0.00833333333333333*x[1]*x[2]*x[3]);
        vals[7] = obj_factor * (-0.00833333333333333*x[0]*x[2]*x[3]);
        vals[8] = obj_factor * (-0.00833333333333333*x[0]*x[1]*x[3]);
        vals[9] = obj_factor * (-0.00833333333333333*x[0]*x[1]*x[2]);
    }
}

pub struct HsTp048;

impl NlpProblem for HsTp048 {
    fn num_variables(&self) -> usize {
        5
    }

    fn num_constraints(&self) -> usize {
        2
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY;
        x_u[1] = f64::INFINITY;
        x_l[2] = f64::NEG_INFINITY;
        x_u[2] = f64::INFINITY;
        x_l[3] = f64::NEG_INFINITY;
        x_u[3] = f64::INFINITY;
        x_l[4] = f64::NEG_INFINITY;
        x_u[4] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = 0.0;
        g_l[1] = 0.0;
        g_u[1] = 0.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 3.0;
        x0[1] = 5.0;
        x0[2] = -3.0;
        x0[3] = 2.0;
        x0[4] = -2.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        -2.0*x[0] - 2.0*x[1]*x[2] - 2.0*x[3]*x[4] + 1.0 + x[0].powi(2) + x[1].powi(2) + x[2].powi(2) + x[3].powi(2) + x[4].powi(2)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0*x[0] - 2.0;
        grad[1] = 2.0*x[1] - 2.0*x[2];
        grad[2] = -2.0*x[1] + 2.0*x[2];
        grad[3] = 2.0*x[3] - 2.0*x[4];
        grad[4] = -2.0*x[3] + 2.0*x[4];
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0] + x[1] + x[2] + x[3] + x[4] - 5.0;
        g[1] = x[2] - 2.0*x[3] - 2.0*x[4] + 3.0;
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 0, 0, 0, 1, 1, 1], vec![0, 1, 2, 3, 4, 2, 3, 4])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 1.0;
        vals[1] = 1.0;
        vals[2] = 1.0;
        vals[3] = 1.0;
        vals[4] = 1.0;
        vals[5] = 1.0;
        vals[6] = -2.00000000000000;
        vals[7] = -2.00000000000000;
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 2, 2, 3, 4, 4], vec![0, 1, 1, 2, 3, 3, 4])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (2.0);
        vals[1] = obj_factor * (2.0);
        vals[2] = obj_factor * (-2.0);
        vals[3] = obj_factor * (2.0);
        vals[4] = obj_factor * (2.0);
        vals[5] = obj_factor * (-2.0);
        vals[6] = obj_factor * (2.0);
    }
}

pub struct HsTp071;

impl NlpProblem for HsTp071 {
    fn num_variables(&self) -> usize {
        4
    }

    fn num_constraints(&self) -> usize {
        2
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 1.0;
        x_u[0] = 5.0;
        x_l[1] = 1.0;
        x_u[1] = 5.0;
        x_l[2] = 1.0;
        x_u[2] = 5.0;
        x_l[3] = 1.0;
        x_u[3] = 5.0;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = f64::INFINITY;
        g_l[1] = 0.0;
        g_u[1] = 0.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 1.0;
        x0[1] = 5.0;
        x0[2] = 5.0;
        x0[3] = 1.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        x[0].powi(2)*x[3] + x[0]*x[1]*x[3] + x[0]*x[2]*x[3] + x[2]
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0*x[0]*x[3] + x[1]*x[3] + x[2]*x[3];
        grad[1] = x[0]*x[3];
        grad[2] = x[0]*x[3] + 1.0;
        grad[3] = x[0].powi(2) + x[0]*x[1] + x[0]*x[2];
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = 0.04*x[0]*x[1]*x[2]*x[3] - 1.0;
        g[1] = 0.025*x[0].powi(2) + 0.025*x[1].powi(2) + 0.025*x[2].powi(2) + 0.025*x[3].powi(2) - 1.0;
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 0, 0, 1, 1, 1, 1], vec![0, 1, 2, 3, 0, 1, 2, 3])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 0.04*x[1]*x[2]*x[3];
        vals[1] = 0.04*x[0]*x[2]*x[3];
        vals[2] = 0.04*x[0]*x[1]*x[3];
        vals[3] = 0.04*x[0]*x[1]*x[2];
        vals[4] = 0.05*x[0];
        vals[5] = 0.05*x[1];
        vals[6] = 0.05*x[2];
        vals[7] = 0.05*x[3];
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1, 2, 2, 2, 3, 3, 3, 3], vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (2.0*x[3]) + lambda[1] * (0.0500000000000000);
        vals[1] = obj_factor * (x[3]) + lambda[0] * (0.04*x[2]*x[3]);
        vals[2] = lambda[1] * (0.0500000000000000);
        vals[3] = obj_factor * (x[3]) + lambda[0] * (0.04*x[1]*x[3]);
        vals[4] = lambda[0] * (0.04*x[0]*x[3]);
        vals[5] = lambda[1] * (0.0500000000000000);
        vals[6] = obj_factor * (2.0*x[0] + x[1] + x[2]) + lambda[0] * (0.04*x[1]*x[2]);
        vals[7] = obj_factor * (x[0]) + lambda[0] * (0.04*x[0]*x[2]);
        vals[8] = obj_factor * (x[0]) + lambda[0] * (0.04*x[0]*x[1]);
        vals[9] = lambda[1] * (0.0500000000000000);
    }
}

pub struct HsTp081;

impl NlpProblem for HsTp081 {
    fn num_variables(&self) -> usize {
        5
    }

    fn num_constraints(&self) -> usize {
        3
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = -2.3;
        x_u[0] = 2.3;
        x_l[1] = -2.3;
        x_u[1] = 2.3;
        x_l[2] = -3.2;
        x_u[2] = 3.2;
        x_l[3] = -3.2;
        x_u[3] = 3.2;
        x_l[4] = -3.2;
        x_u[4] = 3.2;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = 0.0;
        g_l[1] = 0.0;
        g_u[1] = 0.0;
        g_l[2] = 0.0;
        g_u[2] = 0.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = -2.0;
        x0[1] = 2.0;
        x0[2] = 2.0;
        x0[3] = -1.0;
        x0[4] = -1.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        -1.0*x[0].powi(3) - 0.5*x[0].powi(6) - 1.0*x[1].powi(3) - 0.5*x[1].powi(6) - 1.0*x[0].powi(3)*x[1].powi(3) + (x[0]*x[1]*x[2]*x[3]*x[4]).exp() - 0.5
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = -3.0*x[0].powi(5) - 3.0*x[0].powi(2)*x[1].powi(3) - 3.0*x[0].powi(2) + x[1]*x[2]*x[3]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp();
        grad[1] = -3.0*x[0].powi(3)*x[1].powi(2) + x[0]*x[2]*x[3]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() - 3.0*x[1].powi(5) - 3.0*x[1].powi(2);
        grad[2] = x[0]*x[1]*x[3]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp();
        grad[3] = x[0]*x[1]*x[2]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp();
        grad[4] = x[0]*x[1]*x[2]*x[3]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp();
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = -10.0 + x[0].powi(2) + x[1].powi(2) + x[2].powi(2) + x[3].powi(2) + x[4].powi(2);
        g[1] = x[1]*x[2] - 5.0*x[3]*x[4];
        g[2] = 1.0 + x[0].powi(3) + x[1].powi(3);
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2], vec![0, 1, 2, 3, 4, 1, 2, 3, 4, 0, 1])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = 2.0*x[0];
        vals[1] = 2.0*x[1];
        vals[2] = 2.0*x[2];
        vals[3] = 2.0*x[3];
        vals[4] = 2.0*x[4];
        vals[5] = x[2];
        vals[6] = x[1];
        vals[7] = -5.0*x[4];
        vals[8] = -5.0*x[3];
        vals[9] = 3.0*x[0].powi(2);
        vals[10] = 3.0*x[1].powi(2);
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4], vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (-15.0*x[0].powi(4) - 6.0*x[0]*x[1].powi(3) - 6.0*x[0] + x[1].powi(2)*x[2].powi(2)*x[3].powi(2)*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp()) + lambda[0] * (2.0) + lambda[2] * (6.0*x[0]);
        vals[1] = obj_factor * (-9.0*x[0].powi(2)*x[1].powi(2) + x[0]*x[1]*x[2].powi(2)*x[3].powi(2)*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[2]*x[3]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[2] = obj_factor * (-6.0*x[1]*x[0].powi(3) + x[0].powi(2)*x[2].powi(2)*x[3].powi(2)*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() - 15.0*x[1].powi(4) - 6.0*x[1]) + lambda[0] * (2.0) + lambda[2] * (6.0*x[1]);
        vals[3] = obj_factor * (x[0]*x[1].powi(2)*x[2]*x[3].powi(2)*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[1]*x[3]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[4] = obj_factor * (x[0].powi(2)*x[1]*x[2]*x[3].powi(2)*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[0]*x[3]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp()) + lambda[1] * (1.0);
        vals[5] = obj_factor * (x[0].powi(2)*x[1].powi(2)*x[3].powi(2)*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp()) + lambda[0] * (2.0);
        vals[6] = obj_factor * (x[0]*x[1].powi(2)*x[2].powi(2)*x[3]*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[1]*x[2]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[7] = obj_factor * (x[0].powi(2)*x[1]*x[2].powi(2)*x[3]*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[0]*x[2]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[8] = obj_factor * (x[0].powi(2)*x[1].powi(2)*x[2]*x[3]*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[0]*x[1]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[9] = obj_factor * (x[0].powi(2)*x[1].powi(2)*x[2].powi(2)*x[4].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp()) + lambda[0] * (2.0);
        vals[10] = obj_factor * (x[0]*x[1].powi(2)*x[2].powi(2)*x[3].powi(2)*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[1]*x[2]*x[3]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[11] = obj_factor * (x[0].powi(2)*x[1]*x[2].powi(2)*x[3].powi(2)*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[0]*x[2]*x[3]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[12] = obj_factor * (x[0].powi(2)*x[1].powi(2)*x[2]*x[3].powi(2)*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[0]*x[1]*x[3]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp());
        vals[13] = obj_factor * (x[0].powi(2)*x[1].powi(2)*x[2].powi(2)*x[3]*x[4]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp() + x[0]*x[1]*x[2]*(x[0]*x[1]*x[2]*x[3]*x[4]).exp()) + lambda[1] * (-5.00000000000000);
        vals[14] = obj_factor * (x[0].powi(2)*x[1].powi(2)*x[2].powi(2)*x[3].powi(2)*(x[0]*x[1]*x[2]*x[3]*x[4]).exp()) + lambda[0] * (2.0);
    }
}

pub struct HsTp106;

impl NlpProblem for HsTp106 {
    fn num_variables(&self) -> usize {
        8
    }

    fn num_constraints(&self) -> usize {
        6
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 100.0;
        x_u[0] = 10000.0;
        x_l[1] = 1000.0;
        x_u[1] = 10000.0;
        x_l[2] = 1000.0;
        x_u[2] = 10000.0;
        x_l[3] = 10.0;
        x_u[3] = 1000.0;
        x_l[4] = 10.0;
        x_u[4] = 1000.0;
        x_l[5] = 10.0;
        x_u[5] = 1000.0;
        x_l[6] = 10.0;
        x_u[6] = 1000.0;
        x_l[7] = 10.0;
        x_u[7] = 1000.0;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = f64::INFINITY;
        g_l[1] = 0.0;
        g_u[1] = f64::INFINITY;
        g_l[2] = 0.0;
        g_u[2] = f64::INFINITY;
        g_l[3] = 0.0;
        g_u[3] = f64::INFINITY;
        g_l[4] = 0.0;
        g_u[4] = f64::INFINITY;
        g_l[5] = 0.0;
        g_u[5] = f64::INFINITY;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 5000.0;
        x0[1] = 5000.0;
        x0[2] = 5000.0;
        x0[3] = 200.0;
        x0[4] = 350.0;
        x0[5] = 150.0;
        x0[6] = 225.0;
        x0[7] = 425.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        x[0] + x[1] + x[2]
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 1.0;
        grad[1] = 1.0;
        grad[2] = 1.0;
        grad[3] = 0.0;
        grad[4] = 0.0;
        grad[5] = 0.0;
        grad[6] = 0.0;
        grad[7] = 0.0;
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = -0.0025*x[3] - 0.0025*x[5] + 1.0;
        g[1] = 0.0025*x[3] - 0.0025*x[4] - 0.0025*x[6] + 1.0;
        g[2] = 0.01*x[4] - 0.01*x[7] + 1.0;
        g[3] = -100.0*x[0] - 833.33252*x[3] + x[0]*x[5] + 83333.333;
        g[4] = -x[1]*x[3] + x[1]*x[6] + 1250.0*x[3] - 1250.0*x[4];
        g[5] = 2500.0*x[4] - x[2]*x[4] + x[2]*x[7] - 1250000.0;
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5], vec![3, 5, 3, 4, 6, 4, 7, 0, 3, 5, 1, 3, 4, 6, 2, 4, 7])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = -0.00250000000000000;
        vals[1] = -0.00250000000000000;
        vals[2] = 0.00250000000000000;
        vals[3] = -0.00250000000000000;
        vals[4] = -0.00250000000000000;
        vals[5] = 0.0100000000000000;
        vals[6] = -0.0100000000000000;
        vals[7] = x[5] - 100.0;
        vals[8] = -833.332520000000;
        vals[9] = x[0];
        vals[10] = -x[3] + x[6];
        vals[11] = -x[1] + 1250.0;
        vals[12] = -1250.00000000000;
        vals[13] = x[1];
        vals[14] = -x[4] + x[7];
        vals[15] = -x[2] + 2500.0;
        vals[16] = x[2];
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![3, 4, 5, 6, 7], vec![1, 2, 0, 1, 2])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = lambda[4] * (-1.0);
        vals[1] = lambda[5] * (-1.0);
        vals[2] = lambda[3] * (1.0);
        vals[3] = lambda[4] * (1.0);
        vals[4] = lambda[5] * (1.0);
    }
}

pub struct HsTp113;

impl NlpProblem for HsTp113 {
    fn num_variables(&self) -> usize {
        10
    }

    fn num_constraints(&self) -> usize {
        8
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY;
        x_u[1] = f64::INFINITY;
        x_l[2] = f64::NEG_INFINITY;
        x_u[2] = f64::INFINITY;
        x_l[3] = f64::NEG_INFINITY;
        x_u[3] = f64::INFINITY;
        x_l[4] = f64::NEG_INFINITY;
        x_u[4] = f64::INFINITY;
        x_l[5] = f64::NEG_INFINITY;
        x_u[5] = f64::INFINITY;
        x_l[6] = f64::NEG_INFINITY;
        x_u[6] = f64::INFINITY;
        x_l[7] = f64::NEG_INFINITY;
        x_u[7] = f64::INFINITY;
        x_l[8] = f64::NEG_INFINITY;
        x_u[8] = f64::INFINITY;
        x_l[9] = f64::NEG_INFINITY;
        x_u[9] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = f64::INFINITY;
        g_l[1] = 0.0;
        g_u[1] = f64::INFINITY;
        g_l[2] = 0.0;
        g_u[2] = f64::INFINITY;
        g_l[3] = 0.0;
        g_u[3] = f64::INFINITY;
        g_l[4] = 0.0;
        g_u[4] = f64::INFINITY;
        g_l[5] = 0.0;
        g_u[5] = f64::INFINITY;
        g_l[6] = 0.0;
        g_u[6] = f64::INFINITY;
        g_l[7] = 0.0;
        g_u[7] = f64::INFINITY;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 2.0;
        x0[1] = 3.0;
        x0[2] = 5.0;
        x0[3] = 5.0;
        x0[4] = 1.0;
        x0[5] = 2.0;
        x0[6] = 7.0;
        x0[7] = 3.0;
        x0[8] = 6.0;
        x0[9] = 10.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        -14.0*x[0] - 16.0*x[1] - 20.0*x[2] - 40.0*x[3] + 4.0*x[3].powi(2) - 6.0*x[4] - 4.0*x[5] + 2.0*x[5].powi(2) + 5.0*x[6].powi(2) - 154.0*x[7] + 7.0*x[7].powi(2) - 40.0*x[8] + 2.0*x[8].powi(2) - 14.0*x[9] + x[0]*x[1] + 1352.0 + x[0].powi(2) + x[1].powi(2) + x[2].powi(2) + x[4].powi(2) + x[9].powi(2)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0*x[0] + x[1] - 14.0;
        grad[1] = x[0] + 2.0*x[1] - 16.0;
        grad[2] = 2.0*x[2] - 20.0;
        grad[3] = 8.0*x[3] - 40.0;
        grad[4] = 2.0*x[4] - 6.0;
        grad[5] = 4.0*x[5] - 4.0;
        grad[6] = 10.0*x[6];
        grad[7] = 14.0*x[7] - 154.0;
        grad[8] = 4.0*x[8] - 40.0;
        grad[9] = 2.0*x[9] - 14.0;
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = -4.0*x[0] - 5.0*x[1] + 3.0*x[6] - 9.0*x[7] + 105.0;
        g[1] = -10.0*x[0] + 8.0*x[1] + 17.0*x[6] - 2.0*x[7];
        g[2] = 8.0*x[0] - 2.0*x[1] - 5.0*x[8] + 2.0*x[9] + 12.0;
        g[3] = 12.0*x[0] - 3.0*x[0].powi(2) + 24.0*x[1] - 4.0*x[1].powi(2) - 2.0*x[2].powi(2) + 7.0*x[3] + 72.0;
        g[4] = -5.0*x[0].powi(2) - 8.0*x[1] + 12.0*x[2] - x[2].powi(2) + 2.0*x[3] + 4.0;
        g[5] = 8.0*x[0] - 0.5*x[0].powi(2) + 16.0*x[1] - 2.0*x[1].powi(2) - 3.0*x[4].powi(2) + x[5] - 34.0;
        g[6] = -x[0].powi(2) + 8.0*x[1] - 2.0*x[1].powi(2) - 14.0*x[4] + 6.0*x[5] + 2.0*x[0]*x[1] - 8.0;
        g[7] = 3.0*x[0] - 6.0*x[1] + 192.0*x[8] - 12.0*x[8].powi(2) + 7.0*x[9] - 768.0;
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7], vec![0, 1, 6, 7, 0, 1, 6, 7, 0, 1, 8, 9, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 4, 5, 0, 1, 4, 5, 0, 1, 8, 9])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = -4.00000000000000;
        vals[1] = -5.00000000000000;
        vals[2] = 3.00000000000000;
        vals[3] = -9.00000000000000;
        vals[4] = -10.0000000000000;
        vals[5] = 8.00000000000000;
        vals[6] = 17.0000000000000;
        vals[7] = -2.00000000000000;
        vals[8] = 8.00000000000000;
        vals[9] = -2.00000000000000;
        vals[10] = -5.00000000000000;
        vals[11] = 2.00000000000000;
        vals[12] = -6.0*x[0] + 12.0;
        vals[13] = -8.0*x[1] + 24.0;
        vals[14] = -4.0*x[2];
        vals[15] = 7.00000000000000;
        vals[16] = -10.0*x[0];
        vals[17] = -8.00000000000000;
        vals[18] = -2.0*x[2] + 12.0;
        vals[19] = 2.00000000000000;
        vals[20] = -1.0*x[0] + 8.0;
        vals[21] = -4.0*x[1] + 16.0;
        vals[22] = -6.0*x[4];
        vals[23] = 1.0;
        vals[24] = -2.0*x[0] + 2.0*x[1];
        vals[25] = 2.0*x[0] - 4.0*x[1] + 8.0;
        vals[26] = -14.0000000000000;
        vals[27] = 6.00000000000000;
        vals[28] = 3.00000000000000;
        vals[29] = -6.00000000000000;
        vals[30] = -24.0*x[8] + 192.0;
        vals[31] = 7.00000000000000;
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9], vec![0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (2.0) + lambda[3] * (-6.00000000000000) + lambda[4] * (-10.0000000000000) + lambda[5] * (-1.00000000000000) + lambda[6] * (-2.0);
        vals[1] = obj_factor * (1.0) + lambda[6] * (2.00000000000000);
        vals[2] = obj_factor * (2.0) + lambda[3] * (-8.00000000000000) + lambda[5] * (-4.00000000000000) + lambda[6] * (-4.00000000000000);
        vals[3] = obj_factor * (2.0) + lambda[3] * (-4.00000000000000) + lambda[4] * (-2.0);
        vals[4] = obj_factor * (8.00000000000000);
        vals[5] = obj_factor * (2.0) + lambda[5] * (-6.00000000000000);
        vals[6] = obj_factor * (4.00000000000000);
        vals[7] = obj_factor * (10.0000000000000);
        vals[8] = obj_factor * (14.0000000000000);
        vals[9] = obj_factor * (4.00000000000000) + lambda[7] * (-24.0000000000000);
        vals[10] = obj_factor * (2.0);
    }
}

pub struct HsTp116;

impl NlpProblem for HsTp116 {
    fn num_variables(&self) -> usize {
        13
    }

    fn num_constraints(&self) -> usize {
        15
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 0.1;
        x_u[0] = 1.0;
        x_l[1] = 0.1;
        x_u[1] = 1.0;
        x_l[2] = 0.1;
        x_u[2] = 1.0;
        x_l[3] = 0.0001;
        x_u[3] = 0.1;
        x_l[4] = 0.1;
        x_u[4] = 0.9;
        x_l[5] = 0.1;
        x_u[5] = 0.9;
        x_l[6] = 0.1;
        x_u[6] = 1000.0;
        x_l[7] = 0.1;
        x_u[7] = 1000.0;
        x_l[8] = 500.0;
        x_u[8] = 1000.0;
        x_l[9] = 0.1;
        x_u[9] = 500.0;
        x_l[10] = 1.0;
        x_u[10] = 150.0;
        x_l[11] = 0.0001;
        x_u[11] = 150.0;
        x_l[12] = 0.0001;
        x_u[12] = 150.0;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = f64::INFINITY;
        g_l[1] = 0.0;
        g_u[1] = f64::INFINITY;
        g_l[2] = 0.0;
        g_u[2] = f64::INFINITY;
        g_l[3] = 0.0;
        g_u[3] = f64::INFINITY;
        g_l[4] = 0.0;
        g_u[4] = f64::INFINITY;
        g_l[5] = 0.0;
        g_u[5] = f64::INFINITY;
        g_l[6] = 0.0;
        g_u[6] = f64::INFINITY;
        g_l[7] = 0.0;
        g_u[7] = f64::INFINITY;
        g_l[8] = 0.0;
        g_u[8] = f64::INFINITY;
        g_l[9] = 0.0;
        g_u[9] = f64::INFINITY;
        g_l[10] = 0.0;
        g_u[10] = f64::INFINITY;
        g_l[11] = 0.0;
        g_u[11] = f64::INFINITY;
        g_l[12] = 0.0;
        g_u[12] = f64::INFINITY;
        g_l[13] = 0.0;
        g_u[13] = f64::INFINITY;
        g_l[14] = 0.0;
        g_u[14] = f64::INFINITY;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.5;
        x0[1] = 0.8;
        x0[2] = 0.9;
        x0[3] = 0.1;
        x0[4] = 0.14;
        x0[5] = 0.5;
        x0[6] = 489.0;
        x0[7] = 80.0;
        x0[8] = 650.0;
        x0[9] = 450.0;
        x0[10] = 150.0;
        x0[11] = 150.0;
        x0[12] = 150.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        x[10] + x[11] + x[12]
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 0.0;
        grad[1] = 0.0;
        grad[2] = 0.0;
        grad[3] = 0.0;
        grad[4] = 0.0;
        grad[5] = 0.0;
        grad[6] = 0.0;
        grad[7] = 0.0;
        grad[8] = 0.0;
        grad[9] = 0.0;
        grad[10] = 1.0;
        grad[11] = 1.0;
        grad[12] = 1.0;
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = -x[1] + x[2];
        g[1] = -x[0] + x[1];
        g[2] = -0.002*x[6] + 0.002*x[7] + 1.0;
        g[3] = x[10] + x[11] + x[12] - 50.0;
        g[4] = -x[10] - x[11] - x[12] + 250.0;
        g[5] = x[12] + 1.231059*x[2]*x[9] - 1.262626*x[9];
        g[6] = 0.00975*x[1].powi(2) - 0.975*x[1]*x[4] - 0.03475*x[1] + x[4];
        g[7] = 0.00975*x[2].powi(2) - 0.975*x[2]*x[5] - 0.03475*x[2] + x[5];
        g[8] = -x[0]*x[7] - x[3]*x[6] + x[3]*x[7] + x[4]*x[6];
        g[9] = -x[4] - x[5] + 0.002*x[0]*x[7] - 0.002*x[1]*x[8] - 0.002*x[4]*x[7] + 0.002*x[5]*x[8] + 1.0;
        g[10] = x[1]*x[8] + x[1]*x[9] - 500.0*x[1] - x[2]*x[9] - x[5]*x[8] + 500.0*x[5];
        g[11] = x[1] - 0.002*x[1]*x[9] + 0.002*x[2]*x[9] - 0.9;
        g[12] = 0.00975*x[0].powi(2) - 0.975*x[0]*x[3] - 0.03475*x[0] + x[3];
        g[13] = 1.231059*x[0]*x[7] + x[10] - 1.262626*x[7];
        g[14] = 1.231059*x[1]*x[8] + x[11] - 1.262626*x[8];
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 14], vec![1, 2, 0, 1, 6, 7, 10, 11, 12, 10, 11, 12, 2, 9, 12, 1, 4, 2, 5, 0, 3, 4, 6, 7, 0, 1, 4, 5, 7, 8, 1, 2, 5, 8, 9, 1, 2, 9, 0, 3, 0, 7, 10, 1, 8, 11])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = -1.0;
        vals[1] = 1.0;
        vals[2] = -1.0;
        vals[3] = 1.0;
        vals[4] = -0.00200000000000000;
        vals[5] = 0.00200000000000000;
        vals[6] = 1.0;
        vals[7] = 1.0;
        vals[8] = 1.0;
        vals[9] = -1.0;
        vals[10] = -1.0;
        vals[11] = -1.0;
        vals[12] = 1.231059*x[9];
        vals[13] = 1.231059*x[2] - 1.262626;
        vals[14] = 1.0;
        vals[15] = 0.0195*x[1] - 0.975*x[4] - 0.03475;
        vals[16] = 1.0 - 0.975*x[1];
        vals[17] = 0.0195*x[2] - 0.975*x[5] - 0.03475;
        vals[18] = 1.0 - 0.975*x[2];
        vals[19] = -x[7];
        vals[20] = -x[6] + x[7];
        vals[21] = x[6];
        vals[22] = -x[3] + x[4];
        vals[23] = -x[0] + x[3];
        vals[24] = 0.002*x[7];
        vals[25] = -0.002*x[8];
        vals[26] = -0.002*x[7] - 1.0;
        vals[27] = 0.002*x[8] - 1.0;
        vals[28] = 0.002*x[0] - 0.002*x[4];
        vals[29] = -0.002*x[1] + 0.002*x[5];
        vals[30] = x[8] + x[9] - 500.0;
        vals[31] = -x[9];
        vals[32] = -x[8] + 500.0;
        vals[33] = x[1] - x[5];
        vals[34] = x[1] - x[2];
        vals[35] = 1.0 - 0.002*x[9];
        vals[36] = 0.002*x[9];
        vals[37] = -0.002*x[1] + 0.002*x[2];
        vals[38] = 0.0195*x[0] - 0.975*x[3] - 0.03475;
        vals[39] = 1.0 - 0.975*x[0];
        vals[40] = 1.231059*x[7];
        vals[41] = 1.231059*x[0] - 1.262626;
        vals[42] = 1.0;
        vals[43] = 1.231059*x[8];
        vals[44] = 1.231059*x[1] - 1.262626;
        vals[45] = 1.0;
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1, 2, 3, 4, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9], vec![0, 1, 2, 0, 1, 2, 3, 4, 0, 3, 4, 1, 5, 1, 2])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = lambda[12] * (0.0195000000000000);
        vals[1] = lambda[6] * (0.0195000000000000);
        vals[2] = lambda[7] * (0.0195000000000000);
        vals[3] = lambda[12] * (-0.975000000000000);
        vals[4] = lambda[6] * (-0.975000000000000);
        vals[5] = lambda[7] * (-0.975000000000000);
        vals[6] = lambda[8] * (-1.0);
        vals[7] = lambda[8] * (1.0);
        vals[8] = lambda[8] * (-1.0) + lambda[9] * (0.00200000000000000) + lambda[13] * (1.23105900000000);
        vals[9] = lambda[8] * (1.0);
        vals[10] = lambda[9] * (-0.00200000000000000);
        vals[11] = lambda[9] * (-0.00200000000000000) + lambda[10] * (1.0) + lambda[14] * (1.23105900000000);
        vals[12] = lambda[9] * (0.00200000000000000) + lambda[10] * (-1.0);
        vals[13] = lambda[10] * (1.0) + lambda[11] * (-0.00200000000000000);
        vals[14] = lambda[5] * (1.23105900000000) + lambda[10] * (-1.0) + lambda[11] * (0.00200000000000000);
    }
}

pub struct HsTp201;

impl NlpProblem for HsTp201 {
    fn num_variables(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        0
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY;
        x_u[1] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {

    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 8.0;
        x0[1] = 9.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        -40.0*x[0] + 4.0*x[0].powi(2) - 12.0*x[1] + 136.0 + x[1].powi(2)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 8.0*x[0] - 40.0;
        grad[1] = 2.0*x[1] - 12.0;
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let _ = (x, g);
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        let _ = (x, vals);
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (8.00000000000000);
        vals[1] = obj_factor * (2.0);
    }
}

pub struct HsTp325;

impl NlpProblem for HsTp325 {
    fn num_variables(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        3
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = f64::NEG_INFINITY;
        x_u[0] = f64::INFINITY;
        x_l[1] = f64::NEG_INFINITY;
        x_u[1] = f64::INFINITY;
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0;
        g_u[0] = f64::INFINITY;
        g_l[1] = 0.0;
        g_u[1] = f64::INFINITY;
        g_l[2] = 0.0;
        g_u[2] = 0.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.0;
        x0[1] = 0.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        x[0].powi(2) + x[1]
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2.0*x[0];
        grad[1] = 1.0;
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = -x[0] - x[1] + 1.0;
        g[1] = -x[0] - x[1].powi(2) + 1.0;
        g[2] = -9.0 + x[0].powi(2) + x[1].powi(2);
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 1, 1, 2, 2], vec![0, 1, 0, 1, 0, 1])
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = -1.0;
        vals[1] = -1.0;
        vals[2] = -1.0;
        vals[3] = -2.0*x[1];
        vals[4] = 2.0*x[0];
        vals[5] = 2.0*x[1];
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0, 1], vec![0, 1])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        vals[0] = obj_factor * (2.0) + lambda[2] * (2.0);
        vals[1] = lambda[1] * (-2.0) + lambda[2] * (2.0);
    }
}

pub struct HsTp374;

// Helper functions for TP374 trigonometric constraints (from Fortran TP374A, TP374B, TP374G)
fn tp374_a(z: f64, x: &[f64]) -> f64 {
    let mut val = 0.0;
    for k in 1..=9 {
        val += x[k - 1] * (k as f64 * z).cos();
    }
    val
}

fn tp374_b(z: f64, x: &[f64]) -> f64 {
    let mut val = 0.0;
    for k in 1..=9 {
        val += x[k - 1] * (k as f64 * z).sin();
    }
    val
}

fn tp374_gfn(z: f64, x: &[f64]) -> f64 {
    let a = tp374_a(z, x);
    let b = tp374_b(z, x);
    a * a + b * b
}

impl NlpProblem for HsTp374 {
    fn num_variables(&self) -> usize {
        10
    }

    fn num_constraints(&self) -> usize {
        35
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..10 {
            x_l[i] = f64::NEG_INFINITY;
            x_u[i] = f64::INFINITY;
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for i in 0..35 {
            g_l[i] = 0.0;
            g_u[i] = f64::INFINITY;
        }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        for i in 0..10 {
            x0[i] = 0.1;
        }
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        x[9]
    }

    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) {
        for i in 0..9 { grad[i] = 0.0; }
        grad[9] = 1.0;
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        use std::f64::consts::PI;
        // Constraints 0..9: G(z_i, x) - (1 - x[9])^2 >= 0
        for i in 0..10 {
            let z = PI / 4.0 * (i as f64 * 0.1);
            g[i] = tp374_gfn(z, x) - (1.0 - x[9]).powi(2);
        }
        // Constraints 10..19: (1 + x[9])^2 - G(z_i, x) >= 0
        for i in 10..20 {
            let z = PI / 4.0 * ((i - 10) as f64 * 0.1);
            g[i] = (1.0 + x[9]).powi(2) - tp374_gfn(z, x);
        }
        // Constraints 20..34: x[9]^2 - G(z_i, x) >= 0
        for i in 20..35 {
            let z = PI / 4.0 * (1.2 + (i - 20) as f64 * 0.2);
            g[i] = x[9].powi(2) - tp374_gfn(z, x);
        }
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Each of the 35 constraints depends on all 10 variables (dense)
        let mut rows = Vec::with_capacity(350);
        let mut cols = Vec::with_capacity(350);
        for i in 0..35 {
            for j in 0..10 {
                rows.push(i);
                cols.push(j);
            }
        }
        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        use std::f64::consts::PI;
        let mut idx = 0;

        // Constraints 0..9: g_i = G(z, x) - (1 - x[9])^2
        for i in 0..10 {
            let z = PI / 4.0 * (i as f64 * 0.1);
            let a = tp374_a(z, x);
            let b = tp374_b(z, x);
            for k in 1..=9 {
                vals[idx] = 2.0 * (a * (k as f64 * z).cos() + b * (k as f64 * z).sin());
                idx += 1;
            }
            vals[idx] = 2.0 * (1.0 - x[9]);
            idx += 1;
        }

        // Constraints 10..19: g_i = (1 + x[9])^2 - G(z, x)
        for i in 10..20 {
            let z = PI / 4.0 * ((i - 10) as f64 * 0.1);
            let a = tp374_a(z, x);
            let b = tp374_b(z, x);
            for k in 1..=9 {
                vals[idx] = -2.0 * (a * (k as f64 * z).cos() + b * (k as f64 * z).sin());
                idx += 1;
            }
            vals[idx] = 2.0 * (1.0 + x[9]);
            idx += 1;
        }

        // Constraints 20..34: g_i = x[9]^2 - G(z, x)
        for i in 20..35 {
            let z = PI / 4.0 * (1.2 + (i - 20) as f64 * 0.2);
            let a = tp374_a(z, x);
            let b = tp374_b(z, x);
            for k in 1..=9 {
                vals[idx] = -2.0 * (a * (k as f64 * z).cos() + b * (k as f64 * z).sin());
                idx += 1;
            }
            vals[idx] = 2.0 * x[9];
            idx += 1;
        }
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // 10x10 lower triangle
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for i in 0..10 {
            for j in 0..=i {
                rows.push(i);
                cols.push(j);
            }
        }
        (rows, cols)
    }

    fn hessian_values(&self, _x: &[f64], _new_x: bool, _obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        use std::f64::consts::PI;
        // Objective Hessian is zero (f = x[9] is linear).
        for v in vals.iter_mut() { *v = 0.0; }

        // Lower-triangle index: (i,j) -> i*(i+1)/2 + j
        let lt = |i: usize, j: usize| -> usize { i * (i + 1) / 2 + j };

        // Group 1 (constraints 0..9): g = A^2 + B^2 - (1-x9)^2
        //   d2g/dx[j-1]dx[k-1] = 2*cos((j-k)*z) for j,k in 1..9
        //   d2g/dx9^2 = 2
        for ci in 0..10 {
            let lam = lambda[ci];
            if lam == 0.0 { continue; }
            let z = PI / 4.0 * (ci as f64 * 0.1);
            for j in 1..=9usize {
                for k in 1..=j {
                    let h = 2.0 * ((j as f64 - k as f64) * z).cos();
                    vals[lt(j - 1, k - 1)] += lam * h;
                }
            }
            vals[lt(9, 9)] += lam * 2.0;
        }

        // Group 2 (constraints 10..19): g = (1+x9)^2 - A^2 - B^2
        //   d2g/dx[j-1]dx[k-1] = -2*cos((j-k)*z)
        //   d2g/dx9^2 = 2
        for ci in 10..20 {
            let lam = lambda[ci];
            if lam == 0.0 { continue; }
            let z = PI / 4.0 * ((ci - 10) as f64 * 0.1);
            for j in 1..=9usize {
                for k in 1..=j {
                    let h = -2.0 * ((j as f64 - k as f64) * z).cos();
                    vals[lt(j - 1, k - 1)] += lam * h;
                }
            }
            vals[lt(9, 9)] += lam * 2.0;
        }

        // Group 3 (constraints 20..34): g = x9^2 - A^2 - B^2
        //   d2g/dx[j-1]dx[k-1] = -2*cos((j-k)*z)
        //   d2g/dx9^2 = 2
        for ci in 20..35 {
            let lam = lambda[ci];
            if lam == 0.0 { continue; }
            let z = PI / 4.0 * (1.2 + (ci - 20) as f64 * 0.2);
            for j in 1..=9usize {
                for k in 1..=j {
                    let h = -2.0 * ((j as f64 - k as f64) * z).cos();
                    vals[lt(j - 1, k - 1)] += lam * h;
                }
            }
            vals[lt(9, 9)] += lam * 2.0;
        }
    }
}
