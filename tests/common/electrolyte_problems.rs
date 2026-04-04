// Electrolyte thermodynamics test problems for ripopt.
//
// 13 problems capturing deep nonlinearities from activity coefficients:
// sqrt(I) singularities (Debye-Huckel), exp() terms (Pitzer/NRTL),
// extreme variable scaling, and tightly coupled constraints.

use ripopt::NlpProblem;

// ===========================================================================
// Physical constants at 25°C (298.15 K)
// ===========================================================================

const A_DH: f64 = 0.5091; // Debye-Huckel A parameter
const B_DH: f64 = 0.3283; // DH B parameter, 1/(Å·sqrt(mol/kg))
const A_PHI: f64 = 0.3915; // Pitzer osmotic coefficient A
const LN_KW: f64 = -32.2387; // ln(Kw) where Kw = 1.012e-14
const LN10: f64 = 2.302_585_093;
const M_W: f64 = 0.018_015; // molar mass of water, kg/mol

// ===========================================================================
// Helper functions: Extended Debye-Huckel (Truesdell-Jones)
// ===========================================================================

/// ln(gamma_i) = -A*z^2*sqrt(I)/(1+a*B*sqrt(I)) + b_dot*I
fn ln_gamma_dh(z: f64, a: f64, b_dot: f64, ionic_strength: f64) -> f64 {
    if ionic_strength <= 0.0 {
        return 0.0;
    }
    let si = ionic_strength.sqrt();
    let denom = 1.0 + a * B_DH * si;
    -A_DH * z * z * si / denom + b_dot * ionic_strength
}

/// d(ln_gamma)/dI
fn d_ln_gamma_dh_di(z: f64, a: f64, b_dot: f64, ionic_strength: f64) -> f64 {
    if ionic_strength <= 1e-30 {
        return -A_DH * z * z * 1e15 * 0.5; // limit form
    }
    let si = ionic_strength.sqrt();
    let ab = a * B_DH;
    let denom = 1.0 + ab * si;
    // d/dI[-A*z^2*sqrt(I)/(1+ab*sqrt(I))] + b_dot
    // = -A*z^2 * [1/(2*sqrt(I)) * (1+ab*sqrt(I)) - sqrt(I)*ab/(2*sqrt(I))] / denom^2
    // = -A*z^2 / (2*sqrt(I)) * [1+ab*sqrt(I) - ab*sqrt(I)] / denom^2
    // = -A*z^2 / (2*sqrt(I)*denom^2)
    -A_DH * z * z / (2.0 * si * denom * denom) + b_dot
}

/// d²(ln_gamma)/dI²
fn d2_ln_gamma_dh_di2(z: f64, a: f64, _b_dot: f64, ionic_strength: f64) -> f64 {
    if ionic_strength <= 1e-30 {
        return 0.0;
    }
    let si = ionic_strength.sqrt();
    let ab = a * B_DH;
    let denom = 1.0 + ab * si;
    // d/dI[-A*z^2/(2*sqrt(I)*denom^2)]
    // Let u = I^{-1/2}, v = denom^{-2}
    // d(u*v)/dI = u'*v + u*v'
    // u' = -1/(2*I^{3/2})
    // v = (1+ab*I^{1/2})^{-2}, v' = -2*(1+ab*I^{1/2})^{-3} * ab/(2*I^{1/2})
    // = -ab / (I^{1/2} * denom^3)
    let term1 = 1.0 / (4.0 * ionic_strength * si * denom * denom);
    let term2 = ab / (2.0 * ionic_strength * denom * denom * denom);
    A_DH * z * z * (term1 + term2)
}

// ===========================================================================
// Helper functions: Pitzer model for 1:1 electrolyte
// ===========================================================================

/// Mean ionic activity coefficient ln(gamma_pm) for 1:1 salt at molality m.
/// I = m for a 1:1 salt.
fn pitzer_ln_gamma_pm(m: f64, beta0: f64, beta1: f64, c_phi: f64) -> f64 {
    if m <= 0.0 {
        return 0.0;
    }
    let si = m.sqrt();
    // Debye-Huckel term
    let f_gamma = -A_PHI * (si / (1.0 + 1.2 * si) + (2.0 / 1.2) * (1.0 + 1.2 * si).ln());
    // B_gamma term
    let x = 2.0 * si;
    let ex = (-x).exp();
    let b_gamma = 2.0 * beta0 + 2.0 * beta1 / (4.0 * m) * (1.0 - (1.0 + x) * ex);
    // C_gamma
    let c_gamma = 1.5 * c_phi;
    f_gamma + m * b_gamma + m * m * c_gamma
}

/// d(ln_gamma_pm)/dm for 1:1 salt.
fn d_pitzer_ln_gamma_pm_dm(m: f64, beta0: f64, beta1: f64, c_phi: f64) -> f64 {
    if m <= 1e-30 {
        return 0.0;
    }
    let si = m.sqrt();
    // df_gamma/dm
    let d_si = 0.5 / si;
    let denom = 1.0 + 1.2 * si;
    let df_gamma = -A_PHI * (d_si / denom - 1.2 * d_si * si / (denom * denom)
        + (2.0 / 1.2) * 1.2 * d_si / denom);
    // Actually let me recompute more carefully:
    // f_gamma = -A_PHI * [sqrt(m)/(1+1.2*sqrt(m)) + (2/1.2)*ln(1+1.2*sqrt(m))]
    // df/dm = -A_PHI * [d(sqrt(m)/(1+1.2*sqrt(m)))/dm + (2/1.2)*1.2*d_si/denom]
    // d(si/denom)/dm = (d_si*denom - si*1.2*d_si)/denom^2 = d_si*(denom-1.2*si)/denom^2
    //                = d_si * 1/denom^2 = 0.5/(si*denom^2)
    let df_gamma2 = -A_PHI * (0.5 / (si * denom * denom) + (2.0 / 1.2) * 1.2 * 0.5 / (si * denom));
    // = -A_PHI * (0.5/(si*denom^2) + 1/(si*denom))
    let _ = df_gamma; // use the corrected version
    let df_gamma = -A_PHI * (0.5 / (si * denom * denom) + 1.0 / (si * denom));

    // B_gamma(m) * m where B_gamma = 2*beta0 + 2*beta1/(4m) * [1-(1+2*si)*exp(-2*si)]
    // d(m*B_gamma)/dm:
    let x = 2.0 * si;
    let ex = (-x).exp();
    // m*B_gamma = 2*beta0*m + beta1/2 * [1-(1+2*si)*exp(-2*si)]
    // d/dm = 2*beta0 + beta1/2 * d/dm[1-(1+x)*e^{-x}]  where x=2*sqrt(m)
    // d/dm[(1+x)*e^{-x}] = (dx/dm)*e^{-x} + (1+x)*(-dx/dm)*e^{-x}
    //                     = dx/dm * e^{-x} * (1 - 1 - x) = -x*dx/dm*e^{-x}
    // dx/dm = 1/sqrt(m) = 2/x * ... actually dx/dm = 2 * 0.5/sqrt(m) = 1/sqrt(m)
    let dx_dm = 1.0 / si;
    let d_mb = 2.0 * beta0 + beta1 / 2.0 * x * dx_dm * ex;
    // = 2*beta0 + beta1/2 * 2*sqrt(m) / sqrt(m) * exp(-2*sqrt(m))
    // = 2*beta0 + beta1 * exp(-2*sqrt(m))

    // C term: d(m^2 * 1.5*c_phi)/dm = 3*c_phi*m
    let dc = 3.0 * c_phi * m;

    let _ = df_gamma2;
    df_gamma + d_mb + dc
}

/// Pitzer osmotic coefficient phi for 1:1 salt at molality m.
fn pitzer_osmotic(m: f64, beta0: f64, beta1: f64, c_phi: f64) -> f64 {
    if m <= 0.0 {
        return 1.0;
    }
    let si = m.sqrt();
    1.0 - A_PHI * si / (1.0 + 1.2 * si)
        + m * (beta0 + beta1 * (-2.0 * si).exp())
        + m * m * c_phi
}

/// d(phi)/dm
fn d_pitzer_osmotic_dm(m: f64, beta0: f64, beta1: f64, c_phi: f64) -> f64 {
    if m <= 1e-30 {
        return 0.0;
    }
    let si = m.sqrt();
    let denom = 1.0 + 1.2 * si;
    // d/dm[-A_PHI*si/denom] = -A_PHI*(0.5/si * denom - si*1.2*0.5/si)/(denom^2)
    // = -A_PHI * (0.5/(si) - 0.6) / denom^2  ... no, let me redo
    // = -A_PHI * d(si/denom)/dm = -A_PHI * 0.5/(si*denom^2)
    let d_dh = -A_PHI * 0.5 / (si * denom * denom);

    let ex = (-2.0 * si).exp();
    // d/dm[m*(beta0 + beta1*exp(-2*si))]
    // = beta0 + beta1*exp(-2*si) + m*beta1*(-2*0.5/si)*exp(-2*si)
    // = beta0 + beta1*exp(-2*si) - beta1*si*exp(-2*si)  ... wait
    // = beta0 + beta1*exp(-2*si) + m*beta1*(-1/si)*exp(-2*si)
    // = beta0 + beta1*exp(-2*si)*(1 - m/si) = beta0 + beta1*exp(-2*si)*(1-si)
    // since m/si = si
    let d_b = beta0 + beta1 * ex * (1.0 - si);

    let d_c = 2.0 * m * c_phi;

    d_dh + d_b + d_c
}

/// Partial derivatives of pitzer_osmotic w.r.t. parameters.
/// Returns (d_phi/d_beta0, d_phi/d_beta1, d_phi/d_c_phi)
fn pitzer_osmotic_partials(m: f64, _beta0: f64, _beta1: f64, _c_phi: f64) -> (f64, f64, f64) {
    if m <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let si = m.sqrt();
    (m, m * (-2.0 * si).exp(), m * m)
}

// ===========================================================================
// NRTL helpers for Problem 8
// ===========================================================================

/// NRTL activity coefficients for a binary system.
/// Returns (ln_gamma_1, ln_gamma_2) given mole fraction x1.
fn nrtl_binary(x1: f64, tau12: f64, tau21: f64, alpha: f64) -> (f64, f64) {
    let x2 = 1.0 - x1;
    let g12 = (-alpha * tau12).exp();
    let g21 = (-alpha * tau21).exp();
    let a1 = x2 * g21 / (x1 + x2 * g21);
    let a2 = x1 * g12 / (x2 + x1 * g12);
    let ln_g1 = x2 * x2 * (tau21 * (g21 / (x1 + x2 * g21)).powi(2)
        + tau12 * g12 / (x2 + x1 * g12).powi(2));
    let ln_g2 = x1 * x1 * (tau12 * (g12 / (x2 + x1 * g12)).powi(2)
        + tau21 * g21 / (x1 + x2 * g21).powi(2));
    let _ = (a1, a2); // suppress warnings
    (ln_g1, ln_g2)
}

/// d(ln_gamma_1)/d(x1) for NRTL binary.
fn d_nrtl_ln_gamma1_dx1(x1: f64, tau12: f64, tau21: f64, alpha: f64) -> f64 {
    // Numerical derivative for robustness
    let eps = 1e-8;
    let (g1p, _) = nrtl_binary(x1 + eps, tau12, tau21, alpha);
    let (g1m, _) = nrtl_binary(x1 - eps, tau12, tau21, alpha);
    (g1p - g1m) / (2.0 * eps)
}

fn d_nrtl_ln_gamma2_dx1(x1: f64, tau12: f64, tau21: f64, alpha: f64) -> f64 {
    let eps = 1e-8;
    let (_, g2p) = nrtl_binary(x1 + eps, tau12, tau21, alpha);
    let (_, g2m) = nrtl_binary(x1 - eps, tau12, tau21, alpha);
    (g2p - g2m) / (2.0 * eps)
}

// ===========================================================================
// Generic helpers for speciation problems
// ===========================================================================

/// Compute ionic strength from log-transformed molalities and charges.
fn ionic_strength(x: &[f64], charges: &[f64]) -> f64 {
    let mut i_s = 0.0;
    for (xi, &z) in x.iter().zip(charges.iter()) {
        if z != 0.0 {
            i_s += z * z * xi.exp();
        }
    }
    0.5 * i_s
}

/// Lower-triangle indices for dense n×n Hessian.
/// Returns (rows, cols) for entries (0,0), (1,0), (1,1), (2,0), ...
fn dense_lower_triangle(n: usize) -> (Vec<usize>, Vec<usize>) {
    let nnz = n * (n + 1) / 2;
    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);
    for i in 0..n {
        for j in 0..=i {
            rows.push(i);
            cols.push(j);
        }
    }
    (rows, cols)
}

/// Index into dense lower-triangle storage for entry (i, j) where i >= j.
#[inline]
fn lt_idx(i: usize, j: usize) -> usize {
    i * (i + 1) / 2 + j
}

// ===========================================================================
// Gibbs energy speciation problem: shared objective/gradient/hessian logic
// ===========================================================================

/// Evaluate Gibbs energy objective for a speciation problem.
/// f = sum_i exp(x[i]) * (mu0[i] + ln_gamma_i(I) + x[i])
fn gibbs_objective(
    x: &[f64],
    mu0: &[f64],
    charges: &[f64],
    dh_a: &[f64],
    dh_b: &[f64],
) -> f64 {
    let i_s = ionic_strength(x, charges);
    let mut f = 0.0;
    for i in 0..x.len() {
        let m_i = x[i].exp();
        let lg = ln_gamma_dh(charges[i], dh_a[i], dh_b[i], i_s);
        f += m_i * (mu0[i] + lg + x[i]);
    }
    f
}

/// Evaluate gradient of Gibbs energy objective.
fn gibbs_gradient(
    x: &[f64],
    mu0: &[f64],
    charges: &[f64],
    dh_a: &[f64],
    dh_b: &[f64],
    grad: &mut [f64],
) {
    let n = x.len();
    let i_s = ionic_strength(x, charges);

    // S1 = sum_i m_i * d(ln_gamma_i)/dI
    let mut s1 = 0.0;
    for i in 0..n {
        let m_i = x[i].exp();
        s1 += m_i * d_ln_gamma_dh_di(charges[i], dh_a[i], dh_b[i], i_s);
    }

    for j in 0..n {
        let m_j = x[j].exp();
        let lg_j = ln_gamma_dh(charges[j], dh_a[j], dh_b[j], i_s);
        let a_j = mu0[j] + lg_j + x[j] + 1.0;
        let di_dxj = 0.5 * charges[j] * charges[j] * m_j;
        grad[j] = m_j * a_j + di_dxj * s1;
    }
}

/// Evaluate Hessian of Gibbs energy objective (lower triangle, dense).
fn gibbs_hessian(
    x: &[f64],
    mu0: &[f64],
    charges: &[f64],
    dh_a: &[f64],
    dh_b: &[f64],
    obj_factor: f64,
    vals: &mut [f64],
) {
    let n = x.len();
    let i_s = ionic_strength(x, charges);

    let mut m = vec![0.0; n];
    let mut lg = vec![0.0; n];
    let mut dlg = vec![0.0; n];
    let mut d2lg = vec![0.0; n];

    let mut s1 = 0.0;
    let mut s2 = 0.0;
    for i in 0..n {
        m[i] = x[i].exp();
        lg[i] = ln_gamma_dh(charges[i], dh_a[i], dh_b[i], i_s);
        dlg[i] = d_ln_gamma_dh_di(charges[i], dh_a[i], dh_b[i], i_s);
        d2lg[i] = d2_ln_gamma_dh_di2(charges[i], dh_a[i], dh_b[i], i_s);
        s1 += m[i] * dlg[i];
        s2 += m[i] * d2lg[i];
    }

    for j in 0..n {
        let zj2 = charges[j] * charges[j];
        let mj = m[j];
        let a_j = mu0[j] + lg[j] + x[j] + 1.0;

        // Diagonal: d²f/dx_j²
        let h_jj = mj * (a_j + 1.0)
            + zj2 * mj * mj * dlg[j]
            + 0.5 * zj2 * mj * s1
            + 0.5 * zj2 * mj * mj * dlg[j]
            + 0.25 * zj2 * zj2 * mj * mj * s2;
        vals[lt_idx(j, j)] += obj_factor * h_jj;

        // Off-diagonal: d²f/(dx_j dx_k) for k < j
        for k in 0..j {
            let zk2 = charges[k] * charges[k];
            let mk = m[k];
            let h_jk = 0.5 * zk2 * mj * mk * dlg[j]
                + 0.5 * zj2 * mj * mk * dlg[k]
                + 0.25 * zj2 * zk2 * mj * mk * s2;
            vals[lt_idx(j, k)] += obj_factor * h_jk;
        }
    }
}

/// Add constraint Hessian for constraints of the form g = sum(a_i * exp(x_i)) - b.
/// Each such constraint contributes lambda * a_j * exp(x_j) on the diagonal.
fn add_constraint_hessian_exp(
    x: &[f64],
    lambda_val: f64,
    coeffs: &[f64],
    vals: &mut [f64],
) {
    for j in 0..x.len() {
        if coeffs[j] != 0.0 {
            vals[lt_idx(j, j)] += lambda_val * coeffs[j] * x[j].exp();
        }
    }
}

// ===========================================================================
// Problem 1: Water Autoionization (n=1, m=0)
// ===========================================================================

pub struct WaterAutoionization;

impl NlpProblem for WaterAutoionization {
    fn num_variables(&self) -> usize { 1 }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = (-1e-10_f64).ln(); // ln(1e-10) ~ -23
        x_u[0] = (-1e-3_f64).ln().abs().copysign(-1.0); // ln(1e-3) ~ -6.9
        // Fix: use explicit values
        x_l[0] = -23.0;
        x_u[0] = -6.9;
    }

    fn constraint_bounds(&self, _g_l: &mut [f64], _g_u: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = (1e-5_f64).ln(); // ~ -11.5
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        // x[0] = ln(m_H) = ln(m_OH), I = m_H (z=1 for both, equal conc)
        let m = x[0].exp();
        let i_s = m; // I = 0.5*(1^2*m + 1^2*m) = m
        let lg_h = ln_gamma_dh(1.0, 9.0, 0.0, i_s);
        let lg_oh = ln_gamma_dh(1.0, 3.5, 0.0, i_s);
        // f = m*(mu0_H + lg_H + ln(m)) + m*(mu0_OH + lg_OH + ln(m))
        // mu0_H + mu0_OH = -ln(Kw) so split: mu0_H = 0, mu0_OH = -LN_KW
        // f = m*(lg_H + x) + m*(-LN_KW + lg_OH + x)
        m * (lg_h + x[0]) + m * (-LN_KW + lg_oh + x[0])
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let m = x[0].exp();
        let i_s = m;
        let lg_h = ln_gamma_dh(1.0, 9.0, 0.0, i_s);
        let lg_oh = ln_gamma_dh(1.0, 3.5, 0.0, i_s);
        let dlg_h = d_ln_gamma_dh_di(1.0, 9.0, 0.0, i_s);
        let dlg_oh = d_ln_gamma_dh_di(1.0, 3.5, 0.0, i_s);
        // df/dx = df/dm * dm/dx = df/dm * m
        // f = m*(lg_h + x) + m*(-LN_KW + lg_oh + x)
        // df/dm = (lg_h + x + 1) + m*dlg_h + (-LN_KW + lg_oh + x + 1) + m*dlg_oh
        //       = (lg_h + lg_oh - LN_KW + 2*x + 2) + m*(dlg_h + dlg_oh)
        // (dI/dm = 1)
        let df_dm = (lg_h + lg_oh - LN_KW + 2.0 * x[0] + 2.0) + m * (dlg_h + dlg_oh);
        grad[0] = df_dm * m;
    }

    fn constraints(&self, _x: &[f64], _new_x: bool, _g: &mut [f64]) {}
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _x: &[f64], _new_x: bool, _vals: &mut [f64]) {}

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        // d²f/dx² = d/dx[m * df_dm] = m*df_dm + m*d(df_dm)/dx
        // d(df_dm)/dx = d(df_dm)/dm * m
        let m = x[0].exp();
        let i_s = m;
        let lg_h = ln_gamma_dh(1.0, 9.0, 0.0, i_s);
        let lg_oh = ln_gamma_dh(1.0, 3.5, 0.0, i_s);
        let dlg_h = d_ln_gamma_dh_di(1.0, 9.0, 0.0, i_s);
        let dlg_oh = d_ln_gamma_dh_di(1.0, 3.5, 0.0, i_s);
        let d2lg_h = d2_ln_gamma_dh_di2(1.0, 9.0, 0.0, i_s);
        let d2lg_oh = d2_ln_gamma_dh_di2(1.0, 3.5, 0.0, i_s);

        let df_dm = (lg_h + lg_oh - LN_KW + 2.0 * x[0] + 2.0) + m * (dlg_h + dlg_oh);
        // d(df_dm)/dm = (dlg_h + dlg_oh + 2/m) + (dlg_h + dlg_oh) + m*(d2lg_h + d2lg_oh)
        // Wait: d/dm of (lg_h + lg_oh - LN_KW + 2*x + 2) = dlg_h + dlg_oh + 2*dx/dm = dlg_h + dlg_oh + 2/m
        // d/dm of m*(dlg_h+dlg_oh) = (dlg_h+dlg_oh) + m*(d2lg_h+d2lg_oh)
        let d2f_dm2 = (dlg_h + dlg_oh) + 2.0 / m + (dlg_h + dlg_oh) + m * (d2lg_h + d2lg_oh);

        // d²f/dx² = m*df_dm + m^2*d2f_dm2
        vals[0] = obj_factor * (m * df_dm + m * m * d2f_dm2);
    }
}

// ===========================================================================
// Problem 2: CO2-Water Speciation (n=5, m=2)
// ===========================================================================

pub struct Co2WaterSpeciation;

impl Co2WaterSpeciation {
    const N: usize = 5;
    // Species: H2CO3, HCO3-, CO3^2-, H+, OH-
    const CHARGES: [f64; 5] = [0.0, -1.0, -2.0, 1.0, -1.0];
    const DH_A: [f64; 5] = [0.0, 4.0, 5.4, 9.0, 3.5];
    const DH_B: [f64; 5] = [0.0; 5];
    const PK1: f64 = 6.35;
    const PK2: f64 = 10.33;
    const PKW: f64 = 14.0;
    const C_TOTAL: f64 = 0.001;

    fn mu0() -> [f64; 5] {
        [0.0, Self::PK1 * LN10, (Self::PK1 + Self::PK2) * LN10, 0.0, Self::PKW * LN10]
    }
}

impl NlpProblem for Co2WaterSpeciation {
    fn num_variables(&self) -> usize { Self::N }
    fn num_constraints(&self) -> usize { 2 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..Self::N { x_l[i] = -35.0; x_u[i] = -2.0; }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0; g_u[0] = 0.0; // C balance
        g_l[1] = 0.0; g_u[1] = 0.0; // electroneutrality
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = (Self::C_TOTAL / 3.0).ln();
        x0[1] = (Self::C_TOTAL / 3.0).ln();
        x0[2] = (1e-8_f64).ln();
        x0[3] = (1e-6_f64).ln();
        x0[4] = (1e-8_f64).ln();
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        gibbs_objective(x, &Self::mu0(), &Self::CHARGES, &Self::DH_A, &Self::DH_B)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        gibbs_gradient(x, &Self::mu0(), &Self::CHARGES, &Self::DH_A, &Self::DH_B, grad);
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        // g[0]: C balance: m_H2CO3 + m_HCO3 + m_CO3 = C_TOTAL
        g[0] = x[0].exp() + x[1].exp() + x[2].exp() - Self::C_TOTAL;
        // g[1]: electroneutrality: -m_HCO3 - 2*m_CO3 + m_H - m_OH = 0
        g[1] = -x[1].exp() - 2.0 * x[2].exp() + x[3].exp() - x[4].exp();
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // g[0] depends on x[0,1,2], g[1] depends on x[1,2,3,4]
        let rows = vec![0, 0, 0, 1, 1, 1, 1];
        let cols = vec![0, 1, 2, 1, 2, 3, 4];
        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = x[0].exp();        // dg0/dx0
        vals[1] = x[1].exp();        // dg0/dx1
        vals[2] = x[2].exp();        // dg0/dx2
        vals[3] = -x[1].exp();       // dg1/dx1
        vals[4] = -2.0 * x[2].exp(); // dg1/dx2
        vals[5] = x[3].exp();        // dg1/dx3
        vals[6] = -x[4].exp();       // dg1/dx4
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        dense_lower_triangle(Self::N)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        for v in vals.iter_mut() { *v = 0.0; }
        gibbs_hessian(x, &Self::mu0(), &Self::CHARGES, &Self::DH_A, &Self::DH_B, obj_factor, vals);
        // Constraint 0: g0 = e^x0 + e^x1 + e^x2 - C
        // d²g0/dxj² = e^xj for j=0,1,2
        let coeffs0 = [1.0, 1.0, 1.0, 0.0, 0.0];
        add_constraint_hessian_exp(x, lambda[0], &coeffs0, vals);
        // Constraint 1: g1 = -e^x1 - 2*e^x2 + e^x3 - e^x4
        let coeffs1 = [0.0, -1.0, -2.0, 1.0, -1.0];
        add_constraint_hessian_exp(x, lambda[1], &coeffs1, vals);
    }
}

// ===========================================================================
// Problem 3: NaCl Strong Electrolyte (n=4, m=3)
// ===========================================================================

pub struct NaClSpeciation;

impl NaClSpeciation {
    const N: usize = 4;
    // Species: Na+, Cl-, H+, OH-
    const CHARGES: [f64; 4] = [1.0, -1.0, 1.0, -1.0];
    const DH_A: [f64; 4] = [4.0, 3.0, 9.0, 3.5];
    const DH_BDOT: [f64; 4] = [0.075, 0.015, 0.0, 0.0];
    const PKW: f64 = 14.0;

    fn mu0() -> [f64; 4] {
        [0.0, 0.0, 0.0, Self::PKW * LN10]
    }
}

impl NlpProblem for NaClSpeciation {
    fn num_variables(&self) -> usize { Self::N }
    fn num_constraints(&self) -> usize { 3 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..Self::N { x_l[i] = -30.0; x_u[i] = 1.0; }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for i in 0..3 { g_l[i] = 0.0; g_u[i] = 0.0; }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = (0.1_f64).ln();
        x0[1] = (0.1_f64).ln();
        x0[2] = (1e-7_f64).ln();
        x0[3] = (1e-7_f64).ln();
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        gibbs_objective(x, &Self::mu0(), &Self::CHARGES, &Self::DH_A, &Self::DH_BDOT)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        gibbs_gradient(x, &Self::mu0(), &Self::CHARGES, &Self::DH_A, &Self::DH_BDOT, grad);
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        g[0] = x[0].exp() - 0.1;  // Na balance
        g[1] = x[1].exp() - 0.1;  // Cl balance
        g[2] = x[0].exp() + x[2].exp() - x[1].exp() - x[3].exp(); // electroneutrality
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let rows = vec![0, 1, 2, 2, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 3];
        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = x[0].exp();
        vals[1] = x[1].exp();
        vals[2] = x[0].exp();
        vals[3] = -x[1].exp();
        vals[4] = x[2].exp();
        vals[5] = -x[3].exp();
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        dense_lower_triangle(Self::N)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        for v in vals.iter_mut() { *v = 0.0; }
        gibbs_hessian(x, &Self::mu0(), &Self::CHARGES, &Self::DH_A, &Self::DH_BDOT, obj_factor, vals);
        // g0: e^x0 - 0.1 → diagonal at (0,0)
        vals[lt_idx(0, 0)] += lambda[0] * x[0].exp();
        // g1: e^x1 - 0.1 → diagonal at (1,1)
        vals[lt_idx(1, 1)] += lambda[1] * x[1].exp();
        // g2: e^x0 + e^x2 - e^x1 - e^x3
        vals[lt_idx(0, 0)] += lambda[2] * x[0].exp();
        vals[lt_idx(1, 1)] += lambda[2] * (-x[1].exp());
        vals[lt_idx(2, 2)] += lambda[2] * x[2].exp();
        vals[lt_idx(3, 3)] += lambda[2] * (-x[3].exp());
    }
}

// ===========================================================================
// Problem 4: CaCl2+NaCl Mixed (n=6, m=4)
// ===========================================================================

pub struct CaCl2NaClMixed;

impl CaCl2NaClMixed {
    const N: usize = 6;
    // Species: Ca2+, Na+, Cl-, H+, OH-, CaOH+
    const CHARGES: [f64; 6] = [2.0, 1.0, -1.0, 1.0, -1.0, 1.0];
    const DH_A: [f64; 6] = [6.0, 4.0, 3.0, 9.0, 3.5, 4.0];
    const DH_BDOT: [f64; 6] = [0.165, 0.075, 0.015, 0.0, 0.0, 0.0];
    const PKW: f64 = 14.0;
    const LOG_K_CAOH: f64 = 1.3;

    fn mu0() -> [f64; 6] {
        // Ca2+=0, Na+=0, Cl-=0, H+=0, OH-=pKw*LN10,
        // CaOH+: from Ca2+ + OH- → CaOH+, K=10^1.3
        // mu0_CaOH = mu0_Ca + mu0_OH - ln(K) = pKw*LN10 - 1.3*LN10
        [0.0, 0.0, 0.0, 0.0, Self::PKW * LN10, (Self::PKW - Self::LOG_K_CAOH) * LN10]
    }
}

impl NlpProblem for CaCl2NaClMixed {
    fn num_variables(&self) -> usize { Self::N }
    fn num_constraints(&self) -> usize { 4 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..Self::N { x_l[i] = -35.0; x_u[i] = 1.0; }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for i in 0..4 { g_l[i] = 0.0; g_u[i] = 0.0; }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = (0.05_f64).ln();
        x0[1] = (0.1_f64).ln();
        x0[2] = (0.2_f64).ln();
        x0[3] = (1e-7_f64).ln();
        x0[4] = (1e-7_f64).ln();
        x0[5] = (1e-6_f64).ln();
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        gibbs_objective(x, &Self::mu0(), &Self::CHARGES, &Self::DH_A, &Self::DH_BDOT)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        gibbs_gradient(x, &Self::mu0(), &Self::CHARGES, &Self::DH_A, &Self::DH_BDOT, grad);
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        // Ca balance: m_Ca + m_CaOH = 0.05
        g[0] = x[0].exp() + x[5].exp() - 0.05;
        // Na balance: m_Na = 0.1
        g[1] = x[1].exp() - 0.1;
        // Cl balance: m_Cl = 0.2
        g[2] = x[2].exp() - 0.2;
        // Electroneutrality: 2*m_Ca + m_Na + m_H + m_CaOH - m_Cl - m_OH = 0
        g[3] = 2.0 * x[0].exp() + x[1].exp() + x[3].exp() + x[5].exp()
            - x[2].exp() - x[4].exp();
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let rows = vec![0, 0, 1, 2, 3, 3, 3, 3, 3, 3];
        let cols = vec![0, 5, 1, 2, 0, 1, 2, 3, 4, 5];
        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = x[0].exp();         // dg0/dx0
        vals[1] = x[5].exp();         // dg0/dx5
        vals[2] = x[1].exp();         // dg1/dx1
        vals[3] = x[2].exp();         // dg2/dx2
        vals[4] = 2.0 * x[0].exp();   // dg3/dx0
        vals[5] = x[1].exp();         // dg3/dx1
        vals[6] = -x[2].exp();        // dg3/dx2
        vals[7] = x[3].exp();         // dg3/dx3
        vals[8] = -x[4].exp();        // dg3/dx4
        vals[9] = x[5].exp();         // dg3/dx5
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        dense_lower_triangle(Self::N)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        for v in vals.iter_mut() { *v = 0.0; }
        gibbs_hessian(x, &Self::mu0(), &Self::CHARGES, &Self::DH_A, &Self::DH_BDOT, obj_factor, vals);
        // g0: e^x0 + e^x5 - 0.05
        vals[lt_idx(0, 0)] += lambda[0] * x[0].exp();
        vals[lt_idx(5, 5)] += lambda[0] * x[5].exp();
        // g1: e^x1 - 0.1
        vals[lt_idx(1, 1)] += lambda[1] * x[1].exp();
        // g2: e^x2 - 0.2
        vals[lt_idx(2, 2)] += lambda[2] * x[2].exp();
        // g3: 2*e^x0 + e^x1 + e^x3 + e^x5 - e^x2 - e^x4
        vals[lt_idx(0, 0)] += lambda[3] * 2.0 * x[0].exp();
        vals[lt_idx(1, 1)] += lambda[3] * x[1].exp();
        vals[lt_idx(2, 2)] += lambda[3] * (-x[2].exp());
        vals[lt_idx(3, 3)] += lambda[3] * x[3].exp();
        vals[lt_idx(4, 4)] += lambda[3] * (-x[4].exp());
        vals[lt_idx(5, 5)] += lambda[3] * x[5].exp();
    }
}

// ===========================================================================
// Problem 5: Phosphoric Acid H3PO4 (n=6, m=2)
// ===========================================================================

pub struct PhosphoricAcid;

impl PhosphoricAcid {
    const N: usize = 6;
    // Species: H3PO4, H2PO4-, HPO4^2-, PO4^3-, H+, OH-
    const CHARGES: [f64; 6] = [0.0, -1.0, -2.0, -3.0, 1.0, -1.0];
    const DH_A: [f64; 6] = [0.0, 4.5, 4.0, 4.0, 9.0, 3.5];
    const DH_BDOT: [f64; 6] = [0.0; 6];
    const PK1: f64 = 2.148;
    const PK2: f64 = 7.199;
    const PK3: f64 = 12.35;
    const PKW: f64 = 14.0;
    const P_TOTAL: f64 = 0.01;

    fn mu0() -> [f64; 6] {
        [
            0.0,
            Self::PK1 * LN10,
            (Self::PK1 + Self::PK2) * LN10,
            (Self::PK1 + Self::PK2 + Self::PK3) * LN10,
            0.0,
            Self::PKW * LN10,
        ]
    }
}

impl NlpProblem for PhosphoricAcid {
    fn num_variables(&self) -> usize { Self::N }
    fn num_constraints(&self) -> usize { 2 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..Self::N { x_l[i] = -55.0; x_u[i] = 0.0; }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0; g_u[0] = 0.0;
        g_l[1] = 0.0; g_u[1] = 0.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = (0.005_f64).ln();
        x0[1] = (0.004_f64).ln();
        x0[2] = (1e-5_f64).ln();
        x0[3] = (1e-15_f64).ln();
        x0[4] = (0.005_f64).ln();
        x0[5] = (1e-10_f64).ln();
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        gibbs_objective(x, &Self::mu0(), &Self::CHARGES, &Self::DH_A, &Self::DH_BDOT)
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        gibbs_gradient(x, &Self::mu0(), &Self::CHARGES, &Self::DH_A, &Self::DH_BDOT, grad);
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        // P balance
        g[0] = x[0].exp() + x[1].exp() + x[2].exp() + x[3].exp() - Self::P_TOTAL;
        // Electroneutrality
        g[1] = -x[1].exp() - 2.0 * x[2].exp() - 3.0 * x[3].exp() + x[4].exp() - x[5].exp();
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let rows = vec![0, 0, 0, 0, 1, 1, 1, 1, 1];
        let cols = vec![0, 1, 2, 3, 1, 2, 3, 4, 5];
        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        vals[0] = x[0].exp();
        vals[1] = x[1].exp();
        vals[2] = x[2].exp();
        vals[3] = x[3].exp();
        vals[4] = -x[1].exp();
        vals[5] = -2.0 * x[2].exp();
        vals[6] = -3.0 * x[3].exp();
        vals[7] = x[4].exp();
        vals[8] = -x[5].exp();
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        dense_lower_triangle(Self::N)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        for v in vals.iter_mut() { *v = 0.0; }
        gibbs_hessian(x, &Self::mu0(), &Self::CHARGES, &Self::DH_A, &Self::DH_BDOT, obj_factor, vals);
        let c0 = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        add_constraint_hessian_exp(x, lambda[0], &c0, vals);
        let c1 = [0.0, -1.0, -2.0, -3.0, 1.0, -1.0];
        add_constraint_hessian_exp(x, lambda[1], &c1, vals);
    }
}

// ===========================================================================
// Problem 6: HCl Mean Activity (n=1, m=0)
// ===========================================================================

pub struct HclMeanActivity;

impl HclMeanActivity {
    const BETA0: f64 = 0.1775;
    const BETA1: f64 = 0.2945;
    const C_PHI: f64 = 0.00080;
    // Target: gamma_pm(1.0) * 1.0 = 0.796 → ln(0.796) = -0.228
    fn target_ln_a() -> f64 {
        pitzer_ln_gamma_pm(1.0, Self::BETA0, Self::BETA1, Self::C_PHI) + (1.0_f64).ln()
    }
}

impl NlpProblem for HclMeanActivity {
    fn num_variables(&self) -> usize { 1 }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 0.01;
        x_u[0] = 5.0;
    }

    fn constraint_bounds(&self, _: &mut [f64], _: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.5;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let m = x[0];
        let ln_a = pitzer_ln_gamma_pm(m, Self::BETA0, Self::BETA1, Self::C_PHI) + m.ln();
        let r = ln_a - Self::target_ln_a();
        r * r
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let m = x[0];
        let ln_a = pitzer_ln_gamma_pm(m, Self::BETA0, Self::BETA1, Self::C_PHI) + m.ln();
        let r = ln_a - Self::target_ln_a();
        let dr = d_pitzer_ln_gamma_pm_dm(m, Self::BETA0, Self::BETA1, Self::C_PHI) + 1.0 / m;
        grad[0] = 2.0 * r * dr;
    }

    fn constraints(&self, _: &[f64], _new_x: bool, _: &mut [f64]) {}
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _: &[f64], _new_x: bool, _: &mut [f64]) {}

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        let m = x[0];
        let ln_a = pitzer_ln_gamma_pm(m, Self::BETA0, Self::BETA1, Self::C_PHI) + m.ln();
        let r = ln_a - Self::target_ln_a();
        let dr = d_pitzer_ln_gamma_pm_dm(m, Self::BETA0, Self::BETA1, Self::C_PHI) + 1.0 / m;
        // d²f/dm² = 2*dr^2 + 2*r*d²r/dm²
        // d²r/dm² via numerical for simplicity
        let eps = 1e-7 * m.max(1e-10);
        let drp = d_pitzer_ln_gamma_pm_dm(m + eps, Self::BETA0, Self::BETA1, Self::C_PHI) + 1.0 / (m + eps);
        let drm = d_pitzer_ln_gamma_pm_dm(m - eps, Self::BETA0, Self::BETA1, Self::C_PHI) + 1.0 / (m - eps);
        let d2r = (drp - drm) / (2.0 * eps);
        vals[0] = obj_factor * (2.0 * dr * dr + 2.0 * r * d2r);
    }
}

// ===========================================================================
// Problem 7: NaCl Solubility (n=1, m=0)
// ===========================================================================

pub struct NaClSolubility;

impl NaClSolubility {
    const BETA0: f64 = 0.0765;
    const BETA1: f64 = 0.2664;
    const C_PHI: f64 = 0.00127;
    const LN_KSP: f64 = 3.627; // ln(37.584)
}

impl NlpProblem for NaClSolubility {
    fn num_variables(&self) -> usize { 1 }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 0.1;
        x_u[0] = 15.0;
    }

    fn constraint_bounds(&self, _: &mut [f64], _: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 3.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let m = x[0];
        let r = 2.0 * pitzer_ln_gamma_pm(m, Self::BETA0, Self::BETA1, Self::C_PHI)
            + 2.0 * m.ln() - Self::LN_KSP;
        r * r
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let m = x[0];
        let r = 2.0 * pitzer_ln_gamma_pm(m, Self::BETA0, Self::BETA1, Self::C_PHI)
            + 2.0 * m.ln() - Self::LN_KSP;
        let dr = 2.0 * d_pitzer_ln_gamma_pm_dm(m, Self::BETA0, Self::BETA1, Self::C_PHI)
            + 2.0 / m;
        grad[0] = 2.0 * r * dr;
    }

    fn constraints(&self, _: &[f64], _new_x: bool, _: &mut [f64]) {}
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _: &[f64], _new_x: bool, _: &mut [f64]) {}

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![0], vec![0])
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        let m = x[0];
        let r = 2.0 * pitzer_ln_gamma_pm(m, Self::BETA0, Self::BETA1, Self::C_PHI)
            + 2.0 * m.ln() - Self::LN_KSP;
        let dr = 2.0 * d_pitzer_ln_gamma_pm_dm(m, Self::BETA0, Self::BETA1, Self::C_PHI)
            + 2.0 / m;
        let eps = 1e-7 * m.max(1e-10);
        let drp = 2.0 * d_pitzer_ln_gamma_pm_dm(m + eps, Self::BETA0, Self::BETA1, Self::C_PHI) + 2.0 / (m + eps);
        let drm = 2.0 * d_pitzer_ln_gamma_pm_dm(m - eps, Self::BETA0, Self::BETA1, Self::C_PHI) + 2.0 / (m - eps);
        let d2r = (drp - drm) / (2.0 * eps);
        vals[0] = obj_factor * (2.0 * dr * dr + 2.0 * r * d2r);
    }
}

// ===========================================================================
// Problem 8: Water-Butanol-NaCl LLE (n=2, m=2)
// ===========================================================================

pub struct ButanolWaterLle;

impl ButanolWaterLle {
    // NRTL params for 1-butanol(1)/water(2) system
    const TAU12: f64 = 0.50;   // BuOH→Water interaction
    const TAU21: f64 = 4.50;   // Water→BuOH interaction (large = BuOH dislikes water)
    const ALPHA: f64 = 0.40;
    const KS: f64 = 0.19;      // Setchenow coefficient for NaCl
    const M_NACL: f64 = 1.0;   // fixed salt molality
}

impl NlpProblem for ButanolWaterLle {
    fn num_variables(&self) -> usize { 2 }
    fn num_constraints(&self) -> usize { 2 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 1e-4; x_u[0] = 0.05;  // x_BuOH_aq
        x_l[1] = 0.3;  x_u[1] = 0.95;  // x_BuOH_org
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l[0] = 0.0; g_u[0] = 0.0;
        g_l[1] = 0.0; g_u[1] = 0.0;
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.006;
        x0[1] = 0.48;
    }

    // Small regularizer to avoid pure feasibility (helps the IPM)
    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        1e-6 * ((x[0] - 0.006).powi(2) + (x[1] - 0.48).powi(2))
    }
    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        grad[0] = 2e-6 * (x[0] - 0.006);
        grad[1] = 2e-6 * (x[1] - 0.48);
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let (lg1_aq, lg2_aq) = nrtl_binary(x[0], Self::TAU12, Self::TAU21, Self::ALPHA);
        let (lg1_org, lg2_org) = nrtl_binary(x[1], Self::TAU12, Self::TAU21, Self::ALPHA);
        // BuOH equilibrium (with Setchenow salting-out in aqueous phase)
        g[0] = lg1_aq + x[0].ln() + Self::KS * Self::M_NACL - lg1_org - x[1].ln();
        // Water equilibrium
        g[1] = lg2_aq + (1.0 - x[0]).ln() - lg2_org - (1.0 - x[1]).ln();
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // Both constraints depend on both variables
        let rows = vec![0, 0, 1, 1];
        let cols = vec![0, 1, 0, 1];
        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        let dlg1_aq = d_nrtl_ln_gamma1_dx1(x[0], Self::TAU12, Self::TAU21, Self::ALPHA);
        let dlg1_org = d_nrtl_ln_gamma1_dx1(x[1], Self::TAU12, Self::TAU21, Self::ALPHA);
        let dlg2_aq = d_nrtl_ln_gamma2_dx1(x[0], Self::TAU12, Self::TAU21, Self::ALPHA);
        let dlg2_org = d_nrtl_ln_gamma2_dx1(x[1], Self::TAU12, Self::TAU21, Self::ALPHA);
        // g0 = lg1_aq + ln(x0) + k_s*m - lg1_org - ln(x1)
        vals[0] = dlg1_aq + 1.0 / x[0];       // dg0/dx0
        vals[1] = -dlg1_org - 1.0 / x[1];     // dg0/dx1
        // g1 = lg2_aq + ln(1-x0) - lg2_org - ln(1-x1)
        vals[2] = dlg2_aq - 1.0 / (1.0 - x[0]); // dg1/dx0
        vals[3] = -dlg2_org + 1.0 / (1.0 - x[1]); // dg1/dx1
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        dense_lower_triangle(2)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        // Objective Hessian (regularizer): d²f/dx_i² = 2e-6
        vals[0] = obj_factor * 2e-6; vals[1] = 0.0; vals[2] = obj_factor * 2e-6;
        let eps = 1e-7;
        // d²g/dx_i dx_j via finite differences on Jacobian
        let mut jp = [0.0; 4];
        let mut jm = [0.0; 4];
        for var in 0..2 {
            let mut xp = [x[0], x[1]];
            let mut xm = [x[0], x[1]];
            xp[var] += eps;
            xm[var] -= eps;
            self.jacobian_values(&xp, true, &mut jp);
            self.jacobian_values(&xm, true, &mut jm);
            for con in 0..2 {
                // d²g[con]/(dx[var] dx[col]) for col = 0..var
                for col in 0..=var {
                    let jac_idx = con * 2 + col; // g[con] depends on x[col]
                    let d2 = (jp[jac_idx] - jm[jac_idx]) / (2.0 * eps);
                    if var == col {
                        vals[lt_idx(var, col)] += lambda[con] * d2;
                    } else {
                        // Only add if var > col (lower triangle)
                        vals[lt_idx(var, col)] += lambda[con] * d2;
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Problem 9: Saturated Brine VLE+SLE (n=3, m=3)
// ===========================================================================

pub struct SaturatedBrine;

impl SaturatedBrine {
    const BETA0: f64 = 0.0765;
    const BETA1: f64 = 0.2664;
    const C_PHI: f64 = 0.00127;
    const LN_KSP: f64 = 3.627;
    const P_W_PURE: f64 = 3.169; // kPa at 25C
}

impl NlpProblem for SaturatedBrine {
    fn num_variables(&self) -> usize { 3 }
    fn num_constraints(&self) -> usize { 3 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = 0.1;  x_u[0] = 15.0; // m_NaCl
        x_l[1] = 0.5;  x_u[1] = 1.0;  // a_w
        x_l[2] = 1.0;  x_u[2] = 3.5;  // p_w (kPa)
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for i in 0..3 { g_l[i] = 0.0; g_u[i] = 0.0; }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 3.0;
        x0[1] = 0.8;
        x0[2] = 2.5;
    }

    fn objective(&self, _x: &[f64], _new_x: bool) -> f64 { 0.0 }
    fn gradient(&self, _x: &[f64], _new_x: bool, grad: &mut [f64]) { for g in grad.iter_mut() { *g = 0.0; } }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let m = x[0];
        let a_w = x[1];
        let p_w = x[2];
        // SLE: 2*ln_gamma_pm(m) + 2*ln(m) = ln(K_sp)
        g[0] = 2.0 * pitzer_ln_gamma_pm(m, Self::BETA0, Self::BETA1, Self::C_PHI)
            + 2.0 * m.ln() - Self::LN_KSP;
        // Water activity: a_w = exp(-phi * 2 * m * M_W)
        let phi = pitzer_osmotic(m, Self::BETA0, Self::BETA1, Self::C_PHI);
        g[1] = a_w - (-phi * 2.0 * m * M_W).exp();
        // VLE: p_w = a_w * p_w_pure
        g[2] = p_w - a_w * Self::P_W_PURE;
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        // g0 depends on x0; g1 depends on x0, x1; g2 depends on x1, x2
        let rows = vec![0, 1, 1, 2, 2];
        let cols = vec![0, 0, 1, 1, 2];
        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        let m = x[0];
        // dg0/dm = 2*d_pitzer_ln_gamma_pm/dm + 2/m
        vals[0] = 2.0 * d_pitzer_ln_gamma_pm_dm(m, Self::BETA0, Self::BETA1, Self::C_PHI)
            + 2.0 / m;

        // dg1/dm
        let phi = pitzer_osmotic(m, Self::BETA0, Self::BETA1, Self::C_PHI);
        let dphi = d_pitzer_osmotic_dm(m, Self::BETA0, Self::BETA1, Self::C_PHI);
        let exp_val = (-phi * 2.0 * m * M_W).exp();
        // d/dm[-phi*2*m*M_W] = -(dphi*2*m*M_W + phi*2*M_W)
        vals[1] = exp_val * (dphi * 2.0 * m * M_W + phi * 2.0 * M_W);
        // dg1/da_w = 1
        vals[2] = 1.0;
        // dg2/da_w = -p_w_pure
        vals[3] = -Self::P_W_PURE;
        // dg2/dp_w = 1
        vals[4] = 1.0;
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        dense_lower_triangle(3)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, _obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        for v in vals.iter_mut() { *v = 0.0; }
        // Use numerical second derivatives for the constraint Hessian
        let eps = 1e-7;
        let n = 3;
        let mut jp = vec![0.0; 5];
        let mut jm = vec![0.0; 5];
        // Map: which jacobian entries correspond to which (constraint, variable)?
        // jac entry 0: (g0, x0), 1: (g1, x0), 2: (g1, x1), 3: (g2, x1), 4: (g2, x2)
        struct JacEntry { con: usize, var: usize }
        let entries = [
            JacEntry { con: 0, var: 0 },
            JacEntry { con: 1, var: 0 },
            JacEntry { con: 1, var: 1 },
            JacEntry { con: 2, var: 1 },
            JacEntry { con: 2, var: 2 },
        ];

        for d in 0..n {
            let mut xp = [x[0], x[1], x[2]];
            let mut xm = [x[0], x[1], x[2]];
            let h = eps * x[d].abs().max(1e-3);
            xp[d] += h;
            xm[d] -= h;
            self.jacobian_values(&xp, true, &mut jp);
            self.jacobian_values(&xm, true, &mut jm);
            for e in &entries {
                let jidx = entries.iter().position(|en| en.con == e.con && en.var == e.var).unwrap();
                // This gives d²g[e.con]/(dx[e.var] dx[d])
                let d2 = (jp[jidx] - jm[jidx]) / (2.0 * h);
                let (i, j) = if e.var >= d { (e.var, d) } else { (d, e.var) };
                vals[lt_idx(i, j)] += lambda[e.con] * d2;
            }
        }
    }
}

// ===========================================================================
// Problem 10: Pitzer NaCl Parameter Fit (n=3, m=0)
// ===========================================================================

pub struct PitzerNaClFit;

impl PitzerNaClFit {
    const TRUE_PARAMS: [f64; 3] = [0.0765, 0.2664, 0.00127];
    const MOLALITIES: [f64; 11] = [0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0];

    fn phi_data() -> [f64; 11] {
        let mut data = [0.0; 11];
        for (k, &m) in Self::MOLALITIES.iter().enumerate() {
            data[k] = pitzer_osmotic(m, Self::TRUE_PARAMS[0], Self::TRUE_PARAMS[1], Self::TRUE_PARAMS[2]);
        }
        data
    }
}

impl NlpProblem for PitzerNaClFit {
    fn num_variables(&self) -> usize { 3 }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = -1.0; x_u[0] = 1.0;
        x_l[1] = -1.0; x_u[1] = 2.0;
        x_l[2] = -0.1; x_u[2] = 0.1;
    }

    fn constraint_bounds(&self, _: &mut [f64], _: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 0.2;
        x0[1] = 0.5;
        x0[2] = 0.005;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let phi_d = Self::phi_data();
        let mut f = 0.0;
        for (k, &m) in Self::MOLALITIES.iter().enumerate() {
            let r = pitzer_osmotic(m, x[0], x[1], x[2]) - phi_d[k];
            f += r * r;
        }
        f
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let phi_d = Self::phi_data();
        for g in grad.iter_mut() { *g = 0.0; }
        for (k, &m) in Self::MOLALITIES.iter().enumerate() {
            let r = pitzer_osmotic(m, x[0], x[1], x[2]) - phi_d[k];
            let (dp0, dp1, dp2) = pitzer_osmotic_partials(m, x[0], x[1], x[2]);
            grad[0] += 2.0 * r * dp0;
            grad[1] += 2.0 * r * dp1;
            grad[2] += 2.0 * r * dp2;
        }
    }

    fn constraints(&self, _: &[f64], _new_x: bool, _: &mut [f64]) {}
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _: &[f64], _new_x: bool, _: &mut [f64]) {}

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        dense_lower_triangle(3)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        for v in vals.iter_mut() { *v = 0.0; }
        let phi_d = Self::phi_data();
        // f = sum r_k^2, H = 2*sum (J_k^T J_k + r_k * H_k)
        // Since phi is linear in parameters, H_k = 0 → Gauss-Newton: H = 2*J^T*J
        for (k, &m) in Self::MOLALITIES.iter().enumerate() {
            let r = pitzer_osmotic(m, x[0], x[1], x[2]) - phi_d[k];
            let (dp0, dp1, dp2) = pitzer_osmotic_partials(m, x[0], x[1], x[2]);
            let dp = [dp0, dp1, dp2];
            let _ = r; // r*H_k = 0 since phi is linear in params
            for i in 0..3 {
                for j in 0..=i {
                    vals[lt_idx(i, j)] += obj_factor * 2.0 * dp[i] * dp[j];
                }
            }
        }
    }
}

// ===========================================================================
// Problem 11: Multi-Salt DH Fit (n=8, m=0)
// ===========================================================================

pub struct MultiSaltDhFit;

impl MultiSaltDhFit {
    const TRUE_PARAMS: [f64; 8] = [4.0, 0.075, 3.0, 0.015, 6.0, 0.165, 3.0, 0.015];
    const MOLALITIES: [f64; 8] = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0];

    fn compute_ln_gamma_pm_nacl(x: &[f64], m: f64) -> f64 {
        let i_s = m;
        0.5 * (ln_gamma_dh(1.0, x[0], x[1], i_s) + ln_gamma_dh(1.0, x[6], x[7], i_s))
    }

    fn compute_ln_gamma_pm_kcl(x: &[f64], m: f64) -> f64 {
        let i_s = m;
        0.5 * (ln_gamma_dh(1.0, x[2], x[3], i_s) + ln_gamma_dh(1.0, x[6], x[7], i_s))
    }

    fn compute_ln_gamma_pm_cacl2(x: &[f64], m: f64) -> f64 {
        let i_s = 3.0 * m;
        (ln_gamma_dh(2.0, x[4], x[5], i_s) + 2.0 * ln_gamma_dh(1.0, x[6], x[7], i_s)) / 3.0
    }

    fn data() -> Vec<f64> {
        let tp = &Self::TRUE_PARAMS;
        let mut d = Vec::with_capacity(24);
        for &m in &Self::MOLALITIES {
            d.push(Self::compute_ln_gamma_pm_nacl(tp, m));
        }
        for &m in &Self::MOLALITIES {
            d.push(Self::compute_ln_gamma_pm_kcl(tp, m));
        }
        for &m in &Self::MOLALITIES {
            d.push(Self::compute_ln_gamma_pm_cacl2(tp, m));
        }
        d
    }
}

impl NlpProblem for MultiSaltDhFit {
    fn num_variables(&self) -> usize { 8 }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..4 {
            x_l[2 * i] = 1.0;     // a params
            x_u[2 * i] = 10.0;
            x_l[2 * i + 1] = -0.5; // b params
            x_u[2 * i + 1] = 0.5;
        }
    }

    fn constraint_bounds(&self, _: &mut [f64], _: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        for i in 0..4 {
            x0[2 * i] = 3.0;
            x0[2 * i + 1] = 0.0;
        }
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let data = Self::data();
        let mut f = 0.0;
        for (k, &m) in Self::MOLALITIES.iter().enumerate() {
            let r = Self::compute_ln_gamma_pm_nacl(x, m) - data[k];
            f += r * r;
        }
        for (k, &m) in Self::MOLALITIES.iter().enumerate() {
            let r = Self::compute_ln_gamma_pm_kcl(x, m) - data[8 + k];
            f += r * r;
        }
        for (k, &m) in Self::MOLALITIES.iter().enumerate() {
            let r = Self::compute_ln_gamma_pm_cacl2(x, m) - data[16 + k];
            f += r * r;
        }
        f
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        // Numerical gradient
        for g in grad.iter_mut() { *g = 0.0; }
        let f0 = self.objective(x, true);
        let mut xp = x.to_vec();
        for i in 0..8 {
            let h = 1e-7 * x[i].abs().max(1e-5);
            xp[i] = x[i] + h;
            let fp = self.objective(&xp, true);
            grad[i] = (fp - f0) / h;
            xp[i] = x[i];
        }
    }

    fn constraints(&self, _: &[f64], _new_x: bool, _: &mut [f64]) {}
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _: &[f64], _new_x: bool, _: &mut [f64]) {}

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        dense_lower_triangle(8)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        for v in vals.iter_mut() { *v = 0.0; }
        // Numerical Hessian
        let n = 8;
        let mut grad0 = vec![0.0; n];
        self.gradient(x, true, &mut grad0);
        let mut xp = x.to_vec();
        for j in 0..n {
            let h = 1e-6 * x[j].abs().max(1e-4);
            xp[j] = x[j] + h;
            let mut grad_p = vec![0.0; n];
            self.gradient(&xp, true, &mut grad_p);
            for i in j..n {
                vals[lt_idx(i, j)] = obj_factor * (grad_p[i] - grad0[i]) / h;
            }
            xp[j] = x[j];
        }
    }
}

// ===========================================================================
// Problem 12: eNRTL Temperature-Dependent Fit (n=4, m=0)
// ===========================================================================

pub struct EnrtlTempFit;

impl EnrtlTempFit {
    const TRUE_PARAMS: [f64; 4] = [8.045, -3987.0, -4.549, 2216.0];
    const TEMPS: [f64; 4] = [288.15, 298.15, 308.15, 318.15];
    const MOLALITIES: [f64; 8] = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0];

    fn model_ln_gamma(x: &[f64], m: f64, t: f64) -> f64 {
        let tau_ca = x[0] + x[1] / t;
        let tau_wc = x[2] + x[3] / t;
        let si = m.sqrt();
        -A_PHI * si / (1.0 + si)
            + m * tau_ca * (-0.2 * tau_ca).exp()
            + m * m * tau_wc * (-0.2 * tau_wc).exp()
    }

    fn data() -> Vec<f64> {
        let mut d = Vec::with_capacity(32);
        for &t in &Self::TEMPS {
            for &m in &Self::MOLALITIES {
                d.push(Self::model_ln_gamma(&Self::TRUE_PARAMS, m, t));
            }
        }
        d
    }
}

impl NlpProblem for EnrtlTempFit {
    fn num_variables(&self) -> usize { 4 }
    fn num_constraints(&self) -> usize { 0 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l[0] = -20.0; x_u[0] = 20.0;
        x_l[1] = -10000.0; x_u[1] = 10000.0;
        x_l[2] = -20.0; x_u[2] = 20.0;
        x_l[3] = -10000.0; x_u[3] = 10000.0;
    }

    fn constraint_bounds(&self, _: &mut [f64], _: &mut [f64]) {}

    fn initial_point(&self, x0: &mut [f64]) {
        x0[0] = 7.0;
        x0[1] = -3500.0;
        x0[2] = -3.5;
        x0[3] = 1800.0;
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let data = Self::data();
        let mut f = 0.0;
        let mut idx = 0;
        for &t in &Self::TEMPS {
            for &m in &Self::MOLALITIES {
                let r = Self::model_ln_gamma(x, m, t) - data[idx];
                f += r * r;
                idx += 1;
            }
        }
        f
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        for g in grad.iter_mut() { *g = 0.0; }
        let data = Self::data();
        let mut idx = 0;
        for &t in &Self::TEMPS {
            for &m in &Self::MOLALITIES {
                let tau_ca = x[0] + x[1] / t;
                let tau_wc = x[2] + x[3] / t;
                let e_ca = (-0.2 * tau_ca).exp();
                let e_wc = (-0.2 * tau_wc).exp();
                let r = Self::model_ln_gamma(x, m, t) - data[idx];
                // d(model)/d(tau_ca) = m * e_ca * (1 - 0.2*tau_ca)
                let dm_dtca = m * e_ca * (1.0 - 0.2 * tau_ca);
                let dm_dtwc = m * m * e_wc * (1.0 - 0.2 * tau_wc);
                // dtau_ca/dx[0] = 1, dtau_ca/dx[1] = 1/t
                grad[0] += 2.0 * r * dm_dtca;
                grad[1] += 2.0 * r * dm_dtca / t;
                grad[2] += 2.0 * r * dm_dtwc;
                grad[3] += 2.0 * r * dm_dtwc / t;
                idx += 1;
            }
        }
    }

    fn constraints(&self, _: &[f64], _new_x: bool, _: &mut [f64]) {}
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) { (vec![], vec![]) }
    fn jacobian_values(&self, _: &[f64], _new_x: bool, _: &mut [f64]) {}

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        dense_lower_triangle(4)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, _lambda: &[f64], vals: &mut [f64]) {
        // Numerical Hessian via gradient finite differences
        for v in vals.iter_mut() { *v = 0.0; }
        let n = 4;
        let mut grad0 = vec![0.0; n];
        self.gradient(x, true, &mut grad0);
        let mut xp = x.to_vec();
        for j in 0..n {
            let h = 1e-6 * x[j].abs().max(1e-3);
            xp[j] = x[j] + h;
            let mut grad_p = vec![0.0; n];
            self.gradient(&xp, true, &mut grad_p);
            for i in j..n {
                vals[lt_idx(i, j)] = obj_factor * (grad_p[i] - grad0[i]) / h;
            }
            xp[j] = x[j];
        }
    }
}

// ===========================================================================
// Problem 13: Seawater Speciation (n=15, m=8)
// ===========================================================================

pub struct SeawaterSpeciation;

impl SeawaterSpeciation {
    const N: usize = 15;
    // Species: Na+, K+, Mg2+, Ca2+, Cl-, SO4^2-, HCO3-, CO3^2-, H+, OH-,
    //          MgSO4(aq), CaSO4(aq), MgOH+, NaSO4-, KSO4-
    const CHARGES: [f64; 15] = [1.0, 1.0, 2.0, 2.0, -1.0, -2.0, -1.0, -2.0,
                                 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, -1.0];
    const DH_A_PARAMS: [f64; 15] = [4.0, 3.0, 6.0, 6.0, 3.0, 5.0, 4.0, 5.4,
                                      9.0, 3.5, 0.0, 0.0, 4.0, 4.0, 3.5];
    const DH_B_PARAMS: [f64; 15] = [0.075, 0.015, 0.165, 0.165, 0.015, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Total concentrations (mol/kg) - standard seawater
    const NA_TOTAL: f64 = 0.4861;
    const K_TOTAL: f64 = 0.01058;
    const MG_TOTAL: f64 = 0.05474;
    const CA_TOTAL: f64 = 0.01065;
    const CL_TOTAL: f64 = 0.5658;
    const S_TOTAL: f64 = 0.02927;
    const C_TOTAL: f64 = 0.002048;

    const PKW: f64 = 14.0;
    const PK2_CO3: f64 = 10.33;
    const LOG_K_MGSO4: f64 = 2.23;
    const LOG_K_CASO4: f64 = 2.30;
    const LOG_K_MGOH: f64 = 2.58;
    const LOG_K_NASO4: f64 = 0.70;
    const LOG_K_KSO4: f64 = 0.85;

    fn mu0() -> [f64; 15] {
        let mut mu = [0.0; 15];
        // Free ions: reference = 0
        // CO3^2-: from HCO3- ↔ CO3^2- + H+, pK2
        mu[7] = Self::PK2_CO3 * LN10;
        // OH-: from water
        mu[9] = Self::PKW * LN10;
        // Ion pairs: mu0 = mu0_cat + mu0_an - ln(K)
        mu[10] = -Self::LOG_K_MGSO4 * LN10; // MgSO4
        mu[11] = -Self::LOG_K_CASO4 * LN10; // CaSO4
        mu[12] = Self::PKW * LN10 - Self::LOG_K_MGOH * LN10; // MgOH+
        mu[13] = -Self::LOG_K_NASO4 * LN10; // NaSO4-
        mu[14] = -Self::LOG_K_KSO4 * LN10; // KSO4-
        mu
    }
}

impl NlpProblem for SeawaterSpeciation {
    fn num_variables(&self) -> usize { Self::N }
    fn num_constraints(&self) -> usize { 8 }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        for i in 0..Self::N { x_l[i] = -46.0; x_u[i] = 1.0; }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        for i in 0..8 { g_l[i] = 0.0; g_u[i] = 0.0; }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        // Free ions at approximately total concentrations
        x0[0] = (0.48_f64).ln();     // Na+
        x0[1] = (0.01_f64).ln();     // K+
        x0[2] = (0.05_f64).ln();     // Mg2+
        x0[3] = (0.01_f64).ln();     // Ca2+
        x0[4] = (0.56_f64).ln();     // Cl-
        x0[5] = (0.025_f64).ln();    // SO4^2-
        x0[6] = (0.002_f64).ln();    // HCO3-
        x0[7] = (1e-5_f64).ln();     // CO3^2-
        x0[8] = (1e-8_f64).ln();     // H+ (pH~8)
        x0[9] = (1e-6_f64).ln();     // OH-
        x0[10] = (1e-3_f64).ln();    // MgSO4
        x0[11] = (1e-3_f64).ln();    // CaSO4
        x0[12] = (1e-5_f64).ln();    // MgOH+
        x0[13] = (1e-3_f64).ln();    // NaSO4-
        x0[14] = (1e-4_f64).ln();    // KSO4-
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let mu0 = Self::mu0();
        // For neutral species use simple Setchenow: ln_gamma = 0.1 * I
        let i_s = ionic_strength(x, &Self::CHARGES);
        let mut f = 0.0;
        for i in 0..Self::N {
            let m_i = x[i].exp();
            let lg = if Self::CHARGES[i] == 0.0 {
                0.1 * i_s // Setchenow for neutral species
            } else {
                ln_gamma_dh(Self::CHARGES[i], Self::DH_A_PARAMS[i], Self::DH_B_PARAMS[i], i_s)
            };
            f += m_i * (mu0[i] + lg + x[i]);
        }
        f
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let n = Self::N;
        let mu0 = Self::mu0();
        let i_s = ionic_strength(x, &Self::CHARGES);

        // Compute all ln_gamma and their dI derivatives
        let mut lg = vec![0.0; n];
        let mut dlg_di = vec![0.0; n];
        for i in 0..n {
            if Self::CHARGES[i] == 0.0 {
                lg[i] = 0.1 * i_s;
                dlg_di[i] = 0.1;
            } else {
                lg[i] = ln_gamma_dh(Self::CHARGES[i], Self::DH_A_PARAMS[i], Self::DH_B_PARAMS[i], i_s);
                dlg_di[i] = d_ln_gamma_dh_di(Self::CHARGES[i], Self::DH_A_PARAMS[i], Self::DH_B_PARAMS[i], i_s);
            }
        }

        let mut s1 = 0.0;
        for i in 0..n {
            s1 += x[i].exp() * dlg_di[i];
        }

        for j in 0..n {
            let m_j = x[j].exp();
            let a_j = mu0[j] + lg[j] + x[j] + 1.0;
            let di_dxj = 0.5 * Self::CHARGES[j] * Self::CHARGES[j] * m_j;
            grad[j] = m_j * a_j + di_dxj * s1;
        }
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        // g[0]: Na = x0 + x13
        g[0] = x[0].exp() + x[13].exp() - Self::NA_TOTAL;
        // g[1]: K = x1 + x14
        g[1] = x[1].exp() + x[14].exp() - Self::K_TOTAL;
        // g[2]: Mg = x2 + x10 + x12
        g[2] = x[2].exp() + x[10].exp() + x[12].exp() - Self::MG_TOTAL;
        // g[3]: Ca = x3 + x11
        g[3] = x[3].exp() + x[11].exp() - Self::CA_TOTAL;
        // g[4]: Cl = x4
        g[4] = x[4].exp() - Self::CL_TOTAL;
        // g[5]: S = x5 + x10 + x11 + x13 + x14
        g[5] = x[5].exp() + x[10].exp() + x[11].exp() + x[13].exp() + x[14].exp() - Self::S_TOTAL;
        // g[6]: C = x6 + x7
        g[6] = x[6].exp() + x[7].exp() - Self::C_TOTAL;
        // g[7]: Electroneutrality
        g[7] = 0.0;
        for i in 0..Self::N {
            g[7] += Self::CHARGES[i] * x[i].exp();
        }
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        // g0: x0, x13
        rows.push(0); cols.push(0);
        rows.push(0); cols.push(13);
        // g1: x1, x14
        rows.push(1); cols.push(1);
        rows.push(1); cols.push(14);
        // g2: x2, x10, x12
        rows.push(2); cols.push(2);
        rows.push(2); cols.push(10);
        rows.push(2); cols.push(12);
        // g3: x3, x11
        rows.push(3); cols.push(3);
        rows.push(3); cols.push(11);
        // g4: x4
        rows.push(4); cols.push(4);
        // g5: x5, x10, x11, x13, x14
        rows.push(5); cols.push(5);
        rows.push(5); cols.push(10);
        rows.push(5); cols.push(11);
        rows.push(5); cols.push(13);
        rows.push(5); cols.push(14);
        // g6: x6, x7
        rows.push(6); cols.push(6);
        rows.push(6); cols.push(7);
        // g7: all 15 vars
        for i in 0..Self::N {
            rows.push(7);
            cols.push(i);
        }
        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        let mut idx = 0;
        // g0
        vals[idx] = x[0].exp(); idx += 1;
        vals[idx] = x[13].exp(); idx += 1;
        // g1
        vals[idx] = x[1].exp(); idx += 1;
        vals[idx] = x[14].exp(); idx += 1;
        // g2
        vals[idx] = x[2].exp(); idx += 1;
        vals[idx] = x[10].exp(); idx += 1;
        vals[idx] = x[12].exp(); idx += 1;
        // g3
        vals[idx] = x[3].exp(); idx += 1;
        vals[idx] = x[11].exp(); idx += 1;
        // g4
        vals[idx] = x[4].exp(); idx += 1;
        // g5
        vals[idx] = x[5].exp(); idx += 1;
        vals[idx] = x[10].exp(); idx += 1;
        vals[idx] = x[11].exp(); idx += 1;
        vals[idx] = x[13].exp(); idx += 1;
        vals[idx] = x[14].exp(); idx += 1;
        // g6
        vals[idx] = x[6].exp(); idx += 1;
        vals[idx] = x[7].exp(); idx += 1;
        // g7
        for i in 0..Self::N {
            vals[idx] = Self::CHARGES[i] * x[i].exp();
            idx += 1;
        }
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        dense_lower_triangle(Self::N)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        let n = Self::N;
        let nnz = n * (n + 1) / 2;
        for v in vals[..nnz].iter_mut() { *v = 0.0; }

        // Objective Hessian (using the generalized form with Setchenow for neutrals)
        let mu0 = Self::mu0();
        let i_s = ionic_strength(x, &Self::CHARGES);

        let mut m_vec = vec![0.0; n];
        let mut lg = vec![0.0; n];
        let mut dlg = vec![0.0; n];
        let mut d2lg = vec![0.0; n];
        let mut s1 = 0.0;
        let mut s2 = 0.0;

        for i in 0..n {
            m_vec[i] = x[i].exp();
            if Self::CHARGES[i] == 0.0 {
                lg[i] = 0.1 * i_s;
                dlg[i] = 0.1;
                d2lg[i] = 0.0;
            } else {
                lg[i] = ln_gamma_dh(Self::CHARGES[i], Self::DH_A_PARAMS[i], Self::DH_B_PARAMS[i], i_s);
                dlg[i] = d_ln_gamma_dh_di(Self::CHARGES[i], Self::DH_A_PARAMS[i], Self::DH_B_PARAMS[i], i_s);
                d2lg[i] = d2_ln_gamma_dh_di2(Self::CHARGES[i], Self::DH_A_PARAMS[i], Self::DH_B_PARAMS[i], i_s);
            }
            s1 += m_vec[i] * dlg[i];
            s2 += m_vec[i] * d2lg[i];
        }

        for j in 0..n {
            let zj2 = Self::CHARGES[j] * Self::CHARGES[j];
            let mj = m_vec[j];
            let a_j = mu0[j] + lg[j] + x[j] + 1.0;

            let h_jj = mj * (a_j + 1.0)
                + zj2 * mj * mj * dlg[j]
                + 0.5 * zj2 * mj * s1
                + 0.5 * zj2 * mj * mj * dlg[j]
                + 0.25 * zj2 * zj2 * mj * mj * s2;
            vals[lt_idx(j, j)] += obj_factor * h_jj;

            for k in 0..j {
                let zk2 = Self::CHARGES[k] * Self::CHARGES[k];
                let mk = m_vec[k];
                let h_jk = 0.5 * zk2 * mj * mk * dlg[j]
                    + 0.5 * zj2 * mj * mk * dlg[k]
                    + 0.25 * zj2 * zk2 * mj * mk * s2;
                vals[lt_idx(j, k)] += obj_factor * h_jk;
            }
        }

        // Constraint Hessians (all are diagonal since g = sum(c_i * e^{x_i}) - const)
        // g0: Na: e^x0 + e^x13
        vals[lt_idx(0, 0)] += lambda[0] * x[0].exp();
        vals[lt_idx(13, 13)] += lambda[0] * x[13].exp();
        // g1: K: e^x1 + e^x14
        vals[lt_idx(1, 1)] += lambda[1] * x[1].exp();
        vals[lt_idx(14, 14)] += lambda[1] * x[14].exp();
        // g2: Mg: e^x2 + e^x10 + e^x12
        vals[lt_idx(2, 2)] += lambda[2] * x[2].exp();
        vals[lt_idx(10, 10)] += lambda[2] * x[10].exp();
        vals[lt_idx(12, 12)] += lambda[2] * x[12].exp();
        // g3: Ca: e^x3 + e^x11
        vals[lt_idx(3, 3)] += lambda[3] * x[3].exp();
        vals[lt_idx(11, 11)] += lambda[3] * x[11].exp();
        // g4: Cl: e^x4
        vals[lt_idx(4, 4)] += lambda[4] * x[4].exp();
        // g5: S: e^x5 + e^x10 + e^x11 + e^x13 + e^x14
        vals[lt_idx(5, 5)] += lambda[5] * x[5].exp();
        vals[lt_idx(10, 10)] += lambda[5] * x[10].exp();
        vals[lt_idx(11, 11)] += lambda[5] * x[11].exp();
        vals[lt_idx(13, 13)] += lambda[5] * x[13].exp();
        vals[lt_idx(14, 14)] += lambda[5] * x[14].exp();
        // g6: C: e^x6 + e^x7
        vals[lt_idx(6, 6)] += lambda[6] * x[6].exp();
        vals[lt_idx(7, 7)] += lambda[6] * x[7].exp();
        // g7: electroneutrality: sum(z_i * e^x_i)
        for i in 0..n {
            vals[lt_idx(i, i)] += lambda[7] * Self::CHARGES[i] * x[i].exp();
        }
    }
}

// ===========================================================================
// Public list of all problems (for benchmark)
// ===========================================================================

/// Returns a list of (name, problem) pairs for all 13 problems.
pub fn all_problems() -> Vec<(&'static str, Box<dyn NlpProblem>)> {
    vec![
        ("Water autoionization", Box::new(WaterAutoionization)),
        ("CO2-water speciation", Box::new(Co2WaterSpeciation)),
        ("NaCl speciation", Box::new(NaClSpeciation)),
        ("CaCl2+NaCl mixed", Box::new(CaCl2NaClMixed)),
        ("Phosphoric acid", Box::new(PhosphoricAcid)),
        ("HCl mean activity", Box::new(HclMeanActivity)),
        ("NaCl solubility", Box::new(NaClSolubility)),
        ("BuOH-water LLE", Box::new(ButanolWaterLle)),
        ("Saturated brine", Box::new(SaturatedBrine)),
        ("Pitzer NaCl fit", Box::new(PitzerNaClFit)),
        ("Multi-salt DH fit", Box::new(MultiSaltDhFit)),
        ("eNRTL T-dep fit", Box::new(EnrtlTempFit)),
        ("Seawater speciation", Box::new(SeawaterSpeciation)),
    ]
}
