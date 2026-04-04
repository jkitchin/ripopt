// AC Optimal Power Flow (ACOPF) benchmark problems for ripopt.
//
// Implements the polar-form ACOPF formulation for standard IEEE/PGLib test cases:
//   case3_lmbd   (3 buses, 3 generators, 3 branches)
//   case5_pjm    (5 buses, 5 generators, 6 branches)
//   case14_ieee  (14 buses, 5 generators, 20 branches)
//   case30_ieee  (30 buses, 6 generators, 41 branches)

use ripopt::NlpProblem;
use std::f64::consts::PI;

// ===========================================================================
// Data structures
// ===========================================================================

struct BusData {
    pd: f64,
    qd: f64,
    gs: f64,
    bs: f64,
    v_max: f64,
    v_min: f64,
    bus_type: u8,
}

struct GenData {
    bus: usize,
    pg_max: f64,
    pg_min: f64,
    qg_max: f64,
    qg_min: f64,
    c2: f64,
    c1: f64,
    c0: f64,
}

struct BranchData {
    from: usize,
    to: usize,
    r: f64,
    x: f64,
    b_ch: f64,
    rate_a: f64,
    tap: f64,
}

// Precomputed Jacobian entry descriptor
#[derive(Clone)]
struct JacEntry {
    row: usize,
    col: usize,
}

pub struct AcopfProblem {
    #[allow(dead_code)]
    name: &'static str,
    n_bus: usize,
    n_gen: usize,
    #[allow(dead_code)]
    n_branch: usize,
    base_mva: f64,
    buses: Vec<BusData>,
    gens: Vec<GenData>,
    branches: Vec<BranchData>,
    // Derived
    y_bus_re: Vec<f64>, // G matrix (N*N, row-major)
    y_bus_im: Vec<f64>, // B matrix (N*N, row-major)
    ref_bus: usize,
    gen_at_bus: Vec<Vec<usize>>,
    flow_lim_branches: Vec<usize>,
    n_flow_lim: usize,
    // Precomputed branch admittance terms
    branch_g_ff: Vec<f64>,
    branch_b_ff: Vec<f64>,
    branch_g_ft: Vec<f64>,
    branch_b_ft: Vec<f64>,
    branch_g_tt: Vec<f64>,
    branch_b_tt: Vec<f64>,
    branch_g_tf: Vec<f64>,
    branch_b_tf: Vec<f64>,
    // Precomputed adjacency: for each bus, list of buses connected via Y_bus
    adj: Vec<Vec<usize>>,
    // Jacobian structure cache
    jac_entries: Vec<JacEntry>,
}

// ===========================================================================
// Index helpers
// ===========================================================================

impl AcopfProblem {
    fn idx_v(&self, i: usize) -> usize {
        i
    }
    fn idx_theta(&self, i: usize) -> usize {
        self.n_bus + i
    }
    fn idx_pg(&self, g: usize) -> usize {
        2 * self.n_bus + g
    }
    fn idx_qg(&self, g: usize) -> usize {
        2 * self.n_bus + self.n_gen + g
    }
    fn n_var(&self) -> usize {
        2 * self.n_bus + 2 * self.n_gen
    }
    fn n_con(&self) -> usize {
        2 * self.n_bus + 2 * self.n_flow_lim
    }
    fn yidx(&self, i: usize, k: usize) -> usize {
        i * self.n_bus + k
    }
}

// ===========================================================================
// Constructor
// ===========================================================================

impl AcopfProblem {
    fn new(
        name: &'static str,
        base_mva: f64,
        buses: Vec<BusData>,
        gens: Vec<GenData>,
        branches: Vec<BranchData>,
    ) -> Self {
        let n_bus = buses.len();
        let n_gen = gens.len();
        let n_branch = branches.len();

        // Find reference bus
        let ref_bus = buses
            .iter()
            .position(|b| b.bus_type == 3)
            .expect("No reference bus found");

        // Build gen_at_bus
        let mut gen_at_bus = vec![Vec::new(); n_bus];
        for (g, gen) in gens.iter().enumerate() {
            gen_at_bus[gen.bus].push(g);
        }

        // Build Y_bus admittance matrix
        let mut y_bus_re = vec![0.0; n_bus * n_bus];
        let mut y_bus_im = vec![0.0; n_bus * n_bus];

        // Precompute branch admittance terms
        let mut branch_g_ff = vec![0.0; n_branch];
        let mut branch_b_ff = vec![0.0; n_branch];
        let mut branch_g_ft = vec![0.0; n_branch];
        let mut branch_b_ft = vec![0.0; n_branch];
        let mut branch_g_tt = vec![0.0; n_branch];
        let mut branch_b_tt = vec![0.0; n_branch];
        let mut branch_g_tf = vec![0.0; n_branch];
        let mut branch_b_tf = vec![0.0; n_branch];

        for (l, br) in branches.iter().enumerate() {
            let f = br.from;
            let t = br.to;
            let r = br.r;
            let x = br.x;
            let b_ch = br.b_ch;
            let tap = if br.tap == 0.0 { 1.0 } else { br.tap };

            let z_sq = r * r + x * x;
            let g_s = r / z_sq;
            let b_s = -x / z_sq;

            let tap2 = tap * tap;

            // Y_bus contributions
            // Y_ff += (g_s + j*(b_s + b_ch/2)) / tap^2
            y_bus_re[f * n_bus + f] += g_s / tap2;
            y_bus_im[f * n_bus + f] += (b_s + b_ch / 2.0) / tap2;

            // Y_tt += g_s + j*(b_s + b_ch/2)
            y_bus_re[t * n_bus + t] += g_s;
            y_bus_im[t * n_bus + t] += b_s + b_ch / 2.0;

            // Y_ft += -(g_s + j*b_s) / tap
            y_bus_re[f * n_bus + t] += -g_s / tap;
            y_bus_im[f * n_bus + t] += -b_s / tap;

            // Y_tf += -(g_s + j*b_s) / tap
            y_bus_re[t * n_bus + f] += -g_s / tap;
            y_bus_im[t * n_bus + f] += -b_s / tap;

            // Branch admittance terms for flow calculations
            branch_g_ff[l] = g_s / tap2;
            branch_b_ff[l] = (b_s + b_ch / 2.0) / tap2;
            branch_g_ft[l] = -g_s / tap;
            branch_b_ft[l] = -b_s / tap;

            branch_g_tt[l] = g_s;
            branch_b_tt[l] = b_s + b_ch / 2.0;
            branch_g_tf[l] = -g_s / tap;
            branch_b_tf[l] = -b_s / tap;
        }

        // Add bus shunt admittance
        for (i, bus) in buses.iter().enumerate() {
            y_bus_re[i * n_bus + i] += bus.gs / base_mva;
            y_bus_im[i * n_bus + i] += bus.bs / base_mva;
        }

        // Identify branches with flow limits
        let flow_lim_branches: Vec<usize> = (0..n_branch)
            .filter(|&l| branches[l].rate_a > 0.0 && branches[l].rate_a < 9900.0)
            .collect();
        let n_flow_lim = flow_lim_branches.len();

        // Build adjacency list (buses connected via nonzero Y_bus entries)
        let mut adj = vec![Vec::new(); n_bus];
        for i in 0..n_bus {
            for k in 0..n_bus {
                if i != k
                    && (y_bus_re[i * n_bus + k].abs() > 1e-15
                        || y_bus_im[i * n_bus + k].abs() > 1e-15)
                {
                    adj[i].push(k);
                }
            }
        }

        // Build Jacobian structure
        let jac_entries = Self::build_jac_structure(
            n_bus,
            n_gen,
            &gen_at_bus,
            &adj,
            &flow_lim_branches,
            &branches,
        );

        AcopfProblem {
            name,
            n_bus,
            n_gen,
            n_branch,
            base_mva,
            buses,
            gens,
            branches,
            y_bus_re,
            y_bus_im,
            ref_bus,
            gen_at_bus,
            flow_lim_branches,
            n_flow_lim,
            branch_g_ff,
            branch_b_ff,
            branch_g_ft,
            branch_b_ft,
            branch_g_tt,
            branch_b_tt,
            branch_g_tf,
            branch_b_tf,
            adj,
            jac_entries,
        }
    }

    fn build_jac_structure(
        n_bus: usize,
        n_gen: usize,
        gen_at_bus: &[Vec<usize>],
        adj: &[Vec<usize>],
        flow_lim_branches: &[usize],
        branches: &[BranchData],
    ) -> Vec<JacEntry> {
        let mut entries = Vec::new();

        for i in 0..n_bus {
            let p_row = 2 * i;
            let q_row = 2 * i + 1;

            // P balance: depends on V_i, V_k for k adjacent, theta_i, theta_k, Pg_g
            // V_i
            entries.push(JacEntry {
                row: p_row,
                col: i,
            });
            // V_k for adjacent buses (sorted for determinism)
            let mut adj_sorted: Vec<usize> = adj[i].clone();
            adj_sorted.sort();
            for &k in &adj_sorted {
                entries.push(JacEntry {
                    row: p_row,
                    col: k,
                });
            }
            // theta_i
            entries.push(JacEntry {
                row: p_row,
                col: n_bus + i,
            });
            // theta_k
            for &k in &adj_sorted {
                entries.push(JacEntry {
                    row: p_row,
                    col: n_bus + k,
                });
            }
            // Pg_g for generators at bus i
            for &g in &gen_at_bus[i] {
                entries.push(JacEntry {
                    row: p_row,
                    col: 2 * n_bus + g,
                });
            }

            // Q balance: depends on V_i, V_k, theta_i, theta_k, Qg_g
            // V_i
            entries.push(JacEntry {
                row: q_row,
                col: i,
            });
            // V_k for adjacent buses
            for &k in &adj_sorted {
                entries.push(JacEntry {
                    row: q_row,
                    col: k,
                });
            }
            // theta_i
            entries.push(JacEntry {
                row: q_row,
                col: n_bus + i,
            });
            // theta_k
            for &k in &adj_sorted {
                entries.push(JacEntry {
                    row: q_row,
                    col: n_bus + k,
                });
            }
            // Qg_g for generators at bus i
            for &g in &gen_at_bus[i] {
                entries.push(JacEntry {
                    row: q_row,
                    col: 2 * n_bus + n_gen + g,
                });
            }
        }

        // Flow limit constraints
        for (fl_idx, &l) in flow_lim_branches.iter().enumerate() {
            let f = branches[l].from;
            let t = branches[l].to;
            let base_row = 2 * n_bus + 2 * fl_idx;

            // From-end flow constraint: depends on V_f, V_t, theta_f, theta_t
            entries.push(JacEntry {
                row: base_row,
                col: f,
            }); // V_f
            entries.push(JacEntry {
                row: base_row,
                col: t,
            }); // V_t
            entries.push(JacEntry {
                row: base_row,
                col: n_bus + f,
            }); // theta_f
            entries.push(JacEntry {
                row: base_row,
                col: n_bus + t,
            }); // theta_t

            // To-end flow constraint
            entries.push(JacEntry {
                row: base_row + 1,
                col: f,
            }); // V_f
            entries.push(JacEntry {
                row: base_row + 1,
                col: t,
            }); // V_t
            entries.push(JacEntry {
                row: base_row + 1,
                col: n_bus + f,
            }); // theta_f
            entries.push(JacEntry {
                row: base_row + 1,
                col: n_bus + t,
            }); // theta_t
        }

        entries
    }
}

// ===========================================================================
// Flow calculation helpers
// ===========================================================================

impl AcopfProblem {
    /// Compute from-end P and Q flows for branch l
    fn flow_from(&self, l: usize, x: &[f64]) -> (f64, f64) {
        let f = self.branches[l].from;
        let t = self.branches[l].to;
        let vf = x[self.idx_v(f)];
        let vt = x[self.idx_v(t)];
        let theta_ft = x[self.idx_theta(f)] - x[self.idx_theta(t)];
        let c = theta_ft.cos();
        let s = theta_ft.sin();

        let g_ff = self.branch_g_ff[l];
        let b_ff = self.branch_b_ff[l];
        let g_ft = self.branch_g_ft[l];
        let b_ft = self.branch_b_ft[l];

        let p_ft = vf * vf * g_ff + vf * vt * (g_ft * c + b_ft * s);
        let q_ft = -vf * vf * b_ff + vf * vt * (g_ft * s - b_ft * c);

        (p_ft, q_ft)
    }

    /// Compute to-end P and Q flows for branch l
    fn flow_to(&self, l: usize, x: &[f64]) -> (f64, f64) {
        let f = self.branches[l].from;
        let t = self.branches[l].to;
        let vf = x[self.idx_v(f)];
        let vt = x[self.idx_v(t)];
        let theta_tf = x[self.idx_theta(t)] - x[self.idx_theta(f)];
        let c = theta_tf.cos();
        let s = theta_tf.sin();

        let g_tt = self.branch_g_tt[l];
        let b_tt = self.branch_b_tt[l];
        let g_tf = self.branch_g_tf[l];
        let b_tf = self.branch_b_tf[l];

        let p_tf = vt * vt * g_tt + vf * vt * (g_tf * c + b_tf * s);
        let q_tf = -vt * vt * b_tt + vf * vt * (g_tf * s - b_tf * c);

        (p_tf, q_tf)
    }
}

// ===========================================================================
// NlpProblem trait implementation
// ===========================================================================

impl NlpProblem for AcopfProblem {
    fn num_variables(&self) -> usize {
        self.n_var()
    }

    fn num_constraints(&self) -> usize {
        self.n_con()
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        let n = self.n_bus;
        let ng = self.n_gen;
        let bmva = self.base_mva;

        // Voltage magnitudes
        for i in 0..n {
            x_l[self.idx_v(i)] = self.buses[i].v_min;
            x_u[self.idx_v(i)] = self.buses[i].v_max;
        }

        // Voltage angles
        for i in 0..n {
            if i == self.ref_bus {
                x_l[self.idx_theta(i)] = 0.0;
                x_u[self.idx_theta(i)] = 0.0;
            } else {
                x_l[self.idx_theta(i)] = -PI;
                x_u[self.idx_theta(i)] = PI;
            }
        }

        // Generator real power (per-unit)
        for g in 0..ng {
            x_l[self.idx_pg(g)] = self.gens[g].pg_min / bmva;
            x_u[self.idx_pg(g)] = self.gens[g].pg_max / bmva;
        }

        // Generator reactive power (per-unit)
        for g in 0..ng {
            x_l[self.idx_qg(g)] = self.gens[g].qg_min / bmva;
            x_u[self.idx_qg(g)] = self.gens[g].qg_max / bmva;
        }
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        // Power balance: equality constraints
        for i in 0..2 * self.n_bus {
            g_l[i] = 0.0;
            g_u[i] = 0.0;
        }

        // Flow limits: <= 0
        for fl in 0..self.n_flow_lim {
            let base = 2 * self.n_bus + 2 * fl;
            g_l[base] = f64::NEG_INFINITY;
            g_u[base] = 0.0;
            g_l[base + 1] = f64::NEG_INFINITY;
            g_u[base + 1] = 0.0;
        }
    }

    fn initial_point(&self, x0: &mut [f64]) {
        let bmva = self.base_mva;

        // Flat start
        for i in 0..self.n_bus {
            x0[self.idx_v(i)] = 1.0;
            x0[self.idx_theta(i)] = 0.0;
        }

        for g in 0..self.n_gen {
            x0[self.idx_pg(g)] = (self.gens[g].pg_max + self.gens[g].pg_min) / (2.0 * bmva);
            x0[self.idx_qg(g)] = (self.gens[g].qg_max + self.gens[g].qg_min) / (2.0 * bmva);
        }
    }

    fn objective(&self, x: &[f64], _new_x: bool) -> f64 {
        let bmva = self.base_mva;
        let mut f = 0.0;
        for g in 0..self.n_gen {
            let pg_pu = x[self.idx_pg(g)];
            let pg_mw = pg_pu * bmva;
            f += self.gens[g].c2 * pg_mw * pg_mw + self.gens[g].c1 * pg_mw + self.gens[g].c0;
        }
        f
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) {
        let bmva = self.base_mva;
        let nv = self.n_var();
        for i in 0..nv {
            grad[i] = 0.0;
        }
        for g in 0..self.n_gen {
            let pg_pu = x[self.idx_pg(g)];
            // df/d(pg_pu) = 2*c2*baseMVA^2*pg_pu + c1*baseMVA
            grad[self.idx_pg(g)] =
                2.0 * self.gens[g].c2 * bmva * bmva * pg_pu + self.gens[g].c1 * bmva;
        }
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) {
        let n = self.n_bus;
        let bmva = self.base_mva;

        // Power balance constraints
        for i in 0..n {
            let vi = x[self.idx_v(i)];
            let theta_i = x[self.idx_theta(i)];

            // Sum generator injections at bus i
            let mut pg_sum = 0.0;
            let mut qg_sum = 0.0;
            for &gg in &self.gen_at_bus[i] {
                pg_sum += x[self.idx_pg(gg)];
                qg_sum += x[self.idx_qg(gg)];
            }

            // Power flow from Y_bus
            let mut p_flow = 0.0;
            let mut q_flow = 0.0;
            for k in 0..n {
                let vk = x[self.idx_v(k)];
                let theta_ik = theta_i - x[self.idx_theta(k)];
                let c = theta_ik.cos();
                let s = theta_ik.sin();
                let gik = self.y_bus_re[self.yidx(i, k)];
                let bik = self.y_bus_im[self.yidx(i, k)];
                p_flow += vi * vk * (gik * c + bik * s);
                q_flow += vi * vk * (gik * s - bik * c);
            }

            let pd = self.buses[i].pd / bmva;
            let qd = self.buses[i].qd / bmva;

            g[2 * i] = pg_sum - pd - p_flow;
            g[2 * i + 1] = qg_sum - qd - q_flow;
        }

        // Flow limit constraints
        for (fl_idx, &l) in self.flow_lim_branches.iter().enumerate() {
            let rate_a_pu = self.branches[l].rate_a / bmva;
            let limit_sq = rate_a_pu * rate_a_pu;

            let (p_ft, q_ft) = self.flow_from(l, x);
            let (p_tf, q_tf) = self.flow_to(l, x);

            let base = 2 * n + 2 * fl_idx;
            g[base] = p_ft * p_ft + q_ft * q_ft - limit_sq;
            g[base + 1] = p_tf * p_tf + q_tf * q_tf - limit_sq;
        }
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let rows: Vec<usize> = self.jac_entries.iter().map(|e| e.row).collect();
        let cols: Vec<usize> = self.jac_entries.iter().map(|e| e.col).collect();
        (rows, cols)
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) {
        let n = self.n_bus;
        let mut idx = 0;

        for i in 0..n {
            let vi = x[self.idx_v(i)];
            let theta_i = x[self.idx_theta(i)];

            // ===== P balance row =====
            // dP/dV_i = sum_k V_k*(G_ik cos(theta_ik) + B_ik sin(theta_ik))
            //         + V_i * (G_ii cos(0) + B_ii sin(0))
            //         = sum_{k!=i} V_k*(G_ik*c + B_ik*s) + 2*V_i*G_ii
            {
                let mut dp_dvi = 0.0;
                let gii = self.y_bus_re[self.yidx(i, i)];
                // Diagonal term: d/dV_i of V_i^2 * G_ii = 2*V_i*G_ii
                // Plus sum over k!=i of d/dV_i of V_i*V_k*(G_ik*c + B_ik*s) = V_k*(G_ik*c + B_ik*s)
                for k in 0..n {
                    let vk = x[self.idx_v(k)];
                    let theta_ik = theta_i - x[self.idx_theta(k)];
                    let c = theta_ik.cos();
                    let s = theta_ik.sin();
                    let gik = self.y_bus_re[self.yidx(i, k)];
                    let bik = self.y_bus_im[self.yidx(i, k)];
                    if k == i {
                        dp_dvi += 2.0 * vi * gii;
                    } else {
                        dp_dvi += vk * (gik * c + bik * s);
                    }
                }
                // The constraint is Pg - Pd - P_flow, so dg/dV_i = -dP_flow/dV_i
                vals[idx] = -dp_dvi;
                idx += 1;
            }

            // dP/dV_k for adjacent buses
            let mut adj_sorted: Vec<usize> = self.adj[i].clone();
            adj_sorted.sort();
            for &k in &adj_sorted {
                let theta_ik = theta_i - x[self.idx_theta(k)];
                let c = theta_ik.cos();
                let s = theta_ik.sin();
                let gik = self.y_bus_re[self.yidx(i, k)];
                let bik = self.y_bus_im[self.yidx(i, k)];
                // dP_flow/dV_k = V_i * (G_ik*c + B_ik*s)
                vals[idx] = -(vi * (gik * c + bik * s));
                idx += 1;
            }

            // dP/d(theta_i)
            {
                let mut dp_dthi = 0.0;
                for k in 0..n {
                    if k == i {
                        continue;
                    }
                    let vk = x[self.idx_v(k)];
                    let theta_ik = theta_i - x[self.idx_theta(k)];
                    let c = theta_ik.cos();
                    let s = theta_ik.sin();
                    let gik = self.y_bus_re[self.yidx(i, k)];
                    let bik = self.y_bus_im[self.yidx(i, k)];
                    dp_dthi += vi * vk * (-gik * s + bik * c);
                }
                vals[idx] = -dp_dthi;
                idx += 1;
            }

            // dP/d(theta_k) for adjacent buses
            for &k in &adj_sorted {
                let vk = x[self.idx_v(k)];
                let theta_ik = theta_i - x[self.idx_theta(k)];
                let c = theta_ik.cos();
                let s = theta_ik.sin();
                let gik = self.y_bus_re[self.yidx(i, k)];
                let bik = self.y_bus_im[self.yidx(i, k)];
                // dP_flow/d(theta_k) = V_i * V_k * (G_ik*sin(theta_ik) - B_ik*cos(theta_ik))
                vals[idx] = -(vi * vk * (gik * s - bik * c));
                idx += 1;
            }

            // dP/d(Pg_g) = 1
            for &_g in &self.gen_at_bus[i] {
                vals[idx] = 1.0;
                idx += 1;
            }

            // ===== Q balance row =====
            // dQ/dV_i
            {
                let mut dq_dvi = 0.0;
                let bii = self.y_bus_im[self.yidx(i, i)];
                for k in 0..n {
                    let vk = x[self.idx_v(k)];
                    let theta_ik = theta_i - x[self.idx_theta(k)];
                    let c = theta_ik.cos();
                    let s = theta_ik.sin();
                    let gik = self.y_bus_re[self.yidx(i, k)];
                    let bik = self.y_bus_im[self.yidx(i, k)];
                    if k == i {
                        dq_dvi += -2.0 * vi * bii;
                    } else {
                        dq_dvi += vk * (gik * s - bik * c);
                    }
                }
                vals[idx] = -dq_dvi;
                idx += 1;
            }

            // dQ/dV_k
            for &k in &adj_sorted {
                let theta_ik = theta_i - x[self.idx_theta(k)];
                let c = theta_ik.cos();
                let s = theta_ik.sin();
                let gik = self.y_bus_re[self.yidx(i, k)];
                let bik = self.y_bus_im[self.yidx(i, k)];
                vals[idx] = -(vi * (gik * s - bik * c));
                idx += 1;
            }

            // dQ/d(theta_i)
            {
                let mut dq_dthi = 0.0;
                for k in 0..n {
                    if k == i {
                        continue;
                    }
                    let vk = x[self.idx_v(k)];
                    let theta_ik = theta_i - x[self.idx_theta(k)];
                    let c = theta_ik.cos();
                    let s = theta_ik.sin();
                    let gik = self.y_bus_re[self.yidx(i, k)];
                    let bik = self.y_bus_im[self.yidx(i, k)];
                    // d/d(theta_i) of V_i*V_k*(G_ik*sin(theta_ik) - B_ik*cos(theta_ik))
                    dq_dthi += vi * vk * (gik * c + bik * s);
                }
                vals[idx] = -dq_dthi;
                idx += 1;
            }

            // dQ/d(theta_k)
            for &k in &adj_sorted {
                let vk = x[self.idx_v(k)];
                let theta_ik = theta_i - x[self.idx_theta(k)];
                let c = theta_ik.cos();
                let s = theta_ik.sin();
                let gik = self.y_bus_re[self.yidx(i, k)];
                let bik = self.y_bus_im[self.yidx(i, k)];
                // d/d(theta_k) = -d/d(theta_i) for the (i,k) term
                vals[idx] = -(vi * vk * (-gik * c - bik * s));
                idx += 1;
            }

            // dQ/d(Qg_g) = 1
            for &_g in &self.gen_at_bus[i] {
                vals[idx] = 1.0;
                idx += 1;
            }
        }

        // Flow limit constraint Jacobians
        for (_fl_idx, &l) in self.flow_lim_branches.iter().enumerate() {
            let f = self.branches[l].from;
            let t = self.branches[l].to;
            let vf = x[self.idx_v(f)];
            let vt = x[self.idx_v(t)];
            let theta_ft = x[self.idx_theta(f)] - x[self.idx_theta(t)];
            let c = theta_ft.cos();
            let s = theta_ft.sin();

            let g_ff = self.branch_g_ff[l];
            let b_ff = self.branch_b_ff[l];
            let g_ft = self.branch_g_ft[l];
            let b_ft = self.branch_b_ft[l];
            let g_tt = self.branch_g_tt[l];
            let b_tt = self.branch_b_tt[l];
            let g_tf = self.branch_g_tf[l];
            let b_tf = self.branch_b_tf[l];

            // From-end flows
            let p_ft = vf * vf * g_ff + vf * vt * (g_ft * c + b_ft * s);
            let q_ft = -vf * vf * b_ff + vf * vt * (g_ft * s - b_ft * c);

            // Derivatives of P_ft, Q_ft
            let dp_ft_dvf = 2.0 * vf * g_ff + vt * (g_ft * c + b_ft * s);
            let dp_ft_dvt = vf * (g_ft * c + b_ft * s);
            let dp_ft_dthf = vf * vt * (-g_ft * s + b_ft * c);
            let dp_ft_dtht = vf * vt * (g_ft * s - b_ft * c);

            let dq_ft_dvf = -2.0 * vf * b_ff + vt * (g_ft * s - b_ft * c);
            let dq_ft_dvt = vf * (g_ft * s - b_ft * c);
            let dq_ft_dthf = vf * vt * (g_ft * c + b_ft * s);
            let dq_ft_dtht = vf * vt * (-g_ft * c - b_ft * s);

            // Constraint: S_ft^2 = P_ft^2 + Q_ft^2 - limit^2
            // d(S_ft^2)/dx = 2*P_ft * dP_ft/dx + 2*Q_ft * dQ_ft/dx
            // V_f
            vals[idx] = 2.0 * p_ft * dp_ft_dvf + 2.0 * q_ft * dq_ft_dvf;
            idx += 1;
            // V_t
            vals[idx] = 2.0 * p_ft * dp_ft_dvt + 2.0 * q_ft * dq_ft_dvt;
            idx += 1;
            // theta_f
            vals[idx] = 2.0 * p_ft * dp_ft_dthf + 2.0 * q_ft * dq_ft_dthf;
            idx += 1;
            // theta_t
            vals[idx] = 2.0 * p_ft * dp_ft_dtht + 2.0 * q_ft * dq_ft_dtht;
            idx += 1;

            // To-end flows
            let theta_tf = -theta_ft;
            let ct = theta_tf.cos();
            let st = theta_tf.sin();

            let p_tf = vt * vt * g_tt + vf * vt * (g_tf * ct + b_tf * st);
            let q_tf = -vt * vt * b_tt + vf * vt * (g_tf * st - b_tf * ct);

            let dp_tf_dvf = vt * (g_tf * ct + b_tf * st);
            let dp_tf_dvt = 2.0 * vt * g_tt + vf * (g_tf * ct + b_tf * st);
            let dp_tf_dthf = vf * vt * (g_tf * st - b_tf * ct);
            let dp_tf_dtht = vf * vt * (-g_tf * st + b_tf * ct);

            let dq_tf_dvf = vt * (g_tf * st - b_tf * ct);
            let dq_tf_dvt = -2.0 * vt * b_tt + vf * (g_tf * st - b_tf * ct);
            let dq_tf_dthf = vf * vt * (-g_tf * ct - b_tf * st);
            let dq_tf_dtht = vf * vt * (g_tf * ct + b_tf * st);

            // V_f
            vals[idx] = 2.0 * p_tf * dp_tf_dvf + 2.0 * q_tf * dq_tf_dvf;
            idx += 1;
            // V_t
            vals[idx] = 2.0 * p_tf * dp_tf_dvt + 2.0 * q_tf * dq_tf_dvt;
            idx += 1;
            // theta_f
            vals[idx] = 2.0 * p_tf * dp_tf_dthf + 2.0 * q_tf * dq_tf_dthf;
            idx += 1;
            // theta_t
            vals[idx] = 2.0 * p_tf * dp_tf_dtht + 2.0 * q_tf * dq_tf_dtht;
            idx += 1;
        }

        debug_assert_eq!(idx, vals.len());
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let nv = self.n_var();
        dense_lower_triangle(nv)
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        let n = self.n_bus;
        let bmva = self.base_mva;

        // Zero out
        for v in vals.iter_mut() {
            *v = 0.0;
        }

        // ---- Objective Hessian ----
        // f = sum_g c2*(baseMVA*Pg)^2 + c1*baseMVA*Pg + c0
        // d2f/dPg^2 = 2*c2*baseMVA^2
        for g in 0..self.n_gen {
            let pg_idx = self.idx_pg(g);
            let lt = lt_idx(pg_idx, pg_idx);
            vals[lt] += obj_factor * 2.0 * self.gens[g].c2 * bmva * bmva;
        }

        // ---- Power balance constraint Hessian ----
        for i in 0..n {
            let lambda_p = lambda[2 * i];
            let lambda_q = lambda[2 * i + 1];
            let vi = x[self.idx_v(i)];
            let theta_i = x[self.idx_theta(i)];
            let vi_idx = self.idx_v(i);
            let thi_idx = self.idx_theta(i);

            // Diagonal term (k == i): P_flow has V_i^2 * G_ii, Q_flow has -V_i^2 * B_ii
            let gii = self.y_bus_re[self.yidx(i, i)];
            let bii = self.y_bus_im[self.yidx(i, i)];
            // d2P/dV_i^2 = 2*G_ii, so d2g/dV_i^2 = -2*G_ii
            // d2Q/dV_i^2 = -2*B_ii, so d2g/dV_i^2 = 2*B_ii
            {
                let (r, c) = lt_order(vi_idx, vi_idx);
                vals[lt_idx(r, c)] += (-lambda_p) * 2.0 * gii + (-lambda_q) * (-2.0 * bii);
            }

            // Off-diagonal terms (k != i)
            for k in 0..n {
                if k == i {
                    continue;
                }
                let vk = x[self.idx_v(k)];
                let vk_idx = self.idx_v(k);
                let thk_idx = self.idx_theta(k);
                let theta_ik = theta_i - x[self.idx_theta(k)];
                let cos_ik = theta_ik.cos();
                let sin_ik = theta_ik.sin();
                let gik = self.y_bus_re[self.yidx(i, k)];
                let bik = self.y_bus_im[self.yidx(i, k)];

                // P balance term: V_i * V_k * (G_ik*cos(theta_ik) + B_ik*sin(theta_ik))
                let ac = gik * cos_ik + bik * sin_ik;
                let a_s = -gik * sin_ik + bik * cos_ik;

                // Q balance term: V_i * V_k * (G_ik*sin(theta_ik) - B_ik*cos(theta_ik))
                let bc = gik * sin_ik - bik * cos_ik;
                let bs = gik * cos_ik + bik * sin_ik;

                // Constraint is g = Pg - Pd - P_flow, so second derivatives have a minus sign
                // for the P_flow and Q_flow parts

                // H[V_i, V_k] += -lambda_p * Ac - lambda_q * Bc
                {
                    let (r, c) = lt_order(vi_idx, vk_idx);
                    vals[lt_idx(r, c)] += (-lambda_p) * ac + (-lambda_q) * bc;
                }

                // H[V_i, theta_i] += -lambda_p * V_k * As - lambda_q * V_k * Bs
                {
                    let (r, c) = lt_order(vi_idx, thi_idx);
                    vals[lt_idx(r, c)] += (-lambda_p) * vk * a_s + (-lambda_q) * vk * bs;
                }

                // H[V_i, theta_k] += -lambda_p * (-V_k * As) - lambda_q * (-V_k * Bs)
                {
                    let (r, c) = lt_order(vi_idx, thk_idx);
                    vals[lt_idx(r, c)] += (-lambda_p) * (-vk * a_s) + (-lambda_q) * (-vk * bs);
                }

                // H[V_k, theta_i] += -lambda_p * V_i * As - lambda_q * V_i * Bs
                {
                    let (r, c) = lt_order(vk_idx, thi_idx);
                    vals[lt_idx(r, c)] += (-lambda_p) * vi * a_s + (-lambda_q) * vi * bs;
                }

                // H[V_k, theta_k] += -lambda_p * (-V_i * As) - lambda_q * (-V_i * Bs)
                {
                    let (r, c) = lt_order(vk_idx, thk_idx);
                    vals[lt_idx(r, c)] += (-lambda_p) * (-vi * a_s) + (-lambda_q) * (-vi * bs);
                }

                // H[theta_i, theta_i] += -lambda_p * (-V_i * V_k * Ac) - lambda_q * (-V_i * V_k * Bc)
                {
                    let (r, c) = lt_order(thi_idx, thi_idx);
                    vals[lt_idx(r, c)] +=
                        (-lambda_p) * (-vi * vk * ac) + (-lambda_q) * (-vi * vk * bc);
                }

                // H[theta_i, theta_k] += -lambda_p * (V_i * V_k * Ac) - lambda_q * (V_i * V_k * Bc)
                {
                    let (r, c) = lt_order(thi_idx, thk_idx);
                    vals[lt_idx(r, c)] +=
                        (-lambda_p) * (vi * vk * ac) + (-lambda_q) * (vi * vk * bc);
                }

                // H[theta_k, theta_k]: only the contribution from bus i's constraint
                // Note: bus k's constraint will add its own diagonal contribution
                {
                    let (r, c) = lt_order(thk_idx, thk_idx);
                    vals[lt_idx(r, c)] +=
                        (-lambda_p) * (-vi * vk * ac) + (-lambda_q) * (-vi * vk * bc);
                }
            }
        }

        // ---- Flow limit constraint Hessian (numerical) ----
        let eps = 1e-7;
        let nnz_jac = self.jac_entries.len();

        for (fl_idx, &_l) in self.flow_lim_branches.iter().enumerate() {
            let con_from = 2 * n + 2 * fl_idx;
            let con_to = con_from + 1;

            let lam_from = lambda[con_from];
            let lam_to = lambda[con_to];

            if lam_from.abs() < 1e-20 && lam_to.abs() < 1e-20 {
                continue;
            }

            // Find the Jacobian entries for these two constraints
            // The flow constraint Jacobian entries are at the end of jac_entries
            // Each flow-limited branch contributes 8 entries (4 for from, 4 for to)
            let flow_jac_start = nnz_jac - 8 * (self.n_flow_lim - fl_idx);
            let from_start = flow_jac_start;
            let to_start = flow_jac_start + 4;

            // Get the 4 variable indices for this branch's flow constraints
            let var_indices: Vec<usize> = (0..4)
                .map(|j| self.jac_entries[from_start + j].col)
                .collect();

            // Compute Jacobian values at the current point
            let mut jac_base = vec![0.0; nnz_jac];
            self.jacobian_values(x, true, &mut jac_base);

            // For each of the 4 variables, perturb and compute numerical Hessian
            let mut x_pert = x.to_vec();
            for a in 0..4 {
                let va = var_indices[a];
                let x_orig = x_pert[va];
                x_pert[va] = x_orig + eps;

                let mut jac_pert = vec![0.0; nnz_jac];
                self.jacobian_values(&x_pert, true, &mut jac_pert);

                // For variables b <= a (lower triangle)
                for b in 0..=a {
                    let vb = var_indices[b];

                    // Numerical second derivative via finite difference on Jacobian
                    // H[va, vb] ≈ (dg/dvb(x+eps*ea) - dg/dvb(x)) / eps

                    // Find the Jacobian entry for constraint con_from w.r.t. variable vb
                    let jac_idx_from = from_start + b;
                    let jac_idx_to = to_start + b;

                    let h_from = (jac_pert[jac_idx_from] - jac_base[jac_idx_from]) / eps;
                    let h_to = (jac_pert[jac_idx_to] - jac_base[jac_idx_to]) / eps;

                    let (r, c) = lt_order(va, vb);
                    vals[lt_idx(r, c)] += lam_from * h_from + lam_to * h_to;
                }

                x_pert[va] = x_orig;
            }
        }
    }
}

// ===========================================================================
// Dense lower-triangle helpers
// ===========================================================================

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

/// Index into the dense lower-triangle storage for entry (i, j) where i >= j
fn lt_idx(i: usize, j: usize) -> usize {
    debug_assert!(i >= j, "lt_idx requires i >= j, got i={}, j={}", i, j);
    i * (i + 1) / 2 + j
}

/// Ensure (row, col) ordering for lower triangle: returns (max, min)
fn lt_order(a: usize, b: usize) -> (usize, usize) {
    if a >= b {
        (a, b)
    } else {
        (b, a)
    }
}

// ===========================================================================
// Test case constructors
// ===========================================================================

pub fn case3_lmbd() -> AcopfProblem {
    let base_mva = 100.0;

    let buses = vec![
        BusData {
            pd: 110.0,
            qd: 40.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.9,
            v_max: 1.1,
            bus_type: 3,
        },
        BusData {
            pd: 110.0,
            qd: 40.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.9,
            v_max: 1.1,
            bus_type: 2,
        },
        BusData {
            pd: 95.0,
            qd: 50.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.9,
            v_max: 1.1,
            bus_type: 2,
        },
    ];

    let gens = vec![
        GenData {
            bus: 0,
            pg_max: 2000.0,
            pg_min: 0.0,
            qg_max: 1000.0,
            qg_min: -1000.0,
            c2: 0.11,
            c1: 5.0,
            c0: 0.0,
        },
        GenData {
            bus: 1,
            pg_max: 2000.0,
            pg_min: 0.0,
            qg_max: 1000.0,
            qg_min: -1000.0,
            c2: 0.085,
            c1: 1.2,
            c0: 0.0,
        },
        GenData {
            bus: 2,
            pg_max: 0.0,
            pg_min: 0.0,
            qg_max: 1000.0,
            qg_min: -1000.0,
            c2: 0.0,
            c1: 0.0,
            c0: 0.0,
        },
    ];

    let branches = vec![
        BranchData {
            from: 0,
            to: 2,
            r: 0.065,
            x: 0.62,
            b_ch: 0.45,
            rate_a: 9000.0,
            tap: 0.0,
        },
        BranchData {
            from: 2,
            to: 1,
            r: 0.025,
            x: 0.75,
            b_ch: 0.70,
            rate_a: 50.0,
            tap: 0.0,
        },
        BranchData {
            from: 0,
            to: 1,
            r: 0.042,
            x: 0.90,
            b_ch: 0.30,
            rate_a: 9000.0,
            tap: 0.0,
        },
    ];

    AcopfProblem::new("case3_lmbd", base_mva, buses, gens, branches)
}

pub fn case5_pjm() -> AcopfProblem {
    let base_mva = 100.0;

    let buses = vec![
        BusData {
            pd: 0.0,
            qd: 0.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.9,
            v_max: 1.1,
            bus_type: 2,
        },
        BusData {
            pd: 300.0,
            qd: 98.61,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.9,
            v_max: 1.1,
            bus_type: 1,
        },
        BusData {
            pd: 300.0,
            qd: 98.61,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.9,
            v_max: 1.1,
            bus_type: 2,
        },
        BusData {
            pd: 400.0,
            qd: 131.47,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.9,
            v_max: 1.1,
            bus_type: 3,
        },
        BusData {
            pd: 0.0,
            qd: 0.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.9,
            v_max: 1.1,
            bus_type: 2,
        },
    ];

    let gens = vec![
        GenData {
            bus: 0,
            pg_max: 40.0,
            pg_min: 0.0,
            qg_max: 30.0,
            qg_min: -30.0,
            c2: 0.0,
            c1: 14.0,
            c0: 0.0,
        },
        GenData {
            bus: 0,
            pg_max: 170.0,
            pg_min: 0.0,
            qg_max: 127.5,
            qg_min: -127.5,
            c2: 0.0,
            c1: 15.0,
            c0: 0.0,
        },
        GenData {
            bus: 2,
            pg_max: 520.0,
            pg_min: 0.0,
            qg_max: 390.0,
            qg_min: -390.0,
            c2: 0.0,
            c1: 30.0,
            c0: 0.0,
        },
        GenData {
            bus: 3,
            pg_max: 200.0,
            pg_min: 0.0,
            qg_max: 150.0,
            qg_min: -150.0,
            c2: 0.0,
            c1: 40.0,
            c0: 0.0,
        },
        GenData {
            bus: 4,
            pg_max: 600.0,
            pg_min: 0.0,
            qg_max: 450.0,
            qg_min: -450.0,
            c2: 0.0,
            c1: 10.0,
            c0: 0.0,
        },
    ];

    let branches = vec![
        BranchData {
            from: 0,
            to: 1,
            r: 0.00281,
            x: 0.0281,
            b_ch: 0.00712,
            rate_a: 400.0,
            tap: 0.0,
        },
        BranchData {
            from: 0,
            to: 3,
            r: 0.00304,
            x: 0.0304,
            b_ch: 0.00658,
            rate_a: 426.0,
            tap: 0.0,
        },
        BranchData {
            from: 0,
            to: 4,
            r: 0.00064,
            x: 0.0064,
            b_ch: 0.03126,
            rate_a: 426.0,
            tap: 0.0,
        },
        BranchData {
            from: 1,
            to: 2,
            r: 0.00108,
            x: 0.0108,
            b_ch: 0.01852,
            rate_a: 426.0,
            tap: 0.0,
        },
        BranchData {
            from: 2,
            to: 3,
            r: 0.00297,
            x: 0.0297,
            b_ch: 0.00674,
            rate_a: 426.0,
            tap: 0.0,
        },
        BranchData {
            from: 3,
            to: 4,
            r: 0.00297,
            x: 0.0297,
            b_ch: 0.00674,
            rate_a: 240.0,
            tap: 0.0,
        },
    ];

    AcopfProblem::new("case5_pjm", base_mva, buses, gens, branches)
}

pub fn case14_ieee() -> AcopfProblem {
    let base_mva = 100.0;

    let buses = vec![
        BusData {
            pd: 0.0,
            qd: 0.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 3,
        },
        BusData {
            pd: 21.7,
            qd: 12.7,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 2,
        },
        BusData {
            pd: 94.2,
            qd: 19.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 2,
        },
        BusData {
            pd: 47.8,
            qd: -3.9,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 7.6,
            qd: 1.6,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 11.2,
            qd: 7.5,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 2,
        },
        BusData {
            pd: 0.0,
            qd: 0.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 0.0,
            qd: 0.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 2,
        },
        BusData {
            pd: 29.5,
            qd: 16.6,
            gs: 0.0,
            bs: 19.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 9.0,
            qd: 5.8,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 3.5,
            qd: 1.8,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 6.1,
            qd: 1.6,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 13.5,
            qd: 5.8,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 14.9,
            qd: 5.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
    ];

    let gens = vec![
        GenData {
            bus: 0,
            pg_max: 340.0,
            pg_min: 0.0,
            qg_max: 10.0,
            qg_min: 0.0,
            c2: 0.0,
            c1: 7.920951,
            c0: 0.0,
        },
        GenData {
            bus: 1,
            pg_max: 59.0,
            pg_min: 0.0,
            qg_max: 30.0,
            qg_min: -30.0,
            c2: 0.0,
            c1: 23.269494,
            c0: 0.0,
        },
        GenData {
            bus: 2,
            pg_max: 0.0,
            pg_min: 0.0,
            qg_max: 40.0,
            qg_min: 0.0,
            c2: 0.0,
            c1: 0.0,
            c0: 0.0,
        },
        GenData {
            bus: 5,
            pg_max: 0.0,
            pg_min: 0.0,
            qg_max: 24.0,
            qg_min: -6.0,
            c2: 0.0,
            c1: 0.0,
            c0: 0.0,
        },
        GenData {
            bus: 7,
            pg_max: 0.0,
            pg_min: 0.0,
            qg_max: 24.0,
            qg_min: -6.0,
            c2: 0.0,
            c1: 0.0,
            c0: 0.0,
        },
    ];

    let branches = vec![
        BranchData {
            from: 0,
            to: 1,
            r: 0.01938,
            x: 0.05917,
            b_ch: 0.0528,
            rate_a: 472.0,
            tap: 0.0,
        },
        BranchData {
            from: 0,
            to: 4,
            r: 0.05403,
            x: 0.22304,
            b_ch: 0.0492,
            rate_a: 128.0,
            tap: 0.0,
        },
        BranchData {
            from: 1,
            to: 2,
            r: 0.04699,
            x: 0.19797,
            b_ch: 0.0438,
            rate_a: 145.0,
            tap: 0.0,
        },
        BranchData {
            from: 1,
            to: 3,
            r: 0.05811,
            x: 0.17632,
            b_ch: 0.034,
            rate_a: 158.0,
            tap: 0.0,
        },
        BranchData {
            from: 1,
            to: 4,
            r: 0.05695,
            x: 0.17388,
            b_ch: 0.0346,
            rate_a: 161.0,
            tap: 0.0,
        },
        BranchData {
            from: 2,
            to: 3,
            r: 0.06701,
            x: 0.17103,
            b_ch: 0.0128,
            rate_a: 160.0,
            tap: 0.0,
        },
        BranchData {
            from: 3,
            to: 4,
            r: 0.01335,
            x: 0.04211,
            b_ch: 0.0,
            rate_a: 664.0,
            tap: 0.0,
        },
        BranchData {
            from: 3,
            to: 6,
            r: 0.0,
            x: 0.20912,
            b_ch: 0.0,
            rate_a: 141.0,
            tap: 0.978,
        },
        BranchData {
            from: 3,
            to: 8,
            r: 0.0,
            x: 0.55618,
            b_ch: 0.0,
            rate_a: 53.0,
            tap: 0.969,
        },
        BranchData {
            from: 4,
            to: 5,
            r: 0.0,
            x: 0.25202,
            b_ch: 0.0,
            rate_a: 117.0,
            tap: 0.932,
        },
        BranchData {
            from: 5,
            to: 10,
            r: 0.09498,
            x: 0.1989,
            b_ch: 0.0,
            rate_a: 134.0,
            tap: 0.0,
        },
        BranchData {
            from: 5,
            to: 11,
            r: 0.12291,
            x: 0.25581,
            b_ch: 0.0,
            rate_a: 104.0,
            tap: 0.0,
        },
        BranchData {
            from: 5,
            to: 12,
            r: 0.06615,
            x: 0.13027,
            b_ch: 0.0,
            rate_a: 201.0,
            tap: 0.0,
        },
        BranchData {
            from: 6,
            to: 7,
            r: 0.0,
            x: 0.17615,
            b_ch: 0.0,
            rate_a: 167.0,
            tap: 0.0,
        },
        BranchData {
            from: 6,
            to: 8,
            r: 0.0,
            x: 0.11001,
            b_ch: 0.0,
            rate_a: 267.0,
            tap: 0.0,
        },
        BranchData {
            from: 8,
            to: 9,
            r: 0.03181,
            x: 0.0845,
            b_ch: 0.0,
            rate_a: 325.0,
            tap: 0.0,
        },
        BranchData {
            from: 8,
            to: 13,
            r: 0.12711,
            x: 0.27038,
            b_ch: 0.0,
            rate_a: 99.0,
            tap: 0.0,
        },
        BranchData {
            from: 9,
            to: 10,
            r: 0.08205,
            x: 0.19207,
            b_ch: 0.0,
            rate_a: 141.0,
            tap: 0.0,
        },
        BranchData {
            from: 11,
            to: 12,
            r: 0.22092,
            x: 0.19988,
            b_ch: 0.0,
            rate_a: 99.0,
            tap: 0.0,
        },
        BranchData {
            from: 12,
            to: 13,
            r: 0.17093,
            x: 0.34802,
            b_ch: 0.0,
            rate_a: 76.0,
            tap: 0.0,
        },
    ];

    AcopfProblem::new("case14_ieee", base_mva, buses, gens, branches)
}

pub fn case30_ieee() -> AcopfProblem {
    let base_mva = 100.0;

    let buses = vec![
        BusData {
            pd: 0.0,
            qd: 0.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 3,
        },
        BusData {
            pd: 21.7,
            qd: 12.7,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 2,
        },
        BusData {
            pd: 2.4,
            qd: 1.2,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 7.6,
            qd: 1.6,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 94.2,
            qd: 19.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 2,
        },
        BusData {
            pd: 0.0,
            qd: 0.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 22.8,
            qd: 10.9,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 30.0,
            qd: 30.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 2,
        },
        BusData {
            pd: 0.0,
            qd: 0.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 5.8,
            qd: 2.0,
            gs: 0.0,
            bs: 19.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 0.0,
            qd: 0.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 2,
        },
        BusData {
            pd: 11.2,
            qd: 7.5,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 0.0,
            qd: 0.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 2,
        },
        BusData {
            pd: 6.2,
            qd: 1.6,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 8.2,
            qd: 2.5,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 3.5,
            qd: 1.8,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 9.0,
            qd: 5.8,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 3.2,
            qd: 0.9,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 9.5,
            qd: 3.4,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 2.2,
            qd: 0.7,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 17.5,
            qd: 11.2,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 0.0,
            qd: 0.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 3.2,
            qd: 1.6,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 8.7,
            qd: 6.7,
            gs: 0.0,
            bs: 4.3,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 0.0,
            qd: 0.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 3.5,
            qd: 2.3,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 0.0,
            qd: 0.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 0.0,
            qd: 0.0,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 2.4,
            qd: 0.9,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
        BusData {
            pd: 10.6,
            qd: 1.9,
            gs: 0.0,
            bs: 0.0,
            v_min: 0.94,
            v_max: 1.06,
            bus_type: 1,
        },
    ];

    let gens = vec![
        GenData {
            bus: 0,
            pg_max: 271.0,
            pg_min: 0.0,
            qg_max: 10.0,
            qg_min: 0.0,
            c2: 0.0,
            c1: 18.421528,
            c0: 0.0,
        },
        GenData {
            bus: 1,
            pg_max: 92.0,
            pg_min: 0.0,
            qg_max: 46.0,
            qg_min: -40.0,
            c2: 0.0,
            c1: 52.182254,
            c0: 0.0,
        },
        GenData {
            bus: 4,
            pg_max: 0.0,
            pg_min: 0.0,
            qg_max: 40.0,
            qg_min: -40.0,
            c2: 0.0,
            c1: 0.0,
            c0: 0.0,
        },
        GenData {
            bus: 7,
            pg_max: 0.0,
            pg_min: 0.0,
            qg_max: 40.0,
            qg_min: -10.0,
            c2: 0.0,
            c1: 0.0,
            c0: 0.0,
        },
        GenData {
            bus: 10,
            pg_max: 0.0,
            pg_min: 0.0,
            qg_max: 24.0,
            qg_min: -6.0,
            c2: 0.0,
            c1: 0.0,
            c0: 0.0,
        },
        GenData {
            bus: 12,
            pg_max: 0.0,
            pg_min: 0.0,
            qg_max: 24.0,
            qg_min: -6.0,
            c2: 0.0,
            c1: 0.0,
            c0: 0.0,
        },
    ];

    let branches = vec![
        BranchData {
            from: 0,
            to: 1,
            r: 0.0192,
            x: 0.0575,
            b_ch: 0.0528,
            rate_a: 138.0,
            tap: 0.0,
        },
        BranchData {
            from: 0,
            to: 2,
            r: 0.0452,
            x: 0.1652,
            b_ch: 0.0408,
            rate_a: 152.0,
            tap: 0.0,
        },
        BranchData {
            from: 1,
            to: 3,
            r: 0.057,
            x: 0.1737,
            b_ch: 0.0368,
            rate_a: 139.0,
            tap: 0.0,
        },
        BranchData {
            from: 2,
            to: 3,
            r: 0.0132,
            x: 0.0379,
            b_ch: 0.0084,
            rate_a: 135.0,
            tap: 0.0,
        },
        BranchData {
            from: 1,
            to: 4,
            r: 0.0472,
            x: 0.1983,
            b_ch: 0.0418,
            rate_a: 144.0,
            tap: 0.0,
        },
        BranchData {
            from: 1,
            to: 5,
            r: 0.0581,
            x: 0.1763,
            b_ch: 0.0374,
            rate_a: 139.0,
            tap: 0.0,
        },
        BranchData {
            from: 3,
            to: 5,
            r: 0.0119,
            x: 0.0414,
            b_ch: 0.009,
            rate_a: 148.0,
            tap: 0.0,
        },
        BranchData {
            from: 4,
            to: 6,
            r: 0.046,
            x: 0.116,
            b_ch: 0.0204,
            rate_a: 127.0,
            tap: 0.0,
        },
        BranchData {
            from: 5,
            to: 6,
            r: 0.0267,
            x: 0.082,
            b_ch: 0.017,
            rate_a: 140.0,
            tap: 0.0,
        },
        BranchData {
            from: 5,
            to: 7,
            r: 0.012,
            x: 0.042,
            b_ch: 0.009,
            rate_a: 148.0,
            tap: 0.0,
        },
        BranchData {
            from: 5,
            to: 8,
            r: 0.0,
            x: 0.208,
            b_ch: 0.0,
            rate_a: 142.0,
            tap: 0.978,
        },
        BranchData {
            from: 5,
            to: 9,
            r: 0.0,
            x: 0.556,
            b_ch: 0.0,
            rate_a: 53.0,
            tap: 0.969,
        },
        BranchData {
            from: 8,
            to: 10,
            r: 0.0,
            x: 0.208,
            b_ch: 0.0,
            rate_a: 142.0,
            tap: 1.0,
        },
        BranchData {
            from: 8,
            to: 9,
            r: 0.0,
            x: 0.11,
            b_ch: 0.0,
            rate_a: 267.0,
            tap: 1.0,
        },
        BranchData {
            from: 3,
            to: 11,
            r: 0.0,
            x: 0.256,
            b_ch: 0.0,
            rate_a: 115.0,
            tap: 0.932,
        },
        BranchData {
            from: 11,
            to: 12,
            r: 0.0,
            x: 0.14,
            b_ch: 0.0,
            rate_a: 210.0,
            tap: 1.0,
        },
        BranchData {
            from: 11,
            to: 13,
            r: 0.1231,
            x: 0.2559,
            b_ch: 0.0,
            rate_a: 29.0,
            tap: 0.0,
        },
        BranchData {
            from: 11,
            to: 14,
            r: 0.0662,
            x: 0.1304,
            b_ch: 0.0,
            rate_a: 29.0,
            tap: 0.0,
        },
        BranchData {
            from: 11,
            to: 15,
            r: 0.0945,
            x: 0.1987,
            b_ch: 0.0,
            rate_a: 30.0,
            tap: 0.0,
        },
        BranchData {
            from: 13,
            to: 14,
            r: 0.221,
            x: 0.1997,
            b_ch: 0.0,
            rate_a: 20.0,
            tap: 0.0,
        },
        BranchData {
            from: 15,
            to: 16,
            r: 0.0524,
            x: 0.1923,
            b_ch: 0.0,
            rate_a: 38.0,
            tap: 0.0,
        },
        BranchData {
            from: 14,
            to: 17,
            r: 0.1073,
            x: 0.2185,
            b_ch: 0.0,
            rate_a: 29.0,
            tap: 0.0,
        },
        BranchData {
            from: 17,
            to: 18,
            r: 0.0639,
            x: 0.1292,
            b_ch: 0.0,
            rate_a: 29.0,
            tap: 0.0,
        },
        BranchData {
            from: 18,
            to: 19,
            r: 0.034,
            x: 0.068,
            b_ch: 0.0,
            rate_a: 29.0,
            tap: 0.0,
        },
        BranchData {
            from: 9,
            to: 19,
            r: 0.0936,
            x: 0.209,
            b_ch: 0.0,
            rate_a: 30.0,
            tap: 0.0,
        },
        BranchData {
            from: 9,
            to: 16,
            r: 0.0324,
            x: 0.0845,
            b_ch: 0.0,
            rate_a: 33.0,
            tap: 0.0,
        },
        BranchData {
            from: 9,
            to: 20,
            r: 0.0348,
            x: 0.0749,
            b_ch: 0.0,
            rate_a: 30.0,
            tap: 0.0,
        },
        BranchData {
            from: 9,
            to: 21,
            r: 0.0727,
            x: 0.1499,
            b_ch: 0.0,
            rate_a: 29.0,
            tap: 0.0,
        },
        BranchData {
            from: 20,
            to: 21,
            r: 0.0116,
            x: 0.0236,
            b_ch: 0.0,
            rate_a: 29.0,
            tap: 0.0,
        },
        BranchData {
            from: 14,
            to: 22,
            r: 0.1,
            x: 0.202,
            b_ch: 0.0,
            rate_a: 29.0,
            tap: 0.0,
        },
        BranchData {
            from: 21,
            to: 23,
            r: 0.115,
            x: 0.179,
            b_ch: 0.0,
            rate_a: 26.0,
            tap: 0.0,
        },
        BranchData {
            from: 22,
            to: 23,
            r: 0.132,
            x: 0.27,
            b_ch: 0.0,
            rate_a: 29.0,
            tap: 0.0,
        },
        BranchData {
            from: 23,
            to: 24,
            r: 0.1885,
            x: 0.3292,
            b_ch: 0.0,
            rate_a: 27.0,
            tap: 0.0,
        },
        BranchData {
            from: 24,
            to: 25,
            r: 0.2544,
            x: 0.38,
            b_ch: 0.0,
            rate_a: 25.0,
            tap: 0.0,
        },
        BranchData {
            from: 24,
            to: 26,
            r: 0.1093,
            x: 0.2087,
            b_ch: 0.0,
            rate_a: 28.0,
            tap: 0.0,
        },
        BranchData {
            from: 27,
            to: 26,
            r: 0.0,
            x: 0.396,
            b_ch: 0.0,
            rate_a: 75.0,
            tap: 0.968,
        },
        BranchData {
            from: 26,
            to: 28,
            r: 0.2198,
            x: 0.4153,
            b_ch: 0.0,
            rate_a: 28.0,
            tap: 0.0,
        },
        BranchData {
            from: 26,
            to: 29,
            r: 0.3202,
            x: 0.6027,
            b_ch: 0.0,
            rate_a: 28.0,
            tap: 0.0,
        },
        BranchData {
            from: 28,
            to: 29,
            r: 0.2399,
            x: 0.4533,
            b_ch: 0.0,
            rate_a: 28.0,
            tap: 0.0,
        },
        BranchData {
            from: 7,
            to: 27,
            r: 0.0636,
            x: 0.2,
            b_ch: 0.0428,
            rate_a: 140.0,
            tap: 0.0,
        },
        BranchData {
            from: 5,
            to: 27,
            r: 0.0169,
            x: 0.0599,
            b_ch: 0.013,
            rate_a: 149.0,
            tap: 0.0,
        },
    ];

    AcopfProblem::new("case30_ieee", base_mva, buses, gens, branches)
}
