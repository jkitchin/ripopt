use ripopt::nl::{parse_nl_file, write_sol, NlProblem};
use ripopt::SolverOptions;
use std::fs;
use std::io::BufWriter;

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();

    // Handle -v / --version flag for AMPL solver protocol
    if args.len() >= 2 && (args[1] == "-v" || args[1] == "--version" || args[1] == "-AMPL") {
        if args.len() == 2 && (args[1] == "-v" || args[1] == "--version") {
            println!("ripopt {}", env!("CARGO_PKG_VERSION"));
            return;
        }
    }

    // Handle -h / --help flag
    if args.len() >= 2 && (args[1] == "-h" || args[1] == "--help") {
        print_help();
        return;
    }

    if args.len() < 2 {
        eprintln!("Usage: ripopt <problem.nl> [-AMPL] [key=value ...]");
        eprintln!("Try 'ripopt --help' for more information.");
        std::process::exit(1);
    }

    let nl_path = &args[1];
    let mut options = SolverOptions::default();

    // Parse key=value options from command line
    for arg in &args[2..] {
        if arg == "-AMPL" || arg == "--AMPL" {
            continue; // AMPL mode flag, acknowledged
        }
        if let Some((key, value)) = arg.split_once('=') {
            apply_option(&mut options, key.trim(), value.trim());
        }
    }

    // Read and parse NL file
    let content = match fs::read_to_string(nl_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading {}: {}", nl_path, e);
            std::process::exit(1);
        }
    };

    let nl_data = match parse_nl_file(&content) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error parsing NL file: {}", e);
            std::process::exit(1);
        }
    };

    let n_vars = nl_data.header.n_vars;
    let n_constrs = nl_data.header.n_constrs;

    let problem = match NlProblem::from_nl_data(nl_data) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    // Solve
    let result = ripopt::solve(&problem, &options);

    // Print summary to stdout
    println!(
        "ripopt {}: {} after {} iterations",
        env!("CARGO_PKG_VERSION"),
        match result.status {
            ripopt::SolveStatus::Optimal => "Optimal",
            ripopt::SolveStatus::Acceptable => "Acceptable",
            ripopt::SolveStatus::Infeasible => "Infeasible",
            ripopt::SolveStatus::LocalInfeasibility => "LocalInfeasibility",
            ripopt::SolveStatus::MaxIterations => "MaxIterations",
            ripopt::SolveStatus::NumericalError => "NumericalError",
            ripopt::SolveStatus::Unbounded => "Unbounded",
            ripopt::SolveStatus::RestorationFailed => "RestorationFailed",
            ripopt::SolveStatus::InternalError => "InternalError",
            ripopt::SolveStatus::EvaluationError => "EvaluationError",
            ripopt::SolveStatus::UserRequestedStop => "UserRequestedStop",
        },
        result.iterations
    );
    println!("Objective: {:.15e}", result.objective);

    // Print diagnostics to stderr
    result.diagnostics.print_summary(result.status, result.iterations);

    // Write SOL file (replace .nl extension with .sol)
    let sol_path = if nl_path.ends_with(".nl") {
        format!("{}sol", &nl_path[..nl_path.len() - 2])
    } else {
        format!("{}.sol", nl_path)
    };

    let sol_file = match fs::File::create(&sol_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error creating {}: {}", sol_path, e);
            std::process::exit(1);
        }
    };

    let mut writer = BufWriter::new(sol_file);
    if let Err(e) = write_sol(&mut writer, &result, n_vars, n_constrs) {
        eprintln!("Error writing SOL file: {}", e);
        std::process::exit(1);
    }
}

fn print_help() {
    println!("ripopt {} — primal-dual interior point NLP solver", env!("CARGO_PKG_VERSION"));
    println!();
    println!("USAGE:");
    println!("    ripopt <problem.nl> [-AMPL] [key=value ...]");
    println!();
    println!("FLAGS:");
    println!("    -h, --help       Print this help message and exit");
    println!("    -v, --version    Print version and exit");
    println!("    -AMPL            AMPL solver protocol mode");
    println!();
    println!("OPTIONS:");
    println!();
    println!("  Convergence");
    println!("    tol=<float>                          Optimality convergence tolerance [1e-8]");
    println!("    max_iter=<int>                       Maximum iterations [3000]");
    println!("    max_wall_time=<float>                Max wall-clock time in seconds (0=no limit) [0.0]");
    println!("    stall_iter_limit=<int>               Iters without 1% improvement before stall (0=off) [30]");
    println!();
    println!("  Constraint & Dual Tolerances");
    println!("    constr_viol_tol=<float>              Constraint violation tolerance [1e-4]");
    println!("    dual_inf_tol=<float>                 Dual infeasibility tolerance [1.0]");
    println!("    compl_inf_tol=<float>                Complementarity tolerance [1e-4]");
    println!();
    println!("  Barrier Parameter");
    println!("    mu_init=<float>                      Initial barrier parameter [0.1]");
    println!("    mu_min=<float>                       Minimum barrier parameter [1e-11]");
    println!("    mu_strategy=<str>                    Barrier strategy: adaptive or monotone [adaptive]");
    println!("    mu_linear_decrease_factor=<float>    Monotone-mode decrease factor [0.2]");
    println!("    mu_superlinear_decrease_power=<float> Superlinear decrease exponent [1.5]");
    println!("    kappa=<float>                        Adaptive divisor: mu = avg_compl/kappa [10.0]");
    println!("    mu_allow_increase=<bool>             Allow mu increase after restoration [yes]");
    println!("    adaptive_mu_monotone_init_factor=<float> Fixed-mode init factor [0.8]");
    println!("    barrier_tol_factor=<float>           Fixed-mode mu tolerance factor [10.0]");
    println!();
    println!("  Bound Management");
    println!("    bound_push=<float>                   Initial-point bound push [1e-2]");
    println!("    bound_frac=<float>                   Initial-point bound fraction [1e-2]");
    println!("    slack_bound_push=<float>             Slack variable bound push [1e-2]");
    println!("    slack_bound_frac=<float>             Slack variable bound fraction [1e-2]");
    println!("    tau_min=<float>                      Fraction-to-boundary minimum [0.99]");
    println!("    nlp_lower_bound_inf=<float>          Treat bounds below this as -inf [-1e19]");
    println!("    nlp_upper_bound_inf=<float>          Treat bounds above this as +inf [1e19]");
    println!();
    println!("  Warm Start");
    println!("    warm_start_init_point=<bool>         Enable warm-start initialization [no]");
    println!("    warm_start_bound_push=<float>        Warm-start bound push [1e-3]");
    println!("    warm_start_bound_frac=<float>        Warm-start bound fraction [1e-3]");
    println!("    warm_start_mult_bound_push=<float>   Warm-start multiplier bound push [1e-3]");
    println!("    warm_start_target_mu=<float>         Override cold-start mu ramp at warm start [unset]");
    println!();
    println!("  Multiplier Initialization");
    println!("    least_squares_mult_init=<bool>       Least-squares multiplier init [yes]");
    println!("    constr_mult_init_max=<float>         Max abs value for LS multiplier init [1000.0]");
    println!();
    println!("  Step Control & Line Search");
    println!("    max_soc=<int>                        Max second-order correction steps [4]");
    println!("    watchdog_shortened_iter_trigger=<int> Shortened steps before watchdog [10]");
    println!("    watchdog_trial_iter_max=<int>        Max watchdog trial iterations [5]");
    println!();
    println!("  Constraint Handling");
    println!("    constraint_slack_barrier=<bool>      Slack log-barriers in filter merit [yes]");
    println!("    detect_linear_constraints=<bool>     Detect linear constraints [yes]");
    println!("    restoration_max_iter=<int>           Max restoration subproblem iterations [200]");
    println!("    disable_nlp_restoration=<bool>       Disable NLP restoration [no]");
    println!();
    println!("  Hessian & Linear Algebra");
    println!("    hessian_approximation=<str>          exact or limited-memory (L-BFGS) [exact]");
    println!("    linear_solver=<str>                  direct, iterative (MINRES), or hybrid [direct]");
    println!("    sparse_threshold=<int>               Sparse solver if n+m >= threshold [110]");
    println!("    mehrotra_pc=<bool>                   Mehrotra predictor-corrector [no]");
    println!("    gondzio_mcc_max=<int>                Max Gondzio centrality corrections [3]");
    println!();
    println!("  Fallback Strategies");
    println!("    enable_slack_fallback=<bool>         Retry with explicit slack variables [yes]");
    println!("    enable_lbfgs_fallback=<bool>         L-BFGS fallback for unconstrained [yes]");
    println!("    enable_lbfgs_hessian_fallback=<bool> Retry with L-BFGS Hessian [yes]");
    println!();
    println!("  Preprocessing & Diagnostics");
    println!("    enable_preprocessing=<bool>          Eliminate fixed vars & redundant constraints [yes]");
    println!("    proactive_infeasibility_detection=<bool> Early infeasibility detection [no]");
    println!("    print_level=<int>                    Verbosity: 0=silent, 5=verbose [5]");
    println!("    early_stall_timeout=<float>          Max seconds for first 3 iters (0=off) [120.0]");
    println!("    mu_oracle_quality_function=<bool>    Use quality function for mu selection [no]");
    println!();
    println!("  Boolean values accept: yes, true, 1 (anything else is false).");
}

/// Apply a key=value option to SolverOptions.
fn apply_option(opts: &mut SolverOptions, key: &str, value: &str) {
    match key {
        "tol" => {
            if let Ok(v) = value.parse() {
                opts.tol = v;
            }
        }
        "max_iter" => {
            if let Ok(v) = value.parse() {
                opts.max_iter = v;
            }
        }
        "mu_init" => {
            if let Ok(v) = value.parse() {
                opts.mu_init = v;
            }
        }
        "print_level" => {
            if let Ok(v) = value.parse() {
                opts.print_level = v;
            }
        }
        "max_wall_time" => {
            if let Ok(v) = value.parse() {
                opts.max_wall_time = v;
            }
        }
        "bound_push" => {
            if let Ok(v) = value.parse() {
                opts.bound_push = v;
            }
        }
        "bound_frac" => {
            if let Ok(v) = value.parse() {
                opts.bound_frac = v;
            }
        }
        "slack_bound_push" => {
            if let Ok(v) = value.parse() {
                opts.slack_bound_push = v;
            }
        }
        "slack_bound_frac" => {
            if let Ok(v) = value.parse() {
                opts.slack_bound_frac = v;
            }
        }
        "constr_viol_tol" => {
            if let Ok(v) = value.parse() {
                opts.constr_viol_tol = v;
            }
        }
        "dual_inf_tol" => {
            if let Ok(v) = value.parse() {
                opts.dual_inf_tol = v;
            }
        }
        "compl_inf_tol" => {
            if let Ok(v) = value.parse() {
                opts.compl_inf_tol = v;
            }
        }
        "kappa" => {
            if let Ok(v) = value.parse() {
                opts.kappa = v;
            }
        }
        "mu_linear_decrease_factor" => {
            if let Ok(v) = value.parse() {
                opts.mu_linear_decrease_factor = v;
            }
        }
        "mu_superlinear_decrease_power" => {
            if let Ok(v) = value.parse() {
                opts.mu_superlinear_decrease_power = v;
            }
        }
        "mu_min" => {
            if let Ok(v) = value.parse() {
                opts.mu_min = v;
            }
        }
        "tau_min" => {
            if let Ok(v) = value.parse() {
                opts.tau_min = v;
            }
        }
        "mu_allow_increase" => {
            opts.mu_allow_increase = value == "yes" || value == "true" || value == "1";
        }
        "adaptive_mu_monotone_init_factor" => {
            if let Ok(v) = value.parse() {
                opts.adaptive_mu_monotone_init_factor = v;
            }
        }
        "barrier_tol_factor" => {
            if let Ok(v) = value.parse() {
                opts.barrier_tol_factor = v;
            }
        }
        "mu_strategy" => {
            opts.mu_strategy_adaptive = value == "adaptive";
        }
        "least_squares_mult_init" => {
            opts.least_squares_mult_init = value == "yes" || value == "true" || value == "1";
        }
        "constr_mult_init_max" => {
            if let Ok(v) = value.parse() {
                opts.constr_mult_init_max = v;
            }
        }
        "constraint_slack_barrier" => {
            opts.constraint_slack_barrier = value == "yes" || value == "true" || value == "1";
        }
        "sparse_threshold" => {
            if let Ok(v) = value.parse() {
                opts.sparse_threshold = v;
            }
        }
        "warm_start_init_point" => {
            opts.warm_start = value == "yes" || value == "true" || value == "1";
        }
        "warm_start_bound_push" => {
            if let Ok(v) = value.parse() {
                opts.warm_start_bound_push = v;
            }
        }
        "warm_start_bound_frac" => {
            if let Ok(v) = value.parse() {
                opts.warm_start_bound_frac = v;
            }
        }
        "warm_start_mult_bound_push" => {
            if let Ok(v) = value.parse() {
                opts.warm_start_mult_bound_push = v;
            }
        }
        "nlp_lower_bound_inf" => {
            if let Ok(v) = value.parse() {
                opts.nlp_lower_bound_inf = v;
            }
        }
        "nlp_upper_bound_inf" => {
            if let Ok(v) = value.parse() {
                opts.nlp_upper_bound_inf = v;
            }
        }
        "max_soc" => {
            if let Ok(v) = value.parse() {
                opts.max_soc = v;
            }
        }
        "watchdog_shortened_iter_trigger" => {
            if let Ok(v) = value.parse() {
                opts.watchdog_shortened_iter_trigger = v;
            }
        }
        "watchdog_trial_iter_max" => {
            if let Ok(v) = value.parse() {
                opts.watchdog_trial_iter_max = v;
            }
        }
        "restoration_max_iter" => {
            if let Ok(v) = value.parse() {
                opts.restoration_max_iter = v;
            }
        }
        "disable_nlp_restoration" => {
            opts.disable_nlp_restoration = value == "yes" || value == "true" || value == "1";
        }
        "slack_fallback" | "enable_slack_fallback" => {
            opts.enable_slack_fallback = value == "yes" || value == "true" || value == "1";
        }
        "lbfgs_fallback" | "enable_lbfgs_fallback" => {
            opts.enable_lbfgs_fallback = value == "yes" || value == "true" || value == "1";
        }
        "lbfgs_hessian_fallback" | "enable_lbfgs_hessian_fallback" => {
            opts.enable_lbfgs_hessian_fallback = value == "yes" || value == "true" || value == "1";
        }
        "enable_preprocessing" => {
            opts.enable_preprocessing = value == "yes" || value == "true" || value == "1";
        }
        "detect_linear_constraints" => {
            opts.detect_linear_constraints = value == "yes" || value == "true" || value == "1";
        }
        "mehrotra_pc" => {
            opts.mehrotra_pc = value == "yes" || value == "true" || value == "1";
        }
        "gondzio_mcc_max" => {
            if let Ok(v) = value.parse() {
                opts.gondzio_mcc_max = v;
            }
        }
        "proactive_infeasibility_detection" => {
            opts.proactive_infeasibility_detection = value == "yes" || value == "true" || value == "1";
        }
        "hessian_approximation" => {
            match value {
                "limited-memory" => opts.hessian_approximation_lbfgs = true,
                "exact" => opts.hessian_approximation_lbfgs = false,
                _ => eprintln!("Warning: unknown hessian_approximation '{}'", value),
            }
        }
        "linear_solver" => {
            match value {
                "direct" => opts.linear_solver = ripopt::LinearSolverChoice::Direct,
                "iterative" | "minres" => opts.linear_solver = ripopt::LinearSolverChoice::Iterative,
                "hybrid" | "auto" => opts.linear_solver = ripopt::LinearSolverChoice::Hybrid,
                _ => eprintln!("Warning: unknown linear_solver '{}' (use 'direct', 'iterative', or 'hybrid')", value),
            }
        }
        "stall_iter_limit" => {
            if let Ok(v) = value.parse() {
                opts.stall_iter_limit = v;
            }
        }
        "early_stall_timeout" => {
            if let Ok(v) = value.parse() {
                opts.early_stall_timeout = v;
            }
        }
        "mu_oracle_quality_function" => {
            opts.mu_oracle_quality_function = value == "yes" || value == "true" || value == "1";
        }
        "quality_function_centrality" => {
            opts.quality_function_centrality = value == "yes" || value == "true" || value == "1";
        }
        "warm_start_target_mu" => {
            if let Ok(v) = value.parse() {
                opts.warm_start_target_mu = Some(v);
            }
        }
        "user_obj_scaling" => {
            if let Ok(v) = value.parse() {
                opts.user_obj_scaling = Some(v);
            }
        }
        "kkt_dump_dir" => {
            opts.kkt_dump_dir = Some(std::path::PathBuf::from(value));
        }
        "kkt_dump_name" => {
            opts.kkt_dump_name = value.to_string();
        }
        _ => {
            eprintln!("Warning: unknown option '{}'", key);
        }
    }
}
