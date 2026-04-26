// Benchmark binary: solve all HS problems with ripopt and output JSON results.
// Runs multiple timing passes to get stable timing measurements.

#[path = "generated/hs_problems.rs"]
mod hs_problems;

use hs_problems::solve_all;
use ripopt::SolverOptions;

/// Collect system information for benchmark reproducibility.
fn print_system_info() {
    eprintln!("=== System Information ===");
    eprintln!("  OS:           {}", std::env::consts::OS);
    eprintln!("  Arch:         {}", std::env::consts::ARCH);

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            let cpu = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !cpu.is_empty() {
                eprintln!("  CPU:          {}", cpu);
            }
        }
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
        {
            if let Ok(bytes) = String::from_utf8_lossy(&output.stdout).trim().parse::<u64>() {
                eprintln!("  RAM:          {} GB", bytes / (1024 * 1024 * 1024));
            }
        }
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "hw.physicalcpu"])
            .output()
        {
            let cores = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !cores.is_empty() {
                eprintln!("  Cores:        {}", cores);
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if line.starts_with("model name") {
                    if let Some(name) = line.split(':').nth(1) {
                        eprintln!("  CPU:          {}", name.trim());
                        break;
                    }
                }
            }
        }
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal") {
                    if let Some(val) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = val.parse::<u64>() {
                            eprintln!("  RAM:          {} GB", kb / (1024 * 1024));
                        }
                    }
                    break;
                }
            }
        }
    }

    eprintln!("  Rust version: {}", env!("CARGO_PKG_VERSION"));
    eprintln!("  Profile:      {}", if cfg!(debug_assertions) { "debug" } else { "release" });
    eprintln!("=========================");
}

fn main() {
    let n_timing_runs: usize = std::env::var("RIPOPT_TIMING_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let force_sparse = std::env::var("RIPOPT_FORCE_SPARSE").is_ok();
    let options = SolverOptions {
        tol: 1e-8,
        max_iter: 3000,
        print_level: 0,
        mu_strategy_adaptive: true,
        sparse_threshold: if force_sparse { 0 } else { 100 },
        ..SolverOptions::default()
    };

    print_system_info();
    eprintln!("Solving all HS problems with ripopt ({} timing runs)...", n_timing_runs);

    // First run: get correctness results
    let mut results = solve_all(&options);

    // Additional timing runs: keep minimum solve_time per problem
    for run in 1..n_timing_runs {
        eprintln!("  Timing run {}/{}...", run + 1, n_timing_runs);
        let timing_results = solve_all(&options);
        for (r, t) in results.iter_mut().zip(timing_results.iter()) {
            if t.solve_time < r.solve_time {
                r.solve_time = t.solve_time;
            }
        }
    }

    // Summary to stderr
    let total = results.len();
    let optimal = results
        .iter()
        .filter(|r| r.status == "Optimal")
        .count();
    let acceptable = results
        .iter()
        .filter(|r| r.status == "Acceptable")
        .count();
    let solved = optimal + acceptable;
    eprintln!(
        "Solved {}/{} ({} optimal, {} acceptable)",
        solved, total, optimal, acceptable
    );

    // JSON to stdout (preserves the existing Makefile redirection contract).
    let json = serde_json::to_string_pretty(&results).unwrap();
    println!("{}", json);

    // Emit a JSONL companion alongside the canonical JSON file so external
    // tooling can consume the same per-problem records line-by-line. HS
    // solves are sub-millisecond, so per-problem streaming during the run
    // adds no visibility — we just dump JSONL after solve_all returns.
    let jsonl_path: Option<std::path::PathBuf> = std::env::var("RESULTS_JSONL")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var("HS_RESULTS_FILE")
                .ok()
                .map(|p| std::path::PathBuf::from(p).with_extension("jsonl"))
        });
    if let Some(path) = jsonl_path {
        use std::io::Write;
        match std::fs::File::create(&path) {
            Ok(f) => {
                let mut w = std::io::BufWriter::new(f);
                for r in &results {
                    if let Ok(line) = serde_json::to_string(r) {
                        let _ = writeln!(w, "{}", line);
                    }
                }
                let _ = w.flush();
                eprintln!("JSONL stream: {}", path.display());
            }
            Err(e) => eprintln!("WARNING: cannot open {}: {}", path.display(), e),
        }
    }
}
