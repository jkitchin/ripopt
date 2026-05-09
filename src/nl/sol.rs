use crate::result::{SolveResult, SolveStatus};
use std::io::Write;

/// Map SolveStatus to AMPL solve code.
fn solve_code(status: SolveStatus) -> i32 {
    match status {
        SolveStatus::Optimal => 0,
        // AMPL convention: "acceptable" ≈ "solved with warning" ≈ code 100.
        SolveStatus::Acceptable => 100,
        // STOP_AT_TINY_STEP: treat as warning-class (search direction too small).
        SolveStatus::StopAtTinyStep => 100,
        SolveStatus::LocalInfeasibility | SolveStatus::Infeasible => 200,
        SolveStatus::DivergingIterates => 300,
        SolveStatus::MaxIterations => 400,
        // Matches Ipopt AmplTNLP.cpp's `Maximum_CpuTime_Exceeded` mapping.
        // Pyomo's .sol parser collapses 400-499 to maxIterations, but the
        // numeric `solver.id` and message string still distinguish a
        // time-out from an iter-cap.
        SolveStatus::MaxTimeExceeded => 401,
        SolveStatus::NumericalError
        | SolveStatus::EvaluationError
        | SolveStatus::RestorationFailed
        | SolveStatus::InternalError => 500,
        SolveStatus::UserRequestedStop => 400,
    }
}

/// Map SolveStatus to a human-readable message.
fn status_message(status: SolveStatus) -> &'static str {
    match status {
        SolveStatus::Optimal => "Optimal Solution Found",
        SolveStatus::Acceptable => "Solved To Acceptable Level",
        SolveStatus::StopAtTinyStep => "Search Direction Becomes Too Small",
        SolveStatus::Infeasible => "Infeasible Problem Detected",
        SolveStatus::LocalInfeasibility => "Converged to a point of local infeasibility",
        SolveStatus::MaxIterations => "Maximum Number of Iterations Exceeded",
        SolveStatus::MaxTimeExceeded => "Maximum CPU Time Exceeded",
        SolveStatus::NumericalError => "Numerical Difficulties",
        SolveStatus::EvaluationError => "Evaluation Error in User Callbacks",
        SolveStatus::UserRequestedStop => "Optimization Stopped by User",
        SolveStatus::DivergingIterates => "Diverging Iterates",
        SolveStatus::RestorationFailed => "Restoration Failed",
        SolveStatus::InternalError => "Internal Error",
    }
}

/// Write a SOL file for the given solve result.
pub fn write_sol<W: Write>(
    writer: &mut W,
    result: &SolveResult,
    n_vars: usize,
    n_constrs: usize,
) -> std::io::Result<()> {
    // Message section
    writeln!(
        writer,
        "ripopt {}: {}",
        env!("CARGO_PKG_VERSION"),
        status_message(result.status)
    )?;
    writeln!(writer)?; // blank line terminates message

    // Options section
    writeln!(writer, "Options")?;
    writeln!(writer, "3")?;
    writeln!(writer, "0")?;
    writeln!(writer, "1")?;
    writeln!(writer, "0")?;

    // Constraint/variable counts
    writeln!(writer, "{}", n_constrs)?;
    writeln!(writer, "{}", n_constrs)?;
    writeln!(writer, "{}", n_vars)?;
    writeln!(writer, "{}", n_vars)?;

    // Dual values (constraint multipliers)
    for i in 0..n_constrs {
        let val = result
            .constraint_multipliers
            .get(i)
            .copied()
            .unwrap_or(0.0);
        writeln!(writer, "{:.17e}", val)?;
    }

    // Primal values
    for i in 0..n_vars {
        let val = result.x.get(i).copied().unwrap_or(0.0);
        writeln!(writer, "{:.17e}", val)?;
    }

    // Objective line
    writeln!(writer, "objno 0 {}", solve_code(result.status))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_exceeded_distinct_from_iter_exceeded_in_sol() {
        // Issue #36 follow-up: Pyomo collapses 400-499 to maxIterations,
        // but `solver.id` and the message string still need to distinguish
        // a wall-time exit from an iter-cap exit. Mirrors Ipopt's AmplTNLP
        // mapping (Maximum_Iterations_Exceeded=400, Maximum_CpuTime_Exceeded=401).
        assert_eq!(solve_code(SolveStatus::MaxIterations), 400);
        assert_eq!(solve_code(SolveStatus::MaxTimeExceeded), 401);
        assert_eq!(
            status_message(SolveStatus::MaxIterations),
            "Maximum Number of Iterations Exceeded"
        );
        assert_eq!(
            status_message(SolveStatus::MaxTimeExceeded),
            "Maximum CPU Time Exceeded"
        );
    }
}
