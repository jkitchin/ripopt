use crate::result::{SolveResult, SolveStatus};
use std::io::Write;

/// Map SolveStatus to AMPL solve code.
fn solve_code(status: SolveStatus) -> i32 {
    match status {
        SolveStatus::Optimal => 0,
        SolveStatus::LocalInfeasibility | SolveStatus::Infeasible => 200,
        SolveStatus::Unbounded => 300,
        SolveStatus::MaxIterations => 400,
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
        SolveStatus::Infeasible => "Infeasible Problem Detected",
        SolveStatus::LocalInfeasibility => "Converged to a point of local infeasibility",
        SolveStatus::MaxIterations => "Maximum Number of Iterations Exceeded",
        SolveStatus::NumericalError => "Numerical Difficulties",
        SolveStatus::EvaluationError => "Evaluation Error in User Callbacks",
        SolveStatus::UserRequestedStop => "Optimization Stopped by User",
        SolveStatus::Unbounded => "Problem Appears Unbounded",
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
