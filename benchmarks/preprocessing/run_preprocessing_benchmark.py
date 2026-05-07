#!/usr/bin/env python3
"""Benchmark ripopt with auxiliary preprocessing enabled and disabled.

The script runs the same AMPL .nl instances twice:

* enable_preprocessing=yes
* enable_preprocessing=no

It writes a compact JSON artifact and a Markdown table suitable for pasting
into a pull request. The generated reports intentionally omit full solution
vectors from ripopt's JSON output.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

SMOKE_INSTANCES = [
    "tests/fixtures/issue_10/auxiliary_gate.nl",
    "tests/fixtures/issue_23/tutorial_flow_density.nl",
    "tests/fixtures/issue_23/tutorial_flow_density_perturbed.nl",
    "benchmarks/gas/gaslib11_steady.nl",
]

DEFAULT_INSTANCES = [
    *SMOKE_INSTANCES,
    "benchmarks/gas/gaslib11_dynamic.nl",
    "benchmarks/gas/gaslib40_steady.nl",
    "benchmarks/water/water.nl",
    "benchmarks/water/water3.nl",
    "benchmarks/water/water4.nl",
    "benchmarks/water/watersbp.nl",
    "benchmarks/water/waterx.nl",
    "benchmarks/water/waterz.nl",
]

FULL_INSTANCES = [
    *DEFAULT_INSTANCES,
    "benchmarks/gas/gaslib40_dynamic.nl",
    "benchmarks/cho/nl_export_results/cho_parmest.nl",
]

PROFILES = {
    "smoke": SMOKE_INSTANCES,
    "default": DEFAULT_INSTANCES,
    "full": FULL_INSTANCES,
}

SOLVED_STATUSES = {"Optimal", "Solved", "Acceptable"}
STATUS_RANK = {
    "Optimal": 0,
    "Solved": 0,
    "Acceptable": 1,
    "MaxIterations": 2,
    "MaxTimeExceeded": 2,
    "NumericalError": 3,
    "Infeasible": 3,
    "DivergingIterates": 3,
    "ProcessError": 4,
    "Timeout": 4,
    "MissingReport": 4,
}

AUX_SOLVED_RE = re.compile(
    r"Auxiliary preprocessing solved (\d+) block\(s\), max residual ([0-9eE+\-.]+)"
)
AUX_REDUCED_RE = re.compile(
    r"Auxiliary preprocessing reduced problem: "
    r"(\d+) fixed vars, (\d+) removed constraints "
    r"\((\d+)x(\d+) -> (\d+)x(\d+)\)"
)
NESTED_REDUCED_RE = re.compile(
    r"Auxiliary nested preprocessing reduced problem: "
    r"(\d+) fixed vars, (\d+) redundant constraints "
    r"\((\d+)x(\d+) -> (\d+)x(\d+)\)"
)


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_instance(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def read_manifest(path: Path) -> list[str]:
    entries: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


def default_ripopt_binary() -> Path:
    candidates = [
        ROOT / "target" / "release" / "ripopt",
        ROOT / "target" / "debug" / "ripopt",
        shutil.which("ripopt"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return path.resolve()
    raise SystemExit(
        "Could not find ripopt. Build it first with "
        "`cargo build --release --bin ripopt`, or pass --ripopt PATH."
    )


def run_version(ripopt: Path) -> str | None:
    try:
        completed = subprocess.run(
            [str(ripopt), "--version"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = completed.stdout.strip()
    return text or None


def finite_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def median(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return float(statistics.median(finite))


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    problem = report.get("problem", {}) if isinstance(report.get("problem"), dict) else {}
    validation = (
        report.get("validation", {}) if isinstance(report.get("validation"), dict) else {}
    )
    diagnostics = (
        report.get("diagnostics", {}) if isinstance(report.get("diagnostics"), dict) else {}
    )
    options = report.get("options", {}) if isinstance(report.get("options"), dict) else {}
    return {
        "status": report.get("status"),
        "iterations": report.get("iterations"),
        "objective": finite_number(report.get("objective")),
        "wall_time_secs": finite_number(report.get("wall_time_secs")),
        "problem": {
            "name": problem.get("name"),
            "n_variables": problem.get("n_variables"),
            "n_constraints": problem.get("n_constraints"),
        },
        "validation": {
            "max_constraint_violation": finite_number(
                validation.get("max_constraint_violation")
            ),
            "stationarity_inf_norm": finite_number(
                validation.get("stationarity_inf_norm")
            ),
            "kkt_satisfied": validation.get("kkt_satisfied"),
        },
        "diagnostics": {
            "final_primal_inf": finite_number(diagnostics.get("final_primal_inf")),
            "final_dual_inf": finite_number(diagnostics.get("final_dual_inf")),
            "final_compl": finite_number(diagnostics.get("final_compl")),
            "fallback_used": diagnostics.get("fallback_used"),
            "wall_time_secs": finite_number(diagnostics.get("wall_time_secs")),
            "n_obj_evals": diagnostics.get("n_obj_evals"),
            "n_grad_evals": diagnostics.get("n_grad_evals"),
            "n_constr_evals": diagnostics.get("n_constr_evals"),
            "n_jac_evals": diagnostics.get("n_jac_evals"),
            "n_hess_evals": diagnostics.get("n_hess_evals"),
        },
        "options": {
            "enable_preprocessing": options.get("enable_preprocessing"),
            "max_iter": options.get("max_iter"),
            "max_wall_time": options.get("max_wall_time"),
        },
    }


def run_one(
    ripopt: Path,
    instance: Path,
    *,
    enable_preprocessing: bool,
    max_iter: int,
    timeout: float,
    max_wall_time: float | None,
    print_level: int,
    tmpdir: Path,
    keep_stdout: bool = False,
) -> dict[str, Any]:
    report_path = tmpdir / (
        f"{instance.stem}-pre-{int(enable_preprocessing)}-{time.time_ns()}.json"
    )
    cmd = [
        str(ripopt),
        str(instance),
        "-o",
        str(report_path),
        f"print_level={print_level}",
        f"enable_preprocessing={'yes' if enable_preprocessing else 'no'}",
        f"max_iter={max_iter}",
    ]
    if max_wall_time is not None and max_wall_time > 0:
        cmd.append(f"max_wall_time={max_wall_time}")

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - started
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        result = {
            "status": "Timeout",
            "returncode": None,
            "elapsed_secs": timeout,
            "stdout_tail": stdout[-4000:],
            "command": cmd,
        }
        if keep_stdout:
            result["stdout"] = stdout
        return result

    result: dict[str, Any] = {
        "returncode": completed.returncode,
        "elapsed_secs": elapsed,
        "stdout_tail": completed.stdout[-4000:],
        "command": cmd,
    }
    if keep_stdout:
        result["stdout"] = completed.stdout
    if report_path.exists():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            result["status"] = "MissingReport"
            result["report_error"] = str(exc)
        else:
            result.update(compact_report(loaded))
    else:
        result["status"] = "MissingReport"

    if completed.returncode != 0 and result.get("status") in {None, "MissingReport"}:
        result["status"] = "ProcessError"
    return result


def parse_auxiliary_diagnostics(output: str) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    solved = AUX_SOLVED_RE.search(output)
    if solved:
        diagnostics["auxiliary_blocks_solved"] = int(solved.group(1))
        diagnostics["auxiliary_max_residual"] = float(solved.group(2))
    reduced = AUX_REDUCED_RE.search(output)
    if reduced:
        diagnostics["auxiliary_reduction"] = {
            "fixed_variables": int(reduced.group(1)),
            "removed_constraints": int(reduced.group(2)),
            "original_variables": int(reduced.group(3)),
            "original_constraints": int(reduced.group(4)),
            "reduced_variables": int(reduced.group(5)),
            "reduced_constraints": int(reduced.group(6)),
        }
    nested = NESTED_REDUCED_RE.search(output)
    if nested:
        diagnostics["nested_reduction"] = {
            "fixed_variables": int(nested.group(1)),
            "redundant_constraints": int(nested.group(2)),
            "original_variables": int(nested.group(3)),
            "original_constraints": int(nested.group(4)),
            "reduced_variables": int(nested.group(5)),
            "reduced_constraints": int(nested.group(6)),
        }
    return diagnostics


def diagnostic_probe(
    ripopt: Path,
    instance: Path,
    *,
    max_iter: int,
    timeout: float,
    max_wall_time: float | None,
    tmpdir: Path,
) -> dict[str, Any]:
    result = run_one(
        ripopt,
        instance,
        enable_preprocessing=True,
        max_iter=max_iter,
        timeout=timeout,
        max_wall_time=max_wall_time,
        print_level=5,
        tmpdir=tmpdir,
        keep_stdout=True,
    )
    parsed = parse_auxiliary_diagnostics(
        result.get("stdout", "") or result.get("stdout_tail", "")
    )
    return {
        "status": result.get("status"),
        "iterations": result.get("iterations"),
        "returncode": result.get("returncode"),
        **parsed,
    }


def summarize_mode(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    first = runs[0]
    solver_times = [
        first_present(run.get("wall_time_secs"), run.get("diagnostics", {}).get("wall_time_secs"))
        for run in runs
    ]
    elapsed_times = [run.get("elapsed_secs") for run in runs]
    statuses = [run.get("status") for run in runs]
    iterations = [run.get("iterations") for run in runs if isinstance(run.get("iterations"), int)]
    return {
        "status": first.get("status"),
        "statuses": statuses,
        "iterations": first.get("iterations"),
        "median_iterations": median([float(value) for value in iterations]),
        "objective": first.get("objective"),
        "median_solver_wall_time_secs": median(
            [float(value) for value in solver_times if isinstance(value, (int, float))]
        ),
        "median_elapsed_secs": median(
            [float(value) for value in elapsed_times if isinstance(value, (int, float))]
        ),
        "validation": first.get("validation", {}),
        "diagnostics": first.get("diagnostics", {}),
        "returncodes": [run.get("returncode") for run in runs],
    }


def status_rank(status: Any) -> int:
    return STATUS_RANK.get(str(status), 5)


def compare_summaries(pre: dict[str, Any], no_pre: dict[str, Any]) -> dict[str, Any]:
    pre_status = pre.get("status")
    no_pre_status = no_pre.get("status")
    pre_rank = status_rank(pre_status)
    no_pre_rank = status_rank(no_pre_status)
    if pre_rank < no_pre_rank:
        status_delta = "better"
    elif pre_rank > no_pre_rank:
        status_delta = "worse"
    else:
        status_delta = "same"

    pre_iters = pre.get("iterations")
    no_pre_iters = no_pre.get("iterations")
    iter_delta = None
    if isinstance(pre_iters, int) and isinstance(no_pre_iters, int):
        iter_delta = no_pre_iters - pre_iters

    pre_elapsed = pre.get("median_elapsed_secs")
    no_pre_elapsed = no_pre.get("median_elapsed_secs")
    elapsed_ratio = None
    if isinstance(pre_elapsed, (int, float)) and isinstance(no_pre_elapsed, (int, float)):
        if pre_elapsed > 0:
            elapsed_ratio = no_pre_elapsed / pre_elapsed

    pre_solver_time = pre.get("median_solver_wall_time_secs")
    no_pre_solver_time = no_pre.get("median_solver_wall_time_secs")
    solver_time_ratio = None
    if isinstance(pre_solver_time, (int, float)) and isinstance(no_pre_solver_time, (int, float)):
        if pre_solver_time > 0:
            solver_time_ratio = no_pre_solver_time / pre_solver_time

    return {
        "status_delta": status_delta,
        "iteration_delta_no_pre_minus_pre": iter_delta,
        "elapsed_time_ratio_no_pre_over_pre": elapsed_ratio,
        "solver_time_ratio_no_pre_over_pre": solver_time_ratio,
    }


def format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        if value == 0:
            return "0"
        if abs(value) >= 1000 or abs(value) < 0.001:
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"
    return str(value)


def reduction_text(diagnostic: dict[str, Any]) -> str:
    reduction = diagnostic.get("auxiliary_reduction")
    if not isinstance(reduction, dict):
        return "-"
    blocks = diagnostic.get("auxiliary_blocks_solved")
    shape = (
        f"{reduction.get('original_variables')}x{reduction.get('original_constraints')}"
        f" -> {reduction.get('reduced_variables')}x{reduction.get('reduced_constraints')}"
    )
    removed = (
        f"{reduction.get('fixed_variables')} vars, "
        f"{reduction.get('removed_constraints')} cons"
    )
    if isinstance(blocks, int):
        return f"{shape}; {removed}; {blocks} blocks"
    return f"{shape}; {removed}"


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|")


def render_markdown(artifact: dict[str, Any]) -> str:
    metadata = artifact["metadata"]
    lines = [
        "# Preprocessing benchmark",
        "",
        f"- Generated: `{metadata['generated_at']}`",
        f"- ripopt: `{metadata.get('ripopt_version') or metadata['ripopt']}`",
        f"- Profile: `{metadata['profile']}`",
        f"- Repeats per mode: `{metadata['repeat']}`",
        f"- Timeout per solve: `{metadata['timeout_secs']}s`",
        f"- max_iter: `{metadata['max_iter']}`",
        "",
        "| Instance | Shape | Reduction | Pre status | Pre iters | No-pre status | No-pre iters | Iter delta | Elapsed ratio |",
        "| --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for case in artifact["instances"]:
        problem = case.get("problem", {})
        shape = "-"
        if problem.get("n_variables") is not None and problem.get("n_constraints") is not None:
            shape = f"{problem['n_variables']}x{problem['n_constraints']}"
        comparison = case.get("comparison", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{markdown_escape(case['instance'])}`",
                    shape,
                    markdown_escape(reduction_text(case.get("diagnostic_probe", {}))),
                    format_number(case["preprocessing"]["status"]),
                    format_number(case["preprocessing"]["iterations"]),
                    format_number(case["no_preprocessing"]["status"]),
                    format_number(case["no_preprocessing"]["iterations"]),
                    format_number(comparison.get("iteration_delta_no_pre_minus_pre")),
                    format_number(
                        comparison.get("elapsed_time_ratio_no_pre_over_pre"),
                        digits=2,
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Positive iteration delta means preprocessing used fewer IPM iterations.",
            "Elapsed ratio is median no-preprocessing process time divided by median preprocessing process time; values above 1 mean preprocessing was faster end-to-end.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_artifact(
    *,
    args: argparse.Namespace,
    ripopt: Path,
    instances: list[Path],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "instances": len(rows),
        "preprocessing_solved": 0,
        "no_preprocessing_solved": 0,
        "preprocessing_fewer_iterations": 0,
        "preprocessing_slower_iterations": 0,
        "preprocessing_better_status": 0,
        "preprocessing_worse_status": 0,
    }
    for row in rows:
        pre_status = row["preprocessing"].get("status")
        no_pre_status = row["no_preprocessing"].get("status")
        if pre_status in SOLVED_STATUSES:
            summary["preprocessing_solved"] += 1
        if no_pre_status in SOLVED_STATUSES:
            summary["no_preprocessing_solved"] += 1
        comparison = row["comparison"]
        delta = comparison.get("iteration_delta_no_pre_minus_pre")
        if isinstance(delta, int):
            if delta > 0:
                summary["preprocessing_fewer_iterations"] += 1
            elif delta < 0:
                summary["preprocessing_slower_iterations"] += 1
        if comparison.get("status_delta") == "better":
            summary["preprocessing_better_status"] += 1
        elif comparison.get("status_delta") == "worse":
            summary["preprocessing_worse_status"] += 1

    return {
        "metadata": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "root": str(ROOT),
            "ripopt": str(ripopt),
            "ripopt_version": run_version(ripopt),
            "profile": args.profile,
            "repeat": args.repeat,
            "timeout_secs": args.timeout,
            "max_iter": args.max_iter,
            "max_wall_time": args.max_wall_time,
            "diagnostic_probe": args.diagnostic_probe,
            "instances": [relpath(instance) for instance in instances],
        },
        "summary": summary,
        "instances": rows,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ripopt enable_preprocessing=yes/no on AMPL .nl instances."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="default",
        help="Instance profile to run when --instance/--manifest is not supplied.",
    )
    parser.add_argument(
        "--instance",
        action="append",
        default=[],
        help="Run only this .nl instance. May be passed multiple times.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Text file of .nl paths, one per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--list-instances",
        action="store_true",
        help="List the selected instances and exit.",
    )
    parser.add_argument(
        "--ripopt",
        type=Path,
        default=None,
        help="ripopt binary. Defaults to target/release/ripopt, target/debug/ripopt, then PATH.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Timed repeats per preprocessing mode.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Subprocess timeout in seconds for each solve.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3000,
        help="ripopt max_iter option for each solve.",
    )
    parser.add_argument(
        "--max-wall-time",
        type=float,
        default=0.0,
        help="ripopt max_wall_time option. 0 leaves ripopt unlimited.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "preprocessing" / "results" / "benchmark.json",
        help="Path for compact JSON results.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=ROOT / "benchmarks" / "preprocessing" / "results" / "benchmark.md",
        help="Path for Markdown summary.",
    )
    parser.add_argument(
        "--diagnostic-probe",
        dest="diagnostic_probe",
        action="store_true",
        default=True,
        help="Run one extra print_level=5 preprocessing solve to collect reduction diagnostics.",
    )
    parser.add_argument(
        "--no-diagnostic-probe",
        dest="diagnostic_probe",
        action="store_false",
        help="Skip the extra reduction-diagnostic solve.",
    )
    return parser.parse_args(argv)


def select_instances(args: argparse.Namespace) -> list[Path]:
    if args.instance:
        raw_instances = args.instance
    elif args.manifest:
        raw_instances = read_manifest(args.manifest)
    else:
        raw_instances = PROFILES[args.profile]

    instances = [resolve_instance(path) for path in raw_instances]
    missing = [path for path in instances if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing instance: {path}", file=sys.stderr)
        raise SystemExit(2)
    return instances


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    instances = select_instances(args)
    if args.list_instances:
        for instance in instances:
            print(relpath(instance))
        return 0

    ripopt = args.ripopt.resolve() if args.ripopt is not None else default_ripopt_binary()
    if not ripopt.exists():
        raise SystemExit(f"ripopt binary does not exist: {ripopt}")

    output = args.output if args.output.is_absolute() else ROOT / args.output
    markdown = args.markdown if args.markdown.is_absolute() else ROOT / args.markdown
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="ripopt-prebench-") as tmp:
        tmpdir = Path(tmp)
        for index, instance in enumerate(instances, start=1):
            name = relpath(instance)
            print(f"[{index}/{len(instances)}] {name}")
            probe: dict[str, Any] = {}
            if args.diagnostic_probe:
                probe = diagnostic_probe(
                    ripopt,
                    instance,
                    max_iter=args.max_iter,
                    timeout=args.timeout,
                    max_wall_time=args.max_wall_time,
                    tmpdir=tmpdir,
                )

            pre_runs = [
                run_one(
                    ripopt,
                    instance,
                    enable_preprocessing=True,
                    max_iter=args.max_iter,
                    timeout=args.timeout,
                    max_wall_time=args.max_wall_time,
                    print_level=0,
                    tmpdir=tmpdir,
                )
                for _ in range(args.repeat)
            ]
            no_pre_runs = [
                run_one(
                    ripopt,
                    instance,
                    enable_preprocessing=False,
                    max_iter=args.max_iter,
                    timeout=args.timeout,
                    max_wall_time=args.max_wall_time,
                    print_level=0,
                    tmpdir=tmpdir,
                )
                for _ in range(args.repeat)
            ]

            pre_summary = summarize_mode(pre_runs)
            no_pre_summary = summarize_mode(no_pre_runs)
            problem = first_present(
                pre_runs[0].get("problem") if pre_runs else None,
                no_pre_runs[0].get("problem") if no_pre_runs else None,
                {},
            )
            rows.append(
                {
                    "instance": name,
                    "problem": problem,
                    "diagnostic_probe": probe,
                    "preprocessing": pre_summary,
                    "no_preprocessing": no_pre_summary,
                    "comparison": compare_summaries(pre_summary, no_pre_summary),
                    "runs": {
                        "preprocessing": pre_runs,
                        "no_preprocessing": no_pre_runs,
                    },
                }
            )

    artifact = build_artifact(args=args, ripopt=ripopt, instances=instances, rows=rows)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(artifact), encoding="utf-8")
    print(f"Wrote {relpath(output)}")
    print(f"Wrote {relpath(markdown)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
