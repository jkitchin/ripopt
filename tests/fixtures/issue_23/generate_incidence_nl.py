"""Generate and optionally solve the issue #23 incidence-example fixtures.

This script is intentionally optional: the committed `.nl` fixtures let CI test
the ripopt side without Pyomo, IDAES, or the local incidence_examples checkout.

Validated local setup:

    env UV_CACHE_DIR=/tmp/ripopt-issue23-uv-cache uv venv .venv --python 3.11
    env UV_CACHE_DIR=/tmp/ripopt-issue23-uv-cache \
        uv pip install --python .venv/bin/python \
        pyomo idaes-pse scipy networkx matplotlib \
        -e ~/repos/incidence_examples
    env PIXI_HOME=/tmp/ripopt-issue23-pixi pixi global install ipopt
    cargo build --release --bin ripopt
    env PATH=/tmp/ripopt-issue23-pixi/bin:$PATH MPLCONFIGDIR=/tmp/ripopt-issue23-mpl \
        .venv/bin/python tests/fixtures/issue_23/generate_incidence_nl.py \
        --ripopt target/release/ripopt
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    perturbed: bool = False


CASES = (
    Case(
        name="tutorial_flow_density",
        description=(
            "Final incidence_examples tutorial repair: adds the density/flow "
            "relation and leaves particle porosity as a solved variable."
        ),
    ),
    Case(
        name="tutorial_flow_density_perturbed",
        description=(
            "Same tutorial repair at a perturbed operating point with shifted "
            "fixed inlet, holdup, volume, velocity, and initial values."
        ),
        perturbed=True,
    ),
)


def patch_idaes_get_solver() -> None:
    """Keep the older incidence_examples import working with current IDAES."""
    import idaes.core.util as idaes_util
    from idaes.core.solvers import get_solver

    idaes_util.get_solver = get_solver


def build_tutorial_flow_density_model(perturbed: bool = False):
    patch_idaes_get_solver()

    import pyomo.environ as pyo
    from incidence_examples.tutorial.model import make_model

    m = make_model()
    m.particle_porosity.unfix()

    @m.Constraint()
    def flow_density_eqn(b):
        return m.flow_mass == m.velocity * m.area * m.dens_mass_particle

    if perturbed:
        m.flow_mass_in.fix(0.7)
        for comp, value in {"A": 0.2, "B": 0.5, "C": 0.3}.items():
            m.mass_frac_comp_in[comp].fix(value)
        m.enth_mass_in.fix(1.4)
        for comp, value in {"A": 1.1, "B": 0.9, "C": 1.0}.items():
            m.material_holdup[comp].fix(value)
        m.energy_holdup.fix(1.2)
        m.volume.fix(1.5)
        m.velocity.fix(0.8)

        for comp, value in {"A": 0.25, "B": 0.45, "C": 0.30}.items():
            m.mass_frac_comp[comp].set_value(value)
        m.temperature.set_value(350.0)
        m.dens_mass_particle.set_value(1000.0)
        m.dens_mass_skeletal.set_value(1300.0)
        m.flow_mass.set_value(800.0)

    m._ripopt_issue23_objective = pyo.Objective(expr=0.0)
    return m


def model_counts(model) -> tuple[int, int]:
    import pyomo.environ as pyo

    n_constraints = sum(
        1 for _ in model.component_data_objects(pyo.Constraint, active=True)
    )
    n_unfixed_variables = sum(
        1 for var in model.component_data_objects(pyo.Var) if not var.fixed
    )
    return n_constraints, n_unfixed_variables


def export_nl(case: Case, fixture_dir: Path) -> dict[str, Any]:
    from pyomo.opt import ProblemFormat

    model = build_tutorial_flow_density_model(perturbed=case.perturbed)
    n_constraints, n_unfixed_variables = model_counts(model)
    path = fixture_dir / f"{case.name}.nl"
    model.write(
        str(path),
        format=ProblemFormat.nl,
        io_options={"symbolic_solver_labels": False},
    )
    try:
        display_path = str(path.relative_to(ROOT))
    except ValueError:
        display_path = str(path)

    return {
        "name": case.name,
        "description": case.description,
        "nl": display_path,
        "constraints": n_constraints,
        "unfixed_variables": n_unfixed_variables,
        "bytes": path.stat().st_size,
    }


def survey_skips() -> list[dict[str, str]]:
    skipped: list[dict[str, str]] = []

    try:
        importlib.import_module("idaes.gas_solid_contactors")
    except Exception as exc:  # noqa: BLE001 - reported as an optional skip reason.
        reason = (
            "Current idaes-pse environment does not provide "
            f"idaes.gas_solid_contactors ({type(exc).__name__}: {exc})."
        )
        skipped.append({"name": "example1_clc_dm", "reason": reason})
        skipped.append({"name": "example2_clc_scc", "reason": reason})

    return skipped


STATUS_RE = re.compile(r"ripopt\s+\S+:\s+(\w+)\s+after\s+(\d+)\s+iterations")
OBJECTIVE_RE = re.compile(r"Objective:\s+([-+0-9.eE]+)")
DIAGNOSTIC_RE = re.compile(r"^([a-zA-Z_]+):\s+(.+)$")


def parse_ripopt_output(output: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    status = STATUS_RE.search(output)
    if status:
        metrics["status"] = status.group(1)
        metrics["iterations"] = int(status.group(2))
    objective = OBJECTIVE_RE.search(output)
    if objective:
        metrics["objective"] = float(objective.group(1))
    for line in output.splitlines():
        diagnostic = DIAGNOSTIC_RE.match(line.strip())
        if not diagnostic:
            continue
        key, value = diagnostic.groups()
        if key in {"final_primal_inf", "final_dual_inf", "final_compl", "final_mu"}:
            metrics[key] = float(value)
        elif key == "fallback_used":
            metrics[key] = value
    return metrics


def run_ripopt(ripopt: Path, nl: Path, enable_preprocessing: bool) -> dict[str, Any]:
    cmd = [
        str(ripopt),
        str(nl),
        "print_level=0",
        f"enable_preprocessing={'yes' if enable_preprocessing else 'no'}",
        "early_stall_timeout=0",
        "max_iter=500",
    ]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    metrics = parse_ripopt_output(completed.stdout + "\n" + completed.stderr)
    metrics["returncode"] = completed.returncode
    return metrics


def classify(preprocessed: dict[str, Any], fallback: dict[str, Any]) -> str:
    if preprocessed.get("fallback_used") == "auxiliary_preprocessing":
        return "fell back"
    if preprocessed.get("status") != fallback.get("status"):
        return "changed status"
    pre_iters = preprocessed.get("iterations")
    fallback_iters = fallback.get("iterations")
    if isinstance(pre_iters, int) and isinstance(fallback_iters, int):
        if pre_iters + 2 < fallback_iters:
            return "helped"
        if pre_iters > fallback_iters + 2:
            return "slower"
    return "neutral"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=FIXTURE_DIR,
        help="Directory where .nl fixtures are written.",
    )
    parser.add_argument(
        "--ripopt",
        type=Path,
        default=None,
        help="Optional ripopt binary for preprocessing on/off comparison.",
    )
    args = parser.parse_args()

    args.fixture_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"cases": [], "skipped": survey_skips()}

    for case in CASES:
        exported = export_nl(case, args.fixture_dir)
        if args.ripopt is not None:
            nl_path = ROOT / exported["nl"]
            preprocessed = run_ripopt(args.ripopt, nl_path, True)
            fallback = run_ripopt(args.ripopt, nl_path, False)
            exported["preprocessing_enabled"] = preprocessed
            exported["preprocessing_disabled"] = fallback
            exported["preprocessing_effect"] = classify(preprocessed, fallback)
        report["cases"].append(exported)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
