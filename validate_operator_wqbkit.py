#!/usr/bin/env python3
"""Validate FASTEXPR expressions through real WorldQuant BRAIN simulations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence

from config import load_project_env


SIMULATIONS_URL = "https://api.worldquantbrain.com/simulations"


@dataclass(frozen=True)
class ValidationConfig:
    region: str = "USA"
    universe: str = "TOP3000"
    neutralization: str = "SUBINDUSTRY"
    decay: int = 6
    delay: int = 1
    min_sharpe: float = 1.25
    min_fitness: float = 1.0
    max_turnover: float = 0.7
    max_correlation: float = 0.7
    correlation_type: str = "self"
    project_root: Path = Path.cwd()


@dataclass(frozen=True)
class ValidationResult:
    expression: str
    alpha_id: str
    sharpe: float
    fitness: float
    turnover: float
    correlation: float | None
    correlation_type: str
    drawdown: float | None
    two_year_sharpe: float
    failed_checks: int
    passed: bool
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_metrics(
    *,
    sharpe: float,
    fitness: float,
    turnover: float,
    correlation: float | None,
    config: ValidationConfig,
) -> tuple[bool, list[str]]:
    reasons = []
    if not math.isfinite(sharpe) or sharpe < config.min_sharpe:
        reasons.append(
            f"Sharpe {sharpe:.4f} < minimum {config.min_sharpe:.4f}"
        )
    if not math.isfinite(fitness) or fitness < config.min_fitness:
        reasons.append(
            f"Fitness {fitness:.4f} < minimum {config.min_fitness:.4f}"
        )
    if not math.isfinite(turnover) or turnover > config.max_turnover:
        reasons.append(
            f"Turnover {turnover:.4f} > maximum {config.max_turnover:.4f}"
        )
    if correlation is None or not math.isfinite(correlation):
        reasons.append("Correlation is unavailable")
    elif correlation > config.max_correlation:
        reasons.append(
            f"Correlation {correlation:.4f} > maximum "
            f"{config.max_correlation:.4f}"
        )
    return not reasons, reasons


def load_expressions(
    expression: str | None,
    expression_file: Path | None,
) -> list[str]:
    if expression:
        return [expression.strip()]
    if expression_file:
        lines = expression_file.read_text(encoding="utf-8").splitlines()
        return [
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
    raise ValueError("Provide --expression or --file")


def validate_environment(environment: Mapping[str, str]) -> None:
    missing = [
        name
        for name in ("WQB_USERNAME", "WQB_PASSWORD")
        if not environment.get(name)
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    if environment.get("DB_ENABLE", "true").lower() not in {
        "true",
        "1",
        "yes",
    }:
        raise RuntimeError(
            "wqbkit.AlphaSimulator requires DB_ENABLE=true"
        )


def _load_wqbkit(project_root: Path):
    load_project_env(
        project_root / ".env",
        required=("WQB_USERNAME", "WQB_PASSWORD"),
    )
    validate_environment(os.environ)
    from wqbkit import AlphaCalcCorr, AlphaSimulator

    return AlphaSimulator, AlphaCalcCorr


def _close_simulator(simulator) -> None:
    simulator.simulate_shutdown_event.set()
    simulator.executor.shutdown(wait=True, cancel_futures=True)
    simulator.process_shutdown_event.set()
    simulator.result_executor.shutdown(wait=True, cancel_futures=True)


def validate_expression(
    expression: str,
    config: ValidationConfig,
    *,
    simulator_factory: Callable | None = None,
    correlation_factory: Callable | None = None,
) -> ValidationResult:
    if not expression.strip():
        raise ValueError("Expression must not be empty")

    if simulator_factory is None or correlation_factory is None:
        default_simulator, default_correlation = _load_wqbkit(
            config.project_root
        )
        simulator_factory = simulator_factory or default_simulator
        correlation_factory = correlation_factory or default_correlation

    simulator = simulator_factory(1, 1, project_root=config.project_root)
    try:
        payload = simulator.combine_alpha(
            expression,
            config.region,
            config.universe,
            config.neutralization,
            config.decay,
            config.delay,
        )
        response = simulator.post(SIMULATIONS_URL, payload)
        response.raise_for_status()
        progress_url = response.headers.get("Location")
        if not progress_url:
            raise RuntimeError("BRAIN simulation response has no Location header")

        success, _ = simulator._wait_for_simulation(
            progress_url,
            "[operator-validator]",
        )
        if not success:
            raise RuntimeError("BRAIN simulation did not complete successfully")

        simulation = simulator.get_simulation_result(progress_url)
        if not simulation or not simulation.alpha_id:
            raise RuntimeError("BRAIN simulation returned no alpha id")

        detail_response = simulator.wqbs.locate_alpha(
            simulation.alpha_id,
            log=None,
        )
        detail_response.raise_for_status()
        details = detail_response.json()
        turnover = float(details.get("is", {}).get("turnover", math.nan))
    finally:
        _close_simulator(simulator)

    correlator = correlation_factory(project_root=config.project_root)
    correlations = correlator.calculate(
        alpha=simulation.alpha_id,
        calc_type=config.correlation_type,
        skip_cache=False,
        show_detail=False,
    )
    correlation = correlations.get(simulation.alpha_id)
    passed, reasons = evaluate_metrics(
        sharpe=float(simulation.sharpe),
        fitness=float(simulation.fitness),
        turnover=turnover,
        correlation=correlation,
        config=config,
    )
    if simulation.fail_num:
        passed = False
        reasons.append(
            f"BRAIN reported {simulation.fail_num} failed platform checks"
        )

    return ValidationResult(
        expression=expression,
        alpha_id=simulation.alpha_id,
        sharpe=float(simulation.sharpe),
        fitness=float(simulation.fitness),
        turnover=turnover,
        correlation=correlation,
        correlation_type=config.correlation_type,
        drawdown=simulation.drawdown,
        two_year_sharpe=float(simulation.twoyearsharpe),
        failed_checks=int(simulation.fail_num),
        passed=passed,
        reasons=reasons,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Submit FASTEXPR expressions through wqbkit.AlphaSimulator and "
            "validate performance and correlation."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--expression", help="One FASTEXPR expression")
    source.add_argument(
        "--file",
        type=Path,
        help="UTF-8 file containing one expression per line",
    )
    parser.add_argument("--region", default="USA")
    parser.add_argument("--universe", default="TOP3000")
    parser.add_argument("--neutralization", default="SUBINDUSTRY")
    parser.add_argument("--decay", type=int, default=6)
    parser.add_argument("--delay", type=int, choices=(0, 1), default=1)
    parser.add_argument("--min-sharpe", type=float, default=1.25)
    parser.add_argument("--min-fitness", type=float, default=1.0)
    parser.add_argument("--max-turnover", type=float, default=0.7)
    parser.add_argument("--max-correlation", type=float, default=0.7)
    parser.add_argument(
        "--correlation-type",
        choices=("self", "ppac", "prod", "self_web"),
        default="self",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _config_from_args(args) -> ValidationConfig:
    return ValidationConfig(
        region=args.region,
        universe=args.universe,
        neutralization=args.neutralization,
        decay=args.decay,
        delay=args.delay,
        min_sharpe=args.min_sharpe,
        min_fitness=args.min_fitness,
        max_turnover=args.max_turnover,
        max_correlation=args.max_correlation,
        correlation_type=args.correlation_type,
        project_root=Path.cwd(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    try:
        expressions = load_expressions(args.expression, args.file)
        if not expressions:
            raise RuntimeError("No expressions found")

        results = []
        for index, expression in enumerate(expressions, start=1):
            print(f"[{index}/{len(expressions)}] Simulating {expression}")
            result = validate_expression(expression, config)
            results.append(result)
            print(
                f"{'PASS' if result.passed else 'FAIL'} "
                f"alpha={result.alpha_id} sharpe={result.sharpe:.4f} "
                f"fitness={result.fitness:.4f} turnover={result.turnover:.4f} "
                f"{result.correlation_type}_corr={result.correlation}"
            )
            for reason in result.reasons:
                print(f"  - {reason}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    report = [result.to_dict() for result in results]
    if args.output:
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
