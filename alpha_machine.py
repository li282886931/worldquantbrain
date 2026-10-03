#!/usr/bin/env python3
"""Run the complete Alpha Machine workflow without Jupyter."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Sequence

import machine_lib

CHECKPOINT_FILE = Path("alpha_machine_checkpoint.json")


@dataclass(frozen=True)
class WorkflowConfig:
    dataset_id: str = "analyst4"
    field_prefix: str = "anl4"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    neutralization: str = "SUBINDUSTRY"
    initial_decay: int = 6
    simulations_per_batch: int = 3
    start_date: str = "02-27"
    end_date: str = "02-28"
    prune_keep: int = 5
    first_sharpe: float = 1.0
    first_fitness: float = 0.7
    first_alpha_limit: int = 100
    second_sharpe: float = 1.3
    second_fitness: float = 0.8
    second_alpha_limit: int = 200
    submit_sharpe: float = 1.25
    submit_fitness: float = 1.0
    submit_alpha_limit: int = 200
    first_simulation_start: int = 0
    second_simulation_start: int = 0
    third_simulation_start: int = 3
    random_seed: int | None = None
    option_dataset_id: str = "option8"
    sentiment_dataset_id: str = "sentiment1"
    fresh: bool = False


@dataclass(frozen=True)
class WorkflowResult:
    data_field_count: int
    first_order_count: int
    second_order_count: int
    third_order_count: int
    submittable_count: int
    template_count: int


def _require_nonempty(values: Sequence[Any], message: str) -> None:
    if not values:
        raise RuntimeError(message)


def load_checkpoint() -> dict:
    """读取断点文件；不存在或损坏时返回空 dict。"""
    try:
        return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_checkpoint(state: dict) -> None:
    """原子写入断点文件。"""
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CHECKPOINT_FILE)


def _simulate_or_raise(
    api,
    alpha_list,
    config: WorkflowConfig,
    start: int,
    stage: str,
    checkpoint: dict | None = None,
    progress_key: str | None = None,
) -> None:
    pools = api.load_task_pool_single(
        alpha_list,
        config.simulations_per_batch,
    )

    on_batch_done = None
    if checkpoint is not None and progress_key:
        def on_batch_done(batch_index):
            checkpoint[progress_key] = batch_index + 1
            save_checkpoint(checkpoint)

    failures = api.single_simulate(
        pools,
        config.neutralization,
        config.region,
        config.universe,
        start,
        on_batch_done=on_batch_done,
    )
    if failures:
        raise RuntimeError(f"{stage} simulation failed: {failures}")
    if checkpoint is not None and progress_key:
        checkpoint[progress_key] = len(pools)
        save_checkpoint(checkpoint)


def template_factory(sentiment_fields, option_fields):
    expressions = []
    for sentiment_field in sentiment_fields:
        for option_field in option_fields:
            expressions.append(
                "log(1+sigmoid(ts_zscore(%s,30))*sigmoid(ts_zscore(%s,30)))"
                % (sentiment_field, option_field)
            )
    return expressions


def run_workflow(config: WorkflowConfig, api=machine_lib) -> WorkflowResult:
    rng = random.Random(config.random_seed)
    ckpt = {} if config.fresh else load_checkpoint()
    stage = ckpt.get("stage", 0)
    if stage:
        print(f"Resuming from checkpoint (stage {stage})")
    session = api.login()

    # [1/8] 获取主数据集字段
    if stage < 1:
        print("[1/8] Fetching primary data fields")
        data_frame = api.get_datafields(
            session,
            dataset_id=config.dataset_id,
            region=config.region,
            universe=config.universe,
            delay=config.delay,
        )
        processed_fields = api.process_datafields(data_frame)
        _require_nonempty(processed_fields, "No usable data fields were returned")
        ckpt["processed_fields"] = list(processed_fields)
        ckpt["stage"] = 1
        save_checkpoint(ckpt)
    processed_fields = ckpt["processed_fields"]

    # [2/8] 生成并模拟一阶 alpha
    if stage < 2:
        print("[2/8] Generating and simulating first-order alphas")
        if "first_alpha_list" in ckpt:
            first_alpha_list = [tuple(x) for x in ckpt["first_alpha_list"]]
        else:
            first_order = api.first_order_factory(processed_fields, api.ts_ops)
            _require_nonempty(first_order, "No first-order alphas were generated")
            first_alpha_list = [
                (expression, config.initial_decay) for expression in first_order
            ]
            rng.shuffle(first_alpha_list)
            ckpt["first_alpha_list"] = [list(x) for x in first_alpha_list]
            ckpt["first_sim_next"] = 0
            save_checkpoint(ckpt)
        start = max(config.first_simulation_start, ckpt.get("first_sim_next", 0))
        _simulate_or_raise(
            api, first_alpha_list, config, start, "First-order",
            checkpoint=ckpt, progress_key="first_sim_next",
        )
        ckpt["stage"] = 2
        save_checkpoint(ckpt)
    first_alpha_list = [tuple(x) for x in ckpt["first_alpha_list"]]

    # [3/8] 筛选一阶候选
    if stage < 3:
        print("[3/8] Selecting first-order candidates")
        first_tracker = api.get_alphas(
            config.start_date,
            config.end_date,
            config.first_sharpe,
            config.first_fitness,
            config.region,
            config.first_alpha_limit,
            "track",
        )
        first_layer = api.prune(
            first_tracker,
            config.field_prefix,
            config.prune_keep,
        )
        _require_nonempty(first_layer, "No first-order candidates survived pruning")
        ckpt["first_layer"] = [list(x) for x in first_layer]
        ckpt["stage"] = 3
        save_checkpoint(ckpt)
    first_layer = [tuple(x) for x in ckpt["first_layer"]]

    # [4/8] 生成并模拟二阶 alpha
    if stage < 4:
        print("[4/8] Generating and simulating second-order alphas")
        if "second_alpha_list" in ckpt:
            second_alpha_list = [tuple(x) for x in ckpt["second_alpha_list"]]
        else:
            group_ops = ["group_neutralize", "group_rank", "group_zscore"]
            second_alpha_list = []
            for expression, decay in first_layer:
                generated = api.get_group_second_order_factory(
                    [expression],
                    group_ops,
                    config.region,
                )
                second_alpha_list.extend((alpha, decay) for alpha in generated)
            _require_nonempty(second_alpha_list, "No second-order alphas were generated")
            rng.shuffle(second_alpha_list)
            ckpt["second_alpha_list"] = [list(x) for x in second_alpha_list]
            ckpt["second_sim_next"] = 0
            save_checkpoint(ckpt)
        start = max(config.second_simulation_start, ckpt.get("second_sim_next", 0))
        _simulate_or_raise(
            api, second_alpha_list, config, start, "Second-order",
            checkpoint=ckpt, progress_key="second_sim_next",
        )
        ckpt["stage"] = 4
        save_checkpoint(ckpt)
    second_alpha_list = [tuple(x) for x in ckpt["second_alpha_list"]]

    # [5/8] 筛选二阶候选
    if stage < 5:
        print("[5/8] Selecting second-order candidates")
        second_tracker = api.get_alphas(
            config.start_date,
            config.end_date,
            config.second_sharpe,
            config.second_fitness,
            config.region,
            config.second_alpha_limit,
            "track",
        )
        second_layer = api.prune(
            second_tracker,
            config.field_prefix,
            config.prune_keep,
        )
        _require_nonempty(second_layer, "No second-order candidates survived pruning")
        ckpt["second_layer"] = [list(x) for x in second_layer]
        ckpt["stage"] = 5
        save_checkpoint(ckpt)
    second_layer = [tuple(x) for x in ckpt["second_layer"]]

    # [6/8] 生成并模拟三阶 alpha
    if stage < 6:
        print("[6/8] Generating and simulating third-order alphas")
        if "third_alpha_list" in ckpt:
            third_alpha_list = [tuple(x) for x in ckpt["third_alpha_list"]]
        else:
            third_alpha_list = []
            for expression, decay in second_layer:
                generated = api.trade_when_factory(
                    "trade_when",
                    expression,
                    config.region,
                )
                third_alpha_list.extend((alpha, decay) for alpha in generated)
            _require_nonempty(third_alpha_list, "No third-order alphas were generated")
            rng.shuffle(third_alpha_list)
            ckpt["third_alpha_list"] = [list(x) for x in third_alpha_list]
            ckpt["third_sim_next"] = 0
            save_checkpoint(ckpt)
        start = max(config.third_simulation_start, ckpt.get("third_sim_next", 0))
        _simulate_or_raise(
            api, third_alpha_list, config, start, "Third-order",
            checkpoint=ckpt, progress_key="third_sim_next",
        )
        ckpt["stage"] = 6
        save_checkpoint(ckpt)
    third_alpha_list = [tuple(x) for x in ckpt["third_alpha_list"]]

    # [7/8] 检查可提交候选
    if stage < 7:
        print("[7/8] Checking submission candidates")
        submit_tracker = api.get_alphas(
            config.start_date,
            config.end_date,
            config.submit_sharpe,
            config.submit_fitness,
            config.region,
            config.submit_alpha_limit,
            "submit",
        )
        stone_bag = [alpha[0] for alpha in submit_tracker]
        gold_bag = api.check_submission(stone_bag, [], 0) if stone_bag else []
        api.view_alphas(gold_bag)
        ckpt["gold_bag"] = [list(x) for x in gold_bag]
        ckpt["stage"] = 7
        save_checkpoint(ckpt)
    gold_bag = ckpt.get("gold_bag", [])

    # [8/8] 构建模板表达式
    print("[8/8] Building Appendix template expressions")
    option_frame = api.get_datafields(
        session,
        dataset_id=config.option_dataset_id,
        region=config.region,
        universe=config.universe,
        delay=config.delay,
    )
    sentiment_frame = api.get_datafields(
        session,
        dataset_id=config.sentiment_dataset_id,
        region=config.region,
        universe=config.universe,
        delay=config.delay,
    )
    option_fields = option_frame.loc[
        option_frame["type"] == "MATRIX",
        "id",
    ].tolist()
    sentiment_fields = sentiment_frame.loc[
        sentiment_frame["type"] == "MATRIX",
        "id",
    ].tolist()
    template_alphas = template_factory(sentiment_fields, option_fields)

    ckpt["stage"] = 8
    save_checkpoint(ckpt)

    result = WorkflowResult(
        data_field_count=len(processed_fields),
        first_order_count=len(first_alpha_list),
        second_order_count=len(second_alpha_list),
        third_order_count=len(third_alpha_list),
        submittable_count=len(gold_bag),
        template_count=len(template_alphas),
    )
    print("Workflow complete:")
    for name, value in asdict(result).items():
        print(f"  {name}: {value}")
    print(f"Checkpoint saved to {CHECKPOINT_FILE} (use --fresh to restart from scratch)")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete WorldQuant Alpha Machine workflow."
    )
    parser.add_argument("--dataset-id", default="analyst4")
    parser.add_argument("--field-prefix", default="anl4")
    parser.add_argument("--region", default="USA")
    parser.add_argument("--universe", default="TOP3000")
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--neutralization", default="SUBINDUSTRY")
    parser.add_argument("--initial-decay", type=int, default=6)
    parser.add_argument("--simulations-per-batch", type=int, default=3)
    parser.add_argument("--start-date", default="02-27")
    parser.add_argument("--end-date", default="02-28")
    parser.add_argument("--prune-keep", type=int, default=5)
    parser.add_argument("--first-sharpe", type=float, default=1.0)
    parser.add_argument("--first-fitness", type=float, default=0.7)
    parser.add_argument("--first-alpha-limit", type=int, default=100)
    parser.add_argument("--second-sharpe", type=float, default=1.3)
    parser.add_argument("--second-fitness", type=float, default=0.8)
    parser.add_argument("--second-alpha-limit", type=int, default=200)
    parser.add_argument("--submit-sharpe", type=float, default=1.25)
    parser.add_argument("--submit-fitness", type=float, default=1.0)
    parser.add_argument("--submit-alpha-limit", type=int, default=200)
    parser.add_argument("--first-simulation-start", type=int, default=0)
    parser.add_argument("--second-simulation-start", type=int, default=0)
    parser.add_argument("--third-simulation-start", type=int, default=3)
    parser.add_argument("--option-dataset-id", default="option8")
    parser.add_argument("--sentiment-dataset-id", default="sentiment1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="忽略已有断点文件，从头开始运行",
    )
    return parser


def parse_args(argv=None) -> WorkflowConfig:
    args = build_parser().parse_args(argv)
    return WorkflowConfig(
        dataset_id=args.dataset_id,
        field_prefix=args.field_prefix,
        region=args.region,
        universe=args.universe,
        delay=args.delay,
        neutralization=args.neutralization,
        initial_decay=args.initial_decay,
        simulations_per_batch=args.simulations_per_batch,
        start_date=args.start_date,
        end_date=args.end_date,
        prune_keep=args.prune_keep,
        first_sharpe=args.first_sharpe,
        first_fitness=args.first_fitness,
        first_alpha_limit=args.first_alpha_limit,
        second_sharpe=args.second_sharpe,
        second_fitness=args.second_fitness,
        second_alpha_limit=args.second_alpha_limit,
        submit_sharpe=args.submit_sharpe,
        submit_fitness=args.submit_fitness,
        submit_alpha_limit=args.submit_alpha_limit,
        first_simulation_start=args.first_simulation_start,
        second_simulation_start=args.second_simulation_start,
        third_simulation_start=args.third_simulation_start,
        random_seed=args.seed,
        option_dataset_id=args.option_dataset_id,
        sentiment_dataset_id=args.sentiment_dataset_id,
        fresh=args.fresh,
    )


def main(argv=None) -> int:
    run_workflow(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
