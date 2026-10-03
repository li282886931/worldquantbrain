import importlib
import importlib.util

import pandas as pd
import pytest


def load_alpha_machine():
    assert importlib.util.find_spec("alpha_machine") is not None, (
        "alpha_machine.py has not been created"
    )
    return importlib.import_module("alpha_machine")


class FakeMachineLib:
    ts_ops = ["ts_rank"]

    def __init__(self):
        self.calls = []
        self.alpha_queries = 0
        self.prune_calls = 0

    def login(self):
        self.calls.append(("login",))
        return object()

    def get_datafields(self, session, **kwargs):
        self.calls.append(("get_datafields", kwargs["dataset_id"]))
        field_id = {
            "analyst4": "anl4_field",
            "option8": "option_field",
            "sentiment1": "sentiment_field",
        }[kwargs["dataset_id"]]
        return pd.DataFrame([{"id": field_id, "type": "MATRIX"}])

    def process_datafields(self, frame):
        self.calls.append(("process_datafields",))
        return ["processed_field"]

    def first_order_factory(self, fields, operators):
        self.calls.append(("first_order_factory",))
        return ["first_order_alpha"]

    def load_task_pool_single(self, alpha_list, limit):
        self.calls.append(("load_task_pool_single", tuple(alpha_list)))
        return [alpha_list]

    def single_simulate(self, pools, neutralization, region, universe, start, on_batch_done=None):
        alpha = pools[0][0][0]
        self.calls.append(("single_simulate", alpha, start))
        if on_batch_done:
            on_batch_done(0)
        return []

    def get_alphas(self, *args):
        self.alpha_queries += 1
        usage = args[-1]
        self.calls.append(("get_alphas", usage))
        return [[f"alpha-{self.alpha_queries}", "expression", 1.5, 0.2]]

    def prune(self, records, prefix, keep_num):
        self.prune_calls += 1
        self.calls.append(("prune", prefix, keep_num))
        return [(f"layer-{self.prune_calls}", 6)]

    def get_group_second_order_factory(self, first_order, group_ops, region):
        self.calls.append(("group_factory",))
        return ["second_order_alpha"]

    def trade_when_factory(self, operator, expression, region):
        self.calls.append(("trade_when_factory",))
        return ["third_order_alpha"]

    def check_submission(self, stone_bag, gold_bag, start):
        self.calls.append(("check_submission", tuple(stone_bag)))
        gold_bag.append(("alpha-3", 0.2))
        return gold_bag

    def view_alphas(self, gold_bag):
        self.calls.append(("view_alphas", tuple(gold_bag)))


def test_default_config_matches_notebook_values():
    alpha_machine = load_alpha_machine()

    config = alpha_machine.WorkflowConfig()

    assert config.dataset_id == "analyst4"
    assert config.field_prefix == "anl4"
    assert config.region == "USA"
    assert config.universe == "TOP3000"
    assert config.start_date == "02-27"
    assert config.end_date == "02-28"
    assert config.third_simulation_start == 3


def test_run_workflow_executes_all_notebook_stages(tmp_path, monkeypatch):
    alpha_machine = load_alpha_machine()
    monkeypatch.setattr(
        alpha_machine, "CHECKPOINT_FILE", tmp_path / "checkpoint.json"
    )
    fake = FakeMachineLib()

    result = alpha_machine.run_workflow(
        alpha_machine.WorkflowConfig(random_seed=7),
        api=fake,
    )

    stages = [call[0] for call in fake.calls]
    assert stages == [
        "login",
        "get_datafields",
        "process_datafields",
        "first_order_factory",
        "load_task_pool_single",
        "single_simulate",
        "get_alphas",
        "prune",
        "group_factory",
        "load_task_pool_single",
        "single_simulate",
        "get_alphas",
        "prune",
        "trade_when_factory",
        "load_task_pool_single",
        "single_simulate",
        "get_alphas",
        "check_submission",
        "view_alphas",
        "get_datafields",
        "get_datafields",
    ]
    assert result.first_order_count == 1
    assert result.second_order_count == 1
    assert result.third_order_count == 1
    assert result.submittable_count == 1
    assert result.template_count == 1


def test_run_workflow_stops_when_no_data_fields_are_available(tmp_path, monkeypatch):
    alpha_machine = load_alpha_machine()
    monkeypatch.setattr(
        alpha_machine, "CHECKPOINT_FILE", tmp_path / "checkpoint.json"
    )
    fake = FakeMachineLib()
    fake.process_datafields = lambda frame: []

    with pytest.raises(RuntimeError, match="No usable data fields"):
        alpha_machine.run_workflow(alpha_machine.WorkflowConfig(), api=fake)


def test_template_factory_creates_balanced_expression():
    alpha_machine = load_alpha_machine()

    expressions = alpha_machine.template_factory(["sentiment"], ["option"])

    assert expressions == [
        "log(1+sigmoid(ts_zscore(sentiment,30))*"
        "sigmoid(ts_zscore(option,30)))"
    ]


def test_run_workflow_resumes_from_checkpoint(tmp_path, monkeypatch):
    alpha_machine = load_alpha_machine()
    ckpt_file = tmp_path / "checkpoint.json"
    monkeypatch.setattr(alpha_machine, "CHECKPOINT_FILE", ckpt_file)

    # 模拟断点：一阶数据已生成，模拟已做到第 1 批
    ckpt_file.write_text(
        '{"stage": 2, "processed_fields": ["f1"], "first_alpha_list": [["a1", 6], ["a2", 6]], "first_sim_next": 1, "first_layer": [["l1", 6]]}',
        encoding="utf-8",
    )

    fake = FakeMachineLib()
    result = alpha_machine.run_workflow(
        alpha_machine.WorkflowConfig(), api=fake,
    )

    # 应跳过 [1/8] 和 [2/8] 的数据生成，直接从 [3/8] 开始
    assert fake.calls[0][0] == "login"
    assert fake.calls[1][0] == "get_alphas"
    assert fake.calls[2][0] == "prune"
    assert result.first_order_count == 2


def test_parse_args_supports_runtime_overrides():
    alpha_machine = load_alpha_machine()

    config = alpha_machine.parse_args([
        "--dataset-id",
        "custom",
        "--region",
        "EUR",
        "--start-date",
        "2026-01-01",
        "--end-date",
        "2026-01-31",
        "--first-sharpe",
        "1.1",
        "--second-alpha-limit",
        "300",
        "--submit-fitness",
        "1.2",
        "--option-dataset-id",
        "option-custom",
        "--seed",
        "9",
    ])

    assert config.dataset_id == "custom"
    assert config.region == "EUR"
    assert config.start_date == "2026-01-01"
    assert config.end_date == "2026-01-31"
    assert config.first_sharpe == 1.1
    assert config.second_alpha_limit == 300
    assert config.submit_fitness == 1.2
    assert config.option_dataset_id == "option-custom"
    assert config.random_seed == 9
