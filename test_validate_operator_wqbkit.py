import importlib
import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest


def load_validator():
    assert importlib.util.find_spec("validate_operator_wqbkit") is not None, (
        "validate_operator_wqbkit.py has not been created"
    )
    return importlib.import_module("validate_operator_wqbkit")


class FakeResponse:
    headers = {"Location": "https://brain/simulations/1"}

    def raise_for_status(self):
        return None


class FakeWQBSession:
    def locate_alpha(self, alpha_id, log=None):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "id": alpha_id,
                "is": {"turnover": 0.35},
            }
        )


class FakeEvent:
    def __init__(self):
        self.was_set = False

    def set(self):
        self.was_set = True


class FakeExecutor:
    def __init__(self):
        self.shutdown_calls = []

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)


class FakeSimulator:
    instances = []

    def __init__(self, multi_limit, child_limit, project_root=None):
        self.args = (multi_limit, child_limit, project_root)
        self.wqbs = FakeWQBSession()
        self.simulate_shutdown_event = FakeEvent()
        self.process_shutdown_event = FakeEvent()
        self.executor = FakeExecutor()
        self.result_executor = FakeExecutor()
        self.calls = []
        self.__class__.instances.append(self)

    def combine_alpha(self, expression, region, universe, neutralization, decay, delay):
        self.calls.append(("combine_alpha", expression))
        return {"regular": expression}

    def post(self, url, payload):
        self.calls.append(("post", url, payload))
        return FakeResponse()

    def _wait_for_simulation(self, url, prefix):
        self.calls.append(("wait", url))
        return True, object()

    def get_simulation_result(self, url):
        self.calls.append(("result", url))
        return SimpleNamespace(
            alpha_id="alpha-1",
            sharpe=1.6,
            fitness=1.2,
            drawdown=0.08,
            twoyearsharpe=1.3,
            fail_num=0,
        )


class FakeCorrelation:
    def __init__(self, project_root=None):
        self.project_root = project_root

    def calculate(self, alpha, calc_type, skip_cache, show_detail):
        return {alpha: 0.42}


def test_validate_expression_runs_real_simulation_adapter_and_passes_metrics(tmp_path):
    validator = load_validator()
    FakeSimulator.instances.clear()
    config = validator.ValidationConfig(project_root=tmp_path)

    result = validator.validate_expression(
        "rank(close)",
        config,
        simulator_factory=FakeSimulator,
        correlation_factory=FakeCorrelation,
    )

    assert result.passed is True
    assert result.alpha_id == "alpha-1"
    assert result.sharpe == 1.6
    assert result.fitness == 1.2
    assert result.turnover == 0.35
    assert result.correlation == 0.42
    assert result.reasons == []
    simulator = FakeSimulator.instances[0]
    assert [call[0] for call in simulator.calls] == [
        "combine_alpha",
        "post",
        "wait",
        "result",
    ]
    assert simulator.process_shutdown_event.was_set is True


def test_metric_gate_reports_every_failed_threshold():
    validator = load_validator()
    config = validator.ValidationConfig(
        min_sharpe=1.25,
        min_fitness=1.0,
        max_turnover=0.7,
        max_correlation=0.7,
    )

    passed, reasons = validator.evaluate_metrics(
        sharpe=1.0,
        fitness=0.8,
        turnover=0.9,
        correlation=0.8,
        config=config,
    )

    assert passed is False
    assert len(reasons) == 4
    assert any("Sharpe" in reason for reason in reasons)
    assert any("Fitness" in reason for reason in reasons)
    assert any("Turnover" in reason for reason in reasons)
    assert any("Correlation" in reason for reason in reasons)


def test_load_expressions_supports_inline_and_file(tmp_path):
    validator = load_validator()
    expression_file = tmp_path / "expressions.txt"
    expression_file.write_text(
        "# comment\nrank(close)\n\n-ts_delta(close, 5)\n",
        encoding="utf-8",
    )

    assert validator.load_expressions("rank(volume)", None) == ["rank(volume)"]
    assert validator.load_expressions(None, expression_file) == [
        "rank(close)",
        "-ts_delta(close, 5)",
    ]


def test_validation_result_serializes_to_json_compatible_dict():
    validator = load_validator()
    result = validator.ValidationResult(
        expression="rank(close)",
        alpha_id="alpha-1",
        sharpe=1.5,
        fitness=1.1,
        turnover=0.2,
        correlation=0.3,
        correlation_type="self",
        drawdown=0.1,
        two_year_sharpe=1.2,
        failed_checks=0,
        passed=True,
        reasons=[],
    )

    assert result.to_dict()["two_year_sharpe"] == 1.2


def test_validate_environment_rejects_missing_wqb_credentials():
    validator = load_validator()
    environment = {
        "DB_ENABLE": "true",
        "WQB_USERNAME": "",
        "WQB_PASSWORD": "",
    }

    with pytest.raises(RuntimeError, match="WQB_USERNAME.*WQB_PASSWORD"):
        validator.validate_environment(environment)


def test_load_wqbkit_overrides_process_credentials_from_project_dotenv(
    tmp_path,
    monkeypatch,
):
    validator = load_validator()
    (tmp_path / ".env").write_text(
        "DB_ENABLE=true\n"
        "WQB_USERNAME=dotenv-user\n"
        "WQB_PASSWORD=dotenv-password\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WQB_USERNAME", "process-user")
    monkeypatch.setenv("WQB_PASSWORD", "process-password")
    fake_wqbkit = SimpleNamespace(
        AlphaSimulator=object(),
        AlphaCalcCorr=object(),
    )
    monkeypatch.setitem(sys.modules, "wqbkit", fake_wqbkit)

    validator._load_wqbkit(tmp_path)

    assert os.environ["WQB_USERNAME"] == "dotenv-user"
    assert os.environ["WQB_PASSWORD"] == "dotenv-password"


def test_main_reports_runtime_error_without_traceback(monkeypatch, capsys):
    validator = load_validator()

    def fail_validation(expression, config):
        raise RuntimeError("credentials are missing")

    monkeypatch.setattr(validator, "validate_expression", fail_validation)

    exit_code = validator.main(["--expression", "rank(close)"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Error: credentials are missing" in captured.err
    assert "Traceback" not in captured.err
