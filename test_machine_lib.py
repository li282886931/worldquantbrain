from datetime import datetime

import pandas as pd
import pytest
import requests

import machine_lib


class FakeResponse:
    def __init__(self, payload=None, status_code=200, headers=None, content=b""):
        self._payload = payload or {}
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content
        self.url = "https://api.worldquantbrain.com/authentication"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class RecordingSession:
    def __init__(self, responses=None):
        self.auth = None
        self.responses = list(responses or [])
        self.calls = []

    def _request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        return self._request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._request("POST", url, **kwargs)

    def patch(self, url, **kwargs):
        return self._request("PATCH", url, **kwargs)


def test_login_reads_credentials_from_dotenv_loader(monkeypatch):
    session = RecordingSession([FakeResponse(status_code=201)])
    monkeypatch.setattr(
        machine_lib,
        "load_wqb_credentials",
        lambda: ("alice", "secret"),
    )
    monkeypatch.setattr(machine_lib.requests, "Session", lambda: session)

    result = machine_lib.login()

    assert result is session
    assert session.auth == ("alice", "secret")
    assert session.calls[0][2]["timeout"] == machine_lib.DEFAULT_TIMEOUT


def test_login_rejects_missing_dotenv_credentials(monkeypatch):
    def fail_loading():
        raise RuntimeError("Missing required .env values: WQB_USERNAME")

    monkeypatch.setattr(machine_lib, "load_wqb_credentials", fail_loading)
    with pytest.raises(RuntimeError, match="WQB_USERNAME"):
        machine_lib.login()


def test_get_datasets_uses_params_and_checks_response():
    session = RecordingSession([FakeResponse({"results": [{"id": "fundamental"}]})])

    result = machine_lib.get_datasets(session, region="EUR", universe="TOP1200")

    assert result.to_dict("records") == [{"id": "fundamental"}]
    method, url, kwargs = session.calls[0]
    assert method == "GET"
    assert url.endswith("/data-sets")
    assert kwargs["params"]["region"] == "EUR"
    assert kwargs["params"]["universe"] == "TOP1200"
    assert kwargs["timeout"] == machine_lib.DEFAULT_TIMEOUT


def test_get_datafields_search_uses_server_count_for_all_pages():
    responses = [
        FakeResponse({"count": 120, "results": [{"id": "field-0"}]}),
        FakeResponse({"count": 120, "results": [{"id": "field-50"}]}),
        FakeResponse({"count": 120, "results": [{"id": "field-100"}]}),
    ]
    session = RecordingSession(responses)

    result = machine_lib.get_datafields(session, search="earnings surprise")

    assert result["id"].tolist() == ["field-0", "field-50", "field-100"]
    assert [call[2]["params"]["offset"] for call in session.calls] == [0, 50, 100]
    assert all(call[2]["params"]["search"] == "earnings surprise" for call in session.calls)


def test_process_datafields_validates_required_columns():
    with pytest.raises(ValueError, match="id.*type|type.*id"):
        machine_lib.process_datafields(pd.DataFrame({"id": ["x"]}))


def test_group_factory_only_adds_usa_specific_groups_for_usa():
    usa = machine_lib.group_factory("group_rank", "field", "USA")
    eur = machine_lib.group_factory("group_rank", "field", "EUR")

    assert any("pv13_h_min2_3000_sector" in expression for expression in usa)
    assert not any("pv13_" in expression for expression in eur)
    assert len(usa) == len(set(usa))


def test_trade_when_factory_adds_only_requested_region_events():
    usa = machine_lib.trade_when_factory("trade_when", "field", "USA")
    eur = machine_lib.trade_when_factory("trade_when", "field", "EUR")

    assert any("rp_css_business" in expression for expression in usa)
    assert not any("oth429_research" in expression for expression in usa)
    assert any("oth429_research" in expression for expression in eur)
    assert not any("mws82_sentiment" in expression for expression in eur)


def test_set_alpha_properties_checks_and_returns_response():
    response = FakeResponse({"id": "alpha-1"})
    session = RecordingSession([response])

    result = machine_lib.set_alpha_properties(session, "alpha-1", tags=None)

    assert result is response
    assert session.calls[0][2]["json"]["tags"] == ["ace_tag"]
    assert session.calls[0][2]["timeout"] == machine_lib.DEFAULT_TIMEOUT


def test_get_alphas_uses_current_year_for_month_day(monkeypatch):
    session = RecordingSession([FakeResponse({"results": []})])
    monkeypatch.setattr(machine_lib, "login", lambda: session)

    machine_lib.get_alphas("01-02", "02-03", 1.2, 1.0, "USA", 1, "submit")

    params = session.calls[0][2]["params"]
    assert params["dateCreated>"] == f"{datetime.now().year}-01-02T00:00:00-04:00"
    assert params["dateCreated<"] == f"{datetime.now().year}-02-03T00:00:00-04:00"


def test_get_alphas_preserves_explicit_year(monkeypatch):
    session = RecordingSession([FakeResponse({"results": []})])
    monkeypatch.setattr(machine_lib, "login", lambda: session)

    machine_lib.get_alphas("2024-12-01", "2025-01-01", 1.2, 1.0, "USA", 1, "submit")

    params = session.calls[0][2]["params"]
    assert params["dateCreated>"].startswith("2024-12-01")
    assert params["dateCreated<"].startswith("2025-01-01")


def test_task_pool_validates_positive_limits():
    with pytest.raises(ValueError, match="positive"):
        machine_lib.load_task_pool_single([("alpha", 1)], 0)

    with pytest.raises(ValueError, match="positive"):
        machine_lib.load_task_pool([("alpha", 1)], 10, 0)


def test_ts_comp_factory_rejects_unsupported_parameter_types():
    with pytest.raises(TypeError, match="int or float"):
        machine_lib.ts_comp_factory("op", "field", "factor", ["bad"])


def test_single_simulate_reports_failed_posts_without_long_sleep(monkeypatch):
    session = RecordingSession([FakeResponse(status_code=400, content=b"failed")])
    sleeps = []
    monkeypatch.setattr(machine_lib, "login", lambda: session)
    monkeypatch.setattr(machine_lib, "sleep", sleeps.append)

    failures = machine_lib.single_simulate(
        [[("rank(close)", 1)]],
        "SECTOR",
        "USA",
        "TOP3000",
        0,
    )

    assert failures == [("rank(close)", "HTTP 400")]
    # 仅允许短延时（节流用），不允许长休眠
    assert all(s < 1 for s in sleeps)


def test_multi_simulate_handles_empty_pool(monkeypatch):
    session = RecordingSession()
    monkeypatch.setattr(machine_lib, "login", lambda: session)

    assert machine_lib.multi_simulate([[]], "SECTOR", "USA", "TOP3000", 0) == []


def test_check_submission_retries_without_mutating_input(monkeypatch):
    alpha_bag = ["alpha-1"]
    results = iter(["sleep", 0.25])
    monkeypatch.setattr(machine_lib, "login", lambda: object())
    monkeypatch.setattr(machine_lib, "get_check_submission", lambda *_: next(results))
    monkeypatch.setattr(machine_lib, "sleep", lambda _: None)

    gold = machine_lib.check_submission(alpha_bag, [], 0)

    assert alpha_bag == ["alpha-1"]
    assert gold == [("alpha-1", 0.25)]


def test_locate_alpha_checks_response_and_uses_json():
    payload = {
        "dateCreated": "2026-01-01",
        "is": {
            "sharpe": 1.5,
            "fitness": 1.1,
            "turnover": 0.2,
            "margin": 0.01,
        },
        "settings": {"decay": 4},
        "regular": {"code": "rank(close)"},
    }
    session = RecordingSession([FakeResponse(payload)])

    result = machine_lib.locate_alpha(session, "alpha-1")

    assert result == [
        "alpha-1",
        "rank(close)",
        1.5,
        0.2,
        1.1,
        0.01,
        "2026-01-01",
        4,
    ]
    assert session.calls[0][2]["timeout"] == machine_lib.DEFAULT_TIMEOUT


def test_login_hk_reads_credentials_from_dotenv_loader(monkeypatch):
    session = RecordingSession([FakeResponse(status_code=201)])
    monkeypatch.setattr(
        machine_lib,
        "load_wqb_credentials",
        lambda: ("alice", "secret"),
    )
    monkeypatch.setattr(machine_lib.requests, "Session", lambda: session)

    result = machine_lib.login_hk()

    assert result is session
    assert session.auth == ("alice", "secret")
    assert session.calls[0][2]["timeout"] == machine_lib.DEFAULT_TIMEOUT
