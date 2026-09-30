import importlib
import importlib.util

import pytest


def load_config():
    assert importlib.util.find_spec("config") is not None, (
        "config.py has not been created"
    )
    return importlib.import_module("config")


def test_load_wqb_credentials_uses_dotenv_values_over_process_environment(
    tmp_path,
    monkeypatch,
):
    config = load_config()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "WQB_USERNAME=dotenv-user\nWQB_PASSWORD=dotenv-password\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WQB_USERNAME", "process-user")
    monkeypatch.setenv("WQB_PASSWORD", "process-password")

    credentials = config.load_wqb_credentials(env_file)

    assert credentials == ("dotenv-user", "dotenv-password")
    assert config.environ["WQB_USERNAME"] == "dotenv-user"
    assert config.environ["WQB_PASSWORD"] == "dotenv-password"


def test_load_wqb_credentials_rejects_missing_dotenv_values_even_if_set_in_process(
    tmp_path,
    monkeypatch,
):
    config = load_config()
    env_file = tmp_path / ".env"
    env_file.write_text("DB_ENABLE=true\n", encoding="utf-8")
    monkeypatch.setenv("WQB_USERNAME", "process-user")
    monkeypatch.setenv("WQB_PASSWORD", "process-password")

    with pytest.raises(RuntimeError, match="WQB_USERNAME.*WQB_PASSWORD"):
        config.load_wqb_credentials(env_file)
