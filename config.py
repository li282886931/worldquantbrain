"""Load runtime configuration from the project .env file."""

from os import environ
from pathlib import Path
from typing import Iterable

from dotenv import dotenv_values, load_dotenv


PROJECT_ENV_FILE = Path(__file__).resolve().with_name(".env")


def load_project_env(
    env_file: Path | str = PROJECT_ENV_FILE,
    *,
    required: Iterable[str] = (),
) -> dict[str, str]:
    """Load one .env file, overriding the current process environment."""
    path = Path(env_file)
    values = {
        key: value
        for key, value in dotenv_values(path).items()
        if value is not None
    }
    missing = [name for name in required if not values.get(name, "").strip()]
    if missing:
        raise RuntimeError(
            "Missing required .env values: " + ", ".join(missing)
        )

    load_dotenv(path, override=True)
    return values


def load_wqb_credentials(
    env_file: Path | str = PROJECT_ENV_FILE,
) -> tuple[str, str]:
    """Return BRAIN credentials sourced exclusively from the selected .env."""
    load_project_env(
        env_file,
        required=("WQB_USERNAME", "WQB_PASSWORD"),
    )
    return environ["WQB_USERNAME"], environ["WQB_PASSWORD"]
