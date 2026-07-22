import pytest

from app.agent.checkpoint import (
    masked_checkpoint_connection_string,
    resolve_checkpoint_connection_string,
)
from app.config import Settings


def test_derives_psycopg_url_from_asyncpg_url() -> None:
    app_settings = Settings(
        database_url=("postgresql+asyncpg://agentbench:agentbench@localhost:5432/agentbench"),
        checkpoint_database_url=None,
    )

    result = resolve_checkpoint_connection_string(app_settings)

    assert result == ("postgresql://agentbench:agentbench@localhost:5432/agentbench")


def test_explicit_checkpoint_url_takes_priority() -> None:
    app_settings = Settings(
        database_url=("postgresql+asyncpg://app_user:app_password@localhost/app_db"),
        checkpoint_database_url=(
            "postgresql://checkpoint_user:checkpoint_password@localhost/checkpoint_db"
        ),
    )

    result = resolve_checkpoint_connection_string(app_settings)

    assert result.startswith("postgresql://checkpoint_user:")

    assert result.endswith("@localhost/checkpoint_db")


def test_masked_url_does_not_expose_password() -> None:
    app_settings = Settings(
        database_url=("postgresql+asyncpg://agentbench:secret-password@localhost:5432/agentbench"),
        checkpoint_database_url=None,
    )

    result = masked_checkpoint_connection_string(app_settings)

    assert "secret-password" not in result
    assert "***" in result


def test_rejects_non_postgresql_database() -> None:
    app_settings = Settings(
        database_url="sqlite+aiosqlite:///test.db",
        checkpoint_database_url=None,
    )

    with pytest.raises(
        ValueError,
        match="must use PostgreSQL",
    ):
        resolve_checkpoint_connection_string(app_settings)
