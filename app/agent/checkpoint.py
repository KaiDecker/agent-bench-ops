from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver,
)
from sqlalchemy.engine import make_url

from app.config import Settings, settings


def resolve_checkpoint_connection_string(
    app_settings: Settings = settings,
) -> str:
    """
    返回 Psycopg 可识别的 PostgreSQL 连接字符串。

    优先使用 checkpoint_database_url；未配置时从
    SQLAlchemy database_url 派生。
    """

    source_url = app_settings.checkpoint_database_url or app_settings.database_url

    parsed_url = make_url(source_url)

    if parsed_url.get_backend_name() != "postgresql":
        raise ValueError("Checkpoint database must use PostgreSQL")

    # postgresql+asyncpg:// -> postgresql://
    psycopg_url = parsed_url.set(drivername="postgresql")

    return psycopg_url.render_as_string(hide_password=False)


def masked_checkpoint_connection_string(
    app_settings: Settings = settings,
) -> str:
    """返回隐藏密码的连接字符串，用于日志输出。"""

    connection_string = resolve_checkpoint_connection_string(app_settings)

    return make_url(connection_string).render_as_string(hide_password=True)


@asynccontextmanager
async def open_postgres_checkpointer(
    app_settings: Settings = settings,
) -> AsyncIterator[AsyncPostgresSaver]:
    """
    打开 PostgreSQL Checkpointer。

    调用方必须在该上下文内部编译和使用持久化图，
    不能在上下文退出后继续使用 checkpointer。
    """

    connection_string = resolve_checkpoint_connection_string(app_settings)

    async with AsyncPostgresSaver.from_conn_string(
        connection_string,
        pipeline=False,
    ) as checkpointer:
        yield checkpointer


async def setup_postgres_checkpointer(
    app_settings: Settings = settings,
) -> None:
    """创建或升级 LangGraph checkpoint 数据表。"""

    async with open_postgres_checkpointer(app_settings) as checkpointer:
        await checkpointer.setup()


__all__ = [
    "masked_checkpoint_connection_string",
    "open_postgres_checkpointer",
    "resolve_checkpoint_connection_string",
    "setup_postgres_checkpointer",
]
