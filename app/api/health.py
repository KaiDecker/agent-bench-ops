import asyncio
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.persistence.database import engine

router = APIRouter()


@router.get("/live")
async def liveness() -> dict[str, str]:
    """
    存活检查。

    只检查 FastAPI 进程能否响应请求，
    不依赖 PostgreSQL 和 Redis。
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def check_postgresql() -> dict[str, Any]:
    """检查 PostgreSQL 是否可用。"""
    started_at = perf_counter()

    try:
        async with asyncio.timeout(3):
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
    except Exception:
        return {
            "status": "down",
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        }

    return {
        "status": "up",
        "latency_ms": round((perf_counter() - started_at) * 1000, 2),
    }


async def check_redis(request: Request) -> dict[str, Any]:
    """检查 Redis 是否可用。"""
    started_at = perf_counter()

    try:
        async with asyncio.timeout(3):
            redis_client = request.app.state.redis
            await redis_client.ping()
    except Exception:
        return {
            "status": "down",
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        }

    return {
        "status": "up",
        "latency_ms": round((perf_counter() - started_at) * 1000, 2),
    }


@router.get("/ready")
async def readiness(request: Request) -> JSONResponse:
    """
    就绪检查。

    PostgreSQL 和 Redis 都可用时返回 200；
    任意依赖不可用时返回 503。
    """
    postgresql_status, redis_status = await asyncio.gather(
        check_postgresql(),
        check_redis(request),
    )

    dependencies = {
        "postgresql": postgresql_status,
        "redis": redis_status,
    }

    is_ready = all(dependency["status"] == "up" for dependency in dependencies.values())

    return JSONResponse(
        status_code=(status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE),
        content={
            "status": "ready" if is_ready else "not_ready",
            "timestamp": datetime.now(UTC).isoformat(),
            "dependencies": dependencies,
        },
    )
