from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.config import settings
from app.persistence.cache import create_redis_client
from app.persistence.database import engine


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """管理应用级资源的创建与释放。"""
    application.state.redis = create_redis_client()

    try:
        yield
    finally:
        await application.state.redis.aclose()
        await engine.dispose()


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        debug=settings.debug,
        description=("Observable and recoverable benchmark platform for tool-calling agents."),
        lifespan=lifespan,
    )

    application.include_router(
        health_router,
        prefix="/health",
        tags=["health"],
    )

    @application.get("/", tags=["system"])
    async def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "environment": settings.app_env,
            "status": "running",
            "docs": "/docs",
        }

    return application


app = create_app()
