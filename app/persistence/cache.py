from redis.asyncio import Redis

from app.config import settings


def create_redis_client() -> Redis:
    """创建异步 Redis 客户端。"""
    return Redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )
