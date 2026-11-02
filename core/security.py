import hashlib
import time
from collections import defaultdict, deque
from typing import Any

from fastapi import Header, HTTPException, Request

from config import Settings
from core.logging_config import setup_logger

logger = setup_logger("semantic_cache.security")


class SecurityGuard:
    """Handles API key validation (SHA-256 hashed comparison) and sliding-window rate limiting."""

    def __init__(self, settings: Settings, redis_client: Any | None = None) -> None:
        self.settings = settings
        self.redis = redis_client
        self._local_history: dict[str, deque[float]] = defaultdict(deque)
        self._expected_hash = (
            hashlib.sha256(settings.cache_api_key.encode("utf-8")).hexdigest()
            if settings.cache_api_key
            else None
        )

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    async def check_rate_limit(self, identifier: str) -> bool:
        """Sliding-window rate limiter: Redis if available, local memory fallback."""
        now = time.time()
        window = 60.0
        limit = self.settings.rate_limit

        if self.redis:
            try:
                key = f"rate:{identifier}"
                pipe = self.redis.pipeline()
                pipe.zremrangebyscore(key, 0, now - window)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, int(window) + 1)
                results = await pipe.execute()
                current_count = results[2]
                return current_count <= limit
            except Exception as e:
                logger.warning(f"Redis rate limit check error: {e}. Falling back to memory.")

        # In-memory sliding window fallback
        d = self._local_history[identifier]
        while d and now - d[0] > window:
            d.popleft()
        if len(d) >= limit:
            return False
        d.append(now)
        return True

    async def __call__(self, request: Request, x_api_key: str | None = Header(None)) -> None:
        # Only protect /api routes
        if not request.url.path.startswith("/api"):
            return

        # Exempt prometheus metrics and healthcheck if required
        if request.url.path in ("/api/health", "/metrics"):
            return

        # Check API key if configured
        if self._expected_hash:
            if not x_api_key or self._hash_key(x_api_key) != self._expected_hash:
                raise HTTPException(status_code=401, detail="Invalid API key")

        # Identifier for rate limiting: API key hash if provided, else client IP
        identifier = (
            f"key:{self._hash_key(x_api_key)}"
            if x_api_key
            else f"ip:{request.client.host if request.client else 'unknown'}"
        )

        allowed = await self.check_rate_limit(identifier)
        if not allowed:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
