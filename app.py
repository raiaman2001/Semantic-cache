import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Depends, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router as api_router, set_cache_service
from config import settings
from core.cache import SemanticCacheService
from core.embedding import EmbeddingService
from core.llm import LLMService
from core.logging_config import setup_logger
from core.metrics import render_metrics
from core.rerank import RerankService
from core.security import SecurityGuard
from db.database import Database

logger = setup_logger("semantic_cache", settings.log_level)

redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    logger.info("Initializing Semantic Cache services...")

    # Optional async Redis client
    if settings.redis_url:
        try:
            import redis.asyncio as aioredis
            redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
            await redis_client.ping()
            logger.info("Connected to Redis L1 cache.")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis at {settings.redis_url}: {e}. Running with SQLite L1.")
            redis_client = None

    db = Database(settings.sqlite_db_path)
    embedding_svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
    rerank_svc = RerankService(
        model_name="cross-encoder/stsb-TinyBERT-L-4",
        min_score=settings.rerank_min,
        enabled=settings.rerank_enabled,
    )
    llm_svc = LLMService(
        api_key=settings.gemini_api_key,
        default_model=settings.gemini_model,
        fallback_model=settings.gemini_fallback_model,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )

    cache_service = SemanticCacheService(
        settings=settings,
        db=db,
        embedding_service=embedding_svc,
        rerank_service=rerank_svc,
        llm_service=llm_svc,
        redis_client=redis_client,
    )
    set_cache_service(cache_service)
    app.state.cache_service = cache_service

    yield

    logger.info("Shutting down Semantic Cache services...")
    if redis_client:
        await redis_client.close()
    db.close()


def create_app() -> FastAPI:
    guard = SecurityGuard(settings=settings)
    application = FastAPI(
        title="Semantic Cache",
        version="2.0.0",
        description="Production-grade multi-tier LLM semantic cache",
        lifespan=lifespan,
        dependencies=[Depends(guard)],
    )

    application.include_router(api_router, credentials=True if False else None, prefix="/api")

    @application.get("/metrics", include_in_schema=False)
    def metrics():
        data, content_type = render_metrics()
        return Response(content=data, media_type=content_type)

    @application.get("/", include_in_schema=False)
    def index():
        return FileResponse("static/index.html")

    if os.path.exists("static"):
        application.mount("/static", StaticFiles(directory="static"), name="static")

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
