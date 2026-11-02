import pytest
from httpx import AsyncClient, ASGITransport
from api.routes import set_cache_service
from app import create_app
from config import Settings
from core.security import SecurityGuard


@pytest.mark.asyncio
async def test_api_ask_flow(cache_service):
    set_cache_service(cache_service)
    app = create_app()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First call -> L3 LLM
        payload = {"query": "Explain quantum computing in simple words"}
        res1 = await client.post("/api/ask", json=payload)
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["level"] == "L3_llm"
        assert "answer" in data1

        # Second call -> L1 exact hit
        res2 = await client.post("/api/ask", json=payload)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["level"] == "L1_exact"
        assert data2["answer"] == data1["answer"]


@pytest.mark.asyncio
async def test_api_ask_stream(cache_service):
    set_cache_service(cache_service)
    app = create_app()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"query": "Tell me a joke about programming"}
        res = await client.post("/api/ask/stream", json=payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        body = res.text
        assert "data: " in body
        assert '"done": true' in body


@pytest.mark.asyncio
async def test_api_auth_and_rate_limiting(tmp_path):
    # App with API key required
    secure_settings = Settings(
        cache_api_key="secret-key-123",
        rate_limit=2,
        sqlite_db_path=str(tmp_path / "sec_cache.db"),
    )
    guard = SecurityGuard(settings=secure_settings)

    from fastapi import FastAPI, Depends
    test_app = FastAPI(dependencies=[Depends(guard)])

    @test_app.get("/api/test")
    def test_endpoint():
        return {"ok": True}

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # No key -> 401
        r_unauth = await client.get("/api/test")
        assert r_unauth.status_code == 401

        # Wrong key -> 401
        r_bad = await client.get("/api/test", headers={"x-api-key": "wrong"})
        assert r_bad.status_code == 401

        # Correct key -> 200
        r_ok1 = await client.get("/api/test", headers={"x-api-key": "secret-key-123"})
        assert r_ok1.status_code == 200

        r_ok2 = await client.get("/api/test", headers={"x-api-key": "secret-key-123"})
        assert r_ok2.status_code == 200

        # 3rd request exceeds rate limit of 2 -> 429
        r_limit = await client.get("/api/test", headers={"x-api-key": "secret-key-123"})
        assert r_limit.status_code == 429


@pytest.mark.asyncio
async def test_health_and_metrics_endpoints(cache_service):
    set_cache_service(cache_service)
    app = create_app()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_health = await client.get("/api/health")
        assert res_health.status_code == 200
        assert "status" in res_health.json()

        res_metrics = await client.get("/metrics")
        assert res_metrics.status_code == 200
        assert "semantic_cache_requests_total" in res_metrics.text

