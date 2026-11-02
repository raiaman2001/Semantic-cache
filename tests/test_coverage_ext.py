import logging

import pytest
from httpx import ASGITransport, AsyncClient

from api.routes import set_cache_service
from app import create_app
from core.logging_config import JSONFormatter, setup_logger


@pytest.mark.asyncio
async def test_api_routes_extended(cache_service):
    set_cache_service(cache_service)
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Config endpoint
        r_cfg = await client.get("/api/config")
        assert r_cfg.status_code == 200
        assert "l1" in r_cfg.json()

        # Warm endpoint
        r_warm = await client.post("/api/warm", json={"queries": ["warm query 1", "warm query 2"], "ns": "default"})
        assert r_warm.status_code == 200
        assert r_warm.json()["warmed"] == 2

        # Entries endpoint
        r_entries = await client.get("/api/entries")
        assert r_entries.status_code == 200
        entries = r_entries.json()
        assert len(entries) >= 2

        # Invalidate endpoint
        r_inv = await client.post("/api/invalidate", json={"tag": "", "ns": "default"})
        assert r_inv.status_code == 200

        # Stats endpoint
        r_stats = await client.get("/api/stats")
        assert r_stats.status_code == 200
        assert "hit_rate" in r_stats.json()

        # Simulate endpoint
        r_sim = await client.get("/api/simulate?queries=100&hit_rate=0.5")
        assert r_sim.status_code == 200
        assert r_sim.json()["llm_calls_saved"] == 50

        # Threshold endpoint
        r_thr = await client.post("/api/threshold", json={"code": 0.95, "general": 0.89})
        assert r_thr.status_code == 200
        assert r_thr.json()["code"] == 0.95

        # Clear endpoint
        r_clr = await client.post("/api/clear")
        assert r_clr.status_code == 200


def test_logging_formatter():
    setup_logger("test_json_logger")
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.props = {"api_key": "secret_abc", "user_id": 42}
    formatted = formatter.format(record)
    assert "[REDACTED]" in formatted
    assert "secret_abc" not in formatted
