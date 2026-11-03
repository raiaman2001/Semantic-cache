import io
import pytest
from httpx import ASGITransport, AsyncClient
from api.routes import set_cache_service
from app import create_app


@pytest.mark.asyncio
async def test_import_and_cache_hit_flow(cache_service):
    set_cache_service(cache_service)
    app = create_app()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Test sample CSV download
        res_csv = await client.get("/api/import/sample/csv")
        assert res_csv.status_code == 200
        assert "question,answer" in res_csv.text

        # 2. Test sample JSON download
        res_json = await client.get("/api/import/sample/json")
        assert res_json.status_code == 200
        assert "question" in res_json.text

        # 3. Test bulk import via CSV upload
        csv_data = """question,answer
What is the company return policy?,Items can be returned within 30 days for a full refund.
Where is the main headquarters located?,The headquarters is in San Francisco California.
"""
        files = {"file": ("test_history.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
        res_import = await client.post("/api/import?ns=test_ns", files=files)
        assert res_import.status_code == 200
        data = res_import.json()
        assert data["status"] == "success"
        assert data["imported"] == 2

        # 4. Verify that querying the imported question returns CACHE HIT immediately
        res_ask = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "namespace": "test_ns",
                "messages": [{"role": "user", "content": "What is the company return policy?"}],
            },
        )
        assert res_ask.status_code == 200
        assert "HIT" in res_ask.headers.get("x-cache", "")
        assert "Items can be returned" in res_ask.json()["choices"][0]["message"]["content"]
