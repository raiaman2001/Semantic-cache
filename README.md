# Semantic Cache
Three-level LLM cache: L1 exact hash, L2 semantic (FAISS + MiniLM), L3 LLM (Gemini, or a mock if no key).

## Run
```
pip install -r requirements.txt
export GEMINI_API_KEY=your_key      # optional, mock answers otherwise
uvicorn app:app --reload            # open http://localhost:8000
```
Optional env: CACHE_API_KEY, RATE_LIMIT, MAX_ENTRIES, TTL, GEMINI_MODEL.

## Features
Persistence (SQLite), TTL, LRU eviction, namespaces and model-aware keys, doc-version invalidation, tag invalidation,
borderline match verification (blocks enable/disable style false positives), per-category thresholds that adapt from feedback,
cache warming, stats and cost saved, evaluation curve, simulator, API key auth, rate limiting.

## v2 upgrades
- **Qdrant** (local by default, or `QDRANT_URL=http://localhost:6333`) replaces FAISS: incremental inserts, server-side namespace/model/version filters.
- **Redis** (`REDIS_URL=redis://localhost:6379`) serves L1 exact hits with native TTL. `docker compose up -d` starts both.
- **Async + batching:** embeddings arriving within 10 ms are encoded together; identical concurrent misses share one LLM call.
- **Streaming:** `POST /api/ask/stream` (SSE). Cache hits stream too, so hit and miss feel the same.
- **Reranker:** cross-encoder (`RERANK_MIN`, default 0.6) re-checks borderline matches. `RERANK=0` disables it.
- Delete the old `cache.db` before first v2 run.
