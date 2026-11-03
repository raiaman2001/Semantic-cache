<div align="center">

<img src="https://cdn.simpleicons.org/databricks/FF3621" width="64" alt="Semantic Cache logo" />

# Production LLM Semantic Cache

**A multi-tier semantic caching layer for LLM applications.**
Cut API spend by up to 70% and drop p95 latency from seconds to milliseconds.

[![CI](https://github.com/example/semantic-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/example/semantic-cache/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C.svg?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Redis](https://img.shields.io/badge/Redis-L1%20Cache-DC382D.svg?logo=redis&logoColor=white)](https://redis.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Quickstart](#quickstart) &nbsp;|&nbsp;
[Integration](#drop-in-integration) &nbsp;|&nbsp;
[Configuration](#configuration-reference) &nbsp;|&nbsp;
[Benchmarks](#benchmark-results) &nbsp;|&nbsp;
[API](#api-reference)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Capabilities](#core-capabilities)
- [Drop-In Integration](#drop-in-integration)
- [Quickstart](#quickstart)
- [Configuration Reference](#configuration-reference)
- [Benchmark Results](#benchmark-results)
- [API Reference](#api-reference)
- [Verification and Code Quality](#verification-and-code-quality)
- [Technical Design Decisions](#technical-design-decisions)

---

## Overview

The cache sits between your application and the LLM provider. Each request passes through three stages, and only a true miss reaches the model:

| Stage | Mechanism | Typical role |
|:--|:--|:--|
| **Coalescing** | Concurrent identical misses share one LLM task | Prevents stampeding herds |
| **L1 Exact** | SHA-256 normalized-query hash (Redis or SQLite) | Instant repeat questions |
| **L2 Semantic** | Dense vector search (Qdrant) plus cross-encoder verification | Paraphrased questions |
| **L3 Generation** | Gemini 2.5 / OpenAI / custom provider | Genuine misses |

It ships with two extras:

- A drop-in **OpenAI-compatible proxy endpoint** (`/v1/chat/completions`).
- A **historical bulk ingestion engine** that removes cold-start costs.

---

## Architecture

```mermaid
flowchart TD
    Client([Client Application / Chatbot]) -->|POST /v1/chat/completions| SecGuard[Security Guard & Sliding-Window Rate Limiter]
    SecGuard --> Coalesce[Request Coalescer]

    subgraph MultiTierCache ["Multi-Tier Semantic Cache Engine"]
        direction TB
        L1["L1 Exact Match (Redis / SQLite Key-Hash) ~1ms"]
        L2["L2 Semantic Match (Qdrant Dense Cosine Vectors) ~15ms"]
        Guard["Verification Guard (Antonym Lexical Check + Cross-Encoder)"]
        L3["L3 LLM Generation (Gemini 2.5 / OpenAI / Custom Provider)"]
    end

    Coalesce --> L1
    L1 -->|L1 Hit: <5ms| Return([Instant Cached Response])
    L1 -->|Miss| L2
    L2 -->|Similarity >= Threshold| Guard
    Guard -->|Approved: Score >= 0.6| Return
    Guard -->|Rejected / Antonym Conflict| L3
    L2 -->|Miss / Under Threshold| L3
    L3 -->|Generated Response| Store[Async Populate L1 + L2 + Prometheus Metrics]
    Store --> Return
```

---

## Core Capabilities

<table>
<tr>
<td width="50%" valign="top">

### <img src="https://cdn.simpleicons.org/redis/DC382D" width="20" /> Tiered Storage

- **L1 Exact Layer:** O(1) hash lookup on SHA-256 normalized queries via Redis or embedded SQLite. No neural model overhead.
- **L2 Semantic Layer:** Approximate nearest neighbor search in Qdrant using `all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **L3 Model Layer:** LLM invocation with exponential backoff, timeout protection, and automated fallback (Gemini 2.5 Flash, OpenAI, or local vLLM).

</td>
<td width="50%" valign="top">

### <img src="https://cdn.simpleicons.org/openai/412991" width="20" /> OpenAI-Compatible Proxy

- Acts as a reverse-proxy gateway at `/v1/chat/completions`.
- Any bot built on the OpenAI SDK, LangChain, LlamaIndex, or the Next.js AI SDK integrates with a single `base_url` change.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### <img src="https://cdn.simpleicons.org/files/4A90D9" width="20" /> Historical Bulk Ingestion

- Upload past customer conversations (CSV or JSON).
- Questions and answers are bulk-encoded straight into L1 and L2 with **zero LLM API calls**, so the cache starts warm.

</td>
<td width="50%" valign="top">

### <img src="https://cdn.simpleicons.org/speedtest/141526" width="20" /> Batching and Coalescing

- Concurrent embedding queries inside a 10ms window are grouped into a single tensor batch.
- Identical in-flight misses share one LLM execution task.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### <img src="https://cdn.simpleicons.org/hackthebox/9FEF00" width="20" /> Hard Negative Protection

- A lexical guard blocks false hits on opposite concepts (`install` vs `uninstall`, `enable` vs `disable`).
- A cross-encoder (`ms-marco-MiniLM-L-6-v2`) applies deep cross-attention before any cached answer is returned.

</td>
<td width="50%" valign="top">

### <img src="https://cdn.simpleicons.org/letsencrypt/003A70" width="20" /> Security and Multi-Tenancy

- Namespace isolation (`ns`), model tagging, and document versioning (`doc_ver`).
- Constant-time SHA-256 API key authentication.
- Sliding-window rate limiting per API key or client IP.
- Structured JSON log sanitization.

</td>
</tr>
</table>

### <img src="https://cdn.simpleicons.org/prometheus/E6522C" width="20" /> Enterprise Observability

- Prometheus `/metrics` endpoint: request counters, latency histograms by tier, tokens saved, hit distribution.
- Health probe at `/api/health`.

---

## Drop-In Integration

Point any existing client at the proxy. Setup takes under two minutes. Pick your stack:

<details open>
<summary><img src="https://cdn.simpleicons.org/python/3776AB" width="16" /> <b>Python (OpenAI SDK)</b></summary>

```python
from openai import OpenAI

# Point the client to the Semantic Cache Proxy
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Or your CACHE_API_KEY if security is enabled
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "How do I reset my password?"}]
)
print(response.choices[0].message.content)
```

</details>

<details>
<summary><img src="https://cdn.simpleicons.org/langchain/1C3C3C" width="16" /> <b>LangChain / LlamaIndex</b></summary>

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="gpt-4o"
)
response = llm.invoke("What is Retrieval-Augmented Generation?")
print(response.content)
```

</details>

<details>
<summary><img src="https://cdn.simpleicons.org/nodedotjs/5FA04E" width="16" /> <b>Node.js / TypeScript</b></summary>

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "not-needed",
});

const completion = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "How to install Python on Windows?" }],
});
console.log(completion.choices[0].message.content);
```

</details>

<details>
<summary><img src="https://cdn.simpleicons.org/curl/073551" width="16" /> <b>cURL</b></summary>

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "How do I reset my password?"}
    ]
  }'
```

</details>

> **Tip:** Inspect the `X-Cache` response header to see whether a request was served from L1, L2, or the LLM. See the [API Reference](#api-reference).

---

## Quickstart

### 1. Local Setup

<details open>
<summary><b>Linux / macOS</b></summary>

```bash
git clone https://github.com/example/semantic-cache.git
cd semantic-cache
pip install -r requirements.txt
cp .env.example .env
python app.py
```

</details>

<details>
<summary><b>Windows</b></summary>

```bash
git clone https://github.com/example/semantic-cache.git
cd semantic-cache
pip install -r requirements.txt
copy .env.example .env
python app.py
```

</details>

Then open [http://localhost:8000](http://localhost:8000) for the interactive studio, playground, and telemetry dashboard.

### 2. Docker Deployment

Deploy the full production stack (FastAPI + Redis 7 + Qdrant) with one command:

```bash
docker compose up -d --build
```

---

## Configuration Reference

Set values through environment variables or a `.env` file (`pydantic-settings`).

<details open>
<summary><b>Security and limits</b></summary>

| Variable | Type | Default | Description |
|:--|:--|:--|:--|
| `CACHE_API_KEY` | string | `None` | Optional API key required in the `x-api-key` or `Authorization` header |
| `RATE_LIMIT` | int | `120` | Maximum requests per 60s sliding window |
| `MAX_QUERY_LENGTH` | int | `4096` | Maximum query length in characters |

</details>

<details open>
<summary><b>Cache behavior</b></summary>

| Variable | Type | Default | Description |
|:--|:--|:--|:--|
| `MAX_ENTRIES` | int | `500` | LRU eviction threshold for SQLite / Qdrant |
| `TTL` | float | `86400` | Default time-to-live for cache entries, in seconds (24h) |
| `BASE_THRESHOLD_GENERAL` | float | `0.88` | Cosine similarity threshold for general conversational queries |
| `BASE_THRESHOLD_CODE` | float | `0.93` | Stricter cosine threshold for code and technical queries |
| `RERANK` | bool | `true` | Enable the cross-encoder verification guard |
| `RERANK_MIN` | float | `0.6` | Minimum cross-encoder confidence score |

</details>

<details open>
<summary><b>Infrastructure and providers</b></summary>

| Variable | Type | Default | Description |
|:--|:--|:--|:--|
| `REDIS_URL` | string | `None` | Optional Redis URL for the L1 cache (defaults to embedded SQLite) |
| `QDRANT_URL` | string | `None` | Optional Qdrant server URL (defaults to embedded local storage) |
| `GEMINI_API_KEY` | string | `None` | Google Gemini API key |
| `GEMINI_MODEL` | string | `gemini-2.5-flash` | Primary LLM generation model |
| `GEMINI_FALLBACK_MODEL` | string | `gemini-2.5-flash` | Fallback model if the primary degrades |

</details>

---

## Benchmark Results

> Measured with `scripts/benchmark.py` and `scripts/evaluate.py` across exact and semantic queries.

### 1. Latency and Hit Rate

| Metric | p50 | p95 |
|:--|--:|--:|
| **L1 exact hit** | 12.57 ms | 13.54 ms |
| **L2 semantic hit** | 28.40 ms | 34.10 ms |
| **L3 cold generation** | 1,420 ms | 1,840 ms |

- **Cache hit rate:** **82.5%**
- **Net latency reduction:** **over 98% faster** for cached questions.

### 2. Precision and Recall Across a Threshold Sweep

Evaluated on 101 labeled test pairs (positive paraphrases and adversarial hard negatives).

<details open>
<summary><b>View the full sweep table</b></summary>

| Cosine Threshold | Raw Precision | Raw Recall | Raw False Positives | Precision (+ Guard) | Recall (+ Guard) | False Positives (+ Guard) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **0.78** | 70.0% | 42.0% | 17.6% | **70.0%** | 14.0% | **5.9%** |
| **0.82** | 69.6% | 32.0% | 13.7% | **70.0%** | 14.0% | **5.9%** |
| **0.86** | 68.8% | 22.0% | 9.8% | **83.3%** | 10.0% | **2.0%** |
| **0.90** | 42.9% | 6.0% | 7.8% | **66.7%** | 4.0% | **2.0%** |
| **0.94** | 0.0% | 0.0% | 2.0% | **100.0%** | 0.0% | **0.0%** |

</details>

---

## API Reference

<details open>
<summary><code>POST</code> <b>/v1/chat/completions</b>: OpenAI-compatible proxy</summary>

- **Request body:** standard OpenAI payload (`messages`, `model`, `stream`).
- **Response headers:**

| Header | Values |
|:--|:--|
| `X-Cache` | `HIT (L1_exact)`, `HIT (L2_semantic)`, `MISS (L3_llm)` |
| `X-Cache-Level` | `L1_exact`, `L2_semantic`, `L3_llm` |
| `X-Response-Time-Ms` | Execution latency in milliseconds |

</details>

<details>
<summary><code>POST</code> <b>/api/import</b>: bulk historical ingestion</summary>

Uploads historical conversation datasets (CSV / JSON) directly into L1 and L2 without calling any LLM API.

- **Parameters:** `file` (multipart file), `ns` (namespace, default `default`).
- **Returns:** total parsed count, imported count, and elapsed processing time.

</details>

<details>
<summary><code>POST</code> <b>/api/ask</b>: direct cache query</summary>

Semantic cache query endpoint with full telemetry metadata.

</details>

<details>
<summary><code>POST</code> <b>/api/ask/stream</b>: streaming</summary>

Server-Sent Events (SSE) endpoint with chunked token delivery.

</details>

<details>
<summary><code>GET</code> <b>/api/health</b>: health probe</summary>

Reports the status of L1 (Redis / SQLite) and L2 (Qdrant).

</details>

<details>
<summary><code>GET</code> <b>/metrics</b>: Prometheus</summary>

Standard Prometheus exposition format.

</details>

---

## Verification and Code Quality

The codebase has a comprehensive automated test suite:

```bash
pytest -v        # run the test suite
ruff check .     # lint checks
mypy .           # static type checks
```

---

## Technical Design Decisions

<details>
<summary><b>1. Embedding model: <code>all-MiniLM-L6-v2</code></b></summary>

Produces 384-dimensional dense embeddings with high-throughput CPU inference, so no dedicated GPU instance is required.

</details>

<details>
<summary><b>2. Antonym sensitivity resolution</b></summary>

Antonyms often have high cosine similarity in dense vector spaces. To prevent invalid hits (for example `install` vs `uninstall`), the system applies two-stage lexical antonym filtering followed by cross-encoder verification.

</details>

<details>
<summary><b>3. Dual storage flexibility</b></summary>

Runs in **embedded mode** (SQLite + local Qdrant WAL) for zero-dependency local setups, or in **distributed mode** (Redis + remote Qdrant cluster) for high-concurrency cloud deployments.

</details>

---

<div align="center">

Released under the [MIT License](LICENSE).

</div>
