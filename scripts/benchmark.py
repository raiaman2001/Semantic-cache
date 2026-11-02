"""
Benchmark script for Semantic Cache:
Measures hit rate, p50, and p95 latency for Cache (L1/L2) vs LLM generation (L3).
Outputs real benchmark numbers.
"""

import asyncio
import os
import sys
import time
import numpy as np

# Ensure root directory is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Settings
from core.cache import SemanticCacheService, compute_cache_key
from core.embedding import EmbeddingService
from core.llm import LLMService
from core.rerank import RerankService
from db.database import Database

BENCHMARK_QUERIES = [
    # Paraphrased queries designed to trigger hits
    ("How do I reset my password?", "How do I reset my password?"),  # L1 exact
    ("How to reset password on web", "How do I reset my password?"),  # L2 semantic
    ("What is RAG in AI?", "What is RAG in AI?"),  # L1 exact
    ("Explain retrieval augmented generation", "What is RAG in AI?"),  # L2 semantic
    ("How to install Python?", "How to install Python?"),  # L1 exact
    ("Steps to install Python on Windows", "How to install Python?"),  # L2 semantic
    ("What is Docker container?", "What is Docker container?"),  # L1 exact
    ("How do Docker containers work?", "What is Docker container?"),  # L2 semantic
    ("What is an API gateway?", "What is an API gateway?"),  # L1 exact
    ("How does an API gateway route traffic?", "What is an API gateway?"),  # L2 semantic
]

COLD_QUERIES = [
    "What is the airspeed velocity of an unladen swallow?",
    "Explain the history of the Byzantine empire in 5 sentences.",
    "How does nuclear fusion generate energy in stars?",
    "What is Dijkstra's shortest path algorithm?",
    "Explain how a carburetors works in internal combustion engines.",
]


async def run_benchmark():
    db_file = "benchmark_test.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    settings = Settings(
        sqlite_db_path=db_file,
        qdrant_path="bench_qdrant",
        rate_limit=10000,
        max_entries=1000,
    )
    db = Database(db_file)
    embedding_svc = EmbeddingService()
    rerank_svc = RerankService(enabled=True)
    llm_svc = LLMService(api_key=None)  # Uses mock LLM with ~200ms latency

    cache = SemanticCacheService(
        settings=settings,
        db=db,
        embedding_service=embedding_svc,
        rerank_service=rerank_svc,
        llm_service=llm_svc,
        redis_client=None,
    )

    print("--- 1. Warming Cache with Base Entries ---")
    for q, base in BENCHMARK_QUERIES[:5]:
        key = compute_cache_key("default", "test-llm", "v1", base)
        vec = await cache.embedding.embed(base)
        await cache.store(base, f"Answer for {base}", "default", "test-llm", "v1", "bench", 3600, key, vec)

    print("\n--- 2. Benchmarking Cache Hits vs Misses ---")
    l1_latencies = []
    l2_latencies = []
    l3_latencies = []

    # Run 40 cache probe queries
    trials = 40
    hits = 0
    total = 0

    for i in range(trials):
        pair = BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)]
        query = pair[0]
        key = compute_cache_key("default", "test-llm", "v1", query)
        thr = cache.get_thresholds()

        t0 = time.perf_counter()
        hit, vec = await cache.lookup(query, "default", "test-llm", "v1", "general", thr, key)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        total += 1
        if hit:
            hits += 1
            lvl = hit[0]
            if lvl == "L1_exact":
                l1_latencies.append(elapsed_ms)
            else:
                l2_latencies.append(elapsed_ms)
        else:
            # Simulate L3 LLM call
            t_llm = time.perf_counter()
            ans, _ = await cache.coalesce_generate(key, query, "test-llm")
            await cache.store(query, ans, "default", "test-llm", "v1", "bench", 3600, key, vec)
            elapsed_l3 = (time.perf_counter() - t_llm) * 1000.0
            l3_latencies.append(elapsed_l3)

    hit_rate = (hits / total) * 100.0
    print(f"\nResults ({trials} requests):")
    print(f"Total Requests: {total}")
    print(f"Cache Hit Rate: {hit_rate:.1f}%")

    if l1_latencies:
        print(f"L1 Exact Hit Latency:   p50={np.percentile(l1_latencies, 50):.2f}ms, p95={np.percentile(l1_latencies, 95):.2f}ms (n={len(l1_latencies)})")
    if l2_latencies:
        print(f"L2 Semantic Latency:    p50={np.percentile(l2_latencies, 50):.2f}ms, p95={np.percentile(l2_latencies, 95):.2f}ms (n={len(l2_latencies)})")
    if l3_latencies:
        print(f"L3 Mock LLM Latency:    p50={np.percentile(l3_latencies, 50):.2f}ms, p95={np.percentile(l3_latencies, 95):.2f}ms (n={len(l3_latencies)})")

    # Clean up benchmark db
    db.close()
    if os.path.exists(db_file):
        os.remove(db_file)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
