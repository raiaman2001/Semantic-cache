from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

CACHE_REQUESTS_TOTAL = Counter(
    "semantic_cache_requests_total",
    "Total requests processed by the semantic cache",
    ["level", "category", "namespace"],
)

CACHE_LATENCY_SECONDS = Histogram(
    "semantic_cache_latency_seconds",
    "End-to-end request latency in seconds",
    ["level"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

CACHE_TOKENS_SAVED_TOTAL = Counter(
    "semantic_cache_tokens_saved_total",
    "Total tokens saved by cache hits (L1 + L2)",
    ["level", "namespace"],
)

CACHE_ENTRIES_GAUGE = Gauge(
    "semantic_cache_entries_current",
    "Current number of entries stored in the semantic cache",
)

CACHE_COALESCED_TOTAL = Counter(
    "semantic_cache_coalesced_requests_total",
    "Total requests coalesced to an in-flight LLM request",
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST

