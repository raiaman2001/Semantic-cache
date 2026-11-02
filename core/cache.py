import asyncio
import hashlib
import json
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient, models as qm

from config import Settings
from core.embedding import EmbeddingService
from core.llm import LLMService
from core.logging_config import setup_logger
from core.metrics import (
    CACHE_COALESCED_TOTAL,
    CACHE_ENTRIES_GAUGE,
    CACHE_LATENCY_SECONDS,
    CACHE_REQUESTS_TOTAL,
    CACHE_TOKENS_SAVED_TOTAL,
)
from core.rerank import RerankService, normalize_text
from db.database import Database

logger = setup_logger("semantic_cache.core")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def classify_category(query: str) -> str:
    pattern = r"\b(code|python|function|error|bug|api|sql|class|java|script)\b"
    return "code" if re.search(pattern, query.lower()) else "general"


def compute_cache_key(ns: str, model: str, doc_ver: str, query: str) -> str:
    norm_q = normalize_text(query)
    raw = f"{ns}|{model}|{doc_ver}|{norm_q}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SemanticCacheService:
    """Production LLM multi-tier cache orchestrator: L1 (Redis/SQLite) -> L2 (Qdrant) -> L3 (LLM)."""

    def __init__(
        self,
        settings: Settings,
        db: Database,
        embedding_service: EmbeddingService,
        rerank_service: RerankService,
        llm_service: LLMService,
        qdrant_client: Optional[QdrantClient] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.embedding = embedding_service
        self.rerank = rerank_service
        self.llm = llm_service
        self.redis = redis_client
        self._inflight: Dict[str, asyncio.Task[str]] = {}

        # Initialize Qdrant Client if not injected
        if qdrant_client is not None:
            self.qd: Optional[QdrantClient] = qdrant_client
        else:
            try:
                if self.settings.qdrant_url:
                    self.qd = QdrantClient(url=self.settings.qdrant_url)
                else:
                    self.qd = QdrantClient(path=self.settings.qdrant_path)
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant client: {e}. L2 semantic cache will be disabled.")
                self.qd = None

        self._init_qdrant_collection()

    def _init_qdrant_collection(self) -> None:
        if not self.qd:
            return
        try:
            if not self.qd.collection_exists(self.settings.qdrant_collection):
                self.qd.create_collection(
                    self.settings.qdrant_collection,
                    vectors_config=qm.VectorParams(
                        size=self.settings.embedding_dim,
                        distance=qm.Distance.COSINE,
                    ),
                )
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")
            self.qd = None

    def get_thresholds(self) -> Dict[str, float]:
        defaults = {
            "code": self.settings.base_threshold_code,
            "general": self.settings.base_threshold_general,
        }
        return self.db.get_thresholds(defaults)

    def save_thresholds(self, thr: Dict[str, float]) -> None:
        self.db.save_thresholds(thr)

    async def lookup(
        self,
        query: str,
        ns: str,
        model: str,
        doc_ver: str,
        cat: str,
        thr: Dict[str, float],
        key: str,
    ) -> Tuple[Optional[Tuple[str, str, float, int]], Optional[np.ndarray]]:
        """
        Multi-tier lookup.
        Returns: ((level, answer, score, entry_id) or None, embedding_vector or None)
        """
        now = time.time()

        # Tier 1: Exact match lookup (Redis or SQLite)
        if self.redis:
            try:
                v = await self.redis.get("sc:" + key)
                if v:
                    d = json.loads(v)
                    self.db.touch_entry(d["id"])
                    return ("L1_exact", d["answer"], 1.0, d["id"]), None
            except Exception as e:
                logger.warning(f"Redis L1 read error: {e}. Falling back to SQLite.")

        row = self.db.get_entry_by_key(key)
        if row:
            eid, ans, ts, ttl = row
            if now - ts < ttl:
                self.db.touch_entry(eid)
                return ("L1_exact", ans, 1.0, eid), None

        # Tier 2: Vector embedding & semantic similarity search (Qdrant)
        vec = await self.embedding.embed(query)
        if not self.qd:
            return None, vec

        try:
            flt = qm.Filter(
                must=[
                    qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
                    for k, v in (("ns", ns), ("model", model), ("doc_ver", doc_ver))
                ]
            )
            pts = await asyncio.to_thread(
                lambda: self.qd.query_points(
                    self.settings.qdrant_collection,
                    query=vec.tolist(),
                    query_filter=flt,
                    limit=3,
                ).points
            )

            for p in pts:
                if p.score < thr[cat]:
                    break
                entry = self.db.get_entry_by_id(int(p.id))
                if not entry:
                    continue
                eid, stored_query, stored_ans, ts, ttl = entry
                if now - ts >= ttl:
                    continue

                # Borderline check if within +0.05 threshold zone
                if p.score < thr[cat] + 0.05:
                    is_valid = await self.rerank.confirm(query, stored_query)
                    if not is_valid:
                        continue

                self.db.touch_entry(eid)
                return ("L2_semantic", stored_ans, float(p.score), eid), vec
        except Exception as e:
            logger.error(f"L2 Qdrant lookup failed: {e}. Gracefully falling back to L3.")

        return None, vec

    async def store(
        self,
        query: str,
        answer: str,
        ns: str,
        model: str,
        doc_ver: str,
        tag: str,
        ttl: float,
        key: str,
        vec: Optional[np.ndarray],
    ) -> int:
        """Stores entry across L1 (Redis/SQLite) and L2 (Qdrant) with LRU eviction enforcement."""
        if vec is None:
            vec = await self.embedding.embed(query)

        cat = classify_category(query)
        new_id, old_ids = self.db.insert_entry(
            ns=ns,
            model=model,
            doc_ver=doc_ver,
            tag=tag,
            cat=cat,
            key=key,
            query=query,
            answer=answer,
            ttl=ttl,
        )

        # Sync Qdrant deletions for overwritten old IDs
        if self.qd and old_ids:
            try:
                await asyncio.to_thread(
                    self.qd.delete,
                    self.settings.qdrant_collection,
                    points_selector=qm.PointIdsList(points=old_ids),
                )
            except Exception as e:
                logger.error(f"Failed to delete overwritten Qdrant points {old_ids}: {e}")

        # Insert new Qdrant vector point
        if self.qd:
            try:
                await asyncio.to_thread(
                    self.qd.upsert,
                    self.settings.qdrant_collection,
                    points=[
                        qm.PointStruct(
                            id=new_id,
                            vector=vec.tolist(),
                            payload={"ns": ns, "model": model, "doc_ver": doc_ver},
                        )
                    ],
                )
            except Exception as e:
                logger.error(f"Failed to upsert Qdrant vector point {new_id}: {e}")

        # Store in Redis
        if self.redis:
            try:
                await self.redis.setex(
                    "sc:" + key,
                    int(ttl),
                    json.dumps({"id": new_id, "answer": answer}),
                )
            except Exception as e:
                logger.warning(f"Failed to write entry to Redis: {e}")

        # LRU Eviction check
        excess_ids = self.db.get_lru_excess_ids(self.settings.max_entries)
        if excess_ids:
            await self.delete_entries(excess_ids)

        CACHE_ENTRIES_GAUGE.set(self.db.count_entries())
        return new_id

    async def delete_entries(self, ids: List[int]) -> None:
        if not ids:
            return
        keys = self.db.delete_by_ids(ids)
        if self.qd:
            try:
                await asyncio.to_thread(
                    self.qd.delete,
                    self.settings.qdrant_collection,
                    points_selector=qm.PointIdsList(points=ids),
                )
            except Exception as e:
                logger.error(f"Failed to delete Qdrant points {ids}: {e}")

        if self.redis and keys:
            try:
                redis_keys = ["sc:" + k for k in keys]
                await self.redis.delete(*redis_keys)
            except Exception as e:
                logger.warning(f"Failed to delete keys from Redis: {e}")

        CACHE_ENTRIES_GAUGE.set(self.db.count_entries())

    async def invalidate_tag(self, tag: str, ns: str = "") -> int:
        ids = self.db.delete_by_tag_and_ns(tag, ns)
        await self.delete_entries(ids)
        return len(ids)

    async def clear_cache(self) -> None:
        self.db.clear_all()
        if self.qd:
            try:
                await asyncio.to_thread(self.qd.delete_collection, self.settings.qdrant_collection)
                self._init_qdrant_collection()
            except Exception as e:
                logger.error(f"Failed to reset Qdrant collection: {e}")

        if self.redis:
            try:
                async for k in self.redis.scan_iter("sc:*"):
                    await self.redis.delete(k)
            except Exception as e:
                logger.warning(f"Failed to clear Redis keys: {e}")

        CACHE_ENTRIES_GAUGE.set(0)

    async def coalesce_generate(self, key: str, query: str, model: str) -> Tuple[str, bool]:
        """Request coalescing: identical in-flight cache misses share a single LLM execution."""
        shared = key in self._inflight
        if not shared:
            self._inflight[key] = asyncio.create_task(self.llm.complete(query, model))
        else:
            CACHE_COALESCED_TOTAL.inc()

        task = self._inflight[key]
        try:
            ans = await task
            return ans, shared
        finally:
            if not shared:
                self._inflight.pop(key, None)

    def log_and_record_metrics(
        self,
        query: str,
        ns: str,
        level: str,
        cat: str,
        ans: str,
        score: float,
        start_time: float,
        entry_id: Optional[int],
        thr: Dict[str, float],
    ) -> Dict[str, Any]:
        ms = (time.time() - start_time) * 1000
        sec = ms / 1000.0
        toks = estimate_tokens(ans)

        log_id = self.db.log_request(
            ns=ns,
            query=query,
            level=level,
            score=float(score),
            ms=ms,
            tokens=toks,
            entry_id=entry_id,
            cat=cat,
        )

        CACHE_REQUESTS_TOTAL.labels(level=level, category=cat, namespace=ns).inc()
        CACHE_LATENCY_SECONDS.labels(level=level).observe(sec)
        if level in ("L1_exact", "L2_semantic"):
            CACHE_TOKENS_SAVED_TOTAL.labels(level=level, namespace=ns).inc(toks)

        return {
            "answer": ans,
            "level": level,
            "score": round(float(score), 4),
            "ms": round(ms, 1),
            "log_id": log_id,
            "threshold": thr[cat],
            "category": cat,
        }

