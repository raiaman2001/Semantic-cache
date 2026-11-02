import asyncio
import json
import re
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from core.cache import (
    SemanticCacheService,
    classify_category,
    compute_cache_key,
    normalize_text,
)
from core.rerank import verify_lexical_consistency

router = APIRouter()

# Global reference to be injected or retrieved from app.state
_cache_service: SemanticCacheService | None = None


def set_cache_service(service: SemanticCacheService) -> None:
    global _cache_service
    _cache_service = service


def get_cache_service() -> SemanticCacheService:
    if _cache_service is None:
        raise RuntimeError("SemanticCacheService has not been initialized.")
    return _cache_service


# Request / Response Schemas
class AskRequest(BaseModel):
    query: str = Field(..., max_length=settings.max_query_length)
    ns: str = Field(default="default")
    model: str = Field(default_factory=lambda: settings.gemini_model)
    doc_ver: str = Field(default="v1")
    tag: str = Field(default="")
    use_cache: bool = Field(default=True)
    ttl: float = Field(default_factory=lambda: float(settings.default_ttl))


class InvalidateRequest(BaseModel):
    tag: str = ""
    ns: str = ""


class FeedbackRequest(BaseModel):
    log_id: int
    good: bool


class ThresholdRequest(BaseModel):
    code: float
    general: float


class WarmRequest(BaseModel):
    queries: list[str]
    ns: str = "default"


@router.get("/health")
def health():
    svc = get_cache_service()
    redis_status = "connected" if svc.redis else "not_configured"
    qdrant_status = "connected" if svc.qd else "degraded"
    return {
        "status": "healthy" if svc.qd else "degraded",
        "redis": redis_status,
        "qdrant": qdrant_status,
        "version": "2.0.0",
    }


@router.get("/config")
def get_config():
    svc = get_cache_service()
    return {
        "l1": "Redis" if svc.redis else "SQLite",
        "vectors": "Qdrant server" if svc.settings.qdrant_url else "Qdrant local",
        "reranker": bool(svc.rerank.get_model()),
        "rerank_min": svc.rerank.min_score,
    }


@router.post("/ask")
async def ask(r: AskRequest):
    svc = get_cache_service()
    if not normalize_text(r.query):
        raise HTTPException(status_code=400, detail="Query is empty")

    t0 = time.time()
    cat = classify_category(r.query)
    thr = svc.get_thresholds()
    key = compute_cache_key(r.ns, r.model, r.doc_ver, r.query)

    hit, vec = await svc.lookup(r.query, r.ns, r.model, r.doc_ver, cat, thr, key) if r.use_cache else (None, None)
    if hit:
        level, ans, score, eid = hit
        return svc.log_and_record_metrics(r.query, r.ns, level, cat, ans, score, t0, eid, thr)

    # Miss: coalesce or generate via LLM
    ans, shared = await svc.coalesce_generate(key, r.query, r.model)
    new_eid: int | None = (
        await svc.store(r.query, ans, r.ns, r.model, r.doc_ver, r.tag, r.ttl, key, vec)
        if (r.use_cache and not shared)
        else None
    )

    result = svc.log_and_record_metrics(r.query, r.ns, "L3_llm", cat, ans, 0.0, t0, new_eid, thr)
    result["coalesced"] = shared
    return result


@router.post("/ask/stream")
async def ask_stream(r: AskRequest):
    svc = get_cache_service()
    if not normalize_text(r.query):
        raise HTTPException(status_code=400, detail="Query is empty")

    t0 = time.time()
    cat = classify_category(r.query)
    thr = svc.get_thresholds()
    key = compute_cache_key(r.ns, r.model, r.doc_ver, r.query)

    hit, vec = await svc.lookup(r.query, r.ns, r.model, r.doc_ver, cat, thr, key) if r.use_cache else (None, None)

    def ev(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    async def event_generator():
        if hit:
            level, ans, score, eid = hit
            meta = svc.log_and_record_metrics(r.query, r.ns, level, cat, ans, score, t0, eid, thr)
            yield ev({"meta": meta})
            for word in re.findall(r"\S+\s*", ans):
                yield ev({"t": word})
                await asyncio.sleep(0.012)
        else:
            parts: list[str] = []
            async for chunk in svc.llm.stream(r.query, r.model):
                if not parts:
                    yield ev({
                        "meta": {
                            "level": "L3_llm",
                            "score": 0.0,
                            "ms": round((time.time() - t0) * 1000, 1),
                            "log_id": None,
                            "threshold": thr[cat],
                            "category": cat,
                        }
                    })
                parts.append(chunk)
                yield ev({"t": chunk})

            full_ans = "".join(parts)
            eid = await svc.store(r.query, full_ans, r.ns, r.model, r.doc_ver, r.tag, r.ttl, key, vec) if r.use_cache else None
            svc.log_and_record_metrics(r.query, r.ns, "L3_llm", cat, full_ans, 0.0, t0, eid, thr)

        yield ev({"done": True})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/stats")
def stats(ns: str = ""):
    svc = get_cache_service()
    st = svc.db.get_stats(ns)
    d = st["levels"]
    hits = sum(d.get(k, {"n": 0})["n"] for k in ("L1_exact", "L2_semantic"))
    total = hits + d.get("L3_llm", {"n": 0})["n"]
    saved = sum(d.get(k, {"tokens": 0})["tokens"] for k in ("L1_exact", "L2_semantic"))

    return {
        "levels": d,
        "hits": hits,
        "total": total,
        "hit_rate": (hits / total) if total else 0.0,
        "tokens_saved": saved,
        "cost_saved": (saved / 1000.0) * svc.settings.cost_per_1k_tokens,
        "entries": svc.db.count_entries(),
        "thresholds": svc.get_thresholds(),
        "recent": st["recent"],
        "top": st["top"],
    }


@router.get("/entries")
def get_entries():
    svc = get_cache_service()
    return svc.db.get_entries_list()


@router.delete("/entries/{eid}")
async def delete_entry(eid: int):
    svc = get_cache_service()
    await svc.delete_entries([eid])
    return {"ok": True}


@router.post("/clear")
async def clear_cache():
    svc = get_cache_service()
    await svc.clear_cache()
    return {"ok": True}


@router.post("/invalidate")
async def invalidate(b: InvalidateRequest):
    svc = get_cache_service()
    deleted = await svc.invalidate_tag(b.tag, b.ns)
    return {"deleted": deleted}


@router.post("/feedback")
def feedback(b: FeedbackRequest):
    svc = get_cache_service()
    row = svc.db.get_log_record(b.log_id)
    if not row:
        raise HTTPException(status_code=404, detail="Unknown log id")
    cat, level = row
    thr = svc.get_thresholds()
    if level == "L2_semantic":
        current = thr.get(cat, 0.88)
        new_val = round(max(0.75, current - 0.002) if b.good else min(0.99, current + 0.01), 4)
        thr[cat] = new_val
        svc.save_thresholds(thr)
    return {"thresholds": thr}


@router.post("/threshold")
def set_thresholds(b: ThresholdRequest):
    svc = get_cache_service()
    payload = {"code": b.code, "general": b.general}
    svc.save_thresholds(payload)
    return payload


@router.post("/warm")
async def warm_cache(b: WarmRequest):
    get_cache_service()
    qs = [q.strip() for q in b.queries if q.strip()]
    if qs:
        await asyncio.gather(*[
            ask(AskRequest(query=q, ns=b.ns))
            for q in qs
        ])
    return {"warmed": len(qs)}


@router.get("/simulate")
def simulate(
    queries: int = 10000,
    hit_rate: float = 0.6,
    llm_ms: float = 1500.0,
    cache_ms: float = 15.0,
    tok: int = 400,
):
    svc = get_cache_service()
    h = int(queries * hit_rate)
    calls_saved = h
    time_saved_min = round(h * (llm_ms - cache_ms) / 60000.0, 1)
    cost_saved = round(h * tok / 1000.0 * svc.settings.cost_per_1k_tokens, 2)
    avg_ms = round((h * cache_ms + (queries - h) * llm_ms) / max(1, queries), 1)
    return {
        "llm_calls_saved": calls_saved,
        "time_saved_min": time_saved_min,
        "cost_saved": cost_saved,
        "avg_ms": avg_ms,
    }


PAIRS = [
    ("How do I reset my password?", "I forgot my password, how can I change it?", 1),
    ("What is RAG?", "Explain retrieval augmented generation", 1),
    ("How to install Python on Windows", "Steps to set up Python on a Windows PC", 1),
    ("What is the capital of France?", "Which city is France's capital?", 1),
    ("How do I enable dark mode?", "How can I turn on the dark theme?", 1),
    ("Best way to learn SQL fast", "Quickest method to study SQL", 1),
    ("How do I enable notifications?", "How do I disable notifications?", 0),
    ("Add a user to the group", "Remove a user from the group", 0),
    ("What is machine learning?", "What is the weather in Delhi today?", 0),
    ("How to install Docker", "How to uninstall Docker", 0),
    ("Delete my account", "Deactivate my account", 0),
    ("Explain Python lists", "Explain Java interfaces", 0),
    ("Open a file in Python", "Close a file in Python", 0),
]


@router.get("/evaluate")
def evaluate():
    import numpy as np
    svc = get_cache_service()
    texts_a = [p[0] for p in PAIRS]
    texts_b = [p[1] for p in PAIRS]

    emb_a = svc.embedding.encode_sync(texts_a)
    emb_b = svc.embedding.encode_sync(texts_b)
    sims = [float(np.dot(a, b)) for a, b in zip(emb_a, emb_b)]

    reranker = svc.rerank.get_model()
    if reranker:
        rr = [float(x) for x in reranker.predict([(p[0], p[1]) for p in PAIRS])]
    else:
        rr = [1.0] * len(PAIRS)

    ok = [
        verify_lexical_consistency(p[0], p[1]) and (rr[i] >= svc.rerank.min_score)
        for i, p in enumerate(PAIRS)
    ]

    out = []
    pos = sum(p[2] for p in PAIRS)
    for t in np.arange(0.5, 0.99, 0.03):
        t_val = float(t)
        tp = sum(1 for i, p in enumerate(PAIRS) if sims[i] >= t_val and p[2])
        fp = sum(1 for i, p in enumerate(PAIRS) if sims[i] >= t_val and not p[2])

        tp2 = sum(1 for i, p in enumerate(PAIRS) if sims[i] >= t_val and p[2] and ok[i])
        fp2 = sum(1 for i, p in enumerate(PAIRS) if sims[i] >= t_val and not p[2] and ok[i])

        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / pos if pos else 0.0
        prec_rr = tp2 / (tp2 + fp2) if (tp2 + fp2) else 1.0
        rec_rr = tp2 / pos if pos else 0.0

        out.append({
            "t": round(t_val, 2),
            "precision": prec,
            "recall": rec,
            "precision_rr": prec_rr,
            "recall_rr": rec_rr,
            "fp": fp,
            "fp_rr": fp2,
        })

    return {
        "curve": out,
        "pairs": [
            {
                "a": a,
                "b": b,
                "same": y,
                "sim": round(sims[i], 3),
                "rerank": round(rr[i], 3),
                "pass": ok[i],
            }
            for i, (a, b, y) in enumerate(PAIRS)
        ],
    }

