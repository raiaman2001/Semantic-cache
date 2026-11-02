"""Semantic Cache v2: L1 exact (Redis or SQLite) -> L2 semantic (Qdrant) -> L3 LLM.
Async, batched embeddings, request coalescing, SSE streaming, cross-encoder reranker.
Run: uvicorn app:app --reload   (optional: REDIS_URL, QDRANT_URL, GEMINI_API_KEY)"""
import os, re, time, json, sqlite3, hashlib, threading, asyncio
from collections import defaultdict, deque
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient, models as qm

API_KEY = os.getenv("CACHE_API_KEY")
RATE = int(os.getenv("RATE_LIMIT", 120)); MAX_ENTRIES = int(os.getenv("MAX_ENTRIES", 500)); DEFAULT_TTL = int(os.getenv("TTL", 86400))
RR_MIN = float(os.getenv("RERANK_MIN", 0.6)); COL = "semantic_cache"; DIM = 384; COST_PER_1K_TOK = 0.0005
BASE_THR = {"code": 0.93, "general": 0.88}
OPPOSITES = [("enable", "disable"), ("add", "remove"), ("open", "close"), ("start", "stop"),
             ("install", "uninstall"), ("allow", "block"), ("increase", "decrease"), ("encrypt", "decrypt")]

model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = None
if os.getenv("RERANK", "1") == "1":
    try: reranker = CrossEncoder("cross-encoder/stsb-TinyBERT-L-4")
    except Exception as e: print("Reranker disabled:", e)
qd = QdrantClient(url=os.environ["QDRANT_URL"]) if os.getenv("QDRANT_URL") else QdrantClient(path="qdrant_data")
def init_col():
    if not qd.collection_exists(COL):
        qd.create_collection(COL, vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE))
init_col()
redis = None
if os.getenv("REDIS_URL"):
    import redis.asyncio as aioredis
    redis = aioredis.from_url(os.environ["REDIS_URL"], decode_responses=True)

db = sqlite3.connect("cache.db", check_same_thread=False); lock = threading.RLock()
db.executescript("""
CREATE TABLE IF NOT EXISTS entries(id INTEGER PRIMARY KEY, ns TEXT, model TEXT, doc_ver TEXT, tag TEXT,
 cat TEXT, key TEXT, query TEXT, answer TEXT, emb BLOB, ts REAL, ttl REAL, hits INT DEFAULT 0, last REAL);
CREATE INDEX IF NOT EXISTS ix_key ON entries(key);
CREATE TABLE IF NOT EXISTS log(id INTEGER PRIMARY KEY, ts REAL, ns TEXT, query TEXT, level TEXT, score REAL,
 ms REAL, tokens INT, entry_id INT, cat TEXT);
CREATE TABLE IF NOT EXISTS settings(k TEXT PRIMARY KEY, v TEXT);""")

def get_thr():
    r = db.execute("SELECT v FROM settings WHERE k='thr'").fetchone()
    return json.loads(r[0]) if r else dict(BASE_THR)
def save_thr(t):
    with lock: db.execute("REPLACE INTO settings VALUES('thr',?)", (json.dumps(t),)); db.commit()
def norm(q): return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", q.lower())).strip()
def category(q): return "code" if re.search(r"\b(code|python|function|error|bug|api|sql|class|java|script)\b", q.lower()) else "general"
def tokens(s): return max(1, len(s) // 4)
def enc(texts):
    v = model.encode(texts).astype("float32"); return v / np.linalg.norm(v, axis=1, keepdims=True)

class Batcher:
    """Micro-batching: embedding requests arriving within 10 ms are encoded in one model call."""
    def __init__(s): s.q = None
    async def embed(s, text):
        if s.q is None: s.q = asyncio.Queue(); asyncio.create_task(s.run())
        f = asyncio.get_running_loop().create_future(); await s.q.put((text, f)); return await f
    async def run(s):
        while True:
            items = [await s.q.get()]; await asyncio.sleep(0.01)
            while not s.q.empty() and len(items) < 32: items.append(s.q.get_nowait())
            try:
                vs = await asyncio.to_thread(enc, [t for t, _ in items])
                for (_, f), v in zip(items, vs): f.set_result(v)
            except Exception as e:
                for _, f in items: f.set_exception(e)
batcher = Batcher()

def verify(a, b):
    A, B = set(norm(a).split()), set(norm(b).split())
    for x, y in OPPOSITES:
        if (x in A and y in B and x not in B) or (y in A and x in B and y not in B): return False
    return len(A & B) / max(1, len(A | B)) >= 0.2
async def confirm(a, b):
    """Borderline check: lexical guard, then cross-encoder reranker."""
    if not verify(a, b): return False
    if not reranker: return True
    return await asyncio.to_thread(lambda: float(reranker.predict([(a, b)])[0])) >= RR_MIN

def llm_stream(q, mdl):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        for w in f"[mock {mdl} answer] Set GEMINI_API_KEY for real answers. You asked: {q}".split(" "): time.sleep(0.05); yield w + " "
        return
    import google.generativeai as genai; genai.configure(api_key=key)
    for c in genai.GenerativeModel(mdl).generate_content(q, stream=True):
        if c.text: yield c.text
async def agen(it):
    while True:
        c = await asyncio.to_thread(next, it, None)
        if c is None: return
        yield c

inflight = {}
async def generate(r, key):
    """Request coalescing: identical concurrent misses share one LLM call."""
    shared = key in inflight
    if not shared: inflight[key] = asyncio.create_task(asyncio.to_thread(lambda: "".join(llm_stream(r.query, r.model))))
    t = inflight[key]
    try: return await t, shared
    finally: inflight.pop(key, None)

def qsearch(vec, r):
    flt = qm.Filter(must=[qm.FieldCondition(key=k, match=qm.MatchValue(value=v)) for k, v in (("ns", r.ns), ("model", r.model), ("doc_ver", r.doc_ver))])
    return qd.query_points(COL, query=vec.tolist(), query_filter=flt, limit=3).points

class Ask(BaseModel):
    query: str; ns: str = "default"; model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    doc_ver: str = "v1"; tag: str = ""; use_cache: bool = True; ttl: float = DEFAULT_TTL

def mkkey(r): return hashlib.sha256(f"{r.ns}|{r.model}|{r.doc_ver}|{norm(r.query)}".encode()).hexdigest()
def touch(i):
    with lock: db.execute("UPDATE entries SET hits=hits+1, last=? WHERE id=?", (time.time(), i)); db.commit()

async def lookup(r, cat, thr, key):
    """Returns ((level, answer, score, entry_id) | None, query_vector | None)."""
    if redis:
        v = await redis.get("sc:" + key)
        if v: d = json.loads(v); touch(d["id"]); return ("L1_exact", d["answer"], 1.0, d["id"]), None
    else:
        row = db.execute("SELECT id, answer, ts, ttl FROM entries WHERE key=?", (key,)).fetchone()
        if row and time.time() - row[2] < row[3]: touch(row[0]); return ("L1_exact", row[1], 1.0, row[0]), None
    vec = await batcher.embed(r.query)
    for p in await asyncio.to_thread(qsearch, vec, r):
        if p.score < thr[cat]: break
        e = db.execute("SELECT id, query, answer, ts, ttl FROM entries WHERE id=?", (p.id,)).fetchone()
        if not e or time.time() - e[3] >= e[4]: continue
        if p.score < thr[cat] + 0.05 and not await confirm(r.query, e[1]): continue
        touch(e[0]); return ("L2_semantic", e[2], p.score, e[0]), vec
    return None, vec

async def store(r, ans, key, vec):
    if vec is None: vec = await batcher.embed(r.query)
    with lock:
        old = [x[0] for x in db.execute("SELECT id FROM entries WHERE key=?", (key,))]
        cur = db.execute("INSERT INTO entries(ns,model,doc_ver,tag,cat,key,query,answer,emb,ts,ttl,last) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                         (r.ns, r.model, r.doc_ver, r.tag, category(r.query), key, r.query, ans, b"", time.time(), r.ttl, time.time()))
        eid = cur.lastrowid
        if old: db.execute(f"DELETE FROM entries WHERE id IN ({','.join('?' * len(old))})", old)
        db.commit()
    if old: await asyncio.to_thread(qd.delete, COL, points_selector=qm.PointIdsList(points=old))
    await asyncio.to_thread(qd.upsert, COL, points=[qm.PointStruct(id=eid, vector=vec.tolist(), payload={"ns": r.ns, "model": r.model, "doc_ver": r.doc_ver})])
    if redis: await redis.setex("sc:" + key, int(r.ttl), json.dumps({"id": eid, "answer": ans}))
    n = db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    if n > MAX_ENTRIES:   # LRU eviction
        await delete_ids([x[0] for x in db.execute("SELECT id FROM entries ORDER BY last ASC LIMIT ?", (n - MAX_ENTRIES,))])
    return eid

async def delete_ids(ids):
    if not ids: return
    q = ",".join("?" * len(ids))
    with lock:
        keys = [x[0] for x in db.execute(f"SELECT key FROM entries WHERE id IN ({q})", ids)]
        db.execute(f"DELETE FROM entries WHERE id IN ({q})", ids); db.commit()
    await asyncio.to_thread(qd.delete, COL, points_selector=qm.PointIdsList(points=ids))
    if redis and keys: await redis.delete(*["sc:" + k for k in keys])

def finish(r, cat, thr, t0, level, ans, score, eid):
    ms = (time.time() - t0) * 1000
    with lock:
        cur = db.execute("INSERT INTO log(ts,ns,query,level,score,ms,tokens,entry_id,cat) VALUES(?,?,?,?,?,?,?,?,?)",
                         (time.time(), r.ns, r.query, level, float(score), ms, tokens(ans), eid, cat)); db.commit()
    return {"answer": ans, "level": level, "score": round(float(score), 4), "ms": round(ms, 1), "log_id": cur.lastrowid,
            "threshold": thr[cat], "category": cat}

hist = defaultdict(deque)
def guard(request: Request, x_api_key: str = Header(None)):
    if not request.url.path.startswith("/api"): return
    if API_KEY and x_api_key != API_KEY: raise HTTPException(401, "Invalid API key")
    d, now = hist[request.client.host], time.time()
    while d and now - d[0] > 60: d.popleft()
    if len(d) >= RATE: raise HTTPException(429, "Rate limit exceeded")
    d.append(now)
app = FastAPI(title="Semantic Cache", dependencies=[Depends(guard)])

@app.get("/api/config")
def config(): return {"l1": "Redis" if redis else "SQLite", "vectors": "Qdrant server" if os.getenv("QDRANT_URL") else "Qdrant local",
                      "reranker": bool(reranker), "rerank_min": RR_MIN}

@app.post("/api/ask")
async def ask(r: Ask):
    t0, cat, thr, key = time.time(), category(r.query), get_thr(), mkkey(r)
    if not norm(r.query): raise HTTPException(400, "Query is empty")
    hit, vec = await lookup(r, cat, thr, key) if r.use_cache else (None, None)
    if hit: return finish(r, cat, thr, t0, *hit)
    ans, shared = await generate(r, key)
    eid = await store(r, ans, key, vec) if r.use_cache and not shared else None
    return {**finish(r, cat, thr, t0, "L3_llm", ans, 0, eid), "coalesced": shared}

@app.post("/api/ask/stream")
async def ask_stream(r: Ask):
    t0, cat, thr, key = time.time(), category(r.query), get_thr(), mkkey(r)
    if not norm(r.query): raise HTTPException(400, "Query is empty")
    hit, vec = await lookup(r, cat, thr, key) if r.use_cache else (None, None)
    ev = lambda o: f"data: {json.dumps(o)}\n\n"
    async def gen():
        if hit:
            yield ev({"meta": finish(r, cat, thr, t0, *hit)})
            for w in re.findall(r"\S+\s*", hit[1]): yield ev({"t": w}); await asyncio.sleep(0.012)
        else:
            parts = []
            async for c in agen(llm_stream(r.query, r.model)):
                if not parts: yield ev({"meta": {"level": "L3_llm", "score": 0, "ms": round((time.time() - t0) * 1000, 1), "log_id": None, "threshold": thr[cat], "category": cat}})
                parts.append(c); yield ev({"t": c})
            ans = "".join(parts)
            eid = await store(r, ans, key, vec) if r.use_cache else None
            finish(r, cat, thr, t0, "L3_llm", ans, 0, eid)
        yield ev({"done": True})
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/api/stats")
def stats(ns: str = ""):
    w, p = ("WHERE ns=?", (ns,)) if ns else ("", ())
    d = {r[0]: {"n": r[1], "avg_ms": r[2], "tokens": r[3] or 0} for r in db.execute(f"SELECT level, COUNT(*), AVG(ms), SUM(tokens) FROM log {w} GROUP BY level", p)}
    hits = sum(d.get(k, {"n": 0})["n"] for k in ("L1_exact", "L2_semantic")); tot = hits + d.get("L3_llm", {"n": 0})["n"]
    saved = sum(d.get(k, {"tokens": 0})["tokens"] for k in ("L1_exact", "L2_semantic"))
    recent = [dict(zip(("query", "level", "score", "ms"), r)) for r in db.execute(f"SELECT query, level, score, ms FROM log {w} ORDER BY id DESC LIMIT 12", p)]
    top = [dict(zip(("query", "hits"), r)) for r in db.execute("SELECT query, hits FROM entries ORDER BY hits DESC LIMIT 5")]
    return {"levels": d, "hits": hits, "total": tot, "hit_rate": hits / tot if tot else 0, "tokens_saved": saved,
            "cost_saved": saved / 1000 * COST_PER_1K_TOK, "entries": db.execute("SELECT COUNT(*) FROM entries").fetchone()[0],
            "thresholds": get_thr(), "recent": recent, "top": top}

@app.get("/api/entries")
def entries():
    return [dict(zip(("id", "ns", "tag", "cat", "query", "answer", "hits", "ts", "ttl"), r)) for r in db.execute(
        "SELECT id, ns, tag, cat, query, answer, hits, ts, ttl FROM entries ORDER BY last DESC LIMIT 200")]
@app.delete("/api/entries/{eid}")
async def del_entry(eid: int): await delete_ids([eid]); return {"ok": True}
@app.post("/api/clear")
async def clear():
    with lock: db.execute("DELETE FROM entries"); db.commit()
    await asyncio.to_thread(qd.delete_collection, COL); init_col()
    if redis:
        async for k in redis.scan_iter("sc:*"): await redis.delete(k)
    return {"ok": True}

class Inv(BaseModel): tag: str = ""; ns: str = ""
@app.post("/api/invalidate")
async def invalidate(b: Inv):
    ids = [r[0] for r in db.execute("SELECT id FROM entries WHERE (?='' OR tag=?) AND (?='' OR ns=?)", (b.tag, b.tag, b.ns, b.ns))]
    await delete_ids(ids); return {"deleted": len(ids)}

class FB(BaseModel): log_id: int; good: bool
@app.post("/api/feedback")
def feedback(b: FB):
    r = db.execute("SELECT cat, level FROM log WHERE id=?", (b.log_id,)).fetchone()
    if not r: raise HTTPException(404, "Unknown log id")
    t = get_thr()
    if r[1] == "L2_semantic": t[r[0]] = round(max(0.75, t[r[0]] - 0.002) if b.good else min(0.99, t[r[0]] + 0.01), 4)
    save_thr(t); return {"thresholds": t}
class Thr(BaseModel): code: float; general: float
@app.post("/api/threshold")
def set_thr(b: Thr): save_thr(b.dict()); return b.dict()

class Warm(BaseModel): queries: list[str]; ns: str = "default"
@app.post("/api/warm")
async def warm(b: Warm):
    qs = [x.strip() for x in b.queries if x.strip()]
    await asyncio.gather(*[ask(Ask(query=q, ns=b.ns)) for q in qs])   # concurrent: batched embeddings + coalescing
    return {"warmed": len(qs)}

@app.get("/api/simulate")
def simulate(queries: int = 10000, hit_rate: float = 0.6, llm_ms: float = 1500, cache_ms: float = 15, tok: int = 400):
    h = int(queries * hit_rate)
    return {"llm_calls_saved": h, "time_saved_min": round(h * (llm_ms - cache_ms) / 60000, 1),
            "cost_saved": round(h * tok / 1000 * COST_PER_1K_TOK, 2), "avg_ms": round((h * cache_ms + (queries - h) * llm_ms) / queries, 1)}

PAIRS = [("How do I reset my password?", "I forgot my password, how can I change it?", 1), ("What is RAG?", "Explain retrieval augmented generation", 1),
 ("How to install Python on Windows", "Steps to set up Python on a Windows PC", 1), ("What is the capital of France?", "Which city is France's capital?", 1),
 ("How do I enable dark mode?", "How can I turn on the dark theme?", 1), ("Best way to learn SQL fast", "Quickest method to study SQL", 1),
 ("How do I enable notifications?", "How do I disable notifications?", 0), ("Add a user to the group", "Remove a user from the group", 0),
 ("What is machine learning?", "What is the weather in Delhi today?", 0), ("How to install Docker", "How to uninstall Docker", 0),
 ("Delete my account", "Deactivate my account", 0), ("Explain Python lists", "Explain Java interfaces", 0), ("Open a file in Python", "Close a file in Python", 0)]
@app.get("/api/evaluate")
def evaluate():
    A, B = enc([p[0] for p in PAIRS]), enc([p[1] for p in PAIRS])
    sims = [float(np.dot(a, b)) for a, b in zip(A, B)]
    rr = [float(x) for x in reranker.predict([(p[0], p[1]) for p in PAIRS])] if reranker else [1.0] * len(PAIRS)
    ok = [verify(p[0], p[1]) and rr[i] >= RR_MIN for i, p in enumerate(PAIRS)]   # verify + reranker decision
    out = []
    for t in np.arange(0.5, 0.99, 0.03):
        f = lambda gate: (sum(1 for i, p in enumerate(PAIRS) if sims[i] >= t and p[2] and gate(i)),
                          sum(1 for i, p in enumerate(PAIRS) if sims[i] >= t and not p[2] and gate(i)))
        (tp, fp), (tp2, fp2) = f(lambda i: True), f(lambda i: ok[i]); pos = sum(p[2] for p in PAIRS)
        out.append({"t": round(float(t), 2), "precision": tp / (tp + fp) if tp + fp else 1.0, "recall": tp / pos,
                    "precision_rr": tp2 / (tp2 + fp2) if tp2 + fp2 else 1.0, "recall_rr": tp2 / pos, "fp": fp, "fp_rr": fp2})
    return {"curve": out, "pairs": [{"a": a, "b": b, "same": y, "sim": round(sims[i], 3), "rerank": round(rr[i], 3), "pass": ok[i]} for i, (a, b, y) in enumerate(PAIRS)]}

@app.get("/")
def home(): return FileResponse("static/index.html")
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
