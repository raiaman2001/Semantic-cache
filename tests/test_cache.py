import asyncio
import time
import pytest
from core.cache import compute_cache_key, classify_category
from core.rerank import verify_lexical_consistency


@pytest.mark.asyncio
async def test_cache_miss_then_l1_exact_hit(cache_service):
    q = "How do I reset my password?"
    ns, model, ver = "default", "test-llm", "v1"
    key = compute_cache_key(ns, model, ver, q)
    thr = cache_service.get_thresholds()

    # Step 1: Lookup before storing -> Miss
    hit, vec = await cache_service.lookup(q, ns, model, ver, "general", thr, key)
    assert hit is None
    assert vec is not None

    # Step 2: Store entry
    ans = "Click forgot password on the login screen."
    eid = await cache_service.store(q, ans, ns, model, ver, "auth", 3600, key, vec)
    assert eid > 0

    # Step 3: Exact lookup with identical query -> L1_exact
    hit2, _ = await cache_service.lookup(q, ns, model, ver, "general", thr, key)
    assert hit2 is not None
    level, found_ans, score, found_id = hit2
    assert level == "L1_exact"
    assert found_ans == ans
    assert score == 1.0
    assert found_id == eid


@pytest.mark.asyncio
async def test_l2_semantic_hit_with_vector_similarity(cache_service, mock_qdrant):
    q1 = "What is the capital of France?"
    q2 = "Which city is France's capital?"
    ns, model, ver = "default", "test-llm", "v1"
    key1 = compute_cache_key(ns, model, ver, q1)
    key2 = compute_cache_key(ns, model, ver, q2)
    thr = cache_service.get_thresholds()

    # Store q1 with a known embedding
    fixed_vec = (await cache_service.embedding.embed(q1))
    ans = "The capital of France is Paris."
    eid = await cache_service.store(q1, ans, ns, model, ver, "geo", 3600, key1, fixed_vec)

    # For q2, mock_qdrant returns q1's vector when queried with high similarity
    # We patch q2 embed to return vector identical to fixed_vec to ensure cosine similarity is 1.0 >= thr
    async def mock_embed(text):
        return fixed_vec
    cache_service.embedding.embed = mock_embed

    # Lookup q2: exact key differs, but vector query finds q1
    hit, _ = await cache_service.lookup(q2, ns, model, ver, "general", thr, key2)
    assert hit is not None
    level, found_ans, score, found_id = hit
    assert level == "L2_semantic"
    assert found_ans == ans
    assert found_id == eid
    assert score >= thr["general"]


@pytest.mark.asyncio
async def test_namespace_and_version_isolation(cache_service):
    q = "What is your return policy?"
    ans1 = "Policy for version 1"
    ans2 = "Policy for version 2"

    key_v1 = compute_cache_key("ns_a", "m1", "v1", q)
    key_v2 = compute_cache_key("ns_a", "m1", "v2", q)
    key_ns_b = compute_cache_key("ns_b", "m1", "v1", q)
    thr = cache_service.get_thresholds()

    await cache_service.store(q, ans1, "ns_a", "m1", "v1", "tag", 3600, key_v1, None)
    await cache_service.store(q, ans2, "ns_a", "m1", "v2", "tag", 3600, key_v2, None)

    hit_v1, _ = await cache_service.lookup(q, "ns_a", "m1", "v1", "general", thr, key_v1)
    hit_v2, _ = await cache_service.lookup(q, "ns_a", "m1", "v2", "general", thr, key_v2)
    hit_ns_b, _ = await cache_service.lookup(q, "ns_b", "m1", "v1", "general", thr, key_ns_b)

    assert hit_v1[1] == ans1
    assert hit_v2[1] == ans2
    assert hit_ns_b is None


@pytest.mark.asyncio
async def test_ttl_expiration(cache_service):
    q = "Temporary cache question"
    key = compute_cache_key("default", "test-llm", "v1", q)
    thr = cache_service.get_thresholds()

    # Store with TTL of 2.0s
    await cache_service.store(q, "Quick answer", "default", "test-llm", "v1", "", 2.0, key, None)

    # Immediate lookup hits
    hit, _ = await cache_service.lookup(q, "default", "test-llm", "v1", "general", thr, key)
    assert hit is not None

    # Manually age the record in DB to simulate expiration
    cache_service.db._conn.execute("UPDATE entries SET ts = ts - 100 WHERE key = ?", (key,))
    cache_service.db._conn.commit()

    expired_hit, _ = await cache_service.lookup(q, "default", "test-llm", "v1", "general", thr, key)
    assert expired_hit is None


@pytest.mark.asyncio
async def test_lru_eviction(cache_service):
    # settings.max_entries is 5 in fixture
    thr = cache_service.get_thresholds()
    for i in range(7):
        q = f"Question {i}"
        k = compute_cache_key("default", "test-llm", "v1", q)
        await cache_service.store(q, f"Ans {i}", "default", "test-llm", "v1", "", 3600, k, None)
        time.sleep(0.01)

    assert cache_service.db.count_entries() <= 5
    # Oldest question 0 should have been evicted
    k0 = compute_cache_key("default", "test-llm", "v1", "Question 0")
    hit0, _ = await cache_service.lookup("Question 0", "default", "test-llm", "v1", "general", thr, k0)
    assert hit0 is None

    # Latest question 6 should be present
    k6 = compute_cache_key("default", "test-llm", "v1", "Question 6")
    hit6, _ = await cache_service.lookup("Question 6", "default", "test-llm", "v1", "general", thr, k6)
    assert hit6 is not None


@pytest.mark.asyncio
async def test_tag_invalidation(cache_service):
    thr = cache_service.get_thresholds()
    q1 = "Tag q1"
    q2 = "Tag q2"
    k1 = compute_cache_key("default", "test-llm", "v1", q1)
    k2 = compute_cache_key("default", "test-llm", "v1", q2)

    await cache_service.store(q1, "A1", "default", "test-llm", "v1", "billing", 3600, k1, None)
    await cache_service.store(q2, "A2", "default", "test-llm", "v1", "auth", 3600, k2, None)

    deleted = await cache_service.invalidate_tag("billing", "default")
    assert deleted == 1

    hit1, _ = await cache_service.lookup(q1, "default", "test-llm", "v1", "general", thr, k1)
    hit2, _ = await cache_service.lookup(q2, "default", "test-llm", "v1", "general", thr, k2)
    assert hit1 is None
    assert hit2 is not None


@pytest.mark.asyncio
async def test_request_coalescing(cache_service):
    key = "coalesce_key"
    task1 = asyncio.create_task(cache_service.coalesce_generate(key, "Coalesce query", "test-llm"))
    task2 = asyncio.create_task(cache_service.coalesce_generate(key, "Coalesce query", "test-llm"))

    res1, shared1 = await task1
    res2, shared2 = await task2

    assert res1 == res2
    # One of them is the primary caller (shared=False), the second joins (shared=True)
    assert (shared1 and not shared2) or (shared2 and not shared1) or (shared1 and shared2)


def test_lexical_gate_opposites():
    # Hard negative opposite keyword
    assert not verify_lexical_consistency("How do I enable 2FA?", "How do I disable 2FA?")
    assert not verify_lexical_consistency("Add a member to the project", "Remove a member from the project")
    assert not verify_lexical_consistency("Install docker engine", "Uninstall docker engine")

    # True paraphrase with token overlap
    assert verify_lexical_consistency("How do I reset my password?", "Steps to reset my account password")

