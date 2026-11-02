import asyncio
import numpy as np
import pytest
from typing import List, Tuple
from config import Settings
from core.cache import SemanticCacheService
from core.embedding import EmbeddingService
from core.llm import LLMService
from core.rerank import RerankService
from db.database import Database


class StubEmbeddingModel:
    """Deterministic embedding stub based on character frequency / keyword hashing."""
    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            np.random.seed(abs(hash(t.lower())) % (2**31))
            v = np.random.randn(384).astype(np.float32)
            norm = np.linalg.norm(v)
            vecs.append(v / (norm if norm > 0 else 1.0))
        return np.vstack(vecs)


class StubRerankerModel:
    """Cross encoder stub returning predictable scores."""
    def __init__(self, score_mapping: dict = None):
        self.score_mapping = score_mapping or {}

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        results = []
        for a, b in pairs:
            if (a, b) in self.score_mapping:
                results.append(self.score_mapping[(a, b)])
            else:
                results.append(0.85)
        return results


class InMemoryQdrantMock:
    """Lightweight in-memory simulation of Qdrant client."""
    class Point:
        def __init__(self, id, score, payload):
            self.id = id
            self.score = score
            self.payload = payload

    class QueryPointsResult:
        def __init__(self, points):
            self.points = points

    def __init__(self):
        self.points_store = {}
        self.collection_exists_flag = True

    def collection_exists(self, name):
        return self.collection_exists_flag

    def create_collection(self, name, vectors_config):
        self.collection_exists_flag = True

    def delete_collection(self, name):
        self.points_store.clear()

    def upsert(self, collection_name, points):
        for p in points:
            self.points_store[p.id] = (np.array(p.vector, dtype=np.float32), p.payload)

    def delete(self, collection_name, points_selector):
        if hasattr(points_selector, "points"):
            for pid in points_selector.points:
                self.points_store.pop(pid, None)

    def query_points(self, collection_name, query, query_filter=None, limit=3):
        qvec = np.array(query, dtype=np.float32)
        qnorm = np.linalg.norm(qvec)
        scored = []
        for pid, (vec, payload) in self.points_store.items():
            # Check filter
            if query_filter:
                conditions = getattr(query_filter, "must", None) or []
                match = True
                for cond in conditions:
                    k = getattr(cond, "key", None)
                    v = getattr(getattr(cond, "match", None), "value", None)
                    if k is not None and payload.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            vnorm = np.linalg.norm(vec)
            score = float(np.dot(qvec, vec) / (qnorm * vnorm + 1e-12))
            scored.append(self.Point(pid, score, payload))
        scored.sort(key=lambda x: x.score, reverse=True)
        return self.QueryPointsResult(scored[:limit])


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_cache.db"
    db = Database(str(db_file))
    yield db
    db.close()


@pytest.fixture
def stub_settings(tmp_path):
    return Settings(
        cache_api_key=None,
        rate_limit=10,
        max_entries=5,
        default_ttl=86400,
        max_query_length=200,
        sqlite_db_path=str(tmp_path / "test_cache.db"),
        gemini_api_key=None,
        rerank_enabled=True,
    )


@pytest.fixture
def mock_qdrant():
    return InMemoryQdrantMock()


@pytest.fixture
def stub_embedding_svc():
    return EmbeddingService(model_instance=StubEmbeddingModel())


@pytest.fixture
def stub_rerank_svc():
    return RerankService(model_instance=StubRerankerModel(), enabled=True)


@pytest.fixture
def stub_llm_svc():
    return LLMService(api_key=None, default_model="test-llm")


@pytest.fixture
def cache_service(stub_settings, temp_db, stub_embedding_svc, stub_rerank_svc, stub_llm_svc, mock_qdrant):
    return SemanticCacheService(
        settings=stub_settings,
        db=temp_db,
        embedding_service=stub_embedding_svc,
        rerank_service=stub_rerank_svc,
        llm_service=stub_llm_svc,
        qdrant_client=mock_qdrant,
        redis_client=None,
    )

