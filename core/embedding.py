import asyncio
from typing import Any, Protocol

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        ...


class EmbeddingService:
    """Manages text embedding generation, normalization, and micro-batching."""

    def __init__(self, model_instance: Any = None, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model: Any = model_instance
        self._model_name = model_name
        self._q: asyncio.Queue[tuple[str, asyncio.Future[np.ndarray]]] | None = None
        self._batch_task: asyncio.Task[None] | None = None

    def get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode_sync(self, texts: list[str]) -> np.ndarray:
        model = self.get_model()
        vecs = model.encode(texts).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vecs / norms

    async def embed(self, text: str) -> np.ndarray:
        """Micro-batching: requests arriving within 10ms are grouped into one batch."""
        if self._q is None:
            self._q = asyncio.Queue()
            self._batch_task = asyncio.create_task(self._run_batcher())

        loop = asyncio.get_running_loop()
        future: asyncio.Future[np.ndarray] = loop.create_future()
        await self._q.put((text, future))
        return await future

    async def _run_batcher(self) -> None:
        assert self._q is not None
        while True:
            item = await self._q.get()
            items = [item]
            await asyncio.sleep(0.01)
            while not self._q.empty() and len(items) < 32:
                items.append(self._q.get_nowait())

            try:
                texts = [t for t, _ in items]
                vs = await asyncio.to_thread(self.encode_sync, texts)
                for (_, fut), v in zip(items, vs):
                    if not fut.done():
                        fut.set_result(v)
            except Exception as e:
                for _, fut in items:
                    if not fut.done():
                        fut.set_exception(e)

