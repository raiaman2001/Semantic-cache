import asyncio
import time
from typing import AsyncGenerator, Iterator, Optional
from core.logging_config import setup_logger

logger = setup_logger("semantic_cache.llm")


class LLMService:
    """Manages LLM invocation with timeout, retries with exponential backoff, and model fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-2.0-flash",
        fallback_model: str = "gemini-1.5-flash",
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.default_model = default_model
        self.fallback_model = fallback_model
        self.timeout = timeout
        self.max_retries = max_retries

    def _sync_stream(self, query: str, model_name: str) -> Iterator[str]:
        if not self.api_key:
            for w in f"[mock {model_name} answer] Set GEMINI_API_KEY for real answers. You asked: {query}".split(" "):
                time.sleep(0.04)
                yield w + " "
            return

        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(query, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def stream(self, query: str, model_name: Optional[str] = None) -> AsyncGenerator[str, None]:
        chosen_model = model_name or self.default_model
        models_to_try = [chosen_model]
        if self.fallback_model and self.fallback_model != chosen_model:
            models_to_try.append(self.fallback_model)

        last_err: Optional[Exception] = None
        for current_model in models_to_try:
            for attempt in range(self.max_retries + 1):
                try:
                    # Attempt to stream from current model with timeout
                    started = False
                    it = self._sync_stream(query, current_model)
                    while True:
                        chunk = await asyncio.wait_for(
                            asyncio.to_thread(next, it, None),
                            timeout=self.timeout,
                        )
                        if chunk is None:
                            return
                        started = True
                        yield chunk
                except Exception as e:
                    last_err = e
                    logger.warning(
                        f"LLM stream error on {current_model} (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                    )
                    # If we already yielded tokens, do not retry from start as it would corrupt stream
                    if started:
                        raise e
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.5 * (2 ** attempt))

        if last_err:
            raise last_err

    async def complete(self, query: str, model_name: Optional[str] = None) -> str:
        parts: list[str] = []
        async for chunk in self.stream(query, model_name):
            parts.append(chunk)
        return "".join(parts)
