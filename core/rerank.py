import asyncio
import re
from typing import List, Optional, Tuple, Protocol


OPPOSITES: List[Tuple[str, str]] = [
    ("enable", "disable"),
    ("add", "remove"),
    ("open", "close"),
    ("start", "stop"),
    ("install", "uninstall"),
    ("allow", "block"),
    ("increase", "decrease"),
    ("encrypt", "decrypt"),
]


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def verify_lexical_consistency(a: str, b: str) -> bool:
    """
    Fast lexical guard:
    1. Rejects if antonym/opposite keywords are present exclusively in one query.
    2. Enforces minimum Jaccard token overlap (>= 0.2).
    """
    set_a = set(normalize_text(a).split())
    set_b = set(normalize_text(b).split())
    if not set_a or not set_b:
        return False

    for x, y in OPPOSITES:
        if (x in set_a and y in set_b and x not in set_b) or (y in set_a and x in set_b and y not in set_b):
            return False

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return (intersection / max(1, union)) >= 0.2


class CrossEncoderProtocol(Protocol):
    def predict(self, sentences: List[Tuple[str, str]]) -> List[float]:
        ...


class RerankService:
    """Manages cross-encoder verification with fallback and lexical pre-filtering."""

    def __init__(
        self,
        model_instance: Optional[CrossEncoderProtocol] = None,
        model_name: str = "cross-encoder/stsb-TinyBERT-L-4",
        min_score: float = 0.6,
        enabled: bool = True,
    ) -> None:
        self._model = model_instance
        self._model_name = model_name
        self.min_score = min_score
        self.enabled = enabled
        self._load_failed = False

    def get_model(self) -> Optional[CrossEncoderProtocol]:
        if not self.enabled or self._load_failed:
            return None
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name)
            except Exception:
                self._load_failed = True
                return None
        return self._model

    async def confirm(self, query: str, candidate: str) -> bool:
        """Borderline check: lexical guard first, then cross-encoder score."""
        if not verify_lexical_consistency(query, candidate):
            return False

        model = self.get_model()
        if not model:
            return True

        score = await asyncio.to_thread(lambda: float(model.predict([(query, candidate)])[0]))
        return score >= self.min_score

