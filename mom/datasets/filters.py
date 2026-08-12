import hashlib
import logging
import random
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QualityFilter:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = []
        for sample in samples:
            score = self._score(sample)
            if score >= self.threshold:
                filtered.append(sample)
            else:
                logger.debug("Dropped low-quality sample score=%.2f", score)
        return filtered

    def _score(self, sample: Dict[str, Any]) -> float:
        text = sample.get("text") or sample.get("content") or ""
        if not text:
            return 0.0
        length_score = min(len(text) / 200.0, 1.0)
        word_score = min(len(text.split()) / 50.0, 1.0)
        punctuation_ratio = len(re.findall(r"[^\w\s]", text)) / max(len(text), 1)
        punct_score = 1.0 if punctuation_ratio < 0.2 else max(0.0, 1.0 - (punctuation_ratio - 0.2) * 2)
        unique_ratio = len(set(text.split())) / max(len(text.split()), 1)
        unique_score = min(unique_ratio * 2.0, 1.0)
        return (length_score + word_score + punct_score + unique_score) / 4.0


class Deduplicator:
    def __call__(self, samples: List[Dict[str, Any]], key: str = "text") -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for sample in samples:
            value = sample.get(key) or sample.get("content") or ""
            digest = hashlib.md5(value.encode("utf-8")).hexdigest()
            if digest not in seen:
                seen.add(digest)
                deduped.append(sample)
        logger.info("Deduplicated: %d -> %d", len(samples), len(deduped))
        return deduped


class LengthFilter:
    def __init__(self, min_length: int = 10, max_length: int = 100_000):
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, samples: List[Dict[str, Any]], key: str = "text") -> List[Dict[str, Any]]:
        filtered = []
        for sample in samples:
            value = sample.get(key) or sample.get("content") or ""
            if self.min_length <= len(value) <= self.max_length:
                filtered.append(sample)
        return filtered


class LanguageFilter:
    def __init__(self, language: str = "en"):
        self.language = language

    def __call__(self, samples: List[Dict[str, Any]], key: str = "text") -> List[Dict[str, Any]]:
        filtered = []
        for sample in samples:
            value = sample.get(key) or sample.get("content") or ""
            if self._detect_language(value) == self.language:
                filtered.append(sample)
        return filtered

    def _detect_language(self, text: str) -> str:
        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return "en"


class ToxicityFilter:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def __call__(self, samples: List[Dict[str, Any]], key: str = "text") -> List[Dict[str, Any]]:
        filtered = []
        for sample in samples:
            value = sample.get(key) or sample.get("content") or ""
            score = self._score_toxicity(value)
            if score < self.threshold:
                filtered.append(sample)
            else:
                logger.debug("Dropped toxic sample score=%.2f", score)
        return filtered

    def _score_toxicity(self, text: str) -> float:
        toxic_patterns = [
            r"\b(kill|murder|suicide|rape|terrorist)\b",
            r"\b(hate|racist|nazi|bigot)\b",
        ]
        score = 0.0
        for pattern in toxic_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            score += len(matches) * 0.3
        return min(score, 1.0)


class DataFilters:
    def __init__(self):
        self.quality = QualityFilter()
        self.deduplicate = Deduplicator()
        self.length = LengthFilter()
        self.language = LanguageFilter()
        self.toxicity = ToxicityFilter()

    def apply(self, samples: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        if kwargs.get("quality_threshold") is not None:
            self.quality.threshold = kwargs["quality_threshold"]
            samples = self.quality(samples)
        if kwargs.get("deduplicate", False):
            samples = self.deduplicate(samples)
        if kwargs.get("min_length") is not None or kwargs.get("max_length") is not None:
            self.length.min_length = kwargs.get("min_length", 10)
            self.length.max_length = kwargs.get("max_length", 100_000)
            samples = self.length(samples)
        if kwargs.get("language") is not None:
            self.language.language = kwargs["language"]
            samples = self.language(samples)
        if kwargs.get("remove_toxic", False):
            samples = self.toxicity(samples)
        return samples
