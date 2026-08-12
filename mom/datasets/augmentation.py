import logging
import random
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TextAugmenter:
    def __init__(self, synonym_prob: float = 0.1, backtranslation: bool = False):
        self.synonym_prob = synonym_prob
        self.backtranslation = backtranslation
        self._synonym_map = self._build_synonym_map()

    def __call__(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        augmented = []
        for sample in samples:
            text = sample.get("text") or sample.get("content") or ""
            if not text:
                augmented.append(sample)
                continue
            new_text = self._synonym_replace(text)
            if self.backtranslation:
                new_text = self._backtranslate(new_text)
            new_sample = dict(sample)
            new_sample["text"] = new_text
            augmented.append(new_sample)
        return augmented

    def _synonym_replace(self, text: str) -> str:
        words = text.split()
        new_words = []
        for word in words:
            if random.random() < self.synonym_prob and word.lower() in self._synonym_map:
                synonyms = self._synonym_map[word.lower()]
                new_words.append(random.choice(synonyms))
            else:
                new_words.append(word)
        return " ".join(new_words)

    def _backtranslate(self, text: str) -> str:
        return text

    @staticmethod
    def _build_synonym_map() -> Dict[str, List[str]]:
        return {
            "good": ["great", "excellent", "fine"],
            "bad": ["poor", "terrible", "awful"],
            "happy": ["joyful", "cheerful", "pleased"],
            "sad": ["unhappy", "gloomy", "miserable"],
            "fast": ["quick", "rapid", "swift"],
            "slow": ["sluggish", "leisurely", "unhurried"],
            "big": ["large", "huge", "enormous"],
            "small": ["tiny", "little", "miniature"],
            "smart": ["intelligent", "clever", "brilliant"],
            "stupid": ["foolish", "idiotic", "dense"],
        }


class CodeAugmenter:
    def __init__(self, rename_prob: float = 0.2, format_prob: float = 0.3):
        self.rename_prob = rename_prob
        self.format_prob = format_prob

    def __call__(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        augmented = []
        for sample in samples:
            text = sample.get("text") or sample.get("content") or ""
            if not text:
                augmented.append(sample)
                continue
            new_text = self._rename_variables(text)
            if random.random() < self.format_prob:
                new_text = self._reformat(new_text)
            new_sample = dict(sample)
            new_sample["text"] = new_text
            augmented.append(new_sample)
        return augmented

    def _rename_variables(self, text: str) -> str:
        identifiers = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text))
        reserved = {
            "def", "class", "return", "if", "else", "for", "while", "import",
            "from", "as", "try", "except", "with", "lambda", "yield", "True",
            "False", "None", "and", "or", "not", "in", "is",
        }
        candidates = [name for name in identifiers if name not in reserved and len(name) > 2]
        random.shuffle(candidates)
        to_rename = candidates[: max(1, int(len(candidates) * self.rename_prob))]
        mapping = {old: f"var_{random.randint(1000, 9999)}" for old in to_rename}
        for old, new in mapping.items():
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)
        return text

    def _reformat(self, text: str) -> str:
        return text


class MathAugmenter:
    def __init__(self, number_substitute_prob: float = 0.15, equation_variation: bool = True):
        self.number_substitute_prob = number_substitute_prob
        self.equation_variation = equation_variation

    def __call__(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        augmented = []
        for sample in samples:
            text = sample.get("text") or sample.get("content") or ""
            if not text:
                augmented.append(sample)
                continue
            new_text = self._substitute_numbers(text)
            if self.equation_variation:
                new_text = self._vary_equations(new_text)
            new_sample = dict(sample)
            new_sample["text"] = new_text
            augmented.append(new_sample)
        return augmented

    def _substitute_numbers(self, text: str) -> str:
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        substituted = set()
        new_text = text
        for num in numbers:
            if num in substituted:
                continue
            if random.random() < self.number_substitute_prob:
                if "." in num:
                    val = float(num)
                    new_val = round(val + random.uniform(-val * 0.2, val * 0.2), 2)
                    new_num = str(max(0.0, new_val))
                else:
                    val = int(num)
                    new_val = val + random.randint(-max(1, int(val * 0.2)), max(1, int(val * 0.2)))
                    new_num = str(max(0, new_val))
                new_text = new_text.replace(num, new_num)
                substituted.add(num)
        return new_text

    def _vary_equations(self, text: str) -> str:
        return text


class DataAugmentation:
    def __init__(self):
        self.text = TextAugmenter()
        self.code = CodeAugmenter()
        self.math = MathAugmenter()

    def augment(self, samples: List[Dict[str, Any]], domain: str = "text", **kwargs) -> List[Dict[str, Any]]:
        if domain == "text":
            return self.text(samples)
        if domain == "code":
            return self.code(samples)
        if domain == "math":
            return self.math(samples)
        raise ValueError(f"Unknown augmentation domain: {domain}")
