import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DistillationDataset:
    def __init__(self, output_dir: str = "data/distilled"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.teacher = None
        self.verifier = None

    def set_teacher(self, teacher_model: Any) -> None:
        self.teacher = teacher_model

    def set_verifier(self, verifier_model: Any) -> None:
        self.verifier = verifier_model

    def generate_teacher_outputs(
        self,
        prompts: List[Dict[str, Any]],
        batch_size: int = 8,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> List[Dict[str, Any]]:
        if self.teacher is None:
            raise RuntimeError("Teacher model not set")
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            for prompt in batch:
                output = self._call_teacher(prompt, temperature, max_tokens)
                results.append({
                    "prompt": prompt.get("text") or prompt.get("prompt") or "",
                    "response": output,
                    "metadata": {k: v for k, v in prompt.items() if k not in ("text", "prompt")},
                })
        return results

    def verify_and_filter(
        self,
        samples: List[Dict[str, Any]],
        min_quality_score: float = 0.7,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if self.verifier is not None:
            samples = self._verify_with_model(samples, min_quality_score)
        else:
            samples = self._verify_heuristic(samples, min_quality_score)
        random.shuffle(samples)
        if max_samples is not None:
            samples = samples[:max_samples]
        logger.info("Verified samples: %d remaining", len(samples))
        return samples

    def score_quality(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored = []
        for sample in samples:
            score = self._compute_quality_score(sample)
            new_sample = dict(sample)
            new_sample["quality_score"] = score
            scored.append(new_sample)
        scored.sort(key=lambda x: x.get("quality_score", 0.0), reverse=True)
        return scored

    def prepare_student_data(
        self,
        samples: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        format: str = "jsonl",
    ) -> str:
        output = Path(output_path or str(self.output_dir / "student_train.jsonl"))
        output.parent.mkdir(parents=True, exist_ok=True)
        student_samples = []
        for sample in samples:
            student_samples.append({
                "input_ids": sample.get("input_ids", []),
                "labels": sample.get("labels", sample.get("input_ids", [])),
                "attention_mask": sample.get("attention_mask", []),
                "quality_score": sample.get("quality_score", 1.0),
            })
        with open(output, "w", encoding="utf-8") as f:
            for item in student_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Student training data saved to %s (%d samples)", output, len(student_samples))
        return str(output)

    def _call_teacher(self, prompt: Dict[str, Any], temperature: float, max_tokens: int) -> str:
        try:
            if hasattr(self.teacher, "generate"):
                return self.teacher.generate(
                    prompt.get("text") or prompt.get("prompt") or "",
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            if hasattr(self.teacher, "__call__"):
                return str(self.teacher(prompt))
        except Exception as exc:
            logger.error("Teacher generation failed: %s", exc)
        return ""

    def _verify_with_model(self, samples: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        verified = []
        for sample in samples:
            text = sample.get("response") or sample.get("text") or ""
            score = self._model_score(text)
            if score >= threshold:
                sample["verifier_score"] = score
                verified.append(sample)
        return verified

    def _verify_heuristic(self, samples: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        verified = []
        for sample in samples:
            text = sample.get("response") or sample.get("text") or ""
            score = self._heuristic_score(text)
            if score >= threshold:
                sample["quality_score"] = score
                verified.append(sample)
        return verified

    def _model_score(self, text: str) -> float:
        return 1.0

    def _heuristic_score(self, text: str) -> float:
        if not text:
            return 0.0
        length_score = min(len(text) / 200.0, 1.0)
        word_score = min(len(text.split()) / 50.0, 1.0)
        return (length_score + word_score) / 2.0

    def _compute_quality_score(self, sample: Dict[str, Any]) -> float:
        text = sample.get("response") or sample.get("text") or ""
        return self._heuristic_score(text)
