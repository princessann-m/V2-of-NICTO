import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class DatasetPreparer:
    def __init__(
        self,
        output_dir: str = "data/processed",
        tokenizer_name: str = "gpt2",
        max_length: int = 2048,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self._tokenizer = None

    def download(
        self,
        source: Union[str, List[str]],
        output_dir: Optional[str] = None,
        source_type: str = "auto",
    ) -> List[str]:
        if isinstance(source, str):
            source = [source]
        downloaded_paths = []
        output = Path(output_dir or self.output_dir / "raw")
        output.mkdir(parents=True, exist_ok=True)
        for src in source:
            if source_type == "auto":
                if src.startswith("http://") or src.startswith("https://"):
                    source_type = "url"
                elif "/" in src and not Path(src).exists():
                    source_type = "huggingface"
                else:
                    source_type = "local"
            if source_type == "url":
                path = self._download_url(src, output)
            elif source_type == "huggingface":
                path = self._download_huggingface(src, output)
            else:
                path = self._copy_local(src, output)
            downloaded_paths.append(str(path))
            logger.info("Downloaded %s -> %s", src, path)
        return downloaded_paths

    def prepare(
        self,
        input_paths: Union[str, List[str]],
        output_dir: Optional[str] = None,
        clean: bool = True,
        filter_kwargs: Optional[Dict[str, Any]] = None,
        deduplicate: bool = True,
    ) -> str:
        if isinstance(input_paths, str):
            input_paths = [input_paths]
        output = Path(output_dir or self.output_dir / "prepared")
        output.mkdir(parents=True, exist_ok=True)
        combined = []
        for path in input_paths:
            combined.extend(self._read(path))
        if clean:
            combined = self._clean(combined)
        if deduplicate:
            combined = self._deduplicate(combined)
        if filter_kwargs:
            combined = self._apply_filters(combined, filter_kwargs)
        out_path = output / "dataset.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in combined:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Prepared %d samples -> %s", len(combined), out_path)
        return str(out_path)

    def tokenize(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        format: str = "jsonl",
    ) -> str:
        if self._tokenizer is None:
            self._load_tokenizer()
        output = Path(output_dir or self.output_dir / "tokenized")
        output.mkdir(parents=True, exist_ok=True)
        samples = self._read(input_path)
        tokenized = []
        for sample in samples:
            text = sample.get("text", "") or sample.get("content", "")
            if not text:
                continue
            tokens = self._tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokenized.append({
                "input_ids": tokens["input_ids"][0].tolist(),
                "attention_mask": tokens["attention_mask"][0].tolist(),
                "metadata": {k: v for k, v in sample.items() if k not in ("text", "content")},
            })
        out_path = output / f"tokenized.{format}"
        self._save(tokenized, out_path, format=format)
        logger.info("Tokenized %d samples -> %s", len(tokenized), out_path)
        return str(out_path)

    def split(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 42,
    ) -> Dict[str, str]:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        output = Path(output_dir or self.output_dir / "splits")
        output.mkdir(parents=True, exist_ok=True)
        samples = self._read(input_path)
        import random
        random.seed(seed)
        random.shuffle(samples)
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        splits = {
            "train": samples[:train_end],
            "val": samples[train_end:val_end],
            "test": samples[val_end:],
        }
        paths = {}
        for split_name, split_samples in splits.items():
            path = output / f"{split_name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for item in split_samples:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            paths[split_name] = str(path)
            logger.info("Split %s: %d samples -> %s", split_name, len(split_samples), path)
        return paths

    def save(
        self,
        data: Union[str, List[Dict[str, Any]]],
        output_path: str,
        format: str = "jsonl",
    ) -> str:
        if isinstance(data, str):
            data = self._read(data)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._save(data, out, format=format)
        logger.info("Saved %d samples to %s", len(data), out)
        return str(out)

    def _download_url(self, url: str, output_dir: Path) -> Path:
        import urllib.request
        fname = url.split("/")[-1] or "dataset"
        out_path = output_dir / fname
        urllib.request.urlretrieve(url, out_path)
        return out_path

    def _download_huggingface(self, repo_id: str, output_dir: Path) -> Path:
        try:
            from datasets import load_dataset
            ds = load_dataset(repo_id, split="train", trust_remote_code=True)
            out_path = output_dir / f"{repo_id.replace('/', '_')}.jsonl"
            ds.to_json(str(out_path))
            return out_path
        except Exception as exc:
            logger.error("Failed to download %s: %s", repo_id, exc)
            raise

    def _copy_local(self, src: str, output_dir: Path) -> Path:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(src)
        out_path = output_dir / src_path.name
        import shutil
        shutil.copy2(src_path, out_path)
        return out_path

    def _read(self, path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        if p.suffix == ".jsonl":
            with open(p, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        if p.suffix == ".json":
            import json
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]
        if p.suffix == ".csv":
            import csv
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return [dict(row) for row in reader]
        if p.suffix == ".parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(p)
                return df.to_dict(orient="records")
            except Exception as exc:
                raise RuntimeError(f"Parquet read failed: {exc}")
        if p.suffix == ".txt":
            with open(p, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            return [{"text": line} for line in lines]
        raise ValueError(f"Unsupported file format: {p.suffix}")

    def _save(self, data: List[Dict[str, Any]], path: Path, format: str = "jsonl") -> None:
        if format == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif format == "json":
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == "parquet":
            import pandas as pd
            pd.DataFrame(data).to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported save format: {format}")

    def _clean(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned = []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            text = sample.get("text") or sample.get("content") or ""
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < 10:
                continue
            if text:
                sample["text"] = text
                cleaned.append(sample)
        return cleaned

    def _deduplicate(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for sample in samples:
            text = sample.get("text") or sample.get("content") or ""
            key = hashlib.md5(text.encode("utf-8")).hexdigest()
            if key not in seen:
                seen.add(key)
                deduped.append(sample)
        return deduped

    def _apply_filters(self, samples: List[Dict[str, Any]], kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        from .filters import DataFilters
        filters = DataFilters()
        if kwargs.get("quality_threshold") is not None:
            samples = filters.quality(samples, threshold=kwargs["quality_threshold"])
        if kwargs.get("min_length") is not None or kwargs.get("max_length") is not None:
            samples = filters.length(
                samples,
                min_length=kwargs.get("min_length", 0),
                max_length=kwargs.get("max_length", float("inf")),
            )
        if kwargs.get("language") is not None:
            samples = filters.language(samples, language=kwargs["language"])
        if kwargs.get("remove_toxic", False):
            samples = filters.toxicity(samples)
        return samples

    def _load_tokenizer(self) -> None:
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        except Exception:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding(self.tokenizer_name)
            except Exception:
                self._tokenizer = lambda text, **kwargs: {"input_ids": list(text.encode("utf-8")), "attention_mask": [1] * len(text)}
