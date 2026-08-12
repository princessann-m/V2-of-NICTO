import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class JSONLLoader:
    def load(self, path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def save(self, data: List[Dict[str, Any]], path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


class JSONLoader:
    def load(self, path: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, data: Any, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class CSVLoader:
    def load(self, path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    def save(self, data: List[Dict[str, Any]], path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not data:
            return
        with open(p, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)


class ParquetLoader:
    def load(self, path: str) -> List[Dict[str, Any]]:
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            return df.to_dict(orient="records")
        except Exception as exc:
            raise RuntimeError(f"Parquet load failed: {exc}")

    def save(self, data: List[Dict[str, Any]], path: str) -> None:
        try:
            import pandas as pd
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(data).to_parquet(p, index=False)
        except Exception as exc:
            raise RuntimeError(f"Parquet save failed: {exc}")


class TextLoader:
    def load(self, path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return [{"text": line} for line in lines]

    def save(self, data: List[Dict[str, Any]], path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for item in data:
                text = item.get("text") or item.get("content") or ""
                f.write(text + "\n")


class ImageDatasetLoader:
    def load(self, path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        samples = []
        if p.is_dir():
            for img_path in sorted(p.rglob("*")):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                    samples.append({"image_path": str(img_path)})
        elif p.suffix == ".jsonl":
            with open(p, "r", encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]
        return samples

    def save(self, data: List[Dict[str, Any]], path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


class VideoDatasetLoader:
    def load(self, path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        samples = []
        if p.is_dir():
            for vid_path in sorted(p.rglob("*")):
                if vid_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                    samples.append({"video_path": str(vid_path)})
        elif p.suffix == ".jsonl":
            with open(p, "r", encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]
        return samples

    def save(self, data: List[Dict[str, Any]], path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


class DatasetLoaders:
    def __init__(self):
        self.jsonl = JSONLLoader()
        self.json = JSONLoader()
        self.csv = CSVLoader()
        self.parquet = ParquetLoader()
        self.text = TextLoader()
        self.image = ImageDatasetLoader()
        self.video = VideoDatasetLoader()

    def load(self, path: str, format: Optional[str] = None) -> List[Dict[str, Any]]:
        fmt = format or self._infer_format(path)
        loader = {
            "jsonl": self.jsonl,
            "json": self.json,
            "csv": self.csv,
            "parquet": self.parquet,
            "txt": self.text,
            "text": self.text,
            "image": self.image,
            "video": self.video,
        }.get(fmt)
        if loader is None:
            raise ValueError(f"Unsupported format: {fmt}")
        return loader.load(path)

    def save(self, data: List[Dict[str, Any]], path: str, format: Optional[str] = None) -> None:
        fmt = format or self._infer_format(path)
        loader = {
            "jsonl": self.jsonl,
            "json": self.json,
            "csv": self.csv,
            "parquet": self.parquet,
            "txt": self.text,
            "text": self.text,
            "image": self.image,
            "video": self.video,
        }.get(fmt)
        if loader is None:
            raise ValueError(f"Unsupported format: {fmt}")
        loader.save(data, path)

    @staticmethod
    def _infer_format(path: str) -> str:
        ext = Path(path).suffix.lower().lstrip(".")
        if ext in {"txt", "text"}:
            return "txt"
        return ext
