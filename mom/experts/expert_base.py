"""Base expert fallback when no concrete expert is registered."""


class ExpertBase:
    def __init__(self, meta: dict):
        self.meta = meta

    def compute(self, task: dict) -> dict:
        return {
            "answer": f"(placeholder answer by {self.meta.get('name')})",
            "metadata": {"expert": self.meta.get("name")},
        }
