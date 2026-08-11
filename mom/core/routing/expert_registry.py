"""Expert registry with 310 experts and domain taxonomy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

LOG = logging.getLogger(__name__)

DOMAINS = [
    "nlp", "vision", "audio", "video", "multimodal", "reasoning", "math", "code",
    "science", "medicine", "law", "finance", "education", "creative", "robotics",
    "security", "data", "search", "translation", "summarization", "qa", "generation",
    "verification", "planning", "tool_use",
]

SUBDOMAINS = {
    "nlp": ["sentiment", "ner", "parsing", "generation", "embedding"],
    "vision": ["detection", "segmentation", "classification", "ocr", "depth"],
    "audio": ["asr", "tts", "classification", "separation", "music"],
    "video": ["generation", "editing", "classification", "captioning", "tracking"],
    "multimodal": ["alignment", "retrieval", "vqa", "captioning", "reasoning"],
    "reasoning": ["logic", "commonsense", "math_proof", "planning", "abductive"],
    "math": ["algebra", "calculus", "geometry", "number_theory", "stats"],
    "code": ["generation", "repair", "review", "translation", "debug"],
    "science": ["physics", "chemistry", "biology", "astronomy", "earth"],
    "medicine": ["diagnosis", "drug", "imaging", "genomics", "clinical"],
    "law": ["contract", "litigation", "compliance", "ip", "regulatory"],
    "finance": ["trading", "risk", "accounting", "crypto", "macro"],
    "education": ["tutoring", "assessment", "curriculum", "feedback", "adaptive"],
    "creative": ["writing", "art", "music", "design", "storytelling"],
    "robotics": ["navigation", "manipulation", "perception", "planning", "simulation"],
    "security": ["malware", "pentest", "forensics", "crypto_sec", "compliance"],
    "data": ["etl", "analytics", "visualization", "engineering", "governance"],
    "search": ["web", "enterprise", "semantic", "federated", "personalized"],
    "translation": ["text", "speech", "document", "real_time", "low_resource"],
    "summarization": ["extractive", "abstractive", "multilingual", "meeting", "legal"],
    "qa": ["open_domain", "closed_domain", "factual", "conversational", "multi_hop"],
    "generation": ["text", "image", "video", "audio", "code"],
    "verification": ["fact_check", "hallucination", "bias", "robustness", "safety"],
    "planning": ["task", "route", "resource", "conversation", "emergency"],
    "tool_use": ["api", "database", "web", "code_exec", "file_system"],
}


def _generate_experts() -> list[dict[str, Any]]:
    experts = []
    eid = 0
    for domain in DOMAINS:
        subdomains = SUBDOMAINS.get(domain, [domain])
        per_sub = max(1, (310 // len(DOMAINS)) // max(1, len(subdomains)))
        for sub in subdomains:
            for i in range(per_sub):
                experts.append({
                    "id": f"exp_{eid:03d}",
                    "name": f"{domain}_{sub}_{i}",
                    "domain": domain,
                    "subdomain": sub,
                    "capabilities": [domain, sub, f"variant_{i % 4}"],
                    "modality": _modality_for(domain),
                    "latency_baseline_ms": 200 + (i % 10) * 50,
                    "quality_score": 0.7 + ((i * 7) % 30) / 100.0,
                    "compute_cost": 0.1 + (i % 5) * 0.05,
                    "active": True,
                })
                eid += 1
                if eid >= 310:
                    break
            if eid >= 310:
                break
        if eid >= 310:
            break
    while len(experts) < 310:
        experts.append({
            "id": f"exp_{len(experts):03d}",
            "name": f"general_fallback_{len(experts)}",
            "domain": "general",
            "subdomain": "general",
            "capabilities": ["general"],
            "modality": "text",
            "latency_baseline_ms": 500.0,
            "quality_score": 0.6,
            "compute_cost": 0.1,
            "active": True,
        })
    return experts[:310]


def _modality_for(domain: str) -> str:
    if domain in {"vision", "video", "image"}:
        return "image"
    if domain in {"audio", "music"}:
        return "audio"
    if domain in {"multimodal", "generation"}:
        return "multimodal"
    return "text"


@dataclass
class ExpertMetadata:
    id: str
    name: str
    domain: str
    subdomain: str
    capabilities: list[str]
    modality: str
    latency_baseline_ms: float
    quality_score: float
    compute_cost: float
    active: bool = True
    performance_history: list[dict[str, Any]] = field(default_factory=list)

    def record_performance(self, latency_ms: float, quality: float, success: bool = True) -> None:
        self.performance_history.append({
            "latency_ms": latency_ms,
            "quality": quality,
            "success": success,
            "timestamp": time.time(),
        })
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]


class ExpertRegistry:
    def __init__(self) -> None:
        self._experts: dict[str, ExpertMetadata] = {}
        self._domain_index: dict[str, list[str]] = {}
        self._subdomain_index: dict[str, list[str]] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        for raw in _generate_experts():
            meta = ExpertMetadata(
                id=raw["id"],
                name=raw["name"],
                domain=raw["domain"],
                subdomain=raw["subdomain"],
                capabilities=raw["capabilities"],
                modality=raw["modality"],
                latency_baseline_ms=raw["latency_baseline_ms"],
                quality_score=raw["quality_score"],
                compute_cost=raw["compute_cost"],
                active=raw["active"],
            )
            self._experts[meta.id] = meta
            self._domain_index.setdefault(meta.domain, []).append(meta.id)
            self._subdomain_index.setdefault(f"{meta.domain}:{meta.subdomain}", []).append(meta.id)

    def register(self, expert: ExpertMetadata) -> None:
        self._experts[expert.id] = expert
        self._domain_index.setdefault(expert.domain, []).append(expert.id)
        self._subdomain_index.setdefault(f"{expert.domain}:{expert.subdomain}", []).append(expert.id)

    def query(self, domain: str | None = None, subdomain: str | None = None, modality: str | None = None) -> list[ExpertMetadata]:
        ids = set(self._experts.keys())
        if domain:
            ids &= set(self._domain_index.get(domain, []))
        if subdomain and domain:
            ids &= set(self._subdomain_index.get(f"{domain}:{subdomain}", []))
        if modality:
            ids = {eid for eid in ids if self._experts[eid].modality == modality}
        return [self._experts[eid] for eid in ids if self._experts[eid].active]

    def get(self, expert_id: str) -> ExpertMetadata | None:
        return self._experts.get(expert_id)

    def all(self) -> list[ExpertMetadata]:
        return [e for e in self._experts.values() if e.active]

    def domains(self) -> list[str]:
        return list(self._domain_index.keys())

    def subdomains(self, domain: str) -> list[str]:
        return list({self._experts[eid].subdomain for eid in self._domain_index.get(domain, [])})

    def size(self) -> int:
        return len(self._experts)
