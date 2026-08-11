"""Configuration for MoM framework."""

from dataclasses import dataclass, field
import os


@dataclass
class LLMConfig:
    provider: str = "heuristic"
    api_key: str | None = None
    base_url: str | None = None
    model: str = ""
    timeout: float = 30.0
    max_tokens: int = 512
    temperature: float = 0.2


@dataclass
class MoMConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    global_deadline: float = 10.0
    max_retries: int = 1
    log_path: str = "logs/mom_requests.log"


def load_config() -> MoMConfig:
    return MoMConfig(
        llm=LLMConfig(
            provider=os.getenv("MOM_LLM_PROVIDER", "heuristic"),
            api_key=os.getenv("MOM_LLM_API_KEY"),
            base_url=os.getenv("MOM_LLM_BASE_URL"),
            model=os.getenv("MOM_LLM_MODEL", ""),
        ),
        global_deadline=float(os.getenv("MOM_DEADLINE", "10.0")),
        max_retries=int(os.getenv("MOM_MAX_RETRIES", "1")),
    )
