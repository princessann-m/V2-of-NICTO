from mom.core.orchestrator import Orchestrator
from mom.config import MoMConfig, LLMConfig


def run():
    cfg = MoMConfig(
        llm=LLMConfig(provider="heuristic"),
        global_deadline=120.0,
        max_retries=1,
    )
    orch = Orchestrator(cfg)
    prompt = "Calculate 12 * (3 + 4) and explain briefly."
    res = orch.handle_request(prompt, request_id="demo1")
    print("Answer:\n", res.get("answer"))
    print("Metadata:\n", res.get("metadata"))


if __name__ == "__main__":
    run()
