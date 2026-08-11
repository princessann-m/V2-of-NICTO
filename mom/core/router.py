from ..models.model_registry import ModelRegistry


class Router:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def create_mme_config(self, task_repr: dict, seed: int = 0):
        # Simple heuristic: select experts by task_type
        task_type = task_repr.get("task_type", "general")
        candidates = self.registry.find_experts_for(task_type)
        # sparse activation: pick top N depending on complexity
        complexity = task_repr.get("complexity", "medium")
        n = 2
        if complexity == "easy":
            n = 1
        elif complexity == "hard":
            n = 3
        selected = candidates[:n]
        return {"selected_experts": selected, "seed": seed}
