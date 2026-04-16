class ConservativeMemoryWritePolicy:
    def __init__(self, *, min_importance: float, min_novelty: float, min_stability: float) -> None:
        self.min_importance = min_importance
        self.min_novelty = min_novelty
        self.min_stability = min_stability

    async def should_write(self, fact: dict, existing: list[dict]) -> dict:
        if not fact.get("safe", False):
            return {"allow": False, "reason": "unsafe"}
        if float(fact.get("importance", 0.0)) < self.min_importance:
            return {"allow": False, "reason": "low_importance"}
        if float(fact.get("novelty", 0.0)) < self.min_novelty:
            return {"allow": False, "reason": "low_novelty"}
        if float(fact.get("stability", 0.0)) < self.min_stability:
            return {"allow": False, "reason": "low_stability"}
        return {"allow": True, "reason": "accepted"}
