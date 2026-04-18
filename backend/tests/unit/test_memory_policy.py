import asyncio

from app.rag.memory.policies import ConservativeMemoryWritePolicy


def test_memory_write_rejects_low_importance() -> None:
    policy = ConservativeMemoryWritePolicy(
        min_importance=0.7,
        min_novelty=0.6,
        min_stability=0.8,
    )
    decision = asyncio.run(
        policy.should_write(
            fact={"importance": 0.4, "novelty": 0.9, "stability": 0.9, "safe": True},
            existing=[],
        )
    )
    assert decision["allow"] is False
    assert decision["reason"] == "low_importance"


def test_memory_write_accepts_high_signal_fact() -> None:
    policy = ConservativeMemoryWritePolicy(
        min_importance=0.7,
        min_novelty=0.6,
        min_stability=0.8,
    )
    decision = asyncio.run(
        policy.should_write(
            fact={"importance": 0.9, "novelty": 0.9, "stability": 0.9, "safe": True},
            existing=[],
        )
    )
    assert decision["allow"] is True
    assert decision["reason"] == "accepted"
