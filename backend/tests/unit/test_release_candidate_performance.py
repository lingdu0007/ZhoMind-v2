from __future__ import annotations

import math

import pytest

from app.release_candidate import performance_live
from app.release_candidate.performance import (
    PerformanceSample,
    build_performance_profile,
    freeze_regression_envelope,
)


def _sample(
    *,
    ttft_ms: float,
    total_ms: float,
    retrieval_ms: float,
    generation_provider_ms: float,
    embedding_provider_ms: float,
    persistence_ms: float,
    outcome: str = "evidence_gated_answer",
    error_code: str | None = None,
    fallback_hops: int = 0,
) -> PerformanceSample:
    return PerformanceSample(
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        retrieval_ms=retrieval_ms,
        generation_provider_ms=generation_provider_ms,
        embedding_provider_ms=embedding_provider_ms,
        persistence_ms=persistence_ms,
        outcome=outcome,
        error_code=error_code,
        fallback_hops=fallback_hops,
    )


def test_profile_reports_percentiles_and_separates_external_provider_time() -> None:
    profile = build_performance_profile(
        run_id="performance-1-20260807T000000Z",
        concurrency=1,
        samples=(
            _sample(
                ttft_ms=100,
                total_ms=200,
                retrieval_ms=60,
                generation_provider_ms=80,
                embedding_provider_ms=20,
                persistence_ms=10,
            ),
            _sample(
                ttft_ms=200,
                total_ms=300,
                retrieval_ms=70,
                generation_provider_ms=110,
                embedding_provider_ms=30,
                persistence_ms=12,
            ),
            _sample(
                ttft_ms=300,
                total_ms=500,
                retrieval_ms=90,
                generation_provider_ms=170,
                embedding_provider_ms=40,
                persistence_ms=14,
                outcome="generation_unavailable",
                error_code="TimeoutError",
            ),
        ),
    )

    assert profile.load == {"concurrency": 1, "requests": 3}
    assert profile.metrics["ttft_ms"] == {"p50": 200.0, "p95": 300.0, "p99": 300.0}
    assert profile.metrics["total_ms"] == {"p50": 300.0, "p95": 500.0, "p99": 500.0}
    assert profile.metrics["error_rate"] == pytest.approx(1 / 3)
    assert profile.metrics["retrieval_ms"] == pytest.approx((60 + 70 + 90) / 3)
    assert profile.metrics["generation_provider_ms"] == pytest.approx((80 + 110 + 170) / 3)
    assert profile.metrics["embedding_provider_ms"] == pytest.approx((20 + 30 + 40) / 3)
    assert profile.metrics["persistence_ms"] == pytest.approx((10 + 12 + 14) / 3)
    assert profile.metrics["application_controlled_ms"] == pytest.approx((100 + 160 + 290) / 3)
    assert profile.error_counts == {"TimeoutError": 1}


def test_regression_envelope_is_derived_from_observed_project_controlled_variance() -> None:
    profile = build_performance_profile(
        run_id="performance-1-20260807T000000Z",
        concurrency=1,
        samples=(
            _sample(
                ttft_ms=100,
                total_ms=150,
                retrieval_ms=40,
                generation_provider_ms=50,
                embedding_provider_ms=10,
                persistence_ms=5,
            ),
            _sample(
                ttft_ms=120,
                total_ms=170,
                retrieval_ms=45,
                generation_provider_ms=50,
                embedding_provider_ms=10,
                persistence_ms=5,
            ),
            _sample(
                ttft_ms=160,
                total_ms=210,
                retrieval_ms=55,
                generation_provider_ms=50,
                embedding_provider_ms=10,
                persistence_ms=5,
            ),
        ),
    )

    envelope = freeze_regression_envelope((profile,))

    observed = [90.0, 110.0, 150.0]
    expected_spread = math.sqrt(sum((value - 350 / 3) ** 2 for value in observed) / len(observed))
    assert envelope == {
        "metric": "application_controlled_ms",
        "observed_sample_count": 3,
        "center_ms": pytest.approx(350 / 3),
        "spread_ms": pytest.approx(expected_spread),
        "tolerance_ms": float(math.ceil(3 * expected_spread)),
        "threshold_ms": pytest.approx(150 + math.ceil(3 * expected_spread)),
        "derivation": "p95 plus three observed population standard deviations; not derived from the 12-second target",
    }


def test_profile_rejects_provider_fallback_as_latency_evidence() -> None:
    with pytest.raises(ValueError, match="fallback"):
        build_performance_profile(
            run_id="performance-1-20260807T000000Z",
            concurrency=1,
            samples=(
                _sample(
                    ttft_ms=100,
                    total_ms=200,
                    retrieval_ms=60,
                    generation_provider_ms=80,
                    embedding_provider_ms=20,
                    persistence_ms=10,
                    fallback_hops=1,
                ),
            ),
        )


def test_stream_sample_measures_first_sse_body_byte_and_uses_only_diagnostics(monkeypatch) -> None:
    class _Response:
        status = 200

        def __init__(self) -> None:
            self._payload = (
                b'event: outcome\ndata: {"outcome":"evidence_gated_answer"}\n\n'
                b'event: retrieval_diagnostics\ndata: {"retrieval_diagnostics":'
                b'{"timing_ms":{"retrieval_ms":7,"generation_provider_ms":9,"embedding_provider_ms":5,"persistence_ms":3},'
                b'"fallback":{"hops":0}}}\n\n'
            )
            self.calls: list[int] = []

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, amount: int = -1) -> bytes:
            self.calls.append(amount)
            if amount == 1:
                first, self._payload = self._payload[:1], self._payload[1:]
                return first
            remaining, self._payload = self._payload, b""
            return remaining

    response = _Response()
    monkeypatch.setattr(performance_live, "urlopen", lambda *_args, **_kwargs: response)

    sample = performance_live._stream_sample("https://example.invalid", "token", "private question", 5)

    assert response.calls == [1, -1]
    assert sample.error_code is None
    assert sample.outcome == "evidence_gated_answer"
    assert sample.retrieval_ms == 7
    assert sample.generation_provider_ms == 9
    assert sample.embedding_provider_ms == 5
    assert sample.persistence_ms == 3
    assert sample.total_ms >= sample.ttft_ms >= 0


def test_stream_sample_classifies_closed_non_answer_outcomes(monkeypatch) -> None:
    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, amount: int = -1) -> bytes:
            payload = b'event: outcome\ndata: {"outcome":"generation_unavailable"}\n\n'
            if amount == 1:
                return payload[:1]
            return payload[1:]

    monkeypatch.setattr(performance_live, "urlopen", lambda *_args, **_kwargs: _Response())

    sample = performance_live._stream_sample("https://example.invalid", "token", "private question", 5)

    assert sample.outcome == "generation_unavailable"
    assert sample.error_code == "OUTCOME_GENERATION_UNAVAILABLE"
