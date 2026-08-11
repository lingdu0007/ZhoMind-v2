from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from statistics import fmean, pstdev


@dataclass(frozen=True)
class PerformanceSample:
    """One content-free measurement from an authenticated product request."""

    ttft_ms: float
    total_ms: float
    retrieval_ms: float
    generation_provider_ms: float
    embedding_provider_ms: float
    persistence_ms: float
    outcome: str
    error_code: str | None
    fallback_hops: int

    @property
    def application_controlled_ms(self) -> float:
        return max(0.0, self.total_ms - self.generation_provider_ms - self.embedding_provider_ms)


@dataclass(frozen=True)
class PerformanceProfile:
    run_id: str
    concurrency: int
    samples: tuple[PerformanceSample, ...]

    @property
    def load(self) -> dict[str, int]:
        return {"concurrency": self.concurrency, "requests": len(self.samples)}

    @property
    def metrics(self) -> dict[str, object]:
        return {
            "ttft_ms": _percentiles(sample.ttft_ms for sample in self.samples),
            "total_ms": _percentiles(sample.total_ms for sample in self.samples),
            "error_rate": sum(sample.error_code is not None for sample in self.samples) / len(self.samples),
            "retrieval_ms": fmean(sample.retrieval_ms for sample in self.samples),
            "generation_provider_ms": fmean(sample.generation_provider_ms for sample in self.samples),
            "embedding_provider_ms": fmean(sample.embedding_provider_ms for sample in self.samples),
            "persistence_ms": fmean(sample.persistence_ms for sample in self.samples),
            "application_controlled_ms": fmean(sample.application_controlled_ms for sample in self.samples),
        }

    @property
    def error_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(sample.error_code for sample in self.samples if sample.error_code).items()))

    @property
    def successful_application_controlled_ms(self) -> tuple[float, ...]:
        return tuple(sample.application_controlled_ms for sample in self.samples if sample.error_code is None)


def build_performance_profile(
    *,
    run_id: str,
    concurrency: int,
    samples: tuple[PerformanceSample, ...],
) -> PerformanceProfile:
    if not run_id:
        raise ValueError("run_id is required")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if not samples:
        raise ValueError("at least one performance sample is required")
    for sample in samples:
        if sample.fallback_hops:
            raise ValueError("performance evidence rejects provider fallback")
        if sample.fallback_hops < 0:
            raise ValueError("fallback_hops cannot be negative")
        _validate_measurement(sample)
    return PerformanceProfile(run_id=run_id, concurrency=concurrency, samples=samples)


def freeze_regression_envelope(profiles: tuple[PerformanceProfile, ...]) -> dict[str, object]:
    observed = tuple(
        value
        for profile in profiles
        for value in profile.successful_application_controlled_ms
    )
    if not observed:
        raise ValueError("a regression envelope requires successful samples")
    center = fmean(observed)
    spread = pstdev(observed)
    tolerance = float(max(1, math.ceil(3 * spread)))
    p95 = _percentile(observed, 95)
    return {
        "metric": "application_controlled_ms",
        "observed_sample_count": len(observed),
        "center_ms": center,
        "spread_ms": spread,
        "tolerance_ms": tolerance,
        "threshold_ms": p95 + tolerance,
        "derivation": "p95 plus three observed population standard deviations; not derived from the 12-second target",
    }


def _percentiles(values: object) -> dict[str, float]:
    ordered = tuple(sorted(float(value) for value in values))  # type: ignore[arg-type]
    return {f"p{percentile}": _percentile(ordered, percentile) for percentile in (50, 95, 99)}


def _percentile(values: tuple[float, ...], percentile: int) -> float:
    if not values:
        raise ValueError("percentiles require at least one value")
    index = max(0, math.ceil(len(values) * percentile / 100) - 1)
    return float(sorted(values)[index])


def _validate_measurement(sample: PerformanceSample) -> None:
    values = (
        sample.ttft_ms,
        sample.total_ms,
        sample.retrieval_ms,
        sample.generation_provider_ms,
        sample.embedding_provider_ms,
        sample.persistence_ms,
    )
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("performance measurements must be finite non-negative milliseconds")
