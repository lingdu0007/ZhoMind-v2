"""Content-free Pilot observations; historical portfolio metrics stay separate."""

from collections import Counter
from collections.abc import Sequence
from math import ceil, isfinite
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from app.common.canonical_json import canonical_json_sha256
from app.extensions.provider_router import ADVANCE_REASONS, STOP_REASONS

Milliseconds = Annotated[float, Field(ge=0)]
Count = Annotated[int, Field(ge=0)]
ConfigurationIdentity = Annotated[str, Field(pattern=r"^configuration:[0-9a-f]{64}$")]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False, hide_input_in_errors=True)


class WorkloadProfile(Observation):
    schema_version: Literal[1] = 1
    admitted_members: Literal[8] = 8
    executing: Literal[2] = 2
    queued: Literal[2] = 2
    member_executing: Literal[1] = 1
    member_queued: Literal[1] = 1
    background_workers: Literal[1] = 1
    active_entries: Literal[50] = 50
    maximum_eligible_chunks: Literal[10000] = 10000
    c1_requests: Literal[40] = 40
    c2_requests: Literal[40] = 40
    burst_rounds: Literal[10] = 10
    burst_size: Literal[4] = 4


class MeasurementBinding(Observation):
    deployment: Annotated[str, Field(pattern=r"^deployment:[0-9a-f]{64}$")]
    host: Annotated[str, Field(pattern=r"^host:[0-9a-f]{64}$")]
    configuration: ConfigurationIdentity
    provider_route: Annotated[str, Field(pattern=r"^provider_route:[0-9a-f]{64}$")]
    retrieval_profile: Annotated[str, Field(pattern=r"^retrieval_profile:[0-9a-f]{64}$")]
    corpus: Annotated[str, Field(pattern=r"^corpus:[0-9a-f]{64}$")]
    editorial_inputs_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    chunk_envelope: ConfigurationIdentity
    product_revision: Annotated[str, Field(pattern=r"^product_revision:[0-9a-f]{40}$")]
    host_vcpu: Annotated[int, Field(gt=0)]
    host_memory_mib: Annotated[int, Field(gt=0)]
    active_entries: Count
    eligible_chunks: Count
    concurrency: Literal[1, 2, 4]
    window_start: AwareDatetime
    window_end: AwareDatetime
    mode: Literal["local_deterministic", "controlled_live"]

    @model_validator(mode="after")
    def ordered_window(self) -> Self:
        if self.window_end <= self.window_start:
            raise ValueError("invalid observation window")
        return self

    @property
    def identity(self) -> str:
        return "configuration:" + canonical_json_sha256(self.model_dump(mode="json"))


class RequestMeasurement(Observation):
    request_id: UUID
    binding_identity: ConfigurationIdentity
    observed_at: AwareDatetime
    population: Literal["admitted", "invalid", "unauthorized", "canceled", "throttled", "rejected"]
    state: Literal["completed", "failed", "stopped", "throttled", "rejected"]
    outcome: Literal[
        "evidence_gated_answer", "insufficient_evidence_reply", "non_knowledge_base_reply", "generation_unavailable",
    ] | None = None
    acknowledgement_ms: Milliseconds | None = None
    processing_ms: Milliseconds | None = None
    answer_content_ms: Milliseconds | None = None
    closed_outcome_ms: Milliseconds | None = None
    terminal_ms: Milliseconds
    queue_ms: Milliseconds | None = None
    retrieval_ms: Milliseconds | None = None
    provider_ms: Milliseconds | None = None
    persistence_ms: Milliseconds | None = None
    application_controlled_ms: Milliseconds | None = None
    route_attempt_ms: tuple[Milliseconds, ...] = ()
    route_reason: str | None = None
    normalized_failure: Literal["queue", "retrieval", "provider", "persistence", "stream", "application", "authorization"] | None = None
    burst_round: Annotated[int, Field(ge=1, le=10)] | None = None
    queued: bool = False

    @model_validator(mode="after")
    def consistent_result(self) -> Self:
        if (self.state == "completed") != (self.outcome is not None):
            raise ValueError("contradictory closed outcome")
        if self.route_reason is not None and self.route_reason not in ADVANCE_REASONS | STOP_REASONS | {"succeeded"}:
            raise ValueError("invalid route reason")
        if len(self.route_attempt_ms) > 4:
            raise ValueError("invalid route attempts")
        if self.route_attempt_ms and self.provider_ms is not None and sum(self.route_attempt_ms) > self.provider_ms + 5:
            raise ValueError("provider timing omits route attempts")
        for name in ("acknowledgement_ms", "processing_ms", "answer_content_ms", "closed_outcome_ms",
                     "queue_ms", "retrieval_ms", "provider_ms", "persistence_ms", "application_controlled_ms"):
            value = getattr(self, name)
            if value is not None and value > self.terminal_ms:
                raise ValueError("timing exceeds terminal observation")
        if self.closed_outcome_ms is not None and self.state != "completed":
            raise ValueError("non-completed execution has no closed-outcome time")
        if self.outcome == "generation_unavailable" and self.answer_content_ms is not None:
            raise ValueError("generation preview is not answer content")
        return self


class BuildItemMeasurement(Observation):
    item_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    attempt: Annotated[int, Field(ge=1)]
    status: Literal["candidate_ready", "failed", "canceled", "interrupted_retryable", "superseded", "queued", "running"]
    execution_ms: Milliseconds | None = None
    chunks: Count
    frozen_input_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    embedding_configuration_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None


class AdmissionPeak(Observation):
    binding_identity: ConfigurationIdentity
    burst_round: Annotated[int, Field(ge=1, le=10)]
    observed_at: AwareDatetime
    executing: Literal[2]
    queued: Literal[2]


class BuildMeasurement(Observation):
    binding_identity: ConfigurationIdentity
    kind: Literal["ten_item_bundle", "fifty_entry_rebuild"]
    bundle_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    configuration_identity: ConfigurationIdentity
    items: tuple[BuildItemMeasurement, ...]
    elapsed_ms: Milliseconds
    observed_start: AwareDatetime
    observed_end: AwareDatetime

    @model_validator(mode="after")
    def unique_items(self) -> Self:
        if len({item.item_sha256 for item in self.items}) != len(self.items):
            raise ValueError("duplicate build item")
        if any(item.execution_ms is not None and item.execution_ms > self.elapsed_ms for item in self.items):
            raise ValueError("item timing exceeds batch time")
        return self


def summarize_build(binding: MeasurementBinding, observation: BuildMeasurement) -> dict:
    if observation.binding_identity != binding.identity or observation.configuration_identity != binding.configuration:
        raise ValueError("build measurement binding mismatch")
    if not binding.window_start <= observation.observed_start <= observation.observed_end <= binding.window_end:
        raise ValueError("build outside observation window")
    expected, budget = (10, 1800000) if observation.kind == "ten_item_bundle" else (50, 3600000)
    complete = sum(item.status == "candidate_ready" and item.execution_ms is not None for item in observation.items)
    status = "passing" if observation.elapsed_ms <= budget else "at_risk"
    if complete != expected or len(observation.items) != expected:
        status = "unavailable"
    if any(item.status in {"failed", "canceled", "interrupted_retryable", "superseded"} or item.attempt != 1
           for item in observation.items):
        status = "at_risk"
    if observation.kind == "fifty_entry_rebuild" and sum(item.chunks for item in observation.items) > 10000:
        status = "at_risk"
    return {
        "schema": "pilot_build_report/v1", "binding": binding.model_dump(mode="json"),
        "observation": observation.model_dump(mode="json"), "status": status, "complete_items": complete,
        "item_execution": distribution(
            [item.execution_ms for item in observation.items if item.execution_ms is not None], threshold_ms=300000,
        ),
    }


def distribution(values: Sequence[float], *, threshold_ms: float | None = None) -> dict:
    if any(isinstance(value, bool) or not isinstance(value, int | float)
           or not isfinite(value) or value < 0 for value in values):
        raise ValueError("invalid measurement")
    result: dict = {
        "count": len(values), "values_ms": list(values),
        "maximum_ms": max(values) if values else None,
        "status": "observed_with_insufficient_sample" if values else "unavailable",
    }
    if len(values) >= 20:
        ordered = sorted(values)
        result.update({
            "p50_ms": ordered[ceil(len(values) * .50) - 1],
            "p95_ms": ordered[ceil(len(values) * .95) - 1],
            "observational_p99_ms": ordered[ceil(len(values) * .99) - 1],
            "method": "nearest_rank",
        })
        result["status"] = (
            "observed" if threshold_ms is None else
            "passing" if result["p95_ms"] <= threshold_ms else "at_risk"
        )
    return result


def route_counts(samples: Sequence[RequestMeasurement]) -> tuple[int, int]:
    successes = sum(sample.route_reason == "succeeded" and sample.outcome == "evidence_gated_answer" for sample in samples)
    unavailable = sum(sample.outcome == "generation_unavailable" and sample.route_reason in ADVANCE_REASONS for sample in samples)
    return successes, unavailable


def summarize_requests(
    binding: MeasurementBinding, samples: Sequence[RequestMeasurement], *,
    admission_peaks: Sequence[AdmissionPeak] = (),
) -> dict:
    seen: set[UUID] = set()
    for sample in samples:
        if sample.request_id in seen:
            raise ValueError("duplicate request observation")
        seen.add(sample.request_id)
        if sample.binding_identity != binding.identity:
            raise ValueError("measurement binding mismatch")
        if not binding.window_start <= sample.observed_at < binding.window_end:
            raise ValueError("sample outside observation window")
    eligible = [sample for sample in samples if sample.population == "admitted"]
    metrics = {}
    thresholds = {"acknowledgement_ms": 1000, "processing_ms": 2000}
    if binding.concurrency == 2:
        thresholds["closed_outcome_ms"] = 30000
    for name in (
        "acknowledgement_ms", "processing_ms", "answer_content_ms", "closed_outcome_ms",
        "queue_ms", "retrieval_ms", "provider_ms", "persistence_ms", "application_controlled_ms",
    ):
        population = (
            [sample for sample in eligible if sample.outcome != "generation_unavailable"]
            if name == "answer_content_ms" else eligible
        )
        values = [getattr(sample, name) for sample in population if getattr(sample, name) is not None]
        metric = distribution(values, threshold_ms=thresholds.get(name))
        metric["missing_count"] = len(population) - len(values)
        if metric["missing_count"]:
            metric["status"] = "unavailable"
        metrics[name] = metric
    generation = sorted(
        [sample for sample in eligible if sample.route_attempt_ms or sample.outcome in {"evidence_gated_answer", "generation_unavailable"}],
        key=lambda sample: (sample.observed_at, str(sample.request_id)),
    )
    rolling = generation[-20:]
    successes, unavailable = route_counts(rolling)
    success_rate = successes / len(rolling) if rolling else None
    failed_windows = 0
    for index in range(20, len(generation) + 1):
        window = generation[index - 20:index]
        successes, provider_failures = route_counts(window)
        failed_windows += successes < 19 or provider_failures > 1
    missing_attempts = sum(not sample.route_attempt_ms or sample.route_reason is None for sample in generation)
    burst: dict = {"status": "not_applicable"}
    if binding.concurrency == 4:
        rounds = [[sample for sample in eligible if sample.burst_round == index] for index in range(1, 11)]
        complete_rounds = sum(len(items) == 4 and sum(item.queued for item in items) == 2 for items in rounds)
        burst = {
            "rounds_with_two_queued": complete_rounds,
            "status": "passing" if all(sample.closed_outcome_ms is not None and sample.closed_outcome_ms <= 60000
                                       for sample in eligible) else "at_risk",
        }
        peak_rounds = set()
        for peak in admission_peaks:
            if peak.binding_identity != binding.identity or not binding.window_start <= peak.observed_at < binding.window_end:
                raise ValueError("admission peak binding mismatch")
            if peak.burst_round in peak_rounds:
                raise ValueError("duplicate admission peak")
            peak_rounds.add(peak.burst_round)
        burst["observed_two_executing_two_queued"] = len(peak_rounds)
        burst["admission_peaks"] = [peak.model_dump(mode="json") for peak in admission_peaks]
        if complete_rounds != 10 or len(eligible) != 40 or len(peak_rounds) != 10:
            burst["status"] = "unavailable"
    return {
        "schema": "pilot_request_report/v1", "binding": binding.model_dump(mode="json"),
        "binding_identity": binding.identity, "eligible_count": len(eligible),
        "separate_populations": dict(Counter(sample.population for sample in samples if sample.population != "admitted")),
        "admitted_failures": sum(sample.state != "completed" for sample in eligible),
        "normalized_failures": dict(Counter(sample.normalized_failure for sample in samples if sample.normalized_failure)),
        "metrics": metrics,
        "burst": burst,
        "route": {
            "generation_required_count": len(generation), "rolling_20_count": len(rolling),
            "rolling_20_success_rate": success_rate, "provider_unavailable_count": unavailable,
            "failed_windows": failed_windows, "missing_attempt_observations": missing_attempts,
            "status": (
                "unavailable" if missing_attempts else
                "observed_with_insufficient_sample" if len(rolling) < 20 else
                "at_risk" if failed_windows else "passing"
            ),
        },
        "live_acceptance": False,
    }
